"Per-source decode state: ring buffer + background worker. Every source in
the sequence gets its own pool, created on first use (see [`pool`](@ref)).
`source` is what actually gets decoded — the original, or its preview proxy
after [`startproxy!`](@ref) swapped the pool."
struct SourcePool
    source::VideoSource
    ring::FrameRing
    worker::DecodeWorker
end

function SourcePool(source::VideoSource; capacity::Integer = 64)
    ring = FrameRing(source.width, source.height; capacity)
    return SourcePool(source, ring, DecodeWorker(source, ring))
end

"""
    Player(path; capacity=64)

Open a video in a GLMakie editor window: big preview, zoomable thumbnail
timeline over an editable `Sequence`, and playback controls. More videos
can be added by dropping their files onto the window (or `addsource!`);
each source decodes through its own worker/ring pool.

Keys: space = play/pause, ←/→ = step (shift ±10), S = split at playhead,
X/Del = ripple-delete clip at playhead, C = crop mode (drag a rect on the
preview), R = reset crop, Esc = cancel crop mode.
Timeline mouse: left-drag scrubs, Ctrl+left-drag moves a clip (snapping),
scroll zooms, right-drag pans.

The playhead (timeline frames) is the single source of truth; its listener
resolves through the sequence (`locate`) to a source frame, targets that
source's decode worker and presents when the frame is in the ring. Gaps
show black. Crop is non-destructive per clip, applied via axis limits.
"""
mutable struct Player
    const sequence::Sequence
    const pools::Dict{VideoSource, SourcePool}
    const capacity::Int
    const proxyheight::Int     # preview proxy resolution (see startproxy!)
    const proxythreshold::Int  # source pixel count above which a proxy is auto-generated
    const timeline::Timeline
    const frame::Observable{RGBFrame}
    const playhead::Observable{Int}
    const playing::Observable{Bool}
    const status::Observable{String}
    const statusqueue::Channel{String}
    const uiqueue::Channel{Function}   # thread → main-thread actions
    const applytracks::Observable{Bool}
    const stabinfo::Observable{String}
    const analysisbackend::Any  # KA backend for motion analysis (e.g. LavaBackend())
    const fig::Figure
    const previewaxis::Axis
    const cropmode::Base.RefValue{Bool}
    const croprect::Observable{Vector{Point2f}}
    fxtmp1::RGBFrame  # match the active clip's source resolution
    fxtmp2::RGBFrame
    const fxsliders::Dict{Symbol, Slider}
    const fxwidgets::Dict{Symbol, Any}  # panel menu/buttons (dock content is
                                        # invisible to fig.content — tests and
                                        # MCP reach the widgets through here)
    const fxsyncing::Base.RefValue{Bool}
    cropanchor::Union{Nothing, Point2f}
    lastcrop::NTuple{4, Float64}
    lastclip::Union{Nothing, Clip}
    clipmodal::Any
    rctime::Float64
    gpuworker::Any
    onpick::Any  # armed one-shot preview-click callback (object lock)
    previewplot::Any  # the preview image plot (texture swap target, glbridge.jl)
    gpupreview::Any   # GPUPreview when Player(gpupreview = true), else nothing
    audio::Any        # AudioPreview when Player(audiopreview = true), else nothing
    const undostack::Vector{Vector{Clip}}
    const redostack::Vector{Vector{Clip}}
    lastslidersnap::Float64
    screen::Any
    presented::Int
    dropped::Int
    const dockpanels::Dict{Symbol, Any}  # dock key → (; sf::Subfigure, width)
    const dockopen::Observable{Symbol}   # open dock panel, :none when collapsed
    const mediasources::Observable{Vector{VideoSource}}  # media-bin content
    const binrows::Vector{Any}           # bin row buttons (rebuilt on change)
    dragsource::Any                      # bin source mid-drag onto the timeline
    const tool::Observable{Symbol}       # armed tool: :none, :split, :crop
end

"Push the current edit state onto the undo stack (clears redo)."
function snapshot!(player::Player)
    push!(player.undostack, snapshot(player.sequence))
    length(player.undostack) > 100 && popfirst!(player.undostack)
    empty!(player.redostack)
    return nothing
end

function undo!(player::Player)
    isempty(player.undostack) && return nothing
    push!(player.redostack, snapshot(player.sequence))
    restore!(player.sequence, pop!(player.undostack))
    postrestore!(player)
    return nothing
end

function redo!(player::Player)
    isempty(player.redostack) && return nothing
    push!(player.undostack, snapshot(player.sequence))
    restore!(player.sequence, pop!(player.redostack))
    postrestore!(player)
    return nothing
end

function postrestore!(player::Player)
    player.playhead[] = clamp(player.playhead[], 0, max(seqlength(player.sequence) - 1, 0))
    player.lastclip = nothing  # force slider resync (clip identities changed)
    refreshedit!(player)
    return nothing
end

"""
Runs GPU analyses on one pinned thread: Lava's BatchQueue is single-writer
(the owning thread is fixed when the Vulkan context is first created), so
every device dispatch must come from the same thread. The worker's first
job initializes the context on its pinned thread and owns it from then on —
don't touch Lava from other threads in the same session.
"""
struct GPUWorker
    jobs::Channel{Function}
    task::Task

    function GPUWorker()
        jobs = Channel{Function}(8)
        task = Task(() -> for f in jobs
            Base.invokelatest(f)  # jobs may be defined after the worker started
        end)
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, Threads.nthreads() - 1)
        schedule(task)
        return new(jobs, task)
    end
end

function rungpu(f::Function, player::Player)
    player.gpuworker === nothing && (player.gpuworker = GPUWorker())
    put!(player.gpuworker.jobs, f)
    return nothing
end

function Player(path::AbstractString; capacity::Integer = 64,
                background = RGBf(0.114, 0.12, 0.135), accent = RGBf(1.0, 0.47, 0.22),
                analysisbackend = KA.CPU(), gpupreview::Bool = false,
                audiopreview::Bool = true,
                proxyheight::Integer = 720, proxythreshold::Integer = 2_100_000)
    gpupreview && analysisbackend isa KA.CPU &&
        error("gpupreview = true requires a GPU backend, e.g. Player(path; analysisbackend = LavaBackend(), gpupreview = true)")
    # a .toml path opens a saved project (Ctrl+S / saveproject) instead of a video
    sequence = endswith(lowercase(path), ".toml") ? loadproject(path) :
                                                    Sequence(VideoSource(path))
    isempty(sequence.clips) && error("project has no clips: $path")
    source = sequence.clips[1].source
    pools = Dict{VideoSource, SourcePool}(source => SourcePool(source; capacity))

    frame = Observable(zeros(RGB{N0f8}, source.width, source.height))
    playhead = Observable(0)
    playing = Observable(false)

    uicolors = Makie.derive_colors(; background, accent)
    player = Makie.with_theme(colors = Makie.Attributes(; uicolors...),
                              backgroundcolor = background,
                              textcolor = uicolors.text) do
        buildui(sequence, pools, Int(capacity), Int(proxyheight), Int(proxythreshold),
                frame, playhead, playing; background, uicolors, analysisbackend)
    end
    player.screen = display(player.fig)
    gpupreview && (player.gpupreview = GPUPreview())
    audiopreview && (player.audio = AudioPreview())
    retrypresent(player, 0)
    for src in unique(c.source for c in sequence.clips)  # projects may be multi-source
        needsproxy(src; maxpixels = proxythreshold) && startproxy!(player, src)
    end
    Threads.@spawn begin  # keep the regenerable proxy/PCM caches bounded
        pruned = prunecache!()
        pruned > 0 && setstatus!(player, "cache pruned — freed $(round(pruned / 2^30, digits = 1)) GiB")
    end
    return player
end

"The decode pool for `source`, created on first use."
pool(player::Player, source::VideoSource) =
    get!(() -> SourcePool(source; capacity = player.capacity), player.pools, source)

"""
    startproxy!(player, source; height=player.proxyheight)

Generate a preview proxy for `source` in the background (see
[`generateproxy`](@ref)) and switch its decode pool over once ready:
scrubbing and playback then decode the lightweight proxy while analysis,
export and the project file keep using the original. Runs automatically
for sources above `player.proxythreshold` pixels.
"""
function startproxy!(player::Player, source::VideoSource;
                     height::Integer = player.proxyheight)
    Threads.@spawn try
        setstatus!(player, "generating preview proxy for $(basename(source.path))…")
        proxy = generateproxy(source; height)
        put!(player.uiqueue, () -> begin
            old = get(player.pools, source, nothing)
            player.pools[source] = SourcePool(proxy; capacity = player.capacity)
            old === nothing || stop!(old.worker)
            notify(player.playhead)  # re-present through the proxy
            player.playing[] || retrypresent(player, player.playhead[])
            setstatus!(player, "preview proxy ready — $(proxy.width)×$(proxy.height)")
        end)
    catch e
        setstatus!(player, "proxy generation failed: $(sprint(showerror, e))")
        @error "proxy generation failed" exception = (e, catch_backtrace())
    end
    return nothing
end

function buildui(sequence, pools, capacity, proxyheight, proxythreshold,
                 frame, playhead, playing; background, uicolors, analysisbackend)
    fig = Figure(size = (1500, 950))
    # column 1: far-left vertical toolbar; column 2: THE dock — one fixed slot
    # where the effects / media / export panels open (never over the preview);
    # column 3: the preview. Timeline and controls span the full window width.
    toolbar = GridLayout(fig[1, 1]; tellheight = false, valign = :top)
    # DataAspect keeps pixels square and letterboxes the crop inside the cell,
    # so we never show outside the crop (which would re-reveal the warp border
    # a stabilization crop hides). The wide dock-less cell gives the extra width.
    ax = Axis(fig[1, 3], aspect = DataAspect(), yreversed = true,
              backgroundcolor = background)
    hidedecorations!(ax)
    hidespines!(ax)
    deregister_interaction!(ax, :rectanglezoom)  # left-drag is the crop tool
    previewplot = image!(ax, frame; interpolate = true)

    timeline = Timeline(fig[2, 1:3], sequence, playhead, playing)
    rowsize!(fig.layout, 2, Makie.Fixed(96))

    # onboarding hint; replaced by the first real status update
    status = Observable("Space plays · S splits · right-click a clip for all actions")
    controls = GridLayout(fig[3, 1:3], tellwidth = false)
    playbtn = Button(controls[1, 1]; label = map(p -> p ? "Pause" : "Play", playing), width = 80)
    Label(controls[1, 2], map(n -> timecode(sequence, n), playhead); width = 220)
    exportbtn = Button(controls[1, 3]; label = "Export", width = 80)
    muted = Observable(false)
    mutebtn = Button(controls[1, 4]; label = map(m -> m ? "Muted" : "Sound", muted), width = 70)
    Label(controls[1, 5], status; width = 380, halign = :left, fontsize = 12,
          color = uicolors.text_muted)

    player = Player(sequence, pools, capacity, proxyheight, proxythreshold,
                    timeline, frame, playhead,
                    playing, status, Channel{String}(32), Channel{Function}(32),
                    Observable(true), Observable("no analysis yet"), analysisbackend,
                    fig, ax, Ref(false),
                    Observable(Point2f[]),
                    similar(frame[]), similar(frame[]), Dict{Symbol, Slider}(),
                    Dict{Symbol, Any}(),
                    Ref(false), nothing, (0.0, 0.0, 1.0, 1.0), nothing, nothing, 0.0,
                    nothing, nothing, nothing, nothing, nothing,
                    Vector{Clip}[], Vector{Clip}[], 0.0, nothing, 0, 0,
                    Dict{Symbol, Any}(), Observable(:none),
                    Observable(VideoSource[]), Any[], nothing, Observable(:none))
    @async for s in player.statusqueue  # main-thread consumer: threads → observable
        status[] = s
    end
    player.previewplot = previewplot
    @async for f in player.uiqueue      # main-thread consumer: threads → UI actions
        Base.invokelatest(f)
    end
    on(exportbtn.clicks) do _
        toggledock!(player, :export)   # options live in the export dock panel
    end

    lines!(ax, player.croprect; color = :orangered, linewidth = 2)

    fxdock = dockpanel!(player, :effects)
    buildfxpanel!(player, fxdock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[1, 1], "FX", :effects, uicolors)
    player.mediasources[] = unique([c.source for c in sequence.clips])
    mediadock = dockpanel!(player, :media)
    buildmediabin!(player, mediadock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[2, 1], "Bin", :media, uicolors)
    exportdock = dockpanel!(player, :export)
    buildexportpanel!(player, exportdock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[3, 1], "Out", :export, uicolors)
    # tool strip: ✂ split and ▢ crop ARM (change the cursor + act where you
    # click/drag); ✕ ↶ ↷ are one-shot at the playhead
    splitbtn = Button(toolbar[4, 1]; label = "✂", width = 40, height = 40)
    cropbtn = Button(toolbar[5, 1]; label = "▢", width = 40, height = 40)
    on(_ -> armtool!(player, :split), splitbtn.clicks)
    on(_ -> armtool!(player, :crop), cropbtn.clicks)
    oneshots = [("✕", () -> deleteat!(player)),
                ("↶", () -> isempty(player.undostack) ? setstatus!(player, "nothing to undo") :
                            (undo!(player); setstatus!(player, "undone (Ctrl+Z redoes with Shift)"))),
                ("↷", () -> isempty(player.redostack) ? setstatus!(player, "nothing to redo") :
                            (redo!(player); setstatus!(player, "redone")))]
    for (row, (lbl, action)) in enumerate(oneshots)
        on(_ -> action(), Button(toolbar[5 + row, 1]; label = lbl, width = 40, height = 40).clicks)
    end
    # armed tool → cursor, crop mode, hint, button highlight (cropmode is a
    # Ref, not an Observable, so it's driven here and reset in finishcrop!)
    on(player.tool; update = true) do t
        player.cropmode[] = (t === :crop)
        setcursor!(player, t === :none ? :arrow : :crosshair)
        splitbtn.buttoncolor[] = t === :split ? uicolors.accent : uicolors.surface
        cropbtn.buttoncolor[] = t === :crop ? uicolors.accent : uicolors.surface
        t === :split && setstatus!(player, "split tool — click the timeline where you want to cut")
        t === :crop && setstatus!(player, "crop tool — drag a rectangle on the preview")
    end
    # split tool: the next timeline click cuts THERE (not at the playhead)
    on(events(fig).mousebutton; priority = 95) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        player.tool[] === :split || return Consume(false)
        mp = Point2f(events(fig).mouseposition[])
        tlscene = timeline.axis.scene
        mp in tlscene.viewport[] || return Consume(false)
        t = Makie.mouseposition(tlscene)[1]
        frame = clamp(round(Int, t * sequence.framerate), 0, max(seqlength(sequence) - 1, 0))
        snapshot!(player)  # so Ctrl+Z / the undo tool can revert the cut
        split!(sequence, frame)
        refreshedit!(player)
        setstatus!(player, "split at $(timecode(sequence, frame))")
        player.tool[] = :none
        return Consume(true)
    end
    opendock!(player, :effects)   # the working panel starts open

    on(playhead) do n
        loc = locate(sequence, n)
        clip = loc === nothing ? nothing : loc[1]
        if clip !== player.lastclip
            player.lastclip = clip
            if clip !== nothing
                syncsliders!(player, clip)
                # the panel always tells WHICH stabilization this clip carries
                player.stabinfo[] = stabdescription(clip.motiontrack)
            end
        end
        if !present!(player)
            if timeline.scrubbing[]
                loc = locate(sequence, n)
                if loc !== nothing
                    clip, srcframe = loc
                    # the scrub fallback blits a RAW per-second thumbnail. That
                    # only matches the composed preview when the clip is neither
                    # stabilized nor cropped — otherwise it would flicker between
                    # the cropped/stabilized exact frame and the raw thumbnail
                    # (and warping a per-second thumbnail by a per-frame track
                    # just wobbles). For those clips, hold the last exact frame.
                    plain = (clip.motiontrack === nothing || !player.applytracks[]) &&
                            clip.crop == (0.0, 0.0, 1.0, 1.0)
                    thumb = plain ? nearestthumb(cachefor(timeline, clip.source),
                                                 floor(Int, srcframe / clip.source.framerate)) :
                            nothing
                    if thumb !== nothing
                        ensureframesize!(player, pool(player, clip.source).source)
                        blitthumb!(frame[], thumb)
                        notify(frame)
                    end
                end
            end
            playing[] || retrypresent(player, n)
        end
    end

    on(playbtn.clicks) do _
        playing[] ? pause!(player) : play!(player)
    end

    on(mutebtn.clicks) do _
        ap = player.audio
        ap === nothing && return
        muted[] = ap.enabled           # toggling: enabled → will be muted
        ap.enabled = !ap.enabled
        if playing[]
            ap.enabled ? startaudio!(player) : stopaudio!(player)
        end
    end

    # drop video files onto the window to append them to the timeline
    on(events(fig).dropped_files) do files
        for f in files
            try
                addsource!(player, f)
            catch e
                setstatus!(player, "could not add $(basename(f)): $(sprint(showerror, e))")
            end
        end
    end

    timeline.onedit = () -> snapshot!(player)
    wirecroptool(player)
    wirekeys(player)
    wireclipmenu!(player)
    return player
end

function timecode(seq::Sequence, n::Integer)
    t = n / seq.framerate
    minutes = floor(Int, t / 60)
    return @sprintf("%02d:%06.3f  %d/%d", minutes, t - 60minutes, n, max(seqlength(seq) - 1, 0))
end

# ---------------------------------------------------------------- presentation

"Resolve and show timeline frame `n` if possible (gaps show black). Returns success."
function showframe!(player::Player, n::Integer)
    loc = locate(player.sequence, n)
    if loc === nothing
        fill!(player.frame[], RGB{N0f8}(0, 0, 0))
        notify(player.frame)
        return true
    end
    clip, srcframe = loc
    sp = pool(player, clip.source)
    target, protect = decodetarget(player, n, clip, srcframe)
    settarget!(sp.worker, target; protect)
    ensureframesize!(player, sp.source)  # proxy resolution when one is active
    if fetchframe!(player.frame[], sp.ring, srcframe)
        gp = player.gpupreview
        if gp isa GPUPreview && !gp.failed && presentgpu!(player, clip, srcframe)
            applycrop!(player, clip)   # tracks/effects ran on the GPU
            return true
        end
        if player.applytracks[]
            applymotiontrack!(player.frame[], player.fxtmp1, clip, srcframe)
            applycolortrack!(player.frame[], clip, srcframe)
        end
        applyeffects!(player.frame[], player.fxtmp1, player.fxtmp2, clip)
        notify(player.frame)
        applycrop!(player, clip)
        return true
    end
    return false
end

"""
Where the clip's decode worker should aim while showing timeline frame `n`:
normally the current source frame, but when the playhead approaches a cut
into an adjacent clip, prefetch across it. Same source: retarget the worker
at the next clip's first frame once the tail is buffered (its slots
protected from reuse). Different source: warm the other source's worker —
its ring is separate, so nothing needs protecting.
"""
function decodetarget(player::Player, n::Integer, clip::Clip, srcframe::Integer)
    seq = player.sequence
    tail = clipend(clip) - n
    0 < tail <= 30 || return (srcframe, 1:0)
    i = clipat(seq, n)
    (i === nothing || i >= length(seq.clips)) && return (srcframe, 1:0)
    nxt = seq.clips[i + 1]
    nxt.start == clipend(clip) || return (srcframe, 1:0)  # gap: nothing to prefetch
    if nxt.source !== clip.source
        settarget!(pool(player, nxt.source).worker, nxt.src_in)
        return (srcframe, 1:0)
    end
    ring = pool(player, clip.source).ring
    for k in 0:(tail - 1)                                 # tail fully buffered?
        hasframe(ring, srcframe + k) || return (srcframe, 1:0)
    end
    return (nxt.src_in, srcframe:(srcframe + tail - 1))   # protect the tail's slots
end

"Match the preview and effect buffers to the active clip's source resolution."
function ensureframesize!(player::Player, source::VideoSource)
    size(player.frame[]) == (source.width, source.height) && return nothing
    player.frame.val = RGBFrame(undef, source.width, source.height)  # notified once filled
    player.fxtmp1 = similar(player.frame[])
    player.fxtmp2 = similar(player.frame[])
    player.lastcrop = (-1.0, 0.0, 0.0, 0.0)  # force applycrop! (limits are per-source)
    return nothing
end

function present!(player::Player)
    if showframe!(player, player.playhead[])
        player.presented += 1
        return true
    end
    player.dropped += 1
    return false
end

"While paused, keep trying to present frame `n` until it lands or the playhead moves."
function retrypresent(player::Player, n::Integer)
    @async begin
        for _ in 1:400
            (player.playhead[] == n && !player.playing[]) || break
            showframe!(player, n) && (player.presented += 1; break)
            sleep(0.005)
        end
    end
    return nothing
end

"""
Set the preview limits to EXACTLY the given rect (y descending — the axis is
reversed, and Makie's `ylims!` derives `yreversed` from argument order). The
axis's `DataAspect` letterboxes the rect inside the cell, so pixels stay square
and — crucially — nothing OUTSIDE the crop is ever shown (a stabilization
crop hides the warp's replicate-border smear; expanding past it re-reveals it).
The width benefit is the now-wide preview cell, not zooming past the crop.
"""
function coverlimits!(player::Player, xlo::Real, xhi::Real, yfirst::Real, ysecond::Real)
    limits!(player.previewaxis, xlo, xhi, yfirst, ysecond)
    return nothing
end

"Apply a clip's crop to the preview via axis limits (zero-copy, GPU-side).
The crop is normalized, so limits come from the DISPLAYED buffer — which is
the proxy's resolution when one is active."
function applycrop!(player::Player, clip::Clip)
    clip.crop == player.lastcrop && return nothing
    player.lastcrop = clip.crop
    x, y, w, h = clip.crop
    W, H = size(player.frame[])
    coverlimits!(player, x * W, (x + w) * W, (y + h) * H, y * H)
    return nothing
end

"""
Make a framing change visible: glide the preview zoom from crop `from` to
crop `to` (instead of snapping), then outline the new framing just inside
the view for a moment. `player.lastcrop` must equal `to` already so regular
presents don't fight the animation.
"""
function showcrop!(player::Player, from::NTuple{4, Float64}, to::NTuple{4, Float64};
                   duration::Real = 0.5, hold::Real = 1.5)
    W, H = size(player.frame[])
    setlimits(c) = coverlimits!(player, c[1] * W, (c[1] + c[3]) * W,
                                (c[2] + c[4]) * H, c[2] * H)  # y descending: axis is reversed
    @async try
        if from != to
            steps = max(round(Int, duration * 60), 1)
            for k in 1:steps
                s = sin(0.5π * k / steps)^2  # ease in/out
                setlimits(from .+ s .* (to .- from))
                sleep(duration / steps)
            end
        end
        setlimits(to)
        player.cropmode[] && return  # the crop tool owns the overlay
        x, y, w, h = to
        ix, iy = 0.015w * W, 0.015h * H  # sit just inside the view edges
        player.croprect[] = Point2f[(x * W + ix, y * H + iy), ((x + w) * W - ix, y * H + iy),
                                    ((x + w) * W - ix, (y + h) * H - iy),
                                    (x * W + ix, (y + h) * H - iy), (x * W + ix, y * H + iy)]
        sleep(hold)
        player.cropmode[] || (player.croprect[] = Point2f[])
    catch e
        @error "crop feedback failed" exception = (e, catch_backtrace())
    end
    return nothing
end

# ---------------------------------------------------------------- transport

function play!(player::Player)
    player.playing[] && return player
    player.playhead[] >= seqlength(player.sequence) - 1 && (player.playhead[] = 0)
    player.playing[] = true
    startaudio!(player)
    @async playloop(player)
    return player
end

function pause!(player::Player)
    player.playing[] = false
    stopaudio!(player)
    return player
end

function step!(player::Player, delta::Integer = 1)
    pause!(player)
    player.playhead[] = clamp(player.playhead[] + delta, 0, max(seqlength(player.sequence) - 1, 0))
    return player
end

"Wall-clock paced playback: playhead follows elapsed time, dropping frames if needed."
function playloop(player::Player)
    fps = player.sequence.framerate
    base = player.playhead[]
    t0 = time_ns()
    lastset = base
    while player.playing[]
        if player.playhead[] != lastset  # external scrub while playing: rebase the clock
            base = player.playhead[]
            t0 = time_ns()
        end
        n = base + floor(Int, (time_ns() - t0) / 1.0e9 * fps)
        if n >= seqlength(player.sequence)
            player.playhead[] = max(seqlength(player.sequence) - 1, 0)
            pause!(player)
            break
        end
        n == player.playhead[] || (player.playhead[] = n)
        lastset = player.playhead[]
        sleep(0.003)
    end
    return nothing
end

Base.seek(player::Player, t::Real) =
    (player.playhead[] = clamp(round(Int, t * player.sequence.framerate), 0,
                               max(seqlength(player.sequence) - 1, 0)); player)

# ---------------------------------------------------------------- editing

"Split / delete / crop-reset operate on the clip under the playhead."
function split!(player::Player)
    snapshot!(player)
    split!(player.sequence, player.playhead[]) === nothing && return (pop!(player.undostack); nothing)
    refreshedit!(player)
    return nothing
end

function Base.deleteat!(player::Player)
    snapshot!(player)
    deleteclip!(player.sequence, player.playhead[]) === nothing && return (pop!(player.undostack); nothing)
    player.playhead[] = clamp(player.playhead[], 0, max(seqlength(player.sequence) - 1, 0))
    refreshedit!(player)
    return nothing
end

resetcrop!(player::Player) = resetcropat!(player, player.playhead[])

function resetcropat!(player::Player, n::Integer)
    loc = locate(player.sequence, n)
    loc === nothing && return nothing
    snapshot!(player)
    loc[1].crop = (0.0, 0.0, 1.0, 1.0)
    refreshedit!(player)
    return nothing
end

"""
    addsource!(player, path) -> Clip

Append the video at `path` as a new clip at the end of the timeline
(undoable). The source must match the sequence framerate (the edit model is
frame-exact, no resampling); resolution may differ. Also triggered by
dropping video files onto the editor window.
"""
function addsource!(player::Player, path::AbstractString)
    source = VideoSource(path)
    seq = player.sequence
    isapprox(source.framerate, seq.framerate; atol = 0.01) ||
        error("framerate $(source.framerate) doesn't match the sequence ($(seq.framerate))")
    snapshot!(player)
    clip = Clip(source, 0, source.nframes, seqlength(seq), (0.0, 0.0, 1.0, 1.0))
    push!(seq.clips, clip)
    registermedia!(player, source)
    limits!(player.timeline.axis, 0.0, seqduration(seq), 0.0, 1.0)  # reveal the new clip
    refreshedit!(player)
    setstatus!(player, "added $(basename(path)) — $(source.width)×$(source.height), " *
                       "$(round(source.nframes / source.framerate, digits = 1))s")
    needsproxy(source; maxpixels = player.proxythreshold) && startproxy!(player, source)
    return clip
end

"The player's default project file: next to the first source."
projectfile(player::Player) =
    splitext(player.sequence.clips[1].source.path)[1] * ".videoedit.toml"

"""
    saveproject!(player) -> path

Save the edit (Ctrl+S) to [`projectfile`](@ref) — reopen it later with
`Player(path)`.
"""
function saveproject!(player::Player)
    if isempty(player.sequence.clips)
        setstatus!(player, "nothing to save — the timeline is empty")
        return nothing
    end
    path = projectfile(player)
    saveproject(path, player.sequence)
    setstatus!(player, "project saved — $path (open with Player(path))")
    return path
end

"Thread-safe status line update (any thread → main-thread observable)."
setstatus!(player::Player, s::String) = (put!(player.statusqueue, s); nothing)

"One-line description of the stabilization active on a clip, for the panel."
stabdescription(track::MotionTrack) = "$(modelabel(track.mode)) — $(tracksummary(track))"
stabdescription(track::ColorTrack) = "color — $(tracksummary(track))"
stabdescription(::Nothing) = "no stabilization on this clip"
modelabel(mode::Symbol) = get(Dict(:similarity => "camera lock", :objectlock => "object lock",
                                   :tripod => "tripod (affine)", :perspective => "perspective",
                                   :smooth => "smooth", :unknown => "stabilized"),
                              mode, String(mode))

"""
    removestabilization!(player; at=playhead) -> nothing

Undo-able removal of the clip's motion track — and of its auto-crop: the
framing glides back to what it was before the analysis (the track's
`basecrop`). The counterpart every analysis needs; replacing a track
silently is not removal UX.
"""
function removestabilization!(player::Player; at::Integer = player.playhead[])
    loc = locate(player.sequence, at)
    if loc === nothing || !(loc[1].motiontrack isa MotionTrack)
        setstatus!(player, "no stabilization to remove at the playhead")
        return nothing
    end
    clip = loc[1]
    snapshot!(player)
    oldcrop = clip.crop
    base = clip.motiontrack.basecrop
    clip.motiontrack = nothing
    base === nothing || (clip.crop = base)
    player.stabinfo[] = stabdescription(nothing)
    setstatus!(player, "stabilization removed" *
                       (base === nothing ? "" : " — framing restored") * " (Ctrl+Z undoes)")
    shown = locate(player.sequence, player.playhead[])
    if shown !== nothing && shown[1] === clip
        player.lastcrop = clip.crop
        notify(player.playhead)
        showcrop!(player, oldcrop, clip.crop)
    else
        player.lastcrop = (-1.0, 0.0, 0.0, 0.0)
        notify(player.playhead)
    end
    return nothing
end

"Run a stabilization analysis for the clip at frame `at` (default: playhead), in the background."
function analyzeat!(player::Player, analyze!::Function, what::String;
                    at::Integer = player.playhead[])
    loc = locate(player.sequence, at)
    if loc === nothing
        setstatus!(player, "$what: no clip under the playhead — move it onto a clip first")
        return nothing
    end
    clip = loc[1]
    setstatus!(player, "$what: analyzing $(cliplength(clip)) frames…")
    player.stabinfo[] = "analyzing…"
    # replacing a track must not stack auto-crops: the new analysis composes
    # its crop from the framing the OLD track's auto-crop replaced
    prevbase = clip.motiontrack isa MotionTrack ? clip.motiontrack.basecrop : nothing
    job = () -> try
        track = analyze!(clip; progress = (d, t) -> setstatus!(player, "$what: $d/$t"))
        summary = track === nothing ? "clip too short to analyze" : stabdescription(track)
        setstatus!(player, "$what ready — $summary")
        put!(player.uiqueue, () -> begin
            player.stabinfo[] = summary
            if track isa MotionTrack
                # hide the warp's replicate borders: shrink the pre-analysis
                # framing to its overlap with the safe region — a no-op when
                # the framing is already inside it
                oldcrop = clip.crop
                track.basecrop = something(prevbase, oldcrop)
                newcrop = cropintersect(track.basecrop,
                                        bordercrop(track, clip.source.width,
                                                   clip.source.height))
                if newcrop != oldcrop
                    snapshot!(player)
                    clip.crop = newcrop
                    setstatus!(player, "$what ready — $summary · cropped to hide stabilized borders (Ctrl+Z keeps framing)")
                end
                # the framing change should be SEEN, not inferred from the
                # status line: when the view shows this clip, hold the old
                # framing while the (now warped) frame refreshes, then glide
                # to the new one; otherwise apply on the clip's next present
                shown = locate(player.sequence, player.playhead[])
                if shown !== nothing && shown[1] === clip
                    player.lastcrop = clip.crop      # presents keep hands off
                    notify(player.playhead)
                    showcrop!(player, oldcrop, clip.crop)
                else
                    player.lastcrop = (-1.0, 0.0, 0.0, 0.0)
                    notify(player.playhead)
                end
            else
                notify(player.playhead)
            end
        end)
    catch e
        setstatus!(player, "$what failed: $(sprint(showerror, e))")
        put!(player.uiqueue, () -> player.stabinfo[] = "analysis failed")
        @error "$what analysis failed" exception = (e, catch_backtrace())
    end
    if player.analysisbackend isa KA.CPU
        Threads.@spawn job()
    else
        rungpu(job, player)  # GPU dispatches must all come from one pinned thread
    end
    return nothing
end

function tracksummary(track::MotionTrack)
    shift = maximum(M -> hypot(M[1, 3], M[2, 3]), track.transforms)
    rot = maximum(M -> abs(atand(M[2, 1], M[1, 1])), track.transforms)
    return "max correction $(round(shift, digits = 1)) px, $(round(rot, digits = 2))° rotation"
end
function tracksummary(track::ColorTrack)
    dev = maximum(zip(track.gains, track.offsets)) do (g, o)
        max(maximum(abs.(g .- 1.0f0)), maximum(abs.(o)))
    end
    return "max adjustment $(round(Int, 100 * dev))%"
end

"Re-present the playhead frame and repaint the timeline after an edit."
function refreshedit!(player::Player)
    relayout!(player.timeline)
    notify(player.playhead)
    player.playing[] || retrypresent(player, player.playhead[])
    return nothing
end

"Arm a one-shot preview click that picks the subject to lock onto."
function armpick!(player::Player)
    player.onpick = p -> analyzeat!(player,
        (clip; kwargs...) -> begin
            # preview data coords may be proxy pixels; analysis reads the original
            scale = clip.source.width / size(player.frame[], 1)
            analyzeobject!(clip, (p[1] * scale, p[2] * scale);
                           backend = player.analysisbackend, kwargs...)
        end,
        "object lock")
    setstatus!(player, "object lock: click the subject in the preview (Esc cancels)")
    return nothing
end

function wirecroptool(player::Player)
    ax = player.previewaxis
    on(events(ax.scene).mousebutton) do event
        pick = player.onpick
        if pick !== nothing && event.button == Mouse.left &&
           event.action == Mouse.press && is_mouseinside(ax.scene)
            player.onpick = nothing
            pick(Point2f(mouseposition(ax.scene)))
            return Consume(true)
        end
        return Consume(false)
    end
    on(events(ax.scene).mousebutton) do event
        event.button == Mouse.left || return Consume(false)
        player.cropmode[] || return Consume(false)
        if event.action == Mouse.press && is_mouseinside(ax.scene)
            player.cropanchor = Point2f(mouseposition(ax.scene))
            return Consume(true)
        elseif event.action == Mouse.release && player.cropanchor !== nothing
            finishcrop!(player, Point2f(mouseposition(ax.scene)))
            return Consume(true)
        end
        return Consume(false)
    end
    on(events(ax.scene).mouseposition) do _
        anchor = player.cropanchor
        if anchor !== nothing && player.cropmode[]
            pos = Point2f(mouseposition(ax.scene))
            player.croprect[] = Point2f[anchor, Point2f(pos[1], anchor[2]), pos,
                                        Point2f(anchor[1], pos[2]), anchor]
        end
        return Consume(false)
    end
    return nothing
end

function finishcrop!(player::Player, corner::Point2f)
    anchor = player.cropanchor
    player.cropanchor = nothing
    player.cropmode[] = false
    player.tool[] === :none || (player.tool[] = :none)  # disarm the crop tool
    player.croprect[] = Point2f[]
    anchor === nothing && return nothing
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    snapshot!(player)
    W, H = size(player.frame[])  # preview data coords = displayed (maybe proxy) pixels
    x0, x1 = minmax(anchor[1], corner[1])
    y0, y1 = minmax(anchor[2], corner[2])
    x0, x1 = clamp(x0 / W, 0.0, 1.0), clamp(x1 / W, 0.0, 1.0)
    y0, y1 = clamp(y0 / H, 0.0, 1.0), clamp(y1 / H, 0.0, 1.0)
    (x1 - x0 < 0.01 || y1 - y0 < 0.01) && return nothing  # degenerate drag
    clip.crop = (x0, y0, x1 - x0, y1 - y0)
    applycrop!(player, clip)
    return nothing
end

function wirekeys(player::Player)
    on(events(player.fig).keyboardbutton) do event
        event.action in (Keyboard.press, Keyboard.repeat) || return Consume(false)
        shift = ispressed(player.fig, Keyboard.left_shift | Keyboard.right_shift)
        ispress = event.action == Keyboard.press
        if event.key == Keyboard.space && ispress
            player.playing[] ? pause!(player) : play!(player)
        elseif event.key == Keyboard.right
            step!(player, shift ? 10 : 1)
        elseif event.key == Keyboard.left
            step!(player, shift ? -10 : -1)
        elseif event.key == Keyboard.s && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            saveproject!(player)
        elseif event.key == Keyboard.s && ispress
            split!(player)
        elseif (event.key == Keyboard.x || event.key == Keyboard.delete) && ispress
            deleteat!(player)
        elseif event.key == Keyboard.c && ispress
            armtool!(player, :crop)
        elseif event.key == Keyboard.r && ispress
            resetcrop!(player)
        elseif event.key == Keyboard.a && ispress
            analyzeat!(player, analyzecolor!, "color stabilization")
        elseif event.key == Keyboard.m && ispress
            analyzeat!(player, (c; kw...) -> analyzemotion!(c; backend = player.analysisbackend, kw...), "motion stabilization")
        elseif event.key == Keyboard.z && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            shift ? redo!(player) : undo!(player)
        elseif event.key == Keyboard.escape && ispress
            player.cropmode[] = false
            player.cropanchor = nothing
            player.croprect[] = Point2f[]
            player.tool[] === :none || (player.tool[] = :none)
            if player.onpick !== nothing
                player.onpick = nothing
                setstatus!(player, "object lock cancelled")
            end
        else
            return Consume(false)
        end
        return Consume(true)
    end
    return nothing
end

# ---------------------------------------------------------------- left dock

"""
Register (or fetch) the dock panel `Subfigure` for `key`. The dock is the ONE
fixed slot between the toolbar and the preview that the effects, media and
export panels share — panels open there, never over the video. Built hidden;
show it with [`opendock!`](@ref).
"""
function dockpanel!(player::Player, key::Symbol; width::Real = 300)
    haskey(player.dockpanels, key) && return player.dockpanels[key].sf
    sf = Subfigure(player.fig[1, 2]; visible = false, contentpadding = 8)
    player.dockpanels[key] = (; sf, width = Float64(width))
    return sf
end

"Open dock panel `key` (closing any other), or collapse the dock with `:none`."
function opendock!(player::Player, key::Symbol)
    entry = key === :none ? nothing : player.dockpanels[key]
    for (k, e) in player.dockpanels
        e.sf.visible = k === key
    end
    layout = player.fig.layout
    colsize!(layout, 2, Makie.Fixed(entry === nothing ? 0.0 : entry.width))
    colgap!(layout, 2, entry === nothing ? 0.0 : 8.0)
    player.dockopen[] = key
    return nothing
end

closedock!(player::Player) = opendock!(player, :none)

toggledock!(player::Player, key::Symbol) =
    opendock!(player, player.dockopen[] === key ? :none : key)

"Toolbar button that toggles dock panel `key`, highlighted while it is open."
function toolbarbutton!(player::Player, gridpos, label::AbstractString, key::Symbol,
                        uicolors)
    btn = Button(gridpos; label, width = 40, height = 40)
    on(_ -> toggledock!(player, key), btn.clicks)
    on(player.dockopen; update = true) do open
        btn.buttoncolor[] = open === key ? uicolors.accent : uicolors.surface
        btn.labelcolor[] = open === key ? uicolors.text_on_accent : uicolors.text
    end
    return btn
end

# process-wide cache of GLFW standard cursors (immutable handles, shared)
const CURSORS = Dict{Symbol, Any}()

"Set the window's mouse cursor (`:arrow` or `:crosshair`); no-op when headless."
function setcursor!(player::Player, shape::Symbol)
    player.screen === nothing && return nothing
    try  # GLFW cursor calls require the main thread + a real window
        GLFW = GLMakie.GLFW
        cur = get!(CURSORS, shape) do
            GLFW.CreateStandardCursor(shape === :crosshair ? GLFW.CROSSHAIR_CURSOR :
                                      shape === :hand ? GLFW.POINTING_HAND_CURSOR :
                                      GLFW.ARROW_CURSOR)
        end
        GLFW.SetCursor(player.screen.glscreen, cur)
    catch
    end
    return nothing
end

"""
Arm tool `t` (`:split` or `:crop`), or disarm with `:none` — toggling the same
tool disarms it. An armed tool changes the cursor to a crosshair and acts where
you click/drag (split cuts at the clicked timeline position, not the playhead).
"""
function armtool!(player::Player, t::Symbol)
    player.tool[] = player.tool[] === t ? :none : t
    return nothing
end

# ---------------------------------------------------------------- media bin

# process-wide first-frame thumbnail cache, keyed by source path (decoding a
# frame is expensive; the bin rebuilds whenever the source list changes)
const BINTHUMBS = Dict{String, Matrix{RGB{N0f8}}}()

"Decoded, downscaled first frame of `source` (`width` px wide), cached by path."
function firstframethumb(source::VideoSource; width::Integer = 120)
    get!(BINTHUMBS, source.path) do
        f = RGBFrame(undef, source.width, source.height)
        sr = SequentialReader(source)
        try
            readframe!(f, sr, 0)
        finally
            close(sr)
        end
        downscale(f, Int(width), max(round(Int, width * source.height / source.width), 1))
    end
end

"Scale `img` to fit inside a `W × H` canvas (aspect-preserving) and center it,
padding with `fill3` — a uniform letterboxed thumbnail box for the media bin."
function fitbox(img::AbstractMatrix{RGB{N0f8}}, W::Integer, H::Integer, fill3)
    iw, ih = size(img)
    s = min(W / iw, H / ih)
    small = downscale(img, max(round(Int, iw * s), 1), max(round(Int, ih * s), 1))
    return fittile(small, Int(W), Int(H), fill3)
end

"""
Media-bin dock panel: each imported source shows its first-frame thumbnail and
name; press a row and drag onto the timeline (a ghost thumbnail follows the
cursor) — release places the clip at the drop position (an overlapping spot
goes to the end instead). "Import clip…" opens a native file dialog.
"""
function buildmediabin!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1], "Media"; font = :bold, halign = :left)
    importbtn = Button(panel[2, 1]; label = "Import clip…", tellwidth = false)
    Label(panel[3, 1], "press a clip, drag onto the timeline";
          fontsize = 11, halign = :left, color = uicolors.text_muted)
    rows = GridLayout(panel[4, 1])
    on(importbtn.clicks) do _
        Threads.@spawn try   # the dialog blocks — keep the UI thread rendering
            path = Makie.choose_file_dialogue()
            path === nothing ||
                put!(player.uiqueue, () -> importsource!(player, String(path)))
        catch e
            setstatus!(player, "import failed: $(sprint(showerror, e))")
        end
    end
    boxw, boxh = 96, 54
    boxfill = RGB{N0f8}(uicolors.surface_subtle)
    scene = player.dockpanels[:media].sf.scene   # Axis-in-Subfigure won't render;
                                                 # thumbnails draw on this scene
    on(player.mediasources; update = true) do sources
        for (btn, box, tim) in player.binrows
            Makie.delete!(btn); Makie.delete!(box); Makie.delete!(scene, tim)
        end
        empty!(player.binrows)
        for (k, src) in enumerate(sources)
            # a reserved box (renders as the frame) + a label button; the
            # decoded first frame is drawn over the box on the dock scene
            box = Box(rows[k, 1]; width = boxw, height = boxh,
                      color = uicolors.surface_subtle, strokecolor = uicolors.border,
                      strokewidth = 1, cornerradius = 3)
            secs = round(src.nframes / src.framerate; digits = 1)
            btn = Button(rows[k, 2]; label = "$(basename(src.path))\n$(secs)s",
                         tellwidth = false, halign = :left)
            thumb = Observable{Matrix{RGB{N0f8}}}(fill(boxfill, boxw, boxh))
            # position the thumbnail over the box in dock-scene-local pixels
            # (space = :pixel is relative to the scene's viewport origin)
            xy = lift(box.layoutobservables.computedbbox, scene.viewport) do bb, vp
                all(isfinite, bb.origin) && all(isfinite, bb.widths) || return (0.0, 1.0, 0.0, 1.0)
                (bb.origin[1] - vp.origin[1], bb.origin[1] + bb.widths[1] - vp.origin[1],
                 bb.origin[2] - vp.origin[2], bb.origin[2] + bb.widths[2] - vp.origin[2])
            end
            tim = image!(scene, lift(v -> (v[1], v[2]), xy), lift(v -> (v[3], v[4]), xy),
                         thumb; space = :pixel, interpolate = true,
                         visible = lift(d -> d === :media, player.dockopen))
            translate!(tim, 0, 0, 20)
            push!(player.binrows, (btn, box, tim))
            Threads.@spawn try   # decode off the UI thread, publish when ready
                # dock scene is y-up; flip the frame's columns to show it upright
                t = reverse(fitbox(firstframethumb(src), boxw, boxh, boxfill), dims = 2)
                put!(player.uiqueue, () -> (thumb[] = t))
            catch
            end
        end
        return
    end
    # drag preview on the timeline: a translucent accent band showing where the
    # dropped clip would land and how wide it would be (the drop target renders
    # reliably, unlike a fig-scene overlay)
    dropghost = Observable(Rect2f(0, 0, 0, 0))
    dropvis = Observable(false)
    dgp = poly!(player.timeline.axis, dropghost; color = (uicolors.accent, 0.3),
                strokecolor = uicolors.accent, strokewidth = 2, visible = dropvis)
    translate!(dgp, 0, 0, 20)
    function moveghost(mp)
        tlscene = player.timeline.axis.scene
        if mp in tlscene.viewport[]
            t = Makie.mouseposition(tlscene)[1]
            dur = player.dragsource.nframes / player.sequence.framerate
            dropghost[] = Rect2f(t, 0.03, dur, 0.94)
            dropvis[] = true
        else
            dropvis[] = false
        end
        return nothing
    end
    on(events(player.fig).mouseposition) do mp
        player.dragsource === nothing || moveghost(Point2f(mp))
    end
    # press on a row starts the drag; release over the timeline places the clip
    on(events(player.fig).mousebutton; priority = 90) do event
        event.button == Mouse.left || return Consume(false)
        mp = Point2f(events(player.fig).mouseposition[])
        if event.action == Mouse.press && player.dockopen[] === :media
            for ((btn, _, _), src) in zip(player.binrows, player.mediasources[])
                if mp in btn.layoutobservables.computedbbox[]
                    player.dragsource = src
                    moveghost(mp)
                    setstatus!(player, "drop $(basename(src.path)) on the timeline to place it")
                    return Consume(true)
                end
            end
        elseif event.action == Mouse.release && player.dragsource !== nothing
            src = player.dragsource
            player.dragsource = nothing
            dropvis[] = false
            tlscene = player.timeline.axis.scene
            if mp in tlscene.viewport[]
                t = Makie.mouseposition(tlscene)[1]
                placesource!(player, src, round(Int, t * player.sequence.framerate))
            else
                setstatus!(player, "drag cancelled — release over the timeline to place a clip")
            end
            return Consume(true)
        end
        return Consume(false)
    end
    return panel
end

"Register a source in the media bin (no-op when its path is already there)."
function registermedia!(player::Player, source::VideoSource)
    any(s -> s.path == source.path, player.mediasources[]) ||
        (player.mediasources[] = vcat(player.mediasources[], source))
    return nothing
end

"Open `path` and add it to the media bin (the import dialog's action)."
function importsource!(player::Player, path::AbstractString)
    source = try
        VideoSource(path)
    catch e
        setstatus!(player, "couldn't open $(basename(path)): $(sprint(showerror, e))")
        return nothing
    end
    registermedia!(player, source)
    opendock!(player, :media)
    setstatus!(player, "imported $(basename(path)) — drag it onto the timeline")
    return source
end

"""
Place `source` as a new clip starting at frame `at` (undoable). A drop that
would overlap an existing clip lands at the end of the timeline instead —
the status line says which happened.
"""
function placesource!(player::Player, source::VideoSource, at::Integer)
    seq = player.sequence
    if !isapprox(source.framerate, seq.framerate; atol = 0.01)
        setstatus!(player, "framerate $(source.framerate) doesn't match the sequence ($(seq.framerate))")
        return nothing
    end
    start = max(Int(at), 0)
    if any(c -> start < clipend(c) && start + source.nframes > c.start, seq.clips)
        start = seqlength(seq)
        setstatus!(player, "no room at the drop point — $(basename(source.path)) placed at the end")
    else
        setstatus!(player, "placed $(basename(source.path)) at $(timestring(start / seq.framerate))")
    end
    snapshot!(player)
    clip = Clip(source, 0, source.nframes, start, (0.0, 0.0, 1.0, 1.0))
    push!(seq.clips, clip)
    sort!(seq.clips, by = c -> c.start)
    limits!(player.timeline.axis, 0.0, seqduration(seq), 0.0, 1.0)  # reveal it
    refreshedit!(player)
    needsproxy(source; maxpixels = player.proxythreshold) && startproxy!(player, source)
    return clip
end

# -------------------------------------------------------------- export panel

"""
Export dock panel around [`exportvideo`](@ref): output path via the native
save dialog, container, codec, quality (crf), preset and audio. The Export
button in the controls row opens this panel; "Export video" starts the render
in the background with progress in the status line.
"""
function buildexportpanel!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1:2], "Export"; font = :bold, halign = :left)
    clips = player.sequence.clips
    path = Observable(isempty(clips) ? abspath("export.mp4") :
                      splitext(clips[1].source.path)[1] * "_export.mp4")
    Label(panel[2, 1:2], map(basename, path); halign = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = false)
    browsebtn = Button(panel[3, 1:2]; label = "Choose file…", tellwidth = false)
    Label(panel[4, 1], "Format"; halign = :left, fontsize = 12)
    fmtmenu = Menu(panel[4, 2]; options = [("mp4", ".mp4"), ("mkv", ".mkv"),
                                           ("mov", ".mov"), ("gif", ".gif")],
                   tellwidth = false)
    Label(panel[5, 1], "Codec"; halign = :left, fontsize = 12)
    codecmenu = Menu(panel[5, 2]; options = [("H.264", "libx264"), ("H.265", "libx265")],
                     tellwidth = false)
    Label(panel[6, 1], "Quality"; halign = :left, fontsize = 12)
    crfslider = Slider(panel[6, 2]; range = 38:-1:10, startvalue = 20, width = 120)
    Label(panel[7, 1], "Preset"; halign = :left, fontsize = 12)
    presetmenu = Menu(panel[7, 2]; options = ["fast", "medium", "slow"], tellwidth = false)
    Label(panel[8, 1], "Audio"; halign = :left, fontsize = 12)
    audiobox = Checkbox(panel[8, 2]; checked = true, halign = :left)
    # GIF-only options (fps + looping)
    Label(panel[9, 1], "GIF fps"; halign = :left, fontsize = 12)
    fpsslider = Slider(panel[9, 2]; range = 5:1:30, startvalue = 15, width = 120)
    Label(panel[10, 1], "GIF loop"; halign = :left, fontsize = 12)
    loopbox = Checkbox(panel[10, 2]; checked = true, halign = :left)
    gobtn = Button(panel[11, 1:2]; label = "Export video", tellwidth = false)
    # button label follows the format so GIF export is discoverable
    on(fmtmenu.selection; update = true) do fmt
        gobtn.label[] = fmt == ".gif" ? "Export GIF" : "Export video"
    end
    on(browsebtn.clicks) do _
        Threads.@spawn try   # native dialog blocks — keep the UI rendering
            p = Makie.save_file_dialogue()
            p === nothing || put!(player.uiqueue, () -> (path[] = String(p)))
        catch e
            setstatus!(player, "file dialog failed: $(sprint(showerror, e))")
        end
    end
    on(gobtn.clicks) do _
        if isempty(player.sequence.clips)
            setstatus!(player, "nothing to export — the timeline is empty")
            return
        end
        fmt = something(fmtmenu.selection[], ".mp4")
        out = splitext(path[])[1] * fmt
        codec = something(codecmenu.selection[], "libx264")
        opts = (crf = round(Int, crfslider.value[]),
                preset = something(presetmenu.selection[], "medium"))
        audio = audiobox.checked[]
        fps = round(Int, fpsslider.value[])
        loop = loopbox.checked[] ? 0 : -1   # 0 = loop forever, -1 = play once
        setstatus!(player, "exporting $(basename(out))…")
        prog = (d, t) -> d % 120 == 0 && setstatus!(player, "exporting $(round(Int, 100d / t))%")
        Threads.@spawn try
            if fmt == ".gif"
                exportgif(out, player.sequence; fps, loop, progress = prog)
            else
                exportvideo(out, player.sequence; codec_name = codec,
                            encoder_options = opts, audio, progress = prog)
            end
            setstatus!(player, "exported $out")
        catch e
            setstatus!(player, "export failed: $(sprint(showerror, e))")
            @error "export failed" exception = (e, catch_backtrace())
        end
    end
    merge!(player.fxwidgets, Dict{Symbol, Any}(:exportgo => gobtn, :exportpath => path))
    return panel
end

# ------------------------------------------------------------- effect panel

"Sliders editing the effect stack of the clip under the playhead."
function buildfxpanel!(player::Player, gridpos, uicolors)
    specs = [
        (:brightness, "Brightness", -0.5:0.01:0.5, 0.0),
        (:contrast, "Contrast", 0.5:0.01:2.0, 1.0),
        (:saturation, "Saturation", 0.0:0.01:2.0, 1.0),
        (:temperature, "Temperature", -1.0:0.02:1.0, 0.0),
        (:blur, "Blur", 0.0:0.1:8.0, 0.0),
        (:sharpen, "Sharpen", 0.0:0.05:2.0, 0.0),
    ]
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1:2], "Effects"; font = :bold, halign = :left)
    # which clip these controls edit (the clip under the playhead) — the panel
    # acts on it, so name it and its timeline position explicitly
    target = map(player.playhead) do n
        loc = locate(player.sequence, n)
        if loc === nothing
            "▸ no clip at the playhead"
        else
            c = loc[1]
            i = something(findfirst(x -> x === c, player.sequence.clips), 0)
            fps = player.sequence.framerate
            "▸ editing clip $i · $(basename(c.source.path)) " *
            "($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
        end
    end
    Label(panel[2, 1:2], target; halign = :left, fontsize = 11,
          color = uicolors.accent, tellwidth = false)
    for (row, (key, text, range, default)) in enumerate(specs)
        Label(panel[row + 2, 1], text; halign = :left, fontsize = 12)
        slider = Slider(panel[row + 2, 2]; range, startvalue = default, width = 120)
        player.fxsliders[key] = slider
        on(slider.value) do value
            player.fxsyncing[] || applyslider!(player, key, Float32(value))
        end
    end
    nrows = length(specs) + 2
    Label(panel[nrows + 1, 1:2], "Stabilize"; font = :bold, halign = :left)
    modemenu = Menu(panel[nrows + 2, 1:2];
                    options = [("Camera lock — like a tripod", :similarity),
                               ("Object lock — keep a subject still", :objectlock),
                               ("Tripod (affine) — legacy", :tripod),
                               ("Tripod + perspective — legacy", :perspective),
                               ("Smooth — keep camera moves", :smooth)],
                    tellwidth = false)
    analyzebtn = Button(panel[nrows + 3, 1:2]; label = "Stabilize clip", tellwidth = false)
    Label(panel[nrows + 4, 1:2], player.stabinfo; halign = :left, fontsize = 11,
          tellwidth = false)
    comparebtn = Button(panel[nrows + 5, 1:2]; label = "Hold to compare original",
                        tellwidth = false)
    removebtn = Button(panel[nrows + 6, 1:2]; label = "Remove stabilization",
                       tellwidth = false)
    colorbtn = Button(panel[nrows + 7, 1:2]; label = "Fix color flicker", tellwidth = false)
    merge!(player.fxwidgets, Dict{Symbol, Any}(
        :modemenu => modemenu, :analyze => analyzebtn, :compare => comparebtn,
        :remove => removebtn, :color => colorbtn))

    on(analyzebtn.clicks) do _
        mode = something(modemenu.selection[], :similarity)
        if mode === :objectlock
            armpick!(player)
        else
            analyzeat!(player, (clip; kwargs...) ->
                           analyzemotion!(clip; mode, backend = player.analysisbackend, kwargs...),
                       "motion stabilization")
        end
    end
    on(_ -> removestabilization!(player), removebtn.clicks)
    on(_ -> analyzeat!(player, analyzecolor!, "color stabilization"), colorbtn.clicks)

    # press-and-hold on the compare button shows the un-stabilized original.
    # High priority: the Button block consumes presses itself, which would
    # shadow this handler (caught by the interaction tests).
    on(events(player.fig).mousebutton; priority = 100) do event
        event.button == Mouse.left || return Consume(false)
        if event.action == Mouse.press &&
           Point2f(events(player.fig).mouseposition[]) in comparebtn.layoutobservables.computedbbox[]
            player.applytracks[] = false
            notify(player.playhead)
            return Consume(true)
        elseif event.action == Mouse.release && !player.applytracks[]
            player.applytracks[] = true
            notify(player.playhead)
        end
        return Consume(false)
    end
    return panel
end

"Right-click context modal on the timeline: clip actions + shortcut reference."
function wireclipmenu!(player::Player)
    modal = Modal(player.fig; title = "Clip actions", min_size = (300, 100))
    player.clipmodal = modal
    rcframe() = clamp(round(Int, player.rctime * player.sequence.framerate), 0,
                      max(seqlength(player.sequence) - 1, 0))
    actions = [
        ("Split here", "S", () -> (split!(player.sequence, rcframe()); refreshedit!(player))),
        ("Delete clip (ripple)", "X", () -> begin
            deleteclip!(player.sequence, rcframe())
            player.playhead[] = clamp(player.playhead[], 0, max(seqlength(player.sequence) - 1, 0))
            refreshedit!(player)
        end),
        ("Crop (drag on preview)", "C", () -> (player.cropmode[] = true)),
        ("Reset crop", "R", () -> resetcropat!(player, rcframe())),
        ("Stabilize clip", "", () -> analyzeat!(player, (c; kw...) -> analyzemotion!(c; backend = player.analysisbackend, kw...), "motion stabilization"; at = rcframe())),
        ("Remove stabilization", "", () -> removestabilization!(player; at = rcframe())),
        ("Fix color flicker", "", () -> analyzeat!(player, analyzecolor!, "color stabilization"; at = rcframe())),
    ]
    for (i, (text, key, action)) in enumerate(actions)
        btn = Button(modal[i, 1]; label = isempty(key) ? text : "$text   ·  $key",
                     tellwidth = false)
        on(btn.clicks) do _
            close!(modal)
            action()
        end
    end
    Label(modal[length(actions) + 1, 1],
          "Scrub: drag  ·  Move clip: Ctrl+drag  ·  Zoom: scroll\nPlay: Space  ·  Step: ←/→ (Shift ±10)";
          fontsize = 11, halign = :left, justification = :left)
    player.timeline.onrightclick = t -> begin
        player.rctime = t
        modal.title = "Clip @ " * timestring(t)
        open!(modal)
    end
    return nothing
end

function applyslider!(player::Player, key::Symbol, value::Float32)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    if time() - player.lastslidersnap > 1.5  # one undo entry per slider gesture
        snapshot!(player)
        player.lastslidersnap = time()
    end
    sliders = player.fxsliders
    if key in (:brightness, :contrast, :saturation, :temperature)
        seteffect!(clip, ColorEffect(
            brightness = sliders[:brightness].value[], contrast = sliders[:contrast].value[],
            saturation = sliders[:saturation].value[], temperature = sliders[:temperature].value[]))
    elseif key == :blur
        seteffect!(clip, BlurEffect(value))
    elseif key == :sharpen
        seteffect!(clip, SharpenEffect(1.0f0, value))
    end
    player.playing[] || notify(player.playhead)  # live re-present while paused
    return nothing
end

"Reflect `clip`'s effect stack in the sliders (when the playhead enters it)."
function syncsliders!(player::Player, clip::Clip)
    player.fxsyncing[] = true
    color = something(findeffect(clip, ColorEffect), ColorEffect()).adj
    blur = something(findeffect(clip, BlurEffect), BlurEffect(0.0f0))
    sharpen = something(findeffect(clip, SharpenEffect), SharpenEffect(1.0f0, 0.0f0))
    set_close_to!(player.fxsliders[:brightness], color.brightness)
    set_close_to!(player.fxsliders[:contrast], color.contrast)
    set_close_to!(player.fxsliders[:saturation], color.saturation)
    set_close_to!(player.fxsliders[:temperature], color.temperature)
    set_close_to!(player.fxsliders[:blur], blur.σ)
    set_close_to!(player.fxsliders[:sharpen], sharpen.amount)
    player.fxsyncing[] = false
    return nothing
end

function Base.close(player::Player)
    pause!(player)   # also stops the audio feed
    foreach(sp -> stop!(sp.worker), values(player.pools))
    stop!(player.timeline)
    close(player.statusqueue)
    close(player.uiqueue)
    player.screen === nothing || close(player.screen)
    player.screen = nothing
    return nothing
end
