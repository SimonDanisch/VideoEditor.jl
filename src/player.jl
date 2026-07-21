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
    analysisbackend::Any  # KA backend for analysis/GPU playback; set by auto-detect
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
    const playrate::Base.RefValue{Float64}  # JKL shuttle rate: 1.0 normal, <0 reverse, |·|>1 fast
    const kffocus::Observable{Symbol}    # param whose curve the keyframe lane shows
    const kflaneopen::Observable{Bool}   # keyframe curve lane visible
    const gpucache::Dict{Any, Any}       # source → device-resident decoded frames (pure-GPU playback)
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
                analysisbackend = nothing, gpupreview = nothing,
                audiopreview::Bool = true,
                proxyheight::Integer = 720, proxythreshold::Integer = 2_100_000)
    # GPU playback is the default: leaving both `analysisbackend` and `gpupreview`
    # unset auto-detects a video-capable Vulkan device (below) and, if present, runs
    # decode + effects on the GPU. Pass either to force the choice.
    autodetect = analysisbackend === nothing && gpupreview === nothing
    backend = analysisbackend === nothing ? KA.CPU() : analysisbackend
    wantgpu = gpupreview === true
    wantgpu && backend isa KA.CPU &&
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

    # `derive_colors` mixes surfaces only ~3–6% toward the contrast pole — weights
    # tuned for light mode that are nearly invisible on a dark canvas. Widen the steps
    # so the dark UI has real depth (void → panel → button → border read as distinct).
    bgc = RGBf(Makie.to_color(background)); wht = RGBf(1, 1, 1)
    uicolors = merge(Makie.derive_colors(; background, accent),
        (surface_subtle = Makie.lerp_oklab(bgc, wht, 0.05),
         surface        = Makie.lerp_oklab(bgc, wht, 0.11),
         border         = Makie.lerp_oklab(bgc, wht, 0.26)))
    player = Makie.with_theme(colors = Makie.Attributes(; uicolors...),
                              backgroundcolor = background,
                              textcolor = uicolors.text) do
        buildui(sequence, pools, Int(capacity), Int(proxyheight), Int(proxythreshold),
                frame, playhead, playing; background, uicolors, analysisbackend = backend)
    end
    player.screen = display(player.fig)
    wantgpu && (player.gpupreview = GPUPreview())
    audiopreview && (player.audio = AudioPreview())
    retrypresent(player, 0)
    for src in unique(c.source for c in sequence.clips)  # projects may be multi-source
        needsproxy(src; maxpixels = proxythreshold) && startproxy!(player, src)
        # with the GPU preview on, stream-decode each source on the GPU so playback
        # runs purely on the GPU (background; CPU decode until the stream is ready)
        wantgpu && Threads.@spawn preloadgpu!(player, src)
    end
    autodetect && Threads.@spawn autodetectgpu!(player)  # enable GPU playback if capable
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
    # Sidebar surfaces so the controls read as a designed panel, not text floating on
    # the void: a dark rail behind the toolbar, a lighter panel behind the dock slot
    # (both collapse with their columns). Created before their content → drawn behind.
    Box(fig[1, 1]; color = uicolors.surface_subtle, strokewidth = 0, tellwidth = false, tellheight = false)
    Box(fig[1, 2]; color = uicolors.surface_subtle, strokecolor = uicolors.border, strokewidth = 1,
        tellwidth = false, tellheight = false)
    # DataAspect keeps pixels square and letterboxes the crop inside the cell,
    # so we never show outside the crop (which would re-reveal the warp border
    # a stabilization crop hides). The wide dock-less cell gives the extra width.
    ax = Axis(fig[1, 3], aspect = DataAspect(), yreversed = true,
              backgroundcolor = background)
    hidedecorations!(ax)
    hidespines!(ax)
    deregister_interaction!(ax, :rectanglezoom)  # left-drag is the crop tool
    previewplot = image!(ax, frame; interpolate = true)

    # row 2: a full-width, collapsible keyframe curve lane, time-aligned with the
    # timeline directly below it (populated by buildkeyframelane!, shown with ◆)
    Box(fig[3, 1:3]; color = uicolors.surface_subtle, strokewidth = 0, tellwidth = false, tellheight = false)  # timeline zone
    timeline = Timeline(fig[3, 1:3], sequence, playhead, playing)
    rowsize!(fig.layout, 2, Makie.Fixed(0))
    rowsize!(fig.layout, 3, Makie.Fixed(96))

    # onboarding hint; replaced by the first real status update
    status = Observable("Space plays · S splits · right-click a clip for all actions")
    controls = GridLayout(fig[4, 1:3], tellwidth = false)
    Box(fig[4, 1:3]; color = uicolors.surface_subtle, strokewidth = 0, tellwidth = false, tellheight = false)  # footer bar
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
                    Observable(VideoSource[]), Any[], nothing, Observable(:none),
                    Ref(1.0), Observable(:opacity), Observable(false), Dict{Any, Any}())
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

    fxdock = dockpanel!(player, :effects; width = 340)
    buildfxpanel!(player, fxdock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[1, 1], "FX", :effects, uicolors)
    player.mediasources[] = unique([c.source for c in sequence.clips])
    mediadock = dockpanel!(player, :media)
    buildmediabin!(player, mediadock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[2, 1], "Bin", :media, uicolors)
    exportdock = dockpanel!(player, :export)
    buildexportpanel!(player, exportdock[1, 1], uicolors)
    toolbarbutton!(player, toolbar[3, 1], "Out", :export, uicolors)
    buildkeyframelane!(player, uicolors)   # full-width curve lane above the timeline
    buildkeyframeoverlay!(player)          # colored keyframe curves on the thumbnail track
    kflanebtn = Button(toolbar[4, 1]; label = "◆", width = 40, height = 40)
    on(_ -> (player.kflaneopen[] = !player.kflaneopen[]), kflanebtn.clicks)
    on(player.kflaneopen; update = true) do o
        kflanebtn.buttoncolor[] = o ? uicolors.accent : uicolors.surface
        kflanebtn.labelcolor[] = o ? uicolors.text_on_accent : uicolors.text
    end
    # tool strip: ✂ split and ▢ crop ARM (change the cursor + act where you
    # click/drag); ✕ ↶ ↷ are one-shot at the playhead
    splitbtn = Button(toolbar[5, 1]; label = "✂", width = 40, height = 40)
    cropbtn = Button(toolbar[6, 1]; label = "▢", width = 40, height = 40)
    on(_ -> armtool!(player, :split), splitbtn.clicks)
    on(_ -> armtool!(player, :crop), cropbtn.clicks)
    oneshots = [("✕", "Delete  (X)", () -> deleteat!(player)),
                ("↶", "Undo  (Ctrl+Z)", () -> isempty(player.undostack) ? setstatus!(player, "nothing to undo") :
                            (undo!(player); setstatus!(player, "undone (Ctrl+Z redoes with Shift)"))),
                ("↷", "Redo  (Ctrl+⇧+Z)", () -> isempty(player.redostack) ? setstatus!(player, "nothing to redo") :
                            (redo!(player); setstatus!(player, "redone")))]
    onebtns = Makie.Button[]
    for (row, (lbl, _tip, action)) in enumerate(oneshots)
        b = Button(toolbar[6 + row, 1]; label = lbl, width = 40, height = 40)
        on(_ -> action(), b.clicks)
        push!(onebtns, b)
    end
    buildkeyframelegend!(player, uicolors)   # modal: show/hide keyframe tracks
    buildeffecteditor!(player, uicolors)     # modal: add / edit any effect (ParamForm)
    buildstabmodal!(player, uicolors)        # modal: stabilization + loop controls
    # these modals' open buttons live in the effects panel (see buildfxpanel!), not
    # the toolbar — the toolbar must stay short enough to clear the full-width timeline
    # hover tooltips: hovering a toolbar button shows its name + shortcut to the
    # right of it (detected by mouse-vs-bbox; Makie Buttons have no hover attr).
    tiptargets = vcat([(splitbtn, "Blade  (S)"), (cropbtn, "Crop  (C)")],
                      [(onebtns[i], oneshots[i][2]) for i in eachindex(onebtns)])
    tip_txt = Observable(" "); tip_pos = Observable(Point2f(0, 0)); tip_vis = Observable(false)
    Makie.text!(fig.scene, tip_pos; text = tip_txt, visible = tip_vis, space = :pixel,
                align = (:left, :center), fontsize = 15, font = :bold, color = :white,
                strokecolor = (:black, 0.95), strokewidth = 2.5, overdraw = true)
    on(events(fig).mouseposition) do mp
        p = Point2f(mp); hit = nothing
        for (btn, label) in tiptargets
            bb = btn.layoutobservables.computedbbox[]
            if bb.origin[1] <= p[1] <= bb.origin[1] + bb.widths[1] &&
               bb.origin[2] <= p[2] <= bb.origin[2] + bb.widths[2]
                hit = (bb, label); break
            end
        end
        if hit === nothing
            tip_vis[] && (tip_vis[] = false)
        else
            bb, label = hit
            tip_pos[] = Point2f(bb.origin[1] + bb.widths[1] + 10, bb.origin[2] + bb.widths[2] / 2)
            tip_txt[] = label; tip_vis[] = true
        end
        return Consume(false)
    end
    # armed tool → cursor, crop mode, hint, button highlight (cropmode is a
    # Ref, not an Observable, so it's driven here and reset in finishcrop!)
    on(player.tool; update = true) do t
        player.cropmode[] = (t === :crop)
        setcursor!(player, t === :none ? :arrow : t === :split ? :scissor : :crosshair)
        splitbtn.buttoncolor[] = t === :split ? uicolors.accent : uicolors.surface
        cropbtn.buttoncolor[] = t === :crop ? uicolors.accent : uicolors.surface
        t === :split && setstatus!(player, "blade tool — click the timeline to cut (stays active; Esc or ✂ to put it away)")
        t === :crop && setstatus!(player, "crop tool — drag a rectangle on the preview (Esc to put it away)")
    end
    # split tool: the next timeline click cuts THERE (not at the playhead)
    on(events(fig).mousebutton; priority = 95) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        player.tool[] === :split || return Consume(false)
        mp = Point2f(events(fig).mouseposition[])
        tlscene = timeline.axis.scene
        mp in tlscene.viewport[] || return Consume(false)
        t = Makie.mouseposition(tlscene)[1]
        # NB: must NOT be named `frame` — that would rebind the shared preview
        # Observable this scope captures (used by image! + the scrub fallback)
        cutat = clamp(round(Int, t * sequence.framerate), 0, max(seqlength(sequence) - 1, 0))
        snapshot!(player)  # so Ctrl+Z / the undo tool can revert the cut
        split!(sequence, cutat)
        refreshedit!(player)
        # Persistent blade: stays armed so the mouse keeps cutting (DaVinci blade).
        # Esc or clicking ✂ again puts it away.
        setstatus!(player, "cut at $(timecode(sequence, cutat)) — blade still active (Esc to stop)")
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
        elseif clip !== nothing
            syncanimatedsliders!(player, clip)  # follow the curves while scrubbing
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
                            clip.crop == (0.0, 0.0, 1.0, 1.0) && isempty(clip.animations)
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
    # Trim affordance: hovering a clip edge shows a horizontal-resize cursor so it
    # reads as "drag to trim the in/out point" (not "move"). Only when no tool is
    # armed — an armed blade/crop owns the cursor. `edgeline` is the edge marker
    # the timeline sets on edge-hover.
    on(timeline.edgeline) do e
        player.tool[] === :none && setcursor!(player, isempty(e) ? :arrow : :hresize)
    end
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

"""
Preview a cross-dissolve at timeline frame `n`: fetch both clips' frames from
their rings, run each clip's tracks + effects, and blend `(1-p)·A + p·B`.
Best-effort — returns `false` (caller falls back to the plain single-clip path)
if either frame isn't buffered yet, or the two sources differ in size (mismatched
dissolves preview as the outgoing clip; export still blends them via warp).
"""
function showtransition!(player::Player, sample)
    left, srcA, right, srcB, p = sample
    spA = pool(player, left.source)
    spB = pool(player, right.source)
    (spA.source.width, spA.source.height) == (spB.source.width, spB.source.height) || return false
    settarget!(spA.worker, srcA)
    settarget!(spB.worker, srcB)
    ensureframesize!(player, spA.source)
    bufA = player.frame[]
    fetchframe!(bufA, spA.ring, srcA) || return false
    bufB = similar(bufA)
    fetchframe!(bufB, spB.ring, srcB) || return false
    lclip = effectiveclip(left, srcA)   # keyframed params on each side
    rclip = effectiveclip(right, srcB)
    if player.applytracks[]
        applymotiontrack!(bufA, player.fxtmp1, lclip, srcA)
        applycolortrack!(bufA, lclip, srcA)
        applymotiontrack!(bufB, player.fxtmp1, rclip, srcB)
        applycolortrack!(bufB, rclip, srcB)
    end
    applyeffects!(bufA, player.fxtmp1, player.fxtmp2, lclip)
    applyeffects!(bufB, player.fxtmp1, player.fxtmp2, rclip)
    blend!(bufA, bufA, bufB, p)
    notify(player.frame)
    applycrop!(player, lclip)  # both sides share framing in the common (split) case
    return true
end

"""
Composite the stack of clips covering timeline frame `n` (bottom track → top) into
the preview: each layer is decoded, its tracks + effects applied, its crop BAKED into
the shared canvas via `warp!`, then alpha-blended by its opacity (upper over lower).
Returns `false` (caller falls back to the single-clip path) if any layer isn't buffered
yet. CPU preview path — the GPU/export paths still show the top clip for now.
"""
function compositeframe!(player::Player, n::Integer, clips::Vector{Clip})
    ensureframesize!(player, clips[end].source)      # canvas at the top clip's resolution
    canvas = player.frame[]; W, H = size(canvas)
    warpbuf = similar(canvas)
    fill!(canvas, RGB{N0f8}(0, 0, 0))
    for clip in clips
        srcframe = clip.src_in + (n - clip.start)
        sp = pool(player, clip.source)
        settarget!(sp.worker, srcframe)
        clipbuf = RGBFrame(undef, clip.source.width, clip.source.height)
        deadline = time() + 1.0
        while !fetchframe!(clipbuf, sp.ring, srcframe)
            time() > deadline && return false            # not buffered yet → single-clip fallback
            sleep(0.004)
        end
        ec = effectiveclip(clip, srcframe)               # keyframed params at this frame
        st1 = similar(clipbuf); st2 = similar(clipbuf)
        if player.applytracks[]
            applymotiontrack!(clipbuf, st1, ec, srcframe)
            applycolortrack!(clipbuf, ec, srcframe)
        end
        for e in ec.effects                              # opacity is the layer alpha, not scale-to-black
            (e isa OpacityEffect || isneutral(e)) && continue
            applyeffect!(clipbuf, st1, st2, e)
        end
        KA.synchronize(KA.get_backend(clipbuf))
        warp!(warpbuf, clipbuf, ec.crop)                 # bake this layer's crop into canvas space
        α = Float32(clamp(paramvalue(clip, :opacity, srcframe), 0.0, 1.0))
        blend!(canvas, canvas, warpbuf, α)               # (1-α)·below + α·layer
    end
    notify(player.frame)
    player.lastcrop = (0.0, 0.0, 1.0, 1.0)
    coverlimits!(player, 0, W, H, 0)                     # show the full baked canvas
    return true
end

"Resolve and show timeline frame `n` if possible (gaps show black). Returns success."
function showframe!(player::Player, n::Integer)
    tr = transitionat(player.sequence, n)
    if tr !== nothing
        s = transitionsample(player.sequence, tr, n)
        s !== nothing && showtransition!(player, s) && return true
    end
    # multiple stacked tracks → composite the stack (on the GPU if every layer has a
    # stream, else on the CPU)
    if ntracks(player.sequence) > 1
        clips = clipsat(player.sequence, n)
        if length(clips) > 1
            gp = player.gpupreview
            if gp isa GPUPreview && !gp.failed && all(haskey(player.gpucache, c.source) for c in clips)
                presentgpucomposite!(player, clips, n) && return true
            end
            compositeframe!(player, n, clips) && return true
        end
    end
    loc = locate(player.sequence, n)
    if loc === nothing
        fill!(player.frame[], RGB{N0f8}(0, 0, 0))
        notify(player.frame)
        return true
    end
    clip, srcframe = loc
    # PURE-GPU path: a streaming GPU decoder feeds this source — Vulkan-Video decode
    # into a bounded VRAM ring + effects on device, no CPU decode, no upload.
    gp = player.gpupreview
    if gp isa GPUPreview && !gp.failed && haskey(player.gpucache, clip.source)
        stream = player.gpucache[clip.source]
        if 0 <= srcframe < nframes(stream)
            ensureframesize!(player, clip.source)
            eclip = effectiveclip(clip, srcframe)
            if presentgpu!(player, eclip, srcframe; stream = stream)
                applycrop!(player, eclip)
                return true
            end
        end
    end
    sp = pool(player, clip.source)
    target, protect = decodetarget(player, n, clip, srcframe)
    settarget!(sp.worker, target; protect)
    ensureframesize!(player, sp.source)  # proxy resolution when one is active
    if fetchframe!(player.frame[], sp.ring, srcframe)
        eclip = effectiveclip(clip, srcframe)  # keyframed params sampled at this frame
        gp = player.gpupreview
        if gp isa GPUPreview && !gp.failed && presentgpu!(player, eclip, srcframe)
            applycrop!(player, eclip)  # tracks/effects (incl. keyframes) ran on the GPU
            return true
        end
        if player.applytracks[]
            applymotiontrack!(player.frame[], player.fxtmp1, eclip, srcframe)
            applycolortrack!(player.frame[], eclip, srcframe)
        end
        applyeffects!(player.frame[], player.fxtmp1, player.fxtmp2, eclip)
        notify(player.frame)
        applycrop!(player, eclip)
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

function play!(player::Player; rate::Real = 1.0)
    last = max(seqlength(player.sequence) - 1, 0)
    # wrap at the boundary we're heading toward, so play/reverse from an end loops in
    rate >= 0 ? (player.playhead[] >= last && (player.playhead[] = 0)) :
                (player.playhead[] <= 0 && (player.playhead[] = last))
    player.playrate[] = rate
    player.playing[] && return player   # a running loop reads playrate live — just retarget
    player.playing[] = true
    rate == 1.0 && startaudio!(player)  # audio only tracks real-time forward play
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

"Jump the playhead to the sequence start (`n = 0`) or end. `seek!` pauses first."
function seek!(player::Player, n::Integer)
    pause!(player)
    player.playhead[] = clamp(n, 0, max(seqlength(player.sequence) - 1, 0))
    return player
end

"DaVinci ↑/↓: jump to the previous (`dir < 0`) or next (`dir > 0`) edit point —
the start/end boundary of any clip. Nothing happens past the outermost edit."
function jumpedit!(player::Player, dir::Integer)
    pause!(player)
    seq = player.sequence
    bounds = sort!(unique(vcat(Int[c.start for c in seq.clips],
                               Int[clipend(c) for c in seq.clips], 0)))
    cur = player.playhead[]
    target = dir > 0 ? findfirst(>(cur), bounds) : findlast(<(cur), bounds)
    target === nothing && return player   # already at the outermost edit
    player.playhead[] = clamp(bounds[target], 0, max(seqlength(seq) - 1, 0))
    return player
end

"JKL shuttle (`dir = +1` forward / `-1` reverse). Repeated taps in the same
direction ramp the rate 1×→2×→4×→8×; switching direction resets to 1×."
function shuttle!(player::Player, dir::Integer)
    r = player.playrate[]
    rate = (player.playing[] && sign(r) == dir) ? clamp(2r, -8.0, 8.0) : float(dir)
    play!(player; rate = rate)
    setstatus!(player, "shuttle $(rate > 0 ? "▶▶" : "◀◀") $(abs(round(Int, rate)))×  (K to stop)")
    return player
end

"Wall-clock paced playback: playhead follows elapsed time, dropping frames if needed."
function playloop(player::Player)
    fps = player.sequence.framerate
    base = player.playhead[]
    rate = player.playrate[]
    t0 = time_ns()
    lastset = base
    while player.playing[]
        # external scrub OR a shuttle-rate change (J/K/L) → rebase the clock
        if player.playhead[] != lastset || player.playrate[] != rate
            base = player.playhead[]
            rate = player.playrate[]
            t0 = time_ns()
        end
        n = base + floor(Int, (time_ns() - t0) / 1.0e9 * fps * rate)
        last = max(seqlength(player.sequence) - 1, 0)
        if (rate >= 0 && n >= seqlength(player.sequence)) || (rate < 0 && n < 0)
            player.playhead[] = rate < 0 ? 0 : last   # park at the end we reached
            pause!(player)
            break
        end
        n = clamp(n, 0, last)
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

"""
Toggle a cross-dissolve on the cut nearest the playhead: adds a ~0.6 s dissolve
if none is there, removes it otherwise. No-op unless the playhead is near a real
cut between two adjacent clips (split one first with `S`).
"""
function toggletransition!(player::Player)
    seq = player.sequence
    fps = seq.framerate
    cuts = sort!(unique(Int[c.start for c in seq.clips if c.start > 0]))
    isempty(cuts) && return setstatus!(player, "no cut here — split a clip first (S), then press T on the cut")
    at = cuts[argmin(abs.(cuts .- player.playhead[]))]
    if removetransition!(seq, at) !== nothing
        refreshedit!(player)
        return setstatus!(player, "removed transition at $(timecode(seq, at))")
    end
    t = addtransition!(seq, at; duration = round(Int, 0.6fps))
    t === nothing && return setstatus!(player, "can't add a dissolve there — need a real cut between two clips")
    refreshedit!(player)
    setstatus!(player, "cross-dissolve at $(timecode(seq, at)) · $(round(t.duration / fps, digits = 2))s  (T to remove)")
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

"""
    findlooptrim!(player; at, minseconds, maxseconds)

Find the most seamless loop inside the clip under `at` — matching its
STABILIZED, cropped content so the camera is locked and the match is on the
subject's motion (e.g. a bird returning to the same spot after a full cycle) —
and trim the whole timeline down to that loop, so it's ready to export as a
forever-looping GIF. The frame search runs off the UI thread; `maxseconds ≤ 0`
searches up to the clip length (picks the globally most seamless loop, which is
usually the full behavioural cycle, not a short snippet).
"""
function findlooptrim!(player::Player; at::Integer = player.playhead[],
                       minseconds::Real = 2.0, maxseconds::Real = 0.0)
    loc = locate(player.sequence, at)
    if loc === nothing
        setstatus!(player, "make loop: no clip under the playhead — move it onto a clip first")
        return nothing
    end
    clip = loc[1]
    fps = clip.source.framerate
    # cap below the full length so it can't return the whole clip; a small
    # length bias makes a full behavioural cycle win over a short sub-loop
    maxs = maxseconds > 0 ? Float64(maxseconds) : 0.9 * cliplength(clip) / fps
    setstatus!(player, "make loop: matching frames across $(cliplength(clip)) frames…")
    Threads.@spawn try
        a, b, score = findloop(clip; minseconds = 3.0, maxseconds = maxs, lengthbias = 0.0002,
                               progress = (d, t) -> setstatus!(player, "make loop: matching $d/$t"))
        put!(player.uiqueue, () -> begin
            snapshot!(player)
            newin = clip.src_in + a
            newout = clip.src_in + b
            clip.src_in = newin           # motiontrack is keyed by absolute source
            clip.src_out = newout         # frame, so it survives the trim
            clip.start = 0
            filter!(c -> c === clip, player.sequence.clips)   # timeline == the loop
            setstatus!(player, "trimmed to a $(round((b - a) / fps, digits = 1))s seamless loop " *
                               "(seam $(round(score, digits = 4))) — export as a looping GIF")
            refreshedit!(player)
        end)
    catch e
        setstatus!(player, "make loop failed: $(sprint(showerror, e))")
        @error "make loop failed" exception = (e, catch_backtrace())
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
    player.croprect[] = Point2f[]   # clear the in-progress rectangle; stay armed
    # Persistent crop tool: stays armed so you can re-drag to refine (Esc to put away).
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
        elseif event.key == Keyboard.l && ispress   # JKL shuttle: forward / faster
            shuttle!(player, 1)
        elseif event.key == Keyboard.j && ispress   # reverse / faster reverse
            shuttle!(player, -1)
        elseif event.key == Keyboard.k && ispress   # stop shuttle
            pause!(player); player.playrate[] = 1.0
        elseif event.key == Keyboard.up             # previous edit point
            jumpedit!(player, -1)
        elseif event.key == Keyboard.down           # next edit point
            jumpedit!(player, 1)
        elseif event.key == Keyboard.home && ispress
            seek!(player, 0)
        elseif event.key == Keyboard._end && ispress
            seek!(player, seqlength(player.sequence) - 1)
        elseif event.key == Keyboard.s && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            saveproject!(player)
        elseif event.key == Keyboard.s && ispress
            split!(player)
        elseif (event.key == Keyboard.x || event.key == Keyboard.delete) && ispress
            deleteat!(player)
        elseif event.key == Keyboard.c && ispress
            armtool!(player, :crop)
        elseif event.key == Keyboard.t && ispress
            toggletransition!(player)
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

# process-wide cache of cursors (immutable handles, shared). Custom cursors
# (:scissor) build once from an RGBA bitmap; the rest are GLFW standard cursors.
const CURSORS = Dict{Symbol, Any}()

"A 24×24 RGBA scissor cursor (dark blades + white halo so it reads on any
background), for the armed blade tool. Hotspot returned alongside."
function scissor_bitmap()
    S = 24
    img = fill((0x00, 0x00, 0x00, 0x00), S, S)   # (r,g,b,a); row = y, col = x
    ink = (0x18, 0x18, 0x18, 0xff); halo = (0xff, 0xff, 0xff, 0xff)
    put!(x, y, c) = (1 <= x <= S && 1 <= y <= S) && (img[y, x] = c)
    line!(x0, y0, x1, y1) = for t in range(0, 1; length = round(Int, hypot(x1 - x0, y1 - y0)) * 2 + 1)
        x = x0 + t * (x1 - x0); y = y0 + t * (y1 - y0)
        for dx in -1:1, dy in -1:1; put!(round(Int, x) + dx, round(Int, y) + dy, halo); end
        put!(round(Int, x), round(Int, y), ink)
    end
    ring!(cx, cy, r) = for a in range(0, 2pi; length = 48)
        put!(round(Int, cx + r * cos(a)), round(Int, cy + r * sin(a)), halo)
        put!(round(Int, cx + r * cos(a)) + 1, round(Int, cy + r * sin(a)), ink)
    end
    line!(15, 17, 12, 13); line!(12, 13, 8, 2)    # blade 1: handle → pivot → tip
    line!(9, 17, 12, 13);  line!(12, 13, 16, 2)   # blade 2 (crosses at the pivot)
    ring!(6, 19, 3); ring!(18, 19, 3)             # finger holes
    put!(12, 13, halo)                             # pivot rivet
    return img
end

"Set the window's mouse cursor (`:arrow`, `:crosshair`, `:hand`, `:hresize`, or
`:scissor`); no-op when headless."
function setcursor!(player::Player, shape::Symbol)
    player.screen === nothing && return nothing
    try  # GLFW cursor calls require the main thread + a real window
        GLFW = GLMakie.GLFW
        cur = get!(CURSORS, shape) do
            shape === :scissor  ? GLFW.CreateCursor(scissor_bitmap(), (12, 3)) :
            shape === :hresize  ? GLFW.CreateStandardCursor(GLFW.RESIZE_EW_CURSOR) :
            shape === :crosshair ? GLFW.CreateStandardCursor(GLFW.CROSSHAIR_CURSOR) :
            shape === :hand     ? GLFW.CreateStandardCursor(GLFW.POINTING_HAND_CURSOR) :
                                  GLFW.CreateStandardCursor(GLFW.ARROW_CURSOR)
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
    merge!(player.fxwidgets, Dict{Symbol, Any}(:exportgo => gobtn, :exportpath => path,
                                               :exportformat => fmtmenu))
    return panel
end

# ------------------------------------------------------------- keyframe editor

"""
The full-width keyframe curve lane, docked directly above the timeline and locked
to its time axis so a key sits over its frame. Shows the curve of the focused
parameter (`player.kffocus`, set by arming/touching a slider), with a value scale.
Click adds a key, drag a ◆ moves it (time ↔ x, value ↔ y), right-click deletes.
A main-figure `Axis`, so its plots render and its mouse coordinates behave normally.
"""
function buildkeyframelane!(player::Player, uicolors)
    fig = player.fig
    timeline = player.timeline
    seq = player.sequence
    fps = seq.framerate
    lane = Axis(fig[2, 1:3]; backgroundcolor = uicolors.surface, xgridvisible = false,
                ygridvisible = false, xticksvisible = false, yticksvisible = false,
                xticklabelsvisible = false, yticklabelsvisible = false,
                titlealign = :left, titlesize = 11, titlecolor = uicolors.accent, titlegap = 2)
    Makie.hidespines!(lane)
    foreach(i -> Makie.deregister_interaction!(lane, i), (:rectanglezoom, :scrollzoom, :dragpan))
    ylims!(lane, -0.06, 1.06)
    on(timeline.viewrange; update = true) do (x0, x1)
        Makie.xlims!(lane, x0, x1)
    end
    player.fxwidgets[:kflane] = lane
    lanevis = player.kflaneopen
    on(player.kflaneopen; update = true) do o
        rowsize!(fig.layout, 2, Makie.Fixed(o ? 150 : 0))
        lane.titlevisible = o   # don't leak the param title above the timeline when collapsed
    end
    # title = focused parameter; corner labels = its value scale (kept inside the
    # plot so the lane's left edge still lines up with the timeline below)
    numfmt(x) = (r = round(x; digits = 2); r == round(x) ? string(round(Int, r)) : string(r))
    vscale_txt = Observable(["", "", ""])
    text!(lane, [Point2f(0.004, 0.93), Point2f(0.004, 0.5), Point2f(0.004, 0.07)];
          text = vscale_txt, space = :relative, align = (:left, :center), fontsize = 9,
          color = uicolors.text_muted, visible = lanevis)
    on(player.kffocus; update = true) do key
        p = paramspec(key)
        lane.title[] = p.label
        lane.titlecolor[] = paramcolor(key)   # title matches the focused curve's color
        vscale_txt[] = [numfmt(p.hi), numfmt((p.lo + p.hi) / 2), numfmt(p.lo)]
    end

    curveline = Observable(Point2f[]); keypts = Observable(Point2f[])
    curvecolor = Observable{Any}(paramcolor(player.kffocus[]))         # focused param's color
    # one thin line per keyframed parameter (scalar color each → no per-vertex matching)
    MAXCURVES = length(PARAMPALETTE)
    poolpts = [Observable(Point2f[]) for _ in 1:MAXCURVES]
    poolcol = [Observable{Any}(uicolors.text) for _ in 1:MAXCURVES]
    activespan = Observable(Rect2f(0, 0, 0, 1))   # the editable clip's time region
    poly!(lane, activespan; color = (uicolors.accent, 0.07), visible = lanevis)
    hlines!(lane, [0.0, 0.5, 1.0]; color = (uicolors.text, 0.1), visible = lanevis)
    for i in 1:MAXCURVES
        lines!(lane, poolpts[i]; color = poolcol[i], linewidth = 1.5, visible = lanevis)
    end
    lines!(lane, curveline; color = curvecolor, linewidth = 2.5, visible = lanevis) # focused, bolded on top
    vlines!(lane, map(n -> n / fps, player.playhead); color = uicolors.text, linewidth = 1,
            visible = lanevis)
    kfscatter = scatter!(lane, keypts; marker = :diamond, markersize = 15, color = curvecolor,
                         strokecolor = uicolors.background, strokewidth = 1.5, visible = lanevis)
    translate!(kfscatter, 0, 0, 5)
    hint_vis = Observable(false)   # shown when the focused curve has no keys yet
    text!(lane, Point2f(0.5, 0.4); text = "click to add a keyframe   ·   drag a ◆ to move it   ·   right-click to delete",
          space = :relative, align = (:center, :center), fontsize = 11, color = uicolors.text_muted,
          visible = hint_vis)

    focusparam() = player.kffocus[]
    currentclip() = (loc = locate(seq, player.playhead[]); loc === nothing ? nothing : loc[1])
    keytl(clip, f) = (clip.start + (f - clip.src_in)) / fps   # source frame → timeline seconds
    samplecurve(clip, pp, cc, x0, x1) = map(0:80) do i
        s = x0 + (x1 - x0) * i / 80
        sf = clip.src_in + (round(Int, s * fps) - clip.start)
        v = cc === nothing ? pp.get(clip) : something(valueat(cc, sf), pp.get(clip))
        Point2f(s, paramnorm(pp, v))
    end
    function redraw()
        clip = currentclip()
        key = focusparam(); p = paramspec(key)
        if clip === nothing
            curveline[] = Point2f[]; keypts[] = Point2f[]
            foreach(pp -> isempty(pp[]) || (pp[] = Point2f[]), poolpts)
            hint_vis[] = false
            return
        end
        x0 = clip.start / fps; x1 = clipend(clip) / fps
        activespan[] = Rect2f(x0, -0.06, x1 - x0, 1.12)
        c = get(clip.animations, key, nothing)
        curvecolor[] = paramcolor(key)
        curveline[] = samplecurve(clip, p, c, x0, x1)
        keypts[] = c === nothing ? Point2f[] :
                   [Point2f(keytl(clip, k.frame), paramnorm(p, k.value)) for k in c.keys
                    if clip.src_in <= k.frame <= clip.src_out]
        # draw every keyframed parameter as its own colored curve (the focused one is bolded above)
        i = 0
        for (k2, c2) in clip.animations
            i += 1; i > MAXCURVES && break
            poolcol[i][] = paramcolor(k2)
            poolpts[i][] = samplecurve(clip, paramspec(k2), c2, x0, x1)
        end
        foreach(j -> isempty(poolpts[j][]) || (poolpts[j][] = Point2f[]), (i + 1):MAXCURVES)
        hint_vis[] = player.kflaneopen[] && isempty(keypts[]) && i == 0
        return
    end
    on(_ -> redraw(), player.playhead)
    on(_ -> redraw(), player.kffocus)
    on(_ -> redraw(), player.kflaneopen)

    getcurve!(clip, key) = get!(() -> AnimCurve(), clip.animations, key)
    repaint() = (redraw(); player.playing[] || notify(player.playhead))
    drag = Ref{Union{Nothing, AnimCurve}}(nothing)
    dragi = Ref(0)
    laneframe(clip, s) = clip.src_in + (clamp(round(Int, s * fps), clip.start, clipend(clip) - 1) - clip.start)
    function nearestkey(clip, key, s, v)
        c = get(clip.animations, key, nothing); c === nothing && return 0
        p = paramspec(key); x0, x1 = timeline.viewrange[]; span = max(x1 - x0, 1.0e-6)
        best = 0; bestd = 0.05
        for (i, k) in enumerate(c.keys)
            clip.src_in <= k.frame <= clip.src_out || continue
            d = hypot((keytl(clip, k.frame) - s) / span, paramnorm(p, k.value) - v)
            d < bestd && ((best, bestd) = (i, d))
        end
        return best
    end
    readout(clip, key, f, v) = setstatus!(player,
        "$(paramspec(key).label) = $(round(paramdenorm(paramspec(key), v), digits = 3)) @ $(timecode(seq, clip.start + (f - clip.src_in)))")
    on(events(fig).mousebutton; priority = 92) do event
        if event.action == Mouse.release && drag[] !== nothing
            drag[] = nothing; return Consume(true)
        end
        (player.kflaneopen[] && Makie.is_mouseinside(lane.scene)) || return Consume(false)
        clip = currentclip(); clip === nothing && return Consume(false)
        key = focusparam(); p = paramspec(key)
        pos = Makie.mouseposition(lane.scene); s = pos[1]; v = clamp(pos[2], 0.0, 1.0)
        clip.start / fps <= s <= clipend(clip) / fps || return Consume(false)  # only the active clip
        if event.button == Mouse.right && event.action == Mouse.press
            i = nearestkey(clip, key, s, v)
            i == 0 && return Consume(false)
            snapshot!(player); c = clip.animations[key]; deleteat!(c.keys, i)
            isempty(c) && delete!(clip.animations, key); repaint()
            return Consume(true)
        elseif event.button == Mouse.left && event.action == Mouse.press
            i = nearestkey(clip, key, s, v)
            snapshot!(player)
            if i == 0
                cur = getcurve!(clip, key); f = laneframe(clip, s)
                setkey!(cur, f, paramdenorm(p, v))
                drag[] = cur; dragi[] = something(findfirst(k -> k.frame == f, cur.keys), 1)
            else
                drag[] = clip.animations[key]; dragi[] = i
            end
            readout(clip, key, laneframe(clip, s), v); repaint()
            return Consume(true)
        end
        return Consume(false)
    end
    on(events(fig).mouseposition) do _
        cur = drag[]; cur === nothing && return
        clip = currentclip(); clip === nothing && return
        key = focusparam(); p = paramspec(key)
        pos = Makie.mouseposition(lane.scene)
        f = laneframe(clip, pos[1]); v = clamp(pos[2], 0.0, 1.0)
        movekey!(cur, dragi[], f, paramdenorm(p, v))
        dragi[] = something(findfirst(k -> k.frame == f, cur.keys), dragi[])
        readout(clip, key, f, v); repaint()
    end
    redraw()
    return lane
end

# ---------------------------------------------------- keyframe overlay (on thumbnails)

"""
Draw every clip's keyframed parameters as colored curves overlaid on its thumbnail
strip in the timeline. One STABLE `lines!` plot per parameter (across all clips,
NaN-separated) so its `.visible` is settable — the legend modal and the per-param
toggles flip the same plot. Colors come from [`paramcolor`](@ref). Returns the
`param => (plot, points)` registry (also stored in `player.fxwidgets[:kfoverlay]`).
"""
function buildkeyframeoverlay!(player::Player)
    ax = player.timeline.axis
    seq = player.sequence
    fps = seq.framerate
    # a clip's curve band = its track's row (matches the timeline), inset a little
    function clipband(clip)
        ntr = ntracks(seq); span = 0.96 / ntr; g = min(0.02, span * 0.15)
        lo = 0.02 + (clip.track - 1) * span + g; hi = 0.02 + clip.track * span - g
        inset = 0.12 * (hi - lo)
        return (lo + inset, hi - inset)
    end
    yat(clip, p, v) = (b = clipband(clip); b[1] + (b[2] - b[1]) * clamp(paramnorm(p, v), 0.0, 1.0))
    yval(clip, p, y) = (b = clipband(clip); paramdenorm(p, clamp((y - b[1]) / (b[2] - b[1]), 0.0, 1.0)))
    curveplots = Dict{Symbol, Any}()  # param => (plot, points-observable)
    # keyframe ◆ markers for the focused parameter on the clip under the playhead
    focuspts = Observable(Point2f[]); focuscol = Observable{Any}(paramcolor(player.kffocus[]))
    focussc = scatter!(ax, focuspts; marker = :diamond, markersize = 12, color = focuscol,
                       strokecolor = :white, strokewidth = 1.0)
    translate!(focussc, 0, 0, 6)
    function ensureplot!(key)
        get!(curveplots, key) do
            pts = Observable(Point2f[])
            pl = lines!(ax, pts; color = paramcolor(key), linewidth = 2.0)
            translate!(pl, 0, 0, 4)   # above the thumbnails, below the playhead line
            (pl, pts)
        end
    end
    function refresh()
        active = Set{Symbol}()
        for clip in seq.clips, k in keys(clip.animations)
            push!(active, k)
        end
        for key in active
            _, pts = ensureplot!(key)
            p = paramspec(key)
            segs = Point2f[]
            for clip in seq.clips
                c = get(clip.animations, key, nothing)
                c === nothing && continue
                x0 = clip.start / fps; x1 = clipend(clip) / fps
                for i in 0:60
                    s = x0 + (x1 - x0) * i / 60
                    sf = clip.src_in + (round(Int, s * fps) - clip.start)
                    v = something(valueat(c, sf), p.get(clip))
                    push!(segs, Point2f(s, yat(clip, p, v)))
                end
                push!(segs, Point2f(NaN, NaN))   # break between clips
            end
            pts[] = segs
        end
        for (key, (_, pts)) in curveplots       # empty curves for params no longer animated
            key in active || isempty(pts[]) || (pts[] = Point2f[])
        end
        # ◆ markers = the focused param's keys on the clip under the playhead
        key = player.kffocus[]
        loc = locate(seq, player.playhead[])
        if loc !== nothing && clipanimated(loc[1], key)
            clip = loc[1]; p = paramspec(key); c = clip.animations[key]
            focuscol[] = paramcolor(key)
            focuspts[] = [Point2f((clip.start + (k.frame - clip.src_in)) / fps, yat(clip, p, k.value))
                          for k in c.keys if clip.src_in <= k.frame <= clip.src_out]
        else
            focuspts[] = Point2f[]
        end
        return
    end
    on(_ -> refresh(), player.playhead)   # refreshes on edits too (they notify the playhead)
    on(_ -> refresh(), player.kffocus)
    refresh()

    # ---- editing on the overlay: drag a ◆, Alt-click to add, right-click to delete.
    # Conservative: only consumes near a marker (or with Alt), so scrub / clip-drag /
    # trim / the right-click menu are untouched everywhere else.
    tl = player.timeline
    dragref = Ref{Any}(nothing)                          # (curve, index, clip)
    focuscurve() = (loc = locate(seq, player.playhead[]);
                    loc === nothing ? nothing : get(loc[1].animations, player.kffocus[], nothing))
    function nearestmarker(t, y)                         # key index near (t,y), else 0
        pts = focuspts[]; isempty(pts) && return 0
        vp = ax.scene.viewport[]; (x0, x1) = tl.viewrange[]
        sx = (x1 - x0) / max(vp.widths[1], 1); sy = 1.0 / max(vp.widths[2], 1)
        best = 0; bestd = 14.0
        for (i, pt) in enumerate(pts)
            d = hypot((t - pt[1]) / sx, (y - pt[2]) / sy)
            d < bestd && ((best, bestd) = (i, d))
        end
        return best
    end
    on(events(ax.scene).mousebutton; priority = 20) do event
        is_mouseinside(ax.scene) || return Consume(false)
        t, y = mouseposition(ax.scene)
        if event.button == Mouse.left && event.action == Mouse.press
            if ispressed(ax.scene, Keyboard.left_alt | Keyboard.right_alt)   # Alt-click adds a key
                loc = locate(seq, timelineframe(tl, t)); loc === nothing && return Consume(false)
                clip = loc[1]; p = paramspec(player.kffocus[])
                snapshot!(player)
                setkey!(get!(() -> AnimCurve(), clip.animations, player.kffocus[]),
                        clip.src_in + (timelineframe(tl, t) - clip.start), yval(clip, p, y))
                notify(player.playhead); return Consume(true)
            end
            c = focuscurve(); c === nothing && return Consume(false)
            i = nearestmarker(t, y); i == 0 && return Consume(false)   # else fall through to scrub
            snapshot!(player); dragref[] = (c, i, locate(seq, player.playhead[])[1])
            return Consume(true)
        elseif event.button == Mouse.left && event.action == Mouse.release && dragref[] !== nothing
            dragref[] = nothing; return Consume(true)
        elseif event.button == Mouse.right && event.action == Mouse.press
            c = focuscurve(); c === nothing && return Consume(false)
            i = nearestmarker(t, y); i == 0 && return Consume(false)   # else the clip menu opens
            snapshot!(player); deleteat!(c.keys, i)
            isempty(c) && delete!(locate(seq, player.playhead[])[1].animations, player.kffocus[])
            notify(player.playhead); return Consume(true)
        end
        return Consume(false)
    end
    on(events(ax.scene).mouseposition; priority = 20) do _
        dragref[] === nothing && return Consume(false)
        c, i, clip = dragref[]
        t, y = mouseposition(ax.scene)
        p = paramspec(player.kffocus[])
        f = clamp(clip.src_in + (timelineframe(tl, t) - clip.start), clip.src_in, clip.src_out)
        movekey!(c, i, f, yval(clip, p, y))
        j = findfirst(k -> k.frame == f, c.keys); j === nothing || (dragref[] = (c, j, clip))
        notify(player.playhead); return Consume(true)
    end

    player.fxwidgets[:kfoverlay] = curveplots
    player.fxwidgets[:kfrefresh] = refresh
    return curveplots
end

"Register `modal` and open it exclusively — any other registered modal is closed first,
so modals never stack on top of each other."
function openmodal!(player::Player, modal)
    reg = get!(() -> Any[], player.fxwidgets, :modals)
    modal in reg || push!(reg, modal)
    for m in reg
        m === modal || (try; close!(m); catch; end)
    end
    open!(modal)
    return nothing
end

"""
A modal "legend" of every keyframed parameter — a Makie [`Legend`](@ref) connected to
the overlay's per-param curve plots, so its built-in interaction toggles track
visibility: left-click hide/show one, middle-click show-all/hide-all, right-click
toggle all (hidden entries shade out). Rebuilt each open from the current animations.
Call `player.fxwidgets[:kflegendopen]()` to show it.
"""
function buildkeyframelegend!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Keyframed tracks", min_size = (300, 240), halign = :left)
    curveplots = player.fxwidgets[:kfoverlay]
    content = Ref{Any}(nothing)
    function rebuild()
        content[] === nothing || (Makie.clear!(content[]); content[] = nothing)
        player.fxwidgets[:kfrefresh]()   # make sure every active param has a plot
        keys_active = [p.key for p in PARAMS
                       if haskey(curveplots, p.key) &&
                          any(clipanimated(cl, p.key) for cl in player.sequence.clips)]
        gl = GridLayout(modal[1, 1])
        content[] = gl
        if isempty(keys_active)
            Label(gl[1, 1], "No keyframed parameters yet.\nToggle ◆ on a parameter to animate it.";
                  fontsize = 12, halign = :left, tellwidth = false)
            return
        end
        plots = [curveplots[k][1] for k in keys_active]
        labels = [paramspec(k).label for k in keys_active]
        Legend(gl[1, 1], plots, labels; framevisible = false, valign = :top, halign = :left,
               titlevisible = false, rowgap = 3, labelcolor = uicolors.text)
        Label(gl[2, 1], "click: hide / show   ·   middle-click: all   ·   right-click: toggle all";
              fontsize = 10, halign = :left, color = uicolors.text_muted, tellwidth = false)
        return
    end
    player.fxwidgets[:kflegendopen] = () -> (rebuild(); openmodal!(player, modal))
    return modal
end

"""
Modal editor for ANY effect (built-in or plugin), driven by `EffectKind`. Add mode
(`:effectaddopen`) shows a kind menu and appends the chosen effect; edit mode
(`:effecteditopen(i)`) fixes the kind and seeds the form from effect `i`. A `ParamForm`
(auto-built from the kind's params) with the colored ◆ keyframe + ● visibility
accessory per row live-applies to the clip.
"""
function buildeffecteditor!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Effect", min_size = (340, 300), halign = :left)
    formref = Ref{Any}(nothing)
    editclip = Ref{Any}(nothing)
    editindex = Ref(0)                       # 0 = append a new effect; >0 = edit that index
    menu = Menu(modal[1, 1]; prompt = "Choose an effect…", tellwidth = false,
                options = [(k.label, k.name) for k in effectkinds()])
    function showform(kindname)
        formref[] === nothing || (Makie.clear!(formref[]); formref[] = nothing)
        clip = editclip[]; clip === nothing && return
        k = kindbyname(kindname); k === nothing && return
        defs = NamedTuple(pr.name => pr.default for pr in k.params)
        if editindex[] == 0                  # add: append with defaults, then edit it
            snapshot!(player); push!(clip.effects, k.make(defs))
            editindex[] = length(clip.effects); notify(player.playhead)
        elseif !(1 <= editindex[] <= length(clip.effects) && k.matches(clip.effects[editindex[]]))
            snapshot!(player); clip.effects[editindex[]] = k.make(defs); notify(player.playhead)  # kind changed
        end
        cur = k.read(clip.effects[editindex[]])
        if isempty(k.params)
            formref[] = Label(modal[2, 1], "$(k.label) — no parameters."; halign = :left, tellwidth = false)
            return
        end
        spec = NamedTuple(pr.name => (Float64(cur[pr.name]), Makie.Between(pr.min, pr.max)) for pr in k.params)
        accessory = (field, pos) -> begin
            key = k.kfkeys[findfirst(pr -> pr.name === field, k.params)]
            pcol = paramcolor(key)
            gl = GridLayout(pos)
            kf = Button(gl[1, 1]; label = "◆", width = 20, tellwidth = false, labelcolor = pcol)
            ey = Button(gl[1, 2]; label = "●", width = 20, tellwidth = false, labelcolor = pcol)
            on(_ -> armkeyframe!(player, key), kf.clicks)
            on(_ -> (ov = get(player.fxwidgets, :kfoverlay, nothing);
                     ov !== nothing && haskey(ov, key) && (ov[key][1].visible[] = !ov[key][1].visible[])), ey.clicks)
            gl
        end
        pf = Makie.ParamForm(modal[2, 1], spec, accessory; accessorywidth = 48, labelwidth = 96)
        formref[] = pf
        on(pf.graph[:values]) do vals        # live-apply to the edited effect
            clip = editclip[]; clip === nothing && return
            (1 <= editindex[] <= length(clip.effects)) || return
            clip.effects[editindex[]] = k.make(vals); notify(player.playhead)
        end
    end
    on(sel -> sel === nothing || showform(sel), menu.selection)
    player.fxwidgets[:effectaddopen] = () -> begin
        loc = locate(player.sequence, player.playhead[])
        editclip[] = loc === nothing ? nothing : loc[1]
        editindex[] = 0; modal.title = "Add effect"; menu.i_selected[] = 0
        openmodal!(player, modal)
    end
    player.fxwidgets[:effecteditopen] = (i::Integer) -> begin
        loc = locate(player.sequence, player.playhead[]); loc === nothing && return
        clip = loc[1]; (1 <= i <= length(clip.effects)) || return
        k = effectkindfor(clip.effects[i]); k === nothing && return
        editclip[] = clip; editindex[] = i; modal.title = k.label
        openmodal!(player, modal)
        mi = findfirst(o -> o[2] === k.name, menu.options[])
        mi === nothing || (menu.i_selected[] = 0; menu.i_selected[] = mi)   # force showform to re-seed
    end
    player.fxwidgets[:pickermenu] = menu
    player.fxwidgets[:pickerform] = formref
    return modal
end

# ------------------------------------------------------------- effect panel

"""
The effects dock: a compact, scrollable list of the clip's applied effects (the
stack). `+ Add effect` opens the picker modal; clicking an effect opens its editor
modal ([`buildeffecteditor!`](@ref)); `×` removes it. `Tracks…` opens the keyframe
legend, `Stabilize…` the stabilization modal — editing lives in modals, not here.
"""
function buildfxpanel!(player::Player, gridpos, uicolors)
    fxscroll = Subfigure(gridpos; scrollbar_thumb_color = uicolors.border)
    player.fxwidgets[:fxscroll] = fxscroll
    panel = GridLayout(fxscroll[1, 1]; valign = :top)
    Label(panel[1, 1:4], "Effects"; font = :bold, halign = :left)
    target = map(player.playhead) do n
        loc = locate(player.sequence, n)
        loc === nothing && return "▸ no clip at the playhead"
        c = loc[1]; i = something(findfirst(x -> x === c, player.sequence.clips), 0)
        fps = player.sequence.framerate
        "▸ clip $i · $(basename(c.source.path)) ($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
    end
    Label(panel[2, 1:4], target; halign = :left, fontsize = 11, color = uicolors.accent, tellwidth = false)
    addbtn = Button(panel[3, 1:2]; label = "+ Add effect", tellwidth = false)
    on(_ -> player.fxwidgets[:effectaddopen](), addbtn.clicks)
    tracksbtn = Button(panel[3, 3:4]; label = "Tracks…", tellwidth = false)
    on(_ -> player.fxwidgets[:kflegendopen](), tracksbtn.clicks)
    # expose the panel buttons (in a Subfigure → not in fig.content) for tests/demos
    merge!(player.fxwidgets, Dict{Symbol, Any}(:addeffect => addbtn, :tracksbtn => tracksbtn))

    # the applied-effects stack — rebuilt only when the clip or its effect list changes
    listref = Ref{Any}(nothing); lastsig = Ref{Any}(:init)
    effsig(clip) = clip === nothing ? nothing :
        (objectid(clip), Tuple(e isa PluginEffect ? e.name : nameof(typeof(e)) for e in clip.effects))
    function rebuildlist()
        loc = locate(player.sequence, player.playhead[])
        clip = loc === nothing ? nothing : loc[1]
        sig = effsig(clip)
        sig == lastsig[] && return
        lastsig[] = sig
        listref[] === nothing || Makie.clear!(listref[])
        gl = GridLayout(panel[4, 1:4]); listref[] = gl
        rows = Makie.Button[]   # exposed for tests/demos (Subfigure widgets aren't in fig.content)
        if clip === nothing || isempty(clip.effects)
            Label(gl[1, 1], clip === nothing ? "—" : "No effects yet — click “+ Add effect”.";
                  halign = :left, fontsize = 11, color = uicolors.text_muted, tellwidth = false)
        else
            for (i, e) in enumerate(clip.effects)
                k = effectkindfor(e)
                b = Button(gl[i, 1]; label = k === nothing ? string(nameof(typeof(e))) : k.label,
                           halign = :left, tellwidth = false)
                on(_ -> player.fxwidgets[:effecteditopen](i), b.clicks)
                push!(rows, b)
                rm = Button(gl[i, 2]; label = "×", width = 28, tellwidth = false)
                on(rm.clicks) do _
                    snapshot!(player); deleteat!(clip.effects, i); notify(player.playhead); rebuildlist()
                end
            end
        end
        player.fxwidgets[:effectrows] = rows
        length(gl.content) > 1 && rowgap!(gl, 6)
        return
    end
    on(_ -> rebuildlist(), player.playhead)
    rebuildlist()

    Label(panel[5, 1:2], "Stabilize"; font = :bold, halign = :left)
    stabbtn = Button(panel[5, 3:4]; label = "Stabilize…", tellwidth = false)
    on(_ -> player.fxwidgets[:stabmodalopen](), stabbtn.clicks)
    player.fxwidgets[:fxlistrefresh] = rebuildlist
    rowgap!(panel, 12)              # breathing room between the panel's sections
    rowgap!(panel, 4, 22.0)         # extra space above the Stabilize group (a new section)
    return panel
end

"""
Stabilization + loop controls in a modal (opened from the effects panel's Stabilize…
button): mode menu, Stabilize/Compare/Remove/Fix-flicker, and Make-seamless-loop.
"""
function buildstabmodal!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Stabilize", min_size = (320, 320), halign = :left)
    gl = GridLayout(modal[1, 1])
    modemenu = Menu(gl[1, 1]; options = [("Camera lock — like a tripod", :similarity),
                                         ("Object lock — keep a subject still", :objectlock),
                                         ("Tripod (affine) — legacy", :tripod),
                                         ("Tripod + perspective — legacy", :perspective),
                                         ("Smooth — keep camera moves", :smooth)], tellwidth = false)
    analyzebtn = Button(gl[2, 1]; label = "Stabilize clip", tellwidth = false)
    Label(gl[3, 1], player.stabinfo; halign = :left, fontsize = 11, tellwidth = false)
    comparebtn = Button(gl[4, 1]; label = "Hold to compare original", tellwidth = false)
    removebtn = Button(gl[5, 1]; label = "Remove stabilization", tellwidth = false)
    colorbtn = Button(gl[6, 1]; label = "Fix color flicker", tellwidth = false)
    Label(gl[7, 1], "Loop"; font = :bold, halign = :left)
    loopbtn = Button(gl[8, 1]; label = "Make seamless loop", tellwidth = false)
    merge!(player.fxwidgets, Dict{Symbol, Any}(:modemenu => modemenu, :analyze => analyzebtn,
        :compare => comparebtn, :remove => removebtn, :color => colorbtn, :loop => loopbtn))
    on(analyzebtn.clicks) do _
        mode = something(modemenu.selection[], :similarity)
        if mode === :objectlock
            close!(modal); armpick!(player)   # close so the pick click reaches the preview
        else
            analyzeat!(player, (clip; kwargs...) ->
                           analyzemotion!(clip; mode, backend = player.analysisbackend, kwargs...),
                       "motion stabilization")
        end
    end
    on(_ -> removestabilization!(player), removebtn.clicks)
    on(_ -> analyzeat!(player, analyzecolor!, "color stabilization"), colorbtn.clicks)
    on(_ -> (close!(modal); findlooptrim!(player)), loopbtn.clicks)
    on(events(player.fig).mousebutton; priority = 100) do event   # press-and-hold compare
        (modal.open[] && event.button == Mouse.left) || return Consume(false)
        if event.action == Mouse.press &&
           Point2f(events(player.fig).mouseposition[]) in comparebtn.layoutobservables.computedbbox[]
            player.applytracks[] = false; notify(player.playhead); return Consume(true)
        elseif event.action == Mouse.release && !player.applytracks[]
            player.applytracks[] = true; notify(player.playhead)
        end
        return Consume(false)
    end
    player.fxwidgets[:stabmodalopen] = () -> openmodal!(player, modal)
    return modal
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

"Whether param `key` is keyframed on `clip` (has at least one key)."
clipanimated(clip::Clip, key::Symbol) = haskey(clip.animations, key) && !isempty(clip.animations[key])

"Absolute source frame the playhead currently maps to within `clip`."
playheadframe(player::Player, clip::Clip) = clip.src_in + (player.playhead[] - clip.start)

"Effective value of `key` on `clip` at source frame `sf` — the animated curve if
keyframed, otherwise the static value."
paramvalue(clip::Clip, key::Symbol, sf::Integer) =
    clipanimated(clip, key) ? something(valueat(clip.animations[key], sf), paramspec(key).get(clip)) :
    paramspec(key).get(clip)

"""
A slider moved: if the parameter is keyframed on the clip, writes/updates a key
at the playhead (so animating is just scrub-and-adjust); otherwise sets the
static value. Registry-driven, so every `PARAMS` entry with a slider works.
"""
function applyslider!(player::Player, key::Symbol, value::Float32)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    if time() - player.lastslidersnap > 1.5  # one undo entry per slider gesture
        snapshot!(player)
        player.lastslidersnap = time()
    end
    if clipanimated(clip, key)
        setkey!(clip.animations[key], playheadframe(player, clip), Float64(value))
    else
        paramspec(key).set(clip, Float64(value))
    end
    player.playing[] || notify(player.playhead)  # live re-present + lane refresh while paused
    return nothing
end

"Reflect the clip's effective values (animated or static) in every slider."
function syncsliders!(player::Player, clip::Clip)
    player.fxsyncing[] = true
    sf = playheadframe(player, clip)
    for (key, slider) in player.fxsliders
        set_close_to!(slider, paramvalue(clip, key, sf))
    end
    player.fxsyncing[] = false
    return nothing
end

"While scrubbing within one clip, keep only the KEYFRAMED sliders tracking their
curve (cheap no-op when nothing is animated)."
function syncanimatedsliders!(player::Player, clip::Clip)
    isempty(clip.animations) && return nothing
    player.fxsyncing[] = true
    sf = playheadframe(player, clip)
    for (key, slider) in player.fxsliders
        clipanimated(clip, key) && set_close_to!(slider, paramvalue(clip, key, sf))
    end
    player.fxsyncing[] = false
    return nothing
end

"""
The ◆ button next to a slider: begin keyframing this parameter (first key = its
current value) if it isn't animated yet, otherwise just focus the lane on its
curve. Never destructive — clearing is the separate `clearkeyframes!` action, so
a stray click can't wipe your work.
"""
function armkeyframe!(player::Player, key::Symbol)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    p = paramspec(key)
    if !clipanimated(clip, key)
        snapshot!(player)
        setkey!(get!(() -> AnimCurve(), clip.animations, key), playheadframe(player, clip), p.get(clip))
        setstatus!(player, "$(p.label): keyframing on — scrub and move the slider to add keys")
    else
        setstatus!(player, "$(p.label): editing its keyframes on the curve below")
    end
    player.kffocus[] = key
    player.kflaneopen[] = true
    notify(player.playhead)
    return nothing
end

"Clear every keyframe of the focused parameter on the clip under the playhead
(the slider returns to its static value; undoable)."
function clearkeyframes!(player::Player)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    key = player.kffocus[]
    p = paramspec(key)
    clipanimated(clip, key) || return setstatus!(player, "$(p.label): no keyframes to clear")
    snapshot!(player)
    n = length(clip.animations[key].keys)
    delete!(clip.animations, key)
    syncsliders!(player, clip)   # slider drops back to the static value
    notify(player.playhead)
    setstatus!(player, "$(p.label): cleared $n keyframe$(n == 1 ? "" : "s") (Ctrl+Z to restore)")
    return nothing
end

function Base.close(player::Player)
    pause!(player)   # also stops the audio feed
    freegpucache!(player)
    foreach(sp -> stop!(sp.worker), values(player.pools))
    stop!(player.timeline)
    close(player.statusqueue)
    close(player.uiqueue)
    player.screen === nothing || close(player.screen)
    player.screen = nothing
    return nothing
end
