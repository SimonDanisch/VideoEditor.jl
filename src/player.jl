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
    composebuf::RGBFrame  # frames are composed HERE and published to `frame` as one
                          # copy of the finished image — GLMakie samples `frame[]`
                          # lazily at render time, so decoding/warping in place there
                          # flashes raw or half-processed frames on screen during play
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
    const kffocus::Observable{Symbol}    # param whose ◆ markers the timeline overlay shows
    const kfvisible::Observable{Bool}    # keyframe curves on the timeline visible
    const jobprogress::Threads.Atomic{Float64}  # running job fraction 0..1, NaN when idle
                                                # (written from worker threads, polled by the UI)
    const gpucache::Dict{Any, Any}       # source → device-resident decoded frames (pure-GPU playback)
    const gpubusy::Threads.Atomic{Int}   # long GPU jobs queued/running (stream warmup, analyses);
                                         # presents take the CPU lane while > 0 instead of stalling
                                         # behind them on the single-writer worker
    const cpuengine::FxEngine            # the CPU-tier render engine — the SAME effect
                                         # graph as the GPU path, on the KA CPU backend
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

"""
Run a long analysis/render `job` on the right executor: CPU backend → a thread,
GPU backend → the pinned GPU worker. On the GPU the sources' stream rings are
closed for the duration — their VRAM starves the analysis pool otherwise (flaky
pool-block OOM in `goodfeatures`) — and re-opened afterwards; playback falls
back to CPU decode in between and returns to the stream when it's ready.
"""
function runanalysis(job::Function, player::Player)
    if player.analysisbackend isa KA.CPU
        Threads.@spawn job()
        return nothing
    end
    streamed = collect(keys(player.gpucache))
    isempty(streamed) || freegpucache!(player)
    Threads.atomic_add!(player.gpubusy, 1)   # presents take the CPU lane until the job ends
    rungpu(player) do
        try
            job()
        finally
            Threads.atomic_sub!(player.gpubusy, 1)
            for src in streamed
                Threads.@spawn preloadgpu!(player, src)
            end
        end
    end
    return nothing
end

"Like [`runanalysis`](@ref) but synchronous: runs `job` on the right executor,
returns its result (callers that need the answer in-line, e.g. MCP `find_loop`)."
function runanalysissync(job::Function, player::Player)
    player.analysisbackend isa KA.CPU && return job()
    streamed = collect(keys(player.gpucache))
    isempty(streamed) || freegpucache!(player)
    Threads.atomic_add!(player.gpubusy, 1)
    try
        return rungpusync(job, player)
    finally
        Threads.atomic_sub!(player.gpubusy, 1)
        for src in streamed
            Threads.@spawn preloadgpu!(player, src)
        end
    end
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
    # an explicit GPU analysis backend implies GPU playback — `gpupreview = false` opts out
    wantgpu = gpupreview === true || (gpupreview === nothing && !(backend isa KA.CPU))
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
    # thumbnails decode on the GPU too — through the pinned worker (single-writer)
    wantgpu && setgpurun!(player.timeline, (f; long = false) -> rungpusync(f, player; long))
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

    # row 2: the timeline — keyframe curves overlay directly on the clips (ONE
    # keyframe editor, no separate lane); the row grows with the track count
    Box(fig[2, 1:3]; color = uicolors.surface_subtle, strokewidth = 0, tellwidth = false, tellheight = false)  # timeline zone
    timeline = Timeline(fig[2, 1:3], sequence, playhead, playing)
    rowsize!(fig.layout, 2, Makie.Fixed(112))

    # onboarding hint; replaced by the first real status update
    status = Observable("Space plays · S splits · Ctrl+P adds effects · Shift+click marks · right-drag pans")
    controls = GridLayout(fig[3, 1:3], tellwidth = false)
    Box(fig[3, 1:3]; color = uicolors.surface_subtle, strokewidth = 0, tellwidth = false, tellheight = false)  # footer bar
    playbtn = Button(controls[1, 1]; label = map(p -> p ? "Pause" : "Play", playing), width = 80)
    Label(controls[1, 2], map(n -> timecode(sequence, n), playhead); width = 220)
    exportbtn = Button(controls[1, 3]; label = "Export", width = 80)
    muted = Observable(false)
    mutebtn = Button(controls[1, 4]; label = map(m -> m ? "Muted" : "Sound", muted), width = 70)
    Label(controls[1, 5], status; width = 380, halign = :left, fontsize = 12,
          color = uicolors.text_muted)
    # long-running jobs (stabilize / export / loop search) animate a spinner + progress
    # bar here — every heavy computation has visible, moving feedback
    spintxt = Observable(" ")
    progfrac = Observable(0.0)
    progvis = Observable(false)
    Label(controls[1, 6], spintxt; width = 84, halign = :right, fontsize = 12,
          color = uicolors.accent)
    Box(controls[1, 7]; width = 140, height = 8, halign = :left, color = uicolors.surface,
        cornerradius = 4, strokewidth = 0, visible = progvis)
    Box(controls[1, 7]; width = map(f -> max(6.0, 140.0 * f), progfrac), height = 8,
        halign = :left, color = uicolors.accent, cornerradius = 4, strokewidth = 0,
        tellwidth = false, visible = progvis)

    player = Player(sequence, pools, capacity, proxyheight, proxythreshold,
                    timeline, frame, playhead,
                    playing, status, Channel{String}(32), Channel{Function}(32),
                    Observable(true), Observable("no analysis yet"), analysisbackend,
                    fig, ax, Ref(false),
                    Observable(Point2f[]),
                    similar(frame[]), Dict{Symbol, Slider}(),
                    Dict{Symbol, Any}(),
                    Ref(false), nothing, (0.0, 0.0, 1.0, 1.0), nothing, nothing, 0.0,
                    nothing, nothing, nothing, nothing, nothing,
                    Vector{Clip}[], Vector{Clip}[], 0.0, nothing, 0, 0,
                    Dict{Symbol, Any}(), Observable(:none),
                    Observable(VideoSource[]), Any[], nothing, Observable(:none),
                    Ref(1.0), Observable(:opacity), Observable(true),
                    Threads.Atomic{Float64}(NaN), Dict{Any, Any}(),
                    Threads.Atomic{Int}(0), FxEngine(KA.CPU()))
    @async for s in player.statusqueue  # main-thread consumer: threads → observable
        status[] = s
    end
    player.previewplot = previewplot
    @async for f in player.uiqueue      # main-thread consumer: threads → UI actions
        try
            Base.invokelatest(f)
        catch e
            # a throwing action must not kill the consumer — that silently
            # freezes every future cross-thread UI update
            @error "uiqueue action failed" exception = (e, catch_backtrace())
        end
    end
    @async begin  # job-progress poller: worker threads write the atomic, the UI animates
        spin = ('◐', '◓', '◑', '◒')
        i = 0
        while isopen(player.statusqueue)
            v = player.jobprogress[]
            if isnan(v)
                progvis[] && (progvis[] = false; spintxt[] = " ")
            else
                i += 1
                progvis[] || (progvis[] = true)
                f = clamp(v, 0.0, 1.0)
                progfrac[] = f
                spintxt[] = string(spin[mod1(i, 4)], " ", lpad(round(Int, 100f), 3), "%")
            end
            sleep(0.12)
        end
    end
    # the timeline row grows when clips stack on more tracks (edits notify the playhead)
    lastntr = Ref(1)
    on(playhead) do _
        ntr = ntracks(sequence)
        ntr == lastntr[] && return
        lastntr[] = ntr
        rowsize!(fig.layout, 2, Makie.Fixed(112 + 48 * (ntr - 1)))
    end
    on(exportbtn.clicks) do _
        toggledock!(player, :export)   # options live in the export dock panel
    end

    lines!(ax, player.croprect; color = :orangered, linewidth = 2)

    fxdock = dockpanel!(player, :effects; width = 360)
    buildfxpanel!(player, fxdock[1, 1], uicolors)
    fxbtn = toolbarbutton!(player, toolbar[1, 1], "FX", :effects, uicolors)
    player.mediasources[] = unique([c.source for c in sequence.clips])
    mediadock = dockpanel!(player, :media)
    buildmediabin!(player, mediadock[1, 1], uicolors)
    exportdock = dockpanel!(player, :export)
    buildexportpanel!(player, exportdock[1, 1], uicolors)
    # toolbar in three groups: the DOCKS (FX · Tools · Bin · Out), the armable
    # edit tools (✂ ▢), the one-shots (✕ ↶ ↷). The keyframe-curve toggle lives
    # in the Inspector now — it configures the overlay, it is not a tool.
    toolsdock = dockpanel!(player, :tools; width = 300)
    buildtoolspanel!(player, toolsdock[1, 1], uicolors)
    toolsbtn = toolbarbutton!(player, toolbar[2, 1], "⚒", :tools, uicolors)
    binbtn = toolbarbutton!(player, toolbar[3, 1], "Bin", :media, uicolors)
    outbtn = toolbarbutton!(player, toolbar[4, 1], "Out", :export, uicolors)
    buildkeyframeoverlay!(player)   # THE keyframe editor: curves + ◆ on the clips
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
    # hover tooltips: hovering a toolbar button shows its name + shortcut to the
    # right of it (detected by mouse-vs-bbox; Makie Buttons have no hover attr).
    tiptargets = vcat([(fxbtn, "Inspector — effects & stabilize"),
                       (toolsbtn, "Tools — loop finder, blends, …"),
                       (binbtn, "Media bin — import & drag clips"),
                       (outbtn, "Export"),
                       (splitbtn, "Blade  (S)"), (cropbtn, "Crop  (C)")],
                      [(onebtns[i], oneshots[i][2]) for i in eachindex(onebtns)])
    tip_txt = Observable(" "); tip_pos = Observable(Point2f(0, 0)); tip_vis = Observable(false)
    Makie.text!(fig.scene, tip_pos; text = tip_txt, visible = tip_vis, space = :pixel,
                align = (:left, :center), fontsize = 15, font = :bold, color = :white,
                strokecolor = (:black, 0.95), strokewidth = 2.5, overdraw = true)
    # The tool cursor is scoped to where the tool can ACT: the blade scissor only
    # over a cuttable clip on the timeline, the crop crosshair only over the
    # preview — hovering buttons or panels always shows a normal arrow.
    lastcursor = Ref(:arrow)
    function refreshcursor!()
        mp = Point2f(events(fig).mouseposition[])
        t = player.tool[]
        shape = :arrow
        if t === :split
            tlscene = timeline.axis.scene
            if mp in tlscene.viewport[]
                tt = Makie.mouseposition(tlscene)[1]
                shape = clipat(sequence, timelineframe(timeline, tt)) === nothing ?
                        :arrow : :scissor
            end
        elseif t === :crop
            shape = mp in ax.scene.viewport[] ? :crosshair : :arrow
        elseif !isempty(timeline.edgeline[])
            shape = :hresize   # trim handle under the cursor reads as "drag to trim"
        end
        shape === lastcursor[] || (lastcursor[] = shape; setcursor!(player, shape))
        return
    end
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
        refreshcursor!()
        return Consume(false)
    end
    # armed tool → cursor scope, crop mode, hint, button highlight (cropmode is a
    # Ref, not an Observable, so it's driven here and reset in finishcrop!)
    on(player.tool; update = true) do t
        player.cropmode[] = (t === :crop)
        refreshcursor!()
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
                        showcpuframe!(player)
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
    # trim-handle hover changes the cursor too (refreshcursor! reads edgeline)
    on(_ -> refreshcursor!(), timeline.edgeline)
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
    bufA = player.composebuf
    fetchframe!(bufA, spA.ring, srcA) || return false
    bufB = similar(bufA)
    fetchframe!(bufB, spB.ring, srcB) || return false
    lclip = effectiveclip(left, srcA)   # keyframed params on each side
    rclip = effectiveclip(right, srcB)
    if player.applytracks[]   # false = hold-to-compare: original frames, no effects
        render(player.cpuengine, bufA, lclip, Int(srcA)) do out
            copyto!(bufA, out)
        end
        render(player.cpuengine, bufB, rclip, Int(srcB)) do out
            copyto!(bufB, out)
        end
    end
    blend!(bufA, bufA, bufB, p)
    copyto!(player.frame[], bufA)   # publish the finished blend in one copy
    showcpuframe!(player)
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
    canvas = player.composebuf; W, H = size(canvas)
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
        ec = withoutopacity(effectiveclip(clip, srcframe))   # opacity = the layer alpha
        if player.applytracks[]                          # false = hold-to-compare bypass
            render(player.cpuengine, clipbuf, ec, Int(srcframe)) do out
                copyto!(clipbuf, out)
            end
        end
        warp!(warpbuf, clipbuf, ec.crop)                 # bake this layer's crop into canvas space
        KA.synchronize(KA.get_backend(warpbuf))
        α = Float32(clamp(paramvalue(clip, :opacity, srcframe), 0.0, 1.0))
        blend!(canvas, canvas, warpbuf, α)               # (1-α)·below + α·layer
    end
    copyto!(player.frame[], canvas)                      # publish the finished composite
    showcpuframe!(player)
    notify(player.frame)
    player.lastcrop = (0.0, 0.0, 1.0, 1.0)
    coverlimits!(player, 0, W, H, 0)                     # show the full baked canvas
    return true
end

"""
May a present use the GPU path right now? False while a long job (stream
warmup, analysis) owns the single-writer worker — presents then fall back to
the CPU tier for the duration instead of freezing the preview behind it.
"""
function gpuready(player::Player)
    gp = player.gpupreview
    return gp isa GPUPreview && !gp.failed && player.gpubusy[] == 0
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
            if gpuready(player) && all(haskey(player.gpucache, c.source) for c in clips)
                presentgpucomposite!(player, clips, n) && return true
            end
            compositeframe!(player, n, clips) && return true
        end
    end
    loc = locate(player.sequence, n)
    if loc === nothing
        fill!(player.composebuf, RGB{N0f8}(0, 0, 0))
        copyto!(player.frame[], player.composebuf)
        showcpuframe!(player)
        notify(player.frame)
        return true
    end
    clip, srcframe = loc
    # PURE-GPU path: a streaming GPU decoder feeds this source — Vulkan-Video decode
    # into a bounded VRAM ring + effects on device, no CPU decode, no upload.
    if gpuready(player) && haskey(player.gpucache, clip.source)
        stream = player.gpucache[clip.source]
        if 0 <= srcframe < nframes(stream)
            ensureframesize!(player, clip.source)
            eclip = effectiveclip(clip, srcframe)
            if presentgpu!(player, eclip, srcframe; stream = stream)
                applycrop!(player, eclip)
                # a scrub into an undecoded region shows the nearest decoded frame
                # NOW; returning "not presented yet" keeps the paused retry loop
                # refining in ~90 ms steps until the EXACT frame is on screen
                return hasframe(stream, srcframe)
            end
        end
    end
    sp = pool(player, clip.source)
    target, protect = decodetarget(player, n, clip, srcframe)
    settarget!(sp.worker, target; protect)
    ensureframesize!(player, sp.source)  # proxy resolution when one is active
    buf = player.composebuf
    if fetchframe!(buf, sp.ring, srcframe)
        eclip = effectiveclip(clip, srcframe)  # keyframed params sampled at this frame
        if gpuready(player)
            copyto!(player.frame[], buf)   # the GPU upload lane sources from `frame`
            if presentgpu!(player, eclip, srcframe)
                applycrop!(player, eclip)  # tracks/effects (incl. keyframes) ran on the GPU
                return true
            end
        end
        if player.applytracks[]   # false = hold-to-compare: the ORIGINAL frame
            render(player.cpuengine, buf, eclip, Int(srcframe)) do out
                copyto!(buf, out)
            end
        end
        copyto!(player.frame[], buf)       # publish the finished frame in one copy
        showcpuframe!(player)
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
    player.composebuf = similar(player.frame[])
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
    # buffer size is read LIVE each step, not captured: a proxy swap can land
    # mid-glide and change the preview resolution — a glide pinned to the old
    # size would stomp the swapped-in limits with stale coordinates
    setlimits(c) = begin
        W, H = size(player.frame[])
        coverlimits!(player, c[1] * W, (c[1] + c[3]) * W,
                     (c[2] + c[4]) * H, c[2] * H)  # y descending: axis is reversed
    end
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
        W, H = size(player.frame[])
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
    seq = player.sequence
    marked = player.timeline.selection[]
    if length(marked) > 1              # shift-click marks: delete the whole set
        clips = [seq.clips[i] for i in marked if 1 <= i <= length(seq.clips)]
        isempty(clips) && return nothing
        snapshot!(player)
        for c in clips                 # by identity — ripple keeps the rest current
            deleteclip!(seq, c)
        end
        player.timeline.selection[] = Int[]
        player.playhead[] = clamp(player.playhead[], 0, max(seqlength(seq) - 1, 0))
        setstatus!(player, "$(length(clips)) clips deleted — Ctrl+Z undoes")
        refreshedit!(player)
        return nothing
    end
    snapshot!(player)
    deleteclip!(seq, player.playhead[]) === nothing && return (pop!(player.undostack); nothing)
    player.playhead[] = clamp(player.playhead[], 0, max(seqlength(seq) - 1, 0))
    refreshedit!(player)
    return nothing
end

"Join the clip at `at` with the next same-source half (undo-able) — the inverse
of a blade cut. Politely refuses anything that isn't two halves of one cut."
function joinat!(player::Player; at::Integer = player.playhead[])
    snapshot!(player)
    if joinclips!(player.sequence, at) === nothing
        pop!(player.undostack)
        setstatus!(player, "nothing to join here — needs the two halves of one cut, side by side")
        return nothing
    end
    refreshedit!(player)
    setstatus!(player, "joined into one clip again (Ctrl+Z re-splits)")
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
    player.jobprogress[] = 0.0   # footer spinner + bar animate while the job runs
    # replacing a track must not stack auto-crops: the new analysis composes
    # its crop from the framing the OLD track's auto-crop replaced
    prevbase = clip.motiontrack isa MotionTrack ? clip.motiontrack.basecrop : nothing
    job = () -> try
        track = analyze!(clip; progress = (d, t) -> begin
                             player.jobprogress[] = d / max(t, 1)
                             setstatus!(player, "$what: $d/$t")
                         end)
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
    finally
        player.jobprogress[] = NaN
    end
    runanalysis(job, player)
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
    player.jobprogress[] = 0.0
    job = () -> try
        a, b, score = findloop(clip; minseconds = 3.0, maxseconds = maxs, lengthbias = 0.0002,
                               backend = player.analysisbackend,
                               progress = (d, t) -> begin
                                   player.jobprogress[] = d / max(t, 1)
                                   setstatus!(player, "make loop: matching $d/$t")
                               end)
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
    finally
        player.jobprogress[] = NaN
    end
    runanalysis(job, player)   # GPU decode must run on the worker (single-writer)
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
        elseif event.key == Keyboard.p && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            f = get(player.fxwidgets, :paletteopen, nothing)   # Ctrl+P: effect palette
            f === nothing || f()
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
            analyzeat!(player, (c; kw...) -> analyzecolor!(c; backend = player.analysisbackend, kw...),
                       "color stabilization")
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
            activetool(player) === nothing || deactivatetool!(player)
            player.clipmodal !== nothing && player.clipmodal.open[] &&
                close!(player.clipmodal)
            kfm = get(player.fxwidgets, :kfmenu, nothing)
            kfm !== nothing && kfm.open[] && close!(kfm)
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
Media-bin dock panel: each imported source is a card — an ASPECT-CORRECT
first-frame thumbnail beside its left-aligned name and duration. Drag a card
onto a timeline lane (a chip follows the cursor, the target lane ghosts) or
onto the "+ new track" strip. "Import clip…" opens a native file dialog.
"""
function buildmediabin!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1], "Media"; font = :bold, halign = :left, tellwidth = false)
    importbtn = Button(panel[2, 1]; label = "Import clip…", tellwidth = false,
                       width = Makie.Relative(1.0))
    Label(panel[3, 1], "drag a clip onto a lane · top strip = new track";
          fontsize = 11, halign = :left, color = uicolors.text_muted, tellwidth = false)
    rows = GridLayout(panel[4, 1])
    colsize!(panel, 1, Makie.Relative(1.0))
    on(importbtn.clicks) do _
        Threads.@spawn try   # the dialog blocks — keep the UI thread rendering
            path = Makie.choose_file_dialogue()
            path === nothing ||
                put!(player.uiqueue, () -> importsource!(player, String(path)))
        catch e
            setstatus!(player, "import failed: $(sprint(showerror, e))")
        end
    end
    boxh = 58
    boxfill = RGB{N0f8}(uicolors.surface_subtle)
    scene = player.dockpanels[:media].sf.scene   # Axis-in-Subfigure won't render;
                                                 # thumbnails draw on this scene
    on(player.mediasources; update = true) do sources
        for (name, box, tim, extras...) in player.binrows
            Makie.delete!(name); Makie.delete!(box); Makie.delete!(scene, tim)
            foreach(Makie.delete!, extras)
        end
        empty!(player.binrows)
        for (k, src) in enumerate(sources)
            # thumbnail box sized to the SOURCE aspect (no letterbox padding),
            # name + duration left-aligned beside it
            boxw = clamp(round(Int, boxh * src.width / src.height), 26, 104)
            box = Box(rows[k, 1]; width = boxw, height = boxh,
                      color = uicolors.surface_subtle, strokecolor = uicolors.border,
                      strokewidth = 1, cornerradius = 3, halign = :left)
            textgl = GridLayout(rows[k, 2]; valign = :center, halign = :left)
            name = Label(textgl[1, 1], basename(src.path); halign = :left,
                         fontsize = 13, tellwidth = false)
            secs = round(src.nframes / src.framerate; digits = 1)
            dur = Label(textgl[2, 1], "$(secs)s · $(src.width)×$(src.height)";
                        halign = :left, fontsize = 11, color = uicolors.text_muted,
                        tellwidth = false)
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
            push!(player.binrows, (name, box, tim, dur))
            Threads.@spawn try   # decode off the UI thread, publish when ready
                # dock scene is y-up; flip the frame's columns to show it upright
                t = reverse(fitbox(firstframethumb(src), boxw, boxh, boxfill), dims = 2)
                put!(player.uiqueue, () -> (thumb[] = t))
            catch
            end
        end
        if length(rows.content) > 1
            rowgap!(rows, 8)
            colgap!(rows, 12)   # text hugs the (aspect-sized) thumbnail column
        end
        return
    end
    # drag preview on the timeline: a translucent accent band showing where the
    # dropped clip would land — the cursor height picks the TARGET TRACK, and
    # hovering above the top lane shows the "+ new track" hint (stacked clips
    # composite, e.g. picture-in-picture with keyframed Opacity)
    dropghost = Observable(Rect2f(0, 0, 0, 0))
    dropvis = Observable(false)
    droptrack = Ref(1)
    dgp = poly!(player.timeline.axis, dropghost; color = (uicolors.accent, 0.3),
                strokecolor = uicolors.accent, strokewidth = 2, visible = dropvis)
    translate!(dgp, 0, 0, 20)
    # …and a chip that FOLLOWS THE CURSOR from the instant the drag starts at the
    # bin thumbnail — the drag is visible everywhere, not only over the timeline
    dragpos = Observable(Point2f(0, 0))
    draglabel = Observable(" ")
    dragvis = Observable(false)
    chip = Makie.scatter!(player.fig.scene, dragpos; marker = :rect,
                          markersize = (66, 40), color = (uicolors.accent, 0.35),
                          strokecolor = uicolors.accent, strokewidth = 1.5,
                          space = :pixel, visible = dragvis)
    translate!(chip, 0, 0, 500)
    chiptxt = Makie.text!(player.fig.scene, dragpos; text = draglabel, space = :pixel,
                          offset = (0, 26), align = (:center, :bottom), fontsize = 12,
                          color = :white, strokecolor = (:black, 0.85), strokewidth = 2.5,
                          visible = dragvis, overdraw = true)
    translate!(chiptxt, 0, 0, 501)
    function moveghost(mp)
        tl = player.timeline
        tlscene = tl.axis.scene
        dragpos[] = mp                  # the in-hand chip tracks the cursor everywhere
        if mp in tlscene.viewport[]
            pos = Makie.mouseposition(tlscene)
            t = pos[1]
            seq = player.sequence
            ntr = ntracks(seq)
            nfr = player.dragsource.nframes
            # target the hovered lane; if that spot is taken, the ghost snaps to
            # the first free lane above (stacking), never silently to the end
            track = freetrack(seq, round(Int, t * seq.framerate), nfr, trackat(pos[2], ntr))
            droptrack[] = track
            dur = nfr / seq.framerate
            n2 = max(ntr, track)
            g = min(0.02, trackspan(n2) * 0.15)
            lo, hi = trackband(track, n2)
            lo += g; hi -= g
            dropghost[] = Rect2f(t, lo, dur, hi - lo)
            dropvis[] = true
            if track > ntr
                tl.newtrackpos[] = Point2f(t + dur / 2, (lo + hi) / 2)
                tl.newtrackplot.visible = true
            else
                tl.newtrackplot.visible = false
            end
        else
            dropvis[] = false
            tl.newtrackplot.visible = false
        end
        return nothing
    end
    on(events(player.fig).mouseposition) do mp
        player.dragsource === nothing || moveghost(Point2f(mp))
    end
    # press on a row (its NAME or its THUMBNAIL — users grab the picture) starts
    # the drag: hand cursor + the new-track zone lights up; release over the
    # timeline places the clip on the ghosted lane
    on(events(player.fig).mousebutton; priority = 90) do event
        event.button == Mouse.left || return Consume(false)
        mp = Point2f(events(player.fig).mouseposition[])
        if event.action == Mouse.press && player.dockopen[] === :media
            for ((btn, box, _), src) in zip(player.binrows, player.mediasources[])
                if mp in btn.layoutobservables.computedbbox[] ||
                   mp in box.layoutobservables.computedbbox[]
                    player.dragsource = src
                    player.timeline.dragactive[] = true
                    setcursor!(player, :hand)
                    draglabel[] = basename(src.path)
                    dragvis[] = true
                    moveghost(mp)
                    setstatus!(player, "drop $(basename(src.path)) on a timeline lane — or on “+ new track” above them")
                    return Consume(true)
                end
            end
        elseif event.action == Mouse.release && player.dragsource !== nothing
            src = player.dragsource
            player.dragsource = nothing
            dropvis[] = false
            dragvis[] = false
            player.timeline.dragactive[] = false
            player.timeline.newtrackplot.visible = false
            setcursor!(player, :arrow)
            tlscene = player.timeline.axis.scene
            if mp in tlscene.viewport[]
                t = Makie.mouseposition(tlscene)[1]
                placesource!(player, src, round(Int, t * player.sequence.framerate);
                             track = droptrack[])
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
Place `source` as a new clip starting at frame `at` on `track` (undoable) —
dropping above the top lane creates a new track, whose clips composite over the
ones below (multi-track). A drop that would overlap a clip ON THE SAME track
lands at the end of the timeline instead — the status line says which happened.
"""
function placesource!(player::Player, source::VideoSource, at::Integer; track::Integer = 1)
    seq = player.sequence
    if !isapprox(source.framerate, seq.framerate; atol = 0.01)
        setstatus!(player, "framerate $(source.framerate) doesn't match the sequence ($(seq.framerate))")
        return nothing
    end
    start = max(Int(at), 0)
    # if the wanted lane is occupied there, stack on the first free lane above —
    # the drop always lands where the drag ghost showed it
    track = freetrack(seq, start, Int(source.nframes), clamp(Int(track), 1, ntracks(seq) + 1))
    if track > ntracks(seq) && ntracks(seq) > 0 && !isempty(seq.clips)
        setstatus!(player, "placed $(basename(source.path)) on NEW track V$track — it composites over the tracks below")
    else
        setstatus!(player, "placed $(basename(source.path)) at $(timestring(start / seq.framerate))" *
                           (track > 1 ? " on track V$track" : ""))
    end
    snapshot!(player)
    clip = Clip(source, 0, source.nframes, start, (0.0, 0.0, 1.0, 1.0))
    clip.track = track
    push!(seq.clips, clip)
    sort!(seq.clips, by = c -> (c.track, c.start))
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
    panel = GridLayout(gridpos; tellheight = false, valign = :top, halign = :left)
    Label(panel[1, 1:2], "Export"; font = :bold, halign = :left, tellwidth = false)
    clips = player.sequence.clips
    path = Observable(isempty(clips) ? abspath("export.mp4") :
                      splitext(clips[1].source.path)[1] * "_export.mp4")
    Label(panel[2, 1:2], map(basename, path); halign = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = false)
    browsebtn = Button(panel[3, 1:2]; label = "Choose file…", tellwidth = false,
                       width = Makie.Relative(1.0))
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
    gobtn = Button(panel[11, 1:2]; label = "Export video", tellwidth = false,
                   width = Makie.Relative(1.0))
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
        player.jobprogress[] = 0.0
        prog = (d, t) -> begin
            player.jobprogress[] = d / max(t, 1)
            d % 120 == 0 && setstatus!(player, "exporting $(round(Int, 100d / t))%")
        end
        backend = player.analysisbackend   # GPU backend → effects/warp render on device
        runanalysis(player) do
            try
                if fmt == ".gif"
                    exportgif(out, player.sequence; fps, loop, backend, progress = prog)
                else
                    exportvideo(out, player.sequence; codec_name = codec,
                                encoder_options = opts, audio, backend, progress = prog)
                end
                setstatus!(player, "exported $out")
            catch e
                setstatus!(player, "export failed: $(sprint(showerror, e))")
                @error "export failed" exception = (e, catch_backtrace())
            finally
                player.jobprogress[] = NaN
            end
        end
    end
    merge!(player.fxwidgets, Dict{Symbol, Any}(:exportgo => gobtn, :exportpath => path,
                                               :exportformat => fmtmenu))
    return panel
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
        ntr = ntracks(seq)
        g = min(0.02, trackspan(ntr) * 0.15)
        lo, hi = trackband(clip.track, ntr)
        lo += g; hi -= g
        inset = 0.12 * (hi - lo)
        return (lo + inset, hi - inset)
    end
    yat(clip, p, v) = (b = clipband(clip); b[1] + (b[2] - b[1]) * clamp(paramnorm(p, v), 0.0, 1.0))
    yval(clip, p, y) = (b = clipband(clip); paramdenorm(p, clamp((y - b[1]) / (b[2] - b[1]), 0.0, 1.0)))
    curveplots = Dict{Symbol, Any}()  # param => (plot, points-observable)
    # ◆ markers for EVERY animated parameter on the clip under the playhead — what
    # you click is what you edit (the focused param's markers just draw bigger)
    markpts = Observable(Point2f[])           # shared hit-test list (not rendered)
    markmeta = Tuple{Symbol, Int, Clip}[]     # per marker: (param, key index, clip)
    # the ease mode shows in the marker SHAPE: ◆ linear · ● smooth · ■ hold. One
    # scatter per shape (a scatter's marker input type-locks scalar-vs-vector and
    # mixed sdf shape classes don't batch); markpts/markmeta stay the shared
    # hit-test list, the per-shape plots only render.
    markrender = Dict(shape => (pts = Observable(Point2f[]), cols = Observable(RGBAf[]),
                                sizes = Observable(Float64[]))
                      for shape in (:diamond, :circle, :rect))
    for (shape, r) in markrender
        sc = scatter!(ax, r.pts; marker = shape, markersize = r.sizes,
                      color = r.cols, strokecolor = :white, strokewidth = 1.0,
                      visible = player.kfvisible)
        translate!(sc, 0, 0, 6)
    end
    function ensureplot!(key)
        get!(curveplots, key) do
            pts = Observable(Point2f[])
            halo = lines!(ax, pts; color = (:black, 0.55), linewidth = 4.0,
                          visible = player.kfvisible)  # dark under-halo: keeps the
            translate!(halo, 0, 0, 3)                  # curve readable on busy thumbs
            pl = lines!(ax, pts; color = paramcolor(key), linewidth = 2.0,
                        visible = player.kfvisible)   # toolbar ◆ shows/hides all curves
            translate!(pl, 0, 0, 4)   # above the thumbnails, below the playhead line
            (pl, pts, halo)
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
        # ◆ markers for every animated param on EVERY clip under the playhead —
        # multi-track: each lane's clip gets its own editable markers
        empty!(markmeta)
        pts = Point2f[]
        render = Dict(s => (Point2f[], RGBAf[], Float64[]) for s in keys(markrender))
        ph = player.playhead[]
        for clip in seq.clips
            clip.start <= ph < clipend(clip) || continue
            for (key, c) in clip.animations
                isempty(c) && continue
                p = paramspec(key)
                focused = key === player.kffocus[]
                col = RGBAf(Makie.to_color(paramcolor(key)))
                for (i, k) in enumerate(c.keys)
                    clip.src_in <= k.frame <= clip.src_out || continue
                    pt = Point2f((clip.start + (k.frame - clip.src_in)) / fps,
                                 yat(clip, p, k.value))
                    push!(pts, pt)
                    push!(markmeta, (key, i, clip))
                    ease = keyease(c, k)
                    rp, rc, rs = render[ease === :hold ? :rect :
                                        ease === :smooth ? :circle : :diamond]
                    push!(rp, pt); push!(rc, col); push!(rs, focused ? 13.0 : 9.0)
                end
            end
        end
        markpts[] = pts
        for (shape, r) in markrender
            rp, rc, rs = render[shape]
            r.cols[] = rc; r.sizes[] = rs; r.pts[] = rp
        end
        return
    end
    on(_ -> refresh(), player.playhead)   # refreshes on edits too (they notify the playhead)
    on(_ -> refresh(), player.kffocus)
    refresh()

    # ---- editing on the overlay: drag a ◆ (snaps to the playhead, live readout),
    # Alt-click ON a curve to add, Ctrl-click a ◆ to delete, right-click a ◆ for
    # the keyframe menu (delete · ease · clear). PARAM-AWARE: the marker/curve you
    # actually hit picks the parameter (and takes the focus with it) — never a
    # hidden "currently focused" one. Everything is gated on `kfvisible`: hidden
    # curves are INERT, so they can't hijack a scrub. Conservative: only consumes
    # near a marker (or Alt near a curve), so scrub / clip-drag / trim / the
    # right-click menu are untouched everywhere else.
    tl = player.timeline
    dragref = Ref{Any}(nothing)                          # (curve, index, clip, paramspec)
    dragtippos = Observable(Point2f(0, 0))               # value/time readout while dragging
    dragtiptext = Observable("")
    dragtip = text!(ax, dragtippos; text = dragtiptext, visible = false,
                    fontsize = 12, font = :bold, color = :white,
                    strokecolor = (:black, 0.8), strokewidth = 2,
                    offset = (10, 10), align = (:left, :bottom))
    translate!(dragtip, 0, 0, 7)
    function nearestmarker(t, y)                         # markmeta index near (t,y), else 0
        pts = markpts[]; isempty(pts) && return 0
        vp = ax.scene.viewport[]; (x0, x1) = tl.viewrange[]
        sx = (x1 - x0) / max(vp.widths[1], 1); sy = 1.0 / max(vp.widths[2], 1)
        best = 0; bestd = 14.0
        for (i, pt) in enumerate(pts)
            d = hypot((t - pt[1]) / sx, (y - pt[2]) / sy)
            d < bestd && ((best, bestd) = (i, d))
        end
        return best
    end
    # the animated param whose curve passes within `maxpx` of (t, y) on the clicked
    # clip — Alt-adding requires actually AIMING at a curve, like Premiere's pen
    function nearestcurve(clip, sf, y; maxpx = 22.0)
        vp = ax.scene.viewport[]
        best = nothing; bestd = maxpx / max(vp.widths[2], 1)
        for (key, c) in clip.animations
            isempty(c) && continue
            p = paramspec(key)
            d = abs(y - yat(clip, p, something(valueat(c, sf), p.get(clip))))
            d < bestd && ((best, bestd) = (key, d))
        end
        return best
    end
    function deletekey!(clip, key, kidx)
        c = clip.animations[key]
        snapshot!(player); deleteat!(c.keys, kidx)
        isempty(c) && (delete!(clip.animations, key);
                       setstatus!(player, "$(paramspec(key).label): last keyframe removed — back to a static value"))
        notify(player.playhead)
        return
    end
    # right-click ◆ → keyframe menu at the cursor. PER-KEY temporal interpolation,
    # Premiere-style: each key is a linear corner, a smooth (ease in & out) key,
    # or a hold (freeze until the next key).
    kfmenu = Modal(player.fig; title = "Keyframe", min_size = (200, 10),
                   backdrop_color = (:black, 0.15))
    kfmenuctx = Ref{Any}(nothing)                        # (clip, key, kidx)
    easelabel = Observable("Ease in & out")
    holdlabel = Observable("Hold until the next key")
    # per-key edits first bake the legacy curve-wide :smooth into the keys, so
    # un-easing ONE key can actually take effect
    function retoggle(mode)
        clip, key, kidx = kfmenuctx[]
        c = clip.animations[key]
        snapshot!(player)
        materializeease!(c)
        cur = c.keys[kidx].ease
        new = cur === mode ? :linear : mode
        setease!(c, kidx, new)
        setstatus!(player, "$(paramspec(key).label): keyframe is now " *
                           (new === :smooth ? "eased (in & out)" :
                            new === :hold ? "held until the next key" : "linear"))
        notify(player.playhead)
    end
    for (row, (lbl, action)) in enumerate([
        ("Delete keyframe", () -> begin
            clip, key, kidx = kfmenuctx[]
            deletekey!(clip, key, kidx)
        end),
        (easelabel, () -> retoggle(:smooth)),
        (holdlabel, () -> retoggle(:hold)),
        ("Clear all keys of this parameter", () -> begin
            clip, key, _ = kfmenuctx[]
            snapshot!(player)
            n = length(clip.animations[key].keys)
            delete!(clip.animations, key)
            syncsliders!(player, clip)
            setstatus!(player, "$(paramspec(key).label): cleared $n keyframe$(n == 1 ? "" : "s") (Ctrl+Z to restore)")
            notify(player.playhead)
        end)])
        btn = Button(kfmenu[row, 1]; label = lbl, tellwidth = false)
        on(btn.clicks) do _
            close!(kfmenu)
            kfmenuctx[] === nothing || action()
        end
        player.fxwidgets[Symbol(:kfmenubtn, row)] = btn   # delete · ease · hold · clear
    end
    player.fxwidgets[:kfmenu] = kfmenu
    on(events(ax.scene).mousebutton; priority = 20) do event
        player.kfvisible[] || return Consume(false)      # hidden curves are inert
        is_mouseinside(ax.scene) || return Consume(false)
        t, y = mouseposition(ax.scene)
        if event.button == Mouse.left && event.action == Mouse.press
            if ispressed(ax.scene, Keyboard.left_alt | Keyboard.right_alt)   # Alt-click adds a key
                # resolve by time AND track band — aiming at V2 must never key V1
                n = timelineframe(tl, t)
                tr = trackat(y, ntracks(seq))
                ci = findfirst(c -> c.track == tr && c.start <= n < clipend(c), seq.clips)
                ci === nothing && return Consume(false)
                clip = seq.clips[ci]
                sf = clip.src_in + (n - clip.start)
                key = nearestcurve(clip, sf, y)
                if key === nothing                       # not aiming at any curve
                    setstatus!(player, isempty(clip.animations) ?
                        "no animated parameter here — arm one with its ◆ in the Inspector first" :
                        "Alt-click ON a curve to add a keyframe to it")
                    return Consume(true)
                end
                player.kffocus.val = key                 # adding targets the curve you aim at
                p = paramspec(key)
                snapshot!(player)
                setkey!(get!(() -> AnimCurve(), clip.animations, key), sf, yval(clip, p, y))
                notify(player.playhead); return Consume(true)
            end
            i = nearestmarker(t, y); i == 0 && return Consume(false)   # else fall through to scrub
            key, kidx, clip = markmeta[i]
            if ispressed(ax.scene, Keyboard.left_control | Keyboard.right_control)
                deletekey!(clip, key, kidx)              # Ctrl-click = instant delete
                return Consume(true)
            end
            player.kffocus.val = key                     # grabbing a ◆ focuses its param
            snapshot!(player)
            dragref[] = (clip.animations[key], kidx, clip, paramspec(key))
            notify(player.kffocus)
            return Consume(true)
        elseif event.button == Mouse.left && event.action == Mouse.release && dragref[] !== nothing
            dragref[] = nothing
            dragtip.visible = false
            return Consume(true)
        elseif event.button == Mouse.right && event.action == Mouse.press
            i = nearestmarker(t, y); i == 0 && return Consume(false)   # else the clip menu opens
            key, kidx, clip = markmeta[i]
            kfmenuctx[] = (clip, key, kidx)
            c = clip.animations[key]
            k = c.keys[kidx]
            kfmenu.title = "◆ $(paramspec(key).label) · $(timestring((clip.start + (k.frame - clip.src_in)) / fps))"
            easelabel[] = keyease(c, k) === :smooth ? "Make linear (corner)" : "Ease in & out"
            holdlabel[] = k.ease === :hold ? "Interpolate again" : "Hold until the next key"
            mp = events(player.fig).mouseposition[]      # pop up AT the cursor
            vp = player.fig.scene.viewport[]
            kfmenu.halign = clamp(mp[1] / max(vp.widths[1], 1), 0.0, 1.0)
            kfmenu.valign = clamp(mp[2] / max(vp.widths[2], 1), 0.0, 1.0)
            open!(kfmenu)
            return Consume(true)
        end
        return Consume(false)
    end
    on(events(ax.scene).mouseposition; priority = 20) do _
        dragref[] === nothing && return Consume(false)
        c, i, clip, p = dragref[]
        t, y = mouseposition(ax.scene)
        f = clamp(clip.src_in + (timelineframe(tl, t) - clip.start), clip.src_in, clip.src_out)
        # snap to the playhead when close (Premiere-style) — RAW pixel distance, not
        # the frame-quantized one (a frame can already be wider than the threshold)
        vp = ax.scene.viewport[]; (x0, x1) = tl.viewrange[]
        pxpersec = max(vp.widths[1], 1) / max(x1 - x0, 1.0e-9)
        phf = clamp(playheadframe(player, clip), clip.src_in, clip.src_out)
        pht = (clip.start + (phf - clip.src_in)) / fps
        abs(t - pht) * pxpersec < 12 && (f = phf)
        v = yval(clip, p, y)
        movekey!(c, i, f, v)
        j = findfirst(k -> k.frame == f, c.keys); j === nothing || (dragref[] = (c, j, clip, p))
        dragtippos[] = Point2f((clip.start + (f - clip.src_in)) / fps, yat(clip, p, v))
        dragtiptext[] = "$(p.label)  $(round(v; digits = 2)) · $(timestring((clip.start + (f - clip.src_in)) / fps))"
        dragtip.visible = true
        notify(player.playhead); return Consume(true)
    end

    player.fxwidgets[:kfoverlay] = curveplots
    player.fxwidgets[:kfrefresh] = refresh
    return curveplots
end

# ------------------------------------------------------------- inspector panel

"""
The inspector dock, DaVinci-style: the clip under the playhead is a stack of
COLLAPSIBLE sections — one per applied effect (▾ · enable toggle · name · ×,
with the parameter sliders and their ◆ keyframe toggles in the body), plus a
Stabilization section that behaves exactly the same (mode, analyze with busy
feedback, hold-to-compare and flicker fix in the body; × on its header removes
the analysis). Unlike a fixed inspector, "+ Add effect…" appends ANY registered
kind (built-ins and live plugins) and sections can stack and be removed freely.
"""
function buildfxpanel!(player::Player, gridpos, uicolors)
    fxscroll = Subfigure(gridpos; scrollbar_thumb_color = uicolors.border)
    player.fxwidgets[:fxscroll] = fxscroll
    panel = GridLayout(fxscroll[1, 1]; valign = :top)
    head = GridLayout(panel[1, 1])
    Label(head[1, 1], "Inspector"; font = :bold, halign = :left, tellwidth = false)
    # keyframe-curve overlay toggle — it configures the timeline overlay the
    # Inspector's ◆ accessories feed, so it lives here, not in the toolbar
    kfb = Button(head[1, 2]; label = "◆", width = 30, height = 24, halign = :right)
    on(_ -> (player.kfvisible[] = !player.kfvisible[]), kfb.clicks)
    on(player.kfvisible; update = true) do o
        kfb.buttoncolor[] = o ? uicolors.accent : uicolors.surface
        kfb.labelcolor[] = o ? uicolors.text_on_accent : uicolors.text
    end
    target = map(player.playhead) do n
        loc = locate(player.sequence, n)
        loc === nothing && return "▸ no clip at the playhead"
        c = loc[1]; i = something(findfirst(x -> x === c, player.sequence.clips), 0)
        fps = player.sequence.framerate
        "▸ clip $i · $(basename(c.source.path)) ($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
    end
    Label(panel[2, 1], target; halign = :left, fontsize = 11, color = uicolors.accent,
          tellwidth = false)

    # ONE searchable menu with EVERY addable thing: built-in kinds, live plugins,
    # Stabilization and the flicker fix (their sections appear once added, like
    # any other effect)
    staged = Set{Tuple{UInt, Symbol}}()   # (clip id, :stab/:flicker) added pre-analysis
    menuopts() = vcat([(k.label, k.name) for k in effectkinds()],
                      [("Stabilization", :stabilization),
                       ("Color flicker fix", :flicker)])
    addmenu = Menu(panel[3, 1]; prompt = "+  Add effect…", default = nothing,
                   searchable = true, search_placeholder = "type to filter…",
                   options = menuopts(), tellwidth = false)
    on(PLUGINSVERSION) do _
        addmenu.options[] = menuopts()
    end
    function addbyname!(sel::Symbol)
        loc = locate(player.sequence, player.playhead[])
        loc === nothing && return setstatus!(player, "no clip at the playhead — move it onto a clip first")
        clip = loc[1]
        if sel === :stabilization
            push!(staged, (objectid(clip), :stab))
            rebuildstack(force = true)
            setstatus!(player, "Stabilization added — pick a mode and press “Stabilize clip”")
            return
        elseif sel === :flicker
            push!(staged, (objectid(clip), :flicker))
            rebuildstack(force = true)
            setstatus!(player, "Color flicker fix added — press “Analyze + fix” to run it")
            return
        end
        k = kindbyname(sel); k === nothing && return
        snapshot!(player)
        push!(clip.effects, k.make(NamedTuple(pr.name => pr.default for pr in k.params)))
        setstatus!(player, "added $(k.label) — tune it below (Ctrl+Z removes)")
        notify(player.playhead)      # rebuilds the stack + re-presents
        return
    end
    on(addmenu.selection) do sel
        sel === nothing && return
        addmenu.i_selected[] = 0     # back to the prompt; re-fires with nothing
        addbyname!(sel)
    end

    # GLOBAL before/after: holding this bypasses EVERYTHING on the clip — tracks
    # and the whole effect stack — so it always compares against the raw source
    comparebtn = Button(panel[4, 1]; label = "Hold to compare with the original",
                        tellwidth = false, width = Makie.Relative(1.0))
    stackgl = GridLayout(panel[5, 1])
    Label(panel[6, 1], "Loop"; font = :bold, halign = :left, tellwidth = false)
    loopbtn = Button(panel[7, 1]; label = "Make seamless loop", tellwidth = false,
                     width = Makie.Relative(1.0))
    on(_ -> findlooptrim!(player), loopbtn.clicks)

    # ◀◆▶ accessory state: ONE persistent (label, color) pair per param, shared by
    # every rebuild of its button and driven by a single playhead listener —
    # ◇ off-key, ◆ when the playhead sits ON a key, param color while animated
    kfaccstate = Dict{Symbol, NamedTuple}()
    function updatekfaccs()
        loc = locate(player.sequence, player.playhead[])
        clip = loc === nothing ? nothing : loc[1]
        for (key, st) in kfaccstate
            anim = clip !== nothing && clipanimated(clip, key)
            onkey = anim && any(k -> k.frame == playheadframe(player, clip),
                                clip.animations[key].keys)
            lbl = onkey ? "◆" : "◇"
            col = anim ? paramcolor(key) : uicolors.text_muted
            st.label[] == lbl || (st.label[] = lbl)
            st.color[] == col || (st.color[] = col)
        end
        return
    end
    on(_ -> updatekfaccs(), player.playhead)

    # the section stack; rebuilt when the clip, its effects, its keyframed-param
    # set or its analyses change — and on collapse toggles (force)
    collapsed = Dict{Tuple{UInt, Any}, Bool}()   # (clip id, section key) → folded?
    stabmode = Ref(:similarity)                  # survives rebuilds/clip switches
    flickercutoff = Ref(0.5)                     # Hz — the analyze parameter, ditto
    analyzeref = Ref{Any}(nothing)               # the busy label needs the live widget
    listref = Ref{Any}(nothing); lastsig = Ref{Any}(:init)
    effsig(clip) = clip === nothing ? nothing :
        (objectid(clip),
         Tuple((e isa Bypassed, uneffect(e) isa PluginEffect ? uneffect(e).name :
                                nameof(typeof(uneffect(e)))) for e in clip.effects),
         Tuple(sort!(collect(keys(clip.animations)))),
         clip.motiontrack !== nothing, clip.colortrack !== nothing,
         (objectid(clip), :stab) in staged, (objectid(clip), :flicker) in staged)
    function rebuildstack(; force::Bool = false)
        force && (lastsig[] = :force)
        loc = locate(player.sequence, player.playhead[])
        clip = loc === nothing ? nothing : loc[1]
        sig = effsig(clip)
        sig == lastsig[] && return
        lastsig[] = sig
        listref[] === nothing || Makie.clear!(listref[])
        empty!(player.fxsliders)     # re-registered per form below (scrub keeps them synced)
        gl = GridLayout(stackgl[1, 1]); listref[] = gl
        forms = Any[]; rows = Any[]
        cid = clip === nothing ? UInt(0) : objectid(clip)
        row = Ref(0)
        nextrow() = (row[] += 1)

        # One CARD per section: a bordered surface enclosing a visually distinct
        # title bar ([▾/▸] [enable?] Title … [×?]) and, when unfolded, an inset
        # body. Returns the body GridLayout (or `nothing` when folded). The card
        # sits BETWEEN the panel and the button elevation so buttons inside it
        # still read as raised.
        cardbg = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                  RGBf(1, 1, 1), 0.075)
        function section!(key, title; enabled = nothing, onenable = nothing,
                          onremove = nothing, register = nothing, hasbody = true)
            open = hasbody && !get(collapsed, (cid, key), false)
            card = GridLayout(gl[nextrow(), 1])
            # backgrounds first, so they draw behind the same cells' content
            Box(card[1:(open ? 2 : 1), 1]; color = cardbg,
                strokecolor = uicolors.border, strokewidth = 1, cornerradius = 6,
                tellwidth = false, tellheight = false)
            Box(card[1, 1]; color = uicolors.surface, strokewidth = 0,
                cornerradius = 5, tellwidth = false, tellheight = false)
            hgl = GridLayout(card[1, 1]; alignmode = Makie.Outside(8, 8, 5, 5))
            fold = Button(hgl[1, 1]; label = open ? "▾" : "▸", width = 24)
            on(fold.clicks) do _
                collapsed[(cid, key)] = open
                rebuildstack(force = true)
            end
            col = 2
            if enabled !== nothing
                tgl = Toggle(hgl[1, col]; active = enabled, length = 24, markersize = 11)
                on(a -> onenable(a), tgl.active)
                col += 1
            end
            lbl = Label(hgl[1, col], title; font = :bold, fontsize = 13, halign = :left,
                        tellwidth = false)
            push!(rows, lbl)
            col += 1
            if onremove !== nothing
                rm = Button(hgl[1, col]; label = "×", width = 24, halign = :right,
                            tellwidth = false)
                on(_ -> onremove(), rm.clicks)
                register === nothing || (player.fxwidgets[register] = rm)
            end
            colsize!(card, 1, Makie.Relative(1.0))
            open || return nothing
            return GridLayout(card[2, 1]; alignmode = Makie.Outside(10, 10, 10, 6))
        end

        if clip === nothing
            Label(gl[nextrow(), 1], "—"; halign = :left, fontsize = 11,
                  color = uicolors.text_muted, tellwidth = false)
        else
            isempty(clip.effects) &&
                Label(gl[nextrow(), 1], "No effects yet — add one above.";
                      halign = :left, fontsize = 11, color = uicolors.text_muted,
                      tellwidth = false)
            for (i, e0) in enumerate(clip.effects)
                e = uneffect(e0)
                k = effectkindfor(e)
                body = section!((:fx, i), k === nothing ? string(nameof(typeof(e))) : k.label;
                    hasbody = k !== nothing && !isempty(k.params),
                    enabled = !(e0 isa Bypassed),
                    onenable = a -> begin      # bypass keeps the params, all render paths skip it
                        (1 <= i <= length(clip.effects)) || return
                        snapshot!(player)
                        cur = uneffect(clip.effects[i])
                        clip.effects[i] = a ? cur : Bypassed(cur)
                        notify(player.playhead)
                    end,
                    onremove = () -> begin
                        (1 <= i <= length(clip.effects)) || return
                        snapshot!(player); deleteat!(clip.effects, i); notify(player.playhead)
                    end)
                (body !== nothing && k !== nothing && !isempty(k.params)) || continue
                cur0 = k.read(e)
                # form fields are keyed by the DISPLAY label ("Brightness"), mapped
                # back to the param by position — ParamForm labels rows by field name
                fieldsym(pr) = Symbol(pr.label)
                spec = NamedTuple(fieldsym(pr) => (Float64(cur0[pr.name]), Makie.Between(pr.min, pr.max))
                                  for pr in k.params)
                accessory = (field, pos) -> begin   # Premiere-style ◀ ◆ ▶ per parameter
                    key = k.kfkeys[findfirst(pr -> fieldsym(pr) === field, k.params)]
                    st = get!(() -> (label = Observable("◇"),
                                     color = Observable{Any}(uicolors.text_muted)),
                              kfaccstate, key)
                    acc = GridLayout(pos)
                    prevb = Button(acc[1, 1]; label = "◀", width = 16, fontsize = 8,
                                   labelcolor = uicolors.text_muted)
                    kf = Button(acc[1, 2]; label = st.label, width = 22,
                                labelcolor = st.color)
                    nextb = Button(acc[1, 3]; label = "▶", width = 16, fontsize = 8,
                                   labelcolor = uicolors.text_muted)
                    colgap!(acc, 1)
                    on(_ -> gotokey!(player, key, -1), prevb.clicks)
                    on(_ -> togglekey!(player, key), kf.clicks)
                    on(_ -> gotokey!(player, key, 1), nextb.clicks)
                    player.fxwidgets[Symbol(:kfacc_, key)] = (prevb, kf, nextb)
                    acc
                end
                pf = Makie.ParamForm(body[1, 1], spec, accessory; labelwidth = 88,
                                     widgetwidth = 132, accessorywidth = 62, rowgap = 4,
                                     halign = :left)
                updatekfaccs()   # seed the fresh buttons' ◇/◆ state
                push!(forms, pf)
                for (j, pr) in enumerate(k.params)   # scrub-sync registry
                    w = get(pf.widgets, fieldsym(pr), nothing)
                    w isa Slider && (player.fxsliders[k.kfkeys[j]] = w)
                end
                # live-apply (keyframe- and bypass-aware). The handler receives the
                # WHOLE form tuple, so only params whose OWN slider moved since the
                # last fire may act — anything else is a synced echo (scrub keeps
                # animated sliders on their curves, quantized to the slider step);
                # treating those as edits stamped ghost keys on untouched params.
                lastvals = Ref{Any}(nothing)
                on(pf.graph[:values]) do vals
                    prevvals = lastvals[]; lastvals[] = vals
                    player.fxsyncing[] && return         # sync: just refresh the baseline
                    (1 <= i <= length(clip.effects)) || return
                    cur = clip.effects[i]
                    k.matches(uneffect(cur)) || return
                    prev = k.read(uneffect(cur))
                    changedstatic = NamedTuple()
                    for (j, pr) in enumerate(k.params)
                        v = Float64(vals[fieldsym(pr)])
                        key = k.kfkeys[j]
                        moved = prevvals === nothing ?   # before any fire: value baselines
                            abs(v - (clipanimated(clip, key) ?
                                     paramvalue(clip, key, playheadframe(player, clip)) :
                                     Float64(prev[pr.name]))) > 1.0e-9 :
                            v != Float64(prevvals[fieldsym(pr)])
                        moved || continue
                        if time() - player.lastslidersnap > 1.5   # one undo entry per gesture
                            snapshot!(player); player.lastslidersnap = time()
                        end
                        if clipanimated(clip, key)   # animated param: the slider writes a key
                            setkey!(clip.animations[key], playheadframe(player, clip), v)
                        else
                            changedstatic = merge(changedstatic, NamedTuple{(pr.name,)}((v,)))
                        end
                    end
                    if !isempty(changedstatic)
                        newe = k.make(merge(prev, changedstatic))
                        clip.effects[i] = cur isa Bypassed ? Bypassed(newe) : newe
                    end
                    player.playing[] || notify(player.playhead)
                end
            end

            # ---- Stabilization: a section like any other effect — it appears when
            # ADDED from the menu (or when an analysis exists), × removes it
            haskey(player.fxwidgets, :remove) && clip.motiontrack === nothing &&
                delete!(player.fxwidgets, :remove)
            showstab = clip.motiontrack !== nothing || (cid, :stab) in staged
            sgl = !showstab ? nothing : section!(:stab, "Stabilization";
                register = clip.motiontrack === nothing ? nothing : :remove,
                onremove = () -> begin
                    delete!(staged, (cid, :stab))
                    if clip.motiontrack !== nothing
                        removestabilization!(player)   # notifies → rebuild
                    else
                        rebuildstack(force = true)     # just un-stage the section
                    end
                end)
            if sgl !== nothing
                opts = [("Camera lock — like a tripod", :similarity),
                        ("Object lock — keep a subject still", :objectlock),
                        ("Tripod (affine) — legacy", :tripod),
                        ("Tripod + perspective — legacy", :perspective),
                        ("Smooth — keep camera moves", :smooth)]
                modemenu = Menu(sgl[1, 1]; options = opts, tellwidth = false,
                                default = something(findfirst(o -> o[2] === stabmode[], opts), 1))
                on(sel -> sel === nothing || (stabmode[] = sel), modemenu.selection)
                analyzebtn = Button(sgl[2, 1]; tellwidth = false, width = Makie.Relative(1.0),
                                    label = player.stabinfo[] == "analyzing…" ? "⏳ Analyzing…" :
                                            "Stabilize clip")
                Label(sgl[3, 1], player.stabinfo; halign = :left, fontsize = 11,
                      tellwidth = false, color = uicolors.text_muted)
                on(analyzebtn.clicks) do _
                    player.stabinfo[] == "analyzing…" &&
                        return setstatus!(player, "analysis already running — progress in the bottom right")
                    mode = something(modemenu.selection[], stabmode[])
                    if mode === :objectlock
                        armpick!(player)   # the next preview click picks the subject to lock
                    else
                        analyzeat!(player, (c; kwargs...) ->
                                       analyzemotion!(c; mode, backend = player.analysisbackend, kwargs...),
                                   "motion stabilization")
                    end
                end
                analyzeref[] = analyzebtn
                merge!(player.fxwidgets, Dict{Symbol, Any}(
                    :modemenu => modemenu, :analyze => analyzebtn))
                rowgap!(sgl, 8)
            else
                analyzeref[] = nothing
            end

            # ---- Color flicker fix: its own effect-like section (a per-frame
            # exposure/color track) — analyze in the body, × removes the track
            showflicker = clip.colortrack !== nothing || (cid, :flicker) in staged
            fgl = !showflicker ? nothing : section!(:flicker, "Color flicker fix";
                onremove = () -> begin
                    delete!(staged, (cid, :flicker))
                    if clip.colortrack !== nothing
                        snapshot!(player)
                        clip.colortrack = nothing
                        setstatus!(player, "flicker fix removed (Ctrl+Z restores)")
                    end
                    notify(player.playhead)
                    rebuildstack(force = true)
                end)
            if fgl !== nothing
                # cutoff = the ANALYSIS parameter: everything faster than this is
                # treated as flicker, slower intentional changes survive
                Label(fgl[1, 1], "Cutoff (Hz)"; halign = :left, fontsize = 12)
                cutslider = Slider(fgl[1, 2]; range = 0.1:0.05:2.0,
                                   startvalue = flickercutoff[])
                Label(fgl[1, 3], map(v -> string(round(v, digits = 2)), cutslider.value);
                      fontsize = 11, halign = :left, color = uicolors.text_muted, width = 32)
                on(v -> flickercutoff[] = Float64(v), cutslider.value)
                colorbtn = Button(fgl[2, 1:3]; tellwidth = false, width = Makie.Relative(1.0),
                                  label = clip.colortrack === nothing ? "Analyze + fix flicker" :
                                          "Re-analyze with this cutoff")
                on(_ -> analyzeat!(player, (c; kw...) ->
                                       analyzecolor!(c; cutoff = flickercutoff[],
                                                     backend = player.analysisbackend, kw...),
                                   "color stabilization"), colorbtn.clicks)
                if clip.colortrack !== nothing
                    # strength scales the correction at APPLY time — live, no re-analysis
                    Label(fgl[3, 1], "Strength"; halign = :left, fontsize = 12)
                    sslider = Slider(fgl[3, 2]; range = 0.0:0.01:1.0,
                                     startvalue = clip.colortrack.strength)
                    Label(fgl[3, 3], map(v -> string(round(v, digits = 2)), sslider.value);
                          fontsize = 11, halign = :left, color = uicolors.text_muted, width = 32)
                    on(sslider.value) do v
                        ct = clip.colortrack
                        ct === nothing && return
                        ct.strength = Float32(v)
                        player.playing[] || notify(player.playhead)
                    end
                    Label(fgl[4, 1:3], stabdescription(clip.colortrack); halign = :left,
                          fontsize = 11, tellwidth = false, color = uicolors.text_muted)
                else
                    Label(fgl[3, 1:3], "not analyzed yet"; halign = :left, fontsize = 11,
                          tellwidth = false, color = uicolors.text_muted)
                end
                player.fxwidgets[:color] = colorbtn
            end
        end
        player.fxwidgets[:effectrows] = rows
        player.fxwidgets[:effectforms] = forms
        if row[] > 0
            colsize!(gl, 1, Makie.Relative(1.0))   # cards span the full panel width
            row[] > 1 && rowgap!(gl, 10)
        end
        return
    end
    on(_ -> rebuildstack(), player.playhead)
    rebuildstack()

    on(player.stabinfo) do s    # busy feedback at the button that started the job
        b = analyzeref[]
        b === nothing && return
        try
            b.label[] = s == "analyzing…" ? "⏳ Analyzing…" : "Stabilize clip"
        catch
        end
    end
    # press-and-hold compare (Button.clicks only fires on release): while held,
    # EVERYTHING on the clip is bypassed — tracks and the whole effect stack
    on(events(player.fig).mousebutton; priority = 100) do event
        (player.dockopen[] === :effects && event.button == Mouse.left) ||
            return Consume(false)
        if event.action == Mouse.press &&
           Point2f(events(player.fig).mouseposition[]) in comparebtn.layoutobservables.computedbbox[]
            player.applytracks[] = false; notify(player.playhead); return Consume(true)
        elseif event.action == Mouse.release && !player.applytracks[]
            player.applytracks[] = true; notify(player.playhead)
        end
        return Consume(false)
    end

    # ---- Ctrl+P command palette: search EVERYTHING addable and apply it to the
    # clip under the playhead — type to filter, ⏎ takes the top hit
    pal = Modal(player.fig; title = "Add effect — type to search", min_size = (380, 300))
    palquery = Observable("")
    Label(pal[1, 1], map(q -> isempty(q) ? "▏ type to filter…" : q * "▏", palquery);
          halign = :left, font = :bold, tellwidth = false)
    palrows = GridLayout(pal[2, 1])
    pallist = Ref{Any}(nothing)
    function palhits()
        q = lowercase(palquery[])
        hits = [o for o in menuopts() if isempty(q) || occursin(q, lowercase(o[1]))]
        if !isempty(q)   # a parameter label ("brightness") finds its owning kind
            for k in effectkinds(), pr in k.params
                occursin(q, lowercase(pr.label)) || continue
                any(h -> h[2] === k.name, hits) && continue
                push!(hits, ("$(k.label) · $(pr.label)", k.name))
            end
        end
        return hits
    end
    function palrefresh()
        pallist[] === nothing || Makie.clear!(pallist[])
        gl = GridLayout(palrows[1, 1]); pallist[] = gl
        hits = palhits()
        for (r, (label, name)) in enumerate(first(hits, 8))
            b = Button(gl[r, 1]; label = r == 1 ? label * "    ⏎" : label,
                       tellwidth = false, width = Makie.Relative(1.0))
            on(b.clicks) do _
                close!(pal); addbyname!(name)
            end
        end
        isempty(hits) && Label(gl[1, 1], "no match"; fontsize = 11,
                               color = uicolors.text_muted, tellwidth = false)
        return
    end
    on(_ -> palrefresh(), palquery)
    on(events(player.fig).unicode_input) do chars
        pal.open[] || return Consume(false)
        s = chars isa AbstractVector ? String(collect(chars)) : string(chars)
        isempty(s) && return Consume(false)
        palquery[] = palquery[] * s
        return Consume(true)
    end
    on(events(player.fig).keyboardbutton; priority = 30) do ev
        pal.open[] || return Consume(false)
        ev.action in (Keyboard.press, Keyboard.repeat) || return Consume(false)
        if ev.key == Keyboard.backspace
            isempty(palquery[]) || (palquery[] = String(chop(palquery[])))
        elseif ev.key == Keyboard.enter
            hits = palhits()
            close!(pal)
            isempty(hits) ? setstatus!(player, "no effect matches “$(palquery[])”") :
                            addbyname!(hits[1][2])
        elseif ev.key == Keyboard.escape
            close!(pal)
        end
        return Consume(true)   # the palette owns the keyboard while open
    end

    merge!(player.fxwidgets, Dict{Symbol, Any}(
        :addeffect => addmenu, :fxlistrefresh => rebuildstack, :loop => loopbtn,
        :compare => comparebtn,
        :palettemodal => pal, :palettequery => palquery, :paletteapply => addbyname!,
        :paletteopen => () -> (palquery[] = ""; palrefresh(); open!(pal)),
        :stabopen => () -> begin
            opendock!(player, :effects)
            loc = locate(player.sequence, player.playhead[])
            loc === nothing || push!(staged, (objectid(loc[1]), :stab))
            rebuildstack(force = true)
        end))
    rowgap!(panel, 10)
    colsize!(panel, 1, Makie.Relative(1.0))   # content fills the dock width, left-aligned
    return panel
end

"Right-click context modal on the timeline: clip actions + shortcut reference."
function wireclipmenu!(player::Player)
    # SMALL and contextual: only what has no first-class home elsewhere —
    # splitting/cropping live on the toolbar (✂ ▢), effects and stabilization
    # in the Inspector, tools in the Tools dock
    modal = Modal(player.fig; title = "Clip actions", min_size = (220, 10),
                  backdrop_color = (:black, 0.15))   # popup, not a dialog
    player.clipmodal = modal
    rcframe() = clamp(round(Int, player.rctime * player.sequence.framerate), 0,
                      max(seqlength(player.sequence) - 1, 0))
    actions = [
        ("Join with next clip", "", () -> joinat!(player; at = rcframe())),
        ("Delete clip (ripple)", "X", () -> begin
            deleteclip!(player.sequence, rcframe())
            player.playhead[] = clamp(player.playhead[], 0, max(seqlength(player.sequence) - 1, 0))
            refreshedit!(player)
        end),
        ("Reset crop", "R", () -> resetcropat!(player, rcframe())),
    ]
    for (i, (text, key, action)) in enumerate(actions)
        btn = Button(modal[i, 1]; label = isempty(key) ? text : "$text   ·  $key",
                     tellwidth = false)
        on(btn.clicks) do _
            close!(modal)
            action()
        end
    end
    player.timeline.onrightclick = t -> begin
        player.rctime = t
        modal.title = "Clip @ " * timestring(t)
        # pop up AT the cursor (fractional align, clamped into the window)
        mp = events(player.fig).mouseposition[]
        vp = player.fig.scene.viewport[]
        modal.halign = clamp(mp[1] / max(vp.widths[1], 1), 0.0, 1.0)
        modal.valign = clamp(mp[2] / max(vp.widths[2], 1), 0.0, 1.0)
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
        setstatus!(player, "$(p.label): keyframing on — scrub + move the slider to add keys (curve on the clip)")
    else
        setstatus!(player, "$(p.label): its ◆ keys are on the clip — drag moves · Alt-click adds · Ctrl-click deletes · right-click eases")
    end
    player.kffocus[] = key
    player.kfvisible[] = true
    notify(player.playhead)
    return nothing
end

"""
The ◆ of the inspector's ◀◆▶ trio, Premiere semantics: arm the parameter if it
isn't animated yet (first key = current value); otherwise ADD a key at the
playhead — or REMOVE the one sitting there (removing the last key makes the
parameter static again).
"""
function togglekey!(player::Player, key::Symbol)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    clipanimated(clip, key) || return armkeyframe!(player, key)
    p = paramspec(key)
    c = clip.animations[key]
    sf = playheadframe(player, clip)
    snapshot!(player)
    if any(k -> k.frame == sf, c.keys)
        removekey!(c, sf)
        if isempty(c)
            delete!(clip.animations, key)
            syncsliders!(player, clip)
            setstatus!(player, "$(p.label): last keyframe removed — back to a static value")
        else
            setstatus!(player, "$(p.label): keyframe removed")
        end
    else
        setkey!(c, sf, paramvalue(clip, key, sf))
        setstatus!(player, "$(p.label): keyframe added at the playhead")
    end
    player.kffocus[] = key
    notify(player.playhead)
    return nothing
end

"Jump the playhead to the previous (`dir < 0`) or next keyframe of `key` on the
clip under it — the ◀ ▶ of the inspector trio."
function gotokey!(player::Player, key::Symbol, dir::Integer)
    loc = locate(player.sequence, player.playhead[])
    loc === nothing && return nothing
    clip = loc[1]
    p = paramspec(key)
    clipanimated(clip, key) || return setstatus!(player, "$(p.label): no keyframes yet — ◆ adds one")
    sf = playheadframe(player, clip)
    ks = filter(k -> clip.src_in <= k.frame <= clip.src_out, clip.animations[key].keys)
    cand = dir < 0 ? filter(k -> k.frame < sf, ks) : filter(k -> k.frame > sf, ks)
    isempty(cand) && return setstatus!(player,
        "$(p.label): no keyframe $(dir < 0 ? "before" : "after") the playhead")
    k = dir < 0 ? last(cand) : first(cand)
    player.kffocus[] = key
    seek!(player, clamp(clip.start + (k.frame - clip.src_in), 0, seqlength(player.sequence) - 1))
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
    try
        deactivatetool!(player)   # tool handlers reference the dying scenes
    catch
    end
    freegpucache!(player)
    foreach(sp -> stop!(sp.worker), values(player.pools))
    stop!(player.timeline)
    close(player.statusqueue)
    close(player.uiqueue)
    player.screen === nothing || close(player.screen)
    player.screen = nothing
    return nothing
end
