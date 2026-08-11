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
    const matteinfo::Observable{String}          # matte tool status line
    const restoreinfo::Observable{String}        # restore tool status line
    const mattemarks::Dict{UInt64, Dict{Int, Matrix{UInt8}}}  # clip id -> marked frames
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
    lastcanvas::Tuple{Int, Int}   # the published frame's size — the limits' only input
    lastclip::Union{Nothing, Clip}
    clipmodal::Any
    rctime::Float64
    gpuworker::Any
    # true once the render engine's context turns out to be owned by THIS thread
    # (see `runowned`) — discovered, never configured
    engineinline::Bool
    onpick::Any  # one-shot preview-click callback, set while object lock waits for it
    previewplot::Any  # the preview image plot (texture swap target, glbridge.jl)
    gpupreview::Any   # GPUPreview when Player(gpupreview = true), else nothing
    audio::Any        # AudioPreview when Player(audiopreview = true), else nothing
    # What turns matte clicks into a mask (see `seedmask`): SAM 2.1 by default,
    # `nothing` when its weights aren't on disk — then the seed is painted discs.
    # A field rather than a registry: which model an editor segments with is that
    # editor's business, and two open on one desktop can differ.
    segmenter::Any
    const undostack::Vector{Any}
    const redostack::Vector{Any}
    lastslidersnap::Float64
    edits::Int              # bumped by every edit (see `snapshot!`)
    autosaved::Int          # `edits` at the last autosave
    screen::Any
    presented::Int
    dropped::Int
    const dockpanels::Dict{Symbol, Any}  # dock key → (; sf::Subfigure, width)
    const dockopen::Observable{Symbol}   # open dock panel, :none when collapsed
    const mediasources::Observable{Vector{VideoSource}}  # media-bin content
    const binrows::Vector{Any}           # bin row buttons (rebuilt on change)
    dragsource::Any                      # bin source mid-drag onto the timeline
    const tool::Observable{Symbol}       # active tool: :none, :split, :crop
    const playrate::Base.RefValue{Float64}  # JKL shuttle rate: 1.0 normal, <0 reverse, |·|>1 fast
    const kffocus::Observable{Symbol}    # param whose ◆ markers the timeline overlay shows
    const kfvisible::Observable{Bool}    # keyframe curves on the timeline visible
    const jobprogress::Threads.Atomic{Float64}  # running job fraction 0..1, NaN when idle
                                                # (written from worker threads, polled by the UI)
    const gpucache::Dict{Any, Any}       # source → device-resident decoded frames (pure-GPU playback)
    engine::FxEngine                     # THE render engine, on the declared backend.
                                         # Not const: `autodetectgpu!` can upgrade the
                                         # backend, and the engine follows it.
    # Which effects THIS editor offers. Defaults to the module registry, which is
    # where load-time and plugin registration land; a second editor can be given
    # its own and differ.
    const effects::EffectRegistry
end

"""
    docsnapshot(player) -> (clips, mattemarks)

Everything an undo has to put back: the timeline AND the inputs that produced
what is rendered on it. The matte's seed marks live next to the player rather
than on a clip, so a snapshot of the sequence alone restored a clip whose matte
had been removed — with no way to get the marks back.

The masks themselves are shared, not copied: a mark is replaced when it changes,
never written through, so the reference is as good as the value and a hundred
undo steps cost a hundred dictionaries instead of a hundred megabytes.
"""
docsnapshot(player::Player) =
    (snapshot(player.sequence),
     Dict{UInt64, Dict{Int, Matrix{UInt8}}}(k => copy(v) for (k, v) in player.mattemarks))

"Put a [`docsnapshot`](@ref) back."
function docrestore!(player::Player, snap)
    clips, marks = snap
    restore!(player.sequence, clips)
    empty!(player.mattemarks)
    for (k, v) in marks
        player.mattemarks[k] = copy(v)
    end
    return nothing
end

"Push the current edit state onto the undo stack (clears redo)."
function snapshot!(player::Player)
    player.edits += 1        # every edit passes here; the autosave watches it
    push!(player.undostack, docsnapshot(player))
    length(player.undostack) > 100 && popfirst!(player.undostack)
    empty!(player.redostack)
    return nothing
end

function undo!(player::Player)
    isempty(player.undostack) && return nothing
    push!(player.redostack, docsnapshot(player))
    docrestore!(player, pop!(player.undostack))
    postrestore!(player)
    return nothing
end

function redo!(player::Player)
    isempty(player.redostack) && return nothing
    push!(player.undostack, docsnapshot(player))
    docrestore!(player, pop!(player.redostack))
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

"Post `f` to a worker directly — for callers that own a [`GPUWorker`] but no `Player`
(the test suite's GPU beats, which must reach Lava on the same pinned thread the
editor uses, or they fix ownership on main and every later analysis asserts)."
function rungpu(f::Function, w::GPUWorker)
    put!(w.jobs, f)
    return nothing
end

function rungpu(f::Function, player::Player)
    player.gpuworker === nothing && (player.gpuworker = GPUWorker())
    return rungpu(f, player.gpuworker)
end

"""
Run a long analysis/render `job` on the pinned worker — one executor, whatever
the backend is. The sources' stream rings are closed for the duration (their VRAM
starves the analysis pool otherwise: flaky pool-block OOM in `goodfeatures`) and
re-opened afterwards.

Pinned unconditionally because a `BatchQueue` belongs to the thread that first
touched it, so a model built from `Threads.@spawn` dies on "BatchQueue is
single-writer" — and every analysis here drives a Vulkan model.
"""
function runanalysis(job::Function, player::Player)
    streamed = collect(keys(player.gpucache))
    isempty(streamed) || freegpucache!(player)
    rungpu(player) do
        try
            job()
        finally
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
    streamed = collect(keys(player.gpucache))
    isempty(streamed) || freegpucache!(player)
    try
        return rungpusync(job, player)
    finally
        for src in streamed
            Threads.@spawn preloadgpu!(player, src)
        end
    end
end

function Player(path::AbstractString; capacity::Integer = 64,
                background = RGBf(0.114, 0.12, 0.135), accent = RGBf(1.0, 0.47, 0.22),
                analysisbackend = nothing, gpupreview = nothing,
                audiopreview::Bool = true,
                proxyheight::Integer = 720, proxythreshold::Integer = 2_100_000,
                effects::EffectRegistry = EFFECTS)
    # GPU playback is the default: leaving both `analysisbackend` and `gpupreview`
    # unset auto-detects a video-capable Vulkan device (below) and, if present, runs
    # decode + effects on the GPU. Pass either to force the choice.
    autodetect = analysisbackend === nothing && gpupreview === nothing
    backend = analysisbackend === nothing ? KA.CPU() : analysisbackend
    # an explicit GPU analysis backend implies GPU playback — `gpupreview = false` opts out
    wantgpu = gpupreview === true || (gpupreview === nothing && !(backend isa KA.CPU))
    wantgpu && backend isa KA.CPU &&
        error("gpupreview = true requires a GPU backend, e.g. Player(path; analysisbackend = LavaBackend(), gpupreview = true)")
    # a project path (.json, or a .toml from before the format moved) opens the
    # saved edit instead of a video
    isproject = endswith(lowercase(path), ".toml") || endswith(lowercase(path), ".json")
    sequence = isproject ? loadproject(path) : Sequence(VideoSource(path))
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
         border         = Makie.lerp_oklab(bgc, wht, 0.26),
         # SELECTION gets its own colour rather than the accent. The accent
         # already marks the active tool, the playhead, every active control and
         # half the chrome; one more orange thing does not read as "this one".
         select         = RGBf(0.29, 0.60, 0.98),
         select_subtle  = Makie.lerp_oklab(bgc, RGBf(0.29, 0.60, 0.98), 0.30)))
    player = Makie.with_theme(colors = Makie.Attributes(; uicolors...),
                              backgroundcolor = background,
                              textcolor = uicolors.text) do
        buildui(sequence, pools, Int(capacity), Int(proxyheight), Int(proxythreshold),
                frame, playhead, playing; background, uicolors, analysisbackend = backend,
                effects)
    end
    player.screen = display(player.fig)
    wantgpu && (player.gpupreview = GPUPreview(); player.fxwidgets[:lanechip][] = "GPU")
    # thumbnails decode on the GPU too — through the pinned worker (single-writer)
    wantgpu && setgpurun!(player.timeline, f -> rungpusync(f, player))
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

"The decode pool for `source`, created on first use. One reader per source is
enough even when two clips of it overlap in a blend: the ring and the GPU
stream index by GOP, and serving two positions from one reader measured 0
stand-ins and 5.4 ms/frame vs 8.2 ms for a reader per layer (2026-07-28)."
pool(player::Player, source::VideoSource) =
    get!(() -> SourcePool(source; capacity = player.capacity), player.pools, source)

pool(player::Player, clip::Clip) = pool(player, clip.source)

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
                 frame, playhead, playing; background, uicolors, analysisbackend,
                 effects::EffectRegistry = EFFECTS)
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
    # …and BLACK behind it, not the UI grey: everything the picture doesn't cover
    # is a letterbox bar, and the encoder writes black there. A programme monitor
    # that surrounds the frame in grey is off by exactly that lie.
    ax = Axis(fig[1, 3], aspect = DataAspect(), yreversed = true,
              backgroundcolor = RGBf(0, 0, 0))
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
    # render-backend chip: a DISPLAY of `player.gpupreview` (the declared config,
    # set explicitly or by autodetect) — never a state of its own
    lanechip = Observable("CPU")
    Label(controls[1, 8], lanechip; width = 34, halign = :right, fontsize = 11,
          color = map(l -> l == "GPU" ? uicolors.accent : uicolors.text_muted, lanechip))
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
                    Observable(true), Observable("no analysis yet"),
                    Observable("no matte"), Observable("no restoration"),
                    Dict{UInt64, Dict{Int, Matrix{UInt8}}}(),
                    analysisbackend,
                    fig, ax, Ref(false),
                    Observable(Point2f[]),
                    similar(frame[]), Dict{Symbol, Slider}(),
                    Dict{Symbol, Any}(),
                    Ref(false), nothing, (0.0, 0.0, 1.0, 1.0), (0, 0), nothing, nothing, 0.0,
                    nothing, false, nothing, nothing, nothing, nothing, defaultsegmenter(),
                    Any[], Any[], 0.0, 0, 0, nothing, 0, 0,
                    Dict{Symbol, Any}(), Observable(:none),
                    Observable(VideoSource[]), Any[], nothing, Observable(:none),
                    Ref(1.0), Observable(:opacity), Observable(true),
                    Threads.Atomic{Float64}(NaN), Dict{Any, Any}(), FxEngine(analysisbackend),
                    effects)
    player.fxwidgets[:lanechip] = lanechip
    @async for s in player.statusqueue  # main-thread consumer: threads → observable
        status[] = s
    end
    player.previewplot = previewplot
    installtransformgizmo!(player)   # direct manipulation for TransformEffect
    startautosave!(player)
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
    buildpalette!(player, uicolors)      # Ctrl+P: every command, one search box
    buildkeyframemodal!(player, uicolors)   # ◆: what is animated on this clip
    fxbtn = toolbarbutton!(player, toolbar[1, 1], "FX", :effects, uicolors)
    player.mediasources[] = unique([c.source for c in sequence.clips])
    mediadock = dockpanel!(player, :media)
    buildmediabin!(player, mediadock[1, 1], uicolors)
    exportdock = dockpanel!(player, :export)
    buildexportpanel!(player, exportdock[1, 1], uicolors)
    # toolbar in three groups: the DOCKS (FX · Bin · Out · Prj), the edit tools
    # (✂ ▢), the one-shots (✕ ↶ ↷).
    #
    # There is no ⚒ Tools dock. Stabilize, the matte, the flicker fix, restore,
    # the loop finder and blend are effect kinds like Blur, so they are cards in
    # the ONE Effects panel — which is also the only way the two could stop
    # fighting over `:toolslots`, having both built the same card bodies into one
    # shared registry.
    binbtn = toolbarbutton!(player, toolbar[2, 1], "Bin", :media, uicolors)
    outbtn = toolbarbutton!(player, toolbar[3, 1], "Out", :export, uicolors)
    projectdock = dockpanel!(player, :project; width = 300)
    buildprojectpanel!(player, projectdock[1, 1], uicolors)
    prjbtn = toolbarbutton!(player, toolbar[4, 1], "Prj", :project, uicolors)
    buildkeyframeoverlay!(player)   # THE keyframe editor: curves + ◆ on the clips
    splitbtn = Button(toolbar[5, 1]; label = "✂", width = 40, height = 40)
    cropbtn = Button(toolbar[6, 1]; label = "▢", width = 40, height = 40)
    on(_ -> usetool!(player, :split), splitbtn.clicks)
    on(_ -> usetool!(player, :crop), cropbtn.clicks)
    oneshots = [("✕", "Delete  (X)", () -> deleteat!(player)),
                ("↶", "Undo  (Ctrl+Z)", () -> isempty(player.undostack) ? setstatus!(player, "nothing to undo") :
                            (undo!(player); setstatus!(player, "undone (Ctrl+Z redoes with Shift)"))),
                ("↷", "Redo  (Ctrl+⇧+Z)", () -> isempty(player.redostack) ? setstatus!(player, "nothing to redo") :
                            (redo!(player); setstatus!(player, "redone")))]
    onebtns = Makie.Button[]
    for (row, (lbl, _tip, action)) in enumerate(oneshots)
        b = Button(toolbar[7 + row, 1]; label = lbl, width = 40, height = 40)
        on(_ -> action(), b.clicks)
        push!(onebtns, b)
    end
    # hover tooltips: hovering a toolbar button shows its name + shortcut to the
    # right of it (detected by mouse-vs-bbox; Makie Buttons have no hover attr).
    tiptargets = vcat([(fxbtn, "Effects — everything applied to the clip"),
                       (binbtn, "Media bin — import & drag clips"),
                       (outbtn, "Export"),
                       (splitbtn, "Blade  (S)"), (cropbtn, "Crop  (C)")],
                      [(onebtns[i], oneshots[i][2]) for i in eachindex(onebtns)])
    # A tooltip is a SOLID little card, not glowing text: a stroked glyph over the
    # timeline's thumbnails or a bright preview is unreadable (GLMakie's glow is
    # weak), and it has to sit above modals too — the Modal block puts its overlay
    # at z = 1000, so the tip goes higher. One registry for the whole UI: anything
    # that wants a tip pushes (block, text) into `player.fxwidgets[:tips]`.
    tip_txt = Observable(" "); tip_pos = Observable(Point2f(0, 0)); tip_vis = Observable(false)
    tip_box = Observable(Rect2f(0, 0, 0, 0))
    tipplot = poly!(fig.scene, tip_box; color = uicolors.surface, strokewidth = 1,
                    strokecolor = uicolors.border, visible = tip_vis, space = :pixel,
                    overdraw = true)
    translate!(tipplot, 0, 0, 2000)
    tiptext = Makie.text!(fig.scene, tip_pos; text = tip_txt, visible = tip_vis, space = :pixel,
                          align = (:left, :center), fontsize = 13, color = uicolors.text,
                          overdraw = true)
    translate!(tiptext, 0, 0, 2001)
    player.fxwidgets[:tips] = Dict{Any, String}()
    """
    Show `label` beside the block bounding box `bb`, flipping to its left when the
    card would leave the window.
    """
    function showtip!(bb, label)
        # MULTI-LINE: most tips are two words, but an effect card's ? holds its
        # whole description (pre-wrapped by `wraptext`). Sizing on `length` alone
        # drew a 400-wide one-line box off the side of the window.
        lines = split(label, '\n')
        w = 6.6 * maximum(length, lines) + 16
        hgt = 17 * length(lines) + 9
        vp = widths(fig.scene.viewport[])
        x = bb.origin[1] + bb.widths[1] + 10
        x + w > vp[1] && (x = bb.origin[1] - w - 10)
        # …and keep the whole card on screen, or a long description hangs its
        # first lines above the top edge
        y = clamp(bb.origin[2] + bb.widths[2] / 2, hgt / 2 + 4, vp[2] - hgt / 2 - 4)
        tip_box[] = Rect2f(x, y - hgt / 2, w, hgt)
        tip_pos[] = Point2f(x + 8, y)
        tip_txt[] = label
        tip_vis[] = true
        return nothing
    end
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
    for (btn, label) in tiptargets
        player.fxwidgets[:tips][btn] = label
    end
    on(events(fig).mouseposition) do mp
        p = Point2f(mp); hit = nothing
        for (btn, label) in player.fxwidgets[:tips]
            bb = btn.layoutobservables.computedbbox[]
            if bb.origin[1] <= p[1] <= bb.origin[1] + bb.widths[1] &&
               bb.origin[2] <= p[2] <= bb.origin[2] + bb.widths[2]
                hit = (bb, label); break
            end
        end
        if hit === nothing
            tip_vis[] && (tip_vis[] = false)
        else
            showtip!(hit[1], hit[2])
        end
        refreshcursor!()
        return Consume(false)
    end
    # active tool → cursor scope, crop mode, hint, button highlight (cropmode is a
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
        tlscene = timeline.axis.scene
        # is_mouseinside, not `in viewport`: an active blade must not cut THROUGH
        # a dropdown or a modal that happens to hang over the timeline
        Makie.is_mouseinside(tlscene) || return Consume(false)
        t = Makie.mouseposition(tlscene)[1]
        # NB: must NOT be named `frame` — that would rebind the shared preview
        # Observable this scope captures (used by image! + the scrub fallback)
        cutat = clamp(round(Int, t * sequence.framerate), 0, max(seqlength(sequence) - 1, 0))
        snapshot!(player)  # so Ctrl+Z / the undo tool can revert the cut
        split!(sequence, cutat)
        refreshedit!(player)
        # Persistent blade: stays active so the mouse keeps cutting (DaVinci blade).
        # Esc or clicking ✂ again puts it away.
        setstatus!(player, "cut at $(timecode(sequence, cutat)) — blade still active (Esc to stop)")
        return Consume(true)
    end
    opendock!(player, :effects)   # the working panel starts open

    # picking another clip re-aims the inspector even when the playhead stays put
    # (clicking the lane below it is exactly that case)
    on(_ -> notify(playhead), timeline.selected)
    on(playhead) do n
        loc = editclip(player)          # the inspector follows the SELECTED clip
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
                        ensureframesize!(player, pool(player, clip).source)
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

    # (files dropped on the window go to the media bin — see buildmediabin!)
    timeline.onedit = () -> snapshot!(player)
    # trimming shows the frame the cut would land on — exact when it is decoded,
    # the decoder's nearest otherwise so the picture still follows the drag
    timeline.ontrimpreview = (clip, sf) -> begin
        presentclipframe!(player, clip, sf; standin = false) ||
            presentclipframe!(player, clip, sf; standin = true)
        return nothing
    end
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
    spA = pool(player, left)
    spB = pool(player, right)
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
        runowned(player) do    # both sides on the engine's owning thread, one hop
            render(player.engine, bufA, lclip, Int(srcA)) do out
                copyto!(bufA, out)
            end
            render(player.engine, bufB, rclip, Int(srcB)) do out
                copyto!(bufB, out)
            end
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
    canvas = canvassize(player.sequence)   # the SEQUENCE's format, not the top layer's
    ensureframesize!(player, canvas)
    # the CPU tier of ONE composite (see `composite`): its only job is to hand
    # each layer a decoded frame from that source's ring — everything the
    # picture depends on is shared with the GPU tier and the export
    decoded = function (clip, srcframe)
        sp = pool(player, clip)
        settarget!(sp.worker, srcframe)
        buf = RGBFrame(undef, clip.source.width, clip.source.height)
        deadline = time() + 1.0
        while !fetchframe!(buf, sp.ring, srcframe)
            time() > deadline && return nothing          # not buffered yet → single-clip fallback
            sleep(0.004)
        end
        return buf
    end
    ok = runowned(player) do
        composite(player.engine, clips, n, decoded; canvas = canvas,
                  applytracks = player.applytracks[], playing = player.playing[]) do composed
            copyto!(player.frame[], composed)            # publish the finished composite
        end
    end
    ok || return false
    showcpuframe!(player)
    notify(player.frame)
    return true
end

"GPU preview configured? There is no CPU tier to fall back to — a present during
a long job queues behind it on the worker rather than taking a second path."
gpuready(player::Player) = player.gpupreview isa GPUPreview

"""
Is the playhead PARKED — put there by a click, a seek or a step — rather than
dragged or played? A parked playhead owes the user the EXACT frame; a moving one
is happy with the decoder's nearest stand-in (see the `standin` policy of
[`showframe!`](@ref)).
"""
atrest(player::Player) = !player.playing[] && !player.timeline.scrubbing[]

"""
    editclip(player) -> (clip, source_frame) | nothing

The clip the INSPECTOR works on — sliders, effect stack, keyframes, stabilization:
the SELECTED clip while it spans the playhead, else the topmost clip under the
playhead (what [`locate`](@ref) renders). Selection wins because effects and
keyframes are per CLIP, and stacked lanes must stay reachable: the preview can
only ever show the upper clip, but the one below it still needs its parameters
edited — that is exactly how an opacity fade between two stacked clips is built.
"""
function editclip(player::Player)
    seq = player.sequence
    n = player.playhead[]
    i = player.timeline.selected[]
    if 1 <= i <= length(seq.clips)
        c = seq.clips[i]
        c.start <= n < clipend(c) && return (c, sourceframe(c, n))
    end
    return locate(seq, n)
end

"""
Resolve and show timeline frame `n` if possible (gaps show black). Returns success.

`standin` decides what a still-decoding GPU stream may put on screen. While the
playhead MOVES — playback, a scrub drag — [`frameat!`](@ref)'s nearest already-
decoded frame is exactly the feedback wanted: the picture follows the drag
instead of freezing. While the playhead is PARKED it is not: every retry blits a
closer stand-in, so one click into a cold GOP replays it into the preview (8
stand-ins on a 300-frame GOP, measured) and whichever one came last stays on
screen if the settle is cut short. A parked present therefore only advances the
decode and reports failure; the retry loop lands the exact frame.
"""
function showframe!(player::Player, n::Integer; standin::Bool = !atrest(player))
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
            shown = false
            if gpuready(player) && all(haskey(player.gpucache, c.source) for c in clips)
                # composites mix layers: one stand-in among them dates the whole frame
                standin || primecomposite!(player, clips, n) || return false
                shown = presentgpucomposite!(player, clips, n)
            end
            shown || (shown = compositeframe!(player, n, clips))
            if shown
                # every layer's crop is BAKED INTO the composited canvas, so the
                # view shows the canvas WHOLE — one rule, above the tier split
                # (the GPU tier used to leave the single-clip present's crop on
                # the axis, applying a stabilized clip's framing twice for the
                # length of a blend). The canvas is the SEQUENCE's: measuring it
                # by the top layer left the limits describing one format while
                # the buffer held another as soon as the layers differed in size.
                cw, ch = canvassize(player.sequence)
                player.lastcrop = (0.0, 0.0, 1.0, 1.0)
                coverlimits!(player, 0, cw, ch, 0)
                return true
            end
        end
    end
    loc = locate(player.sequence, n)
    if loc === nothing
        # Straight into the published frame. `composebuf` is sized to the DECODE
        # resolution (`ensuredecodesize!`) and `frame[]` to the canvas, so
        # blacking one and copying it into the other is only ever right when a
        # clip happens to match the canvas — with nothing under the playhead
        # there is no clip to make it so, and a 320x180 decode buffer went into a
        # 214x108 frame.
        fill!(player.frame[], RGB{N0f8}(0, 0, 0))
        showcpuframe!(player)
        notify(player.frame)
        return true
    end
    clip, srcframe = loc
    target, protect = decodetarget(player, n, clip, srcframe)
    return presentclipframe!(player, clip, srcframe; standin, target, protect)
end

"""
    presentclipframe!(player, clip, srcframe; standin, target, protect) -> Bool

Put `clip`'s source frame `srcframe` on screen — the whole present path (GPU
stream, GPU effects over a CPU-decoded frame, or the CPU tier), addressed
DIRECTLY instead of resolved from the playhead. [`showframe!`](@ref) uses it for
the playhead's frame; the trim gesture uses it to show the frame at the edge it
is dragging, which is the frame you are deciding about — without moving the
playhead. `target`/`protect` steer the decode worker (see [`decodetarget`](@ref)).
"""
function presentclipframe!(player::Player, clip::Clip, srcframe::Integer;
                           standin::Bool = !atrest(player),
                           target::Integer = srcframe,
                           protect::UnitRange{Int} = 1:0)
    # PURE-GPU path: a streaming GPU decoder feeds this source — Vulkan-Video decode
    # into a bounded VRAM ring + effects on device, no CPU decode, no upload.
    if gpuready(player) && haskey(player.gpucache, clip.source)
        stream = player.gpucache[clip.source]
        if 0 <= srcframe < nframes(stream)
            # parked on an undecoded frame: advance the feed but leave the last
            # exact image up — the retry loop calls back until this frame lands
            if !standin && !hasframe(stream, srcframe)
                primeframe!(player, stream, srcframe) && return false
            end
            ensureframesize!(player, canvassize(player.sequence))
            eclip = effectiveclip(clip, srcframe)
            if presentgpu!(player, clip, srcframe; stream = stream)
                applycrop!(player)
                # a scrub into an undecoded region shows the nearest decoded frame
                # NOW; returning "not presented yet" keeps the paused retry loop
                # refining in ~90 ms steps until the EXACT frame is on screen
                return hasframe(stream, srcframe)
            end
        end
    end
    sp = pool(player, clip)
    settarget!(sp.worker, target; protect)
    ensuredecodesize!(player, sp.source)   # proxy resolution when one is active
    buf = player.composebuf
    if fetchframe!(buf, sp.ring, srcframe)
        eclip = effectiveclip(clip, srcframe)  # keyframed params sampled at this frame
        ensureframesize!(player, canvassize(player.sequence))
        if gpuready(player)
            if presentgpu!(player, clip, srcframe; source = buf)
                applycrop!(player)
                return true
            end
        end
        # ONE layer, placed into the canvas — the same call the stack takes, so a
        # crop, a reframe and a rotation mean here exactly what they mean in the
        # export. Publishing the raw layer and letting the preview's axis limits
        # stand in for the framing made the framing a viewport: the crop removed
        # nothing you could not scroll back to, and a rotation could not show at
        # all, because axis limits do not rotate.
        ok = runowned(player) do        # the engine's context has ONE owning thread
            composite(player.engine, [clip], n_of(player, clip, srcframe),
                      (_, _) -> buf; canvas = canvassize(player.sequence),
                      applytracks = player.applytracks[], playing = player.playing[]) do canvas
                copyto!(player.frame[], canvas)
            end
        end
        ok === true || return false
        showcpuframe!(player)
        notify(player.frame)
        applycrop!(player)
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
        settarget!(pool(player, nxt).worker, nxt.src_in)
        return (srcframe, 1:0)
    end
    ring = pool(player, clip).ring
    for k in 0:(tail - 1)                                 # tail fully buffered?
        hasframe(ring, srcframe + k) || return (srcframe, 1:0)
    end
    return (nxt.src_in, srcframe:(srcframe + tail - 1))   # protect the tail's slots
end

"Match the preview and effect buffers to the active clip's source resolution."
ensureframesize!(player::Player, source::VideoSource) =
    ensureframesize!(player, (source.width, source.height))

"""
Size the PUBLISHED frame — which is the sequence CANVAS, always.

What the preview shows is the canvas, exactly what the export writes: a clip's
crop, its scale/position and its rotation are baked into it by `placelayer!`. It
used to be the rendered LAYER at source resolution, with the preview axis LIMITS
standing in for the framing — which meant the framing was a viewport, not an
edit. Cropping did not remove anything (zoom out and the "removed" material was
still there), and rotation could not be shown at all, because axis limits do not
rotate.
"""
function ensureframesize!(player::Player, wh::Tuple{Integer, Integer})
    size(player.frame[]) == (Int(wh[1]), Int(wh[2])) && return nothing
    player.frame.val = RGBFrame(undef, Int(wh[1]), Int(wh[2]))  # notified once filled
    player.lastcrop = (-1.0, 0.0, 0.0, 0.0)  # force applycrop! (limits are per-source)
    return nothing
end

"""
Size the DECODE buffer — source resolution (or the proxy's), independent of the
canvas: decoding happens in the material's own pixels and placement follows.
"""
function ensuredecodesize!(player::Player, wh::Tuple{Integer, Integer})
    size(player.composebuf) == (Int(wh[1]), Int(wh[2])) && return nothing
    player.composebuf = RGBFrame(undef, Int(wh[1]), Int(wh[2]))
    return nothing
end
ensuredecodesize!(player::Player, source::VideoSource) =
    ensuredecodesize!(player, (source.width, source.height))

function present!(player::Player)
    if showframe!(player, player.playhead[])
        player.presented += 1
        return true
    end
    player.dropped += 1
    return false
end

"""
While the playhead sits at `n`, keep refining until the EXACT frame is on screen
(each attempt advances the stream's decode) — or until the playhead moves on.
When the frame never arrives inside `budget` seconds the preview shows the
decoder's best and SAYS so, rather than holding a stale image without a word.
"""
function retrypresent(player::Player, n::Integer; budget::Real = 20.0)
    @async begin
        deadline = time() + budget
        while player.playhead[] == n && !player.playing[]
            showframe!(player, n) && (player.presented += 1; break)
            if time() > deadline
                showframe!(player, n; standin = true)
                setstatus!(player, "frame $n never finished decoding — showing the nearest decoded frame")
                break
            end
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

"""
    fxselected(player) -> key | nothing

Which inspector card is selected. `nothing` when none is — the resting state, and
what clicking the selected card returns to.
"""
fxselected(player::Player) = get(player.fxwidgets, :fxselected, nothing)

"""
Select an inspector card, or deselect it when it already was.

Rebuilds the stack so the header shows it, and re-runs the preview overlays: a
card's selection is what decides whether its direct-editing tool is on the
picture.
"""
function selectfxcard!(player::Player, key)
    player.fxwidgets[:fxselected] = fxselected(player) == key ? nothing : key
    r = get(player.fxwidgets, :fxlistrefresh, nothing)
    r === nothing || r(force = true)
    loc = editclip(player)
    showtransformgizmo!(player, loc === nothing ? nothing : loc[1])
    return nothing
end

"The timeline frame at which `clip` shows `srcframe` (the inverse of `sourceframe`)."
n_of(::Player, clip::Clip, srcframe::Integer) =
    clip.start + timelineframes(clip, Int(srcframe) - clip.src_in)

"""
Frame the preview on the CANVAS — the whole of it, every time.

There is nothing left for the limits to express: the crop, the reframe and the
rotation are baked into the published frame by `placelayer!`, exactly as the
export bakes them. The limits used to BE the framing, which is what made a crop a
viewport instead of an edit.
"""
function applycrop!(player::Player)
    W, H = size(player.frame[])
    (W, H) == player.lastcanvas && return nothing
    player.lastcanvas = (W, H)
    coverlimits!(player, 0, W, H, 0)
    return nothing
end
applycrop!(player::Player, ::Clip) = applycrop!(player)

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
        # HANDS ON THE RULER WIN: while the button is down on the timeline the user
        # is placing the playhead, so playback holds its clock instead of dragging
        # it away under the cursor (it used to run off between mouse moves, which
        # read as "the playhead can't be moved while playing"). Release resumes
        # from wherever it was put.
        if player.timeline.presspick !== nothing
            base = player.playhead[]
            t0 = time_ns()
            lastset = base
            sleep(0.003)
            continue
        end
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
    # Playback presents stand-ins to keep moving (frameat!'s latency budget), and
    # nothing retries while playing. The moment it stops — Space, K, a GPU render
    # error, the end of the sequence — the screen owes the frame the playhead is ON.
    retrypresent(player, player.playhead[])
    return nothing
end

Base.seek(player::Player, t::Real) =
    (player.playhead[] = clamp(round(Int, t * player.sequence.framerate), 0,
                               max(seqlength(player.sequence) - 1, 0)); player)

# ---------------------------------------------------------------- editing

"Split / delete / crop-reset operate on the clip under the playhead."
function split!(player::Player)
    loc = editclip(player)          # the SELECTED clip, else the one under the playhead
    loc === nothing && return setstatus!(player, "nothing to split at the playhead")
    snapshot!(player)
    if split!(player.sequence, player.playhead[], loc[1].track) === nothing
        pop!(player.undostack)
        return setstatus!(player, "cannot split here — the playhead is at the clip's start")
    end
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
    lc, rc = transitionclips(seq, at)
    t = (lc === nothing || rc === nothing) ? nothing :
        addtransition!(seq, at; duration = defaultdissolve(seq, lc, rc))
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
(undoable). Resolution and framerate may differ from the sequence — a source at
another rate is conformed (see [`placesource!`](@ref)). Also triggered by
dropping video files onto the editor window.
"""
function addsource!(player::Player, path::AbstractString)
    source = VideoSource(path)
    seq = player.sequence
    rate = conformrate(source, seq.framerate)
    snapshot!(player)
    clip = Clip(source, 0, source.nframes, seqlength(seq), (0.0, 0.0, 1.0, 1.0), rate)
    push!(seq.clips, clip)
    registermedia!(player, source)
    limits!(player.timeline.axis, 0.0, seqduration(seq), 0.0, 1.0)  # reveal the new clip
    refreshedit!(player)
    note = conformnote(source, seq.framerate)
    setstatus!(player, "added $(basename(path)) — $(source.width)×$(source.height), " *
                       "$(round(source.nframes / source.framerate, digits = 1))s" *
                       (isempty(note) ? "" : " · " * note))
    needsproxy(source; maxpixels = player.proxythreshold) && startproxy!(player, source)
    return clip
end

"The player's default project file: next to the first source."
projectfile(player::Player) =
    splitext(player.sequence.clips[1].source.path)[1] * ".videoedit.json"

"""
Fill the Project dock: what this edit is called, how to save or open one, and
every kept version as a row you can go back to.

A LIST, one row per version, each with its own action — the same shape the other
panels use, because "restore" belongs to a particular version and not to a
button somewhere else.
"""
function buildprojectpanel!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; valign = :top)
    Label(panel[1, 1], "Project"; font = :bold, halign = :left, tellwidth = false)
    namelbl = Label(panel[2, 1], ""; halign = :left, fontsize = 11,
                    color = uicolors.text_muted, tellwidth = false)
    actions = GridLayout(panel[3, 1])
    savebtn = Button(actions[1, 1]; label = "Save", tellwidth = false)
    openbtn = Button(actions[1, 2]; label = "Open…", tellwidth = false)
    recovergl = GridLayout(panel[4, 1])
    rows = GridLayout(panel[5, 1])
    colsize!(panel, 1, Makie.Relative(1.0))
    built = Any[]

    function refresh()
        foreach(b -> (try Makie.delete!(b) catch end), built); empty!(built)
        Makie.trim!(recovergl); Makie.trim!(rows)
        isempty(player.sequence.clips) && (namelbl.text[] = "no clips yet"; return nothing)
        path = projectfile(player)
        namelbl.text[] = basename(path) * (isfile(path) ? "" : "  ·  not saved yet")
        rec = recoverable(player)
        if rec !== nothing
            l = Label(recovergl[1, 1], "a newer autosave exists"; halign = :left, fontsize = 11,
                      color = uicolors.select, tellwidth = false)
            b = Button(recovergl[1, 2]; label = "Recover", tellwidth = false)
            on(_ -> restoreproject!(player, rec), b.clicks)
            append!(built, (l, b))
        end
        cps = reverse(projectcheckpoints(path))     # newest first
        if isempty(cps)
            l = Label(rows[1, 1], "no saved versions yet"; halign = :left, fontsize = 11,
                      color = uicolors.text_muted, tellwidth = false)
            push!(built, l)
        else
            for (i, cp) in enumerate(cps)
                age = round(Int, (time() - mtime(cp)) / 60)
                l = Label(rows[i, 1], Dates.format(Dates.unix2datetime(mtime(cp)), "HH:MM:SS") *
                                      "  ·  " * (age < 1 ? "just now" : "$(age) min ago");
                          halign = :left, fontsize = 11, tellwidth = false)
                b = Button(rows[i, 2]; label = "restore", tellwidth = false, fontsize = 11)
                on(_ -> restoreproject!(player, cp), b.clicks)
                append!(built, (l, b))
            end
            colsize!(rows, 1, Makie.Auto(false, 1.0))
        end
        return nothing
    end

    on(_ -> (saveproject!(player); refresh()), savebtn.clicks)
    # the native dialog BLOCKS, so it runs off the UI thread and comes back
    # through the queue — the same shape the media bin's browse uses
    on(openbtn.clicks) do _
        Threads.@spawn try
            f = Makie.choose_file_dialogue()
            f === nothing && return nothing
            put!(player.uiqueue, () -> restoreproject!(player, String(f)))
        catch e
            setstatus!(player, "file dialog failed: $(sprint(showerror, e))")
        end
    end
    on(_ -> refresh(), player.playhead)     # the name follows the clip under the cursor
    refresh()
    player.fxwidgets[:projectrefresh] = refresh
    return panel
end

"""
Load a saved version over the running edit.

Snapshot first: going back to a checkpoint is an edit like any other, and Ctrl+Z
has to undo it. The file is READ before anything is thrown away, so a corrupt
checkpoint leaves the session alone.
"""
function restoreproject!(player::Player, file::AbstractString)
    seq = try
        loadproject(file)
    catch e
        setstatus!(player, "could not read $(basename(file)): $(briefly(e))")
        return nothing
    end
    snapshot!(player)
    empty!(player.sequence.clips); append!(player.sequence.clips, seq.clips)
    empty!(player.sequence.transitions); append!(player.sequence.transitions, seq.transitions)
    empty!(player.sequence.overlays); append!(player.sequence.overlays, seq.overlays)
    postrestore!(player)
    r = get(player.fxwidgets, :projectrefresh, nothing); r === nothing || r()
    setstatus!(player, "restored $(basename(file)) — Ctrl+Z puts the edit back")
    return nothing
end

"""
    autosavepath(player) -> String

Where the running edit is parked: a dotfile beside the project (or beside the
first source, before there is a project). Never the project itself — an autosave
must not be able to overwrite a file the user chose to write.
"""
function autosavepath(player::Player)
    p = projectfile(player)
    return joinpath(dirname(p), "." * basename(p) * ".autosave.json")
end

"""
    autosave!(player; force = false) -> path | nothing

Park the edit if it changed since the last time. Cheap enough to call on a timer:
without an edit it returns immediately, and it never checkpoints (an autosave is
not a version, it is a safety net).
"""
function autosave!(player::Player; force::Bool = false)
    isempty(player.sequence.clips) && return nothing
    (force || player.edits != player.autosaved) || return nothing
    path = autosavepath(player)
    try
        saveproject(path, player.sequence; checkpoint = false)
        player.autosaved = player.edits
        return path
    catch e
        @warn "autosave failed" exception = e
        return nothing
    end
end

"""
Run the autosave on a timer for the life of the window.

`every` seconds, and only when something changed. The task dies with the screen,
so closing the editor does not leave it writing.
"""
function startautosave!(player::Player; every::Real = 30)
    @async while player.screen !== nothing && isopen(player.screen)
        sleep(every)
        try
            path = autosave!(player)
            path === nothing || @debug "autosaved" path
        catch e
            @warn "autosave loop" exception = e
        end
    end
    return nothing
end

"""
    recoverable(player) -> String | nothing

An autosave NEWER than the project it belongs to — what is left after a crash.
The caller says so; taking it silently would be its own kind of data loss.
"""
function recoverable(player::Player)
    a = autosavepath(player); p = projectfile(player)
    isfile(a) || return nothing
    (!isfile(p) || mtime(a) > mtime(p)) && return a
    return nothing
end

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
    player.autosaved = player.edits
    rm(autosavepath(player); force = true)   # the project IS the newest copy again
    n = length(projectcheckpoints(path))
    setstatus!(player, "project saved — $path" * (n > 0 ? " · $n checkpoint(s)" : ""))
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
    setmotiontrack!(clip, nothing)
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
    # at the playhead this is the inspector acting → same clip the panel shows
    # (the selected one on stacked lanes); an explicit `at` addresses the frame
    loc = at == player.playhead[] ? editclip(player) : locate(player.sequence, at)
    if loc === nothing
        setstatus!(player, "$what: no clip under the playhead — move it onto a clip first")
        return nothing
    end
    clip = loc[1]
    setstatus!(player, "$what: analyzing $(srclength(clip)) frames…")
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
    maxs = maxseconds > 0 ? Float64(maxseconds) : 0.9 * srclength(clip) / fps
    setstatus!(player, "make loop: matching frames across $(srclength(clip)) frames…")
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
    # lead with what it BOUGHT (measured while analyzing), not just how hard it pushes
    got = track.reduction > 0 ? "flicker −$(round(Int, 100 * track.reduction))%, " : ""
    return "$(got)max adjustment $(round(Int, 100 * dev))%"
end

"Re-present the playhead frame and repaint the timeline after an edit."
function refreshedit!(player::Player)
    relayout!(player.timeline)
    ensurestreams!(player)
    notify(player.playhead)
    player.playing[] || retrypresent(player, player.playhead[])
    return nothing
end

"""
Open a GPU stream for every source the sequence shows and that has none yet —
an edit can bring in a source that was not there when the player started (a
clip dragged in from the bin), and its layers would otherwise fall back to CPU
decode forever. Cheap: one `haskey` per source, the open runs off the UI thread.
"""
function ensurestreams!(player::Player)
    player.gpupreview isa GPUPreview || return nothing
    for source in unique(c.source for c in player.sequence.clips)
        haskey(player.gpucache, source) || Threads.@spawn preloadgpu!(player, source)
    end
    return nothing
end

"""
A canvas point in the clip's MATTE space: normalized within its cropped layer.

`previewtosource`'s sibling, minus the stabilization inverse — the matte is
computed on the rendered (already stabilized, already cropped) frame, so the
click needs the crop fit undone and nothing else.
"""
function previewtomatte(player::Player, clip::Clip, srcframe::Integer, p)
    layer  = (clip.source.width, clip.source.height)
    lclip  = effectiveclip(clip, srcframe)
    q = previewtolayer(player, lclip, layer, p)
    cw, ch = mattelayersize(clip)
    return (clamp((q[1] - lclip.crop[1] * layer[1]) / cw, 0.0, 1.0),
            clamp((q[2] - lclip.crop[2] * layer[2]) / ch, 0.0, 1.0))
end

"The inverse of `previewtomatte`: a normalized matte point in canvas pixels."
function mattetopreview(player::Player, clip::Clip, srcframe::Integer, n)
    layer  = (clip.source.width, clip.source.height)
    lclip  = effectiveclip(clip, srcframe)
    cw, ch = mattelayersize(clip)
    sx = n[1] * cw + lclip.crop[1] * layer[1]
    sy = n[2] * ch + lclip.crop[2] * layer[2]
    return layertopreview(player, lclip, layer, (sx, sy))
end

"""
A preview data coordinate, in the clip's LAYER pixels.

The preview axis plots `player.frame[]` and indexes it in its own pixels, so what
a data coordinate means depends on what that buffer holds. On the single-clip
path it holds the rendered LAYER at source resolution and the axis LIMITS do the
cropping — a data coordinate is already a layer pixel, and putting it through
`layermatrix` crops it a second time. (That is what sent a click on the upper
bird into the middle of the nest box: the error is zero at the centre of the
frame and grows toward its edges, so the bug hid until somebody clicked near an
edge.) On the composite path the buffer is the sequence canvas, and then the
matrix is exactly right.
"""
function previewtolayer(player::Player, lclip::Clip, layer::Tuple{Int, Int}, p)
    canvas = size(player.frame[])
    canvas == layer && return (Float64(p[1]), Float64(p[2]))
    q = layermatrix(lclip, layer, canvas) * Vec3f(p[1], p[2], 1)
    return (Float64(q[1]), Float64(q[2]))
end

"The inverse of [`previewtolayer`](@ref)."
function layertopreview(player::Player, lclip::Clip, layer::Tuple{Int, Int}, q)
    canvas = size(player.frame[])
    canvas == layer && return (Float64(q[1]), Float64(q[2]))
    r = inv(layermatrix(lclip, layer, canvas)) * Vec3f(q[1], q[2], 1)
    return (Float64(r[1]), Float64(r[2]))
end

"""
    previewtosource(player, clip, srcframe, p) -> (x, y)

Where a click in the preview lands, in the clip's SOURCE pixels.

Two hops, and both are *sampling* matrices, so this is a forward multiply rather
than an inversion. `layermatrix` maps a canvas pixel to the layer pixel that was
drawn there — the clip's crop, fitted whole, plus its reframe, letterbox bars
included. The stabilization transform maps a displayed pixel to the source pixel
it was sampled from, which is the same direction. Composing them is the mapping.

What this replaces was `scale = clip.source.width / size(player.frame[], 1)`,
i.e. "the canvas is the source, scaled". That holds only for an uncropped,
unreframed, unstabilized clip on a full-width canvas. Anywhere else the click
landed somewhere other than where the user pointed, and on a stabilized clip it
was wrong by a different amount on every frame — the transform is per frame.
"""
function previewtosource(player::Player, clip::Clip, srcframe::Integer, p)
    canvas = size(player.frame[])
    # Source pixels, not the layer's: `fitmatrix` depends on the input size only
    # through the crop rect's aspect ratio, and a proxy scales both extents
    # together. Feeding the original size therefore lands directly in source
    # pixels and makes the whole mapping proxy-independent — which matters,
    # because a proxy swap replaces the decode pool, not `clip.source`.
    layer = (clip.source.width, clip.source.height)
    lclip = effectiveclip(clip, srcframe)   # crop/reframe can be keyframed
    q = layermatrix(lclip, layer, canvas) * Vec3f(p[1], p[2], 1)   # canvas -> source
    track = clip.motiontrack
    if track !== nothing
        i = srcframe - track.src_in + 1
        if 1 <= i <= length(track.transforms)
            # stored in source pixels, which is what `q` now is
            q = track.transforms[i] * Vec3f(q[1], q[2], 1)         # shown -> pre-stab
        end
    end
    return (q[1], q[2])
end

"""
    sourcetopreview(player, clip, srcframe, s) -> (x, y)

Where a source pixel shows up on the canvas — [`previewtosource`](@ref) the other
way round, so an overlay drawn from source coordinates lands on the pixel it
describes even after a reframe, a crop or a stabilization warp.

Inverting here rather than keeping a second forward mapping: two mappings would
be two things to keep in step, and the one that drifts is always the one the user
sees.
"""
function sourcetopreview(player::Player, clip::Clip, srcframe::Integer, s)
    canvas = size(player.frame[])
    layer = (clip.source.width, clip.source.height)
    q = Vec3f(s[1], s[2], 1)
    track = clip.motiontrack
    if track !== nothing
        i = srcframe - track.src_in + 1
        1 <= i <= length(track.transforms) && (q = inv(track.transforms[i]) * q)
    end
    r = inv(layermatrix(effectiveclip(clip, srcframe), layer, canvas)) * q
    return (r[1], r[2])
end

"Arm a one-shot preview click that picks the subject to lock onto."
function startobjectpick!(player::Player)
    player.onpick = p -> analyzeat!(player,
        (clip; kwargs...) -> begin
            srcframe = sourceframe(clip, player.playhead[])
            analyzeobject!(clip, previewtosource(player, clip, srcframe, p);
                           backend = player.analysisbackend, kwargs...)
        end,
        "object lock")
    setstatus!(player, "object lock: click the subject in the preview (Esc cancels)")
    return nothing
end

function wirecroptool(player::Player)
    ax = player.previewaxis
    # Every handler over the preview asks first whether the pointer is still its
    # to take. An overlay scene that claims it (`captures_mouse`, e.g. the matte
    # marking) turns this false, so a click lands in exactly one place instead of
    # activating a pick AND dropping a crop anchor AND scrubbing.
    mine() = Makie.receives_events(ax.scene)
    on(events(ax.scene).mousebutton) do event
        mine() || return Consume(false)
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
        mine() || return Consume(false)
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
        mine() || return Consume(false)
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
    player.croprect[] = Point2f[]   # clear the in-progress rectangle; stay active
    # Persistent crop tool: stays active so you can re-drag to refine (Esc to put away).
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
            f = get(player.fxwidgets, :paletteopen, nothing)   # Ctrl+P: the command palette
            f === nothing || f()
        elseif event.key == Keyboard.s && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            saveproject!(player)
        elseif event.key == Keyboard.s && ispress
            split!(player)
        elseif (event.key == Keyboard.x || event.key == Keyboard.delete) && ispress
            deleteat!(player)
        elseif event.key == Keyboard.c && ispress
            usetool!(player, :crop)
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
            # While a selection is being marked, Ctrl+Z means "take that point
            # back" — the thing the user just did. Letting it fall through to the
            # project undo edits the TIMELINE instead, which is both surprising
            # and hard to notice: the clip moves behind a preview you are staring
            # at for a matte.
            col = mattecollect(player)
            if col !== nothing && !shift
                dropmattepoint!(col)
            else
                shift ? redo!(player) : undo!(player)
            end
        elseif event.key == Keyboard.enter && ispress && mattecollect(player) !== nothing
            # Enter is the "I am done marking" gesture; it must beat every other
            # Enter binding while a selection is open, hence the guard up here.
            finishmattecollect!(mattecollect(player))
        elseif event.key == Keyboard.backspace && ispress && mattecollect(player) !== nothing
            dropmattepoint!(mattecollect(player))
        elseif event.key == Keyboard.escape && ispress
            col = mattecollect(player)
            if col !== nothing
                cancelmattecollect!(col)
                return Consume(true)
            end
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
    # A VISIBLE scrollbar and a usable wheel step. Makie's defaults are a fully
    # transparent track (`scrollbar_color = RGBAf(0,0,0,0)`) and 15 px per notch,
    # which reads as "this panel does not scroll" right up until you notice the
    # content is cut off — and then takes a dozen notches to get anywhere.
    c = player.timeline.colors
    sf = Subfigure(player.fig[1, 2]; visible = false, contentpadding = 8,
                   scroll_speed = 70,
                   scrollbar_size = 9,
                   scrollbar_color = (c.background, 0.55),
                   scrollbar_thumb_color = Makie.lerp_oklab(RGBf(Makie.to_color(c.background)),
                                                            RGBf(1, 1, 1), 0.34),
                   scrollbar_thumb_color_active = c.accent)
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
background), for the active blade tool. Hotspot returned alongside."
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
            shape === :resize   ? GLFW.CreateStandardCursor(GLFW.RESIZE_NWSE_CURSOR) :
            shape === :move     ? GLFW.CreateStandardCursor(GLFW.RESIZE_ALL_CURSOR) :
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
Start using tool `t` (`:split` or `:crop`), or `:none` to put it away — asking
for the tool already in use puts it away too. A tool in use changes the cursor
to a crosshair and acts where you click or drag (split cuts at the clicked
timeline position, not at the playhead).
"""
function usetool!(player::Player, t::Symbol)
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
onto the "+ new track" strip. The head of the panel is the drop zone: files
dropped anywhere on the window import here (see [`importsources!`](@ref)), and
clicking it opens the native file dialog.
"""
function buildmediabin!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1], "Media"; font = :bold, halign = :left, tellwidth = false)
    # ONE import target instead of a button: files dropped from the file manager
    # land here (any number at once), and a click opens the browse dialog. A
    # drop zone you can see beats a button that hides where dropping is allowed.
    dropstatus = Observable("")     # "" = idle, else the running import's line
    dropaccent = lift(s -> isempty(s) ? uicolors.border : uicolors.accent, dropstatus)
    droparea = Box(panel[2, 1]; height = 60, cornerradius = 6, linestyle = :dash,
                   strokewidth = 1.5, color = uicolors.surface_subtle,
                   strokecolor = dropaccent)
    Label(panel[2, 1], lift(s -> isempty(s) ? "Drop clips here\nor click to browse" : s,
                            dropstatus);
          fontsize = 12, justification = :center, color = dropaccent,
          tellwidth = false, tellheight = false)
    Label(panel[3, 1], "drag a clip onto a lane · top strip = new track";
          fontsize = 11, halign = :left, color = uicolors.text_muted, tellwidth = false)
    rows = GridLayout(panel[4, 1])
    colsize!(panel, 1, Makie.Relative(1.0))
    player.fxwidgets[:dropstatus] = dropstatus
    player.fxwidgets[:droparea] = droparea
    # the browse dialog behind a hook: it is a BLOCKING native window, so tests
    # (the random event storm clicks everywhere) replace it with a no-op
    player.fxwidgets[:browse] = () -> Threads.@spawn try
        path = Makie.choose_file_dialogue()
        path === nothing || importsources!(player, [String(path)])
    catch e
        setstatus!(player, "import failed: $(sprint(showerror, e))")
    end
    # files dropped onto the WINDOW go to the bin, wherever they land — the dock
    # opens itself so the new rows are where the user is looking
    on(events(player.fig).dropped_files) do paths
        isempty(paths) && return
        player.dockopen[] === :media || opendock!(player, :media)
        importsources!(player, paths)
        return
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
            src = player.dragsource
            # the ghost must show what will LAND: a conformed clip keeps its
            # wall-clock duration but occupies a different number of timeline
            # frames. Measuring it in sequence frames drew a 9.3 s clip as 4.7 s.
            nfr = floor(Int, src.nframes / conformrate(src, seq.framerate))
            # target the hovered lane; if that spot is taken, the ghost snaps to
            # the first free lane above (stacking), never silently to the end
            want = trackat(pos[2], ntr)
            track = want == 0 ? 0 :        # 0 = the zone BELOW the bottom lane
                    freetrack(seq, round(Int, t * seq.framerate), nfr, want)
            droptrack[] = track
            dur = nfr / seq.framerate
            n2 = max(ntr, track)
            g = min(0.02, trackspan(n2) * 0.15)
            lo, hi = track == 0 ? (0.008, TRACKBASE - 0.013) : trackband(track, n2)
            lo += g; hi -= g
            dropghost[] = Rect2f(t, lo, dur, hi - lo)
            dropvis[] = true
            if track > ntr || track == 0
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
        # …but only when nothing is drawn over the bin (a modal, a dropdown);
        # the RELEASE is checked against the timeline instead, so it stays out
        # of this guard — the pointer is over the timeline by then
        if event.action == Mouse.press && player.dockopen[] === :media &&
           Makie.receives_events(player.dockpanels[:media].sf.scene)
            if mp in droparea.layoutobservables.computedbbox[]
                player.fxwidgets[:browse]()
                return Consume(true)
            end
            for ((btn, box, _), src) in zip(player.binrows, player.mediasources[])
                if mp in btn.layoutobservables.computedbbox[] ||
                   mp in box.layoutobservables.computedbbox[]
                    player.dragsource = src
                    player.timeline.dragactive[] = true
                    setcursor!(player, :hand)
                    # the chip carries the retime, so the conform is visible BEFORE
                    # the drop rather than after it (or, as it used to be, never)
                    note = conformnote(src, player.sequence.framerate)
                    draglabel[] = isempty(note) ? basename(src.path) :
                                  basename(src.path) * "  ·  " *
                                  "$(round(src.framerate; digits = 2)) → $(round(player.sequence.framerate; digits = 2)) fps"
                    dragvis[] = true
                    moveghost(mp)
                    setstatus!(player, "drop $(basename(src.path)) on a timeline lane — or on “+ new track”, above the stack or below it" *
                                       (isempty(note) ? "" : " · will be " * note))
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

"Text on the bin's drop zone: empty is the idle invitation, anything else the running import."
function dropstate!(player::Player, text::AbstractString)
    o = get(player.fxwidgets, :dropstatus, nothing)
    o === nothing || put!(player.uiqueue, () -> (o[] = String(text)))
    return nothing
end

"What the status line says after an import — every file is accounted for."
function importsummary(added::Vector{VideoSource}, duplicates::Int, failed::Vector{String})
    parts = String[]
    isempty(added) || push!(parts, length(added) == 1 ?
        "imported $(basename(added[1].path))" : "imported $(length(added)) clips")
    duplicates == 0 || push!(parts, "$duplicates already in the bin")
    isempty(failed) || push!(parts, "couldn't open $(join(failed, ", "))")
    isempty(parts) && return "nothing to import"
    return join(parts, " · ") * (isempty(added) ? "" : " — drag one onto a lane")
end

"""
    importsources!(player, paths)

Add every file in `paths` to the media bin — this is what a drop from the file
manager (any number of files at once) and the browse dialog both run. Opening a
source probes the whole container, so the work happens off the UI thread: rows
appear one by one, the drop zone and the footer show the progress, and a file
that won't open names itself in the status line without stopping the others.
"""
function importsources!(player::Player, paths)
    files = unique(String[String(p) for p in paths])
    isempty(files) && return nothing
    n = length(files)
    dropstate!(player, "importing 1/$n…")
    Threads.@spawn try
        added = VideoSource[]
        failed = String[]
        duplicates = 0
        for (i, path) in enumerate(files)
            dropstate!(player, "importing $i/$n…")
            player.jobprogress[] = (i - 1) / n
            if isdir(path)
                push!(failed, basename(path) * " (a folder)")
                continue
            end
            source = try
                VideoSource(path)
            catch
                push!(failed, basename(path))
                continue
            end
            if any(s -> s.path == source.path, player.mediasources[])
                duplicates += 1
            else
                push!(added, source)
                put!(player.uiqueue, () -> registermedia!(player, source))
            end
        end
        player.jobprogress[] = NaN
        dropstate!(player, "")
        setstatus!(player, importsummary(added, duplicates, failed))
    catch e
        player.jobprogress[] = NaN
        dropstate!(player, "")
        setstatus!(player, "import failed: $(sprint(showerror, e))")
    end
    return nothing
end

"""
    conformnote(source, framerate) -> String

What conforming this source will do, for the drag chip and the status line — or
`""` when it runs at the sequence rate and nothing happens to it. A retime that
is never mentioned is a retime the user later finds by wondering why the picture
stutters.
"""
function conformnote(source::VideoSource, framerate::Real)
    r = conformrate(source, framerate)
    r == 1.0 && return ""
    fps(x) = string(round(x; digits = x == round(x) ? 0 : 2))
    how = r < 1 ? "each frame held $(round(1 / r; digits = 1))×" :
                  "every $(round(r; digits = 1))ᵗʰ frame kept"
    return "conformed $(fps(source.framerate)) → $(fps(framerate)) fps ($how)"
end

"""
Place `source` as a new clip starting at frame `at` on `track` (undoable) —
dropping above the top lane creates a new track, whose clips composite over the
ones below (multi-track). A drop onto an occupied spot stacks on the first free
lane above; the status line says which happened.

A source at a different framerate is CONFORMED to the sequence rather than
refused (see [`Clip`](@ref)'s `rate`): its duration is preserved and frames are
held or dropped to fit. Refusing it was a dead end that only announced itself in
the status line — the bin had accepted the file, the drag ghost had shown a
valid lane, and then nothing landed.
"""
function placesource!(player::Player, source::VideoSource, at::Integer; track::Integer = 1)
    seq = player.sequence
    rate = conformrate(source, seq.framerate)
    start = max(Int(at), 0)
    len = floor(Int, source.nframes / rate)   # TIMELINE frames this clip will occupy
    # if the wanted lane is occupied there, stack on the first free lane above —
    # the drop always lands where the drag ghost showed it
    # track 0 = the zone BELOW the bottom lane: make room underneath and land there
    if Int(track) == 0
        # nothing LEFT a lane here — a clip is being added — so no compaction:
        # compacting before the new clip exists would undo the push it needs
        snapshot!(player)
        pushtracksup!(seq)
        track = 1
    else
        track = freetrack(seq, start, len, clamp(Int(track), 1, ntracks(seq) + 1))
    end
    note = conformnote(source, seq.framerate)
    where = if track > ntracks(seq) && ntracks(seq) > 0 && !isempty(seq.clips)
        "placed $(basename(source.path)) on NEW track V$track — it composites over the tracks below"
    else
        "placed $(basename(source.path)) at $(timestring(start / seq.framerate))" *
        (track > 1 ? " on track V$track" : "")
    end
    setstatus!(player, isempty(note) ? where : where * " · " * note)
    snapshot!(player)
    clip = Clip(source, 0, source.nframes, start, (0.0, 0.0, 1.0, 1.0), rate)
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
    # One visibility Observable PER PARAMETER, not one global flag: the keyframe
    # legend in the Effects panel owns these, so a click on an entry hides that
    # one curve. `kfvisible` is the initial state and the toggle-all target.
    function ensureplot!(key)
        get!(curveplots, key) do
            pts = Observable(Point2f[])
            vis = Observable(player.kfvisible[])
            halo = lines!(ax, pts; color = (:black, 0.55), linewidth = 4.0,
                          visible = vis)              # dark under-halo: keeps the
            translate!(halo, 0, 0, 3)                 # curve readable on busy thumbs
            pl = lines!(ax, pts; color = paramcolor(key), linewidth = 2.0, visible = vis,
                        label = paramspec(key).label)
            translate!(pl, 0, 0, 4)   # above the thumbnails, below the playhead line
            on(_ -> refresh(), vis)   # markers follow their curve
            (plot = pl, pts = pts, halo = halo, visible = vis)
        end
    end
    function refresh()
        active = Set{Symbol}()
        for clip in seq.clips, k in keys(clip.animations)
            push!(active, k)
        end
        for key in active
            pts = ensureplot!(key).pts
            p = paramspec(key)
            segs = Point2f[]
            for clip in seq.clips
                c = get(clip.animations, key, nothing)
                c === nothing && continue
                x0 = clip.start / fps; x1 = clipend(clip) / fps
                for i in 0:60
                    s = x0 + (x1 - x0) * i / 60
                    sf = sourceframe(clip, round(Int, s * fps))
                    v = something(valueat(c, sf), p.get(clip))
                    push!(segs, Point2f(s, yat(clip, p, v)))
                end
                push!(segs, Point2f(NaN, NaN))   # break between clips
            end
            pts[] = segs
        end
        for (key, c) in curveplots              # empty curves for params no longer animated
            key in active || isempty(c.pts[]) || (c.pts[] = Point2f[])
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
                # a hidden curve has hidden markers: the legend entry governs both
                cp = get(curveplots, key, nothing)
                cp === nothing || cp.visible[] || continue
                p = paramspec(key)
                focused = key === player.kffocus[]
                col = RGBAf(Makie.to_color(paramcolor(key)))
                for (i, k) in enumerate(c.keys)
                    clip.src_in <= k.frame <= clip.src_out || continue
                    pt = Point2f(timelineframe(clip, k.frame) / fps,
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
    player.fxwidgets[:kfcurves] = curveplots   # the legend in the Effects panel reads these
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
        anycurvevisible(player) || return Consume(false)   # hidden curves are inert
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
                sf = sourceframe(clip, n)
                key = nearestcurve(clip, sf, y)
                if key === nothing                       # not aiming at any curve
                    setstatus!(player, isempty(clip.animations) ?
                        "no animated parameter here — turn one on with its ◆ in the Inspector first" :
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
            kfmenu.title = "◆ $(paramspec(key).label) · $(timestring(timelineframe(clip, k.frame) / fps))"
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
        f = clamp(sourceframe(clip, timelineframe(tl, t)), clip.src_in, clip.src_out)
        # snap to the playhead when close (Premiere-style) — RAW pixel distance, not
        # the frame-quantized one (a frame can already be wider than the threshold)
        vp = ax.scene.viewport[]; (x0, x1) = tl.viewrange[]
        pxpersec = max(vp.widths[1], 1) / max(x1 - x0, 1.0e-9)
        phf = clamp(playheadframe(player, clip), clip.src_in, clip.src_out)
        pht = timelineframe(clip, phf) / fps
        abs(t - pht) * pxpersec < 12 && (f = phf)
        v = yval(clip, p, y)
        movekey!(c, i, f, v)
        j = findfirst(k -> k.frame == f, c.keys); j === nothing || (dragref[] = (c, j, clip, p))
        dragtippos[] = Point2f(timelineframe(clip, f) / fps, yat(clip, p, v))
        dragtiptext[] = "$(p.label)  $(round(v; digits = 2)) · $(timestring(timelineframe(clip, f) / fps))"
        dragtip.visible = true
        notify(player.playhead); return Consume(true)
    end

    player.fxwidgets[:kfoverlay] = curveplots
    player.fxwidgets[:kfrefresh] = refresh
    return curveplots
end

# The Effects panel lives in fxpanel.jl and the command palette in palette.jl —
# one card list for the selected clip, one menu, one search box. What stood
# here was an "Inspector" that listed effects while a Tools dock listed tools,
# with a second card builder and a second set of widget helpers behind it.

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
        # THE SAME call the X key makes. Deleting straight from the menu took no
        # undo snapshot, so the delete was not undoable — and because no snapshot
        # was pushed, Ctrl+Z reached past it to the edit before (the split), and
        # redo then re-did THAT. "Deleting a cut clip cannot be redone."
        ("Delete clip (ripple)", "X", () -> begin
            player.playhead[] = rcframe()
            deleteat!(player)
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
playheadframe(player::Player, clip::Clip) = sourceframe(clip, player.playhead[])

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
    loc = editclip(player)
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
function startanimating!(player::Player, key::Symbol)
    loc = editclip(player)
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
    kfcurves(player) -> Dict{Symbol, NamedTuple}

The keyframe curve plots on the timeline, by parameter key. Each entry carries
its own `visible` Observable — that is what a legend entry toggles.
"""
kfcurves(player::Player) = get(player.fxwidgets, :kfcurves, Dict{Symbol, Any}())

"Whether ANY keyframe curve is currently shown (hidden curves take no clicks)."
anycurvevisible(player::Player) = any(c -> c.visible[], values(kfcurves(player)))

"""
    showcurves!(player, on) -> Bool

Show or hide every keyframe curve at once — what the legend's right-click does,
and what the palette command reaches. Returns what it set them to.
"""
function showcurves!(player::Player, on::Bool)
    player.kfvisible[] = on
    for c in values(kfcurves(player))
        c.visible[] == on || (c.visible[] = on)
    end
    return on
end

"""
The ◆ of the inspector's ◀◆▶ trio, Premiere semantics: start animating the parameter if it
isn't animated yet (first key = current value); otherwise ADD a key at the
playhead — or REMOVE the one sitting there (removing the last key makes the
parameter static again).
"""
function togglekey!(player::Player, key::Symbol)
    loc = editclip(player)
    loc === nothing && return nothing
    clip = loc[1]
    clipanimated(clip, key) || return startanimating!(player, key)
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
    loc = editclip(player)
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
    seek!(player, clamp(timelineframe(clip, k.frame), 0, seqlength(player.sequence) - 1))
    return nothing
end

"Clear every keyframe of the focused parameter on the clip under the playhead
(the slider returns to its static value; undoable)."
function clearkeyframes!(player::Player)
    loc = editclip(player)
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
