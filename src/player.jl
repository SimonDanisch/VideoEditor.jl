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
    # keyed by `readerkey`: a VideoSource when clips of it can share one read
    # head, a clip id when they are too far apart to (see there)
    const pools::Dict{Any, SourcePool}
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
    # clip id -> frames repaired by hand after propagation. Separate from the
    # marks because they are a different KIND of statement: a mark is an input the
    # propagator runs from, a repair is an output that overrules what it produced.
    # `runmatte!` re-applies these after every full analysis, or rebuilding the
    # track would throw them away without a word.
    const matterepairs::Dict{UInt64, Dict{Int, Matrix{UInt8}}}
    # `(clip, srcframe, mask)` while a brush stroke is in flight, else `nothing`.
    # The mask is a COPY of the frame's alpha; the stroke paints into it and only
    # `endmattebrush!` puts it on the undo stack, so one stroke is one step and an
    # abandoned one costs nothing.
    mattebrush::Any
    # The matte brush's radius, as a fraction of the matte's width. A field
    # because it is per-editor state a user adjusts constantly ([ and ]), and
    # because painting with a size you cannot see or change is guesswork.
    brushradius::Float64
    analysisbackend::Any  # KA backend for analysis/GPU playback; set by auto-detect
    const fig::Figure
    const previewaxis::Axis
    const cropmode::Base.RefValue{Bool}
    const croprect::Observable{Vector{Point2f}}
    composebuf::RGBFrame  # frames are composed HERE and published to `frame` as one
                          # copy of the finished image — GLMakie samples `frame[]`
                          # lazily at render time, so decoding/warping in place there
                          # flashes raw or half-processed frames on screen during play
    # keyed by (effect id, parameter name): a parameter is only unique WITHIN
    # its effect now, and two entries of one kind must not share a slider
    const fxsliders::Dict{Tuple{UInt64, Symbol}, Slider}
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
    # Clips taken by Ctrl+C, already detached from the sequence.
    #
    # They are full [`copyclip`](@ref) results rather than references, so the
    # clipboard survives deleting what was copied — the copy shares the source's
    # analysis and nothing that a later edit can invalidate.
    clipboard::Vector{Clip}
    # Whether a `retrypresent` refiner is running. One at a time, so exactly one
    # thing presents and it always targets the current playhead — see there for
    # what many of them did to a scrub.
    const refining::Threads.Atomic{Bool}
    # One trim refiner at a time, and the frame it is chasing. `trimtarget` is
    # overwritten by every trim event so an older loop abandons its frame rather
    # than publishing it late over a newer one.
    const trimming::Threads.Atomic{Bool}
    trimtarget::Any
end

"""
    docsnapshot(player) -> (clips, mattemarks, repairs, captions, canvas, narration)

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
     Dict{UInt64, Dict{Int, Matrix{UInt8}}}(k => copy(v) for (k, v) in player.mattemarks),
     Dict{UInt64, Dict{Int, Matrix{UInt8}}}(k => copy(v) for (k, v) in player.matterepairs),
     # The transcript: `restore!` keeps the same Sequence object, so captions would
     # otherwise survive an undo by accident. Re-transcribing REPLACES a transcript
     # the user may have corrected by hand, and that has to be undoable.
     copy(player.sequence.captions),
     # The canvas and the narration are the two things that live on the SEQUENCE
     # rather than on a clip, and `snapshot(seq)` returns the clip vector alone —
     # so without these, cropping the canvas outward or adding a voiceover was not
     # undoable. Both are edits the user makes with a gesture and expects Ctrl+Z
     # to take back.
     player.sequence.canvas,
     # The vector is copied so add/remove is undoable; the `Narration`s in it are
     # SHARED, on the same terms as the matte marks above — safe only because a
     # re-render REPLACES the element rather than writing through it.
     copy(player.sequence.narration))

"Put a [`docsnapshot`](@ref) back."
function docrestore!(player::Player, snap)
    clips, marks, repairs, caps, canvas, narration = snap
    restore!(player.sequence, clips)
    empty!(player.mattemarks)
    for (k, v) in marks
        player.mattemarks[k] = copy(v)
    end
    # `snapshot` SHARES the mattetrack rather than copying it, so restoring the
    # clips alone would not put a repaired frame's pixels back. `repairmatteat!`
    # installs a fresh track instead of writing through the shared one, which is
    # what makes `restore!` above enough — see the note there.
    empty!(player.matterepairs)
    for (k, v) in repairs
        player.matterepairs[k] = copy(v)
    end
    empty!(player.sequence.captions)
    append!(player.sequence.captions, caps)
    player.sequence.canvas = canvas
    empty!(player.sequence.narration)
    append!(player.sequence.narration, narration)
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

"""
    changesummary(from, to) -> String

What differs between two [`docsnapshot`](@ref)s, in the user's terms.

Undo used to say NOTHING: the picture changed and you were left to work out what
had come back — which on a long timeline, where the change may be off screen, is
the difference between confidence and pressing Ctrl+Z twice to see.

DERIVED rather than labelled. A description per edit would mean touching all 62
`snapshot!` sites and going stale at the 63rd; comparing the states reports what
actually changed no matter which one produced it, including edits added later.
"""
function changesummary(from, to)
    fc, fm, fr, fcap, fcv, fn = from
    tc, tm, tr, tcap, tcv, tn = to
    parts = String[]
    length(fc) == length(tc) || push!(parts, "$(length(tc)) clip(s)")
    if length(fc) == length(tc) &&
       any(((a, b),) -> length(a.effects) != length(b.effects) || a.crop != b.crop ||
                        a.start != b.start || a.track != b.track, zip(fc, tc))
        push!(parts, "a clip's edit")
    end
    fcv == tcv   || push!(parts, "the canvas")
    fcap == tcap || push!(parts, "the transcript")
    length(fn) == length(tn) || push!(parts, "the narration")
    (sum(length, values(fm); init = 0) == sum(length, values(tm); init = 0) &&
     sum(length, values(fr); init = 0) == sum(length, values(tr); init = 0)) ||
        push!(parts, "the matte")
    isempty(parts) && return "the last edit"
    return join(parts, " · ")
end

function undo!(player::Player)
    isempty(player.undostack) && return setstatus!(player, "nothing left to undo")
    now = docsnapshot(player)
    push!(player.redostack, now)
    back = pop!(player.undostack)
    docrestore!(player, back)
    postrestore!(player)
    setstatus!(player, "undid $(changesummary(now, back)) — Ctrl+Shift+Z redoes it")
    return nothing
end

function redo!(player::Player)
    isempty(player.redostack) && return setstatus!(player, "nothing to redo")
    now = docsnapshot(player)
    push!(player.undostack, now)
    fwd = pop!(player.redostack)
    docrestore!(player, fwd)
    postrestore!(player)
    setstatus!(player, "redid $(changesummary(now, fwd))")
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
        # A job that throws must not take the worker with it. It used to: the
        # exception escaped the `for`, the task died, and because the channel
        # stayed open every later `rungpu` posted into a channel nobody read —
        # so the next GPU request did not fail, it HUNG, forever, with no
        # message. `record_loop_demo` sat 15 minutes at 107% CPU twice before an
        # interrupt showed no `analyzemotion!` frame anywhere: the worker had
        # been dead since the thumbnail probe hit a lost device.
        #
        # Reported, not swallowed — `@error` with the backtrace, because a GPU
        # job failing is a bug and the whole point here is that it stops being
        # invisible. `rungpusync` still carries its own exception back to its
        # caller; this only catches what nobody else would ever see.
        task = Task(() -> for f in jobs
            try
                Base.invokelatest(f)  # jobs may be defined after the worker started
            catch err
                @error "GPU worker job failed; worker stays up" exception = (err, catch_backtrace())
            end
        end)
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, Threads.nthreads() - 1)
        schedule(task)
        return new(jobs, task)
    end
end

"""
Post `f` to a worker directly — for callers that own a [`GPUWorker`] but no
`Player` (the test suite's GPU beats, which must reach Lava on the same pinned
thread the editor uses, or they fix ownership on main and every later analysis
asserts).

**Fire and forget: this tells you nothing about whether `f` ran, finished or
threw.** If you need to know, use [`rungpusync`](@ref), which carries the
exception back and rethrows it at the caller. Do NOT hand-roll
`rungpu(...) do; …; flag[] = true; end` followed by `while !flag[]` — that is a
synchronous wait written as an asynchronous one, and when the job throws the
flag is never set and the loop spins forever. Both walkthroughs did exactly
that, and both hung for fifteen minutes at full CPU with nothing in the log.
"""
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
    pools = Dict{Any, SourcePool}(source => SourcePool(source; capacity))

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
    retrypresent(player)
    for src in unique(c.source for c in sequence.clips)  # projects may be multi-source
        needsproxy(src; maxpixels = proxythreshold) && startproxy!(player, src)
        # with the GPU preview on, stream-decode each source on the GPU so playback
        # runs purely on the GPU (background; CPU decode until the stream is ready)
        wantgpu && Threads.@spawn preloadgpu!(player, src)
    end
    # The repairs come back with the project. They live here rather than on the
    # sequence, so `loadproject` cannot restore them — see `loadrepairs!` for what
    # losing them silently cost.
    isproject && loadrepairs!(player, path)
    autodetect && Threads.@spawn autodetectgpu!(player)  # enable GPU playback if capable
    Threads.@spawn begin  # keep the regenerable proxy/PCM caches bounded
        pruned = prunecache!()
        pruned > 0 && setstatus!(player, "cache pruned — freed $(round(pruned / 2^30, digits = 1)) GiB")
    end
    return player
end

"""
    readerkey(seq, clip) -> Any

Which decode reader `clip` should use: its SOURCE when sharing one is fine, or
the clip itself when it is not.

Sharing a reader per source is measured FASTER for a BLEND — 0 stand-ins and
5.4 ms/frame against 8.2 for a reader per layer (2026-07-28) — because the two
layers sit a few frames apart and one ring holds both positions.

It is much slower for a CLONE. Two clips of one file overlapping at an offset
LARGER than the ring make that single reader seek twice for every frame drawn,
which is the one thing a long-GOP codec is worst at: each seek walks from a
keyframe. Measured on the bird clip, a clone pasted at the playhead (150 frames
of offset, against a 120-frame ring) costs 55.7 -> 46.5 fps, and it hits
scrubbing as hard as playback.

So the rule is the DISTANCE, not the count. Clips that overlap within a ring's
worth of each other share a reader and keep the blend result; clips further apart
than that get their own and stop fighting over one read head.
"""
function readerkey(seq::Sequence, clip::Clip)
    for other in seq.clips
        other === clip && continue
        other.source === clip.source && overlapping(clip, other) &&
            farapart(clip, other) && return clip.id
    end
    return clip.source
end

"Do these two clips occupy any timeline frame in common?"
overlapping(a::Clip, b::Clip) = a.start < clipend(b) && b.start < clipend(a)

"""
Are `a` and `b` showing source frames further apart than one reader's ring?

Compared at a frame they SHARE, so a retimed clip is handled by `sourceframe`
rather than by assuming `src_in - start`.
"""
function farapart(a::Clip, b::Clip)
    n = max(a.start, b.start)
    return abs(sourceframe(a, n) - sourceframe(b, n)) > GPU_STREAM_CAPACITY
end

"The decode pool for a clip, created on first use — see [`readerkey`](@ref) for
 when two clips of one file share one and when they must not."
pool(player::Player, source::VideoSource) =
    get!(() -> SourcePool(source; capacity = player.capacity), player.pools, source)

function pool(player::Player, clip::Clip)
    key = readerkey(player.sequence, clip)
    key === clip.source && return pool(player, clip.source)
    return get!(() -> SourcePool(clip.source; capacity = player.capacity),
                player.pools, key)
end

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
            player.playing[] || retrypresent(player)
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
                    Dict{UInt64, Dict{Int, Matrix{UInt8}}}(),
                    nothing, 0.04,
                    analysisbackend,
                    fig, ax, Ref(false),
                    Observable(Point2f[]),
                    similar(frame[]), Dict{Tuple{UInt64, Symbol}, Slider}(),
                    Dict{Symbol, Any}(),
                    Ref(false), nothing, (0.0, 0.0, 1.0, 1.0), (0, 0), nothing, nothing, 0.0,
                    nothing, false, nothing, nothing, nothing, nothing, defaultsegmenter(),
                    Any[], Any[], 0.0, 0, 0, nothing, 0, 0,
                    Dict{Symbol, Any}(), Observable(:none),
                    Observable(VideoSource[]), Any[], nothing, Observable(:none),
                    Ref(1.0), Observable(:opacity), Observable(true),
                    Threads.Atomic{Float64}(NaN), Dict{Any, Any}(), FxEngine(analysisbackend),
                    effects, Clip[], Threads.Atomic{Bool}(false),
                    Threads.Atomic{Bool}(false), nothing)
    # The built-in depth model, so an editor has depth without being asked. Lazy
    # inside — this only points `registerdepth!` at it and builds nothing.
    installdepth!()
    installlook!()
    installtranscribe!()
    installinterpolate!()
    installspeak!()
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
    # The matte brush's footprint, so painting is not done blind. Alt is the
    # gesture (see the preview's mouse handlers), so the circle appears exactly
    # when the click would paint — a cursor that lied about whether the next
    # click paints would be worse than none.
    brushpos = Observable(Point2f(NaN, NaN))
    brushsize = Observable(0.0f0)
    brushcursor = scatter!(ax, brushpos; marker = Makie.Circle, markersize = brushsize,
                           markerspace = :pixel, color = (:white, 0.0),
                           strokecolor = (:white, 0.9), strokewidth = 1.5)
    translate!(brushcursor, 0, 0, 9)
    player.fxwidgets[:brushpos] = brushpos
    player.fxwidgets[:brushsize] = brushsize
    # What the drag is ABOUT to produce, next to the rectangle.
    #
    # Derived from `croprect` alone, so there is no second piece of state that can
    # disagree with the rectangle on screen. Before this the size only appeared in
    # the status bar AFTER releasing, which is the wrong moment: the number is what
    # you are aiming at, and a crop that grows the canvas looks exactly like the
    # letterbox bars a differently-shaped clip already gets.
    cropreadoutpos = Makie.lift(player.croprect) do r
        isempty(r) ? Point2f(0, 0) :
            Point2f(minimum(q[1] for q in r), minimum(q[2] for q in r))
    end
    cropreadouttext = Makie.lift(player.croprect) do r
        isempty(r) && return ""
        loc = editclip(player)
        loc === nothing && return ""
        clip = loc[1]
        W, H = size(player.frame[])
        (W == 0 || H == 0) && return ""
        x0, x1 = extrema(q[1] for q in r)
        y0, y1 = extrema(q[2] for q in r)
        # Preview pixels → the SOURCE fraction the crop stores → canvas pixels,
        # the same chain `finishcrop!` and `canvassize` walk. The preview may be a
        # proxy, so its own size is never the answer.
        cw = max(2 * (round(Int, (x1 - x0) / W * clip.source.width) ÷ 2), 2)
        ch = max(2 * (round(Int, (y1 - y0) / H * clip.source.height) ÷ 2), 2)
        now = canvassize(player.sequence)
        grew = cw > now[1] || ch > now[2]
        return "$(cw)×$(ch)" * (grew ? "  ↑ canvas grows" : "")
    end
    cropreadout = text!(ax, cropreadoutpos; text = cropreadouttext,
                        fontsize = 13, font = :bold, color = :orangered,
                        strokecolor = (:black, 0.85), strokewidth = 2,
                        offset = (6, -18), align = (:left, :top))
    translate!(cropreadout, 0, 0, 9)

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
        # A PENDING PICK first: it claims the next click on the preview whatever
        # the toolbar tool is, and it was the one mode with no cursor at all —
        # "focus: click what should be sharp" in the status bar, then nothing on
        # screen to say the click was still owed. Covers the object-lock pick too,
        # which had the same silence.
        if player.onpick !== nothing
            shape = mp in ax.scene.viewport[] ? :crosshair : :arrow
        elseif t === :split
            tlscene = timeline.axis.scene
            if mp in tlscene.viewport[]
                tt = Makie.mouseposition(tlscene)[1]
                shape = clipat(sequence, timelineframe(timeline, tt)) === nothing ?
                        :arrow : :scissor
            end
        elseif t === :crop
            shape = mp in ax.scene.viewport[] ? :crosshair : :arrow
        elseif ispressed(fig, Keyboard.left_control | Keyboard.right_control) &&
               mp in timeline.axis.scene.viewport[] &&
               clipat(sequence, timelineframe(timeline,
                      Makie.mouseposition(timeline.axis.scene)[1])) !== nothing
            # Ctrl ADVERTISES itself. Ctrl is the only way to drag a clip — a plain
            # press scrubs — and nothing said so, which is the other half of "one
            # accidentally drags the clip": once it stopped happening by accident
            # there was no way to learn it happens on purpose. Holding Ctrl over a
            # clip now shows the move cursor, so the gesture is discovered by
            # reaching for it rather than by being told.
            shape = :move
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
        if t === :crop
            # SHOW the framing you are about to change. Without this every crop is
            # a blind redo: the rectangle you drag has no relationship on screen to
            # the one already in force, so refining a crop means guessing where it
            # currently is and starting over.
            showcurrentcrop!(player)
            setstatus!(player, "crop tool — drag a rectangle on the preview " *
                               "(it may reach outside the picture; Esc to put it away)")
        end
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
                            clip.crop == (0.0, 0.0, 1.0, 1.0) && !isanimated(clip)
                    thumb = plain ? nearestthumb(cachefor(timeline, clip.source),
                                                 floor(Int, srcframe / clip.source.framerate)) :
                            nothing
                    if thumb !== nothing
                        ensureframesize!(player, pool(player, clip).source)
                        blitthumb!(frame[], thumb)
                        # overlays HERE too, approximate stand-in or not: an
                        # overlay blinking out for the length of a scrub drag and
                        # back reads as the overlay being broken, which is worse
                        # than the pass costing a few ms on a frame that is
                        # already known to be a stand-in.
                        publishframe!(player, n)
                    end
                end
            end
            playing[] || retrypresent(player)
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
        if !presentclipframe!(player, clip, sf; standin = false)
            presentclipframe!(player, clip, sf; standin = true)
            retrytrim!(player, clip, sf)      # …and land the exact frame when it decodes
        end
        return nothing
    end
    # …and when the drag ends, stop chasing the edge frame. Without this the
    # retry outlives the release and republishes the edge over the playhead —
    # the same bug as before, pointing the other way.
    timeline.ontrimend = () -> (player.trimtarget = nothing; nothing)
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

# A Player has ~50 fields, and the default `show` walks every one of them. Three
# of those are why this exists rather than being cosmetic: `frame` and
# `composebuf` are whole decoded images, and `gpucache` maps each source to its
# DEVICE-RESIDENT frames — so displaying a Player at the REPL pulled megabytes
# back off the GPU to print them. These two touch scalars and observables only.
function Base.show(io::IO, p::Player)
    seq = p.sequence
    n = length(seq.clips)
    print(io, "Player(", n, " clip", n == 1 ? "" : "s", ", ", timecode(seq, p.playhead[]), ")")
end

function Base.show(io::IO, ::MIME"text/plain", p::Player)
    seq = p.sequence
    nclip = length(seq.clips)
    ntr = ntracks(seq)
    nfx = sum(c -> length(c.effects), seq.clips; init = 0)
    names = unique(basename(c.source.path) for c in seq.clips)
    w, h = p.lastcanvas
    println(io, "Player")
    if nclip == 0
        println(io, "  timeline   empty")
    else
        print(io, "  timeline   ", nclip, " clip", nclip == 1 ? "" : "s",
              " on ", ntr, " track", ntr == 1 ? "" : "s")
        nfx == 0 || print(io, ", ", nfx, " effect", nfx == 1 ? "" : "s")
        println(io)
        # Sources, not clips: a split leaves two clips on one file, and naming the
        # file twice reads as two pieces of media.
        println(io, "  media      ", join(first(names, 3), ", "),
                length(names) > 3 ? " (+$(length(names) - 3) more)" : "")
    end
    println(io, "  position   ", timecode(seq, p.playhead[]),
            p.playing[] ? "  playing" : "  paused")
    println(io, "  canvas     ", w, "x", h, " @ ", round(seq.framerate; digits = 3), " fps")
    println(io, "  preview    ", p.gpupreview === nothing ? "CPU" : "GPU",
            p.analysisbackend === nothing ? "" : "  ·  analysis on $(nameof(typeof(p.analysisbackend)))")
    nundo, nredo = length(p.undostack), length(p.redostack)
    print(io, "  edits      ", p.edits, " (", nundo, " undo, ", nredo, " redo)")
    p.dockopen[] === :none || print(io, "  ·  dock: ", p.dockopen[])
end

# ---------------------------------------------------------------- presentation

"""
Preview a cross-dissolve at timeline frame `n`: fetch both clips' frames from
their rings, run each clip's tracks + effects, and blend `(1-p)·A + p·B`.
Best-effort — returns `false` (caller falls back to the plain single-clip path)
if either frame isn't buffered yet, or the two sources differ in size (mismatched
dissolves preview as the outgoing clip; export still blends them via warp).

`n` is carried in rather than derived from the sample: the blend is addressed by
the two SOURCE frames, which say nothing about where on the timeline this is, and
[`publishframe!`](@ref) needs the timeline frame to know which overlays are up.
"""
function showtransition!(player::Player, sample, n::Integer)
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
    publishframe!(player, n)
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
    publishframe!(player, n)
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
    publishframe!(player, n) -> nothing

Put what is in `player.frame[]` on screen as timeline frame `n` — overlays drawn,
texture re-pointed, observable notified.

THE ONE DOOR. Every preview path ends here, because the overlay pass is the last
stage of a rendered frame and a call site that skips it is a call site that shows
the user a different picture than the export produces. Four of them published by
hand — a straight `showcpuframe!` + `notify` — so `drawoverlays!` ran only from
`export.jl`, and every overlay kind was invisible in the editor while being
correct in the file it wrote. That is the failure `drawoverlays!` documents
itself against ("no call site can accidentally be the one that renders WITHOUT
overlays") and it was true of the whole preview.

Costs nothing when nothing is drawn at `n`: `drawoverlays!` returns before it
touches a pixel, and the round trip is bit-exact when it does not.

Main thread ONLY. Rasterising an overlay needs the GL context, so this must not
be called from inside `runowned` — a `composite` callback runs on the engine's
owning thread and GLFW aborts there (`ThreadAssertionError`, measured). All four
callers reach it after the callback has returned.
"""
function publishframe!(player::Player, n::Integer)
    seq = player.sequence
    drawoverlays!(player.frame[], seq.overlays, n;
                  framerate = seq.framerate, captions = seq.captions)
    showcpuframe!(player)
    notify(player.frame)
    return nothing
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
        s !== nothing && showtransition!(player, s, n) && return true
    end
    # multiple stacked tracks → composite the stack (on the GPU if every layer has a
    # stream, else on the CPU)
    if ntracks(player.sequence) > 1
        clips = clipsat(player.sequence, n)
        if length(clips) > 1
            shown = false
            if gpuready(player) &&
               all(haskey(player.gpucache, readerkey(player.sequence, c)) for c in clips)
                # composites mix layers: one stand-in among them dates the whole frame
                standin || primecomposite!(player, clips, n) || return false
                shown = presentgpucomposite!(player, clips, n;
                                             chunks = standin && !player.playing[] ? 0 : 5)
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
        # black, but NOT bare: an overlay is a property of the sequence, not of a
        # clip, so one standing over a gap is still up. Titles live in gaps.
        publishframe!(player, n)
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
    if gpuready(player) && haskey(player.gpucache, readerkey(player.sequence, clip))
        stream = player.gpucache[readerkey(player.sequence, clip)]
        if 0 <= srcframe < nframes(stream)
            # parked on an undecoded frame: advance the feed but leave the last
            # exact image up — the retry loop calls back until this frame lands
            if !standin && !hasframe(stream, srcframe)
                primeframe!(player, stream, srcframe) && return false
            end
            ensureframesize!(player, canvassize(player.sequence))
            eclip = effectiveclip(clip, srcframe)
            # A STAND-IN MUST NOT DECODE. `frameat!` spends up to 5 decode chunks
            # reaching `srcframe` before it settles for the nearest decoded frame,
            # and on a seek that is ~62 ms of exactly the work this call exists to
            # skip: the point is to put a picture up NOW and let the retry loop
            # fetch the real frame. Measured, this is the difference between a
            # seek answering in ~1 ms and in ~62.
            if presentgpu!(player, clip, srcframe; stream = stream,
                           chunks = standin && !player.playing[] ? 0 : 5)
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
        # this path is addressed by SOURCE frame (it is reached from a trim drag
        # as well as from the playhead), so the timeline frame has to be resolved
        # back — the same `n_of` the composite above is given.
        publishframe!(player, n_of(player, clip, srcframe))
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
    retrytrim!(player, clip, srcframe; budget = 5.0)

Keep trying to put `srcframe` on screen while a trim drag settles there.

`ontrimpreview` is best-effort: if the decoder has not reached the frame yet it
shows the nearest one instead. Nothing then corrected it, because the next
attempt only arrived with the next mouse move — so a trim that ENDED on an
undecoded frame left the picture on whatever was there, which is the playhead's
frame. That is the whole of "trimming shows the playhead instead of the cut".

Single-flight and self-cancelling, like [`retrypresent`](@ref): a newer trim
event overwrites the target and the older loop sees that and stops, so dragging
never accumulates refiners racing to publish different frames.
"""
function retrytrim!(player::Player, clip::Clip, srcframe::Integer; budget::Real = 5.0)
    target = (clip, Int(srcframe))
    player.trimtarget = target
    Threads.atomic_cas!(player.trimming, false, true) === false || return nothing
    @async try
        deadline = time() + budget
        while time() < deadline
            t = player.trimtarget
            t === nothing && break
            c, sf = t
            presentclipframe!(player, c, sf; standin = false) && break
            sleep(0.01)
        end
    finally
        player.trimming[] = false
    end
    return nothing
end

"""
Keep refining until the EXACT frame under the playhead is on screen (each attempt
advances the stream's decode). When it never arrives inside `budget` seconds the
preview shows the decoder's best and SAYS so, rather than holding a stale image
without a word.

**One refiner at a time, and it reads the playhead rather than capturing it.**

This used to spawn a task per call, each closing over the frame it was asked for
and looping `while playhead[] == n`. A scrub calls it once per frame, so dozens
ran at once, and the guard was checked BEFORE the render: a task could pass it,
spend a few milliseconds compositing, and publish its frame after a newer task had
already published a later one. The screen then held a frame from behind the
playhead — intermittently, and more often the heavier the chain, which is why it
surfaced as a matte or a stabilized warp "moving behind" rather than as a plainly
wrong picture.

Publishing cannot be stale if nothing holds a stale target. The single task reads
`playhead[]` fresh on every pass, so it always renders what the user is looking at
and the frame that reaches the screen is the last one asked for. `showframe!` is
left alone: it puts frame `n` on screen, which is all it should ever mean, and
trim preview still uses it to show an edge frame the playhead is nowhere near.
"""
function retrypresent(player::Player; budget::Real = 20.0)
    Threads.atomic_cas!(player.refining, false, true) === false || return nothing
    @async try
        deadline = time() + budget
        drawn = -1                       # the frame a stand-in is already up for
        while !player.playing[]
            n = player.playhead[]        # the CURRENT frame, never a captured one
            # THE PICTURE MOVES FIRST, then sharpens. A full-quality seek lands
            # mid-GOP and has to decode forward from the preceding keyframe:
            # measured at ~2.1 ms per frame on a 250-frame GOP, so 90 ms to 455 ms
            # depending only on where in the GOP it falls — and for all of it the
            # screen still showed the frame the user had just left. That is the
            # whole of "seeking is slow"; the compositor and the panel are not in
            # it (playback renders a stand-in in 0.9 ms).
            #
            # `standin = true` draws the nearest decoded frame in about a
            # millisecond AND returns whether it happened to be the exact one, so
            # one call both answers immediately and says whether there is anything
            # left to refine. A miss leaves a picture on screen while the loop
            # below fetches the real frame — which is what every other NLE does,
            # and why nobody notices a GOP walk in them.
            if n != drawn
                drawn = n
                if showframe!(player, n; standin = true)
                    player.presented += 1
                    player.playhead[] == n && break
                    deadline = time() + budget
                    continue
                end
            end
            if showframe!(player, n)
                player.presented += 1
                player.playhead[] == n && break   # …unless it moved while we rendered
                deadline = time() + budget
            elseif time() > deadline
                # the stand-in is already up — say why it is staying
                setstatus!(player, "frame $n never finished decoding — showing the nearest decoded frame")
                break
            else
                sleep(0.005)
            end
        end
    finally
        player.refining[] = false
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
    showcurrentcrop!(player) -> nothing

Draw the framing currently in force, so the crop tool has something to refine
rather than replace.

Uses the same rectangle the drag does (`croprect`), so there is one outline on
screen and no second one to disagree with it. Cleared by the first press of a new
drag, which is exactly when it has served its purpose.
"""
function showcurrentcrop!(player::Player)
    loc = editclip(player)
    loc === nothing && return nothing
    x, y, w, h = loc[1].crop
    W, H = size(player.frame[])
    (W == 0 || H == 0) && return nothing
    # In PREVIEW pixels, like the drag's own rectangle — the preview may be a
    # proxy, so the crop's source fractions have to come through its size.
    x0, y0 = x * W, y * H
    x1, y1 = (x + w) * W, (y + h) * H
    player.croprect[] = Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return nothing
end

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
    inheritmasks!(player, from::Clip, to::Clip) -> nothing

Give `to` the matte marks and repairs of `from`.

Both stores are keyed by CLIP ID, and every way of making a second clip out of one
— splitting it, copying it, pasting it — hands the new clip a fresh id. The
tracks themselves survive that, because they are keyed by absolute SOURCE frame
and both clips share the source; the player-side stores do not, and silently did
not follow.

What that cost is worst exactly where it is least visible: the alpha comes along,
so the picture is right, and only re-running is broken — a split half met an empty
mark store and refused to propagate, with the repair cards gone too. Splitting is
the single most common act in cutting anything, so this is the corner case most
likely to be hit and least likely to be noticed.

Copied whole rather than filtered to `to`'s range: a mark outside it is harmless
(every lookup is by source frame) and trimming the halves back out later would
otherwise have to put it back.
"""
function inheritmasks!(player::Player, from::Clip, to::Clip)
    from.id == to.id && return nothing
    for store in (player.mattemarks, player.matterepairs)
        src = get(store, from.id, nothing)
        src === nothing || isempty(src) || (store[to.id] = copy(src))
    end
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
    retrypresent(player)
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
    right = split!(player.sequence, player.playhead[], loc[1].track)
    if right === nothing
        pop!(player.undostack)
        return setstatus!(player, "cannot split here — the playhead is at the clip's start")
    end
    inheritmasks!(player, loc[1], right)
    refreshedit!(player)
    return nothing
end

"""
    copyclips!(player) -> Int

Take the marked clips — or the one being edited — onto the clipboard. Ctrl+C.

Copies at the point of taking rather than storing references, so deleting what
you copied does not empty the clipboard. Positions are kept relative to the
earliest clip taken, which is what lets [`pasteclips!`](@ref) rebuild a multi-clip
arrangement at the playhead instead of collapsing it to one point.
"""
function copyclips!(player::Player)
    seq = player.sequence
    marked = player.timeline.selection[]
    clips = if length(marked) > 1
        [seq.clips[i] for i in marked if 1 <= i <= length(seq.clips)]
    else
        loc = editclip(player)
        loc === nothing ? Clip[] : [loc[1]]
    end
    isempty(clips) && (setstatus!(player, "nothing selected to copy"); return 0)
    base = minimum(c.start for c in clips)
    empty!(player.clipboard)
    for c in clips
        dup = copyclip(c; start = c.start - base)
        # the clipboard entry carries the marks too, so a paste can be re-run —
        # see `inheritmasks!`. The chain is original -> clipboard -> pasted, and
        # every link is a fresh id.
        inheritmasks!(player, c, dup)
        push!(player.clipboard, dup)
    end
    setstatus!(player, "copied $(length(clips)) clip$(length(clips) == 1 ? "" : "s")")
    return length(clips)
end

"""
    pasteclips!(player) -> Int

Drop the clipboard at the playhead, each clip on the first lane at or above its
own that has room. Ctrl+V.

Searching upward for a free lane is right HERE and wrong for a drag: a paste has
no target lane under the cursor to honour, so the only alternative to finding one
is refusing. A drag does have one, which is why `dragto!` refuses instead — see
the note there on what riding upward cost when it did.
"""
function pasteclips!(player::Player)
    isempty(player.clipboard) && (setstatus!(player, "clipboard is empty"); return 0)
    seq = player.sequence
    at = player.playhead[]
    snapshot!(player)
    n = 0
    fresh = Clip[]
    for c in player.clipboard
        start = at + c.start                    # relative offsets, restored around the playhead
        track = c.track
        while track <= ntracks(seq) && !canplace(seq, c, start, track)
            track += 1
        end
        pasted = copyclip(c; start, track)
        inheritmasks!(player, c, pasted)
        push!(seq.clips, pasted)
        push!(fresh, pasted)
        n += 1
    end
    sort!(seq.clips, by = c -> (c.track, c.start))
    # SELECT what was just pasted. Paste puts a clip at the playhead on whichever
    # lane was free, which is rarely where it belongs — so the next act is always
    # to move it, and without this the act before THAT is hunting for it. By
    # identity, not by index: the sort above has just moved everything.
    idx = filter(!isnothing, [findfirst(c -> c === f, seq.clips) for f in fresh])
    if !isempty(idx)
        player.timeline.selected[] = first(idx)
        player.timeline.selection[] = Int[i for i in idx]
    end
    refreshedit!(player)
    setstatus!(player, "pasted $n clip$(n == 1 ? "" : "s") — selected, Ctrl-drag to place")
    return n
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
    # ONLY when the checkpoint brought its own. `checkpointproject` copies the
    # JSON and not the `.mattes` directory beside it, so most checkpoints have no
    # masks at all — clearing unconditionally would throw away the session's marks
    # and put nothing back, which is strictly worse than the stale-but-present
    # marks the restore used to leave alone.
    if isdir(mattedir(file))
        empty!(player.matterepairs); empty!(player.mattemarks)
    end
    empty!(player.sequence.clips); append!(player.sequence.clips, seq.clips)
    empty!(player.sequence.transitions); append!(player.sequence.transitions, seq.transitions)
    empty!(player.sequence.overlays); append!(player.sequence.overlays, seq.overlays)
    loadrepairs!(player, file)               # …and the repairs saved beside it
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
        saveproject(path, player; checkpoint = false)
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
    saveproject(path, player; checkpoint = true)

Save the project AND the edit state that lives on the player rather than on the
sequence — currently the matte repairs.

A `Sequence` is not the whole edit. The repairs are on the player because they
must outlive the panel that shows them, so the sequence-only method cannot see
them and quietly wrote a project that was missing work. See `saverepairs`.
"""
function saveproject(path::AbstractString, player::Player; checkpoint::Bool = true)
    saveproject(path, player.sequence; checkpoint)
    saverepairs(path, player)
    return path
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
    saveproject(path, player)
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
    player.playing[] || retrypresent(player)
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
    # per READER, not per source: a clone overlapping its original further apart
    # than the ring needs its own stream or the two fight over one read head
    for clip in player.sequence.clips
        haskey(player.gpucache, readerkey(player.sequence, clip)) ||
            Threads.@spawn preloadgpu!(player, clip)
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
    # Alt+drag on the preview PAINTS the matte: left adds to the subject, right
    # takes away. Alt rather than a tool of its own, and available whenever the
    # clip has a matte, because the moment you want it is the moment you are
    # looking at a bad frame — putting it behind a mode switch means noticing the
    # problem, leaving what you were doing, and coming back.
    brushing() = ispressed(player.fig, Keyboard.left_alt | Keyboard.right_alt)
    on(events(ax.scene).mousebutton) do event
        mine() || return Consume(false)
        if brushing() && event.button in (Mouse.left, Mouse.right)
            if event.action == Mouse.press && is_mouseinside(ax.scene)
                beginmattebrush!(player, event.button == Mouse.left) || return Consume(false)
                mattebrushto!(player, Point2f(mouseposition(ax.scene));
                              radius = player.brushradius)
                return Consume(true)
            elseif event.action == Mouse.release && player.mattebrush !== nothing
                endmattebrush!(player)
                return Consume(true)
            end
        end
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
        # Show (or hide) the brush footprint. Diameter in SCREEN pixels, from the
        # radius' fraction of the matte width and the preview's width — so the
        # circle is the size the stroke will actually be, at any zoom.
        bp = get(player.fxwidgets, :brushpos, nothing)
        if bp !== nothing
            vp = ax.scene.viewport[]
            if brushing() && editclip(player) !== nothing
                bp[] = Point2f(mouseposition(ax.scene))
                player.fxwidgets[:brushsize][] =
                    Float32(2 * player.brushradius * vp.widths[1])
            else
                bp[] = Point2f(NaN, NaN)
            end
        end
        if player.mattebrush !== nothing
            # Direction comes from the stroke, not from the mouse right now — see
            # `beginmattebrush!`.
            mattebrushto!(player, Point2f(mouseposition(ax.scene)); radius = player.brushradius)
            return Consume(true)
        end
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

"""
    setbrushradius!(player, r) -> Float64

Resize the matte brush, clamped, and make the change visible where the brush is.

One function because there are two ways to ask — `[`/`]` and the panel's ± — and
they must agree: the cursor ring on the preview and the percentage in the panel
are both derived from this field, so a caller that set it directly would move the
brush and leave one of the two showing the old size.

Steps are the CALLER's, and geometric by convention: at 1% a fixed step is half
the brush and at 20% it is nothing, so the same gesture has to mean the same
proportion.
"""
function setbrushradius!(player::Player, r::Real)
    player.brushradius = clamp(Float64(r), 0.004, 0.4)
    bp = get(player.fxwidgets, :brushsize, nothing)
    bp === nothing ||
        (bp[] = Float32(2 * player.brushradius *
                        player.previewaxis.scene.viewport[].widths[1]))
    setstatus!(player, "matte brush $(round(100 * player.brushradius; digits = 1))% " *
                       "of the frame width")
    # the panel prints the number too, and only rebuilds when asked
    refreshmattepanel!(player)
    return player.brushradius
end

"""
    cropscope(player) -> Ref{Symbol}

What the crop tool changes: `:canvas`, the whole project, or `:clip`, only the
clip you dragged on.

Both are real requests and they used to be the same gesture. Every drag rewrote
`sequence.canvas`, so "show less of this one shot" could not be said at all —
reframing one clip silently resized the finished video. Photoshop draws the same
line between cropping the canvas and moving a layer inside it; this is the same
distinction with the same default, because resizing the project is the reason the
tool exists.

On `fxwidgets` rather than in a field, like [`mattecardview`](@ref): the dock
rebuilds every panel from scratch on each refresh, so a toggle holding its state
in the context would have it discarded by the rebuild it triggered.
"""
cropscope(player::Player) = get!(() -> Ref(:canvas), player.fxwidgets, :cropscope)

"""
    cropaspect(player) -> Ref{Union{Nothing, Float64}}

The shape the next crop is locked to — width over height of the OUTPUT — or
`nothing` for whatever you drag.

A ratio lock is not a convenience here, it is the only way to hit one. "Make this
vertical for phones" means exactly 9:16, and a rectangle dragged by hand is never
exactly anything; the alternative is typing pixel counts into a project-settings
dialog, which is the thing this tool exists to avoid.
"""
cropaspect(player::Player) =
    get!(() -> Ref{Union{Nothing, Float64}}(nothing), player.fxwidgets, :cropaspect)

"""
    lockaspect(crop, source, a) -> NTuple{4, Float64}

Reshape a dragged crop so the exported picture is `a` wide for every 1 tall.

The crop is in fractions **of the source**, and a fraction is not a pixel: the
output ratio is `w·source.width / h·source.height`, so the fraction ratio that
lands on `a` is `a · height/width`. Getting this wrong gives a 16:9 lock that is
16:9 only on a square source.

The dragged WIDTH is kept and the height derived. One of the two has to give, and
width is the one a framing is judged by — a locked crop that quietly narrowed
would move the subject out of the frame you just drew around it.
"""
function lockaspect(crop::NTuple{4, Float64}, source, a::Real)
    x0, y0, w, h = crop
    want = Float64(a) * source.height / source.width
    return (x0, y0, w, w / want)
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
    # NOT clamped to the picture. A crop rectangle may sit partly — or wholly —
    # outside it, which is how Photoshop's crop tool grows a canvas: drag past the
    # edge and the result is bigger than what you started with, with the new area
    # empty. Clamping to `0..1` made this tool able to shrink a project and never
    # to enlarge one, so "make the canvas taller" had no gesture at all.
    #
    # `canvassize` already multiplies these by the source's pixels, so a width
    # past 1.0 IS a wider canvas; nothing downstream needed a new concept. What
    # falls outside the source renders as the letterbox background, which
    # `placelayer!` already paints for every clip that does not fill the canvas.
    x0, x1 = x0 / W, x1 / W
    y0, y1 = y0 / H, y1 / H
    # A degenerate drag is still degenerate, but the floor is on the SIZE, not on
    # where it sits: a 5%-wide crop hanging off the left edge is a real request.
    (x1 - x0 < 0.01 || y1 - y0 < 0.01) && return nothing
    a = cropaspect(player)[]
    a === nothing || ((x0, y0, w, h) = lockaspect((x0, y0, x1 - x0, y1 - y0), clip.source, a);
                      x1 = x0 + w; y1 = y0 + h)
    before = canvassize(player.sequence)
    clip.crop = (x0, y0, x1 - x0, y1 - y0)
    # In `:canvas` scope the crop tool is how the CANVAS is set — that is the
    # whole gesture, and recording it here is what makes it survive deleting or
    # reordering clips. In `:clip` scope the project keeps its size and only this
    # clip's framing moves, which is what letterboxing and re-framing one shot
    # inside a finished timeline needs.
    wholeproject = cropscope(player)[] === :canvas
    if wholeproject
        player.sequence.canvas =
            (max(2 * (round(Int, (x1 - x0) * clip.source.width) ÷ 2), 2),
             max(2 * (round(Int, (y1 - y0) * clip.source.height) ÷ 2), 2))
    elseif player.sequence.canvas === nothing
        # PIN it. Without an explicit canvas `canvassize` derives one from CLIP
        # ONE's crop — so "crop this clip only", performed on clip one, still
        # resized the project, which is the exact thing the scope exists to
        # prevent. Freezing the size it already has is the only way the promise
        # can hold, and it is invisible: the number does not change, it just
        # stops being a function of a clip the user is now editing.
        player.sequence.canvas = before
    end
    applycrop!(player, clip)
    # Say what the canvas became, and say so LOUDLY when it grew. A crop that only
    # ever removed picture needed no readout — the result was on screen. One that
    # can add empty space does: the new area looks like the letterbox bars a
    # differently-shaped clip already gets, so "did that resize my project or just
    # letterbox this clip?" is a real question the status bar can answer.
    after = canvassize(player.sequence)
    grew = after[1] > before[1] || after[2] > before[2]
    setstatus!(player, !wholeproject ?
        "cropped this clip — the project stays $(after[1])×$(after[2])" :
        grew ?
        "canvas $(before[1])×$(before[2]) → $(after[1])×$(after[2]) — cropped outward, the new area is empty" :
        "canvas $(after[1])×$(after[2])")
    refreshcroppanel!(player)
    return nothing
end

"""
    resetcanvas!(player) -> nothing

Drop the explicit canvas and go back to deriving it from the first clip.

The escape hatch for a canvas you no longer want. Without it the only way out of
a project size was Ctrl+Z, which also takes back the crop that set it — and once
any later edit is on the stack, not even that.
"""
function resetcanvas!(player::Player)
    player.sequence.canvas === nothing && return setstatus!(player, "canvas: already derived from the first clip")
    snapshot!(player)
    player.sequence.canvas = nothing
    applycrop!(player)
    refreshedit!(player)
    refreshcroppanel!(player)
    sz = canvassize(player.sequence)
    setstatus!(player, "canvas back to $(sz[1])×$(sz[2]), derived from the first clip")
    return nothing
end

"Rebuild the crop card, so its scope toggle and size readout follow the edit."
refreshcroppanel!(player::Player) = (EFFECTS.version[] += 1; nothing)

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
        elseif (event.key == Keyboard.left_bracket ||
                event.key == Keyboard.right_bracket) && ispress
            # [ and ] resize the matte brush, as every paint tool does. Geometric
            # steps, not linear: at 1% a fixed step is half the brush and at 20%
            # it is nothing, so the same key has to mean the same PROPORTION.
            f = event.key == Keyboard.right_bracket ? 1.25 : 0.8
            setbrushradius!(player, player.brushradius * f)
        elseif event.key == Keyboard.c && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            copyclips!(player)          # BEFORE bare `c`, which is the crop tool
        elseif event.key == Keyboard.v && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            pasteclips!(player)
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
            #
            # Shift+Enter writes THIS FRAME ONLY. The two are one gesture apart
            # because they are the same decision — "the marking is finished" —
            # answered for different scopes, and the scope is the thing a user
            # picks per frame: propagate when the tracking drifted, repair when
            # one frame in a good matte came out broken.
            shift ? repairmattecollect!(mattecollect(player)) :
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
    # The effects panel skips its rebuilds while hidden (see `rebuildstack`), so
    # it has to catch up on the way back — otherwise it shows the clip that was
    # under the playhead when it was closed.
    if key === :effects
        r = get(player.fxwidgets, :fxlistrefresh, nothing)
        r === nothing || r(force = true)
    end
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
    # All five, not three: the GIF fps and loop controls were the only ones in
    # this panel without a handle, so a walkthrough could drive the format menu
    # and the Export button but not the settings between them.
    merge!(player.fxwidgets, Dict{Symbol, Any}(:exportgo => gobtn, :exportpath => path,
                                               :exportformat => fmtmenu,
                                               :giffps => fpsslider, :gifloop => loopbox))
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
    # THE CURVES OF THE SELECTED EFFECT, and nothing else.
    #
    # This used to sweep `seq.clips`, collect every animated key across the whole
    # timeline into a `Dict{Symbol, …}` of plots, and resolve each bare name back
    # to a parameter through the global index. That sweep existed only because
    # the curves lived on the CLIP under a flat name: with the curve on the
    # parameter of the effect that renders it, the selected card already names
    # exactly what to draw, and `Param.visible` says which of its lanes are open.
    #
    # `visible` is independent of whether a parameter has keys, which is what
    # makes an EMPTY lane drawable — that is where you put the first one.
    ax = player.timeline.axis
    seq = player.sequence
    fps = seq.framerate
    lanes = Observable(Point2f[])          # the curve polylines, NaN-separated
    marks = Observable(Point2f[])          # the ◆ positions
    markmeta = Tuple{Clip, Effect, Param, Int}[]   # what each ◆ IS, by index

    halo = lines!(ax, lanes; color = (:black, 0.55), linewidth = 4.0)
    translate!(halo, 0, 0, 3)
    accent = player.timeline.colors.accent
    curve = lines!(ax, lanes; color = accent, linewidth = 2.0)
    translate!(curve, 0, 0, 4)
    dots = scatter!(ax, marks; marker = :diamond, markersize = 10,
                    color = accent, strokecolor = :white, strokewidth = 1.0)
    translate!(dots, 0, 0, 5)

    # Where a value sits inside its clip's LANE — the same band the timeline draws
    # the clip in, so a curve is read against the thing it belongs to.
    function yat(clip::Clip, p::Param, v)
        lo, hi = trackband(clip.track, player.timeline.ntr[])
        return lo + paramnorm(p, v) * (hi - lo)
    end

    function refresh()
        segs = Point2f[]; pts = Point2f[]; empty!(markmeta)
        sel = selectedeffect(player)
        clip = sel === nothing ? nothing : sel[1]
        fx = sel === nothing ? nothing : sel[2]
        if fx !== nothing
            for p in fx.params
                p.visible || continue
                x0 = clip.start / fps; x1 = clipend(clip) / fps
                for i in 0:60                       # the curve, sampled across the clip
                    t = x0 + (x1 - x0) * i / 60
                    sf = sourceframe(clip, round(Int, t * fps))
                    push!(segs, Point2f(t, yat(clip, p, valueat(p, sf))))
                end
                push!(segs, Point2f(NaN, NaN))      # break between lanes
                isanimated(p) || continue
                for (k, key) in enumerate(p.curve.keys)
                    clip.src_in <= key.frame <= clip.src_out || continue
                    push!(pts, Point2f(timelineframe(clip, key.frame) / fps,
                                       yat(clip, p, key.value)))
                    push!(markmeta, (clip, fx, p, k))
                end
            end
        end
        lanes[] = segs; marks[] = pts
        return
    end

    on(_ -> refresh(), player.playhead)
    on(_ -> refresh(), player.timeline.selected)
    refresh()
    player.fxwidgets[:kfrefresh] = refresh
    return refresh
end

"""
    selectedeffect(player) -> Union{Nothing, Tuple{Clip, Effect}}

The clip and the effect whose card is selected — what the timeline draws lanes
for. `nothing` when no card is selected, which is when there is nothing to draw.
"""
function selectedeffect(player::Player)
    loc = editclip(player)
    loc === nothing && return nothing
    clip = loc[1]
    key = fxselected(player)
    key isa Tuple && length(key) == 2 && key[1] === :fx || return nothing
    i = findfirst(fx -> fx.id == key[2], clip.effects)
    return i === nothing ? nothing : (clip, clip.effects[i])
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

"Whether `clip` has any animated parameter at all."
clipanimated(clip::Clip) = isanimated(clip)

"Absolute source frame the playhead currently maps to within `clip`."
playheadframe(player::Player, clip::Clip) = sourceframe(clip, player.playhead[])

"""
    syncsliders!(player, clip)

Put every card slider back on what its parameter actually says.

`fxsyncing` is raised while it writes, because a slider's own handler treats a
change as an EDIT — and an edit at the playhead stamps a keyframe. Without the
flag, scrubbing across an animated parameter would key it at every frame it
passed.
"""
function syncsliders!(player::Player, clip::Clip)
    player.fxsyncing[] = true
    try
        for fx in clip.effects, prm in fx.params
            sl = get(player.fxsliders, (fx.id, prm.name), nothing)
            sl === nothing && continue
            v = Float64(valueat(prm, playheadframe(player, clip)))
            Makie.set_close_to!(sl, v)
        end
    finally
        player.fxsyncing[] = false
    end
    return nothing
end

"""
    syncanimatedsliders!(player, clip)

The same, but only for parameters that HAVE a curve — what a scrub needs. A
static parameter cannot have moved, so writing it back would be pure work (and
one more chance to trip the edit path).
"""
function syncanimatedsliders!(player::Player, clip::Clip)
    any(fx -> any(isanimated, fx.params), clip.effects) || return nothing
    player.fxsyncing[] = true
    try
        for fx in clip.effects, prm in fx.params
            isanimated(prm) || continue
            sl = get(player.fxsliders, (fx.id, prm.name), nothing)
            sl === nothing && continue
            Makie.set_close_to!(sl, Float64(valueat(prm, playheadframe(player, clip))))
        end
    finally
        player.fxsyncing[] = false
    end
    return nothing
end

"""
The ◆ of the inspector trio, Premiere semantics: start animating the parameter if
it is not animated yet (first key = its current value); otherwise ADD a key at
the playhead — or REMOVE the one sitting there (removing the last key makes the
parameter static again, at the value it was showing).

Takes the `Param` ITSELF. There is nothing to look up: no key, no registry, no
"the first effect of this kind". The row that drew the button holds the object
the button edits.
"""
function togglekey!(player::Player, clip::Clip, p::Param)
    sf = playheadframe(player, clip)
    snapshot!(player)
    if !isanimated(p)
        p.curve === nothing && (p.curve = AnimCurve{typeof(p.value)}())
        setkey!(p.curve, sf, p.value)
        p.visible = true
        setstatus!(player, "$(p.label): keyframing on — scrub and move the slider to add keys")
    elseif any(k -> k.frame == sf, p.curve.keys)
        v = valueat(p, sf)
        removekey!(p.curve, sf)
        if isempty(p.curve)
            p.value = v                       # stays where it was, not at zero
            p.curve = nothing
            setstatus!(player, "$(p.label): last keyframe removed — back to a static value")
        else
            setstatus!(player, "$(p.label): keyframe removed")
        end
    else
        setkey!(p.curve, sf, valueat(p, sf))
        setstatus!(player, "$(p.label): keyframe added at the playhead")
    end
    notify(player.playhead)
    return nothing
end

"Jump the playhead to the previous (`dir < 0`) or next keyframe of `p`."
function gotokey!(player::Player, clip::Clip, p::Param, dir::Integer)
    isanimated(p) || return setstatus!(player, "$(p.label): no keyframes yet — ◆ adds one")
    sf = playheadframe(player, clip)
    ks = filter(k -> clip.src_in <= k.frame <= clip.src_out, p.curve.keys)
    cand = dir < 0 ? filter(k -> k.frame < sf, ks) : filter(k -> k.frame > sf, ks)
    isempty(cand) && return setstatus!(player,
        "$(p.label): no keyframe $(dir < 0 ? "before" : "after") the playhead")
    k = dir < 0 ? last(cand) : first(cand)
    seek!(player, clamp(timelineframe(clip, k.frame), 0, seqlength(player.sequence) - 1))
    return nothing
end

"Clear every keyframe of `p` (it keeps the value it was showing; undoable)."
function clearkeyframes!(player::Player, clip::Clip, p::Param)
    isanimated(p) || return setstatus!(player, "$(p.label): no keyframes to clear")
    snapshot!(player)
    n = length(p.curve.keys)
    p.value = valueat(p, playheadframe(player, clip))
    p.curve = nothing
    notify(player.playhead)
    setstatus!(player, "$(p.label): cleared $n keyframe$(n == 1 ? "" : "s") (Ctrl+Z to restore)")
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

function Base.close(player::Player)
    pause!(player)   # also stops the audio feed
    try
        deactivatetool!(player)   # tool handlers reference the dying scenes
    catch
    end
    # Listeners this player put on GLOBAL observables. They must come off before
    # the queues below close, or the next player in this process shares a notify
    # list with a corpse that throws (see `:fxglobalobs` in fxpanel.jl).
    for o in get(player.fxwidgets, :fxglobalobs, ())
        try
            Makie.Observables.off(o)
        catch e
            @warn "could not remove a global listener on close" exception = e
        end
    end
    delete!(player.fxwidgets, :fxglobalobs)
    freegpucache!(player)
    foreach(sp -> stop!(sp.worker), values(player.pools))
    stop!(player.timeline)
    close(player.statusqueue)
    close(player.uiqueue)
    player.screen === nothing || close(player.screen)
    player.screen = nothing
    return nothing
end
