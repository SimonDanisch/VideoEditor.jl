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
The Effects panel: the widgets that ARE the panel, and the stack its cards are
built into.

A type rather than a dozen entries in `player.fxwidgets`, because these are what
the panel is made of and what its operations need: [`buildcards!`](@ref) needs the
stack to build into, [`showclip!`](@ref) the scroll position to keep, and
`applyfilter!` the query and the label that counts what it hid. Reached through
`player.fxpanel`, so adding an effect can build its card without pulling a
closure out of a `Dict{Symbol, Any}` under a symbol and calling it if it happens
to be there.

`rows` is which row of `stack` a clip's cards live in. A clip's cards last as long
as the clip — they are hidden and shown, never rebuilt — and it keeps its row, so
an undo that brings a deleted clip back builds into the row it had.

Declared here rather than in fxpanel.jl for the reason [`FxGraph`](@ref) is
declared in clips.jl: [`Player`](@ref) has a field of it, and a field typed
`::Any` costs every method that reads it its dispatch.
"""
mutable struct FxPanel
    const scroll::Makie.Subfigure
    const stack::GridLayout           # one row per clip that has cards
    const tools::GridLayout           # …and the tool-only cards, below them
    const query::Observable{String}
    const countlabel::Makie.Label
    const uicolors::NamedTuple
    const rows::Dict{UInt64, Int}     # clip id → its row of `stack`
    lastrow::Int
    # The "no clip" / "no effects" message, or empty. Blocks rather than a Card:
    # it is a sentence, and it is what the panel says INSTEAD of a stack.
    const empty::Vector{Any}
    # Tool-only cards on screen, by kind — the add menu opens one, its × closes
    # it. Cards, not a card set: each is a singleton and owns itself.
    const toolcards::Dict{Symbol, Makie.Card}
end

FxPanel(scroll, stack, tools, query, countlabel, uicolors) =
    FxPanel(scroll, stack, tools, query, countlabel, uicolors,
            Dict{UInt64, Int}(), 0, Any[], Dict{Symbol, Makie.Card}())

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
mutable struct Player <: Editor
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
    # marks because they are a different kind of statement: a mark is an input the
    # propagator runs from, a repair is an output that overrules what it produced.
    # `runmatte!` re-applies these after every full analysis, or rebuilding the
    # track would throw them away without a word.
    const matterepairs::Dict{UInt64, Dict{Int, Matrix{UInt8}}}
    # `(clip, srcframe, mask)` while a brush stroke is in flight, else `nothing`.
    # The mask is a copy of the frame's alpha; the stroke paints into it and only
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
    composebuf::RGBFrame  # frames are composed here and published to `frame` as one
                          # copy of the finished image. GLMakie samples `frame[]`
                          # lazily at render time, so decoding or warping in place
                          # there shows half-processed frames during playback
    # The Effects panel, once it is built. Its cards belong to the effects (see
    # [`Effect`](@ref)`.card`); this is the stack they are built into and the
    # controls above it.
    fxpanel::Union{Nothing, FxPanel}
    # Whose cards are on screen. Switching clips hides one set and shows another
    # — see [`showclip!`](@ref) — so this is what "another" is measured against.
    shownclip::Union{Nothing, Clip}
    # Which card is selected, as `(:fx, effect id)`. An Observable because the
    # cards derive their highlight from it, so selecting one is a write rather
    # than a rebuild of the stack.
    const fxselection::Observable{Union{Nothing, Tuple{Symbol, UInt64}}}
    # The keyframe anchor being edited, as `(parameter, key index)`. One truth for
    # "which anchor is selected"; each lane derives from it whether it draws
    # handles, so at most one ever does.
    const selectedkey::Observable{Union{Nothing, Tuple{Param, Int}}}
    # Which lanes are soloed on the timeline, and what was drawn before — see
    # [`sololanes!`](@ref). An Observable because the ∿ of every row, section and
    # card derives its colour from it: one input, one derivation per eye, rather
    # than a listener per parameter in the group.
    const lanesolo::Observable{Union{Nothing, LaneSolo}}
    const fxwidgets::Dict{Symbol, Any}  # panel menu/buttons (dock content is
                                        # invisible to fig.content — tests and
                                        # MCP reach the widgets through here)
    # Listeners this player put on MODULE-level observables (the effect registry's
    # version). They outlive the player unless taken off: a closed player's
    # handler `put!`s onto a closed `uiqueue` and throws, and `notify` then
    # abandons the rest of the list — which is a live player's menu going stale.
    # `close` walks this.
    const globallisteners::Vector{Observables.ObserverFunction}
    cropanchor::Union{Nothing, Point2f}
    lastcrop::NTuple{4, Float64}
    lastcanvas::Tuple{Int, Int}   # the published frame's size — the limits' only input
    lastclip::Union{Nothing, Clip}
    clipmodal::Any
    rctime::Float64
    gpuworker::Any
    # true once the render engine's context turns out to be owned by this thread
    # (see `runowned`); discovered at run time, not configured
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
    # Bumped by every change to the document — see [`edited!`](@ref). Everything
    # that draws from the edit rather than from the frame listens here, and the
    # autosave compares it against `autosaved`.
    const edited::Observable{Int}
    autosaved::Int          # `edited[]` at the last autosave
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
    const jobprogress::Threads.Atomic{Float64}  # running job fraction 0..1, NaN when idle
                                                # (written from worker threads, polled by the UI)
    const gpucache::Dict{Any, Any}       # source → device-resident decoded frames (pure-GPU playback)
    engine::FxEngine                     # the render engine, on the declared backend.
                                         # Not const: `autodetectgpu!` can upgrade the
                                         # backend, and the engine follows it.
    # Which effects this editor offers. Defaults to the module registry, where
    # load-time and plugin registration land; a second editor can be given its own.
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
    # The project file this edit belongs to, `nothing` until one is opened or
    # saved. Not derived from the first source: two projects can cut
    # the same footage, and deriving meant that opening `lego.videoedit` and
    # pressing Ctrl+S wrote `demo_source.videoedit` instead. The edit forked in
    # silence, and the checkpoint list, the autosave and the recovery prompt all
    # followed the wrong file with it.
    projectpath::Union{Nothing, String}
end

"""
    docsnapshot(player) -> (clips, mattemarks, repairs, captions, canvas, narration)

Everything an undo has to put back: the timeline and the inputs that produced
what is rendered on it. The matte's seed marks live next to the player rather
than on a clip, so a snapshot of the sequence alone restores a clip whose matte
has been removed, with no way to get the marks back.

The masks themselves are shared, not copied: a mark is replaced when it changes
and never written through, so a hundred undo steps cost a hundred dictionaries
rather than a hundred megabytes.
"""
docsnapshot(player::Player) =
    (snapshot(player.sequence),
     Dict{UInt64, Dict{Int, Matrix{UInt8}}}(k => copy(v) for (k, v) in player.mattemarks),
     Dict{UInt64, Dict{Int, Matrix{UInt8}}}(k => copy(v) for (k, v) in player.matterepairs),
     # The transcript: `restore!` keeps the same Sequence object, so captions would
     # otherwise survive an undo by accident. Re-transcribing replaces a transcript
     # the user may have corrected by hand, and that has to be undoable.
     copy(player.sequence.captions),
     # The canvas and the narration live on the sequence rather than on a clip,
     # and `snapshot(seq)` returns the clip vector alone, so without these,
     # cropping the canvas outward or adding a voiceover is not undoable.
     player.sequence.canvas,
     # The vector is copied so add/remove is undoable; the `Narration`s in it are
     # SHARED, on the same terms as the matte marks above — safe only because a
     # re-render replaces the element rather than writing through it.
     copy(player.sequence.narration))

"Put a [`docsnapshot`](@ref) back."
function docrestore!(player::Player, snap)
    clips, marks, repairs, caps, canvas, narration = snap
    restore!(player.sequence, clips)
    empty!(player.mattemarks)
    for (k, v) in marks
        player.mattemarks[k] = copy(v)
    end
    # `snapshot` shares the mattetrack rather than copying it, so restoring the
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
    push!(player.undostack, docsnapshot(player))
    length(player.undostack) > 100 && popfirst!(player.undostack)
    empty!(player.redostack)
    return nothing
end

"""
    changesummary(from, to) -> String

What differs between two [`docsnapshot`](@ref)s, in the user's terms.

Without it an undo says nothing: the picture changes and what came back is left
to work out
had come back — which on a long timeline, where the change may be off screen, is
the difference between confidence and pressing Ctrl+Z twice to see.

Derived rather than labelled: a description per edit would mean touching all 62
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
    # A restore can hand the same id a NEW object — a clip that was deleted and
    # undone. The selection still names it, so the panel has to be re-aimed at the
    # object; an ordinary edit does not replace objects and does not come here.
    player.shownclip = nothing
    showclip!(player)
    redraw!(player)
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

Fire and forget: the return value says nothing about whether `f` ran, finished
or threw. To wait for a result use [`rungpusync`](@ref), which carries the
exception back and rethrows it at the caller.

`rungpu(...) do; …; flag[] = true; end` followed by `while !flag[]` is a
synchronous wait written as an asynchronous one: when the job throws, the flag
is never set and the loop spins at full CPU with nothing logged.
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
    # a project path opens the saved edit instead of a video
    isproject = endswith(lowercase(path), ".videoedit")
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
         # Selection gets its own colour rather than the accent. The accent
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
    # …only the ones with a file: a proxy and a GPU decode stream are both about
    # reading media, and a clip that renders its frames has none.
    for src in unique(c.source for c in sequence.clips if decodable(c.source))
        needsproxy(src; maxpixels = proxythreshold) && startproxy!(player, src)
        # with the GPU preview on, stream-decode each source on the GPU so playback
        # runs purely on the GPU (background; CPU decode until the stream is ready)
        wantgpu && Threads.@spawn preloadgpu!(player, src)
    end
    # The repairs come back with the project. They live here rather than on the
    # sequence, so `loadproject` cannot restore them — see `loadrepairs!` for what
    # losing them silently cost.
    # …and the edit now belongs to that file: Ctrl+S goes back to it rather than
    # to a name derived from the footage.
    isproject && (player.projectpath = String(path); loadrepairs!(player, path))
    autodetect && Threads.@spawn autodetectgpu!(player)  # enable GPU playback if capable
    Threads.@spawn begin  # keep the regenerable proxy/PCM caches bounded
        pruned = prunecache!()
        pruned > 0 && setstatus!(player, "cache pruned — freed $(round(pruned / 2^30, digits = 1)) GiB")
    end
    return player
end

"""
    readerkey(seq, clip) -> Any

Which decode reader `clip` should use: its source when sharing one is fine, or
the clip itself when it is not.

Sharing a reader per source measures faster for a blend — 0 stand-ins and
5.4 ms/frame against 8.2 for a reader per layer (2026-07-28) — because the two
layers sit a few frames apart and one ring holds both positions.

It is much slower for a clone: two clips of one file overlapping at an offset
larger than the ring make that single reader seek twice for every frame drawn,
which is the one thing a long-GOP codec is worst at: each seek walks from a
keyframe. Measured on the bird clip, a clone pasted at the playhead (150 frames
of offset, against a 120-frame ring) costs 55.7 -> 46.5 fps, and it hits
scrubbing as hard as playback.

So the rule is the distance, not the count. Clips that overlap within a ring's
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

Compared at a frame they share, so a retimed clip is handled by `sourceframe`
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
            showplayhead!(player)  # re-present through the proxy
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
    # column 1: far-left vertical toolbar; column 2: the dock, one fixed slot where
    # the effects / media / export panels open (never over the preview); column 3:
    # the preview. Timeline and controls span the full window width.
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

    # row 2: the timeline. Keyframe curves overlay the clips directly (one editor,
    # no separate lane); the row grows with the track count
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
                    similar(frame[]),
                    nothing,          # fxpanel — built below, in `buildfxpanel!`
                    nothing,          # shownclip — no cards yet
                    Observable{Union{Nothing, Tuple{Symbol, UInt64}}}(nothing),
                    Observable{Union{Nothing, Tuple{Param, Int}}}(nothing),
                    Observable{Union{Nothing, LaneSolo}}(nothing),
                    Dict{Symbol, Any}(), Observables.ObserverFunction[],
                    nothing, (0.0, 0.0, 1.0, 1.0), (0, 0), nothing, nothing, 0.0,
                    nothing, false, nothing, nothing, nothing, nothing, defaultsegmenter(),
                    Any[], Any[], 0.0, Observable(0), 0, nothing, 0, 0,
                    Dict{Symbol, Any}(), Observable(:none),
                    Observable(VideoSource[]), Any[], nothing, Observable(:none),
                    Ref(1.0), Observable(:opacity),
                    Threads.Atomic{Float64}(NaN), Dict{Any, Any}(), FxEngine(analysisbackend),
                    effects, Clip[], Threads.Atomic{Bool}(false),
                    Threads.Atomic{Bool}(false), nothing, nothing)
    # The sequence is open in this editor now, and every clip in it reaches the
    # editor through it — which is how putting an effect on one builds that
    # effect's card. See [`Editor`](@ref) and [`editorof`](@ref).
    sequence.editor = player
    # …and the clips that were already in it are drawn, the way a clip added later
    # is drawn by `addclip!`. This is the moment the sequence gained an editor.
    foreach(buildclipview!, sequence.clips)
    drawtransitions!(sequence)   # …and its cross-dissolve markers
    # …and one of them is selected. A panel aimed at nothing is an empty panel,
    # and opening a document is exactly the moment an editor establishes what is
    # being worked on — the same statement pasting and `addsceneclip!` make. Until
    # the refactor this was hidden: `editclip` fell back to the clip under the
    # playhead, so the panel had something to show without anyone having chosen it.
    selectfirstclip!(player)
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
    # The timeline row grows with the track count, and by the same 48 px per unit
    # of lane height when a track is resized (see `settrackedge!`), so enlarging a
    # lane takes space from the preview rather than from its neighbours.
    on(_ -> fittimelinerow!(player), fig.scene.viewport)  # …it follows a window resize
    # …and the track count, which is the fact the row's height is a function of.
    # `relayout!` writes it, and only on a change.
    on(_ -> fittimelinerow!(player), timeline.ntr)
    # An edit that adds or removes a track calls it through `redraw!`. It
    # used to listen to the PLAYHEAD for that — the playhead standing in for
    # "something changed", which is the one thing it does not mean.
    # …and a lane dragged taller calls it itself, through the sequence — see
    # `fittimelinerow!`. It used to be handed in as `timeline.onlayout`.
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
    # What the drag is about to produce, next to the rectangle.
    #
    # Derived from `croprect` alone, so there is no second piece of state that can
    # disagree with the rectangle on screen. Before this the size only appeared in
    # the status bar after releasing, which is the wrong moment: the number is what
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
        # Preview pixels → the source fraction the crop stores → canvas pixels,
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

    # 392, not 360: a parameter row is 256 px (label + slider + the ◀ ◆ ▶ trio) and
    # sits under a section indent, which at 360 put its right end 22 px past what
    # the card's scene draws — the ▶ was clipped away entirely, silently.
    fxdock = dockpanel!(player, :effects; width = 392)
    buildfxpanel!(player, fxdock[1, 1], uicolors)
    buildpalette!(player, uicolors)      # Ctrl+P: every command, one search box
    buildkeyframemodal!(player, uicolors)   # ◆: what is animated on this clip
    fxbtn = toolbarbutton!(player, toolbar[1, 1], "FX", :effects, uicolors)
    # The media bin lists files only. A clip that renders its own frames has
    # nothing to import, re-link or proxy, and is already on the timeline.
    player.mediasources[] =
        unique(VideoSource[c.source for c in sequence.clips if c.source isa VideoSource])
    mediadock = dockpanel!(player, :media)
    buildmediabin!(player, mediadock[1, 1], uicolors)
    exportdock = dockpanel!(player, :export)
    buildexportpanel!(player, exportdock[1, 1], uicolors)
    # toolbar in three groups: the docks (FX · Bin · Out · Prj), the edit tools
    # (✂ ▢), the one-shots (✕ ↶ ↷).
    #
    # There is no separate Tools dock: stabilize, the matte, the flicker fix,
    # restore, the loop finder and blend are effect kinds like Blur, so they are
    # cards in the Effects panel. Two panels building the same card bodies also
    # meant two writers for `:toolslots`.
    binbtn = toolbarbutton!(player, toolbar[2, 1], "Bin", :media, uicolors)
    outbtn = toolbarbutton!(player, toolbar[3, 1], "Out", :export, uicolors)
    projectdock = dockpanel!(player, :project; width = 300)
    buildprojectpanel!(player, projectdock[1, 1], uicolors)
    prjbtn = toolbarbutton!(player, toolbar[4, 1], "Prj", :project, uicolors)
    buildkeyframeeditor!(player)   # the keyframe gestures on the timeline
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
    # A tooltip is a solid little card, not glowing text: a stroked glyph over the
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
        # Multi-line: most tips are two words, but an effect card's ? holds its
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
    # The tool cursor is scoped to where the tool can act: the blade scissor only
    # over a cuttable clip on the timeline, the crop crosshair only over the
    # preview — hovering buttons or panels always shows a normal arrow.
    lastcursor = Ref(:arrow)
    function refreshcursor!()
        mp = Point2f(events(fig).mouseposition[])
        t = player.tool[]
        shape = :arrow
        # A pending pick first: it claims the next click on the preview whatever
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
            # Ctrl advertises itself: it is the only way to drag a clip, and a plain
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
            # Show the framing that is about to change. Without this every crop is
            # a blind redo: the rectangle you drag has no relationship on screen to
            # the one already in force, so refining a crop means guessing where it
            # currently is and starting over.
            showcurrentcrop!(player)
            setstatus!(player, "crop tool — drag a rectangle on the preview " *
                               "(it may reach outside the picture; Esc to put it away)")
        end
    end
    # split tool: the next timeline click cuts there, not at the playhead
    on(events(fig).mousebutton; priority = 95) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        player.tool[] === :split || return Consume(false)
        tlscene = timeline.axis.scene
        # is_mouseinside, not `in viewport`: an active blade must not cut through
        # a dropdown or a modal that happens to hang over the timeline
        Makie.is_mouseinside(tlscene) || return Consume(false)
        t, y = Makie.mouseposition(tlscene)
        # the scrub strip stays the playhead's whatever tool is up
        # ([`SCRUBBAND`](@ref)), so the cut can be lined up before it is made
        inscrubband(y) && return Consume(false)
        # not named `frame`: that would rebind the shared preview Observable this
        # scope captures (used by image! and the scrub fallback)
        cutat = clamp(round(Int, t * sequence.framerate), 0, max(seqlength(sequence) - 1, 0))
        snapshot!(player)  # so Ctrl+Z / the undo tool can revert the cut
        split!(sequence, cutat)
        redraw!(player)
        # Persistent blade: stays active so the mouse keeps cutting (DaVinci blade).
        # Esc or clicking ✂ again puts it away.
        setstatus!(player, "cut at $(timecode(sequence, cutat)) — blade still active (Esc to stop)")
        return Consume(true)
    end
    opendock!(player, :effects)   # the working panel starts open

    # WHICH CLIP IS EDITED IS THE SELECTION'S ANSWER, and only the selection's.
    # The playhead says which PICTURE is on screen; it does not pick what you are
    # working on, and a panel that followed it re-aimed itself every time playback
    # crossed a cut.
    on(_ -> showclip!(player), timeline.selected)
    on(_ -> showplayhead!(player), playhead)

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
# device-resident frames — so displaying a Player at the REPL pulled megabytes
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
    names = unique(basename(sourcepath(c.source)) for c in seq.clips)
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
    print(io, "  edits      ", p.edited[], " (", nundo, " undo, ", nredo, " redo)")
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
the two source frames, which say nothing about where on the timeline this is, and
[`publishframe!`](@ref) needs the timeline frame to know which overlays are up.
"""
function showtransition!(player::Player, sample, n::Integer)
    left, srcA, right, srcB, p = sample
    spA = pool(player, left)
    spB = pool(player, right)
    settarget!(spA.worker, srcA)
    settarget!(spB.worker, srcB)
    bufA = RGBFrame(undef, spA.source.width, spA.source.height)
    fetchframe!(bufA, spA.ring, srcA) || return false
    bufB = RGBFrame(undef, spB.source.width, spB.source.height)
    fetchframe!(bufB, spB.ring, srcB) || return false
    ensureframesize!(player, canvassize(player.sequence))
    # Two layers in one composite, the outgoing under the incoming, with the
    # dissolve's fraction as the incoming layer's opacity. Compositing opaque `B`
    # over `A` at `α` is `α·B + (1-α)·A`, the same picture the host-side lerp
    # produced, on the path every other frame takes.
    #
    # The lerp blended two already-fitted frames and returned false (a black
    # preview) when the clips had different resolutions. Here each layer is
    # fitted to the canvas on its own.
    #
    # `applytracks = false` is hold-to-compare; the chains read it themselves.
    prerenderscenes!([left, right], n)
    ok = runowned(player) do
        composite(player.engine, [left, right], n,
                  (clip, _) -> clip === left ? bufA : bufB;
                  canvas = canvassize(player.sequence),
                  applytracks = player.applytracks[], playing = player.playing[],
                  alphafor = (clip, _) -> clip === right ? Float64(p) : nothing) do canvas
            copyto!(player.frame[], canvas)
        end
    end
    ok === true || return false
    publishframe!(player, n)
    applycrop!(player, left)  # both sides share framing in the common (split) case
    return true
end

"""
Composite the stack of clips covering timeline frame `n` (bottom track → top) into
the preview: each layer is decoded, its tracks and effects applied, its crop baked
into the shared canvas via `warp!`, then alpha-blended by its opacity (upper over
lower).
Returns `false` (caller falls back to the single-clip path) if any layer isn't buffered
yet. CPU preview path — the GPU/export paths still show the top clip for now.
"""
function compositeframe!(player::Player, n::Integer, clips::Vector{Clip})
    canvas = canvassize(player.sequence)   # the sequence's format, not the top layer's
    ensureframesize!(player, canvas)
    # the CPU tier of one composite (see `composite`): its only job is to hand
    # each layer a decoded frame from that source's ring, so everything the
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
    # scenes first, on this thread (see `prerender!`): `runowned` below hands the
    # frame to whichever thread owns the Lava context, and a GLMakie screen cannot
    # follow it there
    prerenderscenes!(clips, n)
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
    fittimelinerow!(player; solorow = 0.35, keepfree = 620.0) -> nothing

Give the timeline's figure row the height its lanes need.

Grows with the track count, and by the same 48 px per unit of lane height when a
track is resized (see `settrackedge!`), so enlarging a lane takes space from the
preview rather than from its neighbours. A soloed lane takes `solorow` of the
figure instead of just the other lanes' share; `keepfree` is the height the rest
of the window keeps, because the tool column is a stack of fixed-size buttons that
Makie does not clip — at 0.45 the last two landed on the ruler and the effect
panel was cut off mid-parameter.

`timeline.rowheight` is what it last asked for: `rowsize!` invalidates the layout
whether or not the number changed, and this runs on every window-resize event.
"""
function fittimelinerow!(player::Player; solorow = 0.35, keepfree = 620.0)
    seq = player.sequence
    tl = player.timeline
    ntr = ntracks(seq)
    base = (112 + 48 * (ntr - 1)) * lanescale(seq, ntr)
    winh = Float64(player.fig.scene.viewport[].widths[2])
    h = seq.solo == 0 ? base : max(base, min(solorow * winh, winh - keepfree))
    isapprox(h, tl.rowheight; atol = 0.5) && return nothing
    tl.rowheight = h
    rowsize!(player.fig.layout, 2, Makie.Fixed(h))
    return nothing
end

"""
    trimpreview!(player, clip, srcframe) -> nothing

While a trim drag is on, show the frame the cut would land on — exact when it is
decoded, the decoder's nearest otherwise so the picture still follows the drag.
The playhead is not moved: what is being looked at is the edge, not the position.
"""
function trimpreview!(player::Player, clip::Clip, sf::Integer)
    if !presentclipframe!(player, clip, sf; standin = false)
        presentclipframe!(player, clip, sf; standin = true)
        retrytrim!(player, clip, sf)      # …and land the exact frame when it decodes
    end
    return nothing
end

"""
    trimend!(player) -> nothing

The trim drag is over: stop chasing the edge frame. Without this the retry
outlives the release and republishes the edge over the playhead — the same bug as
[`trimpreview!`](@ref)'s, pointing the other way.
"""
trimend!(player::Player) = (player.trimtarget = nothing; nothing)

"""
Is the playhead parked (put there by a click, a seek or a step) rather than
dragged or played? A parked playhead is shown the exact frame; a moving one takes
the decoder's nearest stand-in (see the `standin` policy of
[`showframe!`](@ref)).
"""
atrest(player::Player) = !player.playing[] && !player.timeline.scrubbing[]

"""
    selectfirstclip!(player) -> nothing

Aim the editor at the sequence's first clip, unless something is already selected.

What opening a document does. Selection is never derived — see [`editclip`](@ref) —
so a document that has just been read has to be given one, or the panel is aimed
at nothing and shows nothing.
"""
function selectfirstclip!(player::Player)
    clips = player.sequence.clips
    isempty(clips) && return nothing
    selectedclip(player) === nothing || return nothing
    player.timeline.selected[] = first(clips).id
    return nothing
end

"""
    selectedclip(player) -> Union{Nothing, Clip}

The clip that is SELECTED, or `nothing`. What the panel edits.

The playhead says which picture is on screen, and nothing else. Deriving the
edited clip from it meant the panel re-aimed itself every time playback crossed a
cut, and that anything the panel owned had to be rebuilt on a playhead move.
"""
selectedclip(player::Player) = clipbyid(player.sequence, player.timeline.selected[])

"""
    editclip(player) -> (clip, source_frame) | nothing

The clip the inspector works on: [`selectedclip`](@ref), and the source frame the
playhead is asking of it.

Nothing selected is `nothing` — there is no "…else the clip under the playhead".
That fallback made the playhead choose what is edited: crossing a cut re-aimed
every tool, and the panel (which follows the selection) and the tools (which
followed the playhead) could be pointed at two different clips at once — which is
how a card the panel had never built got asked for.

Selection is per clip, so it holds across cuts and across stacked lanes: the
preview shows the upper clip while the lower one stays editable.

The source frame is clamped to the clip's extent, so a playhead outside it reads
the clip's first or last frame instead of a frame the source has not got.
"""
function editclip(player::Player)
    c = selectedclip(player)
    c === nothing && return nothing
    n = clamp(player.playhead[], c.start, max(clipend(c) - 1, c.start))
    return (c, sourceframe(c, n))
end

"""
    publishframe!(player, n) -> nothing

Put what is in `player.frame[]` on screen as timeline frame `n`: texture
re-pointed, observable notified.

Every preview path ends here.

This used to render as well: a second compositing stage that pushed the finished
frame into a Makie scene, drew the overlays over it and read it back. Titles,
subtitles and 3D scenes are clips now, composited by the graph in track order
with an opacity and an effect stack, so there is nothing left to draw here.

Main thread only — it re-points a GL texture.
"""
function publishframe!(player::Player, n::Integer)
    showcpuframe!(player)
    notify(player.frame)
    return nothing
end

"""
Resolve and show timeline frame `n` if possible (gaps show black). Returns success.

`standin` decides what a still-decoding GPU stream may put on screen. While the
playhead moves — playback, a scrub drag — [`frameat!`](@ref)'s nearest already
decoded frame keeps the picture following the drag instead of freezing. While it
is parked, each retry would blit a closer stand-in, so one click into a cold GOP
replays it into the preview (8 stand-ins on a 300-frame GOP, measured) and the
last one stays if the settle is cut short. A parked present therefore only
advances the decode and reports failure; the retry loop lands the exact frame.
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
            if gpuready(player) && streamed(player, clips)
                # composites mix layers: one stand-in among them dates the whole frame
                standin || primecomposite!(player, clips, n) || return false
                shown = presentgpucomposite!(player, clips, n;
                                             chunks = standin && !player.playing[] ? 0 : 5)
            end
            shown || (shown = compositeframe!(player, n, clips))
            if shown
                # every layer's crop is baked into the composited canvas, so the
                # view shows the whole canvas — one rule above the tier split.
                # The GPU tier used to leave the single-clip present's crop on
                # the axis, applying a stabilized clip's framing twice for the
                # length of a blend. The size is the sequence's: measuring it by
                # the top layer described one format while the buffer held
                # another as soon as the layers differed in size.
                cw, ch = canvassize(player.sequence)
                player.lastcrop = (0.0, 0.0, 1.0, 1.0)
                coverlimits!(player, 0, cw, ch, 0)
                return true
            end
        end
    end
    loc = locate(player.sequence, n)
    if loc === nothing
        # Straight into the published frame. `composebuf` is sized to the decode
        # resolution (`ensuredecodesize!`) and `frame[]` to the canvas, so blacking
        # one and copying it into the other only matches when a clip happens to
        # match the canvas — a 320x180 decode buffer went into a 214x108 frame.
        fill!(player.frame[], RGB{N0f8}(0, 0, 0))
        # black but not bare: an overlay belongs to the sequence, not to a clip,
        # so one standing over a gap is still drawn. Titles live in gaps.
        publishframe!(player, n)
        return true
    end
    clip, srcframe = loc
    target, protect = decodetarget(player, n, clip, srcframe)
    return presentclipframe!(player, clip, srcframe; standin, target, protect)
end

"""
    presentclipframe!(player, clip, srcframe; standin, target, protect) -> Bool

Put `clip`'s source frame `srcframe` on screen: the whole present path (GPU
stream, GPU effects over a CPU-decoded frame, or the CPU tier), addressed
directly instead of resolved from the playhead. [`showframe!`](@ref) uses it for
the playhead's frame; the trim gesture uses it to show the frame at the edge it
is dragging without moving the playhead. `target`/`protect` steer the decode
worker (see [`decodetarget`](@ref)).
"""
function presentclipframe!(player::Player, clip::Clip, srcframe::Integer;
                           standin::Bool = !atrest(player),
                           target::Integer = srcframe,
                           protect::UnitRange{Int} = 1:0)
    # pure-GPU path: a streaming decoder feeds this source — Vulkan-Video decode
    # into a bounded VRAM ring plus effects on device, no CPU decode, no upload
    if gpuready(player) && haskey(player.gpucache, readerkey(player.sequence, clip))
        stream = player.gpucache[readerkey(player.sequence, clip)]
        if 0 <= srcframe < nframes(stream)
            # parked on an undecoded frame: advance the feed but leave the last
            # exact image up — the retry loop calls back until this frame lands
            if !standin && !hasframe(stream, srcframe)
                primeframe!(player, stream, srcframe) && return false
            end
            ensureframesize!(player, canvassize(player.sequence))
            # A stand-in must not decode: `frameat!` spends up to 5 decode chunks
            # reaching `srcframe` before settling for the nearest decoded frame,
            # which on a seek is ~62 ms of the work this call exists to skip. With
            # 0 chunks the seek answers in ~1 ms and the retry loop fetches the
            # real frame.
            if presentgpu!(player, clip, srcframe; stream = stream,
                           chunks = standin && !player.playing[] ? 0 : 5)
                applycrop!(player)
                # a scrub into an undecoded region shows the nearest decoded frame
                # immediately; reporting "not presented" keeps the paused retry
                # loop refining in ~90 ms steps until the exact frame is up
                return hasframe(stream, srcframe)
            end
        end
    end
    # A clip that renders its frames has no decoder, no ring and no proxy: it goes
    # straight to the composite, the same call the decoded path makes below once
    # it has a frame in hand.
    if !decodable(clip.source)
        ensureframesize!(player, canvassize(player.sequence))
        prerender!(clip.source, clip, srcframe)      # …while still on this thread
        ok = runowned(player) do
            composite(player.engine, [clip], n_of(player, clip, srcframe), (_, _) -> nothing;
                      canvas = canvassize(player.sequence),
                      applytracks = player.applytracks[], playing = player.playing[]) do canvas
                copyto!(player.frame[], canvas)
            end
        end
        ok === true || return false
        publishframe!(player, n_of(player, clip, srcframe))
        applycrop!(player)
        return true
    end
    sp = pool(player, clip)
    settarget!(sp.worker, target; protect)
    ensuredecodesize!(player, sp.source)   # proxy resolution when one is active
    buf = player.composebuf
    if fetchframe!(buf, sp.ring, srcframe)
        ensureframesize!(player, canvassize(player.sequence))
        if gpuready(player)
            if presentgpu!(player, clip, srcframe; source = buf)
                applycrop!(player)
                return true
            end
        end
        # One layer placed into the canvas, the same call the stack takes, so a
        # crop, a reframe and a rotation mean the same here as in the export.
        # Publishing the raw layer and letting the axis limits stand in for the
        # framing made the framing a viewport: the crop removed nothing that could
        # not be scrolled back to, and a rotation could not be shown at all.
        prerender!(clip.source, clip, srcframe)
        ok = runowned(player) do        # the engine's context has one owning thread
            composite(player.engine, [clip], n_of(player, clip, srcframe),
                      (_, _) -> buf; canvas = canvassize(player.sequence),
                      applytracks = player.applytracks[], playing = player.playing[]) do canvas
                copyto!(player.frame[], canvas)
            end
        end
        ok === true || return false
        # this path is addressed by source frame (a trim drag reaches it as well as
        # the playhead), so the timeline frame is resolved back with the same
        # `n_of` the composite above is given
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
Size the published frame, which is always the sequence canvas.

The preview shows the canvas, exactly what the export writes: a clip's crop, its
scale/position and its rotation are baked into it by the composite. It used to be
the rendered layer at source resolution with the preview's axis limits standing in
for the framing, which made the framing a viewport — cropping removed nothing that
zooming out could not bring back, and rotation could not be shown at all.
"""
function ensureframesize!(player::Player, wh::Tuple{Integer, Integer})
    size(player.frame[]) == (Int(wh[1]), Int(wh[2])) && return nothing
    player.frame.val = RGBFrame(undef, Int(wh[1]), Int(wh[2]))  # notified once filled
    player.lastcrop = (-1.0, 0.0, 0.0, 0.0)  # force applycrop! (limits are per-source)
    return nothing
end

"""
Size the decode buffer: source resolution (or the proxy's), independent of the
canvas, since decoding happens in the material's own pixels and placement follows.
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

[`trimpreview!`](@ref) is best-effort: if the decoder has not reached the frame yet it
shows the nearest one instead. Nothing then corrected it, because the next
attempt only arrived with the next mouse move, so a trim that ended on an
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
    refinepreview!(player) -> nothing

Keep adding samples to a PROGRESSIVE clip's picture while the playhead stands
still.

A path tracer's frame is a running average: the seek returns one sample in
subseconds and this adds to it until it converges or the playhead moves. Rendering
a full sample budget on every playhead move instead is a freeze per click, which
is what this replaces — and it works only because the scene's screen is held
across frames (see `SceneSource`).

Unbounded: it stops when the playhead moves, when playback starts, or when
nothing visible is progressive, and otherwise keeps improving the picture you are
looking at. Shares `player.refining` with
[`retrypresent`](@ref): both mean "the picture on screen is not final yet", and
exactly one of them should be improving it.
"""
function refinepreview!(player::Player)
    atrest(player) || return nothing
    any(c -> refining(c.source), clipsat(player.sequence, player.playhead[])) || return nothing
    Threads.atomic_cas!(player.refining, false, true) === false || return nothing
    @async begin
        n = player.playhead[]
        try
            while atrest(player) && player.playhead[] == n
                clips = clipsat(player.sequence, n)
                any(c -> refining(c.source), clips) || break
                present!(player) || break
                yield()
            end
        catch e
            @warn "progressive refinement stopped" exception = (e, catch_backtrace())
        finally
            player.refining[] = false
        end
        # Whoever asked for the frame this loop just abandoned got nothing: the
        # playhead moved, `showplayhead!` ran for the new frame, and both refiners
        # it can reach found the slot still held here. Ask again now that it is
        # free — without this a scrub over a raytraced clip left every frame after
        # the first at one sample. Only on a move, so a `present!` that keeps
        # failing cannot respawn this forever.
        #
        # Handing the slot back BETWEEN samples instead does not work: the loop
        # then loses it to whatever task happens to run at that yield, and
        # measured, it never survived past its first sample.
        player.playhead[] == n || showplayhead!(player)
    end
    return nothing
end

"""
Keep refining until the exact frame under the playhead is on screen (each attempt
advances the stream's decode). If it does not arrive inside `budget` seconds the
preview keeps the decoder's best frame and the status line says so.

One refiner at a time, reading the playhead rather than capturing it.

This used to spawn a task per call, each closing over the frame it was asked for
and looping `while playhead[] == n`. A scrub calls it once per frame, so dozens
ran at once, and the guard was checked before the render: a task could pass it,
spend a few milliseconds compositing and publish after a newer task had already
published a later frame. The screen then held a frame from behind the playhead,
more often the heavier the chain — visible as a matte or stabilized warp lagging.

The single task reads `playhead[]` fresh on every pass, so the frame that reaches
the screen is the last one asked for. `showframe!` is unchanged: it puts frame
`n` on screen, which the trim preview still uses for an edge frame the playhead
is nowhere near.
"""
function retrypresent(player::Player; budget::Real = 20.0)
    Threads.atomic_cas!(player.refining, false, true) === false || return nothing
    @async begin
        try
            deadline = time() + budget
            drawn = -1                   # the frame a stand-in is already up for
            while !player.playing[]
                n = player.playhead[]    # the current frame, never a captured one
                # Show a stand-in first, then refine. A full-quality seek lands
                # mid-GOP and decodes forward from the preceding keyframe: ~2.1 ms
                # per frame on a 250-frame GOP, so 90 to 455 ms depending on where
                # in the GOP it falls, with the previous frame on screen
                # throughout. The compositor is not in it — playback renders a
                # stand-in in 0.9 ms.
                #
                # `standin = true` draws the nearest decoded frame in about a
                # millisecond and returns whether it was the exact one, so one call
                # both answers immediately and says whether anything is left to
                # refine.
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
                    player.playhead[] == n && break  # …unless it moved meanwhile
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
        # The frame is up — and if anything on it is progressive, it is up at ONE
        # sample. `showplayhead!` reaches exactly one of the two refiners and it
        # chose this one, because the decode was not ready when it asked; handing
        # over here is what lets a raytraced clip go on converging after a seek
        # that had to wait for its footage. Measured before this: every frame
        # after such a seek stayed at one sample, however long the playhead then
        # stood still.
        atrest(player) && refinepreview!(player)
    end
    return nothing
end

"""
Set the preview limits to exactly the given rect (y descending: the axis is
reversed, and Makie's `ylims!` derives `yreversed` from argument order). The
axis's `DataAspect` letterboxes the rect inside the cell, so pixels stay square
and nothing outside the crop is shown — a stabilization crop hides the warp's
replicate-border smear, which expanding past it would re-reveal.
"""
function coverlimits!(player::Player, xlo::Real, xhi::Real, yfirst::Real, ysecond::Real)
    limits!(player.previewaxis, xlo, xhi, yfirst, ysecond)
    return nothing
end

"""
    fxselected(player) -> key | nothing

Which effect card is selected, as `(:fx, id)`. `nothing` when none is — the
resting state, and what clicking the selected card returns to.
"""
fxselected(player::Player) = player.fxselection[]

"""
Select an effect card, or deselect it when it already was.

A write, not a rebuild: each card's highlight is derived from
`player.fxselection`, so the one that gains it and the one that loses it both
follow. The preview overlays are re-run because a card's selection is what
decides whether its direct-editing tool is on the picture.
"""
function selectfxcard!(player::Player, key)
    player.fxselection[] = fxselected(player) == key ? nothing : key
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
    # In preview pixels, like the drag's own rectangle: the preview may be a
    # proxy, so the crop's source fractions have to come through its size.
    x0, y0 = x * W, y * H
    x1, y1 = (x + w) * W, (y + h) * H
    player.croprect[] = Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return nothing
end

"""
Frame the preview on the whole canvas, every time.

There is nothing left for the limits to express: the crop, the reframe and the
rotation are baked into the published frame by the composite, as the export bakes
them. The limits used to be the framing, which made a crop a viewport rather than
an edit.
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

Both stores are keyed by clip id, and every way of making a second clip out of one
(splitting, copying, pasting) hands the new clip a fresh id. The tracks survive
that because they are keyed by absolute source frame and both clips share the
source; the player-side stores are not, so without this they do not follow.

The failure is quiet: the alpha comes along, so the picture is right, and only
re-running breaks — a split half meets an empty mark store, refuses to propagate,
and its repair cards are gone.

Copied whole rather than filtered to `to`'s range: a mark outside it is harmless
(every lookup is by source frame) and trimming the halves later would have to put
it back.
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
    # buffer size is read live each step, not captured: a proxy swap can land
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
        # While the button is down on the timeline the playhead is being placed, so
        # playback holds its clock instead of advancing between mouse moves and
        # pulling it away. Release resumes from wherever it was put.
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
    # Playback presents stand-ins to keep moving (frameat!'s latency budget) and
    # nothing retries while playing, so whatever stops it — Space, K, a GPU render
    # error, the end of the sequence — has to land the playhead's exact frame.
    retrypresent(player)
    return nothing
end

Base.seek(player::Player, t::Real) =
    (player.playhead[] = clamp(round(Int, t * player.sequence.framerate), 0,
                               max(seqlength(player.sequence) - 1, 0)); player)

# ---------------------------------------------------------------- editing

"Split / delete / crop-reset operate on the clip under the playhead."
function split!(player::Player)
    # A razor cuts what is under it, so this one is the playhead's to answer and
    # says so out loud (`editclip` no longer does). A selection only picks the
    # lane when clips are stacked — which one of them the blade goes through.
    loc = editclip(player)
    loc === nothing && (loc = locate(player.sequence, player.playhead[]))
    loc === nothing && return setstatus!(player, "nothing to split at the playhead")
    snapshot!(player)
    right = split!(player.sequence, player.playhead[], loc[1].track)
    if right === nothing
        pop!(player.undostack)
        return setstatus!(player, "cannot split here — the playhead is at the clip's start")
    end
    inheritmasks!(player, loc[1], right)
    redraw!(player)
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
        [c for c in seq.clips if c.id in marked]
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

Searching upward for a free lane fits a paste and not a drag: a paste has no
target lane under the cursor to honour, so the alternative is refusing outright.
A drag does have one, which is why `dragto!` refuses instead.
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
        addclip!(seq, pasted)
        push!(fresh, pasted)
        n += 1
    end
    sort!(seq.clips, by = c -> (c.track, c.start))
    # Select what was pasted: paste lands at the playhead on whichever lane was
    # free, so the next gesture is moving it. By identity, not by index — the sort
    # above has just moved everything.
    if !isempty(fresh)
        player.timeline.selected[] = first(fresh).id
        player.timeline.selection[] = UInt64[c.id for c in fresh]
    end
    redraw!(player)
    setstatus!(player, "pasted $n clip$(n == 1 ? "" : "s") — selected, Ctrl-drag to place")
    return n
end

function Base.deleteat!(player::Player)
    seq = player.sequence
    marked = player.timeline.selection[]
    if length(marked) > 1              # shift-click marks: delete the whole set
        clips = [c for c in seq.clips if c.id in marked]
        isempty(clips) && return nothing
        snapshot!(player)
        for c in clips                 # by identity — ripple keeps the rest current
            deleteclip!(seq, c)
        end
        player.timeline.selection[] = UInt64[]
        player.playhead[] = clamp(player.playhead[], 0, max(seqlength(seq) - 1, 0))
        setstatus!(player, "$(length(clips)) clips deleted — Ctrl+Z undoes")
        redraw!(player)
        return nothing
    end
    snapshot!(player)
    deleteclip!(seq, player.playhead[]) === nothing && return (pop!(player.undostack); nothing)
    player.playhead[] = clamp(player.playhead[], 0, max(seqlength(seq) - 1, 0))
    redraw!(player)
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
    redraw!(player)
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
        redraw!(player)
        return setstatus!(player, "removed transition at $(timecode(seq, at))")
    end
    lc, rc = transitionclips(seq, at)
    t = (lc === nothing || rc === nothing) ? nothing :
        addtransition!(seq, at; duration = defaultdissolve(seq, lc, rc))
    t === nothing && return setstatus!(player, "can't add a dissolve there — need a real cut between two clips")
    redraw!(player)
    setstatus!(player, "cross-dissolve at $(timecode(seq, at)) · $(round(t.duration / fps, digits = 2))s  (T to remove)")
    return nothing
end

resetcrop!(player::Player) = resetcropat!(player, player.playhead[])

function resetcropat!(player::Player, n::Integer)
    loc = locate(player.sequence, n)
    loc === nothing && return nothing
    snapshot!(player)
    loc[1].crop = (0.0, 0.0, 1.0, 1.0)
    redraw!(player)
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
    addclip!(seq, clip)
    registermedia!(player, source)
    # AXISTOP, not 1.0 — the axis runs past 1 to carry the scrub strip above the
    # lanes, and a y-limit of 1 pushes that strip off-screen: the playhead's own
    # band silently disappears the first time a source is added.
    limits!(player.timeline.axis, 0.0, seqduration(seq), 0.0, AXISTOP)  # reveal the new clip
    redraw!(player)
    note = conformnote(source, seq.framerate)
    setstatus!(player, "added $(basename(path)) — $(source.width)×$(source.height), " *
                       "$(round(source.nframes / source.framerate, digits = 1))s" *
                       (isempty(note) ? "" : " · " * note))
    needsproxy(source; maxpixels = player.proxythreshold) && startproxy!(player, source)
    return clip
end

"""
    projectfile(player) -> String

The file this edit is, or would be: the project it was opened from or last
saved to, and otherwise a name beside the first source.
"""
projectfile(player::Player) =
    something(player.projectpath,
              splitext(sourcepath(player.sequence.clips[1].source))[1] * ".videoedit")

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
        foreach(Makie.delete!, built); empty!(built)
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
            put!(player.uiqueue, () -> restoreproject!(player, String(f); adopt = true))
        catch e
            setstatus!(player, "file dialog failed: $(sprint(showerror, e))")
        end
    end
    # Not on the playhead: nothing this panel shows depends on it. `projectfile`
    # reads `player.projectpath` or `clips[1]` and the version rows come off disk,
    # so a playhead listener rebuilt the same Labels and Buttons and re-listed the
    # checkpoint directory every frame — 13.3 ms and 1.79 MB per playhead move
    # with the dock closed, the largest of the ten playhead listeners. It
    # refreshes when it opens (`opendock!`) and on a save, restore or edit.
    refresh()
    player.fxwidgets[:projectrefresh] = refresh
    return panel
end

"""
Load a saved version over the running edit.

Snapshot first: going back to a checkpoint is an edit like any other and Ctrl+Z
has to undo it. The file is read before anything is thrown away, so a corrupt
checkpoint leaves the session alone.

`adopt` says whether `file` becomes the project — true when opening one, false
for a checkpoint or an autosave. Adopting a checkpoint would make the next Ctrl+S
write into the checkpoint directory and leave the project at the older state.
"""
function restoreproject!(player::Player, file::AbstractString; adopt::Bool = false)
    seq = try
        loadproject(file)
    catch e
        setstatus!(player, "could not read $(basename(file)): $(briefly(e))")
        return nothing
    end
    snapshot!(player)
    # only when the checkpoint brought its own: `checkpointproject` copies the
    # project file and not the `.mattes` directory beside it, so clearing
    # unconditionally would drop the session's marks and put nothing back
    if isdir(mattedir(file))
        empty!(player.matterepairs); empty!(player.mattemarks)
    end
    # `addclip!`, not `append!`: the clips come from a sequence that was read on a
    # worker and is open in nothing, and they are joining the one on screen
    empty!(player.sequence)
    foreach(c -> addclip!(player.sequence, c), seq.clips)
    selectfirstclip!(player)   # …a freshly opened document has a clip to work on
    empty!(player.sequence.transitions); append!(player.sequence.transitions, seq.transitions)
    drawtransitions!(player.sequence)   # …the list changed, so the markers do
    loadrepairs!(player, file)               # …and the repairs saved beside it
    adopt && (player.projectpath = String(file))
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
    return joinpath(dirname(p), "." * basename(p) * ".autosave")
end

"""
    autosave!(player; force = false) -> path | nothing

Park the edit if it changed since the last time. Cheap enough to call on a timer:
without an edit it returns immediately, and it never checkpoints (an autosave is
not a version, it is a safety net).
"""
function autosave!(player::Player; force::Bool = false)
    isempty(player.sequence.clips) && return nothing
    (force || player.edited[] != player.autosaved) || return nothing
    path = autosavepath(player)
    try
        saveproject(path, player; checkpoint = false)
        player.autosaved = player.edited[]
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

An autosave newer than the project it belongs to, i.e. what a crash left behind.
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

Save the project and the edit state that lives on the player rather than on the
sequence, currently the matte repairs.

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
    player.autosaved = player.edited[]
    rm(autosavepath(player); force = true)   # the project is the newest copy again
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
        showplayhead!(player)
        showcrop!(player, oldcrop, clip.crop)
    else
        player.lastcrop = (-1.0, 0.0, 0.0, 0.0)
        showplayhead!(player)
    end
    return nothing
end

"Run a stabilization analysis for the clip at frame `at` (default: playhead), in the background."
function analyzeat!(player::Player, analyze!::Function, what::String;
                    at::Integer = player.playhead[])
    # at the playhead this is the inspector acting → same clip the panel shows;
    # with nothing selected, and for an explicit `at`, the frame answers instead
    loc = at == player.playhead[] ? editclip(player) : nothing
    loc === nothing && (loc = locate(player.sequence, at))
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
                # the framing change should be seen, not inferred from the
                # status line: when the view shows this clip, hold the old
                # framing while the (now warped) frame refreshes, then glide
                # to the new one; otherwise apply on the clip's next present
                shown = locate(player.sequence, player.playhead[])
                if shown !== nothing && shown[1] === clip
                    player.lastcrop = clip.crop      # presents keep hands off
                    showplayhead!(player)
                    showcrop!(player, oldcrop, clip.crop)
                else
                    player.lastcrop = (-1.0, 0.0, 0.0, 0.0)
                    showplayhead!(player)
                end
            else
                showplayhead!(player)
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
stabilized, cropped content so the camera is locked and the match is on the
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
            redraw!(player)
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
    # lead with what it achieved (measured while analyzing), not how hard it pushes
    got = track.reduction > 0 ? "flicker −$(round(Int, 100 * track.reduction))%, " : ""
    return "$(got)max adjustment $(round(Int, 100 * dev))%"
end

"""
    showplayhead!(player) -> nothing

Draw the frame the playhead is on, with the stand-ins and the refiner around it —
[`present!`](@ref) is the exact version, this is the whole of what the preview
does with a frame.

What an edit calls when the picture it changed has to be drawn again. The playhead
did not move, so notifying it would say something untrue — and did, at 39 call
sites, until five of the six things listening to it were not about the playhead at
all.

Moving the playhead is [`seek!`](@ref), which writes it and lands here through its
own listener.
"""
function showplayhead!(player::Player)
    n = player.playhead[]
    if present!(player)
        # The frame is up. If any of it is still converging — a raytraced scene
        # clip — keep adding samples while the playhead holds.
        refinepreview!(player)
        return nothing
    end
    if player.timeline.scrubbing[]
        loc = locate(player.sequence, n)
        if loc !== nothing
            clip, srcframe = loc
            # the scrub fallback blits a raw per-second thumbnail. That only
            # matches the composed preview when the clip is neither stabilized nor
            # cropped — otherwise it would flicker between the cropped/stabilized
            # exact frame and the raw thumbnail (and warping a per-second
            # thumbnail by a per-frame track just wobbles). For those clips, hold
            # the last exact frame.
            plain = (clip.motiontrack === nothing || !player.applytracks[]) &&
                    clip.crop == (0.0, 0.0, 1.0, 1.0) && !isanimated(clip)
            thumb = plain ? nearestthumb(cachefor(player.timeline, clip.source),
                                         floor(Int, srcframe / clip.source.framerate)) :
                    nothing
            if thumb !== nothing
                ensureframesize!(player, pool(player, clip).source)
                blitthumb!(player.frame[], thumb)
                # overlays on the stand-in too: an overlay blinking out for the
                # length of a scrub drag costs more than the few ms the pass takes
                # on a frame that is already inexact
                publishframe!(player, n)
            end
        end
    end
    player.playing[] || retrypresent(player)
    return nothing
end

"""
    redraw!(player) -> nothing

The document's SHAPE changed: put what is drawn from it back where it belongs,
and show the playhead's frame again.

It does not go looking for what is out of date, which is what it did as
`redraw!`. Four of the six things it used to do have moved to the edge they
belong to: the timeline row now follows `timeline.ntr`, the panel follows the
selection (which is an id, so an insert no longer re-points it), a clip's plot is
built where the clip joins the sequence, and a card where the effect joins the
clip. What is left is the two consequences of a shape change that nothing else
can derive: where the clips sit on the timeline, and which picture that makes.
"""
function redraw!(player::Player)
    ensurestreams!(player)
    # an edit can change what the Project panel says — its derived name comes from
    # `clips[1]`. It no longer listens to the playhead, so it is told here instead.
    if player.dockopen[] === :project
        r = get(player.fxwidgets, :projectrefresh, nothing)
        r === nothing || r()
    end
    showplayhead!(player)
    player.playing[] || retrypresent(player)
    return nothing
end

"""
Open a GPU stream for every source the sequence shows and that has none yet —
an edit can bring in a source that was not there when the player started (a
clip dragged in from the bin), and its layers would otherwise fall back to CPU
decode forever. Cheap: one `haskey` per source, the open runs off the UI thread.

A scan, and it stays one: [`readerkey`](@ref) is defined over the whole set (two
clips of one source that overlap and are far apart need a reader each), so which
clip needs a stream is not a fact about that clip. MOVING a clip changes another
clip's answer, and no per-clip door would fire. This is a resource pool keyed over
the document, not a view asking whether it is still correct.
"""
function ensurestreams!(player::Player)
    player.gpupreview isa GPUPreview || return nothing
    # per reader, not per source: a clone overlapping its original further apart
    # than the ring needs its own stream or the two fight over one read head
    for clip in player.sequence.clips
        haskey(player.gpucache, readerkey(player.sequence, clip)) ||
            Threads.@spawn preloadgpu!(player, clip)
    end
    return nothing
end

"""
A canvas point in the clip's matte space: normalized within its cropped layer.

`previewtosource`'s sibling, minus the stabilization inverse — the matte is
computed on the rendered (already stabilized, already cropped) frame, so the
click needs the crop fit undone and nothing else.
"""
function previewtomatte(player::Player, clip::Clip, srcframe::Integer, p)
    layer  = (clip.source.width, clip.source.height)
    q = previewtolayer(player, clip, layer, p, srcframe)
    cw, ch = mattelayersize(clip)
    return (clamp((q[1] - clip.crop[1] * layer[1]) / cw, 0.0, 1.0),
            clamp((q[2] - clip.crop[2] * layer[2]) / ch, 0.0, 1.0))
end

"The inverse of `previewtomatte`: a normalized matte point in canvas pixels."
function mattetopreview(player::Player, clip::Clip, srcframe::Integer, n)
    layer  = (clip.source.width, clip.source.height)
    cw, ch = mattelayersize(clip)
    sx = n[1] * cw + clip.crop[1] * layer[1]
    sy = n[2] * ch + clip.crop[2] * layer[2]
    return layertopreview(player, clip, layer, (sx, sy), srcframe)
end

"""
A preview data coordinate, in the clip's layer pixels.

The preview axis plots `player.frame[]` and indexes it in its own pixels, so what
a data coordinate means depends on what that buffer holds. On the single-clip path
it holds the rendered layer at source resolution and the axis limits do the
cropping, so a data coordinate is already a layer pixel and putting it through
`layermatrix` crops it a second time. (That is what sent a click on the upper
bird into the middle of the nest box: the error is zero at the centre of the
frame and grows toward its edges, so the bug hid until somebody clicked near an
edge.) On the composite path the buffer is the sequence canvas, and then the
matrix is exactly right.
"""
function previewtolayer(player::Player, clip::Clip, layer::Tuple{Int, Int}, p,
                       frame::Real = 0)
    canvas = size(player.frame[])
    canvas == layer && return (Float64(p[1]), Float64(p[2]))
    q = layermatrix(clip, layer, canvas, frame) * Vec3f(p[1], p[2], 1)
    return (Float64(q[1]), Float64(q[2]))
end

"The inverse of [`previewtolayer`](@ref)."
function layertopreview(player::Player, clip::Clip, layer::Tuple{Int, Int}, q,
                        frame::Real = 0)
    canvas = size(player.frame[])
    canvas == layer && return (Float64(q[1]), Float64(q[2]))
    r = inv(layermatrix(clip, layer, canvas, frame)) * Vec3f(q[1], q[2], 1)
    return (Float64(r[1]), Float64(r[2]))
end

"""
    previewtosource(player, clip, srcframe, p) -> (x, y)

Where a click in the preview lands, in the clip's source pixels.

Two hops, both *sampling* matrices, so this is a forward multiply rather than an
inversion. `layermatrix` maps a canvas pixel to the layer pixel drawn there (the
clip's crop fitted whole, plus its reframe, letterbox bars included), and the
stabilization transform maps a displayed pixel to the source pixel it was sampled
from. Composing them is the mapping.

This replaces `scale = clip.source.width / size(player.frame[], 1)`, which holds
only for an uncropped, unreframed, unstabilized clip on a full-width canvas.
Anywhere else the click landed off target, and on a stabilized clip by a different
amount on every frame.
"""
function previewtosource(player::Player, clip::Clip, srcframe::Integer, p)
    canvas = size(player.frame[])
    # Source pixels, not the layer's: `fitmatrix` depends on the input size only
    # through the crop rect's aspect ratio, and a proxy scales both extents
    # together. Feeding the original size lands directly in source pixels and makes
    # the mapping proxy-independent, which matters because a proxy swap replaces
    # the decode pool and not `clip.source`.
    layer = (clip.source.width, clip.source.height)
    q = layermatrix(clip, layer, canvas, srcframe) * Vec3f(p[1], p[2], 1)  # canvas -> source
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
    r = inv(layermatrix(clip, layer, canvas, srcframe)) * q
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
    # activating a pick, dropping a crop anchor and scrubbing at once.
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
    # Alt+drag on the preview paints the matte: left adds to the subject, right
    # takes away. A modifier rather than a tool of its own, available whenever the
    # clip has a matte, so a bad frame can be fixed without leaving what you are
    # doing for a mode switch.
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
        # Show (or hide) the brush footprint. Diameter in screen pixels, from the
        # radius' fraction of the matte width and the preview's width, so the
        # circle is the size the stroke will be at any zoom.
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

Steps are the caller's, and geometric by convention: at 1% a fixed step is half
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

The shape the next crop is locked to — width over height of the output — or
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

The crop is in fractions of the source, and a fraction is not a pixel: the
output ratio is `w·source.width / h·source.height`, so the fraction ratio that
lands on `a` is `a · height/width`. Getting this wrong gives a 16:9 lock that is
16:9 only on a square source.

The dragged width is kept and the height derived. One of the two has to give, and
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
    # Not clamped to the picture: a crop rectangle may sit partly or wholly
    # outside it, which is how a crop tool grows a canvas. Clamped to `0..1` the
    # tool could only ever shrink a project.
    #
    # `canvassize` multiplies these by the source's pixels, so a width past 1.0 is
    # a wider canvas. What falls outside the source renders as the letterbox
    # background: the canvas pass clears it and a layer only writes where its warp
    # reaches.
    x0, x1 = x0 / W, x1 / W
    y0, y1 = y0 / H, y1 / H
    # the floor is on the size, not on where the rect sits: a 5%-wide crop
    # hanging off the left edge is a valid request
    (x1 - x0 < 0.01 || y1 - y0 < 0.01) && return nothing
    a = cropaspect(player)[]
    a === nothing || ((x0, y0, w, h) = lockaspect((x0, y0, x1 - x0, y1 - y0), clip.source, a);
                      x1 = x0 + w; y1 = y0 + h)
    before = canvassize(player.sequence)
    clip.crop = (x0, y0, x1 - x0, y1 - y0)
    # In `:canvas` scope the crop sets the project size, recorded on the sequence
    # so it survives deleting or reordering clips. In `:clip` scope the project
    # keeps its size and only this clip's framing moves — letterboxing, or
    # re-framing one shot inside a finished timeline.
    wholeproject = cropscope(player)[] === :canvas
    if wholeproject
        player.sequence.canvas =
            (max(2 * (round(Int, (x1 - x0) * clip.source.width) ÷ 2), 2),
             max(2 * (round(Int, (y1 - y0) * clip.source.height) ÷ 2), 2))
    elseif player.sequence.canvas === nothing
        # Pin the current size. Without an explicit canvas, `canvassize` derives
        # one from the first clip's crop, so cropping that clip in `:clip` scope
        # would still resize the project. The number does not change; it just
        # stops being a function of the clip being edited.
        player.sequence.canvas = before
    end
    applycrop!(player, clip)
    # Report the resulting canvas, and that it grew. Added empty area looks like
    # the letterbox bars a differently-shaped clip already gets, so the status
    # line is what distinguishes a resized project from a letterboxed clip.
    after = canvassize(player.sequence)
    grew = after[1] > before[1] || after[2] > before[2]
    setstatus!(player, !wholeproject ?
        "cropped this clip — the project stays $(after[1])×$(after[2])" :
        grew ?
        "canvas $(before[1])×$(before[2]) → $(after[1])×$(after[2]) — cropped outward, the new area is empty" :
        "canvas $(after[1])×$(after[2])")
    rebuildtoolcard!(player, :crop)
    return nothing
end

"""
    resetcanvas!(player) -> nothing

Drop the explicit canvas and go back to deriving it from the first clip.

Without it the only way out of a project size is Ctrl+Z, which also takes back
the crop that set it, and nothing at all once a later edit is on the stack.
"""
function resetcanvas!(player::Player)
    player.sequence.canvas === nothing && return setstatus!(player, "canvas: already derived from the first clip")
    snapshot!(player)
    player.sequence.canvas = nothing
    applycrop!(player)
    redraw!(player)
    rebuildtoolcard!(player, :crop)
    sz = canvassize(player.sequence)
    setstatus!(player, "canvas back to $(sz[1])×$(sz[2]), derived from the first clip")
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
        elseif (event.key == Keyboard.left_bracket ||
                event.key == Keyboard.right_bracket) && ispress
            # [ and ] resize the matte brush, as every paint tool does. Geometric
            # steps, not linear: at 1% a fixed step is half the brush and at 20%
            # it is nothing, so the same key has to mean the same PROPORTION.
            f = event.key == Keyboard.right_bracket ? 1.25 : 0.8
            setbrushradius!(player, player.brushradius * f)
        elseif event.key == Keyboard.c && ispress &&
               ispressed(player.fig, Keyboard.left_control | Keyboard.right_control)
            copyclips!(player)          # before bare `c`, which is the crop tool
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
            # While a selection is being marked, Ctrl+Z takes the last point back.
            # Falling through to the project undo would edit the timeline behind a
            # preview the user is watching for the matte.
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
            # Shift+Enter writes this frame only. Same decision, different scope:
            # propagate when the tracking drifted, repair when one frame in an
            # otherwise good matte came out broken.
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
Register (or fetch) the dock panel `Subfigure` for `key`. The dock is the fixed
slot between the toolbar and the preview that the effects, media and export
panels share, so a panel never opens over the video. Built hidden; show it with
[`opendock!`](@ref).
"""
function dockpanel!(player::Player, key::Symbol; width::Real = 300)
    haskey(player.dockpanels, key) && return player.dockpanels[key].sf
    # A visible scrollbar and a usable wheel step. Makie's defaults are a fully
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
    if key === :effects
        # Its cards belong to the clips, so nothing went stale while it was
        # hidden — but the clip under the playhead may have changed, and which
        # cards are shown follows that.
        showclip!(player)
    elseif key === :project
        # same rule as the effects panel: a panel that does not update while it is
        # hidden has to catch up when it comes back
        r = get(player.fxwidgets, :projectrefresh, nothing)
        r === nothing || r()
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
    screen = player.screen
    screen === nothing && return nothing
    GLFW = GLMakie.GLFW
    win = screen.glscreen
    win isa GLFW.Window || return nothing   # a headless screen has no cursor to set
    cur = get!(CURSORS, shape) do
        shape === :scissor  ? GLFW.CreateCursor(scissor_bitmap(), (12, 3)) :
        shape === :hresize  ? GLFW.CreateStandardCursor(GLFW.RESIZE_EW_CURSOR) :
        shape === :resize   ? GLFW.CreateStandardCursor(GLFW.RESIZE_NWSE_CURSOR) :
        shape === :move     ? GLFW.CreateStandardCursor(GLFW.RESIZE_ALL_CURSOR) :
        shape === :crosshair ? GLFW.CreateStandardCursor(GLFW.CROSSHAIR_CURSOR) :
        shape === :hand     ? GLFW.CreateStandardCursor(GLFW.POINTING_HAND_CURSOR) :
                              GLFW.CreateStandardCursor(GLFW.ARROW_CURSOR)
    end
    GLFW.SetCursor(win, cur)
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
Media-bin dock panel: each imported source is a card: an aspect-correct
first-frame thumbnail beside its left-aligned name and duration. Drag a card
onto a timeline lane (a chip follows the cursor, the target lane ghosts) or
onto the "+ new track" strip. The head of the panel is the drop zone: files
dropped anywhere on the window import here (see [`importsources!`](@ref)), and
clicking it opens the native file dialog.
"""
function buildmediabin!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1], "Media"; font = :bold, halign = :left, tellwidth = false)
    # One import target instead of a button: files dropped from the file manager
    # land here (any number at once), and a click opens the browse dialog.
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
    # the browse dialog behind a hook: it is a blocking native window, so tests
    # (the random event storm clicks everywhere) replace it with a no-op
    player.fxwidgets[:browse] = () -> Threads.@spawn try
        path = Makie.choose_file_dialogue()
        path === nothing || importsources!(player, [String(path)])
    catch e
        setstatus!(player, "import failed: $(sprint(showerror, e))")
    end
    # files dropped anywhere on the window go to the bin; the dock opens itself so
    # the new rows are visible
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
            # thumbnail box sized to the source aspect (no letterbox padding),
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
            catch e   # a task failure is silent otherwise; the bin row stays blank
                setstatus!(player, "no thumbnail for $(basename(src.path)): $(briefly(e))")
            end
        end
        if length(rows.content) > 1
            rowgap!(rows, 8)
            colgap!(rows, 12)   # text hugs the (aspect-sized) thumbnail column
        end
        return
    end
    # drag preview on the timeline: a translucent accent band showing where the
    # dropped clip would land. The cursor height picks the target track, and
    # hovering above the top lane shows the "+ new track" hint.
    dropghost = Observable(Rect2f(0, 0, 0, 0))
    dropvis = Observable(false)
    droptrack = Ref(1)
    dgp = poly!(player.timeline.axis, dropghost; color = (uicolors.accent, 0.3),
                strokecolor = uicolors.accent, strokewidth = 2, visible = dropvis)
    translate!(dgp, 0, 0, 20)
    # …and a chip following the cursor from the moment the drag starts at the bin
    # thumbnail, so the drag is visible outside the timeline too
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
            # the ghost shows what will land: a conformed clip keeps its wall-clock
            # duration but occupies a different number of timeline frames.
            # Measuring it in sequence frames drew a 9.3 s clip as 4.7 s.
            nfr = floor(Int, src.nframes / conformrate(src, seq.framerate))
            # target the hovered lane; if that spot is taken, the ghost snaps to
            # the first free lane above (stacking), never silently to the end
            want = trackat(seq, pos[2], ntr)
            track = want == 0 ? 0 :        # 0 = the zone below the bottom lane
                    freetrack(seq, round(Int, t * seq.framerate), nfr, want)
            droptrack[] = track
            dur = nfr / seq.framerate
            n2 = max(ntr, track)
            g = min(0.02, trackspan(n2) * 0.15)
            lo, hi = track == 0 ? (0.008, TRACKBASE - 0.013) : trackband(seq, track, n2)
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
    # press on a row (its name or its thumbnail, since the picture is what gets
    # grabbed) starts
    # the drag: hand cursor + the new-track zone lights up; release over the
    # timeline places the clip on the ghosted lane
    on(events(player.fig).mousebutton; priority = 90) do event
        event.button == Mouse.left || return Consume(false)
        mp = Point2f(events(player.fig).mouseposition[])
        # …but only when nothing is drawn over the bin (a modal, a dropdown);
        # the release is checked against the timeline instead, so it stays out
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
                    # the chip carries the retime, so the conform is visible before
                    # the drop rather than after it
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
            catch e   # the status line names the file; without the reason it says nothing
                push!(failed, basename(path) * " ($(briefly(e)))")
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

A source at a different framerate is conformed to the sequence rather than
refused (see [`Clip`](@ref)'s `rate`): its duration is preserved and frames are
held or dropped to fit. Refusing it was a dead end that only announced itself in
the status line — the bin had accepted the file, the drag ghost had shown a
valid lane, and then nothing landed.
"""
function placesource!(player::Player, source::VideoSource, at::Integer; track::Integer = 1)
    seq = player.sequence
    rate = conformrate(source, seq.framerate)
    start = max(Int(at), 0)
    len = floor(Int, source.nframes / rate)   # timeline frames this clip will occupy
    # if the wanted lane is occupied there, stack on the first free lane above —
    # the drop always lands where the drag ghost showed it
    # track 0 = the zone below the bottom lane: make room underneath and land there
    if Int(track) == 0
        # no compaction here: a clip is being added, and compacting before it
        # exists would undo the push it needs
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
    addclip!(seq, clip)
    sort!(seq.clips, by = c -> (c.track, c.start))
    limits!(player.timeline.axis, 0.0, seqduration(seq), 0.0, AXISTOP)  # reveal it (see `AXISTOP`)
    redraw!(player)
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
                      splitext(sourcepath(clips[1].source))[1] * "_export.mp4")
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
    # All five: without handles for the GIF fps and loop controls, a walkthrough
    # can drive the format menu and the Export button but not the settings
    # between them.
    merge!(player.fxwidgets, Dict{Symbol, Any}(:exportgo => gobtn, :exportpath => path,
                                               :exportformat => fmtmenu,
                                               :giffps => fpsslider, :gifloop => loopbox))
    return panel
end

# ------------------------------------------ the keyframe editor (on the timeline)
#
# The curves themselves are not here: each is a `lanecurve!` plot owned by the
# parameter it draws (see [`ParamView`](@ref) and lane.jl), so an edit that
# notifies `p.curve` redraws it. What is here is the mouse — grabbing a ◆,
# dragging a handle, Alt-clicking a lane to add a key — and that needs to find
# the parameter under the cursor, which is a walk over what is on screen.

"""
    shownlanes(player) -> Vector{Tuple{Clip, Effect, Param}}

Every parameter whose lane is on the timeline right now.

The clip whose cards are up owns them, so this is a walk over what is being shown
rather than a registry that has to be added to when a lane opens and taken from
when it closes — and that could therefore disagree with the screen.
"""
function shownlanes(player::Player)
    out = Tuple{Clip, Effect, Param}[]
    clip = player.shownclip
    clip === nothing && return out
    for fx in clip.effects, p in fx.params
        p.view === nothing && continue
        p.visible[] || continue
        push!(out, (clip, fx, p))
    end
    return out
end

"Axis units per pixel at the current zoom — what turns a grab radius into a distance."
function axisperpixel(player::Player)
    vp = player.timeline.axis.scene.viewport[]
    (x0, x1) = player.timeline.viewrange[]
    return ((x1 - x0) / max(vp.widths[1], 1), 1.0 / max(vp.widths[2], 1))
end

"""
    nearestmarker(player, t, y) -> (clip, fx, param, key index) | nothing

The ◆ near `(t, y)`.

Walks the lanes and reads each one's own `markpoints`/`markkeys`, so a gesture
edits the object it hit: the index is into that curve, not into a shared list
that would have to be kept in step with what is drawn.
"""
function nearestmarker(player::Player, t, y)
    sx, sy = axisperpixel(player)
    best = nothing; bestd = 14.0
    for (clip, fx, p) in shownlanes(player)
        lane = p.view.lane
        idx = lane.markkeys[]
        for (i, pt) in enumerate(lane.markpoints[])
            d = hypot((t - pt[1]) / sx, (y - pt[2]) / sy)
            d < bestd && ((best, bestd) = ((clip, fx, p, idx[i]), d))
        end
    end
    return best
end

"""
    nearesthandle(player, t, y) -> (param, key index, :in | :out) | nothing

The Bézier grip near `(t, y)`.

Only the selected anchor has grips, so only its lane is asked. They win over ◆:
they are drawn on top, they are the smaller target, and a short handle puts one
on top of its own anchor.
"""
function nearesthandle(player::Player, t, y)
    sel = player.selectedkey[]
    sel === nothing && return nothing
    p, kidx = sel
    v = p.view
    v === nothing && return nothing
    tips = v.lane.handletips[]
    isempty(tips) && return nothing
    sides = v.lane.handlesides[]
    sx, sy = axisperpixel(player)
    best = 0; bestd = 12.0
    for (i, pt) in enumerate(tips)
        d = hypot((t - pt[1]) / sx, (y - pt[2]) / sy)
        d < bestd && ((best, bestd) = (i, d))
    end
    return best == 0 ? nothing : (p, kidx, sides[best])
end

"The shown parameter whose lane passes within `maxpx` of `(sf, y)`."
function nearestlane(player::Player, sf::Integer, y; maxpx::Real = 22.0)
    _, sy = axisperpixel(player)
    best = nothing; bestd = maxpx * sy
    for (clip, fx, p) in shownlanes(player)
        lane = p.view.lane
        d = abs(y - laney(lane.band[], lane.valuerange[], valueat(p, sf)))
        d < bestd && ((best, bestd) = (p, d))
    end
    return best
end

"""
    selectkey!(player, p, i) -> nothing

Select keyframe `i` of `p` — which is what puts its Bézier handles on screen.

One truth for the whole timeline: every lane derives from `player.selectedkey`
whether it is the one showing handles, so selecting an anchor deselects the
previous one without anything being told to redraw.
"""
selectkey!(player::Player, p::Param, i::Integer) =
    (player.selectedkey[] = (p, Int(i)); nothing)
clearselectedkey!(player::Player) = (player.selectedkey[] = nothing; nothing)

"""
    buildkeyframeeditor!(player) -> nothing

Wire the timeline's keyframe gestures: drag a ◆ (snapping to the playhead, with a
live readout), Alt-click a lane to add one, Ctrl-click to delete, right-click for
the keyframe menu, and drag a grip to shape the curve.

[`nearestmarker`](@ref) answers with the lane's own `(clip, effect, parameter, key
index)`, so the gesture edits the object it hit — there is no key to resolve and
no "currently focused parameter" to guess at.
"""
function buildkeyframeeditor!(player::Player)
    ax = player.timeline.axis
    tl = player.timeline
    seq = player.sequence
    fps = seq.framerate

    dragref = Ref{Any}(nothing)                 # (clip, param, key index)
    handledrag = Ref{Any}(nothing)              # (param, key index, :in | :out)
    dragtippos = Observable(Point2f(0, 0))
    dragtiptext = Observable("")
    dragtip = text!(ax, dragtippos; text = dragtiptext, visible = false,
                    fontsize = 12, font = :bold, color = :white,
                    strokecolor = (:black, 0.8), strokewidth = 2,
                    offset = (10, 10), align = (:left, :bottom))
    translate!(dragtip, 0, 0, 7)

    kfmenu = Modal(player.fig; title = "Keyframe", min_size = (200, 10),
                   backdrop_color = (:black, 0.15))
    kfmenuctx = Ref{Any}(nothing)               # (clip, param, key index)
    easelabel = Observable("Ease in & out")
    holdlabel = Observable("Hold until the next key")
    function retoggle(mode)
        clip, p, kidx = kfmenuctx[]
        snapshot!(player)
        materializeease!(p)
        cur = p.curve[].keys[kidx].ease
        new = cur === mode ? :linear : mode
        setease!(p, kidx, new)
        setstatus!(player, "$(p.label): keyframe is now " *
                           (new === :smooth ? "eased (in & out)" :
                            new === :hold ? "held until the next key" : "linear"))
        editedcurve!(player, clip)
    end
    for (row, (lbl, action)) in enumerate([
        ("Delete keyframe", () -> (t = kfmenuctx[]; deletekey!(player, t[1], t[2], t[3]))),
        (easelabel, () -> retoggle(:smooth)),
        (holdlabel, () -> retoggle(:hold)),
        ("Simplify to Bézier anchors", () -> begin
            clip, p, _ = kfmenuctx[]
            snapshot!(player)
            before = length(p.curve[].keys)
            materializeease!(p)
            simplify!(p)
            after = length(p.curve[].keys)
            clearselectedkey!(player)
            setstatus!(player, "$(p.label): $before keyframe$(before == 1 ? "" : "s") → " *
                               "$after anchor$(after == 1 ? "" : "s") with handles" *
                               (after < before ? " (Ctrl+Z to restore)" : " — nothing to remove"))
            editedcurve!(player, clip)
        end),
        ("Clear all keys of this parameter", () -> begin
            clip, p, _ = kfmenuctx[]
            snapshot!(player)
            n = length(p.curve[].keys)
            clearkeys!(p, playheadframe(player, clip))
            clearselectedkey!(player)
            setstatus!(player, "$(p.label): cleared $n keyframe$(n == 1 ? "" : "s") (Ctrl+Z to restore)")
            editedcurve!(player, clip)
        end)])
        btn = Button(kfmenu[row, 1]; label = lbl, tellwidth = false)
        on(btn.clicks) do _
            close!(kfmenu)
            kfmenuctx[] === nothing || action()
        end
        player.fxwidgets[Symbol(:kfmenubtn, row)] = btn
    end
    player.fxwidgets[:kfmenu] = kfmenu

    on(events(ax.scene).mousebutton; priority = 20) do event
        isempty(shownlanes(player)) && return Consume(false)   # no lane = inert
        is_mouseinside(ax.scene) || return Consume(false)
        t, y = mouseposition(ax.scene)
        # The strip is the playhead's, whatever is drawn below it: `nearestmarker`
        # works in SCREEN distance, and with a lane soloed its top edge sits ~18 px
        # under the strip — so a ◆ near a lane's ceiling swallowed the scrub press
        # at its own x and nowhere else, which reads as "the strip scrubs, except
        # sometimes". See `SCRUBBAND` and the same rule in `timeline.jl`.
        inscrubband(y) && return Consume(false)
        if event.button == Mouse.left && event.action == Mouse.press
            # The lane divider stays the timeline's. A dense curve puts anchors at
            # the lane's bottom edge, inside the resize grip's grab zone, and this
            # handler runs first: the grip worked on the sparse lego project (16
            # anchors) and not at all on the per-frame one.
            trackedgeat(seq, y, ntracks(seq); grab = grabzone(tl)) === nothing ||
                return Consume(false)
            # Handle grips are hit before the Alt branch below: Alt-dragging one is
            # how a handle pair gets broken, so inside that branch Alt over a grip
            # falls through to "add a keyframe here".
            grip = nearesthandle(player, t, y)
            if grip !== nothing
                snapshot!(player)
                handledrag[] = grip
                return Consume(true)
            end
            if ispressed(ax.scene, Keyboard.left_alt | Keyboard.right_alt)
                # Alt on an anchor converts it between smooth and corner, as a pen
                # tool does; only Alt on an empty lane adds a key. The reverse
                # order added a key on top of the ◆ being aimed at.
                hit = nearestmarker(player, t, y)
                if hit !== nothing
                    clip, _, p, kidx = hit
                    snapshot!(player)
                    materializeease!(p)
                    p.curve[].keys[kidx].ease === :bezier ?
                        cornerkey!(p, kidx) : smoothkey!(p, kidx)
                    selectkey!(player, p, kidx)
                    setstatus!(player, "$(p.label): anchor is now " *
                                       (p.curve[].keys[kidx].ease === :bezier ?
                                        "smooth — its handles stay in line" :
                                        "a corner — its handles move on their own"))
                    editedcurve!(player, clip)
                    return Consume(true)
                end
                # the clip whose lanes are drawn, without a search: the lanes belong
                # to the clip whose cards are up, so it owns every one on screen
                clip = player.shownclip
                clip === nothing && return Consume(false)
                n = timelineframe(tl, t)
                clip.start <= n < clipend(clip) || return Consume(false)
                sf = sourceframe(clip, n)
                p = nearestlane(player, sf, y)
                if p === nothing
                    setstatus!(player,
                        any(fx -> any(q -> q.visible[], fx.params), clip.effects) ?
                        "Alt-click ON a lane to add a keyframe to it" :
                        "no lane shown — open one with its ◆ in the Effects panel first")
                    return Consume(true)
                end
                snapshot!(player)
                lane = p.view.lane
                setkey!(p, sf, lanevalue(lane.band[], lane.valuerange[], y))
                editedcurve!(player, clip)
                return Consume(true)
            end
            hit = nearestmarker(player, t, y)
            hit === nothing && return Consume(false)
            clip, fx, p, kidx = hit
            if ispressed(ax.scene, Keyboard.left_control | Keyboard.right_control)
                deletekey!(player, clip, p, kidx)
                return Consume(true)
            end
            snapshot!(player)
            # touching an anchor selects it, which puts its handles on screen
            selectkey!(player, p, kidx)
            dragref[] = (clip, p, kidx)
            return Consume(true)
        elseif event.button == Mouse.left && event.action == Mouse.release &&
               (dragref[] !== nothing || handledrag[] !== nothing)
            dragref[] = nothing
            handledrag[] = nothing
            dragtip.visible = false
            return Consume(true)
        elseif event.button == Mouse.right && event.action == Mouse.press
            hit = nearestmarker(player, t, y)
            hit === nothing && return Consume(false)
            clip, _, p, kidx = hit
            kfmenuctx[] = (clip, p, kidx)
            k = p.curve[].keys[kidx]
            kfmenu.title = "◆ $(p.label) · $(timestring(timelineframe(clip, k.frame) / fps))"
            easelabel[] = keyease(p.curve[], k) === :smooth ?
                          "Make linear (corner)" : "Ease in & out"
            holdlabel[] = k.ease === :hold ? "Interpolate again" : "Hold until the next key"
            mp = events(player.fig).mouseposition[]
            vp = player.fig.scene.viewport[]
            kfmenu.halign = clamp(mp[1] / max(vp.widths[1], 1), 0.0, 1.0)
            kfmenu.valign = clamp(mp[2] / max(vp.widths[2], 1), 0.0, 1.0)
            open!(kfmenu)
            return Consume(true)
        end
        return Consume(false)
    end

    on(events(ax.scene).mouseposition; priority = 20) do _
        if handledrag[] !== nothing
            p, kidx, side = handledrag[]
            t, y = mouseposition(ax.scene)
            lane = p.view.lane
            fm = lane.framemap[]
            k = p.curve[].keys[kidx]
            scale = perframe(fm, k.frame)
            df = (t * fps - timelineframe(fm, k.frame)) / (scale == 0 ? 1 : scale)
            # a grip stays on its own side of the anchor; dragging one through the
            # anchor would turn the segment inside out, and Photoshop pins it too
            df = side === :out ? max(df, 0.0) : min(df, 0.0)
            dv = lanevalue(lane.band[], lane.valuerange[], y) - Float64(k.value)
            alt = ispressed(ax.scene, Keyboard.left_alt | Keyboard.right_alt)
            sethandle!(p, kidx, side, Handle(df, dv); couple = !alt)
            dragtippos[] = Point2f(t, y)
            dragtiptext[] = "$(p.label)  handle " * (alt ? "(broken)" : "(smooth)")
            dragtip.visible = true
            editedcurve!(player, player.shownclip)
            return Consume(true)
        end
        dragref[] === nothing && return Consume(false)
        clip, p, i = dragref[]
        t, y = mouseposition(ax.scene)
        lane = p.view.lane
        f = clamp(sourceframe(clip, timelineframe(tl, t)), clip.src_in, clip.src_out)
        # snap to the playhead when close, in raw pixels, since one frame can
        # already be wider than the threshold
        sx, _ = axisperpixel(player)
        phf = clamp(playheadframe(player, clip), clip.src_in, clip.src_out)
        abs(t - timelineframe(clip, phf) / fps) / sx < 12 && (f = phf)
        v = lanevalue(lane.band[], lane.valuerange[], y)
        movekey!(p, i, f, v)
        j = findfirst(k -> k.frame == f, p.curve[].keys)
        # `movekey!` re-sorts, so the index can move under us — the selection has to
        # follow it or the handles would be drawn for whichever key took its place
        if j !== nothing
            dragref[] = (clip, p, j)
            selectkey!(player, p, j)
        end
        dragtippos[] = Point2f(timelineframe(clip, f) / fps,
                               laney(lane.band[], lane.valuerange[], v))
        dragtiptext[] = "$(p.label)  $(round(Float64(v); digits = 2)) · " *
                        timestring(timelineframe(clip, f) / fps)
        dragtip.visible = true
        editedcurve!(player, clip)
        return Consume(true)
    end
    return nothing
end

"""
    editedcurve!(player, clip) -> nothing

A curve on `clip` was edited: the bake no longer describes what it renders, and
the preview draws this frame again.

Not "the curve changed" — the curve said that itself, through the Observable
every lane, ◆ and slider showing it derives from. This is the rest of what an
edit means, and it is the same two lines whichever gesture made it.
"""
function editedcurve!(player::Player, clip)
    clip isa Clip && bakedirty!(clip)
    showplayhead!(player)
    return nothing
end

"""
    deletekey!(player, clip, p, kidx) -> nothing

Take keyframe `kidx` off `p`, undoably. The last one stays: a parameter always has
a value, and a curve of one key is exactly how a constant is written — so removing
the second-to-last is where animation stops.
"""
function deletekey!(player::Player, clip, p::Param, kidx::Integer)
    snapshot!(player)
    if !removekeyat!(p, kidx)
        setstatus!(player, "$(p.label): its last keyframe is its value — " *
                           "the ◆ in the panel switches keyframing off")
        return nothing
    end
    isanimated(p) ||
        setstatus!(player, "$(p.label): last keyframe removed — back to a static value")
    clearselectedkey!(player)
    editedcurve!(player, clip)
    return nothing
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
    # Small and contextual: only what has no first-class home elsewhere —
    # splitting/cropping live on the toolbar (✂ ▢), effects and stabilization
    # in the Inspector, tools in the Tools dock
    modal = Modal(player.fig; title = "Clip actions", min_size = (220, 10),
                  backdrop_color = (:black, 0.15))   # popup, not a dialog
    player.clipmodal = modal
    rcframe() = clamp(round(Int, player.rctime * player.sequence.framerate), 0,
                      max(seqlength(player.sequence) - 1, 0))
    actions = [
        ("Join with next clip", "", () -> joinat!(player; at = rcframe())),
        # the same call the X key makes. Deleting straight from the menu took no
        # undo snapshot, so Ctrl+Z reached past the delete to the edit before it
        # (the split), and redo re-did that instead.
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
    return nothing
end

"""
    openclipmenu!(player, t) -> nothing

Pop the clip context menu up at the cursor, for the clip at timeline second `t`.

A no-op before [`wireclipmenu!`](@ref) has built the modal — the timeline is alive
while the editor is still being assembled.
"""
function openclipmenu!(player::Player, t::Real)
    modal = player.clipmodal
    modal === nothing && return nothing
    player.rctime = t
    modal.title = "Clip @ " * timestring(t)
    # pop up at the cursor (fractional align, clamped into the window)
    mp = events(player.fig).mouseposition[]
    vp = player.fig.scene.viewport[]
    modal.halign = clamp(mp[1] / max(vp.widths[1], 1), 0.0, 1.0)
    modal.valign = clamp(mp[2] / max(vp.widths[2], 1), 0.0, 1.0)
    open!(modal)
    return nothing
end

"Whether `clip` has any animated parameter at all."
clipanimated(clip::Clip) = isanimated(clip)

"Absolute source frame the playhead currently maps to within `clip`."
playheadframe(player::Player, clip::Clip) = sourceframe(clip, player.playhead[])

"""
    editparam!(player, clip, p, value; frame = playheadframe(player, clip)) -> nothing

Set `p` to `value` at `frame`: the one edit a control makes.

Undo, the value, the bake and the picture, in that order, and nothing else — the
slider, the ◆ and the lane are derived from the curve [`setvalue!`](@ref)
notifies, so none of them is written here.
"""
function editparam!(player::Player, clip, p::Param, value;
                    frame::Integer = playheadframe(player, clip))
    # One undo entry per gesture: a slider drag fires per pixel, and a step per
    # pixel makes Ctrl+Z useless. `lastslidersnap` is when the last one was taken.
    if time() - player.lastslidersnap > 1.5
        snapshot!(player); player.lastslidersnap = time()
    end
    setvalue!(p, value, frame)
    editedcurve!(player, clip)
    return nothing
end

"""
    bindinput!(player, p, source; op = :copy) -> Bool

Drive parameter `p` from `source` — `(clip, effect, name)` — instead of from its
own curve. See [`ParamInput`](@ref).

`p` keeps its curve: unbinding puts it back exactly as it was, so trying an edge
out costs nothing.
"""
function bindinput!(player::Player, p::Param, clip::Clip, fx::Effect, name::Symbol;
                    op::Symbol = :copy)
    snapshot!(player)
    p.input = ParamInput(op, ParamRef(name; clip = clip.id, effect = fx.id))
    bindinputs!(player.sequence)
    if !isdriven(p)
        p.input = nothing
        setstatus!(player, "$(p.label): nothing to drive it from there")
        return false
    end
    setstatus!(player, "$(p.label): driven by $name — ⇥ cuts it")
    rebuildcard!(player, clip, fx)   # a driven parameter has no slider
    editedcurve!(player, clip)
    return true
end

"""
    unbindinput!(player, clip, fx, p) -> Bool

Cut `p`'s edge. It keeps the value it was showing, so the picture does not jump
when the link goes, and its card is rebuilt because a parameter that sets its own
value has a slider again.
"""
function unbindinput!(player::Player, clip::Clip, fx::Effect, p::Param)
    p.input === nothing && return false
    snapshot!(player)
    sf = playheadframe(player, clip)
    v = valueat(p, sf)
    p.input = nothing
    setvalue!(p, v, sf)
    setstatus!(player, "$(p.label): no longer driven — it keeps the value it had")
    rebuildcard!(player, clip, fx)
    editedcurve!(player, clip)
    return true
end

"""
The ◆ of the panel's trio, Premiere semantics: start animating the parameter if it
is not animated yet (a second key at the playhead, so the constant it was becomes
a curve); otherwise add a key at the playhead, or remove the one sitting there.
Removing the second-to-last makes it a constant again, at the value it was
showing.

Takes the `Param` itself. There is nothing to look up: no key, no registry, no
"the first effect of this kind". The row that drew the button holds the object the
button edits.
"""
function togglekey!(player::Player, target::Clip, p::Param)
    sf = playheadframe(player, target)
    snapshot!(player)
    if !isanimated(p)
        v = valueat(p, sf)
        # A constant is one key, so animating it needs a second: the value it held
        # is anchored at the clip's first frame and the playhead gets the key the
        # user asked for. Standing ON the first frame there is nothing to anchor
        # before it, so the anchor goes to the frame after — the same statement,
        # read the other way round. The lane appears with `visible`, which is what
        # its plot is drawn with.
        movekey!(p, 1, sf == target.src_in ? sf + 1 : target.src_in, v)
        setkey!(p, sf, v)
        p.visible[] = true
        setstatus!(player, "$(p.label): keyframing on — scrub and move the slider to add keys")
    elseif haskeyat(p, sf)
        removekey!(p, sf)
        setstatus!(player, isanimated(p) ? "$(p.label): keyframe removed" :
                   "$(p.label): last keyframe removed — back to a static value")
    else
        setkey!(p, sf, valueat(p, sf))
        setstatus!(player, "$(p.label): keyframe added at the playhead")
    end
    editedcurve!(player, target)
    return nothing
end

"Jump the playhead to the previous (`dir < 0`) or next keyframe of `p`."
function gotokey!(player::Player, target, p::Param, dir::Integer)
    isanimated(p) || return setstatus!(player, "$(p.label): no keyframes yet — ◆ adds one")
    sf = playheadframe(player, target)
    ks = filter(k -> target.src_in <= k.frame <= target.src_out, p.curve[].keys)
    cand = dir < 0 ? filter(k -> k.frame < sf, ks) : filter(k -> k.frame > sf, ks)
    isempty(cand) && return setstatus!(player,
        "$(p.label): no keyframe $(dir < 0 ? "before" : "after") the playhead")
    k = dir < 0 ? last(cand) : first(cand)
    seek!(player, clamp(timelineframe(target, k.frame), 0, seqlength(player.sequence) - 1))
    return nothing
end

"Clear every keyframe of `p` (it keeps the value it was showing; undoable)."
function clearkeyframes!(player::Player, target, p::Param)
    isanimated(p) || return setstatus!(player, "$(p.label): no keyframes to clear")
    snapshot!(player)
    n = length(p.curve[].keys)
    clearkeys!(p, playheadframe(player, target))
    editedcurve!(player, target)
    setstatus!(player, "$(p.label): cleared $n keyframe$(n == 1 ? "" : "s") (Ctrl+Z to restore)")
    return nothing
end

"""
    shownparams(player) -> Vector{Param}

Every parameter of the clip whose cards are up.

`player.shownclip` is what those cards were built for and what the lanes are drawn
against, so "which parameters are on screen" has one answer. It used to have two:
a `:kfcurves` dictionary of per-name plot handles, written by a sweep that drew
every clip's curves at once. The sweep went and nothing wrote that dictionary
again, so `showcurves!` iterated an empty `Dict` and the palette's "Show/hide
keyframe curves" did nothing at all, silently.
"""
shownparams(player::Player) =
    player.shownclip === nothing ? Param[] :
    Param[p for fx in player.shownclip.effects for p in fx.params]

"Whether any keyframe lane is open right now. A hidden lane takes no clicks."
anycurvevisible(player::Player) = any(p -> p.visible[], shownparams(player))

"""
    showcurves!(player, on) -> Bool

Open or close every keyframe lane at once — what the palette command reaches.

Opening draws the CURVES, not every parameter: a clip's constants are straight
lines that say nothing and bury the curves among them. Closing closes everything,
including a constant whose lane somebody opened by hand.
"""
showcurves!(player::Player, on::Bool) =
    (showlanes!(player, on ? curvesof(shownparams(player)) : shownparams(player), on); on)

"""
    showlanes!(player, params, on) -> Bool

Draw or hide `params`' lanes.

Ends a solo, because this is the user picking lanes by hand: from here what is on
screen is the answer, and there is nothing left to put back. `Param.visible` is
what `lanecurve!` is drawn with, so setting it is the whole operation.
"""
function showlanes!(player::Player, params, on::Bool)
    player.lanesolo[] = nothing
    for p in params
        p.visible[] = on
    end
    return on
end

"""
    togglelanes!(player, p::Param) -> Bool
    togglelanes!(player, params::Vector{Param}) -> Bool

Flip a lane, or a group of them as one: if any of the group is drawn they all go,
otherwise its CURVES come — see [`curvesof`](@ref). Returns what they now are.

A group is mixed more often than not — one lane opened by a keyframe, the rest
closed — and "hide them all" is what the eye of a group with anything showing
should do.
"""
togglelanes!(player::Player, p::Param) = showlanes!(player, (p,), !p.visible[])

function togglelanes!(player::Player, params::Vector{Param})
    any(p -> p.visible[], params) && return showlanes!(player, params, false)
    return showlanes!(player, curvesof(params), true)
end

"""
    sololanes!(player, p::Param) -> Bool
    sololanes!(player, params::Vector{Param}) -> Bool

Draw this lane, or this group's curves, and nothing else. `true` while the solo is
on.

Alt-click on any ∿. Repeating it on the same row or group ends the solo and puts
back exactly what was drawn before it started; aiming it somewhere else moves the
solo and keeps the same restore point. This is the answer to a scene clip's two
hundred parameters: one gesture for one curve on its own, one to get everything
back.
"""
sololanes!(player::Player, p::Param) = sololanes!(player, Param[p], Param[p])

sololanes!(player::Player, params::Vector{Param}) =
    sololanes!(player, params, curvesof(params))

# `mark` is what the solo IS — the row or group whose ∿ was clicked, and what its
# ∿ compares itself against, so a group keeps one identity whether or not all of
# it turned out to be curves. `show` is what actually gets drawn.
function sololanes!(player::Player, mark::Vector{Param}, show::Vector{Param})
    isempty(mark) && return false
    want = Set{Param}(mark)
    s = player.lanesolo[]
    s !== nothing && s.on == want && return unsolo!(player)
    # The restore point survives the solo moving: it is taken only when there is
    # none, so walking a card row by row still ends where it started.
    before = s === nothing ?
             IdDict{Param, Bool}(p => p.visible[] for p in shownparams(player)) : s.before
    player.lanesolo[] = LaneSolo(want, before)
    drawn = Set{Param}(show)
    for p in shownparams(player)
        p.visible[] = p in drawn
    end
    for p in drawn             # …a group from a card that is not the shown one
        p.visible[] = true
    end
    setstatus!(player, isempty(drawn) ?
               "nothing here is animated yet — ◆ starts a curve" :
               "lane solo — alt-click the same ∿ again for the rest")
    return true
end

"""
    unsolo!(player) -> Bool

Put back the lanes a solo hid. `false` when none was on.
"""
function unsolo!(player::Player)
    s = player.lanesolo[]
    s === nothing && return false
    player.lanesolo[] = nothing
    for p in shownparams(player)
        p.visible[] = get(s.before, p, p.visible[])
    end
    setstatus!(player, "lane solo off — the other lanes are back as they were")
    return false
end

function Base.close(player::Player)
    pause!(player)   # also stops the audio feed
    try
        deactivatetool!(player)   # tool handlers reference the dying scenes
    catch e                       # …and a tool that cannot be torn down is a bug,
        @error "deactivating the tool on close failed" exception = (e, catch_backtrace())
    end
    # Listeners this player put on module-level observables. They come off before
    # the queues below close, or the next player in this process shares a notify
    # list with a corpse that throws — see `globallisteners`.
    foreach(Observables.off, player.globallisteners)
    empty!(player.globallisteners)
    freegpucache!(player)
    foreach(sp -> stop!(sp.worker), values(player.pools))
    stop!(player.timeline)
    close(player.statusqueue)
    close(player.uiqueue)
    player.screen === nothing || close(player.screen)
    player.screen = nothing
    return nothing
end
