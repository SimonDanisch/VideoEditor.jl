# Three names the document needs before the things they stand for exist. Each is
# a back-reference out of the document into what is looking at it, and each of
# those is declared files later — a `Player`, and the tool state inside one of its
# cards. A field left untyped instead would cost every method that reads it its
# dispatch, which is what `Clip.graph` used to pay.

"""
The editor a sequence is open in — a [`Player`](@ref), and only that.

A card is built against the EDITOR (its panel, its timeline, its playhead) while
the effect the card belongs to lives on the clip. Without a way back, every caller
that adds an effect would have to remember to ask the panel afterwards — which is
the reconcile pass in another costume.

The reference sits on the [`Sequence`](@ref), one per editor, and a clip reads it
through the sequence it is in — see [`editorof`](@ref). On the clip it would have
to be handed out again at every route a clip arrives by, and the one that forgot
would be a clip whose effects silently have no cards.
"""
abstract type Editor end

"""
The edit a clip belongs to — a [`Sequence`](@ref), and only that.

One direction further than [`Editor`](@ref): a sequence holds a `Vector{Clip}`,
so a clip cannot name its sequence by its concrete type. It is what
[`editorof`](@ref) walks through.
"""
abstract type Edit end

"""
The live state of a tool inside a card — a `ToolContext`, and only that.

An effect holds its own so that dropping the card drops the overlay plots, the
list rows and the listeners the tool put there. One that outlives its card keeps
firing, and its next redraw builds a widget into a layout whose scene is gone.
"""
abstract type ToolState end

"""
Per-source-frame color stabilization corrections (see `analyzecolor!`).
Keyed by absolute source frame via `src_in`, so tracks survive clip splits.
`strength` scales the correction toward identity (1 = full fix, 0 = off) at apply
time, so it is adjustable live without re-analysis.
"""
mutable struct ColorTrack
    const gains::Vector{Vec3f}
    const offsets::Vector{Vec3f}
    const src_in::Int
    strength::Float32
    # how much frame-to-frame brightness variation the correction takes out
    # (0..1), measured while analyzing — the number the UI reports so applying
    # the fix is not an act of faith
    reduction::Float32
end
ColorTrack(gains::Vector{Vec3f}, offsets::Vector{Vec3f}, src_in::Integer,
           strength::Real = 1.0f0, reduction::Real = 0.0f0) =
    ColorTrack(gains, offsets, src_in, Float32(strength), Float32(reduction))

"""
Per-source-frame camera stabilization warps in source pixel coordinates
(see `analyzemotion!`): projective sampling matrices. Same absolute-frame
keying as `ColorTrack`. `mode` records which analysis produced the track
(shown in the stabilize panel); `basecrop` remembers the clip's framing
from before the track's auto-crop was applied, so removing or replacing
the stabilization can restore it (`nothing` until an auto-crop happens).
"""
mutable struct MotionTrack
    const transforms::Vector{Mat3f}
    const src_in::Int
    const mode::Symbol
    basecrop::Union{Nothing, NTuple{4, Float64}}
end
MotionTrack(transforms::Vector{Mat3f}, src_in::Integer, mode::Symbol = :unknown) =
    MotionTrack(transforms, src_in, mode, nothing)

"""
Per-source-frame subject matte (see `analyzematte!`), same absolute-frame keying
as the other tracks.

`alpha` is `(w, h, nframes)` at *matte* resolution — usually smaller than the
source — and is sampled bilinearly when applied, so the matte does not have to
carry full-resolution pixels for every frame of a clip.

`seeds` are the frames the user marked: the matte keyframes. They are the edit,
`alpha` is only a cache of what propagating from them produced, which is why
`seeds` is what the project file stores and `alpha` goes to a sidecar that can be
regenerated. Sorted, absolute source frames.
"""
mutable struct MatteTrack
    const alpha::Array{UInt8, 3}
    const src_in::Int
    const seeds::Vector{Int}
end
MatteTrack(alpha::Array{UInt8, 3}, src_in::Integer, seeds::AbstractVector{<:Integer} = Int[]) =
    MatteTrack(alpha, Int(src_in), sort!(Int.(collect(seeds))))

mattesize(t::MatteTrack) = (size(t.alpha, 1), size(t.alpha, 2))

"""
Per-source-frame depth, keyed absolutely like the other tracks.

`depth` is `(w, h, nframes)` of `UInt8`, near = 255, far = 0, normalized per
frame. Eight bits because nothing here measures distance — it *orders* pixels
front to back so an effect can decide how much of one to apply, and a blur radius
resolved to one part in 256 is finer than any edge it produces.

Per frame, not per clip, and that is a real cost: a monocular depth model has no
scale, so its output is only comparable within one frame. Two frames of the same
shot can disagree about what "far" means, which is why an effect reading this
must be smooth in depth — a hard threshold on it flickers, and the flicker is the
model's, not the effect's.

No `seeds` field: depth is not marked, it is estimated. There is nothing the user
authored to keep, so the whole track is a cache and regenerating it is the only
thing a project file ever needs to record.
"""
mutable struct DepthTrack
    const depth::Array{UInt8, 3}
    const src_in::Int
end
DepthTrack(depth::Array{UInt8, 3}, src_in::Integer) = DepthTrack(depth, Int(src_in))

depthsize(t::DepthTrack) = (size(t.depth, 1), size(t.depth, 2))

"""
Restored frames for one clip, bounded (see `restore.jl`).

Keyed by absolute source frame like the tracks above, and here for the same
reason: `Clip` has a field of it. Unlike a track it is a cache — a clip's worth
of 4x frames is too much to keep or save — so `order` is an insertion queue that
evicts the oldest window past `limit`. An LRU would be better only if playback
ran backwards.
"""
mutable struct RestoreCache
    const frames::Dict{Int, Matrix{RGB{N0f8}}}
    const order::Vector{Int}
    limit::Int
end
RestoreCache(limit::Integer = 96) =
    RestoreCache(Dict{Int, Matrix{RGB{N0f8}}}(), Int[], Int(limit))

"""
Source of stable identities for clips and effect slots. Position in a vector and
`objectid` both die on the first sort, undo or project reload, so anything that
points at a clip or an effect (a blend at its partner, the inspector at a stack
entry, MCP at either) needs an id that survives them.
"""
const NEXTID = Threads.Atomic{UInt64}(0)
freshid() = UInt64(Threads.atomic_add!(NEXTID, UInt64(1)) + 1)

"""
One entry in a clip's effect stack: the effect, a stable `id` so the panel, an
edge or MCP can point at this entry, `enabled`, and the curves of its own
parameters. Several entries of one kind may coexist and are told apart by id.

There used to be a `links::Vector{FxLink}` here: a reference from one slot to
another with a role, so a blend could name its partner. It described a
relationship between two effects and nothing about what flows between them, so
nothing could render it. A [`ParamInput`](@ref) on the parameter says it at the
granularity where a value lives, and covers the dissolve, a caption fed by the
transcript and an audio-reactive number with one mechanism.

The curves belong on the entry rather than on the clip. `Clip` used to carry one
flat `Dict{Symbol, AnimCurve}`, so a curve knew a bare name and not which effect
it animated. That required a global `key -> ParamSpec` index, `registerparams!`
to fill it at registration, and resolution through `findeffect`, which returns
the first effect of a kind.

Measured on two Blur slots with one keyframe of 12 on `:blur`:
`[(blur = 12.0,), (blur = 0.0,)]` — the first animates, the second keeps its
static value, while each card's slider writes to its own slot. The slider and the
diamond of one row pointed at different objects.

A curve keyed by the effect's own parameter name on the slot that owns it cannot
be ambiguous, so none of that machinery is needed: the slot has the effect, the
effect's kind has the parameters, and the parameter has its curve. No global is
involved at any step.
"""
mutable struct Effect
    const id::UInt64
    const kind::Symbol                  # which kind this is (:blur, :color, …)
    # An Observable because the eye in the card header and the render both read
    # it: the button's glyph is derived from this rather than written next to the
    # flag by the click handler.
    const enabled::Observable{Bool}
    const params::Vector{Param}         # this entry's parameters, each with its curve
    # This entry's card in the Effects panel, or `nothing` while it has none.
    # Here rather than in a per-clip cache the panel keeps: the card is a property
    # of the effect the way its parameters are, so adding one builds a card,
    # removing one deletes it, and selecting another clip is a visibility flip.
    card::Union{Nothing, Makie.Card}
    # What the tool inside that card is holding — its overlay plots, its list rows
    # and its listeners (see [`ToolState`](@ref)). `nothing` for a card with no
    # tool in it, which is most of them.
    #
    # Beside `card` because its life IS the card's: a context that outlived one
    # kept listening, and its next redraw built a row into a layout whose scene
    # had been freed ("Can only use scenes with PixelCamera as topscene").
    tool::Union{Nothing, ToolState}
end
Effect(id::Integer, kind::Symbol, enabled::Bool, params::Vector{Param}) =
    Effect(UInt64(id), kind, Observable(enabled), params, nothing, nothing)

"""
    op(fx, frame = 0) -> FxOp

`fx` as the typed payload the render graph dispatches on (`BlurEffect`, …), with
every parameter sampled at `frame`.

Built on demand, not stored. As a field it was a second home for a value the
parameters also owned, the two joined by a name and a global index. The render
path is unchanged — it receives a `BlurEffect` and dispatches on its type — and
builds one per node per frame from the parameters.
"""
function op(fx::Effect, frame::Real = 0)
    k = kindbyname(fx.kind)
    k === nothing && error("no effect kind named $(repr(fx.kind)) is registered")
    return k.make(NamedTuple(p.name => valueat(p, frame) for p in fx.params))
end

"The parameters of `fx`, in the order its kind declares them."
params(fx::Effect) = fx.params

"`fx`'s parameter called `name`, or `nothing`."
param(fx::Effect, name::Symbol) =
    (i = findfirst(p -> p.name === name, fx.params); i === nothing ? nothing : fx.params[i])

"""
    copy(fx::Effect; id = fx.id) -> Effect

An independent copy of `fx`: its own `Param` objects, with their own curves and
their own Observables, so writing a value into one cannot reach the other and a
snapshot does not share listeners with the live entry.

No card comes across. A card belongs to the entry that is on screen, and a copy
is not on screen — the panel builds one for it if it ever becomes so.

The only copier. Of the three there were, one rebuilt the entry from `op(fx)` —
the payload, which carries a value at one frame and no curve — and so dropped
every animation into the undo stack.
"""
Base.copy(fx::Effect; id::Integer = fx.id) =
    Effect(UInt64(id), fx.kind, fx.enabled[], Param[copyparam(p) for p in fx.params])

"""
    paramsfor(kind, values) -> Vector{Param}

A kind's parameters at `values` (a NamedTuple), falling back to its declared
defaults. The one place a `Param` is minted, so a fresh effect and one built
from a payload cannot drift apart.
"""
paramsfor(k, values = NamedTuple()) =
    Param[Param(p.name, p.label, Float64(get(values, p.name, p.default));
                range = (Float64(p.min), Float64(p.max)))
          for p in k.params]

"""
    expandparams!(fx) -> fx

Add the parameters `fx`'s own data contributes, for every parameter that has any.

The kind declares a fixed list of scalars, which cannot describe a 3D scene: what
is animatable there is every number of every object. A parameter whose value is
structured data contributes its own parameters (see `dataparams`), and this is
where they join the list.

Idempotent, and it never overwrites: a path that already has a parameter keeps
the one it has, curve and all. That is what lets it run again after a scene is
edited, adding what is new without disturbing what was keyframed.
"""
function expandparams!(fx::Effect)
    for p in copy(fx.params), q in dataparams(valueat(p, 0))
        param(fx, q.name) === nothing && push!(fx.params, q)
    end
    return fx
end

"""
    Effect(payload::FxOp; enabled = true) -> Effect

An entry holding what `payload` says, for the call sites that build an effect by
constructing its typed struct (`Effect(BlurEffect(3f0))`). The kind is the one
that recognises it and the values are read back out through the kind, so this is
the exact inverse of [`op`](@ref).
"""
function Effect(payload; enabled::Bool = true)
    k = effectkindfor(payload)
    k === nothing && error("no registered effect kind recognises $(typeof(payload))")
    return Effect(freshid(), k.name, enabled, paramsfor(k, k.read(payload)))
end

function Effect(id::Integer, payload, enabled::Bool)
    k = effectkindfor(payload)
    k === nothing && error("no registered effect kind recognises $(typeof(payload))")
    return Effect(UInt64(id), k.name, enabled, paramsfor(k, k.read(payload)))
end

"""
A line of synthesized narration: what to say, when to say it (seconds from the
sequence's start), and which voice.

`samples` is a cache of `text`, not an edit — a project file carries the words
and re-renders, which is what keeps a minute of speech out of the JSON. `rate` is
the synthesizer's, kept because resampling on the way into the mixer needs it and
because a model that changes rate must not silently pitch-shift what is stored.

Here rather than in `narration.jl` for the reason [`Caption`](@ref) is: `Sequence` has a field of it,
and it is declared in this file.
"""
mutable struct Narration
    text::String
    at::Float64
    voice::String
    const samples::Vector{Float32}
    rate::Int
end
Narration(text::AbstractString, at::Real = 0.0, voice::AbstractString = "af_heart") =
    Narration(String(text), Float64(at), String(voice), Float32[], 0)

"""
One spoken line: `start` and `stop` in seconds from the sequence's beginning, and
what was said.

Here rather than in `captions.jl` because `Sequence` has a field of it, and it is
declared in this file. A caption is document data — the transcript is what a user
corrects when the model mishears — not a detail of the model that produced it.

Seconds, not frames, because that is what a speech model reports and what
survives the sequence's frame rate changing under it.
"""
struct Caption
    start::Float64
    stop::Float64
    text::String
end


"A node in a clip's render chain: produces one device image from the previous one."
abstract type FxNode end

"""
A clip's render as STRUCTURE: which nodes, in which order, and which effect each
one came from.

THE STRUCTURE, and nothing that changes per frame. Every value a node carries is
overwritten by [`update!`](@ref) before each run, so what is stored in `nodes` is
only ever the shape of the thing. `slots` is parallel to `nodes[2:end]` and is how
`update!` finds the effect to re-read.

Identity is the point: a graph object is REPLACED when the clip's structure
changes and never edited in place, so `objectid` of it IS the clip's structural
version. That is what a composition keys on — no fingerprint to compute, no hash
of the stack per frame, and nothing that can agree by accident.

Declared here rather than in gpugraph.jl, with the nodes it is made of, because
[`Clip`](@ref) has a field of one. A field left untyped to break an include cycle
costs every method that reads it its dispatch.
"""
struct FxGraph
    nodes::Vector{FxNode}
    slots::Vector{Effect}       # parallel to nodes[2:end]
    dims::Tuple{Int, Int}
end

"""
A clip's pre-rendered frames.

`dir` is where they are, `frames` is the source-frame range they cover, `canvas`
is what they were rendered at. `enabled` is the switch, and what invalidation
touches: a bake that no longer matches what the clip would render is switched off
and kept rather than deleted, because the picture may be exactly what was wanted
and re-rendering costs minutes.

`dirty` is set at the edit, not derived by comparison. There is no fingerprint
here: nothing walks the effect stack per frame to discover something the edit
already knew. The one thing that cannot be caught at an edit site is a file
changing under us — a source video replaced, a mesh re-exported — so the external
inputs are `stat`ed once on load, and that is the whole of what is checked.

Here rather than in bake.jl for the reason [`FxGraph`](@ref) is here: `Clip` has
a field of it. What baking DOES is still bake.jl's.
"""
mutable struct Bake
    dir::String
    frames::UnitRange{Int}
    canvas::Tuple{Int, Int}
    enabled::Bool
    dirty::Bool
    # mtime+size of every file the render read, as of the bake. Only files: an
    # edit inside the editor sets `dirty` where it happens.
    inputs::Dict{String, Tuple{Float64, Int}}
end
Bake(dir::AbstractString, frames::UnitRange{Int}, canvas::Tuple{Int, Int}) =
    Bake(String(dir), frames, canvas, true, false, Dict{String, Tuple{Float64, Int}}())

"""
What draws one clip on the timeline, while it is in a sequence that is open.

The `ClipView` plot and the three inputs that feed it, held by the clip — the
same shape as [`ParamView`](@ref), and for the same reason. The timeline used to
keep five parallel vectors (plots, ranges, source starts, states, sources), grow
and shrink them to `length(seq.clips)` on every edit and then write each clip by
INDEX. Every insert, delete and sort had to be reconciled; a plot could end up
drawing another clip's source, which is why there was a check for exactly that.

Held by the clip, none of it is a question: the plot a clip is drawn by is the
one it owns, and it goes away with the clip.
"""
mutable struct ClipPlot
    plot::Makie.Plot
    range::Observable{Tuple{Float64, Float64}}   # (start, stop) in timeline seconds
    srcstart::Observable{Float64}                # its in-point, in source seconds
    state::Observable{Symbol}                    # :idle | :hovered | :selected
end

"""
    Clip(source; src_in=0, src_out=source.nframes, start=0)

A non-destructive reference into a source video: frames
`src_in:(src_out-1)` of `source`, placed at timeline frame `start`.
`crop` is a normalized `(x, y, w, h)` rect ((0,0,1,1) = full frame,
y measured from the top).

`rate` is how many source frames one timeline frame advances: the conform factor
for a source that does not run at the sequence rate (a 30 fps clip in a 60 fps
timeline has `rate = 0.5` and shows each frame twice). It is
`source.framerate / seq.framerate`, set once when the clip is placed, and the only
place the two frame worlds differ — `src_in`/`src_out` count source frames,
`start`/[`cliplength`](@ref) count timeline frames, and [`sourceframe`](@ref) is
the conversion.

Conforming preserves wall-clock duration (`srclength/source.framerate` seconds
either way), so anything working in time — audio, the filmstrip, thumbnails —
needs no rate at all.
"""
mutable struct Clip <: FrameSource
    id::UInt64                  # stable across sorting, undo and save/load
                                # (settable so a project file restores its own)
    # Any source: a file on disk, or a Makie scene that renders its own frames. A
    # title, a lower third and a 3D animation are clips on a track like everything
    # else — they trim, take an effect stack and composite — rather than a second
    # system beside the timeline. See `ClipSource`.
    const source::ClipSource
    src_in::Int
    src_out::Int
    start::Int
    crop::NTuple{4, Float64}
    const effects::Vector{Effect}  # ordered effect stack (see effects.jl)
    colortrack::Union{Nothing, ColorTrack}
    motiontrack::Union{Nothing, MotionTrack}
    mattetrack::Union{Nothing, MatteTrack}
    # Estimated depth, `nothing` until something asks for it. A field beside the
    # other tracks rather than a global keyed by clip id — that is what made
    # `split!` drop a clip's restoration silently, since a fresh id follows
    # nothing (see `restorecache` below).
    depthtrack::Union{Nothing, DepthTrack}
    # A learned colour grade as a (D,D,D,3) table, host-side. Per clip, not per
    # frame: a look that drifted within a shot would be a fault (see `look.jl`).
    # Host-side because it is project-file data, and because a device array would
    # be on the wrong device after `autodetectgpu!`.
    look::Union{Nothing, Array{Float32, 4}}
    # How this clip fills timeline frames its source has no frame for — i.e. what
    # a slowed clip does between source frames. `:sample` repeats the nearest
    # (the default, and what every editor does with no model); `:flow`
    # synthesizes the in-between frame (see `flow.jl`). A property of the clip
    # like `rate`, not an effect: it decides what the source is, before any effect
    # runs, and an effect only ever sees one frame.
    timeinterp::Symbol
    # The fourth analysis result, and a field like the other three. It used to be
    # a module global keyed by clip id, which is how `split!` came to drop a
    # clip's restoration silently while taking explicit care of its matte: an id
    # is minted fresh for the right half, so nothing followed. `nothing` until a
    # window is restored; a cache, so it is never written to a project file.
    restorecache::Union{Nothing, RestoreCache}
    track::Int                  # stacking layer; higher = on top (1 = base)
    rate::Float64               # source frames per timeline frame (1 = native)
    # This clip's render as structure — see `FxGraph`. `nothing` until something
    # renders it, and reset by `dirtygraph!` when the structure changes. No value
    # is in it, so moving a slider or a keyframe leaves it alone. Never written to
    # a project file.
    graph::Union{Nothing, FxGraph}
    # Pre-rendered frames of this clip's chain, or `nothing` — see bake.jl. An
    # edit's output rather than a cache: minutes of raytracing should not be
    # recomputed on the next open, so it is saved with the project.
    bake::Union{Nothing, Bake}
    # Where this clip's effect cards sit while it is the one being edited — one
    # row of the Effects panel's stack, built once (see `buildcards!`). `nothing`
    # until the panel has drawn it.
    cardlayout::Union{Nothing, GridLayout}
    # The sequence this clip is in, and through it the editor it is open in, so
    # that putting an effect on the clip can build that effect's card — see
    # [`editorof`](@ref). `nothing` for a clip that is in no sequence: one being
    # built, one in an undo snapshot, one on the clipboard.
    sequence::Union{Nothing, Edit}
    # What draws this clip on the timeline, or `nothing` — see [`ClipPlot`](@ref).
    # Built by `addclip!`, dropped by `removeclip!`, exactly as a card is built by
    # `addslot!` and dropped by `removeslotat!`.
    view::Union{Nothing, ClipPlot}
end

"""
    editorof(clip) -> Union{Nothing, Editor}

The editor a clip is open in: the one its sequence is open in.

`nothing` where there is no answer — a clip in no sequence, or a sequence no
player has opened. Both are ordinary (a clip being built, a project read on a
worker), so every caller handles it rather than asserting.
"""
editorof(clip::Clip) = editorof(clip.sequence)
editorof(::Nothing) = nothing

Clip(source::ClipSource, src_in, src_out, start, crop, rate::Real = 1.0,
     reframe::Union{Nothing, NTuple{<:Any, <:Real}} = nothing) =
    withreframe!(Clip(freshid(), source, src_in, src_out, start, crop, Effect[],
                      # colortrack, motiontrack, mattetrack, depthtrack, look
                      nothing, nothing, nothing, nothing, nothing,
                      :sample,          # timeinterp
                      nothing,          # restorecache
                      1, Float64(rate),
                      nothing,          # graph — built on the first render
                      nothing,          # bake — none until somebody asks
                      nothing,          # cardlayout — until the panel draws it
                      nothing,          # sequence — until `addclip!` puts it in one
                      nothing),         # view — …which is also where it is drawn
                 reframe)

function Clip(source::ClipSource; src_in::Integer = 0, src_out::Integer = source.nframes,
              start::Integer = 0, rate::Real = 1.0)
    return Clip(source, src_in, src_out, start, (0.0, 0.0, 1.0, 1.0), rate)
end

"""
Fields of a [`Clip`](@ref) that decide its render structure rather than a value
inside it: an analysis that is present or absent (and so a pass that exists or
does not), and how the clip fills frames between source frames.
"""
const STRUCTURALFIELDS = (:motiontrack, :colortrack, :mattetrack, :depthtrack,
                          :look, :timeinterp)

"""
Writing one of [`STRUCTURALFIELDS`](@ref) drops the clip's compiled structure.

Here rather than at the twenty-odd assignment sites: `clip.mattetrack = track` is
written by the matte tool, the brush, the collect loop, three undo paths and the
project reader. Missing it at one leaves a graph with no matte pass under a clip
that has a matte, which renders a plausible picture with the effect absent.
"""
const PLACEFIELDS = (:start, :src_in, :src_out, :track, :rate)

function Base.setproperty!(clip::Clip, name::Symbol, x)
    name in STRUCTURALFIELDS && setfield!(clip, :graph, nothing)
    # the convert the default `setproperty!` would have done, which overloading
    # takes away: `clip.crop = corner` hands an `NTuple{4, Float32}` to an
    # `NTuple{4, Float64}` field, and `setfield!` alone throws a `TypeError` from
    # inside the crop tool naming neither the field nor the caller
    r = setfield!(clip, name, convert(fieldtype(Clip, name), x))
    # …and where it is drawn follows the write, here rather than at the thirty-odd
    # assignment sites — a trim, a drag, a ripple, a split and an undo all just
    # write these fields. `sort!` moves nothing and so places nothing.
    if name in PLACEFIELDS
        placeclip!(clip)
        retitle!(clip)   # …the panel header names the span
    end
    # Moving a clip to another track can change how many tracks there ARE, and
    # every lane's height is a share of the stack — so this is the one place a
    # write to one clip moves all the others. Placing only the clip that changed
    # left the rest drawn with the bands they had when there was one track fewer:
    # clip 1 went on covering both lanes, and a press anywhere in its old height
    # still found it.
    name === :track && placetracks!(clip)
    # …and the panel's bake row follows the field it reports on, for the same
    # reason: a bake finishing, an undo, a project load all just write this.
    name === :bake && showbake!(clip)
    return r
end

"""
    addslot!(clip, fx) -> fx
    removeslotat!(clip, i) -> clip

Put an effect on the stack, or take one off. The stack is a `const` field mutated
in place, so `setproperty!` cannot catch these; every caller goes through a named
function rather than `push!`ing into `clip.effects`.
"""
addslot!(clip::Clip, fx::Effect) =
    # …and its card, because this is where the entry joins the document and the
    # clip knows the editor it is open in. Whichever route put the effect here —
    # the add menu, `seteffect!`, an undo, a project being read — the card comes
    # with it, and nothing goes looking afterwards for effects that lack one.
    (push!(clip.effects, fx); buildcardfor!(clip, fx); dirtygraph!(clip); fx)

"""
Every slot off the clip at once.

`empty!(clip.effects)` cannot do this: the field is `const`, so a write to it is
invisible to [`setproperty!`](@ref) and the clip would keep a compiled graph with
passes for effects that are gone.
"""
Base.empty!(clip::Clip) =
    (foreach(dropcard!, clip.effects); empty!(clip.effects); dirtygraph!(clip); clip)
removeslotat!(clip::Clip, i::Integer) =
    (dropcard!(clip.effects[i]); deleteat!(clip.effects, i); dirtygraph!(clip); clip)

"""
Seed a clip's placement from a (scale, x, y[, rotation°]) tuple — the shape a
project file written before the transform became an effect holds. `nothing`, and
anything identity, adds no effect at all.
"""
function withreframe!(clip::Clip, r)
    r === nothing && return clip
    t = reframe4(r)
    t == NEUTRALFRAME && return clip
    seteffect!(clip, TransformEffect(t[1], t[2], t[3], t[4]))
    return clip
end

"The identity placement: fitted whole, centred, unrotated, untouched by the user."
const NEUTRALFRAME = (1.0, 0.0, 0.0, 0.0)

"""
A reframe tuple as (scale, x, y, rotation°). Takes the 3-tuple too: projects
saved before rotation existed hold one, and so does any caller that never cared.
"""
reframe4(r::NTuple{4, <:Real}) = Float64.(r)
reframe4(r::NTuple{3, <:Real}) = (Float64(r[1]), Float64(r[2]), Float64(r[3]), 0.0)

"Whether this clip is placed by the plain fit, with no manual zoom or shift."
neutralframe(clip::Clip) = transformof(clip) == NEUTRALFRAME

"""
    conformrate(source, framerate) -> Float64

The [`Clip`](@ref) `rate` that puts `source` into a timeline running at
`framerate`. Exactly 1.0 when the rates agree (to a hundredth of a frame), so a
matching source never picks up conform arithmetic — or its rounding.
"""
conformrate(source::ClipSource, framerate::Real) =
    isapprox(source.framerate, framerate; atol = 0.01) ? 1.0 :
    Float64(source.framerate) / Float64(framerate)

"Frames of source this clip spans — what every per-source-frame analysis iterates."
srclength(clip::Clip) = clip.src_out - clip.src_in

"""
Frames of timeline this clip occupies — its extent in the edit. Equal to
[`srclength`](@ref) on a native clip; `floor` (not `round`) so the last timeline
frame always maps inside the source range.
"""
cliplength(clip::Clip) = max(floor(Int, srclength(clip) / clip.rate), 0)
clipend(clip::Clip) = clip.start + cliplength(clip)
hascrop(clip::Clip) = clip.crop != (0.0, 0.0, 1.0, 1.0)

"Whether this clip is retimed to the sequence rate rather than running natively."
conformed(clip::Clip) = clip.rate != 1.0

"""
    sourceframe(clip, n) -> Int

The source frame showing at timeline frame `n`: the conversion between the edit's
frame world and the media's. Not clamped, since transition handles ask past a
clip's own range.
"""
sourceframe(clip::Clip, n::Integer) =
    clip.rate == 1.0 ? clip.src_in + (Int(n) - clip.start) :
    clip.src_in + floor(Int, (Int(n) - clip.start) * clip.rate)

"""
    sourcephase(clip, n) -> Float64

How far timeline frame `n` sits between [`sourceframe`](@ref)`(clip, n)` and the
one after it, in `0..1`.

Exactly the fraction `sourceframe` throws away with its `floor`. It is zero for
an unconformed clip, and for a slowed one it is the position a frame
interpolator would synthesize at: `rate = 0.5` gives 0, 0.5, 0, 0.5… — the
alternating half-steps that are shown as repeated frames without one, which is
what makes slow motion judder.
"""
sourcephase(clip::Clip, n::Integer) =
    clip.rate == 1.0 ? 0.0 :
    (x = (Int(n) - clip.start) * clip.rate; Float64(x - floor(x)))

"Timeline frames that `nsrc` source frames of this clip's media occupy."
timelineframes(clip::Clip, nsrc::Integer) = floor(Int, Int(nsrc) / clip.rate)

"""
    timelineframe(clip, sf) -> Int

Where source frame `sf` of `clip` shows on the timeline — the inverse of
[`sourceframe`](@ref). Every keyframe marker, readout and ◆ jump needs it: keys
are stored on source frames but drawn against the timeline.
"""
timelineframe(clip::Clip, sf::Integer) =
    clip.start + timelineframes(clip, Int(sf) - clip.src_in)

"""
A transition centered on the cut at timeline frame `at` (== `clipend(left)` ==
`right.start` of the two adjacent clips), blending the outgoing clip into the
incoming one over `duration` frames. `kind` is `:dissolve` (cross-dissolve) for
now — the outgoing clip is extended past its out-point and the incoming clip is
pulled in before its in-point (using each source's handle frames), then the two
are mixed `(1-p)·A + p·B` with `p` sweeping 0→1 across the region.
"""
mutable struct Transition
    kind::Symbol
    at::Int
    duration::Int
end

"Half-width of a transition (frames on each side of the cut)."
transhalf(t::Transition) = t.duration ÷ 2
transstart(t::Transition) = t.at - transhalf(t)
transstop(t::Transition) = transstart(t) + t.duration   # exclusive

"""
    Sequence(source) / Sequence(clips, framerate)

An edited timeline: non-overlapping clips sorted by start. All edits are
metadata operations on this structure; frames are resolved on demand via
`locate`. `transitions` overlay cross-dissolves on clip cuts.
"""
mutable struct Sequence <: Edit
    const clips::Vector{Clip}
    framerate::Float64
    const transitions::Vector{Transition}
    # What is spoken, and when. An edit rather than a cache: a user fixes it when
    # the model mishears a word, so it is saved with the project and restored by
    # undo, and regenerating it is minutes of Whisper.
    const captions::Vector{Caption}
    # Spoken narration mixed over the clips, on both the preview and the export.
    # An edit like the captions: the words are saved, the samples are a cache.
    const narration::Vector{Narration}
    # The output resolution, once something has set it — the crop tool does.
    #
    # `nothing` derives it from the first clip, which was the only behaviour and
    # is a trap: deleting or reordering clips then changes the project's
    # resolution, and cropping clip 2 resized nothing while cropping clip 1
    # resized everything. Kept as the fallback so older projects open unchanged.
    canvas::Union{Nothing, Tuple{Int, Int}}
    # How tall each track is drawn, as a weight: 1.0 is its equal share, 2.0 twice
    # its neighbours. Not pixels — the timeline fills the space it is given, so a
    # resized track keeps its proportion when the window changes. Saved with the
    # project, since a track made tall to work on is part of the layout.
    #
    # Shorter than the track count is normal (and is what every older project
    # reads as): a track with no entry weighs 1.0. See `trackweight`.
    const trackheights::Vector{Float64}
    # The track filling the whole lane area and hiding the others; 0 for the normal
    # stacked view. Unlike `trackheights` this is not saved: it is a way of looking
    # at the edit (double-click a lane), and a project opening with every track but
    # one missing would read as data loss. See `solotrack!`.
    solo::Int
    # The editor this sequence is open in — see [`Editor`](@ref). Written once,
    # when a `Player` opens it, and read by every clip in it. Not saved and not
    # copied: it is what the document is being looked at through, not part of it.
    editor::Union{Nothing, Editor}

    function Sequence(clips::Vector{Clip}, framerate::Real,
                      transitions::Vector{Transition}, captions::Vector{Caption},
                      narration::Vector{Narration},
                      canvas::Union{Nothing, Tuple{Int, Int}},
                      trackheights::Vector{Float64}, solo::Integer)
        seq = new(clips, Float64(framerate), transitions, captions, narration,
                  canvas, trackheights, Int(solo), nothing)
        # the clips handed in join it exactly as a later one does — see `addclip!`
        foreach(c -> (c.sequence = seq), clips)
        return seq
    end
end

Sequence(clips::Vector{Clip}, framerate::Real) =
    Sequence(clips, framerate, Transition[], Caption[], Narration[], nothing, Float64[], 0)
Sequence(clips::Vector{Clip}, framerate::Real, transitions::Vector{Transition}) =
    Sequence(clips, framerate, transitions, Caption[], Narration[], nothing, Float64[], 0)
Sequence(clips::Vector{Clip}, framerate::Real, transitions::Vector{Transition},
         captions::Vector{Caption}, narration::Vector{Narration},
         canvas::Union{Nothing, Tuple{Int, Int}}) =
    Sequence(clips, framerate, transitions, captions, narration, canvas, Float64[], 0)
Sequence(clips::Vector{Clip}, framerate::Real, transitions::Vector{Transition},
         captions::Vector{Caption}, narration::Vector{Narration},
         canvas::Union{Nothing, Tuple{Int, Int}}, trackheights::Vector{Float64}) =
    Sequence(clips, framerate, transitions, captions, narration, canvas, trackheights, 0)
Sequence(source::ClipSource) = Sequence([Clip(source)], source.framerate)

editorof(seq::Sequence) = seq.editor

"""
    addclip!(seq, clip) -> clip
    addclip!(seq, i, clip) -> clip

Put a clip in the sequence, at the end or at index `i`. The only way one gets
there.

`push!(seq.clips, clip)` cannot do this: the clip has to learn which sequence it
is in, because that is how it reaches the editor it is open in (see
[`editorof`](@ref)), and an effect put on a clip that cannot reach one builds no
card. There were seven `push!`es and two of them were the paths a user takes most
— dropping a file in, and pasting.
"""
function addclip!(seq::Sequence, clip::Clip)
    clip.sequence = seq
    push!(seq.clips, clip)
    buildclipview!(clip)
    restack!(seq)
    return clip
end
function addclip!(seq::Sequence, i::Integer, clip::Clip)
    clip.sequence = seq
    insert!(seq.clips, i, clip)
    buildclipview!(clip)
    restack!(seq)
    return clip
end

"""
    removeclip!(seq, i) -> Clip
    empty!(seq) -> seq

Take a clip out of the sequence, or all of them. The only way one leaves.

`deleteat!(seq.clips, i)` cannot do this: what the timeline draws the clip with
belongs to the clip ([`ClipPlot`](@ref)), so it has to go when the clip does —
the same statement [`removeslotat!`](@ref) makes about a card.
"""
function removeclip!(seq::Sequence, i::Integer)
    clip = seq.clips[i]
    deleteat!(seq.clips, i)
    dropclipview!(clip)
    clip.sequence = nothing
    prunetransitions!(seq)   # a cut that no longer exists carries no dissolve
    restack!(seq)
    return clip
end

"The stack gained or lost a lane: every clip's band is a function of the count."
function restack!(seq::Sequence)
    player = editorof(seq)
    player === nothing || placetracks!(player.timeline)
    return nothing
end

function Base.empty!(seq::Sequence)
    foreach(dropclipview!, seq.clips)
    foreach(c -> (c.sequence = nothing), seq.clips)
    empty!(seq.clips)
    return seq
end

"The clip with `id`, or `nothing` — how anything refers to a clip across sorting,
undo and reloads (indices shift, `objectid` dies on the first copy)."
function clipbyid(seq::Sequence, id::Integer)
    i = findfirst(c -> c.id == id, seq.clips)
    return i === nothing ? nothing : seq.clips[i]
end

"""
    bindinputs!(seq) -> seq

Resolve every [`ParamInput`](@ref) in the sequence: turn the ids a node is
written in into the objects [`valueat`](@ref) follows.

Run on every structural change — a project loaded, an undo restored, a clip or an
effect added or removed — which is when an id can start or stop naming something.
Editing a value or moving a keyframe changes nothing here.

An edge whose target is gone is left unresolved rather than deleted: the ids are
what the user wrote, and a clip coming back through undo has to bring the edge
that pointed at it back with it. Unresolved reads as the parameter's own value.

Cycles are refused at the last edge that would close one, so a parameter driven in
a loop keeps its own value instead of hanging the render. Refused here rather than
at each read: this runs once per edit, `valueat` per parameter per frame.
"""
function bindinputs!(seq::Sequence)
    for clip in seq.clips, fx in clip.effects, p in fx.params
        n = p.input
        n === nothing && continue
        n.to = clip
        n.resolved = Any[resolveinput(seq, clip, fx, p, r) for r in n.inputs]
        n.from = Union{Nothing, FrameSource}[inputclip(seq, clip, r) for r in n.inputs]
    end
    for clip in seq.clips, fx in clip.effects, p in fx.params
        drivencycle(p) && (p.input.resolved = Any[nothing for _ in p.input.inputs])
    end
    return seq
end

"""
    resolveinput(seq, clip, fx, p, ref) -> what `readinput` reads, or `nothing`

One address turned into the thing it names. `nothing` where it names nothing —
the target was deleted, the file is gone — because an edit that removes something
must leave the project openable and the dangling input visible, not throw.
"""
function resolveinput(seq::Sequence, clip::Clip, fx::Effect, p::Param, r::ParamRef)
    target = r.clip == 0 ? clip : clipbyid(seq, r.clip)
    target === nothing && return nothing
    if r.effect != 0
        slot = findslot(target, r.effect)
        slot === nothing && return nothing
        q = param(slot, r.param)
        return (q === nothing || q === p) ? nothing : q
    end
    # `effect = 0` means "by name on that clip": across a clip boundary "the same
    # effect" names nothing, and within one clip a parameter name is unique enough.
    # The parameter's own effect is tried first, so a self-reference resolves the
    # way it reads.
    for slot in (target === clip ? (fx, target.effects...) : (target.effects...,))
        q = param(slot, r.param)
        (q === nothing || q === p) && continue          # nothing drives itself
        return q
    end
    return nothing
end
resolveinput(::Sequence, ::Clip, ::Effect, ::Param, r::FileRef) =
    isfile(r.path) ? loadinputfile(r.path) : nothing
resolveinput(seq::Sequence, ::Clip, ::Effect, ::Param, r::ClipRef) = clipbyid(seq, r.clip)

"""
A clip's picture is not a value but a transient that exists while a composition
runs, so the graph wires it and a value lookup never fetches it. Refused rather
than returning the clip itself, which would reach a kernel as something it cannot
use.
"""
readinput(::Clip, ::Integer) =
    error("a clip's picture is a graph input, not a value — the composition wires \
           it, `valueat` does not read it")

"What frames an input's values are counted in, or `nothing` when it has none."
inputclip(seq::Sequence, clip::Clip, r::ParamRef) = r.clip == 0 ? clip : clipbyid(seq, r.clip)
inputclip(seq::Sequence, ::Clip, r::ClipRef) = clipbyid(seq, r.clip)
inputclip(::Sequence, ::Clip, ::FileRef) = nothing

"""
    loadinputfile(path) -> data

A file input's contents, held so a per-frame read does not touch the disk.

Dispatch by extension is deliberately absent: `FileIO.load` already knows what a
`.stl` and a `.png` are, and a table here would be a second, worse copy of that
registry which a new format would have to be added to twice.
"""
const INPUTFILES = Dict{String, Any}()
const INPUTFILELOCK = ReentrantLock()

function loadinputfile(path::AbstractString)
    key = string(abspath(path), ":", mtime(path), ":", filesize(path))
    lock(INPUTFILELOCK) do
        get!(() -> Makie.FileIO.load(path), INPUTFILES, key)
    end
end

"Whether following `p`'s edges comes back to `p`. Bounded by the walk itself: it
stops the moment it revisits anything, so a loop that does not include `p` ends it
too."
function drivencycle(p::Param)
    seen = Base.IdSet{Any}((p,))
    stack = Any[p]
    while !isempty(stack)
        q = pop!(stack)
        n = q isa Param ? q.input : nothing
        n === nothing && continue
        for r in n.resolved
            r isa Param || continue
            r === p && return true
            r in seen && continue
            push!(seen, r); push!(stack, r)
        end
    end
    return false
end

seqlength(seq::Sequence) = maximum(clipend, seq.clips; init = 0)
seqduration(seq::Sequence) = seqlength(seq) / seq.framerate

"Number of stacking layers (1-based; the base track is 1)."
ntracks(seq::Sequence) = isempty(seq.clips) ? 1 : maximum(c.track for c in seq.clips)

"Every clip covering timeline frame `n`, bottom track first (base → top)."
clipsat(seq::Sequence, n::Integer) =
    sort!([c for c in seq.clips if c.start <= n < clipend(c)]; by = c -> c.track)

"Index of the topmost clip containing timeline frame `n`, or `nothing` (gap)."
function clipat(seq::Sequence, n::Integer)
    best = nothing; besttrack = typemin(Int)
    for (i, c) in enumerate(seq.clips)
        if c.start <= n < clipend(c) && c.track > besttrack
            best = i; besttrack = c.track
        end
    end
    return best
end

"""
Index of the clip covering frame `n` on `track`: the lane-aware `clipat`. The
preview renders the topmost clip, but a click (and the inspector behind it) has
to reach the one stacked below it.
"""
function clipat(seq::Sequence, n::Integer, track::Integer)
    for (i, c) in enumerate(seq.clips)
        c.track == track && c.start <= n < clipend(c) && return i
    end
    return nothing
end

"Resolve timeline frame `n` to `(clip, source_frame)`, or `nothing` in a gap."
function locate(seq::Sequence, n::Integer)
    i = clipat(seq, n)
    i === nothing && return nothing
    clip = seq.clips[i]
    return (clip, sourceframe(clip, n))
end

"The transition whose region contains timeline frame `n`, or `nothing`."
function transitionat(seq::Sequence, n::Integer)
    for t in seq.transitions
        transstart(t) <= n < transstop(t) && return t
    end
    return nothing
end

"The outgoing clip (ending at `at`) and incoming clip (starting at `at`), or `nothing`s."
function transitionclips(seq::Sequence, at::Integer)
    l = findfirst(c -> clipend(c) == at, seq.clips)
    r = findfirst(c -> c.start == at, seq.clips)
    return (l === nothing ? nothing : seq.clips[l], r === nothing ? nothing : seq.clips[r])
end

"""
    transitionsample(seq, t, n) -> (left, srcA, right, srcB, p) | nothing

Resolve timeline frame `n` inside transition `t`: the two clips, the source
frame each contributes (extended into its handle past the cut, clamped to
available source), and the mix `p` ∈ [0,1] (0 = fully outgoing, 1 = incoming).
"""
function transitionsample(seq::Sequence, t::Transition, n::Integer)
    left, right = transitionclips(seq, t.at)
    (left === nothing || right === nothing) && return nothing
    p = clamp((n - transstart(t) + 0.5) / t.duration, 0.0, 1.0)
    srcA = clamp(sourceframe(left, n), 0, left.source.nframes - 1)
    srcB = clamp(sourceframe(right, n), 0, right.source.nframes - 1)
    return (left, srcA, right, srcB, p)
end

"Largest even duration a dissolve on this cut can take without overrunning either clip."
clamptransition(left::Clip, right::Clip, duration::Integer) =
    2 * max(min(duration ÷ 2, cliplength(left), cliplength(right)), 0)

"""
The dissolve length a one-click blend uses on this cut: 0.6 s, and never more than
half the shorter clip, so each side keeps three quarters of itself un-blended. The hard limit ([`clamptransition`](@ref)) allows a dissolve twice
the shorter clip — on short clips (loop cuts!) that swallows both of them whole
and the timeline is one big bowtie with no clip left to see.
"""
defaultdissolve(seq::Sequence, left::Clip, right::Clip) =
    max(min(round(Int, 0.6 * seq.framerate),
            cliplength(left) ÷ 2, cliplength(right) ÷ 2), 2)

"""
    addtransition!(seq, at; duration, kind=:dissolve) -> Union{Transition, Nothing}

Add (or resize) a cross-dissolve on the cut at timeline frame `at`. No-op unless
`at` is a real cut between two adjacent clips; `duration` is clamped to fit both.
"""
function addtransition!(seq::Sequence, at::Integer; duration::Integer, kind::Symbol = :dissolve)
    left, right = transitionclips(seq, at)
    (left === nothing || right === nothing) && return nothing
    dur = clamptransition(left, right, duration)
    dur >= 2 || return nothing
    i = findfirst(t -> t.at == at, seq.transitions)
    if i === nothing
        t = Transition(kind, at, dur)
        push!(seq.transitions, t)
        drawtransitions!(seq)
        return t
    end
    seq.transitions[i].kind = kind
    seq.transitions[i].duration = dur
    drawtransitions!(seq)
    return seq.transitions[i]
end

"Remove the transition on the cut at `at` (returns it, or `nothing`)."
function removetransition!(seq::Sequence, at::Integer)
    i = findfirst(t -> t.at == at, seq.transitions)
    i === nothing && return nothing
    t = seq.transitions[i]
    deleteat!(seq.transitions, i)
    drawtransitions!(seq)
    return t
end

"Drop transitions whose cut no longer exists (after edits that move/merge clips)."
function prunetransitions!(seq::Sequence)
    n = length(seq.transitions)
    filter!(seq.transitions) do t
        left, right = transitionclips(seq, t.at)
        left !== nothing && right !== nothing
    end
    length(seq.transitions) == n || drawtransitions!(seq)
    return seq
end

"""
    drawtransitions!(seq) -> nothing

Redraw the cross-dissolve markers, because the LIST of them changed.

The boxes are two shared plots fed by two vectors, so this recomputes both from
`seq.transitions` — a pure function of the document, not a pass that goes looking
for what is out of date. It is called where a transition is added, re-keyed,
removed or pruned, and nowhere else: it used to hang off `relayout!`, and then off
`placetracks!`, which fires on the STACK's geometry — a fact these markers do not
depend on (their band is fixed) and which is silent about the one that they do.
"""
function drawtransitions!(seq::Sequence)
    player = editorof(seq)
    player === nothing || refreshtransitions!(player.timeline)
    return nothing
end

"""
    split!(seq, n) -> Union{Clip, Nothing}

Split the clip containing timeline frame `n` at `n`; the right half is
returned. No-op at a clip start or in a gap.
"""
function split!(seq::Sequence, n::Integer, track::Union{Nothing, Integer} = nothing)
    # `track` names the lane to cut. Without it `clipat` answers with the topmost
    # clip at `n`, so pressing S with V2 selected cuts V3.
    i = track === nothing ? clipat(seq, n) : clipat(seq, n, Int(track))
    i === nothing && return nothing
    clip = seq.clips[i]
    n == clip.start && return nothing
    cut = sourceframe(clip, n)            # the cut in source frames; both halves share it
    cut > clip.src_in || return nothing
    # …and the halves have to meet: on a conformed clip several timeline frames
    # show the same source frame, so the cut snaps back to where that frame
    # starts. Cutting at the raw `n` leaves the left half one frame short of the
    # right, a hole that only appears on retimed material.
    at = clip.start + timelineframes(clip, cut - clip.src_in)
    right = Clip(clip.source, cut, clip.src_out, at, clip.crop, clip.rate)
    right.track = clip.track              # both halves stay on the same stacking layer
    # …and the blend pairing, which lives on the opacity parameter (an edge with
    # `op = :pairedwith`) and is copied with the stack below.
    # each half owns its stack: same effects, own slot ids, so the inspector can
    # address one half's entry without touching the other's
    # …and a data entry (the `:scene`) is copied as itself: it has no payload to
    # rebuild from, and `copy` gives it its own parameters and curves.
    for s in clip.effects
        addslot!(right, renderable(s) ? Effect(freshid(), op(s), s.enabled[]) :
                                        copy(s; id = freshid()))
    end
    # keyed by absolute source frame, so both halves stay valid. Assigned directly:
    # the stack was copied above, slots and all, so `setmotiontrack!` would prepend
    # a second Stabilize slot.
    right.colortrack = clip.colortrack
    right.motiontrack = clip.motiontrack
    right.mattetrack = clip.mattetrack    # ditto — cutting a clip must not lose its matte
    right.depthtrack = clip.depthtrack    # …nor its depth, which is keyed the same way
    right.look = clip.look                # both halves of a cut keep the shot's grade
    right.timeinterp = clip.timeinterp    # …and how it fills in between frames
    # Shared, not copied, exactly as the tracks are: the frames are keyed by
    # absolute source frame, so one cache indexes correctly from both halves. The
    # two then share the eviction budget, which is the bargain a shared track
    # makes anyway.
    right.restorecache = clip.restorecache
    # Keys are absolute-frame, so both halves keep every one, but each half needs
    # its own curve object or editing one reaches into the other. `fi`/`pi`, not
    # `i`: `i` is the clip's index in the sequence and `insert!` below needs it.
    for (fi, fx) in enumerate(clip.effects)
        for (pi, prm) in enumerate(fx.params)
            right.effects[fi].params[pi].curve[] =
                AnimCurve(copy(prm.curve[].keys), prm.curve[].interp)
        end
    end
    clip.src_out = cut
    addclip!(seq, i + 1, right)   # …and with that the right half is in the editor too
    return right
end

"""
    copyclip(clip; start = clip.start, track = clip.track) -> Clip

An independent copy of `clip` — the same source range, the same effect stack, the
same analysis — placed at `start` on `track`. Not inserted into any sequence.

What is copied and what is shared follows [`split!`](@ref): a derived clip reads
the same source frames, so anything keyed by absolute source frame is correct to
share and expensive to duplicate.

* Fresh: the clip `id`, and one `id` per [`Effect`](@ref) with its own copy of the
  links. Two slots sharing an id would make the inspector and the blend card
  address both at once. The `Effect` inside a slot is shared, which is safe —
  every effect is an immutable `struct`, so a parameter change replaces it.
* Shared: `colortrack`, `motiontrack`, `mattetrack` and `restorecache`. All four
  are keyed by absolute source frame and the copy covers the same frames, so one
  analysis indexes correctly from both; re-running a matte would cost minutes.
* Copied: the animation curves, which are edited per clip and have to move
  independently.
* Dropped: the blend pairing. It names another clip by id, and a copy elsewhere in
  the timeline is not in that transition. `split!` keeps it because its left half
  continues the same blend.
"""
copyclip(clip::Clip; start::Integer = clip.start, track::Integer = clip.track) =
    withfields(clip; id = freshid(), start = Int(start), track = Int(track),
               # Fresh slot ids and copied link vectors: the copy's stack is its
               # own, so unlinking on one must not reach into the other.
               # …and every parameter's edge goes unresolved with it, so the copy's
               # blend pairing points at nothing until `bindinputs!` runs — which
               # is right: a copy is not in the original's transition.
               effects = [copy(fx; id = freshid()) for fx in clip.effects])

"""
    trimclip!(seq, clip, i, side, n) -> clip

Move one edge of `clip` (`i` = its index in `seq.clips`) to timeline frame `n`.
The left edge shifts `start` and `src_in` together so the content stays anchored;
the right edge moves `src_out`. Clamped to the available source and to the
neighbours on the same lane.

The edge walks timeline frames while the in/out points count source frames, which
on a conformed clip is not the same step — hence one function rather than
arithmetic in the drag handler.
"""
function trimclip!(seq::Sequence, clip::Clip, i::Integer, side::Symbol, n::Integer)
    if side === :right
        maxend = clip.start + timelineframes(clip, clip.source.nframes - clip.src_in)
        nxt = nextontrack(seq, clip)
        nxt === nothing || (maxend = min(maxend, nxt.start))
        # `max(…, clip.start + 1)`: a clip is never trimmed out of existence, and
        # the bound it is clamped against must not invert. Taking the limit from
        # `seq.clips[i + 1]` did invert it — that list is sorted by (track, start),
        # so on a stack the "next clip" was usually one on another track, often
        # starting earlier, and `src_out` landed at or before `src_in`.
        newend = clamp(Int(n), clip.start + 1, max(maxend, clip.start + 1))
        clip.src_out = sourceframe(clip, newend)
    else
        prv = prevontrack(seq, clip)
        minstart = max(prv === nothing ? 0 : clipend(prv),
                       clip.start - timelineframes(clip, clip.src_in))  # src_in stays ≥ 0
        newstart = clamp(Int(n), min(minstart, clipend(clip) - 1), clipend(clip) - 1)
        delta = newstart - clip.start
        clip.src_in += round(Int, delta * clip.rate)
        clip.start += delta
    end
    return clip
end

"The clip that follows `clip` on its own track, or `nothing`."
function nextontrack(seq::Sequence, clip::Clip)
    best = nothing
    for c in seq.clips
        c === clip && continue
        c.track == clip.track || continue
        c.start >= clipend(clip) || continue
        (best === nothing || c.start < best.start) && (best = c)
    end
    return best
end

"The clip that precedes `clip` on its own track, or `nothing`."
function prevontrack(seq::Sequence, clip::Clip)
    best = nothing
    for c in seq.clips
        c === clip && continue
        c.track == clip.track || continue
        clipend(c) <= clip.start || continue
        (best === nothing || clipend(c) > clipend(best)) && (best = c)
    end
    return best
end

"""
    joinclips!(seq, n) -> Union{Clip, Nothing}

Merge the clip at frame `n` with the clip that follows it on the same track,
when the two are halves of one cut: same source, timeline-contiguous and
source-contiguous (`left.src_out == right.src_in`). The left half's effects
and analyses win; the right half's keyframes (keyed by absolute source frame)
carry over where the left has none. The inverse of [`split!`](@ref).
"""
function joinclips!(seq::Sequence, n::Integer)
    i = clipat(seq, n)
    i === nothing && return nothing
    c = seq.clips[i]
    j = findfirst(o -> o !== c && o.track == c.track && o.source === c.source &&
                       o.rate == c.rate &&
                       o.start == clipend(c) && o.src_in == c.src_out, seq.clips)
    j === nothing && return nothing
    nxt = seq.clips[j]
    removetransition!(seq, nxt.start)     # a dissolve on the joined cut is gone with it
    # Carry the right half's keys over, matched by kind rather than position: the
    # halves need not have the same stack, so indexing by position writes a curve
    # onto a different effect or drops it (the joined clip lost its fade).
    for fx in nxt.effects
        any(isanimated, fx.params) || continue
        i = findfirst(g -> g.kind === fx.kind, c.effects)
        if i === nothing
            addslot!(c, copy(fx; id = freshid()))
            continue
        end
        for prm in fx.params
            isanimated(prm) || continue     # a constant is the left half's to keep
            dst = param(c.effects[i], prm.name)
            dst === nothing && continue
            foreach(k -> setkey!(dst, k.frame, k.value, k.ease), prm.curve[].keys)
        end
    end
    c.src_out = nxt.src_out
    removeclip!(seq, j)
    return c
end

"""
    deleteclip!(seq, n; ripple=true) -> Union{Clip, Nothing}

Delete the clip containing timeline frame `n`. With `ripple`, later clips
shift left to close the gap.
"""
function deleteclip!(seq::Sequence, n::Integer; ripple::Bool = true)
    i = clipat(seq, n)
    i === nothing && return nothing
    return deleteclip!(seq, seq.clips[i]; ripple)
end

"""
    rippledoc!(seq, from, by) -> seq

Shift the document-level timings at or after `from` seconds earlier by `by`.

The captions and the narration are pinned to the picture, not to the wall clock.
Without this, `deleteclip!` ripples the clips and leaves both where they were, so
every subtitle and voiceover after the cut goes out of sync with its shot.

Both are replaced rather than mutated: `docsnapshot` shares these objects with
every undo step holding them, so shifting one in place would rewrite the history
meant to put it back. The narration's samples come along — the words have not
changed, only when they are said.
"""
function rippledoc!(seq::Sequence, from::Real, by::Real)
    for (i, c) in enumerate(seq.captions)
        c.start >= from && (seq.captions[i] = Caption(c.start - by, c.stop - by, c.text))
    end
    for (i, nar) in enumerate(seq.narration)
        nar.at >= from || continue
        fresh = Narration(nar.text, nar.at - by, nar.voice)
        append!(fresh.samples, nar.samples)
        fresh.rate = nar.rate
        seq.narration[i] = fresh
    end
    return seq
end

"Delete `clip` by identity (track-safe where a frame is ambiguous), with the same
ripple semantics."
function deleteclip!(seq::Sequence, clip::Clip; ripple::Bool = true)
    i = findfirst(c -> c === clip, seq.clips)
    i === nothing && return nothing
    removeclip!(seq, i)   # …and with it what the timeline drew it with
    dropcards!(clip)      # it is out of the document; its panel row goes with it
    if ripple
        len = cliplength(clip)
        for other in seq.clips
            other.start >= clip.start && (other.start -= len)
        end
        # …and the timings that are not on a clip. See `rippledoc!`.
        seq.framerate > 0 &&
            rippledoc!(seq, clip.start / seq.framerate, len / seq.framerate)
    end
    return clip
end

"""
    withfields(clip; kw...) -> Clip

`clip` with named fields replaced, everything else shared.

Adding a field to [`Clip`](@ref) otherwise means finding every positional
construction of it — there were five across two files, and the ones a test does
not reach fail at runtime, in the render path, as a `MethodError` about argument
counts. This is the same shape as Mantle's `DeviceCaps(c; kw...)` and exists for
the same reason.

Shares the effect vector by default. Callers that hand the copy somewhere it may
be mutated pass their own — `snapshot` copies the slots, so an undo step cannot be
written into by a later edit.
"""
withfields(clip::Clip;
           id = clip.id, source = clip.source, src_in = clip.src_in,
           src_out = clip.src_out, start = clip.start, crop = clip.crop,
           effects = clip.effects, colortrack = clip.colortrack,
           motiontrack = clip.motiontrack, mattetrack = clip.mattetrack,
           depthtrack = clip.depthtrack, look = clip.look,
           timeinterp = clip.timeinterp, restorecache = clip.restorecache,
           track = clip.track,
           rate = clip.rate, bake = clip.bake) =
    # Never the graph: a derived clip may carry a different effect stack
    # (`withoutmatte`), and a compiled chain reached through two clips would be one
    # clip's `update!` writing into the other's passes. Never the cards either —
    # they are built for the clip the panel is showing, and a copy is not it —
    # and never the sequence: a copy is in none until `addclip!` puts it in one,
    # and a snapshot that claimed membership would reach a live editor from inside
    # the undo stack. Never the timeline plot either, for the reason the cards are
    # not copied: it is drawn for the clip that is in the sequence, and a copy is
    # not it.
    Clip(id, source, src_in, src_out, start, crop, effects, colortrack, motiontrack,
         mattetrack, depthtrack, look, timeinterp, restorecache, track,
         rate, nothing, bake, nothing, nothing, nothing)

"Copy of the edit state for undo/redo. Sources and analysis tracks are shared."
snapshot(seq::Sequence) =
    [withfields(c;
                effects = [copy(fx) for fx in c.effects])
     for c in seq.clips]

"""
    restoreinto!(into, from) -> into

Overwrite `into`'s edit state with `from`'s, keeping the object.

Matching is by what the document already identifies things with: clips and
effects by `id`, parameters by `name` within their effect. Anything without a
counterpart is built fresh; anything the snapshot no longer lists is dropped.

Identity is what the editor's bindings are — a slider holds its `Param`, a
keyframe lane its curve, a card its clip. Replacing the objects (which is what
`restore!` used to do) left all 110 rows of the lego project's panel pointing at
parameters that were no longer in the sequence.
"""
function restoreinto!(into::Clip, from::Clip)
    # NAMED, not derived by excluding what must not travel. A list built as
    # "every non-const field except these" includes anything added later by
    # default, so a field like `sequence` or `cardlayout` joins the undo stack
    # because nobody objected. Written out, a new field is simply not restored
    # until somebody says it should be — the mistake falls the harmless way, and
    # it is the same reason `to_msgpack` names an `Effect`'s fields and has
    # therefore never written a widget into a project file.
    into.src_in = from.src_in
    into.src_out = from.src_out
    into.start = from.start
    into.crop = from.crop
    into.colortrack = from.colortrack
    into.motiontrack = from.motiontrack
    into.mattetrack = from.mattetrack
    into.depthtrack = from.depthtrack
    into.look = from.look
    into.timeinterp = from.timeinterp
    into.restorecache = from.restorecache
    into.track = from.track
    into.rate = from.rate
    into.bake = from.bake
    into.id = from.id
    # …and not `graph` (the compiled chain, rebuilt on the next render),
    # `cardlayout`, `sequence` or `view` — the clip keeps the ones it is open with.
    into.graph = nothing
    old = Dict{UInt64, Effect}(fx.id => fx for fx in into.effects)
    empty!(into.effects)
    # One door, here as everywhere: an effect gets onto a clip through
    # `addslot!`, which is what makes sure it has a card. This was a bare `push!`,
    # and undoing the removal of an effect put it back on the clip with no row in
    # the panel — the panel does not diff, so nothing ever built it, and only
    # selecting another clip and coming back drew it. An effect that kept its card
    # keeps it (`addslot!` builds only for one that has none), so a restore does
    # not churn the rows of everything it did not touch.
    for fx in from.effects
        cur = pop!(old, fx.id, nothing)
        addslot!(into, cur === nothing ? copy(fx) : restoreinto!(cur, fx))
    end
    # What the snapshot does not list is gone from the document, and its card goes
    # with it. Nothing else would ever delete it: the panel builds cards for the
    # effects a clip has and never diffs them against what is on screen.
    foreach(dropcard!, values(old))
    return into
end

function restoreinto!(into::Effect, from::Effect)
    into.enabled[] = from.enabled[]
    old = Dict{Symbol, Param}(p.name => p for p in into.params)
    empty!(into.params)
    newrows = false
    for p in from.params
        cur = pop!(old, p.name, nothing)
        # a parameter whose value type changed is a different parameter, its curve
        # being of the old type — copied rather than converted
        fresh = cur === nothing || typeof(cur) !== typeof(p)
        newrows |= fresh
        push!(into.params, fresh ? copyparam(p) : restoreinto!(cur, p))
    end
    # A parameter that came or went makes this a different card — the rows are
    # built with it, and a copied parameter has no view at all. Dropping it here
    # is what makes `addslot!` build a fresh one; the card is the effect's, so the
    # effect is where the call belongs, and `dropcard!` needs no panel to reach
    # (it asks each plot where it is drawn). A scene clip whose parameters appear
    # only once it has rendered is the case this is really for.
    (newrows || !isempty(old)) && dropcard!(into)
    return into
end

"""
Restore one parameter in place, through its Observables.

Written through rather than replaced, so the widgets holding this object keep
working — and, because the curve is an Observable, so that everything derived
from it redraws. An undo updates the lane and the ◆ because it writes the same
node an edit does; there is nothing else to tell.
"""
function restoreinto!(into::Param{T}, from::Param{T}) where {T}
    # …its own curve: the snapshot stays reusable, so a redo restores the same
    # keys again rather than the ones an edit has moved since.
    into.curve[] = AnimCurve(copy(from.curve[].keys), from.curve[].interp)
    into.visible[] = from.visible[]
    into.input = from.input === nothing ? nothing :
        ParamInput(from.input.op, from.input.inputs...)
    return into
end

"""
An independent `Param`, on the same terms as [`copy(::Effect)`](@ref): its own
curve, its own Observables, its own unresolved edge — the ids are the edge, and
the copy is about to be placed somewhere the next `bindinputs!` resolves them
against. No view: the widgets belong to the parameter that is on screen.
"""
copyparam(p::Param{T}) where {T} =
    Param{T}(p.name, p.label,
             Observable(AnimCurve(copy(p.curve[].keys), p.curve[].interp)),
             Observable(p.visible[]), p.range,
             p.input === nothing ? nothing : ParamInput(p.input.op, p.input.inputs...),
             nothing)

"""
Restore a [`snapshot`](@ref) (the snapshot itself stays reusable).

Clips that are still there are restored in place (see [`restoreinto!`](@ref)), so
an undo does not invalidate what the editor is holding. Only what the snapshot
adds is built and only what it drops goes away. The edges are re-resolved either
way: a restored `input` is unresolved by construction.
"""
function restore!(seq::Sequence, snap::Vector{Clip})
    old = Dict{UInt64, Clip}(c.id => c for c in seq.clips)
    empty!(seq)
    for c in snap
        cur = pop!(old, c.id, nothing)
        addclip!(seq, cur === nothing ?
                 withfields(c; effects = [copy(fx) for fx in c.effects]) :
                 restoreinto!(cur, c))
    end
    # a clip the snapshot does not have is out of the document, and its cards go
    # with it — the same statement `restoreinto!` makes about an effect
    foreach(dropcards!, values(old))
    return bindinputs!(seq)
end

"Timeline frames of all clip edges (starts and ends), for snapping and drawing."
function clipedges(seq::Sequence)
    edges = Int[]
    for clip in seq.clips
        push!(edges, clip.start, clipend(clip))
    end
    return sort!(unique!(edges))
end

"""
    snappedstart(newstart, len, snap, targets) -> (start, didsnap)

Snap a clip of length `len` so either edge aligns to a target frame within
`snap` frames; clamps to ≥ 0.
"""
function snappedstart(newstart::Integer, len::Integer, snap::Integer, targets::Vector{Int})
    best, bestdist, didsnap = newstart, snap + 1, false
    for target in targets
        for cand in (target, target - len)  # snap left or right clip edge
            dist = abs(cand - newstart)
            dist < bestdist && ((best, bestdist, didsnap) = (cand, dist, true))
        end
    end
    return max(best, 0), didsnap
end

"""
First track at or above `want` where `[at, at + len)` is free, so dropping onto an
occupied spot stacks the clip on the lane above rather than failing. A new top
track always fits, so the result is at most `ntracks + 1`.
"""
function freetrack(seq::Sequence, at::Integer, len::Integer, want::Integer)
    for tr in max(Int(want), 1):(ntracks(seq) + 1)
        any(c -> c.track == tr && at < clipend(c) && at + len > c.start, seq.clips) ||
            return tr
    end
    return ntracks(seq) + 1
end

"""
    pushtracksup!(seq) -> seq

Make room for a track underneath: every existing clip moves up one lane, leaving
lane 1 free for the clip about to land there.
"""
function pushtracksup!(seq::Sequence)
    for c in seq.clips
        c.track += 1
    end
    return seq
end

"""
    compacttracks!(seq) -> seq

Close gaps in the lane numbering, keeping the order. Inserting a track underneath
moves everything up, and if the clip that moved down was the only one on the old
bottom lane, that lane is left empty in the middle of the stack.
"""
function compacttracks!(seq::Sequence)
    used = sort!(unique(c.track for c in seq.clips))
    rank = Dict(t => i for (i, t) in enumerate(used))
    for c in seq.clips
        c.track = rank[c.track]
    end
    return seq
end

"Whether `clip` can sit at `newstart` without overlapping another clip."
function canplace(seq::Sequence, clip::Clip, newstart::Integer, track::Integer = clip.track)
    len = cliplength(clip)
    for other in seq.clips
        other === clip && continue
        other.track == track || continue          # different tracks may overlap in time
        newstart < clipend(other) && other.start < newstart + len && return false
    end
    return true
end

"""
    moveclip!(seq, clip, newstart; snap=0, snaptargets=Int[]) -> Bool

Move `clip` so it starts at `newstart` (frames), snapping either clip edge
to `snaptargets` within `snap` frames. Returns `false` (no move) if the
new position would overlap another clip.
"""
function moveclip!(seq::Sequence, clip::Clip, newstart::Integer;
                   snap::Integer = 0, snaptargets::Vector{Int} = Int[])
    if snap > 0
        newstart, _ = snappedstart(newstart, cliplength(clip), snap, snaptargets)
    end
    newstart = max(newstart, 0)
    canplace(seq, clip, newstart) || return false
    clip.start = newstart
    sort!(seq.clips, by = c -> c.start)
    return true
end
