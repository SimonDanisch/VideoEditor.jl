"""
Per-source-frame color stabilization corrections (see `analyzecolor!`).
Keyed by absolute source frame via `src_in`, so tracks survive clip splits.
`strength` scales the correction toward identity (1 = full fix, 0 = off) at
APPLY time — adjustable live, no re-analysis needed.
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
Per-source-frame DEPTH, keyed absolutely like the other tracks.

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
reason `FxLink` is: `Clip` has a field of it. Unlike a track it is a CACHE — a
clip's worth of 4x frames is far too much to keep or to save — so `order` is an
insertion queue that evicts the oldest window past `limit`. A plain LRU would be
better if playback ever ran backwards, which it does not.
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
`objectid` both die on the first sort, undo or project reload — anything that has
to POINT at a clip or an effect (a blend at its partner, the inspector at a stack
entry, MCP at either) needs an id that survives those.
"""
const NEXTID = Threads.Atomic{UInt64}(0)
freshid() = UInt64(Threads.atomic_add!(NEXTID, UInt64(1)) + 1)

"""
A reference from one effect slot to another, possibly on another clip — see
links.jl for what they mean and how they are followed. The type lives here
because `FxSlot` has a field of it.
"""
struct FxLink
    clip::UInt64      # target clip id; 0 = the same clip
    slot::UInt64      # target slot id
    role::Symbol      # what the target IS to the parent ("fades into", …)
end

"""
One entry in a clip's effect stack: the effect, a STABLE `id` so anything can
point at exactly this entry (a linked card, the panel, MCP), `enabled` — the
honest form of what wrapping an effect in `Bypassed` used to express — any
`links` to other slots, and the curves of ITS OWN parameters. Several entries of
the same kind may coexist; they are told apart by their id.

THE CURVES BELONG HERE, not on the clip — and that last sentence is why. `Clip`
used to carry one flat
`Dict{Symbol, AnimCurve}` for everything on it, which meant a curve knew only a
bare name and not which effect it animated. Everything else followed from that:
a global `key -> ParamSpec` index had to exist to make the name resolvable
again, `registerparams!` had to fill it at registration, and resolution went
through `findeffect`, which returns the FIRST effect of a kind.

Measured on two Blur slots with one keyframe of 12 on `:blur`:
`[(blur = 12.0,), (blur = 0.0,)]` — the first slot animates, the second silently
keeps its static value, while each card's SLIDER writes to its own slot
correctly. The slider and the diamond of the same row pointed at different
objects.

A curve keyed by the effect's own parameter name on the slot that owns it cannot
be ambiguous, so none of that machinery is needed: the slot has the effect, the
effect's kind has the parameters, and the parameter has its curve. No global is
involved at any step.
"""
mutable struct FxSlot
    const id::UInt64
    effect::Any        # an Effect — effects.jl is included after this file
    enabled::Bool
    const links::Vector{FxLink}
    # parameter name (as the effect's own kind names it) -> its curve
    const animations::Dict{Symbol, AnimCurve}
end
FxSlot(effect; enabled::Bool = true) =
    FxSlot(freshid(), effect, enabled, FxLink[], Dict{Symbol, AnimCurve}())
FxSlot(id::Integer, effect, enabled::Bool) =
    FxSlot(UInt64(id), effect, enabled, FxLink[], Dict{Symbol, AnimCurve}())
FxSlot(id::Integer, effect, enabled::Bool, links::Vector{FxLink}) =
    FxSlot(UInt64(id), effect, enabled, links, Dict{Symbol, AnimCurve}())

"""
    Clip(source; src_in=0, src_out=source.nframes, start=0)

A non-destructive reference into a source video: frames
`src_in:(src_out-1)` of `source`, placed at timeline frame `start`.
`crop` is a normalized `(x, y, w, h)` rect ((0,0,1,1) = full frame,
y measured from the top).

`rate` is how many SOURCE frames one TIMELINE frame advances — the conform
factor for a source that doesn't run at the sequence rate (a 30 fps clip in a
60 fps timeline has `rate = 0.5` and shows each of its frames twice). It is
`source.framerate / seq.framerate`, set once when the clip is placed, and it is
the ONLY place the two frame worlds differ: `src_in`/`src_out` count source
frames, `start`/[`cliplength`](@ref) count timeline frames, and
[`sourceframe`](@ref) is the one conversion between them.

Conforming preserves WALL-CLOCK duration (`srclength/source.framerate` seconds
either way), which is why everything that works in time — audio, the filmstrip,
thumbnails — needs no rate at all.
"""
mutable struct Clip
    id::UInt64                  # stable across sorting, undo and save/load
                                # (settable so a project file restores its own)
    const source::VideoSource
    src_in::Int
    src_out::Int
    start::Int
    crop::NTuple{4, Float64}
    const effects::Vector{FxSlot}  # ordered effect stack (see effects.jl)
    colortrack::Union{Nothing, ColorTrack}
    motiontrack::Union{Nothing, MotionTrack}
    mattetrack::Union{Nothing, MatteTrack}
    # Estimated depth, `nothing` until something asks for it. A field beside the
    # other tracks rather than a global keyed by clip id — that is what made
    # `split!` drop a clip's restoration silently, since a fresh id follows
    # nothing (see `restorecache` below).
    depthtrack::Union{Nothing, DepthTrack}
    # A learned colour grade as a (D,D,D,3) table, host-side. Per CLIP, not per
    # frame: a look that drifted within a shot would be a fault, not a feature
    # (see `look.jl`). Host-side because it is project-file data and because a
    # device array would be on the wrong device after `autodetectgpu!`.
    look::Union{Nothing, Array{Float32, 4}}
    # How this clip fills timeline frames its source has no frame for — i.e. what
    # a slowed clip does between source frames. `:sample` repeats the nearest
    # (the default, and what every editor does with no model); `:flow`
    # synthesizes the in-between frame (see `flow.jl`). A property of the clip
    # like `rate`, not an effect: it decides what the SOURCE is, before any
    # effect runs, and an effect only ever sees one frame.
    timeinterp::Symbol
    # The fourth analysis result, and a field like the other three. It used to be
    # a module global keyed by clip id, which is how `split!` came to drop a
    # clip's restoration silently while taking explicit care of its matte: an id
    # is minted fresh for the right half, so nothing followed. `nothing` until a
    # window is restored; a cache, so it is never written to a project file.
    restorecache::Union{Nothing, RestoreCache}
    const animations::Dict{Symbol, AnimCurve}  # keyframed params (see keyframes.jl)
    track::Int                  # stacking layer; higher = on top (1 = base)
    blendfrom::UInt64           # clip this one blends away FROM (0 = nothing)
    rate::Float64               # source frames per timeline frame (1 = native)
end

Clip(source::VideoSource, src_in, src_out, start, crop, rate::Real = 1.0,
     reframe::Union{Nothing, NTuple{<:Any, <:Real}} = nothing) =
    withreframe!(Clip(freshid(), source, src_in, src_out, start, crop, FxSlot[],
                      # colortrack, motiontrack, mattetrack, depthtrack, look
                      nothing, nothing, nothing, nothing, nothing,
                      :sample,          # timeinterp
                      nothing,          # restorecache
                      Dict{Symbol, AnimCurve}(), 1, UInt64(0), Float64(rate)),
                 reframe)

function Clip(source::VideoSource; src_in::Integer = 0, src_out::Integer = source.nframes,
              start::Integer = 0, rate::Real = 1.0)
    return Clip(source, src_in, src_out, start, (0.0, 0.0, 1.0, 1.0), rate)
end

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
conformrate(source::VideoSource, framerate::Real) =
    isapprox(source.framerate, framerate; atol = 0.01) ? 1.0 :
    Float64(source.framerate) / Float64(framerate)

"Frames of SOURCE this clip spans — what every per-source-frame analysis iterates."
srclength(clip::Clip) = clip.src_out - clip.src_in

"""
Frames of TIMELINE this clip occupies — its extent in the edit. Equal to
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

The source frame showing at timeline frame `n` — the ONE conversion between the
edit's frame world and the media's. Not clamped: transition handles deliberately
ask past a clip's own range.
"""
sourceframe(clip::Clip, n::Integer) =
    clip.rate == 1.0 ? clip.src_in + (Int(n) - clip.start) :
    clip.src_in + floor(Int, (Int(n) - clip.start) * clip.rate)

"""
    sourcephase(clip, n) -> Float64

How far timeline frame `n` sits BETWEEN [`sourceframe`](@ref)`(clip, n)` and the
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
mutable struct Sequence
    const clips::Vector{Clip}
    framerate::Float64
    const transitions::Vector{Transition}
    # Plots drawn over the finished canvas (see overlays.jl). They belong to the
    # SEQUENCE, not to a clip: an overlay sits above the composite, so a cut
    # underneath it changes nothing about where or when it is drawn.
    const overlays::Vector{Overlay}
    # What is spoken, and when. An EDIT like the overlays above — the transcript
    # is what a user fixes when the model mishears a word, so it is saved with the
    # project and restored by undo. Regenerating it is minutes of Whisper, which
    # is the other half of why it is not a cache.
    const captions::Vector{Caption}
    # Spoken narration mixed OVER the clips, on both the preview and the export.
    # An edit like the captions: the words are saved, the samples are a cache.
    const narration::Vector{Narration}
    # The output resolution, once something has set it — the crop tool does.
    #
    # `nothing` means "derive it from the first clip", which is what this did
    # ALWAYS and is a trap: deleting or reordering clips then silently changes the
    # project's resolution, and cropping clip 2 resized nothing while cropping
    # clip 1 resized everything. Kept as the fallback so projects written before
    # this open unchanged.
    canvas::Union{Nothing, Tuple{Int, Int}}
end

Sequence(clips::Vector{Clip}, framerate::Real) =
    Sequence(clips, framerate, Transition[], Overlay[], Caption[], Narration[], nothing)
Sequence(clips::Vector{Clip}, framerate::Real, transitions::Vector{Transition}) =
    Sequence(clips, framerate, transitions, Overlay[], Caption[], Narration[], nothing)
Sequence(clips::Vector{Clip}, framerate::Real, transitions::Vector{Transition},
         overlays::Vector{Overlay}) =
    Sequence(clips, framerate, transitions, overlays, Caption[], Narration[], nothing)
Sequence(source::VideoSource) = Sequence([Clip(source)], source.framerate)

"The clip with `id`, or `nothing` — how anything refers to a clip across sorting,
undo and reloads (indices shift, `objectid` dies on the first copy)."
function clipbyid(seq::Sequence, id::Integer)
    i = findfirst(c -> c.id == id, seq.clips)
    return i === nothing ? nothing : seq.clips[i]
end

seqlength(seq::Sequence) = maximum(clipend, seq.clips; init = 0)
seqduration(seq::Sequence) = seqlength(seq) / seq.framerate

"Number of stacking layers (1-based; the base track is 1)."
ntracks(seq::Sequence) = isempty(seq.clips) ? 1 : maximum(c.track for c in seq.clips)

"Every clip covering timeline frame `n`, bottom track first (base → top)."
clipsat(seq::Sequence, n::Integer) =
    sort!([c for c in seq.clips if c.start <= n < clipend(c)]; by = c -> c.track)

"Index of the TOP-most clip containing timeline frame `n`, or `nothing` (gap)."
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
Index of the clip covering frame `n` ON `track` — the lane-aware `clipat`. The
topmost clip is what the preview renders, but a click (and the inspector behind
it) must be able to reach the one stacked BELOW it.
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
The dissolve length a one-click blend should use on this cut: 0.6 s, but never
more than HALF the shorter clip, so each side keeps three quarters of itself
un-blended. The hard limit ([`clamptransition`](@ref)) allows a dissolve twice
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
        return t
    end
    seq.transitions[i].kind = kind
    seq.transitions[i].duration = dur
    return seq.transitions[i]
end

"Remove the transition on the cut at `at` (returns it, or `nothing`)."
function removetransition!(seq::Sequence, at::Integer)
    i = findfirst(t -> t.at == at, seq.transitions)
    i === nothing && return nothing
    t = seq.transitions[i]
    deleteat!(seq.transitions, i)
    return t
end

"Drop transitions whose cut no longer exists (after edits that move/merge clips)."
function prunetransitions!(seq::Sequence)
    filter!(seq.transitions) do t
        left, right = transitionclips(seq, t.at)
        left !== nothing && right !== nothing
    end
    return seq
end

"""
    split!(seq, n) -> Union{Clip, Nothing}

Split the clip containing timeline frame `n` at `n`; the right half is
returned. No-op at a clip start or in a gap.
"""
function split!(seq::Sequence, n::Integer, track::Union{Nothing, Integer} = nothing)
    # `track` names the LANE to cut. Without it `clipat` answers with the topmost
    # clip at `n`, so pressing S while V2 was selected cut V3 — the selection was
    # never consulted at all.
    i = track === nothing ? clipat(seq, n) : clipat(seq, n, Int(track))
    i === nothing && return nothing
    clip = seq.clips[i]
    n == clip.start && return nothing
    cut = sourceframe(clip, n)            # the cut in SOURCE frames — both halves share it
    cut > clip.src_in || return nothing
    # …and the halves must MEET: on a conformed clip several timeline frames show
    # the same source frame, so the cut is snapped back to where that frame starts.
    # Cutting at the raw `n` left the left half one frame short of the right one —
    # a hole in the timeline that only appears on retimed material.
    at = clip.start + timelineframes(clip, cut - clip.src_in)
    right = Clip(clip.source, cut, clip.src_out, at, clip.crop, clip.rate)
    right.track = clip.track              # both halves stay on the same stacking layer
    right.blendfrom = clip.blendfrom
    # each half owns its stack: same effects, own slot ids, so the inspector and
    # the blend card can address one half's entry without touching the other's
    # each half owns its stack, with its own slot ids — and its own copy of the
    # links, so cutting a blended clip does not give two slots the same partner
    append!(right.effects, [FxSlot(freshid(), s.effect, s.enabled, copy(s.links))
                            for s in clip.effects])
    # keyed by absolute source frame, so both halves stay valid. Assigned
    # directly: the stack was already copied above, slots and all, so going
    # through `setmotiontrack!` would prepend a SECOND Stabilize slot.
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
    for (key, curve) in clip.animations   # absolute-frame keyed, but each half gets
        right.animations[key] = AnimCurve(copy(curve.keys), curve.interp)
    end                                   # its OWN copy — halves must edit independently
    clip.src_out = cut
    insert!(seq.clips, i + 1, right)
    return right
end

"""
    copyclip(clip; start = clip.start, track = clip.track) -> Clip

An independent copy of `clip` — the same source range, the same effect stack, the
same analysis — placed at `start` on `track`. Not inserted into any sequence.

**What is copied and what is shared follows [`split!`](@ref) exactly**, because
the question is the same one: a derived clip reads the same source frames, so
anything keyed by absolute source frame is correct to share and expensive to
duplicate.

* **Fresh:** the clip `id`, and one `id` per [`FxSlot`](@ref) with its own copy of
  the links. Two slots sharing an id would make the inspector and the blend card
  address both at once. The `Effect` inside a slot is shared and that is safe —
  every effect is an immutable `struct`, so a parameter change replaces it rather
  than mutating what the other clip reads.
* **Shared:** `colortrack`, `motiontrack`, `mattetrack` and `restorecache`. All
  four are keyed by absolute source frame and the copy covers the same frames, so
  one analysis indexes correctly from both — and re-running a matte to duplicate
  it would cost minutes.
* **Copied:** the animation curves. Keyframes are the one thing you edit per
  clip, so the two must move independently.
* **Dropped:** `blendfrom`. It names another clip by id, and a copy landing
  somewhere else in the timeline has no business blending away from that clip's
  partner. `split!` keeps it because its left half genuinely continues the same
  blend; a copy does not.
"""
copyclip(clip::Clip; start::Integer = clip.start, track::Integer = clip.track) =
    withfields(clip; id = freshid(), start = Int(start), track = Int(track),
               # Fresh slot ids and copied link vectors: the copy's stack is its
               # own, so unlinking on one must not reach into the other.
               effects = [FxSlot(freshid(), s.effect, s.enabled, copy(s.links))
                          for s in clip.effects],
               animations = Dict{Symbol, AnimCurve}(
                   k => AnimCurve(copy(c.keys), c.interp) for (k, c) in clip.animations),
               # NOT the original's blend partner — a copy is not in that transition.
               blendfrom = UInt64(0))

"""
    trimclip!(seq, clip, i, side, n) -> clip

Move one edge of `clip` (`i` = its index in `seq.clips`) to timeline frame `n`.
The left edge shifts `start` and `src_in` together so the content stays anchored;
the right edge moves `src_out`. Clamped to the available source and to the
neighbours on the same lane.

The edge walks TIMELINE frames while the in/out points count SOURCE frames — on a
conformed clip those are not the same step, which is why this is one function
and not arithmetic inlined in the drag handler.
"""
function trimclip!(seq::Sequence, clip::Clip, i::Integer, side::Symbol, n::Integer)
    if side === :right
        maxend = clip.start + timelineframes(clip, clip.source.nframes - clip.src_in)
        nxt = nextontrack(seq, clip)
        nxt === nothing || (maxend = min(maxend, nxt.start))
        # `max(…, clip.start + 1)`: a clip is never trimmed out of existence, and
        # the bound it is clamped against must not invert. It did — the limit used
        # to come from `seq.clips[i + 1]`, and that list is sorted by (track,
        # start), so on a stack the "next clip" was usually one on ANOTHER track,
        # often starting earlier. `src_out` then landed at or before `src_in` and
        # the clip vanished mid-drag.
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

"The clip that follows `clip` ON ITS OWN TRACK, or `nothing`."
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

"The clip that precedes `clip` ON ITS OWN TRACK, or `nothing`."
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
    for (key, curve) in nxt.animations    # carry the right half's keys over
        if haskey(c.animations, key)
            foreach(k -> setkey!(c.animations[key], k.frame, k.value, k.ease), curve.keys)
        else
            c.animations[key] = curve
        end
    end
    c.src_out = nxt.src_out
    deleteat!(seq.clips, j)
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

The captions and the narration are pinned to the PICTURE, not to the wall clock.
`deleteclip!` rippled the clips and left these exactly where they were, so the
first ripple delete slid every subtitle and every voiceover after the cut out of
sync with the shot it belonged to — and cutting anything is mostly ripple
deletes, so this went wrong on essentially the first real edit.

Both are REPLACED rather than mutated. `docsnapshot` shares these objects with
every undo step holding them, so shifting one in place would rewrite the history
that is supposed to put it back. The narration's samples come along: the words
have not changed, only when they are said.
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

"Delete `clip` ITSELF (by identity — track-safe where a frame is ambiguous)
with the same ripple semantics."
function deleteclip!(seq::Sequence, clip::Clip; ripple::Bool = true)
    i = findfirst(c -> c === clip, seq.clips)
    i === nothing && return nothing
    deleteat!(seq.clips, i)
    if ripple
        len = cliplength(clip)
        for other in seq.clips
            other.start >= clip.start && (other.start -= len)
        end
        # …and the timings that are NOT on a clip. See `rippledoc!`.
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
be mutated pass their own (`snapshot` copies the slots, `effectiveclip` gives the
copy its own so a sampled parameter cannot write back).
"""
withfields(clip::Clip;
           id = clip.id, source = clip.source, src_in = clip.src_in,
           src_out = clip.src_out, start = clip.start, crop = clip.crop,
           effects = clip.effects, colortrack = clip.colortrack,
           motiontrack = clip.motiontrack, mattetrack = clip.mattetrack,
           depthtrack = clip.depthtrack, look = clip.look,
           timeinterp = clip.timeinterp, restorecache = clip.restorecache,
           animations = clip.animations, track = clip.track,
           blendfrom = clip.blendfrom, rate = clip.rate) =
    Clip(id, source, src_in, src_out, start, crop, effects, colortrack, motiontrack,
         mattetrack, depthtrack, look, timeinterp, restorecache, animations, track,
         blendfrom, rate)

"Copy of the edit state for undo/redo. Sources and analysis tracks are shared."
snapshot(seq::Sequence) =
    # links are copied, not shared: an undo that put back a slot whose link vector
    # was the live one would not restore a removed partner
    [withfields(c;
                effects = [FxSlot(s.id, s.effect, s.enabled, copy(s.links)) for s in c.effects],
                animations = deepcopy(c.animations))
     for c in seq.clips]

"Restore a [`snapshot`](@ref) (the snapshot itself stays reusable)."
function restore!(seq::Sequence, snap::Vector{Clip})
    empty!(seq.clips)
    append!(seq.clips, snapshot(Sequence(snap, seq.framerate)))
    return seq
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
First track at or above `want` where `[at, at + len)` is free — dropping onto an
occupied spot stacks the clip on the lane above instead of failing (a NEW top
track always fits, so this always returns ≤ `ntracks + 1`).
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

Make room for a track UNDERNEATH: every existing clip moves up one lane, so lane
1 is free for the clip that is about to land there.
"""
function pushtracksup!(seq::Sequence)
    for c in seq.clips
        c.track += 1
    end
    return seq
end

"""
    compacttracks!(seq) -> seq

Close gaps in the lane numbering, keeping the order. Inserting a track
underneath moves everything up, and if the clip that moved DOWN was the only one
on the old bottom lane, that lane is left empty — an empty lane in the middle of
a stack is a hole in the timeline nobody asked for.
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
