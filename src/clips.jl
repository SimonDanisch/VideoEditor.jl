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
Source of stable identities for clips and effect slots. Position in a vector and
`objectid` both die on the first sort, undo or project reload — anything that has
to POINT at a clip or an effect (a blend at its partner, the inspector at a stack
entry, MCP at either) needs an id that survives those.
"""
const NEXTID = Threads.Atomic{UInt64}(0)
freshid() = UInt64(Threads.atomic_add!(NEXTID, UInt64(1)) + 1)

"""
One entry in a clip's effect stack: the effect, a STABLE `id` so anything can
point at exactly this entry (the blend card, the inspector, MCP), and `enabled` —
the honest form of what wrapping an effect in `Bypassed` used to express. Several
entries of the same kind may coexist; they are told apart by their id.
"""
mutable struct FxSlot
    const id::UInt64
    effect::Any        # an Effect — effects.jl is included after this file
    enabled::Bool
end
FxSlot(effect; enabled::Bool = true) = FxSlot(freshid(), effect, enabled)

"""
    Clip(source; src_in=0, src_out=source.nframes, start=0)

A non-destructive reference into a source video: frames
`src_in:(src_out-1)` of `source`, placed at timeline frame `start`.
`crop` is a normalized `(x, y, w, h)` rect ((0,0,1,1) = full frame,
y measured from the top).
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
    const animations::Dict{Symbol, AnimCurve}  # keyframed params (see keyframes.jl)
    track::Int                  # stacking layer; higher = on top (1 = base)
    blendfrom::UInt64           # clip this one blends away FROM (0 = nothing)
end

Clip(source::VideoSource, src_in, src_out, start, crop) =
    Clip(freshid(), source, src_in, src_out, start, crop, FxSlot[], nothing, nothing,
         nothing, Dict{Symbol, AnimCurve}(), 1, UInt64(0))

function Clip(source::VideoSource; src_in::Integer = 0, src_out::Integer = source.nframes,
              start::Integer = 0)
    return Clip(source, src_in, src_out, start, (0.0, 0.0, 1.0, 1.0))
end

cliplength(clip::Clip) = clip.src_out - clip.src_in
clipend(clip::Clip) = clip.start + cliplength(clip)
hascrop(clip::Clip) = clip.crop != (0.0, 0.0, 1.0, 1.0)

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
end

Sequence(clips::Vector{Clip}, framerate::Real) = Sequence(clips, framerate, Transition[])
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
    return (clip, clip.src_in + (n - clip.start))
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
    srcA = clamp(left.src_in + (n - left.start), 0, left.source.nframes - 1)
    srcB = clamp(right.src_in + (n - right.start), 0, right.source.nframes - 1)
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
function split!(seq::Sequence, n::Integer)
    i = clipat(seq, n)
    i === nothing && return nothing
    clip = seq.clips[i]
    n == clip.start && return nothing
    offset = n - clip.start
    right = Clip(clip.source, clip.src_in + offset, clip.src_out, n, clip.crop)
    right.track = clip.track              # both halves stay on the same stacking layer
    right.blendfrom = clip.blendfrom
    # each half owns its stack: same effects, own slot ids, so the inspector and
    # the blend card can address one half's entry without touching the other's
    append!(right.effects, [FxSlot(s.effect; enabled = s.enabled) for s in clip.effects])
    right.colortrack = clip.colortrack    # keyed by absolute source frame, still valid
    right.motiontrack = clip.motiontrack
    for (key, curve) in clip.animations   # absolute-frame keyed, but each half gets
        right.animations[key] = AnimCurve(copy(curve.keys), curve.interp)
    end                                   # its OWN copy — halves must edit independently
    clip.src_out = clip.src_in + offset
    insert!(seq.clips, i + 1, right)
    return right
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
    end
    return clip
end

"Copy of the edit state for undo/redo. Sources and analysis tracks are shared."
snapshot(seq::Sequence) =
    [Clip(c.id, c.source, c.src_in, c.src_out, c.start, c.crop,
          [FxSlot(s.id, s.effect, s.enabled) for s in c.effects],
          c.colortrack, c.motiontrack, c.mattetrack, deepcopy(c.animations), c.track,
          c.blendfrom)
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
