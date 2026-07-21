"""
Per-source-frame color stabilization corrections (see `analyzecolor!`).
Keyed by absolute source frame via `src_in`, so tracks survive clip splits.
"""
struct ColorTrack
    gains::Vector{Vec3f}
    offsets::Vector{Vec3f}
    src_in::Int
end

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
    Clip(source; src_in=0, src_out=source.nframes, start=0)

A non-destructive reference into a source video: frames
`src_in:(src_out-1)` of `source`, placed at timeline frame `start`.
`crop` is a normalized `(x, y, w, h)` rect ((0,0,1,1) = full frame,
y measured from the top).
"""
mutable struct Clip
    const source::VideoSource
    src_in::Int
    src_out::Int
    start::Int
    crop::NTuple{4, Float64}
    const effects::Vector{Any}  # ordered Effect stack (see effects.jl)
    colortrack::Union{Nothing, ColorTrack}
    motiontrack::Union{Nothing, MotionTrack}
    const animations::Dict{Symbol, AnimCurve}  # keyframed params (see keyframes.jl)
    track::Int                  # stacking layer; higher = on top (1 = base)
end

Clip(source::VideoSource, src_in, src_out, start, crop) =
    Clip(source, src_in, src_out, start, crop, [], nothing, nothing, Dict{Symbol, AnimCurve}(), 1)

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
    append!(right.effects, clip.effects)  # elements are immutable, sharing is safe
    right.colortrack = clip.colortrack    # keyed by absolute source frame, still valid
    right.motiontrack = clip.motiontrack
    merge!(right.animations, clip.animations)  # curves too are absolute-frame keyed
    clip.src_out = clip.src_in + offset
    insert!(seq.clips, i + 1, right)
    return right
end

"""
    deleteclip!(seq, n; ripple=true) -> Union{Clip, Nothing}

Delete the clip containing timeline frame `n`. With `ripple`, later clips
shift left to close the gap.
"""
function deleteclip!(seq::Sequence, n::Integer; ripple::Bool = true)
    i = clipat(seq, n)
    i === nothing && return nothing
    clip = seq.clips[i]
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
    [Clip(c.source, c.src_in, c.src_out, c.start, c.crop, copy(c.effects),
          c.colortrack, c.motiontrack, deepcopy(c.animations), c.track) for c in seq.clips]

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

"Whether `clip` can sit at `newstart` without overlapping another clip."
function canplace(seq::Sequence, clip::Clip, newstart::Integer)
    len = cliplength(clip)
    for other in seq.clips
        other === clip && continue
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
