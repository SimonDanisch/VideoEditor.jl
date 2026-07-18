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
end

Clip(source::VideoSource, src_in, src_out, start, crop) =
    Clip(source, src_in, src_out, start, crop, [], nothing, nothing)

function Clip(source::VideoSource; src_in::Integer = 0, src_out::Integer = source.nframes,
              start::Integer = 0)
    return Clip(source, src_in, src_out, start, (0.0, 0.0, 1.0, 1.0))
end

cliplength(clip::Clip) = clip.src_out - clip.src_in
clipend(clip::Clip) = clip.start + cliplength(clip)
hascrop(clip::Clip) = clip.crop != (0.0, 0.0, 1.0, 1.0)

"""
    Sequence(source) / Sequence(clips, framerate)

An edited timeline: non-overlapping clips sorted by start. All edits are
metadata operations on this structure; frames are resolved on demand via
`locate`.
"""
mutable struct Sequence
    const clips::Vector{Clip}
    framerate::Float64
end

Sequence(source::VideoSource) = Sequence([Clip(source)], source.framerate)

seqlength(seq::Sequence) = maximum(clipend, seq.clips; init = 0)
seqduration(seq::Sequence) = seqlength(seq) / seq.framerate

"Index of the clip containing timeline frame `n`, or `nothing` (gap)."
clipat(seq::Sequence, n::Integer) = findfirst(c -> c.start <= n < clipend(c), seq.clips)

"Resolve timeline frame `n` to `(clip, source_frame)`, or `nothing` in a gap."
function locate(seq::Sequence, n::Integer)
    i = clipat(seq, n)
    i === nothing && return nothing
    clip = seq.clips[i]
    return (clip, clip.src_in + (n - clip.start))
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
          c.colortrack, c.motiontrack) for c in seq.clips]

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
