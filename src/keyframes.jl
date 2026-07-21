"""
Keyframed animation for a single clip parameter. Control points are `(frame,
value)` pairs sorted by ABSOLUTE source frame — the same keying as `ColorTrack`
/`MotionTrack`, so a curve survives clip splits — linearly interpolated and held
flat past the first/last key. The engine is parameter-agnostic: what a key's
`value` means is decided entirely by the matching [`ParamSpec`](@ref).
"""
struct Keyframe
    frame::Int
    value::Float64
end

mutable struct AnimCurve
    const keys::Vector{Keyframe}
    interp::Symbol   # :linear (constant velocity) or :smooth (ease in/out at each key)
end
AnimCurve() = AnimCurve(Keyframe[], :linear)

Base.isempty(c::AnimCurve) = isempty(c.keys)
Base.length(c::AnimCurve) = length(c.keys)

"Interpolated value at absolute source frame `f`, or `nothing` if the curve is empty."
function valueat(c::AnimCurve, f::Real)
    ks = c.keys
    isempty(ks) && return nothing
    f <= ks[1].frame && return ks[1].value
    f >= ks[end].frame && return ks[end].value
    i = findlast(k -> k.frame <= f, ks)::Int
    a, b = ks[i], ks[i + 1]
    b.frame == a.frame && return b.value
    t = (f - a.frame) / (b.frame - a.frame)
    c.interp === :smooth && (t = t * t * (3.0 - 2.0 * t))   # smoothstep ease
    return a.value + t * (b.value - a.value)
end

"Insert or replace the key at `frame` (keeps `keys` sorted). Returns the curve."
function setkey!(c::AnimCurve, frame::Integer, value::Real)
    i = findfirst(k -> k.frame == frame, c.keys)
    if i === nothing
        j = findfirst(k -> k.frame > frame, c.keys)
        insert!(c.keys, j === nothing ? length(c.keys) + 1 : j, Keyframe(frame, value))
    else
        c.keys[i] = Keyframe(frame, value)
    end
    return c
end

"Remove the key at `frame` if present. Returns whether one was removed."
function removekey!(c::AnimCurve, frame::Integer)
    i = findfirst(k -> k.frame == frame, c.keys)
    i === nothing && return false
    deleteat!(c.keys, i)
    return true
end

"Move the key at index `i` to `(frame, value)` and re-sort. Returns the curve."
function movekey!(c::AnimCurve, i::Integer, frame::Integer, value::Real)
    c.keys[i] = Keyframe(frame, value)
    sort!(c.keys; by = k -> k.frame)
    return c
end

"""
Descriptor for one animatable clip parameter — the ONLY place that knows how a
named parameter maps to clip state. `get(clip)` reads its current static value;
`set(clip, v)` writes `v` onto the clip. Everything else (keyframe storage, the
render override, the editor UI) enumerates [`PARAMS`](@ref) and works through
these accessors, so adding a new animatable parameter is a single table row.
"""
struct ParamSpec
    key::Symbol
    label::String
    group::Symbol
    lo::Float64
    hi::Float64
    default::Float64
    get::Function   # clip -> Float64
    set::Function   # (clip, value) -> nothing
end

"Fraction of the param's range (for the editor's normalized lanes), clamped to [0,1]."
paramnorm(p::ParamSpec, v::Real) = clamp((v - p.lo) / (p.hi - p.lo), 0.0, 1.0)
"Inverse of [`paramnorm`](@ref): a [0,1] lane fraction back to a parameter value."
paramdenorm(p::ParamSpec, u::Real) = p.lo + clamp(u, 0.0, 1.0) * (p.hi - p.lo)
