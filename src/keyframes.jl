"""
Keyframed animation for a single clip parameter. Control points are `(frame,
value)` pairs sorted by ABSOLUTE source frame — the same keying as `ColorTrack`
/`MotionTrack`, so a curve survives clip splits — interpolated between keys and
held flat past the first/last one. Each key carries its own `ease`, Premiere's
per-keyframe temporal interpolation:

  - `:linear` — a corner: constant velocity into and out of the key
  - `:smooth` — a flat tangent: the value eases in AND out at the key
  - `:hold`   — a step: the value freezes until the next key

The engine is parameter-agnostic: what a key's `value` means is decided
entirely by the matching [`ParamSpec`](@ref).
"""
struct Keyframe
    frame::Int
    value::Float64
    ease::Symbol     # :linear · :smooth · :hold
end
Keyframe(frame::Integer, value::Real) = Keyframe(frame, value, :linear)

mutable struct AnimCurve
    const keys::Vector{Keyframe}
    interp::Symbol   # legacy curve-wide default: :smooth eases EVERY key (old projects)
end
AnimCurve() = AnimCurve(Keyframe[], :linear)

Base.isempty(c::AnimCurve) = isempty(c.keys)
Base.length(c::AnimCurve) = length(c.keys)

"Effective ease of key `k` on curve `c` — the legacy curve-wide `:smooth` still
eases every key, a per-key mode wins otherwise."
keyease(c::AnimCurve, k::Keyframe) =
    k.ease === :linear && c.interp === :smooth ? :smooth : k.ease

"Interpolated value at absolute source frame `f`, or `nothing` if the curve is empty."
function valueat(c::AnimCurve, f::Real)
    ks = c.keys
    isempty(ks) && return nothing
    f <= ks[1].frame && return ks[1].value
    f >= ks[end].frame && return ks[end].value
    i = findlast(k -> k.frame <= f, ks)::Int
    a, b = ks[i], ks[i + 1]
    b.frame == a.frame && return b.value
    keyease(c, a) === :hold && return a.value            # step until the next key
    t = (f - a.frame) / (b.frame - a.frame)
    # cubic hermite with unit (linear) or flat (eased) end tangents: m==1 at a
    # linear corner, m==0 at a smooth key — :smooth/:smooth is exactly smoothstep
    m0 = keyease(c, a) === :smooth ? 0.0 : 1.0
    m1 = keyease(c, b) === :smooth ? 0.0 : 1.0
    h = (t^3 - 2t^2 + t) * m0 + (-2t^3 + 3t^2) + (t^3 - t^2) * m1
    return a.value + h * (b.value - a.value)
end

"Insert or replace the key at `frame` (keeps `keys` sorted and the replaced
key's ease). Returns the curve."
function setkey!(c::AnimCurve, frame::Integer, value::Real)
    i = findfirst(k -> k.frame == frame, c.keys)
    if i === nothing
        j = findfirst(k -> k.frame > frame, c.keys)
        insert!(c.keys, j === nothing ? length(c.keys) + 1 : j, Keyframe(frame, value))
    else
        c.keys[i] = Keyframe(frame, value, c.keys[i].ease)
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

"Move the key at index `i` to `(frame, value)` — its ease travels with it —
and re-sort. Returns the curve."
function movekey!(c::AnimCurve, i::Integer, frame::Integer, value::Real)
    c.keys[i] = Keyframe(frame, value, c.keys[i].ease)
    sort!(c.keys; by = k -> k.frame)
    return c
end

"Insert or replace the key at `frame` with an explicit ease mode."
function setkey!(c::AnimCurve, frame::Integer, value::Real, ease::Symbol)
    setkey!(c, frame, value)
    i = findfirst(k -> k.frame == frame, c.keys)::Int
    c.keys[i] = Keyframe(frame, value, ease)
    return c
end

"Set the ease mode of the key at index `i` (`:linear` · `:smooth` · `:hold`)."
function setease!(c::AnimCurve, i::Integer, mode::Symbol)
    k = c.keys[i]
    c.keys[i] = Keyframe(k.frame, k.value, mode)
    return c
end

"Bake the legacy curve-wide `:smooth` into per-key eases so per-key edits can
take effect (a `:linear` marked key would otherwise stay overridden)."
function materializeease!(c::AnimCurve)
    c.interp === :smooth || return c
    for (i, k) in enumerate(c.keys)
        k.ease === :linear && (c.keys[i] = Keyframe(k.frame, k.value, :smooth))
    end
    c.interp = :linear
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
