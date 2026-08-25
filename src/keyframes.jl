"""
Keyframed animation for a single clip parameter. Control points are `(frame,
value)` pairs sorted by ABSOLUTE source frame — the same keying as `ColorTrack`
/`MotionTrack`, so a curve survives clip splits — interpolated between keys and
held flat past the first/last one. Each key carries its own `ease`, Premiere's
per-keyframe temporal interpolation:

  - `:linear` — a corner: constant velocity into and out of the key
  - `:smooth` — a flat tangent: the value eases in AND out at the key
  - `:hold`   — a step: the value freezes until the next key

The engine is parameter-agnostic in the VALUE too: a key holds a `T`, and how
two of them blend is [`lerp`](@ref). That is what lets a rotation be a curve of
quaternions rather than three curves of Euler angles — interpolating a rotation
component-wise is simply wrong, and decomposing it here would make that the only
option.
"""
struct Keyframe{T}
    frame::Int
    value::T
    ease::Symbol     # :linear · :smooth · :hold
end
Keyframe(frame::Integer, value) = Keyframe(Int(frame), value, :linear)

mutable struct AnimCurve{T}
    const keys::Vector{Keyframe{T}}
    interp::Symbol   # curve-wide default: :smooth eases EVERY key
end
AnimCurve{T}() where {T} = AnimCurve{T}(Keyframe{T}[], :linear)
AnimCurve() = AnimCurve{Float64}()

"""
    lerp(a, b, t) -> typeof(a)

Blend two keyframe values. The ONLY thing [`valueat`](@ref) needs to know about
a parameter's type, so a new animatable type is one method and nothing else.
"""
lerp(a::Real, b::Real, t::Real) = a + t * (b - a)
lerp(a::AbstractVector, b::AbstractVector, t::Real) = a .+ t .* (b .- a)
lerp(a::Colorant, b::Colorant, t::Real) =
    Makie.lerp_oklab(RGBf(a), RGBf(b), Float32(t))

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
    # `h` is the eased position between the two keys; `lerp` is what the VALUE
    # type says blending means. `h` may leave [0,1] at a linear corner, which is
    # the overshoot a hermite is supposed to have — every `lerp` above is an
    # affine combination, so that carries through unchanged.
    return lerp(a.value, b.value, h)
end

"Insert or replace the key at `frame` (keeps `keys` sorted and the replaced
key's ease). Returns the curve."
function setkey!(c::AnimCurve{T}, frame::Integer, value) where {T}
    i = findfirst(k -> k.frame == frame, c.keys)
    if i === nothing
        j = findfirst(k -> k.frame > frame, c.keys)
        insert!(c.keys, j === nothing ? length(c.keys) + 1 : j,
                Keyframe(Int(frame), convert(T, value), :linear))
    else
        c.keys[i] = Keyframe(Int(frame), convert(T, value), c.keys[i].ease)
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
function movekey!(c::AnimCurve{T}, i::Integer, frame::Integer, value) where {T}
    c.keys[i] = Keyframe(Int(frame), convert(T, value), c.keys[i].ease)
    sort!(c.keys; by = k -> k.frame)
    return c
end

"Insert or replace the key at `frame` with an explicit ease mode."
function setkey!(c::AnimCurve{T}, frame::Integer, value, ease::Symbol) where {T}
    setkey!(c, frame, value)
    i = findfirst(k -> k.frame == frame, c.keys)::Int
    c.keys[i] = Keyframe(Int(frame), convert(T, value), ease)
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
One tunable parameter — a scalar with a slider range and a default. Effect
plugins ([`registerplugin!`](@ref)) and overlays ([`registeroverlay!`](@ref))
both declare their parameters with this, and both get keyframing for free
because a declared scalar is all the curve engine needs.
"""
struct FxParam
    name::Symbol
    label::String
    min::Float64
    max::Float64
    default::Float64
end
FxParam(name, label; min = 0.0, max = 1.0, default = min) =
    FxParam(Symbol(name), String(label), Float64(min), Float64(max), Float64(default))

"""
One parameter of one effect: what it is called, what it is now, and — if it is
animated — its curve.

THE VALUE AND THE CURVE LIVE TOGETHER, and that is the whole point. They used to
be a field on a typed effect struct and an entry in a flat `Dict{Symbol,
AnimCurve}` on the CLIP, joined by a name and a global index. A curve therefore
knew only a bare name, not which effect it animated, and resolution took the
first effect of a kind: two Blur entries with one keyframe of 12 rendered
`[(blur = 12.0,), (blur = 0.0,)]` — the second silently static, while its own
card's slider wrote to it correctly. Slider and diamond of one row pointed at
different objects. Holding the parameter itself makes that unsayable.

`visible` is its lane on the timeline, and it is INDEPENDENT of `curve`: showing
an empty lane is how you get somewhere to put the first key. Coupling the two —
which a registry of animated-parameters-only forces — means you must keyframe
something before you can see where its keyframes would go.

`T` is whatever the parameter IS. A rotation is a curve of quaternions rather
than three curves of Euler angles, because interpolating a rotation
component-wise is wrong; [`lerp`](@ref) is the only thing the engine needs to
know about a type.
"""
mutable struct Param{T}
    const name::Symbol      # as the EFFECT names it — no global uniqueness needed
    const label::String
    value::T
    curve::Union{Nothing, AnimCurve{T}}
    visible::Bool
    const range::Any        # (lo, hi) for a number, `nothing` where it means nothing
end
Param(name::Symbol, label::AbstractString, value::T;
      curve = nothing, visible = false, range = nothing) where {T} =
    Param{T}(name, String(label), value, curve, visible, range)

"Whether `p` is animated — has a curve with at least one key."
isanimated(p::Param) = p.curve !== nothing && !isempty(p.curve)

"""
    valueat(p::Param, frame) -> T

`p`'s value at `frame`: its curve where it has one, its static value otherwise.
The static value is also the fallback for a curve that exists but is empty, so
clearing the last key leaves the parameter where it was rather than at zero.
"""
valueat(p::Param, frame::Real) =
    isanimated(p) ? something(valueat(p.curve, frame), p.value) : p.value

"Fraction of `p`'s range, clamped to [0,1] — where its value sits in its lane."
function paramnorm(p::Param, v)
    p.range === nothing && return 0.5
    lo, hi = p.range
    return hi > lo ? clamp((Float64(v) - lo) / (hi - lo), 0.0, 1.0) : 0.5
end

"Inverse of [`paramnorm`](@ref): a [0,1] lane fraction back to a value."
function paramdenorm(p::Param, u::Real)
    p.range === nothing && return p.value
    lo, hi = p.range
    return lo + clamp(u, 0.0, 1.0) * (hi - lo)
end
