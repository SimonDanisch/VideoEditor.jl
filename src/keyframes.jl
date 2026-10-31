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

"""
An IMAGE blends per pixel. Reached through a `:mix` input, never through a curve:
two pictures as keyframe values would sit in the project file and allocate a third
one per frame, where what the user has is two sources and a number between them.
"""
lerp(a::AbstractMatrix{<:Colorant}, b::AbstractMatrix{<:Colorant}, t::Real) =
    (size(a) == size(b) ||
         error("cannot blend images of $(size(a)) and $(size(b)) — resize one first");
     lerp.(a, b, Float32(t)))

"""
A MESH blends vertex by vertex, which is only meaningful when the two describe the
same thing in two poses.

Refused otherwise, loudly: interpolating between meshes of different topology is a
resampling problem with its own cost and its own choices, not this. Silently
producing something for "a cube becomes a sphere" would produce garbage that looks
like an animation.
"""
function lerp(a::GeometryBasics.Mesh, b::GeometryBasics.Mesh, t::Real)
    (length(a.position) == length(b.position) && faces(a) == faces(b)) ||
        error("cannot morph meshes with different topology ($(length(a.position)) vs \
               $(length(b.position)) vertices) — that is a resampling problem, not a blend")
    pos = lerp.(a.position, b.position, Float32(t))
    return GeometryBasics.mesh(pos, faces(a); normal = normals(a))
end

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
plugins ([`registerplugin!`](@ref)) declare their parameters with this and get
keyframing for free, because a declared scalar is all the curve engine needs.
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
What a parameter can READ FROM: an address that survives sorting, undo and a
project round trip.

Three kinds, and they are all just inputs — that is the point. A number somewhere
else in the sequence, a file on disk, another clip's picture. "The texture is that
video", "the mesh is this STL" and "the opacity is one minus that clip's" are the
same statement, so they are the same mechanism.

Ids, not objects: position and `objectid` both die on the first sort or reload.
[`bindinputs!`](@ref) turns these into the objects that get read.
"""
abstract type InputRef end

"Another parameter. `clip`/`effect` of `0` mean the one this input is on."
struct ParamRef <: InputRef
    clip::UInt64
    effect::UInt64
    param::Symbol
end
ParamRef(param::Symbol; clip::Integer = 0, effect::Integer = 0) =
    ParamRef(UInt64(clip), UInt64(effect), param)

"A file: a mesh, an image, a LUT. Read once and held, not per frame."
struct FileRef <: InputRef
    path::String
end
FileRef(path::AbstractString) = FileRef(String(path))

"""
Another clip's finished picture.

Resolved by the GRAPH, not by [`valueat`](@ref): a clip's picture is a transient
that exists while a composition runs, and asking for it outside one would mean
rendering a second clip in the middle of reading a number. So this is the input
kind that says "an image arrives here" — [`readinput`](@ref) refuses it, and the
composition wires it.
"""
struct ClipRef <: InputRef
    clip::UInt64
end
ClipRef(clip::Integer) = ClipRef(UInt64(clip))

"""
Where a parameter's value comes from when it does not come from the parameter
itself: a small NODE with an operation and one or more inputs.

A parameter is `value`, `curve` **or** `input` — a static number, a curve of its
own, or the output of this node. That third case is what a cross-dissolve, a
caption fed by the transcript, an audio-reactive scale, a texture that is another
clip and a mesh morph all are, and they are one mechanism rather than five.

A NODE and not a single reference, because a blend needs two things and a factor.
`:mix` is the reason: it takes `a`, `b` and a numeric `t`, and `t` is an ordinary
parameter with an ordinary slider and an ordinary curve. That is what keeps the
keyframe engine numeric — a curve of meshes would put two meshes in the project
file as keyframe VALUES and allocate one per frame in `valueat`, where what the
user actually has is two sources and a number between them.

`resolved` is filled by [`bindinputs!`](@ref) at the one moment ids can change
meaning: a structural edit. It is not a cache that could disagree — nothing else
reads it. An entry of `nothing` is a DANGLING input, and a node with one reads as
the parameter's own value rather than failing.
"""
mutable struct ParamInput
    const op::Symbol
    const inputs::Vector{InputRef}
    resolved::Vector{Any}
    # What each resolved input's frames are counted in, and what the READER's
    # are, so a value crossing a clip boundary is converted rather than assumed.
    from::Vector{Any}
    to::Any
end
ParamInput(op::Symbol, inputs::InputRef...) =
    ParamInput(op, collect(InputRef, inputs), Any[nothing for _ in inputs],
               Any[nothing for _ in inputs], nothing)

"""
    drives(op) -> Bool

Whether a node of this kind supplies the parameter's VALUE, or only points at
another one.

`:pairedwith` is the second case, and it is the whole of what `Clip.blendfrom`
used to be: "these two are one edit". A cross-dissolve is keyed on ONE side —
fading both would darken the middle, because the outgoing clip fades against
nothing while the incoming one only partly covers it — so the pairing genuinely
carries no value, and saying so is a predicate rather than a second mechanism.
"""
drives(::Val{:pairedwith}) = false
drives(::Val) = true
drives(op::Symbol) = drives(Val(op))

"""
    inputvalue(Val(op), values) -> value

What the node computes from what its inputs delivered.

`:mix` is [`lerp`](@ref) — the same function a curve uses between two keys, which
is why a mesh or an image blends here without the engine learning anything new.
It is evaluated on values that just ARRIVED, never on values that were stored.
"""
inputvalue(::Val{:copy}, v) = v[1]
inputvalue(::Val{:invert}, v) = invertvalue(v[1])
inputvalue(::Val{:mix}, v) = lerp(v[1], v[2], Float64(v[3]))
inputvalue(op::Symbol, v) = inputvalue(Val(op), v)

invertvalue(v::Real) = oneunit(v) - v
invertvalue(v::AbstractVector) = oneunit(eltype(v)) .- v

"How many inputs an operation takes — what the panel offers, and what a project
file is checked against."
inputarity(::Val{:copy}) = 1
inputarity(::Val{:invert}) = 1
inputarity(::Val{:mix}) = 3          # a, b, and the factor between them
inputarity(::Val{:pairedwith}) = 1
inputarity(op::Symbol) = inputarity(Val(op))

"""
    inputframe(node, i, frame) -> Int

`frame`, counted in input `i`'s frames instead of the reader's. Identity within
one clip; across a clip boundary it goes out to the timeline and back, because
each clip keys against its own source and the two need not run at the same rate.
"""
function inputframe(n::ParamInput, i::Integer, frame::Real)
    f = n.from[i]
    (f === n.to || f === nothing || n.to === nothing) && return round(Int, frame)
    return sourceframe(f, timelineframe(n.to, round(Int, frame)))
end

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
    # Where the value comes from when it comes from somewhere else — see
    # [`ParamInput`](@ref). The third of the three ways a parameter can have a
    # value, and the reason there is no separate machinery for a cross-dissolve,
    # a caption bound to the transcript, an audio-reactive number, a texture that
    # is another clip, or a mesh morph.
    input::Union{Nothing, ParamInput}
end
Param(name::Symbol, label::AbstractString, value::T;
      curve = nothing, visible = false, range = nothing, input = nothing) where {T} =
    Param{T}(name, String(label), value, curve, visible, range, input)

"Whether `p` is animated — has a curve with at least one key."
isanimated(p::Param) = p.curve !== nothing && !isempty(p.curve)

"""
Whether `p` is DRIVEN — takes its value from another parameter.

A parameter with a `:pairedwith` edge is not: it points at another one and keeps
its own curve. See [`drives`](@ref).
"""
isdriven(p::Param) =
    p.input !== nothing && drives(p.input.op) && all(!isnothing, p.input.resolved)

"""
    valueat(p::Param, frame) -> T

`p`'s value at `frame` — the one question the whole animation system answers.

Three sources, in order: an EDGE if it has one (the value resolved at the far end
and mapped through `op`), its CURVE if it has one, its static value otherwise. The
static value is also the fallback for a curve that exists but is empty and for an
edge whose target is gone, so clearing the last key — or deleting what drove a
parameter — leaves it where it was rather than at zero.
"""
function valueat(p::Param{T}, frame::Real) where {T}
    n = p.input
    if n !== nothing && drives(n.op) && all(!isnothing, n.resolved)
        vals = ntuple(i -> readinput(n.resolved[i], inputframe(n, i, frame)),
                      length(n.resolved))
        return convert(T, inputvalue(n.op, vals))
    end
    return isanimated(p) ? something(valueat(p.curve, frame), p.value) : p.value
end

"""
    readinput(resolved, frame) -> value

What one resolved input delivers at `frame`.

A parameter is read at that frame; a file's contents are what they are and do not
depend on one. A CLIP's picture is refused here on purpose — it lives as a
transient inside a running composition, so it is wired by the graph and never
fetched from inside a value lookup.
"""
readinput(p::Param, frame::Integer) = valueat(p, frame)
readinput(x, ::Integer) = x
# …and the refusal for a `Clip` is in clips.jl, where that type exists.

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

"""
One parameter row of the effects panel, BOUND to its parameter.

A row is not a place a value is pushed to. What it shows — where the slider sits,
whether the ◆ is filled — is DERIVED from `param` and `playheadframe(player,
target)`, and re-derived by [`refreshfxrows!`](@ref) whenever either can have
moved. The other direction is the slider's own handler, which edits the curve.

The row exists so that derivation has one place to happen per on-screen
parameter. Before it, `paramform!` registered an `on(player.playhead)` per row:
118 of them on the lego project, none ever unregistered, and a fresh set with
every card rebuild.

`colors` is the ◆'s three states (a key here / animated / not animated) — the
panel's palette, carried rather than looked up, because the row is drawn once and
the refresh runs on every playhead move.
"""
struct ParamRow
    target::Any                  # the Clip the parameter's frames are counted in
    param::Param
    slider::Any                  # its Slider, or `nothing` for a row without one
    kf::Any                      # the ◆ Button, or `nothing`
    colors::NTuple{3, Any}
end
