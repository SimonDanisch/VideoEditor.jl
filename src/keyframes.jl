"""
A keyframe's tangent handle, in the curve's own plane: `x` in frames, `y` in the
parameter's value units. Exactly what the user drags — see [`Keyframe`](@ref).

`NOHANDLE` means "this side has no handle", which is what `:linear`, `:smooth`
and `:hold` keys carry; the handle only becomes the truth for `:bezier` and
`:corner` keys. A zero-length handle is therefore distinguishable from an absent
one, which matters when a user drags a handle all the way onto its anchor.
"""
const Handle = Vec2f
const NOHANDLE = Vec2f(NaN, NaN)
hashandle(h::Handle) = !isnan(h[1])

"""
One control point of an [`AnimCurve`](@ref).

`ease` is the key's kind. The two Bézier kinds are a pen tool's two anchor kinds,
and the gestures match:

  - `:linear` — a corner with no handles: constant velocity into and out of it
  - `:smooth` — a flat tangent: the value eases in and out (no handles either)
  - `:hold`   — a step: the value freezes until the next key
  - `:bezier` — a smooth anchor: `inhandle`/`outhandle` are kept collinear, so
    dragging one rotates the other and the curve passes through without a kink
  - `:corner` — a corner anchor: the two handles move independently

`inhandle` reaches back toward the previous key (`x <= 0`), `outhandle` forward
toward the next (`x >= 0`), both relative to this key. A segment is a
cubic Bézier through `a`, `a + a.outhandle`, `b + b.inhandle`, `b`.

Handles shape the curve only where the value is a `Real` — see
[`segmentvalue`](@ref). A quaternion has no plane to drag a handle in, so those
curves keep the tangent model, which is the same reason [`lerp`](@ref) is the
only thing the engine knows about a value.
"""
struct Keyframe{T}
    frame::Int
    value::T
    ease::Symbol
    inhandle::Handle
    outhandle::Handle
end
Keyframe(frame::Integer, value, ease::Symbol) =
    Keyframe(Int(frame), value, ease, NOHANDLE, NOHANDLE)
Keyframe(frame::Integer, value) = Keyframe(Int(frame), value, :linear)
Keyframe{T}(frame::Integer, value, ease::Symbol) where {T} =
    Keyframe{T}(Int(frame), convert(T, value), ease, NOHANDLE, NOHANDLE)

"Whether this key's handles are what shapes the curve around it."
isbezier(k::Keyframe) = k.ease === :bezier || k.ease === :corner

"""
Keyframed animation for a single clip parameter. Control points are `(frame,
value)` pairs sorted by absolute source frame — the same keying as `ColorTrack`
/`MotionTrack`, so a curve survives clip splits — interpolated between keys (see
[`Keyframe`](@ref) for the kinds) and held flat past the first/last one.

The engine is agnostic in the value: a key holds a `T`, and how two of
them blend is [`lerp`](@ref). That is what lets a rotation be a curve of
quaternions rather than three curves of Euler angles — interpolating a rotation
component-wise is simply wrong, and decomposing it here would make that the only
option.

`interp` is the legacy curve-wide default; per-key kinds win over it, and
[`materializeease!`](@ref) bakes it away the first time somebody edits a key.
"""
mutable struct AnimCurve{T}
    const keys::Vector{Keyframe{T}}
    interp::Symbol   # curve-wide default: :smooth eases EVERY key
end
AnimCurve{T}() where {T} = AnimCurve{T}(Keyframe{T}[], :linear)
AnimCurve() = AnimCurve{Float64}()

"""
The same curve in another value type — what a project file needs when the kind
has changed a parameter's type since it was written (a `Float32` angle read back
into a `Float64` one). Keys convert one by one; the handles are already `Float32`
pairs in the curve's own plane and carry over.
"""
AnimCurve{T}(c::AnimCurve) where {T} =
    AnimCurve{T}([Keyframe{T}(k.frame, convert(T, k.value), k.ease, k.inhandle, k.outhandle)
                  for k in c.keys], c.interp)

"""
    lerp(a, b, t) -> typeof(a)

Blend two keyframe values: the only thing [`valueat`](@ref) needs to know about a
parameter's type, so a new animatable type is one method.
"""
lerp(a::Real, b::Real, t::Real) = a + t * (b - a)
lerp(a::AbstractVector, b::AbstractVector, t::Real) = a .+ t .* (b .- a)
lerp(a::Colorant, b::Colorant, t::Real) =
    Makie.lerp_oklab(RGBf(a), RGBf(b), Float32(t))

"""
An image blends per pixel. Reached through a `:mix` input, never through a curve:
two pictures as keyframe values would sit in the project file and allocate a third
per frame, where what the user has is two sources and a number between them.
"""
lerp(a::AbstractMatrix{<:Colorant}, b::AbstractMatrix{<:Colorant}, t::Real) =
    (size(a) == size(b) ||
         error("cannot blend images of $(size(a)) and $(size(b)) — resize one first");
     lerp.(a, b, Float32(t)))

"""
A mesh blends vertex by vertex, which is only meaningful when the two describe the
same thing in two poses.

Refused otherwise: interpolating between meshes of different topology is a
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
    return segmentvalue(c, a, b, f)
end

"""
    segmentvalue(curve, a, b, f) -> value

The value between two adjacent keys, and the one place a value type's shape
matters, so it is a dispatch rather than a branch.

The generic method is a cubic hermite with unit (linear) or flat (eased) end
tangents — `m == 1` at a linear corner, `m == 0` at a smooth key, so
`:smooth`/`:smooth` is exactly smoothstep. `h` is the eased position between the
keys and [`lerp`](@ref) is what the value type says blending means. `h` may leave
`[0, 1]` at a linear corner, which is the overshoot a hermite is supposed to
have — every `lerp` is an affine combination, so that carries through unchanged.
"""
function segmentvalue(c::AnimCurve, a::Keyframe, b::Keyframe, f::Real)
    t = (f - a.frame) / (b.frame - a.frame)
    m0 = keyease(c, a) === :smooth ? 0.0 : 1.0
    m1 = keyease(c, b) === :smooth ? 0.0 : 1.0
    h = (t^3 - 2t^2 + t) * m0 + (-2t^3 + 3t^2) + (t^3 - t^2) * m1
    return lerp(a.value, b.value, h)
end

"""
A real-valued parameter has a plane, so its keys can carry handles and the segment
is the cubic Bézier through `a`, `a + a.outhandle`, `b + b.inhandle`, `b` — the
curve the user sees and drags. Without handles on either end it falls back to the
tangent model above, so a curve changes shape only once a handle is touched.
"""
function segmentvalue(c::AnimCurve{T}, a::Keyframe{T}, b::Keyframe{T}, f::Real) where {T <: Real}
    p1 = isbezier(a) && hashandle(a.outhandle) ? a.outhandle : NOHANDLE
    p2 = isbezier(b) && hashandle(b.inhandle) ? b.inhandle : NOHANDLE
    (hashandle(p1) || hashandle(p2)) ||
        return invoke(segmentvalue, Tuple{AnimCurve, Keyframe, Keyframe, Real}, c, a, b, f)
    span = b.frame - a.frame
    # A handle that reaches past the neighbouring key would fold the curve back on
    # itself, and a curve that is not a function of time has no value AT a frame.
    # Photoshop lets a path loop; an animation curve cannot, so the x components
    # are clamped into the segment. Only x — the y overshoot is the whole point of
    # a handle and stays free.
    # A side without a handle puts its control point on the anchor — a retracted
    # handle — so the curve leaves that end straight at the other control point.
    # Guarding only the y, and letting `clamp` see `NOHANDLE`'s NaN x, put a NaN
    # into the solve and returned the far endpoint for every frame of the segment.
    x1 = a.frame + (hashandle(p1) ? clamp(p1[1], 0.0f0, Float32(span)) : 0.0f0)
    x2 = b.frame + (hashandle(p2) ? clamp(p2[1], -Float32(span), 0.0f0) : 0.0f0)
    y0, y3 = Float64(a.value), Float64(b.value)
    y1 = y0 + (hashandle(p1) ? Float64(p1[2]) : 0.0)
    y2 = y3 + (hashandle(p2) ? Float64(p2[2]) : 0.0)
    u = beziersolve(Float64(a.frame), Float64(x1), Float64(x2), Float64(b.frame), Float64(f))
    return convert(T, bezier1(y0, y1, y2, y3, u))
end

"One coordinate of a cubic Bézier at parameter `u`."
bezier1(p0, p1, p2, p3, u) =
    (v = 1 - u; v * v * v * p0 + 3v * v * u * p1 + 3v * u * u * p2 + u * u * u * p3)

"Its derivative, for the Newton step in [`beziersolve`](@ref)."
bezier1′(p0, p1, p2, p3, u) =
    (v = 1 - u; 3v * v * (p1 - p0) + 6v * u * (p2 - p1) + 3u * u * (p3 - p2))

"""
    beziersolve(x0, x1, x2, x3, x) -> u

The curve parameter at which the Bézier's x reaches `x`. Newton from the linear
guess, with bisection as the fallback — `x` is monotone by construction (the
caller clamps the control x into the span), so a bracket always exists and the
loop cannot diverge. Eight iterations put it well under a thousandth of a frame.
"""
function beziersolve(x0, x1, x2, x3, x)
    x3 == x0 && return 0.0
    u = clamp((x - x0) / (x3 - x0), 0.0, 1.0)
    lo, hi = 0.0, 1.0
    for _ in 1:8
        e = bezier1(x0, x1, x2, x3, u) - x
        abs(e) < 1.0e-6 && return u
        e > 0 ? (hi = u) : (lo = u)
        d = bezier1′(x0, x1, x2, x3, u)
        step = d == 0 ? NaN : u - e / d
        u = (isnan(step) || step <= lo || step >= hi) ? 0.5 * (lo + hi) : step
    end
    return u
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
        k = c.keys[i]
        # replacing a key keeps its kind and its handles, so retyping a value does
        # not straighten a curve that was shaped
        c.keys[i] = Keyframe{T}(Int(frame), convert(T, value), k.ease, k.inhandle, k.outhandle)
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
    k = c.keys[i]
    c.keys[i] = Keyframe{T}(Int(frame), convert(T, value), k.ease, k.inhandle, k.outhandle)
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
    # handles survive a conversion, so switching an anchor to a corner and back
    # gives the shape back rather than a straight line
    c.keys[i] = Keyframe(k.frame, k.value, mode, k.inhandle, k.outhandle)
    return c
end

"""
    autohandles!(c, i; strength = 1/3) -> curve

Give key `i` the handles it would have had if somebody had drawn it with the pen
tool: collinear, along the chord through its neighbours, each reaching
`strength` of the way to that neighbour. Makes it a `:bezier` (smooth) anchor.

This is what a `:linear` key turns into when the user first grabs a handle, and
what [`simplify!`](@ref) starts every anchor from. The chord tangent is
Catmull-Rom's: it reproduces a straight run exactly and rounds a corner the way
the eye expects.
"""
function autohandles!(c::AnimCurve{T}, i::Integer; strength::Real = 1 / 3) where {T <: Real}
    ks = c.keys
    k = ks[i]
    prev = i > 1 ? ks[i - 1] : k
    next = i < length(ks) ? ks[i + 1] : k
    dx = Float64(next.frame - prev.frame)
    dy = Float64(next.value) - Float64(prev.value)
    slope = dx == 0 ? 0.0 : dy / dx                       # value units per frame
    back = Float32(strength * (k.frame - prev.frame))
    fwd = Float32(strength * (next.frame - k.frame))
    ks[i] = Keyframe{T}(k.frame, k.value, :bezier,
                        Handle(-back, -back * slope), Handle(fwd, fwd * slope))
    return c
end

"""
    sethandle!(c, i, side, h; couple = true) -> curve

Move one handle of key `i` (`side` is `:in` or `:out`), Photoshop's way.

On a `:bezier` anchor the other handle follows: it stays collinear — opposite
direction — and keeps its own length, which is what makes dragging one side
rotate the whole tangent without resizing the other. `couple = false` is
Alt-dragging: the anchor becomes a `:corner` and only the grabbed side moves.
"""
function sethandle!(c::AnimCurve{T}, i::Integer, side::Symbol, h::Handle;
                    couple::Bool = true) where {T <: Real}
    k = c.keys[i]
    isbezier(k) || autohandles!(c, i)                     # first grab makes it an anchor
    k = c.keys[i]
    inh, outh = k.inhandle, k.outhandle
    mode = couple ? (k.ease === :corner ? :corner : :bezier) : :corner
    if side === :out
        outh = h
        if mode === :bezier && hashandle(inh)
            inh = mirrorhandle(h, inh)
        end
    else
        inh = h
        if mode === :bezier && hashandle(outh)
            outh = mirrorhandle(h, outh)
        end
    end
    c.keys[i] = Keyframe{T}(k.frame, k.value, mode, inh, outh)
    return c
end

"The partner of handle `h`: opposite direction, `other`'s own length. A handle
dragged onto its anchor has no direction left, so the partner is left alone."
function mirrorhandle(h::Handle, other::Handle)
    n = hypot(h[1], h[2])
    n == 0 && return other
    return Handle(-h[1] / n, -h[2] / n) * hypot(other[1], other[2])
end

"""
    smoothkey!(c, i) / cornerkey!(c, i) -> curve

Convert an anchor, which is Alt-clicking it in Photoshop. `smoothkey!` re-couples
the handles (deriving them from the neighbours when there are none yet);
`cornerkey!` keeps the handles where they are and stops coupling them.
"""
smoothkey!(c::AnimCurve{<:Real}, i::Integer) = autohandles!(c, i)

function cornerkey!(c::AnimCurve{T}, i::Integer) where {T <: Real}
    k = c.keys[i]
    isbezier(k) || autohandles!(c, i)
    k = c.keys[i]
    c.keys[i] = Keyframe{T}(k.frame, k.value, :corner, k.inhandle, k.outhandle)
    return c
end

"""
    simplify!(c; tol = 0.005, minkeys = 2) -> curve

Replace a densely sampled curve with the few Bézier anchors that reproduce it.

This turns a baked animation back into an editable one. The lego project arrived
with a key on every frame — 181 per parameter, 1267 in all, for four sine waves,
one ramp and one constant — where the lane is a wall of diamonds and moving one
changes a single frame.

Schneider's fit ("An Algorithm for Automatically Fitting Digitized Curves",
Graphics Gems), specialised to a curve that is a function of the frame: anchors
keep their sampled position, the two handle lengths come from a least-squares
solve against the samples between them, and a run whose worst error exceeds `tol`
is split at that worst sample and fitted again. `tol` is relative to the
parameter's own value range, because a joint angle in radians and an offset in
pixels have nothing comparable about their absolute error.

Refuses to touch anything but a `Real` curve, for the same reason handles are
`Real`-only: there is no plane to fit in.
"""
function simplify!(c::AnimCurve{T}; tol::Real = 0.005, minkeys::Integer = 2) where {T <: Real}
    ks = c.keys
    length(ks) > max(minkeys, 2) || return c
    xs = Float64[k.frame for k in ks]
    ys = Float64[k.value for k in ks]
    span = maximum(ys) - minimum(ys)
    # A constant curve is one key. Nothing below can discover that, because every
    # fit of a flat run is exact and the recursion never splits — but it would
    # still keep both endpoints.
    if span <= tol * max(abs(maximum(ys)), 1.0)
        first_ = ks[1]
        empty!(ks); push!(ks, Keyframe{T}(first_.frame, first_.value, :linear))
        return c
    end
    eps = max(tol * span, 1.0e-9)
    out = Tuple{Int, Float64, Handle, Handle}[]     # frame, value, in, out
    # A held key is a wall: the value is meant to jump there, and a fit across it
    # would round the step off.
    runs = Vector{UnitRange{Int}}()
    lo = 1
    for i in 1:length(ks)
        if ks[i].ease === :hold && i > lo
            push!(runs, lo:i); lo = i
        end
    end
    push!(runs, lo:length(ks))
    for r in runs
        fitrun!(out, xs, ys, first(r), last(r), eps)
    end
    # Every interior anchor comes back twice: once as the end of a segment carrying
    # only its `in`, once as the start of the next carrying only its `out`. They
    # have to be merged — dropping the duplicate throws away half of every tangent
    # and leaves a curve that misses its own samples by 92% of their range.
    empty!(ks)
    for (f, v, inh, outh) in out
        if !isempty(ks) && ks[end].frame == f
            half = ks[end]
            ks[end] = Keyframe{T}(f, half.value, :bezier,
                                  hashandle(half.inhandle) ? half.inhandle : inh,
                                  hashandle(half.outhandle) ? half.outhandle : outh)
        else
            push!(ks, Keyframe{T}(f, convert(T, v), :bezier, inh, outh))
        end
    end
    return c
end

"Recursive half of [`simplify!`](@ref): fit `i:j`, or split at the worst sample."
function fitrun!(out, xs, ys, i::Int, j::Int, eps::Real)
    dx = Float32((xs[j] - xs[i]) / 3)
    if j - i < 2                                          # a segment with nothing between
        dy = Float32((ys[j] - ys[i]) / 3)
        push!(out, (round(Int, xs[i]), ys[i], NOHANDLE, Handle(dx, dy)))
        push!(out, (round(Int, xs[j]), ys[j], Handle(-dx, -dy), NOHANDLE))
        return out
    end
    t0, t1, a1, a2 = fithandles(xs, ys, i, j)
    err, worst = fiterror(xs, ys, i, j, t0, t1, a1, a2)
    if err <= eps
        push!(out, (round(Int, xs[i]), ys[i], NOHANDLE, Handle(a1 * t0[1], a1 * t0[2])))
        push!(out, (round(Int, xs[j]), ys[j], Handle(a2 * t1[1], a2 * t1[2]), NOHANDLE))
        return out
    end
    fitrun!(out, xs, ys, i, worst, eps)
    fitrun!(out, xs, ys, worst, j, eps)
    return out
end

"""
    fithandles(xs, ys, i, j) -> (t0, t1, a1, a2)

Least squares for the two handle lengths of segment `i:j`, along directions taken
from the data — Schneider's normal equations, with the parameterisation fixed
between rounds.

That fixing is the whole difficulty. The residual has to be measured at the
curve parameter where the Bézier actually reaches the sample's frame, and that
parameter depends on the lengths being solved for. Solved once against `u`
guessed from the frame, the system fights itself: measured on a half-sine it
returned lengths 20% off where 1.3% was available. So it alternates — solve,
then recompute every `u` with [`beziersolve`](@ref), which is exact here because
x is monotone — and converges in a couple of rounds.

Pinning the handles' x to thirds of the span instead would make x linear in `u`
and the whole thing one clean linear solve. It is also strictly worse: the curve
is then a cubic polynomial in the frame, and a cubic polynomial cannot follow a
half-sine closer than about 4%. The freedom in the x components is exactly what
buys the accuracy.
"""
function fithandles(xs, ys, i::Int, j::Int)
    x0, y0, x3, y3 = xs[i], ys[i], xs[j], ys[j]
    span = x3 - x0
    t0 = tangentat(xs, ys, i, +1)
    t1 = tangentat(xs, ys, j, -1)
    span == 0 && return (t0, t1, 0.0, 0.0)
    # a handle pointing backwards folds the segment; the x limit is the same one
    # `segmentvalue` enforces when it draws
    lim0 = t0[1] > 0 ? span / t0[1] : span
    lim1 = t1[1] < 0 ? span / -t1[1] : span
    a1 = a2 = span / 3
    best = first(fiterror(xs, ys, i, j, t0, t1, a1, a2))
    step = span / 3
    for _ in 1:9                                   # halving from a third of the span
        for side in 1:2
            for s in (step, -step)
                b1 = side == 1 ? clamp(a1 + s, 0.0, lim0) : a1
                b2 = side == 2 ? clamp(a2 + s, 0.0, lim1) : a2
                e = first(fiterror(xs, ys, i, j, t0, t1, b1, b2))
                e < best && ((best, a1, a2) = (e, b1, b2))
            end
        end
        step /= 2
    end
    return (t0, t1, a1, a2)
end

"Unit tangent at sample `i`, looking in direction `dir`; the central chord where
there is one, so a peak gets a flat tangent and a ramp keeps its slope."
function tangentat(xs, ys, i::Int, dir::Int)
    lo = max(firstindex(xs), i - 1)
    hi = min(lastindex(xs), i + 1)
    dx, dy = xs[hi] - xs[lo], ys[hi] - ys[lo]
    n = hypot(dx, dy)
    n == 0 && return (Float64(dir), 0.0)
    return (dir * dx / n, dir * dy / n)
end

"Worst vertical distance between the fitted segment and the samples, and where."
function fiterror(xs, ys, i::Int, j::Int, t0, t1, a1::Real, a2::Real)
    x0, y0, x3, y3 = xs[i], ys[i], xs[j], ys[j]
    x1 = clamp(x0 + a1 * t0[1], x0, x3); y1 = y0 + a1 * t0[2]
    x2 = clamp(x3 + a2 * t1[1], x0, x3); y2 = y3 + a2 * t1[2]
    err, worst = 0.0, (i + j) ÷ 2
    for k in (i + 1):(j - 1)
        u = beziersolve(x0, x1, x2, x3, xs[k])
        d = abs(bezier1(y0, y1, y2, y3, u) - ys[k])
        d > err && ((err, worst) = (d, k))
    end
    return err, worst
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
What a parameter can read from: an address that survives sorting, undo and a
project round trip.

Three kinds, and they are all just inputs — that is the point. A number somewhere
else in the sequence, a file on disk, another clip's picture. "The texture is that
video", "the mesh is that STL" and "the opacity is one minus that clip's" are the
same statement, so they are the same mechanism.

Ids, not objects: position and `objectid` both die on the first sort or reload.
[`bindinputs!`](@ref) turns these into the objects that get read.
"""
abstract type InputRef end

"""
Something a value's frames can be counted in: a [`Clip`](@ref), and only that.

Abstract because `Clip` is declared a file later — it has effects, which have
parameters, which have the [`ParamInput`](@ref)s that hold one of these. The
alternative was an untyped field, which costs [`inputframe`](@ref) its dispatch on
every read of a driven parameter.
"""
abstract type FrameSource end

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

Resolved by the graph, not by [`valueat`](@ref): a clip's picture is a transient
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
itself: a small node with an operation and one or more inputs.

A parameter is `value`, `curve` or `input`: a static number, a curve of its
own, or the output of this node. That third case is what a cross-dissolve, a
caption fed by the transcript, an audio-reactive scale, a texture that is another
clip and a mesh morph all are, and they are one mechanism rather than five.

A node rather than a single reference, because a blend needs two things and a
factor.
`:mix` is the reason: it takes `a`, `b` and a numeric `t`, and `t` is an ordinary
parameter with an ordinary slider and an ordinary curve. That is what keeps the
keyframe engine numeric — a curve of meshes would put two meshes in the project
file as keyframe values and allocate one per frame in `valueat`, where what the
user actually has is two sources and a number between them.

`resolved` is filled by [`bindinputs!`](@ref) at the one moment ids can change
meaning: a structural edit. It is not a cache that could disagree — nothing else
reads it. An entry of `nothing` is a dangling input, and a node with one reads as
the parameter's own value rather than failing.
"""
mutable struct ParamInput
    const op::Symbol
    const inputs::Vector{InputRef}
    # What each id resolved to: a `Param`, or a file's contents — a mesh, an
    # image, a LUT. Genuinely open, which is the point of a file input.
    resolved::Vector{Any}
    # What each resolved input's frames are counted in, and what the reader's
    # are, so a value crossing a clip boundary is converted rather than assumed.
    from::Vector{Union{Nothing, FrameSource}}
    to::Union{Nothing, FrameSource}
end
ParamInput(op::Symbol, inputs::InputRef...) =
    ParamInput(op, collect(InputRef, inputs), Any[nothing for _ in inputs],
               Union{Nothing, FrameSource}[nothing for _ in inputs], nothing)

"""
    drives(op) -> Bool

Whether a node of this kind supplies the parameter's value, or only points at
another one.

`:pairedwith` is the second case, and it is the whole of what `Clip.blendfrom`
used to be: "these two are one edit". A cross-dissolve is keyed on one side —
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
It is evaluated on values that just arrived, never on values that were stored.
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
What draws one parameter while its card is on screen: the control that sets it,
the ◆ that keys it, and its curve on the timeline.

On the parameter rather than in a registry keyed by `(effect id, name)`. A
registry has to be filled when a card is built, emptied when it goes, copied when
a card set is kept and looked up on every edit — four places to keep in step with
one truth. Reached from the parameter, an edit that changes a value already holds
everything that shows it.

`control` is `nothing` for a parameter with no widget of its own: one driven down
a [`ParamInput`](@ref) has no value for a slider to set.
"""
mutable struct ParamView
    control::Union{Nothing, Makie.Block}   # Slider, Menu, Checkbox, colour picker
    kf::Makie.Button                       # the ◆
    lane::Makie.Plot                       # its `lanecurve!` on the timeline
    # What this view registered on observables that outlive it — the lane's
    # `selectedkey`, derived from the editor's single selection. A `map` leaves
    # its listener on the source forever, so what a view hooks up it hands back
    # here and `dropcard!` unhooks. See `derive`.
    regs::Vector{Observables.ObserverFunction}
end

"""
One parameter of one effect: what it is called, and the curve that is its value.

The curve is the value. A parameter is never "a number, or else a curve": a
constant is a curve with one key, held over the whole clip, so there is one place
a value lives and `valueat` has one answer to give. A value beside the curve meant
every edit site had to fork on which of the two was in force and write to a
different place in each branch, and every reader had to make the same decision.

That it lives on the parameter at all is the older half of the same point. The
value used to be a field on a typed effect struct and the curve an entry in a flat
`Dict{Symbol, AnimCurve}` on the clip, joined by a name and a global index. A
curve therefore knew only a bare name, not which effect it animated, and
resolution took the first effect of a kind: two Blur entries with one keyframe of
12 rendered `[(blur = 12.0,), (blur = 0.0,)]` — the second silently static, while
its own card's slider wrote to it correctly. Slider and diamond of one row pointed
at different objects.

An `Observable`, because everything that shows a parameter derives from it: the
lane plot, the ◆, the slider. An edit writes the curve and notifies, and there is
nothing to refresh. Mutate the `AnimCurve` in place only through the `Param`
methods below ([`setkey!`](@ref), [`setvalue!`](@ref), …) — they are what
notifies.

`visible` is its lane on the timeline, independent of the keys: showing an empty
lane is how you get somewhere to put the first key. Coupling the two — which a
registry of animated-parameters-only forces — means you must keyframe something
before you can see where its keyframes would go.

`T` is whatever the parameter IS. A rotation is a curve of quaternions rather
than three curves of Euler angles, because interpolating a rotation
component-wise is wrong; [`lerp`](@ref) is the only thing the engine needs to
know about a type.
"""
mutable struct Param{T}
    const name::Symbol      # as the EFFECT names it — no global uniqueness needed
    const label::String
    const curve::Observable{AnimCurve{T}}
    const visible::Observable{Bool}
    const range::Union{Nothing, Tuple{Float64, Float64}}
    # Where the value comes from when it comes from somewhere else — see
    # [`ParamInput`](@ref). The other way a parameter can have a value, and the
    # reason there is no separate machinery for a cross-dissolve, a caption bound
    # to the transcript, an audio-reactive number, a texture that is another clip,
    # or a mesh morph. Not a curve, because "take that value, through this op" is
    # not something a curve can say.
    input::Union{Nothing, ParamInput}
    view::Union{Nothing, ParamView}       # while its card is on screen
end

"""
    Param(name, label, value; curve, visible, range, input) -> Param

`value` seeds the curve's single key — a constant over the clip. An explicit
`curve` wins, and an empty one is seeded the same way, so the invariant "a
parameter always has a value" holds however it was built (a project file written
before this can carry an empty curve).
"""
function Param(name::Symbol, label::AbstractString, value::T;
               curve::Union{Nothing, AnimCurve{T}} = nothing, visible::Bool = false,
               range = nothing, input = nothing) where {T}
    c = curve === nothing ? AnimCurve{T}() : curve
    isempty(c.keys) && push!(c.keys, Keyframe{T}(0, value, :linear))
    return Param{T}(name, String(label), Observable(c), Observable(visible),
                    range === nothing ? nothing :
                    (Float64(range[1]), Float64(range[2])),
                    input, nothing)
end

"""
Whether `p` is animated: more than one key, so its value depends on the frame.

One key is a constant — the shape every parameter starts as — which is why this
is a count and not "does it have a curve".
"""
isanimated(p::Param) = length(p.curve[].keys) > 1

"""
    isconstant(p::Param) -> Bool

Whether `p` is just one value — the shape every parameter starts as.

The negation of [`isanimated`](@ref), and the discriminator the timeline filters
on: a constant draws as a straight line across the clip, which says nothing and
costs a lane. Opening a whole card's lanes used to put two hundred of those on the
timeline and bury the six curves among them.

Asked of the curve rather than kept beside it. A single key IS "one value", so
there is nothing a stored flag could say that the keys do not, and nothing to go
stale when an edit adds the second one.
"""
isconstant(p::Param) = !isanimated(p)

"""
    curvesof(params) -> Vector{Param}

The ones that are actually a curve — what a group's ∿ draws.

A single row's ∿ is not filtered: opening the lane of a constant is how you get
somewhere to put its first key. Only asking for a whole group means "show me the
animation", and a group is where the straight lines pile up.
"""
curvesof(params) = Param[p for p in params if isanimated(p)]

"""
A solo on the timeline: only `on`'s lanes are drawn, and `before` is what was
drawn when it started.

The restore point is taken ONCE, when the solo begins, and kept while the solo
moves from one row or group to the next — so alt-clicking around a card never
loses the state you started from. A plain click on an eye ends the solo instead of
moving it: that is the user picking lanes by hand, and what is on screen is then
the answer, with nothing left to restore.

`before` covers every parameter of the shown clip, not just the hidden ones,
because "put it back" has to be able to hide a lane the solo turned on.
"""
struct LaneSolo
    on::Set{Param}
    before::IdDict{Param, Bool}
end

"""
    paramcolor(p::Param) -> RGBf

The colour that stands for `p` wherever it is drawn: its ◆ in the card, and its
lane on the timeline.

Both used to draw in the editor's accent, so a card of two hundred rows put two
hundred identical orange curves on the timeline and there was no way to tell which
row owned which lane.

The colour comes from the name rather than from a counter or a draw: a parameter
keeps it across a rebuilt card, a reordered stack, a reloaded project and a
restarted session, which is the whole point of colouring by identity. The hue is
the name's hash, at a fixed saturation and value chosen to read on the dark panel.
"""
paramcolor(p::Param) =
    RGBf(HSV(360.0 * (hash(p.name) % 1024) / 1024, 0.62, 0.98))

"Whether `p` has a key exactly at source frame `f` — what fills the row's ◆."
haskeyat(p::Param, f::Integer) = any(k -> k.frame == f, p.curve[].keys)

"What `p` is a parameter OF: the type its keys hold and [`lerp`](@ref) blends."
Base.eltype(::Param{T}) where {T} = T

"""
Whether `p` is driven, i.e. takes its value from another parameter.

A parameter with a `:pairedwith` edge is not: it points at another one and keeps
its own curve. See [`drives`](@ref).
"""
isdriven(p::Param) =
    p.input !== nothing && drives(p.input.op) && all(!isnothing, p.input.resolved)

"""
    valueat(p::Param, frame) -> T

`p`'s value at `frame` — the one question the whole animation system answers.

Two sources: an edge if it has one (the value resolved at the far end and mapped
through `op`), its curve otherwise. An edge whose target is gone reads as the
curve, so deleting what drove a parameter leaves it where it was rather than at
zero.
"""
function valueat(p::Param{T}, frame::Real) where {T}
    n = p.input
    if n !== nothing && drives(n.op) && all(!isnothing, n.resolved)
        vals = ntuple(i -> readinput(n.resolved[i], inputframe(n, i, frame)),
                      length(n.resolved))
        return convert(T, inputvalue(n.op, vals))
    end
    return valueat(p.curve[], frame)::T
end

"""
    readinput(resolved, frame) -> value

What one resolved input delivers at `frame`.

A parameter is read at that frame; a file's contents are what they are and do not
depend on one. A clip's picture is refused here on purpose: it lives as a
transient inside a running composition, so it is wired by the graph and never
fetched from inside a value lookup.
"""
readinput(p::Param, frame::Integer) = valueat(p, frame)
readinput(x, ::Integer) = x
# …and the refusal for a `Clip` is in clips.jl, where that type exists.

"""
    paramnorm(range, v) -> Float64

Where `v` sits in `range`, clamped to [0,1] — a value as a lane height.

Takes the range rather than the parameter, because the lane plot is given one:
what the band means is the drawing's business, and a recipe that reached into a
`Param` for it could not be handed anything else.
"""
paramnorm(range::Tuple{Float64, Float64}, v) =
    range[2] > range[1] ?
    clamp((Float64(v) - range[1]) / (range[2] - range[1]), 0.0, 1.0) : 0.5
paramnorm(::Nothing, v) = 0.5
paramnorm(p::Param, v) = paramnorm(p.range, v)

"Inverse of [`paramnorm`](@ref): a [0,1] lane fraction back to a value."
paramdenorm(range::Tuple{Float64, Float64}, u::Real) =
    range[1] + clamp(u, 0.0, 1.0) * (range[2] - range[1])
paramdenorm(p::Param, u::Real) =
    # Without a range a fraction means nothing, and there is nothing to invert —
    # a parameter with no range has no slider and no lane either.
    p.range === nothing ? valueat(p, 0) : paramdenorm(p.range, u)

# ------------------------------------------------------------- editing a parameter
#
# Every one of these mutates `p.curve[]` in place and notifies it. That is the
# whole reason they exist as methods on `Param` beside the `AnimCurve` ones: an
# edit site says what it changed, and everything drawn from the curve — the lane,
# the ◆, the slider — follows because it is derived from that Observable. Nothing
# refreshes anything.

"""
    setvalue!(p, value, frame) -> p

Set what `p` is at `frame`.

The one value edit, whatever shape the curve is in: animated, it keys at `frame`;
constant, the single key takes the new value and stays where it is. Every call
site used to spell that fork out, and the two halves wrote to different places.
"""
function setvalue!(p::Param{T}, value, frame::Integer) where {T}
    c = p.curve[]
    if isanimated(p)
        setkey!(c, frame, value)
    else
        movekey!(c, 1, c.keys[1].frame, value)
    end
    notify(p.curve)
    return p
end

"Insert or replace `p`'s key at `frame` (see [`setkey!`](@ref AnimCurve))."
setkey!(p::Param, frame::Integer, value) =
    (setkey!(p.curve[], frame, value); notify(p.curve); p)
setkey!(p::Param, frame::Integer, value, ease::Symbol) =
    (setkey!(p.curve[], frame, value, ease); notify(p.curve); p)

"""
    removekey!(p, frame) -> Bool
    removekeyat!(p, i) -> Bool

Take a key off `p`. Refused for the last one: a parameter always has a value, and
a curve with a single key is exactly how a constant is written, so the key before
last is where animation stops.
"""
function removekey!(p::Param, frame::Integer)
    length(p.curve[].keys) > 1 || return false
    removekey!(p.curve[], frame) || return false
    notify(p.curve)
    return true
end

function removekeyat!(p::Param, i::Integer)
    ks = p.curve[].keys
    length(ks) > 1 || return false
    deleteat!(ks, i)
    notify(p.curve)
    return true
end

"""
    clearkeys!(p, frame) -> p

Collapse `p` to the constant it shows at `frame`: one key, the value it had,
held over the whole clip. What "clear the keyframes" means when a value is
always a curve.
"""
function clearkeys!(p::Param{T}, frame::Integer) where {T}
    c = p.curve[]
    v = convert(T, valueat(p, frame))
    empty!(c.keys)
    push!(c.keys, Keyframe{T}(Int(frame), v, :linear))
    notify(p.curve)
    return p
end

"Move `p`'s `i`-th key to `(frame, value)` — see [`movekey!`](@ref AnimCurve)."
movekey!(p::Param, i::Integer, frame::Integer, value) =
    (movekey!(p.curve[], i, frame, value); notify(p.curve); p)

"Set the ease mode of `p`'s `i`-th key."
setease!(p::Param, i::Integer, mode::Symbol) =
    (setease!(p.curve[], i, mode); notify(p.curve); p)

"Move one handle of `p`'s `i`-th key — see [`sethandle!`](@ref AnimCurve)."
sethandle!(p::Param, i::Integer, side::Symbol, h::Handle; couple::Bool = true) =
    (sethandle!(p.curve[], i, side, h; couple); notify(p.curve); p)

"Make `p`'s `i`-th key a smooth anchor / a corner anchor."
smoothkey!(p::Param, i::Integer) = (smoothkey!(p.curve[], i); notify(p.curve); p)
cornerkey!(p::Param, i::Integer) = (cornerkey!(p.curve[], i); notify(p.curve); p)

"Bake `p`'s legacy curve-wide ease into its keys — see [`materializeease!`](@ref)."
materializeease!(p::Param) = (materializeease!(p.curve[]); notify(p.curve); p)

"""
    simplify!(p; tol) -> p

Refit `p`'s curve to the few Bézier anchors that reproduce it — see
[`simplify!`](@ref AnimCurve). `tol` is relative to the parameter's own range,
which is what makes one tolerance mean the same thing for an angle and an offset.
"""
simplify!(p::Param; tol::Real = 0.005) =
    (simplify!(p.curve[]; tol); notify(p.curve); p)
