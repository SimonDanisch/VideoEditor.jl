"""
A keyframe's tangent handle, in the curve's OWN plane: `x` in frames, `y` in the
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

`ease` is the key's kind, and the two BÉZIER kinds are Photoshop's two anchor
kinds — the vocabulary is deliberate, because the gestures are the same:

  - `:linear` — a corner with no handles: constant velocity into and out of it
  - `:smooth` — a flat tangent: the value eases in AND out (no handles either)
  - `:hold`   — a step: the value freezes until the next key
  - `:bezier` — a SMOOTH ANCHOR: `inhandle`/`outhandle` are kept collinear, so
    dragging one rotates the other and the curve passes through without a kink
  - `:corner` — a CORNER ANCHOR: the two handles move independently

`inhandle` reaches BACK toward the previous key (`x <= 0`), `outhandle` reaches
FORWARD toward the next (`x >= 0`), both relative to this key. A segment is a
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
value)` pairs sorted by ABSOLUTE source frame — the same keying as `ColorTrack`
/`MotionTrack`, so a curve survives clip splits — interpolated between keys (see
[`Keyframe`](@ref) for the kinds) and held flat past the first/last one.

The engine is parameter-agnostic in the VALUE: a key holds a `T`, and how two of
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
    return segmentvalue(c, a, b, f)
end

"""
    segmentvalue(curve, a, b, f) -> value

The value between two adjacent keys. **The one place a value type's shape
matters**, so it is a dispatch and not a branch.

The generic method is a cubic hermite with unit (linear) or flat (eased) end
tangents — `m == 1` at a linear corner, `m == 0` at a smooth key, so
`:smooth`/`:smooth` is exactly smoothstep. `h` is the eased position between the
keys and [`lerp`](@ref) is what the VALUE type says blending means. `h` may leave
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
A REAL parameter has a plane, so its keys can carry handles and the segment is
the cubic Bézier through `a`, `a + a.outhandle`, `b + b.inhandle`, `b` — the
curve the user sees and drags. Without handles on either end this falls back to
the tangent model above, so a curve only changes shape once somebody has actually
touched a handle.
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
    # A SIDE WITHOUT A HANDLE PUTS ITS CONTROL POINT ON THE ANCHOR — a retracted
    # handle, exactly as in Photoshop, so the curve leaves that end straight at the
    # other control point. Guarding only the y (and letting `clamp` see the NaN x
    # of `NOHANDLE`) put a NaN into the solve, and every frame of such a segment
    # came back as the far endpoint.
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
        # replacing a key keeps its kind AND its handles: retyping a value must not
        # silently straighten a curve somebody shaped
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
    # handles SURVIVE a conversion, as they do in Photoshop: switching an anchor to
    # a corner and back must give the shape back, not a straight line
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
direction — and **keeps its own length**, which is what makes dragging one side
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

This is what turns a BAKED animation back into an editable one. The lego project
arrived with a key on every single frame — 181 per parameter, 1267 in all, for
four sine waves, one straight ramp and one constant. Nobody can steer that: the
lane is a wall of diamonds and moving one changes a single frame.

Schneider's fit ("An Algorithm for Automatically Fitting Digitized Curves",
Graphics Gems), specialised to a curve that is a FUNCTION of the frame: anchors
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
        keep = ks[1]
        empty!(ks); push!(ks, Keyframe{T}(keep.frame, keep.value, :linear))
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
    # EVERY INTERIOR ANCHOR COMES BACK TWICE — once as the end of a segment,
    # carrying only its `in`, and once as the start of the next, carrying only its
    # `out`. They have to be MERGED. Dropping the second (which is what "skip the
    # duplicate" does) throws away half of every tangent and leaves a curve that
    # misses its own samples by 92% of their range.
    empty!(ks)
    for (f, v, inh, outh) in out
        if !isempty(ks) && ks[end].frame == f
            p = ks[end]
            ks[end] = Keyframe{T}(f, p.value, :bezier,
                                  hashandle(p.inhandle) ? p.inhandle : inh,
                                  hashandle(p.outhandle) ? p.outhandle : outh)
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

Least squares for the two handle LENGTHS of segment `i:j`, along directions taken
from the data — Schneider's normal equations, **with the parameterisation fixed
between rounds**.

That fixing is the whole difficulty. The residual has to be measured at the
curve parameter where the Bézier actually reaches the sample's frame, and that
parameter depends on the lengths being solved for. Solved once against `u`
guessed from the frame, the system fights itself: measured on a half-sine it
returned lengths 20% off where 1.3% was available. So it alternates — solve,
then recompute every `u` with [`beziersolve`](@ref), which is EXACT here because
x is monotone — and converges in a couple of rounds.

Pinning the handles' x to thirds of the span instead would make x linear in `u`
and the whole thing one clean linear solve. It is also strictly worse: the curve
is then a cubic POLYNOMIAL in the frame, and a cubic polynomial cannot follow a
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
