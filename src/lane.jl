# One parameter's keyframe curve, drawn over the timeline.
#
# A recipe, so that the curve, its ◆ and the selected anchor's handles are three
# derivations of one input and a zoom, an edit and a selection in the same frame
# recompute once. The keyframe editor used to be a builder function holding a
# `Dict{Param, Lane}` of plot handles, a per-lane `dirty` counter to poke them
# with and a reconcile pass over "what should be on screen"; a plot that follows
# its parameter's curve needs none of the three.

"""
How a clip's frames sit on the timeline, as data.

[`sourceframe`](@ref) and [`timelineframe`](@ref) are `Clip` methods, and a plot
cannot hold a `Clip`: a mutable document object is not an input a compute graph
can notice changing, so a clip that moved would leave its curve behind. These
four numbers are such an input — [`placelanes!`](@ref) writes a new one where the
timeline already re-places the clip's filmstrip.
"""
struct FrameMap
    start::Int          # timeline frame the clip begins at
    src_in::Int         # …and the source frame showing there
    rate::Float64       # source frames per timeline frame
    framerate::Float64  # timeline frames per second
end
FrameMap() = FrameMap(0, 0, 1.0, 25.0)
FrameMap(clip::Clip, framerate::Real) =
    FrameMap(clip.start, clip.src_in, clip.rate, Float64(framerate))

sourceframe(m::FrameMap, n::Integer) =
    m.rate == 1.0 ? m.src_in + (Int(n) - m.start) :
    m.src_in + floor(Int, (Int(n) - m.start) * m.rate)

timelineframe(m::FrameMap, sf::Integer) =
    m.start + floor(Int, (Int(sf) - m.src_in) / m.rate)

"Where source frame `sf` of this clip shows, in timeline seconds."
seconds(m::FrameMap, sf::Integer) = timelineframe(m, sf) / m.framerate

"The timeline frame at `t` seconds, and the source frame showing there."
sourceframeat(m::FrameMap, t::Real) = sourceframe(m, round(Int, t * m.framerate))

"""
Timeline frames per source frame around `f` — the scale a handle's `x` is in. On
a conformed clip a key's handle reaches further across the timeline than across
the source, and by exactly this.
"""
perframe(m::FrameMap, f::Integer) = timelineframe(m, f + 1) - timelineframe(m, f)

"Where value `v` sits in `band`, given what the band means."
laney(band::Tuple{Float64, Float64}, valuerange, v) =
    band[1] + (band[2] - band[1]) * paramnorm(valuerange, v)

"The value a lane's y coordinate means — the inverse of [`laney`](@ref)."
lanevalue(band::Tuple{Float64, Float64}, valuerange, y::Real) =
    paramdenorm(valuerange, (y - band[1]) / max(band[2] - band[1], 1.0e-12))

"""
    lanecurve!(ax, curve; clipspan, viewrange, band, valuerange, framemap, …)

One parameter's curve over the timeline: the polyline, the ◆ that are far enough
apart to be told apart, and — for the one selected anchor — its Bézier handles.

Everything is derived from `curve`, so an edit that notifies it redraws exactly
this lane and nothing else. That the curve is a parameter's is the caller's
business: what is drawn is a curve, a span to draw it across and a band to draw it
in, which is also why a parameter driven down a [`ParamInput`](@ref) shows the
curve it keeps rather than the value arriving on the edge — that value is the
other parameter's, and unbinding gives this one back.

`markkeys` and `handlesides` are outputs, not decoration: a gesture reads them to
learn which key it hit, so there is no second copy of what is drawn to keep in
step with the drawing.
"""
@recipe LaneCurve (curve,) begin
    "The clip's extent in timeline seconds — where the lane starts and ends."
    clipspan = (0.0, 1.0)
    "Visible x-range of the timeline axis, in seconds."
    viewrange = (0.0, 1.0e9)
    "The axis-y band the values map onto — this clip's track lane, inset."
    band = (0.0, 1.0)
    "The parameter's own value range, i.e. what the band means. `nothing` centres it."
    valuerange = nothing
    "How this clip's frames sit on the timeline — see [`FrameMap`](@ref)."
    framemap = FrameMap()
    "On-screen pixels per second, which decides how finely the curve is sampled."
    pixelspersecond = 100.0
    "Pixels two ◆ must be apart to be drawn as two."
    markgap = 9.0
    "Index of this curve's selected anchor, or 0 — only it shows handles."
    selectedkey = 0
    color = RGBAf(1, 0.47, 0.22, 1)
    handlecolor = :white
    # `visible` among them, which is what a parameter's lane is switched off with
    # — a recipe declares its own attributes and inherits none of the generic ones
    # unless it says so.
    Makie.mixin_generic_plot_attributes()...
end

function Makie.plot!(p::LaneCurve)
    map!(p, [:curve, :clipspan, :viewrange, :band, :valuerange, :framemap,
             :pixelspersecond], :points) do c, span, vrange, band, vals, fm, pps
        return lanepoints(c, span, vrange, band, vals, fm, pps)
    end
    map!(p, [:curve, :clipspan, :viewrange, :band, :valuerange, :framemap,
             :pixelspersecond, :markgap],
         [:markpoints, :markkeys]) do c, span, vrange, band, vals, fm, pps, gap
        return lanemarks(c, span, vrange, band, vals, fm, pps, gap)
    end
    map!(p, [:curve, :selectedkey, :band, :valuerange, :framemap],
         [:handlebars, :handletips, :handlesides]) do c, kidx, band, vals, fm
        return handlegeometry(c, kidx, band, vals, fm)
    end
    # Drawn in this order and at one z: a dark halo under the curve so it reads
    # over a filmstrip, the curve, its anchors, then the handles of the selected
    # one on top of all of it.
    # `visible` is passed to every child EXPLICITLY. Declaring it on the recipe
    # only puts it in the parent's attributes; the children are their own plots and
    # keep drawing, so a lane switched off went on being drawn — which meant
    # `Param.visible` never actually took a curve off the timeline.
    vis = p.visible
    lines!(p, p.points; color = (:black, 0.55), linewidth = 4, visible = vis)
    lines!(p, p.points; color = p.color, linewidth = 2, visible = vis)
    scatter!(p, p.markpoints; marker = :diamond, markersize = 10, color = p.color,
             strokecolor = :white, strokewidth = 1, visible = vis)
    lines!(p, p.handlebars; color = p.handlecolor, linewidth = 1.2, visible = vis)
    scatter!(p, p.handletips; marker = :circle, markersize = 8, color = p.handlecolor,
             strokecolor = p.color, strokewidth = 1.5, visible = vis)
    return p
end

"A curve is its own argument — `lanecurve!(ax, p.curve)` takes the parameter's."
Makie.convert_arguments(::Type{<:LaneCurve}, c::AnimCurve) = (c,)

"""
    lanepoints(curve, clipspan, viewrange, band, valuerange, framemap, pps)

The lane's polyline over what is on screen.

Sampled per pixel, and never finer than the frames the animation plays. A fixed
count — 61 points across the whole clip — is four samples per Bézier segment on
the lego walk, which draws the curve as a chain of chords.
"""
function lanepoints(c::AnimCurve, clipspan, viewrange, band, valuerange, fm::FrameMap, pps)
    isempty(c) && return Point2f[]
    x0 = max(clipspan[1], viewrange[1])
    x1 = min(clipspan[2], viewrange[2])
    x1 > x0 || return Point2f[]
    npx = (x1 - x0) * pps
    nsamp = clamp(round(Int, min(npx / 2, (x1 - x0) * fm.framerate)), 40, 4000)
    return [(t = x0 + (x1 - x0) * i / nsamp;
             Point2f(t, laney(band, valuerange, valueat(c, sourceframeat(fm, t)))))
            for i in 0:nsamp]
end

"""
    lanemarks(curve, clipspan, viewrange, band, valuerange, framemap, pps, gap)

The ◆ of one lane, and which key of `curve` each one is.

Thinned to `gap` pixels: a baked animation has a key on every frame (the lego
walk: 181 on each of seven curves), and a grab reaches 14 px, so anything closer
than that is not a separate target either. Below the separation the curve alone
shows where the keys are.

A one-key curve draws none: that key is the parameter's constant value, not a
keyframe, and a ◆ sitting on it would say the opposite.
"""
function lanemarks(c::AnimCurve, clipspan, viewrange, band, valuerange, fm::FrameMap,
                   pps, gap)
    pts = Point2f[]; idx = Int[]
    length(c.keys) > 1 || return (pts, idx)
    gapsec = gap / max(pps, 1.0e-12)
    lastx = -Inf
    for (k, key) in enumerate(c.keys)
        x = seconds(fm, key.frame)
        clipspan[1] <= x <= clipspan[2] || continue
        x - lastx < gapsec && continue
        lastx = x
        push!(pts, Point2f(x, laney(band, valuerange, key.value)))
        push!(idx, k)
    end
    return (pts, idx)
end

"""
    handlegeometry(curve, kidx, band, valuerange, framemap)

The selected anchor's handles: the bars from anchor to grip (NaN-separated), the
grips, and which side each one is.

One anchor at a time. Every anchor's handles at once is a thicket you cannot aim
in, and `kidx == 0` — nothing selected — is the resting state.
"""
function handlegeometry(c::AnimCurve, kidx::Integer, band, valuerange, fm::FrameMap)
    bars = Point2f[]; tips = Point2f[]; sides = Symbol[]
    (1 <= kidx <= length(c.keys)) || return (bars, tips, sides)
    k = c.keys[kidx]
    scale = perframe(fm, k.frame)
    ax0 = seconds(fm, k.frame)
    ay0 = laney(band, valuerange, k.value)
    for (side, h) in ((:in, k.inhandle), (:out, k.outhandle))
        hashandle(h) || continue
        tx = (timelineframe(fm, k.frame) + scale * h[1]) / fm.framerate
        ty = laney(band, valuerange, k.value + h[2])
        append!(bars, (Point2f(ax0, ay0), Point2f(tx, ty), Point2f(NaN, NaN)))
        push!(tips, Point2f(tx, ty))
        push!(sides, side)
    end
    return (bars, tips, sides)
end
