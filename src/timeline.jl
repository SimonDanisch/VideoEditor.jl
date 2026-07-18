"""
    Timeline(gridpos, sequence, playhead, playing)

Zoomable timeline over an edited `Sequence`. Each clip is one [`ClipView`](@ref)
plot (band + composed thumbnail strip); the timeline reconciles plots with
the clip list on edits and feeds shared view observables — zooming or
panning only updates `viewrange`/`pixelspersecond` and the recipes recompute
their strips themselves.

Interaction: left-drag scrubs (and selects the clip under the cursor),
Ctrl+left-drag moves the selected clip with a translucent ghost (snap lines,
red tint when the drop would overlap; commits on release), scroll zooms
(x only), right-drag pans, right-click calls `onrightclick(time)`. A time
tooltip follows the cursor; hover brightens a clip's border. Clip edges are
trim handles: hovering one shows a handle bar, dragging it adjusts the
in/out point (the tooltip switches to the clip's length).
"""
mutable struct Timeline
    const axis::Axis
    const sequence::Sequence
    const caches::Dict{VideoSource, ThumbnailCache}
    const playhead::Observable{Int}
    const scrubbing::Base.RefValue{Bool}
    const colors::NamedTuple
    # shared recipe inputs
    const viewrange::Observable{Tuple{Float64, Float64}}
    const pps::Observable{Float64}
    const bandheight::Observable{Float64}  # on-screen px of the thumbnail band
    const refresh::Observable{Int}
    # one ClipView per clip
    const clipplots::Vector{Any}
    const clipranges::Vector{Observable{Tuple{Float64, Float64}}}
    const clipstarts::Vector{Observable{Float64}}
    const clipstates::Vector{Observable{Symbol}}
    const plotsources::Vector{Any}  # VideoSource behind each plot's thumbs
    # interaction feedback
    const selected::Observable{Int}          # clip index, 0 = none
    const hovered::Base.RefValue{Int}
    const ghost_rect::Observable{Rect2f}
    const ghost_color::Observable{Any}
    const snapline::Observable{Vector{Float64}}
    const edgeline::Observable{Vector{Point2f}}  # trim-handle hover/drag marker
    const tooltip_text::Observable{String}
    const tooltip_pos::Observable{Point2f}
    const running::Threads.Atomic{Bool}
    ghost_plot::Any
    tooltip_plot::Any
    dragclip::Union{Nothing, Tuple{Clip, Int}}  # (clip, grab offset in frames)
    dragstart::Int                              # committed on release
    dragvalid::Bool
    trimclip::Union{Nothing, Tuple{Clip, Symbol, Int}}  # (clip, :left/:right, index)
    onrightclick::Function
    onedit::Function                            # called before a gesture mutates
    refreshtask::Task

    function Timeline(gridpos, sequence::Sequence, playhead::Observable{Int},
                      playing::Observable{Bool})
        colors = timelinecolors()
        axis = Axis(gridpos; yzoomlock = true, ypanlock = true, yrectzoom = false,
                    xautolimitmargin = (0.0, 0.0), xgridvisible = false,
                    backgroundcolor = colors.background)
        hideydecorations!(axis)
        hidespines!(axis, :l, :r)
        deregister_interaction!(axis, :rectanglezoom)  # left-drag is scrubbing
        limits!(axis, 0.0, max(seqduration(sequence), 1.0), 0.0, 1.0)

        timeline = new(axis, sequence, Dict{VideoSource, ThumbnailCache}(),
                       playhead, Ref(false), colors,
                       Observable((0.0, max(seqduration(sequence), 1.0))),
                       Observable(100.0), Observable(72.0), Observable(0),
                       Any[], Observable{Tuple{Float64, Float64}}[],
                       Observable{Float64}[], Observable{Symbol}[], Any[],
                       Observable(0), Ref(0),
                       Observable(Rect2f(0, 0, 0, 0)), Observable{Any}(colors.accent_subtle),
                       Observable(Float64[]), Observable(Point2f[]),
                       Observable(""), Observable(Point2f(0, 0)),
                       Threads.Atomic{Bool}(true),
                       nothing, nothing, nothing, 0, false, nothing, identity, identity)

        timeline.ghost_plot = poly!(axis, timeline.ghost_rect; color = timeline.ghost_color,
                                    strokecolor = colors.accent, strokewidth = 1.5,
                                    visible = false)
        translate!(timeline.ghost_plot, 0, 0, 5)
        snap = vlines!(axis, timeline.snapline; color = colors.accent, linewidth = 2)
        translate!(snap, 0, 0, 8)
        edge = linesegments!(axis, timeline.edgeline; color = colors.accent, linewidth = 5)
        translate!(edge, 0, 0, 9)
        phline = vlines!(axis, map(n -> [n / sequence.framerate], playhead);
                         color = colors.accent, linewidth = 2)
        translate!(phline, 0, 0, 10)
        timeline.tooltip_plot = text!(axis, timeline.tooltip_pos; text = timeline.tooltip_text,
                                      color = colors.text, fontsize = 12,
                                      align = (:center, :top), offset = (0, -2),
                                      space = :data, visible = false, overdraw = true)
        translate!(timeline.tooltip_plot, 0, 0, 12)

        onany(axis.finallimits, axis.scene.viewport) do lims, vp
            x0, x1 = minimum(lims)[1], maximum(lims)[1]
            x1 > x0 || return
            timeline.viewrange[] = (x0, x1)
            timeline.pps[] = vp.widths[1] / (x1 - x0)
            timeline.bandheight[] = 0.86 * vp.widths[2]  # strip spans y 0.07..0.93
            return
        end
        on(_ -> setstates!(timeline), timeline.selected)
        wiretimelinemouse(timeline, playhead)

        # re-pull thumbnails as the background decoders fill the caches
        timeline.refreshtask = @async while timeline.running[]
            for cache in collect(values(timeline.caches))
                if cache.dirty[]
                    cache.dirty[] = false
                    timeline.refresh[] += 1
                end
            end
            sleep(0.1)
        end
        # initialize the shared view inputs directly — notify(finallimits) on
        # this branch triggers a full compute-graph walk that can hang
        lims = axis.finallimits[]
        span = max(maximum(lims)[1] - minimum(lims)[1], 1.0e-9)
        timeline.viewrange[] = (minimum(lims)[1], maximum(lims)[1])
        timeline.pps[] = axis.scene.viewport[].widths[1] / span
        relayout!(timeline)
        return timeline
    end
end

"UI role colors from the theme's `:colors` group (with a dark fallback)."
function timelinecolors()
    pick(key, fallback) = try
        Makie.to_color(Makie.to_value(Makie.theme(:colors)[key]))
    catch
        Makie.to_color(fallback)
    end
    return (background = pick(:background, RGBf(0.12, 0.13, 0.15)),
            surface = pick(:surface, RGBf(0.18, 0.19, 0.21)),
            border = pick(:border, RGBf(0.35, 0.36, 0.38)),
            text = pick(:text, RGBf(0.92, 0.92, 0.92)),
            accent = pick(:accent, RGBf(1.0, 0.45, 0.2)),
            accent_subtle = pick(:accent_subtle, RGBf(0.5, 0.32, 0.2)))
end

function stop!(timeline::Timeline)
    timeline.running[] = false
    foreach(stop!, values(timeline.caches))
    return nothing
end

"Thumbnail cache for `source`, created (with its worker) on first use."
cachefor(timeline::Timeline, source::VideoSource) =
    get!(() -> ThumbnailCache(source), timeline.caches, source)

"On-demand thumbnail provider for a `ClipView` showing `source`."
function thumbsfor(timeline::Timeline, source::VideoSource)
    cache = cachefor(timeline, source)
    return second -> begin
        thumb = getthumb(cache, second)
        thumb === nothing && requestone!(cache, second)
        thumb
    end
end

function timelineframe(timeline::Timeline, t::Real)
    return clamp(round(Int, t * timeline.sequence.framerate), 0,
                 max(seqlength(timeline.sequence) - 1, 0))
end

# ------------------------------------------------------------------ layout

"Reconcile one ClipView plot per clip and push current ranges/states."
function relayout!(timeline::Timeline)
    seq = timeline.sequence
    fps = seq.framerate
    while length(timeline.clipplots) < length(seq.clips)
        source = seq.clips[length(timeline.clipplots) + 1].source
        cache = cachefor(timeline, source)
        rng = Observable((0.0, 0.0))
        srcstart = Observable(0.0)
        state = Observable(:idle)
        plt = clipview!(timeline.axis, rng;
                        viewrange = timeline.viewrange, pixelspersecond = timeline.pps,
                        bandheight = timeline.bandheight,
                        sourcestart = srcstart, state = state,
                        thumbs = thumbsfor(timeline, source),
                        refresh = timeline.refresh,
                        color = timeline.colors.surface,
                        strokecolor_idle = timeline.colors.border,
                        strokecolor_hovered = timeline.colors.accent_subtle,
                        strokecolor_selected = timeline.colors.accent,
                        thumbsize = (cache.thumbwidth, cache.thumbheight))
        push!(timeline.clipplots, plt)
        push!(timeline.clipranges, rng)
        push!(timeline.clipstarts, srcstart)
        push!(timeline.clipstates, state)
        push!(timeline.plotsources, source)
    end
    while length(timeline.clipplots) > length(seq.clips)
        delete!(timeline.axis, pop!(timeline.clipplots))
        pop!(timeline.clipranges)
        pop!(timeline.clipstarts)
        pop!(timeline.clipstates)
        pop!(timeline.plotsources)
    end
    for (i, clip) in enumerate(seq.clips)
        timeline.clipranges[i][] = (clip.start / fps, clipend(clip) / fps)
        timeline.clipstarts[i][] = clip.src_in / clip.source.framerate
        if timeline.plotsources[i] !== clip.source  # edits shift clips across plots
            timeline.plotsources[i] = clip.source
            cache = cachefor(timeline, clip.source)
            timeline.clipplots[i].thumbs = thumbsfor(timeline, clip.source)
            timeline.clipplots[i].thumbsize = (cache.thumbwidth, cache.thumbheight)
        end
    end
    cliplimits!(timeline)
    setstates!(timeline)
    return nothing
end

"After an edit shrinks the sequence, pull the view back to the content —
ripple deletes otherwise leave the timeline showing dead space. Keeps the
zoom span; only fires on edits (relayout!), never against manual zooming."
function cliplimits!(timeline::Timeline)
    dur = max(seqduration(timeline.sequence), 1.0)
    lims = timeline.axis.finallimits[]
    x0, x1 = minimum(lims)[1], maximum(lims)[1]
    x1 > dur * 1.05 || return nothing
    newx1 = dur * 1.03  # a sliver of headroom keeps the last clip edge grabbable
    limits!(timeline.axis, max(0.0, newx1 - (x1 - x0)), newx1, 0.0, 1.0)
    return nothing
end

"Hover/selection feedback without touching geometry."
function setstates!(timeline::Timeline)
    for i in eachindex(timeline.clipstates)
        state = i == timeline.selected[] ? :selected :
                i == timeline.hovered[] ? :hovered : :idle
        timeline.clipstates[i][] = state
    end
    return nothing
end

# ------------------------------------------------------------------ mouse

function wiretimelinemouse(timeline::Timeline, playhead::Observable{Int})
    axis, seq = timeline.axis, timeline.sequence
    on(events(axis.scene).mousebutton) do event
        if event.button == Mouse.right && event.action == Mouse.press &&
           is_mouseinside(axis.scene)
            timeline.onrightclick(mouseposition(axis.scene)[1])
            return Consume(true)
        end
        event.button == Mouse.left || return Consume(false)
        if event.action == Mouse.press && is_mouseinside(axis.scene)
            t = mouseposition(axis.scene)[1]
            n = timelineframe(timeline, t)
            i = clipat(seq, n)
            timeline.selected[] = something(i, 0)
            edge = edgeat(timeline, t)
            if ispressed(axis.scene, Keyboard.left_control | Keyboard.right_control) && i !== nothing
                clip = seq.clips[i]
                timeline.dragclip = (clip, n - clip.start)
                timeline.dragstart = clip.start
                timeline.dragvalid = true
            elseif edge !== nothing
                timeline.selected[] = edge[1]
                timeline.onedit()
                timeline.trimclip = (seq.clips[edge[1]], edge[2], edge[1])
            else
                timeline.scrubbing[] = true
                n == playhead[] || (playhead[] = n)
            end
            return Consume(true)
        elseif event.action == Mouse.release
            timeline.scrubbing[] = false
            finishdrag!(timeline)
            if timeline.trimclip !== nothing
                timeline.trimclip = nothing
                relayout!(timeline)
            end
            return Consume(false)
        end
        return Consume(false)
    end

    on(events(axis.scene).mouseposition) do _
        inside = is_mouseinside(axis.scene)
        if timeline.dragclip !== nothing
            dragto!(timeline, mouseposition(axis.scene)[1])
        elseif timeline.trimclip !== nothing
            trimto!(timeline, mouseposition(axis.scene)[1])
        elseif timeline.scrubbing[]
            t = mouseposition(axis.scene)[1]
            n = timelineframe(timeline, t)
            n == playhead[] || (playhead[] = n)
        elseif inside
            hoverat!(timeline, mouseposition(axis.scene)[1])
        end
        if inside
            t = clamp(mouseposition(axis.scene)[1], 0.0, seqduration(seq))
            trim = timeline.trimclip
            timeline.tooltip_text[] = trim === nothing ? timestring(t) :
                                      "clip " * timestring(cliplength(trim[1]) / seq.framerate)
            timeline.tooltip_pos[] = Point2f(t, 1.0)
            timeline.tooltip_plot.visible = true
        else
            timeline.tooltip_plot.visible = false
            timeline.hovered[] == 0 || (timeline.hovered[] = 0; setstates!(timeline))
            isempty(timeline.edgeline[]) || (timeline.edgeline[] = Point2f[])
        end
        return Consume(false)
    end
    return nothing
end

function timestring(t::Real)
    minutes = floor(Int, t / 60)
    return @sprintf("%02d:%05.2f", minutes, t - 60minutes)
end

"Half-width of the clip-edge trim grab zone, in seconds at the current zoom."
edgezone(timeline::Timeline) = 8 / max(timeline.pps[], 1.0e-9)

"""
The trim edge within grab range of time `t`: `(clipindex, :left/:right)`, or
`nothing`. Finds the nearest edge of ANY clip — a `clipat`-based test misses
the exact edge (the boundary frame already belongs to the NEXT clip) and
misses presses just outside the last clip. Where two edges coincide at a cut
the cursor side picks the clip, so dragging always trims the edge you are
visually grabbing.
"""
function edgeat(timeline::Timeline, t::Real)
    seq = timeline.sequence
    fps = seq.framerate
    zone = edgezone(timeline)
    best = nothing
    bestd = Inf
    for (i, clip) in enumerate(seq.clips)
        for (side, e) in ((:left, clip.start / fps), (:right, clipend(clip) / fps))
            d = abs(t - e)
            d <= zone || continue
            better = d < bestd - 1.0e-12 ||
                     (d <= bestd + 1.0e-12 && ((t < e) == (side === :right)))
            better && (best = (i, side); bestd = d)
        end
    end
    return best
end

"Hover feedback: brighten the border of the clip under the cursor (the edge's
clip when a trim handle is grabbable), and mark that edge with a handle bar."
function hoverat!(timeline::Timeline, t::Real)
    edge = edgeat(timeline, t)
    hovered = edge !== nothing ? edge[1] :
              something(clipat(timeline.sequence, timelineframe(timeline, t)), 0)
    if hovered != timeline.hovered[]
        timeline.hovered[] = hovered
        setstates!(timeline)
    end
    edgemark!(timeline, edge)
    return nothing
end

"Show the trim-handle bar on the edge a press would grab (same `edgeat` as
the press handler, so the affordance and the action can never disagree)."
function edgemark!(timeline::Timeline, edge)
    pts = Point2f[]
    if edge !== nothing
        clip = timeline.sequence.clips[edge[1]]
        fps = timeline.sequence.framerate
        e = edge[2] === :left ? clip.start / fps : clipend(clip) / fps
        append!(pts, (Point2f(e, 0.02), Point2f(e, 0.98)))
    end
    (isempty(pts) && isempty(timeline.edgeline[])) || (timeline.edgeline[] = pts)
    return nothing
end

"Ctrl-drag: move a translucent ghost to the (snapped) drop position."
function dragto!(timeline::Timeline, t::Real)
    drag = timeline.dragclip
    drag === nothing && return nothing
    clip, offset = drag
    seq = timeline.sequence
    fps = seq.framerate
    rawstart = round(Int, t * fps) - offset

    snapframes = max(round(Int, 10 / timeline.pps[] * fps), 1)
    targets = Int[0, timeline.playhead[]]
    for other in seq.clips
        other === clip && continue
        push!(targets, other.start, clipend(other))
    end

    snapped, didsnap = snappedstart(rawstart, cliplength(clip), snapframes, targets)
    timeline.dragstart = snapped
    timeline.dragvalid = canplace(seq, clip, snapped)
    timeline.ghost_rect[] = Rect2f(snapped / fps, 0.05, cliplength(clip) / fps, 0.9)
    timeline.ghost_color[] = timeline.dragvalid ? (timeline.colors.accent_subtle, 0.55) :
                             (RGBf(0.75, 0.2, 0.2), 0.4)
    timeline.ghost_plot.visible = true
    timeline.snapline[] = didsnap && timeline.dragvalid ? [snapped / fps] : Float64[]
    return nothing
end

"""
Edge trim: dragging a clip border adjusts its in/out point. The left edge
shifts `start` and `src_in` together (content stays anchored); the right
edge moves `src_out`. Clamped to the source length and the neighbors.
"""
function trimto!(timeline::Timeline, t::Real)
    trim = timeline.trimclip
    trim === nothing && return nothing
    clip, side, i = trim
    seq = timeline.sequence
    fps = seq.framerate
    n = round(Int, t * fps)
    if side === :right
        maxend = clip.start + (clip.source.nframes - clip.src_in)
        i < length(seq.clips) && (maxend = min(maxend, seq.clips[i + 1].start))
        newend = clamp(n, clip.start + 1, maxend)
        clip.src_out = clip.src_in + (newend - clip.start)
    else
        minstart = max(i > 1 ? clipend(seq.clips[i - 1]) : 0,
                       clip.start - clip.src_in)  # src_in must stay ≥ 0
        newstart = clamp(n, minstart, clipend(clip) - 1)
        delta = newstart - clip.start
        clip.src_in += delta
        clip.start += delta
    end
    timeline.clipranges[i][] = (clip.start / fps, clipend(clip) / fps)
    e = (side === :right ? clipend(clip) : clip.start) / fps
    timeline.edgeline[] = Point2f[Point2f(e, 0.02), Point2f(e, 0.98)]  # handle follows
    notify(timeline.playhead)  # live preview while trimming
    return nothing
end

"Commit (or cancel) a clip drag on mouse release."
function finishdrag!(timeline::Timeline)
    drag = timeline.dragclip
    drag === nothing && return nothing
    clip, _ = drag
    timeline.dragclip = nothing
    timeline.ghost_plot.visible = false
    timeline.snapline[] = Float64[]
    if timeline.dragvalid && timeline.dragstart != clip.start
        timeline.onedit()
        clip.start = max(timeline.dragstart, 0)
        sort!(timeline.sequence.clips, by = c -> c.start)
        timeline.selected[] = something(findfirst(c -> c === clip, timeline.sequence.clips), 0)
        notify(timeline.playhead)  # frame under the playhead may have changed
    end
    relayout!(timeline)
    return nothing
end
