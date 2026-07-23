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
# ---- track geometry (shared by the timeline, drag targeting, the media-bin drop
# ghost and the keyframe overlay): lanes fill axis-y 0.02..0.86; the strip above
# (0.875..0.99) is the ALWAYS-VISIBLE "+ new track" drop zone.
"Vertical share of the axis one lane takes with `ntr` stacked tracks."
trackspan(ntr::Integer) = 0.84 / max(ntr, 1)

"`(lo, hi)` axis-y band of `track` (1 = bottom) out of `ntr` lanes."
function trackband(track::Integer, ntr::Integer)
    s = trackspan(ntr)
    lo = 0.02 + (track - 1) * s
    return (lo, lo + s)
end

"Track a drop at axis-y `y` targets; anything above the top lane (the marked
zone) is `ntr + 1` — a new track."
trackat(y::Real, ntr::Integer) =
    clamp(floor(Int, (Float64(y) - 0.02) / trackspan(ntr)) + 1, 1, ntr + 1)

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
    const selected::Observable{Int}          # clip index, 0 = none (primary/last clicked)
    const selection::Observable{Vector{Int}} # shift-click multi-select (clip indices)
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
    dragtrack::Int                              # target track, committed on release
    dragvalid::Bool
    trimclip::Union{Nothing, Tuple{Clip, Symbol, Int}}  # (clip, :left/:right, index)
    onrightclick::Function
    onedit::Function                            # called before a gesture mutates
    refreshtask::Task
    transbox::Observable{Vector{Rect2f}}        # cross-dissolve span boxes
    transx::Observable{Vector{Point2f}}         # the bowtie X inside each box
    transplot::Any
    transxplot::Any
    ntr::Observable{Int}                        # track count (drives lane labels)
    tracklabelpos::Vector{Observable{Point2f}}  # one V1/V2/… badge per lane
    tracklabelplots::Vector{Any}
    newtrackpos::Observable{Point2f}            # "+ new track" hint while dragging
    newtrackplot::Any
    newtrackzone::Observable{Rect2f}            # the permanent drop-zone strip above the lanes
    dragactive::Observable{Bool}                # a clip/bin drag is in flight → highlight the zone
    zonelabelpos::Observable{Point2f}           # left-anchored zone caption
    presspick::Union{Nothing, Tuple{Int, Float64, Float64}}  # (clipindex, t, y) of a scrub press —
                                                # dragging out of the lane converts it to a clip move
    gpurun::Any   # synchronous GPU-worker runner for thumbnail decoding (nothing = CPU)
    rightpress::Any   # (t, px) of a right press — release decides menu vs pan

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
                       Observable(0), Observable(Int[]), Ref(0),
                       Observable(Rect2f(0, 0, 0, 0)), Observable{Any}(colors.accent_subtle),
                       Observable(Float64[]), Observable(Point2f[]),
                       Observable(""), Observable(Point2f(0, 0)),
                       Threads.Atomic{Bool}(true),
                       nothing, nothing, nothing, 0, 1, false, nothing, identity, identity)
        timeline.gpurun = nothing
        timeline.rightpress = nothing

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
        # cross-dissolve markers: a translucent span with a bowtie X over the cut
        timeline.transbox = Observable(Rect2f[])
        timeline.transx = Observable(Point2f[])
        timeline.transplot = poly!(axis, timeline.transbox; color = (colors.accent, 0.20),
                                   strokecolor = colors.accent, strokewidth = 1.0)
        translate!(timeline.transplot, 0, 0, 6)
        timeline.transxplot = linesegments!(axis, timeline.transx;
                                            color = colors.accent, linewidth = 1.5)
        translate!(timeline.transxplot, 0, 0, 7)
        # multi-track affordances: V1/V2 lane badges, a "+ new track" hint over the
        # drag ghost, and a PERMANENT drop-zone strip above the lanes (dotted
        # outline + caption; fills accent while a drag is in flight) — adding a
        # track must be visible before you know the gesture, not only during it
        timeline.presspick = nothing
        timeline.ntr = Observable(1)
        timeline.tracklabelpos = Observable{Point2f}[]
        timeline.tracklabelplots = Any[]
        timeline.newtrackpos = Observable(Point2f(0, 0))
        timeline.newtrackplot = text!(axis, timeline.newtrackpos; text = "+ new track",
                                      color = colors.accent, fontsize = 12, font = :bold,
                                      align = (:center, :center), visible = false)
        translate!(timeline.newtrackplot, 0, 0, 11)
        timeline.newtrackzone = Observable(Rect2f(0, 0.875, 1, 0.115))
        timeline.dragactive = Observable(false)
        timeline.zonelabelpos = Observable(Point2f(0, 0.9325))
        zonefill = poly!(axis, timeline.newtrackzone;
                         color = map(a -> a ? (colors.accent, 0.12) : (colors.text, 0.0),
                                     timeline.dragactive), strokewidth = 0)
        translate!(zonefill, 0, 0, 2)
        zoneline = lines!(axis, map(timeline.newtrackzone) do r
                              x0, y0 = r.origin; w, h = r.widths
                              Point2f[(x0, y0), (x0 + w, y0), (x0 + w, y0 + h),
                                      (x0, y0 + h), (x0, y0)]
                          end; linestyle = :dot, linewidth = 1,
                          color = map(a -> a ? (colors.accent, 1.0) : (colors.text, 0.3),
                                      timeline.dragactive))
        translate!(zoneline, 0, 0, 2)
        zonelabel = text!(axis, timeline.zonelabelpos;
                          text = "+  new track — drop a clip here",
                          fontsize = 10, align = (:left, :center),
                          color = map(a -> a ? (colors.accent, 1.0) : (colors.text, 0.35),
                                      timeline.dragactive))
        translate!(zonelabel, 0, 0, 3)

        onany(axis.finallimits, axis.scene.viewport) do lims, vp
            x0, x1 = minimum(lims)[1], maximum(lims)[1]
            x1 > x0 || return
            timeline.viewrange[] = (x0, x1)
            timeline.pps[] = vp.widths[1] / (x1 - x0)
            timeline.bandheight[] = 0.84 * vp.widths[2]  # the lanes' share of the axis
            timeline.newtrackzone[] = Rect2f(x0, 0.875, x1 - x0, 0.115)
            updatetracklabels!(timeline)                 # badges stick to the left edge
            return
        end
        on(_ -> setstates!(timeline), timeline.selected)
        on(_ -> setstates!(timeline), timeline.selection)
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
    get!(() -> ThumbnailCache(source; gpurun = timeline.gpurun), timeline.caches, source)

"""
Switch thumbnail decoding to the GPU runner `f`. Caches created before the
player's GPU worker existed (the initial sources — `relayout!` runs inside the
Timeline constructor) restart their workers on the GPU loop; decoded thumbs
and pending requests survive the swap.
"""
function setgpurun!(timeline::Timeline, f)
    timeline.gpurun = f
    for cache in values(timeline.caches)
        stop!(cache)
        cache.gpurun = f
        cache.running[] = true
        cache.task = Threads.@spawn gputhumbloop(cache)
    end
    return nothing
end

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
    # edits shift clip indices — drop marks that no longer point at a clip
    sel = filter(i -> 1 <= i <= length(seq.clips), timeline.selection[])
    length(sel) == length(timeline.selection[]) || (timeline.selection[] = sel)
    ntr = ntracks(seq)
    g = min(0.02, trackspan(ntr) * 0.15)    # gap between stacked tracks
    for (i, clip) in enumerate(seq.clips)
        timeline.clipranges[i][] = (clip.start / fps, clipend(clip) / fps)
        timeline.clipstarts[i][] = clip.src_in / clip.source.framerate
        # higher track sits higher up the axis (on top)
        lo, hi = trackband(clip.track, ntr)
        timeline.clipplots[i].bandlo = lo + g
        timeline.clipplots[i].bandhi = hi - g
        timeline.clipplots[i].ntracks = ntr
        if timeline.plotsources[i] !== clip.source  # edits shift clips across plots
            timeline.plotsources[i] = clip.source
            cache = cachefor(timeline, clip.source)
            timeline.clipplots[i].thumbs = thumbsfor(timeline, clip.source)
            timeline.clipplots[i].thumbsize = (cache.thumbwidth, cache.thumbheight)
        end
    end
    timeline.ntr[] = ntr
    updatetracklabels!(timeline)
    prunetransitions!(seq)
    refreshtransitions!(timeline)
    cliplimits!(timeline)
    setstates!(timeline)
    return nothing
end

"Lane badges (V1, V2, …) at the left edge of each track, and the new-track-zone
caption pinned to the view's left edge."
function updatetracklabels!(timeline::Timeline)
    ntr = timeline.ntr[]
    while length(timeline.tracklabelplots) < ntr
        k = length(timeline.tracklabelplots) + 1
        pos = Observable(Point2f(0, 0))
        pl = text!(timeline.axis, pos; text = "V$k", color = (timeline.colors.text, 0.6),
                   fontsize = 11, font = :bold, align = (:left, :center))
        translate!(pl, 0, 0, 11)
        push!(timeline.tracklabelpos, pos)
        push!(timeline.tracklabelplots, pl)
    end
    (x0, _) = timeline.viewrange[]
    xpad = 8 / max(timeline.pps[], 1.0e-9)
    span = trackspan(ntr)
    for (k, pl) in enumerate(timeline.tracklabelplots)
        show = k <= ntr
        pl.visible = show
        show || continue
        timeline.tracklabelpos[k][] = Point2f(x0 + xpad, 0.02 + (k - 0.5) * span)
    end
    timeline.zonelabelpos[] = Point2f(x0 + xpad, 0.9325)
    return nothing
end

"Rebuild the cross-dissolve span boxes + bowtie Xs from `seq.transitions`."
function refreshtransitions!(timeline::Timeline)
    seq = timeline.sequence
    fps = seq.framerate
    boxes = Rect2f[]
    xs = Point2f[]
    y0, y1 = 0.06, 0.82   # inside the lane area, below the new-track zone
    for t in seq.transitions
        x0 = transstart(t) / fps
        x1 = transstop(t) / fps
        push!(boxes, Rect2f(x0, y0, x1 - x0, y1 - y0))
        push!(xs, Point2f(x0, y0), Point2f(x1, y1), Point2f(x0, y1), Point2f(x1, y0))
    end
    timeline.transbox[] = boxes
    timeline.transx[] = xs
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
        state = i == timeline.selected[] || i in timeline.selection[] ? :selected :
                i == timeline.hovered[] ? :hovered : :idle
        timeline.clipstates[i][] = state
    end
    return nothing
end

# ------------------------------------------------------------------ mouse

function wiretimelinemouse(timeline::Timeline, playhead::Observable{Int})
    axis, seq = timeline.axis, timeline.sequence
    on(events(axis.scene).mousebutton) do event
        if event.button == Mouse.right
            # right-DRAG pans the view (the axis interaction) — the clip menu
            # must only open on a STILL click, decided at release, or panning
            # over a clip would move the view AND pop the menu at once
            if event.action == Mouse.press && is_mouseinside(axis.scene)
                lims = axis.finallimits[]
                timeline.rightpress = (mouseposition(axis.scene)[1],
                                       Point2f(events(axis.scene).mouseposition[]),
                                       (minimum(lims)[1], maximum(lims)[1]))
                return Consume(true)
            elseif event.action == Mouse.release && timeline.rightpress !== nothing
                t0, px0, _ = timeline.rightpress
                timeline.rightpress = nothing
                mp = events(axis.scene).mouseposition[]
                if hypot(mp[1] - px0[1], mp[2] - px0[2]) < 4
                    timeline.onrightclick(t0)
                end
                return Consume(true)
            end
            return Consume(false)
        end
        event.button == Mouse.left || return Consume(false)
        if event.action == Mouse.press && is_mouseinside(axis.scene)
            t = mouseposition(axis.scene)[1]
            n = timelineframe(timeline, t)
            i = clipat(seq, n)
            # Shift+click MARKS clips (toggle in the multi-selection) without
            # scrubbing; a plain click collapses the marks to the one clip
            if ispressed(axis.scene, Keyboard.left_shift | Keyboard.right_shift) && i !== nothing
                sel = copy(timeline.selection[])
                isempty(sel) && timeline.selected[] > 0 && timeline.selected[] != i &&
                    push!(sel, timeline.selected[])   # extend FROM the primary
                j = findfirst(==(i), sel)
                j === nothing ? push!(sel, i) : deleteat!(sel, j)
                timeline.selection[] = sel
                timeline.selected[] = i
                return Consume(true)
            end
            isempty(timeline.selection[]) || (timeline.selection[] = Int[])
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
                # remember what the press landed on: dragging OUT of that clip's
                # lane converts the scrub into a clip move (no Ctrl needed)
                y = mouseposition(axis.scene)[2]
                timeline.presspick = (something(i, 0), Float64(t), Float64(y))
                n == playhead[] || (playhead[] = n)
            end
            return Consume(true)
        elseif event.action == Mouse.release
            timeline.scrubbing[] = false
            timeline.presspick = nothing
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
        if timeline.rightpress !== nothing        # right-drag PANS the view
            _, px0, lims0 = timeline.rightpress
            mp = events(axis.scene).mouseposition[]
            vp = axis.scene.viewport[]
            dt = (mp[1] - px0[1]) / max(vp.widths[1], 1) * (lims0[2] - lims0[1])
            abs(mp[1] - px0[1]) > 3 &&
                limits!(axis, lims0[1] - dt, lims0[2] - dt, 0.0, 1.0)
            return Consume(true)
        end
        if timeline.dragclip !== nothing
            mp = mouseposition(axis.scene); dragto!(timeline, mp[1], mp[2])
        elseif timeline.trimclip !== nothing
            trimto!(timeline, mouseposition(axis.scene)[1])
        elseif timeline.scrubbing[]
            mp = mouseposition(axis.scene)
            pk = timeline.presspick
            if pk !== nothing && pk[1] != 0 && inside
                # the press grabbed a clip and the cursor DELIBERATELY left its
                # lane → a MOVE (e.g. lifting a cut clip into the new-track
                # zone), not a scrub. Deliberate = well past the band AND well
                # below/above where the press started: a few pixels of vertical
                # wobble during a horizontal scrub or an edge trim used to
                # convert into a surprise clip move ("it moved my clip!")
                clip = seq.clips[pk[1]]
                blo, bhi = trackband(clip.track, ntracks(seq))
                if (mp[2] > bhi + 0.04 || mp[2] < blo - 0.04) && abs(mp[2] - pk[3]) > 0.18
                    timeline.scrubbing[] = false
                    timeline.presspick = nothing
                    timeline.dragclip = (clip, timelineframe(timeline, pk[2]) - clip.start)
                    timeline.dragstart = clip.start
                    timeline.dragtrack = clip.track
                    timeline.dragvalid = true
                    dragto!(timeline, mp[1], mp[2])
                    return Consume(false)
                end
            end
            t = mp[1]
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
edgezone(timeline::Timeline) = 12 / max(timeline.pps[], 1.0e-9)

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
        append!(pts, (Point2f(e, 0.02), Point2f(e, 0.86)))
    end
    (isempty(pts) && isempty(timeline.edgeline[])) || (timeline.edgeline[] = pts)
    return nothing
end

"Ctrl-drag: move a translucent ghost to the (snapped) drop position; cursor height
picks the target track (drag above the top row to create a new one)."
function dragto!(timeline::Timeline, t::Real, y::Real = NaN)
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

    # target track from cursor height (bands match relayout!); the marked zone
    # above the top lane targets a NEW track
    ntr = ntracks(seq)
    track = isnan(y) ? clip.track : trackat(y, ntr)
    timeline.dragstart = snapped
    timeline.dragtrack = track
    timeline.dragvalid = canplace(seq, clip, snapped, track)
    timeline.dragactive[] = true

    n2 = max(ntr, track)                            # if dropping on a new track, shrink to fit
    g = min(0.02, trackspan(n2) * 0.15)
    lo, hi = trackband(track, n2)
    lo += g; hi -= g
    timeline.ghost_rect[] = Rect2f(snapped / fps, lo, cliplength(clip) / fps, hi - lo)
    timeline.ghost_color[] = timeline.dragvalid ? (timeline.colors.accent_subtle, 0.55) :
                             (RGBf(0.75, 0.2, 0.2), 0.4)
    timeline.ghost_plot.visible = true
    timeline.snapline[] = didsnap && timeline.dragvalid ? [snapped / fps] : Float64[]
    # say it, don't imply it: dropping above the top lane creates a NEW track
    if track > ntr
        timeline.newtrackpos[] = Point2f(snapped / fps + cliplength(clip) / fps / 2,
                                         (lo + hi) / 2)
        timeline.newtrackplot.visible = true
    else
        timeline.newtrackplot.visible = false
    end
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
    timeline.edgeline[] = Point2f[Point2f(e, 0.02), Point2f(e, 0.86)]  # handle follows
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
    timeline.newtrackplot.visible = false
    timeline.dragactive[] = false
    timeline.snapline[] = Float64[]
    if timeline.dragvalid && (timeline.dragstart != clip.start || timeline.dragtrack != clip.track)
        timeline.onedit()
        clip.start = max(timeline.dragstart, 0)
        clip.track = timeline.dragtrack
        sort!(timeline.sequence.clips, by = c -> (c.track, c.start))
        timeline.selected[] = something(findfirst(c -> c === clip, timeline.sequence.clips), 0)
        notify(timeline.playhead)  # frame under the playhead may have changed
    end
    relayout!(timeline)
    return nothing
end
