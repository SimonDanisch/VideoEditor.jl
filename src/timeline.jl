"""
    Timeline(gridpos, sequence, playhead, playing)

Zoomable timeline over an edited `Sequence`. Each clip is one [`ClipView`](@ref)
plot (band + composed thumbnail strip); the timeline reconciles plots with
the clip list on edits and feeds shared view observables — zooming or
panning only updates `viewrange`/`pixelspersecond` and the recipes recompute
their strips themselves.

Interaction: the strip above the lanes ([`SCRUBBAND`](@ref)) always scrubs.
Pressing a clip selects it without moving the playhead; empty lane space scrubs.
Ctrl+left-drag moves the clip with a translucent ghost (snap lines, red tint when
the drop would overlap; commits on release), scroll zooms (x only), right-drag
pans, right-click opens the clip menu ([`openclipmenu!`](@ref)). A time tooltip
follows the cursor;
hover brightens a clip's border. Clip edges are trim handles: hovering one shows
a handle bar, dragging it adjusts the in/out point (the tooltip switches to the
clip's length). A lane's top edge is a resize grip — dragging it makes that track
taller, and the height is saved with the project (see [`settrackheight!`](@ref)).
"""
# ---- track geometry, shared by the timeline, drag targeting, the media-bin drop
# ghost and the keyframe overlay: lanes fill axis-y `TRACKBASE`..`TRACKTOP`, with
# a "+ new track" drop zone at each end so a stack can be built in either
# direction.
"Axis-y where the lanes start; below it is the drop zone for a track underneath."
const TRACKBASE = 0.09

"Axis-y the lanes end at; above it is the drop zone for a track on top."
const TRACKTOP = 0.86

"""
The scrub strip: the band where the playhead can always be moved, whatever is
selected and whatever tool is up.

It sits above the lanes rather than taking a slice out of them, which is why the
axis runs past 1. Scrubbing and editing are then separate gestures — a press in
the lanes selects, trims or drags without touching the playhead.
"""
const SCRUBBAND = (1.0, 1.10)
const AXISTOP = 1.12


"How tall track `track` is drawn relative to its neighbours; 1.0 unless resized."
trackweight(seq::Sequence, track::Integer) =
    (1 <= track <= length(seq.trackheights)) ? max(seq.trackheights[track], 0.15) : 1.0

"Axis-y the lanes have between them: the whole stack, or a soloed lane's band."
lanearea(seq::Sequence) = seq.solo == 0 ? TRACKTOP - TRACKBASE : SOLOBAND[2] - SOLOBAND[1]

"Sum of the weights of `ntr` tracks — the unit the lane heights divide."
totalweight(seq::Sequence, ntr::Integer) = sum(trackweight(seq, t) for t in 1:max(ntr, 1))

"Vertical share of the axis one lane takes with `ntr` stacked tracks, unweighted."
trackspan(ntr::Integer) = (TRACKTOP - TRACKBASE) / max(ntr, 1)

"…and what `track` actually gets once the tracks have their own heights."
trackspan(seq::Sequence, track::Integer, ntr::Integer) =
    (TRACKTOP - TRACKBASE) * trackweight(seq, track) / totalweight(seq, ntr)

"`(lo, hi)` axis-y band of `track` (1 = bottom) out of `ntr` equally tall lanes."
function trackband(track::Integer, ntr::Integer)
    s = trackspan(ntr)
    lo = TRACKBASE + (track - 1) * s
    return (lo, lo + s)
end

"""
Where hidden content goes: a band past the axis' top, outside the visible range
and out of hit testing. Solo parks the other lanes and the "+ new track" strips
here. A zero-height band at `TRACKTOP` would still draw the clip border as a line
across the lanes.
"""
const HIDDENBAND = (AXISTOP + 0.4, AXISTOP + 0.5)

"The band a soloed lane fills: everything under the scrub strip."
const SOLOBAND = (0.02, 0.99)

"`(lo, hi)` axis-y band of `track`, honouring the sequence's own track heights —
or, while one track is [`solotrack!`](@ref)ed, that one filling the whole stack."
function trackband(seq::Sequence, track::Integer, ntr::Integer)
    if seq.solo != 0
        return track == seq.solo ? SOLOBAND : HIDDENBAND
    end
    tot = totalweight(seq, ntr)
    lo = TRACKBASE
    for t in 1:(track - 1)
        lo += (TRACKTOP - TRACKBASE) * trackweight(seq, t) / tot
    end
    return (lo, lo + (TRACKTOP - TRACKBASE) * trackweight(seq, track) / tot)
end

"""
Track a drop at axis-y `y` targets, out of `ntr` lanes.

`ntr + 1` is a new track above the stack (the zone over the top lane), `0` a new
track underneath it (the zone below the bottom lane).
"""
function trackat(y::Real, ntr::Integer)
    yy = Float64(y)
    yy < TRACKBASE && return 0
    return clamp(floor(Int, (yy - TRACKBASE) / trackspan(ntr)) + 1, 1, ntr + 1)
end

"…and the same with the sequence's own track heights, which is what a drop on a
resized stack has to use."
function trackat(seq::Sequence, y::Real, ntr::Integer)
    yy = Float64(y)
    if seq.solo != 0                    # the soloed lane is everything under the strip
        return SOLOBAND[1] <= yy < SOLOBAND[2] ? seq.solo : 0
    end
    yy < TRACKBASE && return 0
    yy >= TRACKTOP && return ntr + 1
    for t in 1:ntr
        _, hi = trackband(seq, t, ntr)
        yy < hi && return t
    end
    return ntr + 1
end

"Whether axis-y `y` is in the [`SCRUBBAND`](@ref) — where a press always scrubs."
inscrubband(y::Real) = SCRUBBAND[1] <= Float64(y) <= SCRUBBAND[2]

"""
    trackedgeat(seq, y, ntr; grab = 0.03) -> track | nothing

The track whose top edge `y` is within `grab` of — the grip that resizes it.

Only the dividers (`1:ntr-1`): the stack's outer edges border the drop zones, not
another lane, so there is no height to trade across them.
"""
function trackedgeat(seq::Sequence, y::Real, ntr::Integer; grab::Real = 0.03)
    seq.solo == 0 || return nothing     # one lane, no divider to grab
    yy = Float64(y)
    for t in 1:(ntr - 1)
        _, hi = trackband(seq, t, ntr)
        abs(yy - hi) <= grab && return t
    end
    return nothing
end

"""
    settrackheight!(seq, track, weight) -> seq

Set one track's height weight, keeping the vector long enough to hold it. Clamped
to a band where a lane is still both grabbable and not the whole timeline.
"""
function settrackheight!(seq::Sequence, track::Integer, weight::Real)
    track >= 1 || return seq
    while length(seq.trackheights) < track
        push!(seq.trackheights, 1.0)
    end
    seq.trackheights[track] = clamp(Float64(weight), 0.25, 6.0)
    return seq
end

"""
    solotrack!(seq, track) -> seq

Give `track` the whole lane area and move the others to [`HIDDENBAND`](@ref), or
pass `0` to put the stack back. Toggled by double-clicking a lane.

The scrub strip stays where it is, so the playhead remains reachable. Solo is
view state and is not written to the file — see `Sequence.solo`.
"""
function solotrack!(seq::Sequence, track::Integer)
    seq.solo = (1 <= track <= ntracks(seq)) ? Int(track) : 0
    return seq
end

"""
    lanescale(seq, ntr) -> factor >= 1

How much taller the timeline row has to be for the largest lane to reach the
pixel height its weight asks for, without the others losing theirs.

The lane heights are shares of the row, so inside a fixed row enlarging one lane
just squashes its neighbours. The row grows instead ([`fittimelinerow!`](@ref)).
"""
function lanescale(seq::Sequence, ntr::Integer)
    ntr >= 1 || return 1.0
    tot = totalweight(seq, ntr)
    tot > 0 || return 1.0
    return max(1.0, ntr * maximum(t -> trackweight(seq, t), 1:ntr) / tot)
end

"""
    settrackedge!(seq, track, y, ntr) -> seq

Put the divider above `track` at axis-y `y`, so the edge lands under the cursor.

Lane heights are weights, so this is a solve, not an offset. With `u` the share of
the stack below the edge and `S` the weights above and below the dragged lane,
`hi(w) = TRACKBASE + H (S_below + w) / (S_below + w + S_above)` inverts to
`w = u S_above / (1 - u) - S_below`.

Adding the drag distance to the weight instead moves the edge by roughly a third
of the cursor's travel (27 px of drag, 11 px of edge), because normalisation
returns most of the gain to the neighbouring lanes.
"""
function settrackedge!(seq::Sequence, track::Integer, y::Real, ntr::Integer)
    above = sum(t -> trackweight(seq, t), (track + 1):ntr; init = 0.0)
    above > 0 || return seq            # the top lane's edge IS the stack's top
    below = sum(t -> trackweight(seq, t), 1:(track - 1); init = 0.0)
    u = clamp((Float64(y) - TRACKBASE) / (TRACKTOP - TRACKBASE), 0.02, 0.98)
    return settrackheight!(seq, track, u * above / (1 - u) - below)
end

mutable struct Timeline
    const axis::Axis
    const sequence::Sequence
    const caches::Dict{VideoSource, ThumbnailCache}
    const playhead::Observable{Int}
    const scrubbing::Base.RefValue{Bool}  # a drag is under way (the press moved). A
                                          # click alone does not set it, so the preview
                                          # settles on the exact frame instead of
                                          # showing decode stand-ins (see `atrest`)
    const colors::NamedTuple
    # shared recipe inputs
    const viewrange::Observable{Tuple{Float64, Float64}}
    const pps::Observable{Float64}
    const bandheight::Observable{Float64}  # on-screen px of the thumbnail band
    const refresh::Observable{Int}
    # one ClipView per clip
    # interaction feedback
    # BY ID, not by index. `seq.clips` is inserted into, deleted from and sorted,
    # so an index silently comes to mean another clip — which is why every edit
    # used to have to re-aim the panel and prune the marks. An id names one clip
    # for as long as the document has it, survives undo (`restoreinto!` keeps
    # ids), and a stale one simply matches nothing. 0 = none.
    const selected::Observable{UInt64}          # primary / last clicked
    const selection::Observable{Vector{UInt64}} # shift-click multi-select
    const hovered::Base.RefValue{UInt64}
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
    # The figure-row height this timeline last asked for — see `fittimelinerow!`.
    # Here rather than in a `Ref` the player closes over, because it is a fact
    # about this timeline and the function that needs it is a method now.
    rowheight::Float64
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
    newtrackzonelo::Observable{Rect2f}          # …and its mirror below them
    dragactive::Observable{Bool}                # a clip/bin drag is in flight → highlight the zone
    zonelabelpos::Observable{Point2f}           # left-anchored zone caption
    zonelabelposlo::Observable{Point2f}
    # (clipindex, t, y) of a scrub press. Only its presence is read now — Ctrl is
    # the sole way to move a clip, so a press never becomes anything but a scrub.
    presspick::Union{Nothing, Tuple{Int, Float64, Float64}}
    # (track, how far the grab landed from that lane's top edge) while a divider
    # is being dragged. The offset is what keeps the edge from jumping the few
    # pixels you were off when you grabbed it.
    resizetrack::Union{Nothing, Tuple{Int, Float64}}
    # (wall clock, t, axis-y) of the last press in the lanes — the second one on
    # the same spot within `wiretimelinemouse`'s `doubleclick` is a double-click,
    # which solos the lane
    lastclick::Union{Nothing, Tuple{Float64, Float64, Float64}}
    scrubband::Observable{Rect2f}   # the always-live playhead strip above the lanes
    gpurun::Any   # synchronous GPU-worker runner for thumbnail decoding (nothing = CPU)
    rightpress::Any   # (t, px) of a right press — release decides menu vs pan

    function Timeline(gridpos, sequence::Sequence, playhead::Observable{Int},
                      playing::Observable{Bool})
        colors = timelinecolors()
        # Ticks on top, against the scrub strip, so the scale sits next to the band
        # it applies to and the two read as one ruler.
        axis = Axis(gridpos; yzoomlock = true, ypanlock = true, yrectzoom = false,
                    xautolimitmargin = (0.0, 0.0), xgridvisible = false,
                    xaxisposition = :top, backgroundcolor = colors.background)
        hideydecorations!(axis)
        hidespines!(axis, :l, :r)
        deregister_interaction!(axis, :rectanglezoom)  # left-drag is scrubbing
        limits!(axis, 0.0, max(seqduration(sequence), 1.0), 0.0, AXISTOP)

        timeline = new(axis, sequence, Dict{VideoSource, ThumbnailCache}(),
                       playhead, Ref(false), colors,
                       Observable((0.0, max(seqduration(sequence), 1.0))),
                       Observable(100.0), Observable(72.0), Observable(0),
                       Observable(UInt64(0)), Observable(UInt64[]), Ref(UInt64(0)),
                       Observable(Rect2f(0, 0, 0, 0)), Observable{Any}(colors.accent_subtle),
                       Observable(Float64[]), Observable(Point2f[]),
                       Observable(""), Observable(Point2f(0, 0)),
                       Threads.Atomic{Bool}(true),
                       nothing, nothing, nothing, 0, 1, false, nothing, 0.0)
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
                                      align = (:center, :center),
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
        # drag ghost, and a permanent drop-zone strip above the lanes (dotted
        # outline + caption; fills accent while a drag is in flight)
        timeline.presspick = nothing
        timeline.resizetrack = nothing
        timeline.lastclick = nothing
        # The scrub strip (see `SCRUBBAND`): its own band above the lanes, in the
        # playhead's colour so the band that moves it also looks like it.
        timeline.scrubband = Observable(Rect2f(0, SCRUBBAND[1], 1, SCRUBBAND[2] - SCRUBBAND[1]))
        scrubfill = poly!(axis, timeline.scrubband;
                          color = (colors.accent, 0.13), strokewidth = 0)
        translate!(scrubfill, 0, 0, 2)
        scrubedge = lines!(axis, map(r -> Point2f[(r.origin[1], r.origin[2]),
                                                  (r.origin[1] + r.widths[1], r.origin[2])],
                                     timeline.scrubband);
                           color = (colors.accent, 0.5), linewidth = 1.0)
        translate!(scrubedge, 0, 0, 3)
        # Caption naming the gesture, anchored to the band's own rect so it rides
        # the left edge as the view pans, without a second observable to keep in step.
        scrublabel = text!(axis, map(r -> Point2f(r.origin[1], r.origin[2] + r.widths[2] / 2),
                                     timeline.scrubband);
                           text = "playhead — drag here", fontsize = 10,
                           align = (:left, :center), offset = (6, 0),
                           color = (colors.accent, 0.55))
        translate!(scrublabel, 0, 0, 3)
        timeline.ntr = Observable(1)
        timeline.tracklabelpos = Observable{Point2f}[]
        timeline.tracklabelplots = Any[]
        timeline.newtrackpos = Observable(Point2f(0, 0))
        # Stroked: this label also carries the "lane taken" message, which lands on
        # top of a clip's filmstrip where unstroked text is unreadable.
        timeline.newtrackplot = text!(axis, timeline.newtrackpos; text = "+ new track",
                                      color = colors.accent, fontsize = 12, font = :bold,
                                      align = (:center, :center), visible = false,
                                      strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5)
        translate!(timeline.newtrackplot, 0, 0, 11)
        timeline.newtrackzone = Observable(Rect2f(0, 0.875, 1, 0.115))
        timeline.newtrackzonelo = Observable(Rect2f(0, 0.005, 1, TRACKBASE - 0.01))
        timeline.dragactive = Observable(false)
        timeline.zonelabelpos = Observable(Point2f(0, 0.9325))
        timeline.zonelabelposlo = Observable(Point2f(0, TRACKBASE / 2))
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

        # the same zone below the lanes, so a track can be added underneath the
        # stack instead of on top and then moved
        zonefilllo = poly!(axis, timeline.newtrackzonelo;
                           color = map(a -> a ? (colors.accent, 0.10) : (colors.text, 0.04),
                                       timeline.dragactive))
        translate!(zonefilllo, 0, 0, 1)
        zonelinelo = lines!(axis, map(timeline.newtrackzonelo) do r
                                x0, y0 = r.origin; w, h = r.widths
                                Point2f[(x0, y0), (x0 + w, y0), (x0 + w, y0 + h),
                                        (x0, y0 + h), (x0, y0)]
                            end; linestyle = :dot, linewidth = 1,
                            color = map(a -> a ? (colors.accent, 1.0) : (colors.text, 0.3),
                                        timeline.dragactive))
        translate!(zonelinelo, 0, 0, 2)
        zonelabello = text!(axis, timeline.zonelabelposlo;
                            text = "+  new track underneath",
                            fontsize = 10, align = (:left, :center),
                            color = map(a -> a ? (colors.accent, 1.0) : (colors.text, 0.35),
                                        timeline.dragactive))
        translate!(zonelabello, 0, 0, 3)

        onany(axis.finallimits, axis.scene.viewport) do lims, vp
            x0, x1 = minimum(lims)[1], maximum(lims)[1]
            x1 > x0 || return
            timeline.viewrange[] = (x0, x1)
            timeline.pps[] = vp.widths[1] / (x1 - x0)
            # on-screen height of the whole lane stack; each clip takes its lane's
            # share of it (`bandshare`) for the tile pitch
            timeline.bandheight[] = lanearea(timeline.sequence) / AXISTOP * vp.widths[2]
            updatezones!(timeline)
            timeline.scrubband[] = Rect2f(x0, SCRUBBAND[1], x1 - x0, SCRUBBAND[2] - SCRUBBAND[1])
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
        placetracks!(timeline)
        return timeline
    end
end

"UI role colors from the theme's `:colors` group (with a dark fallback)."
function timelinecolors()
    # A theme without the group, or without one of its roles, takes the fallback;
    # a role that is there and is not a colour is a broken theme and says so.
    group = Makie.to_value(Makie.theme(:colors))
    pick(key, fallback) = Makie.to_color(group === nothing ? fallback :
                                         Makie.to_value(get(group, key, fallback)))
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
A source that has no file to scan has no thumbnail cache and never gets one.

`nothing`, not an empty cache: a cache is a decode worker plus a ring, and
starting one for a clip that renders its frames would spawn a thread to seek a
file that does not exist. The band draws as a plain block instead — see
[`thumbsfor`](@ref).
"""
cachefor(::Timeline, ::ClipSource) = nothing

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

"A source with no thumbnails asks for none: its band is a plain block."
thumbsfor(::Timeline, ::ClipSource) = _ -> nothing

"""
The thumbnail size a cache reports.

A source with no cache still needs a size: the filmstrip's tile pitch is
`bandheight * (w/h) / pixelspersecond`, so a zero divides the lane into
infinitely many tiles and `floor(Int, NaN)` throws inside the plot. 16:9 is the
stand-in — there are no thumbnails to be the wrong shape, and the band is drawn
as one plain block at that pitch.
"""
thumbdims(cache) = (cache.thumbwidth, cache.thumbheight)
thumbdims(::Nothing) = (16, 9)

function timelineframe(timeline::Timeline, t::Real)
    return clamp(round(Int, t * timeline.sequence.framerate), 0,
                 max(seqlength(timeline.sequence) - 1, 0))
end

# ------------------------------------------------------------------ layout

"""
    buildclipview!(clip) -> Union{Nothing, ClipPlot}

Draw `clip` on the timeline of the editor its sequence is open in.

Called from [`addclip!`](@ref), where a clip joins the document — whichever route
put it there. A clip in no sequence, or in one no editor has opened, has nothing
to be drawn on and gets nothing; the `Player` builds the views of the clips that
were already in the sequence when it opens it.

A no-op for a clip that already has one: `addclip!` is also how a restore puts
the clips back, and those keep the plot they are being drawn with.
"""
function buildclipview!(clip::Clip)
    clip.view === nothing || return clip.view
    player = editorof(clip)
    player === nothing && return nothing
    timeline = player.timeline
    cache = cachefor(timeline, clip.source)
    rng = Observable((0.0, 0.0))
    srcstart = Observable(0.0)
    state = Observable(:idle)
    plt = clipview!(timeline.axis, rng;
                    # The x range is the SEQUENCE's, not this clip's. With the clip
                    # contributing, writing `src_out` and inserting the other half
                    # of a split as two statements let the axis fit itself to the
                    # document as it stood BETWEEN them — measured: 0→4.0 became
                    # 0→2.06 on the write, and `cliplimits!` only ever shrinks, so
                    # it stayed there and every time→pixel figure was on half the
                    # scale. `cliplimits!` owns this range.
                    xautolimits = false,
                    viewrange = timeline.viewrange, pixelspersecond = timeline.pps,
                    bandheight = timeline.bandheight,
                    sourcestart = srcstart, state = state,
                    thumbs = thumbsfor(timeline, clip.source),
                    refresh = timeline.refresh,
                    color = timeline.colors.surface,
                    strokecolor_idle = timeline.colors.border,
                    strokecolor_hovered = timeline.colors.accent_subtle,
                    strokecolor_selected = timeline.colors.accent,
                    thumbsize = thumbdims(cache))
    clip.view = ClipPlot(plt, rng, srcstart, state)
    # …and placed, in the same breath. Creating the plot and giving it its extent
    # used to be one pass of `relayout!`; split in two, a clip built here and not
    # relayouted afterwards stays at the `(0.0, 0.0)` it was born with — drawn,
    # hit-testable and invisible.
    placeclip!(clip)
    return clip.view
end

"""
    placeclip!(clip) -> nothing

Write where `clip` is onto the plot it owns: its span, its in-point, its lane —
and the keyframe lanes drawn against it, which are placed by the same numbers.
"""
function placeclip!(clip::Clip)
    v = clip.view
    v === nothing && return nothing
    player = editorof(clip)
    player === nothing && return nothing
    timeline = player.timeline
    seq = timeline.sequence
    ntr = ntracks(seq)
    fps = seq.framerate
    g = min(0.02, trackspan(ntr) * 0.15)   # gutter, sized from an equal lane
    v.range[] = (clip.start / fps, clipend(clip) / fps)
    v.srcstart[] = clip.src_in / clip.source.framerate
    lo, hi = trackband(seq, clip.track, ntr)   # higher track sits higher up the axis
    v.plot.bandlo = lo + g
    v.plot.bandhi = hi - g
    # the filmstrip's tile pitch is per lane, not per stack, so a resized track
    # gets bigger frames instead of the same ones pulled tall
    v.plot.bandshare = (hi - lo) / lanearea(seq)
    placelanes!(timeline, clip, lo + g, hi - g)
    return nothing
end

"Take `clip`'s plot off the timeline — from wherever it is drawn, so this can be
said from `clips.jl`, where a clip leaves the document and no timeline is in reach."
function dropclipview!(clip::Clip)
    v = clip.view
    v === nothing && return nothing
    Makie.delete!(Makie.parent_scene(v.plot), v.plot)
    clip.view = nothing
    return nothing
end

"""
    placetracks!(clip) -> nothing

…said from `clips.jl`, where a clip's track is written and no timeline is in
reach. A clip in no editor has no geometry to place.
"""
function placetracks!(clip::Clip)
    player = editorof(clip)
    player === nothing && return nothing
    return placetracks!(player.timeline)
end

"""
    placetracks!(timeline) -> nothing

The STACK\'s geometry changed — a track gained or lost, resized, soloed.

Every clip\'s band is a function of it, so every clip is placed. That is not a
pass that goes looking: it is one quantity with N dependants, and the clips are
the N. Called where that quantity is written — `settrackheight!`, `settrackedge!`,
`solotrack!`, and `addclip!`/`removeclip!`, which can add or drop a lane.

A clip moving along its own lane does not come here — that is `placeclip!`, called
by the write itself. A clip changing TRACK does, because it can add or drop a lane
and every other clip's band is a share of the stack (see `Clip`\'s `setproperty!`).
"""
function placetracks!(timeline::Timeline)
    seq = timeline.sequence
    ntr = ntracks(seq)
    foreach(placeclip!, seq.clips)
    timeline.ntr[] == ntr || (timeline.ntr[] = ntr)
    updatezones!(timeline)
    updatetracklabels!(timeline)
    # …and the axis extent, which is a function of the whole set. Not the
    # transitions: their band is fixed, so the stack's geometry says nothing about
    # them — they are drawn where the LIST changes (`drawtransitions!`).
    cliplimits!(timeline)
    return nothing
end

"""
    placelanes!(timeline, clip) -> nothing

Place `clip`'s lanes from its own track band, for a lane that has just been built
— [`relayout!`](@ref) says the same thing to every clip at once and already has
the numbers, this works them out for the one clip.
"""
function placelanes!(timeline::Timeline, clip::Clip)
    seq = timeline.sequence
    ntr = ntracks(seq)
    g = min(0.02, trackspan(ntr) * 0.15)   # the gutter `relayout!` uses
    lo, hi = trackband(seq, clip.track, ntr)
    return placelanes!(timeline, clip, lo + g, hi - g)
end

"""
    placelanes!(timeline, clip, lo, hi) -> nothing

Tell the curves drawn against `clip` where it now is: its extent in seconds, how
its frames map onto the timeline, and the band between `lo` and `hi` its values
occupy.

The same statement [`relayout!`](@ref) makes to a clip's filmstrip one line
above, and made in the same place for the same reason: a clip that is moved,
trimmed or sent to another track has to take its keyframe curves with it, and
these are the numbers that say where they went.

Inset from the lane's own edges, so a value at 0 or 1 does not sit exactly on the
divider — where it reads as belonging to the neighbouring track and cannot be
grabbed without hitting the resize grip.
"""
function placelanes!(timeline::Timeline, clip::Clip, lo::Real, hi::Real)
    fps = timeline.sequence.framerate
    span = (clip.start / fps, clipend(clip) / fps)
    fm = FrameMap(clip, fps)
    inset = 0.12 * (hi - lo)
    band = (Float64(lo + inset), Float64(hi - inset))
    for fx in clip.effects, p in fx.params
        v = p.view
        v === nothing && continue
        v.lane.clipspan = span
        v.lane.framemap = fm
        v.lane.band = band
    end
    return nothing
end

"""
The "+ new track" drop strips above and below the lanes, pinned to the view's
left edge. While a lane is soloed they move to [`HIDDENBAND`](@ref): the soloed
lane fills their space (see [`SOLOBAND`](@ref)) and there is nowhere to drop.
"""
function updatezones!(timeline::Timeline)
    x0, x1 = timeline.viewrange[]
    w = x1 - x0
    if timeline.sequence.solo != 0
        timeline.newtrackzone[] = Rect2f(x0, HIDDENBAND[1], w, 0.02)
        timeline.newtrackzonelo[] = Rect2f(x0, HIDDENBAND[1], w, 0.02)
    else
        timeline.newtrackzone[] = Rect2f(x0, 0.875, w, 0.115)
        timeline.newtrackzonelo[] = Rect2f(x0, 0.005, w, TRACKBASE - 0.01)
    end
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
    seq = timeline.sequence
    for (k, pl) in enumerate(timeline.tracklabelplots)
        show = k <= ntr
        pl.visible = show
        show || continue
        # the badge rides its lane's own band: with per-track heights the equal
        # share it used to be computed from is not where the lane is. A soloed
        # lane says so, and the hidden ones' badges go off screen with them.
        lo, hi = trackband(seq, k, ntr)
        # the way out of solo has to be on screen: the other lanes are gone, and
        # nothing else says the double-click is a toggle
        pl.text[] = seq.solo == k ? "V$k · solo — double-click to show all tracks" : "V$k"
        timeline.tracklabelpos[k][] = Point2f(x0 + xpad, (lo + hi) / 2)
    end
    if seq.solo == 0
        timeline.zonelabelpos[] = Point2f(x0 + xpad, 0.9325)
        timeline.zonelabelposlo[] = Point2f(x0 + xpad, TRACKBASE / 2)
    else                                    # …the captions leave with their strips
        timeline.zonelabelpos[] = Point2f(x0 + xpad, HIDDENBAND[1])
        timeline.zonelabelposlo[] = Point2f(x0 + xpad, HIDDENBAND[1])
    end
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
    limits!(timeline.axis, max(0.0, newx1 - (x1 - x0)), newx1, 0.0, AXISTOP)
    return nothing
end

"Hover/selection feedback without touching geometry."
function setstates!(timeline::Timeline)
    sel = timeline.selected[]
    marked = timeline.selection[]
    hov = timeline.hovered[]
    for clip in timeline.sequence.clips
        v = clip.view
        v === nothing && continue
        v.state[] = (clip.id == sel || clip.id in marked) ? :selected :
                    clip.id == hov ? :hovered : :idle
    end
    return nothing
end

# ------------------------------------------------------------------ mouse

"""
    editorof(timeline) -> Union{Nothing, Editor}

The editor this timeline is in, reached through the sequence it shows.

This is what replaced `Timeline`'s five `::Function` fields — `onedit`,
`onlayout`, `ontrimpreview`, `ontrimend`, `onrightclick`, each a closure the
player pushed in after construction because the timeline could not see it. It can
now: the sequence knows its editor ([`Editor`](@ref)), so a gesture calls the
method it means by name.

`nothing` while the player is still being assembled — the timeline is alive
before the `Player` exists, and a gesture cannot arrive in that window.
"""
editorof(timeline::Timeline) = editorof(timeline.sequence)

"""
    snapshotedit!(timeline) -> nothing

Put the document on the undo stack, before a gesture in the timeline changes it.

The three gestures that mutate — resizing a lane, trimming an edge, dropping a
dragged clip — all call this first, which is what `timeline.onedit` was.
"""
function snapshotedit!(timeline::Timeline)
    ed = editorof(timeline)
    ed === nothing || snapshot!(ed)
    return nothing
end

function wiretimelinemouse(timeline::Timeline, playhead::Observable{Int};
                           doubleclick::Real = 0.4)
    axis, seq = timeline.axis, timeline.sequence
    # Double-clicking a lane solos it (again to restore the stack); the scrub strip
    # is above the lanes and unaffected.
    #
    # Priority 30, above the keyframe overlay's 20: the overlay consumes presses
    # that land on a ◆, and a lane full of anchors is the one worth soloing. This
    # handler consumes the second click of a pair and nothing else.
    # The strip is the playhead's, at a priority that says so. This was a branch
    # of the general press handler at priority 0, which is under Makie's own axis
    # interactions (1) and under every tool (25), so "the strip scrubs whatever is
    # up" held only as long as nothing above it took the press first. Measured: in
    # the suite a press in the strip reached a probe at priority 2 and never
    # arrived at 0 — the axis' rectangle-zoom machinery had it. Nothing else in
    # the editor claims this band, so nothing is being cut in front of.
    on(events(axis.scene).mousebutton; priority = 35) do event
        (event.button == Mouse.left && event.action == Mouse.press) ||
            return Consume(false)
        is_mouseinside(axis.scene) || return Consume(false)
        t, ypos = mouseposition(axis.scene)
        inscrubband(ypos) || return Consume(false)
        # `presspick` marks the button as down; `scrubbing` only turns on at the
        # first move, so a click gets the exact frame instead of a decode stand-in
        timeline.presspick = (0, Float64(t), Float64(ypos))
        n = timelineframe(timeline, t)
        n == playhead[] || (playhead[] = n)
        return Consume(true)
    end
    on(events(axis.scene).mousebutton; priority = 30) do event
        event.button == Mouse.left || return Consume(false)
        is_mouseinside(axis.scene) || return Consume(false)
        t, ypos = mouseposition(axis.scene)
        ntr = ntracks(seq)
        tr = (TRACKBASE <= ypos < TRACKTOP) ? trackat(seq, ypos, ntr) : 0
        if event.action == Mouse.press
            1 <= tr <= ntr || return Consume(false)
            last = timeline.lastclick
            if last !== nothing && time() - last[1] < doubleclick &&
               abs(t - last[2]) * timeline.pps[] < 6 && abs(ypos - last[3]) < grabzone(timeline)
                timeline.lastclick = nothing
                solotrack!(seq, seq.solo == 0 ? tr : 0)
                ed = editorof(timeline)
                ed === nothing || fittimelinerow!(ed)
                placetracks!(timeline)
                return Consume(true)
            end
            timeline.lastclick = (time(), Float64(t), Float64(ypos))
        elseif event.action == Mouse.release && timeline.lastclick !== nothing
            # A press that moved before release does not start a double-click pair;
            # otherwise scrubbing away from a clip and back solos the lane.
            lc = timeline.lastclick
            (abs(t - lc[2]) * timeline.pps[] < 6 && abs(ypos - lc[3]) < grabzone(timeline)) ||
                (timeline.lastclick = nothing)
        end
        return Consume(false)
    end
    on(events(axis.scene).mousebutton) do event
        if event.button == Mouse.right
            # right-drag pans the view (the axis interaction), so the clip menu
            # opens only on a still click, decided at release — otherwise panning
            # over a clip pans and pops the menu at once
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
                    ed = editorof(timeline)
                    ed === nothing || openclipmenu!(ed, t0)
                end
                return Consume(true)
            end
            return Consume(false)
        end
        event.button == Mouse.left || return Consume(false)
        if event.action == Mouse.press && is_mouseinside(axis.scene)
            t, ypos = mouseposition(axis.scene)
            n = timelineframe(timeline, t)
            # the strip never gets here — it is taken above, at priority 35
            # …and a lane's top edge is the grip that makes it taller
            let ntr = ntracks(seq)
                te = trackedgeat(seq, ypos, ntr; grab = grabzone(timeline))
                if te !== nothing
                    snapshotedit!(timeline)
                    timeline.resizetrack = (te, Float64(ypos) - trackband(seq, te, ntr)[2])
                    return Consume(true)
                end
            end
            # the clip in the band under the cursor, not the topmost one: the
            # preview shows the upper clip, but the lower one stays selectable
            lane = clipat(seq, n, trackat(seq, ypos, ntracks(seq)))
            i = lane === nothing ? clipat(seq, n) : lane
            # Shift+click toggles a clip in the multi-selection without scrubbing;
            # a plain click collapses the selection to the one clip
            if ispressed(axis.scene, Keyboard.left_shift | Keyboard.right_shift) && i !== nothing
                cid = seq.clips[i].id
                sel = copy(timeline.selection[])
                isempty(sel) && timeline.selected[] != 0 && timeline.selected[] != cid &&
                    push!(sel, timeline.selected[])   # extend from the primary
                j = findfirst(==(cid), sel)
                j === nothing ? push!(sel, cid) : deleteat!(sel, j)
                timeline.selection[] = sel
                timeline.selected[] = cid
                return Consume(true)
            end
            isempty(timeline.selection[]) || (timeline.selection[] = UInt64[])
            timeline.selected[] = i === nothing ? UInt64(0) : seq.clips[i].id
            edge = edgeat(timeline, axis.scene.events.mouseposition[])
            if ispressed(axis.scene, Keyboard.left_control | Keyboard.right_control) && i !== nothing
                clip = seq.clips[i]
                timeline.dragclip = (clip, n - clip.start)
                timeline.dragstart = clip.start
                timeline.dragvalid = true
            elseif edge !== nothing
                timeline.selected[] = seq.clips[edge[1]].id
                snapshotedit!(timeline)
                timeline.trimclip = (seq.clips[edge[1]], edge[2], edge[1])
            elseif i === nothing
                # empty lane space scrubs, having nothing to edit. `presspick` marks
                # the button as down; `scrubbing` only turns on at the first move,
                # so a click gets the exact frame instead of decode stand-ins
                timeline.presspick = (0, Float64(t), Float64(ypos))
                n == playhead[] || (playhead[] = n)
            end
            # on a clip: select it and leave the playhead where it is, so starting
            # a trim or drag does not move the frame being worked at
            return Consume(true)
        elseif event.action == Mouse.release
            timeline.scrubbing[] = false
            timeline.presspick = nothing
            if timeline.resizetrack !== nothing
                timeline.resizetrack = nothing
                # the row grows on release, not per mouse move: its height decides
                # how many pixels an axis-y is, so growing it mid-drag would pull
                # the divider out from under the cursor
                ed = editorof(timeline)
                ed === nothing || fittimelinerow!(ed)
            end
            finishdrag!(timeline)
            if timeline.trimclip !== nothing
                timeline.trimclip = nothing
                # before the notify: the trim preview retries in the background
                # until the exact frame decodes, and a retry outliving the drag
                # would land the edge frame on top of the playhead's
                ed = editorof(timeline)
                ed === nothing || trimend!(ed)
                placetracks!(timeline)
                notify(timeline.playhead)   # the preview returns to the playhead frame
            end
            return Consume(false)
        end
        return Consume(false)
    end

    on(events(axis.scene).mouseposition) do _
        inside = is_mouseinside(axis.scene)
        if timeline.rightpress !== nothing        # right-drag pans the view
            _, px0, lims0 = timeline.rightpress
            mp = events(axis.scene).mouseposition[]
            vp = axis.scene.viewport[]
            dt = (mp[1] - px0[1]) / max(vp.widths[1], 1) * (lims0[2] - lims0[1])
            abs(mp[1] - px0[1]) > 3 &&
                limits!(axis, lims0[1] - dt, lims0[2] - dt, 0.0, AXISTOP)
            return Consume(true)
        end
        if timeline.resizetrack !== nothing
            # dragging a divider: the edge goes where the cursor is, which for
            # weights is a solve — see `settrackedge!`
            t, grab = timeline.resizetrack
            settrackedge!(seq, t, mouseposition(axis.scene)[2] - grab, ntracks(seq))
            placetracks!(timeline)
            laneedgemark!(timeline, t)   # the grip stays under the cursor while dragging
            return Consume(true)
        end
        if timeline.dragclip !== nothing
            mp = mouseposition(axis.scene); dragto!(timeline, mp[1], mp[2])
        elseif timeline.trimclip !== nothing
            trimto!(timeline, mouseposition(axis.scene)[1])
        elseif timeline.presspick !== nothing   # button down on the ruler → moving = scrubbing
            mp = mouseposition(axis.scene)
            pk = timeline.presspick
            # A press without Ctrl is a scrub and stays one, however far the cursor
            # wanders. It used to turn into a clip drag once the cursor left the
            # clip's lane, but that gesture is the scrub — both are "press and
            # move" — and the distance thresholds separating them kept misfiring.
            # Ctrl is the only drag modifier (see the press handler).
            t = mp[1]
            n = timelineframe(timeline, t)
            timeline.scrubbing[] = true    # the press became a drag
            n == playhead[] || (playhead[] = n)
        elseif inside
            mp = mouseposition(axis.scene)
            hoverat!(timeline, mp[1], mp[2], axis.scene.events.mouseposition[])
        end
        if inside
            t = clamp(mouseposition(axis.scene)[1], 0.0, seqduration(seq))
            trim = timeline.trimclip
            timeline.tooltip_text[] = trim === nothing ? timestring(t) :
                                      "clip " * timestring(cliplength(trim[1]) / seq.framerate)
            # the time readout sits in the scrub strip, which is the ruler; below it
            # the text landed on the "+ new track" caption
            timeline.tooltip_pos[] = Point2f(t, (SCRUBBAND[1] + SCRUBBAND[2]) / 2)
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

"How far from the cursor [`edgeat`](@ref) looks for a clip, in pixels — the same
reach as the grab zone, so a press just outside a clip still finds it."
const PICKRADIUS = 12

"""
    grabzone(timeline; px = 6) -> axis-y half-width of the lane-edge grip

Constant in pixels, converted to axis units through the current viewport height.
A constant in axis units instead scales with the window: [`trackedgeat`](@ref)'s
fallback is about six pixels on the ~200 px tall axis here and a sliver on a
taller one.
"""
function grabzone(timeline::Timeline; px::Real = 6)
    return AXISTOP * px / max(timeline.axis.scene.viewport[].widths[2], 1)
end

"""
    clipof(seq, plot) -> Clip | nothing

The clip a picked plot belongs to. Picking hands back the leaf plot a recipe drew
with, so this walks up the parents to the one a clip holds as its view.
"""
function clipof(seq::Sequence, plot)
    p = plot
    while p isa Makie.Plot
        for c in seq.clips
            v = c.view
            v === nothing || v.plot !== p || return c
        end
        p = p.parent
    end
    return nothing
end

"""
    edgeat(timeline, xy) -> (clipindex, :left/:right) | nothing

The trim edge within grab range of the window position `xy`, in pixels.

Asks Makie which plot is there and maps it back to its clip, rather than looking
for a clip whose numbers put it near the cursor. Everything the old scan had to
reason about falls out of that: the lane is wherever the plot was drawn, a press
just past a clip's end still finds it (picking searches a neighbourhood), and at a
cut the clip under the cursor is the one whose edge you get — no tie-break on
which side of the boundary the cursor sits.

`pick_sorted`, not `pick`: keyframe lanes and the drag ghost are drawn OVER the
clips, so the nearest plot is often not one. The first hit that belongs to a clip
is the one being pointed at.

Being near the clip is not being near its EDGE — that is asked of the span the
clip's view was drawn with, which is the same span on screen.
"""
function edgeat(timeline::Timeline, xy)
    seq = timeline.sequence
    clip = pickclip(timeline, xy)
    clip === nothing && return nothing
    i = findfirst(c -> c === clip, seq.clips)
    i === nothing && return nothing
    x0, x1 = clip.view.range[]
    # …from `xy`, not from the live cursor: the position asked about is the one
    # answered about, which is also what lets a test say where it points.
    t = Makie.to_world(timeline.axis.scene,
                       Makie.screen_relative(timeline.axis.scene, Point2f(xy)))[1]
    dl, dr = abs(t - x0), abs(t - x1)
    min(dl, dr) <= edgezone(timeline) || return nothing  # on the clip, off its edges
    return (i, dl <= dr ? :left : :right)
end

"""
    pickclip(timeline, xy) -> Clip | nothing

The clip drawn at window position `xy`, reaching [`PICKRADIUS`](@ref) pixels to
either side ALONG THE TIME AXIS.

Horizontally only, and by three point picks rather than one square neighbourhood:
a square of twelve pixels also reaches into the lane above, so the drop zone over
the stack offered the top clip's edge, and a point in an empty lane offered the
edge of a clip one lane down. The reach exists for the other case — grabbing an
edge from just outside the clip — and that one is horizontal by nature.

`pick_sorted`, not `pick`: keyframe lanes and the drag ghost are drawn OVER the
clips, so the nearest plot at a point is often not one.
"""
function pickclip(timeline::Timeline, xy)
    seq = timeline.sequence
    for dx in (0, -PICKRADIUS, PICKRADIUS)
        at = Point2f(xy[1] + dx, xy[2])
        for (plot, _) in Makie.pick_sorted(timeline.axis.scene, at, 1)
            c = clipof(seq, plot)
            c === nothing || return c
        end
    end
    return nothing
end

"Hover feedback: brighten the border of the clip under the cursor (the edge's
clip when a trim handle is grabbable), and mark that edge with a handle bar
across that clip's lane.

A lane's top edge takes precedence and shows the resize grip instead, in the same
order the press handler decides in."
function hoverat!(timeline::Timeline, t::Real, y::Real, xy)
    seq = timeline.sequence
    te = trackedgeat(seq, y, ntracks(seq); grab = grabzone(timeline))
    if te !== nothing
        laneedgemark!(timeline, te)
        timeline.hovered[] == 0 || (timeline.hovered[] = 0; setstates!(timeline))
        return nothing
    end
    edge = edgeat(timeline, xy)
    i = edge !== nothing ? edge[1] :
        clipat(timeline.sequence, timelineframe(timeline, t))
    hovered = i === nothing ? UInt64(0) : timeline.sequence.clips[i].id
    if hovered != timeline.hovered[]
        timeline.hovered[] = hovered
        setstates!(timeline)
    end
    edgemark!(timeline, edge)
    return nothing
end

"Show the trim-handle bar on the edge a press would grab, using the same `edgeat`
as the press handler."
function edgemark!(timeline::Timeline, edge)
    pts = Point2f[]
    if edge !== nothing
        seq = timeline.sequence
        clip = seq.clips[edge[1]]
        e = edge[2] === :left ? clip.start / seq.framerate : clipend(clip) / seq.framerate
        # Across the clip's OWN lane. It used to run 0.02..0.86 — the whole stack
        # of tracks — so the handle for a cut on one lane was drawn over every
        # other lane as well, and read as belonging to whichever clip you were
        # looking at.
        lo, hi = trackband(seq, clip.track, ntracks(seq))
        append!(pts, (Point2f(e, lo), Point2f(e, hi)))
    end
    (isempty(pts) && isempty(timeline.edgeline[])) || (timeline.edgeline[] = pts)
    return nothing
end

"The lane-resize grip where a press would grab it: a bar across the view at that
lane's top edge. Shares the trim handle's plot; the two grips are horizontal and
vertical, so only one is ever under the cursor."
function laneedgemark!(timeline::Timeline, track::Integer)
    _, hi = trackband(timeline.sequence, track, ntracks(timeline.sequence))
    x0, x1 = timeline.viewrange[]
    timeline.edgeline[] = Point2f[Point2f(x0, hi), Point2f(x1, hi)]
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
    # above the top lane targets a new track
    ntr = ntracks(seq)
    track = isnan(y) ? clip.track : trackat(seq, y, ntr)
    # A drop onto an occupied lane is refused, not relocated. Searching upward for
    # a free lane made downward moves impossible: aiming below a stacked clip found
    # that lane occupied, walked back to the clip's own lane (a clip never blocks
    # itself), reported valid, and committed a move that changed nothing.
    #
    # The two new-track zones are valid by construction: 0 is a new lane below the
    # stack, anything past `ntr` a new lane above it.
    timeline.dragstart = snapped
    timeline.dragtrack = track
    timeline.dragvalid = track == 0 || track > ntr || canplace(seq, clip, snapped, track)
    timeline.dragactive[] = true

    # the ghost sits in the zone it targets. Drawing it in the band it will occupy
    # after the relayout puts it on top of the current bottom lane, which reads as
    # the opposite of what the drop does.
    n2 = max(ntr, track)
    g = min(0.02, trackspan(n2) * 0.15)
    lo, hi = track == 0 ? (0.008, TRACKBASE - 0.013) : trackband(seq, track, n2)
    lo += g; hi -= g
    timeline.ghost_rect[] = Rect2f(snapped / fps, lo, cliplength(clip) / fps, hi - lo)
    timeline.ghost_color[] = timeline.dragvalid ? (timeline.colors.accent_subtle, 0.55) :
                             (RGBf(0.75, 0.2, 0.2), 0.4)
    timeline.ghost_plot.visible = true
    timeline.snapline[] = didsnap && timeline.dragvalid ? [snapped / fps] : Float64[]
    # The red ghost only says "no", so the label says why and what would work.
    # Both messages share one plot, so at most one hint is ever on screen.
    if !timeline.dragvalid
        lo2, hi2 = trackband(seq, track, n2)
        timeline.newtrackpos[] = Point2f(snapped / fps + cliplength(clip) / fps / 2,
                                         (lo2 + hi2) / 2)
        # a vector, matching how the plot was created: Makie locks an attribute's
        # scalar-vs-vector form at creation, so a bare String does not fit here
        timeline.newtrackplot.text[] = ["lane taken here — slide along, or drop on a new track"]
        timeline.newtrackplot.color[] = RGBf(1.0, 0.72, 0.68)
        timeline.newtrackplot.visible = true
    # a drop outside the stack creates a new track; the label says so
    elseif track > ntr || track == 0
        timeline.newtrackplot.text[] = ["+ new track"]
        timeline.newtrackplot.color[] = timeline.colors.accent
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
    trimclip!(seq, clip, i, side, round(Int, t * fps))
    v = clip.view
    if v !== nothing
        v.range[] = (clip.start / fps, clipend(clip) / fps)
        # the filmstrip follows too: trimming the left edge walks `src_in`, so the
        # thumbnails start at the new in-point. Updating only the time range keeps
        # the old head under a shrinking band, which reads as trimming the far end.
        v.srcstart[] = clip.src_in / clip.source.framerate
    end
    e = (side === :right ? clipend(clip) : clip.start) / fps
    timeline.edgeline[] = Point2f[Point2f(e, 0.02), Point2f(e, 0.86)]  # handle follows
    # preview the frame at the edge, not the one under the playhead: the question
    # while trimming is which frame the cut lands on
    let ed = editorof(timeline)
        ed === nothing ||
            trimpreview!(ed, clip, side === :right ? clip.src_out - 1 : clip.src_in)
    end
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
        snapshotedit!(timeline)
        clip.start = max(timeline.dragstart, 0)
        if timeline.dragtrack == 0        # the zone below the bottom lane
            pushtracksup!(timeline.sequence)
            clip.track = 1
            compacttracks!(timeline.sequence)   # no empty lane where it came from
        else
            clip.track = timeline.dragtrack
        end
        sort!(timeline.sequence.clips, by = c -> (c.track, c.start))
        timeline.selected[] = clip.id
        notify(timeline.playhead)  # frame under the playhead may have changed
    end
    placetracks!(timeline)
    return nothing
end
