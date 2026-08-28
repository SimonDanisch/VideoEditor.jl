"""
    Timeline(gridpos, sequence, playhead, playing)

Zoomable timeline over an edited `Sequence`. Each clip is one [`ClipView`](@ref)
plot (band + composed thumbnail strip); the timeline reconciles plots with
the clip list on edits and feeds shared view observables — zooming or
panning only updates `viewrange`/`pixelspersecond` and the recipes recompute
their strips themselves.

Interaction, and **where the playhead lives is the first thing to know**: the
strip above the lanes ([`SCRUBBAND`](@ref)) scrubs, always. Pressing a CLIP
selects it and leaves the playhead alone — picking the thing you want to work on
must not move the frame you are working at. Empty lane space still scrubs, having
nothing else to mean.

The rest: Ctrl+left-drag moves the clip with a translucent ghost (snap lines, red
tint when the drop would overlap; commits on release), scroll zooms (x only),
right-drag pans, right-click calls `onrightclick(time)`. A time tooltip follows
the cursor; hover brightens a clip's border. Clip edges are trim handles:
hovering one shows a handle bar, dragging it adjusts the in/out point (the
tooltip switches to the clip's length). A lane's TOP EDGE is a resize grip —
dragging it makes that track taller, and the height is saved with the project
(see [`settrackheight!`](@ref)).
"""
# ---- track geometry (shared by the timeline, drag targeting, the media-bin drop
# ghost and the keyframe overlay): lanes fill axis-y `TRACKBASE`..0.86, with an
# ALWAYS-VISIBLE "+ new track" drop zone at each end — above the top lane, and
# below the bottom one. Stacking upward only was half an editor: a clip that
# belongs UNDER everything had to be added on top and then every other clip moved.
"Axis-y where the lanes start; below it is the drop zone for a track UNDERNEATH."
const TRACKBASE = 0.09

"Axis-y the lanes end at; above it is the drop zone for a track ON TOP."
const TRACKTOP = 0.86

"""
The SCRUB STRIP: the band where the playhead can always be moved, whatever is
selected and whatever tool is up.

It sits ABOVE everything rather than taking a slice out of the lanes, which is
why the axis runs past 1. Dragging the playhead used to be the default meaning of
a press anywhere in the timeline, so selecting a clip to work on it moved the
playhead too — and while trimming or dragging you had to keep clear of a gesture
you never wanted. Now the two live in different places: this strip scrubs, the
lanes edit.
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
Where anything hidden goes: a band above the axis' own top, so it costs nothing
to draw and cannot be hit. Solo parks the other lanes AND the "+ new track"
strips here.

A zero-height band at `TRACKTOP` would do neither — the clip's border stays as a
line across the lanes, with a degenerate filmstrip image behind it.
"""
const HIDDENBAND = (AXISTOP + 0.4, AXISTOP + 0.5)

"""
The band a SOLOED lane fills: everything under the scrub strip.

The "+ new track" drop zones go off screen with the other lanes — adding a track
is not what you are doing while one of them is blown up to fill the window, and
keeping their strips would leave the lane pinched between two captions about
tracks that are not visible.
"""
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

`ntr + 1` is a new track ABOVE the stack (the zone over the top lane) and **0** is
a new track UNDERNEATH it (the zone below the bottom lane) — the two ends of the
same gesture, so building a stack downward costs a drag rather than a rebuild.
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

The track whose TOP edge `y` is within `grab` of — the grip that resizes it.

Only the **dividers** (`1:ntr-1`): the top lane's own upper edge is the stack's
boundary, not a border between two lanes, and there is nothing above it to trade
height with. The bottom lane's base is the drop zone's boundary and stays out of
it for the same reason.
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

Give `track` the WHOLE lane area and take the others off screen, or pass `0` to
put the stack back. Toggled by double-clicking a lane.

The scrub strip is untouched, which is the point: a curve you are editing gets
the full height AND you can still put the playhead anywhere. Nothing is written
to the file — see `Sequence.solo`.
"""
function solotrack!(seq::Sequence, track::Integer)
    seq.solo = (1 <= track <= ntracks(seq)) ? Int(track) : 0
    return seq
end

"""
    lanescale(seq, ntr) -> factor >= 1

How much taller the timeline panel has to be for the BIGGEST lane to keep the
pixels an equal share would have given it.

Making a lane taller is asking for ROOM TO WORK IN, so the panel gives up the
space and the other lanes keep their pixels. Held inside the fixed row instead,
"taller" only ever meant "squashes its neighbour" — with two tracks the one you
enlarged gained about as much as the other lost, which is not what the gesture
promises.
"""
function lanescale(seq::Sequence, ntr::Integer)
    ntr >= 1 || return 1.0
    tot = totalweight(seq, ntr)
    tot > 0 || return 1.0
    return max(1.0, ntr * maximum(t -> trackweight(seq, t), 1:ntr) / tot)
end

"""
    settrackedge!(seq, track, y, ntr) -> seq

Put the divider above `track` AT axis-y `y` — where the cursor is.

Lane heights are weights, so this is a solve rather than an offset. With `u` the
share of the stack below the edge and `S` the weights above and below the dragged
lane, `hi(w) = TRACKBASE + H (S_below + w) / (S_below + w + S_above)` inverts to
`w = u S_above / (1 - u) - S_below`.

**Adding the drag to the weight instead — the obvious version — moves the edge by
about a third of the cursor's travel**, because the normalisation hands most of
what the lane gains straight back to its neighbours. Measured in the walkthrough:
27 px of drag, 11 px of edge. A grip that lags the hand like that reads as broken
long before anyone works out why.
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
    const scrubbing::Base.RefValue{Bool}  # a DRAG is under way (the press moved) — a
                                          # click alone is not scrubbing: the preview
                                          # then settles on the exact frame instead of
                                          # showing decode stand-ins (see `atrest`)
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
    ontrimpreview::Function                     # (clip, srcframe) while trimming:
                                                # show the EDGE frame, playhead untouched
    ontrimend::Function                         # …and the drag is over: stop chasing it,
                                                # or the retry publishes the edge frame
                                                # back over the playhead on release
    onlayout::Function                          # the lanes' TOTAL height changed — the
                                                # panel gives up the space (see the row
                                                # rule in the player). NOT the playhead:
                                                # notifying that per mouse move rebuilds
                                                # the inspector on every pixel of a drag
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
        # TICKS ON TOP, right against the scrub strip: the numbers and the band you
        # scrub in are then one ruler, the way every editor puts it. With the ticks
        # underneath, the one place the playhead can be moved had no scale next to
        # it and the scale you read had nothing to do with the gesture.
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
                       Any[], Observable{Tuple{Float64, Float64}}[],
                       Observable{Float64}[], Observable{Symbol}[], Any[],
                       Observable(0), Observable(Int[]), Ref(0),
                       Observable(Rect2f(0, 0, 0, 0)), Observable{Any}(colors.accent_subtle),
                       Observable(Float64[]), Observable(Point2f[]),
                       Observable(""), Observable(Point2f(0, 0)),
                       Threads.Atomic{Bool}(true),
                       nothing, nothing, nothing, 0, 1, false, nothing, identity, identity,
                       (_, _) -> nothing, () -> nothing, () -> nothing)
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
        # drag ghost, and a PERMANENT drop-zone strip above the lanes (dotted
        # outline + caption; fills accent while a drag is in flight) — adding a
        # track must be visible before you know the gesture, not only during it
        timeline.presspick = nothing
        timeline.resizetrack = nothing
        timeline.lastclick = nothing
        # THE SCRUB STRIP — see `SCRUBBAND`. Drawn as its own band above the lanes
        # with the playhead's own colour, so the one place that moves the playhead
        # is also the one place that looks like the playhead.
        timeline.scrubband = Observable(Rect2f(0, SCRUBBAND[1], 1, SCRUBBAND[2] - SCRUBBAND[1]))
        scrubfill = poly!(axis, timeline.scrubband;
                          color = (colors.accent, 0.13), strokewidth = 0)
        translate!(scrubfill, 0, 0, 2)
        scrubedge = lines!(axis, map(r -> Point2f[(r.origin[1], r.origin[2]),
                                                  (r.origin[1] + r.widths[1], r.origin[2])],
                                     timeline.scrubband);
                           color = (colors.accent, 0.5), linewidth = 1.0)
        translate!(scrubedge, 0, 0, 3)
        # …and it says so. The gesture MOVED — a strip that looks like decoration
        # is a strip nobody presses. Anchored to the band's own rect, so it rides
        # the left edge as the view pans without a second observable to keep in step.
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
        # STROKED. "+ new track" sits on a plain empty band and reads fine bare,
        # but the same label also carries the "lane taken" message — and that one
        # lands on top of a clip's filmstrip, where unstroked text is unreadable.
        # One outline makes both legible on whatever they happen to cover.
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

        # the SAME zone below the lanes: a stack you can only grow upward is half
        # an editor — putting a clip under everything meant adding it on top and
        # moving every other clip out of its way
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
            # on-screen height of the WHOLE lane stack; each clip takes its lane's
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

A source with no cache still needs a SIZE — the filmstrip's tile pitch is
`bandheight * (w/h) / pixelspersecond`, so a zero here divides the lane into
infinitely many tiles and `floor(Int, NaN)` throws inside the plot. 16:9 is the
honest stand-in: there are no thumbnails to be the wrong shape, and the band is
drawn as one plain block at that pitch.
"""
thumbdims(cache) = (cache.thumbwidth, cache.thumbheight)
thumbdims(::Nothing) = (16, 9)

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
                        thumbsize = thumbdims(cache))
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
    g = min(0.02, trackspan(ntr) * 0.15)    # gap between stacked tracks (of an EQUAL lane:
                                            # a resized one keeps the same gutter)
    for (i, clip) in enumerate(seq.clips)
        timeline.clipranges[i][] = (clip.start / fps, clipend(clip) / fps)
        timeline.clipstarts[i][] = clip.src_in / clip.source.framerate
        # higher track sits higher up the axis (on top)
        lo, hi = trackband(seq, clip.track, ntr)
        timeline.clipplots[i].bandlo = lo + g
        timeline.clipplots[i].bandhi = hi - g
        # the filmstrip's tile pitch is per LANE, not per stack: a resized track
        # gets bigger frames, not the same ones pulled tall
        timeline.clipplots[i].bandshare = (hi - lo) / lanearea(seq)
        if timeline.plotsources[i] !== clip.source  # edits shift clips across plots
            timeline.plotsources[i] = clip.source
            cache = cachefor(timeline, clip.source)
            timeline.clipplots[i].thumbs = thumbsfor(timeline, clip.source)
            timeline.clipplots[i].thumbsize = thumbdims(cache)
        end
    end
    timeline.ntr[] = ntr
    updatezones!(timeline)
    updatetracklabels!(timeline)
    prunetransitions!(seq)
    refreshtransitions!(timeline)
    cliplimits!(timeline)
    setstates!(timeline)
    return nothing
end

"""
The "+ new track" drop strips, above the lanes and below them — pinned to the
view's left edge, and OFF SCREEN while a lane is soloed: the soloed lane fills
their space (see [`SOLOBAND`](@ref)), and a caption about adding a track next to
one blown-up lane is an offer for a gesture that has nowhere to land.
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
        # the badge rides its lane's OWN band — with per-track heights the equal
        # share it used to be computed from is not where the lane is. A soloed
        # lane says so, and the hidden ones' badges go off screen with them.
        lo, hi = trackband(seq, k, ntr)
        # the way OUT of solo has to be on screen: the other lanes are gone, and
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
    for i in eachindex(timeline.clipstates)
        state = i == timeline.selected[] || i in timeline.selection[] ? :selected :
                i == timeline.hovered[] ? :hovered : :idle
        timeline.clipstates[i][] = state
    end
    return nothing
end

# ------------------------------------------------------------------ mouse

function wiretimelinemouse(timeline::Timeline, playhead::Observable{Int};
                           doubleclick::Real = 0.4)
    axis, seq = timeline.axis, timeline.sequence
    # DOUBLE-CLICK A LANE = SOLO IT: that lane fills the whole stack, the others go
    # off screen. The scrub strip sits above the lanes and is untouched, so the
    # playhead is still yours. Double-click again to put the stack back.
    #
    # ABOVE THE KEYFRAME OVERLAY (priority 20), which consumes a press that lands
    # on a ◆ — and a lane carpeted in anchors is exactly the one worth blowing up,
    # so at the timeline's own priority the gesture never arrived. Everything else
    # passes straight through: this handler consumes the second click and nothing
    # else.
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
                timeline.onlayout()
                relayout!(timeline)
                return Consume(true)
            end
            timeline.lastclick = (time(), Float64(t), Float64(ypos))
        elseif event.action == Mouse.release && timeline.lastclick !== nothing
            # A CLICK THAT MOVED IS NOT A CLICK. Without this, a press, a drag and
            # a press back at the start counted as a double-click — scrub away from
            # a clip, touch it again, and the lane soloed itself.
            lc = timeline.lastclick
            (abs(t - lc[2]) * timeline.pps[] < 6 && abs(ypos - lc[3]) < grabzone(timeline)) ||
                (timeline.lastclick = nothing)
        end
        return Consume(false)
    end
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
            t, ypos = mouseposition(axis.scene)
            n = timelineframe(timeline, t)
            # THE STRIP SCRUBS, ALWAYS — whatever is selected, whatever is up.
            if inscrubband(ypos)
                timeline.presspick = (0, Float64(t), Float64(ypos))
                n == playhead[] || (playhead[] = n)
                return Consume(true)
            end
            # …and a lane's top edge is the grip that makes it taller
            let ntr = ntracks(seq)
                te = trackedgeat(seq, ypos, ntr; grab = grabzone(timeline))
                if te !== nothing
                    timeline.onedit()
                    timeline.resizetrack = (te, Float64(ypos) - trackband(seq, te, ntr)[2])
                    return Consume(true)
                end
            end
            # the clip you POINT AT: on stacked lanes that is the one in the band
            # under the cursor, not the topmost — the preview shows the upper clip,
            # but the lower one has to be selectable (and thus editable) too
            lane = clipat(seq, n, trackat(seq, ypos, ntracks(seq)))
            i = lane === nothing ? clipat(seq, n) : lane
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
            elseif i === nothing
                # EMPTY LANE SPACE still scrubs: there is nothing to edit there, so
                # the gesture has only one sensible meaning. `presspick` says the
                # button is down; `scrubbing` turns on at the first move, so a click
                # gets the exact frame instead of the decoder's stand-ins.
                timeline.presspick = (0, Float64(t), Float64(ypos))
                n == playhead[] || (playhead[] = n)
            end
            # ON A CLIP: select it and leave the playhead where it is. Pressing a
            # clip used to scrub as well, so picking the thing you wanted to work
            # on moved the frame you were working at — and every trim or drag
            # started with a jump you had to undo by scrubbing back. The strip
            # above the lanes is where a deliberate scrub lives now.
            return Consume(true)
        elseif event.action == Mouse.release
            timeline.scrubbing[] = false
            timeline.presspick = nothing
            if timeline.resizetrack !== nothing
                timeline.resizetrack = nothing
                # THE PANEL GROWS ON RELEASE, not per mouse move: the row's height
                # decides how many pixels an axis-y is, so growing it mid-drag pulls
                # the divider out from under the cursor it is supposed to follow.
                timeline.onlayout()
            end
            finishdrag!(timeline)
            if timeline.trimclip !== nothing
                timeline.trimclip = nothing
                # BEFORE the notify: the trim preview retries in the background
                # until the exact frame decodes, and a retry that outlives the drag
                # lands the edge frame on top of the playhead one.
                timeline.ontrimend()
                relayout!(timeline)
                notify(timeline.playhead)   # the preview returns to the playhead frame
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
                limits!(axis, lims0[1] - dt, lims0[2] - dt, 0.0, AXISTOP)
            return Consume(true)
        end
        if timeline.resizetrack !== nothing
            # DRAGGING A DIVIDER: the edge goes where the cursor is, which for
            # weights is a solve — see `settrackedge!`.
            t, grab = timeline.resizetrack
            settrackedge!(seq, t, mouseposition(axis.scene)[2] - grab, ntracks(seq))
            relayout!(timeline)
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
            # NO implicit conversion into a clip move. A press without Ctrl is a
            # scrub and stays one, however far the cursor wanders.
            #
            # It used to convert: a press that grabbed a clip and then left that
            # clip's lane became a drag, so a clip could be lifted to another
            # track with no modifier. That gesture overlaps the scrub exactly —
            # both are "press on the timeline and move" — and only how far you
            # strayed told them apart. Two thresholds were added to separate them
            # and it still fired by accident, which is what a heuristic over an
            # ambiguous gesture does.
            #
            # Ctrl is the drag modifier (see the press handler) and now the only
            # one: without it the playhead follows the cursor and nothing in the
            # sequence can move.
            t = mp[1]
            n = timelineframe(timeline, t)
            timeline.scrubbing[] = true    # the press became a drag
            n == playhead[] || (playhead[] = n)
        elseif inside
            mp = mouseposition(axis.scene)
            hoverat!(timeline, mp[1], mp[2])
        end
        if inside
            t = clamp(mouseposition(axis.scene)[1], 0.0, seqduration(seq))
            trim = timeline.trimclip
            timeline.tooltip_text[] = trim === nothing ? timestring(t) :
                                      "clip " * timestring(cliplength(trim[1]) / seq.framerate)
            # the time readout rides IN the scrub strip — that band is the ruler now,
            # and hung under it the text landed on the "+ new track" caption
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

"""
    grabzone(timeline; px = 6) -> axis-y half-width of the lane-edge grip

A grab zone is a GESTURE, so it has to be a constant number of PIXELS. Written as
a constant in axis units it silently depends on the window: the timeline axis is
around 200 px tall here, where [`trackedgeat`](@ref)'s own fallback works out to
some six pixels — but on a taller window the same number is a grip nobody can hit.
"""
function grabzone(timeline::Timeline; px::Real = 6)
    return AXISTOP * px / max(timeline.axis.scene.viewport[].widths[2], 1)
end

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
clip when a trim handle is grabbable), and mark that edge with a handle bar.

With `y`, a lane's top edge takes precedence and shows the resize grip instead —
the same order the press handler decides in, so what is drawn and what a press
does can never disagree."
function hoverat!(timeline::Timeline, t::Real, y::Real)
    seq = timeline.sequence
    te = trackedgeat(seq, y, ntracks(seq); grab = grabzone(timeline))
    if te !== nothing
        laneedgemark!(timeline, te)
        timeline.hovered[] == 0 || (timeline.hovered[] = 0; setstates!(timeline))
        return nothing
    end
    return hoverat!(timeline, t)
end

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

"The lane-resize grip where a press would grab it: a bar across the view at that
lane's top edge. It shares the trim handle's plot — the two grips live in
different directions and can never both be under the cursor."
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
    # above the top lane targets a NEW track
    ntr = ntracks(seq)
    track = isnan(y) ? clip.track : trackat(seq, y, ntr)
    # A drop onto an occupied lane is REFUSED, not relocated.
    #
    # It used to ride upward looking for a free lane, which made moving a clip
    # DOWN impossible in the one arrangement where you always want to: a clip
    # stacked over another. Aiming at the lane below finds it occupied, rides back
    # up to the lane the clip is already on — a clip never blocks itself — and
    # reports `dragvalid = true`, so the ghost went green, the release committed,
    # and nothing moved. With the lanes above full it walked off the top into the
    # new-track zone, which is why that looked like the only drop target there was.
    #
    # The two new-track zones stay always-valid by construction: 0 is a new lane
    # below the stack, anything past `ntr` a new lane above it.
    timeline.dragstart = snapped
    timeline.dragtrack = track
    timeline.dragvalid = track == 0 || track > ntr || canplace(seq, clip, snapped, track)
    timeline.dragactive[] = true

    # the ghost sits IN the zone it targets. Drawing it in the band it will occupy
    # AFTER the relayout put it straight on top of the current bottom lane, which
    # reads as "it will land on V1" — the opposite of what the drop does.
    n2 = max(ntr, track)
    g = min(0.02, trackspan(n2) * 0.15)
    lo, hi = track == 0 ? (0.008, TRACKBASE - 0.013) : trackband(seq, track, n2)
    lo += g; hi -= g
    timeline.ghost_rect[] = Rect2f(snapped / fps, lo, cliplength(clip) / fps, hi - lo)
    timeline.ghost_color[] = timeline.dragvalid ? (timeline.colors.accent_subtle, 0.55) :
                             (RGBf(0.75, 0.2, 0.2), 0.4)
    timeline.ghost_plot.visible = true
    timeline.snapline[] = didsnap && timeline.dragvalid ? [snapped / fps] : Float64[]
    # SAY WHY, where the eye already is. The red ghost says "no" and nothing said
    # what would work — which is exactly the confusion that produced "seems like
    # only new track is a drop target": aiming at an occupied lane looked like the
    # lane simply refused clips, rather than refusing THIS one HERE. One label
    # carries both messages, so there is never more than one hint on screen.
    if !timeline.dragvalid
        lo2, hi2 = trackband(seq, track, n2)
        timeline.newtrackpos[] = Point2f(snapped / fps + cliplength(clip) / fps / 2,
                                         (lo2 + hi2) / 2)
        # a VECTOR, matching how the plot was created: Makie type-locks an
        # attribute scalar-vs-vector at creation, and assigning a bare String to
        # one that started as `["+ new track"]` is the trap that costs an hour.
        timeline.newtrackplot.text[] = ["lane taken here — slide along, or drop on a new track"]
        timeline.newtrackplot.color[] = RGBf(1.0, 0.72, 0.68)
        timeline.newtrackplot.visible = true
    # say it, don't imply it: dropping above the top lane creates a NEW track
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
    timeline.clipranges[i][] = (clip.start / fps, clipend(clip) / fps)
    # the STRIP has to follow too: trimming the left edge walks `src_in`, so the
    # thumbnails must start at the new in-point. Updating only the time range drew
    # the old head under a shrinking band — it looked as if the far end were being
    # cut instead of the one under the cursor.
    timeline.clipstarts[i][] = clip.src_in / clip.source.framerate
    e = (side === :right ? clipend(clip) : clip.start) / fps
    timeline.edgeline[] = Point2f[Point2f(e, 0.02), Point2f(e, 0.86)]  # handle follows
    # show the frame AT THE EDGE, not the one under the playhead: while dragging `[`
    # or `]` the question is which frame the cut lands on, and the playhead has no
    # business moving for that
    timeline.ontrimpreview(clip, side === :right ? clip.src_out - 1 : clip.src_in)
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
        if timeline.dragtrack == 0        # the zone below the bottom lane
            pushtracksup!(timeline.sequence)
            clip.track = 1
            compacttracks!(timeline.sequence)   # no empty lane where it came from
        else
            clip.track = timeline.dragtrack
        end
        sort!(timeline.sequence.clips, by = c -> (c.track, c.start))
        timeline.selected[] = something(findfirst(c -> c === clip, timeline.sequence.clips), 0)
        notify(timeline.playhead)  # frame under the playhead may have changed
    end
    relayout!(timeline)
    return nothing
end
