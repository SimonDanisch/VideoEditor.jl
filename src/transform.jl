# Direct manipulation of a clip's placement, in the preview, with handles —
# rather than hunting for the right slider and guessing what a number does.
#
# The gizmo owns no state. It reads `transformof(clip)` and writes through
# `applyslider!`, the SAME call the sliders make, which is what makes the two
# agree: dragging moves the sliders, a slider moves the gizmo, one undo entry per
# gesture either way, and a keyframed parameter gets a key at the playhead instead
# of a static value — so drag-to-animate is just scrub-and-drag.

"How near a handle the pointer must be, in canvas pixels, to grab it."
const GIZMOGRAB = 14.0

"""
    layerquad(player, clip, srcframe) -> Vector{Point2f}

The clip's picture as it sits on the canvas: the four corners of its crop rect,
in preview coordinates.

Through `inv(layermatrix)`, so the quad is whatever the placement says it is —
crop, scale, position and rotation included — instead of a second opinion about
where the picture went.
"""
function layerquad(player::Player, clip::Clip, srcframe::Integer)
    lay = (clip.source.width, clip.source.height)
    can = size(player.frame[])
    # the clip AT THIS FRAME: with the transform keyframed, the static effect is
    # not where the picture is — the handles would sit on the base values and stay
    # there while the animation moved underneath them
    Minv = inv(layermatrix(effectiveclip(clip, srcframe), lay, can))
    x, y, w, h = clip.crop
    src = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    return [Point2f(begin
                        q = Minv * Vec3f(u * lay[1] + 0.5f0, v * lay[2] + 0.5f0, 1.0f0)
                        (q[1], q[2])
                    end) for (u, v) in src]
end

"The quad's centre — what scaling and rotation turn about."
quadcentre(q::Vector{Point2f}) = Point2f(sum(p[1] for p in q) / 4, sum(p[2] for p in q) / 4)

"""
Where the rotate handle sits: off the top edge's midpoint, along the quad's own
up direction, so it follows the picture round as it turns.
"""
function rotatehandle(q::Vector{Point2f})
    mid = Point2f((q[1][1] + q[2][1]) / 2, (q[1][2] + q[2][2]) / 2)
    c = quadcentre(q)
    d = Point2f(mid[1] - c[1], mid[2] - c[2])
    n = sqrt(d[1]^2 + d[2]^2)
    n < 1 && return mid
    return Point2f(mid[1] + d[1] / n * 46, mid[2] + d[2] / n * 46)
end

"Edge midpoints, in the corner order the quad uses."
edgepoints(q::Vector{Point2f}) =
    [Point2f((q[i][1] + q[mod1(i + 1, 4)][1]) / 2, (q[i][2] + q[mod1(i + 1, 4)][2]) / 2)
     for i in 1:4]

"Whether `p` is inside the (convex) quad — the move zone."
function insidequad(q::Vector{Point2f}, p)
    sgn = 0
    for i in 1:4
        a, b = q[i], q[mod1(i + 1, 4)]
        cr = (b[1] - a[1]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[1] - a[1])
        cr == 0 && continue
        s = cr > 0 ? 1 : -1
        sgn == 0 ? (sgn = s) : (s == sgn || return false)
    end
    return true
end

"""
    transformgizmo(player) -> (; scene, quad, handles, rotate, pivot)

The overlay, built once per player and reused.

A SIBLING of the preview axis, not a child: `receives_events` lets a covering
scene through to anything on its own root-to-leaf path, so a child would take the
pointer and still leave the axis' own crop/scrub handlers firing — the conflict
this exists to end. Same viewport and camera, so a handle sits where the picture
does.
"""
function transformgizmo(player::Player)
    return get!(player.fxwidgets, :transformgizmo) do
        ax = player.previewaxis
        sc = Makie.Scene(player.fig.scene; viewport = ax.scene.viewport,
                         camera = ax.scene.camera, clear = false, visible = false)
        Makie.translate!(sc, 0, 0, 12)
        colors = player.timeline.colors
        sel = get(player.fxwidgets, :uicolors, nothing)
        selcol = sel === nothing ? RGBf(0.29, 0.60, 0.98) : sel.select
        quad = Observable(Point2f[])
        handles = Observable(Point2f[])
        edges = Observable(Point2f[])
        rotate = Observable(Point2f[])
        pivot = Observable(Point2f[])
        stalk = Observable(Point2f[])
        lines!(sc, quad; color = (selcol, 0.95), linewidth = 2)
        lines!(sc, stalk; color = (selcol, 0.7), linewidth = 1.5)
        scatter!(sc, edges; color = (:white, 0.9), marker = :rect, markersize = 8,
                 strokewidth = 1, strokecolor = (:black, 0.7))
        scatter!(sc, handles; color = (:white, 0.95), marker = :rect, markersize = 11,
                 strokewidth = 1, strokecolor = (:black, 0.7))
        scatter!(sc, rotate; color = (selcol, 0.95), marker = :circle, markersize = 12,
                 strokewidth = 1, strokecolor = (:black, 0.7))
        scatter!(sc, pivot; color = (:white, 0.8), marker = :cross, markersize = 10)
        (scene = sc, quad = quad, handles = handles, edges = edges, rotate = rotate,
         pivot = pivot, stalk = stalk)
    end
end

"Redraw the gizmo for `clip`, or hide it when there is nothing to show."
function showtransformgizmo!(player::Player, clip::Union{Nothing, Clip},
                             srcframe::Union{Nothing, Integer} = nothing)
    g = transformgizmo(player)
    # The handles belong to the SELECTED card, not to the effect's mere presence.
    # Tying them to the effect meant a clip with a transform always wore handles,
    # with nothing to click to put them away.
    sel = fxselected(player)
    slot = clip === nothing || sel === nothing ? nothing :
           findfirst(s -> (:fx, s.id) == sel && op(s) isa TransformEffect, clip.effects)
    if slot === nothing
        g.scene.visible[] = false
        g.scene.captures_mouse = false
        return nothing
    end
    sf = srcframe === nothing ? (loc = editclip(player); loc === nothing ? 0 : loc[2]) : Int(srcframe)
    q = layerquad(player, clip, sf)
    c = quadcentre(q)
    r = rotatehandle(q)
    mid = Point2f((q[1][1] + q[2][1]) / 2, (q[1][2] + q[2][2]) / 2)
    g.quad[] = vcat(q, [q[1]])
    g.handles[] = q
    g.edges[] = edgepoints(q)
    g.rotate[] = [r]
    g.stalk[] = [mid, r]
    g.pivot[] = [c]
    g.scene.visible[] = true
    g.scene.captures_mouse = true
    return nothing
end

"""
What a press at `p` grabs: `:corner`, `:edge`, `:rotate`, `:move`, or `:none`.

Handles first, biggest target last: a corner sits ON the quad's outline, so
hit-testing the body before the handles would make the corners unreachable.
"""
function gizmozone(player::Player, clip::Clip, srcframe::Integer, p)
    q = layerquad(player, clip, srcframe)
    near(a) = (a[1] - p[1])^2 + (a[2] - p[2])^2 <= GIZMOGRAB^2
    near(rotatehandle(q)) && return :rotate
    any(near, q) && return :corner
    any(near, edgepoints(q)) && return :edge
    insidequad(q, p) && return :move
    return :none
end

"A gesture in flight: what was grabbed, and the state it started from."
mutable struct TransformDrag
    clip::Clip
    zone::Symbol
    start::Point2f          # where the press landed, canvas px
    centre::Point2f         # the quad's centre at press
    scale0::Float64
    pos0::Tuple{Float64, Float64}
    rot0::Float64
    dist0::Float64          # |press − centre|, for the scale ratio
    ang0::Float64           # atan of the same, for the rotation delta
end

transformdrag(player::Player) = get(player.fxwidgets, :transformdrag, nothing)

"""
Write one of the gizmo's parameters and SHOW it everywhere at once.

`applyslider!` is the sliders' own path: it takes the undo snapshot, and it
writes a KEY at the playhead when the parameter is keyframed rather than
overwriting the curve. The slider widget is then pushed to the new value under
`fxsyncing`, which is the flag that stops that echo from firing the slider's
handler straight back at us.
"""
function setgizmoparam!(player::Player, key::Symbol, v::Real)
    applyslider!(player, key, Float32(v))
    s = get(player.fxsliders, key, nothing)
    if s !== nothing
        player.fxsyncing[] = true
        Makie.set_close_to!(s, Float64(v))
        player.fxsyncing[] = false
    end
    return nothing
end

"Begin a gesture at `p`; returns whether the gizmo took the press."
function starttransformdrag!(player::Player, p)
    loc = editclip(player)
    loc === nothing && return false
    clip = loc[1]
    g = transformgizmo(player)
    g.scene.visible[] || return false        # not the selected card: not our press
    sf = loc[2]
    zone = gizmozone(player, clip, sf, p)
    zone === :none && return false
    q = layerquad(player, clip, sf)
    c = quadcentre(q)
    # the values the gesture starts from are the EFFECTIVE ones: on a keyframed
    # parameter the static value is not what is on screen, so a drag that started
    # from it jumped the moment it was picked up
    s, px, py, rot = transformof(effectiveclip(clip, sf))
    d = sqrt((p[1] - c[1])^2 + (p[2] - c[2])^2)
    player.fxwidgets[:transformdrag] =
        TransformDrag(clip, zone, Point2f(p), c, s, (px, py), rot, max(d, 1e-3),
                      atan(p[2] - c[2], p[1] - c[1]))
    setcursor!(player, zone === :rotate ? :hand : zone === :move ? :move : :resize)
    return true
end

"""
Continue the gesture — LIVE: every move writes the parameter, so the picture and
the sliders follow the pointer instead of jumping when it is let go.
"""
function transformdragto!(player::Player, p; shift::Bool = false, alt::Bool = false)
    d = transformdrag(player)
    d === nothing && return false
    can = size(player.frame[])
    if d.zone === :move
        dx = (p[1] - d.start[1]) / can[1]
        dy = (p[2] - d.start[2]) / can[2]
        if shift                            # one axis at a time
            abs(dx) > abs(dy) ? (dy = 0.0) : (dx = 0.0)
        end
        # The preview axis is y-DOWN (`limits!(…, H, 0)`), the same direction the
        # placement's position runs, so the drag delta carries straight through.
        # Negating it here — "screen y is up" — dragged the picture the opposite
        # way from the pointer.
        setgizmoparam!(player, :pos_x, clamp(d.pos0[1] + dx, -1.0, 1.0))
        setgizmoparam!(player, :pos_y, clamp(d.pos0[2] + dy, -1.0, 1.0))
        setstatus!(player, "transform: position " *
                           "$(round(d.pos0[1] + dx; digits = 3)), $(round(d.pos0[2] + dy; digits = 3))")
    elseif d.zone === :rotate
        a = atan(p[2] - d.centre[2], p[1] - d.centre[1])
        deg = d.rot0 + rad2deg(a - d.ang0)   # y-down axis: the angle already reads clockwise
        shift && (deg = round(deg / 15) * 15)
        deg = clamp(rem(deg, 360, RoundNearest), -180.0, 180.0)
        setgizmoparam!(player, :rotation, deg)
        setstatus!(player, "transform: $(round(deg; digits = 1))°")
    else
        dist = sqrt((p[1] - d.centre[1])^2 + (p[2] - d.centre[2])^2)
        sc = clamp(d.scale0 * dist / d.dist0, 0.1, 4.0)
        shift && (sc = round(sc * 20) / 20)
        setgizmoparam!(player, :scale, sc)
        setstatus!(player, "transform: scale $(round(sc; digits = 3))")
    end
    showtransformgizmo!(player, d.clip)
    return true
end

"End the gesture."
function finishtransformdrag!(player::Player)
    transformdrag(player) === nothing && return false
    delete!(player.fxwidgets, :transformdrag)
    setcursor!(player, :arrow)
    return true
end

"""
Wire the gizmo to the preview: press to grab, move to transform, release to let
go, and a redraw whenever the picture could have moved under it.
"""
function installtransformgizmo!(player::Player)
    g = transformgizmo(player)
    sc = g.scene
    ax = player.previewaxis
    on(events(sc).mousebutton; priority = 25) do event
        Makie.receives_events(sc) || return Consume(false)
        event.button === Mouse.left || return Consume(false)
        p = Point2f(mouseposition(ax.scene))
        if event.action === Mouse.press
            return Consume(starttransformdrag!(player, p))
        elseif event.action === Mouse.release
            return Consume(finishtransformdrag!(player))
        end
        return Consume(false)
    end
    on(events(sc).mouseposition) do _
        transformdrag(player) === nothing && return Consume(false)
        p = Point2f(mouseposition(ax.scene))
        shift = ispressed(sc, Keyboard.left_shift | Keyboard.right_shift)
        alt = ispressed(sc, Keyboard.left_alt | Keyboard.right_alt)
        return Consume(transformdragto!(player, p; shift = shift, alt = alt))
    end
    # follow the picture: a slider, a keyframe, a scrub, another clip
    on(player.playhead) do _
        loc = editclip(player)
        showtransformgizmo!(player, loc === nothing ? nothing : loc[1])
    end
    return nothing
end
