# A LOOK at the matte card's object pills, which is the only way to judge the
# complaint that produced them: "the object group pill is misaligned with the
# (n points) pills, also the pill has some kind of rendering bug and doesn't look
# good. I think it's also way too round."
#
# Three things this is here to show, none of which a test assertion can:
#   * the "+ object" button lines up with the pills above it (it sits in the
#     pills' OWN grid, column 1, so it stops short by the `×` column like they do)
#   * the corner radius reads as the same design language as the rest (5, not a
#     full stadium at half the height)
#   * the row holding them is tall enough for every pill, so nothing overlaps the
#     control underneath as objects are added
#
# Mouse-driven: the points come from clicks on the preview and the second object
# from pressing `+ object`, so what gets photographed is what a user would make.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie
import VideoEditor as VE

const SRC  = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const SHOT = joinpath(@__DIR__, "..", "..", "..", "media", "matte_pills.png")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

seq.clips[1].src_out = 90
VE.refreshedit!(player)
player.playhead[] = 0
player.timeline.selected[] = 1
VE.opendock!(player, :effects)
sleep(0.5)

# Events go to the FIGURE in FIGURE PIXELS, exactly as the suite synthesizes
# them. Not the axis scene in data coordinates: the marking overlay is a sibling
# scene that captures the mouse, so a click posted to the axis is a click nothing
# is listening for — which is why the first attempt photographed an empty card.
ev = Makie.events(fig)
"Preview viewport fraction -> figure pixel."
pv(fx, fy) = begin
    vp = player.previewaxis.scene.viewport[]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + fy * vp.widths[2])
end
function clickpreview(fx, fy)
    ev.mouseposition[] = Tuple(pv(fx, fy))
    ev.mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    ev.mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    return nothing
end

# `showkind!` FIRST. The matte card is the `:matte` kind's card, and a kind's
# card exists only where its effect is on the clip — so activating the tool alone
# starts a marking session with nothing on screen to draw the pills into. In the
# real UI you never hit this: you press "Mark subject" ON the card, so the card
# is already there. It cost two empty screenshots to notice.
VE.showkind!(player, :matte)
sleep(1.0)
VE.activatetool!(player, :matte)
sleep(1.5)
col = VE.mattecollect(player)
col === nothing && error("matte tool did not start a marking session")

# First object: two clicks. The FIRST one pays the cold model — SAM 2 plus the
# live MatAnyone preview — so it gets the long wait; the rest are warm.
clickpreview(0.52, 0.45); sleep(45.0)
clickpreview(0.56, 0.50); sleep(6.0)

# Second object, by PRESSING the button that makes one.
card = get(player.fxwidgets, :mattecard, nothing)
if card !== nothing && card.newobj !== nothing
    card.newobj.clicks[] = card.newobj.clicks[] + 1
    sleep(1.0)
    clickpreview(0.35, 0.62); sleep(8.0)
else
    @warn "no live matte card — the pills cannot be photographed" card
end

# …and a third, so the row height is exercised past two.
card = get(player.fxwidgets, :mattecard, nothing)
if card !== nothing && card.newobj !== nothing
    card.newobj.clicks[] = card.newobj.clicks[] + 1
    sleep(1.0)
    clickpreview(0.68, 0.35); sleep(8.0)
end

card = get(player.fxwidgets, :mattecard, nothing)
Makie.save(SHOT, Makie.colorbuffer(player.screen))
info = card === nothing ? (; nobj = 0) :
       (; nobj = card.nobj, npoints = length(col.points),
          objects = sort(unique(q[4] for q in col.points)))
close(player)
@info "matte pill check" SHOT info
info
