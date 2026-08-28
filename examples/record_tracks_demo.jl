# Record the TRACK/PLAYHEAD walkthrough at media/tracks_demo.mp4.
#
# Same rules as record_demo.jl: everything on screen is caused by a mouse or
# keyboard event, never by a script-side mutation. What this one shows:
#
#   [1] the SCRUB STRIP above the lanes — press and drag moves the playhead
#   [2] a press on a CLIP selects it and leaves the frame you are at alone
#   [3] S splits, Ctrl+drag lifts the right half onto a new track
#   [4] a lane's TOP EDGE is a resize grip: hover shows the bar, drag makes
#       the lane taller — and its filmstrip shows bigger frames, not stretched
#   [5] the blade cuts on a LANE and still scrubs in the strip

using VideoEditor, GLMakie, Makie
import VideoEditor as VE

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, KeyPress,
                        KeyDown, KeyUp

# ---------------------------------------------------------------- off-camera

demosource = joinpath(@__DIR__, "..", "..", "..", "media", "demo_source.mp4")
isfile(demosource) || error("missing $demosource — run record_demo.jl once to cut it")

# visible=true would let GLFW poll the real mouse and clobber the synthetic one
GLMakie.activate!(; visible = false, framerate = 30)
player = Player(demosource)
fig = player.fig
resize!(fig, 1440, 900)
sleep(4.0)                       # let the thumbnail strip fill before recording
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# ------------------------------------------------------------------ helpers

"Pixel position of timeline time `t` at axis-y `y` (uses the CURRENT zoom and
track layout — always Lazy these)."
function timeline_pos(t::Real, y::Real)
    ax = player.timeline.axis
    lims = ax.finallimits[]
    vp = ax.scene.viewport[]
    frac = (Float64(t) - lims.origin[1]) / lims.widths[1]
    return Point2f(vp.origin[1] + frac * vp.widths[1],
                   vp.origin[2] + Float64(y) / VE.AXISTOP * vp.widths[2])
end

strippos(t) = timeline_pos(t, (VE.SCRUBBAND[1] + VE.SCRUBBAND[2]) / 2)
function lanepos(t, track)
    seq = player.sequence
    lo, hi = VE.trackband(seq, track, VE.ntracks(seq))
    return timeline_pos(t, (lo + hi) / 2)
end
function edgepos(t, track)
    seq = player.sequence
    return timeline_pos(t, VE.trackband(seq, track, VE.ntracks(seq))[2])
end
newtrackpos(t) = timeline_pos(t, 0.93)     # the "+ new track" zone above the lanes

button(lbl) = first(b for b in fig.content if b isa Makie.Button && b.label[] == lbl)
blade = button("✂")
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))

dur = VE.seqduration(player.sequence)
t1, t2, t3 = 0.25dur, 0.5dur, 0.75dur

# ------------------------------------------------------------------- events

events = [
    # [1] THE STRIP SCRUBS. Press in the band above the lanes and drag: the
    # playhead follows the cursor, nothing gets selected.
    Lazy(_ -> MouseTo(strippos(t1))), Wait(0.8),
    LeftDown(),
    Lazy(_ -> MouseTo(strippos(t3))), Wait(0.3),
    Lazy(_ -> MouseTo(strippos(t2))),
    LeftUp(), Wait(1.0),

    # [2] A PRESS ON A CLIP SELECTS IT and leaves the playhead where it is —
    # the accent border appears, the orange line does not move.
    Lazy(_ -> MouseTo(lanepos(t1, 1))), LeftClick(), Wait(1.2),

    # [3] split at the playhead, then Ctrl+drag the right half onto a new track
    KeyPress(Makie.Keyboard.s), Wait(1.0),
    KeyDown(Makie.Keyboard.left_control),
    Lazy(_ -> MouseTo(lanepos(t3, 1))), LeftDown(), Wait(0.4),
    Lazy(_ -> MouseTo(newtrackpos(t3))), Wait(0.6),
    LeftUp(), KeyUp(Makie.Keyboard.left_control), Wait(1.4),

    # [4] THE LANE'S TOP EDGE IS A GRIP. Hovering it shows the bar; dragging it
    # up makes V1 taller — and its filmstrip shows BIGGER frames, not the same
    # ones pulled tall.
    Lazy(_ -> MouseTo(edgepos(0.35dur, 1))), Wait(1.0),
    LeftDown(),
    Lazy(_ -> MouseTo(timeline_pos(0.35dur, 0.60))), Wait(0.4),
    Lazy(_ -> MouseTo(timeline_pos(0.35dur, 0.66))),
    LeftUp(), Wait(1.4),

    # [5] the blade cuts on a LANE, and the strip still scrubs while it is out
    MouseTo(block_center(blade)), LeftClick(), Wait(0.8),
    Lazy(_ -> MouseTo(strippos(t1))), LeftDown(),
    Lazy(_ -> MouseTo(strippos(0.35dur))), LeftUp(), Wait(1.0),   # scrubs, no cut
    Lazy(_ -> MouseTo(lanepos(0.62dur, 1))), LeftClick(), Wait(1.2),  # …this cuts
    KeyPress(Makie.Keyboard.escape), Wait(1.0),
]

video_path = joinpath(@__DIR__, "..", "..", "..", "media", "tracks_demo.mp4")
# `visible = false` REACHES THE SCREEN CONFIG, and it has to: `record` opens the
# figure's screen with the default config, which SHOWS the window — and
# `glfwShowWindow` never returns on this Xwayland display (100% CPU in
# `_glfwShowWindowX11` waiting for a VisibilityNotify). The recording is
# offscreen either way.
FakeInteraction.interaction_record(fig, video_path, events; fps = 30, px_per_unit = 1,
                                   visible = false)
println("Saved: ", video_path)
close(player)
