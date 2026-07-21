# Mouse-driven walkthrough of the KEYFRAME workflow — "animate the controls you
# already use". Every step is a real on-screen mouse/keyboard event (FakeInteraction
# draws a live cursor + click state); nothing is poked through observables except the
# caption/instrumentation. A methodical little story:
#
#   open Effects → fade the clip in by arming ◆ Opacity and dragging the slider as we
#   scrub → smooth the fade (ease menu) → retime it by dragging the keyframe on the
#   lane → compose a second animation (a cool→warm colour reveal) → make a mistake and
#   fix it with Undo → play the polished result.
#
# RULES: everything VISIBLE is a mouse/keyboard event — never mutate an editor
# observable to make the demoed thing happen. Only the caption + the off-camera setup
# (trim to a short working clip) are script-side.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")                       # XAUTHORITY rotates across reboots
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie
import VideoEditor as VE
import FFMPEG_jll

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, WaitUntil, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, KeyPress

const SRC     = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const RAW_MP4 = joinpath(tempdir(), "keyframe_walkthrough_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "keyframe_walkthrough.mp4")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)                                           # let the thumbnail strip fill
Makie.disconnect!(player.screen, Makie.mouse_position)   # keep the real OS mouse out
fig.scene.events.hasfocus[] = false

# off-camera setup: work on a short 6 s clip so the fades read quickly, and start
# with the dock closed so the on-camera "FX" click OPENS it (it's open by default)
seq.clips[1].src_out = 360
VE.refreshedit!(player)
VE.closedock!(player)
player.playhead[] = 0

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    frac = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + frac * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
function menu_pos(menu, rely)
    bb = menu.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + bb.widths[1] / 2, bb.origin[2] + rely * bb.widths[2])
end
sliderfrac(sl) = (r = sl.range[]; (sl.value[] - first(r)) / (last(r) - first(r)))
val2frac(sl, v) = (r = sl.range[]; (Float64(v) - first(r)) / (last(r) - first(r)))
function slider_x(sl, frac)
    bb = sl.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + clamp(frac, 0, 1) * bb.widths[1], bb.origin[2] + bb.widths[2] / 2)
end
# a VISIBLE slider drag to value v: grab the handle where it is, drag to v, release
drag(sl, v) = [Lazy(_ -> MouseTo(slider_x(sl, sliderfrac(sl)))), LeftDown(), Wait(0.15),
               Lazy(_ -> MouseTo(slider_x(sl, val2frac(sl, v)))), Wait(0.1),
               Lazy(_ -> MouseTo(slider_x(sl, val2frac(sl, v)))), LeftUp(), Wait(0.35)]
function lane_pos(s, vnorm)
    lane = player.fxwidgets[:kflane]; vp = lane.scene.viewport[]; L = lane.finallimits[]
    lx0, lx1 = minimum(L)[1], maximum(L)[1]; ly0, ly1 = minimum(L)[2], maximum(L)[2]
    Point2f(vp.origin[1] + (Float64(s) - lx0) / (lx1 - lx0) * vp.widths[1],
            vp.origin[2] + (Float64(vnorm) - ly0) / (ly1 - ly0) * vp.widths[2])
end

buttons = [c for c in fig.content if c isa Makie.Button]
btn(lbl) = first(b for b in buttons if b.label[] == lbl)
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
fx_btn = btn("FX"); undo_btn = btn("↶"); redo_btn = btn("↷")
op_sl = player.fxsliders[:opacity]; temp_sl = player.fxsliders[:temperature]
kf_op = player.fxwidgets[:kf_opacity]; kf_temp = player.fxwidgets[:kf_temperature]
ease_menu = player.fxwidgets[:easemenu]

# caption overlay (narration) — an on-screen title updated per beat
caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = (RGBAf(0, 0, 0, 0.85)), strokewidth = 2.5, overdraw = true)

# THE reveal: grab the playhead (press on the timeline, away from any edge) and
# drag it back to the start then forward through the animation, twice. Each MouseTo
# animates the cursor smoothly through intermediate positions, so a single move is
# a smooth scrub — the preview, slider and lane playhead all update live.
showanim(hi, cap) = [
    Lazy(_ -> (caption[] = cap; MouseTo(timeline_pos(hi * 0.7)))), LeftDown(), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(0.06))), Wait(0.5),      # sweep back to the start
    Lazy(_ -> MouseTo(timeline_pos(hi))), Wait(0.6),        # forward reveal — the payoff
    Lazy(_ -> MouseTo(timeline_pos(0.06))), Wait(0.4),      # once more
    Lazy(_ -> MouseTo(timeline_pos(hi))), Wait(0.5),
    LeftUp(), Wait(0.5)]

recfunc = (i, t) -> nothing
K = Makie.Keyboard
events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "A plain 6-second clip"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(1.8), KeyPress(K.space), Wait(0.5),

    # 1 — open the Effects panel: sliders you already use, each with a ◆ keyframe toggle
    Lazy(_ -> (caption[] = "Every effect slider has a ◆ keyframe toggle"; MouseTo(block_center(fx_btn)))),
    LeftClick(), Wait(1.2),

    # 2 — set the fade: arm ◆ Opacity, drag to 0 at the start, drag to 1 two seconds in
    Lazy(_ -> (caption[] = "To fade in, arm ◆ next to Opacity"; MouseTo(timeline_pos(0.1)))),
    LeftClick(), Wait(0.4), Lazy(_ -> MouseTo(block_center(kf_op))), LeftClick(), Wait(1.0),
    Lazy(_ -> (caption[] = "At the start, drag Opacity down to 0"; MouseTo(slider_x(op_sl, sliderfrac(op_sl))))),
    drag(op_sl, 0.0)..., Wait(0.6),
    Lazy(_ -> (caption[] = "Two seconds in, drag it up to 1 — a keyframe is set"; MouseTo(timeline_pos(2.0)))),
    LeftClick(), Wait(0.4), drag(op_sl, 1.0)..., Wait(0.5),

    # 3 — SEE IT: drag the playhead through the fade, watching the clip appear from black
    showanim(2.4, "Drag the playhead through it — the clip fades in from black")...,

    # 4 — smooth the curve, then scrub again to feel the difference
    Lazy(_ -> (caption[] = "Make it graceful — set the curve to Smooth"; MouseTo(menu_pos(ease_menu, 0.5)))),
    LeftClick(), Wait(0.9), Lazy(_ -> MouseTo(menu_pos(ease_menu, -1.5))), LeftClick(), Wait(0.9),
    showanim(2.4, "Scrub again — now it eases in, gently")...,

    # 5 — compose a SECOND animation: a cool→warm colour reveal on Temperature
    Lazy(_ -> (caption[] = "Compose more — arm ◆ Temperature"; MouseTo(timeline_pos(0.1)))),
    LeftClick(), Wait(0.4), Lazy(_ -> MouseTo(block_center(kf_temp))), LeftClick(), Wait(0.9),
    Lazy(_ -> (caption[] = "Cool at the start…"; MouseTo(slider_x(temp_sl, sliderfrac(temp_sl))))),
    drag(temp_sl, -0.7)..., Wait(0.5),
    Lazy(_ -> (caption[] = "…warm by the end"; MouseTo(timeline_pos(4.0)))),
    LeftClick(), Wait(0.4), drag(temp_sl, 0.7)..., Wait(0.5),
    showanim(4.4, "Scrub through — it fades in AND warms up")...,

    # 6 — mistakes are cheap: Undo the last change, then Redo it
    Lazy(_ -> (caption[] = "Changed your mind? Undo…"; MouseTo(block_center(undo_btn)))),
    LeftClick(), Wait(1.2),
    Lazy(_ -> (caption[] = "…and Redo bring it right back"; MouseTo(block_center(redo_btn)))),
    LeftClick(), Wait(1.0),

    # 7 — the finale: it plays back in real time
    Lazy(_ -> (caption[] = "And it all plays back in real time"; MouseTo(timeline_pos(0.1)))),
    LeftClick(), Wait(0.3), Lazy(_ -> MouseTo(block_center(play_btn))), LeftClick(), Wait(4.5),
    KeyPress(K.space), Wait(0.8),
    Lazy(_ -> (caption[] = ""; MouseTo(block_center(play_btn)))), Wait(0.5),
]

FakeInteraction.interaction_record(recfunc, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
close(player)

# straight re-encode — NO mpdecimate here: the whole point of this demo is the
# gradual fades/colour ramps during playback, and mpdecimate drops near-identical
# consecutive frames, which is exactly what a slow animation looks like (it would
# collapse each fade into a jump). The script's own waits keep the pacing tight.
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved keyframe walkthrough" OUT_MP4
