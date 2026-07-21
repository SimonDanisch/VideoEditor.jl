# Mouse-driven walkthrough of the KEYFRAME workflow (rewritten for the 2026-07 UI:
# the ◆ keyframe LANE, not the old inline sliders). Every step is a real on-screen
# mouse/keyboard event (FakeInteraction draws a live cursor); only the caption and the
# off-camera trim-to-a-short-clip are script-side.
#
#   play a plain clip → open the ◆ lane → click two points to fade Opacity in from 0→1
#   → scrub the playhead through the fade → play it back.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie
import VideoEditor as VE
import FFMPEG_jll

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, KeyPress

const SRC     = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const RAW_MP4 = joinpath(tempdir(), "keyframe_walkthrough_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "keyframe_walkthrough.mp4")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# off-camera: a short working clip so the fade reads quickly
seq.clips[1].src_out = 240
VE.refreshedit!(player)
player.playhead[] = 0

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    fx = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
function lane_pos(t, vf)   # keyframe-lane time (s) + value-fraction (0..1) → figure px
    lane = player.fxwidgets[:kflane]; vp = lane.scene.viewport[]; L = lane.finallimits[]
    lx0, lx1 = minimum(L)[1], maximum(L)[1]; ly0, ly1 = minimum(L)[2], maximum(L)[2]
    Point2f(vp.origin[1] + (Float64(t) - lx0) / (lx1 - lx0) * vp.widths[1],
            vp.origin[2] + (Float64(vf) - ly0) / (ly1 - ly0) * vp.widths[2])
end
buttons = [c for c in fig.content if c isa Makie.Button]
btn(lbl) = first(b for b in buttons if b.label[] == lbl)
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
dia_btn = btn("◆")

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

fade = 2.5   # seconds
K = Makie.Keyboard
# scrub the playhead through the fade (plain timeline press = scrub), twice
showfade(cap) = [
    Lazy(_ -> (caption[] = cap; MouseTo(timeline_pos(fade * 0.7)))), LeftDown(), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(0.05))), Wait(0.5),
    Lazy(_ -> MouseTo(timeline_pos(fade))), Wait(0.6),
    Lazy(_ -> MouseTo(timeline_pos(0.05))), Wait(0.4),
    Lazy(_ -> MouseTo(timeline_pos(fade))), Wait(0.5),
    LeftUp(), Wait(0.4)]

events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "A plain clip"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(2.0), KeyPress(K.space), Wait(0.5),

    Lazy(_ -> (caption[] = "Open the ◆ keyframe lane"; MouseTo(block_center(dia_btn)))),
    LeftClick(), Wait(1.1),

    Lazy(_ -> (caption[] = "Click at the start, low — Opacity 0"; MouseTo(lane_pos(0.12, 0.06)))),
    LeftClick(), Wait(0.9),
    Lazy(_ -> (caption[] = "Click later, high — Opacity 1: a fade-in"; MouseTo(lane_pos(fade, 0.94)))),
    LeftClick(), Wait(0.9),

    showfade("Scrub through — the clip fades in from black")...,

    Lazy(_ -> (caption[] = "…and it plays back in real time"; MouseTo(timeline_pos(0.05)))),
    LeftClick(), Wait(0.3), Lazy(_ -> MouseTo(block_center(play_btn))), LeftClick(), Wait(3.5),
    KeyPress(K.space), Wait(0.6),
    Lazy(_ -> (caption[] = ""; MouseTo(block_center(play_btn)))), Wait(0.4),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
clip = seq.clips[1]
result = (nkeys = haskey(clip.animations, :opacity) ? length(clip.animations[:opacity].keys) : 0,)
close(player)
# no mpdecimate — the fade is a gradual reveal that frame-dedup would collapse
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved keyframe walkthrough" OUT_MP4 result
result
