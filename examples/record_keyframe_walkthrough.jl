# Mouse-driven walkthrough of the KEYFRAME workflow (2026-07 UI: the inline
# Inspector with the Premiere-style ◀◆▶ trio, curves + ◆ markers directly on the
# clip, and the right-click keyframe menu). Every step is a real on-screen
# mouse/keyboard event (FakeInteraction draws a live cursor); only the caption
# and the off-camera trim-to-a-short-clip are script-side.
#
#   play a plain clip → Ctrl+P adds a Color effect → ◆ activates Brightness
#   → scrub + slider writes a second key → drag a ◆ (readout + playhead snap)
#   → right-click a ◆ → Ease curve → scrub the ramp → play it back

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
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, RightClick,
                        Lazy, KeyPress, KeyDown, KeyUp, TypeText

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

# off-camera: a short working clip so the ramp reads quickly
seq.clips[1].src_out = min(seq.clips[1].src_out, 240)
VE.refreshedit!(player)
player.playhead[] = 0
T = (seq.clips[1].src_out - seq.clips[1].src_in) / seq.framerate   # clip length (s)

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    fx = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
function marker_pos(key, sf)   # figure pixel of `key`'s ◆ at source frame sf
    clip = seq.clips[1]
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    ntr = VE.ntracks(seq)
    lo, hi = VE.trackband(clip.track, ntr)
    g = min(0.02, VE.trackspan(ntr) * 0.15); lo += g; hi -= g
    inset = 0.12 * (hi - lo); lo += inset; hi -= inset
    c = clip.animations[key]; pr = VE.paramspec(key)
    v = something(VE.valueat(c, sf), pr.get(clip))
    y = lo + (hi - lo) * clamp(VE.paramnorm(pr, v), 0.0, 1.0)
    t = (clip.start + (sf - clip.src_in)) / seq.framerate
    Point2f(vp.origin[1] + (t - lims.origin[1]) / lims.widths[1] * vp.widths[1],
            vp.origin[2] + (y - lims.origin[2]) / lims.widths[2] * vp.widths[2])
end
function slider_at(key, frac)   # figure pixel inside a registered fx slider
    bb = player.fxsliders[key].layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + frac * bb.widths[1], bb.origin[2] + bb.widths[2] / 2)
end
trio_pos() = block_center(player.fxwidgets[:kfacc_brightness][2])   # re-fetch: rebuilds
menubtn_pos(i) = block_center(player.fxwidgets[Symbol(:kfmenubtn, i)])
srcframe() = VE.playheadframe(player, seq.clips[1])
buttons = [c for c in fig.content if c isa Makie.Button]
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

K = Makie.Keyboard
armframe = Ref(0)
events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "A plain clip"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(1.6), KeyPress(K.space), Wait(0.5),

    # park the playhead early in the clip, on camera
    Lazy(_ -> (caption[] = "Ctrl+P — add a Color effect"; MouseTo(timeline_pos(0.12T)))),
    LeftClick(), Wait(0.5),
    KeyDown(K.left_control), KeyPress(K.p), KeyUp(K.left_control), Wait(0.8),
    TypeText("col"), Wait(0.9), KeyPress(K.enter), Wait(1.2),

    # ◆ activates Brightness — first key at the playhead, curve lands on the clip
    Lazy(_ -> (caption[] = "◆ activates Brightness — its curve lands on the clip";
               armframe[] = srcframe(); MouseTo(trio_pos()))),
    LeftClick(), Wait(1.3),

    # scrub ahead, then pull the slider: scrub-and-adjust writes a key each move
    Lazy(_ -> (caption[] = "Scrub ahead…"; MouseTo(timeline_pos(0.6T)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (caption[] = "…move the slider — a new key right at the playhead";
               MouseTo(slider_at(:brightness, 0.5)))),
    LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(slider_at(:brightness, 0.86))), Wait(0.3), LeftUp(), Wait(1.2),

    # drag a ◆ on the clip: live value·time readout, snaps onto the playhead
    Lazy(_ -> (caption[] = "Drag a ◆ — live readout, snaps to the playhead";
               MouseTo(marker_pos(:brightness, armframe[])))),
    LeftDown(), Wait(0.3),
    Lazy(_ -> MouseTo(marker_pos(:brightness, armframe[]) .+ Point2f(60, 10))), Wait(0.5),
    Lazy(_ -> MouseTo(marker_pos(:brightness, armframe[]) .+ Point2f(-20, 4))), Wait(0.5),
    LeftUp(), Wait(0.8),

    # right-click a ◆ → the keyframe menu → Ease curve
    Lazy(_ -> (caption[] = "Right-click a ◆ — ease it in & out";
               MouseTo(marker_pos(:brightness, srcframe())))),
    RightClick(), Wait(1.1),
    Lazy(_ -> MouseTo(menubtn_pos(2))),
    LeftClick(), Wait(1.0),

    # scrub through the eased ramp, then play it back
    Lazy(_ -> (caption[] = "Scrub through — the brightness ramps in";
               MouseTo(timeline_pos(0.75T)))),
    LeftDown(), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(0.05T))), Wait(0.7),
    Lazy(_ -> MouseTo(timeline_pos(0.85T))), Wait(0.8),
    LeftUp(), Wait(0.4),
    Lazy(_ -> (caption[] = "…and it plays back in real time"; MouseTo(timeline_pos(0.05)))),
    LeftClick(), Wait(0.3), Lazy(_ -> MouseTo(block_center(play_btn))), LeftClick(), Wait(3.5),
    KeyPress(K.space), Wait(0.6),
    Lazy(_ -> (caption[] = ""; MouseTo(block_center(play_btn)))), Wait(0.4),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
clip = seq.clips[1]
result = (nkeys = haskey(clip.animations, :brightness) ? length(clip.animations[:brightness].keys) : 0,
          eased = haskey(clip.animations, :brightness) && clip.animations[:brightness].interp === :smooth)
close(player)
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved keyframe walkthrough" OUT_MP4 result
result
