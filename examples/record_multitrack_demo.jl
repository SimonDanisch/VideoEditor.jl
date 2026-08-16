# Mouse-driven walkthrough of MULTI-TRACK compositing — "stack a clip on top and
# see it blend through". Every step is a real on-screen mouse event (FakeInteraction
# draws a live cursor). Off-camera we only make two adjacent clips and pre-tint the
# second (so the composite reads); the DEMOED action — lifting the clip onto a second
# track and watching it composite — is all mouse.

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
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, KeyPress, KeyDown, KeyUp

const SRC     = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const RAW_MP4 = joinpath(tempdir(), "multitrack_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "multitrack_walkthrough.mp4")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# off-camera setup: two adjacent clips; tint + half-opacity the second so that,
# once it's stacked over the first, the composite is obviously a blend.
half = max(VE.seqlength(seq) ÷ 2, 30)
top = VE.split!(seq, half)
VE.seteffect!(top, VE.ColorEffect(saturation = 1.8f0, temperature = 0.5f0))
VE.seteffect!(top, VE.OpacityEffect(0.5f0))     # `effects` holds FxSlots, not effects
VE.refreshedit!(player)
player.playhead[] = 0

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)   # timeline time (s) + axis y-fraction → figure px
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    fx = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
buttons = [c for c in fig.content if c isa Makie.Button]
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

fps = seq.framerate
tmid = (top.start + VE.clipend(top)) / 2 / fps   # CENTER of the second clip (its edges would trim, not move)
tover = (half ÷ 2) / fps           # drop the cursor here → grab-offset snaps clip B's start to 0 (full overlap)
K = Makie.Keyboard

events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "Two clips, back to back on one track"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(3.0), KeyPress(K.space), Wait(0.6),

    # Ctrl-drag the SECOND clip onto a new track, overlapping the first (DaVinci-style move)
    Lazy(_ -> (caption[] = "Ctrl-drag the second clip…"; MouseTo(timeline_pos(tmid; yfrac = 0.5)))),
    KeyDown(K.left_control), LeftDown(), Wait(0.5),
    Lazy(_ -> (caption[] = "…lift it onto a track of its own, over the first"; MouseTo(timeline_pos(tmid; yfrac = 0.9)))),
    Wait(0.4),
    Lazy(_ -> MouseTo(timeline_pos(tover; yfrac = 0.985))), Wait(0.5),   # top of the axis → new track
    Lazy(_ -> MouseTo(timeline_pos(tover; yfrac = 0.985))), Wait(0.2),
    LeftUp(), KeyUp(K.left_control), Wait(0.8),

    # play the stack — the tinted top clip now composites over the base
    Lazy(_ -> (caption[] = "Now they're stacked — and they composite"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(3.2), KeyPress(K.space), Wait(0.8),
    Lazy(_ -> (caption[] = ""; MouseTo(timeline_pos(tover; yfrac = 0.5)))), Wait(0.5),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
result = (ntracks = VE.ntracks(seq), ntop = count(c -> c.track == 2, seq.clips),
          nclips = length(seq.clips))
close(player)
# straight re-encode — NO mpdecimate: the composited playback is a gradual reveal that
# frame-dedup would collapse; the script's own waits keep the pacing tight.
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved multi-track walkthrough" OUT_MP4 result
result
