# Mouse-driven tour of the CORE editing workflow (CPU-only, no GPU stabilization so it
# records fast): blade-cut a clip, crop the framing on the preview, then export from the
# dock. Every step is a real on-screen mouse/keyboard event (FakeInteraction cursor).

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
const RAW_MP4 = joinpath(tempdir(), "editing_demo_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "editing_walkthrough.mp4")
const EXPORTED = joinpath(tempdir(), "editing_demo_export.mp4")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

seq.clips[1].src_out = 240
player.fxwidgets[:exportpath][] = EXPORTED   # preset off-camera → no blocking save dialog
VE.refreshedit!(player)
player.playhead[] = 0

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    fx = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
function pv(fx, fy)   # preview viewport fraction → figure px
    vp = player.previewaxis.scene.viewport[]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + fy * vp.widths[2])
end
buttons = [c for c in fig.content if c isa Makie.Button]
btn(lbl) = first(b for b in buttons if b.label[] == lbl)
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
blade_btn = btn("✂"); export_btn = btn("Export")
export_go() = player.fxwidgets[:exportgo]

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

K = Makie.Keyboard
events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "A plain clip"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(1.6), KeyPress(K.space), Wait(0.5),

    # blade: arm ✂, click the timeline to cut there
    Lazy(_ -> (caption[] = "Blade tool — click ✂"; MouseTo(block_center(blade_btn)))),
    LeftClick(), Wait(0.9),
    Lazy(_ -> (caption[] = "…then click the timeline to cut"; MouseTo(timeline_pos(2.5)))),
    LeftClick(), Wait(1.0), KeyPress(K.escape), Wait(0.6),   # Esc puts the persistent blade away

    # crop: C arms crop, drag a rectangle on the preview
    Lazy(_ -> (caption[] = "Crop — press C and drag on the preview"; MouseTo(pv(0.28, 0.28)))),
    KeyPress(K.c), Wait(0.5),
    Lazy(_ -> MouseTo(pv(0.28, 0.28))), LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(pv(0.72, 0.72))), Wait(0.2), LeftUp(), Wait(0.8),
    KeyPress(K.escape), Wait(0.8),

    # export from the dock (path preset off-camera)
    Lazy(_ -> (caption[] = "Export — open the panel"; MouseTo(block_center(export_btn)))),
    LeftClick(), Wait(1.0),
    Lazy(_ -> (caption[] = "…and render the cut"; MouseTo(block_center(export_go())))),
    LeftClick(), Wait(3.5),
    Lazy(_ -> (caption[] = ""; MouseTo(block_center(play_btn)))), Wait(0.5),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
result = (nclips = length(seq.clips),
          cropped = seq.clips[1].crop != (0.0, 0.0, 1.0, 1.0),
          exported = isfile(EXPORTED))
close(player)
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved editing walkthrough" OUT_MP4 result
result
