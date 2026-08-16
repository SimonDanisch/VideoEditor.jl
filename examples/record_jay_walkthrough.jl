# The full walkthrough: one clip becomes two birds.
#
#   grade it → cut it in half → stack the halves → key the jay out of its
#   background → place it beside itself
#
# Every step on screen is a real mouse or keyboard event (FakeInteraction draws a
# live cursor). Off camera we only cut a short proxy out of the 110 s source —
# the rule `record_demo.jl` sets out, and the reason the video is worth anything:
# what you watch is the editor being used, not a script mutating state.
#
# The matte is the beat to watch. Clicking the bird runs SAM 2.1 to seed a mask
# and MatAnyone to propagate it, both on the GPU through Lava, while the CPU is
# busy rendering this very recording.

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
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy,
                        KeyPress, KeyDown, KeyUp

const SOURCE  = "/home/simon/Videos/test-jay-gpu.mp4"
const MEDIA   = normpath(joinpath(@__DIR__, "..", "..", "..", "media"))
const PROXY   = joinpath(MEDIA, "jay_proxy.mp4")
const RAW_MP4 = joinpath(tempdir(), "jay_raw.mp4")
const OUT_MP4 = joinpath(MEDIA, "jay_walkthrough.mp4")

# ---------------------------------------------------------------- off-camera
# 8 s at 720 px. The source is 110 s of 1080p60 and the matte propagates frame by
# frame, so the full clip would spend the demo waiting instead of showing.
mkpath(MEDIA)
isfile(PROXY) || run(pipeline(
    `$(FFMPEG_jll.ffmpeg()) -y -ss 2 -t 8 -i $SOURCE -vf scale=720:-2 -c:v libx264
     -g 30 -crf 20 -preset fast -an $PROXY`, stdout = devnull, stderr = devnull))

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(PROXY)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
# Hidden screen, but GLFW would still poll the real mouse and fight the synthetic
# cursor, and a focused window would eat real keystrokes.
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# ---------------------------------------------------------------- helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    fx = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
"A point on the preview, in figure pixels, from a fraction of the frame."
function preview_pos(fx, fy)
    vp = player.previewaxis.scene.viewport[]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + (1 - fy) * vp.widths[2])
end
button(label) = first(b for b in fig.content if b isa Makie.Button && b.label[] == label)
"The Add-effect dropdown, and where its `name` row lands once open."
const ADDFX = player.fxwidgets[:addeffect]
addfx_center() = let bb = ADDFX.layoutobservables.computedbbox[]
    Point2f(bb.origin .+ bb.widths ./ 2)
end
function addfx_row(name)
    i = findfirst(o -> first(o) == name, ADDFX.options[])
    sc = ADDFX.blockscene.children[end]
    rects = sc.plots[1][1][]; tr = Makie.translation(sc)[]
    Point2f(sum(extrema(rects[i])) ./ 2 .+ Point2f(tr[1], tr[2]))
end

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

fps = seq.framerate
half = max(VE.seqlength(seq) ÷ 2, 30)
K = Makie.Keyboard

# Where the jay sits in frame. A click that misses it seeds the mask on the
# branch and the demo keys out the wrong thing, so this is measured off the
# proxy's own frame rather than guessed.
const JAY = (0.52, 0.46)

events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "One clip, straight off the camera: a jay";
               MouseTo(block_center(button("Play"))))),
    LeftClick(), Wait(3.0), KeyPress(K.space), Wait(0.8),

    # ---------------------------------------------------------- grade it
    Lazy(_ -> (caption[] = "Open the effects dock"; MouseTo(block_center(button("FX"))))),
    LeftClick(), Wait(1.0),
    Lazy(_ -> (caption[] = "Add a Color grade"; MouseTo(timeline_pos(1.0)))),
    LeftClick(), Wait(0.6),
    Lazy(_ -> MouseTo(addfx_center())), LeftClick(), Wait(0.8),
    Lazy(_ -> MouseTo(addfx_row("Color"))), LeftClick(), Wait(1.6),

    # ---------------------------------------------------------- cut in half
    Lazy(_ -> (caption[] = "Park the playhead halfway…";
               MouseTo(timeline_pos(half / fps)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (caption[] = "…and S cuts the clip in two"; MouseTo(timeline_pos(half / fps)))),
    KeyPress(K.s), Wait(1.2),

    # ---------------------------------------------------- stack the halves
    Lazy(_ -> (caption[] = "Ctrl-drag the second half…";
               MouseTo(timeline_pos((half + VE.seqlength(seq)) / 2 / fps)))),
    KeyDown(K.left_control), LeftDown(), Wait(0.6),
    Lazy(_ -> (caption[] = "…onto a track of its own, over the first";
               MouseTo(timeline_pos(half / 2 / fps; yfrac = 0.9)))), Wait(0.5),
    Lazy(_ -> MouseTo(timeline_pos(0.0; yfrac = 0.985))), Wait(0.6),
    Lazy(_ -> MouseTo(timeline_pos(0.0; yfrac = 0.985))), Wait(0.3),
    LeftUp(), KeyUp(K.left_control), Wait(1.2),

    # ------------------------------------------------------- key the bird
    Lazy(_ -> (caption[] = "The top copy needs its background gone";
               MouseTo(addfx_center()))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> MouseTo(addfx_row("Matte"))), LeftClick(), Wait(1.5),
    Lazy(_ -> (caption[] = "Click the bird — SAM 2.1 finds its edge";
               MouseTo(preview_pos(JAY...)))),
    Wait(0.8), LeftClick(), Wait(3.0),
    Lazy(_ -> (caption[] = "…and MatAnyone carries the matte across the clip, on the GPU";
               MouseTo(preview_pos(JAY[1] + 0.08, JAY[2] - 0.05)))),
    Wait(8.0),

    # ------------------------------------------------- place it beside itself
    Lazy(_ -> (caption[] = "Now it is only the jay — so move it";
               MouseTo(addfx_center()))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> MouseTo(addfx_row("Transform"))), LeftClick(), Wait(1.5),
    Lazy(_ -> (caption[] = "Drag the gizmo: a second jay, perched on the first";
               MouseTo(preview_pos(0.5, 0.5)))),
    LeftDown(), Wait(0.4),
    Lazy(_ -> MouseTo(preview_pos(0.30, 0.30))), Wait(0.5),
    Lazy(_ -> MouseTo(preview_pos(0.22, 0.24))), Wait(0.4),
    LeftUp(), Wait(1.2),

    Lazy(_ -> (caption[] = "Two birds, one clip";
               MouseTo(block_center(button("Play"))))),
    LeftClick(), Wait(4.0), KeyPress(K.space), Wait(1.0),
    Lazy(_ -> (caption[] = ""; MouseTo(preview_pos(0.5, 0.5)))), Wait(0.5),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events;
                                   fps = 30, px_per_unit = 1)
result = (ntracks = VE.ntracks(seq), nclips = length(seq.clips),
          matted = count(c -> c.mattetrack !== nothing, seq.clips),
          effects = [length(c.effects) for c in seq.clips])
close(player)
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved the jay walkthrough" OUT_MP4 result
result
