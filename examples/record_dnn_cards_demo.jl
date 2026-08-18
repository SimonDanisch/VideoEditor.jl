# Mouse-driven walkthrough of the two DNN effect cards — Depth blur and Look —
# added the way a user adds any other effect: from the Add-effect menu.
#
# The point of recording this one is that both effects were, until now, only
# reachable by typing a command in the palette. They rendered, saved and
# keyframed correctly and still had no card, no sliders and no menu entry. A
# screenshot is the only thing that shows the difference between "the effect
# works" and "a person can use it".
#
#   pick Depth blur from the menu → its card appears with Focus/Defocus and an
#   Estimate depth button → pick Look → its card appears with Learn look.

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
using .FakeInteraction: Wait, MouseTo, LeftClick, Lazy

const SRC     = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const RAW_MP4 = joinpath(tempdir(), "dnn_cards_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "dnn_cards_walkthrough.mp4")
const SHOT    = joinpath(@__DIR__, "..", "..", "..", "media", "dnn_cards.png")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

seq.clips[1].src_out = 180
VE.refreshedit!(player)
player.fxwidgets[:fxlistrefresh]()
player.playhead[] = 0

block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))

# The Add-effect control is a Menu whose options come from `addablekinds()` — the
# thing that was empty of these two kinds until they were registered. It
# registers itself under `:addeffect`, which is a sturdier handle than scanning
# `fig.content` for "the Menu" — the panel grew a second one once already.
addmenu() = player.fxwidgets[:addeffect]
cardnamed(t) = first(c for c in player.fxwidgets[:fxcards] if c.title[] == t)

"Pick an option by LABEL rather than by index: the menu is built from a registry
 whose order is registration order, and an index would silently pick the wrong
 effect the next time a kind is added."
function pick!(menu, label)
    i = findfirst(o -> (o isa Tuple ? first(o) : o) == label, menu.options[])
    i === nothing && error("no menu option $label — have $(menu.options[])")
    menu.i_selected[] = i
    return i
end

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "The Add-effect menu — Depth blur is in it now";
               MouseTo(block_center(addmenu())))),
    LeftClick(), Wait(0.9),
    Lazy(_ -> (pick!(addmenu(), "Depth blur"); caption[] = "Its card: Focus, Defocus, and Estimate depth";
               MouseTo(block_center(addmenu())))),
    Wait(1.8),
    Lazy(_ -> (caption[] = "Look, the same way"; MouseTo(block_center(addmenu())))),
    LeftClick(), Wait(0.7),
    Lazy(_ -> (pick!(addmenu(), "Look"); caption[] = "A grade learned from one frame of this shot";
               MouseTo(block_center(addmenu())))),
    Wait(2.0),
    Lazy(_ -> (caption[] = ""; MouseTo(Point2f(750, 500)))), Wait(0.6),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)

# What ACTUALLY landed on the clip, and whether each card exists — the two
# questions the video is being recorded to answer.
kinds = [typeof(s.effect) for s in seq.clips[1].effects]
titles = [c.title[] for c in player.fxwidgets[:fxcards]]
Makie.save(SHOT, Makie.colorbuffer(player.screen))
close(player)
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved dnn card walkthrough" OUT_MP4 SHOT kinds titles
(; kinds, titles)
