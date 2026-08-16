# Mouse-driven walkthrough of the EFFECT STACK: the FX panel is a stack of cards,
# one per applied effect, each showing its parameters as live sliders — the
# preview updates as you drag. All on-screen mouse.
#
#   a clip already carries a Color effect → click its card to select it → drag
#   Saturation up (vivid) then down (grayscale) then to a rich look, live.

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
const RAW_MP4 = joinpath(tempdir(), "effects_demo_raw.mp4")
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "effects_walkthrough.mp4")

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(SRC)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# off-camera: short clip + a neutral Color effect already in the stack (starts vivid-neutral)
seq.clips[1].src_out = 180
VE.seteffect!(seq.clips[1], VE.ColorEffect())   # `effects` holds FxSlots, not effects
VE.refreshedit!(player)
player.fxwidgets[:fxlistrefresh]()
player.playhead[] = 0

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
sliderfrac(sl) = (r = sl.range[]; (sl.value[] - first(r)) / (last(r) - first(r)))
val2frac(sl, v) = (r = sl.range[]; (Float64(v) - first(r)) / (last(r) - first(r)))
function slider_x(sl, frac)
    bb = sl.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + clamp(frac, 0, 1) * bb.widths[1], bb.origin[2] + bb.widths[2] / 2)
end
drag(getsl, v) = [Lazy(_ -> (sl = getsl(); MouseTo(slider_x(sl, sliderfrac(sl))))), LeftDown(), Wait(0.15),
                  Lazy(_ -> (sl = getsl(); MouseTo(slider_x(sl, val2frac(sl, v))))), Wait(0.1),
                  Lazy(_ -> (sl = getsl(); MouseTo(slider_x(sl, val2frac(sl, v))))), LeftUp(), Wait(0.35)]

buttons = [c for c in fig.content if c isa Makie.Button]
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
# The FX panel is a stack of `Makie.Card`s, one per applied effect, and a card
# opens IN PLACE — there is no modal picker any more, and no `:effectrows`.
colorcard() = first(c for c in player.fxwidgets[:fxcards] if c.title[] == "Color")
"A point on the card's header bar: its top strip, left of the accessory cell
(the ?/eye/× buttons live there and take their own presses)."
function header_point(card)
    bb = card.layoutobservables.computedbbox[]
    h = card.headerheight[]
    Point2f(bb.origin[1] + 0.3 * bb.widths[1], bb.origin[2] + bb.widths[2] - h / 2)
end
# Parameter sliders register themselves by parameter name as the card body is
# built (`fxpanel.jl`: `player.fxsliders[kind.kfkeys[j]] = w`).
satslider() = player.fxsliders[:saturation]

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

K = Makie.Keyboard
events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "A clip with a Color effect in the stack"; MouseTo(block_center(play_btn)))),
    LeftClick(), Wait(1.6), KeyPress(K.space), Wait(0.5),

    # The card's parameters are already unfolded; what the click visibly does is
    # SELECT it — accent outline and tinted header — which is what makes it the
    # card the keyframe and tool actions target. Narrate that, not a fold.
    Lazy(_ -> (caption[] = "Click the effect's card to select it"; MouseTo(header_point(colorcard())))),
    LeftClick(), Wait(1.2),

    Lazy(_ -> (caption[] = "Drag Saturation up — vivid, live in the preview"; MouseTo(slider_x(satslider(), sliderfrac(satslider()))))),
    Wait(0.3), drag(satslider, 2.0)..., Wait(1.1),
    Lazy(_ -> (caption[] = "…or down to grayscale"; MouseTo(slider_x(satslider(), sliderfrac(satslider()))))),
    drag(satslider, 0.0)..., Wait(1.1),
    Lazy(_ -> (caption[] = "…and settle on a rich look"; MouseTo(slider_x(satslider(), sliderfrac(satslider()))))),
    drag(satslider, 1.6)..., Wait(1.2),
    Lazy(_ -> (caption[] = ""; MouseTo(block_center(play_btn)))), Wait(0.5),
]

FakeInteraction.interaction_record((i, t) -> nothing, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
ce = seq.clips[1].effects[end].effect      # a slot, not the effect itself
result = (saturation = ce isa VE.ColorEffect ? round(Float64(ce.adj.saturation), digits = 2) : nothing,)
close(player)
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved effects walkthrough" OUT_MP4 result
result
