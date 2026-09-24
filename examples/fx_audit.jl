# Does each effect actually reach the PICTURE? A model that runs and a card that
# renders still do not mean the frame changes.
#
# Two things make this script trustworthy, and it was wrong without both:
#
#  1. A POSITIVE CONTROL. Reading `player.frame[]` reported "no change" for every
#     effect INCLUDING a brightness-0.4 ColorEffect — the observable was not
#     updated by the script's `showframe!`. Without the control that reads as two
#     broken effects rather than one broken harness.
#  2. A BARE ENGINE, no Player. Touching the GPU on the main thread beside a live
#     Player's workers trips `BatchQueue is single-writer`.
#
# `framereader(clip, engine)` renders the clip through its effect graph, which is
# exactly the question. Run on :1.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")
using VideoEditor, GLMakie
import VideoEditor as VE
import MatAnyoneRunner
GLMakie.activate!(; visible = false)
VE.registermatte!(MatAnyoneRunner.matanyonepropagator())
VE.installdepth!(); VE.installlook!()

# NO Player: a bare engine, so nothing contends with its GPU workers.
engine = VE.FxEngine(VE.Mantle.defaultbackend())
src  = VE.VideoSource("/sim/Programmieren/VideoEdit/media/demo_loop.mp4")
clip = VE.Clip(src, 0, 16, 0, (0.0, 0.0, 1.0, 1.0), 1.0)
FRAME = 4
# `framereader` renders the clip THROUGH its effect graph — its own docstring
# says so ("the models get the frame the user sees"). That is precisely the
# question here: does adding the effect change the picture?
grab() = VE.framereader(clip, engine; maxside = 240)(FRAME)
mdiff(a, b) = size(a) != size(b) ? Inf :
    sum(abs.(Float64.(VE.ColorTypes.red.(a)) .- Float64.(VE.ColorTypes.red.(b)))) / length(a)

base = grab(); println("baseline ", size(base))

# POSITIVE CONTROL — if this does not move, nothing below means anything
VE.seteffect!(clip, VE.ColorEffect(brightness = 0.4, contrast = 1.0,
                                   saturation = 1.0, temperature = 0.0))
println("CONTROL ColorEffect  mean Δ = ", round(mdiff(grab(), base); digits = 4))
filter!(s -> !(s.effect isa VE.ColorEffect), clip.effects)

# DEPTH BLUR
reader = VE.framereader(clip, engine; maxside = 240)
VE.analyzedepth!(clip, reader; maxside = 240)
println("depth track: ", clip.depthtrack !== nothing)
b1 = grab()
VE.seteffect!(clip, VE.DepthBlurEffect(0.0f0, 1.0f0))
println("DepthBlurEffect      mean Δ = ", round(mdiff(grab(), b1); digits = 4))
filter!(s -> !(s.effect isa VE.DepthBlurEffect), clip.effects)

# LOOK
f0 = grab()
VE.analyzelook!(clip, f0)
println("look LUT: ", clip.look !== nothing)
b2 = grab()
VE.seteffect!(clip, VE.LookEffect(1.0f0))
println("LookEffect           mean Δ = ", round(mdiff(grab(), b2); digits = 4))
