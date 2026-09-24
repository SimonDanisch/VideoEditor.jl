# Does every DNN actually RUN? Not "is it wired" — run it and see.
#
# This found three integrations that shipped green and could not execute:
#   * depth threw `MethodError: depthbytes(::Array{Float16,4})` on every press —
#     the runner returns the model's raw tensor, `depthbytes` takes a matrix
#   * `depthblur_kernel!` and `lookmix_kernel!` both built colours with the
#     VALIDATING `RGB{N0f8}(::Float32, …)`, which the shader compiler rejects, so neither could
#     compile on the GPU they were written for (both passed every CPU test)
#   * `WhisperRunner.transcribe` returns `(text, segments)`; iterating the tuple
#     walked the String and died on `s.text`
#
# Every one of those seams was invisible to the suite, because the tests feed the
# consumers synthetic data and never call the model. Run this on :1 — on :99 the
# model path segfaults.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")
using VideoEditor, GLMakie, Makie
import VideoEditor as VE
GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player("/sim/Programmieren/VideoEdit/media/demo_loop.mp4")
fig = player.fig; resize!(fig, 1400, 900); sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false
seq = player.sequence; clip = seq.clips[1]
clip.src_out = 16
player.playhead[] = 4; player.timeline.selected[] = 1
VE.refreshedit!(player); sleep(0.5)

"Run one analysis and say plainly whether it produced anything."
function probe(name, go, done)
    println("---- ", name)
    # the error is PRINTED, never swallowed: the point of this script is to
    # surface exactly the failures a green suite has been hiding
    try
        go()
    catch e
        println("  THREW immediately: ", sprint(showerror, e))
        return
    end
    for _ in 1:150
        done() && (println("  OK"); return)
        sleep(1.0)
    end
    println("  NO RESULT after 150 s")
end

probe("NeuralLUT (look)", () -> VE.runlook!(player), () -> clip.look !== nothing)
probe("Whisper (transcribe)", () -> VE.runtranscribe!(player), () -> !isempty(seq.captions))
probe("Kokoro (narration)", () -> VE.addnarration!(player, "one two three"),
      () -> !isempty(seq.narration) && !isempty(seq.narration[1].samples))
probe("RIFE (smooth slow motion)", () -> VE.smoothslowmo!(player),
      () -> clip.timeinterp === :flow)
close(player)
