ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")
using VideoEditor, GLMakie, Makie
import VideoEditor as VE
import FFMPEG_jll
GLMakie.activate!(; visible = false, framerate = 30)

dir = mktempdir()
# 1. Kokoro SPEAKS a known sentence
player0 = VE.Player("/sim/Programmieren/VideoEdit/media/demo_loop.mp4")
sleep(3.0)
said = "the quick brown fox jumps over the lazy dog"
# through `addnarration!`, i.e. the analysis executor — calling `render!` on the
# main thread beside a live Player trips the GPU's single-writer submit channel
VE.addnarration!(player0, said)
for _ in 1:180
    (!isempty(player0.sequence.narration) &&
     !isempty(player0.sequence.narration[1].samples)) && break
    sleep(1.0)
end
nar = player0.sequence.narration[1]
println("kokoro produced ", length(nar.samples), " samples at ", nar.rate, " Hz")
close(player0)

raw = joinpath(dir, "speech.pcm"); wav = joinpath(dir, "speech.wav")
write(raw, reinterpret(UInt8, round.(Int16, clamp.(nar.samples, -1, 1) .* 32767)))
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f s16le -ac 1 -ar $(nar.rate) -i $raw $wav`,
             stdout = devnull, stderr = devnull))
# 2. mux it onto a video so the editor sees a source WITH audio
vid = joinpath(dir, "spoken.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=320x180:rate=30
              -i $wav -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac $vid`,
             stdout = devnull, stderr = devnull))
println("built a source with audio: ", isfile(vid))

# 3. Whisper HEARS it
player = VE.Player(vid); sleep(3.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
VE.runtranscribe!(player)
ok = false
for _ in 1:180
    isempty(player.sequence.captions) || (global ok = true; break)
    sleep(1.0)
end
println("transcribed: ", ok)
if ok
    for c in player.sequence.captions
        println("  [", round(c.start; digits=2), "-", round(c.stop; digits=2), "] ", c.text)
    end
end
println("status: ", player.status[])
close(player)
