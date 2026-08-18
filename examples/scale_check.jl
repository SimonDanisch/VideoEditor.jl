# Does the editor hold up at the size a real edit actually is?
#
# Everything in the suite runs on one to three clips. A real timeline is dozens
# across several tracks, and the things that break at that size are exactly the
# ones a small fixture cannot show: panel rebuild cost per playhead move, the
# timeline's per-clip plots, and whether the fx panel's signature work is O(1) or
# O(clips) on every frame change.
#
# Run on :1.

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
fig = player.fig; resize!(fig, 1600, 1000); sleep(4.0)
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false
seq = player.sequence

# 24 clips over 3 tracks — a modest documentary sequence, not a stress test
base = seq.clips[1]
base.src_out = base.src_in + 20
at = VE.clipend(base)
for k in 1:23
    track = 1 + (k % 3)
    c = VE.copyclip(base; start = at, track = track)
    push!(seq.clips, c)
    track == 1 && (global at = at + VE.cliplength(base))
end
sort!(seq.clips, by = c -> (c.track, c.start))
VE.refreshedit!(player); sleep(2.0)
println("clips = ", length(seq.clips), "   tracks = ", VE.ntracks(seq),
        "   length = ", VE.seqlength(seq), " frames")

VE.opendock!(player, :effects); sleep(1.5)
player.fxwidgets[:fxlistrefresh](); sleep(1.5)

# how long does one playhead move cost, with the panel open?
player.timeline.selected[] = 1
VE.refreshedit!(player); sleep(0.5)
# time the LOOP only — a trailing sleep in the measured window turns 37 ms/move
# into "4.25 s" and hides what the number actually is
player.playhead[] = 0; sleep(0.5)
t0 = time()
for n in 0:59
    player.playhead[] = n
end
el = time() - t0
println("60 playhead moves with the panel open: ", round(1000 * el; digits = 1), " ms total, ",
        round(1000 * el / 60; digits = 1), " ms each")
sleep(1.5)

# WITHIN one clip: no clip boundary is crossed, so `effsig` never changes and
# `rebuildstack` must early-return. If this is cheap and the 0..59 sweep is not,
# the cost is rebuilding the panel at each boundary, not the panel per se.
first_len = VE.cliplength(seq.clips[1])
player.playhead[] = seq.clips[1].start; sleep(0.5)
t3 = time()
for k in 0:59
    player.playhead[] = seq.clips[1].start + (k % max(first_len - 1, 1))
end
println("60 moves INSIDE one clip, panel open: ",
        round(1000 * (time() - t3) / 60; digits = 1), " ms each")
sleep(1.0)

# …and with the panel CLOSED, to say whether the cost is the panel or the decode
VE.opendock!(player, :none); sleep(1.0)
player.playhead[] = 0; sleep(0.5)
t2 = time()
for n in 0:59
    player.playhead[] = n
end
el2 = time() - t2
println("60 playhead moves with the panel CLOSED: ", round(1000 * el2 / 60; digits = 1), " ms each")
VE.opendock!(player, :effects); sleep(1.0)

# and the rebuild signature itself, which runs on EVERY move
clip = VE.editclip(player) === nothing ? seq.clips[1] : VE.editclip(player)[1]
VE.docsig(seq, clip); VE.effsig(clip)          # warm
t1 = time(); for _ in 1:1000; VE.docsig(seq, clip); VE.effsig(clip); end
println("1000 signature computations: ", round(1000 * (time() - t1); digits = 2), " ms")

Makie.save("/sim/Programmieren/VideoEdit/media/scale_24clips.png",
           Makie.colorbuffer(player.screen))
println("timeline clip plots: ", length(player.timeline.clipplots))
close(player)
