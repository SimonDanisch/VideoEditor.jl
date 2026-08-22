# Is a seek's cost the DISTANCE past the last keyframe?
#
# The theory after playback_check.jl: a full-quality seek must decode forward from
# the preceding keyframe, so its cost is (frames since that keyframe) x per-frame
# decode. This clip's keyframes are 4.17 s apart — a 250-frame GOP at 60 fps — and
# the measured median seek was 250 ms, which fits too neatly to trust unaudited.
#
# The prediction is specific and easy to falsify: land ON a keyframe and the seek
# should be near-free; land just BEFORE the next one and it should be worst. If
# cost is flat across the GOP, the theory is wrong and the time is somewhere else.
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

source  = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
offsets = [0, 10, 30, 60, 120, 180, 240]   # frames past a keyframe
reps    = 5

GLMakie.activate!(; visible = true, framerate = 60)
p = VE.Player(source)
sleep(6.0)
seq = p.sequence
src = seq.clips[1].source
fps = src.framerate

kf = [round(Int, t * fps) for t in src.keyframe_times]
println("keyframes at frames: ", kf[1:min(8, end)], " …  GOP ≈ ",
        length(kf) > 2 ? round(Int, (kf[end] - kf[2]) / (length(kf) - 2)) : 0, " frames")

for n in 0:25:300                     # warm
    VE.showframe!(p, n; standin = false)
end
sleep(1.5)

"Time until the frame at `n` is actually on screen, retrying as the editor does."
function untilshown(p, n; budget = 4.0)
    t0 = time_ns()
    got = false
    while !got && (time_ns() - t0) / 1e9 < budget
        got = VE.showframe!(p, n; standin = false) === true
        got || sleep(0.002)
    end
    return ((time_ns() - t0) / 1e6, got)
end

println("\noffset-past-keyframe   median ms   landed")
for off in offsets
    ts = Float64[]
    ok = 0
    for r in 1:reps
        k = kf[2 + (r % max(length(kf) - 3, 1))]        # a different GOP each rep
        n = min(k + off, VE.seqlength(seq) - 2)
        VE.showframe!(p, 0; standin = false)            # leave the GOP between samples
        sleep(0.25)
        t, got = untilshown(p, n)
        push!(ts, t); ok += got
    end
    sort!(ts)
    println(rpad("+$off", 22), rpad(string(round(ts[cld(end, 2)]; digits = 1)), 12),
            "$ok/$reps")
end

close(p)
