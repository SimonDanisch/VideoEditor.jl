# How fast do playback and seeking ACTUALLY feel, on the clip Simon edits?
#
# Three traps this is built around, each of which produced a confident wrong
# number before:
#
#  1. `playloop` is WALL-CLOCK driven — it computes the target frame from elapsed
#     time and sets `playhead` to it, so a slow renderer DROPS frames rather than
#     running slower. Timing the playhead across the timeline measures the
#     TIMELINE, not the frame rate: 90 frames over 6 s is "15.0 fps" however the
#     renderer did.
#  2. `player.frame` is NOT the presentation signal. The GPU preview path
#     (`presentgpu!`) swaps a texture and sets `requires_update`; it never
#     notifies `frame`. Counting `on(p.frame)` over six seconds of playback on a
#     1080x1920 clip counts ZERO — measured, and it is the harness that is wrong.
#  3. `visible = false` composites nothing.
#
# So: time `showframe!` itself. It is synchronous, returns whether the frame
# reached the screen, and is exactly what the playhead observer calls. Its
# `standin` default is `!atrest(player)`, so playback and seeking really do take
# different paths and are measured separately rather than assumed equal.
#
# Run on :1. Re-assignable params, no ARGS.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie
import VideoEditor as VE

source = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"   # 1080x1920, 60 fps
nplay  = 240      # sequential frames per playback sample
nseek  = 40       # scattered frames per seek sample

# The A/B for `readerkey`. With VE_FORCE_SHARE=1 every clip of one file shares one
# read head, which is what the code did before the split — so the "far apart"
# block below can be compared against itself instead of against a guess. Patched
# BEFORE the Player exists, so no call is left in an older world age (an @eval'd
# redefinition is invisible to a function already running, and that trap has
# reported a clean 0.0% difference here before).
if get(ENV, "VE_FORCE_SHARE", "0") == "1"
    VE.eval(:(farapart(a::Clip, b::Clip) = false))
    println("[forcing ONE read head per source — pre-split behaviour]")
end

GLMakie.activate!(; visible = true, framerate = 60)

"""
Milliseconds per `showframe!`, plus how many landed on screen.

`shown` below 100% is the number that matters as much as the time: a frame that
returned `false` fell back or gave up, and a fast miss is not a fast frame.
"""
function presentcost(p, frames; standin::Bool, playing::Bool = false)
    # `playing` selects the real PLAYBACK path: the stand-in decode budget is
    # gated on being at rest, so measuring playback with the flag down measures
    # the seek path instead and quietly reports 92% where the editor gets 100%.
    p.playing[] = playing
    VE.showframe!(p, first(frames); standin = standin)   # warm this path
    sleep(0.2)
    ts = Float64[]
    ok = 0
    g0 = Base.gc_num().total_time
    for n in frames
        t = time_ns()
        got = VE.showframe!(p, n; standin = standin)
        push!(ts, (time_ns() - t) / 1e6)
        ok += got === true
    end
    gcms = (Base.gc_num().total_time - g0) / 1e6
    p.playing[] = false
    sort!(ts)
    return (med = ts[cld(end, 2)], p90 = ts[cld(9 * end, 10)], worst = last(ts),
            shown = ok / length(ts), gcfrac = gcms / sum(ts))
end

"""
Seek latency as the USER experiences it: playhead set → frame actually on screen.

`presentcost` measures one `showframe!` call, and on a full-quality seek that call
returns `false` most of the time (measured: 15% shown on one clip) because the
frame is not decoded yet. The time of a FAILED attempt is not the latency of a
seek — `retrypresent` keeps trying and the picture appears later. This retries the
way the editor does and times until it lands.
"""
function seektoshown(p, frames; budget = 3.0)
    ts = Float64[]
    landed = 0
    for n in frames
        t0 = time_ns()
        got = false
        while !got && (time_ns() - t0) / 1e9 < budget
            got = VE.showframe!(p, n; standin = false) === true
            got || sleep(0.002)
        end
        push!(ts, (time_ns() - t0) / 1e6)
        landed += got
        sleep(0.05)          # let the ring settle, as a human between seeks would
    end
    sort!(ts)
    return (med = ts[cld(end, 2)], p90 = ts[cld(9 * end, 10)], worst = last(ts),
            shown = landed / length(ts), gcfrac = 0.0)
end

function report(tag, c)
    println(rpad(tag, 24),
            rpad(string(round(c.med; digits = 1)), 7), "ms med  (",
            rpad(string(round(1000 / c.med; digits = 1)), 6), "fps)   p90 ",
            rpad(string(round(c.p90; digits = 1)), 7), "worst ",
            rpad(string(round(c.worst; digits = 1)), 8),
            "shown ", round(Int, 100 * c.shown), "%  GC ", round(Int, 100 * c.gcfrac), "%")
end

p = VE.Player(source)
sleep(6.0)
resize!(p.fig, 1500, 950)
sleep(2.0)
seq = p.sequence
println("source ", VE.seqlength(seq), " frames @ ", seq.framerate, " fps   budget ",
        round(1000 / seq.framerate; digits = 1), " ms/frame")

# warm the whole path once — first touch compiles shaders and fills the ring
for n in 0:25:400
    VE.showframe!(p, n; standin = false)
end
sleep(1.5)

playframes(seq) = 1:min(nplay, VE.seqlength(seq) - 1)
# scattered, deterministic (golden-ratio hops), so every seek is a ring MISS
seekframes(seq) = [round(Int, (VE.seqlength(seq) - 2) * ((k * 0.6180339887) % 1.0)) + 1
                   for k in 1:nseek]

function sample(tag, seq)
    println("\n── ", tag, " ──   readerkeys: ",
            unique(VE.readerkey(seq, c) === c.source ? :shared : :own for c in seq.clips))
    report("playback (standin)", presentcost(p, playframes(seq); standin = true, playing = true))
    # what the user now WAITS for: `retrypresent` draws the nearest decoded frame
    # first, so this is the felt latency of a seek…
    report("seek → picture",     presentcost(p, seekframes(seq); standin = true))
    # …and this is when it finishes sharpening to the exact frame, underneath it
    report("seek → exact",       seektoshown(p, seekframes(seq)))
end

sample("1 clip", seq)

# The case Simon actually hit: a CLONE of the same clip, overlapping. At the SAME
# source position the two read heads want the same frames, so `readerkey` shares
# one — the arrangement its docstring says is faster to share.
base = seq.clips[1]
clone = VE.copyclip(base; start = base.start, track = 2)
push!(seq.clips, clone)
VE.refreshedit!(p)
sleep(2.0)
sample("2 clips, same position", seq)

# …and the arrangement `readerkey` EXISTS for: overlapping in time but far apart
# in the source, so one read head would have to seek back and forth across the
# ring for every frame. This is the case that has to justify the split.
clone.src_in += 4 * VE.GPU_STREAM_CAPACITY
clone.src_out += 4 * VE.GPU_STREAM_CAPACITY
VE.refreshedit!(p)
sleep(2.0)
sample("2 clips, far apart", seq)

close(p)
