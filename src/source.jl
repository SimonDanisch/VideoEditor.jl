"""
    VideoSource(path)

Probed metadata for a video file: dimensions, framerate, duration, frame count,
the keyframe index and the timestamp of EVERY frame (scanned from packet flags,
no decoding).

Frame indices are 0-based throughout. `n / framerate` is only the display time of
frame `n` on constant-rate material — phone clips drop frames (measured on a
"60 fps" clip: median period 0.0167 s, longest gap 0.2 s), and then the n-th
DECODED frame and the frame at time `n/60` are different pictures. Analysis reads
sequentially while the preview seeks by time, so every per-frame track (flicker,
stabilization) landed on the wrong frame — 11 frames off at frame 100 on that
clip. `frametimes` makes the mapping exact for both.
"""
struct VideoSource
    path::String
    width::Int
    height::Int
    framerate::Float64
    duration::Float64
    nframes::Int
    keyframe_times::Vector{Float64}
    frametimes::Vector{Float64}   # display time of every frame, in display order
end

function VideoSource(path::AbstractString)
    isfile(path) || error("no such file: $path")
    reader = VideoIO.openvideo(path)
    width, height = VideoIO.out_frame_size(reader)
    fps = Float64(VideoIO.framerate(reader))
    close(reader)
    duration = VideoIO.get_duration(path)
    keyframes, frametimes = scan_packets(path)
    npackets = length(frametimes)
    counted = VideoIO.get_number_frames(path)
    nframes = something(counted, npackets > 0 ? npackets : round(Int, duration * fps))
    # containers may claim a track rate the stream doesn't deliver (YouTube mkv
    # remuxes report 29.97 while frames actually arrive at 23.976) — when the
    # true frame count disagrees with duration × claimed rate, the EFFECTIVE
    # rate is the one every frame↔time mapping (and the proxy check) must use.
    # Snap it to the nearest standard video rate: count/duration carries the
    # container's rounding noise, and a raw float like 23.975288… later
    # explodes into a 2^47 denominator when converted to ffmpeg's Int32
    # AVRational at export time.
    if nframes > 0 && duration > 0 && abs(nframes / duration - fps) / fps > 0.01
        eff = nframes / duration
        std = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0,
               60000 / 1001, 60.0, 120000 / 1001, 120.0)
        near = findfirst(r -> abs(eff - r) / r < 0.002, std)
        fps = near === nothing ? eff : std[near]
    end
    return VideoSource(String(path), width, height, fps, duration, nframes, keyframes,
                       length(frametimes) == nframes ? frametimes : Float64[])
end

"""
    scan_packets(path) -> (keyframe_times, frame_times)

Timestamps of every frame in DISPLAY order, plus the subset that are keyframes —
read from packet flags via ffprobe. Demux only, no decoding, fast even for long
files. The frame count implied here is authoritative where containers (mkv!)
carry no `nb_frames`, and the per-frame times are what makes index↔time exact on
variable-rate material.
"""
function scan_packets(path::AbstractString)
    cmd = `$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $path`
    keys = Float64[]
    times = Float64[]
    for line in eachline(cmd)
        parts = split(line, ',')
        length(parts) >= 2 || continue
        t = tryparse(Float64, parts[1])
        t === nothing && continue
        push!(times, t)
        occursin('K', parts[2]) && push!(keys, t)
    end
    # packets arrive in DECODE order (B-frames!); display order is by timestamp
    sort!(times)
    sort!(keys)
    return keys, times
end

"Display time of frame `n` — the scanned timestamp when we have one, else the
constant-rate assumption."
frametime(src::VideoSource, n::Integer) =
    1 <= n + 1 <= length(src.frametimes) ? src.frametimes[n + 1] : n / src.framerate

"Frame displayed at time `t` — the inverse of [`frametime`](@ref)."
function frameindex(src::VideoSource, t::Real)
    isempty(src.frametimes) && return clamp(round(Int, t * src.framerate), 0, src.nframes - 1)
    i = searchsortedlast(src.frametimes, Float64(t) + 1.0e-6)
    return clamp(i - 1, 0, src.nframes - 1)
end

"Time of the last keyframe at or before `t` (falls back to 0.0)."
function nearest_keyframe(src::VideoSource, t::Real)
    isempty(src.keyframe_times) && return 0.0
    i = searchsortedlast(src.keyframe_times, t)
    return i < 1 ? 0.0 : src.keyframe_times[i]
end
