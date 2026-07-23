"""
    VideoSource(path)

Probed metadata for a video file: dimensions, framerate, duration, frame count
and the keyframe index (scanned from packet flags, no decoding).

Frame indices are 0-based throughout: frame `n` is displayed at `n / framerate`.
"""
struct VideoSource
    path::String
    width::Int
    height::Int
    framerate::Float64
    duration::Float64
    nframes::Int
    keyframe_times::Vector{Float64}
end

function VideoSource(path::AbstractString)
    isfile(path) || error("no such file: $path")
    reader = VideoIO.openvideo(path)
    width, height = VideoIO.out_frame_size(reader)
    fps = Float64(VideoIO.framerate(reader))
    close(reader)
    duration = VideoIO.get_duration(path)
    keyframes, npackets = scan_packets(path)
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
    return VideoSource(String(path), width, height, fps, duration, nframes, keyframes)
end

"""
    scan_packets(path) -> (keyframe_times::Vector{Float64}, npackets::Int)

Keyframe timestamps in seconds plus the TRUE packet count, read from packet
flags via ffprobe. Demux only — no decoding, fast even for long files. The
count is authoritative where containers (mkv!) carry no `nb_frames`.
"""
function scan_packets(path::AbstractString)
    cmd = `$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $path`
    times = Float64[]
    n = 0
    for line in eachline(cmd)
        parts = split(line, ',')
        length(parts) >= 2 || continue
        n += 1
        if occursin('K', parts[2])
            t = tryparse(Float64, parts[1])
            t === nothing || push!(times, t)
        end
    end
    sort!(times)
    return times, n
end

frametime(src::VideoSource, n::Integer) = n / src.framerate
frameindex(src::VideoSource, t::Real) = clamp(round(Int, t * src.framerate), 0, src.nframes - 1)

"Time of the last keyframe at or before `t` (falls back to 0.0)."
function nearest_keyframe(src::VideoSource, t::Real)
    isempty(src.keyframe_times) && return 0.0
    i = searchsortedlast(src.keyframe_times, t)
    return i < 1 ? 0.0 : src.keyframe_times[i]
end
