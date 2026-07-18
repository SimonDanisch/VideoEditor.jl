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
    counted = VideoIO.get_number_frames(path)
    nframes = counted === nothing ? round(Int, duration * fps) : counted
    return VideoSource(String(path), width, height, fps, duration, nframes, scan_keyframes(path))
end

"""
    scan_keyframes(path) -> Vector{Float64}

Keyframe timestamps in seconds, read from packet flags via ffprobe.
Demux only — no decoding, fast even for long files.
"""
function scan_keyframes(path::AbstractString)
    cmd = `$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $path`
    times = Float64[]
    for line in eachline(cmd)
        parts = split(line, ',')
        length(parts) >= 2 || continue
        if occursin('K', parts[2])
            t = tryparse(Float64, parts[1])
            t === nothing || push!(times, t)
        end
    end
    sort!(times)
    return times
end

frametime(src::VideoSource, n::Integer) = n / src.framerate
frameindex(src::VideoSource, t::Real) = clamp(round(Int, t * src.framerate), 0, src.nframes - 1)

"Time of the last keyframe at or before `t` (falls back to 0.0)."
function nearest_keyframe(src::VideoSource, t::Real)
    isempty(src.keyframe_times) && return 0.0
    i = searchsortedlast(src.keyframe_times, t)
    return i < 1 ? 0.0 : src.keyframe_times[i]
end
