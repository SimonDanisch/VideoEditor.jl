"""
    exportvideo(path, seq; kwargs...) -> path

Render the sequence to a video file. Uses the same resolution machinery as
the preview: timeline frame → `locate` → source frame → effect stack →
crop warp — only here the crop is baked into the pixels and every timeline
frame is written.

Keywords:
- `size`: output `(width, height)`; defaults to [`canvassize`](@ref) —
  the first clip's crop region in source pixels.
- `framerate`: defaults to the sequence framerate.
- `codec_name` / `encoder_options`: passed through to `VideoIO.open_video_out`
  (default H.264, `crf=20, preset="medium"`).
- `audio`: mux the sources' audio along the cut list (see [`muxaudio`](@ref));
  on by default, skipped automatically when no source has an audio stream.
- `progress`: called as `progress(done, total)` every 30 frames.

Gaps render black (and sound silent).
"""
function exportvideo(path::AbstractString, seq::Sequence;
                     size::Union{Nothing, Tuple{Int, Int}} = nothing,
                     framerate::Real = seq.framerate,
                     codec_name::Union{Nothing, String} = nothing,
                     encoder_options::NamedTuple = (crf = 20, preset = "medium"),
                     audio::Bool = true,
                     progress = nothing)
    total = seqlength(seq)
    total > 0 || error("empty sequence")   # before canvassize: it indexes clips[1]
    canvas = something(size, canvassize(seq))
    wantaudio = audio && any(hasaudio, unique(c.source.path for c in seq.clips))
    videopath = wantaudio ? tempname() * ".mp4" : path

    outbuf = zeros(RGB{N0f8}, canvas[1], canvas[2])
    tmp1 = RGBFrame(undef, canvas[1], canvas[2])
    tmp2 = RGBFrame(undef, canvas[1], canvas[2])
    readers = Dict{String, SequentialReader}()
    fxbufs = Dict{String, NTuple{3, RGBFrame}}()  # per-source full-res scratch

    writer = VideoIO.open_video_out(videopath, RGB{N0f8}, (canvas[2], canvas[1]);
                                    framerate = framerate, codec_name = codec_name,
                                    encoder_options = encoder_options)
    try
        for n in 0:(total - 1)
            loc = locate(seq, n)
            if loc === nothing
                fill!(outbuf, RGB{N0f8}(0, 0, 0))
            else
                clip, srcframe = loc
                sr = get!(() -> SequentialReader(clip.source), readers, clip.source.path)
                frame, fx1, fx2 = get!(() -> ntuple(_ -> RGBFrame(undef, clip.source.width, clip.source.height), 3),
                                       fxbufs, clip.source.path)
                readframe!(frame, sr, srcframe)
                applymotiontrack!(frame, fx1, clip, srcframe)
                applycolortrack!(frame, clip, srcframe)
                applyeffects!(frame, fx1, fx2, clip)
                if clip.crop == (0.0, 0.0, 1.0, 1.0) && Base.size(frame) == Base.size(outbuf)
                    copyto!(outbuf, frame)
                else
                    warp!(outbuf, frame, clip.crop)
                    KA.synchronize(KA.get_backend(outbuf))
                end
            end
            write(writer, PermutedDimsArray(outbuf, (2, 1)))
            progress === nothing || n % 30 == 0 && progress(n + 1, total)
        end
    finally
        VideoIO.close_video_out!(writer)
        foreach(close, values(readers))
    end
    if wantaudio
        muxaudio(videopath, seq, path)
        rm(videopath)
    end
    progress === nothing || progress(total, total)
    return path
end

"""
    exportgif(path, seq; fps=15, loop=0, width=nothing, progress=nothing) -> path

Export the sequence to an animated GIF. Renders the timeline (video-only,
through the same pipeline as [`exportvideo`](@ref)) to a temp file, then builds
an optimized palette (`palettegen`/`paletteuse`, Lanczos scaling) for clean
colors. `fps` sets the GIF frame rate, `width` optionally downscales (height
follows the aspect), and `loop` sets looping: `0` loops forever, `-1` plays
once, `n` repeats `n` extra times.
"""
function exportgif(path::AbstractString, seq::Sequence; fps::Real = 15,
                   loop::Integer = 0, width::Union{Nothing, Integer} = nothing,
                   progress = nothing)
    seqlength(seq) > 0 || error("empty sequence")
    tmp = tempname() * ".mp4"
    palette = tempname() * ".png"
    try
        exportvideo(tmp, seq; audio = false, progress = progress)
        scale = width === nothing ? "scale=trunc(iw/2)*2:-2:flags=lanczos" :
                "scale=$(Int(width)):-2:flags=lanczos"
        vf = "fps=$(fps),$(scale)"
        ff = FFMPEG_jll.ffmpeg()
        run(pipeline(`$ff -y -i $tmp -vf "$vf,palettegen=stats_mode=diff" $palette`;
                     stdout = devnull, stderr = devnull))
        run(pipeline(`$ff -y -i $tmp -i $palette
                      -lavfi "$vf [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3"
                      -loop $(Int(loop)) $path`; stdout = devnull, stderr = devnull))
    finally
        isfile(tmp) && rm(tmp)
        isfile(palette) && rm(palette)
    end
    return path
end

"""
    canvassize(seq) -> (width, height)

Default export canvas: the first clip's crop region in source pixels
(even multiples, as encoders require), or the source size when uncropped.
"""
function canvassize(seq::Sequence)
    clip = seq.clips[1]
    w = round(Int, clip.crop[3] * clip.source.width)
    h = round(Int, clip.crop[4] * clip.source.height)
    return (max(2 * (w ÷ 2), 2), max(2 * (h ÷ 2), 2))
end

"""
Synchronous decoder for export: sequential reads, seeking only when the
requested frame isn't the next one (clip boundaries, gaps).
"""
mutable struct SequentialReader
    const source::VideoSource
    const reader::VideoIO.VideoReader
    position::Int  # next frame the reader will produce

    function SequentialReader(source::VideoSource)
        reader = VideoIO.openvideo(source.path, target_format = VideoIO.AV_PIX_FMT_RGB24)
        return new(source, reader, 0)
    end
end

function readframe!(dest::RGBFrame, sr::SequentialReader, n::Integer)
    if n != sr.position
        seek(sr.reader, frametime(sr.source, n))
        sr.position = n
    end
    read!(sr.reader, PermutedDimsArray(dest, (2, 1)))
    sr.position += 1
    return dest
end

Base.close(sr::SequentialReader) = close(sr.reader)
