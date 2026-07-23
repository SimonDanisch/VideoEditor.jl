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
- `backend`: KA backend for the render chain. A GPU backend (e.g. `LavaBackend()`)
  runs tracks, effects, blends and the crop warp on device buffers — decode and
  encode stay on the CPU with one upload/download per frame. Same kernels either
  way, so the output is identical.
- `progress`: called as `progress(done, total)` every 30 frames.

Gaps render black (and sound silent).
"""
function exportvideo(path::AbstractString, seq::Sequence;
                     size::Union{Nothing, Tuple{Int, Int}} = nothing,
                     framerate::Real = seq.framerate,
                     codec_name::Union{Nothing, String} = nothing,
                     encoder_options::NamedTuple = (crf = 20, preset = "medium"),
                     audio::Bool = true,
                     backend = KA.CPU(),
                     progress = nothing)
    total = seqlength(seq)
    total > 0 || error("empty sequence")   # before canvassize: it indexes clips[1]
    canvas = something(size, canvassize(seq))
    wantaudio = audio && any(hasaudio, unique(c.source.path for c in seq.clips))
    videopath = wantaudio ? tempname() * ".mp4" : path

    outbuf = alloccanvas(backend, (canvas[1], canvas[2]))
    transbuf = allocframe(backend, (canvas[1], canvas[2]))   # incoming side of a transition
    layerbuf = allocframe(backend, (canvas[1], canvas[2]))   # one track layer while compositing
    hostout = zeros(RGB{N0f8}, canvas[1], canvas[2])         # encode staging (download target)
    blackhost = zeros(RGB{N0f8}, canvas[1], canvas[2])       # gap/composite base for device canvases
    readers = Dict{String, Any}()   # per-source decoder: GpuVideoStream or SequentialReader
    fxbufs = Dict{String, NTuple{4, AnyRGBFrame}}()  # per-source (host, chain triple)

    # ffmpeg takes the rate as an Int32 AVRational: a raw measured float
    # (23.975288…) converts to an exact Rational with a 2^47 denominator and
    # overflows — rationalize to the smallest fraction within a millihertz
    fr = rationalize(Float64(framerate); tol = 1e-6)
    writer = VideoIO.open_video_out(videopath, RGB{N0f8}, (canvas[2], canvas[1]);
                                    framerate = fr, codec_name = codec_name,
                                    encoder_options = encoder_options)
    try
        for n in 0:(total - 1)
            tr = transitionat(seq, n)
            sample = tr === nothing ? nothing : transitionsample(seq, tr, n)
            if sample !== nothing
                left, srcA, right, srcB, p = sample
                rendercanvas!(outbuf, left, srcA, readers, fxbufs)
                rendercanvas!(transbuf, right, srcB, readers, fxbufs)
                blend!(outbuf, outbuf, transbuf, p)
            elseif ntracks(seq) > 1 && length(clipsat(seq, n)) > 1
                fillblack!(outbuf, blackhost)                   # composite the track stack
                for clip in clipsat(seq, n)                     # bottom → top
                    sf = clip.src_in + (n - clip.start)
                    rendercanvas!(layerbuf, clip, sf, readers, fxbufs; skipopacity = true)
                    blend!(outbuf, outbuf, layerbuf, Float32(clamp(paramvalue(clip, :opacity, sf), 0.0, 1.0)))
                end
            else
                loc = locate(seq, n)
                loc === nothing ? fillblack!(outbuf, blackhost) :
                                  rendercanvas!(outbuf, loc[1], loc[2], readers, fxbufs)
            end
            writeframe!(writer, outbuf, hostout, backend)
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
                   backend = KA.CPU(), progress = nothing)
    seqlength(seq) > 0 || error("empty sequence")
    tmp = tempname() * ".mp4"
    palette = tempname() * ".png"
    try
        exportvideo(tmp, seq; audio = false, backend = backend, progress = progress)
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

"Frame-sized working buffer on `backend` (a plain host frame on the CPU)."
allocframe(::KA.CPU, dims::NTuple{2, Int}) = RGBFrame(undef, dims...)
allocframe(backend, dims::NTuple{2, Int}) = KA.allocate(backend, RGB{N0f8}, dims)

"The export canvas (downloaded to the host every frame; stays device-local —
BAR/unified memory reads ~70 MB/s from the CPU, 8× slower than staged readback)."
alloccanvas(::KA.CPU, dims::NTuple{2, Int}) = zeros(RGB{N0f8}, dims...)
alloccanvas(backend, dims::NTuple{2, Int}) = KA.allocate(backend, RGB{N0f8}, dims)

"Black canvas for gaps and the composite base (device canvases copy a host zero frame)."
fillblack!(buf::RGBFrame, blackhost) = fill!(buf, RGB{N0f8}(0, 0, 0))
fillblack!(buf, blackhost) = copyto!(buf, blackhost)

"Publish the composed canvas to the encoder (device canvases download via `hostout`)."
writeframe!(writer, canvas::RGBFrame, hostout, backend) =
    write(writer, PermutedDimsArray(canvas, (2, 1)))
function writeframe!(writer, canvas, hostout, backend)
    KA.synchronize(backend)
    copyto!(hostout, canvas)
    write(writer, PermutedDimsArray(hostout, (2, 1)))
end

"""
The per-source export decoder: a [`GpuVideoStream`](@ref) on a GPU backend when the
source hardware-decodes (frames land device-resident through the chunked session —
no host round-trip, the biggest export cost on the CPU path), else the CPU
[`SequentialReader`](@ref).
"""
function opendecoder(source::VideoSource, backend)
    backend isa KA.CPU && return SequentialReader(source)
    try
        return openstream(backend, source.path, source.width, source.height)
    catch e
        @warn "GPU stream unavailable for export — CPU decode" source = source.path exception = e
        return SequentialReader(source)
    end
end

"""
    decodeinto!(dest, host, decoder, srcframe)

Decode source frame `srcframe` into `dest`, EXACTLY — an export must never get a
nearest-frame stand-in. Dispatches on the decoder: a [`GpuVideoStream`](@ref)
decodes device-resident via [`exactframeat!`](@ref); a [`SequentialReader`](@ref)
decodes into the `host` staging buffer and uploads only when `dest` lives on a
device.
"""
function decodeinto!(dest::AnyRGBFrame, host::RGBFrame, s::GpuVideoStream, srcframe::Integer)
    f = exactframeat!(s, srcframe)
    nv12torgb!(dest, f.y, f.uv; bt601 = s.bt601)
    return dest
end

function decodeinto!(dest::AnyRGBFrame, host::RGBFrame, sr::SequentialReader, srcframe::Integer)
    readframe!(host, sr, srcframe)
    dest === host || copyto!(dest, host)
    return dest
end

"""
Render `clip` at source frame `srcframe` — decode, motion/color tracks, effect
stack, then warp its crop into `dest` (a canvas-sized buffer). `readers`/`fxbufs`
cache one decoder and one scratch triple per source path.
"""
function rendercanvas!(dest::AnyRGBFrame, clip::Clip, srcframe::Integer,
                       readers::Dict{String, Any},
                       fxbufs::Dict{String, NTuple{4, AnyRGBFrame}}; skipopacity::Bool = false)
    bk = KA.get_backend(dest)
    sr = get!(() -> opendecoder(clip.source, bk), readers, clip.source.path)
    host, frame, fx1, fx2 = get!(fxbufs, clip.source.path) do
        w, h = clip.source.width, clip.source.height
        host = RGBFrame(undef, w, h)                  # decode target (VideoIO needs host memory)
        mk() = bk isa KA.CPU ? RGBFrame(undef, w, h) : KA.allocate(bk, RGB{N0f8}, (w, h))
        (host, bk isa KA.CPU ? host : mk(), mk(), mk())   # on CPU the chain runs in `host` itself
    end
    decodeinto!(frame, host, sr, srcframe)
    clip = effectiveclip(clip, srcframe)  # keyframed params baked at this frame
    applymotiontrack!(frame, fx1, clip, srcframe)
    applycolortrack!(frame, clip, srcframe)
    if skipopacity                        # compositing: opacity is the layer alpha, not fade-to-black
        for e in clip.effects
            (e isa OpacityEffect || isneutral(e)) && continue
            applyeffect!(frame, fx1, fx2, e)
        end
        KA.synchronize(KA.get_backend(frame))
    else
        applyeffects!(frame, fx1, fx2, clip)
    end
    if clip.crop == (0.0, 0.0, 1.0, 1.0) && Base.size(frame) == Base.size(dest)
        copyto!(dest, frame)
    else
        warp!(dest, frame, clip.crop)
        KA.synchronize(KA.get_backend(dest))
    end
    return dest
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
