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
- `pixel_format`: the encoded chroma layout, `yuv420p` by default. Left to
  ffmpeg it would be `yuv444p` — RGB in, so 4:4:4 is the "best match" and no
  chroma is thrown away — but that lands the file in H.264 *High 4:4:4
  Predictive*, which no phone, TV or browser hardware decoder will touch. An
  export that won't play on the device it was shot on is not an export. Pass
  `VideoIO.AV_PIX_FMT_YUV444P` if you want the extra chroma and control the
  player.
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
                     pixel_format = VideoIO.AV_PIX_FMT_YUV420P,
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
    engine = FxEngine(backend)      # the SAME effect graph the preview runs, exact policy

    # ffmpeg takes the rate as an Int32 AVRational: a raw measured float
    # (23.975288…) converts to an exact Rational with a 2^47 denominator and
    # overflows — rationalize to the smallest fraction within a millihertz
    fr = rationalize(Float64(framerate); tol = 1e-6)
    writer = VideoIO.open_video_out(videopath, RGB{N0f8}, (canvas[2], canvas[1]);
                                    framerate = fr, codec_name = codec_name,
                                    encoder_options = encoder_options,
                                    target_pix_fmt = pixel_format)
    try
        for n in 0:(total - 1)
            renderframe!(outbuf, seq, n, readers, engine; scratch = transbuf, black = blackhost)
            writeframe!(writer, outbuf, hostout, backend)
            progress === nothing || n % 30 == 0 && progress(n + 1, total)
        end
    finally
        VideoIO.close_video_out!(writer)
        foreach(close, values(readers))
        emptyengine!(engine)
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
requested frame isn't the next one (clip boundaries, gaps). A graph source —
[`sourceinto!`](@ref) decodes into its host staging buffer and uploads only
when the graph runs on a device.
"""
mutable struct SequentialReader
    const source::VideoSource
    const reader::VideoIO.VideoReader
    const host::RGBFrame   # decode staging (VideoIO needs host memory)
    position::Int  # next frame the reader will produce

    function SequentialReader(source::VideoSource)
        reader = VideoIO.openvideo(source.path, target_format = VideoIO.AV_PIX_FMT_RGB24)
        return new(source, reader,
                   RGBFrame(undef, source.width, source.height), 0)
    end
end

framesize(sr::SequentialReader) = (sr.source.width, sr.source.height)

function sourceinto!(out, sr::SequentialReader, frame; prefetch::Bool = false,
                     served = nothing, exact::Bool = false)
    readframe!(sr.host, sr, frame)   # always exact — sequential decode by construction
    out === sr.host || copyto!(out, sr.host)
    return out
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
The per-source export decoder — a DETERMINISTIC lane, decided by capability, not
by catching errors: a GPU backend streams every codec the hardware session
decodes (frames land device-resident through the chunked session — no host
round-trip); everything else reads sequentially on the CPU. A failure to open
the declared lane fails the export loudly instead of silently switching.
"""
opendecoder(source::VideoSource, ::KA.CPU) = SequentialReader(source)
function opendecoder(source::VideoSource, backend)
    videocodec(source.path) in (:h264, :hevc) || return SequentialReader(source)
    # A mezzanine, when one exists, IS the decodable version of this source — the
    # streaming preview already opens on it, and this path did not, so a source
    # that plays fine (open-GOP HEVC, transcoded once) still threw
    # "not GOP-seekable" at every reader that came through here: the matte's frame
    # reader, the agent views, the export.
    mezz = mezzaninepath(source)
    path = isfile(mezz) ? mezz : source.path
    try
        return openstream(backend, path, source.width, source.height)
    catch e
        # a decoder that cannot open is not a reason to fail the render — the CPU
        # reader can read anything ffmpeg can, it is only slower
        @warn "GPU decode unavailable for $(basename(source.path)); reading on the CPU" exception = e
        return SequentialReader(source)
    end
end

"""
    renderframe!(dest, seq, n, readers, engine; scratch, black) -> dest

Timeline frame `n` of `seq`, finished: transition, track composite or single
clip, every layer through the effect graph under the export policy (`exact`).

This is the ONE definition of "what frame `n` looks like" — the encoder writes
it, and the agent views ([`contactsheet`](@ref), [`framegrab`](@ref)) show it,
so what an agent sees is by construction what the export produces. `readers`
caches one decoder per source path; pass `scratch`/`black` canvas-sized buffers
to avoid per-frame allocation in a loop.
"""
function renderframe!(dest::AnyRGBFrame, seq::Sequence, n::Integer,
                      readers::Dict{String, Any}, engine::FxEngine;
                      scratch::Union{Nothing, AnyRGBFrame} = nothing,
                      black::Union{Nothing, AbstractMatrix} = nothing)
    tr = transitionat(seq, n)
    sample = tr === nothing ? nothing : transitionsample(seq, tr, n)
    if sample !== nothing
        left, srcA, right, srcB, p = sample
        incoming = scratch === nothing ? similar(dest) : scratch
        rendercanvas!(dest, left, srcA, readers, engine)
        rendercanvas!(incoming, right, srcB, readers, engine)
        blend!(dest, dest, incoming, p)
    elseif ntracks(seq) > 1 && length(clipsat(seq, n)) > 1
        # the export tier of ONE composite (see `composite`): same layer loop the
        # preview runs, only under the exact-decode policy
        composite(engine, clipsat(seq, n), n,
                  (clip, _) -> get!(() -> opendecoder(clip.source, engine.backend),
                                    readers, clip.source.path);
                  canvas = Base.size(dest), exact = true) do canvas
            copyto!(dest, canvas)
        end || error("composite at frame $n could not be rendered")
    else
        loc = locate(seq, n)
        if loc === nothing
            black === nothing ? fill!(dest, RGB{N0f8}(0, 0, 0)) : fillblack!(dest, black)
        else
            rendercanvas!(dest, loc[1], loc[2], readers, engine)
        end
    end
    # Plots go on LAST, over the finished canvas — including over a gap, so a
    # title can carry a black hold. Unconditional: on a sequence without
    # overlays it returns without touching a pixel.
    drawoverlays!(dest, seq.overlays, n; framerate = seq.framerate)
    return dest
end

"""
Render `clip` at source frame `srcframe` through the SAME effect graph the
preview uses — only under the export policy (`exact = true`: no nearest-frame
stand-ins) — then warp its crop into `dest` (a canvas-sized buffer). `readers`
caches one decoder per source path; the engine pools every working buffer.
"""
function rendercanvas!(dest::AnyRGBFrame, clip::Clip, srcframe::Integer,
                       readers::Dict{String, Any}, engine::FxEngine;
                       skipopacity::Bool = false)
    dec = get!(() -> opendecoder(clip.source, engine.backend), readers, clip.source.path)
    ec = effectiveclip(clip, srcframe)    # keyframed params baked at this frame
    skipopacity && (ec = withoutopacity(ec))   # compositing: opacity = layer alpha
    render(engine, dec, ec, Int(srcframe); exact = true) do layer
        if ec.crop == (0.0, 0.0, 1.0, 1.0) && neutralframe(ec) &&
           Base.size(layer) == Base.size(dest)
            copyto!(dest, layer)          # already exactly the canvas — no resample
        else
            fill!(dest, RGB{N0f8}(0, 0, 0))   # whatever the fit doesn't cover is a bar
            placelayer!(dest, layer, ec)
            KA.synchronize(KA.get_backend(dest))
        end
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
