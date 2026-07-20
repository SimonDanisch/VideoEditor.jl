# GPU video decode — hardware H.264 decode via Lava's Vulkan Video engine.
#
# ffmpeg (already a dep) DEMUXes the container to an H.264 Annex-B elementary
# stream (no decode); Lava decodes it on the GPU's dedicated video-decode queue
# (VK_KHR_video_decode) and returns luma (Y) planes — exactly the grayscale the
# motion tracker consumes — kept device-resident as `LavaArray`s. VideoEditor
# doesn't hard-depend on Lava, so we reach the decoder through the backend's own
# module (the same trick the GPU-preview bridge uses).

"Demux `path` to an in-memory H.264 Annex-B elementary stream (no re-encode)."
function demux_annexb(path::AbstractString)
    mktemp() do f, io
        close(io)
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -v error -i $path -map 0:v:0 -c:v copy -bsf:v h264_mp4toannexb -f h264 $f`))
        read(f)
    end
end

"""
    gpu_decode_luma(backend, path; maxframes = typemax(Int)) -> (width, height, frames)

Hardware-decode the H.264 video at `path` on the GPU and return its luma (Y)
planes in display order, kept device-resident as `LavaArray{UInt8,2}` (grayscale =
the NV12 Y plane). `backend` must be a `LavaBackend` on a video-capable device.
Throws for unsupported streams (non-4:2:0 chroma, or `max_num_ref_frames < 2`);
use [`gpu_decodable`](@ref) to probe first.
"""
function gpu_decode_luma(backend, path::AbstractString; maxframes::Integer = typemax(Int))
    lava = parentmodule(typeof(backend))
    isdefined(lava, :decode_h264_gpu) ||
        error("GPU video decode requires a LavaBackend whose Lava exposes decode_h264_gpu")
    return lava.decode_h264_gpu(demux_annexb(path); maxframes = Int(maxframes))
end

"Whether `backend`'s device exposes hardware video decode at all."
gpu_decode_available(backend) = isdefined(parentmodule(typeof(backend)), :decode_h264_gpu)

"Release the device memory of GPU-resident decode frames promptly (don't wait for GC)."
gpu_free_frames!(frames) = (for f in frames; parentmodule(typeof(f)).unsafe_free!(f); end; nothing)
gpu_free_frames!(::Nothing) = nothing

"BT.601 (`true`) vs BT.709 (`false`) YUV→RGB, matching ffmpeg/VideoIO's choice
from the stream's color tag — defaulting to BT.601 for untagged content, as
swscale does."
function video_is_bt601(path::AbstractString)
    cs = try
        strip(read(pipeline(`$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0 -show_entries stream=color_space -of default=noprint_wrappers=1:nokey=1 $path`), String))
    catch
        "unknown"
    end
    return !occursin("bt709", cs)
end

"""
    gpu_decode_rgb(backend, path; maxframes = typemax(Int)) -> (width, height, frames)

Hardware-decode `path` and return RGB frames kept device-resident as
`LavaArray{RGB{N0f8},2}` — the NV12 planes are decoded and converted to RGB
entirely on the GPU (see [`nv12torgb!`]), matching VideoIO's RGB to within a couple
of levels. Feeding these to `grayscale!` gives the Rec.709 grayscale the tracker
expects, identical to the CPU decode path. Throws for unsupported streams.
"""
function gpu_decode_rgb(backend, path::AbstractString; maxframes::Integer = typemax(Int))
    lava = parentmodule(typeof(backend))
    isdefined(lava, :decode_h264_nv12) ||
        error("GPU RGB decode requires a LavaBackend whose Lava exposes decode_h264_nv12")
    w, h, ys, uvs = lava.decode_h264_nv12(demux_annexb(path); maxframes = Int(maxframes))
    bt601 = video_is_bt601(path)
    rgbs = [KA.allocate(backend, RGB{N0f8}, (w, h)) for _ in eachindex(ys)]
    for i in eachindex(ys)
        nv12torgb!(rgbs[i], ys[i], uvs[i]; bt601 = bt601)
    end
    KA.synchronize(backend)
    foreach(a -> lava.unsafe_free!(a), ys)
    foreach(a -> lava.unsafe_free!(a), uvs)
    return (w, h, rgbs)
end

"""
    gpu_decodable(backend, path) -> Bool

Whether `path` can actually be GPU-decoded here — the device has a video-decode
queue AND the stream is a supported form (4:2:0, multi-reference H.264). Probes by
decoding a single frame, so it catches the `decode_h264_gpu` guards (4:4:4 chroma,
single-reference/all-intra) rather than only checking device capability.
"""
function gpu_decodable(backend, path::AbstractString)
    gpu_decode_available(backend) || return false
    try
        gpu_decode_luma(backend, path; maxframes = 1)
        return true
    catch
        return false
    end
end
