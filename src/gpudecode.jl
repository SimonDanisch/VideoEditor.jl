# GPU video decode — hardware H.264 decode via Lava's Vulkan Video engine.
#
# ffmpeg (already a dep) DEMUXes the container to an H.264 Annex-B elementary
# stream (no decode); Lava decodes it on the GPU's dedicated video-decode queue
# (VK_KHR_video_decode) and returns luma (Y) planes — which are exactly the
# grayscale the motion tracker consumes. VideoEditor doesn't hard-depend on Lava,
# so we reach the decoder through the backend's own module (the same trick the
# GPU-preview bridge uses).

"""
    gpu_decode_luma(backend, path; maxframes = typemax(Int)) -> (width, height, frames)

Hardware-decode the H.264 video at `path` to luma (Y) `Matrix{UInt8}` frames in
display order on the GPU, via Lava's Vulkan Video decoder. `backend` must be a
`LavaBackend` from a device built with video-decode support. Returns the display
`width`, `height`, and the vector of frames.
"""
function gpu_decode_luma(backend, path::AbstractString; maxframes::Integer = typemax(Int))
    lava = parentmodule(typeof(backend))
    isdefined(lava, :decode_h264_luma) ||
        error("GPU video decode requires a LavaBackend whose Lava exposes decode_h264_luma")
    annexb = mktemp() do f, io
        close(io)
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -v error -i $path -map 0:v:0 -c:v copy -bsf:v h264_mp4toannexb -f h264 $f`))
        read(f)
    end
    return lava.decode_h264_luma(annexb; maxframes = Int(maxframes))
end

"Whether `backend` can hardware-decode H.264 (a LavaBackend with Vulkan Video)."
gpu_decode_available(backend) = isdefined(parentmodule(typeof(backend)), :decode_h264_luma)
