# disk → VRAM: a streaming, seekable GPU H.264 decoder with a bounded device ring.
#
# The compressed Annex-B elementary stream is `mmap`'d from a demux temp file, so it
# is disk-backed and paged in by the OS — a GOP is a byte-slice and only the bytes
# actually decoded are ever resident, however long the clip. The stream is split into
# GOPs at IDR boundaries; a GOP's bytes are handed to Lava's Vulkan-Video decoder,
# which returns NV12 planes kept GPU-resident. `frameat!` decodes the owning GOP on a
# miss and evicts the least-recently-used GOP, so at most `capacity` frames live in
# VRAM at once. NV12 → RGB conversion happens once per shown frame, downstream (the
# ring stays in the compact native decode format). Every method takes the stream
# explicitly and runs on whichever thread owns the Lava context (the player's GPU
# worker) — the type is pure device logic, threading is the caller's concern.

"One GOP: its byte range in the elementary stream, the display index of its first
frame, and how many frames it holds."
struct Gop
    bytes::UnitRange{Int}
    firstframe::Int
    nframes::Int
end

"An NV12 frame resident in VRAM — luma and interleaved-chroma planes."
struct Nv12Frame
    y::LavaArray{UInt8, 2}
    uv::LavaArray{UInt8, 2}
end

"""
    GpuVideoStream

A seekable streaming GPU decoder over one video file (see the file header). Address
frames by display index via [`frameat!`](@ref); the ring keeps recently-used GOPs'
frames VRAM-resident up to `capacity` frames. `close` frees the ring and unmaps the
bitstream.
"""
mutable struct GpuVideoStream
    backend   ::LavaBackend
    tmpfile   ::String                    # demux temp file (mmap-backed); removed on close
    bitstream ::Vector{UInt8}             # Mmap view of `tmpfile` — disk-paged
    leadparams::Vector{UInt8}             # first SPS+PPS, prepended to GOPs that lack them
    gops      ::Vector{Gop}
    width     ::Int
    height    ::Int
    bt601     ::Bool
    ring      ::Dict{Int, Nv12Frame}      # display-frame index → VRAM NV12 frame
    resident  ::Vector{Int}               # resident GOP indices, LRU order (front = oldest)
    capacity  ::Int
end

"Total display frames in the stream."
nframes(s::GpuVideoStream) = isempty(s.gops) ? 0 : s.gops[end].firstframe + s.gops[end].nframes

"""
    openstream(backend, path, width, height; capacity=120) -> GpuVideoStream

Demux `path` to an Annex-B temp file, `mmap` it, and index its GOPs. `width`×`height`
is the display size (from the source). `capacity` bounds VRAM-resident frames. Pure
CPU — no decode happens until [`frameat!`](@ref).
"""
function openstream(backend::LavaBackend, path::AbstractString, width::Integer, height::Integer;
                    capacity::Integer = 120)
    tmp = tempname() * ".h264"
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -v error -i $path -map 0:v:0 -c:v copy -bsf:v h264_mp4toannexb -f h264 $tmp`))
    io = open(tmp, "r")
    bits = Mmap.mmap(io, Vector{UInt8})
    close(io)
    gops, lead = index_gops(bits)
    return GpuVideoStream(backend, tmp, bits, lead, gops, Int(width), Int(height),
                          video_is_bt601(path), Dict{Int, Nv12Frame}(), Int[], Int(capacity))
end

"NAL type of the NAL whose 3-byte start code begins at `i` (0 if out of range)."
naltype(bits, i) = i + 3 <= length(bits) ? (bits[i + 3] & 0x1f) : 0x00

"""
Index the Annex-B `bits` into closed GOPs at IDR (type 5) boundaries — each GOP backed
up over any immediately-preceding SPS/PPS so its byte-slice is self-decodable — plus
the leading SPS+PPS to prepend for streams that carry parameter sets only once.
Frame count per GOP = coded pictures (NAL types 1 and 5).
"""
function index_gops(bits::AbstractVector{UInt8})
    starts = Int[]; types = UInt8[]
    i = 1; n = length(bits)
    @inbounds while i <= n - 3
        if bits[i] == 0x00 && bits[i + 1] == 0x00 && bits[i + 2] == 0x01
            push!(starts, i); push!(types, bits[i + 3] & 0x1f); i += 3
        else
            i += 1
        end
    end
    isempty(starts) && error("no NAL units found in elementary stream")
    # leading SPS(7)+PPS(8): copy their bytes to prepend to GOPs that lack params
    sps = findfirst(==(0x07), types); pps = findfirst(==(0x08), types)
    lead = UInt8[]
    if sps !== nothing && pps !== nothing
        nalbytes(k) = bits[starts[k]:(k < length(starts) ? starts[k + 1] - 1 : n)]
        lead = vcat(nalbytes(sps), nalbytes(pps))
    end
    # GOP starts = each IDR, backed up over contiguous leading SPS/PPS NALs
    gopstart = Int[]
    for j in eachindex(types)
        types[j] == 0x05 || continue
        k = j
        while k > 1 && (types[k - 1] == 0x07 || types[k - 1] == 0x08); k -= 1; end
        push!(gopstart, k)
    end
    isempty(gopstart) && error("no IDR frames found (stream not seekable)")
    gops = Gop[]; frame0 = 0
    for (gi, k) in enumerate(gopstart)
        khi = gi < length(gopstart) ? gopstart[gi + 1] - 1 : length(types)
        b0 = starts[k]
        b1 = gi < length(gopstart) ? starts[gopstart[gi + 1]] - 1 : n
        nf = count(t -> t == 0x01 || t == 0x05, @view types[k:khi])
        push!(gops, Gop(b0:b1, frame0, nf))
        frame0 += nf
    end
    return gops, lead
end

"Index of the GOP owning display frame `n` (clamped to the stream)."
function gopof(s::GpuVideoStream, n::Integer)
    for (g, gop) in enumerate(s.gops)
        gop.firstframe <= n < gop.firstframe + gop.nframes && return g
    end
    return n < 0 ? 1 : length(s.gops)
end

"""
    frameat!(s, n) -> Nv12Frame

The device-resident NV12 frame at display index `n`. Decodes its owning GOP into the
ring on a miss (evicting the least-recently-used GOP to stay within `capacity`), and
prefetches the next GOP for smooth forward playback. Runs on the caller's GPU thread.
"""
function frameat!(s::GpuVideoStream, n::Integer)
    g = gopof(s, n)
    haskey(s.ring, n) ? touchgop!(s, g) : decodegop!(s, g)
    g < length(s.gops) && prefetchgop!(s, g + 1)   # keep playback ahead by one GOP
    return get(s.ring, n, s.ring[clamp(n, s.gops[g].firstframe, s.gops[g].firstframe + s.gops[g].nframes - 1)])
end

"Decode GOP `g` fully into the ring (no-op if already resident), then evict LRU GOPs."
function decodegop!(s::GpuVideoStream, g::Integer)
    if g in s.resident
        touchgop!(s, g); return nothing
    end
    gop = s.gops[g]
    bytes = s.bitstream[gop.bytes]
    naltype(bytes, 1) == 0x07 || (bytes = vcat(s.leadparams, bytes))   # ensure SPS/PPS present
    _, _, ys, uvs = Lava.decode_h264_nv12(bytes)
    for k in eachindex(ys)
        idx = gop.firstframe + k - 1
        s.ring[idx] = Nv12Frame(ys[k], uvs[k])
    end
    push!(s.resident, g)
    evict!(s)
    return nothing
end

"Decode GOP `g` ahead of time if not resident and there's room (best-effort prefetch)."
function prefetchgop!(s::GpuVideoStream, g::Integer)
    (g in s.resident || length(s.ring) + s.gops[g].nframes > s.capacity) && return nothing
    decodegop!(s, g)
    return nothing
end

"Move GOP `g` to the most-recently-used end of the LRU order."
function touchgop!(s::GpuVideoStream, g::Integer)
    i = findfirst(==(g), s.resident)
    i === nothing && return nothing
    deleteat!(s.resident, i); push!(s.resident, g)
    return nothing
end

"Free the oldest resident GOPs until the ring fits `capacity` (keeps at least one)."
function evict!(s::GpuVideoStream)
    while length(s.ring) > s.capacity && length(s.resident) > 1
        g = popfirst!(s.resident)
        gop = s.gops[g]
        for k in 0:(gop.nframes - 1)
            f = pop!(s.ring, gop.firstframe + k, nothing)
            f === nothing && continue
            Lava.unsafe_free!(f.y); Lava.unsafe_free!(f.uv)
        end
    end
    return nothing
end

function Base.close(s::GpuVideoStream)
    for f in values(s.ring); Lava.unsafe_free!(f.y); Lava.unsafe_free!(f.uv); end
    empty!(s.ring); empty!(s.resident)
    finalize(s.bitstream)                       # unmap
    isfile(s.tmpfile) && rm(s.tmpfile; force = true)
    return nothing
end
