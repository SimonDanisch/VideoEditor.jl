# disk → VRAM: a streaming, seekable GPU H.264 decoder with a bounded device ring.
#
# The compressed Annex-B elementary stream is `mmap`'d from a demux temp file, so it
# is disk-backed and paged in by the OS — a GOP is a byte-slice and only the bytes
# actually decoded are ever resident, however long the clip. The stream is split into
# GOPs at IDR boundaries; a GOP's bytes are handed to Mantle's video decoder,
# which returns NV12 planes kept GPU-resident. `frameat!` decodes the owning GOP on a
# miss and evicts the least-recently-used GOP, so at most `capacity` frames live in
# VRAM at once. NV12 → RGB conversion happens once per shown frame, downstream (the
# ring stays in the compact native decode format). Every method takes the stream
# explicitly and runs on whichever thread owns the GPU context (the player's GPU
# worker) — the type is pure device logic, threading is the caller's concern.

"One GOP: its byte range in the elementary stream, the display index of its first
frame, and how many frames it holds."
struct Gop
    bytes::UnitRange{Int}
    firstframe::Int
    nframes::Int
end

"""
An NV12 frame resident in VRAM — luma and interleaved-chroma planes.

PARAMETERISED on the array the decoder handed back rather than naming one
backend's: the editor talks to the runtime, and which device array that is
belongs to the platform. A field typed abstractly would read the same and cost a
dynamic lookup per plane per frame, which is why this is a parameter and not
`Any`.
"""
struct Nv12Frame{A<:AbstractMatrix{UInt8}}
    y::A
    uv::A
end

"""
    GpuVideoStream

A seekable streaming GPU decoder over one video file (see the file header). Address
frames by display index via [`frameat!`](@ref); the ring keeps recently-used GOPs'
frames VRAM-resident up to `capacity` frames. `close` frees the ring and unmaps the
bitstream.
"""
mutable struct GpuVideoStream{B<:KA.GPU}
    backend   ::B
    codec     ::Symbol                    # :h264 | :hevc — selects parser + decode session
    tmpfile   ::String                    # demux temp file (mmap-backed); removed on close
    bitstream ::Vector{UInt8}             # Mmap view of `tmpfile` — disk-paged
    leadparams::Vector{UInt8}             # first parameter sets, prepended to GOPs that lack them
    gops      ::Vector{Gop}
    width     ::Int
    height    ::Int
    bt601     ::Bool
    ring      ::Dict{Int, Nv12Frame}      # display-frame index → VRAM NV12 frame
    resident  ::Vector{Int}               # resident GOP indices, LRU order (front = oldest)
    capacity  ::Int
    dec       ::Any                       # lazy persistent Mantle H264Decoder (chroma)
    feedgop   ::Int                       # GOP being incrementally decoded (0 = none)
    feednext  ::Int                       # display index the active feed emits next
end

"Total display frames in the stream."
nframes(s::GpuVideoStream) = isempty(s.gops) ? 0 : s.gops[end].firstframe + s.gops[end].nframes

"Whether display frame `n` is decoded and VRAM-resident right now."
hasframe(s::GpuVideoStream, n::Integer) = haskey(s.ring, n)

"""
    openstream(backend, path, width, height; capacity=120, vrambudget=3 * 2^30) -> GpuVideoStream

Demux `path` to an Annex-B temp file, `mmap` it, and index its GOPs. `width`×`height`
is the display size (from the source). Pure CPU — no decode happens until
[`frameat!`](@ref).

`capacity` is a FLOOR for the VRAM-resident frame count: a GOP decodes as a unit, so
the ring is sized to hold TWO of the stream's largest GOPs (current + prefetched next)
— a smaller ring would evict the very GOP being played and re-decode it every present.
`vrambudget` caps that in bytes — but grows itself to the decode-ahead need when the
driver reports enough free device memory (a stingy default must not hitch a 20 GB
card). If even one GOP exceeds the final budget the stream refuses to open (the
caller falls back to CPU decode) rather than thrash.
"""
function openstream(backend::KA.GPU, path::AbstractString, width::Integer, height::Integer;
                    capacity::Integer = 120, vrambudget::Integer = 3 * 2^30)
    codec = videocodec(path)
    codec in (:h264, :hevc) ||
        error("GPU stream: codec $codec has no hardware decode session yet — transcode first")
    bsf = codec === :hevc ? "hevc_mp4toannexb" : "h264_mp4toannexb"
    tmp = tempname() * "." * String(codec)
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -v error -i $path -map 0:v:0 -c:v copy -bsf:v $bsf -f $(codec === :hevc ? "hevc" : "h264") $tmp`))
    io = open(tmp, "r")
    bits = Mmap.mmap(io, Vector{UInt8})
    close(io)
    gops, lead = codec === :hevc ? index_gops_h265(bits) : index_gops(bits)
    # honor the container's EDIT LIST: the mp4 may cut leading coded frames (phone
    # footage typically trims a couple) — every other decoder (ffmpeg, VideoIO,
    # players) shows the video WITHOUT them, so the raw Annex-B stream must shift
    # its display indexing accordingly or the app's two decode paths disagree
    skip = editlistlead(path)
    skip > 0 && (gops = [Gop(g.bytes, g.firstframe - skip, g.nframes) for g in gops])
    maxgop = maximum(g -> g.nframes, gops)
    framebytes = Int(width) * Int(height) * 3 ÷ 2
    # decode-ahead needs 2 GOPs (+ prefetch margin) resident: a conservative
    # `vrambudget` must not cause GOP-boundary hitches when the card has the
    # headroom — grow it whenever the driver reports twice the need free
    budget = Int(vrambudget)
    need = (2 * maxgop + 8) * framebytes
    if need > budget
        free = freedevicevram(backend)
        free > 2 * need && (budget = need)
    end
    fit = max(budget ÷ framebytes, 1)   # NV12 frames in budget
    maxgop <= fit || error("largest GOP ($maxgop frames) exceeds the VRAM budget " *
                           "($fit frames) — staying on CPU decode")
    cap = min(max(Int(capacity), 2 * maxgop), fit)
    cap < 2 * maxgop &&
        @warn "VRAM too tight for decode-ahead — expect brief GOP-boundary hitches" maxgop fit
    return GpuVideoStream(backend, codec, tmp, bits, lead, gops, Int(width), Int(height),
                          video_is_bt601(path), Dict{Int, Nv12Frame}(), Int[], cap,
                          nothing, 0, 0)
end

"Video codec of `path` (`:h264`, `:hevc`, …) via ffprobe."
function videocodec(path::AbstractString)
    out = read(`$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0
                -show_entries stream=codec_name -of csv=p=0 $path`, String)
    # FIRST FIELD, not the whole line. ffprobe's csv writer pads the row with an
    # empty trailing field when the file carries extra streams — an iPhone MOV has
    # seven — so this comes back as "hevc," and `Symbol("hevc,")` matches no
    # supported codec. A perfectly streamable file was then declared
    # un-streamable, sent to the mezzanine transcoder it did not need, and the
    # retry parsed the result the same way and failed again.
    return Symbol(first(split(strip(out), ',')))
end

"""
Free device-local VRAM in bytes — the driver's budget minus its usage
(VK_EXT_memory_budget); half the heap size when the extension is missing,
0 when no context is up (callers treat that as "don't grow").
"""
function freedevicevram(backend::KA.GPU)
    try
        heaps = Mantle.probe_device_memory_budget(Mantle.vk_context())
        free = 0
        for h in heaps
            h.device_local || continue
            avail = h.budget > 0 ? h.budget - h.usage : h.size ÷ 2
            free = max(free, avail)
        end
        return free
    catch e
        # 0 reads as "no VRAM" everywhere upstream, so say why it could not be read
        @warn "could not read the device memory budget" exception = (e, catch_backtrace())
        return 0
    end
end

"NAL type of the NAL whose 3-byte start code begins at `i` (0 if out of range)."
naltype(bits, i) = i + 3 <= length(bits) ? (bits[i + 3] & 0x1f) : 0x00

"""
    editlistlead(path) -> Int

Leading video frames the mp4's EDIT LIST actually cuts off. An `elst` whose
`media_time` merely equals the first sample's composition offset (`ctts`) is the
standard B-frame pts alignment and trims NOTHING — only the excess beyond that
offset is a real trim: `(media_time - first_ctts) ÷ sample_delta`. Read straight
from the container boxes; 0 without an edit list (or a non-mp4 container).
"""
function editlistlead(path::AbstractString)
    bytes = try
        open(io -> Mmap.mmap(io, Vector{UInt8}), path)   # moov sits at either end — walk it all
    catch e
        e isa SystemError || rethrow()   # unreadable file: no edit list to read
        return 0
    end
    # only ISO-BMFF containers carry edit lists; walking an mkv's EBML as boxes
    # would read garbage sizes
    (length(bytes) >= 12 && String(bytes[5:8]) == "ftyp") || return 0
    be32(o) = (Int(bytes[o]) << 24) | (Int(bytes[o+1]) << 16) | (Int(bytes[o+2]) << 8) | Int(bytes[o+3])
    boxtype(o) = String(bytes[o+4:o+7])
    # walk boxes for the VIDEO track's edts/elst media_time and stts sample delta
    function walk(f, lo, hi, want)
        o = lo
        while o + 8 <= hi
            sz = be32(o)
            body = o + 8                                    # after size + type
            if sz == 1                                      # 64-bit largesize (mdat)
                o + 16 <= hi || return
                sz = (Int(be32(o + 8)) << 32) | be32(o + 12)
                body = o + 16
            elseif sz == 0                                  # box extends to EOF
                sz = hi - o + 1
            end
            sz < 8 && return
            boxtype(o) == want && f(body, min(o + sz - 1, hi))
            o += sz
        end
    end
    lead = 0
    walk(1, length(bytes) - 7, "moov") do mlo, mhi
        walk(mlo, mhi, "trak") do tlo, thi
            isvideo = Ref(false); mtime = Ref(0); delta = Ref(0); ctts0 = Ref(0)
            walk(tlo, thi, "mdia") do dlo, dhi
                walk(dlo, dhi, "hdlr") do hlo, hhi
                    hlo + 11 <= hhi && String(bytes[hlo+8:hlo+11]) == "vide" && (isvideo[] = true)
                end
                walk(dlo, dhi, "minf") do ilo, ihi
                    walk(ilo, ihi, "stbl") do slo, shi
                        walk(slo, shi, "stts") do xlo, xhi
                            xlo + 15 <= xhi && (delta[] = be32(xlo + 12))   # first entry's sample delta
                        end
                        walk(slo, shi, "ctts") do xlo, xhi
                            xlo + 15 <= xhi && (ctts0[] = be32(xlo + 12))  # first composition offset
                        end
                    end
                end
            end
            walk(tlo, thi, "edts") do elo, ehi
                walk(elo, ehi, "elst") do llo, lhi
                    llo + 15 <= lhi || return
                    version = bytes[llo]
                    mtime[] = version == 1 ? Int(be32(llo + 16)) << 32 | be32(llo + 20) :
                                             be32(llo + 12)    # first entry's media_time
                end
            end
            isvideo[] && delta[] > 0 && (lead = max(0, mtime[] - ctts0[]) ÷ delta[])
        end
    end
    return lead
end

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

"""
The HEVC sibling of [`index_gops`](@ref): 2-byte NAL headers, GOPs at IDR
boundaries (types 19/20) backed up over leading VPS/SPS/PPS/prefix-SEI; frame
count counts FIRST slice segments so multi-slice pictures stay one frame.
CRA recovery points (type 21, open GOPs) are refused — their RASL pictures
reference across the GOP cut, so per-GOP seeking would mis-decode; the caller
falls back (and the ingest transcode produces closed GOPs anyway).
"""
function index_gops_h265(bits::AbstractVector{UInt8})
    starts = Int[]; types = UInt8[]; firsts = Bool[]
    i = 1; n = length(bits)
    @inbounds while i <= n - 3
        if bits[i] == 0x00 && bits[i + 1] == 0x00 && bits[i + 2] == 0x01
            t = (bits[i + 3] >> 1) & 0x3f
            push!(starts, i); push!(types, t)
            push!(firsts, t <= 0x15 && i + 5 <= n && (bits[i + 5] & 0x80) != 0)
            i += 3
        else
            i += 1
        end
    end
    isempty(starts) && error("no NAL units found in elementary stream")
    any(==(0x15), types) &&
        error("open-GOP HEVC (CRA) is not GOP-seekable — transcode to closed GOPs first")
    nalbytes(k) = bits[starts[k]:(k < length(starts) ? starts[k + 1] - 1 : n)]
    vps = findfirst(==(0x20), types); sps = findfirst(==(0x21), types); pps = findfirst(==(0x22), types)
    lead = (vps === nothing || sps === nothing || pps === nothing) ? UInt8[] :
           vcat(nalbytes(vps), nalbytes(sps), nalbytes(pps))
    gopstart = Int[]
    for j in eachindex(types)
        types[j] in (0x13, 0x14) || continue
        k = j
        while k > 1 && types[k - 1] in (0x20, 0x21, 0x22, 0x27); k -= 1; end
        push!(gopstart, k)
    end
    isempty(gopstart) && error("no IDR frames found (stream not seekable)")
    gops = Gop[]; frame0 = 0
    for (gi, k) in enumerate(gopstart)
        khi = gi < length(gopstart) ? gopstart[gi + 1] - 1 : length(types)
        b0 = starts[k]
        b1 = gi < length(gopstart) ? starts[gopstart[gi + 1]] - 1 : n
        nf = count(@view firsts[k:khi])
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
    frameat!(s, n; prefetch=false, served=nothing) -> Nv12Frame

The device-resident NV12 frame at display index `n`, decoded INCREMENTALLY: a miss
starts (or continues) a chunked feed of `n`'s GOP through the stream's persistent
[`Mantle.VideoDecode.H264Decoder`] — at most ~5 chunks (≈90 ms) per call, then the
nearest already-decoded frame is served. Callers that need exactness poll
[`hasframe`](@ref) and re-present (the player's retry loop refines a scrub to the
exact frame across a few calls instead of stalling one call for a whole GOP).

`chunks` is how many ~10 ms decode chunks this call may spend reaching `n` before
it settles for the nearest decoded frame. `chunks = 0` never decodes: it returns
whatever the ring already holds, which is what a SEEK wants for its first draw —
put a picture up now, let the retry loop fetch the real one. The default 5 is the
scrub budget; measured, a miss that spends it costs ~62 ms, and that was the
whole difference between a seek answering in 1 ms and in 62.

With `prefetch` (pass it during sequential playback) the NEXT GOP starts feeding
once `n` is half-way into the current one, and every present advances the feed by a
~10 ms chunk — a 250-frame GOP decodes spread invisibly across ~2 s of playback
instead of as one ~600 ms stall at the boundary. Runs on the caller's GPU thread.
"""
function frameat!(s::GpuVideoStream, n::Integer; prefetch::Bool = false,
                  served::Union{Nothing, Base.RefValue{Int}} = nothing,
                  chunks::Integer = 5)
    g = gopof(s, n)
    gop = s.gops[g]
    if haskey(s.ring, n)
        touchgop!(s, g)
    else
        if s.feedgop != g
            abandonfeed!(s)
            startfeed!(s, g)
        end
        spent = 0
        while !haskey(s.ring, n) && s.feedgop == g && spent < chunks
            decodechunk!(s) == 0 && break
            spent += 1
        end
    end
    if prefetch
        # as soon as nothing is in flight, chain the NEXT GOP's feed — maximal runway
        # (startfeed!'s LRU eviction makes the room; the played GOP stays touched-recent)
        if s.feedgop == 0 && g < length(s.gops) && !(g + 1 in s.resident) &&
           s.gops[g + 1].nframes <= s.capacity
            startfeed!(s, g + 1)
        end
        s.feedgop == 0 || decodechunk!(s; frames = 4)   # ~10 ms of ahead-work per present
    end
    if haskey(s.ring, n)
        served === nothing || (served[] = n)
        return s.ring[n]
    end
    # nearest decoded frame of this GOP (feeds fill front-to-back, so search outward).
    # `served` tells the caller WHICH frame this really is — per-frame consumers
    # (track transforms) must follow it, or they warp the wrong image.
    for k in 1:gop.nframes
        for m in (n - k, n + k)
            if gop.firstframe <= m < gop.firstframe + gop.nframes && haskey(s.ring, m)
                served === nothing || (served[] = m)
                return s.ring[m]
            end
        end
    end
    decodechunk!(s)                       # freshly-started feed: land the first chunk
    served === nothing || (served[] = gop.firstframe)
    return s.ring[gop.firstframe]
end

"""
    exactframeat!(s, n) -> Nv12Frame

The frame at display index `n`, decoding as much as it takes — the sequential-
export counterpart to the latency-bounded [`frameat!`](@ref): no approximate
serves, no decode-ahead margin. Errors if the stream cannot produce the frame
(truncated bitstream) instead of silently substituting a neighbour.
"""
function exactframeat!(s::GpuVideoStream, n::Integer)
    g = gopof(s, n)
    if !haskey(s.ring, n)
        s.feedgop == g || (abandonfeed!(s); startfeed!(s, g))
        while !haskey(s.ring, n) && s.feedgop == g
            # no latency budget here — large chunks amortize the per-submit wait
            decodechunk!(s; frames = 32) == 0 && break
        end
        haskey(s.ring, n) || error("stream could not decode frame $n (GOP $g)")
    end
    touchgop!(s, g)
    return s.ring[n]
end

"First NAL of a GOP slice is a parameter set (so the GOP is self-decodable)."
startswithparams(s::GpuVideoStream, bytes) =
    s.codec === :hevc ? ((bytes[4] >> 1) & 0x3f) == 0x20 : naltype(bytes, 1) == 0x07

"Begin the incremental feed of GOP `g` on the stream's persistent decode session."
function startfeed!(s::GpuVideoStream, g::Integer)
    if s.dec === nothing
        params = isempty(s.leadparams) ? s.bitstream[s.gops[1].bytes] : s.leadparams
        ctx = Mantle.vk_context()
        s.dec = s.codec === :hevc ? Mantle.VideoDecode.H265Decoder(ctx, params; chroma = true) :
                                    Mantle.VideoDecode.H264Decoder(ctx, params; chroma = true)
    end
    gop = s.gops[g]
    bytes = s.bitstream[gop.bytes]
    startswithparams(s, bytes) || (bytes = vcat(s.leadparams, bytes))
    Mantle.VideoDecode.feed!(s.dec, bytes)
    s.feedgop = g
    s.feednext = gop.firstframe
    g in s.resident || push!(s.resident, g)   # partially resident from the first chunk on
    evict!(s)   # abandoned partial GOPs must not accumulate (scrubs retarget feeds often)
    return nothing
end

"""
Advance the active feed by up to `frames` access units, moving display-ready frames
into the ring. Emitted frames are display-contiguous (verified bit-exact against the
batch decoder), so ring indices just count up from the GOP's first frame. Returns
the number of frames landed; completes the feed (and applies eviction) at GOP end.
"""
function decodechunk!(s::GpuVideoStream; frames::Integer = 7)
    s.feedgop == 0 && return 0
    out = Mantle.VideoDecode.decodemore!(s.dec, frames)
    for (y, uv) in out
        idx = s.feednext
        s.feednext += 1
        idx < 0 && continue   # edit-list lead-in: decoded (refs need it), never shown
        s.ring[idx] = Nv12Frame(y, uv)
    end
    if Mantle.VideoDecode.remaining(s.dec) == 0
        s.feedgop = 0
        evict!(s)
    end
    return length(out)
end

"Drop an unfinished feed (scrub retargeted): decoded-but-unemitted frames can't be
placed without their display predecessors, so they are discarded with the tail."
function abandonfeed!(s::GpuVideoStream)
    s.feedgop == 0 && return nothing
    empty!(s.dec.pending)
    empty!(s.dec.aus)
    s.dec.nextau = 1
    s.feedgop = 0
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
            Mantle.unsafe_free!(f.y); Mantle.unsafe_free!(f.uv)
        end
    end
    return nothing
end

function Base.close(s::GpuVideoStream)
    # WAIT FIRST. `unsafe_free!` returns a buffer to the pool immediately, and the
    # GPU may still be READING these planes — a colour convert or a thumbnail
    # downscale recorded moments ago. Freeing under an in-flight read faults the
    # driver, and the crash lands here, in `close`, with nothing naming the
    # dispatch that was still using it: seen twice as
    # `signal (11): Segmentation fault … close at gpustream.jl … gputhumbloop`.
    # Same shape as the buffer-lifetime bug in the allocator's own pool.
    #
    # BEFORE the decoder too, not only before the planes: the observed crash is in
    # `close(s.dec)` itself, and a decode session owns device memory the same
    # dispatches are reading. Draining once, first, covers both.
    KA.synchronize(s.backend)
    s.dec === nothing || (close(s.dec); s.dec = nothing)
    s.feedgop = 0
    for f in values(s.ring); Mantle.unsafe_free!(f.y); Mantle.unsafe_free!(f.uv); end
    empty!(s.ring); empty!(s.resident)
    finalize(s.bitstream)                       # unmap
    isfile(s.tmpfile) && rm(s.tmpfile; force = true)
    return nothing
end
