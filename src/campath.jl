# Production camera lock — the DaVinci-parity virtual tripod. The established offline
# stabilizer architecture, so nothing can death-spiral (no velocity / no coasting) and
# nothing drifts (anchored to frame 1):
#
#   Pass 1  — estimate each frame's motion by tracking Shi-Tomasi corners with sparse
#             GPU Lucas-Kanade (gradient-based → no NCC peak-locking on the repetitive
#             texture), forward-backward validated, RANSAC-fit to a **similarity** and
#             accumulated FRAME-TO-FRAME. Per-step scale/rotation are ≈1/0 → stable, so a
#             real zoom or roll accumulates smoothly instead of the spurious spikes a
#             large-baseline fit produces.
#   Pass 2  — refine translation drift-free: match the FIRST frame's NCC templates at each
#             frame's pass-1-predicted position (tiny search → no aliasing) and RANSAC the
#             residual translation onto the estimate. Anchored to frame 1 ⇒ zero drift.
#   Smooth  — median-filter single-frame outliers + light Gaussian smooth of the
#             (scale, rotation, translation) trajectory (motion inpainting).
#
# Runs at full resolution (downscaling blurs fine texture and aliases on repeating
# structures). The per-feature LK is a KA kernel, so it runs on the Vulkan/Lava GPU.

"""
    deoutlier(v, w, thresh) -> Vector

Replace ONLY the samples that deviate more than `thresh` from their local median
(window `2w+1`) with that median. Removes single-frame fit glitches (a bird crossing
the patches, a momentary bad RANSAC set) without attenuating real motion — a plain
median or Gaussian would blur out genuine fast camera movement, which for a *lock*
leaves it visibly un-cancelled.
"""
function deoutlier(v::Vector{Float64}, w::Integer, thresh::Real)
    n = length(v)
    out = copy(v)
    for i in 1:n
        m = median(@view v[max(1, i - w):min(n, i + w)])
        abs(v[i] - m) > thresh && (out[i] = m)
    end
    return out
end

# Pass 1: frame-to-frame similarity accumulation via GPU Lucas-Kanade tracking.
"""
CPU grayscale source for the analysis passes: VideoIO decode + host grayscale.
The GPU counterpart is a [`GpuVideoStream`](@ref) — see [`grayinto!`](@ref).
"""
struct GrayReader
    sr::SequentialReader
    frame::RGBFrame
    host::Matrix{Float32}
end
GrayReader(source::VideoSource) =
    GrayReader(SequentialReader(source), RGBFrame(undef, source.width, source.height),
               Matrix{Float32}(undef, source.width, source.height))
Base.close(r::GrayReader) = close(r.sr)

"""
GPU grayscale source for the analysis passes: the chunked [`GpuVideoStream`](@ref)
plus device scratch. Decoded NV12 converts to RGB on-device and THEN to grayscale —
the same gray the CPU path computes (grayscale of chroma-reconstructed RGB), so the
CPU/GPU tracks stay in sub-pixel parity; tracking straight on the luma plane
differs at chroma edges and drifts the comparison.
"""
struct StreamGraySource
    s::GpuVideoStream
    rgb::Any    # device RGB scratch (source-sized)
    gray::Any   # device Float32 scratch for host handoff
end
StreamGraySource(s::GpuVideoStream, backend) =
    StreamGraySource(s, KA.allocate(backend, RGB{N0f8}, (s.width, s.height)),
                     KA.allocate(backend, Float32, (s.width, s.height)))
Base.close(src::StreamGraySource) = close(src.s)

"""
    grayinto!(g, decoder, srcframe) -> g

Source frame `srcframe` as Float32 grayscale in `g` (usually a device array), by
decoder type: a [`StreamGraySource`](@ref) decodes and converts entirely on-device;
a [`GrayReader`](@ref) decodes via VideoIO on the host and uploads; a prefetched
device-frame vector converts its RGB in place. All land in the same 0..1 range.
"""
function grayinto!(g, src::StreamGraySource, sf::Integer)
    f = exactframeat!(src.s, sf)
    nv12torgb!(src.rgb, f.y, f.uv; bt601 = src.s.bt601)
    grayscale!(g, src.rgb)
    return g
end
grayinto!(g, r::GrayReader, sf::Integer) =
    (readframe!(r.frame, r.sr, sf); grayscale!(r.host, r.frame); copyto!(g, r.host); g)
grayinto!(g, frames::AbstractVector, sf::Integer) = (grayscale!(g, frames[sf + 1]); g)

"Like [`grayinto!`](@ref) but into a HOST matrix (NCC template cutting needs it)."
hostgray!(host::Matrix{Float32}, src::StreamGraySource, sf::Integer) =
    (grayinto!(src.gray, src, sf); copyto!(host, src.gray); host)
hostgray!(host::Matrix{Float32}, r::GrayReader, sf::Integer) =
    (readframe!(r.frame, r.sr, sf); grayscale!(host, r.frame); host)
hostgray!(host::Matrix{Float32}, frames::AbstractVector, sf::Integer) =
    (grayscale!(host, Array(frames[sf + 1])); host)

"""
    graysource(backend, source) -> StreamGraySource | GrayReader

Frame source for an analysis pass over `source`: a GPU backend decodes through
its own chunked [`GpuVideoStream`](@ref) (falling back to VideoIO with a warning
when the stream won't open — unsupported codec, VRAM); the CPU backend reads via
VideoIO. Feed frames with [`grayinto!`](@ref)/[`hostgray!`](@ref)/[`rgbinto!`](@ref);
`close` when done.
"""
graysource(backend::KA.CPU, source::VideoSource) = GrayReader(source)
function graysource(backend, source::VideoSource)
    mezz = mezzaninepath(source)
    for path in (source.path, mezz)   # prefer the original; fall back to a mezzanine
        (path == mezz && !isfile(mezz)) && continue
        try
            s = openstream(backend, path, source.width, source.height)
            try
                exactframeat!(s, 0)   # probe: `openstream` only demuxes — unsupported
            catch                     # profiles (e.g. 4:4:4) surface at the first decode
                close(s)
                rethrow()
            end
            return StreamGraySource(s, backend)
        catch e
            path == mezz &&
                @warn "GPU stream unavailable for analysis — CPU decode" source exception = e
        end
    end
    isfile(mezz) || @warn "GPU stream unavailable for analysis — CPU decode" source = source.path
    return GrayReader(source)
end

"Decode source frame `sf` into the HOST RGB buffer `frame` (color/loop analyses)."
rgbinto!(frame::RGBFrame, r::GrayReader, sf::Integer) = (readframe!(frame, r.sr, sf); frame)
function rgbinto!(frame::RGBFrame, src::StreamGraySource, sf::Integer)
    f = exactframeat!(src.s, sf)
    nv12torgb!(src.rgb, f.y, f.uv; bt601 = src.s.bt601)
    copyto!(frame, src.rgb)
    return frame
end

"RGB frame of `sf` ON the decoder's backend for whole-frame reductions: the
stream converts into its device scratch (nothing crosses the bus); the reader
returns its host frame."
function rgbframe!(src::StreamGraySource, sf::Integer)
    f = exactframeat!(src.s, sf)
    nv12torgb!(src.rgb, f.y, f.uv; bt601 = src.s.bt601)
    return src.rgb
end
rgbframe!(r::GrayReader, sf::Integer) = (readframe!(r.frame, r.sr, sf); r.frame)

"""
    smallrgbinto!(small, ws, decoder, sf) -> small

Area-averaged downscale of source frame `sf` into the HOST `small` (loop
matching): the stream converts + downscales on-device and only the small
frame crosses the bus; the reader decodes and downscales on the host.
"""
function smallrgbinto!(small::RGBFrame, ws, src::StreamGraySource, sf::Integer)
    f = exactframeat!(src.s, sf)
    nv12torgb!(src.rgb, f.y, f.uv; bt601 = src.s.bt601)
    areadownscale!(ws.small, src.rgb)
    copyto!(small, ws.small)
    return small
end
smallrgbinto!(small::RGBFrame, ws, r::GrayReader, sf::Integer) =
    (readframe!(r.frame, r.sr, sf); copyto!(small, downscale(r.frame, size(small)...)); small)

"Downscale staging for [`smallrgbinto!`](@ref) — device scratch for a stream."
smallrgbws(src::StreamGraySource, backend, dims) =
    (small = KA.allocate(backend, RGB{N0f8}, dims),)
smallrgbws(r::GrayReader, backend, dims) = NamedTuple()

"""
    smallgrayinto!(g, ws, decoder, sf) -> g

Downscaled Float32 grayscale of source frame `sf` into `g` for the flow
analyses, staged through the workspace from [`smallgrayws`](@ref): a stream
decodes, downscales and grays entirely on-device; a [`GrayReader`](@ref) stays
on the host and uploads only the small gray.
"""
function smallgrayinto!(g, ws, src::StreamGraySource, sf::Integer)
    f = exactframeat!(src.s, sf)
    nv12torgb!(src.rgb, f.y, f.uv; bt601 = src.s.bt601)
    warp!(ws.small, src.rgb, (0.0, 0.0, 1.0, 1.0))
    grayscale!(g, ws.small)
    return g
end
function smallgrayinto!(g, ws, r::GrayReader, sf::Integer)
    readframe!(r.frame, r.sr, sf)
    warp!(ws.small, r.frame, (0.0, 0.0, 1.0, 1.0))
    grayscale!(ws.gray, ws.small)
    copyto!(g, ws.gray)
    return g
end

"Downscale staging buffers for [`smallgrayinto!`](@ref), on the decoder's side of the bus."
smallgrayws(src::StreamGraySource, backend, dims) =
    (small = KA.allocate(backend, RGB{N0f8}, dims),)
smallgrayws(r::GrayReader, backend, dims) =
    (small = RGBFrame(undef, dims...), gray = Matrix{Float32}(undef, dims...))

"Host copy of an analysis frame for host-side work (template cutting) —
already-host arrays pass through untouched."
hostcopy(g::Matrix{Float32}) = g
hostcopy(g) = Array(g)

function trackcampath(clip::Clip, n::Integer; window::Integer, iters::Integer,
                      minfeatures::Integer, maxfeatures::Integer, ransacpx::Real,
                      fbmax::Real, redetectevery::Integer, backend, progress, frames = nothing)
    source = clip.source
    W, H = source.width, source.height
    g0 = KA.allocate(backend, Float32, (W, H)); g1 = similar(g0)
    ix0 = similar(g0); iy0 = similar(g0); ix1 = similar(g0); iy1 = similar(g0)
    dec = frames === nothing ? GrayReader(source) : frames
    loadgray!(g, k) = grayinto!(g, dec, clip.src_in + k - 1)
    transforms = fill(Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), n)
    try
        loadgray!(g0, 1)
        gradients!(ix0, iy0, g0)
        pts = goodfeatures(backend, g0; maxpoints = maxfeatures, border = window + 4)
        px = Float64[p[1] for p in pts]; py = Float64[p[2] for p in pts]
        for k in 2:n
            loadgray!(g1, k)
            gradients!(ix1, iy1, g1)
            nf = length(px)
            dpx = KA.allocate(backend, Float32, nf); copyto!(dpx, Float32.(px))
            dpy = KA.allocate(backend, Float32, nf); copyto!(dpy, Float32.(py))
            dqx = similar(dpx); dqy = similar(dpx); dv = KA.allocate(backend, Int32, nf)
            lucaskanade!(dqx, dqy, dv, g0, g1, ix0, iy0, dpx, dpy; win = window, iters = iters)
            dbx = similar(dpx); dby = similar(dpx); dbv = KA.allocate(backend, Int32, nf)
            lucaskanade!(dbx, dby, dbv, g1, g0, ix1, iy1, dqx, dqy; win = window, iters = iters)
            hqx = Array(dqx); hqy = Array(dqy); hv = Array(dv)
            hbx = Array(dbx); hby = Array(dby); hbv = Array(dbv)
            opx = Float64[]; opy = Float64[]; nqx = Float64[]; nqy = Float64[]
            for i in 1:nf
                (hv[i] == 1 && hbv[i] == 1) || continue
                hypot(hbx[i] - px[i], hby[i] - py[i]) > fbmax && continue
                push!(opx, px[i]); push!(opy, py[i]); push!(nqx, hqx[i]); push!(nqy, hqy[i])
            end
            if length(opx) >= 4
                tinc = ransacsimilarity(opx, opy, nqx .- opx, nqy .- opy; thresh = ransacpx)
                transforms[k] = Mat3f(tinc * transforms[k - 1])
            else
                transforms[k] = transforms[k - 1]
            end
            px = nqx; py = nqy
            if length(px) < minfeatures || k % redetectevery == 0
                for p in goodfeatures(backend, g1; maxpoints = maxfeatures, border = window + 4)
                    qx = Float64(p[1]); qy = Float64(p[2]); near = false
                    for j in eachindex(px)
                        if (qx - px[j])^2 + (qy - py[j])^2 < window * window
                            near = true; break
                        end
                    end
                    near || (push!(px, qx); push!(py, qy))
                end
            end
            g0, g1 = g1, g0; ix0, ix1 = ix1, ix0; iy0, iy1 = iy1, iy0
            progress === nothing || (k % 30 == 0 && progress(k, 2n))
        end
    finally
        frames === nothing && close(dec)
    end
    return transforms
end

# Pass 2: drift-free translation refinement against frame-1 NCC templates.
function refinecampath(clip::Clip, seed::Vector{Mat3f}, n::Integer; window::Integer,
                       radius::Integer, minscore::Real, minmargin::Real, thresh::Real,
                       maxfeatures::Integer, backend, progress, frames = nothing)
    source = clip.source
    W, H = source.width, source.height
    dec = frames === nothing ? GrayReader(source) : frames
    host = Matrix{Float32}(undef, W, H)   # PatchTracker (template cut) needs host
    dg = KA.allocate(backend, Float32, (W, H))
    A = copy(seed)
    try
        hostgray!(host, dec, clip.src_in)
        dg0 = KA.allocate(backend, Float32, (W, H)); copyto!(dg0, host)
        feats = goodfeatures(backend, dg0; maxpoints = maxfeatures, border = radius + window)
        length(feats) >= 6 || return A
        tracker = PatchTracker(backend, host, feats; window = window,
                               maxradius = radius + 18, minstd = 0.0)
        for k in 2:n
            gray = grayinto!(dg, dec, clip.src_in + k - 1)
            good = [m for m in matchpatches!(tracker, gray, seed[k]; radius = radius)
                    if m.score >= minscore && m.margin >= minmargin]
            if length(good) >= 6
                dx, dy = ransactranslation([m.dx for m in good], [m.dy for m in good]; thresh = thresh)
                A[k] = Mat3f(seed[k] * Mat3f(1, 0, 0, 0, 1, 0, dx, dy, 1))
            end
            progress === nothing || (k % 30 == 0 && progress(n + k, 2n))
        end
    finally
        frames === nothing && close(dec)
    end
    return A
end

"""
    similaritypath!(clip; backend=KA.CPU(), progress=nothing, ...) -> MotionTrack

Camera lock via the two-pass Lucas-Kanade + RANSAC-similarity pipeline (see file
header). Estimates a smooth per-frame **similarity** (translation + rotation + uniform
scale — so it takes out a real zoom or roll, not just pan/tilt), drift-free against
frame 1, and stores it as the clip's [`MotionTrack`]. This is the mode that matches a
dedicated NLE's *Similarity + Camera Lock*; `analyzemotion!(:similarity)` calls it.
"""
function similaritypath!(clip::Clip; window::Integer = 11, iters::Integer = 15,
                         minfeatures::Integer = 220, maxfeatures::Integer = 520,
                         ransacpx::Real = 2.0, fbmax::Real = 0.7, redetectevery::Integer = 5,
                         refwindow::Integer = 28, refradius::Integer = 10,
                         outlierwindow::Integer = 3, outliertrans::Real = 20.0,
                         outlierscale::Real = 0.1, outlierrot::Real = 5.0,
                         backend = KA.CPU(), progress = nothing)
    n = srclength(clip)
    n >= 2 || return nothing
    # Feed both passes from the backend's frame source: on the GPU the chunked
    # stream decodes on-device with bounded VRAM. Replaces the old whole-source
    # RGB predecode, which needed ~GBs and fell back to CPU decode on long
    # clips — leaving the analysis decode-bound at high resolutions.
    frames = graysource(backend, clip.source)
    A = try
        A = trackcampath(clip, n; window, iters, minfeatures, maxfeatures, ransacpx,
                         fbmax, redetectevery, backend, progress, frames)
        refinecampath(clip, A, n; window = refwindow, radius = refradius, minscore = 0.5,
                      minmargin = 0.05, thresh = 2.0, maxfeatures = 400, backend, progress, frames)
    finally
        close(frames)
    end
    # scrub single-frame glitches in (scale, rotation, translation) — WITHOUT
    # attenuating real motion, which a lock must fully cancel rather than smooth
    sc = deoutlier([Float64(hypot(M[1, 1], M[2, 1])) for M in A], outlierwindow, outlierscale)
    th = deoutlier([Float64(atand(M[2, 1], M[1, 1])) for M in A], outlierwindow, outlierrot)
    tx = deoutlier([Float64(M[1, 3]) for M in A], outlierwindow, outliertrans)
    ty = deoutlier([Float64(M[2, 3]) for M in A], outlierwindow, outliertrans)
    transforms = [begin
                      s = Float32(sc[k]); co = Float32(cosd(th[k])); si = Float32(sind(th[k]))
                      Mat3f(s * co, s * si, 0, -s * si, s * co, 0, Float32(tx[k]), Float32(ty[k]), 1)
                  end for k in 1:n]
    clip.motiontrack = MotionTrack(transforms, clip.src_in, :similarity)
    progress === nothing || progress(2n, 2n)
    return clip.motiontrack
end
