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
function trackcampath(clip::Clip, n::Integer; window::Integer, iters::Integer,
                      minfeatures::Integer, maxfeatures::Integer, ransacpx::Real,
                      fbmax::Real, redetectevery::Integer, backend, progress, frames = nothing)
    source = clip.source
    W, H = source.width, source.height
    g0 = KA.allocate(backend, Float32, (W, H)); g1 = similar(g0)
    ix0 = similar(g0); iy0 = similar(g0); ix1 = similar(g0); iy1 = similar(g0)
    # `frames` (device-resident RGB, whole source) → Rec.709 grayscale on the GPU;
    # otherwise CPU decode + the same grayscale + upload. Identical grayscale on both
    # paths (GPU RGB matches VideoIO to ~1 level), so tracking is unchanged.
    usevio = frames === nothing
    sr = usevio ? SequentialReader(source) : nothing
    frame = usevio ? RGBFrame(undef, W, H) : nothing
    host = usevio ? Matrix{Float32}(undef, W, H) : nothing
    loadgray!(g, k) = usevio ?
        (readframe!(frame, sr, clip.src_in + k - 1); grayscale!(host, frame); copyto!(g, host)) :
        grayscale!(g, frames[clip.src_in + k])
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
        usevio && close(sr)
    end
    return transforms
end

# Pass 2: drift-free translation refinement against frame-1 NCC templates.
function refinecampath(clip::Clip, seed::Vector{Mat3f}, n::Integer; window::Integer,
                       radius::Integer, minscore::Real, minmargin::Real, thresh::Real,
                       maxfeatures::Integer, backend, progress, frames = nothing)
    source = clip.source
    W, H = source.width, source.height
    usevio = frames === nothing
    sr = usevio ? SequentialReader(source) : nothing
    frame = usevio ? RGBFrame(undef, W, H) : nothing
    host = Matrix{Float32}(undef, W, H)   # PatchTracker (template cut) needs host
    # GPU path: grayscale each RGB frame on-device and hand matchpatches! the device
    # array — it copies device→device (no per-frame host download / re-upload).
    dg = usevio ? nothing : KA.allocate(backend, Float32, (W, H))
    A = copy(seed)
    try
        if usevio
            readframe!(frame, sr, clip.src_in); grayscale!(host, frame)
        else
            grayscale!(host, Array(frames[clip.src_in + 1]))   # one download, for the templates
        end
        dg0 = KA.allocate(backend, Float32, (W, H)); copyto!(dg0, host)
        feats = goodfeatures(backend, dg0; maxpoints = maxfeatures, border = radius + window)
        length(feats) >= 6 || return A
        tracker = PatchTracker(backend, host, feats; window = window,
                               maxradius = radius + 18, minstd = 0.0)
        for k in 2:n
            gray = if usevio
                readframe!(frame, sr, clip.src_in + k - 1); grayscale!(host, frame); host
            else
                grayscale!(dg, frames[clip.src_in + k]); dg
            end
            good = [m for m in matchpatches!(tracker, gray, seed[k]; radius = radius)
                    if m.score >= minscore && m.margin >= minmargin]
            if length(good) >= 6
                dx, dy = ransactranslation([m.dx for m in good], [m.dy for m in good]; thresh = thresh)
                A[k] = Mat3f(seed[k] * Mat3f(1, 0, 0, 0, 1, 0, dx, dy, 1))
            end
            progress === nothing || (k % 30 == 0 && progress(n + k, 2n))
        end
    finally
        usevio && close(sr)
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
    n = cliplength(clip)
    n >= 2 || return nothing
    # Decode the whole source GPU-resident once (RGB), reused by both passes (no
    # per-frame CPU decode / grayscale / upload). Falls back to CPU decode for
    # unsupported streams (non-4:2:0, single-reference) or non-Lava backends.
    frames = nothing
    if gpu_decode_available(backend)
        try
            w, h, fr = gpu_decode_rgb(backend, clip.source.path)
            if (w, h) == (clip.source.width, clip.source.height) && length(fr) >= clip.src_in + n
                frames = fr
            end
        catch
            frames = nothing
        end
    end
    A = try
        A = trackcampath(clip, n; window, iters, minfeatures, maxfeatures, ransacpx,
                         fbmax, redetectevery, backend, progress, frames)
        refinecampath(clip, A, n; window = refwindow, radius = refradius, minscore = 0.5,
                      minmargin = 0.05, thresh = 2.0, maxfeatures = 400, backend, progress, frames)
    finally
        gpu_free_frames!(frames)   # release the ~GBs of GPU-resident frames now
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
