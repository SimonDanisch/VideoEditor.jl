# Global-bundle camera lock — the production, DaVinci-parity stabilizer.
#
# A causal per-frame tracker cannot lock this class of footage (near-static
# camera, periodic background texture, an independently moving subject): it
# either chases the subject, freezes when its reference templates go stale, or
# aliases on the repetitive texture. DaVinci's stabilizer instead tracks long
# point trajectories and solves for a globally-consistent camera path, keeping
# only the tracks that agree (its green inliers) and discarding the moving
# subject's (its red outliers). This does the same, in two stages:
#
#   1. `extracttracks` — Shi-Tomasi seeds tracked frame-to-frame by sub-pixel
#      NCC, respawned as they die, yielding many overlapping trajectories.
#      Frame-to-frame matching never goes stale, so coverage survives the
#      moments a fixed reference would lose (a bird crossing the frame).
#   2. `bundleadjust` — a global 2D-similarity bundle adjustment: solve every
#      frame's transform `W_f` (sampling matrix, `W_1 = I` anchors the lock to
#      frame 1) and every track's world position `X_i` jointly, by alternating
#      robust (Huber IRLS) least squares. The moving subject's tracks are
#      high-residual, so IRLS downweights them automatically — no manual mask.
#      Because the solve is global and frame-1 anchored, there is no causal
#      drift and no staleness freeze.

"A feature trajectory over contiguous frames `f0 .. f0+length-1` in analysis pixels."
struct Track
    f0::Int
    xs::Vector{Float32}
    ys::Vector{Float32}
end
Base.length(t::Track) = length(t.xs)

@inline function trackable(x::Real, y::Real, gw::Int, gh::Int, half::Int, R::Int)
    rx = round(Int, x); ry = round(Int, y)
    return rx - half - R >= 1 && rx + half - 1 + R <= gw &&
           ry - half - R >= 1 && ry + half - 1 + R <= gh
end

"""
    extracttracks(backend, clip, gw, gh; window, searchradius, ...) -> Vector{Track}

Decode `clip` (streaming — only the previous and current analysis frame are
held), downscale each frame to `gw × gh`, and track Shi-Tomasi features
frame-to-frame with sub-pixel NCC ([`matchpatches!`](@ref)). Tracks that lose
their match (low score / margin, or leaving the frame) are retired; fresh
features are seeded every `respawn` frames and whenever live tracks fall below
`mintracks`, so coverage never collapses. `backend` runs the detection/NCC
kernels AND the decode: a GPU backend feeds frames from the chunked GPU
stream (grayscale on-device, only the gray crosses to the host).
"""
function extracttracks(backend, clip::Clip, gw::Int, gh::Int;
                       window::Integer = 32, searchradius::Integer = 8,
                       minscore::Real = 0.5, minmargin::Real = 0.08,
                       maxfeatures::Integer = 240, respawn::Integer = 8,
                       mintracks::Integer = 120, minlen::Integer = 6,
                       progress = nothing)
    source = clip.source
    n = srclength(clip)
    w = Int(window); iseven(w) || (w -= 1)
    half = w ÷ 2
    R = Int(searchradius)
    margin = half + R + 3
    ident = Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1)
    dec = graysource(backend, source)   # GPU decode+grayscale on a GPU backend
    # full-res gray and the resize stay on the analysis backend; only the tiny
    # gw×gh working image crosses to the host
    fullgray = KA.allocate(backend, Float32, (source.width, source.height))
    sgray = KA.allocate(backend, Float32, (gw, gh))
    bufa = Matrix{Float32}(undef, gw, gh)
    bufb = Matrix{Float32}(undef, gw, gh)
    prev, cur = bufa, bufb
    readgray!(dst, f) = begin
        grayinto!(fullgray, dec, clip.src_in + f - 1)
        bilinearresize!(sgray, fullgray)
        copyto!(dst, sgray)
    end
    px = Float32[]; py = Float32[]
    active = Track[]
    finished = Track[]
    seed!(f, gray) = begin
        for (qx, qy) in detectfeatures(backend, gray, maxfeatures, margin)
            x = Float32(qx); y = Float32(qy)
            near = false
            for k in eachindex(px)
                if (x - px[k])^2 + (y - py[k])^2 < (w * 0.6f0)^2
                    near = true; break
                end
            end
            near && continue
            push!(px, x); push!(py, y); push!(active, Track(f, Float32[x], Float32[y]))
        end
    end
    try
        readgray!(prev, 1)
        seed!(1, prev)
        for f in 2:n
            readgray!(cur, f)
            idx = [k for k in eachindex(px) if trackable(px[k], py[k], gw, gh, half, R)]
            alive = falses(length(px))
            if !isempty(idx)
                centers = [(round(Int, px[k]), round(Int, py[k])) for k in idx]
                tr = PatchTracker(backend, prev, centers; window = w,
                                  maxradius = R + 4, minstd = 0.0)
                ms = matchpatches!(tr, cur, ident; radius = R)
                for (j, k) in enumerate(idx)
                    m = ms[j]
                    if m.score >= minscore && m.margin >= minmargin &&
                       abs(m.dx) <= R && abs(m.dy) <= R
                        px[k] += m.dx; py[k] += m.dy
                        push!(active[k].xs, px[k]); push!(active[k].ys, py[k])
                        alive[k] = true
                    end
                end
            end
            for k in eachindex(px)
                alive[k] || (length(active[k]) >= minlen && push!(finished, active[k]))
            end
            keep = findall(alive)
            px = px[keep]; py = py[keep]; active = active[keep]
            (f % respawn == 0 || length(px) < mintracks) && seed!(f, cur)
            prev, cur = cur, prev
            progress === nothing || (f % 60 == 0 && progress(f, n))
        end
    finally
        close(dec)
    end
    for t in active
        length(t) >= minlen && push!(finished, t)
    end
    return finished
end

"""
    bundleadjust(tracks, n; iters=25, huber=1.5) -> (A, B, TX, TY)

Global 2D-similarity bundle adjustment over `n` frames. Solves per-frame
sampling similarities `W_f = [A -B TX; B A TY]` (with `W_1 = I`) and per-track
world positions jointly, minimising `Σ ρ(|W_f·X_i − q_{i,f}|)` with a Huber
loss (threshold `huber`, analysis pixels) by alternating reweighted least
squares. Independently-moving points are high-residual and get downweighted,
so the fit locks to the dominant (background) motion. Returns the per-frame
similarity parameters in analysis-pixel coordinates.
"""
function bundleadjust(tracks::Vector{Track}, n::Int; iters::Integer = 25, huber::Real = 1.5)
    ntr = length(tracks)
    Xx = Float64[mean(t.xs) for t in tracks]
    Xy = Float64[mean(t.ys) for t in tracks]
    A = ones(Float64, n); B = zeros(Float64, n)
    TX = zeros(Float64, n); TY = zeros(Float64, n)
    perframe = [Tuple{Int, Float64, Float64}[] for _ in 1:n]
    for i in 1:ntr
        t = tracks[i]
        for l in 1:length(t)
            push!(perframe[t.f0 + l - 1], (i, Float64(t.xs[l]), Float64(t.ys[l])))
        end
    end
    δ = Float64(huber)
    for _ in 1:iters
        # solve every frame's similarity from the current world points (robust)
        for f in 2:n
            lst = perframe[f]
            length(lst) < 3 && continue
            M = zeros(4, 4); b = zeros(4)
            for (i, qx, qy) in lst
                x = Xx[i]; y = Xy[i]
                rx = A[f] * x - B[f] * y + TX[f] - qx
                ry = B[f] * x + A[f] * y + TY[f] - qy
                r = hypot(rx, ry); ω = r <= δ ? 1.0 : δ / r
                r1 = (x, -y, 1.0, 0.0); r2 = (y, x, 0.0, 1.0)
                for a in 1:4, c in 1:4
                    M[a, c] += ω * (r1[a] * r1[c] + r2[a] * r2[c])
                end
                for a in 1:4
                    b[a] += ω * (r1[a] * qx + r2[a] * qy)
                end
            end
            c = M \ b
            A[f] = c[1]; B[f] = c[2]; TX[f] = c[3]; TY[f] = c[4]
        end
        A[1] = 1.0; B[1] = 0.0; TX[1] = 0.0; TY[1] = 0.0
        # solve every world point from the current frame similarities (robust)
        for i in 1:ntr
            g = 0.0; s1 = 0.0; s2 = 0.0
            t = tracks[i]
            for l in 1:length(t)
                f = t.f0 + l - 1; qx = Float64(t.xs[l]); qy = Float64(t.ys[l])
                a = A[f]; bb = B[f]
                rx = a * Xx[i] - bb * Xy[i] + TX[f] - qx
                ry = bb * Xx[i] + a * Xy[i] + TY[f] - qy
                r = hypot(rx, ry); ω = r <= δ ? 1.0 : δ / r
                rqx = qx - TX[f]; rqy = qy - TY[f]; gg = a * a + bb * bb
                g += ω * gg; s1 += ω * (a * rqx + bb * rqy); s2 += ω * (-bb * rqx + a * rqy)
            end
            g > 0 && (Xx[i] = s1 / g; Xy[i] = s2 / g)
        end
    end
    return A, B, TX, TY
end

"""
    bundlelock!(clip; analysis_width=480, backend=KA.CPU(), progress=nothing) -> MotionTrack

Production camera lock: extract long feature trajectories ([`extracttracks`])
and solve a global similarity bundle adjustment ([`bundleadjust`]) for a
drift-free, subject-rejecting virtual tripod. Runs at `analysis_width`; the
resulting per-frame similarity sampling matrices are rescaled to source pixels
and stored as the clip's [`MotionTrack`]. This is the mode that matches
DaVinci's *Similarity + Camera Lock*.
"""
function bundlelock!(clip::Clip; analysis_width::Integer = 480, window::Integer = 32,
                     searchradius::Integer = 8, huber::Real = 1.5, iters::Integer = 25,
                     backend = KA.CPU(), progress = nothing, kwargs...)
    n = srclength(clip)
    n >= 2 || return nothing
    source = clip.source
    aw = clamp(Int(analysis_width), 120, source.width)
    ah = max(round(Int, source.height / source.width * aw), 24)
    scale = Float64(source.width / aw)
    tracks = extracttracks(backend, clip, aw, ah; window, searchradius, progress)
    if length(tracks) < 8
        # too textureless to bundle — leave an identity lock
        transforms = fill(Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), n)
        setmotiontrack!(clip, MotionTrack(transforms, clip.src_in, :similarity))
        return clip.motiontrack
    end
    A, B, TX, TY = bundleadjust(tracks, n; iters, huber)
    transforms = [Mat3f(A[f], B[f], 0, -B[f], A[f], 0, scale * TX[f], scale * TY[f], 1)
                  for f in 1:n]
    setmotiontrack!(clip, MotionTrack(transforms, clip.src_in, :similarity))
    progress === nothing || progress(n, n)
    return clip.motiontrack
end
