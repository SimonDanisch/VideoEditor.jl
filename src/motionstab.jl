"""
    analyzemotion!(clip; mode=:similarity, analysis_width=320, cutoff=1.0,
                   backend=KA.CPU(), progress=nothing) -> MotionTrack

Camera stabilization. Four modes:

- `:similarity` (default): virtual tripod via NCC patch tracking — see
  [`cameralock!`](@ref). Translation + rotation + uniform scale only, so
  measurement noise cannot warp the image; drift-free, blur-tolerant,
  device-accelerated. This is the mode that matches a dedicated NLE's
  "camera lock".
- `:tripod`: the older dense-flow variant — every frame's flow is computed
  directly against the reference and a global **affine** (adds shear) is
  fitted (trimmed least squares). Kept for scenes where the extra degrees
  of freedom genuinely help; noisier than `:similarity` on real footage.
- `:perspective`: tripod with a full **homography** fit — adds the two
  keystone terms. On scenes with depth the fit absorbs parallax it cannot
  model into keystone, so the terms are median-filtered in time and capped
  at a physically plausible budget ([`limitkeystone!`](@ref)).
- `:smooth`: classic path smoothing — accumulate frame-to-frame global
  translation, low-pass the path (`cutoff` Hz): intentional pans survive,
  shake doesn't.

`backend` selects the device for the patch kernels / flow buffers; decode
and grayscale stay on the CPU. Like `ColorTrack`, keyed by absolute source
frame → survives splits. Applied as a projective warp before color/effects
in preview and export. Saved with the project.
"""
function analyzemotion!(clip::Clip; mode::Symbol = :similarity, analysis_width::Integer = 320,
                        cutoff::Real = 1.0, levels::Integer = 4,
                        backend = KA.CPU(), progress = nothing)
    mode in (:similarity, :tripod, :perspective, :smooth) ||
        error("mode must be :similarity, :tripod, :perspective or :smooth")
    mode === :similarity && return similaritypath!(clip; backend, progress)
    mode === :bundle && return bundlelock!(clip; analysis_width, backend, progress)
    n = cliplength(clip)
    n >= 24 || return nothing
    source = clip.source
    aw = Int(analysis_width)
    ah = max(round(Int, source.height / source.width * aw), 24)
    scale = Float32(source.width / aw)

    sr = SequentialReader(source)
    frame = RGBFrame(undef, source.width, source.height)
    small = RGBFrame(undef, aw, ah)
    ghost = Matrix{Float32}(undef, aw, ah)          # CPU staging
    gref = KA.allocate(backend, Float32, (aw, ah))  # device buffers
    gprev = KA.allocate(backend, Float32, (aw, ah))
    gcur = KA.allocate(backend, Float32, (aw, ah))
    u = KA.allocate(backend, Float32, (aw, ah))
    v = KA.allocate(backend, Float32, (aw, ah))
    flowws = FlowWorkspace(backend, (aw, ah); levels)  # reused across all frames

    transforms = fill(Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), n)
    path = zeros(Float32, n, 2)  # smooth mode: translation path relative to frame 1
    try
        for i in 1:n
            readframe!(frame, sr, clip.src_in + i - 1)
            warp!(small, frame, (0.0, 0.0, 1.0, 1.0))  # downscale
            grayscale!(ghost, small)
            copyto!(gcur, ghost)
            if i == 1
                copyto!(gref, gcur)
            elseif mode !== :smooth
                opticalflow!(flowws, u, v, gref, gcur)  # direct to reference: no drift
                T = mode === :perspective ? fithomography(u, v) : fitaffine(u, v)
                transforms[i] = scaletosource(T, scale)
            else
                opticalflow!(flowws, u, v, gprev, gcur)
                uh, vh = u isa Matrix ? u : Array(u), v isa Matrix ? v : Array(v)
                path[i, 1] = path[i - 1, 1] + median(vec(uh))
                path[i, 2] = path[i - 1, 2] + median(vec(vh))
            end
            gprev, gcur = gcur, gprev
            progress === nothing || i % 60 == 0 && progress(i, n)
        end
    finally
        close(sr)
    end

    if mode === :smooth
        fps = source.framerate
        normalized = clamp(2 * cutoff / fps, 1.0e-3, 0.95)
        lowpass = DSP.digitalfilter(DSP.Lowpass(normalized), DSP.Butterworth(2))
        sx = DSP.filtfilt(lowpass, Float64.(path[:, 1]))
        sy = DSP.filtfilt(lowpass, Float64.(path[:, 2]))
        for i in 1:n
            transforms[i] = translationmatrix((sx[i] - path[i, 1]) * scale,
                                              (sy[i] - path[i, 2]) * scale)
        end
    end
    mode === :perspective && limitkeystone!(transforms, source.width, source.height)
    clip.motiontrack = MotionTrack(transforms, clip.src_in, mode)
    progress === nothing || progress(n, n)
    return clip.motiontrack
end

"""
    limitkeystone!(transforms, W, H; budget=0.05) -> transforms

Sanity-limit the projective terms of a transform series: replace
single-frame SPIKES (a value deviating hard from its temporal neighbors —
a fit failure, not camera motion) by the local median, and cap the corner
displacement keystone may cause at `budget` × the frame diagonal —
hand-held orientation drift keystones by a few percent at most; anything
larger is the homography fit absorbing parallax it cannot model, which
shows as frame-to-frame jelly. Genuine smooth keystone, fast or slow,
passes through untouched (the perspective testsuite depends on that).
"""
function limitkeystone!(transforms::Vector{Mat3f}, W::Integer, H::Integer;
                        budget::Real = 0.05)
    cap = budget * hypot(W, H)
    n = length(transforms)
    p1 = Float64[M[3, 1] for M in transforms]
    p2 = Float64[M[3, 2] for M in transforms]
    m1 = copy(p1)
    m2 = copy(p2)
    for i in 2:(n - 1)
        med1 = median((p1[i - 1], p1[i], p1[i + 1]))
        med2 = median((p2[i - 1], p2[i], p2[i + 1]))
        abs(p1[i] - med1) > 5.0e-5 + abs(med1) && (m1[i] = med1)
        abs(p2[i] - med2) > 5.0e-5 + abs(med2) && (m2[i] = med2)
    end
    disp(M, q1, q2) = maximum(((x, y),) -> begin
            den = q1 * x + q2 * y + M[3, 3]
            ax = M[1, 1] * x + M[1, 2] * y + M[1, 3]
            ay = M[2, 1] * x + M[2, 2] * y + M[2, 3]
            hypot(ax / den - ax, ay / den - ay)
        end, ((1, 1), (W, 1), (1, H), (W, H)))
    for i in 1:n
        M = transforms[i]
        q1, q2 = m1[i], m2[i]
        k = 0
        while disp(M, q1, q2) > cap && k < 40   # den is nonlinear: iterate down
            s = cap / disp(M, q1, q2)
            q1 *= s
            q2 *= s
            k += 1
        end
        transforms[i] = Mat3f(M[1, 1], M[2, 1], q1, M[1, 2], M[2, 2], q2,
                              M[1, 3], M[2, 3], M[3, 3])
    end
    return transforms
end

"""
Rescale a fitted transform from analysis to source resolution: conjugation
by `diag(s, s, 1)` — translation scales by `s`, perspective terms by `1/s`,
the linear part is resolution-independent (affine fits pass through with
their zero bottom row untouched).
"""
scaletosource(T::Mat3f, s::Real) = Mat3f(T[1, 1], T[2, 1], T[3, 1] / s,
                                         T[1, 2], T[2, 2], T[3, 2] / s,
                                         T[1, 3] * s, T[2, 3] * s, T[3, 3])

"Decompose a similarity step and clamp it to physically plausible per-frame
bounds — hand tremor between two frames at 60 fps is fractions of a degree,
of a percent of zoom, and a bounded shift. A measurement outside these
bounds is a tracking error, not camera motion."
function clampsimilarity(F::Mat3f; maxdθ::Real = 0.5, maxds::Real = 0.005,
                         maxt::Real = 25.0)
    s = clamp(hypot(F[1, 1], F[2, 1]), 1 - maxds, 1 + maxds)
    θ = clamp(atand(F[2, 1], F[1, 1]), -maxdθ, maxdθ)
    tn = hypot(F[1, 3], F[2, 3])
    f = tn > maxt ? maxt / tn : 1.0
    return Mat3f(s * cosd(θ), s * sind(θ), 0, -s * sind(θ), s * cosd(θ), 0,
                 F[1, 3] * f, F[2, 3] * f, 1)
end

"Detect Shi-Tomasi corners on `backend`; uploads the host gray for a GPU pass."
function detectfeatures(backend, gray::AbstractMatrix{Float32}, maxpoints::Integer,
                        border::Integer)
    gdev = backend isa KA.CPU ? gray :
           (d = KA.allocate(backend, Float32, size(gray)); copyto!(d, gray); d)
    return goodfeatures(backend, gdev; maxpoints = maxpoints, border = border)
end

"""
    cameralock!(clip; point=nothing, ...) -> MotionTrack

Virtual tripod, the way DaVinci's "Similarity + Camera Lock" works: track
a grid of reference patches from the clip's first frame by normalized
cross-correlation ([`PatchTracker`](@ref) — device kernels, `backend`
selects CPU or GPU) and fit a trimmed 4-DOF similarity to the surviving
matches. Per frame the search sits on a constant-velocity prediction and
spans `searchradius` px (`lostradius` while reacquiring); matches must
pass `minscore` AND a uniqueness `minmargin` (repetitive texture matches
itself well somewhere — only a unique peak is a lock). Fewer than
`mintemplates` good matches (heavy motion blur, occlusion) coasts instead
of composing garbage, and the cumulative lock is clamped to `maxscale` /
`maxangle` — a locked handheld camera neither zooms far nor rolls far.
Every measurement is against the FIRST frame, so nothing drifts.

With `point` (source pixels in the first frame), one extra patch pins the
content under it exactly — object lock: the scene lock removes rotation
and scale, the object patch's remaining parallax offset composes on top.

Measured on the real birdhouse footage (1889 frames, 1080×1920@60): zero
coast frames, rotation jitter 0.014°/frame, time-slices ruler-straight —
matching a DaVinci Similarity+Camera Lock export of the same clip. Keyed
by absolute source frame → survives splits; saved with the project.
"""
function cameralock!(clip::Clip; point = nothing, window::Integer = 96,
                     searchradius::Integer = 16, lostradius::Integer = 48,
                     gridcols::Integer = 4, gridrows::Integer = 7,
                     minscore::Real = 0.55, minmargin::Real = 0.08,
                     mintemplates::Integer = 5, maxscale::Real = 1.25,
                     maxangle::Real = 15.0, stepmaxt::Real = 25.0,
                     stepmaxs::Real = 0.005, stepmaxθ::Real = 0.5,
                     reacqradius::Integer = 64, coastframes::Integer = 90,
                     maxfeatures::Integer = 120, replenishfrac::Real = 0.5,
                     redetectevery::Integer = 6,
                     backend = KA.CPU(), progress = nothing, debug = nothing)
    n = cliplength(clip)
    n >= 2 || return nothing
    source = clip.source
    # small sources get proportionally smaller patches and searches
    window = clamp(min(Int(window), source.width ÷ 3, source.height ÷ 3), 24, Int(window))
    iseven(window) || (window -= 1)
    lostradius = min(Int(lostradius), window)
    searchradius = min(Int(searchradius), lostradius)
    sr = SequentialReader(source)
    frame = RGBFrame(undef, source.width, source.height)
    gray = Matrix{Float32}(undef, source.width, source.height)
    transforms = fill(Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), n)
    anchor = nothing
    fillt = nothing
    obj = nothing
    objd = (0.0, 0.0)            # the pinned object's offset in the locked domain
    ow = min(64, window)
    px = point === nothing ? 0 : clamp(round(Int, point[1]), ow ÷ 2 + 1, source.width - ow ÷ 2)
    py = point === nothing ? 0 : clamp(round(Int, point[2]), ow ÷ 2 + 1, source.height - ow ÷ 2)
    exr = (window + ow) / 2 + searchradius   # grid patch sees the subject within this
    Ma = transforms[1]
    vel = Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1)   # constant-velocity coast estimate
    lost = 0
    detmargin = window ÷ 2 + reacqradius + 4
    lastdetect = 1
    try
        for i in 1:n
            readframe!(frame, sr, clip.src_in + i - 1)
            grayscale!(gray, frame)
            if i == 1
                margin = window ÷ 2 + reacqradius + 4
                # DETECTED corners (Shi-Tomasi) are trackable through blur/fast
                # motion where an arbitrary grid isn't; fall back to a grid only
                # if the frame is too textureless to yield enough.
                centers = detectfeatures(backend, gray, maxfeatures, margin)
                if length(centers) < 2 * mintemplates
                    centers = [(round(Int, x), round(Int, y))
                               for x in range(margin, source.width - margin; length = gridcols)
                               for y in range(margin, source.height - margin; length = gridrows)]
                end
                # `anchor` = frame-1 features (a drift-FREE absolute reference —
                # they pin the true frame-1 position whenever they match, and
                # snap the lock back when the camera returns near start, e.g. at
                # a loop end). `fill` = fresh re-detected features that carry the
                # track through the flight where the anchor set is unmatchable.
                anchor = PatchTracker(backend, gray, centers; window, maxradius = reacqradius)
                fillt = anchor
                if point !== nothing
                    # center-weighted: follow the SUBJECT under the click, not
                    # the background ring around it
                    obj = PatchTracker(backend, gray, [(px, py)];
                                       window = ow, maxradius = 2 * searchradius,
                                       minstd = 0.0, centerweight = true)
                end
                continue
            end
            # CONSTANT-VELOCITY prediction, applied EVERY frame — including while
            # lost. Coasting through unmatched (motion-blurred / whip-pan) frames
            # at the last known velocity keeps the search tracking the camera, so
            # the reference patches are reacquired the instant they're matchable
            # again (holding still instead loses the camera permanently — the
            # 16 s flight where every patch dropped and never came back).
            pred = Mat3f(Ma * vel)
            M = pred
            nmatch = 0
            rad = lost == 0 ? searchradius : lostradius
            # the pinned subject moves independently of the camera — patches that
            # currently see it would bend the similarity fit (a one-sided cluster
            # of coherent motion reads as rotation)
            keep(m) = m.score >= minscore && m.margin >= minmargin &&
                      (obj === nothing || hypot(m.x - (px + objd[1]), m.y - (py + objd[2])) > exr)
            for _ in 1:2
                good = [m for m in matchpatches!(fillt, gray, M; radius = rad) if keep(m)]
                nmatch = length(good)
                nmatch < mintemplates && break
                M = Mat3f(M * fitsimilarity([m.x for m in good], [m.y for m in good],
                                            [m.dx for m in good], [m.dy for m in good]))
                maximum(m -> max(abs(m.dx), abs(m.dy)), good) < 1 && break
            end
            if nmatch >= mintemplates
                lost = 0
                # PER-FRAME increment clamp (not just the cumulative rail): cap
                # how far the lock can move in ONE frame, so a single bad fit —
                # a bird crossing the grid, aliased wood grain — can only nudge
                # it. Frames are matched against frame 1, so good frames then
                # pull the lock back instead of it railing to the scale limit.
                step = clampsimilarity(Mat3f(inv(Ma) * M);
                                       maxdθ = stepmaxθ, maxds = stepmaxs, maxt = stepmaxt)
                Mc = Mat3f(Ma * step)
                s = clamp(hypot(Mc[1, 1], Mc[2, 1]), 1 / maxscale, maxscale)
                θ = clamp(atand(Mc[2, 1], Mc[1, 1]), -maxangle, maxangle)
                newMa = Mat3f(s * cosd(θ), s * sind(θ), 0, -s * sind(θ), s * cosd(θ), 0,
                              Mc[1, 3], Mc[2, 3], 1)
                # velocity = the committed frame-to-frame increment, TRANSLATION
                # ONLY (a virtual tripod has no sustained zoom/rotation, so a
                # scale/rotation velocity would run coasting away)
                inc = Mat3f(inv(Ma) * newMa)
                vel = Mat3f(1, 0, 0, 0, 1, 0, inc[1, 3], inc[2, 3], 1)
                Ma = newMa
            else
                lost += 1
                Ma = pred                      # coast forward at constant velocity
                lost > coastframes && (vel = Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1))  # give up extrapolating
            end
            # RE-DETECTION: when too few reference patches still match (the
            # templates went stale through blur / a big appearance change),
            # detect fresh corners in THIS frame and re-anchor them to frame-1
            # space via the current transform (Ma maps frame-1 → current, so a
            # fresh point at pₖ references Ma⁻¹·pₖ). Fresh templates match the
            # current look, the frame-1 references keep it drift-free, and it
            # never runs out of points — the flight where the fixed grid died.
            if nmatch < replenishfrac * maxfeatures && i - lastdetect >= redetectevery
                pcur = detectfeatures(backend, gray, maxfeatures, detmargin)
                if length(pcur) >= 2 * mintemplates
                    Mi = inv(Ma)
                    ref = [(round(Int, Mi[1, 1] * px2 + Mi[1, 2] * py2 + Mi[1, 3]),
                            round(Int, Mi[2, 1] * px2 + Mi[2, 2] * py2 + Mi[2, 3]))
                           for (px2, py2) in pcur]
                    fillt = PatchTracker(backend, gray, pcur; window, maxradius = reacqradius,
                                        refcenters = ref)
                    lastdetect = i
                    nmatch >= mintemplates || (lost = 1)  # give the fresh set a chance
                end
            end
            debug === nothing || push!(debug, (i, nmatch, lost))
            transforms[i] = Ma
            if obj !== nothing
                m = only(matchpatches!(obj, gray,
                                       Mat3f(Ma * translationmatrix(-objd[1], -objd[2]));
                                       radius = searchradius))
                m.score >= 0.35 && (objd = (objd[1] + m.dx, objd[2] + m.dy))
                transforms[i] = Mat3f(Ma * translationmatrix(-objd[1], -objd[2]))
            end
            progress === nothing || i % 60 == 0 && progress(i, n)
        end
    finally
        close(sr)
    end
    clip.motiontrack = MotionTrack(transforms, clip.src_in,
                                   point === nothing ? :similarity : :objectlock)
    progress === nothing || progress(n, n)
    return clip.motiontrack
end

"""
    analyzeobject!(clip, point; backend=KA.CPU(), progress=nothing) -> MotionTrack

Object lock: pin the content under `point` (source pixels in the clip's
first frame) so it does not move a pixel — [`cameralock!`](@ref) with an
extra reference patch at `point` whose parallax offset composes onto the
scene lock. Measured on real handheld footage: the pinned patch stays
within 0.15 px of its first-frame position over 960 frames.
"""
analyzeobject!(clip::Clip, point; kwargs...) = cameralock!(clip; point, kwargs...)

"""
    bordercrop(track, W, H) -> (x, y, w, h)

Normalized centered crop that cuts off the unfilled (black/replicate) border a
stabilization warp produces — the auto-"Zoom" of a dedicated NLE. Binary-searches
the smallest zoom whose four corners still sample inside the source frame for EVERY
frame's transform, so a real zoom or large excursion is fully hidden (capped at a
2.5× / 40%-per-side zoom so one pathological frame can't crop the shot away).
"""
function bordercrop(track::MotionTrack, W::Integer, H::Integer)
    Wf, Hf = Float64(W), Float64(H)
    cx, cy = (Wf + 1) / 2, (Hf + 1) / 2
    covered(z) = begin
        hw = (Wf - 1) / (2z); hh = (Hf - 1) / (2z)
        for M in track.transforms
            for (sx, sy) in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
                x = cx + sx * hw; y = cy + sy * hh
                d = M[3, 1] * x + M[3, 2] * y + M[3, 3]      # 1 for affine/similarity
                qx = (M[1, 1] * x + M[1, 2] * y + M[1, 3]) / d
                qy = (M[2, 1] * x + M[2, 2] * y + M[2, 3]) / d
                (1.0 <= qx <= Wf && 1.0 <= qy <= Hf) || return false
            end
        end
        return true
    end
    z = 1.0
    if !covered(1.0)
        lo, hi = 1.0, 1.0
        while !covered(hi) && hi < 2.5
            hi *= 1.15
        end
        for _ in 1:26
            mid = (lo + hi) / 2
            covered(mid) ? (hi = mid) : (lo = mid)
        end
        z = hi
    end
    frac = 1 / z
    off = (1 - frac) / 2
    return (off, off, frac, frac)
end

"""
    cropintersect(a, b) -> (x, y, w, h)

Intersection of two normalized crops — composes a user crop with the
stabilization [`bordercrop`](@ref): shrinks only where the crops disagree,
so a crop already inside the safe region passes through unchanged.
"""
function cropintersect(a::NTuple{4, Float64}, b::NTuple{4, Float64})
    # containment short-circuits return the input EXACTLY — the caller
    # compares crops with `!=`, so float noise here would read as a change
    inside(p, q) = p[1] >= q[1] && p[2] >= q[2] &&
                   p[1] + p[3] <= q[1] + q[3] && p[2] + p[4] <= q[2] + q[4]
    inside(a, b) && return a
    inside(b, a) && return b
    x = max(a[1], b[1])
    y = max(a[2], b[2])
    w = min(a[1] + a[3], b[1] + b[3]) - x
    h = min(a[2] + a[4], b[2] + b[4]) - y
    return (x, y, max(w, 0.01), max(h, 0.01))
end

"""
    findloop(clip; minseconds=1.5, maxseconds=6.0, matchwidth=64, step=2, progress) -> (a, b, score)

Find the best seamless-loop cut inside `clip`: two clip-relative frame offsets
`a < b` whose STABILIZED, brightness-normalized content matches most closely,
so playing `a…b` and jumping back to `a` loops with the least visible seam. The
clip's `motiontrack` (if present) locks the camera and its `crop` frames the
subject before matching; brightness normalization defeats exposure flicker so
the match is on content (e.g. a bird's pose), not lighting. Returns the offsets
(0-based, into the clip's source range) and the seam score (lower = better).
Search is sub-sampled by `step` frames for speed. This is a real content match,
not a reverse/boomerang.
"""
function findloop(clip::Clip; minseconds::Real = 1.5, maxseconds::Real = 6.0,
                  matchwidth::Integer = 64, step::Integer = 2, progress = nothing)
    src = clip.source
    fps = src.framerate
    n = cliplength(clip)
    n >= 4 || throw(ArgumentError("clip too short to loop"))
    W, H = src.width, src.height
    # warp at a small working resolution (≈90× less work than full res) —
    # applymotiontrack! rescales the transform to the buffer size for us
    dw = clamp(W ÷ 5, 160, 480); dh = max(round(Int, dw * H / W), 1)
    x, y, w, h = clip.crop
    cx0 = clamp(round(Int, x * dw) + 1, 1, dw); cx1 = clamp(round(Int, (x + w) * dw), cx0, dw)
    cy0 = clamp(round(Int, y * dh) + 1, 1, dh); cy1 = clamp(round(Int, (y + h) * dh), cy0, dh)
    gw = Int(matchwidth)
    gh = max(round(Int, gw * (cy1 - cy0 + 1) / (cx1 - cx0 + 1)), 1)
    frames = Array{Float32, 3}(undef, gw, gh, n)
    sr = SequentialReader(src)
    big = RGBFrame(undef, W, H)
    stmp = RGBFrame(undef, dw, dh)
    try
        for i in 1:n                           # sequential decode (no per-frame seeks)
            readframe!(big, sr, clip.src_in + i - 1)
            small = downscale(big, dw, dh)
            applymotiontrack!(small, stmp, clip, clip.src_in + i - 1)
            reg = downscale(collect(@view small[cx0:cx1, cy0:cy1]), gw, gh)
            g = @view frames[:, :, i]
            @inbounds for b in 1:gh, a in 1:gw
                c = reg[a, b]
                g[a, b] = 0.299f0 * Float32(c.r) + 0.587f0 * Float32(c.g) + 0.114f0 * Float32(c.b)
            end
            m = sum(g) / length(g)
            sd = sqrt(sum(abs2, g .- m) / length(g)) + 1.0f-6
            @inbounds for p in eachindex(g)
                g[p] = (g[p] - m) / sd
            end
            progress === nothing || i % 60 == 0 && progress(i, n)
        end
    finally
        close(sr)
    end
    minL = max(round(Int, minseconds * fps), 1)
    maxL = max(round(Int, maxseconds * fps), minL)
    npx = gw * gh
    best = (Inf32, 1, 1 + minL)
    st = Int(step)
    @inbounds for a in 1:st:(n - minL), L in minL:st:maxL
        b = a + L
        b > n && break
        d = 0.0f0
        for p in 1:npx
            d += (frames[p + (a - 1) * npx] - frames[p + (b - 1) * npx])^2
        end
        d < best[1] && (best = (d / npx, a, b))
    end
    score, a, b = best
    return (a - 1, b - 1, score)               # 0-based clip-relative offsets
end

"Apply the clip's motion stabilization for `srcframe`: affine warp via `tmp`.
Transforms are stored in the original source's pixels; when `buf` is a
lower-resolution preview proxy the transform is rescaled by conjugation."
function applymotiontrack!(buf::AnyRGBFrame, tmp::AnyRGBFrame, clip::Clip, srcframe::Integer)
    track = clip.motiontrack
    track === nothing && return buf
    i = srcframe - track.src_in + 1
    1 <= i <= length(track.transforms) || return buf
    M = track.transforms[i]
    M == Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1) && return buf
    s = size(buf, 1) / clip.source.width
    s ≈ 1 || (M = scaletosource(M, s))
    warp!(tmp, buf, M)
    KA.synchronize(KA.get_backend(buf))
    return copyto!(buf, tmp)
end
