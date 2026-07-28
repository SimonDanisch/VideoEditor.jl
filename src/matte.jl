"""
Subject matting: isolate a subject and key the background out, driven by frames
the user marks.

The shape mirrors stabilization rather than the effect stack, and for the same
reason. Producing a matte is an *analysis* — expensive, sequential, and it wants
the whole clip — so it runs once into a [`MatteTrack`](@ref) and the render path
only ever samples that track. Nothing per-frame runs a model; there is no
per-clip model state to keep alive across scrubs, transitions and export, and the
GPU and CPU tiers stay one code path because both call [`applymatte!`](@ref).

The user's edit is the set of *seed* frames (matte keyframes) plus a rough
selection at each. Propagation fills the frames between them. `MatteTrack.alpha`
is therefore a cache: deleting it costs a recompute, not an edit.

The propagation itself is pluggable — see [`registermatte!`](@ref). A model
runner (LavaDNN's MatAnyone) registers the good one; the built-in fallback keeps
the feature, its UI and its tests working with no model installed.
"""

# ---------------------------------------------------------------- applying

using KernelAbstractions: @kernel, @index, @Const

@kernel function matte_kernel!(buf, @Const(alpha), mw::Int32, mh::Int32,
                               strength::Float32, feather::Float32, bg::Vec3f)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w, h = size(buf, 1), size(buf, 2)
        # sample the (usually smaller) matte bilinearly in normalized coords
        u = (Float32(i) - 0.5f0) / Float32(w) * Float32(mw) + 0.5f0
        v = (Float32(j) - 0.5f0) / Float32(h) * Float32(mh) + 0.5f0
        x0 = clamp(floor(Int32, u), Int32(1), mw); x1 = min(x0 + Int32(1), mw)
        y0 = clamp(floor(Int32, v), Int32(1), mh); y1 = min(y0 + Int32(1), mh)
        fx = clamp(u - Float32(x0), 0.0f0, 1.0f0)
        fy = clamp(v - Float32(y0), 0.0f0, 1.0f0)
        a00 = Float32(alpha[x0, y0]); a10 = Float32(alpha[x1, y0])
        a01 = Float32(alpha[x0, y1]); a11 = Float32(alpha[x1, y1])
        a = ((a00 * (1.0f0 - fx) + a10 * fx) * (1.0f0 - fy) +
             (a01 * (1.0f0 - fx) + a11 * fx) * fy) / 255.0f0
        # feather: push alpha toward a soft ramp around 0.5, so a hard mask can be
        # softened without re-running the analysis
        if feather > 0.0f0
            t = clamp((a - 0.5f0) / max(feather, 1.0f-3) * 0.5f0 + 0.5f0, 0.0f0, 1.0f0)
            a = t * t * (3.0f0 - 2.0f0 * t)
        end
        a = 1.0f0 - strength * (1.0f0 - a)     # strength 0 = matte off
        c = buf[i, j]
        src = Vec3f(red(c), green(c), blue(c))
        out = src .* a .+ bg .* (1.0f0 - a)
        buf[i, j] = RGB{N0f8}(clamp(out[1], 0.0f0, 1.0f0),
                              clamp(out[2], 0.0f0, 1.0f0),
                              clamp(out[3], 0.0f0, 1.0f0))
    end
end

"""
    applymatte!(buf, clip, srcframe; strength, feather, bg)

Key `buf` against the clip's matte for `srcframe`, in place. A no-op when the
clip has no track or the frame is outside it, so scrubbing past the analyzed
range shows the plain frame rather than a hole.

Called by the GPU graph's `MatteNode` *and* the CPU stack — the same function on
both, which is what keeps preview and export identical.
"""
function applymatte!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer;
                     strength::Real = 1.0, feather::Real = 0.0,
                     bg::Vec3f = Vec3f(0, 0, 0))
    track = clip.mattetrack
    track === nothing && return buf
    s = Float32(clamp(strength, 0.0, 1.0))
    s <= 0.0f0 && return buf
    i = srcframe - track.src_in + 1
    1 <= i <= size(track.alpha, 3) || return buf
    backend = KA.get_backend(buf)
    mw, mh = mattesize(track)
    plane = matteplane!(track, i, backend)
    matte_kernel!(backend)(buf, plane, Int32(mw), Int32(mh), s,
                           Float32(clamp(feather, 0.0, 1.0)), bg; ndrange = size(buf))
    KA.synchronize(backend)
    return buf
end

"""
One matte frame on `backend`, uploaded on demand and cached.

The track holds every frame's alpha on the host — it is the thing that gets
saved and edited — but the kernel needs the current frame's plane on whatever
device is rendering. Caching per `(track, backend, frame)` keeps a scrub from
re-uploading the same plane, and keys on the backend because the CPU and GPU
engines alternate frame to frame while the GPU warms up.
"""
const MATTEPLANES = IdDict{MatteTrack, Dict{Any, Tuple{Int, Any}}}()

function matteplane!(track::MatteTrack, i::Int, backend)
    per = get!(() -> Dict{Any, Tuple{Int, Any}}(), MATTEPLANES, track)
    hit = get(per, backend, nothing)
    hit === nothing || hit[1] == i && return hit[2]
    host = @view track.alpha[:, :, i]
    dev = if backend isa KA.CPU
        collect(host)
    else
        d = KA.allocate(backend, UInt8, size(host)...)
        copyto!(d, collect(host))
        d
    end
    per[backend] = (i, dev)
    return dev
end

"""
    warmmatte!() -> Bool

Run the propagator once on a tiny synthetic clip, to pay its first-call cost
somewhere the user is not waiting.

The first propagation of a session costs ~99 s on the MatAnyone propagator, and
**none of it is shader compilation** — the SPIR-V disk cache is fully hit (683
entries in, zero added). It is Julia specializing the graph's execution paths.

Most of that is size-independent, but not all: warming at 64x48 absorbed 97.9 s
and still left 12.5 s on the first real 240x136 clip, because the
cooperative-matrix GEMM specializes per tile shape (`Val{BLK}`). So warm at the
size that will actually be used — `w`/`h` should be the matte resolution the
tool will ask for, not a token.

Returns whether it ran (false when no propagator is installed, or it already has).
"""
const MATTEWARMED = Ref(false)

function warmmatte!(w::Integer = 480, h::Integer = 270)
    (MATTEWARMED[] || !hasmattemodel()) && return false
    MATTEWARMED[] = true
    w, h = Int(w), Int(h)
    frames = [fill(RGB{N0f8}(0.4, 0.5, 0.6), w, h) for _ in 1:2]
    qw, qh = max(1, w ÷ 4), max(1, h ÷ 4)
    for f in frames
        f[qw:2qw, qh:2qh] .= RGB{N0f8}(0.9, 0.3, 0.2)
    end
    seed = zeros(UInt8, w, h)
    seed[qw:2qw, qh:2qh] .= 0xff
    try
        MATTEPROPAGATOR[](frames, Dict(1 => seed); progress = nothing)
    catch e
        MATTEWARMED[] = false
        rethrow()
    end
    return true
end

"Drop cached device planes for `track` (or all of them)."
freematteplanes!(track::MatteTrack) = (delete!(MATTEPLANES, track); nothing)
freematteplanes!() = (empty!(MATTEPLANES); nothing)

# ---------------------------------------------------------------- producing

"""
    registermatte!(f)

Install the matte propagator. `f(frames, seeds; progress) -> Array{UInt8,3}`
receives the clip's frames as `Vector{Matrix{RGB{N0f8}}}` at matte resolution and
`seeds::Dict{Int, Matrix{UInt8}}` (index into `frames` → the user's rough
selection, 0/255), and returns `(w, h, nframes)` alpha.

This is the seam a model runner plugs into. VideoEditor deliberately does not
depend on one: the editor owns the track, the UI and the render path, and the
propagator is whatever is installed — the same reasoning that keeps Lava behind
`parentmodule(typeof(player.analysisbackend))` in `glbridge.jl`.
"""
const MATTEPROPAGATOR = Ref{Any}(nothing)
registermatte!(f) = (MATTEPROPAGATOR[] = f; MATTEWARMED[] = false; nothing)
hasmattemodel() = MATTEPROPAGATOR[] !== nothing
mattebackendname() = hasmattemodel() ? "model" : "built-in"

"""
Fallback propagator: per-frame colour-distance segmentation seeded by the marked
frames, propagated by carrying the seed's colour statistics forward.

It is not a matting model and does not pretend to be — it exists so that the
matte track, its keyframes, the inspector, export and the tests all work on a
machine with no weights installed, and so the good propagator has something to be
compared against.
"""
function fallbackpropagate(frames::Vector{<:AbstractMatrix{RGB{N0f8}}},
                           seeds::Dict{Int, Matrix{UInt8}}; progress = nothing)
    n = length(frames)
    w, h = size(frames[1])
    out = zeros(UInt8, w, h, n)
    isempty(seeds) && return out
    order = sort!(collect(keys(seeds)))
    # statistics of the marked region at each seed, carried to the frames it owns
    for (si, sf) in enumerate(order)
        mask = seeds[sf]
        fg = Vec3f(0, 0, 0); nf = 0
        bgc = Vec3f(0, 0, 0); nb = 0
        @inbounds for j in 1:h, i in 1:w
            c = frames[sf][i, j]
            v = Vec3f(red(c), green(c), blue(c))
            if mask[i, j] > 0x7f
                fg = fg .+ v; nf += 1
            else
                bgc = bgc .+ v; nb += 1
            end
        end
        nf == 0 && continue
        fg = fg ./ nf
        bgc = nb == 0 ? Vec3f(1, 1, 1) .- fg : bgc ./ nb
        lo = si == 1 ? 1 : order[si - 1] + (sf - order[si - 1]) ÷ 2 + 1
        hi = si == length(order) ? n : sf + (order[si + 1] - sf) ÷ 2
        for f in lo:hi
            @inbounds for j in 1:h, i in 1:w
                c = frames[f][i, j]
                v = Vec3f(red(c), green(c), blue(c))
                df = sum((v .- fg) .^ 2)
                db = sum((v .- bgc) .^ 2)
                a = df + db <= 1.0f-6 ? 0.5f0 : db / (df + db)
                out[i, j, f] = round(UInt8, clamp(a, 0.0f0, 1.0f0) * 255)
            end
            progress === nothing || progress(f, n)
        end
    end
    return out
end

"""
    analyzematte!(clip, readframe, seeds; mattewidth, progress) -> MatteTrack

Propagate `seeds` across the clip and store the result on it.

`readframe(srcframe) -> Matrix{RGB{N0f8}}` supplies source frames (the caller
decides decoder and tier); `seeds` maps an absolute source frame to a rough
selection at *source* resolution. The matte is computed at `mattewidth` and
sampled back up when applied, because a matte that tracks a subject does not need
per-pixel source detail and a full-resolution one costs a clip's worth of memory.
"""
function analyzematte!(clip::Clip, readframe, seeds::Dict{Int, <:AbstractMatrix};
                       mattewidth::Integer = 480, progress = nothing)
    n = cliplength(clip)
    n > 0 || error("cannot matte an empty clip")
    isempty(seeds) && error("matting needs at least one marked frame")
    first = readframe(clip.src_in)
    sw, sh = size(first)
    mw = min(Int(mattewidth), sw)
    mh = max(1, round(Int, sh * mw / sw))
    frames = Vector{Matrix{RGB{N0f8}}}(undef, n)
    for k in 1:n
        f = k == 1 ? first : readframe(clip.src_in + k - 1)
        frames[k] = downscale(f, mw, mh)   # area-average, from thumbnails.jl
        progress === nothing || progress(k, 2n)
    end
    localseeds = Dict{Int, Matrix{UInt8}}()
    for (sf, m) in seeds
        k = sf - clip.src_in + 1
        1 <= k <= n || continue
        localseeds[k] = mattemaskscale(m, mw, mh)
    end
    isempty(localseeds) && error("no marked frame falls inside the clip")
    prop = something(MATTEPROPAGATOR[], fallbackpropagate)
    alpha = prop(frames, localseeds;
                 progress = (d, t) -> progress === nothing ? nothing : progress(n + d, 2n))
    size(alpha) == (mw, mh, n) ||
        error("matte propagator returned $(size(alpha)), expected $((mw, mh, n))")
    track = MatteTrack(alpha, clip.src_in, sort!(collect(keys(seeds))))
    clip.mattetrack = track
    progress === nothing || progress(2n, 2n)
    return track
end

function mattemaskscale(src::AbstractMatrix, w::Int, h::Int)
    sw, sh = size(src)
    out = Matrix{UInt8}(undef, w, h)
    @inbounds for j in 1:h, i in 1:w
        v = src[clamp(round(Int, (i - 0.5) * sw / w + 0.5), 1, sw),
                clamp(round(Int, (j - 0.5) * sh / h + 0.5), 1, sh)]
        out[i, j] = v > 0 ? 0xff : 0x00
    end
    return out
end

"""
    seedmask(clip, rect) -> Matrix{UInt8}

A rectangular selection at source resolution: the cheapest thing a user can mark,
and enough to seed propagation. `rect` is normalized `(x, y, w, h)` with y from
the top, matching `Clip.crop`.
"""
function seedmask(clip::Clip, rect::NTuple{4, <:Real})
    sw, sh = clip.source.width, clip.source.height
    m = zeros(UInt8, sw, sh)
    x0 = clamp(round(Int, rect[1] * sw) + 1, 1, sw)
    x1 = clamp(round(Int, (rect[1] + rect[3]) * sw), 1, sw)
    y0 = clamp(round(Int, rect[2] * sh) + 1, 1, sh)
    y1 = clamp(round(Int, (rect[2] + rect[4]) * sh), 1, sh)
    m[x0:max(x0, x1), y0:max(y0, y1)] .= 0xff
    return m
end


"""
    framereader(player, clip) -> (srcframe -> RGBFrame)

Host RGB frames for analysis, one decoder reused across the clip.

Deliberately the CPU `SequentialReader` rather than the GPU stream: matting reads
every frame of the clip once, in order, which is exactly what a sequential
decoder is good at, and it keeps the analysis off the single-writer Vulkan queue
that the preview is using to stay responsive.
"""
function framereader(player, clip::Clip)
    sr = SequentialReader(clip.source)
    buf = RGBFrame(undef, clip.source.width, clip.source.height)
    # Hands back the decode buffer itself, not a copy. `analyzematte!` downscales
    # each frame before asking for the next, so the buffer is always consumed
    # before it is overwritten — and copying it first meant a full-resolution
    # allocation per frame (6 MB at 1080p) whose only use was to be shrunk to
    # 240x136 and dropped. That garbage cost more than the decode did.
    return sf -> (readframe!(buf, sr, Int(sf)); buf)
end
