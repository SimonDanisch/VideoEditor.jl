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

The propagation itself is pluggable — see [`registermatte!`](@ref). MatAnyone is
registered at load; there is no fallback, and a missing model fails loudly.
"""

# ---------------------------------------------------------------- applying

using KernelAbstractions: @kernel, @index, @Const

"""
    unitn0f8(x) -> N0f8

A 0…1 float as `N0f8`, clamped, without the range check.

`N0f8(x)` (and therefore `RGB{N0f8}(::Float32, …)`) validates and calls
`throw_colorerror`, which builds its message with `repr`/`string`. That drags
string allocation and dynamic dispatch into the kernel's IR, and Lava rejects the
whole kernel for it — a compile error on the *error path*, reached by no input.
The clamp here is the check.
"""
@inline unitn0f8(x::Real) =
    reinterpret(N0f8, unsafe_trunc(UInt8, clamp(Float32(x), 0.0f0, 1.0f0) * 255.0f0 + 0.5f0))

"""
The matte plane sampled bilinearly at plane coordinates `(u, v)`, as 0…1.

Split out of [`mattealphaat`](@ref) because feathering takes SEVERAL taps of the
same plane, and a second copy of the interpolation is a second matte.
"""
@inline function sampleplane(alpha, u::Float32, v::Float32, mw::Int32, mh::Int32)
    @inbounds begin
        x0 = clamp(floor(Int32, u), Int32(1), mw); x1 = min(x0 + Int32(1), mw)
        y0 = clamp(floor(Int32, v), Int32(1), mh); y1 = min(y0 + Int32(1), mh)
        fx = clamp(u - Float32(x0), 0.0f0, 1.0f0)
        fy = clamp(v - Float32(y0), 0.0f0, 1.0f0)
        a00 = Float32(alpha[x0, y0]); a10 = Float32(alpha[x1, y0])
        a01 = Float32(alpha[x0, y1]); a11 = Float32(alpha[x1, y1])
        return ((a00 * (1.0f0 - fx) + a10 * fx) * (1.0f0 - fy) +
                (a01 * (1.0f0 - fx) + a11 * fx) * fy) / 255.0f0
    end
end

"One axis of a 5-tap binomial (1 4 6 4 1) kernel — a Gaussian without a table."
@inline binom5(d::Int32) =
    d == Int32(0) ? 6.0f0 : (d == Int32(1) || d == Int32(-1) ? 4.0f0 : 1.0f0)

"""
The matte's alpha at one pixel — ONE definition, shared by the two things that
need it: keying the clip against a background, and telling the compositor how
much of what is underneath must show through. A second copy of this arithmetic
is a second matte, and they drift.

`feather` is a SPATIAL softening: it spreads the sampling footprint across the
edge, so the transition band widens while the interior stays fully opaque and the
background stays fully out. It used to be a smoothstep over `[0.5-f, 0.5+f]` in
alpha space, which at `f = 1` mapped alpha 1 to 0.84 and alpha 0 to 0.16 — the
subject went translucent and the background bled through everywhere. That is what
`strength` does, one slider up, so the two controls did the same thing and
neither one softened an edge.
"""
@inline function mattealphaat(alpha, i, j, w, h, mw::Int32, mh::Int32,
                              strength::Float32, feather::Float32,
                              cx::Float32, cy::Float32, cw::Float32, ch::Float32)
    # The matte covers the clip's CROP, the buffer is the whole layer: map
    # layer normalized coords into the crop rect before sampling.
    un = ((Float32(i) - 0.5f0) / Float32(w) - cx) / cw
    vn = ((Float32(j) - 0.5f0) / Float32(h) - cy) / ch
    u = un * Float32(mw) + 0.5f0
    v = vn * Float32(mh) + 0.5f0
    a = if feather > 0.0f0
        # 5×5 binomial taps, spaced up to 2 plane texels apart — the plane is 480
        # wide against a 1080-wide source, so full feather is a ±9 source-pixel
        # ramp. At feather 0 the spacing is 0, every tap lands on the same place
        # and this collapses to the single sample below: continuous, and it can
        # only ever soften.
        s = feather * 2.0f0
        acc = 0.0f0
        for dy in Int32(-2):Int32(2), dx in Int32(-2):Int32(2)
            acc += binom5(dx) * binom5(dy) *
                   sampleplane(alpha, u + Float32(dx) * s, v + Float32(dy) * s, mw, mh)
        end
        acc / 256.0f0
    else
        sampleplane(alpha, u, v, mw, mh)
    end
    return 1.0f0 - strength * (1.0f0 - a)     # strength 0 = matte off
end

@kernel function mattealpha_kernel!(out, @Const(alpha), mw::Int32, mh::Int32,
                                    strength::Float32, feather::Float32,
                                    cx::Float32, cy::Float32, cw::Float32, ch::Float32)
    i, j = @index(Global, NTuple)
    @inbounds begin
        a = mattealphaat(alpha, i, j, size(out, 1), size(out, 2), mw, mh,
                         strength, feather, cx, cy, cw, ch)
        out[i, j] = RGB{N0f8}(unitn0f8(a), unitn0f8(a), unitn0f8(a))
    end
end

@kernel function matte_kernel!(buf, @Const(alpha), mw::Int32, mh::Int32,
                               strength::Float32, feather::Float32, bg::Vec3f,
                               cx::Float32, cy::Float32, cw::Float32, ch::Float32)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w, h = size(buf, 1), size(buf, 2)
        # The matte covers the clip's CROP, `buf` is the whole layer: map layer
        # normalized coords into the crop rect before sampling. Outside it the
        # clamps below hold the edge, and those pixels are cropped away anyway.
        a = mattealphaat(alpha, i, j, w, h, mw, mh, strength, feather, cx, cy, cw, ch)
        c = buf[i, j]
        src = Vec3f(red(c), green(c), blue(c))
        out = src .* a .+ bg .* (1.0f0 - a)
        buf[i, j] = RGB{N0f8}(unitn0f8(out[1]), unitn0f8(out[2]), unitn0f8(out[3]))
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
    cr = clip.crop
    matte_kernel!(backend)(buf, plane, Int32(mw), Int32(mh), s,
                           Float32(clamp(feather, 0.0, 1.0)), bg,
                           Float32(cr[1]), Float32(cr[2]), Float32(cr[3]), Float32(cr[4]);
                           ndrange = size(buf))
    KA.synchronize(backend)
    return buf
end

"""
    mattealpha!(dst, clip, srcframe; strength, feather) -> dst | nothing

The clip's matte as a COVERAGE image the size of `dst` (white = keep, black =
show what is underneath), or `nothing` when this clip has no matte at this frame.

This is what makes a matte transparent instead of black. Keying paints the
removed background with `bg`, which is right when nothing is underneath and
wrong the moment there is: the black is opaque and covers the track below.
`applymatte!` and this share [`mattealphaat`](@ref), so the coverage the
compositor honours is exactly the coverage that was keyed.
"""
function mattealpha!(dst::AnyRGBFrame, clip::Clip, srcframe::Integer;
                     strength::Real = 1.0, feather::Real = 0.0)
    track = clip.mattetrack
    track === nothing && return nothing
    s = Float32(clamp(strength, 0.0, 1.0))
    s <= 0.0f0 && return nothing
    i = srcframe - track.src_in + 1
    1 <= i <= size(track.alpha, 3) || return nothing
    backend = KA.get_backend(dst)
    mw, mh = mattesize(track)
    plane = matteplane!(track, i, backend)
    cr = clip.crop
    mattealpha_kernel!(backend)(dst, plane, Int32(mw), Int32(mh), s,
                                Float32(clamp(feather, 0.0, 1.0)),
                                Float32(cr[1]), Float32(cr[2]), Float32(cr[3]), Float32(cr[4]);
                                ndrange = size(dst))
    KA.synchronize(backend)
    return dst
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
    # One path for every backend — no `backend isa CPU`.
    host = collect(@view track.alpha[:, :, i])
    dev = KA.allocate(backend, UInt8, size(host)...)
    copyto!(dev, host)
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
    MATTEWARMED[] && return false
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
        matteprop()(frames, Dict(1 => seed); progress = nothing)
    catch e
        MATTEWARMED[] = false
        rethrow()
    end
    return true
end

"The clip's cropped layer size — the space the matte lives in."
mattelayersize(clip::Clip) =
    (max(1, round(Int, clip.crop[3] * clip.source.width)),
     max(1, round(Int, clip.crop[4] * clip.source.height)))

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
"The installed propagator, or a loud failure. There is no fallback: a matte
without the model would be a different feature wearing its name."
function matteprop()
    p = MATTEPROPAGATOR[]
    p === nothing && error("no matte propagator installed — MatAnyone's assets are " *
                           "missing or its package failed to load")
    return p
end

"""
# Segmenters

`f(frame, points; key) -> Matrix{UInt8}` receives one source-resolution frame,
the marked `(x, y, foreground)` points in normalized coordinates, and a value
identifying the frame, and returns a 0/255 mask the size of the frame. `key` is
what lets an implementation cache per-frame work across the clicks of one
marking; it is `nothing` when the caller cannot identify the frame.

A segmenter is a plain function passed to [`seedmask`](@ref), and the editor's is
[`Player`](@ref)`.segmenter` — not a registry, and not a global switch. Pass a
different one to use a different model.

A separate seam from [`registermatte!`](@ref), because they answer different
questions. A propagator carries a mask through a clip; it does **not** find one —
measured on both this implementation and PyTorch's, MatAnyone's matte tracks its
seed (IoU against the seed 0.95 falling to 0.82 across a clip) rather than
segmenting the subject. So the quality of the whole matte is decided by the seed,
and a seed painted as discs around clicks is a rough one.
"""

"Whether SAM 2's exported graphs and weights are on disk."
sam2ready() = isfile(joinpath(SAM2Runner.assetdir(), "weights.safetensors"))

# The ONE piece of genuinely global state here: the built-in model itself. It is
# ~5 GB of VRAM and a Vulkan context — one per process, whatever else is going
# on — so it is a singleton because the GPU is, not for convenience.
#
# Built on FIRST USE and never at load: a `BatchQueue` is single-writer and
# belongs to whichever thread first touches the context, and the editor calls a
# segmenter from `runanalysis` — its pinned GPU worker. Constructing it at load
# would bind the queue to whoever loaded the package, and every later click would
# die on "BatchQueue is single-writer".
const SAM2MODEL = Ref{Any}(nothing)

"""
    sam2seed(frame, points; key) -> Matrix{UInt8}

The built-in segmenter: SAM 2.1 on Lava, turning clicks into an object boundary.

`pick = :confident` takes the highest predicted IoU but breaks a near-tie by
logit magnitude, which on measured clicks cuts the seed's boundary fragmentation
from 13.6× a compact blob to 3.6×; SAM's own argmax (`:best`) is what PyTorch
does. A click is genuinely ambiguous — a windowpane, the window, the wall — and
the model says so by returning three proposals.
"""
function sam2seed(frame, points; key = nothing)
    if SAM2MODEL[] === nothing
        # `replaydecode = false`: the decoder's captured sequence does not survive
        # a garbage collection, and marking a matte is precisely the pattern that
        # provokes one — a click, host work to show the result, another click. The
        # editor was losing the device on the second mark. Recording each decode
        # fresh costs about 6% of a click and is the only version that runs.
        model = SAM2Runner.sam2model(; backend = Lava.LavaBackend(), replaydecode = false)
        SAM2MODEL[] = SAM2Runner.sam2segmenter(model; pick = :confident)
    end
    return SAM2MODEL[](frame, points; key)
end

"What a new [`Player`](@ref) segments with. SAM 2.1, always — the editor depends
on it, and a missing model is a loud failure, not a quieter mode."
defaultsegmenter() = sam2seed

"""
Whether `seg` can answer a click straight away, or the caller is about to wait
for a model to be built. Only the built-in one is knowable — anything else is
assumed ready, since a custom segmenter's laziness is its own business.
"""
segmenterready(seg) = seg !== sam2seed || SAM2MODEL[] !== nothing

"""
    briefly(e) -> String

One line of an exception, for a status bar.

A Vulkan out-of-memory report is twenty lines of per-heap budgets. That belongs
in the log; in a footer it buries the one sentence the user can act on, and a
status nobody reads is the same as no status at all — which is how "it failed"
became "nothing happens".
"""
function briefly(e)
    s = sprint(showerror, e)
    occursin("Out of GPU memory", s) &&
        return "out of GPU memory — another process is likely holding the card"
    occursin("BatchQueue is single-writer", s) &&
        return "the GPU model was built on the wrong thread (restart the editor)"
    return first(split(s, '\n'))
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
    n = srclength(clip)
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
    prop = matteprop()
    alpha = propagateboth(prop, frames, localseeds, mw, mh, n, progress)
    track = MatteTrack(alpha, clip.src_in, sort!(collect(keys(seeds))))
    clip.mattetrack = track
    progress === nothing || progress(2n, 2n)
    return track
end

"""
    propagateboth(prop, frames, seeds, mw, mh, n, progress) -> alpha

Propagate a seed over the WHOLE clip: forward from it, and backward over what
comes before.

The propagator is causal — it carries a memory forward, one frame at a time — so
handed a clip whose seed sits in the middle it can only answer for the frames
after it, and every earlier frame comes back empty. Empty alpha is not "no
matte", it is a matte that keeps NOTHING: the first half of the clip goes black,
which is exactly what it looks like when somebody marks a subject halfway in and
then cuts at the mark.

So the prefix is propagated as its own sequence, reversed — same model, same
seed, running backwards in time — and the two halves are stitched at the seed.
"""
function propagateboth(prop, frames::Vector{<:AbstractMatrix}, seeds::Dict{Int, <:AbstractMatrix},
                       mw::Integer, mh::Integer, n::Integer, progress)
    k = minimum(keys(seeds))
    alpha = Array{UInt8}(undef, mw, mh, n)
    # total work for the progress bar: the tail plus the prefix, once each
    tail, head = n - k + 1, k
    done = Ref(0)
    step = (d, t) -> progress === nothing ? nothing : progress(n + done[] + d, 2n)
    if k > 1
        back = prop(frames[k:-1:1], Dict(k - sf + 1 => m for (sf, m) in seeds if sf <= k);
                    progress = step)
        size(back) == (mw, mh, head) ||
            error("matte propagator returned $(size(back)), expected $((mw, mh, head))")
        for j in 1:head
            @inbounds alpha[:, :, j] = @view back[:, :, head - j + 1]
        end
        done[] = head
    end
    fwd = prop(frames[k:end], Dict(sf - k + 1 => m for (sf, m) in seeds if sf >= k);
               progress = step)
    size(fwd) == (mw, mh, tail) ||
        error("matte propagator returned $(size(fwd)), expected $((mw, mh, tail))")
    @inbounds alpha[:, :, k:n] = fwd
    return alpha
end

"""
    mattecoverage(track) -> Float64

Fraction of the clip's pixels the matte keeps, sampled across its frames.

Reported to the user because both ends of the range look identical on screen —
"keeps everything" and "keeps nothing" are equally a picture with no visible
change — and the difference between a working model and a stand-in is exactly
this number.
"""
function mattecoverage(track::MatteTrack; samples::Integer = 12)
    a = track.alpha
    n = size(a, 3)
    n == 0 && return 0.0
    ks = unique(round.(Int, range(1, n; length = min(samples, n))))
    return sum(count(>(0x7f), view(a, :, :, k)) for k in ks) /
           (length(ks) * size(a, 1) * size(a, 2))
end

"""
    previewmatte(clip, frame, mask; mattewidth = 480) -> Matrix{UInt8}

The matte for ONE frame, at matte resolution — what the current selection would
produce right here, without touching the rest of the clip.

The same propagator as [`analyzematte!`](@ref), called with a one-element clip:
a live preview must not be a second, differently-behaving implementation of what
propagation does, or the picture that talks the user into stopping is not the
picture they get. It is also why the seeded frame had to start returning a
segmented matte instead of the seed — a single-frame call was otherwise a
very expensive way to hand the box back.

Cost is one seeded frame's worth of model work (~0.2 s on the GPU tier), which is
what makes it usable between clicks.
"""
function previewmatte(clip::Clip, frame::AbstractMatrix{<:RGB}, mask::AbstractMatrix;
                      mattewidth::Integer = 480)
    sw, sh = size(frame)
    mw = min(Int(mattewidth), sw)
    mh = max(1, round(Int, sh * mw / sw))
    prop = matteprop()
    alpha = prop([downscale(frame, mw, mh)], Dict(1 => mattemaskscale(mask, mw, mh)))
    size(alpha) == (mw, mh, 1) ||
        error("matte propagator returned $(size(alpha)), expected $((mw, mh, 1))")
    return alpha[:, :, 1]
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
    seedmask(clip, frame, points; segmenter = defaultsegmenter(), radius = 0.06, key = nothing)

The selection for one marked frame: `segmenter`'s answer — SAM 2.1 unless the
caller passes another. There is no no-segmenter case; the disc form is the
separate two-argument method, asked for by name.

Taking the frame is what makes the segmenter possible at all — the disc form
needs only coordinates, which is why it never asked for a picture. Callers read
the frame on the analysis executor anyway (it is the same frame the preview
mattes), so this costs nothing extra.

`key` identifies the frame so a segmenter can cache whatever it derives from it.
That is not an optimisation detail for SAM 2: embedding a frame costs 0.6 s and
answering a click against a cached embedding costs 0.02 s, and marking a subject
means a dozen clicks on the *same* frame. Without the key every click pays the
embedding again and the live preview stops being live.
"""
function seedmask(clip::Clip, frame::AbstractMatrix{<:RGB},
                  points::AbstractVector{<:Tuple{<:Real, <:Real, Bool}};
                  segmenter = defaultsegmenter(), radius::Real = 0.06, key = nothing)
    segmenter === nothing &&
        error("seedmask needs a segmenter; call seedmask(clip, points) for discs")
    m = segmenter(frame, points; key)
    size(m) == size(frame) ||
        error("segmenter returned $(size(m)), expected $(size(frame))")
    return m
end

"""
    seedmask(clip, frame, groups; segmenter, key) -> Matrix{UInt8}

Several objects in one selection: each group is one object's points, gets its own
segmenter call, and the masks are unioned.

Grouping is the CALLER's, because it is the user's. SAM 2 reads several positive
points as several hints about ONE object — that is what makes refinement work —
so nothing in a flat list of clicks says whether two of them mean "also this bird"
or "no, more like this". Only the person clicking knows, so they say it (the
matte card's `+ object`), and each group is then an ordinary single-object call.
"""
function seedmask(clip::Clip, frame::AbstractMatrix{<:RGB},
                  groups::AbstractVector{<:AbstractVector{<:Tuple{<:Real, <:Real, Bool}}};
                  segmenter = defaultsegmenter(), radius::Real = 0.06, key = nothing)
    return first(seedmasks(clip, frame, groups; segmenter, radius, key))
end

"""
    seedmasks(clip, frame, groups; …) -> (union, per_object)

As [`seedmask`](@ref), and also every object's own mask — what the SAM 2 view
outlines, each in its object's colour.
"""
function seedmasks(clip::Clip, frame::AbstractMatrix{<:RGB},
                   groups::AbstractVector{<:AbstractVector{<:Tuple{<:Real, <:Real, Bool}}};
                   segmenter = defaultsegmenter(), radius::Real = 0.06, key = nothing)
    isempty(groups) && error("matting needs at least one marked point")
    per = [seedmask(clip, frame, g; segmenter, radius, key) for g in groups if !isempty(g)]
    isempty(per) && error("matting needs at least one marked point")
    out = copy(per[1])
    for m in per[2:end]
        out .= max.(out, m)
    end
    return (out, per)
end

"""
    seedmask(clip, points; radius = 0.06) -> Matrix{UInt8}

A selection built from marked points: `(x, y, foreground)` in normalized source
coordinates, painted as discs of `radius` (a fraction of frame width) in the
order given.

Order is the whole point of taking a list rather than a set. A background point
placed after a foreground one erases where they overlap, so "not that bit" is a
click rather than a restart — which is the correction users actually reach for
when one disc swallows an arm or the floor next to it.
"""
function seedmask(clip::Clip, points::AbstractVector{<:Tuple{<:Real, <:Real, Bool}};
                  radius::Real = 0.06)   # of frame width: points are placed several at a
                                       # time, so a disc large enough to hit the
                                       # subject from ONE click would swallow the
                                       # background between them
    sw, sh = clip.source.width, clip.source.height
    m = zeros(UInt8, sw, sh)
    r = max(1, round(Int, radius * sw))
    r2 = r * r
    for (nx, ny, fg) in points
        cx = round(Int, clamp(nx, 0.0, 1.0) * sw)
        cy = round(Int, clamp(ny, 0.0, 1.0) * sh)
        v = fg ? 0xff : 0x00
        @inbounds for j in max(1, cy - r):min(sh, cy + r), i in max(1, cx - r):min(sw, cx + r)
            (i - cx)^2 + (j - cy)^2 <= r2 && (m[i, j] = v)
        end
    end
    return m
end



"""
    framereader(clip, engine) -> (srcframe -> RGBFrame)

Host RGB frames for analysis, one decoder reused across the clip.

Deliberately the CPU `SequentialReader` rather than the GPU stream: matting reads
every frame of the clip once, in order, which is exactly what a sequential
decoder is good at, and it keeps the analysis off the single-writer Vulkan queue
that the preview is using to stay responsive.
"""
function framereader(clip::Clip, engine::FxEngine)
    # Through the effect graph, not around it: the models get the frame the user
    # sees — stabilised, colour-corrected, cropped — so click, mask and pixels
    # share one space.
    readers = Dict{String, Any}()
    base    = withoutmatte(clip)
    w, h    = mattelayersize(clip)      # per clip, not the sequence canvas
    # The render target lives on the ENGINE's backend — rendering into a host
    # Matrix on a GPU engine hands the warp kernel a non-bitstype argument. The
    # models want host frames, so copy back; on a CPU backend that copy is a
    # plain one and needs no branch to say so.
    dev  = KA.allocate(engine.backend, RGB{N0f8}, w, h)
    host = RGBFrame(undef, w, h)
    return sf -> (rendercanvas!(dev, base, Int(sf), readers, engine);
                  copyto!(host, dev); host)
end
