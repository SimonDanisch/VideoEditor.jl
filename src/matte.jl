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
        # 5×5 binomial taps, spaced up to 2 plane texels apart, so full feather
        # reaches ±4 texels. At feather 0 the spacing is 0, every tap lands on the
        # same place and this collapses to the single sample below: continuous,
        # and it can only ever soften.
        #
        # NOTE the reach is in PLANE texels, so it scales with the matte's
        # resolution — and the matte is now the clip's full cropped source
        # (`mattereadsize` defaults to no cap), not the 480-wide plane this was
        # tuned against. Full feather used to be ±9 source pixels (4 texels ×
        # 1080/480); it is ±4 now, and less of the picture still on a 4K source.
        # The control's meaning should not depend on the analysis resolution.
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

const MATTEWARMED = Ref(false)

"""
    warmmatte!(w, h) -> Bool

Run the propagator once on a tiny synthetic clip, to pay its first-call cost
somewhere the user is not waiting.

The first propagation of a session costs ~99 s on the MatAnyone propagator, and
**none of it is shader compilation** — the SPIR-V disk cache is fully hit (683
entries in, zero added). It is Julia specializing the graph's execution paths.

Most of that is size-independent, but not all: warming at 64x48 absorbed 97.9 s
and still left 12.5 s on the first real 240x136 clip, because the
cooperative-matrix GEMM specializes per tile shape (`Val{BLK}`). So warm at the
size that will actually be used — `w`/`h` should be `mattereadsize` of the clip
the tool is about to matte, not a token. The defaults are for the caller that
has no clip yet and can only choose to load the model at all.

Returns whether it ran (false when no propagator is installed, or it already has).
"""
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
receives the clip's frames at matte resolution and `seeds::Dict{Int,
Matrix{UInt8}}` (index into `frames` → the user's rough selection, 0/255), and
returns `(w, h, nframes)` alpha.

`frames` is an **`AbstractVector{Matrix{RGB{N0f8}}}`, not a `Vector`** — what
arrives is a `view` of a [`MatteFrames`](@ref), which decodes on index and keeps
nothing. So a runner must use it as a sequence (`length`, `size(frames[1])`,
`frames[k]`) and must NOT `collect` it: that materialises the whole clip and puts
back the 763 MB, plus the two-phase progress bar, that streaming removed. Index
FORWARD — the readers are sequential decoders and random access costs a seek.

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
The clip's frames at matte resolution, decoded when asked for and not before.

**Nothing here caches a clip.** The propagator says what it does per frame and
this hands it one; indexing decodes that frame and keeps nothing, so a
propagation costs one frame of memory rather than all of them. It used to
build the whole `Vector` up front, which on the birds clip (869 frames at
480x610) was 763 MB held for the three minutes the model ran, and split the job
into a read phase and a propagate phase that no progress bar could weight
honestly.

`AbstractVector` rather than an iterator because that is the propagator's
contract — `length`, `size(frames[1])`, `frames[k]` — so a streaming source
drops in without every model runner learning a new protocol.

Indexing is expected to walk FORWARD: the readers are sequential decoders and
random access costs a seek. [`propagateboth`](@ref) is arranged around that.
"""
struct MatteFrames{F} <: AbstractVector{Matrix{RGB{N0f8}}}
    readframe::F     # srcframe -> frame at SOURCE resolution
    src_in::Int      # the source frame index 1 maps to
    n::Int
    mw::Int
    mh::Int
end
Base.size(fs::MatteFrames) = (fs.n,)
Base.IndexStyle(::Type{<:MatteFrames}) = IndexLinear()
function Base.getindex(fs::MatteFrames, i::Int)
    @boundscheck 1 <= i <= fs.n || throw(BoundsError(fs, i))
    f = fs.readframe(fs.src_in + i - 1)
    # A reader asked for the matte's width resized ON THE DEVICE before the
    # download, so there is normally nothing left to do — but `readframe` is the
    # caller's, and one that hands back the full layer still has to work. Then
    # `downscale` is the fallback, not the plan.
    #
    # Either way the result is a FRESH matrix: `framereader` reuses one host
    # buffer, so handing that straight to a consumer that keeps a frame across
    # iterations would alias. `copy` is the price of the contract.
    size(f) == (fs.mw, fs.mh) ? copy(f) : downscale(f, fs.mw, fs.mh)
end

"""
    analyzematte!(clip, readframe, seeds; maxside, progress) -> MatteTrack

Propagate `seeds` across the clip and store the result on it.

`readframe(srcframe) -> Matrix{RGB{N0f8}}` supplies source frames (the caller
decides decoder and tier); `seeds` maps an absolute source frame to a rough
selection at *source* resolution. The matte is computed at [`mattereadsize`](@ref)
and sampled back up when applied; `maxside` caps it and defaults to no cap, so by
default the matte is as fine as the layer it will be drawn into.

Pass the same `maxside` the reader was built with — [`mattereadsize`](@ref) is
the one definition of what it means, so agreeing costs nothing and disagreeing
costs a resize on every frame.

Frames stream (see [`MatteFrames`](@ref)); reading and propagating interleave, so
there is one phase and the progress bar needs no weighting between two.
"""
function analyzematte!(clip::Clip, readframe, seeds::Dict{Int, <:AbstractMatrix};
                       maxside::Union{Nothing,Integer} = nothing, progress = nothing)
    n = srclength(clip)
    n > 0 || error("cannot matte an empty clip")
    isempty(seeds) && error("matting needs at least one marked frame")
    mw, mh = mattereadsize(clip, maxside)
    report = progress === nothing ? nothing :
             frac -> progress(round(Int, 1000 * clamp(frac, 0.0, 1.0)), 1000)
    # Frames are FETCHED, not collected. Reading and propagating interleave, so
    # there is one phase to report and no share to tune between two — which is
    # what `decodeshare` existed for and why it is gone.
    frames = MatteFrames(readframe, clip.src_in, n, mw, mh)
    localseeds = Dict{Int, Matrix{UInt8}}()
    for (sf, m) in seeds
        k = sf - clip.src_in + 1
        1 <= k <= n || continue
        localseeds[k] = mattemaskscale(m, mw, mh)
    end
    isempty(localseeds) && error("no marked frame falls inside the clip")
    prop = matteprop()
    onstep = report === nothing ? nothing : (d, t) -> report(d / max(t, 1))
    alpha = propagateboth(prop, frames, localseeds, mw, mh, n, onstep)
    track = MatteTrack(alpha, clip.src_in, sort!(collect(keys(seeds))))
    clip.mattetrack = track
    report === nothing || report(1.0)
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
function propagateboth(prop, frames::AbstractVector, seeds::Dict{Int, <:AbstractMatrix},
                       mw::Integer, mh::Integer, n::Integer, onstep)
    k = minimum(keys(seeds))
    alpha = Array{UInt8}(undef, mw, mh, n)
    # total work: the tail plus the prefix, once each. That is `n + 1`, not `n`
    # — the seed frame is propagated by both halves — and reporting it against
    # `n` used to walk the bar slightly past its own end on a mid-clip seed.
    tail, head = n - k + 1, k
    total = head + tail
    done = Ref(0)
    step = onstep === nothing ? nothing : (d, t) -> onstep(done[] + d, total)
    if k > 1
        # The prefix is the one part that cannot stream: it is propagated
        # BACKWARDS and the readers are sequential decoders, so asking for
        # frames k, k-1, … 1 in that order is a seek per frame. Decode it
        # forwards — which they are good at — and hand the model a reversed
        # view. That buffers `k` frames, not `n`, and nothing at all when the
        # seed is on the first frame, which is the ordinary case.
        pre = [frames[j] for j in 1:k]
        back = prop(view(pre, k:-1:1), Dict(k - sf + 1 => m for (sf, m) in seeds if sf <= k);
                    progress = step)
        size(back) == (mw, mh, head) ||
            error("matte propagator returned $(size(back)), expected $((mw, mh, head))")
        for j in 1:head
            @inbounds alpha[:, :, j] = @view back[:, :, head - j + 1]
        end
        done[] = head
        empty!(pre)              # the tail does not need it, and it is `k` frames
    end
    # …and the tail streams: `k:n` is forward order, so each `frames[j]` is the
    # decoder's next frame and nothing is held but the one being propagated.
    fwd = prop(view(frames, k:n), Dict(sf - k + 1 => m for (sf, m) in seeds if sf >= k);
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
    previewmatte(clip, frame, mask; maxside = nothing) -> Matrix{UInt8}

The matte for ONE frame, at matte resolution — what the current selection would
produce right here, without touching the rest of the clip.

The same propagator as [`analyzematte!`](@ref), called with a one-element clip:
a live preview must not be a second, differently-behaving implementation of what
propagation does, or the picture that talks the user into stopping is not the
picture they get. It is also why the seeded frame had to start returning a
segmented matte instead of the seed — a single-frame call was otherwise a
very expensive way to hand the box back.

`maxside` therefore has to match what [`analyzematte!`](@ref) will be run with,
and defaults to the same no-cap: resolution is not a free dial on a causal
propagator, so previewing at 480 and propagating at the layer's 756 shows a matte
that tracks the subject differently from the one the user ends up with. That is
the second implementation this function exists to avoid, in a slower disguise.

Cost is NOT one frame of propagation throughput — measured on a 756x960 layer,
a warm call is 5.74 s against propagation's 0.44 s/frame, because a one-frame
call pays the propagator's per-sequence setup in full. Raising the resolution is
the small part of that: capping to a 480 short side gives 5.10 s warm, and the
cold first call is 52.8 s either way (it is loading MatAnyone, not sizing it).
The 5 s of setup, not the resolution, is what a faster live preview would have
to attack.
"""
function previewmatte(clip::Clip, frame::AbstractMatrix{<:RGB}, mask::AbstractMatrix;
                      maxside::Union{Nothing,Integer} = nothing)
    mw, mh = mattereadsize(frame, maxside)
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
    mattereadsize(clip, maxside) -> (w, h)

The resolution the matte is computed at. One definition: the frame reader has to
produce frames at it and [`analyzematte!`](@ref) has to agree, and computing it
twice would drift into a silent resize on every frame.

`maxside === nothing` — the default — is the clip's own layer resolution. The
matte is sampled back up to the layer when applied, so anything smaller is an
edge reconstructed from fewer samples than the layer can show; MatAnyone's own
entry points all default to `max_size = -1`, no limit, for the same reason.
Downscaling is also not a quality dial: the propagator is causal, so changing its
input resolution changes what it TRACKS. Measured on the birds clip, 640 wide
diverged into a matte with a 57 px transition band where 480 gave 35 px and the
layer's own 756 gave 25.6 px — not an ordering, a different answer.

A cap applies to the SHORT side, as upstream's does. Capping width instead made
one setting mean two resolutions: 480 wide is a 480 short side on a portrait clip
and 270 on a landscape one, a 1.8x swing in what the model sees from nothing but
how the camera was held.

Cost is linear in matte area — 607 ms/megapixel measured across four widths, flat
to within 3% once kernel compilation is excluded — so a cap buys time back
proportionally and predictably.
"""
mattereadsize(clip::Clip, maxside::Union{Nothing,Integer}) =
    mattereadsize(mattelayersize(clip), maxside)
mattereadsize(frame::AbstractMatrix, maxside::Union{Nothing,Integer}) =
    mattereadsize(size(frame), maxside)

function mattereadsize(layer::Tuple{Integer,Integer}, maxside::Union{Nothing,Integer})
    w, h = layer
    maxside === nothing && return (Int(w), Int(h))
    short = min(w, h)
    short <= maxside && return (Int(w), Int(h))
    scale = maxside / short
    return (max(1, round(Int, w * scale)), max(1, round(Int, h * scale)))
end

"""
    framereader(clip, engine; maxside) -> (srcframe -> RGBFrame)

Host RGB frames for analysis, one decoder reused across the clip.

Deliberately the CPU `SequentialReader` rather than the GPU stream: matting reads
every frame of the clip once, in order, which is exactly what a sequential
decoder is good at, and it keeps the analysis off the single-writer Vulkan queue
that the preview is using to stay responsive.

**The default is NO cap and nothing in the editor passes one**, so what the model
reads is the clip's whole cropped source — full resolution, no resample anywhere
on this path. That is deliberate: the matte's edge is the product, and a matte
reconstructed from fewer samples than the layer can show is a worse edge, not a
cheaper one. See [`mattereadsize`](@ref) for why downscaling is not a quality
dial at all here.

`maxside` therefore exists for one case that has not come up: a clip too slow to
matte at all. When it does bite it caps the short side and the resize happens
**on the device, before the download** — a host-side one would pull the full
layer across the bus and throw most of it away (756x960 is 2.18 MB a frame to
produce 878 KB, plus a host resize, 869 times). `areadownscale!` is the same
kernel the thumbnail worker downloads through, for the same reason.
"""
function framereader(clip::Clip, engine::FxEngine;
                     maxside::Union{Nothing,Integer} = nothing)
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
    tw, th = mattereadsize(clip, maxside)
    if (tw, th) == (w, h)
        host = RGBFrame(undef, w, h)
        return sf -> (rendercanvas!(dev, base, Int(sf), readers, engine);
                      copyto!(host, dev); host)
    end
    # Resize BEFORE the download. `areadownscale!` runs on `small`'s backend, so
    # on a GPU engine only the small frame crosses the bus and no host-side
    # resize happens at all; on a CPU engine both buffers are host arrays and it
    # is the same area-average that `downscale` would have done, once.
    small = KA.allocate(engine.backend, RGB{N0f8}, tw, th)
    host  = RGBFrame(undef, tw, th)
    return sf -> (rendercanvas!(dev, base, Int(sf), readers, engine);
                  areadownscale!(small, dev); copyto!(host, small); host)
end
