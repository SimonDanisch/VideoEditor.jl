"""
Subject matting: isolate a subject and key the background out, driven by frames
the user marks.

The shape mirrors stabilization rather than the effect stack, and for the same
reason. Producing a matte is an *analysis* — expensive, sequential, and it wants
the whole clip — so it runs once into a [`MatteTrack`](@ref) and the render path
only ever samples that track. Nothing per-frame runs a model; there is no
per-clip model state to keep alive across scrubs, transitions and export, and the
GPU and CPU tiers stay one code path because the track reaches the kernel the one
way every per-frame plane does — as a [`PlaneOp`](@ref), through the graph.

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

`feather = 1` reaches ±1/120 of the crop's width — about 32 pixels of transition
band on a 1920-wide render, whatever resolution the matte was analyzed at.
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
        # 5×5 binomial taps. At feather 0 the spacing is 0, every tap lands on
        # the same place and this collapses to the single sample below:
        # continuous, and it can only ever soften.
        #
        # The reach is a FRACTION OF THE PICTURE, not a texel count. It used to
        # be `feather * 2` texels of spacing — ±4 texels at full feather — which
        # made the control mean whatever the analysis happened to be sized at:
        # `mattereadsize` no longer caps the plane at 480 wide, so the same
        # slider softened a 34-pixel band on the plane it was tuned against and
        # a 4-pixel band on a 4K source. Dividing by 240 keeps the tuned value
        # (mw/240 = 2 at mw = 480) and makes full feather reach ±mw/120, i.e.
        # 1/120 of the crop's width, at every resolution.
        #
        # Five taps over a widening reach means the taps themselves spread
        # apart on a fine plane — the softening is a scaled copy of the tuned
        # kernel, not a better-sampled one.
        s = feather * Float32(mw) / 240.0f0
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

# ── the matte as a plane op (see `PlaneOp` in gpugraph.jl) ────────────────────
#
# The track holds every frame's alpha on the host — it is the thing that gets
# saved and edited — and the kernel needs the current frame's plane on whatever
# device is rendering. That upload is now the graph's, through the one route
# every per-frame plane takes; it used to be a `KA.allocate` per frame behind a
# module global keyed by `(track, backend)`, outside the pool and never freed.

planeeltype(::MatteOp) = UInt8
passname(::MatteOp) = "matte"

"""
The object whose bytes a matte plane holds: the track. Compared by identity, so a
re-propagation — which is always a NEW `MatteTrack` under an unchanged clip and
frame — writes the plane again. This is what replaced eight `freematteplanes!`
calls scattered through the matte tools, whose job was to drop a device cache
that a new track had made stale.
"""
planesource(::MatteOp, clip::Clip, ::Integer) = clip.mattetrack

"The analysed matte's resolution — usually smaller than the source, and sampled
bilinearly when applied. `nothing` when the clip has no matte."
planeshape(::MatteOp, clip::Clip) =
    clip.mattetrack === nothing ? nothing : mattesize(clip.mattetrack)

function planedata(::MatteOp, clip::Clip, srcframe::Integer)
    t = clip.mattetrack
    t === nothing && return nothing
    i = Int(srcframe) - t.src_in + 1
    1 <= i <= size(t.alpha, 3) || return nothing
    # One frame of a `(w, h, n)` array is contiguous, so this is a view into the
    # track and not a copy of it.
    n = size(t.alpha, 1) * size(t.alpha, 2)
    return view(reshape(t.alpha, :), ((i - 1) * n + 1):(i * n))
end

"""
    applyplane!(buf, plane, op::MatteOp, clip)

Key `buf` against the matte plane, in place: the subject survives, the background
goes to `bg`.
"""
function applyplane!(buf::AnyRGBFrame, plane, op::MatteOp, clip::Clip;
                     bg::Vec3f = Vec3f(0, 0, 0))
    s = clamp(op.strength, 0.0f0, 1.0f0)
    s <= 0.0f0 && return buf
    backend = KA.get_backend(buf)
    cr = clip.crop
    matte_kernel!(backend)(buf, plane, Int32(size(plane, 1)), Int32(size(plane, 2)), s,
                           clamp(op.feather, 0.0f0, 1.0f0), bg,
                           Float32(cr[1]), Float32(cr[2]), Float32(cr[3]), Float32(cr[4]);
                           ndrange = size(buf))
    return buf
end

"""
    mattealpha!(dst, plane, op, clip) -> dst

The matte as a COVERAGE image the size of `dst` (white = keep, black = show what
is underneath).

This is what makes a matte transparent instead of black. Keying paints the
removed background with `bg`, which is right when nothing is underneath and
wrong the moment there is: the black is opaque and covers the track below.
This and [`applyplane!`](@ref) share [`mattealphaat`](@ref) *and now the plane
itself* — the compositor is handed the buffer the keying read, so the coverage it
honours cannot be a frame off the coverage that was keyed.
"""
function mattealpha!(dst::AnyRGBFrame, plane, op::MatteOp, clip::Clip)
    backend = KA.get_backend(dst)
    cr = clip.crop
    mattealpha_kernel!(backend)(dst, plane,
                                Int32(size(plane, 1)), Int32(size(plane, 2)),
                                clamp(op.strength, 0.0f0, 1.0f0),
                                clamp(op.feather, 0.0f0, 1.0f0),
                                Float32(cr[1]), Float32(cr[2]), Float32(cr[3]), Float32(cr[4]);
                                ndrange = size(dst))
    return dst
end

"""
    applymatte!(buf, clip, srcframe; strength, feather, bg)

The host-side form: key a HOST buffer straight from the clip's track. A no-op
when the clip has no matte or the frame is outside it, so scrubbing past the
analyzed range shows the plain frame rather than a hole.

In the render path the plane is a graph resource and the node calls
[`applyplane!`](@ref); this exists so a tool or a test can key one frame without
building a graph for it.
"""
function applymatte!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer;
                     strength::Real = 1.0, feather::Real = 0.0,
                     bg::Vec3f = Vec3f(0, 0, 0))
    op = MatteOp(Float32(clamp(strength, 0.0, 1.0)), Float32(clamp(feather, 0.0, 1.0)))
    d = planedata(op, clip, srcframe)
    d === nothing && return buf
    return applyplane!(buf, reshape(d, planeshape(op, clip)), op, clip; bg)
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

`propagator` defaults to the installed one and is a kwarg for the same reason the
segmenter is: `src/precompile.jl` has to drive this function to trace it, and
`registermatte!`ing a built model during precompilation would serialise that
model's device buffers — and the context they belong to — into the package image.
"""
function analyzematte!(clip::Clip, readframe, seeds::Dict{Int, <:AbstractMatrix};
                       maxside::Union{Nothing,Integer} = nothing, progress = nothing,
                       propagator = matteprop())
    n = srclength(clip)
    n > 0 || error("cannot matte an empty clip")
    isempty(seeds) && error("matting needs at least one marked frame")
    mw, mh = mattereadsize(clip, maxside)
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
    alpha = propagateboth(propagator, frames, localseeds, mw, mh, n, progress)
    track = MatteTrack(alpha, clip.src_in, sort!(collect(keys(seeds))))
    clip.mattetrack = track
    progress === nothing || progress(1000, 1000)
    return track
end

"""
Progress reporting into the propagator, as ONE type.

The propagator takes a `progress` callback, and Julia specializes its whole body
on that callback's type — so a closure over the caller's UI state made every
caller its own specialization, and `src/precompile.jl` could only ever trace one
of them. That is what left ~54 s of inference on the *first* propagate tick, with
the bar frozen at 0 while it ran.

`report` is `Any` on purpose. One `MatteProgress` type means one specialization
of the propagator; the dynamic call it costs is once per frame, against ~100 ms
of model work. `done` is the frames already finished by the other half (see
[`propagateboth`](@ref)), so the two halves report one continuous scale.
"""
struct MatteProgress
    report::Any                 # (done::Int, total::Int) -> anything, or `nothing`
    done::Base.RefValue{Int}
    total::Int
end
function (p::MatteProgress)(d, t)
    p.report === nothing && return nothing
    frac = clamp((p.done[] + d) / max(p.total, 1), 0.0, 1.0)
    p.report(round(Int, 1000 * frac), 1000)
    return nothing
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
                       mw::Integer, mh::Integer, n::Integer, progress)
    k = minimum(keys(seeds))
    alpha = mattebuffer(mw, mh, n)
    # total work: the tail plus the prefix, once each. That is `n + 1`, not `n`
    # — the seed frame is propagated by both halves — and reporting it against
    # `n` used to walk the bar slightly past its own end on a mid-clip seed.
    tail, head = n - k + 1, k
    # ONE type, whatever the caller's callback is — see [`MatteProgress`](@ref).
    # Passing `nothing` here instead would be a second specialization again.
    step = MatteProgress(progress, Ref(0), head + tail)
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
        step.done[] = head
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
    repairframe!(clip, srcframe, mask) -> Bool

Overwrite ONE frame of an existing matte, leaving every other frame alone.

The propagator is causal, so the ordinary way to fix a bad frame — mark it and
re-analyse — rebuilds the whole clip from its seeds and takes as long as the
first run did. That is the right tool when the tracking went wrong and everything
after it drifted. It is the wrong one for a single frame that came out broken in
an otherwise good matte, which is the common case and the one with no answer
before this.

`mask` is at any resolution; it is scaled to the track's. Returns `false` when
the clip has no matte, or when `srcframe` is outside it — a repair has nothing to
repair in either case.

**Not recorded as a seed.** A seed is an INPUT the propagator runs from; this is
an OUTPUT written after the fact. Adding it to `track.seeds` would mean the next
full analysis propagates from a frame the user painted, which is a different and
much stronger claim than "this one frame should look like this". See
[`repairedframes`](@ref) for what does remember them.
"""
function repairframe!(clip::Clip, srcframe::Integer, mask::AbstractMatrix)
    t = clip.mattetrack
    t === nothing && return false
    k = Int(srcframe) - t.src_in + 1
    1 <= k <= size(t.alpha, 3) || return false
    w, h = mattesize(t)
    view(t.alpha, :, :, k) .= mattemaskscale(mask, w, h)
    return true
end

"""
    brushmatte!(mask, nx, ny, foreground; radius = 0.04) -> mask

Stamp one round brush dab into `mask` at normalized `(nx, ny)`, in place.

`foreground` writes 0xff (keep this), otherwise 0x00 (drop this) — the two things
a matte can say about a pixel, so an eraser is the same gesture with the other
button rather than a second tool.

`radius` is a fraction of the mask's WIDTH, so the brush is the same size on
screen whatever resolution the matte was analysed at. Painting is the repair of
last resort: the model gets the subject nearly right and leaves a hole, or takes a
bite out of an edge, and no arrangement of clicks talks it out of that — at which
point saying "this bit, here" directly is the shortest path from wrong to right.
"""
function brushmatte!(mask::AbstractMatrix{UInt8}, nx::Real, ny::Real, foreground::Bool;
                     radius::Real = 0.04)
    w, h = size(mask, 1), size(mask, 2)
    cx, cy = nx * w, ny * h
    r = max(1.0, radius * w)
    r2 = r * r
    v = foreground ? 0xff : 0x00
    x0 = max(1, floor(Int, cx - r));  x1 = min(w, ceil(Int, cx + r))
    y0 = max(1, floor(Int, cy - r));  y1 = min(h, ceil(Int, cy + r))
    @inbounds for y in y0:y1, x in x0:x1
        dx = x - cx; dy = y - cy
        dx * dx + dy * dy <= r2 && (mask[x, y] = v)
    end
    return mask
end

"""
    matteframe(clip, srcframe) -> Matrix{UInt8} | nothing

A COPY of one frame of the clip's matte, at the track's own resolution.

A copy because it is what a brush stroke paints into: strokes are committed
through [`repairframe!`](@ref) on release, so the track must not change under a
stroke the user may still abandon.
"""
function matteframe(clip::Clip, srcframe::Integer)
    t = clip.mattetrack
    t === nothing && return nothing
    k = Int(srcframe) - t.src_in + 1
    1 <= k <= size(t.alpha, 3) || return nothing
    return Matrix{UInt8}(view(t.alpha, :, :, k))
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

"""
    mattebytes(clip, maxside = nothing) -> Int

How big this clip's alpha will be — `width * height * srclength` bytes.

Reported rather than enforced. It grows with the SHOT: 3.5 GB a minute at 1080p,
17 GB for five. Worth showing on the button that spends it, which is why the
matte card prints it past a gigabyte.
"""
mattebytes(clip::Clip, maxside::Union{Nothing,Integer} = nothing) =
    (sz = mattereadsize(clip, maxside); sz[1] * sz[2] * max(srclength(clip), 0))

"""
Above this, a matte's alpha is FILE-BACKED rather than resident.

256 MB: below it the mapping's own cost is not worth paying, above it the
resident footprint is what stops a long shot being mattable at all.
"""
const MATTEINRAM = 256 * 2^20

"""
Where this process keeps its file-backed matte buffers.

**Not `mktempdir()`.** That follows `TMPDIR`, and `/tmp` here is a tmpfs — RAM
with a path. Backing an "avoid holding gigabytes resident" buffer with RAM is
exactly backwards, and writing a 4 GB alpha into it filled the filesystem and
took the shell down with it. The Julia depot's scratch space is on real disk,
which is the only property this needs.

A per-process subdirectory, and NOT one `prunecache!` manages: that one deletes
least-recently-used files to stay under a budget, and pulling the file out from
under a LIVE mapping is not a risk worth taking. `atexit` removes it; a crash
leaves one behind, which `clearmattescratch!` sweeps on the next start.
"""
const MATTESCRATCH = Ref{String}("")

function mattescratch()
    if isempty(MATTESCRATCH[])
        dir = joinpath(cachedir("matte_alpha"), string(getpid()))
        isdir(dir) || mkpath(dir)
        MATTESCRATCH[] = dir
        atexit(() -> rm(dir; recursive = true, force = true))
    end
    return MATTESCRATCH[]
end

"""
    clearmattescratch!() -> Int

Delete matte buffers left by processes that are no longer running, returning how
many bytes went.

A mapping's backing file is only garbage once its process is gone, so the sweep
is keyed on the PID in the directory name rather than on age — an
age-based prune would eventually delete the file under a long-running session's
own live matte.
"""
function clearmattescratch!()
    root = cachedir("matte_alpha")
    freed = 0
    for name in readdir(root; join = true)
        isdir(name) || continue
        pid = tryparse(Int, basename(name))
        (pid === nothing || pid == getpid()) && continue
        # `/proc/<pid>` rather than a signal probe: a plain directory test, so
        # there is nothing to throw and nothing to swallow.
        isdir("/proc/$pid") && continue
        for f in readdir(name; join = true)
            freed += filesize(f)
        end
        rm(name; recursive = true, force = true)
    end
    return freed
end

"""
    mattebuffer(mw, mh, n) -> Array{UInt8, 3}

Storage for an alpha: a plain array when it is small, a file-backed one when it
is not.

Holding the whole clip resident was a STORAGE decision, never a requirement. The
propagator writes forward, once; the renderer reads one frame. Nothing needs all
of it at the same time, and pretending otherwise is what made a five-minute 1080p
shot (17 GB) unmattable on a 31 GB machine.

`Mmap.mmap` hands back an `Array{UInt8, 3}` — precisely the type `MatteTrack`
already declares — so no reader, no kernel and no save path changes. What changes
is that the resident set becomes the pages actually touched, and the kernel
reclaims them under pressure instead of the process dying.
"""
function mattebuffer(mw::Integer, mh::Integer, n::Integer)
    dims = (Int(mw), Int(mh), Int(n))
    prod(dims) <= MATTEINRAM && return Array{UInt8}(undef, dims)
    io = open(joinpath(mattescratch(), string(hash(dims), "-", time_ns(), ".alpha")), "w+")
    a = Mmap.mmap(io, Array{UInt8, 3}, dims)
    close(io)          # the mapping outlives the handle
    return a
end

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
