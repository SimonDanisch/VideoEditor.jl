# A GPU effect graph: a clip's render — source → tracks → effect stack — as a
# Mantle graph, executed from a cached plan, so after warm-up there is no
# per-frame allocation and every intermediate is a transient the placer can
# alias. The whole thing is free functions and multiple dispatch.
#
# Two layers, so most effects are not kernels:
#   • the callback layer: a `Pointwise`/`Stencil` effect is a pure function mapped
#     by a framework-owned kernel (`GPUFiltering.pointwise!`/`stencil!`), and the
#     same function runs the CPU stack and the GPU graph. This is what plugins use.
#   • the node layer: an `FxNode` plus a `chainpass!`, for ops that own a
#     specialized kernel or several passes (decode, motion warp, separable blur,
#     blend). The escape hatch.
#
# Every node becomes one `custom!` pass, because the bodies are multi-launch or
# host-branching (a decode, a model call, a separable blur with a scratch buffer)
# and `dispatch!` expresses exactly one kernel. The pass declares what it touches
# (`use`); the graph derives lifetimes and barriers from that and stays out of
# the body.
#
# The engine caches one compiled plan per (node structure, frame size). What
# changes every frame — the source, the clip, the frame index, and every node
# parameter — flows through `Ref`s the pass bodies read at record time, so a
# keyframed σ costs a store, not a recompile.

using KernelAbstractions: @kernel, @index, @Const

# ---------------------------------------------------------------- per-frame inputs

"""
The decoded source frame's size, from whatever `render` was handed: a
`GpuVideoStream` for disk→VRAM decode, an export reader, or a CPU `RGBFrame`.
"""
framesize(s) = size(s)                                   # a CPU RGBFrame
framesize(s::GpuVideoStream) = (s.width, s.height)
framesize(s::ClipSource) = (s.width, s.height)           # a source that renders itself

"""
    resize!(source, wh) -> source

Change the size a source delivers, where that is a thing it has.

A scene renders at whatever it is asked for, so its size is a format: a document
property like a clip's rate. A decoder delivers the frames it has and this is a
no-op on it — the chain's buffers are sized by the decode, and asking for another
size would be a resample dressed as a render.

This is what the bake modal's canvas field edits, and the same number the preview
renders at, so a bake cannot land at a size the preview will not use.
"""
Base.resize!(s::ClipSource, ::Tuple{Integer, Integer}) = s

"""
    decodesource(source, frame; playing, served, exact) -> what the source pass reads

Get the frame before the plan runs. `frameat!` is latency-bounded: under a scrub
it serves the nearest already-decoded frame rather than `frame` and reports which
through `served`. Every per-frame result — a stabilization warp, a matte plane —
has to be sampled at that index, or it lands on a different picture and the
preview jerks while the decode catches up.

This used to be the first thing the source pass body did, so `served` was not
settled until the plan was already running: the decode's own submits landed in
the middle of the plan's recording and a plane upload had nowhere in the schedule
to go. Out here it is settled before `run!`.

`exact` is the export policy — precisely `frame`, cost what it may — and cannot
serve anything else, so it leaves `served` alone.
"""
decodesource(s::GpuVideoStream, frame::Integer; playing::Bool = false,
             served = nothing, exact::Bool = false, chunks::Integer = 5) =
    exact ? exactframeat!(s, frame) :
            frameat!(s, frame; prefetch = playing, served, chunks)
decodesource(s, frame::Integer; playing::Bool = false, served = nothing,
             exact::Bool = false, chunks::Integer = 5) = s

"Fill `out` with the frame `decodesource` handed back: a colour convert off the
decoder's NV12 planes, or one upload of a CPU frame."
sourceinto!(out, s::GpuVideoStream, f) = (nv12torgb!(out, f.y, f.uv; bt601 = s.bt601); out)
sourceinto!(out, s, f) = copyto!(out, f)

"""
What a pass body reads at record time, so the plan it belongs to can be replayed
for another frame — or another clip with the same structure — without
recompiling. `served` is a `Ref` because that is what [`frameat!`](@ref) writes
into; by the time any body runs it holds the frame the source really delivered.
"""
mutable struct FxState
    source::Any
    decoded::Any                      # what `decodesource` handed back
    # The frame after `decoded`, and how far between them this timeline frame
    # sits. Both are only set for a clip whose time interpolation is `:flow`;
    # `phase == 0` means this frame lands exactly on a source frame and nothing
    # has to be synthesized.
    decoded2::Any
    # A pre-rendered picture for this frame, or `nothing` — see bake.jl. When it
    # is there the source pass hands it straight on and every other pass is
    # inactive: the bake already has them applied. Per frame, because a bake
    # covers a range and the clip is scrubbed in and out of it.
    baked::Any
    phase::Float64
    clip::Any
    frame::Int
    served::Base.RefValue{Int}
    # How a host frame gets onto the device: the transient the source pass
    # stores into, and whether it was written this frame. A `copyto!` in the pass body
    # would do the same copy — but a host→device upload inside a recorded batch
    # forces a `vkQueueSubmit`, one per layer per frame, and the CPU-decode path
    # is the one the editor falls back to whenever the Vulkan decoder is not
    # available. `nothing` for a source pass that has no update (the retime
    # path), and unfired for a decoder that serves device planes.
    upload::Any
    uploaded::Bool
    # What gets stored when there is no picture. A transient keeps nothing
    # between runs, so a frame the source could not produce must still write
    # something or the chain reads whatever the arena last held — a flash of
    # another layer, not a black frame. Cached because allocating a full frame
    # of zeros per miss is the kind of per-frame allocation this path exists to
    # avoid.
    blank::Any
    # Whether a frame with no host picture stores black. True for a scene, whose
    # picture is the only writer; false for a video source, where a miss is the
    # NORMAL case — the decoder served device planes and the conversion pass
    # fills the buffer instead, so storing black first would be a full frame
    # written twice per frame.
    blankonmiss::Bool
    # The chain's NV12 destination, see `planebuffers!`.
    planes::Any
    # …and its decoded picture, see `rawbuffer!`.
    raw::Any
    # The two frames retiming interpolates between, see `synthscratch!`.
    synth::Any
    # The finished-output policy of the frame being rendered (see `update!`):
    # a scene reads it to decide between one progressive sample and the full
    # budget. Per frame, like `uploaded` — the pass body outlives any one frame.
    exact::Bool
end

# By NAME, because this grew four fields during the declared-graph migration and
# a positional list of sixteen is a bug waiting for the next one: the first
# version of it put `false` where `planes` goes and `nothing` where `exact`
# does, and that typechecked as far as the constructor.
FxState(; source = nothing, decoded = nothing, decoded2 = nothing, baked = nothing,
        phase = 0.0, clip = nothing, frame = 0, served = Ref(0), upload = nothing,
        uploaded = false, blank = nothing, blankonmiss = false, planes = nothing,
        raw = nothing, synth = nothing, exact = false) =
    FxState(source, decoded, decoded2, baked, phase, clip, frame, served, upload,
            uploaded, blank, blankonmiss, planes, raw, synth, exact)

"""
    blankpixels!(st, dims) -> Vector{PlanePixel}

A frame of nothing, allocated once per chain and handed back on every miss.

Transparent and not opaque black: the plane is premultiplied, so zero alpha with
zero colour is "no picture here", and the compositor leaves whatever is under it
alone. Opaque black would be a layer that covers the track below.
"""
function blankpixels!(st::FxState, dims::Tuple{Int, Int})
    b = st.blank
    b isa Vector{PlanePixel} && length(b) == prod(dims) && return b
    st.blank = zeros(PlanePixel, prod(dims))
    return st.blank
end

"""
    sourcepicture!(st, dims) -> host image | nothing

What this frame's source hands the graph as a host image, or `nothing` when there
is nothing to upload — device planes a kernel converts in the pass, or a frame
that is on the device already.

Dispatch is on the source, which is what knows: a decoder answers with the frame
it decoded (`hostframe`), a scene with its picture (`scenepicture!`, in
scenesource.jl). Asked once per frame, so both kinds of source reach the device
through the `Update` the pass reserved rather than a `copyto!` in a recorded body.
"""
sourcepicture!(st, dims::Tuple{Int, Int}; exact::Bool = false) =
    st.baked !== nothing ? st.baked : sourcepicture!(st, dims, st.source; exact)
sourcepicture!(st, ::Tuple{Int, Int}, source; exact::Bool = false) =
    hostframe(source, st.decoded)

"""
    hostframe(source, decoded) -> Union{Nothing, Matrix{RGB{N0f8}}}

The host picture a source handed over, or `nothing` when there is nothing to
upload: a decoder's device-resident NV12 planes (a kernel converts those in the
pass, where they already are) or a frame that is on the device already.

`Array{RGB{N0f8}, 2}` exactly — a view or another element type falls through to
the pass body's `copyto!`, which is correct but not free.
"""
hostframe(::Any, f::Array{RGB{N0f8}, 2}) = f
hostframe(::Any, ::Any) = nothing

"""
The 2-D view a kernel gets over a resource's 1-D storage — a transient's arena
slice or a persistent `Mantle.Buffer`'s region alike. `KA.get_backend` walks
`parent`, so the view resolves to the right backend on the GPU and on the host.
"""
frameview(x, dims) = reshape(Mantle.storage(x), dims)


"""
Fill and copy, as dispatches.

The host path spelled these `fill!` and `copyto!` and the graph cannot: a pass is
a kernel the walk reads its accesses off, and `copyto!` on two device arrays is
the ad-hoc form that submits its own work. One kernel each, declared like
anything else.
"""
@kernel function fillpixels_kernel!(dst, value)
    I = @index(Global, Cartesian)
    @inbounds dst[I] = value
end

@doc (@doc fillpixels_kernel!)
@kernel function copypixels_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds dst[I] = src[I]
end

# ---------------------------------------------------------------- effect callbacks

"""
How a per-pixel/neighborhood effect renders, as a pure callback the framework maps
with one kernel (on CPU and GPU alike). An effect opts in by defining [`fxkind`];
then it needs no node, no pass builder, and no CPU/GPU split.
"""
abstract type FxKind end
"`f(c::Vec3f, uv::Vec2f) -> Vec3f` per pixel (`uv ∈ [0,1]²`); may run in place."
struct Pointwise{F} <: FxKind
    f::F
end
"`f(sample, radius, uv) -> Vec3f` over an n×m neighborhood; `sample(di,dj) -> Vec3f`."
struct Stencil{F} <: FxKind
    f::F
    radius::Int
end

"""
    fxkind(e::FxOp) -> Union{Nothing, FxKind}

The render callback for a per-pixel/neighborhood effect. Define this (plus a struct
and [`isneutral`](@ref)) and the effect renders on the CPU stack AND the GPU graph —
no kernel required. Effects that own a specialized/multi-pass kernel skip this and
define [`nodefor`](@ref) + `chainpass!` instead.

`nothing` for an op that did not opt in — and a stack entry is not always a
picture: the loop finder is a slot on the clip so that it has a card and a place
for its reference list, and it draws nothing at all. Stated as a fallback rather
than left undefined: undefined raised a `MethodError` from inside the presenter,
so putting the tool on a clip stopped the preview.
"""
fxkind(::FxOp) = nothing

# run a kind into `out` from `inp` (device images)
applykind!(out, inp, k::Pointwise) = pointwise!(out, inp, k.f)
applykind!(out, inp, k::Stencil) = stencil!(out, inp, k.f, k.radius)
# a Pointwise op may overwrite its input; a Stencil reads neighbors so it cannot
needsfresh(::Pointwise) = false
needsfresh(::Stencil) = true

# `applykind!` is the ONLY one. There was an `applykindcpu!` beside it that called
# the identical kernels and differed only in buffer discipline (in-place with a
# caller's scratch, rather than the pool's `out`) — a second name for one
# operation, on a second traversal that nothing in `src/` called. Both deleted.

# built-in opacity is just a pointwise scale toward black — no dedicated kernel
fxkind(e::OpacityEffect) = (a = e.α; Pointwise((c, uv) -> c * a))

# ---------------------------------------------------------------- nodes

# `FxNode` and `FxGraph` are declared in clips.jl, where `Clip` has a field of
# one. The nodes themselves are here, with the passes they stand for.

struct SourceNode <: FxNode end                                      # the decoded frame

"""
The source pass for a clip whose time interpolation is `:flow`: the frame is
synthesized between the two the decoder produced, at [`sourcephase`](@ref).

A node rather than an effect because it changes what the source is, before any
effect runs — the same category as `rate`, and for the same reason: an effect
sees one frame and this needs two.
""" 
struct SmoothSourceNode <: FxNode end
struct MotionNode <: FxNode; input::Int; end                         # stabilization warp
struct ColorTrackNode <: FxNode                                      # per-frame color stabilization
    input::Int
    strength::Float32
end
struct ColorNode <: FxNode; input::Int; adj::ColorAdjustments; end
"""
How many Gaussian taps a blur's weight buffer holds: `2·64 + 1`.

A cap on the radius at 64, so σ up to 21.3 against a declared slider maximum of
12. Fixed rather than derived from σ, because the count follows the value and a
buffer that resizes with a slider resizes the graph with it. 516 bytes.
"""
const BLURTAPS = 2 * 64 + 1

struct BlurNode <: FxNode; input::Int; σ::Float32; end
struct SharpenNode <: FxNode; input::Int; σ::Float32; amount::Float32; end

# What those two write into their weight edge. Always the full buffer, padded, so
# the write is one fixed-size staged copy whatever σ is doing this frame — and the
# kernel reads only the front of it, `weights[k + radius + 1]`.
planedata(n::Union{BlurNode, SharpenNode}, ::Clip, ::Integer) =
    first(gaussianweights(n.σ; maxradius = (BLURTAPS - 1) ÷ 2))
planeshape(::Union{BlurNode, SharpenNode}, ::Clip) = (BLURTAPS, 1)
struct PixelNode{K <: FxKind} <: FxNode; input::Int; kind::K; end    # a callback effect

# ---------------------------------------------------------------- plane ops
"""
What a node reads besides the picture, when what it reads is a whole image:
a matte's alpha, a restoration model's finished frame. The other kind of
per-frame analysis result — a colour gain, a warp matrix — is a handful of
numbers and rides along as a kernel argument, which is why those stay
ordinary nodes.

One type for all of them, because everything around a plane is the same: which
store slot holds it, when it has to be rewritten, that the rewrite is the
graph's business and not the kernel's, and that a frame the analysis does not
cover renders nothing rather than keying against whatever was there last. What
differs is four small methods — [`planeeltype`](@ref), [`planeshape`](@ref),
[`planedata`](@ref), [`applyplane!`](@ref) — defined next to the kernel that
needs them, in `matte.jl` and `restore.jl`.

Before this the two were separate nodes with a `KA.allocate` apiece behind a
module global, a per-frame device allocation outside the pool that nothing ever
freed, and the compositor did its own third lookup to find out whether the frame
it was placing had been keyed.
"""
abstract type PlaneOp end

"Key the subject out of the background against the clip's matte."
struct MatteOp <: PlaneOp
    strength::Float32
    feather::Float32
end

"Blend a restoration model's finished picture over the decoded one."
struct RestoreOp <: PlaneOp
    strength::Float32
end

"""
Defocus by distance from a focus plane, against the clip's estimated depth.

`focus` is the depth that stays sharp, on the 0..1 scale `DepthTrack` stores
(1 = nearest); `strength` scales the blur radius reached at the far end of that
distance. Between them is a smooth ramp — a hard threshold on a monocular depth
estimate flickers, because the estimate's scale drifts frame to frame and the
threshold lands somewhere different on each one.

Blur, deliberately, and not a key. Depth from one camera has soft, unreliable
edges exactly where a matte needs hard ones — and defocus is soft at its edges
anyway, so the model's weakness lands where a lens would put one. Anything
wanting a hard edge should mark a matte instead.
"""
struct DepthBlurOp <: PlaneOp
    focus::Float32
    strength::Float32
end

"""
The node a [`PlaneOp`](@ref) becomes. `shape` is the plane's size, carried here
because it is part of the plan signature: the plane is a graph resource, so a
clip whose matte was analysed at another resolution needs its own plan rather
than a buffer of the wrong size.
"""
struct PlaneNode{O <: PlaneOp} <: FxNode
    input::Int
    op::O
    shape::Tuple{Int, Int}
end

"""
    planeshape(node) -> (w, h) | nothing

The plane this node reads, for the plan signature. `nothing` for every node that
reads none, which is most of them.
"""
planeshape(::FxNode) = nothing
planeshape(n::PlaneNode) = n.shape


# The planes are graph resources: there is no `BufferStore` and no `PlaneSlot`.
#
# A per-frame analysis image — a matte's alpha, a depth map, a colour table — used
# to live in a persistent `Mantle.Buffer` beside the graph, because `Update` wrote
# a whole buffer by renaming it and a transient's arena slice cannot be renamed.
# Mantle stages into a transient now (`stagewrite!`), so a plane is an ordinary
# transient: the placer can alias it against anything already dead, its lifetime
# comes from its use like everything else, and nothing reaches around the graph to
# find it.

"""
A node's second input, as an edge of the graph.

A matte's alpha, a depth map, a colour table: the node that reads one declared it
with `use`, the same way it declared the picture, so the plane is ordered against
the write that fills it and there is nothing to reach around the graph for. What
this holds is the edge itself — the transient the placer gave it, the `Update`
that writes it at the head of the schedule, the node `Ref` that says what to
sample, and whether there is anything to apply this frame.

`active` is written by [`update!`](@ref) and read by the pass body, because "this
frame is outside the analysed range" has to reach the kernel. `dims` is what the
edge was sized for; an analysis re-propagated at another resolution is a
structural change, and comparing the two here makes a missed one render nothing
rather than garbage.
"""
struct PlaneEdge
    buf::Any                          # the transient the plane is stored into
    node::Any                         # Ref{<:FxNode} — the live node
    dims::Tuple{Int, Int}
    active::Base.RefValue{Bool}
end

"""
    update!(e::PlaneEdge, clip, frame) -> Bool

Store this frame's plane into the transient the chain reads it from, and say
whether there is one.

Unconditional: a new frame is a new plane, and a transient holds nothing between
runs — the placer is free to alias its slice against anything already dead, which
is the whole point of the plane being an ordinary resource. Fired before `run!`,
because the store lands at the update pass, which the graph puts at the head of
the schedule.

The data is a view into the track, retained rather than copied until the write
happens later in the same `run!`; nothing mutates an analysis result during a
render.
"""
function update!(e::PlaneEdge, clip::Clip, frame::Integer)
    n = e.node[]
    d = planedata(n, clip, frame)
    e.active[] = d !== nothing && planeshape(n, clip) == e.dims
    e.active[] || return false
    e.buf[:] = d
    return true
end

# What a node samples for its plane edge, and how big that is. Both forward to the
# op, where every analysis states them (`matte.jl`, `depth.jl`, `restore.jl`): a
# node is the op plus its wiring, and the wiring says nothing about the data.
planedata(n::FxNode, clip::Clip, frame::Integer) = planedata(n.op, clip, frame)
planeshape(n::FxNode, clip::Clip) = planeshape(n.op, clip)

# ---------------------------------------------------------------- the chain as passes
#
# chainpass!(graph, node, params, active, cur, ctx, dims) -> transient
#
# Adds one pass to the graph and returns the transient the next node reads.
# `cur` is the incoming image; `params` is a `Ref{typeof(node)}` the body reads at
# record time, which is what makes a changed parameter a store rather than a new
# plan. `active` is read the same way: an effect that is switched off, neutral at
# this frame, or bypassed by the global compare toggle is a pass that does nothing
# — the node stays in the structure, because whether a blur's σ is zero right now
# is a value, and values must not change which graph is compiled.
#
# In-place nodes declare `read + write` on `cur` itself — in a chain every node is
# its input's last consumer, so there is nothing to copy. Their inactive body is
# then genuinely free. An out-of-place node has to copy instead: the next pass
# reads its destination, and a buffer nothing wrote is a black frame, not a
# missing effect.

"""
What building one chain needs beyond the graph: the state its bodies read, the
plane bindings collected on the way out, and the device parameters its dispatches
read.

`stores` is how a per-frame NUMBER reaches a recorded plan. A pass used to read
`pr[].strength` inside its body, which worked because the body ran every frame;
a `Dispatch` is packed with its arguments once, at `record!`, so a value that
changes between frames has to be a [`Mantle.GPURef`](@ref) whose store lands at
the update pass. One closure per parameter, run by [`update!`](@ref) beside the
planes, and for the same reason: everything this frame hands the device goes the
same way.
"""
struct ChainBuild
    state::FxState
    edges::Vector{PlaneEdge}
    stores::Vector{Any}                 # () -> write one device parameter
end
ChainBuild() = ChainBuild(FxState(), PlaneEdge[], Any[])


"""
    param!(f, ctx, g, neutral) -> GPURef

A device parameter this frame's value is stored into, and the ref the dispatches
read it through.

`f()` is called once per frame by [`update!`](@ref) and returns the value. It is a
closure rather than a value because what it reads — a node's `Ref`, an `active`
flag, a track at this frame — is only settled when the frame is.

`neutral` is the value the ref starts at and gives the parameter its TYPE, which
is why it is a value rather than the type: half of these are structs with no
`zero`, and the do-nothing value is the one worth writing down anyway — it is
what a pass reads before the first frame is stored.
"""
function param!(f, ctx::ChainBuild, g, neutral::T) where {T}
    r = Mantle.GPURef(g.dev, neutral)
    push!(ctx.stores, () -> (r[] = convert(T, f()); nothing))
    return r
end

"""
    gate!(f, ctx, g) -> Mantle.Buffer{UInt32}

The flag a gated pass runs on: `f()` says whether this pass does anything at this
frame, and the pass is wrapped in `Mantle.repeat!(g, 1; while_nonzero = gate)`.

For a pass that writes ITS INPUT — a colour adjustment, a matte — where doing
nothing is the correct result of being switched off. A pass that writes a
separate destination cannot be gated this way: the destination would go unwritten
and the next pass would read a buffer nothing filled, which is a black frame
rather than a missing effect. Those carry their switch as a neutral PARAMETER
instead, so the one path they have is the only path.
"""
function gate!(f, ctx::ChainBuild, g)
    b = Mantle.Buffer(g.dev, UInt32[0])
    push!(ctx.stores, () -> (b[:] = UInt32[f() ? 1 : 0]; nothing))
    return b
end

"""
The source pass: colour-convert (or upload) the frame `decodesource` produced,
and give it full coverage.

Two buffers, because what arrives is `RGB{N0f8}` — a decoder's NV12 planes or a
host frame — and what the chain works in is [`PlanePixel`](@ref). `raw` is a
transient like any other, so the placer aliases it against something already dead
and it costs no memory of its own.

A decoded video frame is opaque: alpha 1 everywhere. Coverage enters the chain
where something REMOVES picture — a matte, a crop, a rendered scene's background.
"""
function chainpass!(g, ::SourceNode, ::Nothing, ::Nothing, ctx::ChainBuild, dims)
    cur = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    st = ctx.state
    # The decoder's own NV12 planes are a DIFFERENT pair of buffers every frame —
    # the ring allocates per decoded frame — so they cannot be dispatch arguments,
    # which are packed once when the plan records. What has to be stable is the
    # DESTINATION: these two are the chain's, declared once, and `update!` copies
    # the chosen ring entry into them. Persistent rather than transient, because
    # that copy happens outside the graph and a transient's storage is arena
    # memory only valid inside a run.
    raw = rawbuffer!(st, g.dev, dims)
    st.upload = raw                       # a host frame is stored into this
    luma, chroma = planebuffers!(st, g.dev, dims)
    convert = gate!(ctx, g) do
        st.baked === nothing && !st.uploaded && st.decoded !== nothing
    end
    Mantle.repeat!(g, 1; while_nonzero = convert) do _
        Mantle.dispatch!(g, GPUFiltering.nv12torgb_kernel!,
                         (raw, luma, chroma, sourceisbt601(st)), dims; name = "source/nv12")
    end
    # ALWAYS, and this is the one pass the chain is guaranteed: whatever filled
    # `raw` — a stored host frame, a stored bake, or the conversion above — a
    # decoded frame is opaque and the chain works in premultiplied coverage.
    Mantle.dispatch!(g, opaque_kernel!, (cur, raw), dims; name = "source")
    return cur
end

"""
    planebuffers!(st, dev, dims) -> (luma, chroma)

The chain's own NV12 planes, made once and kept.

The pair a GPU-decoded frame is copied into before the run, so the conversion
pass has a stable resource to read. Held on the state rather than rebuilt per
frame: their address is what the recorded plan packed, and a new buffer each
frame would be a new address the recording never sees.
"""
function planebuffers!(st::FxState, dev, dims::Tuple{Int, Int})
    p = st.planes
    p === nothing || return p
    st.planes = (Mantle.Buffer(dev, zeros(UInt8, dims[1] * dims[2])),
                 Mantle.Buffer(dev, zeros(UInt8, dims[1] * (dims[2] ÷ 2))))
    return st.planes
end

"""
    synthesize!(st, dims) -> Bool

Retime this frame, writing the synthesized picture into `raw`, and say whether it
did.

Here and not in a pass because the interpolator is a MODEL — RIFE, with a plan of
its own — and a plan cannot be a dispatch inside another. It was called from
inside a recorded pass body through an installed global, which is exactly the
shape `custom!` allowed and nothing else does.

Nothing to do unless this timeline frame lands BETWEEN two source frames
(`phase > 0`), the next one decoded, and an interpolator is installed. Half of a
slowed clip's frames land exactly on a source frame and are shown as they are —
that fallback is the normal case, not an error path.
"""
function synthesize!(st::FxState, dims::Tuple{Int, Int})
    (st.phase > 0.0 && st.decoded2 !== nothing && hasinterpolator() &&
     st.raw !== nothing && st.baked === nothing) || return false
    a, b = synthscratch!(st, dims)
    sourceinto!(a, st.source, st.decoded)
    sourceinto!(b, st.source, st.decoded2)
    INSTALLED.interpolate(Mantle.storage(st.raw), a, b, st.phase)
    return true
end

"The two decoded frames retiming interpolates between, made once and kept."
function synthscratch!(st::FxState, dims::Tuple{Int, Int})
    p = st.synth
    p === nothing || return p
    dev = Mantle.todevice(Mantle.defaultbackend())
    st.synth = (Mantle.storage(Mantle.Buffer(dev, RGB{N0f8}, dims)),
                Mantle.storage(Mantle.Buffer(dev, RGB{N0f8}, dims)))
    return st.synth
end

"""
    rawbuffer!(st, dev, dims) -> Buffer{RGB{N0f8},2}

The chain's decoded picture, before coverage — made once and kept.

PERSISTENT, where every other step of the chain is a transient, and that is the
whole design of the source pass. Three different things fill it — a stored host
frame, the NV12 conversion pass, the retiming interpolator — and two of those
run outside the graph. A transient's storage is arena memory valid only inside a
run, so anything written from outside would be aliasing corruption; a buffer the
chain owns has an address the recorded plan can name and anyone can write.

One frame of VRAM per chain (6 MB at 1080p), which is what a plan that records
costs here.
"""
function rawbuffer!(st::FxState, dev, dims::Tuple{Int, Int})
    b = st.raw
    b === nothing || return b
    st.raw = Mantle.Buffer(dev, RGB{N0f8}, dims)
    return st.raw
end

"""
    copyplanes!(st, frame) -> nothing

Put a decoded frame's NV12 planes where the conversion pass reads them.

Outside the graph, and safely so: the destination is a persistent `Mantle.Buffer`
the chain owns, not arena memory. A `nothing` frame or a source that serves host
pictures is a no-op — the conversion pass is gated off in both cases.
"""
copyplanes!(::FxState, ::Any) = nothing
function copyplanes!(st::FxState, f::Nv12Frame)
    p = st.planes
    p === nothing && return nothing
    copyto!(Mantle.storage(p[1]), vec(f.y))
    copyto!(Mantle.storage(p[2]), vec(f.uv))
    return nothing
end

"Whether this source's NV12 is BT.601 — a property of the stream, read per frame."
sourceisbt601(st::FxState) = (s = st.source; s isa GpuVideoStream ? s.bt601 : false)

"""
The optical-flow source pass. Two decoded frames in, one synthesized frame out.

Falls back to the plain conversion whenever there is nothing to synthesize — the
phase is zero (this timeline frame lands exactly on a source frame), the next
frame could not be decoded (the clip's last), or no interpolator is installed.
That fallback is not an error path: half of a slowed clip's frames land exactly
on a source frame and must be shown as they are.
"""
function chainpass!(g, ::SmoothSourceNode, ::Nothing, ::Nothing, ctx::ChainBuild, dims)
    # THE SAME GRAPH as the plain source. Retiming is not a different picture
    # path, it is a different producer of the same buffer: the interpolator is a
    # whole model with a plan of its own and cannot be a dispatch inside this
    # one, so it runs on the host side of `update!` and writes `raw` there, the
    # way a stored host frame and the conversion pass both do.
    #
    # That is what `raw` being PERSISTENT buys. Three producers, one consumer,
    # one recorded plan — where this used to be a pass body branching three ways
    # and calling an installed global from inside a recorded batch.
    return chainpass!(g, SourceNode(), nothing, nothing, ctx, dims)
end

"""
Stabilization: three dispatches behind one gate.

What the host still does is LOOK THE TRACK UP — a warp matrix at this frame,
scaled to the render size — because a track is a host object and reading it is
not GPU work. What it no longer does is decide whether the pass runs from inside
the pass: the matrix is a parameter and the lookup's answer feeds the gate.

Clear, warp, copy back, in that order and for the reason `applymotiontrack!`
gives: `skipoutside` leaves pixels outside the source alone, so the destination
has to start black or the previous frame shows through the borders.
"""
function chainpass!(g, ::MotionNode, pr, act, cur, ctx::ChainBuild, dims)
    tmp = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    st = ctx.state
    M = param!(ctx, g, one(Mat3f)) do
        motionwarp(st.clip, st.served[], dims)
    end
    go = gate!(ctx, g) do
        act[] && motionwarp(st.clip, st.served[], dims) != one(Mat3f)
    end
    Mantle.repeat!(g, 1; while_nonzero = go) do _
        Mantle.dispatch!(g, fillpixels_kernel!, (tmp, zero(PlanePixel)), dims;
                         name = "stabilize/clear")
        Mantle.dispatch!(g, GPUFiltering.warp_kernel!,
                         (tmp, cur, M, true,
                          Vec4{Int32}(1, 1, dims[1], dims[2]), GPUFiltering.Replace()),
                         dims; name = "stabilize/warp")
        Mantle.dispatch!(g, copypixels_kernel!, (cur, tmp), dims; name = "stabilize/back")
    end
    return cur
end

"""
The colour track's gain and offset at this frame, as two parameters.

Same shape as stabilization: the host reads the track, the numbers are refs, and
the gate is the same question the host used to ask inside the body. Gated rather
than neutral-valued because it writes its input, so off is nothing done.
"""
function chainpass!(g, ::ColorTrackNode, pr, act, cur, ctx::ChainBuild, dims)
    st = ctx.state
    gain = param!(ctx, g, Vec3f(1)) do
        first(colortrackgainoffset(st.clip, st.served[], pr[].strength))
    end
    offset = param!(ctx, g, Vec3f(0)) do
        last(colortrackgainoffset(st.clip, st.served[], pr[].strength))
    end
    go = gate!(ctx, g) do
        act[] && colortrackgainoffset(st.clip, st.served[], pr[].strength) !==
                 (Vec3f(1), Vec3f(0))
    end
    Mantle.repeat!(g, 1; while_nonzero = go) do _
        Mantle.dispatch!(g, GPUFiltering.channellinear_kernel!, (cur, gain, offset), dims;
                         name = "colour track")
    end
    return cur
end

"""
One pass for every [`PlaneOp`](@ref): the plane is a declared `read`, so the
graph orders it against the update that wrote it and nothing has to reach around
the graph for a second input. Everything specific to the op is behind
`applyplane!`.
"""
function chainpass!(g, n::PlaneNode, pr, act, cur, ctx::ChainBuild, dims)
    plane = Mantle.Transient.Buffer(g, planeeltype(n.op), n.shape...; hostwritten = true)
    b = PlaneEdge(plane, pr, n.shape, Ref(false))
    push!(ctx.edges, b)
    go = gate!(ctx, g) do
        act[] && b.active[]
    end
    Mantle.repeat!(g, 1; while_nonzero = go) do _
        planepass!(g, n.op, cur, plane, n, pr, ctx, dims)
    end
    return cur
end

"""
    planepass!(g, op, cur, plane, node, pr, ctx, dims)

The dispatch one [`PlaneOp`](@ref) renders as, declared for the graph.

Dispatch on the op, exactly as `applyplane!` did — the op says which kernel reads
its plane and what that kernel is told about this frame. It is a separate function
and not a branch for the same reason `applyplane!` was: an op that arrives later
adds a method and touches nothing here.

These write their INPUT, so the caller gates them on `active`: switched off is
nothing done, and there is no destination left unwritten.
"""
function planepass!(g, ::MatteOp, cur, plane, n, pr, ctx::ChainBuild, dims)
    st = ctx.state
    p = param!(ctx, g, MatteParams()) do
        op = pr[].op
        cr = st.clip.crop
        MatteParams(clamp(op.strength, 0.0f0, 1.0f0), clamp(op.feather, 0.0f0, 1.0f0),
                    Vec4f(cr[1], cr[2], cr[3], cr[4]))
    end
    Mantle.dispatch!(g, matte_kernel!,
                     (cur, plane, Int32(n.shape[1]), Int32(n.shape[2]), p), dims;
                     name = "matte")
    return nothing
end

function planepass!(g, ::RestoreOp, cur, plane, n, pr, ctx::ChainBuild, dims)
    s = param!(ctx, g, 0.0f0) do
        clamp(Float32(pr[].op.strength), 0.0f0, 1.0f0)
    end
    Mantle.dispatch!(g, restore_kernel!,
                     (cur, plane, Int32(n.shape[1]), Int32(n.shape[2]), s), dims;
                     name = "restore")
    return nothing
end

"""
The depth blur's node. A type of its own rather than a plain [`PlaneNode`](@ref)
because it gathers: it reads neighbours of the pixel it writes, so it needs a
destination buffer, and `PlaneNode`'s pass is in place.
"""
struct DepthBlurNode <: FxNode
    input::Int
    op::DepthBlurOp
    shape::Tuple{Int, Int}
end
planeshape(n::DepthBlurNode) = n.shape

"""
One dispatch, and the RADIUS carries the switch.

Inactive means switched off, neutral, or this frame outside the depth track — and
the picture still has to reach `dst`, or the rest of the chain reads a buffer
nothing wrote, which is a black frame rather than a missing effect. So this
cannot be gated, and does not need to be: at `maxr = 0` the kernel's per-pixel
radius clamps to zero, the tap loop runs once over the pixel itself and divides
by `n = 1`, which is the picture unchanged. An UNWEIGHTED box average is what
makes that exact — the separable blur next door cannot do the same, because its
taps are weighted and tap one is the far edge (see [`blurweights!`](@ref)).
"""
function chainpass!(g, n::DepthBlurNode, pr, act, cur, ctx::ChainBuild, dims)
    plane = Mantle.Transient.Buffer(g, planeeltype(n.op), n.shape...; hostwritten = true)
    b = PlaneEdge(plane, pr, n.shape, Ref(false))
    push!(ctx.edges, b)
    dst = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    focus = param!(ctx, g, 0.0f0) do
        Float32(pr[].op.focus)
    end
    maxr = param!(ctx, g, Int32(0)) do
        (act[] && b.active[]) ? Int32(depthblurradius(pr[].op, dims)) : Int32(0)
    end
    Mantle.dispatch!(g, depthblur_kernel!,
                     (dst, cur, plane, Int32(n.shape[1]), Int32(n.shape[2]), focus, maxr),
                     dims; name = "depth blur")
    return dst
end

"""
The learned grade's node. `dim` is the LUT's edge length and is part of the
structure, because the graph reserves a buffer of exactly `dim^3 * 3` floats — a
clip graded at another table size needs its own graph, not one holding a buffer of
the wrong size.
"""
struct LookNode <: FxNode
    input::Int
    strength::Float32
    dim::Int
end
planeshape(n::LookNode) = (n.dim, n.dim)
# The table is this node's plane edge. Written per frame like every other plane:
# a transient keeps nothing between runs, and 431 KB staged once per frame is what
# an aliasable slice costs. `planeshape` answers in the same units `dim` is in, so
# a re-grade at another table size shows up as the structural change it is.
planedata(n::LookNode, clip::Clip, ::Integer) = (l = clip.look; l === nothing ? nothing : vec(l))
planeshape(::LookNode, clip::Clip) = (d = lookdim(clip); d === nothing ? nothing : (d, d))

"""
Two dispatches and no branch: the grade, then the mix back toward the original.

The STRENGTH carries the switch, and that is why this pass needs no gate. At
`s = 0` the mix is `out = 1·img + 0·out`, which is the bypass exactly — so
"switched off", "keyframed to nothing" and "not analysed yet" are all the same
number rather than a second path, and there is no `copyto!` anywhere. A gate
would not do here anyway: `dst` is a separate destination, and a gated pass that
runs nothing leaves it unwritten, which the next pass reads as a black frame.

`binsize` and `dim` are the table's, so they are constants of this graph — a clip
graded at another table size already needs its own graph (see [`lookdim`](@ref)).
"""
function chainpass!(g, n::LookNode, pr, act, cur, ctx::ChainBuild, dims)
    lut = Mantle.Transient.Buffer(g, Float32, n.dim, n.dim, n.dim, 3; hostwritten = true)
    b = PlaneEdge(lut, pr, (n.dim, n.dim), Ref(false))
    push!(ctx.edges, b)
    dst = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    s = param!(ctx, g, 0.0f0) do
        (act[] && b.active[]) ? clamp(Float32(pr[].strength), 0.0f0, 1.0f0) : 0.0f0
    end
    binsize = Float32(1.000001 / (n.dim - 1))
    Mantle.dispatch!(g, GPUFiltering.lut3d_kernel!,
                     (dst, cur, lut, binsize, Int32(n.dim)), dims; name = "look")
    Mantle.dispatch!(g, lookmix_kernel!, (dst, cur, s), dims; name = "look/mix")
    return dst
end

"""
One dispatch, gated.

This pass writes ITS INPUT, so "switched off" is "did nothing" and a gate says
exactly that — no destination goes unwritten, and a discarded pass costs its
barriers and the gate dispatch rather than a full-frame copy. The adjustment
itself is a `GPURef`: four floats that change per frame, packed once at
`record!` and stored per frame like every other parameter.
"""
function chainpass!(g, ::ColorNode, pr, act, cur, ctx::ChainBuild, dims)
    adj = param!(ctx, g, ColorAdjustments()) do
        pr[].adj
    end
    go = gate!(ctx, g) do
        # GPUFiltering's, qualified: `isneutral` here is the editor's, and it
        # answers for an EFFECT rather than for an adjustment's four numbers.
        act[] && !GPUFiltering.isneutral(pr[].adj)
    end
    Mantle.repeat!(g, 1; while_nonzero = go) do _
        Mantle.dispatch!(g, GPUFiltering.coloradjust_kernel!, (cur, adj), dims; name = "colour")
    end
    return cur
end

"""
The blur's taps are an edge, like a matte's alpha.

They used to be uploaded from the pass body, once per blur per frame — and a
host→device upload inside a recorded batch forces a `vkQueueSubmit`, so a
four-layer composite made five submits where the graph promises one. As an edge
they are written where every other per-frame input is: at the head of the
schedule, staged, with nothing to drain.

[`BLURTAPS`](@ref) is fixed, because the tap count follows σ and σ is a value: a
buffer that resized with a slider would resize the graph with it.
"""
function blurweights!(g, pr, act, ctx::ChainBuild)
    wb = Mantle.Transient.Buffer(g, Float32, BLURTAPS; hostwritten = true)
    r = Mantle.GPURef(g.dev, Int32(0))
    # ONE closure for both, because they are one decision. `convpass_kernel!`
    # reads `weights[k + radius + 1]`, so the taps are indexed RELATIVE to the
    # radius: a radius that does not match the weights it was built with reads
    # the wrong end of the kernel. At radius 0 that is `weights[1]`, the far
    # edge, whose weight is about zero — the frame would come out black rather
    # than unblurred.
    #
    # So the bypass is not "radius 0" but "radius 0 AND a centre tap of one",
    # which is the identity kernel and is what switched-off stores.
    push!(ctx.stores, () -> begin
        if act[]
            w, rad = gaussianweights(pr[].σ; maxradius = (BLURTAPS - 1) ÷ 2)
            wb[:] = w
            r[] = Int32(rad)
        else
            taps = zeros(Float32, BLURTAPS)
            taps[1] = 1.0f0
            wb[:] = taps
            r[] = Int32(0)
        end
        nothing
    end)
    return wb, r
end

"""
A separable blur is two dispatches, and the RADIUS carries the switch.

`convpass_kernel!` sums `2r+1` taps, so the IDENTITY KERNEL — radius 0 and a
centre tap of one — reads the pixel under it and writes it back: the horizontal
pass copies `cur` into `tmp` and the vertical copies `tmp` into `dst`. That is
the bypass, exactly, with no `copyto!` and no second path — the same argument
the look pass makes for `s = 0`, and the reason this pass needs no gate even
though it writes a separate destination. See [`blurweights!`](@ref) for why the
radius alone will not do it.

What a switched-off blur costs is two full-frame passes reading one tap each.
"""
function chainpass!(g, ::BlurNode, pr, act, cur, ctx::ChainBuild, dims)
    dst = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    tmp = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    wb, r = blurweights!(g, pr, act, ctx)
    Mantle.dispatch!(g, GPUFiltering.convpass_kernel!, (tmp, cur, wb, r, Val(1)), dims;
                     name = "blur/h")
    Mantle.dispatch!(g, GPUFiltering.convpass_kernel!, (dst, tmp, wb, r, Val(2)), dims;
                     name = "blur/v")
    return dst
end

"""
The blur, then the difference added back: three dispatches and an AMOUNT.

`unsharp_kernel!` writes `img + amount·(img - blurred)`, so `amount = 0` is
`img` whatever the blur produced — the bypass again, and the reason the blur
underneath it need not be switched off separately. One more transient than the
blur has, because the unsharp pass reads both the picture and its blur and may
not write over either while doing it.
"""
function chainpass!(g, ::SharpenNode, pr, act, cur, ctx::ChainBuild, dims)
    blurred = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    tmp = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    dst = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    wb, r = blurweights!(g, pr, act, ctx)
    amount = param!(ctx, g, 0.0f0) do
        act[] ? Float32(pr[].amount) : 0.0f0
    end
    Mantle.dispatch!(g, GPUFiltering.convpass_kernel!, (tmp, cur, wb, r, Val(1)), dims;
                     name = "sharpen/h")
    Mantle.dispatch!(g, GPUFiltering.convpass_kernel!, (blurred, tmp, wb, r, Val(2)), dims;
                     name = "sharpen/v")
    Mantle.dispatch!(g, GPUFiltering.unsharp_kernel!, (dst, cur, blurred, amount), dims;
                     name = "sharpen/mix")
    return dst
end

"""
A callback effect, declared — and the CALLBACK ITSELF is the parameter.

`fxkind` returns a closure over the effect's numbers (`Pointwise((c, uv) -> c * a)`
for an opacity of `a`), and the thing worth knowing is that a new `a` is the same
closure TYPE with a different captured value. It is isbits, so it goes in a
`Mantle.GPURef` like any other parameter and the plan neither recompiles nor
re-records when the slider moves. That is why this needed no change to the
callback API: `(c, uv)` still, and the value it closes over is stored per frame.

A `Stencil` reads neighbours so it cannot write its input, and a destination left
unwritten is a black frame. Its bypass is therefore a second gated pass rather
than a neutral value: "identity" is not expressible as a value of the effect's
own closure type, and making it so would cost the type stability this rests on.
"""
function chainpass!(g, n::PixelNode, pr, act, cur, ctx::ChainBuild, dims)
    invsz = Vec2f(1.0f0 / dims[1], 1.0f0 / dims[2])
    f = param!(ctx, g, kindcallback(n.kind)) do
        kindcallback(pr[].kind)
    end
    go = gate!(ctx, g) do
        act[]
    end
    if needsfresh(n.kind)
        out = Mantle.Transient.Buffer(g, PlanePixel, dims...)
        r = Int32(n.kind.radius)
        Mantle.repeat!(g, 1; while_nonzero = go) do _
            Mantle.dispatch!(g, GPUFiltering.stencil_kernel!,
                             (out, cur, f, r, Int32(dims[1]), Int32(dims[2]), invsz), dims;
                             name = "effect")
        end
        off = gate!(ctx, g) do
            !act[]
        end
        Mantle.repeat!(g, 1; while_nonzero = off) do _
            Mantle.dispatch!(g, copypixels_kernel!, (out, cur), dims; name = "effect/bypass")
        end
        return out
    end
    Mantle.repeat!(g, 1; while_nonzero = go) do _
        Mantle.dispatch!(g, GPUFiltering.pointwise_kernel!, (cur, cur, f, invsz), dims;
                         name = "effect")
    end
    return cur
end

"""
    kindcallback(kind) -> f

The device callback a [`FxKind`](@ref) renders with, separated from how it is
mapped. `Pointwise` and `Stencil` differ in the kernel that maps them, not in
what they carry, which is why the pass reads this and dispatches on the kind.
"""
kindcallback(k::Pointwise) = k.f
kindcallback(k::Stencil) = k.f

# ---------------------------------------------------------------- a clip's structure

"""
    nodefor(effect, input, clip) -> FxNode | nothing

The node an effect renders as, or `nothing` when it renders as none: its analysis
is missing (a stabilize slot whose track was removed, a matte never analysed), or
it is not a pixel operation at all.

This is structure, so it says what the graph will read. It is not where "switched
off" or "neutral right now" is decided: those are values, they change per frame,
and a graph that recompiled when a blur's σ reached zero would recompile mid-drag.
They are the node's `active` flag instead.
"""
nodefor(::StabilizeEffect, input, clip) =
    clip.motiontrack === nothing ? nothing : MotionNode(input)
nodefor(e::FlickerEffect, input, clip) =
    clip.colortrack === nothing ? nothing : ColorTrackNode(input, e.strength)
nodefor(e::RestoreEffect, input, clip) = planenode(RestoreOp(e.strength), input, clip)
function nodefor(e::LookEffect, input, clip)
    d = lookdim(clip)
    return d === nothing ? nothing : LookNode(input, e.strength, d)
end

function nodefor(e::DepthBlurEffect, input, clip)
    op = DepthBlurOp(e.focus, e.strength)
    sh = planeshape(op, clip)
    return sh === nothing ? nothing : DepthBlurNode(input, op, sh)
end
nodefor(e::MatteEffect, input, clip) = planenode(MatteOp(e.strength, e.feather), input, clip)
nodefor(e::ColorEffect, input, clip) = ColorNode(input, e.adj)       # specialized kernels
nodefor(e::BlurEffect, input, clip) = BlurNode(input, e.σ)
nodefor(e::SharpenEffect, input, clip) = SharpenNode(input, e.σ, e.amount)
# Placement, not pixels: `layermatrix` reads it where the layer is placed.
nodefor(::TransformEffect, input, clip) = nothing
# Opacity is the layer's alpha, applied where the layer meets the canvas — never a
# fade to black inside the chain, which is what it would be with nothing under it.
nodefor(::OpacityEffect, input, clip) = nothing
# Callback effects (including plugins), and the fallback for an op that renders
# nothing. A stack entry is not always a picture: the loop finder is a slot on the
# clip so that it gets a card and a place for its reference list, and draws
# nothing. Without this it reached `fxkind`, which has no method for it, and the
# MethodError surfaced from inside the presenter.
function nodefor(e::FxOp, input, clip)   # callback effects, incl. plugins
    k = fxkind(e)
    return k === nothing ? nothing : PixelNode(input, k)
end

"A plane node for `op`, or `nothing` when the clip has no plane for it to read."
function planenode(op::PlaneOp, input::Int, clip::Clip)
    sh = planeshape(op, clip)
    sh === nothing && return nothing
    return PlaneNode(input, op, sh)
end

"""
    graphof!(clip, dims) -> FxGraph

`clip`'s structure at frame size `dims`, built if it is not there.

Built ONCE per structural change, because that is the only thing it depends on.
`clip.graph = nothing` is how a change says so — see [`dirtygraph!`](@ref) — and a
different `dims` (a proxy swapped in, a source replaced) is the other way, since
the buffers a recorded chain reserves are sized by it.
"""
function graphof!(clip::Clip, dims::Tuple{Int, Int})
    g = clip.graph
    g === nothing || (g.dims == dims && return g)
    nodes = FxNode[sourcenode(clip)]
    slots = Effect[]
    for fx in clip.effects
        renderable(fx) || continue       # data, not a pass — see `renderable`
        n = nodefor(op(fx), length(nodes), clip)
        n === nothing && continue        # not a pixel op, or its analysis is not there
        push!(nodes, n); push!(slots, fx)
    end
    fresh = FxGraph(nodes, slots, dims)
    clip.graph = fresh
    return fresh
end

"""
    dirtygraph!(clip) -> clip

Say that `clip`'s STRUCTURE changed: an effect added, removed or reordered, an
analysis attached or dropped, the time interpolation switched.

Dropping the graph rather than raising a flag, because the next render builds one
anyway and two states ("stale" and "absent") would be one too many. Moving a
slider is not this — a value flows through a `Ref` into a chain that is already
compiled.
"""
dirtygraph!(clip::Clip) = (clip.graph = nothing; bakedirty!(clip); clip)

# ---------------------------------------------------------------- one clip in a composition

"""
One clip's chain, recorded into a composition's graph.

Everything here is per-frame state with a compiled home: `params` is one `Ref` per
node holding that node re-read at this frame, `active` says whether each pass does
its work, `planes` are the analysis images to write before the run, and `matrix`/
`bounds`/`alpha` are how the finished layer meets the canvas. [`update!`](@ref)
writes all of them, unconditionally.
"""
struct ClipChain
    clip::Clip
    nodes::Vector{FxNode}
    slots::Vector{Effect}
    params::Vector{Any}                 # Ref{<:FxNode}, parallel to `slots`
    active::Vector{Base.RefValue{Bool}} # …and so is this
    edges::Vector{PlaneEdge}
    stores::Vector{Any}                 # per-frame device parameters, see `param!`
    state::FxState
    out::Any                            # the transient holding this clip's picture
    dims::Tuple{Int, Int}
    matrix::Base.RefValue{Mat3f}        # where the layer lands on the canvas
    bounds::Base.RefValue{NTuple{4, Int}}  # …and which part of it is picture at all
    alpha::Base.RefValue{Float32}       # the layer's opacity
end

"""
    layerparam!(f, ch, g, neutral) -> GPURef

A per-frame parameter of the COMPOSITION's pass for one layer.

The same thing [`param!`](@ref) makes, hung on the chain instead of the build
context: `buildcomposition` has no `ChainBuild` — it is stitching finished chains
together — and `update!(ch, …)` is what runs a chain's stores, once per clip per
frame, which is when a placement is settled.
"""
function layerparam!(f, ch::ClipChain, g, neutral::T) where {T}
    r = Mantle.GPURef(g.dev, neutral)
    push!(ch.stores, () -> (r[] = convert(T, f()); nothing))
    return r
end

"""
Record `fg`'s passes into `g` and return the chain that drives them.

The source pass is built first and every other node hangs off what the previous
one returned, so the wiring is the node list and there is nothing to look up.
"""
function buildchain!(g, clip::Clip, fg::FxGraph)
    dims = fg.dims
    ctx = ChainBuild()
    cur = chainpass!(g, fg.nodes[1], nothing, nothing, ctx, dims)
    params, active = Any[], Base.RefValue{Bool}[]
    for (i, n) in enumerate(Iterators.drop(fg.nodes, 1))
        pr, act = Ref(n), Ref(true)
        push!(params, pr); push!(active, act)
        cur = chainpass!(g, n, pr, act, cur, ctx, dims)
    end
    return ClipChain(clip, fg.nodes, fg.slots, params, active, ctx.edges, ctx.stores,
                     ctx.state, cur, dims, Ref(one(Mat3f)), Ref((1, 1, dims[1], dims[2])),
                     Ref(1.0f0))
end

"""
    update!(ch, sf, phase, source; canvas, applytracks, playing, exact, chunks) -> nothing

Write everything about source frame `sf` into the chain: the decoded source, each
node's parameters at this frame, which passes do anything, the analysis planes, and
where the layer goes on the canvas.

Addressed by SOURCE frame, because that is what a keyframe, a stabilization warp
and a matte are all keyed on, and because the callers that are not the timeline —
a trim preview, a frame grab, the matte pipeline — have only that. `phase` is how
far between two source frames a retimed clip sits (see [`sourcephase`](@ref)); it
is zero for everything that is not slowed.

UNCONDITIONAL. Nothing here asks whether a write is necessary — it is being made
because something changed, and the alternative is a second account of what the
graph already holds. That account is what `plansignature`, `graphof`-per-frame and
`PlaneSlot`'s "what is in this buffer" all were.

The decode happens here rather than inside the source pass, because what a
streaming source really serves decides which frame every per-frame result is
sampled at — see [`decodesource`](@ref) — and it has to be settled before the plan
records anything.
"""
function update!(ch::ClipChain, sf::Integer, phase::Real, source;
                 canvas::Tuple{Int, Int}, applytracks::Bool = true, playing::Bool = false,
                 exact::Bool = false, chunks::Integer = 5,
                 alpha::Union{Nothing, Real} = nothing)
    clip = ch.clip
    sf = Int(sf)
    st = ch.state
    st.source = source
    st.clip = clip
    st.frame = sf
    st.served[] = sf
    # A BAKED frame is the chain's output, so the chain does not run: the source
    # pass hands the picture on and every effect after it is inactive.
    st.baked = bakedframe(clip, sf)
    for (i, fx) in enumerate(ch.slots)
        e = op(fx, sf)
        ch.active[i][] = st.baked === nothing && applytracks && fx.enabled[] && !isneutral(e)
        node = nodefor(e, ch.nodes[i + 1].input, clip)
        node === nothing || (ch.params[i][] = node)
    end
    # Whatever the SOURCE needs to know about this frame: nothing for a decoder,
    # this frame's animated numbers for a scene.
    updatesource!(clip.source, clip, sf)
    st.phase = clip.timeinterp === :flow ? Float64(phase) : 0.0
    st.decoded = decodesource(source, sf; playing, served = st.served, exact, chunks)
    # …and the frame after it, only when one will be synthesized AND one exists.
    # `served` is deliberately not passed: this decode must not move the frame the
    # rest of the chain was told it is rendering.
    st.decoded2 = (st.phase > 0.0 && sf + 1 < clip.src_out) ?
        decodesource(source, sf + 1; playing, exact, chunks) : nothing
    # A HOST frame goes in through the update the source pass reserved for it —
    # see `FxState.upload`. Fired here, with the planes, so everything that
    # reaches the device this frame reaches it the same way. A SCENE goes the same
    # route: its picture is a host image too, whether it was drawn, pre-rendered
    # or read off a bake.
    st.exact = exact
    hf = st.upload === nothing ? nothing : sourcepicture!(st, ch.dims; exact)
    st.uploaded = hf !== nothing && size(hf) == ch.dims
    # UNCONDITIONAL where there is a destination: a transient holds nothing
    # between runs, so a frame with no picture stores black rather than leaving
    # the chain to read the arena's last tenant.
    if st.upload !== nothing
        st.uploaded ? (st.upload[:] = vec(hf)) :
            st.blankonmiss && (st.upload[:] = blankpixels!(st, ch.dims))
    end
    # A decoder that served DEVICE planes: copy them into the chain's own, which
    # is what the conversion pass reads. Device to device, ~3 MB at 1080p, and it
    # buys a plan that records — the decoder's planes are a new pair of buffers
    # every frame and a recording names one.
    st.uploaded || copyplanes!(st, st.decoded)
    # …and retiming, which is a MODEL and so cannot be a pass: it runs here and
    # writes `raw` itself. Reports whether it did, because if it did there is
    # nothing for the conversion pass to do.
    st.uploaded |= synthesize!(st, ch.dims)
    for e in ch.edges                     # `served` is settled: the planes can be written
        update!(e, clip, st.served[])
    end
    ch.matrix[] = layermatrix(clip, ch.dims, canvas, sf)
    ch.bounds[] = croppixels(clip.crop, ch.dims)
    # The layer's opacity: its own, unless the CALLER dictates one. A dissolve is
    # the case that dictates — the fraction lives on the transition, not on either
    # clip — and it is a parameter of the call rather than a second compositor,
    # which is what it used to be: two `render`s, two full-canvas copies down to
    # the host, and a host-side lerp.
    prm = opacityparam(clip)
    own = prm === nothing ? 1.0 : valueat(prm, sf)
    ch.alpha[] = Float32(clamp(alpha === nothing ? own : alpha, 0.0, 1.0))
    # LAST. Every dispatch's per-frame numbers, stored the way the planes are —
    # and after everything they read, which is the whole of this function:
    # `params` and `active` above, and `matrix`/`bounds`/`alpha` on the three
    # lines before this one. Run earlier, the composition's own stores carried
    # the PREVIOUS frame's placement and a dissolve came out one step behind.
    for w in ch.stores
        w()
    end
    return nothing
end

# ---------------------------------------------------------------- placement

"""
    croppixels(crop, dims) -> (x0, y0, x1, y1)

A normalized crop rect as the inclusive pixel box it names. The crop is the SOURCE
RECT, not merely where the fit samples from: cropping removes picture and makes the
clip smaller, so scaling the result down must show canvas around it — never the
material the crop took away, nor the stabilizer's border, which lives out there too.
"""
function croppixels(crop::NTuple{4, <:Real}, dims::Tuple{Int, Int})
    w, h = dims
    x0 = clamp(floor(Int, crop[1] * w) + 1, 1, w)
    y0 = clamp(floor(Int, crop[2] * h) + 1, 1, h)
    x1 = clamp(ceil(Int, (crop[1] + crop[3]) * w), x0, w)
    y1 = clamp(ceil(Int, (crop[2] + crop[4]) * h), y0, h)
    return (x0, y0, x1, y1)
end

"""
    layermatrix(clip, layersize, canvas, frame = 0) -> Mat3f

How this clip's rendered layer lands on the canvas: its crop rect, FITTED WHOLE
(aspect preserved, so material of another shape gets letterbox bars rather than
being stretched), then the clip's manual placement on top, sampled at `frame`. The
one definition — the preview's axis limits and the composited pixels both come
from it, so a reframe cannot mean two things.
"""
function layermatrix(clip::Clip, layersize::Tuple{Int, Int}, canvas::Tuple{Int, Int},
                     frame::Real = 0)
    s, px, py, rot = transformof(clip, frame)
    return fitmatrix(clip.crop, layersize, canvas;
                     scale = s, position = (px, py), rotation = rot)
end

"""
    canvasrect(clip, layersize, canvas, frame = 0) -> (x, y, w, h)

Which part of a `layersize` buffer the canvas shows, normalized — the inverse of
[`layermatrix`](@ref), and what the preview puts on its axis so a single-clip
present frames itself exactly as the composite would bake it. Equal to the clip's
crop when it fills the canvas; reaching past 0..1 is precisely where the letterbox
bars are.
"""
function canvasrect(clip::Clip, layersize::Tuple{Int, Int}, canvas::Tuple{Int, Int},
                    frame::Real = 0)
    M = layermatrix(clip, layersize, canvas, frame)
    # ALL FOUR corners, bounding-boxed. Two opposite corners describe the mapped
    # region only while the map is axis-aligned; with a rotation in it they
    # describe a diagonal, and the axis limits derived from that framed empty
    # space — the preview went black the moment a clip was turned.
    cs = (Vec3f(0.5f0, 0.5f0, 1),
          Vec3f(Float32(canvas[1]) + 0.5f0, 0.5f0, 1),
          Vec3f(0.5f0, Float32(canvas[2]) + 0.5f0, 1),
          Vec3f(Float32(canvas[1]) + 0.5f0, Float32(canvas[2]) + 0.5f0, 1))
    ps = map(q -> M * q, cs)
    x0 = minimum(q -> (q[1] - 0.5) / layersize[1], ps)
    x1 = maximum(q -> (q[1] - 0.5) / layersize[1], ps)
    y0 = minimum(q -> (q[2] - 0.5) / layersize[2], ps)
    y1 = maximum(q -> (q[2] - 0.5) / layersize[2], ps)
    return (x0, y0, x1 - x0, y1 - y0)
end

# ---------------------------------------------------------------- the composition

"""
The whole picture at one frame, as ONE Mantle graph: every visible clip's chain,
the coverage each contributes, and the compositing that puts them on the canvas.

One graph and one submit, so the placer aliases transients ACROSS clip boundaries
— a two-clip stack no longer holds two full sets of intermediates alive at once,
because Mantle derives liveness from use and the first clip's chain is dead by the
time the second one's runs.

`signature` is what it was built for: the canvas and each clip's structural
identity, which is the identity of its [`FxGraph`](@ref) object. Nothing is
hashed and nothing is compared field by field — a structural change replaces that
object, and this stops matching.
"""
struct Composition
    chains::Vector{ClipChain}
    plan::Any
    accum::Any                  # the canvas transient
    canvas::Tuple{Int, Int}
    signature::Any
end

"The canvas as a 2-D view. Valid until the same plan runs again — consume it
(blit/download) before then, never hold it."
compositeimage(c::Composition) = frameview(c.accum, c.canvas)

"""
    buildcomposition(engine, clips, graphs, canvas) -> Composition

Record the whole frame: a black canvas, then for each clip bottom-to-top its
chain, its coverage, its placement and one `over` against the canvas.

The canvas is read AND written by every layer's compose pass, so the DAG orders
them bottom to top by itself — the stacking order is the declaration order and
there is nothing else stating it.
"""
function buildcomposition(engine, clips, graphs, canvas::Tuple{Int, Int})
    g = Mantle.Graph(engine.device)
    W, H = canvas
    # The CANVAS has no alpha: it is what is delivered — to the screen, to the
    # encoder — and there is nothing behind it to show through. Coverage is a
    # property of a layer on its way here, not of the finished picture.
    accum = Mantle.Transient.Buffer(g, RGB{N0f8}, W, H)
    Mantle.dispatch!(g, fillpixels_kernel!, (accum, RGB{N0f8}(0, 0, 0)), canvas;
                     name = "canvas")
    chains = ClipChain[]
    for (clip, fg) in zip(clips, graphs)
        ch = buildchain!(g, clip, fg)
        push!(chains, ch)
        dims = ch.dims
        # THE CLIP WRITES INTO THE CANVAS. One pass: resample the layer through
        # its placement matrix and composite the result where it lands. Every
        # layer reads and writes `accum`, so the DAG orders them bottom to top by
        # itself — the stacking order IS the declaration order, and the region a
        # layer touches is an overlapping slice of the one canvas.
        #
        # THE PLACEMENT IS THE COVERAGE. `warp!` resamples colour and alpha
        # together through one matrix, so where the layer does not reach — the
        # letterbox bars, the material a crop removed, the background a matte
        # keyed out — nothing is written and the canvas survives.
        #
        # This used to be three images of one fact: render the matte again into a
        # white-on-black coverage image, warp THAT through the same matrix, warp
        # the picture into a second canvas-sized buffer, and composite the two.
        #
        # The matrix, the bounds and the opacity are all per frame, so all three
        # are parameters: `Over` carries the layer's alpha and is itself the
        # value, which is why it goes in a ref rather than being baked at
        # `record!` like the `skipoutside` flag beside it.
        # The stores go on the CHAIN, because `update!(ch, …)` is what runs them
        # and it runs once per clip per frame — which is exactly when the matrix,
        # the crop rect and the opacity are settled.
        M = layerparam!(ch, g, one(Mat3f)) do
            ch.matrix[]
        end
        rect = layerparam!(ch, g, Vec4{Int32}(1, 1, canvas[1], canvas[2])) do
            b = ch.bounds[]
            Vec4{Int32}(b[1], b[2], b[3], b[4])
        end
        over = layerparam!(ch, g, Over(1.0f0)) do
            Over(ch.alpha[])
        end
        Mantle.dispatch!(g, GPUFiltering.warp_kernel!,
                         (accum, ch.out, M, true, rect, over), canvas; name = "place")
    end
    return Composition(chains, Mantle.record!(Mantle.Plan(g)), accum, canvas,
                       compositionsignature(clips, graphs, canvas))
end


"""
What a composition was built for: the canvas, and each clip's structural identity.

`objectid` of the clip's [`FxGraph`](@ref), because a structural change replaces
that object. The alternative is a fingerprint — walk the stack, hash the kinds,
the enabled flags, the plane shapes — computed on every frame to discover
something the edit already knew.
"""
compositionsignature(clips, graphs, canvas) =
    (canvas, Tuple((c.id, objectid(g)) for (c, g) in zip(clips, graphs)))

"""
How many compiled compositions the engine keeps.

A bound, not a policy: distinct compositions come from distinct visible-clip sets,
which a timeline has few of — but every structural edit makes the previous ones
unreachable, and a plan holds pool regions until it is freed. Past this the lot is
dropped and rebuilt, which costs one compile on the next frame.
"""
const MAXCOMPOSITIONS = 8

"""
One clip's chain on its own: the layer as the chain made it, at the SOURCE's size,
with the crop blacked out and no placement.

For the callers that want a clip's own picture rather than a composed canvas — the
matte pipeline, the transition preview, the agent's frame grabs. Everything that
puts a picture on SCREEN goes through [`composite`](@ref) instead, because a crop
and a reframe have to mean the same thing in the preview as in the file.
"""
struct LayerPlan
    chain::ClipChain
    plan::Any
    signature::Any
    out::Any                 # the flattened RGB picture — see `buildlayer`
end

"The layer's finished picture, over black. Valid until the same plan runs again."
layerimage(lp::LayerPlan) = frameview(lp.out, lp.chain.dims)

"The same layer as a PLANE — RGBA, premultiplied, with its own coverage. What a
bake keeps, so a pre-rendered scene still shows the clip underneath it."
layerplane(lp::LayerPlan) = frameview(lp.chain.out, lp.chain.dims)

function buildlayer(engine, clip::Clip, fg::FxGraph)
    g = Mantle.Graph(engine.device)
    ch = buildchain!(g, clip, fg)
    out = Mantle.Transient.Buffer(g, RGB{N0f8}, ch.dims...)
    # The crop REMOVES picture, and a caller handed this buffer gets the clip,
    # not the material the crop took away. `zero(PlanePixel)` is uncovered, so
    # the crop states it once and the flatten turns it into the black a caller
    # with nothing behind the layer expects. In a composite the placement's
    # bounds say the same thing; here there is no placement to say it.
    #
    # Two dispatches where there was one body, and the rect is a parameter — a
    # crop is edited, so it changes without the graph's structure changing.
    rect = layerparam!(ch, g, Vec4{Int32}(1, 1, ch.dims[1], ch.dims[2])) do
        x0, y0, x1, y1 = croppixels(ch.clip.crop, ch.dims)
        Vec4{Int32}(x0, y0, x1, y1)
    end
    Mantle.dispatch!(g, cropaway_kernel!, (ch.out, rect), ch.dims; name = "crop")
    Mantle.dispatch!(g, flatten_kernel!, (out, ch.out), ch.dims; name = "flatten")
    return LayerPlan(ch, Mantle.record!(Mantle.Plan(g)), layersignature(clip, fg), out)
end

layersignature(clip::Clip, fg::FxGraph) = (clip.id, objectid(fg))

# ---------------------------------------------------------------- engine

"""
Owns the render resources so callers pass one handle. `device` is the Mantle
device — it owns the pool every transient comes from — and `compositions` /
`layers` hold the compiled graphs.

There is no store beside them. Everything a render touches is a resource of the
graph that touches it, so what an engine owns is plans and a device, and freeing
a plan is what gives its memory back.
"""
mutable struct FxEngine
    backend::Any
    device::Any
    compositions::Dict{Any, Composition}
    layers::Dict{Any, LayerPlan}
end
FxEngine(backend) = FxEngine(backend, Mantle.Device(backend),
                             Dict{Any, Composition}(), Dict{Any, LayerPlan}())

"""
Give every compiled composition's regions back to the pool. Explicit — Mantle
frees nothing by finalizer, so dropping an engine without this leaks its regions
until the pool is trimmed.
"""
function emptyengine!(e::FxEngine)
    for c in values(e.compositions)
        Mantle.free!(c.plan)
    end
    for l in values(e.layers)
        Mantle.free!(l.plan)
    end
    empty!(e.compositions)
    empty!(e.layers)
    return nothing
end

"The compiled layer plan for this clip's structure, built once."
function layer!(engine::FxEngine, clip::Clip, fg::FxGraph)
    sig = layersignature(clip, fg)
    hit = get(engine.layers, sig, nothing)
    hit === nothing || return hit
    length(engine.layers) >= MAXCOMPOSITIONS && emptyengine!(engine)
    fresh = buildlayer(engine, clip, fg)
    engine.layers[sig] = fresh
    return fresh
end

"""
    composition!(engine, clips, graphs, canvas) -> Composition

The compiled composition for this set of clips at this canvas, built once.
"""
function composition!(engine::FxEngine, clips, graphs, canvas::Tuple{Int, Int})
    sig = compositionsignature(clips, graphs, canvas)
    hit = get(engine.compositions, sig, nothing)
    hit === nothing || return hit
    length(engine.compositions) >= MAXCOMPOSITIONS && emptyengine!(engine)
    fresh = buildcomposition(engine, clips, graphs, canvas)
    engine.compositions[sig] = fresh
    return fresh
end

"""
The default `alphafor`: nobody dictates an opacity, so every layer uses its own.

Shaped like `sourcefor` — a function of `(clip, srcframe)` — because that is what
the composite already speaks, and a `Dict` here would be a second vocabulary for
the same question.
"""
nothingfor(::Clip, ::Integer) = nothing

"""
    composite(f, engine, clips, n, sourcefor; canvas, applytracks, playing, exact) -> Bool

The stack of `clips` (bottom track → top) at timeline frame `n` as ONE image,
handed to `f(canvas)`; returns whether it was rendered.

The whole frame is one Mantle graph and one submit. Per clip: its effect chain,
its coverage, its crop and placement baked into the canvas, and an alpha blend by
the layer's opacity.

`canvas` is `(width, height)` — the SEQUENCE's canvas ([`canvassize`](@ref)),
which every caller passes. Taking it from the top layer instead was a second
definition of the output format: on mixed-resolution material the preview
composited at the top clip's size while the export wrote the first clip's.

`sourcefor(clip, srcframe)` is the ONE thing that differs between the tiers: a
`GpuVideoStream` for the GPU preview, a decoded `RGBFrame` from the ring for the
CPU preview, an export decoder for the render. Returning `nothing` means "that
layer isn't there yet" and aborts the composite (`false`) — the caller falls back.
Everything the PICTURE depends on lives here, once, so preview and export cannot
drift apart.

`alphafor(clip, srcframe)` overrides a layer's opacity — `nothing` for "use its
own", which is every caller but a DISSOLVE. A dissolve's fraction lives on the
transition rather than on either clip, and it is a parameter of this call for
exactly that reason: it used to be a second compositor (two `render`s, both
copied down to the host, and a host-side lerp), in the preview and again in the
export, which is two more places for the picture to drift.
"""
function composite(f, engine::FxEngine, clips, n::Integer, sourcefor;
                   canvas::Tuple{Integer, Integer},
                   applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                   chunks::Integer = 5, alphafor = nothingfor)
    can = (Int(canvas[1]), Int(canvas[2]))
    # The sources first, and all of them: a source is what says how big a clip's
    # chain is, so the structure cannot be settled before they are in hand — and a
    # layer that is not decoded yet must abort the frame rather than composite a
    # hole into it.
    sources = Any[]
    graphs = FxGraph[]
    for clip in clips
        # A source that renders its own frames IS what its chain reads; only a
        # decodable one has to be asked for.
        s = decodable(clip.source) ? sourcefor(clip, sourceframe(clip, n)) : clip.source
        s === nothing && return false
        push!(sources, s)
        push!(graphs, graphof!(clip, framesize(s)))
    end
    comp = composition!(engine, clips, graphs, can)
    for (ch, source) in zip(comp.chains, sources)
        sf = sourceframe(ch.clip, n)
        update!(ch, sf, sourcephase(ch.clip, n), source;
                canvas = can, applytracks, playing, exact, chunks,
                alpha = alphafor(ch.clip, sf))
    end
    Mantle.run!(comp.plan)
    KA.synchronize(engine.backend)
    f(compositeimage(comp))
    return true
end

"""
    render(f, engine, source, clip, frame; applytracks=true, playing=false)

Render `clip` at absolute source frame `frame` from `source` (a `GpuVideoStream`
or a CPU `RGBFrame`) and call `f(out)` with the finished device image, at the
SOURCE's size, crop blacked out and no placement. `f` must consume it
(blit/download) before returning — never hold it: the next run of the same plan
overwrites it.
"""
function render(f, engine::FxEngine, source, clip::Clip, frame::Integer; kw...)
    return runlayer(lp -> f(layerimage(lp)), engine, source, clip, frame; kw...)
end

"""
    renderplane(f, engine, source, clip, frame; …)

[`render`](@ref), but `f` gets the layer as a PLANE — coverage included.

For the callers that are going to composite the result themselves rather than
look at it. A bake is the only one today, and it is the reason this exists: a
baked scene flattened over black would come back opaque and hide the clip below
it the moment it was read back.
"""
renderplane(f, engine::FxEngine, source, clip::Clip, frame::Integer; kw...) =
    runlayer(lp -> f(layerplane(lp)), engine, source, clip, frame; kw...)

"Run one clip's layer plan for `frame` and hand the finished plan to `f`."
function runlayer(f, engine::FxEngine, source, clip::Clip, frame::Integer;
                  applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                  chunks::Integer = 5, phase::Real = 0.0)
    dims = framesize(source)
    lp = layer!(engine, clip, graphof!(clip, dims))
    update!(lp.chain, frame, phase, source; canvas = dims, applytracks, playing, exact,
            chunks)
    Mantle.run!(lp.plan)
    KA.synchronize(engine.backend)
    return f(lp)
end

@kernel function cropaway_kernel!(buf, rectp)
    rect = paramvalue(rectp)
    x0, y0, x1, y1 = Int32(rect[1]), Int32(rect[2]), Int32(rect[3]), Int32(rect[4])
    i, j = @index(Global, NTuple)
    @inbounds if i < x0 || i > x1 || j < y0 || j > y1
        buf[i, j] = zero(eltype(buf))
    end
end

# NONE OF THE THREE BELOW SYNCHRONIZE. They run only inside a graph pass body,
# where Mantle already orders the passes — and a `KA.synchronize` there SUBMITS
# the batch, so each one cost a `vkQueueSubmit` in the middle of the frame.
# Measured on an 8-layer 1080p composite: 9 submits per frame, one per layer's
# source pass plus the one `composite` makes at the end. The plan says one.
# The caller that reads the picture is the one that waits — `composite` and
# `runlayer` do it once, right before they hand the image over.

"""
    cropaway!(buf, crop) -> buf

Black out everything outside the normalized `crop` rect, in place. A no-op for an
uncropped clip, which is the common case and the reason this is a branch on the
rect rather than a kernel that always runs.
"""
function cropaway!(buf, crop::NTuple{4, <:Real})
    crop == (0.0, 0.0, 1.0, 1.0) && return buf
    w, h = size(buf, 1), size(buf, 2)
    x0, y0, x1, y1 = croppixels(crop, (w, h))
    (x0 == 1 && y0 == 1 && x1 == w && y1 == h) && return buf
    backend = KA.get_backend(buf)
    cropaway_kernel!(backend)(buf, Vec4{Int32}(x0, y0, x1, y1); ndrange = (w, h))
    return buf
end

@kernel function opaque_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds begin
        c = src[I]
        dst[I] = topixel(eltype(dst), Float32(red(c)), Float32(green(c)),
                         Float32(blue(c)), 1.0f0)
    end
end

"""
    opaque!(dst, src) -> dst

An RGB picture as a fully covered plane. The one place a frame ENTERS the chain's
format, which is why it is a named operation rather than a `copyto!` that happens
to convert.
"""
function opaque!(dst::AnyRGBFrame, src::AnyRGBFrame)
    backend = KA.get_backend(dst)
    opaque_kernel!(backend)(dst, src; ndrange = size(dst))
    return dst
end

@kernel function flatten_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds begin
        c = src[I]
        dst[I] = RGB{N0f8}(red(c), green(c), blue(c))
    end
end

"""
    flatten!(dst, src) -> dst

A plane as a picture with nothing behind it.

There is no arithmetic: the plane is PREMULTIPLIED, so its stored colour already
IS itself composited over black. Dropping the alpha channel is the whole
operation, and it is named rather than written as a converting `copyto!` because
"over black" is a decision — it is what the callers of [`render`](@ref) mean, and
it is not what the compositor does.
"""
function flatten!(dst::AbstractMatrix{RGB{N0f8}}, src::AnyRGBFrame)
    backend = KA.get_backend(dst)
    flatten_kernel!(backend)(dst, src; ndrange = size(dst))
    return dst
end
