# A GPU effect graph: a clip's render — source → tracks → effect stack — as a
# MANTLE graph, executed from a cached plan so there is ZERO per-frame
# allocation after warm-up and every intermediate is a transient the placer can
# alias. The whole thing is free functions + multiple dispatch.
#
# Two layers, so most effects are NOT kernels:
#   • the CALLBACK layer — a `Pointwise`/`Stencil` effect is a pure function mapped
#     by a framework-owned kernel (`GPUFiltering.pointwise!`/`stencil!`), and the
#     SAME function runs the CPU stack and the GPU graph. This is what plugins use.
#   • the NODE layer — an `FxNode` + a `chainpass!` for ops that own a specialized
#     kernel or multiple passes (decode, motion warp, separable blur, blend). This
#     is the escape hatch.
#
# Every node becomes ONE `custom!` pass, because the bodies are multi-launch or
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

"""
    decodesource(source, frame; playing, served, exact) -> what the source pass reads

Get the frame BEFORE the plan runs. `frameat!` is latency-bounded: under a scrub
it serves the nearest already-decoded frame rather than `frame` and reports which
through `served`, and every per-frame result — a stabilization warp, a matte
plane — has to be sampled at THAT index or it lands on a different picture and
the preview jerks while the decode catches up.

This used to be the first thing the source pass body did, which meant `served`
was not settled until the plan was already running: the decode's own submits
landed in the middle of the plan's recording, and a plane upload had nowhere in
the schedule to go. Out here it is settled before `run!`, so the planes are
written at the position the graph reserved for them.

`exact` is the EXPORT policy — precisely `frame`, cost what it may — and it
cannot serve anything else, so it leaves `served` alone.
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
    # The frame AFTER `decoded`, and how far between them this timeline frame
    # sits. Both are only set for a clip whose time interpolation is `:flow`;
    # `phase == 0` means this frame lands exactly on a source frame and nothing
    # has to be synthesized.
    decoded2::Any
    phase::Float64
    clip::Any
    frame::Int
    served::Base.RefValue{Int}
end
FxState() = FxState(nothing, nothing, nothing, 0.0, nothing, 0, Ref(0))

"""
The 2-D view a kernel gets over a resource's 1-D storage — a transient's arena
slice or a persistent `Mantle.Buffer`'s region alike. `KA.get_backend` walks
`parent`, so the view resolves to the right backend on Lava and on the host.
"""
frameview(x, dims) = reshape(Mantle.storage(x), dims)

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
    fxkind(e::FxOp) -> FxKind

The render callback for a per-pixel/neighborhood effect. Define this (plus a struct
and [`isneutral`](@ref)) and the effect renders on the CPU stack AND the GPU graph —
no kernel required. Effects that own a specialized/multi-pass kernel skip this and
define [`nodefor`](@ref) + `chainpass!` instead.
"""
function fxkind end

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

"A node in a clip's render chain: produces one device image from the previous one."
abstract type FxNode end

struct SourceNode <: FxNode end                                      # the decoded frame

"""
The source pass for a clip whose time interpolation is `:flow`: the frame is
SYNTHESIZED between the two the decoder produced, at [`sourcephase`](@ref).

A node rather than an effect because it changes what the source IS, before any
effect runs — the same category as `rate`, and for the same reason it cannot be
one: an effect sees one frame and this needs two.
""" 
struct SmoothSourceNode <: FxNode end
struct MotionNode <: FxNode; input::Int; end                         # stabilization warp
struct ColorTrackNode <: FxNode                                      # per-frame color stabilization
    input::Int
    strength::Float32
end
struct ColorNode <: FxNode; input::Int; adj::ColorAdjustments; end
struct BlurNode <: FxNode; input::Int; σ::Float32; end
struct SharpenNode <: FxNode; input::Int; σ::Float32; amount::Float32; end
struct PixelNode{K <: FxKind} <: FxNode; input::Int; kind::K; end    # a callback effect

# ---------------------------------------------------------------- plane ops
"""
What a node reads BESIDES the picture, when what it reads is a whole image:
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
because it is part of the PLAN SIGNATURE: the plane is a graph resource, so a
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

# ---------------------------------------------------------------- the buffer store

"""
One device buffer, and what is currently in it.

`source`/`frame` are why a scrub that comes back to a frame costs no upload: they
say which analysis result the bytes already are. A scratch slot leaves them empty
and never asks.
"""
mutable struct PlaneSlot
    buf::Any
    source::Any
    frame::Int
end

"""
Every device buffer the editor owns that is not a graph transient: the
compositor's scratch, and the per-frame analysis planes the nodes read.

Keyed by role and shape, so two clips whose mattes were analysed at the same size
share one buffer — and, because the shape is part of the plan signature, one
plan. The bytes come from the Mantle pool, which is the point of the type. There
were five schemes for "a device buffer that is not a transient": two
`KA.allocate` caches behind module globals (the matte's and the restoration's,
neither of them ever freed), the canvas scratch, the alpha layers, and the
restored frames themselves. None of them was visible to the allocator that
places everything else.
"""
struct BufferStore
    device::Any
    slots::Dict{Any, PlaneSlot}
end
BufferStore(device) = BufferStore(device, Dict{Any, PlaneSlot}())

"The slot for `key`, `n` elements of `T`, created on first use."
slot!(s::BufferStore, key, ::Type{T}, n::Integer) where {T} =
    get!(() -> PlaneSlot(Mantle.Buffer(s.device, T, Int(n)), nothing, 0), s.slots, key)

"""
    scratch!(store, role, T, dims) -> 2-D view

A working buffer of fixed role and size — the compositor's accumulator, its warp
target, a coverage layer. Persistent because the compositor is not inside a graph
yet; pool-backed because everything else is.
"""
scratch!(s::BufferStore, role::Symbol, ::Type{T}, dims::Tuple{Int, Int}) where {T} =
    frameview(slot!(s, (role, T, dims), T, prod(dims)).buf, dims)

function Base.empty!(s::BufferStore)
    for sl in values(s.slots)
        Mantle.free!(sl.buf)
    end
    empty!(s.slots)
    return s
end

"""
A node's plane, bound into one compiled chain: the store slot that holds it, the
`Update` that writes it at the position the graph reserved, the node's parameter
`Ref` (where the live op is), and whether the node has anything to apply this
frame.
"""
struct PlaneBinding
    key::Any
    update::Any                       # a Mantle UpdateRef
    params::Any                       # Ref{<:PlaneNode}
    dims::Tuple{Int, Int}
    active::Base.RefValue{Bool}
end

"The plane's buffer as a 2-D view — what the kernel gets, and what the compositor
is handed so its coverage comes from the same bytes the keying read."
planeview(s::BufferStore, b::PlaneBinding) = frameview(s.slots[b.key].buf, b.dims)

"""
Put this frame's plane in its buffer, unless it is already there.

Fired from [`runchain!`](@ref) BEFORE `run!`, because an `Update`'s write
position is at the head of the schedule — which is exactly why the decode had to
move out of the source pass: `served` has to be settled by now.

"Already there" is decided by [`planesource`](@ref) — the analysis result's
IDENTITY, not the clip and the frame, which do not change when a matte is
re-propagated under them.

The data handed to the update is a VIEW into the track (or the restore cache),
retained rather than copied until the write happens later in the same `run!`;
nothing mutates an analysis result during a render.
"""
function loadplane!(s::BufferStore, b::PlaneBinding, clip::Clip, frame::Int)
    op = b.params[].op
    d = planedata(op, clip, frame)
    b.active[] = d !== nothing
    d === nothing && return false
    sl = s.slots[b.key]
    src = planesource(op, clip, frame)
    sl.source === src && sl.frame == frame && return true
    b.update(d)
    sl.source, sl.frame = src, frame
    return true
end

# ---------------------------------------------------------------- the chain as passes
#
# chainpass!(graph, node, params, cur, ctx, dims) -> transient
#
# Adds ONE pass to the graph and returns the transient the next node reads.
# `cur` is the incoming image; `params` is a `Ref{typeof(node)}` the body reads
# at record time, which is what makes a changed parameter a store rather than a
# new plan. In-place nodes declare `read + write` on `cur` itself — in a chain
# every node is its input's last consumer, so there is nothing to copy.

"""
What building one chain needs beyond the graph: the store its planes come from,
the state its bodies read, and the plane bindings collected on the way out. A
context rather than four arguments threaded through every `chainpass!`.
"""
struct ChainBuild
    store::BufferStore
    state::FxState
    planes::Vector{PlaneBinding}
end
ChainBuild(store::BufferStore) = ChainBuild(store, FxState(), PlaneBinding[])

"The source pass: colour-convert (or upload) the frame `decodesource` produced
into the chain's first transient."
function chainpass!(g, ::SourceNode, ::Nothing, ctx::ChainBuild, dims)
    cur = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    st = ctx.state
    Mantle.custom!(g, "source") do p
        Mantle.use(p, cur; write = true)
        () -> sourceinto!(frameview(cur, dims), st.source, st.decoded)
    end
    return cur
end

"""
The optical-flow source pass. Two decoded frames in, one synthesized frame out.

Falls back to the plain conversion whenever there is nothing to synthesize — the
phase is zero (this timeline frame lands exactly on a source frame), the next
frame could not be decoded (the clip's last), or no interpolator is installed.
That fallback is not an error path: half of a slowed clip's frames land exactly
on a source frame and must be shown as they are.
"""
function chainpass!(g, ::SmoothSourceNode, ::Nothing, ctx::ChainBuild, dims)
    cur = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    fa  = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    fb  = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    st = ctx.state
    Mantle.custom!(g, "source-flow") do p
        Mantle.use(p, cur; write = true)
        Mantle.use(p, fa; read = true, write = true)
        Mantle.use(p, fb; read = true, write = true)
        () -> begin
            d = frameview(cur, dims)
            if st.phase <= 0.0 || st.decoded2 === nothing || !hasinterpolator()
                sourceinto!(d, st.source, st.decoded)
            else
                a, b = frameview(fa, dims), frameview(fb, dims)
                sourceinto!(a, st.source, st.decoded)
                sourceinto!(b, st.source, st.decoded2)
                INTERPOLATOR[](d, a, b, st.phase)
            end
        end
    end
    return cur
end

function chainpass!(g, ::MotionNode, pr, cur, ctx::ChainBuild, dims)
    tmp = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    st = ctx.state
    Mantle.custom!(g, "stabilize") do p
        Mantle.use(p, cur; read = true, write = true)
        Mantle.use(p, tmp; write = true)
        () -> applymotiontrack!(frameview(cur, dims), frameview(tmp, dims),
                                st.clip, st.served[])
    end
    return cur
end

function chainpass!(g, ::ColorTrackNode, pr, cur, ctx::ChainBuild, dims)
    st = ctx.state
    Mantle.custom!(g, "colour track") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> applycolortrack!(frameview(cur, dims), st.clip, st.served[];
                               strength = pr[].strength)
    end
    return cur
end

"""
One pass for every [`PlaneOp`](@ref): the plane is a declared `read`, so the
graph orders it against the update that wrote it and nothing has to reach around
the graph for a second input. Everything specific to the op is behind
`applyplane!`.
"""
function chainpass!(g, n::PlaneNode, pr, cur, ctx::ChainBuild, dims)
    key = (:plane, typeof(n.op), n.shape)
    sl = slot!(ctx.store, key, planeeltype(n.op), prod(n.shape))
    b = PlaneBinding(key, Mantle.Update(g, sl.buf), pr, n.shape, Ref(false))
    push!(ctx.planes, b)
    st = ctx.state
    Mantle.custom!(g, passname(n.op)) do p
        Mantle.use(p, cur; read = true, write = true)
        Mantle.use(p, sl.buf; read = true)
        () -> b.active[] &&
              applyplane!(frameview(cur, dims), frameview(sl.buf, n.shape), pr[].op, st.clip)
    end
    return cur
end

"""
The depth blur's node. A type of its own rather than a plain [`PlaneNode`](@ref)
because it GATHERS: it reads neighbours of the pixel it writes, so it needs a
destination buffer, and `PlaneNode`'s pass is in place.

The alternative was copying the input inside `applyplane!` — a device allocation
per frame, outside the pool, which is the exact thing the note on `PlaneNode`'s
buffer records as already having been fixed once.
"""
struct DepthBlurNode <: FxNode
    input::Int
    op::DepthBlurOp
    shape::Tuple{Int, Int}
end
planeshape(n::DepthBlurNode) = n.shape

function chainpass!(g, n::DepthBlurNode, pr, cur, ctx::ChainBuild, dims)
    key = (:plane, DepthBlurOp, n.shape)
    sl = slot!(ctx.store, key, planeeltype(n.op), prod(n.shape))
    b = PlaneBinding(key, Mantle.Update(g, sl.buf), pr, n.shape, Ref(false))
    push!(ctx.planes, b)
    dst = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    Mantle.custom!(g, passname(n.op)) do p
        Mantle.use(p, cur; read = true)
        Mantle.use(p, dst; write = true)
        Mantle.use(p, sl.buf; read = true)
        () -> begin
            d, c = frameview(dst, dims), frameview(cur, dims)
            # Inactive means this frame is outside the depth track. The picture
            # still has to reach `dst`, or the rest of the chain reads a buffer
            # nothing wrote — which is a black frame, not a missing effect.
            b.active[] ? depthblur!(d, c, frameview(sl.buf, n.shape), pr[].op) :
                         copyto!(d, c)
        end
    end
    return dst
end

"""
The learned grade's node. `dim` is the LUT's edge length and is part of the plan
signature, because the graph reserves a buffer of exactly `dim^3 * 3` floats — a
clip graded at another table size needs its own plan, not one holding a buffer of
the wrong size.
"""
struct LookNode <: FxNode
    input::Int
    strength::Float32
    dim::Int
end
planeshape(n::LookNode) = (n.dim, n.dim)

function chainpass!(g, n::LookNode, pr, cur, ctx::ChainBuild, dims)
    key = (:lut, n.dim)
    sl = slot!(ctx.store, key, Float32, n.dim^3 * 3)
    dst = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    st = ctx.state
    Mantle.custom!(g, "look") do p
        Mantle.use(p, cur; read = true)
        Mantle.use(p, dst; write = true)
        Mantle.use(p, sl.buf; read = true)
        () -> begin
            d, c = frameview(dst, dims), frameview(cur, dims)
            lut = st.clip.look
            # No look on the clip is not an error — the effect may sit in the
            # stack while the analysis has not run, exactly as the matte's does.
            # The picture still has to reach `dst` or the rest of the chain reads
            # a buffer nothing wrote.
            if lut === nothing || size(lut, 1) != n.dim
                copyto!(d, c)
            else
                applylook!(d, c, loadlut!(ctx.store, key, lut), pr[].strength)
            end
        end
    end
    return dst
end

function chainpass!(g, ::ColorNode, pr, cur, ctx::ChainBuild, dims)
    Mantle.custom!(g, "colour") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> coloradjust!(frameview(cur, dims), pr[].adj)
    end
    return cur
end

function chainpass!(g, ::BlurNode, pr, cur, ctx::ChainBuild, dims)
    dst = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    tmp = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    Mantle.custom!(g, "blur") do p
        Mantle.use(p, cur; read = true)
        Mantle.use(p, dst; write = true)
        Mantle.use(p, tmp; write = true)
        () -> gaussianblur!(frameview(dst, dims), frameview(cur, dims), pr[].σ;
                            tmp = frameview(tmp, dims))
    end
    return dst
end

function chainpass!(g, ::SharpenNode, pr, cur, ctx::ChainBuild, dims)
    dst = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    tmp = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    Mantle.custom!(g, "sharpen") do p
        Mantle.use(p, cur; read = true)
        Mantle.use(p, dst; write = true)
        Mantle.use(p, tmp; write = true)
        () -> unsharpmask!(frameview(dst, dims), frameview(cur, dims),
                           pr[].σ, pr[].amount; tmp = frameview(tmp, dims))
    end
    return dst
end

function chainpass!(g, n::PixelNode, pr, cur, ctx::ChainBuild, dims)
    if needsfresh(n.kind)
        out = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
        Mantle.custom!(g, "effect") do p
            Mantle.use(p, cur; read = true)
            Mantle.use(p, out; write = true)
            () -> applykind!(frameview(out, dims), frameview(cur, dims), pr[].kind)
        end
        return out
    end
    Mantle.custom!(g, "effect") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> applykind!(frameview(cur, dims), frameview(cur, dims), pr[].kind)
    end
    return cur
end

# ---------------------------------------------------------------- build from a clip

"""
    nodefor(effect, input, clip) -> FxNode | nothing

The node an effect renders as, or `nothing` when its ANALYSIS is not there: a
stabilize slot whose track was removed, a matte that was never analysed, a
restoration whose first window has not come back yet. Saying so in the structure
is what keeps the plan honest about what it will read — and it is one rule now,
where the matte and the restoration used to answer it a second time inside their
kernels and the compositor a third time on its own.
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
nodefor(e::FxOp, input, clip) = PixelNode(input, fxkind(e))        # callback effects (incl. plugins)

"A plane node for `op`, or `nothing` when the clip has no plane for it to read.
The shape comes from the same `planeshape` the node carries into the plan
signature, so what the plan reserves and what the analysis produced cannot
disagree."
function planenode(op::PlaneOp, input::Int, clip::Clip)
    sh = planeshape(op, clip)
    sh === nothing && return nothing
    return PlaneNode(input, op, sh)
end

"""
A clip's render chain as a node list, source first. The compiled form is the
Mantle plan an `FxEngine` builds from it; this is the structure that decides
which plan a render uses.
"""
struct FxGraph
    nodes::Vector{FxNode}
end

"""
    graphof(clip; applytracks=true) -> FxGraph

Compile `clip`'s render — source → motion/colour stabilization → effect stack — into
a node list. Pass the keyframe-sampled `effectiveclip` so the effect params are current.
Neutral effects are dropped. `applytracks = false` is the hold-to-compare bypass:
it shows the ORIGINAL frame, so it skips the effect stack too, not just the tracks.
"""
function graphof(clip::Clip; applytracks::Bool = true)
    # Frame sampling repeats a source frame when the clip is slowed; optical flow
    # synthesizes the frame in between. The choice belongs to the CLIP, like
    # `rate`, and is read here so the plan signature carries it — the two produce
    # different graphs and must not share a compiled chain.
    nodes = FxNode[clip.timeinterp === :flow ? SmoothSourceNode() : SourceNode()]
    cur = 1
    if applytracks
        for e in liveeffects(clip)          # enabled, non-neutral entries in stack order
            e isa TransformEffect && continue   # placement, not pixels — see `layermatrix`
            n = nodefor(e, cur, clip)
            n === nothing && continue           # its analysis is not there: renders nothing
            push!(nodes, n); cur = length(nodes)
        end
    end
    return FxGraph(nodes)
end

# ---------------------------------------------------------------- engine

"""
One compiled chain: the Mantle plan, the `FxState` its pass bodies read, one
parameter `Ref` per node after the source, and the output transient. Cached on
the engine and replayed per frame.
"""
struct ChainPlan
    plan::Any                 # a Mantle Plan — concrete per backend
    state::FxState
    params::Vector{Any}       # Ref{<:FxNode}, parallel to the nodes after the source
    planes::Vector{PlaneBinding}   # the per-frame planes this chain reads
    out::Any                  # the output transient
    dims::Tuple{Int,Int}
end

"The chain's finished picture: a 2-D view over the plan's output transient. It
stays valid until the same plan runs again — consume it (blit/download/compose)
before then, never hold it."
chainimage(cp::ChainPlan) = frameview(cp.out, cp.dims)

"""
The chain's binding for a kind of [`PlaneOp`](@ref), or `nothing` when it has
none. How the compositor gets at the very buffer the keying read, instead of
asking the clip a second time and hoping the two answers agree.
"""
function planebinding(cp::ChainPlan, ::Type{O}) where {O <: PlaneOp}
    for b in cp.planes
        b.params[].op isa O && return b
    end
    return nothing
end

"""
What makes two node lists the same plan: the frame size, and the node types in
order together with the shape of any plane they read. Values are deliberately not
part of it — they flow through the parameter `Ref`s, so a keyframed parameter
costs a store and not a recompile. A plane's SHAPE is not a value in that sense:
it sizes a graph resource, so a matte analysed at another resolution has to get
its own plan.
"""
plansignature(nodes, dims, applytracks) =
    (dims, applytracks, Tuple((typeof(n), planeshape(n)) for n in nodes))

function buildchain(engine, nodes::Vector{FxNode}, dims)
    g = Mantle.Graph(engine.device)
    ctx = ChainBuild(engine.store)
    cur = chainpass!(g, nodes[1], nothing, ctx, dims)
    params = Any[]
    for n in Iterators.drop(nodes, 1)
        pr = Ref(n)
        push!(params, pr)
        cur = chainpass!(g, n, pr, cur, ctx, dims)
    end
    return ChainPlan(Mantle.Plan(g), ctx.state, params, ctx.planes, cur, dims)
end

"""
Owns the render resources so callers pass one handle. `device` is the Mantle
device — it owns the pool every transient, every plane and every scratch buffer
comes from — `plans` caches one compiled chain per (structure, size) so a stable
timeline renders with no allocation and no compilation after warm-up, and `store`
is every device buffer that is not a transient.
"""
mutable struct FxEngine
    backend::Any
    device::Any
    plans::Dict{Any,ChainPlan}
    store::BufferStore
end
function FxEngine(backend)
    dev = Mantle.Device(backend)
    return FxEngine(backend, dev, Dict{Any,ChainPlan}(), BufferStore(dev))
end

"""
Give every cached plan's regions and every buffer in the store back to the pool.
Explicit — Mantle frees nothing by finalizer, so dropping an engine without this
leaks its regions until the pool is trimmed.
"""
function emptyengine!(e::FxEngine)
    for cp in values(e.plans)
        Mantle.free!(cp.plan)          # the plans first: they hold updates into the store
    end
    empty!(e.plans)
    empty!(e.store)
    return nothing
end

"""
    runchain!(engine, source, clip, frame; applytracks, playing, exact) -> ChainPlan

Run `clip`'s chain for one frame and return the compiled chain that ran: it holds
the output ([`chainimage`](@ref)) and the planes the compositor needs.

The decode happens here rather than inside the source pass, because what a
streaming source really serves decides which frame every per-frame result is
sampled at — see [`decodesource`](@ref).
"""
function runchain!(engine::FxEngine, source, clip::Clip, frame::Integer;
                   applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                   chunks::Integer = 5, phase::Real = 0.0)
    nodes = graphof(clip; applytracks).nodes
    dims = framesize(source)
    cp = get!(engine.plans, plansignature(nodes, dims, applytracks)) do
        buildchain(engine, nodes, dims)
    end
    st = cp.state
    st.source = source
    st.clip = clip
    st.frame = Int(frame)
    st.served[] = Int(frame)
    for i in eachindex(cp.params)            # the store a changed parameter costs
        cp.params[i][] = nodes[i + 1]
    end
    st.phase = clip.timeinterp === :flow ? Float64(phase) : 0.0
    st.decoded = decodesource(source, st.frame; playing, served = st.served, exact, chunks)
    # The frame after it, only when one will actually be synthesized. `served` is
    # deliberately NOT passed: this decode must not move the frame the rest of the
    # chain was told it is rendering, and a streaming source is free to refuse.
    # …and the frame after it, only when one will be synthesized AND one exists.
    # The bound is checked rather than discovered: `src_out` is exclusive, so
    # `frame + 1 < src_out` is exactly "there is a next frame in this clip". A
    # decoder that then fails is a real fault and must surface — catching around
    # this would turn a broken source into silently juddering playback.
    #
    # `served` is deliberately not passed: this decode must not move the frame the
    # rest of the chain was told it is rendering.
    st.decoded2 = (st.phase > 0.0 && st.frame + 1 < clip.src_out) ?
        decodesource(source, st.frame + 1; playing, exact, chunks) : nothing
    for b in cp.planes                       # `served` is settled: the planes can be written
        loadplane!(engine.store, b, clip, st.served[])
    end
    Mantle.run!(cp.plan)
    return cp
end

"""
    render(f, engine, source, clip, frame; applytracks=true, playing=false)

Render `clip` at `frame` from `source` (a `GpuVideoStream` or a CPU `RGBFrame`) and
call `f(out)` with the finished device image. `f` must consume it (blit/download)
before returning — never hold it: the next run of the same plan overwrites it.
All buffer management is internal. `playing` marks sequential playback — a
streaming source then prefetches its next GOP (see [`frameat!`](@ref)).
"""
function render(f, engine::FxEngine, source, clip::Clip, frame::Integer;
                applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                chunks::Integer = 5)
    out = chainimage(runchain!(engine, source, clip, frame; applytracks, playing, exact, chunks))
    # The crop REMOVES picture. Doing it here — once, for every caller — is
    # what makes that true: the preview blits this buffer straight to the
    # screen, so a crop that only told the canvas placement where to sample
    # was no crop at all in the preview. Zoom the viewer out and the material
    # the crop had removed was still sitting there, stabilizer smear and all,
    # while the export (which goes through `placelayer!`) had genuinely
    # dropped it. Two answers to "what is this clip".
    cropaway!(out, clip.crop)
    return f(out)
end

@kernel function cropaway_kernel!(buf, x0::Int32, y0::Int32, x1::Int32, y1::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i < x0 || i > x1 || j < y0 || j > y1
        buf[i, j] = zero(eltype(buf))
    end
end

"""
    cropaway!(buf, crop) -> buf

Black out everything outside the normalized `crop` rect, in place. A no-op for an
uncropped clip, which is the common case and the reason this is a branch on the
rect rather than a kernel that always runs.
"""
function cropaway!(buf, crop::NTuple{4, <:Real})
    crop == (0.0, 0.0, 1.0, 1.0) && return buf
    w, h = size(buf, 1), size(buf, 2)
    x0 = clamp(floor(Int, crop[1] * w) + 1, 1, w)
    y0 = clamp(floor(Int, crop[2] * h) + 1, 1, h)
    x1 = clamp(ceil(Int, (crop[1] + crop[3]) * w), x0, w)
    y1 = clamp(ceil(Int, (crop[2] + crop[4]) * h), y0, h)
    (x0 == 1 && y0 == 1 && x1 == w && y1 == h) && return buf
    backend = KA.get_backend(buf)
    cropaway_kernel!(backend)(buf, Int32(x0), Int32(y0), Int32(x1), Int32(y1);
                              ndrange = (w, h))
    KA.synchronize(backend)
    return buf
end

"""
    layermatrix(clip, layersize, canvas) -> Mat3f

How this clip's rendered layer lands on the canvas: its crop rect, FITTED WHOLE
(aspect preserved, so material of another shape gets letterbox bars rather than
being stretched), then the clip's manual `reframe` on top. The one definition —
the preview's axis limits and the composited pixels both come from it, so a
reframe cannot mean two things.
"""
layermatrix(clip::Clip, layersize::Tuple{Int, Int}, canvas::Tuple{Int, Int}) =
    fitmatrix(clip.crop, layersize, canvas;
              scale = transformof(clip)[1],
              position = (transformof(clip)[2], transformof(clip)[3]),
              rotation = transformof(clip)[4])

"""
    placelayer!(dest, layer, clip) -> dest

Draw one rendered layer into the canvas through [`layermatrix`](@ref). Pixels the
layer's CROP does not cover are LEFT AS THEY ARE — so the bars show whatever the caller
put in `dest` first: black underneath the base layer, the canvas so far under a
layer stacked above one (whose bars must stay clear, not paint black over the
track below).
"""
function placelayer!(dest, layer, clip::Clip)
    # The crop is the SOURCE RECT, not just where the fit samples from: cropping
    # removes picture and makes the clip smaller, so scaling the result down must
    # show canvas around it, never the material the crop took away (nor the
    # stabilizer's border, which lives out there too).
    w, h = size(layer)
    x0 = clamp(floor(Int, clip.crop[1] * w) + 1, 1, w)
    y0 = clamp(floor(Int, clip.crop[2] * h) + 1, 1, h)
    x1 = clamp(ceil(Int, (clip.crop[1] + clip.crop[3]) * w), x0, w)
    y1 = clamp(ceil(Int, (clip.crop[2] + clip.crop[4]) * h), y0, h)
    warp!(dest, layer, layermatrix(clip, size(layer), size(dest));
          skipoutside = true, bounds = (x0, y0, x1, y1))
    return dest
end

"""
    canvasrect(clip, layersize, canvas) -> (x, y, w, h)

Which part of a `layersize` buffer the canvas shows, normalized — the inverse of
[`layermatrix`](@ref), and what the preview puts on its axis so a single-clip
present frames itself exactly as the composite would bake it. Equal to the
clip's crop when it fills the canvas; reaching past 0..1 is precisely where the
letterbox bars are.
"""
function canvasrect(clip::Clip, layersize::Tuple{Int, Int}, canvas::Tuple{Int, Int})
    M = layermatrix(clip, layersize, canvas)
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

"""
    renderlayer!(engine, lclip, srcframe, source; …) -> ChainPlan

One clip's finished layer: its chain run, its crop already removed. The chain is
returned rather than the image because the compositor needs its matte plane too;
both are valid until the same plan runs again, so compose before then.
"""
function renderlayer!(engine::FxEngine, lclip::Clip, srcframe::Integer, source;
                      applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                      chunks::Integer = 5, phase::Real = 0.0)
    cp = runchain!(engine, source, lclip, srcframe; applytracks, playing, exact, chunks,
                   phase)
    cropaway!(chainimage(cp), lclip.crop)
    return cp
end

"""
    composite(f, engine, clips, n, sourcefor; canvas, applytracks, playing, exact) -> Bool

The stack of `clips` (bottom track → top) at timeline frame `n` as ONE image,
handed to `f(canvas)`; returns whether it was rendered. Per layer: [`render`]'s
effect chain, then its crop BAKED into the canvas with `warp!`, then an alpha
blend by the layer's opacity — opacity is the layer alpha here, not a fade to
black inside the graph.

`canvas` is `(width, height)` — the SEQUENCE's canvas ([`canvassize`](@ref)),
which every caller passes. Taking it from the top layer instead was a second
definition of the output format: on mixed-resolution material the preview
composited at the top clip's size while the export wrote the first clip's, so
the two disagreed about the picture (and `copyto!` into the export buffer had
no matching size to copy into).

`sourcefor(clip, srcframe)` is the ONE thing that differs between the tiers: a
`GpuVideoStream` for the GPU preview, a decoded `RGBFrame` from the ring for the
CPU preview, an export decoder for the render. Returning `nothing` means "that
layer isn't there yet" and aborts the composite (`false`) — the caller falls
back. Everything the PICTURE depends on lives here, once: preview and export,
CPU and GPU, cannot drift apart (they did: the GPU preview forgot that a baked
canvas must be shown whole, so a stabilized clip's crop was applied twice for
the length of a blend).
"""
function composite(f, engine::FxEngine, clips, n::Integer, sourcefor;
                   canvas::Tuple{Integer, Integer},
                   applytracks::Bool = true, playing::Bool = false, exact::Bool = false,
                   chunks::Integer = 5)
    W, H = Int(canvas[1]), Int(canvas[2])
    accum = scratch!(engine.store, :accum, RGB{N0f8}, (W, H))
    for (k, clip) in enumerate(clips)
        srcframe = sourceframe(clip, n)
        source = sourcefor(clip, srcframe)
        source === nothing && return false
        lclip = withoutopacity(effectiveclip(clip, srcframe))
        cp = renderlayer!(engine, lclip, srcframe, source; applytracks, playing, exact,
                          chunks, phase = sourcephase(clip, n))
        layer = chainimage(cp)
        prm = opacityparam(clip)
        α = Float32(clamp(prm === nothing ? 1.0 : valueat(prm, srcframe), 0.0, 1.0))
        # COVERAGE, per pixel: white where this layer is opaque, black where
        # what is underneath must show through — the letterbox bars, and the
        # background a matte removed. Keying paints that background black,
        # which is right over nothing and wrong over a track: the black is
        # opaque and hides the clip below. The coverage is placed through the
        # same matrix as the picture, so the two cannot disagree.
        cover = scratch!(engine.store, :cover, RGB{N0f8}, (W, H))
        alphalayer = scratch!(engine.store, :alphalayer, RGB{N0f8}, size(layer))
        # the chain's OWN matte binding: same plane, same frame, same strength as
        # the keying that just ran. Asking the clip a second time was a second
        # answer to "is this layer matted", and the two could differ by a frame.
        mb = planebinding(cp, MatteOp)
        if mb !== nothing && mb.active[]
            mattealpha!(alphalayer, planeview(engine.store, mb), mb.params[].op, lclip)
        else
            fill!(alphalayer, RGB{N0f8}(1, 1, 1))
        end
        fill!(cover, RGB{N0f8}(0, 0, 0))                 # outside the layer: fully clear
        placelayer!(cover, alphalayer, lclip)
        if k == 1                                        # the bottom layer sits on black
            fill!(accum, RGB{N0f8}(0, 0, 0))
            placelayer!(accum, layer, lclip)
            α < 0.999f0 && channellinear!(accum, Vec3f(α), Vec3f(0))
        else
            warpbuf = scratch!(engine.store, :warpbuf, RGB{N0f8}, (W, H))
            fill!(warpbuf, RGB{N0f8}(0, 0, 0))           # the layer ALONE, premultiplied
            placelayer!(warpbuf, layer, lclip)
            overcompose!(accum, warpbuf, cover, α)
        end
    end
    KA.synchronize(engine.backend)
    f(accum)
    return true
end

@kernel function overcompose_kernel!(dst, @Const(layer), @Const(cover), α::Float32)
    i, j = @index(Global, NTuple)
    @inbounds begin
        a = α * Float32(red(cover[i, j]))
        b = dst[i, j]
        l = layer[i, j]
        dst[i, j] = RGB{N0f8}(
            unitn0f8(α * Float32(red(l))   + Float32(red(b))   * (1.0f0 - a)),
            unitn0f8(α * Float32(green(l)) + Float32(green(b)) * (1.0f0 - a)),
            unitn0f8(α * Float32(blue(l))  + Float32(blue(b))  * (1.0f0 - a)))
    end
end

"""
    overcompose!(dst, layer, cover, α) -> dst

`layer` over `dst`, where `cover` says how much of `dst` each pixel hides.

`layer` is PREMULTIPLIED — keying already multiplied it by the matte and left the
rest black — so this is `α·layer + dst·(1 − α·cover)`, not a lerp. That is what
makes a keyed-out background transparent rather than black, and it subsumes the
letterbox rule: outside the layer `cover` is 0 and `dst` survives untouched.
"""
function overcompose!(dst::AnyRGBFrame, layer::AnyRGBFrame, cover::AnyRGBFrame, α::Real)
    backend = KA.get_backend(dst)
    overcompose_kernel!(backend)(dst, layer, cover, Float32(α); ndrange = size(dst))
    KA.synchronize(backend)
    return dst
end
