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

# fill `out` with the decoded source frame (device-resident decode, or one upload);
# `served` reports which frame the stream really delivered. `exact` selects the
# export policy (exactframeat!) over the preview's latency-bounded serve.
function sourceinto!(out, s::GpuVideoStream, frame; prefetch::Bool = false,
                     served = nothing, exact::Bool = false)
    f = exact ? exactframeat!(s, frame) : frameat!(s, frame; prefetch, served)
    nv12torgb!(out, f.y, f.uv; bt601 = s.bt601)
    return out
end
sourceinto!(out, s, frame; prefetch::Bool = false, served = nothing, exact::Bool = false) =
    copyto!(out, s)

"""
What a pass body reads at record time, so the plan it belongs to can be
replayed for another frame — or another clip with the same structure — without
recompiling. `served` is set by the source pass: a streaming source under its
latency budget may serve the nearest already-decoded frame instead of `frame`
(see `frameat!`), and the PER-FRAME track transforms must then be sampled at
the SERVED index, or a stabilization transform for `frame` lands on a different
image and the preview jerks wildly while the decode catches up.
"""
mutable struct FxState
    source::Any
    clip::Any
    frame::Int
    served::Base.RefValue{Int}
    playing::Bool                     # sequential playback: prefetch the next GOP
    exact::Bool                       # the EXPORT policy: precisely `frame`, cost what it may
end
FxState() = FxState(nothing, nothing, 0, Ref(0), false, false)

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
    fxkind(e::Effect) -> FxKind

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
struct MotionNode <: FxNode; input::Int; end                         # stabilization warp
struct ColorTrackNode <: FxNode                                      # per-frame color stabilization
    input::Int
    strength::Float32
end
struct RestoreNode <: FxNode                                         # model-restored frame
    input::Int
    strength::Float32
end
struct MatteNode <: FxNode                                           # per-frame subject matte
    input::Int
    strength::Float32
    feather::Float32
end
struct ColorNode <: FxNode; input::Int; adj::ColorAdjustments; end
struct BlurNode <: FxNode; input::Int; σ::Float32; end
struct SharpenNode <: FxNode; input::Int; σ::Float32; amount::Float32; end
struct PixelNode{K <: FxKind} <: FxNode; input::Int; kind::K; end    # a callback effect

# ---------------------------------------------------------------- the chain as passes
#
# chainpass!(graph, node, params, cur, state, dims) -> transient
#
# Adds ONE pass to the graph and returns the transient the next node reads.
# `cur` is the incoming image; `params` is a `Ref{typeof(node)}` the body reads
# at record time, which is what makes a changed parameter a store rather than a
# new plan. In-place nodes declare `read + write` on `cur` itself — in a chain
# every node is its input's last consumer, so there is nothing to copy.

"The source pass: decode (or upload) the frame into the chain's first transient."
function chainpass!(g, ::SourceNode, ::Nothing, state::FxState, dims)
    cur = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    Mantle.custom!(g, "source") do p
        Mantle.use(p, cur; write = true)
        () -> sourceinto!(frameview(cur, dims), state.source, state.frame;
                          prefetch = state.playing, served = state.served,
                          exact = state.exact)
    end
    return cur
end

function chainpass!(g, ::MotionNode, pr, cur, state::FxState, dims)
    tmp = Mantle.Transient.Buffer(g, RGB{N0f8}, prod(dims))
    Mantle.custom!(g, "stabilize") do p
        Mantle.use(p, cur; read = true, write = true)
        Mantle.use(p, tmp; write = true)
        () -> applymotiontrack!(frameview(cur, dims), frameview(tmp, dims),
                                state.clip, state.served[])
    end
    return cur
end

function chainpass!(g, ::ColorTrackNode, pr, cur, state::FxState, dims)
    Mantle.custom!(g, "colour track") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> applycolortrack!(frameview(cur, dims), state.clip, state.served[];
                               strength = pr[].strength)
    end
    return cur
end

# like the matte, restored frames are keyed by the frame actually served
function chainpass!(g, ::RestoreNode, pr, cur, state::FxState, dims)
    Mantle.custom!(g, "restore") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> applyrestore!(frameview(cur, dims), state.clip, state.served[];
                            strength = pr[].strength)
    end
    return cur
end

# the matte is per-frame data like the tracks above, so it samples `served` too
function chainpass!(g, ::MatteNode, pr, cur, state::FxState, dims)
    Mantle.custom!(g, "matte") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> applymatte!(frameview(cur, dims), state.clip, state.served[];
                          strength = pr[].strength, feather = pr[].feather)
    end
    return cur
end

function chainpass!(g, ::ColorNode, pr, cur, state::FxState, dims)
    Mantle.custom!(g, "colour") do p
        Mantle.use(p, cur; read = true, write = true)
        () -> coloradjust!(frameview(cur, dims), pr[].adj)
    end
    return cur
end

function chainpass!(g, ::BlurNode, pr, cur, state::FxState, dims)
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

function chainpass!(g, ::SharpenNode, pr, cur, state::FxState, dims)
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

function chainpass!(g, n::PixelNode, pr, cur, state::FxState, dims)
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

nodefor(::StabilizeEffect, input) = MotionNode(input)
nodefor(e::FlickerEffect, input) = ColorTrackNode(input, e.strength)
nodefor(e::RestoreEffect, input) = RestoreNode(input, e.strength)
nodefor(e::MatteEffect, input) = MatteNode(input, e.strength, e.feather)
nodefor(e::ColorEffect, input) = ColorNode(input, e.adj)             # specialized kernels
nodefor(e::BlurEffect, input) = BlurNode(input, e.σ)
nodefor(e::SharpenEffect, input) = SharpenNode(input, e.σ, e.amount)
nodefor(e::Effect, input) = PixelNode(input, fxkind(e))              # callback effects (incl. plugins)

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
    nodes = FxNode[SourceNode()]
    cur = 1
    if applytracks
        for e in liveeffects(clip)          # enabled, non-neutral entries in stack order
            e isa TransformEffect && continue   # placement, not pixels — see `layermatrix`
            # a stabilize/flicker slot whose analysis was removed renders nothing
            e isa StabilizeEffect && clip.motiontrack === nothing && continue
            e isa FlickerEffect && clip.colortrack === nothing && continue
            push!(nodes, nodefor(e, cur)); cur = length(nodes)
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
    out::Any                  # the output transient
    dims::Tuple{Int,Int}
end

"""
What makes two node lists the same plan: the frame size and the node TYPES in
order. Values are deliberately not part of it — they flow through the parameter
`Ref`s, so a keyframed parameter costs a store, not a recompile.
"""
plansignature(nodes, dims, applytracks) =
    (dims, applytracks, Tuple(typeof(n) for n in nodes))

function buildchain(engine, nodes::Vector{FxNode}, dims)
    g = Mantle.Graph(engine.device)
    state = FxState()
    cur = chainpass!(g, nodes[1], nothing, state, dims)
    params = Any[]
    for n in Iterators.drop(nodes, 1)
        pr = Ref(n)
        push!(params, pr)
        cur = chainpass!(g, n, pr, cur, state, dims)
    end
    return ChainPlan(Mantle.Plan(g), state, params, cur, dims)
end

"""
The compositor's scratch, as persistent `Mantle.Buffer`s: fixed roles, canvas
sized, created on first use. These are outside any graph — stage 4 of the move
brings the compositor itself in; what matters today is that their bytes come
from the same pool as everything else.
"""
mutable struct CanvasScratch
    accum::Any
    warpbuf::Any
    cover::Any
end
CanvasScratch() = CanvasScratch(nothing, nothing, nothing)

function canvasbuffer!(cs::CanvasScratch, field::Symbol, dev, dims)
    b = getfield(cs, field)
    if b === nothing
        b = Mantle.Buffer(dev, RGB{N0f8}, prod(dims))
        setfield!(cs, field, b)
    end
    return frameview(b, dims)
end

"""
Owns the render resources so callers pass one handle. `device` is the Mantle
device — it owns the pool every transient and every scratch buffer comes from —
and `plans` caches one compiled chain per (structure, size) so a stable
timeline renders with no allocation and no compilation after warm-up.
"""
mutable struct FxEngine
    backend::Any
    device::Any
    plans::Dict{Any,ChainPlan}
    canvases::Dict{Tuple{Int,Int},CanvasScratch}
    alphalayers::Dict{Tuple{Int,Int},Any}
end
FxEngine(backend) = FxEngine(backend, Mantle.Device(backend), Dict{Any,ChainPlan}(),
                             Dict{Tuple{Int,Int},CanvasScratch}(), Dict{Tuple{Int,Int},Any}())

"""
Give every cached plan's regions and every scratch buffer back to the pool.
Explicit — Mantle frees nothing by finalizer, so dropping an engine without
this leaks its regions until the pool is trimmed.
"""
function emptyengine!(e::FxEngine)
    for cp in values(e.plans)
        Mantle.free!(cp.plan)
    end
    empty!(e.plans)
    for cs in values(e.canvases), b in (cs.accum, cs.warpbuf, cs.cover)
        b === nothing || Mantle.release!(Mantle.region(b.store))
    end
    empty!(e.canvases)
    for b in values(e.alphalayers)
        Mantle.release!(Mantle.region(b.store))
    end
    empty!(e.alphalayers)
    return nothing
end

"""
    runchain!(engine, source, clip, frame; applytracks, playing, exact) -> device image

Run `clip`'s chain for one frame and return the output: a 2-D view over the
plan's output transient. It stays valid until the same plan runs again —
consume it (blit/download/compose) before then, never hold it.
"""
function runchain!(engine::FxEngine, source, clip::Clip, frame::Integer;
                   applytracks::Bool = true, playing::Bool = false, exact::Bool = false)
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
    st.playing = playing
    st.exact = exact
    for i in eachindex(cp.params)            # the store a changed parameter costs
        cp.params[i][] = nodes[i + 1]
    end
    Mantle.run!(cp.plan)
    return frameview(cp.out, cp.dims)
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
                applytracks::Bool = true, playing::Bool = false, exact::Bool = false)
    out = runchain!(engine, source, clip, frame; applytracks, playing, exact)
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
    renderlayer!(engine, lclip, srcframe, source; …) -> device image

One clip's finished layer: its chain run, its crop already removed. The view is
valid until the same plan runs again — compose it before then.
"""
function renderlayer!(engine::FxEngine, lclip::Clip, srcframe::Integer, source;
                      applytracks::Bool = true, playing::Bool = false, exact::Bool = false)
    layer = runchain!(engine, source, lclip, srcframe; applytracks, playing, exact)
    cropaway!(layer, lclip.crop)
    return layer
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
                   applytracks::Bool = true, playing::Bool = false, exact::Bool = false)
    W, H = Int(canvas[1]), Int(canvas[2])
    cs = get!(CanvasScratch, engine.canvases, (W, H))
    accum = canvasbuffer!(cs, :accum, engine.device, (W, H))
    for (k, clip) in enumerate(clips)
        srcframe = sourceframe(clip, n)
        source = sourcefor(clip, srcframe)
        source === nothing && return false
        lclip = withoutopacity(effectiveclip(clip, srcframe))
        layer = renderlayer!(engine, lclip, srcframe, source; applytracks, playing, exact)
        α = Float32(clamp(paramvalue(clip, :opacity, srcframe), 0.0, 1.0))
        # COVERAGE, per pixel: white where this layer is opaque, black where
        # what is underneath must show through — the letterbox bars, and the
        # background a matte removed. Keying paints that background black,
        # which is right over nothing and wrong over a track: the black is
        # opaque and hides the clip below. The coverage is placed through the
        # same matrix as the picture, so the two cannot disagree.
        cover = canvasbuffer!(cs, :cover, engine.device, (W, H))
        abuf = get!(engine.alphalayers, size(layer)) do
            Mantle.Buffer(engine.device, RGB{N0f8}, prod(size(layer)))
        end
        alphalayer = frameview(abuf, size(layer))
        # the SAME effect entry the graph keyed with, so the coverage matches
        me = findeffect(lclip, MatteEffect)
        hasmatte = me !== nothing && !isneutral(me) &&
                   mattealpha!(alphalayer, lclip, srcframe;
                               strength = me.strength, feather = me.feather) !== nothing
        hasmatte || fill!(alphalayer, RGB{N0f8}(1, 1, 1))
        fill!(cover, RGB{N0f8}(0, 0, 0))                 # outside the layer: fully clear
        placelayer!(cover, alphalayer, lclip)
        if k == 1                                        # the bottom layer sits on black
            fill!(accum, RGB{N0f8}(0, 0, 0))
            placelayer!(accum, layer, lclip)
            α < 0.999f0 && channellinear!(accum, Vec3f(α), Vec3f(0))
        else
            warpbuf = canvasbuffer!(cs, :warpbuf, engine.device, (W, H))
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
