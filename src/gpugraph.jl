# A GPU effect graph: a DAG of device-image operations, executed with a pooled,
# liveness-minimal set of VRAM buffers so there is ZERO per-frame allocation after
# warm-up. The whole thing is free functions + multiple dispatch.
#
# Two layers, so most effects are NOT kernels:
#   • the CALLBACK layer — a `Pointwise`/`Stencil` effect is a pure function mapped
#     by a framework-owned kernel (`GPUFiltering.pointwise!`/`stencil!`), and the
#     SAME function runs the CPU stack and the GPU graph. This is what plugins use.
#   • the NODE layer — an `FxNode` + `eval_node!` for ops that own a specialized
#     kernel or multiple passes (decode, motion warp, separable blur, blend). This
#     is the escape hatch.
# An `FxEngine` owns the buffer pool; `render` hides acquire/release entirely.

# ---------------------------------------------------------------- buffer pool

"""
Reuse device RGB images across frames and nodes. `acquire!` hands out a buffer of
the requested size (allocating only when none is free); `release!` returns it. The
pool persists across frames, so a stable graph allocates only on the first frame
(and the peak-concurrency frame) and never again.
"""
mutable struct BufferPool
    backend::Any
    free::Dict{Tuple{Int, Int}, Vector{Any}}
    allocations::Int
end
BufferPool(backend) = BufferPool(backend, Dict{Tuple{Int, Int}, Vector{Any}}(), 0)

function acquire!(pool::BufferPool, sz::Tuple{Int, Int})
    stack = get!(() -> [], pool.free, sz)
    if isempty(stack)
        pool.allocations += 1
        return KA.allocate(pool.backend, RGB{N0f8}, sz)
    end
    return pop!(stack)
end
release!(pool::BufferPool, buf) = (push!(get!(() -> [], pool.free, size(buf)), buf); nothing)
copyacquire!(pool::BufferPool, src) = (b = acquire!(pool, size(src)); copyto!(b, src); b)

"Release one pooled buffer's memory: host arrays belong to the GC, device
buffers free eagerly."
freebuffer!(::Array) = nothing
freebuffer!(b) = Lava.unsafe_free!(b)

"Free every pooled buffer (device pools on the GPU worker; CPU pools just drop)."
function emptypool!(pool::BufferPool)
    foreach(stack -> foreach(freebuffer!, stack), values(pool.free))
    empty!(pool.free)
    return nothing
end

# ---------------------------------------------------------------- per-frame inputs

"""
Per-frame inputs the graph pulls from: the frame `source` (a `GpuVideoStream` for
disk→VRAM decode, or a CPU `RGBFrame` uploaded once), the `clip` (for track
transforms and effect params), the frame index, and whether this render is part of
sequential `playing` (a streaming source then prefetches the next GOP; scrubs and
paused presents skip that speculative decode). Size and colour range are derived
from the source — nothing else to pass.
"""
struct FxContext
    source::Any
    clip::Clip
    frame::Int
    playing::Bool
    # The frame the source ACTUALLY delivered. A streaming source under its
    # latency budget may serve the nearest already-decoded frame instead of
    # `frame` (see `frameat!`) — the PER-FRAME track transforms must then be
    # sampled at the SERVED index, or a stabilization transform for `frame`
    # lands on a different image and the preview jerks wildly while the decode
    # catches up (export was always exact, so only playback showed it).
    served::Base.RefValue{Int}
    # exact = the EXPORT policy: the source must deliver precisely `frame`
    # (`exactframeat!`), cost what it may — no latency budget, no stand-ins.
    exact::Bool
end
FxContext(source, clip::Clip, frame::Integer, playing::Bool = false; exact::Bool = false) =
    FxContext(source, clip, Int(frame), playing, Ref(Int(frame)), exact)
framesize(s) = size(s)                                   # a CPU RGBFrame
framesize(s::GpuVideoStream) = (s.width, s.height)
frameisbt601(::Any) = false
frameisbt601(s::GpuVideoStream) = s.bt601

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

# ---------------------------------------------------------------- effect callbacks

"""
How a per-pixel/neighborhood effect renders, as a pure callback the framework maps
with one kernel (on CPU and GPU alike). An effect opts in by defining [`fxkind`];
then it needs no node, no `eval_node!`, and no CPU/GPU split.
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
define [`nodefor`](@ref) + `eval_node!` instead.
"""
function fxkind end

# run a kind into `out` from `inp` on the GPU graph (device images)
applykind!(out, inp, k::Pointwise) = pointwise!(out, inp, k.f)
applykind!(out, inp, k::Stencil) = stencil!(out, inp, k.f, k.radius)
# a Pointwise op may overwrite its input; a Stencil reads neighbors so it cannot
needsfresh(::Pointwise) = false
needsfresh(::Stencil) = true

# CPU effect stack (`applyeffects!`): the same callbacks, run in place with scratch
applykindcpu!(buf, tmp, k::Pointwise) = pointwise!(buf, buf, k.f)
applykindcpu!(buf, tmp, k::Stencil) = (stencil!(tmp, buf, k.f, k.radius); copyto!(buf, tmp); buf)

# built-in opacity is just a pointwise scale toward black — no dedicated kernel
fxkind(e::OpacityEffect) = (a = e.α; Pointwise((c, uv) -> c * a))

# ---------------------------------------------------------------- nodes

"A node in a [`FxGraph`]: produces one device image from the results of `inputs`."
abstract type FxNode end
inputs(::FxNode) = ()
"Whether the node is a per-pixel op that could be fused with adjacent pointwise nodes."
pointwiseop(::FxNode) = false

struct SourceNode <: FxNode end                                      # the decoded frame
struct MotionNode <: FxNode; input::Int; end                         # stabilization warp
struct ColorTrackNode <: FxNode; input::Int; end                     # per-frame color stabilization
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
struct BlendNode <: FxNode; a::Int; b::Int; t::Float32; end          # cross-dissolve (2 inputs)
struct PixelNode{K <: FxKind} <: FxNode; input::Int; kind::K; end    # a callback effect

inputs(n::MotionNode) = (n.input,)
inputs(n::ColorTrackNode) = (n.input,)
inputs(n::MatteNode) = (n.input,)
inputs(n::RestoreNode) = (n.input,)
inputs(n::ColorNode) = (n.input,)
inputs(n::BlurNode) = (n.input,)
inputs(n::SharpenNode) = (n.input,)
inputs(n::BlendNode) = (n.a, n.b)
inputs(n::PixelNode) = (n.input,)
pointwiseop(::ColorNode) = true
pointwiseop(::BlendNode) = true
pointwiseop(n::PixelNode) = n.kind isa Pointwise

# eval_node!(node, ins, pool, canmutate, ctx) -> device image.
# `ins` are the input nodes' device images; `canmutate` is true when this node may
# overwrite `ins[1]` in place (its last consumer is this node).
function eval_node!(::SourceNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = acquire!(pool, framesize(ctx.source))
    return sourceinto!(out, ctx.source, ctx.frame; prefetch = ctx.playing,
                       served = ctx.served, exact = ctx.exact)
end
# the per-frame tracks sample at the SERVED index (what's actually on screen),
# not the requested one — the source node runs first, so `served` is set
function eval_node!(n::MotionNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    tmp = acquire!(pool, size(out))
    applymotiontrack!(out, tmp, ctx.clip, ctx.served[])   # warps `out`, `tmp` = scratch
    release!(pool, tmp)
    return out
end
function eval_node!(n::ColorTrackNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    applycolortrack!(out, ctx.clip, ctx.served[])
    return out
end
# like the matte, restored frames are keyed by the frame actually served
function eval_node!(n::RestoreNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    applyrestore!(out, ctx.clip, ctx.served[]; strength = n.strength)
    return out
end
# the matte is per-frame data like the tracks above, so it samples `served` too
function eval_node!(n::MatteNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    applymatte!(out, ctx.clip, ctx.served[]; strength = n.strength, feather = n.feather)
    return out
end
function eval_node!(n::ColorNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    coloradjust!(out, n.adj)
    return out
end
function eval_node!(n::BlurNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    dst = acquire!(pool, size(ins[1]))
    tmp = acquire!(pool, size(ins[1]))
    gaussianblur!(dst, ins[1], n.σ; tmp = tmp)
    release!(pool, tmp)
    return dst
end
function eval_node!(n::SharpenNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    dst = acquire!(pool, size(ins[1]))
    tmp = acquire!(pool, size(ins[1]))
    unsharpmask!(dst, ins[1], n.σ, n.amount; tmp = tmp)
    release!(pool, tmp)
    return dst
end
function eval_node!(n::BlendNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : acquire!(pool, size(ins[1]))
    blend!(out, ins[1], ins[2], n.t)   # element-wise, safe when out aliases ins[1]
    return out
end
function eval_node!(n::PixelNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    reuse = canmutate && !needsfresh(n.kind)
    out = reuse ? ins[1] : acquire!(pool, size(ins[1]))
    return applykind!(out, ins[1], n.kind)
end

# ---------------------------------------------------------------- graph

"A compiled effect graph: topo-ordered nodes, the output node's index, and per-node
last-use (the last node index that consumes it) for buffer liveness."
struct FxGraph
    nodes::Vector{FxNode}
    output::Int
    lastuse::Vector{Int}
end

"Compile a topo-ordered node list (each node's inputs precede it) into an `FxGraph`."
function compilegraph(nodes::Vector{FxNode}, output::Integer)
    lastuse = zeros(Int, length(nodes))
    for (j, node) in enumerate(nodes), i in inputs(node)
        lastuse[i] = max(lastuse[i], j)
    end
    lastuse[output] = length(nodes) + 1   # the output survives the whole run
    return FxGraph(nodes, Int(output), lastuse)
end

"""
    execute!(graph, pool, ctx) -> device image

Run the graph for one frame, acquiring/releasing pooled buffers by liveness. The
returned image is the output node's buffer; use [`render`](@ref) to have it released
for you. One `KA.synchronize` at the end.
"""
function execute!(graph::FxGraph, pool::BufferPool, ctx::FxContext)
    results = Vector{Any}(undef, length(graph.nodes))
    for (j, node) in enumerate(graph.nodes)
        ins = map(i -> results[i], inputs(node))
        canmutate = !isempty(ins) && graph.lastuse[inputs(node)[1]] == j
        out = eval_node!(node, ins, pool, canmutate, ctx)
        results[j] = out
        for i in inputs(node)                       # release inputs whose life ends here
            graph.lastuse[i] == j && results[i] !== out && release!(pool, results[i])
        end
    end
    KA.synchronize(pool.backend)
    return results[graph.output]
end

# ---------------------------------------------------------------- build from a clip

nodefor(e::RestoreEffect, input) = RestoreNode(input, e.strength)
nodefor(e::MatteEffect, input) = MatteNode(input, e.strength, e.feather)
nodefor(e::ColorEffect, input) = ColorNode(input, e.adj)             # specialized kernels
nodefor(e::BlurEffect, input) = BlurNode(input, e.σ)
nodefor(e::SharpenEffect, input) = SharpenNode(input, e.σ, e.amount)
nodefor(e::Effect, input) = PixelNode(input, fxkind(e))              # callback effects (incl. plugins)

"""
    graphof(clip; applytracks=true) -> FxGraph

Compile `clip`'s render — source → motion/colour stabilization → effect stack — into
a graph. Pass the keyframe-sampled `effectiveclip` so the effect params are current.
Neutral effects are dropped. `applytracks = false` is the hold-to-compare bypass:
it shows the ORIGINAL frame, so it skips the effect stack too, not just the tracks.
"""
function graphof(clip::Clip; applytracks::Bool = true)
    nodes = FxNode[SourceNode()]
    cur = 1
    if applytracks && clip.motiontrack !== nothing
        push!(nodes, MotionNode(cur)); cur = length(nodes)
    end
    if applytracks && clip.colortrack !== nothing
        push!(nodes, ColorTrackNode(cur)); cur = length(nodes)
    end
    if applytracks
        for e in liveeffects(clip)          # enabled, non-neutral entries in stack order
            push!(nodes, nodefor(e, cur)); cur = length(nodes)
        end
    end
    return compilegraph(nodes, cur)
end

# ---------------------------------------------------------------- engine

"""
Owns the render resources so callers pass one handle, not a pool. `render` acquires,
runs the graph, hands you the output, and releases everything back to the pool.
"""
mutable struct FxEngine
    backend::Any
    pool::BufferPool
end
FxEngine(backend) = FxEngine(backend, BufferPool(backend))
emptyengine!(e::FxEngine) = emptypool!(e.pool)

"""
    render(f, engine, source, clip, frame; applytracks=true, playing=false)

Render `clip` at `frame` from `source` (a `GpuVideoStream` or a CPU `RGBFrame`) and
call `f(out)` with the finished device image. The buffer is released to the engine's
pool afterwards, so `f` must consume it (blit/download) before returning — never hold
it. All buffer management is internal. `playing` marks sequential playback — a
streaming source then prefetches its next GOP (see [`frameat!`](@ref)).
"""
function render(f, engine::FxEngine, source, clip::Clip, frame::Integer;
                applytracks::Bool = true, playing::Bool = false, exact::Bool = false)
    graph = graphof(clip; applytracks = applytracks)
    out = execute!(graph, engine.pool, FxContext(source, clip, Int(frame), playing; exact))
    try
        return f(out)
    finally
        release!(engine.pool, out)
    end
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
              scale = clip.reframe[1], position = (clip.reframe[2], clip.reframe[3]))

"""
    placelayer!(dest, layer, clip) -> dest

Draw one rendered layer into the canvas through [`layermatrix`](@ref). Pixels the
layer does not cover are LEFT AS THEY ARE — so the bars show whatever the caller
put in `dest` first: black underneath the base layer, the canvas so far under a
layer stacked above one (whose bars must stay clear, not paint black over the
track below).
"""
function placelayer!(dest, layer, clip::Clip)
    warp!(dest, layer, layermatrix(clip, size(layer), size(dest)); skipoutside = true)
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
    lo = M * Vec3f(0.5f0, 0.5f0, 1.0f0)
    hi = M * Vec3f(Float32(canvas[1]) + 0.5f0, Float32(canvas[2]) + 0.5f0, 1.0f0)
    x0 = (lo[1] - 0.5) / layersize[1]
    y0 = (lo[2] - 0.5) / layersize[2]
    return (x0, y0, (hi[1] - 0.5) / layersize[1] - x0, (hi[2] - 0.5) / layersize[2] - y0)
end

"""
    composite(f, engine, clips, n, sourcefor; canvas, applytracks, playing, exact) -> Bool

The stack of `clips` (bottom track → top) at timeline frame `n` as ONE image,
handed to `f(canvas)`; returns whether it was rendered. Per layer: [`render`]'s
effect graph, then its crop BAKED into the canvas with `warp!`, then an alpha
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
    pool = engine.pool
    W, H = Int(canvas[1]), Int(canvas[2])
    accum = acquire!(pool, (W, H))
    warpbuf = nothing
    try
        for (k, clip) in enumerate(clips)
            srcframe = sourceframe(clip, n)
            source = sourcefor(clip, srcframe)
            source === nothing && return false
            lclip = withoutopacity(effectiveclip(clip, srcframe))
            layer = execute!(graphof(lclip; applytracks = applytracks), pool,
                             FxContext(source, lclip, srcframe, playing; exact))
            α = Float32(clamp(paramvalue(clip, :opacity, srcframe), 0.0, 1.0))
            if k == 1                                   # the bottom layer sits on black:
                fill!(accum, RGB{N0f8}(0, 0, 0))        # (1-α)·0 + α·layer = α·layer,
                placelayer!(accum, layer, lclip)        # and the bars stay that black
                α < 0.999f0 && channellinear!(accum, Vec3f(α), Vec3f(0))
            else
                warpbuf === nothing && (warpbuf = acquire!(pool, (W, H)))
                # start from the canvas so far: where THIS layer doesn't reach,
                # `warpbuf` still holds what is below and the blend leaves it be
                # (blend(a, a, α) = a). A letterboxed upper layer therefore shows
                # the track underneath through its bars instead of blacking it out.
                copyto!(warpbuf, accum)
                placelayer!(warpbuf, layer, lclip)
                blend!(accum, accum, warpbuf, α)
            end
            release!(pool, layer)
        end
        KA.synchronize(pool.backend)
        f(accum)
        return true
    finally
        warpbuf === nothing || release!(pool, warpbuf)
        release!(pool, accum)
    end
end
