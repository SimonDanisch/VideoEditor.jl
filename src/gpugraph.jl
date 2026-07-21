# A GPU effect graph: a DAG of device-image operations, executed with a pooled,
# liveness-minimal set of VRAM buffers so there is ZERO per-frame allocation after
# warm-up. Each node produces one device image from its inputs; the graph is topo-
# ordered and carries a precomputed last-use per node. Adding an effect = one
# `FxNode` subtype + one `eval_node!` method — the whole thing is free functions +
# multiple dispatch. The pixel kernels are the existing GPUFiltering calls; this is
# pure orchestration + buffer management.

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

"Free every pooled device buffer (call on the GPU worker)."
function emptypool!(pool::BufferPool)
    for stack in values(pool.free), b in stack
        Lava.unsafe_free!(b)
    end
    empty!(pool.free)
    return nothing
end

# ---------------------------------------------------------------- nodes

"A node in a [`FxGraph`]: produces one device image from the results of `inputs`."
abstract type FxNode end
inputs(::FxNode) = ()
"Whether the node is a per-pixel op that could be fused with adjacent pointwise nodes."
pointwise(::FxNode) = false

struct SourceNode <: FxNode end                                      # the decoded frame
struct MotionNode <: FxNode; input::Int; end                         # stabilization warp
struct ColorTrackNode <: FxNode; input::Int; end                     # per-frame color stabilization
struct ColorNode <: FxNode; input::Int; adj::ColorAdjustments; end
struct BlurNode <: FxNode; input::Int; σ::Float32; end
struct SharpenNode <: FxNode; input::Int; σ::Float32; amount::Float32; end
struct OpacityNode <: FxNode; input::Int; α::Float32; end
struct BlendNode <: FxNode; a::Int; b::Int; t::Float32; end          # cross-dissolve (2 inputs)

inputs(n::MotionNode) = (n.input,)
inputs(n::ColorTrackNode) = (n.input,)
inputs(n::ColorNode) = (n.input,)
inputs(n::BlurNode) = (n.input,)
inputs(n::SharpenNode) = (n.input,)
inputs(n::OpacityNode) = (n.input,)
inputs(n::BlendNode) = (n.a, n.b)
pointwise(::ColorNode) = true
pointwise(::OpacityNode) = true
pointwise(::BlendNode) = true

"Per-frame inputs the graph pulls from: the frame source (a `GpuVideoStream` or a CPU
frame), the clip (for track transforms), the frame index, size, and colour range."
struct FxContext
    stream::Any        # GpuVideoStream, or `nothing` for the CPU-frame source
    cpuframe::Any      # the CPU frame when `stream === nothing`
    clip::Clip
    frame::Int
    width::Int
    height::Int
    bt601::Bool
end

# eval_node!(node, ins, pool, canmutate, ctx) -> device image.
# `ins` are the input nodes' device images; `canmutate` is true when this node may
# overwrite `ins[1]` in place (its last consumer is this node).
function eval_node!(::SourceNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = acquire!(pool, (ctx.width, ctx.height))
    if ctx.stream === nothing
        copyto!(out, ctx.cpuframe)
    else
        f = frameat!(ctx.stream, ctx.frame)
        nv12torgb!(out, f.y, f.uv; bt601 = ctx.bt601)
    end
    return out
end
function eval_node!(n::MotionNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    tmp = acquire!(pool, size(out))
    applymotiontrack!(out, tmp, ctx.clip, ctx.frame)   # warps `out` using `tmp` as scratch
    release!(pool, tmp)
    return out
end
function eval_node!(n::ColorTrackNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    applycolortrack!(out, ctx.clip, ctx.frame)
    return out
end
function eval_node!(n::ColorNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    coloradjust!(out, n.adj)
    return out
end
function eval_node!(n::OpacityNode, ins, pool::BufferPool, canmutate, ctx::FxContext)
    out = canmutate ? ins[1] : copyacquire!(pool, ins[1])
    channellinear!(out, Vec3f(n.α), Vec3f(0))   # scale toward black
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
returned image is the output node's buffer — the caller uses it (blit/download) and
`release!`s it back to the pool for the next frame. One `KA.synchronize` at the end.
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

nodefor(e::ColorEffect, input) = ColorNode(input, e.adj)
nodefor(e::BlurEffect, input) = BlurNode(input, e.σ)
nodefor(e::SharpenEffect, input) = SharpenNode(input, e.σ, e.amount)
nodefor(e::OpacityEffect, input) = OpacityNode(input, e.α)

"""
    graphof(clip; applytracks=true) -> FxGraph

Compile `clip`'s render — source → motion/colour stabilization → effect stack — into
a graph. Pass the keyframe-sampled `effectiveclip` so the effect params are current.
Neutral effects are dropped.
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
    for e in clip.effects
        isneutral(e) && continue
        push!(nodes, nodefor(e, cur)); cur = length(nodes)
    end
    return compilegraph(nodes, cur)
end
