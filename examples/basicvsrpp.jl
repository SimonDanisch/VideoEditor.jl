"""
Drive the editor's restore effect with BasicVSR++ running on LavaDNN.

Same opt-in shape as `matanyone.jl`: no dependency in either direction, the
editor owns the effect and the cache, and including this file installs the model.

    include("examples/basicvsrpp.jl")
    usebasicvsrpp!(; backend = LavaBackend())

BasicVSR++ is temporal — it propagates along the clip in both directions — so it
takes a *window* of frames, which is exactly the contract `registerrestore!`
defines. The window the editor sends is what the model gets to reason over;
shorter windows restore faster and align worse.

The graph is exported at a fixed frame count and resolution (see
`tools/export_basicvsrpp.py`), so the window length and the source size must
match what was exported. Passing a different count silently produces a graph
whose `select` indices run off the end, which is why this checks up front.
"""

using VideoEditor, LavaDNN, KernelAbstractions
using VideoEditor: RGB, N0f8, red, green, blue
const KA = KernelAbstractions

const GRAPHDIR = joinpath(@__DIR__, "..", "..", "..", "gen", "graphs", "basicvsrpp-fp32")

"""
    basicvsrpprestorer(; graphdir, backend) -> (f, frames, scale)

A restorer matching [`VideoEditor.registerrestore!`](@ref)'s contract, plus the
window length the exported graph expects and the model's upscale factor.
"""
function basicvsrpprestorer(; graphdir = GRAPHDIR, backend = KA.CPU())
    isdir(graphdir) || error("no exported BasicVSR++ graph at $graphdir — " *
                             "run tools/export_basicvsrpp.py first")
    g = LavaDNN.loadgraph(joinpath(graphdir, "basicvsrpp.json"))
    w0 = LavaDNN.readsafetensors(joinpath(graphdir, "weights.safetensors"))
    weights = Dict{String, Any}(k => LavaDNN.toback(backend, v) for (k, v) in w0)
    # The exported input pins (W, H, C, T, N); read the window length and size
    # back off it so the caller cannot silently disagree with the graph.
    inb = g.buffers[g.inputs[1]]
    shp = Tuple(reverse(Int.(inb.shape)))          # (W, H, C, T, N)
    W, H, T = shp[1], shp[2], shp[4]

    f = function (frames; progress = nothing)
        length(frames) == T ||
            error("BasicVSR++ graph expects $T frames per window, got $(length(frames))")
        size(frames[1]) == (W, H) ||
            error("BasicVSR++ graph expects $(W)x$(H) frames, got $(size(frames[1]))")
        host = Array{Float32}(undef, W, H, 3, T, 1)
        @inbounds for t in 1:T, j in 1:H, i in 1:W
            c = frames[t][i, j]
            host[i, j, 1, t, 1] = Float32(red(c))
            host[i, j, 2, t, 1] = Float32(green(c))
            host[i, j, 3, t, 1] = Float32(blue(c))
        end
        lqs = LavaDNN.toback(backend, host)
        vals = LavaDNN.execute!(g, Dict{String, Any}(g.inputs[1] => lqs), weights;
                                dims = (;), backend = backend)
        out = LavaDNN.value(LavaDNN.Ctx(vals, g, (;), backend), g.outputs[1])
        KA.synchronize(backend)
        a = Array(out)                              # (4W, 4H, 3, T, 1)
        OW, OH = size(a, 1), size(a, 2)
        res = Vector{Matrix{RGB{N0f8}}}(undef, T)
        @inbounds for t in 1:T
            img = Matrix{RGB{N0f8}}(undef, OW, OH)
            for j in 1:OH, i in 1:OW
                img[i, j] = RGB{N0f8}(clamp(a[i, j, 1, t, 1], 0f0, 1f0),
                                      clamp(a[i, j, 2, t, 1], 0f0, 1f0),
                                      clamp(a[i, j, 3, t, 1], 0f0, 1f0))
            end
            res[t] = img
            progress === nothing || progress(t, T)
        end
        return res
    end
    return f, T, OW_SCALE
end

"Upscale factor of the REDS4 model."
const OW_SCALE = 4

"""
    usebasicvsrpp!(; backend) -> Int

Install BasicVSR++ as the editor's restorer; returns the window length the
exported graph wants, which is what `restorewindow!` should be called with.
"""
function usebasicvsrpp!(; kwargs...)
    f, frames, scale = basicvsrpprestorer(; kwargs...)
    VideoEditor.registerrestore!(f; scale = scale, window = frames)
    @info "restore: BasicVSR++ installed" window = frames scale = scale
    return frames
end
