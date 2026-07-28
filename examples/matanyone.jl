"""
Drive the editor's matte tool with MatAnyone running on Lava.

Opt-in glue, deliberately not a dependency in either direction: VideoEditor owns
the track, the UI and the render path and asks a propagator to fill in the
frames; LavaDNN knows nothing about editors. Including this file is what joins
them.

    include("examples/matanyone.jl")
    usematanyone!(; backend = LavaBackend())     # once, before marking a subject

After that the Matte tool's status line reads "model" instead of "built-in" and
every propagation runs the network.

Two shape rules the model imposes, both handled here rather than pushed onto the
caller:

  * frames are padded up to a multiple of 16 (the encoder downsamples by 16 and
    `initstate` sizes the memory bank from `W ÷ 16`), and the alpha is cropped
    back, so any matte resolution works;
  * propagation is forward-only and stateful, so the frames are fed in order and
    a marked frame re-injects its mask via `step!(...; mask = ...)` — the first
    mark also resets the memory bank with `firstframe = true`.

The seed mask is **0..255**, not 0..1. A normalized mask is 255x too faint and
produces an all-zero matte with no error anywhere — no exception, correct shape,
empty result. The image operand *is* 0..1, so the two use different scales.
"""

using VideoEditor, LavaDNN, KernelAbstractions
using VideoEditor: red, green, blue
const KA = KernelAbstractions

"Round up to the encoder's stride so `initstate`'s `W ÷ 16` is exact."
padto16(n::Integer) = ((Int(n) + 15) ÷ 16) * 16

"""
    matanyonepropagator(; graphdir, weights, backend) -> f

A propagator matching [`VideoEditor.registermatte!`](@ref)'s contract. The model
is built once and captured, because loading weights per propagation would cost
more than the propagation.
"""
function matanyonepropagator(;
        graphdir = joinpath(@__DIR__, "..", "..", "..", "gen", "graphs", "aten-autocast"),
        weights = joinpath(@__DIR__, "..", "..", "..", "gen", "weights.safetensors"),
        backend = KA.CPU())
    isdir(graphdir) || error("no exported graphs at $graphdir — run tools/ first")
    isfile(weights) || error("no weights at $weights")
    model = LavaDNN.Model(graphdir, weights; backend)

    return function (frames, seeds; progress = nothing)
        n = length(frames)
        w, h = size(frames[1])
        W, H = padto16(w), padto16(h)
        img = KA.allocate(backend, Float32, W, H, 3, 1)
        host = zeros(Float32, W, H, 3, 1)
        maskhost = zeros(Float32, W, H)
        state = LavaDNN.initstate(model, W, H)
        out = zeros(UInt8, w, h, n)
        # One device buffer per frame, downloaded once at the end. `collect`ing
        # each frame's alpha as it came cost a full queue drain per frame — the
        # host waits for the GPU, then the GPU waits for the host to ask for the
        # next frame, and neither overlaps. Device-to-device copies do not
        # synchronise, so the pipeline stays full.
        planes = [KA.allocate(backend, Float32, W, H) for _ in 1:n]
        got = falses(n)
        order = sort!(collect(keys(seeds)))
        isempty(order) && return out
        first = order[1]

        for k in 1:n
            # pad by edge replication rather than zeros: a black border reads as
            # background the model has to explain away, and it leaks into the
            # matte at the frame edge
            frame = frames[k]
            @inbounds for j in 1:H, i in 1:W
                c = frame[min(i, w), min(j, h)]
                host[i, j, 1, 1] = Float32(red(c))
                host[i, j, 2, 1] = Float32(green(c))
                host[i, j, 3, 1] = Float32(blue(c))
            end
            copyto!(img, host)

            alpha = if haskey(seeds, k)
                m = seeds[k]
                fill!(maskhost, 0.0f0)
                @inbounds for j in 1:min(h, H), i in 1:min(w, W)
                    # 0..255, NOT 0..1: the reference mask the model was
                    # validated against is `(0.0, 255.0)`, and a 0/1 mask is 255x
                    # too faint to register — it produces an all-zero matte with
                    # no error anywhere, which is the whole difficulty of this bug.
                    maskhost[i, j] = m[i, j] > 0x7f ? 255.0f0 : 0.0f0
                end
                dev = KA.allocate(backend, Float32, W, H)
                copyto!(dev, maskhost)
                # Mask ingest is its own step, WITHOUT `firstframe`. The two
                # together hit a path where `State`'s `lastpixfeat`/`lastmskvalue`
                # are still `nothing` and reach a broadcast, which fails to
                # compile ("call to jl_f_throw_methoderror" inside
                # `lava_broadcast_flat!`). The verified driver order is
                # ingest-then-run, so do that.
                # `firstframe` on the FIRST mark only: it resets the memory
                # bank, which is right when seeding and wrong when correcting a
                # drifting matte further into the clip.
                LavaDNN.step!(model, state, img; mask = dev, firstframe = k == first)
            elseif k < first
                # nothing marked yet — leave these frames transparent rather than
                # running the model on a bank that has never been seeded
                progress === nothing || progress(k, n)
                continue
            else
                LavaDNN.step!(model, state, img)
            end

            copyto!(planes[k], alpha)
            got[k] = true
            progress === nothing || progress(k, n)
        end
        for k in 1:n
            got[k] || continue
            a = collect(planes[k])
            @inbounds for j in 1:h, i in 1:w
                out[i, j, k] = round(UInt8, clamp(a[i, j], 0.0f0, 1.0f0) * 255)
            end
        end
        return out
    end
end

"""
    usematanyone!(; kwargs...)

Install the MatAnyone propagator into the editor. Same keywords as
[`matanyonepropagator`](@ref).
"""
function usematanyone!(; kwargs...)
    VideoEditor.registermatte!(matanyonepropagator(; kwargs...))
    @info "matte: MatAnyone propagator installed"
    return nothing
end
