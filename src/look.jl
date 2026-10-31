"""
A learned colour grade, as a 3-D LUT (see `NeuralLUTRunner`).

The fourth JuliaVision model in the editor. It is the cheapest of them to apply
and the odd one out in shape: the model looks at ONE frame and predicts a table,
and that table then grades every frame of the shot. Predicting costs ~1.8 ms and
applying ~0.8 ms at 4K, so re-predicting per frame would fit a 60 fps budget —
but it would also make the look *drift within a shot*, which is the one thing a
grade must not do. A cut is where a look may change; a pan is not.

So the LUT is per clip, held host-side and uploaded once, and `analyzelook!` runs
on the frame the user is looking at — you grade from the frame you chose, which
is how a colourist works and also how the result stops being a surprise.
"""

"""
    registerlook!(f)

Install the look model. `f(img) -> Array{Float32,4}` takes one host RGB frame and
returns a `(D, D, D, 3)` table. Pluggable for the same reason the depth and
restoration models are: the editor has to run with it absent.
"""
const LOOKMODEL = Ref{Any}(nothing)

registerlook!(f) = (LOOKMODEL[] = f; nothing)
haslookmodel() = LOOKMODEL[] !== nothing

"""
The built-in look model: NeuralLUT, from `NeuralLUTRunner`.

Built on first use and kept, like the depth and SAM 2 models. `Array(...)`
because the table is stored on the clip and belongs in a project file — a
device-resident LUT could not be saved, and would be on the wrong device the
moment `autodetectgpu!` upgrades the backend under it.
"""
const NEURALLUT = Ref{Any}(nothing)

function neurallutlook(img)
    if NEURALLUT[] === nothing
        NEURALLUT[] = NeuralLUTRunner.neurallut(; backend = Lava.LavaBackend())
    end
    return Array(NeuralLUTRunner.predictlut(NEURALLUT[], img))
end

"""
    installlook!()

Point [`registerlook!`](@ref) at the built-in model. Called by `Player`; builds
nothing until a look is actually asked for.
"""
installlook!() = registerlook!(neurallutlook)

"""
    analyzelook!(clip, img) -> Array{Float32,4}

Predict a look for `clip` from `img` and hang it on the clip.

`img` is a frame the CALLER chose — normally the one under the playhead. Not the
clip's first frame: a shot often opens on black or on a whip, and grading from
that produces a look for a frame nobody will ever see.
"""
function analyzelook!(clip::Clip, img::AbstractMatrix{<:AbstractRGB})
    haslookmodel() || error("no look model installed — see registerlook!")
    lut = LOOKMODEL[](img)
    ndims(lut) == 4 && size(lut, 4) == 3 ||
        error("a look must be (D, D, D, 3), got $(size(lut))")
    clip.look = Array{Float32, 4}(lut)
    return clip.look
end

"""
    lookdim(clip) -> Int | nothing

The LUT's edge length, or `nothing` when the clip has no look. It is part of the
plan signature — the graph reserves a buffer of exactly this size — so a clip
graded at another table size needs its own plan rather than a buffer of the
wrong one.
"""
lookdim(clip::Clip) = (l = clip.look; l === nothing ? nothing : size(l, 1))

"""
    applylook!(out, img, lut, strength)

Grade `img` into `out`, mixed back toward the original by `strength`.

At full strength this is `lut3d!` and nothing else. Below it a second pointwise
pass lerps `out` back toward `img`.

**Not `GPUFiltering.blend!`**, which would express this in one call and does
sanction the aliasing — but it synchronizes, and this runs inside a graph pass
body where Mantle orders the passes and the kernels already there
(`coloradjust!`, `lut3d!`) do not drain the pipeline. Draining it mid-graph is a
measured -7% elsewhere in this project.

`out` is read and written by the mix, so it is not `@Const` there — the same
reason `lut3d_kernel!` leaves its input unannotated when the two-argument form
aliases it.
"""
function applylook!(out, img, lut, strength::Real)
    s = clamp(Float32(strength), 0.0f0, 1.0f0)
    lut3d!(out, img, lut)
    s >= 0.999f0 && return out
    lookmix_kernel!(KA.get_backend(out))(out, img, s; ndrange = size(out))
    return out
end

"`out = (1-s)·img + s·out`, pointwise, where `out` already holds the graded frame."
@kernel function lookmix_kernel!(out, @Const(img), s::Float32)
    I = @index(Global, Cartesian)
    @inbounds begin
        a = out[I]
        b = img[I]
        q = 1.0f0 - s
        # `topixel`, not the validating `RGB{N0f8}(::Float32, …)` — the latter's
        # error path builds a message with `repr`, and Lava rejects the whole
        # kernel for the string allocation, so this compiled on the CPU and could
        # never have run on the GPU it was written for.
        out[I] = topixel(eltype(out),
                         q * Float32(red(b))   + s * Float32(red(a)),
                         q * Float32(green(b)) + s * Float32(green(a)),
                         q * Float32(blue(b))  + s * Float32(blue(a)),
                         alphaof(b))
    end
end
