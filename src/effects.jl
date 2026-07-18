"""
Non-destructive per-clip effects, interpreted in stack order by
`applyeffects!`. All pixel work is done by GPUFiltering kernels through
KernelAbstractions — the same code runs on whatever backend the buffers
live on (CPU today; LavaArrays once frames are GPU-resident).
"""
abstract type Effect end

struct ColorEffect <: Effect
    adj::ColorAdjustments
end
ColorEffect(; kwargs...) = ColorEffect(ColorAdjustments(; kwargs...))

struct BlurEffect <: Effect
    σ::Float32
end

struct SharpenEffect <: Effect
    σ::Float32
    amount::Float32
end

isneutral(e::ColorEffect) = GPUFiltering.isneutral(e.adj)
isneutral(e::BlurEffect) = e.σ <= 0
isneutral(e::SharpenEffect) = e.amount <= 0

"Apply `clip`'s effect stack to `buf` in place, using two same-size scratch buffers."
function applyeffects!(buf::AnyRGBFrame, tmp1::AnyRGBFrame, tmp2::AnyRGBFrame, clip::Clip)
    any(e -> !isneutral(e), clip.effects) || return buf
    for e in clip.effects
        isneutral(e) && continue
        applyeffect!(buf, tmp1, tmp2, e)
    end
    KA.synchronize(KA.get_backend(buf))
    return buf
end

applyeffect!(buf, tmp1, tmp2, e::ColorEffect) = coloradjust!(buf, e.adj)
applyeffect!(buf, tmp1, tmp2, e::BlurEffect) =
    (gaussianblur!(tmp1, buf, e.σ; tmp = tmp2); copyto!(buf, tmp1))
applyeffect!(buf, tmp1, tmp2, e::SharpenEffect) =
    (unsharpmask!(tmp1, buf, e.σ, e.amount; tmp = tmp2); copyto!(buf, tmp1))

# ------------------------------------------------------------- serialization

effectdict(e::ColorEffect) = Dict{String, Any}(
    "type" => "color", "brightness" => e.adj.brightness, "contrast" => e.adj.contrast,
    "saturation" => e.adj.saturation, "temperature" => e.adj.temperature)
effectdict(e::BlurEffect) = Dict{String, Any}("type" => "blur", "sigma" => e.σ)
effectdict(e::SharpenEffect) =
    Dict{String, Any}("type" => "sharpen", "sigma" => e.σ, "amount" => e.amount)

function effectfromdict(d::AbstractDict)
    t = d["type"]
    t == "color" && return ColorEffect(; brightness = d["brightness"], contrast = d["contrast"],
                                       saturation = d["saturation"], temperature = d["temperature"])
    t == "blur" && return BlurEffect(Float32(d["sigma"]))
    t == "sharpen" && return SharpenEffect(Float32(d["sigma"]), Float32(d["amount"]))
    error("unknown effect type: $t")
end

# ------------------------------------------------------- fixed-stack helpers

"The clip's effect of type `T`, or `nothing`."
function findeffect(clip::Clip, ::Type{T}) where {T <: Effect}
    i = findfirst(e -> e isa T, clip.effects)
    return i === nothing ? nothing : clip.effects[i]::T
end

"Replace the clip's effect of the same type, or append it."
function seteffect!(clip::Clip, e::Effect)
    i = findfirst(x -> typeof(x) == typeof(e), clip.effects)
    i === nothing ? push!(clip.effects, e) : (clip.effects[i] = e)
    return clip
end
