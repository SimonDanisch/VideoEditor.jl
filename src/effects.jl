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

"Composite opacity: scales the frame toward black by `α` (1 = opaque). The main
use is a keyframed fade in/out; on a single track α<1 fades to black."
struct OpacityEffect <: Effect
    α::Float32
end

isneutral(e::ColorEffect) = GPUFiltering.isneutral(e.adj)
isneutral(e::BlurEffect) = e.σ <= 0
isneutral(e::SharpenEffect) = e.amount <= 0
isneutral(e::OpacityEffect) = e.α >= 0.999f0

"""
    Bypassed(effect)

A disabled effect in the stack: it is `isneutral`, so every render path (the
CPU stack, the GPU graph, the compositor, export) skips it — but it keeps its
parameters, so the inspector's enable toggle switches it back on losslessly.
"""
struct Bypassed <: Effect
    e::Effect
end
isneutral(::Bypassed) = true
"The effect itself, seen through a [`Bypassed`](@ref) wrapper."
uneffect(e::Effect) = e
uneffect(b::Bypassed) = b.e

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

# Effects with a specialized/multi-pass kernel apply it directly.
applyeffect!(buf, tmp1, tmp2, e::ColorEffect) = coloradjust!(buf, e.adj)
applyeffect!(buf, tmp1, tmp2, e::BlurEffect) =
    (gaussianblur!(tmp1, buf, e.σ; tmp = tmp2); copyto!(buf, tmp1))
applyeffect!(buf, tmp1, tmp2, e::SharpenEffect) =
    (unsharpmask!(tmp1, buf, e.σ, e.amount; tmp = tmp2); copyto!(buf, tmp1))
# Callback effects (`fxkind`, incl. plugins) run the SAME kernel the GPU graph uses.
applyeffect!(buf, tmp1, tmp2, e::Effect) = applykindcpu!(buf, tmp1, fxkind(e))

# ------------------------------------------------------------- serialization

effectdict(e::ColorEffect) = Dict{String, Any}(
    "type" => "color", "brightness" => e.adj.brightness, "contrast" => e.adj.contrast,
    "saturation" => e.adj.saturation, "temperature" => e.adj.temperature)
effectdict(e::BlurEffect) = Dict{String, Any}("type" => "blur", "sigma" => e.σ)
effectdict(e::SharpenEffect) =
    Dict{String, Any}("type" => "sharpen", "sigma" => e.σ, "amount" => e.amount)
effectdict(e::OpacityEffect) = Dict{String, Any}("type" => "opacity", "alpha" => e.α)
effectdict(b::Bypassed) = Dict{String, Any}("type" => "bypassed", "inner" => effectdict(b.e))

function effectfromdict(d::AbstractDict)
    t = d["type"]
    t == "bypassed" && return Bypassed(effectfromdict(d["inner"]))
    t == "color" && return ColorEffect(; brightness = d["brightness"], contrast = d["contrast"],
                                       saturation = d["saturation"], temperature = d["temperature"])
    t == "blur" && return BlurEffect(Float32(d["sigma"]))
    t == "sharpen" && return SharpenEffect(Float32(d["sigma"]), Float32(d["amount"]))
    t == "opacity" && return OpacityEffect(Float32(d["alpha"]))
    t == "plugin" && return plugineffectfromdict(d)   # requires the plugin registered
    error("unknown effect type: $t")
end

# ------------------------------------------------------- fixed-stack helpers

"The clip's effect of type `T`, or `nothing`."
function findeffect(clip::Clip, ::Type{T}) where {T <: Effect}
    i = findfirst(e -> e isa T, clip.effects)
    return i === nothing ? nothing : clip.effects[i]::T
end

# Effects upsert by identity: one instance per type — except plugin effects, which
# share a type, so they upsert per plugin name (see plugins.jl). A bypassed
# effect keeps its inner key, so writing that kind replaces (and re-enables) it.
effectkey(e::Effect) = typeof(e)
effectkey(b::Bypassed) = effectkey(b.e)

"Replace the clip's effect with the same key, or append it."
function seteffect!(clip::Clip, e::Effect)
    i = findfirst(x -> effectkey(x) == effectkey(e), clip.effects)
    i === nothing ? push!(clip.effects, e) : (clip.effects[i] = e)
    return clip
end

# ---------------------------------------------------- animatable parameters

clampunit(v::Real) = clamp(Float64(v), 0.0, 1.0)
curadj(clip::Clip) = (e = findeffect(clip, ColorEffect); e === nothing ? ColorAdjustments() : e.adj)
withcolor(clip::Clip, adj::ColorAdjustments) = seteffect!(clip, ColorEffect(adj))

"""
The animatable-parameter registry — the single table the keyframe engine and the
editor iterate over. Each [`ParamSpec`](@ref) declares how one named parameter
reads from / writes to a `Clip` (color and blur/sharpen live in the effect stack,
opacity is an `OpacityEffect`, pan/zoom are the crop rect). Add a row here and the
parameter is immediately keyframeable and shows up in the editor — nothing else
in the pipeline needs to know about it.
"""
const PARAMS = ParamSpec[
    ParamSpec(:opacity, "Opacity", :composite, 0.0, 1.0, 1.0,
        c -> (e = findeffect(c, OpacityEffect); e === nothing ? 1.0 : Float64(e.α)),
        (c, v) -> seteffect!(c, OpacityEffect(Float32(v)))),
    ParamSpec(:brightness, "Brightness", :color, -0.5, 0.5, 0.0,
        c -> Float64(curadj(c).brightness),
        (c, v) -> withcolor(c, ColorAdjustments(Float32(v), curadj(c).contrast, curadj(c).saturation, curadj(c).temperature))),
    ParamSpec(:contrast, "Contrast", :color, 0.0, 2.0, 1.0,
        c -> Float64(curadj(c).contrast),
        (c, v) -> withcolor(c, ColorAdjustments(curadj(c).brightness, Float32(v), curadj(c).saturation, curadj(c).temperature))),
    ParamSpec(:saturation, "Saturation", :color, 0.0, 2.0, 1.0,
        c -> Float64(curadj(c).saturation),
        (c, v) -> withcolor(c, ColorAdjustments(curadj(c).brightness, curadj(c).contrast, Float32(v), curadj(c).temperature))),
    ParamSpec(:temperature, "Temperature", :color, -1.0, 1.0, 0.0,
        c -> Float64(curadj(c).temperature),
        (c, v) -> withcolor(c, ColorAdjustments(curadj(c).brightness, curadj(c).contrast, curadj(c).saturation, Float32(v)))),
    ParamSpec(:blur, "Blur", :blur, 0.0, 12.0, 0.0,
        c -> (e = findeffect(c, BlurEffect); e === nothing ? 0.0 : Float64(e.σ)),
        (c, v) -> seteffect!(c, BlurEffect(Float32(v)))),
    ParamSpec(:sharpen, "Sharpen", :sharpen, 0.0, 2.0, 0.0,
        c -> (e = findeffect(c, SharpenEffect); e === nothing ? 0.0 : Float64(e.amount)),
        (c, v) -> seteffect!(c, SharpenEffect(2.0f0, Float32(v)))),
    ParamSpec(:crop_x, "Pan X", :geometry, 0.0, 1.0, 0.0,
        c -> c.crop[1], (c, v) -> (c.crop = (clampunit(v), c.crop[2], c.crop[3], c.crop[4]))),
    ParamSpec(:crop_y, "Pan Y", :geometry, 0.0, 1.0, 0.0,
        c -> c.crop[2], (c, v) -> (c.crop = (c.crop[1], clampunit(v), c.crop[3], c.crop[4]))),
    ParamSpec(:crop_w, "Zoom W", :geometry, 0.05, 1.0, 1.0,
        c -> c.crop[3], (c, v) -> (c.crop = (c.crop[1], c.crop[2], clamp(Float64(v), 0.05, 1.0), c.crop[4]))),
    ParamSpec(:crop_h, "Zoom H", :geometry, 0.05, 1.0, 1.0,
        c -> c.crop[4], (c, v) -> (c.crop = (c.crop[1], c.crop[2], c.crop[3], clamp(Float64(v), 0.05, 1.0)))),
]
const PARAMBYKEY = Dict(p.key => p for p in PARAMS)

"The [`ParamSpec`](@ref) for `key` (throws if unknown)."
paramspec(key::Symbol) = PARAMBYKEY[key]

# A stable, visually distinct color per animatable parameter — shared by its keyframe
# curve and its ◆ toggle so a parameter's control and its line are easy to match.
const PARAMPALETTE = map(Makie.to_color,
    ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
     "#B279A2", "#FF9DA6", "#9D755D", "#5C6BC0", "#26A69A", "#8E24AA"])

"A distinct, stable display color for parameter `key` (by its registry position)."
function paramcolor(key::Symbol)
    i = findfirst(p -> p.key == key, PARAMS)
    i === nothing && (i = abs(hash(key)) % length(PARAMPALETTE) + 1)
    return PARAMPALETTE[mod1(i, length(PARAMPALETTE))]
end

"Whether any registered parameter is keyframed on `clip`."
isanimated(clip::Clip) = !isempty(clip.animations)

"""
    effectiveclip(clip, srcframe) -> Clip

The clip as it renders at absolute source frame `srcframe`: a shallow copy with
every animated parameter overridden by its curve's sampled value (via the
parameter's `set`). Returns `clip` unchanged when it has no animations — the
static fast path. The copy shares the source and analysis tracks; only `crop`
and a copied effect stack are mutated.
"""
function effectiveclip(clip::Clip, srcframe::Integer)
    isempty(clip.animations) && return clip
    ec = Clip(clip.source, clip.src_in, clip.src_out, clip.start, clip.crop,
              copy(clip.effects), clip.colortrack, clip.motiontrack, clip.animations, clip.track)
    for (key, curve) in clip.animations
        haskey(PARAMBYKEY, key) || continue
        v = valueat(curve, srcframe)
        v === nothing || paramspec(key).set(ec, v)
    end
    return ec
end
