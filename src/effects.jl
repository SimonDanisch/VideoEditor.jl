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

"""
Keys the frame against the clip's subject matte (see `analyzematte!`).

Holds no matte itself — the pixels live on the clip's `MatteTrack`, so this stays
a couple of scalars that the keyframe registry can animate like any other. That
is the point of the split: `strength` fading 0→1 is a keyframed reveal, and
`feather` softening an edge over time is a keyframed edge, both with no
re-analysis.
"""
struct MatteEffect <: Effect
    strength::Float32
    feather::Float32
end
MatteEffect(; strength = 1.0, feather = 0.0) = MatteEffect(Float32(strength), Float32(feather))

"""
Replaces the frame with a restoration model's output (see `restore.jl`).

Holds only `strength`, for the same reason `MatteEffect` does: the pixels live in
the clip's restore cache, so this stays a scalar the keyframe registry can
animate. Cross-fading it in is a legitimate edit — a restoration is a judgement
call, and half of one is often what you want.
"""
struct RestoreEffect <: Effect
    strength::Float32
end
RestoreEffect(; strength = 1.0) = RestoreEffect(Float32(strength))

"Composite opacity: scales the frame toward black by `α` (1 = opaque). The main
use is a keyframed fade in/out; on a single track α<1 fades to black."
struct OpacityEffect <: Effect
    α::Float32
end

isneutral(e::ColorEffect) = GPUFiltering.isneutral(e.adj)
isneutral(e::BlurEffect) = e.σ <= 0
isneutral(e::SharpenEffect) = e.amount <= 0
isneutral(e::OpacityEffect) = e.α >= 0.999f0
isneutral(e::MatteEffect) = e.strength <= 0.001f0
isneutral(e::RestoreEffect) = e.strength <= 0.001f0

"""
    liveeffects(clip)

The effects of `clip` that actually render: enabled slots whose effect isn't
neutral, in stack order. Disabling a slot ([`FxSlot`](@ref)`.enabled`) keeps its
parameters — the inspector's toggle is lossless — while every render path (CPU
stack, GPU graph, compositor, export) simply skips it.
"""
liveeffects(clip::Clip) = (s.effect for s in clip.effects if s.enabled && !isneutral(s.effect))

"The slot with `id` on `clip`, or `nothing` — how anything points at ONE entry."
function findslot(clip::Clip, id::Integer)
    i = findfirst(s -> s.id == id, clip.effects)
    return i === nothing ? nothing : clip.effects[i]
end

"Apply `clip`'s effect stack to `buf` in place, using two same-size scratch buffers."
function applyeffects!(buf::AnyRGBFrame, tmp1::AnyRGBFrame, tmp2::AnyRGBFrame, clip::Clip)
    for e in liveeffects(clip)
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
effectdict(e::RestoreEffect) =
    Dict{String, Any}("type" => "restore", "strength" => e.strength)
effectdict(e::MatteEffect) =
    Dict{String, Any}("type" => "matte", "strength" => e.strength, "feather" => e.feather)

"A stack entry as a project-file dict: the effect plus its id and enabled state."
slotdict(s::FxSlot) = merge(effectdict(s.effect),
                            Dict{String, Any}("id" => string(s.id), "enabled" => s.enabled))

"""
Read a stack entry back. Files written before effects had ids (and before
`enabled` replaced the `bypassed` wrapper) still load: the entry gets a fresh id,
and a wrapped effect becomes a disabled slot.
"""
function slotfromdict(d::AbstractDict)
    if d["type"] == "bypassed"
        return FxSlot(effectfromdict(d["inner"]); enabled = false)
    end
    id = haskey(d, "id") ? parse(UInt64, d["id"]) : freshid()
    return FxSlot(id, effectfromdict(d), get(d, "enabled", true))
end

function effectfromdict(d::AbstractDict)
    t = d["type"]
    t == "bypassed" && return effectfromdict(d["inner"])
    t == "color" && return ColorEffect(; brightness = d["brightness"], contrast = d["contrast"],
                                       saturation = d["saturation"], temperature = d["temperature"])
    t == "blur" && return BlurEffect(Float32(d["sigma"]))
    t == "sharpen" && return SharpenEffect(Float32(d["sigma"]), Float32(d["amount"]))
    t == "opacity" && return OpacityEffect(Float32(d["alpha"]))
    t == "restore" && return RestoreEffect(Float32(d["strength"]))
    t == "matte" && return MatteEffect(Float32(d["strength"]), Float32(get(d, "feather", 0.0)))
    t == "plugin" && return plugineffectfromdict(d)   # requires the plugin registered
    error("unknown effect type: $t")
end

# ------------------------------------------------------- fixed-stack helpers

"The clip's effect of type `T`, or `nothing` — a DISABLED slot still answers, so
the inspector shows a switched-off effect's real parameters."
function findeffect(clip::Clip, ::Type{T}) where {T <: Effect}
    i = findfirst(s -> s.effect isa T, clip.effects)
    return i === nothing ? nothing : clip.effects[i].effect::T
end

"The clip's slot holding an effect of type `T`, or `nothing`."
function findslot(clip::Clip, ::Type{T}) where {T <: Effect}
    i = findfirst(s -> s.effect isa T, clip.effects)
    return i === nothing ? nothing : clip.effects[i]
end

# Effects upsert by kind: one entry per type — except plugin effects, which share
# a type, so they upsert per plugin name (see plugins.jl). Writing a kind that is
# switched off replaces its effect AND switches it back on.
effectkey(e::Effect) = typeof(e)

"""
    seteffect!(clip, e) -> clip

Replace the effect in the slot of the same kind (keeping that slot's id, so
anything pointing at it still points at it) or append a new slot.
"""
function seteffect!(clip::Clip, e::Effect)
    i = findfirst(s -> effectkey(s.effect) == effectkey(e), clip.effects)
    if i === nothing
        push!(clip.effects, FxSlot(e))
    else
        clip.effects[i].effect = e
        clip.effects[i].enabled = true
    end
    return clip
end

"Drop the slot with `id` (returns whether one went)."
function removeslot!(clip::Clip, id::Integer)
    i = findfirst(s -> s.id == id, clip.effects)
    i === nothing && return false
    deleteat!(clip.effects, i)
    return true
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
    ParamSpec(:restore_strength, "Restore", :restore, 0.0, 1.0, 1.0,
        c -> (e = findeffect(c, RestoreEffect); e === nothing ? 1.0 : Float64(e.strength)),
        (c, v) -> seteffect!(c, RestoreEffect(Float32(v)))),
    ParamSpec(:matte_strength, "Matte", :matte, 0.0, 1.0, 1.0,
        c -> (e = findeffect(c, MatteEffect); e === nothing ? 1.0 : Float64(e.strength)),
        (c, v) -> (e = findeffect(c, MatteEffect);
                   seteffect!(c, MatteEffect(Float32(v), e === nothing ? 0.0f0 : e.feather)))),
    ParamSpec(:matte_feather, "Feather", :matte, 0.0, 1.0, 0.0,
        c -> (e = findeffect(c, MatteEffect); e === nothing ? 0.0 : Float64(e.feather)),
        (c, v) -> (e = findeffect(c, MatteEffect);
                   seteffect!(c, MatteEffect(e === nothing ? 1.0f0 : e.strength, Float32(v))))),
    ParamSpec(:crop_x, "Pan X", :geometry, 0.0, 1.0, 0.0,
        c -> c.crop[1], (c, v) -> (c.crop = (clampunit(v), c.crop[2], c.crop[3], c.crop[4]))),
    ParamSpec(:crop_y, "Pan Y", :geometry, 0.0, 1.0, 0.0,
        c -> c.crop[2], (c, v) -> (c.crop = (c.crop[1], clampunit(v), c.crop[3], c.crop[4]))),
    ParamSpec(:crop_w, "Zoom W", :geometry, 0.05, 1.0, 1.0,
        c -> c.crop[3], (c, v) -> (c.crop = (c.crop[1], c.crop[2], clamp(Float64(v), 0.05, 1.0), c.crop[4]))),
    ParamSpec(:crop_h, "Zoom H", :geometry, 0.05, 1.0, 1.0,
        c -> c.crop[4], (c, v) -> (c.crop = (c.crop[1], c.crop[2], c.crop[3], clamp(Float64(v), 0.05, 1.0)))),
    # …and where the cropped picture SITS in the canvas. The fit is automatic
    # (whole, centred, black bars); these three are the manual override for
    # material that doesn't share the sequence's shape — 1.0 fits, >1 fills past
    # the edges, and the shift moves it inside the frame.
    ParamSpec(:scale, "Scale", :geometry, 0.1, 4.0, 1.0,
        c -> c.reframe[1],
        (c, v) -> (c.reframe = (clamp(Float64(v), 0.1, 4.0), c.reframe[2], c.reframe[3]))),
    ParamSpec(:pos_x, "Position X", :geometry, -1.0, 1.0, 0.0,
        c -> c.reframe[2],
        (c, v) -> (c.reframe = (c.reframe[1], clamp(Float64(v), -1.0, 1.0), c.reframe[3]))),
    ParamSpec(:pos_y, "Position Y", :geometry, -1.0, 1.0, 0.0,
        c -> c.reframe[3],
        (c, v) -> (c.reframe = (c.reframe[1], c.reframe[2], clamp(Float64(v), -1.0, 1.0)))),
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

"`clip` without its opacity effects — compositing reads opacity as the LAYER
alpha, not a per-pixel fade to black."
withoutopacity(clip::Clip) =
    Clip(clip.id, clip.source, clip.src_in, clip.src_out, clip.start, clip.crop,
         filter(s -> !(s.effect isa OpacityEffect), clip.effects),
         clip.colortrack, clip.motiontrack, clip.mattetrack, clip.animations, clip.track,
              clip.blendfrom, clip.rate, clip.reframe)

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
    # own slots (same ids, same on/off) so a sampled value never writes into the
    # clip the user is editing
    ec = Clip(clip.id, clip.source, clip.src_in, clip.src_out, clip.start, clip.crop,
              [FxSlot(s.id, s.effect, s.enabled) for s in clip.effects],
              clip.colortrack, clip.motiontrack, clip.mattetrack, clip.animations, clip.track,
              clip.blendfrom, clip.rate, clip.reframe)
    for (key, curve) in clip.animations
        haskey(PARAMBYKEY, key) || continue
        v = valueat(curve, srcframe)
        v === nothing || paramspec(key).set(ec, v)
    end
    return ec
end
