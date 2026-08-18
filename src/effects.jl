"""
Non-destructive per-clip effects, interpreted in stack order by `graphof` and run
as a Mantle graph — the one renderer, whatever the tier. All pixel work is
GPUFiltering kernels through KernelAbstractions, and the buffers are transients
placed by the engine's Mantle device, so the same code runs on host memory under
`KA.CPU()` and on the GPU through Lava. There is no CPU stack and no GPU stack;
there is one graph and a backend parameter.
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

"""
Defocus the background against estimated depth (see `depth.jl`).

`focus` is the depth that stays sharp (0 = farthest, 1 = nearest) and `strength`
how much everything else softens. Both are keyframable, which is the point of
making this an effect rather than a bake: a rack focus is a `focus` curve.

Declines with no depth track, exactly as the matte declines with no matte — the
analysis is the expensive part and the effect is free to sit in the stack
waiting for it.
"""
struct DepthBlurEffect <: Effect
    focus::Float32
    strength::Float32
end
DepthBlurEffect(; focus = 1.0, strength = 0.6) =
    DepthBlurEffect(Float32(focus), Float32(strength))

"""
Applies the clip's learned colour grade (see `look.jl`).

Holds only `strength`, like `MatteEffect` and `RestoreEffect`: the table lives on
the clip, so this stays a scalar the keyframe registry can animate. Dialling a
grade in over a few frames is a real edit, and re-predicting to do it would be
absurd — the look is the same, the amount of it is what changes.
"""
struct LookEffect <: Effect
    strength::Float32
end
LookEffect(; strength = 1.0) = LookEffect(Float32(strength))

"Composite opacity: scales the frame toward black by `α` (1 = opaque). The main
use is a keyframed fade in/out; on a single track α<1 fades to black."
struct OpacityEffect <: Effect
    α::Float32
end

"""
A loop search on this clip: the reference frames the user marked and the cut
points found from them (see the loop finder's card).

Renders NOTHING — `isneutral` is true, so the graph never sees it. It exists so
that a search has a card on the clip it searches, like everything else the editor
does to a clip.
"""
struct LoopFinderEffect <: Effect end
isneutral(::LoopFinderEffect) = true

"""
The handle for a cross-dissolve into this clip: `seconds` is the shared length of
the two halves.

Also renders nothing — the fade itself is an `OpacityEffect` curve on each half,
and `Transition` is what the composite reads. This is the entry in the stack that
those belong to, and the slot a [`FxLink`](@ref) points at, so the two halves of
one blend can find each other.
"""
struct BlendEffect <: Effect
    seconds::Float64
end
BlendEffect(; seconds = 0.6) = BlendEffect(Float64(seconds))
isneutral(::BlendEffect) = true

"""
Applies the clip's camera stabilization (see `analyzemotion!`).

Holds nothing: the per-frame warps live on the clip's `MotionTrack`, for the same
reason `MatteEffect` holds no pixels. What it adds is a PLACE IN THE STACK — the
analysis used to be applied ahead of every effect by the graph builder, which
meant there was no card to fold, no toggle to compare with, and no way to say
"stabilize the cropped picture, not the raw one".
"""
struct StabilizeEffect <: Effect end

"""
Applies the clip's colour/exposure stabilization (see `analyzecolor!`).

`strength` scales the correction toward identity, so the fix can be dialled back
— or keyframed — without re-analyzing. It lives here rather than on the track
because a parameter you tune belongs to the thing in the stack you tune it on.
"""
struct FlickerEffect <: Effect
    strength::Float32
end
FlickerEffect(; strength = 1.0) = FlickerEffect(Float32(strength))

isneutral(e::ColorEffect) = GPUFiltering.isneutral(e.adj)
isneutral(e::BlurEffect) = e.σ <= 0
isneutral(e::SharpenEffect) = e.amount <= 0
isneutral(e::OpacityEffect) = e.α >= 0.999f0
"""
Where the clip sits on the canvas: scale, position and rotation on top of the
automatic fit.

An EFFECT, so it is added, folded, toggled, removed and keyframed like every
other one — and so the transform gizmo has a card to belong to. It holds no
pixels; `layermatrix` reads it when it places the layer, which is why there is no
`TransformNode` in the graph.
"""
struct TransformEffect <: Effect
    scale::Float64
    x::Float64
    y::Float64
    rotation::Float64      # degrees, positive = clockwise on screen
end
TransformEffect(; scale = 1.0, x = 0.0, y = 0.0, rotation = 0.0) =
    TransformEffect(Float64(scale), Float64(x), Float64(y), Float64(rotation))

isneutral(e::TransformEffect) =
    e.scale ≈ 1.0 && e.x == 0.0 && e.y == 0.0 && e.rotation == 0.0

"""
    transformof(clip) -> (scale, x, y, rotation°)

The clip's placement. ONE reader, so "where is this clip" has one answer: the
`TransformEffect` when it has one, the identity fit otherwise.
"""
transformof(clip::Clip) =
    (e = findeffect(clip, TransformEffect);
     e === nothing ? NEUTRALFRAME : (e.scale, e.x, e.y, e.rotation))

"Write one component of the clip's placement, creating the effect if needed."
function settransform(clip::Clip; scale = nothing, x = nothing, y = nothing, rotation = nothing)
    s, px, py, r = transformof(clip)
    seteffect!(clip, TransformEffect(scale === nothing ? s : clamp(Float64(scale), 0.1, 4.0),
                                     x === nothing ? px : clamp(Float64(x), -1.0, 1.0),
                                     y === nothing ? py : clamp(Float64(y), -1.0, 1.0),
                                     rotation === nothing ? r : clamp(Float64(rotation), -180.0, 180.0)))
    return nothing
end

isneutral(e::MatteEffect) = e.strength <= 0.001f0
isneutral(e::FlickerEffect) = e.strength <= 0.001f0
isneutral(::StabilizeEffect) = false   # the warp is either applied or the slot is off
isneutral(e::RestoreEffect) = e.strength <= 0.001f0
isneutral(e::DepthBlurEffect) = e.strength <= 0.001f0
isneutral(e::LookEffect) = e.strength <= 0.001f0

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

# There is no second walker over the stack here. `runchain!` (gpugraph.jl) is
# the ONE renderer, and it is not GPU-specific: the engine's Mantle device backs
# its transients with host memory on `KA.CPU()` and VRAM through Lava, and the
# passes run the identical kernels either way. The tier is the engine's backend,
# a parameter — never a second code path.
#
# There used to be one anyway: `applyeffects!` walked `liveeffects` with two
# scratch buffers, plus `applyeffect!` methods that re-stated what Color, Blur
# and Sharpen do, plus an `applykindcpu!` that differed from `applykind!` only in
# buffer discipline. Nothing in `src/` ever called it — the whole thing was dead,
# kept alive by three tests, so the suite was exercising a second definition of
# every built-in that shipped to nobody. Deleted 2026-08-07.

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
effectdict(e::DepthBlurEffect) =
    Dict{String, Any}("type" => "depthblur", "focus" => e.focus, "strength" => e.strength)
effectdict(e::LookEffect) = Dict{String, Any}("type" => "look", "strength" => e.strength)
effectdict(e::TransformEffect) =
    Dict{String, Any}("type" => "transform", "scale" => e.scale, "x" => e.x, "y" => e.y,
                      "rotation" => e.rotation)
effectdict(e::MatteEffect) =
    Dict{String, Any}("type" => "matte", "strength" => e.strength, "feather" => e.feather)
effectdict(::StabilizeEffect) = Dict{String, Any}("type" => "stabilize")
effectdict(::LoopFinderEffect) = Dict{String, Any}("type" => "loopfinder")
effectdict(e::BlendEffect) = Dict{String, Any}("type" => "blend", "seconds" => e.seconds)
effectdict(e::FlickerEffect) = Dict{String, Any}("type" => "flicker", "strength" => e.strength)

"A stack entry as a project-file dict: the effect plus its id, enabled state and links."
function slotdict(s::FxSlot)
    d = merge(effectdict(s.effect),
              Dict{String, Any}("id" => string(s.id), "enabled" => s.enabled))
    isempty(s.links) || (d["links"] = [linkdict(l) for l in s.links])
    return d
end

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
    links = FxLink[linkfromdict(l) for l in get(d, "links", [])]
    return FxSlot(id, effectfromdict(d), get(d, "enabled", true), links)
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
    t == "depthblur" && return DepthBlurEffect(Float32(get(d, "focus", 1.0)),
                                               Float32(get(d, "strength", 0.6)))
    t == "look" && return LookEffect(Float32(get(d, "strength", 1.0)))
    t == "matte" && return MatteEffect(Float32(d["strength"]), Float32(get(d, "feather", 0.0)))
    t == "stabilize" && return StabilizeEffect()
    t == "loopfinder" && return LoopFinderEffect()
    t == "blend" && return BlendEffect(Float64(get(d, "seconds", 0.6)))
    t == "flicker" && return FlickerEffect(Float32(get(d, "strength", 1.0)))
    t == "transform" && return TransformEffect(scale = get(d, "scale", 1.0), x = get(d, "x", 0.0),
                                               y = get(d, "y", 0.0), rotation = get(d, "rotation", 0.0))
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
# a type, so they upsert per plugin name (see registry.jl). Writing a kind that is
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

"""
    prependeffect!(clip, e) -> clip

Put `e` at the FRONT of the stack, or update the existing slot of its kind in
place (keeping its id, so anything pointing at it still does). This is where an
ANALYSIS lands: stabilizing the raw picture and then colour-grading it is the
order that was hard-wired into the graph builder before analyses had slots, so it
stays the default — the user can drag it elsewhere afterwards.
"""
function prependeffect!(clip::Clip, e::Effect)
    i = findfirst(s -> effectkey(s.effect) == effectkey(e), clip.effects)
    if i === nothing
        pushfirst!(clip.effects, FxSlot(e))
    else
        clip.effects[i].effect = e
        clip.effects[i].enabled = true
    end
    return clip
end

"Drop every slot holding an effect of type `T` (returns how many went)."
function removeeffects!(clip::Clip, ::Type{T}) where {T <: Effect}
    n = count(s -> s.effect isa T, clip.effects)
    filter!(s -> !(s.effect isa T), clip.effects)
    return n
end

"""
    setmotiontrack!(clip, track)
    setcolortrack!(clip, track)

Attach (or clear) an analysis AND the stack slot that applies it, together.

Five different analyses produce a `MotionTrack`; every one goes through here, so
"the clip is stabilized" and "the panel shows a Stabilize card" can never
disagree. `nothing` removes both.
"""
function setmotiontrack!(clip::Clip, track)
    clip.motiontrack = track
    track === nothing ? removeeffects!(clip, StabilizeEffect) :
                        prependeffect!(clip, StabilizeEffect())
    return track
end

function setcolortrack!(clip::Clip, track)
    clip.colortrack = track
    if track === nothing
        removeeffects!(clip, FlickerEffect)
    else
        prependeffect!(clip, FlickerEffect(track.strength))
    end
    return track
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

"""
How far outside the source a crop rect may reach, as a fraction of the source.

`1.0` means the rect can start a full frame-width left of the picture and end a
full frame-width right of it, so a canvas can be grown to three times the
source's size in each axis before a bound is hit. A bound exists at all only
because these are sliders and a slider needs ends; the crop TOOL is not limited
by it, since dragging a rectangle states the size directly.

Not unbounded, and not a "canvas size" property either: the rect IS the canvas,
so growing it is the same gesture as shrinking it — which is what makes "crop
outward" mean "make the project bigger" without a second concept.
"""
const CROPREACH = 1.0

"A crop origin: outside the picture is allowed, absurdly far outside is not."
cropunit(v::Real) = clamp(Float64(v), -CROPREACH, 1.0 + CROPREACH)

"A crop extent: never degenerate, and free to exceed the source."
cropextent(v::Real) = clamp(Float64(v), 0.05, 1.0 + 2CROPREACH)
curadj(clip::Clip) = (e = findeffect(clip, ColorEffect); e === nothing ? ColorAdjustments() : e.adj)
withcolor(clip::Clip, adj::ColorAdjustments) = seteffect!(clip, ColorEffect(adj))

"""
The animatable parameters of the built-in effects and of a clip's placement.
Each [`ParamSpec`](@ref) declares how one named parameter reads from / writes to
a `Clip` (color and blur/sharpen live in the effect stack, opacity is an
`OpacityEffect`, pan/zoom are the crop rect).

These are seeded into [`EFFECTS`](@ref) at load; ask the registry
([`paramspecs`](@ref), [`paramspec`](@ref)) rather than this list, which is only
the built-in half. A registered kind's parameters get their specs generated —
these are hand-written because several of them (the crop rect, the placement)
are not an effect's fields at all.
"""
const BUILTINPARAMS = ParamSpec[
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
    ParamSpec(:depth_focus, "Focus", :depth, 0.0, 1.0, 1.0,
        c -> (e = findeffect(c, DepthBlurEffect); e === nothing ? 1.0 : Float64(e.focus)),
        (c, v) -> (e = findeffect(c, DepthBlurEffect);
                   seteffect!(c, DepthBlurEffect(Float32(v),
                                                 e === nothing ? 0.6f0 : e.strength)))),
    ParamSpec(:depth_strength, "Defocus", :depth, 0.0, 1.0, 0.6,
        c -> (e = findeffect(c, DepthBlurEffect); e === nothing ? 0.6 : Float64(e.strength)),
        (c, v) -> (e = findeffect(c, DepthBlurEffect);
                   seteffect!(c, DepthBlurEffect(e === nothing ? 1.0f0 : e.focus,
                                                 Float32(v))))),
    ParamSpec(:look_strength, "Look", :look, 0.0, 1.0, 1.0,
        c -> (e = findeffect(c, LookEffect); e === nothing ? 1.0 : Float64(e.strength)),
        (c, v) -> seteffect!(c, LookEffect(Float32(v)))),
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
    # The crop rect may sit OUTSIDE the picture, which is how the canvas grows:
    # the rect is what gets rendered, `canvassize` is its size in source pixels,
    # and the warp leaves whatever it cannot reach as background. So the ranges
    # run past the image rather than stopping at its edges — clamped to `0..1`
    # these sliders could only ever shrink a project, and there was no way at all
    # to make one taller. `CROPREACH` is how far past the edge they go.
    ParamSpec(:crop_x, "Pan X", :geometry, -CROPREACH, 1.0 + CROPREACH, 0.0,
        c -> c.crop[1], (c, v) -> (c.crop = (cropunit(v), c.crop[2], c.crop[3], c.crop[4]))),
    ParamSpec(:crop_y, "Pan Y", :geometry, -CROPREACH, 1.0 + CROPREACH, 0.0,
        c -> c.crop[2], (c, v) -> (c.crop = (c.crop[1], cropunit(v), c.crop[3], c.crop[4]))),
    ParamSpec(:crop_w, "Zoom W", :geometry, 0.05, 1.0 + 2CROPREACH, 1.0,
        c -> c.crop[3], (c, v) -> (c.crop = (c.crop[1], c.crop[2], cropextent(v), c.crop[4]))),
    ParamSpec(:crop_h, "Zoom H", :geometry, 0.05, 1.0 + 2CROPREACH, 1.0,
        c -> c.crop[4], (c, v) -> (c.crop = (c.crop[1], c.crop[2], c.crop[3], cropextent(v)))),
    # …and where the cropped picture SITS in the canvas. The fit is automatic
    # (whole, centred, black bars); these three are the manual override for
    # material that doesn't share the sequence's shape — 1.0 fits, >1 fills past
    # the edges, and the shift moves it inside the frame.
    ParamSpec(:scale, "Scale", :geometry, 0.1, 4.0, 1.0,
        c -> transformof(c)[1], (c, v) -> settransform(c; scale = v)),
    ParamSpec(:pos_x, "Position X", :geometry, -1.0, 1.0, 0.0,
        c -> transformof(c)[2], (c, v) -> settransform(c; x = v)),
    ParamSpec(:pos_y, "Position Y", :geometry, -1.0, 1.0, 0.0,
        c -> transformof(c)[3], (c, v) -> settransform(c; y = v)),
    ParamSpec(:rotation, "Rotation", :geometry, -180.0, 180.0, 0.0,
        c -> transformof(c)[4], (c, v) -> settransform(c; rotation = v)),
]

# A stable, visually distinct color per animatable parameter — shared by its keyframe
# curve and its ◆ toggle so a parameter's control and its line are easy to match.
const PARAMPALETTE = map(Makie.to_color,
    ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
     "#B279A2", "#FF9DA6", "#9D755D", "#5C6BC0", "#26A69A", "#8E24AA"])

"A distinct, stable display color for parameter `key` (by its registry position)."
function paramcolor(key::Symbol)
    specs = paramspecs()
    i = findfirst(p -> p.key == key, specs)
    i === nothing && (i = abs(hash(key)) % length(PARAMPALETTE) + 1)
    return PARAMPALETTE[mod1(i, length(PARAMPALETTE))]
end

"Whether any registered parameter is keyframed on `clip`."
isanimated(clip::Clip) = !isempty(clip.animations)

"`clip` without its matte — what the matte pipeline renders through, so the seed
is not computed from a frame the previous matte already keyed."
withoutmatte(clip::Clip) =
    withfields(clip; mattetrack = nothing,
               effects = filter(s -> !(s.effect isa MatteEffect), clip.effects))

"`clip` without its opacity effects — compositing reads opacity as the LAYER
alpha, not a per-pixel fade to black."
withoutopacity(clip::Clip) =
    withfields(clip; effects = filter(s -> !(s.effect isa OpacityEffect), clip.effects))

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
    ec = withfields(clip;
                    effects = [FxSlot(s.id, s.effect, s.enabled, s.links) for s in clip.effects])
    for (key, curve) in clip.animations
        # a project can hold a curve for a parameter this session has no effect
        # registered for — skip it rather than fail the render
        spec = paramspec(key, nothing)
        spec === nothing && continue
        v = valueat(curve, srcframe)
        v === nothing || spec.set(ec, v)
    end
    return ec
end
