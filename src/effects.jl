"""
Non-destructive per-clip effects, interpreted in stack order by `graphof` and run
as a Mantle graph — the one renderer, whatever the tier. All pixel work is
GPUFiltering kernels through KernelAbstractions, and the buffers are transients
placed by the engine's Mantle device, so the same code runs on host memory under
`KA.CPU()` and on the GPU through Lava. There is no CPU stack and no GPU stack;
there is one graph and a backend parameter.
"""
abstract type FxOp end

struct ColorEffect <: FxOp
    adj::ColorAdjustments
end
ColorEffect(; kwargs...) = ColorEffect(ColorAdjustments(; kwargs...))

struct BlurEffect <: FxOp
    σ::Float32
end

struct SharpenEffect <: FxOp
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
struct MatteEffect <: FxOp
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
struct RestoreEffect <: FxOp
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
struct DepthBlurEffect <: FxOp
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
struct LookEffect <: FxOp
    strength::Float32
end
LookEffect(; strength = 1.0) = LookEffect(Float32(strength))

"Composite opacity: scales the frame toward black by `α` (1 = opaque). The main
use is a keyframed fade in/out; on a single track α<1 fades to black."
struct OpacityEffect <: FxOp
    α::Float32
end

"""
A loop search on this clip: the reference frames the user marked and the cut
points found from them (see the loop finder's card).

Renders NOTHING — `isneutral` is true, so the graph never sees it. It exists so
that a search has a card on the clip it searches, like everything else the editor
does to a clip.
"""
struct LoopFinderEffect <: FxOp end
isneutral(::LoopFinderEffect) = true

"""
The handle for a cross-dissolve into this clip: `seconds` is the shared length of
the two halves.

Also renders nothing — the fade itself is an `OpacityEffect` curve on each half,
and `Transition` is what the composite reads. This is the entry in the stack that
those belong to, and the slot a [`FxLink`](@ref) points at, so the two halves of
one blend can find each other.
"""
struct BlendEffect <: FxOp
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
struct StabilizeEffect <: FxOp end

"""
Applies the clip's colour/exposure stabilization (see `analyzecolor!`).

`strength` scales the correction toward identity, so the fix can be dialled back
— or keyframed — without re-analyzing. It lives here rather than on the track
because a parameter you tune belongs to the thing in the stack you tune it on.
"""
struct FlickerEffect <: FxOp
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
struct TransformEffect <: FxOp
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
neutral, in stack order. Disabling a slot ([`Effect`](@ref)`.enabled`) keeps its
parameters — the inspector's toggle is lossless — while every render path (CPU
stack, GPU graph, compositor, export) simply skips it.
"""
liveeffects(clip::Clip) = (op(s) for s in clip.effects if s.enabled && !isneutral(op(s)))

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

"""
    effectdict(fx) -> Dict

One effect as project-file data: its kind, and its PARAMETERS — each with its
value and, if it has one, its curve.

There is nothing type-specific here. This used to be thirteen hand-written
methods (`effectdict(e::BlurEffect) = ... "sigma" => e.σ ...`) mirrored by a
`t == "blur" && return BlurEffect(...)` chain on the way back, so adding an
effect meant writing its parameters down three times: once in its kind, once
out, once in. The parameters ARE the effect now, so writing them is writing it,
and a new kind needs no serialization code at all.
"""
function effectdict(fx::Effect)
    d = Dict{String, Any}("kind" => String(fx.kind),
                          "id" => string(fx.id),
                          "enabled" => fx.enabled,
                          "params" => [paramdict(p) for p in fx.params])
    isempty(fx.links) || (d["links"] = [linkdict(l) for l in fx.links])
    return d
end

"One parameter: what it is now, its lane, and its keys when it has any."
function paramdict(p::Param)
    d = Dict{String, Any}("name" => String(p.name), "value" => tojson(p.value))
    p.visible && (d["lane"] = true)
    isanimated(p) && (d["curve"] = Dict{String, Any}(
        "interp" => String(p.curve.interp),
        "frames" => [k.frame for k in p.curve.keys],
        "values" => [tojson(k.value) for k in p.curve.keys],
        "eases"  => [String(k.ease) for k in p.curve.keys]))
    return d
end

# A parameter value as JSON. Scalars go as themselves; anything with components
# goes as an array, so a `Vec3f` offset or an `RGBf` colour needs no special case
# on either side — `fromjson` puts it back into the type the parameter already
# has, which is the only place that type is known.
tojson(v::Real) = Float64(v)
tojson(v) = collect(Float64.(v))
tojson(v::Colorant) = [Float64(red(v)), Float64(green(v)), Float64(blue(v))]

fromjson(::Type{T}, v) where {T <: Real} = T(v)
fromjson(::Type{T}, v) where {T} = T(v...)
fromjson(::Type{T}, v) where {T <: Colorant} = T(v...)

"""
    effectfromdict(d) -> Effect

`effectdict`'s inverse. The kind supplies the parameter LIST (names, labels,
ranges and types); the file supplies the values and curves. A parameter the kind
no longer has is skipped, and one the file does not mention keeps its default —
so a kind may gain or lose parameters without stranding a project.
"""
function effectfromdict(d::AbstractDict)
    kind = kindbyname(Symbol(d["kind"]))
    kind === nothing && error("unknown effect kind: $(d["kind"])")
    fx = Effect(freshid(), Symbol(d["kind"]), get(d, "enabled", true),
                FxLink[linkfromdict(l) for l in get(d, "links", [])],
                paramsfor(kind))
    haskey(d, "id") && (fx = Effect(parse(UInt64, d["id"]), fx.kind, fx.enabled,
                                    fx.links, fx.params))
    for pd in get(d, "params", [])
        prm = param(fx, Symbol(pd["name"]))
        prm === nothing && continue          # a parameter this kind no longer has
        T = typeof(prm.value)
        haskey(pd, "value") && (prm.value = fromjson(T, pd["value"]))
        get(pd, "lane", false) && (prm.visible = true)
        haskey(pd, "curve") || continue
        cd = pd["curve"]
        eases = get(cd, "eases", fill("linear", length(cd["frames"])))
        prm.curve = AnimCurve{T}(
            [Keyframe(Int(f), fromjson(T, v), Symbol(e))
             for (f, v, e) in zip(cd["frames"], cd["values"], eases)],
            Symbol(get(cd, "interp", "linear")))
    end
    return fx
end


# ------------------------------------------------------- fixed-stack helpers

"The clip's effect of type `T`, or `nothing` — a DISABLED slot still answers, so
the inspector shows a switched-off effect's real parameters."
function findeffect(clip::Clip, ::Type{T}) where {T <: FxOp}
    i = findfirst(s -> op(s) isa T, clip.effects)
    return i === nothing ? nothing : op(clip.effects[i])::T
end

"The clip's slot holding an effect of type `T`, or `nothing`."
function findslot(clip::Clip, ::Type{T}) where {T <: FxOp}
    i = findfirst(s -> op(s) isa T, clip.effects)
    return i === nothing ? nothing : clip.effects[i]
end

# Effects upsert by kind: one entry per type — except plugin effects, which share
# a type, so they upsert per plugin name (see registry.jl). Writing a kind that is
# switched off replaces its effect AND switches it back on.
effectkey(e::FxOp) = typeof(e)

"""
    seteffect!(clip, e) -> clip

Replace the effect in the slot of the same kind (keeping that slot's id, so
anything pointing at it still points at it) or append a new slot.
"""
function seteffect!(clip::Clip, e::FxOp)
    i = findfirst(s -> effectkey(op(s)) == effectkey(e), clip.effects)
    if i === nothing
        push!(clip.effects, Effect(e))
    else
        setparams!(clip.effects[i], effectkindfor(e).read(e))
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
function prependeffect!(clip::Clip, e::FxOp)
    i = findfirst(s -> effectkey(op(s)) == effectkey(e), clip.effects)
    if i === nothing
        pushfirst!(clip.effects, Effect(e))
    else
        setparams!(clip.effects[i], effectkindfor(e).read(e))
        clip.effects[i].enabled = true
    end
    return clip
end

"Drop every slot holding an effect of type `T` (returns how many went)."
function removeeffects!(clip::Clip, ::Type{T}) where {T <: FxOp}
    n = count(s -> op(s) isa T, clip.effects)
    filter!(s -> !(op(s) isa T), clip.effects)
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
Each parameter declares how it reads from / writes to
a `Clip` (color and blur/sharpen live in the effect stack, opacity is an
`OpacityEffect`, pan/zoom are the crop rect).

These are seeded into [`EFFECTS`](@ref) at load; ask the registry
([`paramspecs`](@ref), [`paramspec`](@ref)) rather than this list, which is only
the built-in half. A registered kind's parameters get their specs generated —
these are hand-written because several of them (the crop rect, the placement)
are not an effect's fields at all.
"""

"Whether any registered parameter is keyframed on `clip`."
isanimated(clip::Clip) = any(fx -> any(isanimated, fx.params), clip.effects)
isanimated(fx::Effect) = any(isanimated, fx.params)

"`clip` without its matte — what the matte pipeline renders through, so the seed
is not computed from a frame the previous matte already keyed."
withoutmatte(clip::Clip) =
    withfields(clip; mattetrack = nothing,
               effects = filter(s -> !(op(s) isa MatteEffect), clip.effects))

"`clip` without its opacity effects — compositing reads opacity as the LAYER
alpha, not a per-pixel fade to black."
withoutopacity(clip::Clip) =
    withfields(clip; effects = filter(s -> !(op(s) isa OpacityEffect), clip.effects))

"""
    effectiveclip(clip, srcframe) -> Clip

The clip as it renders at absolute source frame `srcframe`: a shallow copy with
every animated parameter overridden by its curve's sampled value (via the
parameter's `set`). Returns `clip` unchanged when it has no animations — the
static fast path. The copy shares the source and analysis tracks; only `crop`
and a copied effect stack are mutated.
"""
function effectiveclip(clip::Clip, srcframe::Integer)
    isanimated(clip) || return clip
    # own entries (same ids, same on/off) so a sampled value never writes into
    # the clip the user is editing. The PARAMS are copied; the curves are shared,
    # because sampling never touches them.
    # `curves = false`: the copy IS the sampled state, so nothing may re-sample it
    ec = withfields(clip; effects = [copy(fx; curves = false) for fx in clip.effects])
    for (i, fx) in enumerate(clip.effects)
        any(isanimated, fx.params) || continue
        # THIS entry's parameters, sampled on THIS entry. Resolving a bare name
        # through a global index instead sent every curve to the FIRST effect of
        # its kind — two Blurs, one animated radius, and the second stayed at its
        # static value (measured: [(blur = 12.0,), (blur = 0.0,)]).
        for p in ec.effects[i].params
            src = param(fx, p.name)
            src === nothing || (p.value = valueat(src, srcframe))
        end
    end
    return ec
end
