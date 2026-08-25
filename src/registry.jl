# ONE registry for everything the editor can put on a clip.
#
# There used to be three descriptors for one idea. `EffectKind` described the
# built-ins, `FxPlugin` described registered effects, and `EditorTool` described
# what the Tools dock listed — each with its own lookup table, its own version
# Observable, and, because the two panels were built separately, its own card
# builder and its own set of slider/menu/checkbox helpers.
#
# There is no line to draw between them. "Blur" takes a parameter and shows a
# slider; "Stabilize" takes a mode and runs an analysis; "Matte" wants a click in
# the preview first. Those are differences in what a kind DOES, not in what a
# kind IS: something you add to a clip, tune, and see in the render.

"""
One kind of effect: what the add-menu offers, what the palette can run, what a
card in the Effects panel is built from, and what the render graph turns into a
node.

`make(params) -> Effect` builds an instance; `matches(effect)` recognises one and
`read(effect)` reads its parameters back, so the panel can show any effect
without knowing its type. `params` are the tunable scalars — declared once here
and animatable everywhere for free.

Three optional callbacks cover what used to make something "a tool":

  * `body(ctx)` adds to the card below the parameter sliders — a result readout,
    a list of references, an image.
  * `activate(ctx)` is the card's action, if it has one: run the analysis, start
    collecting clicks in the preview. Nothing happens on its own; the user asks.
  * `deactivate(ctx)` is extra teardown when the card stops being the active one.

`analysis = true` marks a kind whose instance means "an analysis was run and its
result is applied" rather than "these numbers are applied" — the card body says
what it found, and the parameters tune how much of it is used.
"""
struct EffectKind
    name::Symbol
    label::String
    description::String
    params::Vector{FxParam}
    make::Any        # (nt::NamedTuple) -> Effect
    matches::Any     # (e::FxOp) -> Bool
    read::Any        # (e::FxOp) -> NamedTuple
    body::Any        # (ctx::EffectContext) -> nothing, or nothing
    activate::Any    # (ctx::EffectContext) -> nothing, or nothing
    deactivate::Any  # (ctx::EffectContext) -> nothing
    analysis::Bool
end

"""
    EffectKind(name, label; params, make, matches, read, ...)

Keyword constructor.

There is no `kfkeys` any more. It existed to give each parameter a GLOBALLY
unique name, because a curve was stored on the clip under a bare symbol and had
to be resolvable from it alone. A curve now lives on the parameter of the effect
it animates, so two kinds may both call a parameter `:strength` and nothing has
to tell them apart.
"""
function EffectKind(name::Symbol, label::AbstractString;
                    description::AbstractString = "",
                    params::Vector{FxParam} = FxParam[],
                    make = nothing, matches = nothing, read = nothing,
                    body = nothing, activate = nothing,
                    deactivate = ctx -> nothing, analysis::Bool = false)
    return EffectKind(name, String(label), String(description), params,
                      make, matches, read, body, activate, deactivate, analysis)
end

"""
    Effect(k::EffectKind; enabled = true) -> Effect

A fresh entry of `k`, its parameters at their declared defaults. Lives here
rather than in clips.jl because that file is included before `EffectKind`
exists, and an untyped single-argument method there would be ambiguous with
`Effect(payload)`.
"""
Effect(k::EffectKind; enabled::Bool = true) =
    Effect(freshid(), k.name, enabled, FxLink[], paramsfor(k))

"""
    setparams!(fx, values)

Write `values` (a NamedTuple) onto `fx`'s parameters, leaving their curves and
lane visibility alone. What a slider does, and what replacing a payload used to
do by overwriting the whole struct.
"""
function setparams!(fx::Effect, values::NamedTuple)
    for (name, v) in pairs(values)
        p = param(fx, name)
        p === nothing || (p.value = convert(typeof(p.value), v))
    end
    return fx
end

"Default parameter values of `k`, as the NamedTuple `make` takes."
defaults(k::EffectKind) = NamedTuple(p.name => p.default for p in k.params)

"""
Whether `k` can be added by name. Anything that can BUILD an effect can: adding
"Stabilize" puts a Stabilize card on the clip whose action runs the analysis,
which is the same shape as adding "Blur" and then moving its slider. A kind with
no `make` (the loop finder, which searches rather than applies) is not addable.
"""
addable(k::EffectKind) = k.make !== nothing

"""
Every effect kind the editor knows, and every animatable parameter they declare.

One registry object rather than six module globals (`PLUGINS`, `PLUGINBYNAME`,
`PLUGINSVERSION`, `TOOLS`, `TOOLBYNAME`, `TOOLSVERSION`) that had to be kept in
step by hand. `version` is bumped by every registration, so menus and panels
rebuild themselves live.

A `Player` holds a reference to one of these — [`EFFECTS`](@ref) by default.
"""
struct EffectRegistry
    kinds::Vector{EffectKind}
    byname::Dict{Symbol, EffectKind}
    version::Observable{Int}
end
EffectRegistry() = EffectRegistry(EffectKind[], Dict{Symbol, EffectKind}(), Observable(0))

"""
The registry a `Player` gets unless it is given another.

Module-level because registration happens at load time — the built-ins register
while this file is being read, and a package that adds an effect does it in its
own `__init__`, both long before any window exists. Which effects an editor
offers is still per-editor state: `Player(...; effects = EffectRegistry())` gets
its own, and two open editors can differ.
"""
const EFFECTS = EffectRegistry()

Base.length(r::EffectRegistry) = length(r.kinds)
Base.iterate(r::EffectRegistry, s...) = iterate(r.kinds, s...)

"""
    registereffect!(kind; registry = EFFECTS) -> EffectKind

Add or replace an effect kind. Live: callable at any time, from any package or
from the MCP agent, and every menu, palette and panel picks it up immediately.
Registering the same `name` again replaces the old kind in place, so a plugin can
be edited and re-registered without restarting the editor.
"""
function registereffect!(kind::EffectKind; registry::EffectRegistry = EFFECTS)
    registry.byname[kind.name] = kind
    i = findfirst(k -> k.name === kind.name, registry.kinds)
    i === nothing ? push!(registry.kinds, kind) : (registry.kinds[i] = kind)
    registry.version[] = registry.version[] + 1
    return kind
end

"All registered kinds, in registration order."
effectkinds(registry::EffectRegistry = EFFECTS) = registry.kinds
"The kinds offered by name in the add-menu and the palette."
addablekinds(registry::EffectRegistry = EFFECTS) = filter(addable, registry.kinds)
"The kind named `name`, or `nothing`."
kindbyname(name::Symbol, registry::EffectRegistry = EFFECTS) = get(registry.byname, name, nothing)
"The kind that recognises effect `e`, or `nothing`."
function effectkindfor(e, registry::EffectRegistry = EFFECTS)
    for k in registry.kinds
        k.matches !== nothing && k.matches(e) && return k
    end
    return nothing
end

# ---------------------------------------------------------------- parameters


"The first effect of `kind` on `clip`, or `nothing`."
function findeffect(clip::Clip, kind::EffectKind)
    kind.matches === nothing && return nothing
    for s in clip.effects
        kind.matches(op(s)) && return op(s)
    end
    return nothing
end

"Read parameter `name` off `clip`'s instance of `kind`, or `fallback` if it has none."
function paramof(clip::Clip, kind::EffectKind, name::Symbol, fallback::Real)
    e = findeffect(clip, kind)
    e === nothing && return Float64(fallback)
    return Float64(get(kind.read(e), name, fallback))
end

"""
Write parameter `name` of `clip`'s instance of `kind`. Adds the effect at its
defaults if the clip has none — setting a parameter is how a slider says "apply
this", and refusing to would make the first move on a fresh clip do nothing.
"""
function setparam!(clip::Clip, kind::EffectKind, name::Symbol, v::Real)
    e = findeffect(clip, kind)
    cur = e === nothing ? defaults(kind) : kind.read(e)
    seteffect!(clip, kind.make(merge(cur, NamedTuple{(name,)}((Float64(v),)))))
    return nothing
end

# ------------------------------------------------------- effects from a kind

"""
An effect instance for a kind that has no struct of its own: the kind's name plus
its parameter values. Built-ins have their own types (`ColorEffect` and friends,
which the render graph dispatches specialized kernels on); everything registered
later is one of these, rendered through its kind's callback.
"""
struct PluginEffect <: FxOp
    name::Symbol
    params::NamedTuple
end
fxkind(e::PluginEffect) = KINDCALLBACKS[e.name](e.params)
isneutral(::PluginEffect) = false                    # a registered op applies whenever present
effectkey(e::PluginEffect) = (PluginEffect, e.name)  # upsert per kind, not per (shared) type
effectdict(e::PluginEffect) = Dict{String, Any}("type" => "plugin", "name" => String(e.name),
    "params" => Dict{String, Any}(String(k) => Float64(v) for (k, v) in pairs(e.params)))
plugineffectfromdict(d::AbstractDict) = PluginEffect(Symbol(d["name"]),
    NamedTuple(Symbol(k) => Float64(v) for (k, v) in d["params"]))

"""
Render callback per registered kind, keyed by name.

Separate from the registry because [`fxkind`](@ref) is called per frame from the
render path, which has no registry in hand — the effect instance carries only its
name. Written by [`registerplugin!`](@ref) alone.
"""
const KINDCALLBACKS = Dict{Symbol, Any}()

"Construct kind `name`'s effect, overriding parameters by keyword."
function plugineffect(name::Symbol; kw...)
    k = kindbyname(name)
    k === nothing && error("no effect kind named :$name is registered")
    return PluginEffect(name, merge(defaults(k), values(kw)))
end

"""
    registerplugin!(name, label, params, kind; description = "") -> EffectKind

Register an effect from a render callback: `params::Vector{FxParam}` and
`kind(p::NamedTuple) -> FxKind` (a [`Pointwise`](@ref) or [`Stencil`](@ref)).
The effect then shows up in the add-menu and the palette, is keyframable, and is
callable over MCP.

The short form of [`registereffect!`](@ref), which is what to reach for when the
kind needs a card body or an action of its own.
"""
function registerplugin!(name::Symbol, label::AbstractString, params::Vector{FxParam}, kind;
                         description::AbstractString = "", registry::EffectRegistry = EFFECTS)
    KINDCALLBACKS[name] = kind
    # No key namespacing: a plugin's parameters keep the names it gave them, and
    # they only have to be unique WITHIN the plugin.
    return registereffect!(EffectKind(name, label; description, params,
        make = nt -> PluginEffect(name, nt),
        matches = e -> e isa PluginEffect && e.name === name,
        read = e -> NamedTuple(p.name => Float64(get(e.params, p.name, p.default)) for p in params));
        registry)
end

"""
    definepluginfromcode!(code::AbstractString)

Evaluate `code` (a Julia snippet that calls [`registerplugin!`](@ref)) inside this
module, so an MCP agent can author a brand-new effect at runtime. The snippet has
`registerplugin!`, `FxParam`, `Pointwise`, `Stencil`, `Vec2f`, `Vec3f` in scope.
"""
definepluginfromcode!(code::AbstractString) = Base.eval(@__MODULE__, Meta.parseall(String(code)))

# ------------------------------------------------------------ what ships here

# The placement and crop parameters are not any effect's fields, and the
# built-in effects' accessors are hand-written, so they are seeded rather than
# generated. Everything registered after this point generates its own.

registereffect!(EffectKind(:color, "Color";
    description = "Brightness, contrast, saturation and warmth.",
    params = [FxParam(:brightness, "Brightness"; min = -0.5, max = 0.5, default = 0.0),
              FxParam(:contrast, "Contrast"; min = 0.0, max = 2.0, default = 1.0),
              FxParam(:saturation, "Saturation"; min = 0.0, max = 2.0, default = 1.0),
              FxParam(:temperature, "Temperature"; min = -1.0, max = 1.0, default = 0.0)],
    make = nt -> ColorEffect(brightness = nt.brightness, contrast = nt.contrast,
                             saturation = nt.saturation, temperature = nt.temperature),
    matches = e -> e isa ColorEffect,
    read = e -> (brightness = Float64(e.adj.brightness), contrast = Float64(e.adj.contrast),
                 saturation = Float64(e.adj.saturation), temperature = Float64(e.adj.temperature))))

registereffect!(EffectKind(:opacity, "Opacity";
    description = "How much of the clip below shows through.",
    params = [FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    make = nt -> OpacityEffect(Float32(nt.opacity)),
    matches = e -> e isa OpacityEffect,
    read = e -> (opacity = Float64(e.α),)))

registereffect!(EffectKind(:blur, "Blur";
    description = "Gaussian blur.",
    params = [FxParam(:blur, "Blur"; min = 0.0, max = 12.0, default = 0.0)],
    make = nt -> BlurEffect(Float32(nt.blur)),
    matches = e -> e isa BlurEffect,
    read = e -> (blur = Float64(e.σ),)))

registereffect!(EffectKind(:sharpen, "Sharpen";
    description = "Unsharp mask.",
    params = [FxParam(:sharpen, "Sharpen"; min = 0.0, max = 2.0, default = 0.0)],
    make = nt -> SharpenEffect(2.0f0, Float32(nt.sharpen)),
    matches = e -> e isa SharpenEffect,
    read = e -> (sharpen = Float64(e.amount),)))

# ONE card, not two. The sliders and the "click what should be sharp" action are
# the same feature — `EffectKind` carries `params` AND `activate`, so splitting
# them into an effect plus a tool would put two cards on the clip for one thing
# and leave the user to work out that they are related.
registereffect!(EffectKind(:depthblur, "Depth blur";
    description = "Defocus by distance from a focus plane, against estimated depth. " *
                  "Focus is a DEPTH and nothing on screen is labelled with one, so " *
                  "click the picture to set it to whatever you clicked.",
    params = [FxParam(:focus, "Focus"; min = 0.0, max = 1.0, default = 1.0),
              FxParam(:defocus, "Defocus"; min = 0.0, max = 1.0, default = 0.6)],
    make = nt -> DepthBlurEffect(Float32(nt.focus), Float32(nt.defocus)),
    matches = e -> e isa DepthBlurEffect,
    read = e -> (focus = Float64(e.focus), defocus = Float64(e.strength)),
    # `body` only — NOT `activate` too. The panel draws the generic action button
    # only for a kind with no body (see `fxpanel.jl`), so an `activate` here would
    # never be reachable from the card and would only fire via `activatetool!`.
    body = ctx -> depthbody!(ctx)))

registereffect!(EffectKind(:look, "Look";
    description = "A colour grade learned from one frame of this shot. " *
                  "Strength dials it back — the look is the same, the amount is " *
                  "what changes, so a fade-in is a curve on it and never a re-run.",
    params = [FxParam(:look, "Look"; min = 0.0, max = 1.0, default = 1.0)],
    make = nt -> LookEffect(Float32(nt.look)),
    matches = e -> e isa LookEffect,
    read = e -> (look = Float64(e.strength),),
    body = ctx -> lookbody!(ctx)))

registereffect!(EffectKind(:transform, "Transform";
    description = "Where the picture sits in the canvas — scale, position, rotation. " *
                  "Drag the handles in the preview.",
    params = [FxParam(:scale, "Scale"; min = 0.1, max = 4.0, default = 1.0),
              FxParam(:x, "Position X"; min = -1.0, max = 1.0, default = 0.0),
              FxParam(:y, "Position Y"; min = -1.0, max = 1.0, default = 0.0),
              FxParam(:rotation, "Rotation"; min = -180.0, max = 180.0, default = 0.0)],
    make = nt -> TransformEffect(scale = nt.scale, x = nt.x, y = nt.y, rotation = nt.rotation),
    matches = e -> e isa TransformEffect,
    read = e -> (scale = e.scale, x = e.x, y = e.y, rotation = e.rotation)))

# Two effects defined through the public API, so the stock set exercises the same
# path a package or an MCP-authored effect takes.
registerplugin!(:vignette, "Vignette", [FxParam(:strength, "Strength"; max = 1.0, default = 0.6)],
    p -> (s = Float32(p.strength); Pointwise() do c, uv
        d = uv - Vec2f(0.5f0)
        c * clamp(1.0f0 - s * 2.0f0 * (d[1] * d[1] + d[2] * d[2]), 0.0f0, 1.0f0)
    end); description = "Darkens the corners.")

registerplugin!(:soften, "Soften", [FxParam(:radius, "Radius"; min = 1, max = 8, default = 2)],
    p -> Stencil(round(Int, p.radius)) do sample, r, uv
        acc = Vec3f(0, 0, 0); n = 0.0f0
        for dj in -r:r, di in -r:r
            acc = acc + sample(di, dj); n += 1.0f0
        end
        acc / n
    end; description = "Box blur — a cheap softening.")
