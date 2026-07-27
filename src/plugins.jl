# Plugin registry — any package, or a live session (including the MCP agent), can
# register an effect. A registered plugin:
#   • shows up in the editor's "Add effect" menu (live, via `PLUGINSVERSION`),
#   • becomes keyframable + slider-visible (an auto-registered `ParamSpec` per param),
#   • is callable and AUTHORABLE over MCP (`effect_<name>` / `define_effect`).
# Registration is live — no restart. An effect is a pure callback ([`fxkind`]); the
# registry just adds discoverability and generic param wiring on top.

"One tunable parameter of a plugin effect (a scalar with a slider range)."
struct FxParam
    name::Symbol
    label::String
    min::Float64
    max::Float64
    default::Float64
end
FxParam(name, label; min = 0.0, max = 1.0, default = min) =
    FxParam(Symbol(name), String(label), Float64(min), Float64(max), Float64(default))

"""
A registered effect: a name, display label, tunable params, and `kind(p::NamedTuple)`
returning the render callback ([`Pointwise`](@ref)/[`Stencil`](@ref)). Register with
[`registerplugin!`](@ref); it is then discoverable (menu + MCP) and keyframable.
"""
struct FxPlugin
    name::Symbol
    label::String
    params::Vector{FxParam}
    kind::Any        # (p::NamedTuple) -> FxKind
end

const PLUGINS = FxPlugin[]
const PLUGINBYNAME = Dict{Symbol, FxPlugin}()
const PLUGINSVERSION = Observables.Observable(0)   # bumped on every (re)registration

"An effect instance from a registered plugin, carrying its parameter values."
struct PluginEffect <: Effect
    name::Symbol
    params::NamedTuple
end
fxkind(e::PluginEffect) = PLUGINBYNAME[e.name].kind(e.params)
isneutral(::PluginEffect) = false                  # a plugin op is applied whenever present
effectkey(e::PluginEffect) = (PluginEffect, e.name)  # upsert per plugin, not per (shared) type
effectdict(e::PluginEffect) = Dict{String, Any}("type" => "plugin", "name" => String(e.name),
    "params" => Dict{String, Any}(String(k) => Float64(v) for (k, v) in pairs(e.params)))
plugineffectfromdict(d::AbstractDict) = PluginEffect(Symbol(d["name"]),
    NamedTuple(Symbol(k) => Float64(v) for (k, v) in d["params"]))

defaults(p::FxPlugin) = NamedTuple{Tuple(pr.name for pr in p.params)}(Tuple(pr.default for pr in p.params))
"Construct plugin `name`'s effect, overriding params by keyword (`plugineffect(:vignette; strength=0.8)`)."
plugineffect(name::Symbol; kw...) = (p = PLUGINBYNAME[name]; PluginEffect(name, merge(defaults(p), values(kw))))

function findplugineffect(clip::Clip, name::Symbol)
    for s in clip.effects
        s.effect isa PluginEffect && s.effect.name == name && return s.effect
    end
    return nothing
end

# one param → its own key (:vignette); many → namespaced (:duotone_mix)
pluginparamkey(p::FxPlugin, pr::FxParam) = length(p.params) == 1 ? p.name : Symbol(p.name, :_, pr.name)

pluginget(clip::Clip, name::Symbol, pr::Symbol, default::Float64) =
    (e = findplugineffect(clip, name); e === nothing ? default : Float64(get(e.params, pr, default)))
function pluginset!(clip::Clip, name::Symbol, pr::Symbol, v::Real)
    p = PLUGINBYNAME[name]
    cur = (e = findplugineffect(clip, name)) === nothing ? defaults(p) : e.params
    seteffect!(clip, PluginEffect(name, merge(cur, NamedTuple{(pr,)}((Float64(v),)))))
end

# make each plugin param keyframable + slider-visible — the same registry the built-ins use
function registerpluginparams!(p::FxPlugin)
    for pr in p.params
        key = pluginparamkey(p, pr)
        haskey(PARAMBYKEY, key) && continue
        label = length(p.params) == 1 ? p.label : "$(p.label) $(pr.label)"
        spec = ParamSpec(key, label, :plugin, pr.min, pr.max, pr.default,
            c -> pluginget(c, p.name, pr.name, pr.default),
            (c, v) -> pluginset!(c, p.name, pr.name, v))
        push!(PARAMS, spec)
        PARAMBYKEY[key] = spec
    end
end

"""
    registerplugin!(name, label, params, kind) -> FxPlugin

Register an effect so it appears in the editor's Add-effect menu, is keyframable, and
is callable over MCP. `params::Vector{FxParam}`; `kind(p::NamedTuple) -> FxKind` returns
the render callback (a [`Pointwise`](@ref) or [`Stencil`](@ref)). Live — call it any
time, from any package or from the MCP agent.
"""
function registerplugin!(name::Symbol, label::AbstractString, params::Vector{FxParam}, kind)
    p = FxPlugin(name, String(label), params, kind)
    PLUGINBYNAME[name] = p
    i = findfirst(q -> q.name == name, PLUGINS)
    i === nothing ? push!(PLUGINS, p) : (PLUGINS[i] = p)
    registerpluginparams!(p)
    PLUGINSVERSION[] = PLUGINSVERSION[] + 1
    return p
end

"""
    definepluginfromcode!(code::AbstractString)

Evaluate `code` (a Julia snippet that calls [`registerplugin!`](@ref)) inside this
module, so an MCP agent can author a brand-new effect at runtime. The snippet has
`registerplugin!`, `FxParam`, `Pointwise`, `Stencil`, `Vec2f`, `Vec3f` in scope.
"""
definepluginfromcode!(code::AbstractString) = Base.eval(@__MODULE__, Meta.parseall(String(code)))

# ---------------------------------------------------------------- stock plugins

# Two useful effects, defined through the public plugin API (nothing hard-wired):
registerplugin!(:vignette, "Vignette", [FxParam(:strength, "Strength"; max = 1.0, default = 0.6)],
    p -> (s = Float32(p.strength); Pointwise() do c, uv
        d = uv - Vec2f(0.5f0)
        c * clamp(1.0f0 - s * 2.0f0 * (d[1] * d[1] + d[2] * d[2]), 0.0f0, 1.0f0)
    end))

registerplugin!(:soften, "Soften", [FxParam(:radius, "Radius"; min = 1, max = 8, default = 2)],
    p -> Stencil(round(Int, p.radius)) do sample, r, uv
        acc = Vec3f(0, 0, 0); n = 0.0f0
        for dj in -r:r, di in -r:r
            acc = acc + sample(di, dj); n += 1.0f0
        end
        acc / n
    end)

# ---------------------------------------------------------------- effect kinds

"""
A uniform, editable descriptor for ANY effect — built-in or plugin — so the effect
list, the add-picker and the modal editor treat them identically. `params` are the
tunable fields; `kfkeys` is the keyframe-registry key per param (so the ◆ toggle
animates the right curve); `make(nt)` builds the `Effect`; `matches(e)` and `read(e)`
recognise and read back an existing instance.
"""
struct EffectKind
    name::Symbol
    label::String
    params::Vector{FxParam}
    kfkeys::Vector{Symbol}
    make::Any
    matches::Any
    read::Any
end

# built-in effects, edited exactly like plugins (their param names ARE their PARAMS keys)
const BUILTIN_KINDS = EffectKind[
    EffectKind(:color, "Color",
        [FxParam(:brightness, "Brightness"; min = -0.5, max = 0.5, default = 0.0),
         FxParam(:contrast, "Contrast"; min = 0.0, max = 2.0, default = 1.0),
         FxParam(:saturation, "Saturation"; min = 0.0, max = 2.0, default = 1.0),
         FxParam(:temperature, "Temperature"; min = -1.0, max = 1.0, default = 0.0)],
        [:brightness, :contrast, :saturation, :temperature],
        nt -> ColorEffect(brightness = nt.brightness, contrast = nt.contrast,
                          saturation = nt.saturation, temperature = nt.temperature),
        e -> e isa ColorEffect,
        e -> (brightness = Float64(e.adj.brightness), contrast = Float64(e.adj.contrast),
              saturation = Float64(e.adj.saturation), temperature = Float64(e.adj.temperature))),
    EffectKind(:opacity, "Opacity", [FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
        [:opacity], nt -> OpacityEffect(Float32(nt.opacity)), e -> e isa OpacityEffect,
        e -> (opacity = Float64(e.α),)),
    EffectKind(:blur, "Blur", [FxParam(:blur, "Blur"; min = 0.0, max = 12.0, default = 0.0)],
        [:blur], nt -> BlurEffect(Float32(nt.blur)), e -> e isa BlurEffect, e -> (blur = Float64(e.σ),)),
    EffectKind(:sharpen, "Sharpen", [FxParam(:sharpen, "Sharpen"; min = 0.0, max = 2.0, default = 0.0)],
        [:sharpen], nt -> SharpenEffect(2.0f0, Float32(nt.sharpen)), e -> e isa SharpenEffect,
        e -> (sharpen = Float64(e.amount),)),
]

"The `EffectKind` for a registered plugin (its keyframe keys come from `pluginparamkey`)."
pluginkind(p::FxPlugin) = EffectKind(p.name, p.label, p.params,
    [pluginparamkey(p, pr) for pr in p.params],
    nt -> plugineffect(p.name; nt...),
    e -> e isa PluginEffect && e.name === p.name,
    e -> NamedTuple(pr.name => Float64(get(e.params, pr.name, pr.default)) for pr in p.params))

"All editable effect kinds — built-ins then registered plugins."
effectkinds() = vcat(BUILTIN_KINDS, EffectKind[pluginkind(p) for p in PLUGINS])
"The kind matching effect `e`, or `nothing`."
effectkindfor(e::Effect) = (for k in effectkinds(); k.matches(e) && return k; end; nothing)
"The kind named `name`, or `nothing`."
kindbyname(name::Symbol) = (for k in effectkinds(); k.name === name && return k; end; nothing)
