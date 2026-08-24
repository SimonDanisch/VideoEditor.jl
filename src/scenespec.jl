# A Makie scene AS DATA — the thing an LLM writes, a human corrects, and the
# project file round-trips.
#
# Why data and not code: the editor has to be able to WRITE it back. A scene
# built by a Julia function can be rendered but not edited — the moment someone
# drags a sphere in the viewport there is nowhere to put the new position, and
# one of the two authors (the model or the person) always loses. Makie's own
# `PlotSpec` is already exactly this shape:
#
#     struct PlotSpec; type::Symbol; args::Vector{Any}; kwargs::Dict{Symbol,Any}; end
#
# so a scene is a list of them plus the backend and theme they render under.
# Backend settings ARE theme entries (`theme[:RayMakie][:samples]` and so on), so
# there is one dictionary rather than a second parallel notion of settings.
#
# What makes the animation generic is that `kwargs` is a Dict: every attribute of
# every plot is addressable by a PATH, and a path is what a keyframe curve keys
# on. No kind has to declare in advance which numbers are animatable, which is
# what `FxParam` lists forced and why they could never describe a 3D scene.

"""
A scene as data: what to draw, with which backend, under which theme.

`plots` are Makie `PlotSpec`s. `backend` names the renderer (`:GLMakie`,
`:RayMakie`, …) and `theme` is an ordinary Makie theme dictionary — the backend's
own settings live in it under the backend's name, exactly as Makie's default
theme carries them, so there is no second place to look for "backend options".
"""
struct SceneSpec
    plots::Vector{Makie.PlotSpec}
    backend::Symbol
    theme::Dict{Symbol, Any}
end
SceneSpec(plots::Vector{Makie.PlotSpec} = Makie.PlotSpec[];
          backend::Symbol = :GLMakie, theme::Dict{Symbol, Any} = Dict{Symbol, Any}()) =
    SceneSpec(plots, backend, theme)

# ---------------------------------------------------------------- paths

"""
    scenepath(str) -> (index, key)

Parse `"plots[2].markersize"` into the plot it names and the attribute on it.

The string form is what a keyframe curve keys on and what a project file stores,
so it has to survive a round trip through both — which rules out anything that
is not plain text. Returns `nothing` for a path this cannot address, rather than
throwing: a project written by a newer editor may name plots this one has not
got, and a curve pointing nowhere is a curve to ignore, not a crash on open.
"""
function scenepath(str::AbstractString)
    m = match(r"^plots\[(\d+)\]\.([A-Za-z_][A-Za-z0-9_]*)$", String(str))
    m === nothing && return nothing
    return (parse(Int, m.captures[1]), Symbol(m.captures[2]))
end
scenepath(s::Symbol) = scenepath(String(s))

"""
    scenepathvalue(spec, path)

The value at `path`, or `nothing` when it does not address anything in `spec`.
"""
function scenepathvalue(spec::SceneSpec, path)
    p = scenepath(path)
    p === nothing && return nothing
    i, key = p
    checkbounds(Bool, spec.plots, i) || return nothing
    return get(spec.plots[i].kwargs, key, nothing)
end

"""
    setscenepath!(spec, path, value) -> Bool

Write `value` at `path`. `false` when the path addresses nothing, so a caller
can tell "did not apply" from "applied a false".

This is the whole of parameter animation: a curve holds one `Float64` per frame
(see `Keyframe`), a path says where it goes, and one curve per component is how
After Effects separates dimensions too — so a position is three curves, not a
new keyframe value type.
"""
function setscenepath!(spec::SceneSpec, path, value)
    p = scenepath(path)
    p === nothing && return false
    i, key = p
    checkbounds(Bool, spec.plots, i) || return false
    spec.plots[i].kwargs[key] = value
    return true
end

"""
    animatedspec(spec, state) -> SceneSpec

`spec` with every path in `state` written into a COPY of it.

The copy is the point: `state` is sampled per frame, and mutating the stored
spec would make the animation cumulative — frame 10's value would still be there
at frame 11 if its curve had ended. Only the plots are rebuilt; the theme and
backend are shared, since nothing per-frame writes them.
"""
function animatedspec(spec::SceneSpec, state)
    out = SceneSpec([Makie.PlotSpec(p.type, p.args...; copy(p.kwargs)...)
                     for p in spec.plots], spec.backend, spec.theme)
    for (key, value) in pairs(state)
        setscenepath!(out, key, value)
    end
    return out
end

# ---------------------------------------------------------------- serialisation

"""
    specvalue(x)

A Makie attribute value as something a project file can hold.

Symbols keep Julia's own spelling (`:red` ⇢ `":red"`), which is both readable and
unambiguous against a plain string — an LLM writing one gets it right by writing
what it would write in Julia. Everything else that JSON already models (numbers,
strings, bools, arrays) passes through.
"""
specvalue(x::Symbol) = ":" * String(x)
# COLOURS ARE NOT SYMBOLS BY THE TIME THEY GET HERE. `PlotSpec` normalises its
# keywords as it is built — `PlotSpec(:Scatter; color = :red)` already holds
# `RGBA{Float32}(1,0,0,1)`, deliberately, so that spelling a colour two ways
# yields the same spec. Without a method here that would fall to `string(x)` and
# come back as the text "RGBA{Float32}(...)". `#rrggbbaa` is unambiguous against
# a plain string, round-trips exactly, and is what someone writing one by hand
# would reach for anyway.
specvalue(x::Colorant) = (c = RGBA{Float64}(x);
                          "#" * join(lpad(string(round(Int, 255 * v); base = 16), 2, '0')
                                     for v in (c.r, c.g, c.b, c.alpha)))
specvalue(x::Union{Real, AbstractString, Bool}) = x
specvalue(x::AbstractVector) = [specvalue(v) for v in x]
specvalue(x::Tuple) = [specvalue(v) for v in x]
specvalue(x::AbstractDict) = Dict{String, Any}(String(k) => specvalue(v) for (k, v) in x)
specvalue(x) = string(x)          # last resort: readable, and round-trips as text

"The inverse of [`specvalue`](@ref)."
function fromspecvalue(x::AbstractString)
    startswith(x, ":") && length(x) > 1 && return Symbol(x[2:end])
    m = match(r"^#([0-9A-Fa-f]{8})$", x)
    m === nothing && return x
    b = [parse(Int, m.captures[1][i:(i + 1)]; base = 16) / 255 for i in 1:2:7]
    return RGBAf(b[1], b[2], b[3], b[4])
end
fromspecvalue(x::AbstractVector) = [fromspecvalue(v) for v in x]
fromspecvalue(x::AbstractDict) =
    Dict{Symbol, Any}(Symbol(k) => fromspecvalue(v) for (k, v) in x)
fromspecvalue(x) = x

"`spec` as a plain dictionary, ready for the project file."
scenedict(spec::SceneSpec) = Dict{String, Any}(
    "backend" => String(spec.backend),
    "theme" => specvalue(spec.theme),
    "plots" => [Dict{String, Any}("type" => String(p.type),
                                  "args" => specvalue(p.args),
                                  "kwargs" => specvalue(p.kwargs))
                for p in spec.plots])

"`scenedict`'s inverse — the shape an LLM writes by hand."
function scenefromdict(d::AbstractDict)
    plots = Makie.PlotSpec[]
    for pd in get(d, "plots", [])
        kwargs = fromspecvalue(get(pd, "kwargs", Dict{String, Any}()))
        args = fromspecvalue(get(pd, "args", []))
        push!(plots, Makie.PlotSpec(Symbol(pd["type"]), args...; kwargs...))
    end
    theme = fromspecvalue(get(d, "theme", Dict{String, Any}()))
    return SceneSpec(plots, Symbol(get(d, "backend", "GLMakie")),
                     Dict{Symbol, Any}(theme))
end
