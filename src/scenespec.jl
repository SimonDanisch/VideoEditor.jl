# A SCENE AS DATA.
#
# What a scene clip draws is a Makie SPEC — `S.Scene(; camera = cam3d!, plots =
# [...])` — and that is the whole model. There is no second description of it here.
#
# There used to be one: `ScenePart` with `parent`/`origin`/`axis`/`angle`/`offset`,
# `LightSpec`, a camera struct, and a path system over all three. It existed so the
# panel could list what is animatable without looking at the scene — which meant it
# could only ever offer what it had learned to describe, and had to be kept in step
# with what was actually built. Two statements about one thing.
#
# The scene is the truth now. `sceneattributes` (scenesource.jl) walks the real
# plots and offers whatever is there; a path addresses a plot by name and an
# attribute by name, which is the syntax the curves already used, so nothing about
# a saved keyframe changed.

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

"""
    scenepath(str) -> (name, field, component)

Parse `"arm_left.angle"` or `"torso.offset[1]"` into what it addresses.

BY NAME, not by index. A path is what a keyframe curve keys on and what the
project file stores, so it has to survive an edit as well as a round trip — and
reordering the parts must not silently re-aim every curve in the scene. It is
also the difference between a path a model can write (`"arm_left.angle"`) and
one it has to count out (`"plots[4].rotation"`).

`component` is `nothing` unless the path indexes one, as `offset[1]` does.
Returns `nothing` for anything unaddressable rather than throwing: a project
from a newer editor may name parts this one has not got, and a curve pointing
nowhere is one to ignore, not a crash on open.
"""
function scenepath(str::AbstractString)
    m = match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$", String(str))
    m === nothing && return nothing
    comp = m.captures[3] === nothing ? nothing : parse(Int, m.captures[3])
    return (Symbol(m.captures[1]), Symbol(m.captures[2]), comp)
end
scenepath(s::Symbol) = scenepath(String(s))

"""
    dataparams(value) -> Vector{Param}
    datasections(value) -> Vector{NamedTuple}

What a piece of effect DATA contributes to its card.

Nothing, for everything — and a scene no longer answers here either. What a scene
offers depends on the scene that was BUILT, so the card asks the clip's source
(`sceneattributes`) rather than a value that would have to describe itself in
advance.

Kept as the seam, because an effect whose data does contribute parameters is a
reasonable thing to have and this is where it would say so.
"""
dataparams(::Any) = Param[]
datasections(::Any) = NamedTuple[]
