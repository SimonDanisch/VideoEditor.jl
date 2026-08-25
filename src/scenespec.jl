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
One part of a scene: a plot, where it hangs, and the joint it turns on.

A LEGO figure is the reason this exists. It is a tree — `torso → arm_right →
hand_right`, `torso → belt → leg_right` — and the animation only works because
of it: one rotation on `arm_left` takes the hand with it. A flat list of plots
cannot say that, so a part names its `parent` and Makie's `Transformation`
chain does the rest.

The joint is deliberately ONE angle around a FIXED axis, not a quaternion: a
keyframe carries one `Float64` (see `Keyframe`), and a limb swinging on a hinge
is exactly one number per frame. `origin` is the pivot — the shoulder, not the
model's centre — because rotating a limb about the wrong point is the classic
way to tear a figure apart.

`plot.args` may name a MESH FILE as a string; see [`partmesh`](@ref). A project
file cannot hold a vertex array, and should not: the asset is the asset.
"""
struct ScenePart
    name::Symbol
    plot::Makie.PlotSpec
    parent::Union{Nothing, Symbol}     # `nothing` = straight onto the scene
    origin::NTuple{3, Float64}         # pivot, in the parent's frame
    axis::NTuple{3, Float64}           # what `angle` turns around
    angle::Float64                     # ← keyframeable
    offset::NTuple{3, Float64}         # ← three curves
end
ScenePart(name, plot::Makie.PlotSpec; parent = nothing,
          origin = (0.0, 0.0, 0.0), axis = (0.0, 0.0, 1.0),
          angle = 0.0, offset = (0.0, 0.0, 0.0)) =
    ScenePart(Symbol(name), plot, parent === nothing ? nothing : Symbol(parent),
              Float64.(Tuple(origin)), Float64.(Tuple(axis)),
              Float64(angle), Float64.(Tuple(offset)))

"""
A light, as data.

`type` is `:ambient`, `:point` or `:directional`. Intensity rides in `color`,
which is what Makie's own light constructors take — a point light at distance
150 needs a radiance around 150² to read as unit brightness at the subject,
because the falloff is inverse-square.
"""
struct LightSpec
    type::Symbol
    color::NTuple{3, Float64}
    position::NTuple{3, Float64}
end
LightSpec(type; color = (1.0, 1.0, 1.0), position = (0.0, 0.0, 0.0)) =
    LightSpec(Symbol(type), Float64.(Tuple(color)), Float64.(Tuple(position)))

"""
Where the camera is, as data — and therefore keyframeable like anything else.

`"camera.eye[1]"` is a path, so a camera move is three curves and needs no
separate notion of a camera animation. That is the point of putting it here
rather than leaving it to `update_cam!` at build time.
"""
struct CameraSpec
    eye::NTuple{3, Float64}
    lookat::NTuple{3, Float64}
    up::NTuple{3, Float64}
end
CameraSpec(; eye = (3.0, 3.0, 3.0), lookat = (0.0, 0.0, 0.0), up = (0.0, 0.0, 1.0)) =
    CameraSpec(Float64.(Tuple(eye)), Float64.(Tuple(lookat)), Float64.(Tuple(up)))

"""
A scene as data: what to draw, with which backend, under which theme.

`parts` are [`ScenePart`](@ref)s — plots plus the tree they hang in. `backend`
names the renderer (`:GLMakie`, `:RayMakie`, …) and `theme` is an ordinary Makie
theme dictionary; a backend's own settings live in it under the backend's name,
exactly as Makie's default theme carries them, so there is no second place to
look for "backend options".
"""
mutable struct SceneSpec
    parts::Vector{ScenePart}
    lights::Vector{LightSpec}
    camera::CameraSpec
    backend::Symbol
    theme::Dict{Symbol, Any}
end
function SceneSpec(parts::Vector{ScenePart} = ScenePart[];
                   lights::Vector{LightSpec} = LightSpec[],
                   camera::CameraSpec = CameraSpec(),
                   backend::Symbol = :GLMakie,
                   theme::Dict{Symbol, Any} = Dict{Symbol, Any}())
    # NAMES MUST BE UNIQUE, and this is the only place that can insist on it.
    # Paths address a part by name (`"arm_left.angle"`), which is what lets the
    # parts be reordered without re-aiming every curve — but it also means a
    # duplicate name makes the second part unreachable, silently, because the
    # lookup is a `findfirst`. Better to refuse the scene than to animate the
    # wrong arm.
    names = [p.name for p in parts]
    if length(unique(names)) != length(names)
        dupes = unique([n for n in names if count(==(n), names) > 1])
        error("scene part names must be unique; repeated: $(join(dupes, ", "))")
    end
    # `camera` is a reserved name for the same reason — see `scenepathvalue`.
    :camera in names && error("`camera` is reserved as a path name; rename that part")
    return SceneSpec(parts, lights, camera, backend, theme)
end

"The part called `name`, or `nothing`."
partbyname(spec::SceneSpec, name::Symbol) =
    (i = findfirst(p -> p.name === name, spec.parts); i === nothing ? nothing : spec.parts[i])

"""
    partmesh(arg) -> mesh

A plot argument that may be a FILE PATH. A project file names an asset
(`"lego_figure_torso.stl"`), and this is where that becomes geometry — resolved
against Makie's asset artifact when it is not an absolute path, so a scene an
LLM wrote refers to the same file on any machine.
"""
partmesh(arg::AbstractString) =
    isabspath(arg) ? Makie.FileIO.load(arg) : Makie.loadasset(arg)
partmesh(arg) = arg

# ---------------------------------------------------------------- paths

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

"The animatable fields of a part, and how many numbers each takes."
const PARTFIELDS = (angle = 1, offset = 3, origin = 3, axis = 3)

"""
    scenepathvalue(spec, path)

The value at `path`, or `nothing` when it addresses nothing in `spec`.
"""
function scenepathvalue(spec::SceneSpec, path)
    p = scenepath(path)
    p === nothing && return nothing
    name, field, comp = p
    # `camera` is a reserved part name, so a camera move is three ordinary
    # curves ("camera.eye[1]" …) and needs no separate animation concept.
    if name === :camera
        hasfield(CameraSpec, field) || return nothing
        v = getfield(spec.camera, field)
        return comp === nothing ? v : (checkbounds(Bool, collect(v), comp) ? v[comp] : nothing)
    end
    part = partbyname(spec, name)
    part === nothing && return nothing
    if haskey(PARTFIELDS, field)
        v = getfield(part, field)
        comp === nothing && return v
        return checkbounds(Bool, collect(v), comp) ? v[comp] : nothing
    end
    # …anything else is a plot ATTRIBUTE, which is why no part has to declare
    # up front what is animatable: `kwargs` is a Dict and every key in it counts.
    return get(part.plot.kwargs, field, nothing)
end

"""
    setscenepath!(spec, path, value) -> Bool

Write `value` at `path`. `false` when it addresses nothing, so a caller can tell
"did not apply" from "applied a false".

A `ScenePart` is immutable, so writing a joint field REPLACES the part in
`spec.parts` — which is what makes [`animatedspec`](@ref)'s copy cheap and its
isolation real: the original's parts are never touched.

This is the whole of parameter animation: a curve holds one `Float64` per frame
(see `Keyframe`), a path says where it goes, and one curve per component is how
After Effects separates dimensions too — so an offset is three curves, not a new
keyframe value type.
"""
function setscenepath!(spec::SceneSpec, path, value)
    p = scenepath(path)
    p === nothing && return false
    name, field, comp = p
    if name === :camera
        hasfield(CameraSpec, field) || return false
        old = getfield(spec.camera, field)
        new = comp === nothing ? Float64.(Tuple(value)) :
              (checkbounds(Bool, collect(old), comp) ?
               ntuple(k -> k == comp ? Float64(value) : old[k], 3) : return false)
        spec.camera = CameraSpec(field === :eye ? new : spec.camera.eye,
                                 field === :lookat ? new : spec.camera.lookat,
                                 field === :up ? new : spec.camera.up)
        return true
    end
    i = findfirst(q -> q.name === name, spec.parts)
    i === nothing && return false
    part = spec.parts[i]
    if haskey(PARTFIELDS, field)
        old = getfield(part, field)
        new = if comp === nothing
            field === :angle ? Float64(value) : Float64.(Tuple(value))
        elseif field === :angle
            Float64(value)                      # `angle[1]` is just the angle
        else
            checkbounds(Bool, collect(old), comp) || return false
            ntuple(k -> k == comp ? Float64(value) : old[k], length(old))
        end
        spec.parts[i] = ScenePart(part.name, part.plot, part.parent,
                                  field === :origin ? new : part.origin,
                                  field === :axis   ? new : part.axis,
                                  field === :angle  ? new : part.angle,
                                  field === :offset ? new : part.offset)
        return true
    end
    part.plot.kwargs[field] = value
    return true
end

"""
    animatedspec(spec, state) -> SceneSpec

`spec` with every path in `state` written into a COPY of it.

The copy is the point: `state` is sampled per frame, and mutating the stored spec
would make the animation cumulative — frame 10's value would still be sitting
there at frame 11 if its curve had ended. Theme and backend are shared, since
nothing per-frame writes them.
"""
function animatedspec(spec::SceneSpec, state)
    out = SceneSpec([ScenePart(p.name,
                               Makie.PlotSpec(p.plot.type, p.plot.args...; copy(p.plot.kwargs)...),
                               p.parent, p.origin, p.axis, p.angle, p.offset)
                     for p in spec.parts], copy(spec.lights), spec.camera,
                    spec.backend, spec.theme)
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

"""
`spec` as a plain dictionary, ready for the project file.

`"type" => "scene"` is a TAG, not decoration: an overlay's settings are a bag of
values of no fixed type, so when the project is read back something has to say
which of them is a scene rather than a dictionary that happens to have parts.
Sniffing for a `"parts"` key would work today and quietly mis-fire the first
time another setting has one. See `scenesetting` in overlays.jl.
"""
scenedict(spec::SceneSpec) = Dict{String, Any}(
    "type" => "scene",
    "backend" => String(spec.backend),
    "theme" => specvalue(spec.theme),
    "lights" => [Dict{String, Any}("type" => String(l.type),
                                   "color" => collect(l.color),
                                   "position" => collect(l.position)) for l in spec.lights],
    "camera" => Dict{String, Any}("eye" => collect(spec.camera.eye),
                                  "lookat" => collect(spec.camera.lookat),
                                  "up" => collect(spec.camera.up)),
    "parts" => [Dict{String, Any}("name" => String(p.name),
                                  "type" => String(p.plot.type),
                                  "args" => specvalue(p.plot.args),
                                  "kwargs" => specvalue(p.plot.kwargs),
                                  "parent" => p.parent === nothing ? nothing : String(p.parent),
                                  "origin" => collect(p.origin),
                                  "axis" => collect(p.axis),
                                  "angle" => p.angle,
                                  "offset" => collect(p.offset))
               for p in spec.parts])

# A scene is STRUCTURE, and `tomlvalue`'s fallback is `string(v)` — saving would
# have written its `show` form, which nothing reads back. The overlay then
# reloads with `spec` as a `String`, and `:scene`'s draw begins
# `spec isa SceneSpec || return`, so the project opens with the scene silently
# ABSENT rather than visibly broken. Both methods live here rather than in
# overlays.jl because that file is included long before this type exists.
tomlvalue(v::SceneSpec) = scenedict(v)

richsetting(v::AbstractDict) =
    get(v, "type", nothing) == "scene" ? scenefromdict(v) : v

"""
`scenedict`'s inverse — and the shape somebody writes by hand:

    {"name": "arm_left", "type": "Mesh", "parent": "torso",
     "args": ["lego_figure_arm_left.stl"],
     "origin": [0.1427, 6.2127, 5.7342], "axis": [0, 0.9828, 0.1848]}
"""
function scenefromdict(d::AbstractDict)
    parts = ScenePart[]
    for pd in get(d, "parts", [])
        kwargs = fromspecvalue(get(pd, "kwargs", Dict{String, Any}()))
        args = fromspecvalue(get(pd, "args", []))
        parent = get(pd, "parent", nothing)
        push!(parts, ScenePart(Symbol(pd["name"]),
                               Makie.PlotSpec(Symbol(pd["type"]), args...; kwargs...);
                               parent = parent === nothing ? nothing : Symbol(parent),
                               origin = Tuple(get(pd, "origin", (0.0, 0.0, 0.0))),
                               axis = Tuple(get(pd, "axis", (0.0, 0.0, 1.0))),
                               angle = get(pd, "angle", 0.0),
                               offset = Tuple(get(pd, "offset", (0.0, 0.0, 0.0)))))
    end
    lights = LightSpec[LightSpec(Symbol(l["type"]);
                                color = Tuple(get(l, "color", (1.0, 1.0, 1.0))),
                                position = Tuple(get(l, "position", (0.0, 0.0, 0.0))))
                       for l in get(d, "lights", [])]
    cd_ = get(d, "camera", Dict{String, Any}())
    cam = CameraSpec(; eye = Tuple(get(cd_, "eye", (3.0, 3.0, 3.0))),
                     lookat = Tuple(get(cd_, "lookat", (0.0, 0.0, 0.0))),
                     up = Tuple(get(cd_, "up", (0.0, 0.0, 1.0))))
    theme = fromspecvalue(get(d, "theme", Dict{String, Any}()))
    return SceneSpec(parts; lights = lights, camera = cam,
                     backend = Symbol(get(d, "backend", "GLMakie")),
                     theme = Dict{Symbol, Any}(theme))
end
