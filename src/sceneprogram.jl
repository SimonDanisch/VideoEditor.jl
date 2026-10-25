"""
A saved Julia scene recipe. Its GPU objects are constructed lazily on the render thread.

The recipe is either a file (`file`, included into a fresh module) or a function
of an installed package (`package`, loaded through the active environment).
A package recipe is portable: a farm worker resolves it from its own Pkg
environment, so no source file travels with a job.
"""
mutable struct SceneProgram
    file::String
    entry::Symbol
    args::Dict{String, Any}
    builder::Any  # loaded recipe, reused when its screen is rebuilt
    package::String  # "" for a file recipe
end

SceneProgram(file::String, entry::Symbol, args::Dict{String, Any}) =
    SceneProgram(file, entry, args, nothing, "")

function SceneProgram(d::AbstractDict)
    entry = Symbol(get(d, "entry", "buildscene"))
    args = Dict{String, Any}(get(d, "args", Dict{String, Any}()))
    haskey(d, "package") && return SceneProgram("", entry, args, nothing, String(d["package"]))
    return SceneProgram(abspath(String(d["file"])), entry, args, nothing, "")
end

"""
    packagescene(package, entry; args=Dict(), objects=nothing, markers=nothing) -> Dict

A project-saveable recipe that calls `package.entry(canvas, args)`. The contract
is that of [`programscene`](@ref). `package` is a module or its name; it must
be installed in the environment that opens the project or renders its frames.
"""
function packagescene(package::Union{Module, Symbol, AbstractString}, entry::Symbol;
                      args::AbstractDict = Dict{String, Any}(), objects = nothing,
                      markers = nothing)
    name = package isa Module ? String(nameof(package)) : String(package)
    d = Dict{String, Any}("kind" => "program", "package" => name,
                          "entry" => String(entry), "args" => Dict{String, Any}(args))
    objects === nothing || (d["objects"] = objects)
    markers === nothing || (d["markers"] = markers)
    return d
end

"""
    programscene(file; entry=:buildscene, args=Dict()) -> Dict

A project-saveable recipe for a Makie scene. The Julia file defines
`entry(canvas, args)`, returning at least `(scene = scene,)`. The scene is built
once; the clip's keyframes animate it. Every named plot's attributes and
transforms, the camera (eye, target, up, field of view) and the native lights
are keyframable without further declarations. A scene with a 3D camera inside a
figure layout (an `LScene` beside axes) exposes the first 3D camera it holds.

The builder's `args` dictionary contains construction settings. Animation belongs
to native plot, camera and light property tracks. Custom geometric and medium
parameters are inputs on the owning Makie plot's attribute graph. Frames with
equal parameter values share the same traced source master.

Optional `objects` describes inspector groups as dictionaries with `label`,
`plots` (stable names; first is the pivot), and optional `attributes` to expose.
Transforms move group members together. Omit it to expose all named plots.
The builder may return the same `objects` (NamedTuples or dictionaries) beside
its scene instead, where the code that names the plots also groups them; groups
saved with the recipe take precedence.

An optional `preview!(pixel_scale)` callback can adjust expensive procedural
detail when preview quality changes. It runs before the next animation update,
once per quality change. Full preview and finished output always pass `1.0`;
the callback must restore original detail at that scale. It does not edit the
saved recipe arguments or the clip's parameters.

The file runs in its own module and may include other files relative to itself.
Only open recipes whose Julia code you trust, just as with a Julia project.
"""
function programscene(file::AbstractString; entry::Symbol = :buildscene,
                      args::AbstractDict = Dict{String, Any}(), objects = nothing,
                      markers = nothing)
    d = Dict{String, Any}("kind" => "program", "file" => abspath(file),
                          "entry" => String(entry), "args" => Dict{String, Any}(args))
    objects === nothing || (d["objects"] = objects)
    markers === nothing || (d["markers"] = markers)
    return d
end

struct SceneOverrides
    defaults::Dict{Tuple{Symbol, Symbol}, Any}
    touched::Set{Tuple{Symbol, Symbol}}
end

mutable struct ProgramInstance
    scene::Makie.Scene
    overrides::SceneOverrides
    frame::Int
    preview!::Any
    previewscale::Float64
    recordings::Dict{Symbol,RecordedTrack}
    # Inspector groups the builder returned (see `programscene`), in the saved
    # form: `label`, `plots` and optional `attributes`, as strings.
    objects::Vector{Dict{String, Any}}
    # The scene whose 3D camera the editor animates: the root, or the first
    # `LScene`-like child of a figure. `nothing` for a scene without one.
    camerascene::Union{Nothing, Makie.Scene}
end

"""
    camerascene(scene) -> Union{Nothing, Scene}

The scene whose `Camera3D` the editor reads and animates: `scene` itself, or
the first descendant, depth first. A figure's root scene has a pixel camera;
its `LScene` holds the 3D one.
"""
function camerascene(scene::Makie.Scene)
    Makie.cameracontrols(scene) isa Makie.Camera3D && return scene
    for child in scene.children
        found = camerascene(child)
        found === nothing || return found
    end
    return nothing
end

"""
    programobjects(objects) -> Vector{Dict{String, Any}}

Inspector groups a builder returned, in the form a saved recipe holds them.
"""
programobjects(objects) = Dict{String, Any}[programobject(o) for o in objects]
function programobject(o)
    d = Dict{String, Any}("label" => String(o[:label]), "plots" => String.(collect(o[:plots])))
    attributes = get(o, :attributes, nothing)
    attributes === nothing || (d["attributes"] = String.(collect(attributes)))
    return d
end
programobject(o::AbstractDict) = programobject((; (Symbol(k) => v for (k, v) in o)...))

function SceneOverrides(scene)
    defaults = Dict{Tuple{Symbol, Symbol}, Any}()
    camscene = camerascene(scene)
    camscene === nothing || (defaults[(:camera, :camera)] = cameravalues(camscene))
    for (i,camera) in enumerate(nativecameras(scene))
        i==1 || (defaults[(Symbol("camera[",i,"]"),:camera)]=cameravalues(camera))
    end
    for (i, light) in [(0, Makie.AmbientLight(scene.compute[:ambient_color][]));
                       collect(enumerate(Makie.get_lights(scene)))]
        for key in fieldnames(typeof(light))
            defaults[(Symbol("lights[$i]"), key)] = Makie.to_value(getfield(light, key))
        end
    end
    for (name, plot) in sceneplots(scene)
        for key in (:translation, :rotation, :scale)
            defaults[(name, key)] = getproperty(plot.transformation, key)[]
        end
        for key in sceneinputkeys(plot)
            v = Makie.to_value(plot.attributes[key])
            (v===nothing || v isa Bool || rowkind(v) in (:number, :vector, :colour,:data)) || continue
            defaults[(name, key)] = v
        end
    end
    return SceneOverrides(defaults, Set{Tuple{Symbol, Symbol}}())
end

"""
    restoresceneprogram!(target, keep = Set()) -> nothing

Return every property an editor override touched to its recipe value, except
the plot attributes in `keep`: those are still keyed and are about to be written
again, and restoring them first would make everything derived from them
recompute twice per frame. Transforms are always restored, because grouped
objects move by the difference to their current placement.
"""
function restoresceneprogram!(target::ProgramInstance, keep = Set{Tuple{Symbol, Symbol}}())
    u = target.overrides
    for address in u.touched
        # Kept addresses are re-applied this frame, which touches them again.
        (haskey(u.defaults, address) && address ∉ keep) || continue
        name, key = address
        if scenelight(target.scene, name) !== nothing
            setscenelight!(target.scene, name, key, u.defaults[address])
            continue
        end
        if name === :camera && key === :camera
            setcameravalues!(target.camerascene, u.defaults[address])
            continue
        end
        camera=nativecamera(target.scene,name)
        if camera !== nothing && key===:camera
            setcameravalues!(camera,u.defaults[address])
            continue
        end
        plot = Makie.findplot(target.scene, name)
        plot === nothing && continue
        v = u.defaults[address]
        if key === :translation
            Makie.translate!(plot, v)
        elseif key === :rotation
            Makie.rotate!(plot, v)
        elseif key === :scale
            Makie.scale!(plot, v)
        else
            Makie.update!(plot; NamedTuple{(key,)}((v,))...)
        end
    end
    empty!(u.touched)
    return nothing
end

"""
    keyedattributes(clip) -> Set{Tuple{Symbol, Symbol}}

The plot attributes this clip's scene effect writes every frame: the ones
[`restoresceneprogram!`](@ref) need not restore first. Transforms, the camera,
lights are not included.
"""
function keyedattributes(clip)
    keep = Set{Tuple{Symbol, Symbol}}()
    fx = findslot(clip, :scene)
    (fx === nothing || !fx.enabled[]) && return keep
    for p in fx.params
        isfollowing(p) && continue
        address = scenepath(p.name)
        address === nothing && continue
        name, key, _ = address
        (name === :camera || key in (:translation, :rotation, :scale) ||
         startswith(String(name), "lights[")) && continue
        push!(keep, (name, key))
    end
    return keep
end

function remembersceneoverride!(src, path)
    src.live.target isa ProgramInstance || return nothing
    u = src.live.target.overrides
    address = scenepath(path)
    address === nothing && return nothing
    name, key, _ = address
    if scenelight(targetscene(src.live.target), name) !== nothing
        push!(u.touched, (name, key))
        return nothing
    end
    name === :camera && (push!(u.touched, (:camera, :camera)); return nothing)
    nativecamera(targetscene(src.live.target),name)===nothing ||
        (push!(u.touched,(name,:camera));return nothing)
    plots = key in (:translation, :rotation, :scale) ? sceneobjectplots(src, name) :
            filter(!isnothing, [Makie.findplot(targetscene(src.live.target), name)])
    for plot in plots
        n = Symbol(Makie.to_value(plot.attributes[:name]))
        push!(u.touched, (n, key))
        key in (:rotation, :scale) && push!(u.touched, (n, :translation))
    end
    return nothing
end

"""
    recipemodule(name) -> Module

The package a recipe names: already loaded, or loaded now from the active
environment, the same way `using` would find it.
"""
function recipemodule(name::AbstractString)
    for (id, mod) in Base.loaded_modules
        id.name == name && return mod
    end
    return Base.require(Main, Symbol(name))
end

# Loading a script defines methods in a new world. invokelatest is confined to
# this boundary; screens and the compositor remain the normal editor path.
function programbuilder!(root::SceneProgram)
    root.builder === nothing || return root.builder
    if !isempty(root.package)
        mod = recipemodule(root.package)
        isdefined(mod, root.entry) || error("package $(root.package) does not define $(root.entry)")
        root.builder = getfield(mod, root.entry)
        return root.builder
    end
    isfile(root.file) || error("scene program is missing: $(root.file)")
    mod = Module(gensym(:VideoEditorScene))
    Core.eval(mod, :(include(path) = Base.include(@__MODULE__, path)))
    Base.include(mod, root.file)
    isdefined(mod, root.entry) || error("$(root.file) does not define $(root.entry)")
    root.builder = Core.eval(mod, root.entry)
    return root.builder
end

function realize(root::SceneProgram, canvas::NTuple{2, Int})
    built = Base.invokelatest(programbuilder!(root), canvas, deepcopy(root.args))
    built.scene isa Makie.Scene || error("scene program must return a Makie.Scene")
    for name in (:update!, :controls, :activeparams, :sampleparams, :camera, :args)
        haskey(built, name) && error("$(root.entry) returned obsolete procedural hook `$name`; key native Makie plot, camera and light properties instead")
    end
    instance = ProgramInstance(built.scene, SceneOverrides(built.scene), -1,
                               get(built, :preview!, nothing), NaN,
                               Dict{Symbol,RecordedTrack}(),
                               programobjects(get(built, :objects, ())), camerascene(built.scene))
    return built.scene, instance
end

targetscene(p::ProgramInstance) = p.scene
preparepreview!(target, scale) = false
function preparepreview!(target::ProgramInstance, scale)
    target.preview! === nothing && return false
    target.previewscale == scale && return false
    Base.invokelatest(target.preview!, Float64(scale))
    target.previewscale = scale
    return true
end

updatesceneprogram!(target, src, clip, frame, fps; reset = false) = false
function updatesceneprogram!(target::ProgramInstance, src, clip, frame, fps; reset = false)
    changed = reset || target.frame != frame
    if changed
        restoresceneprogram!(target, keyedattributes(clip))
        replayrecordedscene!(target, src, frame)
        target.frame = frame
    end
    target.camerascene === nothing || (src.camera = cameravalues(target.camerascene))
    return changed
end

"A keyed number as the type it lands in: held steps for discrete values."
leafvalue(old::Bool, x) = x >= 0.5
leafvalue(old::Integer, x) = round(typeof(old), x)
leafvalue(old::Enum, x) = typeof(old)(round(Int, x))
leafvalue(old::AbstractString, x::AbstractString) = String(x)
# `Observable{Any}(nothing)` declares an initially unbound data input. Its
# file-backed value need not have a type known before the provider is loaded.
leafvalue(::Nothing, x) = x
leafvalue(old, x) = convert(typeof(old), x)


"Set finished-output scene settings for a scoped operation, restoring preview settings on exit."
function withfinalscenes(f::Function, seq::Sequence)
    sources = unique(c.source for c in seq.clips if c.source isa SceneSource)
    modes = [s.mode for s in sources]
    try
        for s in sources
            s.mode = :bake
            s.pending = nothing
            s.pendingat = -1
        end
        return f()
    finally
        for (s, mode) in zip(sources, modes)
            s.mode = mode
        end
    end
end
