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

`args` declares the recipe's own animated inputs: a NamedTuple (or dictionary)
of `Observable`s the scene derives its plots from. Each becomes an editable
group with timeline lanes, addressed as `"args.name"`. Numbers, `Vec`/`Point`
values and colours key per component; integers, `Bool`s and enums key as held
steps; plain structs key per numeric field (`"args.pose.eye[1]"`). Arrays,
meshes and other data are shown as data inputs. The editor writes each
argument once per frame, only when its value changes, so expensive derived
plots (a wave simulation, a ray trace) recompute only on an actual change.
Keyed arguments belong to the editor: an `update!` must not also set them.

`update! = (frame, fps) -> ...` is optional, for procedural animation. It
receives absolute source frames, so scrubbing, cuts, and farm frames all evaluate
the same animation. It must support seeking in arbitrary order. It runs before
the clip's ordinary scene keyframes, which can override values.

Optional `objects` describes inspector groups as dictionaries with `label`,
`plots` (stable names; first is the pivot), and optional `attributes` to expose.
Transforms move group members together. Omit it to expose all named plots.
The builder may return the same `objects` (NamedTuples or dictionaries) beside
its scene instead, where the code that names the plots also groups them; groups
saved with the recipe take precedence.

The builder may also return `controls`: groups with `name`, `label`,
`sample(frame, fps)` returning a NamedTuple of numeric/vector/colour values, and
`apply!(overrides, frame, fps)`. Each frame samples the original performance,
then applies the clip's keyed overrides before calling the scene updater.
`overrides` contains only edited fields; an empty dictionary restores following
the recipe. These controls use the same inspector, curves and saved parameters
as plot attributes, without serializing live callbacks or GPU objects.
Set a control group's optional `object` to the stable name of its object's pivot
plot to show those controls when that object is picked in a preview. Optional
`sections = [(label = "Face", fields = [:smile, :eyeopen])]` separates related
controls in the inspector without changing their saved parameter paths.
Optional `detail` says what the group is (`"Material"`); the default,
`"Performance"`, is what the inspector's Animation view lists.

An optional `preview!(pixel_scale)` callback can adjust expensive procedural
detail when preview quality changes. It runs before the next animation update,
once per quality change. Full preview and finished output always pass `1.0`;
the callback must restore original detail at that scale. It does not edit the
saved recipe arguments or the clip's parameters.

An optional pure `activeparams(firstframe, lastframe, fps)` callback returns
editable property paths varying over that source-frame interval. The inspector
uses a small interval around the playhead, clamped inside the selected clip.
It must not seek/render the scene or mutate animation state. Without this query,
the active-parameter filter can list editor keyframes and driven values; the
checkbox still exposes every unkeyed recipe control.

The file runs in its own module and may include other files relative to itself.
An optional pure `sampleparams(frame, fps)` returns a dictionary from property
paths to original scalar values. Picking an object fits just its curves over
the selected shot to sparse Bézier anchors using the existing curve fitter
(relative tolerance `1e-4`), retaining their `SceneRef` inputs. Sampling here
supplies the fit, not timeline keys. A gesture edits those same curves and
detaches only the touched property. This query must
not render, seek, or mutate the live performance. Camera curves can use `camera`
without this callback. Recipes without either query remain editable in the
inspector; the editor does not invent their original animation.
An optional pure `camera(frame, fps)` callback returns `(eye, lookat, up)` for
camera-path previews. It must sample without seeking or rendering the scene.
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

mutable struct ProgramControlGroup
    name::Symbol
    label::String
    sample::Any
    apply!::Any
    values::Dict{Symbol, Any}
    object::Union{Nothing, Symbol}
    sections::Vector{NamedTuple}
    detail::String
end

mutable struct ProgramInstance
    scene::Makie.Scene
    update!::Any
    frame::Int
    controls::Vector{ProgramControlGroup}
    preview!::Any
    previewscale::Float64
    recordings::Dict{Symbol,RecordedTrack}
    # The recipe's animated inputs, in declaration order, and the values they were
    # built with: what an argument returns to when its keys are removed.
    args::Vector{Pair{Symbol, Observable}}
    argdefaults::Dict{Symbol, Any}
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
    recipeargs(built) -> Vector{Pair{Symbol, Observable}}

A recipe's declared arguments in declaration order. A NamedTuple keeps its
order; a dictionary is sorted by name, so paths and lanes are stable.
"""
recipeargs(::Nothing) = Pair{Symbol, Observable}[]
recipeargs(args::NamedTuple) = Pair{Symbol, Observable}[k => args[k] for k in keys(args)]
recipeargs(args::AbstractDict) =
    Pair{Symbol, Observable}[Symbol(k) => args[k] for k in sort!(collect(keys(args)); by = string)]

function recipeargument(p::ProgramInstance, name::Symbol)
    for (k, obs) in p.args
        k === name && return obs
    end
    return nothing
end

function programcontrols(built, scene)
    names = Set(first.(sceneplots(scene)))
    push!(names, :camera)
    groups = ProgramControlGroup[]
    for c in get(built, :controls, ())
        name = Symbol(c.name)
        occursin(r"^[A-Za-z_][A-Za-z_0-9]*$", String(name)) ||
            error("invalid scene control name: $name")
        name in names && error("duplicate scene control name: $name")
        push!(names, name)
        owner = get(c, :object, nothing)
        push!(groups, ProgramControlGroup(name, String(c.label), c.sample, c.apply!, Dict(),
                                          owner === nothing ? nothing : Symbol(owner),
                                          NamedTuple[(label = String(s.label), fields = Symbol.(collect(s.fields)))
                                              for s in get(c, :sections, ())],
                                          String(get(c, :detail, "Performance"))))
    end
    return groups
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

programcontrol(::Any, name) = nothing
function programcontrol(p::ProgramInstance, name)
    i = findfirst(g -> g.name === name, p.controls)
    return i === nothing ? nothing : p.controls[i]
end

function updatescenecontrols!(target::ProgramInstance, clip, frame, fps)
    fx = findslot(clip, :scene)
    lookup = Dict(g.name => g for g in target.controls)
    changes = Dict(g.name => Dict{Symbol, Any}() for g in target.controls)
    for group in target.controls
        values = Base.invokelatest(group.sample, frame, fps)
        empty!(group.values)
        for (key, value) in pairs(values)
            rowkind(value) in (:number, :vector, :colour) ||
                error("scene control $(group.name).$key must be numeric, vector or colour")
            group.values[Symbol(key)] = value
        end
    end
    # Walk the clip's parameters once, rather than once per actor/control group.
    if fx !== nothing && fx.enabled[]
        for p in fx.params
            isfollowing(p) && continue
            address = scenepath(p.name)
            address === nothing && continue
            name, key, component = address
            group = get(lookup, name, nothing)
            group !== nothing && haskey(group.values, key) || continue
            overrides = changes[name]
            original = get(overrides, key, group.values[key])
            v = valueat(p, frame)
            overrides[key] = component === nothing ? convert(typeof(original), v) :
                                                    withcomponent(original, component, v)
        end
    end
    for group in target.controls
        Base.invokelatest(group.apply!, changes[group.name], frame, fps)
    end
    return nothing
end

"A scene updater with an optional pure animation-activity query."
struct ProgramAnimation
    update!::Any
    activeparams::Any
    camera::Any
    sampleparams::Any
end
ProgramAnimation(update!,activeparams,camera=nothing) = ProgramAnimation(update!,activeparams,camera,nothing)
(a::ProgramAnimation)(frame, fps) = Base.invokelatest(a.update!, frame, fps)

"Optional pure recipe query for property paths animated in a source-frame interval."
programactiveparams(::Any, firstframe, lastframe, fps) = Symbol[]
function programactiveparams(p::ProgramInstance, firstframe, lastframe, fps)
    u = p.update!
    a = u isa SceneUpdater ? u.callback : u
    a isa ProgramAnimation || return Symbol[]
    a.activeparams === nothing && return Symbol[]
    return Base.invokelatest(a.activeparams, firstframe, lastframe, fps)
end

"Pure original values for timeline curves, without seeking the live scene."
function programsampleparams(p::ProgramInstance, frame, fps)
    u = p.update! isa SceneUpdater ? p.update!.callback : p.update!
    u isa ProgramAnimation && u.sampleparams !== nothing || return nothing
    return Base.invokelatest(u.sampleparams, frame, fps)
end

"Original values for properties touched by editor overrides, held with the live scene."
struct SceneUpdater
    callback::Any
    defaults::Dict{Tuple{Symbol, Symbol}, Any}
    touched::Set{Tuple{Symbol, Symbol}}
end

function SceneUpdater(callback, scene)
    defaults = Dict{Tuple{Symbol, Symbol}, Any}()
    camscene = camerascene(scene)
    camscene === nothing || (defaults[(:camera, :camera)] = cameravalues(camscene))
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
            (key === :visible || rowkind(v) in (:number, :vector, :colour)) || continue
            defaults[(name, key)] = v
        end
    end
    return SceneUpdater(callback, defaults, Set{Tuple{Symbol, Symbol}}())
end
(u::SceneUpdater)(frame, fps) = Base.invokelatest(u.callback, frame, fps)

"""
    restoresceneprogram!(target, keep = Set()) -> nothing

Return every property an editor override touched to its recipe value, except
the plot attributes in `keep`: those are still keyed and are about to be written
again, and restoring them first would make everything derived from them
recompute twice per frame. Transforms are always restored, because grouped
objects move by the difference to their current placement.
"""
function restoresceneprogram!(target::ProgramInstance, keep = Set{Tuple{Symbol, Symbol}}())
    u = target.update!
    u isa SceneUpdater || return nothing
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
lights and arguments are not included.
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
        (name === :camera || name === :args || key in (:translation, :rotation, :scale) ||
         startswith(String(name), "lights[")) && continue
        push!(keep, (name, key))
    end
    return keep
end

function remembersceneoverride!(src, path)
    src.live.target isa ProgramInstance || return nothing
    u = src.live.target.update!
    u isa SceneUpdater || return nothing
    address = scenepath(path)
    address === nothing && return nothing
    name, key, _ = address
    if scenelight(targetscene(src.live.target), name) !== nothing
        push!(u.touched, (name, key))
        return nothing
    end
    name === :camera && (push!(u.touched, (:camera, :camera)); return nothing)
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

"No procedural animation: the scene is moved by its keyframes alone."
staticscene(frame, fps) = nothing

function realize(root::SceneProgram, canvas::NTuple{2, Int})
    built = Base.invokelatest(programbuilder!(root), canvas, deepcopy(root.args))
    built.scene isa Makie.Scene || error("scene program must return a Makie.Scene")
    activity = get(built, :activeparams, nothing)
    camera = get(built, :camera, nothing)
    samples = get(built, :sampleparams, nothing)
    update = get(built, :update!, staticscene)
    updater = activity === nothing && camera === nothing && samples === nothing ? update :
        ProgramAnimation(update, activity, camera, samples)
    args = recipeargs(get(built, :args, nothing))
    for (name, obs) in args
        occursin(r"^[A-Za-z_][A-Za-z_0-9]*$", String(name)) || error("invalid scene argument name: $name")
    end
    defaults = Dict{Symbol, Any}(name => deepcopy(obs[]) for (name, obs) in args)
    instance = ProgramInstance(built.scene, SceneUpdater(updater, built.scene), -1,
                               programcontrols(built, built.scene), get(built, :preview!, nothing), NaN,
                               Dict{Symbol,RecordedTrack}(), args, defaults,
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
        if !replayrecordedscene!(target,src,frame)
            updatescenecontrols!(target, clip, frame, fps)
            Base.invokelatest(target.update!, frame, fps)
        end
        target.frame = frame
    end
    target.camerascene === nothing || (src.camera = cameravalues(target.camerascene))
    return changed
end

# ------------------------------------------------------------ recipe arguments

"""
    argleaves(value) -> Vector{Tuple{Symbol, Union{Nothing, Int}, Real, Bool}}

The keyframable numbers in an argument's value: `(field, component, number,
discrete)`. `field` is the dotted field path inside a struct (`Symbol("")` for
the value itself), `component` indexes a vector or colour. Integers, `Bool`s and
enums are `discrete`: they key as held steps, never as blends.

Arrays and other data have no leaves; they are inputs, not numbers.
"""
argleaves(v) = argleaves!(Tuple{Symbol, Union{Nothing, Int}, Real, Bool}[], Symbol(""), v)

argleaves!(out, field, v::Bool) = push!(out, (field, nothing, Float64(v), true))
argleaves!(out, field, v::Integer) = push!(out, (field, nothing, Float64(v), true))
argleaves!(out, field, v::Enum) = push!(out, (field, nothing, Float64(Int(v)), true))
argleaves!(out, field, v::Real) = push!(out, (field, nothing, Float64(v), false))
function argleaves!(out, field, v::Union{GeometryBasics.Vec, GeometryBasics.Point})
    for i in eachindex(v)
        push!(out, (field, i, Float64(v[i]), false))
    end
    return out
end
function argleaves!(out, field, v::Colorant)
    c = RGBAf(v)
    for (i, x) in enumerate((c.r, c.g, c.b, c.alpha))
        push!(out, (field, i, Float64(x), false))
    end
    return out
end
argleaves!(out, field, ::Union{AbstractArray, AbstractString, Symbol, Function, Nothing}) = out
function argleaves!(out, field, v)
    isstructtype(typeof(v)) || return out
    for name in fieldnames(typeof(v))
        argleaves!(out, field === Symbol("") ? name : Symbol(field, ".", name), getfield(v, name))
    end
    return out
end

"""
    withleaf(value, fields, component, x) -> value

`value` with one keyframable number replaced: the field reached by the dotted
`fields` path, and its `component` if it is a vector or colour. Structs are
rebuilt through their default constructor, field by field.
"""
withleaf(old, fields::Tuple{}, component::Nothing, x) = leafvalue(old, x)
withleaf(old, fields::Tuple{}, component::Integer, x) = withcomponent(old, component, x)
function withleaf(old::NamedTuple, fields::Tuple{Symbol, Vararg{Symbol}}, component, x)
    name = first(fields)
    return merge(old, NamedTuple{(name,)}((withleaf(getfield(old, name), Base.tail(fields), component, x),)))
end
function withleaf(old, fields::Tuple{Symbol, Vararg{Symbol}}, component, x)
    name = first(fields)
    T = typeof(old)
    values = ntuple(fieldcount(T)) do i
        f = fieldname(T, i)
        f === name ? withleaf(getfield(old, f), Base.tail(fields), component, x) : getfield(old, f)
    end
    return T(values...)
end

"A keyed number as the type it lands in: held steps for discrete values."
leafvalue(old::Bool, x) = x >= 0.5
leafvalue(old::Integer, x) = round(typeof(old), x)
leafvalue(old::Enum, x) = typeof(old)(round(Int, x))
leafvalue(old::AbstractString, x::AbstractString) = String(x)
leafvalue(old, x) = convert(typeof(old), x)

argfieldpath(key::Symbol) = key === Symbol("") ? () : Tuple(Symbol.(split(String(key), '.')))

"""
    argpath(name, field, component) -> Symbol

The parameter path of one argument number: `args.speed`, `args.pose.eye[2]`.
"""
argpath(name::Symbol, field::Symbol, component) =
    Symbol("args.", name, field === Symbol("") ? "" : ".", field,
           component === nothing ? "" : "[$component]")

"The argument a path addresses, and the rest of the path inside it."
function argaddress(key::Symbol)
    parts = split(String(key), '.'; limit = 2)
    return Symbol(parts[1]), length(parts) == 1 ? () : Tuple(Symbol.(split(parts[2], '.')))
end

"""
    applyargs!(target, fx, frame) -> nothing

Write every keyed recipe argument for this frame: its default value with each
keyed number replaced. One assignment per argument and only when the value
changed, so a plot derived from it recomputes once, and not at all on a frame
that does not move it.
"""
function applyargs!(target::ProgramInstance, fx, frame::Integer)
    isempty(target.args) && return nothing
    # The recipe's types (a struct, an enum) were defined when its file loaded:
    # building and comparing their values happens in the latest world.
    Base.invokelatest(writeargs!, target, fx, frame)
    return nothing
end

function writeargs!(target::ProgramInstance, fx, frame::Integer)
    values = Dict{Symbol, Any}()
    if fx !== nothing && fx.enabled[]
        for p in fx.params
            isfollowing(p) && continue
            address = scenepath(p.name)
            address === nothing && continue
            name, key, component = address
            name === :args || continue
            arg, fields = argaddress(key)
            haskey(target.argdefaults, arg) || continue
            current = get(values, arg, target.argdefaults[arg])
            values[arg] = withleaf(current, fields, component, valueat(p, frame))
        end
    end
    for (name, obs) in target.args
        new = get(values, name, target.argdefaults[name])
        isequal(obs[], new) || (obs[] = new)
    end
    return nothing
end

"""
The recipe's own value of one argument number, before this clip's keys: what a
parameter that follows the recipe reads.
"""
function argnumber(target::ProgramInstance, key::Symbol, component)
    arg, fields = argaddress(key)
    haskey(target.argdefaults, arg) || return nothing
    v = target.argdefaults[arg]
    for f in fields
        hasfield(typeof(v), f) || return nothing
        v = getfield(v, f)
    end
    component === nothing || (v = v isa Colorant ? (red(v), green(v), blue(v), alpha(v))[component] : v[component])
    return leafnumber(v)
end
leafnumber(v::Real) = Float64(v)
leafnumber(v::Enum) = Float64(Int(v))
leafnumber(::Any) = nothing

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
