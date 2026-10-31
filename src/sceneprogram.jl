"A saved Julia scene recipe. Its GPU objects are constructed lazily on the render thread."
mutable struct SceneProgram
    file::String
    entry::Symbol
    args::Dict{String, Any}
    builder::Any  # loaded recipe, reused when its screen is rebuilt
end

SceneProgram(file::String, entry::Symbol, args::Dict{String, Any}) =
    SceneProgram(file, entry, args, nothing)

SceneProgram(d::AbstractDict) = SceneProgram(abspath(String(d["file"])),
    Symbol(get(d, "entry", "buildscene")),
    Dict{String, Any}(get(d, "args", Dict{String, Any}())))

"""
    programscene(file; entry=:buildscene, args=Dict()) -> Dict

A project-saveable recipe for a procedural Makie animation. The Julia file
defines `entry(canvas, args)`, returning `(scene=scene, update! = (frame, fps)->...)`.
The updater receives absolute source frames, so scrubbing, cuts, and farm frames
all evaluate the same animation. It must support seeking in arbitrary order.
It runs before the clip's ordinary scene keyframes, which can override values.

Optional `objects` describes inspector groups as dictionaries with `label`,
`plots` (stable names; first is the pivot), and optional `attributes` to expose.
Transforms move group members together. Omit it to expose all named plots.

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
end

mutable struct ProgramInstance
    scene::Makie.Scene
    update!::Any
    frame::Int
    controls::Vector{ProgramControlGroup}
    preview!::Any
    previewscale::Float64
    recordings::Dict{Symbol,RecordedTrack}
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
                                              for s in get(c, :sections, ())]))
    end
    return groups
end

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
    camera = Makie.cameracontrols(scene)
    if camera isa Makie.Camera3D
        defaults[(:camera, :camera)] = (eye = Vec3f(camera.eyeposition[]),
            lookat = Vec3f(camera.lookat[]), up = Vec3f(camera.upvector[]))
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
            rowkind(v) in (:number, :vector, :colour) || continue
            defaults[(name, key)] = v
        end
    end
    return SceneUpdater(callback, defaults, Set{Tuple{Symbol, Symbol}}())
end
(u::SceneUpdater)(frame, fps) = Base.invokelatest(u.callback, frame, fps)

function restoresceneprogram!(target::ProgramInstance)
    u = target.update!
    u isa SceneUpdater || return nothing
    for address in u.touched
        haskey(u.defaults, address) || continue
        name, key = address
        if scenelight(target.scene, name) !== nothing
            setscenelight!(target.scene, name, key, u.defaults[address])
            continue
        end
        if name === :camera && key === :camera
            v = u.defaults[address]
            Makie.update_cam!(target.scene, Makie.cameracontrols(target.scene), v.eye, v.lookat, v.up)
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

# Loading a script defines methods in a new world. invokelatest is confined to
# this boundary; screens and the compositor remain the normal editor path.
function programbuilder!(root::SceneProgram)
    root.builder === nothing || return root.builder
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
    activity = get(built, :activeparams, nothing)
    camera = get(built, :camera, nothing)
    samples = get(built, :sampleparams, nothing)
    updater = activity === nothing && camera === nothing && samples === nothing ? built.update! :
        ProgramAnimation(built.update!, activity, camera, samples)
    instance = ProgramInstance(built.scene, SceneUpdater(updater, built.scene), -1,
                               programcontrols(built, built.scene), get(built, :preview!, nothing), NaN,
                               Dict{Symbol,RecordedTrack}())
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
        restoresceneprogram!(target)
        if !replayrecordedscene!(target,src,frame)
            updatescenecontrols!(target, clip, frame, fps)
            Base.invokelatest(target.update!, frame, fps)
        end
        target.frame = frame
    end
    camera = Makie.cameracontrols(target.scene)
    if camera isa Makie.Camera3D
        src.camera = (eye = Vec3f(camera.eyeposition[]), lookat = Vec3f(camera.lookat[]),
                      up = Vec3f(camera.upvector[]))
    end
    return changed
end

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
