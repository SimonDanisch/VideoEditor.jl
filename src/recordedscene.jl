"Recordings are owned by the retained scene, shared across its cuts."
function scenerecordings!(target::ProgramInstance,src::SceneSource)
    files = src.build isa AbstractDict ? get(src.build,"recordings",nothing) : nothing
    files === nothing && return nothing
    for (path,file) in files
        get!(() -> openrecording(file),target.recordings,Symbol(path))
    end
    return target.recordings
end

function replayrecordedscene!(target::ProgramInstance,src::SceneSource,frame)
    tracks = scenerecordings!(target,src)
    tracks === nothing && return false
    camera = Makie.cameracontrols(target.scene)
    if src.camera === nothing && camera isa Makie.Camera3D
        src.camera = (eye=Vec3f(camera.eyeposition[]),lookat=Vec3f(camera.lookat[]),up=Vec3f(camera.upvector[]))
    end
    for (path,track) in tracks
        setscenevalue!(src,path,valueat(track,frame)) ||
            error("recorded scene property no longer exists: $path")
    end
    return true
end

"A component can follow a whole-vector/colour recording without another track."
function recordedproperty(tracks,path,frame)
    haskey(tracks,path) && return valueat(tracks[path],frame)
    address = scenepath(path)
    address === nothing && return nothing
    name,key,component = address
    whole = Symbol(name,".",key)
    component === nothing || !haskey(tracks,whole) ? nothing : begin
        value = valueat(tracks[whole],frame)
        value isa Colorant ? (red(value),green(value),blue(value),alpha(value))[component] : value[component]
    end
end

"Test whole-array activity from the seek index, without reading large samples."
function recordedactive(tracks,path,a,b)
    haskey(tracks,path) && return recordedchanged(tracks[path],a,b)
    va,vb = recordedproperty(tracks,path,a),recordedproperty(tracks,path,b)
    return va !== nothing && vb !== nothing && !isequal(va,vb)
end

"A recording already restored by this source is its original performance."
function followsrecording(src::SceneSource,p::Param)
    n = p.input
    n === nothing || n.op !== :copy || length(n.inputs) != 1 ? false : begin
        ref = only(n.inputs)
        ref isa RecordedRef && src.build isa AbstractDict &&
            get(get(src.build,"recordings",Dict()),String(p.name),nothing) == ref.path
    end
end

function readsceneproperty(scene,path)
    address = scenepath(path)
    address === nothing && error("invalid scene property path: $path")
    name,key,component = address
    if name === :camera
        camera = Makie.cameracontrols(scene)
        camera isa Makie.Camera3D || error("recorded camera requires a 3D camera")
        property = Dict(:eye=>:eyeposition,:lookat=>:lookat,:up=>:upvector)[key]
        value = getproperty(camera,property)[]
    else
        light = scenelight(scene,name)
        if light !== nothing
            value = Makie.to_value(getfield(light,key))
        else
            plot = Makie.findplot(scene,name)
            plot === nothing && error("scene property not found: $path")
            value = key === :rotation ? rotationdegrees(plot.transformation.rotation[]) :
                key in (:translation,:scale) ? getproperty(plot.transformation,key)[] :
                Makie.to_value(plot.attributes[key])
        end
    end
    return component === nothing ? value : value isa Colorant ?
        (red(value),green(value),blue(value),alpha(value))[component] : value[component]
end

"""
    recordsceneanimation!(clip, directory; progress=nothing)

Construct a fresh instance of the clip's saved scene recipe, advance it once
through all source frames, and stream its native numeric/array inputs to disk.
Discover writable scene properties. Playback constructs the scene but bypasses its simulation
updater, restoring recorded samples before normal editor overrides. Recordings
belong to the source, so existing cuts share the same source clock and samples.
Use a new directory for a new recording. The recipe factory must create fresh
simulation state rather than reuse mutable global state.
"""
function recordsceneanimation!(clip::Clip,directory::AbstractString;
                               progress=nothing)
    src = clip.source
    src isa SceneSource && src.root isa SceneProgram ||
        throw(ArgumentError("recording requires a saved procedural scene"))
    get(src.build,"recordings",nothing) === nothing || error("source already uses a recording")
    # Reuse the loaded Julia factory, not its live scene or mutable simulation.
    root = SceneProgram(src.root.file,src.root.entry,deepcopy(src.root.args))
    root.builder = programbuilder!(src.root)
    backend = getbackend(src.backend)
    tracks = onthread(renderthread(backend)) do
        lock(SCENELOCK) do
            scene,target = realize(root,(src.width,src.height))
            try
                # Only record the source performance; clip overrides remain
                # separately editable and apply after these recorded values.
                unkeyed = Clip(src)
                candidates = Symbol[]
                camera = Makie.cameracontrols(scene)
                if camera isa Makie.Camera3D
                    for field in keys(CAMERAFIELDS), i in 1:3
                        push!(candidates,Symbol("camera.$field[$i]"))
                    end
                end
                for (name,plot) in sceneplots(scene)
                    for field in (:translation,:rotation,:scale), i in 1:3
                        push!(candidates,Symbol("$name.$field[$i]"))
                    end
                    for key in sceneinputkeys(plot)
                        value = Makie.to_value(plot.attributes[key])
                        T = value isa AbstractArray ? eltype(value) : typeof(value)
                        T in values(RECORDEDTYPES) || continue
                        push!(candidates,Symbol("$name.$key"))
                    end
                end
                for (i,light) in [(0,Makie.AmbientLight(scene.compute[:ambient_color][]));
                                  collect(enumerate(Makie.get_lights(scene)))]
                    for key in fieldnames(typeof(light))
                        value = Makie.to_value(getfield(light,key))
                        typeof(value) in values(RECORDEDTYPES) && push!(candidates,Symbol("lights[$i].$key"))
                    end
                end
                chosen = unique(candidates)
                capture = (frame,fps) -> begin
                    updatescenecontrols!(target,unkeyed,frame,fps)
                    Base.invokelatest(target.update!,frame,fps)
                    Dict(path=>readsceneproperty(scene,path) for path in chosen)
                end
                recordanimation(capture,directory,0:src.nframes-1;framerate=src.framerate,progress)
            finally
                Makie.free(scene)
            end
        end
    end
    # Publish the new source behavior only after every recording is complete.
    close(src)
    src.build["recordings"] = Dict(String(path)=>track.path for (path,track) in tracks)
    bakedirty!(clip)
    clip.sequence === nothing || foreach(c -> c.source === src && bakedirty!(c),clip.sequence.clips)
    return tracks
end
