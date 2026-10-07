"An editor selection addresses an existing object, never a second scene graph."
struct SceneSelection
    clip::UInt64
    object::Symbol
end

"A reusable independent camera and raster target; all edits go to clip parameters."
mutable struct SceneView
    player::Player
    clip::Clip
    fig::Makie.Figure
    axis::Makie.Axis
    camera::Makie.Scene
    image::Observable{Matrix{RGBA{Float32}}}
    background::Any
    frame::Any
    projectionview::Makie.Mat4f
    handles::Observable{Vector{Point2f}}
    stems::Observable{Vector{Point2f}}
    outline::Observable{Vector{Point2f}}
    path::Observable{Vector{Point2f}}
    pathhandles::Observable{Vector{Point2f}}
    pathframes::Vector{Int}
    mode::Symbol
    drag::Any
    busy::Bool
    dirty::Bool
    closed::Bool
    dimensions::Tuple{Int, Int}
end

sceneselection(player::Player) = get!(
    () -> Observable{Union{Nothing, SceneSelection}}(nothing),
    player.fxwidgets, :scene_selection
)

function selectedscenesections(player::Player, clip::Clip)
    selected = sceneselection(player)[]
    (selected === nothing || selected.clip != clip.id) && return Set{Symbol}()
    return sceneobjectsections(clip.source, selected.object)
end

function sceneobjectsections(src::SceneSource, object)
    object === nothing && return Set{Symbol}()
    names = Set([Symbol(object)])
    if src isa SceneSource && src.live !== nothing && src.live.target isa ProgramInstance
        for group in src.live.target.controls
            group.object === object && push!(names, group.name)
        end
    end
    return names
end

function sceneobjectowner(src::SceneSource, hit)
    hit === nothing && return nothing
    for object in sceneobjects(src), plot in sceneobjectplots(src, object.name)
        found = Ref(false)
        Makie.for_each_atomic_plot(plot) do child
            child === hit && (found[] = true)
        end
        (plot === hit || found[]) && return object.name
    end
    return nothing
end

function selectsceneobject!(player::Player, object; clip = selectedclip(player))
    clip isa Clip && clip.source isa SceneSource || return false
    player.timeline.selected[] == clip.id || (player.timeline.selected[] = clip.id)
    selected = object === nothing ? nothing : SceneSelection(clip.id, Symbol(object))
    current = sceneselection(player)[]
    selected !== nothing && current === selected && return true
    opendock!(player, :effects)
    box = player.fxwidgets[:fxfilterbox]
    box.stored_string[] == "scene" || Makie.set!(box, "scene")
    fx = findslot(clip, :scene)
    fx === nothing && return false
    fx.card === nothing || fx.card.open[] || (fx.card.open = true)
    sections = get(player.fxwidgets, Symbol(:fxsections_, fx.id), nothing)
    # Sample and place the original curves before notifying the shared selection.
    # Its inspector listener then filters once, with the final animation metadata.
    showsceneanimation!(player, clip; owners = sceneobjectsections(clip.source, object), refresh = false)
    sceneselection(player)[] = selected
    fittimelinerow!(player)
    sections === nothing || scrollinspectorto!(player, sections.box)
    setstatus!(player, object === nothing ? "all scene objects" : "selected $(object) · drag XYZ handles in Scene view")
    return true
end

"Fit selected procedural animations to editable Bézier curves; keep recipe inputs intact."
function samplesceneparams!(player, clip, params)
    src = clip.source
    src.live !== nothing && src.live.target isa ProgramInstance || return
    ranges = get!(() -> IdDict{Param,Any}(), player.fxwidgets, :recipe_curve_ranges)
    pending = filter(params) do p
        eltype(p) <: Real && isfollowing(p) && get(ranges, p, nothing) != (clip.src_in, clip.src_out, p.input)
    end
    isempty(pending) && return
    samples = Dict(p.name => Keyframe{Float64}[] for p in pending)
    recordings = scenerecordings!(src.live.target,src)
    for f in clip.src_in:(clip.src_out - 1)
        values = programsampleparams(src.live.target, f, src.framerate)
        for p in pending
            v = if recordings !== nothing
                recordedproperty(recordings,p.name,f)
            elseif values !== nothing && haskey(values, p.name)
                values[p.name]
            elseif startswith(String(p.name), "camera.")
                _, field, component = scenepath(p.name)
                getfield(scenecamerasample(clip, f), field)[component]
            else
                nothing
            end
            v === nothing || push!(samples[p.name], Keyframe(f, Float64(v)))
        end
    end
    for p in pending
        keys = samples[p.name]
        ranges[p] = (clip.src_in, clip.src_out, p.input)
        isempty(keys) && continue
        # Constant properties do not masquerade as animation lanes.
        all(k -> k.value == first(keys).value, keys) && resize!(keys, 1)
        curve = AnimCurve{Float64}(keys, :linear)
        simplify!(curve; tol = 1.0e-4)
        p.curve[] = curve
    end
end

"One selection in the inspector and timeline; picking never changes the performance."
function showsceneanimation!(player, clip; owners = selectedscenesections(player, clip), refresh = true)
    fx = findslot(clip, :scene)
    fx === nothing && return
    chosen = filter(fx.params) do p
        path = scenepath(p.name)
        path !== nothing && first(path) in owners
    end
    samplesceneparams!(player, clip, chosen)
    for p in fx.params
        show = p in chosen && isanimated(p)
        p.visible[] == show || (p.visible[] = show)
        p.view === nothing && continue
        lane = p.view.lane
        lane.highlighted[] == (p in chosen) || (lane.highlighted = p in chosen)
        lane.valuerange[] == paramdisplayrange(p) || (lane.valuerange = paramdisplayrange(p))
    end
    sections = get(player.fxwidgets, Symbol(:fxsections_,fx.id), nothing)
    !refresh || sections === nothing || sections.apply()
    placelanes!(player.timeline, clip)
end

"The scene view displays the frame on screen, including when another clip was selected."
function sceneviewclip(player)
    clips = clipsat(player.sequence, player.playhead[])
    i = findlast(c -> c.source isa SceneSource, clips)
    return i === nothing ? nothing : clips[i]
end

"Pick the raster frame already displayed; ignore seeks and active preview tools."
function picksceneobject!(player::Player, point)
    player.playing[] && return false
    get(player.fxwidgets, :publishedframe, -1) == player.playhead[] || return false
    n = player.playhead[]
    clip = selectedclip(player)
    if clip === nothing || !(clip.start <= n < clipend(clip))
        index = clipat(player.sequence, n)
        index === nothing && return false
        clip = player.sequence.clips[index]
    end
    sf = sourceframe(clip, n)
    src = clip.source
    src isa SceneSource && src.live !== nothing || return false
    backend = getbackend(src.live.backend)
    isdefined(backend, :raster_pick_data) || return false
    result = onthread(renderthread(backend)) do
        lock(SCENELOCK) do
            src.at == sf || return nothing
            frame = backend.raster_pick_data(src.live.screen)
            frame === nothing && return nothing
            # The layer matrix includes placement and crop, even when source and
            # canvas dimensions happen to be equal.
            q = layermatrix(clip, (src.width, src.height), size(player.frame[]), sf) *
                Vec3f(point[1], point[2], 1)
            hit, _ = backend.pick_frame(frame, Vec2d(q[1] - 0.5, src.height - q[2] + 0.5))
            return (owner = sceneobjectowner(src, hit),)
        end
    end
    result === nothing && return false
    player.timeline.selected[] == clip.id || (player.timeline.selected[] = clip.id)
    selectsceneobject!(player, result.owner; clip)
    return true
end

function installsceneselection!(player::Player)
    on(events(player.previewaxis.scene).mousebutton; priority = 15) do event
        event.button === Mouse.left && event.action === Mouse.press || return Consume(false)
        (
            player.cropmode[] || player.onpick !== nothing || player.mattebrush !== nothing ||
                ispressed(player.fig, Keyboard.left_alt | Keyboard.right_alt | Keyboard.left_shift | Keyboard.right_shift)
        ) &&
            return Consume(false)
        Makie.receives_events(player.previewaxis.scene) && is_mouseinside(player.previewaxis.scene) ||
            return Consume(false)
        return Consume(picksceneobject!(player, Point2f(mouseposition(player.previewaxis.scene))))
    end
    on(player.playhead) do _
        refreshsceneview!(player)
    end
    on(player.edited) do _
        refreshsceneview!(player)
    end
    on(player.frame) do _
        refreshsceneview!(player)
    end
    register_scene_view_button!(player)
    return nothing
end

function sceneviewselection(view)
    s = sceneselection(view.player)[]
    return s !== nothing && s.clip == view.clip.id ? s.object : nothing
end

"Pure camera sampling; recipe values first, then the clip's existing overrides."
function scenecamerasample(clip::Clip, frame::Int)
    src = clip.source
    baseline = src.camera
    target = src.live.target
    if target isa ProgramInstance
        updater = target.update! isa SceneUpdater ? target.update!.callback : target.update!
        if updater isa ProgramAnimation && updater.camera !== nothing
            baseline = Base.invokelatest(updater.camera, frame, src.framerate)
        end
    end
    baseline === nothing && return nothing
    fx = findslot(clip, :scene)
    return NamedTuple{(:eye, :lookat, :up)}(
        ntuple(3) do j
            field = (:eye, :lookat, :up)[j]
            original = get(baseline, field, Vec3f(0, 0, 1))
            Vec3f(
                ntuple(3) do i
                    p = fx === nothing ? nothing : param(fx, Symbol("camera.", field, "[", i, "]"))
                    p === nothing || isfollowing(p) ? original[i] : valueat(p, frame)
                end
            )
        end
    )
end

function scenevieworigin(view, name)
    src = view.clip.source
    if name === :camera
        return Vec3f(scenecamerasample(view.clip, sourceframe(view.clip, view.player.playhead[])).eye)
    end
    plots = sceneobjectplots(src, name)
    isempty(plots) && return nothing
    q = first(plots).transformation.model[] * Vec4f(0, 0, 0, 1)
    return Vec3f(q[1:3]) / q[4]
end

"Frame the selected object with the working camera; leave the film camera untouched."
function framesceneselection!(view::SceneView)
    name = sceneviewselection(view)
    name === nothing && return false
    if name === :camera
        origin = scenevieworigin(view, name)
        bounds = Makie.Rect3f(origin - Vec3f(.5), Vec3f(1))
    else
        plots = sceneobjectplots(view.clip.source, name)
        isempty(plots) && return false
        bounds = reduce(union, Makie.boundingbox.(plots))
    end
    padding = Vec3f(max(.1, .08norm(bounds.widths)))
    bounds = Makie.Rect3f(bounds.origin - padding, bounds.widths + 2padding)
    Makie.update_cam!(view.camera, Makie.cameracontrols(view.camera), bounds, true)
    return true
end

function sceneviewaxes(view, name, origin)
    length = Float32(max(norm(Makie.cameracontrols(view.camera).eyeposition[] - origin) * 0.14, 0.01))
    basis = Makie.Mat3f(1.0I)
    if name !== :camera
        plot = first(sceneobjectplots(view.clip.source, name))
        basis = Makie.Mat3f(plot.transformation.parent_model[][1:3, 1:3])
    end
    return [(direction = Vec3f(basis[:, i]), units = length / max(norm(basis[:, i]), 1.0f-5)) for i in 1:3]
end

function updatescenehandles!(view::SceneView)
    view.closed && return
    name = sceneviewselection(view)
    empty!(view.pathframes)
    if name === nothing || view.clip.source.live === nothing
        view.handles[] = fill(Point2f(NaN), 3);view.stems[] = fill(Point2f(NaN), 6);view.outline[] = Point2f[]
        view.path[] = Point2f[];view.pathhandles[] = Point2f[]
        return
    end
    origin = scenevieworigin(view, name)
    origin === nothing && return
    axes = sceneviewaxes(view, name, origin)
    start = viewproject(view, origin)
    ends = [viewproject(view, origin + a.direction * a.units) for a in axes]
    view.handles[] = ends
    view.stems[] = reduce(vcat, [[start, p] for p in ends])
    return if name === :camera
        # At most 65 CPU samples per selected shot, plus all authored keys.
        clip = view.clip;fx = findslot(clip, :scene)
        keys = sort!(
            unique(
                [
                    k.frame for p in fx.params if startswith(String(p.name), "camera.eye[")
                        && !isfollowing(p) for k in p.curve[].keys if clip.src_in <= k.frame < clip.src_out
                ]
            )
        )
        frames = sort!(unique([round.(Int, range(clip.src_in, clip.src_out - 1; length = min(65, clip.src_out - clip.src_in)));keys]))
        view.path[] = [viewproject(view, scenecamerasample(clip, f).eye) for f in frames]
        view.pathframes = keys
        view.pathhandles[] = [viewproject(view, scenecamerasample(clip, f).eye) for f in keys]
        view.outline[] = Point2f[]
    else
        view.path[] = Point2f[];view.pathhandles[] = Point2f[]
        corners = Point2f[]
        for plot in sceneobjectplots(view.clip.source, name)
            box = Makie.boundingbox(plot)
            lo, hi = extrema(box)
            points = [Point3f(x, y, z) for x in (lo[1], hi[1]),y in (lo[2], hi[2]),z in (lo[3], hi[3])]
            for i in CartesianIndices(points),axis in 1:3
                i[axis] == 1 || continue
                j = CartesianIndex(ntuple(k -> k == axis ? 2 : i[k], 3))
                append!(corners, [viewproject(view, points[i]), viewproject(view, points[j]), Point2f(NaN)])
            end
        end
        view.outline[] = corners
    end
end

function startscenehandledrag!(view, point)
    name = sceneviewselection(view)
    name === nothing && return false
    origin = scenevieworigin(view, name)
    origin === nothing && return false
    selected = findfirst(p -> norm(p - point) < 14, view.handles[])
    selected === nothing && return false
    ax = sceneviewaxes(view, name, origin)[selected]
    screenstart = viewproject(view, origin)
    screenaxis = view.handles[][selected] - screenstart
    norm(screenaxis) > 3 || return false
    frame = sourceframe(view.clip, view.player.playhead[])
    field = name === :camera ? :eye : view.mode
    path = Symbol(name, ".", field, "[", selected, "]")
    fx = findslot(view.clip, :scene);p = param(fx, path)
    # Actor position belongs to its performance when the recipe exposes it.
    # Moving the rendered plot alone would freeze its world-space animation.
    if name !== :camera && field === :translation && view.clip.source.live.target isa ProgramInstance
        group = findfirst(g -> g.object === name, view.clip.source.live.target.controls)
        if group !== nothing
            localpath = Symbol(view.clip.source.live.target.controls[group].name, ".local_position[", selected, "]")
            localparam = param(fx, localpath)
            if localparam !== nothing
                p, path = localparam, localpath
            end
        end
    end
    p === nothing && return false
    selectsceneobject!(view.player, name; clip = view.clip)
    p.visible[] = true
    sections = get(view.player.fxwidgets, Symbol(:fxsections_, fx.id), nothing)
    sections === nothing || sections.apply()
    p.view === nothing || scrollinspectorto!(view.player, p.view.control)
    placelanes!(view.player.timeline, view.clip)
    view.drag = (;
        kind = :axis, clip = view.clip, object = name, parameter = path, axis = selected,
        field, frame, start = Point2f(point), screenaxis, units = ax.units,
        original = valueat(p, frame), snapped = Ref(false),
    )
    return true
end

function viewunproject(view, point, depth)
    w, h = view.dimensions
    p = inv(view.projectionview) * Vec4f(2point[1] / w - 1, 1 - 2point[2] / h, depth, 1)
    return Vec3f(p[1:3]) / p[4]
end

function startscenepathdrag!(view, point)
    sceneviewselection(view) === :camera || return false
    i = findfirst(p -> norm(p - point) < 12, view.pathhandles[])
    i === nothing && return false
    frame = view.pathframes[i]
    original = scenecamerasample(view.clip, frame).eye
    p = view.projectionview * Vec4f(original[1], original[2], original[3], 1)
    view.drag = (;
        kind = :path, clip = view.clip, frame, original, depth = p[3] / p[4],
        start = Point2f(point), snapped = Ref(false),
    )
    return true
end

function scenehandledragto!(view, point)
    d = view.drag
    d === nothing && return false
    delta = Point2f(point) - d.start
    norm(delta) > 1 || return true
    if !d.snapped[]
        snapshot!(view.player);d.snapped[] = true
    end
    if d.kind === :path
        shift = viewunproject(view, point, d.depth) - viewunproject(view, d.start, d.depth)
        fx = findslot(d.clip, :scene)
        changed = Param[]
        for i in 1:3
            p = param(fx, Symbol("camera.eye[", i, "]"))
            setkey!(p, d.frame, Float64(d.original[i] + shift[i]))
            push!(changed, p)
        end
        view.player.lastslidersnap = Inf
        editedcurve!(view.player, d.clip; parameter = first(changed))
        return true
    end
    amount = dot(delta, d.screenaxis) / dot(d.screenaxis, d.screenaxis)
    value = d.field === :rotation ? d.original + 90 * amount :
        d.field === :scale ? max(0.001, d.original * (1 + amount)) : d.original + d.units * amount
    p = param(findslot(d.clip, :scene), d.parameter)
    view.player.lastslidersnap = Inf
    editparam!(view.player, d.clip, p, Float64(value); frame = d.frame)
    return true
end

function finishscenehandledrag!(view)
    view.drag === nothing && return false
    view.drag = nothing
    view.player.lastslidersnap = -Inf
    return true
end

function wirescenehandles!(view)
    scene = view.camera
    on(events(view.fig).mousebutton; priority = 60) do event
        event.button === Mouse.left || return Consume(false)
        if event.action === Mouse.release
            return Consume(finishscenehandledrag!(view))
        end
        event.action === Mouse.press && is_mouseinside(scene) || return Consume(false)
        view.busy && return Consume(true)
        point = Point2f(mouseposition(view.axis.scene))
        startscenepathdrag!(view, point) && return Consume(true)
        startscenehandledrag!(view, point) && return Consume(true)
        view.frame === nothing && return Consume(false)
        src = view.clip.source;backend = getbackend(src.live.backend)
        owner = onthread(renderthread(backend)) do
            lock(SCENELOCK) do
                hit, _ = backend.pick_frame(view.frame, Vec2d(point[1], view.dimensions[2] - point[2]))
                sceneobjectowner(src, hit)
            end
        end
        selectsceneobject!(view.player, owner; clip = view.clip)
        return Consume(true)
    end
    on(events(view.fig).mouseposition; priority = 150) do _
        view.drag === nothing && return Consume(false)
        return Consume(scenehandledragto!(view, Point2f(mouseposition(view.axis.scene))))
    end
    return on(events(view.fig).keyboardbutton; priority = 150) do event
        (is_mouseinside(view.camera) && !editingtext(view.fig.scene)) || return Consume(false)
        event.action === Keyboard.press && event.key === Keyboard.z &&
            ispressed(view.fig, Keyboard.left_control | Keyboard.right_control) || return Consume(false)
        ispressed(view.fig, Keyboard.left_shift | Keyboard.right_shift) ? redo!(view.player) : undo!(view.player)
        return Consume(true)
    end
end


function register_scene_view_button!(player)
    monitor = GridLayout(player.fig[1, 3])
    toolbar = GridLayout(monitor[1, 1]; tellheight = true)
    panes = GridLayout(monitor[2, 1]; default_colgap = 8)
    panes[1, 1] = player.previewaxis
    work = GridLayout(panes[1, 2])
    colsize!(panes, 2, Makie.Fixed(0))
    mode = Menu(toolbar[1, 1]; options = [("Film preview", :film), ("Scene view", :scene),
        ("Film + scene", :both)], default = 1, width = 150, fontsize = 12)
    btn = Button(toolbar[1, 2]; label = "Scene view", fontsize = 12, width = 100)
    clear = Button(toolbar[1, 3]; label = "All objects", fontsize = 12, width = 100)
    Label(toolbar[1, 4], "Click an object to edit it"; tellwidth = false,
        halign = :left, fontsize = 11)
    merge!(player.fxwidgets, Dict(:previewpanes => panes, :sceneworkspace => work,
        :previewlayout => mode, :sceneviewbutton => btn, :sceneclearbutton => clear))
    on(mode.selection) do value
        value === :film ? setpreviewlayout!(player, :film) : opensceneview!(player; layout = value)
    end
    on(_ -> opensceneview!(player), btn.clicks)
    on(_ -> selectsceneobject!(player, nothing), clear.clicks)
    return nothing
end

function setpreviewlayout!(player, mode)
    panes = player.fxwidgets[:previewpanes]
    view = get(player.fxwidgets, :sceneview, nothing)
    Makie.GridLayoutBase.with_updates_suspended(panes) do
        player.previewaxis.blockscene.visible[] = mode !== :scene
        colsize!(panes, 1, mode === :scene ? Makie.Fixed(0) : mode === :both ? Makie.Relative(0.4) : Makie.Relative(1))
        colsize!(panes, 2, mode === :film ? Makie.Fixed(0) : mode === :both ? Makie.Relative(0.6) : Makie.Relative(1))
        if view !== nothing
            Makie.set_content_visible!(player.fxwidgets[:sceneworkspace], mode !== :film)
            view.camera.visible[] = mode !== :film
        end
    end
    menu = player.fxwidgets[:previewlayout]
    index = findfirst(o -> last(o) === mode, menu.options[])
    menu.i_selected[] == index || (menu.i_selected[] = index)
    return nothing
end

"Show an editable 3D view without changing the film camera or duplicating geometry."
function opensceneview!(player::Player; layout = :both)
    clip = sceneviewclip(player)
    if clip === nothing
        setstatus!(player, "select a live scene clip to open its Scene view")
        return nothing
    end
    src = clip.source
    if src.live === nothing || !isdefined(getbackend(src.live.backend), :raster_view)
        setstatus!(player, "Scene view needs a RayMakie raster preview")
        return nothing
    end
    old = get(player.fxwidgets, :sceneview, nothing)
    if old !== nothing && !old.closed
        setsceneviewclip!(old, clip)
        setpreviewlayout!(player, layout)
        requestsceneview!(old)
        return old
    end
    fig = player.fig
    workspace = player.fxwidgets[:sceneworkspace]
    workspace.tellwidth[] = false
    workspace.tellheight[] = false
    workspace.width[] = Makie.Relative(1)
    colsize!(workspace, 1, Makie.Relative(1))
    toolbar = GridLayout(workspace[1, 1])
    toolbar.tellwidth[] = false
    mode = Menu(
        toolbar[1, 1]; options = [("Move", :translation), ("Rotate", :rotation), ("Scale", :scale)],
        default = 1, width = 100
    )
    player.fxwidgets[:sceneviewmode] = mode
    camera = Button(toolbar[1, 2]; label = "Camera path", width = 100)
    match = Button(toolbar[1, 3]; label = "Match film view", width = 110)
    focus = Button(toolbar[1, 4]; label = "Frame selection", width = 110)
    ax = Axis(workspace[2, 1]; yreversed = true, backgroundcolor = :black)
    rowsize!(workspace, 2, Makie.Auto(1))
    hidedecorations!(ax);hidespines!(ax)
    for interaction in (:rectanglezoom, :dragpan, :scrollzoom)
        deregister_interaction!(ax, interaction)
    end
    image = Observable(fill(RGBA{Float32}(0, 0, 0, 1), 640, 480))
    image!(ax, 0 .. 640, 0 .. 480, image; interpolate = true)
    limits!(ax, 0, 640, 480, 0)
    camera_scene = Makie.Scene(fig.scene; viewport = ax.scene.viewport, clear = false, camera = cam3d!)
    Makie.cam3d!(
        camera_scene; rotation_button = Mouse.right, translation_button = Mouse.middle,
        near = 0.01, far = 10000.0, clipping_mode = :static
    )
    eye = src.camera.eye; look = src.camera.lookat
    update_cam!(camera_scene, eye, look, src.camera.up)
    handles = Observable(fill(Point2f(NaN), 3));stems = Observable(fill(Point2f(NaN), 6));outline = Observable(Point2f[])
    path = Observable(Point2f[]);pathhandles = Observable(Point2f[])
    lines!(ax, outline; color = (:cyan, 0.8), linewidth = 1.5)
    lines!(ax, stems; color = [:red, :red, :green, :green, :dodgerblue, :dodgerblue], linewidth = 3)
    scatter!(ax, handles; color = [:red, :green, :dodgerblue], markersize = 14)
    lines!(ax, path; color = :gold, linewidth = 2)
    scatter!(ax, pathhandles; color = :gold, markersize = 11)
    Label(
        workspace[3, 1], "Drag XYZ to edit · right-drag orbits · middle-drag pans · wheel zooms";
        fontsize = 12, color = :white, tellwidth = false
    )
    view = SceneView(
        player, clip, fig, ax, camera_scene, image, nothing, nothing,
        Makie.Mat4f(1.0I), handles, stems, outline, path, pathhandles, Int[], :translation,
        nothing, false, false, false, (640, 480)
    )
    player.fxwidgets[:sceneview] = view
    on(mode.selection) do selected
        selected === nothing || (view.mode = selected; updatescenehandles!(view))
    end
    function syncobjects!()
        selected = sceneviewselection(view)
        modes = selected === :camera ? [("Move", :translation)] :
            [("Move", :translation), ("Rotate", :rotation), ("Scale", :scale)]
        mode.options[] == modes || (mode.options[] = modes)
        selected === :camera && mode.i_selected[] != 1 && (mode.i_selected[] = 1)
    end
    on(camera.clicks) do _
        selectsceneobject!(player, :camera; clip = view.clip)
        c = view.clip.source.camera
        offset = c.eye - c.lookat
        norm(offset) > 0 || (offset = Vec3f(3, -4, 3))
        update_cam!(camera_scene, c.eye + offset * 0.8, c.eye, c.up)
        requestsceneview!(view)
    end
    on(_ -> framesceneselection!(view), focus.clicks)
    on(match.clicks) do _
        c = view.clip.source.camera
        update_cam!(camera_scene, c.eye, c.lookat, c.up)
    end
    on(_ -> requestsceneview!(view), camera_scene.camera.projectionview)
    on(sceneselection(player)) do _
        view.closed || (syncobjects!(); updatescenehandles!(view))
    end
    push!(
        fig.scene.deregister_callbacks, on(player.frame) do _
            view.closed || syncobjects!()
        end
    )
    on(events(fig).window_open) do open
        open || (view.closed = true; freesceview!(view))
    end
    wirescenehandles!(view)
    syncobjects!()
    setpreviewlayout!(player, layout)
    requestsceneview!(view)
    return view
end

"Capture recipe motion as ordinary curves, retaining the new key as an anchor."
function scenecameracurve(samples, field, i, lo, at, hi)
    curves = map(((lo, at), (at, hi))) do (a, b)
        curve = AnimCurve{Float64}([Keyframe(f, Float64(getfield(samples[f], field)[i])) for f in a:b], :linear)
        simplify!(curve; tol = 1.0e-4)
        setkey!(curve, at, Float64(getfield(samples[at], field)[i]))
        curve
    end
    left, right = curves
    a, b = last(left.keys), first(right.keys)
    ease = hashandle(a.inhandle) || hashandle(b.outhandle) ? :bezier : :linear
    anchor = Keyframe{Float64}(at, a.value, ease, a.inhandle, b.outhandle)
    return AnimCurve{Float64}([left.keys[1:(end - 1)];anchor;right.keys[2:end]], :linear)
end

function addscenecamerakey!(view)
    clip = view.clip;fx = findslot(clip, :scene)
    frame = sourceframe(clip, view.player.playhead[])
    snapshot!(view.player)
    # Sample on the CPU, then use the existing curve simplifier. Three linear
    # anchors alone would straighten the recipe's original eased camera motion.
    frames = clip.src_in:(clip.src_out - 1)
    samples = Dict(f => scenecamerasample(clip, f) for f in frames)
    changed = Param[]
    for field in (:eye, :lookat, :up),i in 1:3
        p = param(fx, Symbol("camera.", field, "[", i, "]"))
        if isfollowing(p)
            curve = scenecameracurve(samples, field, i, clip.src_in, frame, clip.src_out - 1)
            p.input = nothing
            p.curve[] = curve
        else
            setkey!(p, frame, Float64(getfield(samples[frame], field)[i]))
        end
        push!(changed, p)
    end
    editedcurve!(view.player, clip; parameter = first(changed))
    selectsceneobject!(view.player, :camera; clip)
    requestsceneview!(view)
    return nothing
end

function freesceview!(view)
    view.clip.source.live === nothing && return
    backend = getbackend(view.clip.source.live.backend)
    onthread(renderthread(backend)) do
        lock(SCENELOCK) do
            backend.close_raster_view!(view.clip.source.live.screen)
            view.background === nothing || Mantle.free!(view.background.buffer)
        end
    end
    view.background = nothing
    return view.frame = nothing
end

function refreshsceneview!(player)
    view = get(player.fxwidgets, :sceneview, nothing)
    (view === nothing || view.closed) && return
    clip = sceneviewclip(player)
    return if clip !== nothing
        setsceneviewclip!(view, clip)
        requestsceneview!(view)
    end
end

function setsceneviewclip!(view, clip)
    view.clip.source === clip.source || freesceview!(view)
    view.clip = clip
    return nothing
end

"Coalesce pointer/camera updates; never queue one full render per mouse event."
function requestsceneview!(view::SceneView)
    (view.closed || !view.camera.visible[]) && return
    view.dirty = true
    view.busy && return
    view.busy = true
    return @async try
        while view.dirty && !view.closed
            view.dirty = false
            clip = view.clip
            src = clip.source
            src.live === nothing && break
            backend = getbackend(src.live.backend)
            sf = sourceframe(clip, view.player.playhead[])
            result = onthread(renderthread(backend)) do
                lock(SCENELOCK) do
                    src.at == sf || return nothing
                    w, h = view.dimensions
                    if view.background === nothing
                        buffer = Mantle.Buffer(
                            Mantle.todevice(src.live.screen.config.device),
                            fill(RGBA{Float32}(0.025, 0.025, 0.03, 1), h * w)
                        )
                        view.background = (; buffer, pixels = reshape(Mantle.storage(buffer), h, w))
                    end
                    backend.raster_view(src.live.screen, view.camera, view.background.pixels)
                end
            end
            if result !== nothing && !view.closed && view.clip === clip
                view.frame = result.frame
                view.projectionview = result.projectionview
                # The monitor's coordinates count from the image's top left.
                view.image[] = permutedims(result.image)
                updatescenehandles!(view)
            end
            yield()
        end
    catch error
        setstatus!(view.player, "Scene view: $(sprint(showerror, error))")
        @error "Scene view failed" exception = (error, catch_backtrace())
    finally
        view.busy = false
    end
end

function viewproject(view, point)
    q = view.projectionview * Vec4f(point[1], point[2], point[3], 1)
    q[4] > 0 || return Point2f(NaN)
    w, h = view.dimensions
    return Point2f((q[1] / q[4] + 1) * w / 2, (1 - q[2] / q[4]) * h / 2)
end

# Implemented below with the same Param curves used by the inspector.
