# A Makie scene as a clip source.
#
# Everything drawn that is not decoded video comes through here: a 3D animation, a
# title, a lower third, a progress bar, a subtitle. They are clips on a track,
# with an in point, an out point, an effect stack, a placement and keyframes,
# because that is what they always were in everything but implementation.
#
# What they used to be was an `Overlay`: a separate list on the `Sequence`, drawn
# over the finished canvas by a separate pass with a separate state type, its own
# keyframe world (timeline frames, not source frames), its own card builder, its
# own serialization, its own span arithmetic and its own bake. Every feature of a
# clip had to be written a second time for it, or it simply did not exist for
# overlays — which is why a 3D scene could animate for weeks with no card to see
# a single one of its numbers on.


"""
    scenespecof(clip) -> Union{Nothing, SceneSpec}

The scene a clip draws, which is its source's: that is what a source is.

It used to be a `Param{SceneSpec}` on the clip's `:scene` effect, so that it would
be saved. But a spec is not a parameter: nothing keyframes it, no widget edits it,
and it contributes no rows — those come from the scene that gets built. It is what
the clip shows, which is the definition of its source.
"""
scenespecof(clip::Clip) = scenespecof(clip.source)
scenespecof(src::SceneSource) = src.root
scenespecof(::ClipSource) = nothing

"Release a source's live render resources; its saved recipe can be realized again."
function Base.close(src::SceneSource)
    live=src.live
    if live!==nothing
        closeretired!(live)
        onthread(renderthread(getbackend(live.backend))) do
            lock(SCENELOCK) do
                applicable(close,live.screen) && close(live.screen)
            end
        end
        src.live=nothing
    end
    src.pending=nothing
    src.pendingat=-1
    src.at=-1
    src.samples=0
    return nothing
end

"""
    prerender!(source, clip, frame) -> nothing

Have the source draw this frame before the graph runs.

Nothing for a decoder: `decodesource` is its version of the same idea, called
from `update!` for the reason spelled out there.

A scene draws with resources that belong to ONE thread — GLMakie's screen to
thread 1, a Mantle-backed renderer's Vulkan context to the worker's — while the
composite runs on whichever thread owns the render engine's GPU context.
Drawing inside the pass body therefore zigzags caller → owner → caller once per
frame, and the hop back waits for the editor's renderloop to reach a yield.
Measured on the lego project: 22.5 ms of waiting against 7.3 ms of drawing, and
playback of a 60 fps timeline at 10–12 fps.

Called while the caller is still on thread 1, right before `runowned` hands the
frame to the worker: the hop inside [`sceneframe!`](@ref) is then a no-op for a
GLMakie scene and one straight hop to the worker for a Mantle-backed one — the
zigzag never happens.
"""
prerender!(::ClipSource, ::Clip, ::Integer) = nothing

function prerender!(src::SceneSource, clip::Clip, sf::Integer; pixel_scale::Real = 1,
                    detail_scale::Real = pixel_scale)
    # Between frames, on thread 1: the one point where a screen a backend switch
    # replaced can be destroyed without taking GL objects out from under a frame
    # that is being drawn. See [`closeretired!`](@ref).
    closeretired!(src.live)
    updatesource!(src, clip, sf)          # settles `at`, and resets the sample count
    src.pending = sceneframe!(src, clip, (src.width, src.height); pixel_scale,detail_scale)
    src.pendingat = Int(sf)
    return nothing
end

"""
    prerenderscenes!(clips, n) -> nothing

[`prerender!`](@ref) every clip of a frame, by timeline frame.

The three places that hand a frame to the render engine call this immediately
before they do — `compositeframe!`, `presentclipframe!` and `presentgpu!`. Not
inside `composite`: by then the thread has already changed, which is the whole
problem this avoids.
"""
function prerenderscenes!(clips, n::Integer; pixel_scale::Real = 1)
    for clip in clips
        if clip.source isa SceneSource
            prerender!(clip.source, clip, sourceframe(clip, n); pixel_scale)
        else
            prerender!(clip.source, clip, sourceframe(clip, n))
        end
    end
    return nothing
end

"""
    takepending!(src, sf) -> image | nothing

The frame `prerender!` drew for `sf`, taken (so it is used once), or `nothing`
when nobody pre-rendered — the export and the bake run on one thread and draw in
the pass body, which is correct there and costs them nothing.
"""
function takepending!(src::SceneSource, sf::Integer)
    src.pendingat == Int(sf) || return nothing
    img = src.pending
    src.pending = nothing
    src.pendingat = -1
    return img
end

"""
One scene render at a time, per process.

Two tasks reach here — the frame the caller is producing and whatever the ui
queue is draining — and on a Mantle backend both record into the SAME submit
channel. A channel's hold frames are a STACK, pushed by `acquire!` and popped by
the submission, which is only correct while recordings nest. Interleaved they do
not: the inner submission pops the outer's frame, and the outer's next `hold!`
throws `no recording is open on this channel` from inside a render that was
otherwise fine.

`onthread` does not prevent it. When the caller is already on the render thread
it runs INLINE, and when it is not it waits on a channel — which yields, so
the other task runs on the same thread and interleaves anyway.

Held across the whole render, not just the emit: building the screen and writing
the frame's values both record too.
"""
const SCENELOCK = ReentrantLock()

"""
    sceneframe!(src, spec, dims; exact = false) -> Matrix{RGBA{N0f8}}

One frame of the scene, at `dims`.

The standing scene is reused unless the canvas or the backend changed; only the
values this frame carries are written onto it (see [`liveframe!`](@ref)). The
result is already the plane format — RGBA with the scene's own coverage, so what
it did not draw on is uncovered and the clip below shows through.

`exact` is the finished-output policy (the bake, the export): the frame renders
the integrator's full sample budget in one go. The live preview renders ONE
sample per read instead — a playhead move must not block on a path tracer's
whole budget — and accumulates the rest while the playhead stands still.
"""
function sceneframe!(src::SceneSource, clip::Clip, dims::Tuple{Int, Int};
                     exact::Bool = false, pixel_scale::Real = 1,detail_scale::Real = pixel_scale)
    # On the thread the renderer's resources belong to, wherever the composite
    # runs — see [`renderthread`](@ref). Everything from here down touches the
    # screen: building it, writing this frame's numbers onto its plots (which
    # walks Makie's compute graph and can reach the renderer), and reading the
    # film back.
    name = renderwith(src)
    backend = getbackend(name)
    img = onthread(renderthread(backend)) do; lock(SCENELOCK) do
        # The bake backend is a different renderer, so it is a different standing
        # scene: `livescene!` rebuilds when the backend changes, which is exactly
        # what switching modes is.
        src.live = livescene!(src.live, src.root, dims, name, backend;
                              opts = renderopts(src))
        quality_changed = preparepreview!(src.live.target, exact ? 1.0 : detail_scale)
        changed = updatesceneprogram!(src.live.target, src, clip, src.at, src.framerate;
                                      reset = src.samples == 0 || exact || src.mode === :bake || quality_changed)
        changed && syncsceneinputs!(src, clip)
        # …and now there is a scene to write this frame's numbers onto.
        # Refinement adds samples to the same scene; applying grouped transforms
        # again would do extra work and accumulate floating-point roundoff.
        (changed || !(src.live.target isa ProgramInstance)) && applysceneparams!(src, clip, src.at)
        # The first read at a position clears the film; every read after it adds
        # to it. Without the clear a path tracer would keep averaging the previous
        # frame's picture into this one and the animation would smear.
        liveframe!(src.live; clear = src.samples == 0,
                   samples = exact ? nothing : 1, pixel_scale = exact ? 1 : pixel_scale)
    end; end
    src.samples += 1
    return img
end

"""
    renderwith(src) -> Symbol

Which renderer draws this frame: the scene's own while previewing, the source's
`bakewith` while baking. `:auto` means there is no second one.
"""
renderwith(src::SceneSource) =
    src.mode === :bake && src.bakewith !== :auto ? src.bakewith : src.backend

"""
    renderopts(src) -> Dict

Which settings this frame is drawn with: the preview's own while live, and while
baking the preview's overridden by whatever the bake dialog set. The merge is
what the bake dialog shows — its form is seeded from the preview's values, and a
field it never touched keeps following the preview.

A NEW dict either way: the dialogs edit the source's dicts in place, and handing
`livescene!` the very dict the standing screen was built with would make every
edit compare equal to itself there — a changed setting that never rebuilds the
screen.
"""
renderopts(src::SceneSource) =
    src.mode === :bake ? merge(src.screenopts, src.bakescreenopts) : copy(src.screenopts)

"""
    refining(src) -> Bool

Whether this source has more samples to add at the frame it is showing.

This is what makes a raytraced preview usable: a seek returns one sample in
subseconds and standing still adds to it, instead of every playhead move costing
the full budget. It only works because the screen is held across frames — see
`SceneSource`.

No sample bound, which is the whole point: a path tracer's picture keeps getting
better and a number that stops it is a number that has to be right for every
scene. It stops when the playhead moves or playback starts — which is the only
thing that makes the picture wrong — and how much a FINISHED frame is worth is
the bake's sample count, not the preview's.
"""
refining(src::SceneSource) =
    src.live !== nothing && progressive(src.live.screen)
refining(::ClipSource) = false

"""
    updatesource!(source, clip, sf) -> nothing

Write frame `sf` into whatever the source needs to know about it.

Nothing for a video: a decoder is told which frame by the decode itself. For a
scene it is where this frame's animated numbers land — straight into the standing
spec, at the paths they name.

THIS IS WHAT REPLACED `animatedspec`. That built a full COPY of the scene per
frame — every part, every light, the camera — so that the sampled values had
somewhere to live that was not the stored spec. The values have somewhere now: the
parameters own them, and the spec is where they are written to be rendered.
"""
updatesource!(::ClipSource, ::Clip, ::Integer) = nothing

function updatesource!(src::SceneSource, ::Clip, sf::Integer)
    # A progressive renderer refines while the position holds and starts over when
    # it moves. Moving is what resets the count, so the first read at a new frame
    # clears the film and every read after it adds to it.
    if src.at != Int(sf)
        src.at = Int(sf)
        src.samples = 0
    end
    return nothing
end

"""
    applysceneparams!(src, clip, sf) -> nothing

Write this frame's numbers onto the live scene.

Here rather than in `updatesource!`, because that runs before the plan does and
the scene is built inside the source pass — there is nothing to write onto yet.
Values go straight onto the plots' attributes; a path that addresses nothing is
skipped, which is what makes a project from a newer editor open rather than throw.
"""
function applysceneparams!(src::SceneSource, clip::Clip, sf::Integer)
    src.live === nothing && return nothing
    fx = findslot(clip, :scene)
    if fx === nothing || !fx.enabled[]
        # A bypassed scene effect shows the recipe's own arguments.
        src.live.target isa ProgramInstance && applyargs!(src.live.target, nothing, sf)
        return nothing
    end
    applyscenecamera!(src)          # the eye first: a path may then move it
    for p in fx.params
        (isfollowing(p) || followsrecording(src,p) || isargpath(p.name)) && continue
        setscenevalue!(src, p.name, valueat(p, sf)) && remembersceneoverride!(src, p.name)
    end
    applyjoints!(src, fx, sf)       # …and the rig, once every number is known
    # Arguments last and whole: one write per argument, after every number of it
    # is known, so a plot derived from it recomputes once.
    src.live.target isa ProgramInstance && applyargs!(src.live.target, fx, sf)
    return nothing
end

isargpath(path::Symbol) = startswith(String(path), "args.")

"Read an editable scalar without changing the scene or advancing its animation."
function scenevalue(src::SceneSource, path::Symbol)
    a = scenepath(path)
    a === nothing && return nothing
    name, key, comp = a
    if name === :camera && src.camera !== nothing && haskey(CAMERAFIELDS, key)
        v = getfield(src.camera, key)
    elseif name === :args
        target = src.live === nothing ? nothing : src.live.target
        return target isa ProgramInstance ? argnumber(target, key, comp) : nothing
    else
        src.live === nothing && return nothing
        group = programcontrol(src.live.target, name)
        if group !== nothing
            haskey(group.values, key) || return nothing
            v = group.values[key]
            comp === nothing || (v = v isa Colorant ?
                (red(v), green(v), blue(v), alpha(v))[comp] : v[comp])
            return v isa Real ? Float64(v) : nothing
        end
        light = scenelight(targetscene(src.live.target), name)
        plot = light === nothing ? Makie.findplot(targetscene(src.live.target), name) : nothing
        if light !== nothing
            hasfield(typeof(light), key) || return nothing
            v = Makie.to_value(getfield(light, key))
        elseif plot === nothing
            return nothing
        elseif key in (:translation, :scale)
            v = getproperty(plot.transformation, key)[]
        elseif key === :rotation
            v = rotationdegrees(plot.transformation.rotation[])
        elseif haskey(plot.attributes.inputs, key)
            v = Makie.to_value(plot.attributes[key])
        else
            return nothing
        end
    end
    v isa AbstractString && return comp === nothing ? String(v) : nothing
    if comp !== nothing
        v = v isa Colorant ? (red(v), green(v), blue(v), alpha(v))[comp] : v[comp]
    end
    return v isa Real ? Float64(v) : nothing
end

"Sample source inputs after the original animation and before editor overrides."
function syncsceneinputs!(src::SceneSource, clip::Clip)
    fx = findslot(clip, :scene)
    fx === nothing && return nothing
    for p in fx.params
        isfollowing(p) || continue
        n = p.input
        v = scenevalue(src, only(n.inputs).path)
        v === nothing && continue
        isempty(n.resolved) || !(n.resolved[1] isa SceneValue) || (n.resolved[1].value = v)
        # The curve remains a fallback when the scene has not been realized yet.
        # Do not notify widgets from the render thread; publication does that.
        # A selected recipe property can carry a sampled original curve for the
        # timeline. Its input still follows the recipe until an actual edit.
        length(p.curve[].keys) == 1 && movekey!(p.curve[], 1, 0, v)
    end
    return nothing
end

# ------------------------------------------------- what a scene offers to animate
#
# From the live scene, not from a description of it beside the spec.
#
# The scene is the truth: it is what the renderer draws, so it is what the panel
# should list. A parallel model — our own parts, our own lights, our own camera —
# had to be kept in step with what was actually built, and could only ever offer
# what it had learned to describe. Walking the real plots offers whatever is
# there, including things nobody wrote a description for.
#
# The scene is built once per clip and its structure does not change (see
# `visible` — that is the only structural control a clip has), so a plot's
# identity is stable for the clip's whole life. That is why the cards can hang off
# the plots and do not need GLMakie's render-object list to survive anything.

"""
    sceneplots(scene) -> Vector{Pair{Symbol, Plot}}

Every named plot in the scene, depth first.

Named only, and that is the contract: a path addresses a plot by its name
(`"arm_left.rotation"`), and a name is what makes a keyframe survive the scene
being rebuilt. An unnamed plot is not addressable, so it is not animatable — say
`name = :something` in the scene code and it becomes so.
"""
function sceneplots(scene)
    out = Pair{Symbol, Any}[]
    walk(p) = begin
        n = Makie.to_value(get(p.attributes, :name, nothing))
        # Recipe internals can retain the unresolved default. It is not an
        # addressable object name and would alias unrelated child plots.
        n isa Symbol && n !== :automatic && push!(out, n => p)
        foreach(walk, p.plots)
    end
    walkscene(sc) = (foreach(walk, sc.plots); foreach(walkscene, sc.children))
    walkscene(scene)
    return out
end

"""
    rowkind(value) -> Symbol

What kind of row an attribute of this value gets: `:number` (a slider and a ◆),
`:vector`/`:colour` (one of those per component), `:data` (no slider — a mesh or a
picture is not a number, it comes down an input), `:none` (not offered).

Dispatch on the value, not a list of attribute names we maintain. A plot attribute
we have never heard of gets the right row because of what it IS.
"""
# The rig's own bookkeeping is not something to put a slider on: `jointaxis` and
# `jointbase` describe how a joint moves, not where it is now.
rowkind(::Real) = :number
rowkind(::Bool) = :none                     # `visible` is a toggle, not a slider
rowkind(::AbstractString) = :text           # a label's words: a text field, keyed as held steps
rowkind(::Colorant) = :colour
rowkind(::GeometryBasics.Vec) = :vector
rowkind(::GeometryBasics.Point) = :vector
rowkind(::AbstractMatrix{<:Colorant}) = :data
rowkind(::AbstractArray{<:Number}) = :data
rowkind(::AbstractArray{<:Colorant}) = :data
rowkind(::GeometryBasics.Mesh) = :data
rowkind(::AbstractVector{<:GeometryBasics.Point}) = :data
rowkind(::Any) = :none

# Makie currently also registers the transformation's derived matrix/function as
# graph inputs. Edit the transformation itself, never its computed result.
# `:cycled` is an unset input, not an explicit colour. Restoring its computed
# palette colour would override a material that never requested that colour.
sceneinputkeys(plot) = filter(k -> k ∉ (:name,:transformation,:dim_conversions,:cycle,
                                      :model,:transform_func) &&
                                  plot.attributes.inputs[k].value !== :cycled,
                             collect(keys(plot.attributes.inputs)))

"How many numbers a row kind has, and what they are called."
rowcomponents(::Val{:number}, v) = ((Symbol(""), Float64(v)),)
rowcomponents(::Val{:vector}, v) =
    ntuple(i -> (Symbol("[", i, "]"), Float64(v[i])), length(v))
rowcomponents(::Val{:colour}, v) =
    ((Symbol("[1]"), Float64(red(v))), (Symbol("[2]"), Float64(green(v))),
     (Symbol("[3]"), Float64(blue(v))), (Symbol("[4]"), Float64(alpha(v))))
# An opaque colour has no alpha to key.
rowcomponents(::Val{:colour}, v::Color) =
    ((Symbol("[1]"), Float64(red(v))), (Symbol("[2]"), Float64(green(v))),
     (Symbol("[3]"), Float64(blue(v))))

"""
    rowlabel(name, suffix, value) -> String

A row's label: the field's name and, for one component of a value, which one
in the value's own terms: R G B A of a colour, X Y Z W of a vector. A plain
number is its name alone.
"""
rowlabel(name::AbstractString, suffix::Symbol, v) =
    suffix === Symbol("") ? String(name) :
    string(name, " ", componentname(v, parse(Int, strip(String(suffix), ['[', ']']))))
componentname(::Colorant, i::Integer) = ("R", "G", "B", "A")[i]
componentname(::Any, i::Integer) = ("X", "Y", "Z", "W")[i]

"""
    sceneattributes(src) -> Vector{NamedTuple}

One entry per named plot in the live scene, with the paths its attributes offer.

`nothing` scene means nothing to offer, which is why the card is lazy: it
asks when it is opened, by which time the clip has rendered at least once and the
scene exists.
"""
function sceneattributes(src::SceneSource)
    src.live === nothing && return NamedTuple[]
    out = NamedTuple[]
    # The camera first, then the plots — reading order for a scene: where you look
    # from, then what is there.
    if src.camera !== nothing
        rows = vcat([comprows(:camera, f, n, getfield(src.camera, f))
                     for (f, n) in pairs(CAMERAFIELDS) if n == 3]...)
        push!(rows, (path = Symbol("camera.fov"), label = "Field of view °", kind = :number,
                     value = Float64(src.camera.fov)))
        push!(out, (name = :camera, label = "Camera", detail = "eye · look-at · up · fov",
                    rows = rows))
    end
    target = src.live.target
    if target isa ProgramInstance && !isempty(target.args)
        push!(out, (name = :args, label = "Arguments", detail = "Scene inputs",
                    rows = argrows(target)))
    end
    objects = sceneobjects(src)
    for object in objects
        name, plot = object.name, object.plot
        rows = NamedTuple[]
        # Transformations are not plot attributes, but they are the primary
        # editable numbers of a procedural actor or prop.
        append!(rows, comprows(name, :translation, 3, plot.transformation.translation[]))
        append!(rows, comprows(name, :rotation, 3, rotationdegrees(plot.transformation.rotation[])))
        append!(rows, comprows(name, :scale, 3, plot.transformation.scale[]))
        # A joint's rows come from the description, not from the plot: `angle` and
        # `offset` are not attributes Makie knows, they are how the part is hung.
        # Their values live in the parameters, so a fresh row starts at rest.
        if haskey(src.joints, name)
            push!(rows, (path = Symbol(name, ".angle"), label = "Angle",
                         kind = :number, value = 0.0))
            append!(rows, comprows(name, :offset, 3, Vec3f(0)))
        end
        # The inputs, not everything the graph holds. A plot's attributes are a
        # `ComputeGraph` and most of what is in it is derived — `eyeposition`,
        # `view_direction`, `N_lights` are answers Makie computed from the scene,
        # and a slider on an answer is a slider that will be overwritten. The
        # graph already distinguishes the two; asking it is one rule rather than a
        # list of exceptions to keep up to date.
        for key in sort!(sceneinputkeys(plot))
            object.attributes === nothing || String(key) in object.attributes || continue
            v = Makie.to_value(plot.attributes[key])
            if key === :visible && v isa Bool
                # Shown/hidden is keyed as a held step, never blended.
                push!(rows, (path = Symbol(name, ".visible"), label = "Visible", kind = :number,
                             value = Float64(v), discrete = true))
                continue
            end
            kind = rowkind(v)
            kind === :none && continue
            if kind === :text
                push!(rows, (path = Symbol(name, ".", key), label = titlecase(String(key)),
                             kind = kind, value = String(v)))
                continue
            end
            if kind === :data
                push!(rows, (path = Symbol(name, ".", key), label = titlecase(String(key)),
                             kind = kind, value = v))
                continue
            end
            for (suffix, num) in rowcomponents(Val(kind), v)
                push!(rows, (path = Symbol(name, ".", key, suffix),
                             label = rowlabel(titlecase(String(key)), suffix, v),
                             kind = :number, value = num))
            end
        end
        isempty(rows) ||
            push!(out, (name = name, label = object.label,
                        detail = string(nameof(typeof(plot))), rows = rows))
    end
    if src.live.target isa ProgramInstance
        for group in src.live.target.controls
            motion = (:animation_time,:time,:local_position,:position,:translation,
                      :yaw_degrees,:lean_degrees,:roll_degrees,:rotation,:scale,:grow,:squash)
            order = key -> (something(findfirst(==(key),motion),length(motion)+1),String(key))
            fields = sort!(collect(keys(group.values)); by = order)
            assigned = Set{Symbol}()
            for section in group.sections, field in section.fields
                field in assigned && error("duplicate inspector field $(group.name).$field")
                push!(assigned, field)
            end
            sections = [(label = group.label, fields = filter(k -> k ∉ assigned, fields)); group.sections]
            for section in sections
                rows = NamedTuple[]
                for key in section.fields
                    haskey(group.values, key) || continue
                    v = group.values[key]
                    for (suffix, num) in rowcomponents(Val(rowkind(v)), v)
                        push!(rows, (path = Symbol(group.name, ".", key, suffix),
                            label = rowlabel(titlecase(replace(String(key), '_' => ' ')), suffix, v),
                            kind = :number, value = num))
                    end
                end
                isempty(rows) || push!(out, (name = group.name, label = section.label,
                    detail = group.detail, rows = rows))
            end
        end
    end
    scene = targetscene(src.live.target)
    for (i, light) in [(0, Makie.AmbientLight(scene.compute[:ambient_color][]));
                       collect(enumerate(Makie.get_lights(scene)))]
        rows = NamedTuple[]
        name = Symbol("lights[$i]")
        for key in fieldnames(typeof(light))
            value = Makie.to_value(getfield(light, key))
            kind = rowkind(value)
            kind in (:number, :vector, :colour) || continue
            for (suffix, num) in rowcomponents(Val(kind), value)
                kind === :colour && suffix === Symbol("[4]") && continue
                push!(rows, (path = Symbol(name, ".", key, suffix),
                    label = titlecase(replace(String(key), "_" => " ")) * String(suffix),
                    kind = :number, value = num))
            end
        end
        title = replace(String(nameof(typeof(light))), "Light" => "")
        push!(out, (name, label = "Light · $title" * (i == 0 ? "" : " $i"),
                    detail = "Lighting", rows))
    end
    return out
end

"""
    argrows(target) -> Vector{NamedTuple}

Inspector rows for a recipe's arguments: one per keyframable number (see
[`argleaves`](@ref)), `discrete` for those that key as held steps, and one data
row for an argument without numbers (an array, a mesh).
"""
function argrows(target::ProgramInstance)
    rows = NamedTuple[]
    for (name, obs) in target.args
        v = obs[]
        leaves = argleaves(v)
        if isempty(leaves)
            push!(rows, (path = Symbol("args.", name), label = argtitle(name), kind = :data, value = v))
            continue
        end
        for (field, component, number, discrete) in leaves
            label = join(argtitle.(filter(!=(Symbol("")), [name; Symbol.(split(String(field), '.'))])), " · ")
            component === nothing ||
                (label *= " " * (getfieldpath(v, field) isa Colorant ?
                                 ("R", "G", "B", "A") : ("X", "Y", "Z", "W"))[component])
            push!(rows, (path = argpath(name, field, component), label = label, kind = :number,
                         value = number, discrete = discrete))
        end
    end
    return rows
end
argtitle(s) = titlecase(replace(String(s), '_' => ' '))
getfieldpath(v, field::Symbol) =
    field === Symbol("") ? v : foldl(getfield, Symbol.(split(String(field), '.')); init = v)

"""
    objectdescriptions(src) -> descriptions or `nothing`

The inspector groups of a scene: saved with its recipe, or else returned by its
builder (see `programscene`). `nothing` when there are none.
"""
function objectdescriptions(src::SceneSource)
    saved = src.build isa AbstractDict ? get(src.build, "objects", nothing) : nothing
    saved === nothing || return saved
    target = src.live === nothing ? nothing : src.live.target
    return target isa ProgramInstance && !isempty(target.objects) ? target.objects : nothing
end

"Authored object groups keep an actor's body, face and parts together in the inspector."
function sceneobjects(src::SceneSource)
    scene = targetscene(src.live.target)
    descriptions = objectdescriptions(src)
    if descriptions === nothing
        return [(name = name, plot = plot, label = String(name), attributes = nothing)
                for (name, plot) in sceneplots(scene)]
    end
    out = NamedTuple[]
    grouped = Set{Symbol}()
    for d in descriptions
        union!(grouped, Symbol.(d["plots"]))
        name = Symbol(first(d["plots"]))
        plot = Makie.findplot(scene, name)
        plot === nothing && continue
        push!(out, (name = name, plot = plot, label = String(d["label"]),
                    attributes = get(d, "attributes", nothing)))
    end
    # Groups consolidate known actors; they must not hide other named geometry.
    for (name,plot) in sceneplots(scene)
        name in grouped && continue
        push!(out,(;name,plot,label=String(name),attributes=nothing))
    end
    return out
end

function sceneobjectplots(src::SceneSource, name::Symbol)
    scene = targetscene(src.live.target)
    for d in something(objectdescriptions(src), ())
        Symbol(first(d["plots"])) === name || continue
        return filter(!isnothing, [Makie.findplot(scene, Symbol(n)) for n in d["plots"]])
    end
    plot = Makie.findplot(scene, name)
    return plot === nothing ? Makie.Plot[] : [plot]
end

"XYZ angles in degrees, composed as Z * Y * X. The stored plot rotation stays a quaternion."
function rotationdegrees(q)
    x, y, z, w = q.data
    return Vec3f(rad2deg(atan(2(w*x + y*z), 1 - 2(x*x + y*y))),
                 rad2deg(asin(clamp(2(w*y - z*x), -1, 1))),
                 rad2deg(atan(2(w*z + x*y), 1 - 2(y*y + z*z))))
end
rotationfromdegrees(v) = Makie.qrotation(Vec3f(0, 0, 1), deg2rad(v[3])) *
                         Makie.qrotation(Vec3f(0, 1, 0), deg2rad(v[2])) *
                         Makie.qrotation(Vec3f(1, 0, 0), deg2rad(v[1]))

"`n` numbered rows for one compound value — a joint's offset, the camera's eye."
comprows(name, field::Symbol, n::Integer, v) =
    NamedTuple[(path = Symbol(name, ".", field, "[", i, "]"),
                label = string(field === :eye ? "Position" : field === :lookat ? "Target" :
                               field === :translation ? "Position" :
                               field === :rotation ? "Rotation °" : titlecase(String(field)),
                               " ", ("X", "Y", "Z")[i]),
                kind = :number, value = Float64(v[i])) for i in 1:n]

# `scenepath` itself is in scenespec.jl and is unchanged: `"plot.attribute[i]"`
# parsed into its three parts. It used to address a `ScenePart`; it now addresses a
# plot in the live scene. Same syntax, same keyframes, different thing on the far
# end — which is the whole of what moved.


"""
One joint of a rig, as the project description states it: an axis to turn about and
the translation the part was built with.

Authored data, not runtime state. `angle` and `offset` are not here — those are
keyframed `Param`s and belong to the effect like every other animated number. What
is left is what the description says about how the part is hung, which is exactly
the input `rotate!(plot, axis, angle)` and `translate!(plot, base + offset)` need.

`base` is where a part starts: one with an origin sits at its pivot, one without
has to undo its parent's translation: its mesh is already in world coordinates and
would otherwise inherit it twice, which is what made the hands float beside the
figure. Computed once while the rig is built, because the parent chain carries
every animated offset by itself.

(Makie will grow real joints and skinning; this is the small thing that keeps a rig
working until it does.)
"""
struct RigJoint
    axis::Vec3f
    base::Vec3f
end

"What a joint offers to animate, and how many numbers each takes."
const JOINTFIELDS = (angle = 1, offset = 3)

"""
    applyjoints!(src, fx, sf) -> nothing

Turn and place every jointed part for frame `sf`.

One `rotate!`/`translate!` per part, from the parameters that hold its numbers and
the description that says how it is hung. The parent chain does the rest: turning
`arm_left` carries `hand_left` with it, so nothing here walks the tree.

Together rather than one number at a time, because an angle and an offset are one
placement of the part — writing them separately would apply half a pose.
"""
function applyjoints!(src::SceneSource, fx::Effect, sf::Integer)
    src.live === nothing && return nothing
    scene = targetscene(src.live.target)
    for (name, j) in src.joints
        plot = Makie.findplot(scene, name)
        plot === nothing && continue
        angle = jointnumber(fx, name, :angle, nothing, sf)
        off = Vec3f(ntuple(i -> jointnumber(fx, name, :offset, i, sf), 3))
        Makie.translate!(plot, j.base .+ off)
        Makie.rotate!(plot, j.axis, Float32(angle))
    end
    return nothing
end

"One of a joint's numbers at `sf`; zero when nothing keyframes it."
function jointnumber(fx::Effect, name::Symbol, field::Symbol, comp, sf::Integer)
    path = comp === nothing ? Symbol(name, ".", field) :
           Symbol(name, ".", field, "[", comp, "]")
    p = param(fx, path)
    return p === nothing ? 0.0 : Float64(valueat(p, sf))
end

"""
    setscenevalue!(src, path, v) -> Bool

Write one number into the live scene at `path`. `false` when it addresses nothing.

Three things a path can name, tried in order: a joint of the rig
(`"arm_left.angle"`, applied through the plot's transformation), the camera
(`"camera.eye[1]"` — placed after the scene exists, which is the only time
`cam3d!` can be told where to look), and any plot attribute (`"title.fontsize"` —
straight onto the attribute, because between frames a scene only ever changes
values and re-specifying is the slower path).
"""
function setscenevalue!(src::SceneSource, path, v)
    a = scenepath(path)
    a === nothing && return false
    name, key, comp = a
    # A joint's numbers are applied together, by `applyjoints!`, once every
    # parameter has been read — an angle and an offset are one placement.
    haskey(src.joints, name) && haskey(JOINTFIELDS, key) && return true
    if name === :camera && src.camera !== nothing
        haskey(CAMERAFIELDS, key) || return false
        old = getfield(src.camera, key)
        new = comp !== nothing ? withcomponent(old, comp, v) : old isa Real ? leafvalue(old, v) : Vec3f(v)
        src.camera = merge(src.camera, NamedTuple{(key,)}((new,)))
        applyscenecamera!(src)
        return true
    end
    src.live === nothing && return false
    if name === :args
        target = src.live.target
        target isa ProgramInstance || return false
        arg, fields = argaddress(key)
        obs = recipeargument(target, arg)
        obs === nothing && return false
        new = Base.invokelatest(withleaf, obs[], fields, comp, v)
        Base.invokelatest(isequal, obs[], new) || (obs[] = new)
        return true
    end
    scene = targetscene(src.live.target)
    light = scenelight(scene, name)
    if light !== nothing
        hasfield(typeof(light), key) || return false
        old = Makie.to_value(getfield(light, key))
        new = comp === nothing ? convert(typeof(old), v) : withcomponent(old, comp, v)
        setscenelight!(scene, name, key, new)
        return true
    end
    plot = Makie.findplot(targetscene(src.live.target), name)
    plot === nothing && return false
    if key in (:translation, :rotation, :scale)
        plots = sceneobjectplots(src, name)
        pivot = plot.transformation.translation[]
        if key === :rotation
            old = plot.transformation.rotation[]
            angles = rotationdegrees(old)
            new = rotationfromdegrees(comp === nothing ? Vec3f(v) : withcomponent(angles, comp, v))
            delta = new * inv(old)
            for part in plots
                Makie.translate!(part, pivot + delta * (part.transformation.translation[] - pivot))
                Makie.rotate!(part, delta * part.transformation.rotation[])
            end
            return true
        end
        old = getproperty(plot.transformation, key)[]
        new = comp === nothing ? Vec3f(v) : withcomponent(old, comp, v)
        if key === :translation
            delta = new - old
            for part in plots
                Makie.translate!(part, part.transformation.translation[] + delta)
            end
        else
            ratio = Vec3f(ntuple(i -> abs(old[i]) < 1f-8 ? 1f0 : new[i] / old[i], 3))
            for part in plots
                Makie.translate!(part, pivot + (part.transformation.translation[] - pivot) .* ratio)
                Makie.scale!(part, part === plot ? new : part.transformation.scale[] .* ratio)
            end
        end
        return true
    end
    haskey(plot.attributes.inputs, key) || return false
    old = Makie.to_value(plot.attributes[key])
    new = comp === nothing ? leafvalue(old, v) : withcomponent(old, comp, v)
    isequal(old, new) && return true
    # `update!`, not an assignment into the graph: a plot's attributes are a
    # `ComputeGraph`, and setting an input is what makes everything derived from it
    # recompute. Writing the dict entry would leave the derived values stale.
    Makie.update!(plot; NamedTuple{(key,)}((new,))...)
    return true
end

"Native Makie lights addressed by index; zero is Makie's separate ambient light."
function scenelight(scene, name::Symbol)
    m = match(r"^lights\[(\d+)\]$", String(name))
    m === nothing && return nothing
    i = parse(Int, m[1])
    i == 0 && return Makie.AmbientLight(scene.compute[:ambient_color][])
    lights = Makie.get_lights(scene)
    return i <= length(lights) ? lights[i] : nothing
end

function setscenelight!(scene, name::Symbol, key::Symbol, value)
    i = parse(Int, match(r"^lights\[(\d+)\]$", String(name))[1])
    if i == 0
        Makie.set_ambient_light!(scene, value)
    else
        Makie.set_light!(scene, i; NamedTuple{(key,)}((value,))...)
    end
    return nothing
end

"""
Where a 3-D scene looks from: the three vectors `cam3d!` is told after the fact,
and its vertical field of view in degrees. The number is how many components
each has.
"""
const CAMERAFIELDS = (eye = 3, lookat = 3, up = 3, fov = 1)

"The camera of a scene with a `Camera3D`, as the editor keys it."
function cameravalues(scene::Makie.Scene)
    c = Makie.cameracontrols(scene)
    return (eye = Vec3f(c.eyeposition[]), lookat = Vec3f(c.lookat[]),
            up = Vec3f(c.upvector[]), fov = Float32(c.fov[]))
end

"Place `v` (see [`cameravalues`](@ref)) on the scene's `Camera3D`."
function setcameravalues!(scene::Makie.Scene, v)
    c = Makie.cameracontrols(scene)
    c.fov[] = v.fov
    Makie.update_cam!(scene, c, v.eye, v.lookat, v.up)
    return scene
end

"The scene holding a source's 3D camera, if it has one."
camerascene(target::ProgramInstance) = target.camerascene
camerascene(target) = camerascene(targetscene(target))

"Place the source's camera on its live scene, if it has both."
function applyscenecamera!(src::SceneSource)
    (src.live === nothing || src.camera === nothing) && return src
    sc = camerascene(src.live.target)
    sc === nothing && return src
    setcameravalues!(sc, src.camera)
    return src
end

"One component of a compound attribute replaced — a `Vec`'s axis, a colour's channel."
withcomponent(old::GeometryBasics.Vec{N, T}, i::Integer, v) where {N, T} =
    GeometryBasics.Vec{N, T}(ntuple(k -> k == i ? T(v) : old[k], N))
withcomponent(old::GeometryBasics.Point{N, T}, i::Integer, v) where {N, T} =
    GeometryBasics.Point{N, T}(ntuple(k -> k == i ? T(v) : old[k], N))
function withcomponent(old::Colorant, i::Integer, v)
    c = RGBAf(old)
    return typeof(old)(RGBAf(i == 1 ? v : c.r, i == 2 ? v : c.g,
                             i == 3 ? v : c.b, i == 4 ? v : c.alpha))
end
withcomponent(old, ::Integer, v) = convert(typeof(old), v)

# ---------------------------------------------------------------- the source pass

"""
The source pass of a clip that renders its frames. No decode, no upload of a
decoded picture — the renderer hands back a host image in the plane's own format
and it goes straight in.
"""
struct SceneNode <: FxNode
    inputdims::Tuple{Int, Int}
end

"""
    sourcenode(clip) -> FxNode

Which source pass this clip's chain begins with. Dispatch on the source, because
"where does a frame come from" is exactly what a source is.
"""
sourcenode(clip::Clip) = sourcenode(clip.source, clip)
sourcenode(::VideoSource, clip::Clip) =
    clip.timeinterp === :flow ? SmoothSourceNode() : SourceNode()
sourcenode(src::SceneSource, ::Clip) = SceneNode((src.width, src.height))

function previewnode(clip::Clip, sf::Integer; exact::Bool = false)
    src = clip.source
    if !exact && src isa SceneSource && src.pending !== nothing && src.pendingat == sf &&
       !hasbakedframe(clip, sf)
        return SceneNode(size(src.pending))
    end
    return sourcenode(clip)
end

function chainpass!(g, node::SceneNode, ::Nothing, ::Nothing, ctx::ChainBuild, dims)
    inputdims = node.inputdims
    cur = Mantle.Transient.Buffer(g, PlanePixel, inputdims...; hostwritten = true)
    st = ctx.state
    # A scene's picture is a host frame and is STORED into the transient for
    # exactly the reason a decoded one is: a `copyto!` into a device transient
    # from inside a recorded pass body is a host→device upload mid-batch, which
    # forces a `vkQueueSubmit` and stalls. A store lands at the update pass
    # instead, as a command in the run's own submission. This was the video source's bug too —
    # it is the same bug, and it survived here because the scene pass has its own
    # body. Measured on the lego project: playback of a timeline with a scene ran
    # at 10.6 fps against 32.4 without one, and baking the scene (so the body only
    # reads a PNG) changed nothing, which is what puts the cost on the upload and
    # not the drawing.
    st.upload = cur
    st.uploaddims = inputdims
    st.blankonmiss = true   # nothing else writes this one
    # No pass at all. The picture is a host image and `update!` STORES it into
    # this transient before the run, which is the whole of what the old body did
    # — it only ever had work left when nobody had filled the update, and that
    # case is now a store of `blankpixels!` rather than a draw inside a recorded
    # pass. What reads `cur` is the next node in the chain, which is what gives
    # the transient its interval.
    inputdims == dims && return cur
    # Upscale once on the GPU, before effects. Their pixel units, the clip's
    # placement and editor hit coordinates keep their full-resolution meaning.
    out = Mantle.Transient.Buffer(g, PlanePixel, dims...)
    sx, sy = Float32(inputdims[1] / dims[1]), Float32(inputdims[2] / dims[2])
    matrix = Mat3f(sx, 0, 0, 0, sy, 0, (1 - sx) / 2, (1 - sy) / 2, 1)
    Mantle.dispatch!(g, GPUFiltering.warp_kernel!,
        (out, cur, matrix, false, Vec4{Int32}(1, 1, inputdims...), GPUFiltering.Replace()),
        dims; name = "scene/preview upscale")
    return out
end

"""
    scenepicture!(st, dims) -> Matrix{PlanePixel}

This frame of the scene as a host image: the bake if there is one, the frame
`prerender!` drew if there is one, and otherwise drawn now.

The one place that answers "what does this scene look like at this frame", so the
update below and the pass body above cannot disagree about it.
"""
function scenepicture!(st, dims::Tuple{Int, Int}; exact::Bool = false)
    img = takepending!(st.source, st.served[])
    img === nothing || return img
    # Settle the frame first. `sceneframe!` draws whatever `src.at` currently
    # says, and reaching here means the prerender did not match the frame being
    # composited — so `at` is some other frame, and drawing it would put the
    # scene one pose out of step with the layers under it.
    #
    # `updatesource!` also resets the sample count, so the film is cleared rather
    # than accumulating this pose on top of the last one.
    updatesource!(st.source, st.clip, st.served[])
    return sceneframe!(st.source, st.clip, dims; exact)
end

# …and that is what `sourcepicture!` answers for a scene clip. `update!` asks it
# once per frame, before the plan runs, so the picture goes in through the update
# instead of a `copyto!` inside a recorded pass.
sourcepicture!(st, dims::Tuple{Int, Int}, ::SceneSource; exact::Bool = false) =
    scenepicture!(st, dims; exact)

# ---------------------------------------------------------------- making one

"""
    sceneclip(spec; start, frames, canvas, framerate) -> Clip

A clip that draws `spec`, placed at timeline frame `start`.

The `:scene` effect it carries is where the spec lives and where its animatable
numbers come from — one parameter per number, exactly as a 3D scene's card shows
them. Nothing about this is special-cased downstream: it trims, it takes a blur,
it composites, it keys by source frame.
"""
function sceneclip(built; build = nothing, start::Integer = 0, frames::Integer = 90,
                   canvas::Tuple{Integer, Integer} = (1920, 1080),
                   framerate::Real = 30.0, backend::Symbol = :GLMakie)
    src = SceneSource(built.root; joints = built.joints, camera = built.camera,
                      build, backend, width = canvas[1], height = canvas[2],
                      framerate, nframes = frames)
    clip = Clip(src; src_in = 0, src_out = Int(frames), start = Int(start))
    # The `:scene` entry holds the clip's scene parameters — the curves — and
    # nothing else. It starts empty: what is animatable is discovered from the
    # scene when the card is opened, and a parameter is minted then or read from
    # the project file, whichever comes first.
    addslot!(clip, Effect(freshid(), :scene, true, Param[]))
    return clip
end

"""
    keyframes!(clip, path, keys; ease = :smooth, discrete = false, label) -> Vector{Param}

Key scene numbers from code, exactly as the inspector would: `path` names them
as the inspector does (`"camera.eye[1]"`, `"camera.fov"`, `"callout.alpha"`,
`"args.time"`, `"args.pose.eye[2]"`), `keys` are `frame => value` pairs in the
clip's source frames, or `(frame, value, ease)` triples. A `Vec`, `Point` or
colour value keys every component (`path[1]`, `path[2]`, …). Existing keys on
those paths are replaced.

`ease` is a key's kind (see [`Keyframe`](@ref)): `:smooth` eases in and out,
`:linear` is a constant rate, `:hold` steps. `discrete` makes the whole curve
step, for counts, switches and `visible`. The scene need not be built yet: the
keys wait on the clip's scene effect and are saved with the project.
"""
function keyframes!(clip::Clip, path, keys; ease::Symbol = :smooth, discrete::Bool = false,
                    label::AbstractString = String(path))
    fx = findslot(clip, :scene)
    fx === nothing && error("clip has no scene effect to key")
    entries = [k isa Pair ? (k.first, k.second, ease) : k for k in keys]
    isempty(entries) && error("no keys for $path")
    first(entries)[2] isa Union{GeometryBasics.Vec, GeometryBasics.Point, Colorant} ||
        return [keyscalar!(fx, Symbol(path), entries, discrete, label)]
    n = first(entries)[2] isa Colorant ? 4 : length(first(entries)[2])
    return [keyscalar!(fx, Symbol(path, "[", i, "]"),
                       [(f, v isa Colorant ? (red(v), green(v), blue(v), alpha(v))[i] : v[i], e)
                        for (f, v, e) in entries], discrete, "$label[$i]") for i in 1:n]
end

function keyscalar!(fx::Effect, path::Symbol, entries, discrete::Bool, label::AbstractString)
    ks = sort!([Keyframe{Float64}(Int(f), Float64(v), e) for (f, v, e) in entries]; by = k -> k.frame)
    allunique(k.frame for k in ks) || error("two keys on one frame for $path")
    values = [k.value for k in ks]
    span = occursin(".rotation[", String(path)) ? (-180.0, 180.0) :
           (min(0.0, 2minimum(values)), max(1.0, 2maximum(values)))
    fresh = Param(path, label, first(values); curve = AnimCurve{Float64}(ks, discrete ? :hold : :linear),
                  range = span)
    i = findfirst(q -> q.name === path, fx.params)
    i === nothing ? push!(fx.params, fresh) : (fx.params[i] = fresh)
    return fresh
end

# ---------------------------------------------------------------- the kind

# The `:scene` entry is data, not a pixel operation: it has no `make`, so
# `renderable` is false for it and the chain never asks it for a payload. It is
# registered all the same, because being a registered kind is what gives it a
# label in the panel, a card with its parameter sections, and a name a project
# file can write and read back (`anykind`).
registereffect!(EffectKind(:scene, "Scene";
    description = "What this clip draws: a Makie scene. Every number in it — a \
                   joint angle, a light, the camera, a font size — is a parameter \
                   here, so it keyframes like any other. How it is drawn — the \
                   renderer and its settings — is not one of those numbers: it \
                   belongs to the clip, and it lives in the rendering dialog on \
                   the clip's bake row."))

"""
    addcaptionclip!(seq; track, canvas) -> Clip

Put the transcript on the timeline as a clip spanning the whole sequence.

What the transcript IS stays on the sequence; this is where it is drawn. The text
is written per frame by [`captiontext!`](@ref) from the clip's own position.
"""
function addcaptionclip!(seq::Sequence; canvas = something(seq.canvas, (1920, 1080)))
    n = max(seqlength(seq), 1)
    build = Dict{String, Any}("kind" => "captions",
                              "args" => Dict{String, Any}("canvas" => [canvas[1], canvas[2]]))
    clip = sceneclip(buildscene(build); build, start = 0, frames = n, canvas,
                     framerate = seq.framerate)
    clip.track = ntracks(seq) + 1
    addclip!(seq, clip)
    return clip
end

# ---------------------------------------------------------------- putting one down

"""
    addsceneclip!(player, spec; label, seconds, track) -> Clip

Place a scene clip at the playhead and select it.

The one place a scene reaches the timeline: the title command, the bar, the
timecode, the captions and "add a 3D scene" all come through here, so they cannot
drift apart on where it lands, how long it is, or whether it is undoable.

On its own lane above whatever is there, because a graphic that replaced the shot
under it is not what anybody meant by adding a title.
"""
function addsceneclip!(player::Player, build::AbstractDict; label::AbstractString = "scene",
                       seconds::Real = 3.0, track::Union{Nothing, Integer} = nothing)
    seq = player.sequence
    canvas = canvassize(seq)
    fps = seq.framerate > 0 ? seq.framerate : 30.0
    frames = max(round(Int, seconds * fps), 1)
    at = player.playhead[]
    snapshot!(player)
    clip = sceneclip(buildscene(build); build, start = at, frames, canvas, framerate = fps)
    clip.track = track === nothing ? freetrack(seq, at, frames, ntracks(seq) + 1) : Int(track)
    addclip!(seq, clip)
    sort!(seq.clips; by = c -> c.start)
    bindinputs!(seq)
    redraw!(player)
    player.timeline.selected[] = clip.id
    fx = findslot(clip, :scene)
    fx === nothing || selectfxcard!(player, (:fx, fx.id))
    setstatus!(player, "$label added on track $(clip.track) — " *
                       "$(round(frames / fps, digits = 1))s, trim it like any clip")
    showplayhead!(player)
    return clip
end
