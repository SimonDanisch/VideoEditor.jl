# A Makie scene AS A CLIP SOURCE.
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

The scene a clip draws — its SOURCE's, because that is what a source is.

It used to be a `Param{SceneSpec}` on the clip's `:scene` effect, so that it would
be saved. But a spec is not a parameter: nothing keyframes it, no widget edits it,
and it contributes no rows — those come from the scene that gets built. It is what
the clip shows, which is the definition of its source.
"""
scenespecof(clip::Clip) = scenespecof(clip.source)
scenespecof(src::SceneSource) = src.root
scenespecof(::ClipSource) = nothing

"""
    prerender!(source, clip, frame) -> nothing

Have the source draw this frame BEFORE the graph runs.

Nothing for a decoder: `decodesource` is its version of the same idea, called
from `update!` for the reason spelled out there.

A SCENE draws with GLMakie, and its screen belongs to thread 1 — while the
composite runs on whichever thread owns the Lava context, the pinned GPU worker.
Drawing inside the pass body is therefore thread 1 → worker → thread 1 → worker,
once per frame, and the hop back has to wait for the editor's own renderloop to
reach a yield. Measured on the lego project: **22.5 ms of waiting against 7.3 ms
of drawing**, and playback of a 60 fps timeline at 10–12 fps.

Called where the caller is STILL on thread 1 — right before `runowned` hands the
frame to the worker — so `onmainthread` inside is a no-op and the wait is gone.
"""
prerender!(::ClipSource, ::Clip, ::Integer) = nothing

function prerender!(src::SceneSource, clip::Clip, sf::Integer)
    updatesource!(src, clip, sf)          # settles `at`, and resets the sample count
    src.pending = sceneframe!(src, clip, (src.width, src.height))
    src.pendingat = Int(sf)
    return nothing
end

"""
    prerenderscenes!(clips, n) -> nothing

[`prerender!`](@ref) every clip of a frame, by TIMELINE frame.

The three places that hand a frame to the render engine call this immediately
before they do — `compositeframe!`, `presentclipframe!` and `presentgpu!`. Not
inside `composite`: by then the thread has already changed, which is the whole
problem this avoids.
"""
function prerenderscenes!(clips, n::Integer)
    for clip in clips
        prerender!(clip.source, clip, sourceframe(clip, n))
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
    sceneframe!(src, spec, dims) -> Matrix{RGBA{N0f8}}

One frame of the scene, at `dims`.

The standing scene is reused unless the canvas or the backend changed; only the
VALUES this frame carries are written onto it (see [`liveframe!`](@ref)). The
result is already the plane format — RGBA with the scene's own coverage, so what
it did not draw on is uncovered and the clip below shows through.
"""
function sceneframe!(src::SceneSource, clip::Clip, dims::Tuple{Int, Int})
    # ON THREAD 1, wherever the composite runs — see [`onmainthread`](@ref).
    # Everything from here down touches the screen: building it, writing this
    # frame's numbers onto its plots (which walks Makie's compute graph and can
    # reach the renderer), and reading the film back.
    img = onmainthread() do
        name = renderwith(src)
        # The BAKE backend is a different renderer, so it is a different standing
        # scene: `livescene!` rebuilds when the backend changes, which is exactly
        # what switching modes is.
        src.live = livescene!(src.live, src.root, dims, name, getbackend(name);
                              theme = rendertheme(src))
        # …and NOW there is a scene to write this frame's numbers onto.
        applysceneparams!(src, clip, src.at)
        # The first read at a position CLEARS the film; every read after it adds
        # to it. Without the clear a path tracer would keep averaging the previous
        # frame's picture into this one and the animation would smear.
        liveframe!(src.live; clear = src.samples == 0)
    end
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
    rendertheme(src) -> Dict

Which SETTINGS this frame is drawn with: the live ones while previewing, the
bake's while baking. An empty bake theme means there is nothing to change — the
scene draws the same either way, which is the common case for a rasteriser.
"""
rendertheme(src::SceneSource) =
    src.mode === :bake && !isempty(src.baketheme) ? src.baketheme : src.theme

"""
How many accumulated samples a live preview stops at.

A bound, not a quality setting: past it the picture stops changing visibly and the
GPU is better left to the next seek. A BAKE has no such bound — it renders what
its integrator is configured for.
"""
const MAXSAMPLES = 64

"""
    refining(src) -> Bool

Whether this source has more samples to add at the frame it is showing.

This is what makes a raytraced preview usable: a seek returns ONE sample in
subseconds and standing still adds to it, instead of every playhead move costing
the full budget. It only works because the screen is held across frames — see
`SceneSource`.
"""
refining(src::SceneSource) =
    src.live !== nothing && progressive(src.live.screen) && src.samples < MAXSAMPLES
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

Write this frame's numbers onto the LIVE scene.

Here rather than in `updatesource!`, because that runs before the plan does and
the scene is built inside the source pass — there is nothing to write onto yet.
Values go straight onto the plots' attributes; a path that addresses nothing is
skipped, which is what makes a project from a newer editor open rather than throw.
"""
function applysceneparams!(src::SceneSource, clip::Clip, sf::Integer)
    src.live === nothing && return nothing
    fx = findslot(clip, :scene)
    fx === nothing && return nothing
    applyscenecamera!(src)          # the eye first: a path may then move it
    for p in fx.params
        setscenevalue!(src, p.name, valueat(p, sf))
    end
    applyjoints!(src, fx, sf)       # …and the rig, once every number is known
    return nothing
end

# ------------------------------------------------- what a scene offers to animate
#
# FROM THE LIVE SCENE, not from a description of it beside the spec.
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

Every NAMED plot in the scene, depth first.

Named only, and that is the contract: a path addresses a plot by its name
(`"arm_left.rotation"`), and a name is what makes a keyframe survive the scene
being rebuilt. An unnamed plot is not addressable, so it is not animatable — say
`name = :something` in the scene code and it becomes so.
"""
function sceneplots(scene)
    out = Pair{Symbol, Any}[]
    walk(p) = begin
        n = Makie.to_value(get(p.attributes, :name, nothing))
        n isa Symbol && push!(out, n => p)
        foreach(walk, p.plots)
    end
    walkscene(sc) = (foreach(walk, sc.plots); foreach(walkscene, sc.children))
    walkscene(scene)
    return out
end

"""
    rowkind(value) -> Symbol

What KIND of row an attribute of this value gets: `:number` (a slider and a ◆),
`:vector`/`:colour` (one of those per component), `:data` (no slider — a mesh or a
picture is not a number, it comes down an input), `:none` (not offered).

Dispatch on the value, not a list of attribute names we maintain. A plot attribute
we have never heard of gets the right row because of what it IS.
"""
# The rig's own bookkeeping is not something to put a slider on: `jointaxis` and
# `jointbase` describe HOW a joint moves, not where it is now.
rowkind(::Real) = :number
rowkind(::Bool) = :none                     # `visible` is a toggle, not a slider
rowkind(::Colorant) = :colour
rowkind(::GeometryBasics.Vec) = :vector
rowkind(::GeometryBasics.Point) = :vector
rowkind(::AbstractMatrix{<:Colorant}) = :data
rowkind(::GeometryBasics.Mesh) = :data
rowkind(::AbstractVector{<:GeometryBasics.Point}) = :data
rowkind(::Any) = :none

"How many numbers a row kind has, and what they are called."
rowcomponents(::Val{:number}, v) = ((Symbol(""), Float64(v)),)
rowcomponents(::Val{:vector}, v) =
    ntuple(i -> (Symbol("[", i, "]"), Float64(v[i])), length(v))
rowcomponents(::Val{:colour}, v) =
    ((Symbol("[1]"), Float64(red(v))), (Symbol("[2]"), Float64(green(v))),
     (Symbol("[3]"), Float64(blue(v))), (Symbol("[4]"), Float64(alpha(v))))

"""
    sceneattributes(src) -> Vector{NamedTuple}

One entry per named plot in the live scene, with the paths its attributes offer.

`nothing` scene → nothing to offer. The card is LAZY for exactly this reason: it
asks when it is opened, by which time the clip has rendered at least once and the
scene exists.
"""
function sceneattributes(src::SceneSource)
    src.live === nothing && return NamedTuple[]
    out = NamedTuple[]
    # The camera first, then the plots — reading order for a scene: where you look
    # from, then what is there.
    if src.camera !== nothing
        push!(out, (name = :camera, label = "Camera", detail = "eye · look-at · up",
                    rows = vcat([comprows(:camera, f, n, getfield(src.camera, f))
                                 for (f, n) in pairs(CAMERAFIELDS)]...)))
    end
    for (name, plot) in sceneplots(targetscene(src.live.target))
        rows = NamedTuple[]
        # A JOINT's rows come from the description, not from the plot: `angle` and
        # `offset` are not attributes Makie knows, they are how the part is hung.
        # Their values live in the parameters, so a fresh row starts at rest.
        if haskey(src.joints, name)
            push!(rows, (path = Symbol(name, ".angle"), label = "Angle",
                         kind = :number, value = 0.0))
            append!(rows, comprows(name, :offset, 3, Vec3f(0)))
        end
        # THE INPUTS, not everything the graph holds. A plot's attributes are a
        # `ComputeGraph`, and most of what is in it is DERIVED — `eyeposition`,
        # `view_direction`, `N_lights` are answers Makie computed from the scene,
        # and a slider on an answer is a slider that will be overwritten. The
        # graph already distinguishes the two; asking it is one rule rather than a
        # list of exceptions to keep up to date.
        for key in sort!(collect(keys(plot.attributes.inputs)))
            key in (:name, :transformation, :arg1, :dim_conversions, :cycle) && continue
            v = Makie.to_value(plot.attributes[key])
            kind = rowkind(v)
            kind === :none && continue
            if kind === :data
                push!(rows, (path = Symbol(name, ".", key), label = titlecase(String(key)),
                             kind = kind, value = v))
                continue
            end
            for (suffix, num) in rowcomponents(Val(kind), v)
                push!(rows, (path = Symbol(name, ".", key, suffix),
                             label = string(titlecase(String(key)), " ", suffix),
                             kind = :number, value = num))
            end
        end
        isempty(rows) ||
            push!(out, (name = name, label = String(name),
                        detail = string(nameof(typeof(plot))), rows = rows))
    end
    return out
end

"`n` numbered rows for one compound value — a joint's offset, the camera's eye."
comprows(name, field::Symbol, n::Integer, v) =
    NamedTuple[(path = Symbol(name, ".", field, "[", i, "]"),
                label = string(titlecase(String(field)), " ", ("X", "Y", "Z")[i]),
                kind = :number, value = Float64(v[i])) for i in 1:n]

# `scenepath` itself is in scenespec.jl and is unchanged: `"plot.attribute[i]"`
# parsed into its three parts. It used to address a `ScenePart`; it now addresses a
# plot in the live scene. Same syntax, same keyframes, different thing on the far
# end — which is the whole of what moved.


"""
One joint of a rig, as the PROJECT DESCRIPTION states it: an axis to turn about and
the translation the part was built with.

Authored data, not runtime state. `angle` and `offset` are not here — those are
keyframed `Param`s and belong to the effect like every other animated number. What
is left is what the description says about how the part is hung, which is exactly
the input `rotate!(plot, axis, angle)` and `translate!(plot, base + offset)` need.

`base` is where a part starts: one with an origin sits at its pivot, one without
has to UNDO its parent's translation — its mesh is already in world coordinates and
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

Together rather than one number at a time, because an angle and an offset are ONE
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

Three things a path can name, tried in order and all of them real: a JOINT of the
rig (`"arm_left.angle"` — applied through the plot's transformation), the CAMERA
(`"camera.eye[1]"` — placed after the scene exists, which is the only time
`cam3d!` can be told where to look), and any PLOT ATTRIBUTE (`"title.fontsize"` —
straight onto the attribute, because between frames a scene only ever changes
values and re-specifying is the slower path).
"""
function setscenevalue!(src::SceneSource, path, v)
    a = scenepath(path)
    a === nothing && return false
    name, key, comp = a
    # A joint's numbers are applied TOGETHER, by `applyjoints!`, once every
    # parameter has been read — an angle and an offset are one placement.
    haskey(src.joints, name) && haskey(JOINTFIELDS, key) && return true
    if name === :camera && src.camera !== nothing
        haskey(CAMERAFIELDS, key) || return false
        old = getfield(src.camera, key)
        new = comp === nothing ? Vec3f(v) : withcomponent(old, comp, v)
        src.camera = merge(src.camera, NamedTuple{(key,)}((new,)))
        applyscenecamera!(src)
        return true
    end
    src.live === nothing && return false
    plot = Makie.findplot(targetscene(src.live.target), name)
    plot === nothing && return false
    haskey(plot.attributes.inputs, key) || return false
    old = Makie.to_value(plot.attributes[key])
    new = comp === nothing ? convert(typeof(old), v) : withcomponent(old, comp, v)
    # `update!`, not an assignment into the graph: a plot's attributes are a
    # `ComputeGraph`, and setting an input is what makes everything derived from it
    # recompute. Writing the dict entry would leave the derived values stale.
    Makie.update!(plot; NamedTuple{(key,)}((new,))...)
    return true
end

"Where a 3-D scene looks from — the three vectors `cam3d!` is told after the fact."
const CAMERAFIELDS = (eye = 3, lookat = 3, up = 3)

"Place the source's camera on its live scene, if it has both."
function applyscenecamera!(src::SceneSource)
    (src.live === nothing || src.camera === nothing) && return src
    sc = targetscene(src.live.target)
    c = Makie.cameracontrols(sc)
    c isa Makie.Camera3D || return src
    Makie.update_cam!(sc, c, src.camera.eye, src.camera.lookat, src.camera.up)
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
The source pass of a clip that RENDERS its frames. No decode, no upload of a
decoded picture — the renderer hands back a host image in the plane's own format
and it goes straight in.
"""
struct SceneNode <: FxNode end

"""
    sourcenode(clip) -> FxNode

Which source pass this clip's chain begins with. Dispatch on the source, because
"where does a frame come from" is exactly what a source is.
"""
sourcenode(clip::Clip) = sourcenode(clip.source, clip)
sourcenode(::VideoSource, clip::Clip) =
    clip.timeinterp === :flow ? SmoothSourceNode() : SourceNode()
sourcenode(::SceneSource, ::Clip) = SceneNode()

function chainpass!(g, ::SceneNode, ::Nothing, ::Nothing, ctx::ChainBuild, dims)
    cur = Mantle.Transient.Buffer(g, PlanePixel, prod(dims))
    st = ctx.state
    # A SCENE'S PICTURE IS A HOST FRAME, and it comes in through an `Update` for
    # exactly the reason a decoded one does: a `copyto!` into a device transient
    # from inside a recorded pass body is a host→device upload mid-batch, which
    # forces a `vkQueueSubmit` and stalls. This was the video source's bug too —
    # it is the same bug, and it survived here because the scene pass has its own
    # body. Measured on the lego project: playback of a timeline WITH a scene ran
    # at 10.6 fps against 32.4 without one, and BAKING the scene (so the body only
    # reads a PNG) changed nothing — which is what says the cost is the upload and
    # not the drawing.
    st.upload = Mantle.Update(g, cur)
    Mantle.custom!(g, "scene") do p
        Mantle.use(p, cur; read = true, write = true)
        # …and the body only has work left when nobody filled the update: the
        # export and the bake are single-threaded and draw right here.
        () -> st.uploaded || copyto!(frameview(cur, dims), scenepicture!(st, dims))
    end
    return cur
end

"""
    scenepicture!(st, dims) -> Matrix{PlanePixel}

This frame of the scene as a host image: the bake if there is one, the frame
`prerender!` drew if there is one, and otherwise drawn now.

The one place that answers "what does this scene look like at this frame", so the
update below and the pass body above cannot disagree about it.
"""
function scenepicture!(st, dims::Tuple{Int, Int})
    img = takepending!(st.source, st.served[])
    return img === nothing ? sceneframe!(st.source, st.clip, dims) : img
end

# …and that is what `sourcepicture!` answers for a scene clip. `update!` asks it
# once per frame, before the plan runs, so the picture goes in through the update
# instead of a `copyto!` inside a recorded pass.
sourcepicture!(st, dims::Tuple{Int, Int}, ::SceneSource) = scenepicture!(st, dims)

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
    # The `:scene` entry holds the clip's scene PARAMETERS — the curves — and
    # nothing else. It starts empty: what is animatable is discovered from the
    # scene when the card is opened, and a parameter is minted then or read from
    # the project file, whichever comes first.
    addslot!(clip, Effect(freshid(), :scene, true, Param[]))
    return clip
end

# ---------------------------------------------------------------- the kind

# The `:scene` entry is DATA, not a pixel operation: it has no `make`, so
# `renderable` is false for it and the chain never asks it for a payload. It is
# registered all the same, because being a registered kind is what gives it a
# label in the panel, a card with its parameter sections, and a name a project
# file can write and read back (`anykind`).
registereffect!(EffectKind(:scene, "Scene";
    description = "What this clip draws: a Makie scene. Every number in it — a \
                   joint angle, a light, the camera, a font size — is a parameter \
                   here, so it keyframes like any other."))

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
    push!(seq.clips, clip)
    return clip
end

# ---------------------------------------------------------------- putting one down

"""
    addsceneclip!(player, spec; label, seconds, track) -> Clip

Place a scene clip at the playhead and select it.

THE ONE PLACE a scene reaches the timeline — the title command, the bar, the
timecode, the captions and "add a 3D scene" all come through here, so they cannot
drift apart on where it lands, how long it is, or whether it is undoable.

On its OWN lane above whatever is there, because a graphic that replaced the shot
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
    push!(seq.clips, clip)
    sort!(seq.clips; by = c -> c.start)
    bindinputs!(seq)
    refreshedit!(player)
    i = findfirst(c -> c === clip, seq.clips)
    i === nothing || (player.timeline.selected[] = i)
    fx = findslot(clip, :scene)
    fx === nothing || selectfxcard!(player, (:fx, fx.id))
    setstatus!(player, "$label added on track $(clip.track) — " *
                       "$(round(frames / fps, digits = 1))s, trim it like any clip")
    notify(player.playhead)
    return clip
end
