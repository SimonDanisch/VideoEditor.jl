# THERE ARE NO OVERLAYS. There are clips.
#
# What used to be here was a parallel timeline: `Sequence.overlays`, a list of
# `Overlay`s drawn over the finished canvas by a pass of their own. It had its own
# registry (`OverlayKind`, `registeroverlay!`), its own per-frame state bundle
# (`overlaystate`), its own keyframe world (timeline frames, where a clip keys by
# source frame), its own card builder, its own span arithmetic, its own
# serialization, its own bake, and its own compositing scene. Every feature of a
# clip had to be written a second time to exist for an overlay — or it simply did
# not: a 3D scene could animate for weeks with no card to see one of its numbers
# on, because the panel only knew how to build a card for a clip.
#
# A title IS a clip. It occupies a span on a track, it has an in point and an out
# point, it takes an effect stack, it composites with an opacity, its numbers are
# keyframed. The only thing that made it different was that it drew instead of
# decoding, and that is a property of its SOURCE (see `SceneSource`).
#
# So this file is what is left: the stock things you can put on a timeline that
# are not footage, each as a function returning a `SceneSpec`. A preset, not a
# kind — nothing registers, nothing dispatches, and adding one is writing a
# function that returns plots.

"Colour from a setting, faded by an opacity factor (settings may carry strings)."
fadedcolor(c, α::Real) = (col = Makie.to_color(c);
                          Makie.RGBAf(col.r, col.g, col.b, col.alpha * Float32(α)))

"""
    pixelscene(plots...; kw...) -> Makie.SceneSpec

A scene whose coordinates ARE canvas pixels — what every 2-D graphic here wants.

`campixel!`, so a position of `(960, 540)` is that pixel and a font size of `72` is
72 pixels. Authoring against the canvas rather than against fractions of it is the
honest unit: the numbers in the card are the numbers on the screen, and a scene
clip carries its own canvas size, so there is nothing to rescale against.
"""
pixelscene(plots::Makie.PlotSpec...; kw...) =
    Makie.SpecApi.Scene(; camera = Makie.campixel!, plots = collect(Makie.PlotSpec, plots),
                        kw...)

"""
    textscene(text; canvas, x, y, size, color, rotation) -> Makie.SceneSpec

A title, as a scene. Every number here is a keyframable path on the resulting clip
(`title.fontsize`, `title.rotation`, …) because every numeric plot attribute is —
the card reads them off the plot, so nothing had to declare them.
"""
function textscene(text::AbstractString;
                   canvas::Tuple{Integer, Integer} = (1920, 1080),
                   x::Real = 0.5, y::Real = 0.85, size::Real = 0.08,
                   color = :white, rotation::Real = 0.0)
    W, H = canvas
    return pixelscene(Makie.PlotSpec(:Text, [Makie.Point2f(x * W, y * H)];
                                     text = String(text), fontsize = Float32(size * H),
                                     color = Makie.to_color(color),
                                     rotation = Float32(deg2rad(rotation)),
                                     align = (:center, :center), name = :title))
end

"""
    barscene(; canvas, x, y, width, height, color, opacity) -> Makie.SceneSpec

A solid rectangle — the lower third's backing, a letterbox band, a wipe.
"""
function barscene(; canvas::Tuple{Integer, Integer} = (1920, 1080),
                  x::Real = 0.0, y::Real = 0.0, width::Real = 1.0, height::Real = 0.18,
                  color = :black, opacity::Real = 0.6)
    W, H = canvas
    return pixelscene(Makie.PlotSpec(:Poly,
        [Makie.Rect2f(x * W, y * H, width * W, height * H)];
        color = fadedcolor(color, opacity), name = :bar))
end

"Frame `n` at `fps` as `HH:MM:SS:FF` (SMPTE-style, non-drop)."
function timecodestring(n::Integer, fps::Real)
    f = fps > 0 ? fps : 25.0
    total = max(Int(n), 0)
    frames = round(Int, f)
    s, ff = divrem(total, frames)
    m, ss = divrem(s, 60)
    h, mm = divrem(m, 60)
    return string(lpad(h, 2, '0'), ':', lpad(mm, 2, '0'), ':',
                  lpad(ss, 2, '0'), ':', lpad(ff, 2, '0'))
end

"""
    timecodescene(; canvas, x, y, size, color, framerate) -> Makie.SceneSpec

A running timecode. The TEXT is what changes per frame, and it is not a number, so
it does not come from a curve — it is written per frame from the clip's own
position (see `timecodetext!`).
"""
function timecodescene(; canvas::Tuple{Integer, Integer} = (1920, 1080),
                       x::Real = 0.5, y::Real = 0.05, size::Real = 0.04,
                       color = :white, framerate::Real = 25.0)
    W, H = canvas
    return pixelscene(Makie.PlotSpec(:Text, [Makie.Point2f(x * W, y * H)];
                                     text = timecodestring(0, framerate),
                                     fontsize = Float32(size * H), font = :bold,
                                     color = Makie.to_color(color),
                                     align = (:center, :center), name = :timecode))
end

"""
`values` resampled to `n` points by linear interpolation — the correspondence
rule that lets two series of different length morph into one another.
"""
function resampleseries(v::AbstractVector{<:Real}, n::Integer)
    length(v) == n && return Float64.(v)
    length(v) == 1 && return fill(Float64(v[1]), n)
    n == 1 && return [Float64(v[1])]
    return [(t = (i - 1) * (length(v) - 1) / (n - 1) + 1;
             lo = floor(Int, t); hi = min(lo + 1, length(v));
             Float64(v[lo]) + (t - lo) * Float64(v[hi] - v[lo])) for i in 1:n]
end

"""
The two series interpolated at `morph`, both resampled to the longer length first.
`morph = 0` is `values`, `1` is `values2`; a keyframed `morph` is a plot growing
into another plot, which is all "morphing" needs to be as long as the two shapes
are sampled onto a common parameterization.
"""
function morphseries(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, morph::Real)
    isempty(b) && return Float64.(a)
    isempty(a) && return Float64.(b)
    n = max(length(a), length(b))
    ra, rb = resampleseries(a, n), resampleseries(b, n)
    t = clamp(Float64(morph), 0.0, 1.0)
    return (1 - t) .* ra .+ t .* rb
end

"""
    curvescene(values; canvas, x, y, width, height, linewidth, color) -> Makie.SceneSpec

A data plot laid into the canvas: `values` mapped onto the rect, drawn as a line.
"""
function curvescene(values::AbstractVector{<:Real};
                    canvas::Tuple{Integer, Integer} = (1920, 1080),
                    x::Real = 0.08, y::Real = 0.12, width::Real = 0.84, height::Real = 0.3,
                    linewidth::Real = 4.0, color = :orangered)
    W, H = canvas
    vals = Float64.(values)
    pts = if isempty(vals)
        Makie.Point2f[]
    else
        lo, hi = extrema(vals)
        span = hi - lo
        [Makie.Point2f((x + width * (length(vals) == 1 ? 0.0 : (i - 1) / (length(vals) - 1))) * W,
                       (y + height * (span == 0 ? 0.5 : (vals[i] - lo) / span)) * H)
         for i in eachindex(vals)]
    end
    return pixelscene(Makie.PlotSpec(:Lines, [pts]; linewidth = Float32(linewidth),
                                     color = Makie.to_color(color), name = :curve))
end

# ------------------------------------------------------------ a scene, rebuilt
#
# A `Makie.SceneSpec` holds live objects — `Transformation`s for a rig, loaded
# meshes — so it is not something a project file can contain. What the file holds
# is the RECIPE: which builder, with which arguments. `buildscene` runs it.
#
# This is a placeholder for serialising the scene itself, and it is deliberately a
# small closed table rather than an `eval`: a project file must not be able to name
# a function into existence. When scene serialisation lands, `"build"` becomes the
# scene and this goes.

"""
    buildscene(d) -> (root, joints, camera)

The scene a project file's `"build"` entry describes, plus what drives it.

`joints` and `camera` are empty for everything but a rig: a title has plots and
nothing else. They come back HERE rather than being fished out afterwards, because
they are made while the scene is made and there is no second place that knows both.
"""
plainscene(root) = (root = root, joints = Dict{Symbol, Any}(), camera = nothing)

function buildscene(d::AbstractDict)
    kind = Symbol(get(d, "kind", "rig"))
    a = fromspecvalue(get(d, "args", Dict{String, Any}()))
    canvas = Tuple(Int.(get(a, :canvas, (1920, 1080))))
    kind === :text && return plainscene(textscene(String(get(a, :text, "")); canvas,
                                       x = get(a, :x, 0.5), y = get(a, :y, 0.85),
                                       size = get(a, :size, 0.08),
                                       color = get(a, :color, :white),
                                       rotation = get(a, :rotation, 0.0)))
    kind === :bar && return plainscene(barscene(; canvas, x = get(a, :x, 0.0),
                                     y = get(a, :y, 0.0), width = get(a, :width, 1.0),
                                     height = get(a, :height, 0.18),
                                     color = get(a, :color, :black),
                                     opacity = get(a, :opacity, 0.6)))
    kind === :timecode && return plainscene(timecodescene(; canvas, x = get(a, :x, 0.5),
                                               y = get(a, :y, 0.05),
                                               size = get(a, :size, 0.04),
                                               color = get(a, :color, :white),
                                               framerate = get(a, :framerate, 25.0)))
    kind === :captions && return plainscene(pixelscene(Makie.PlotSpec(:Text,
        [Makie.Point2f(0.5 * canvas[1], get(a, :y, 0.12) * canvas[2])];
        text = "", fontsize = Float32(get(a, :size, 0.05) * canvas[2]),
        color = Makie.to_color(:white), align = (:center, :center), name = :captions)))
    kind === :curve && return plainscene(curvescene(Float64[v for v in get(a, :values, Float64[])];
                                         canvas, x = get(a, :x, 0.08), y = get(a, :y, 0.12),
                                         width = get(a, :width, 0.84),
                                         height = get(a, :height, 0.3),
                                         linewidth = get(a, :linewidth, 4.0)))
    kind === :rig && return rigscene(get(d, "rig", Dict{String, Any}()))
    error("project names an unknown scene builder: $(repr(kind))")
end

"""
    rigscene(d) -> (root, joints, camera)

A jointed 3-D figure: meshes hanging in a transformation tree.

This is the old `ScenePart` model, kept as a BUILDER. It was the data model —
every scene had to be describable as parts with a parent, an origin, an axis and
an angle, and the panel listed those four fields whatever the scene actually was.
Now it is one way to make a scene, callable from scene code like any other mesh
loader, and what the panel offers comes from the plots it produced.

The parent chain is the rig: rotating `arm_left` carries `hand_left` with it. A
part with an origin sits at its pivot; one without must UNDO its parent's
translation, because its mesh is already in world coordinates and would otherwise
inherit it twice — that is what made the hands float beside the figure.
"""
function rigscene(d::AbstractDict)
    parent0 = Makie.Scene()                     # a parent for the root transformations
    joints = Dict{Symbol, Any}()
    # The transformations, while we build: a part hangs off its parent's, and that
    # chain is what makes turning `arm_left` carry `hand_left`. Build-time only —
    # afterwards each one lives on its plot, which is where the animation finds it.
    transforms = Dict{Symbol, Any}()
    plots = Makie.PlotSpec[]
    for pd in get(d, "parts", ())
        name = Symbol(pd["name"])
        parentname = get(pd, "parent", nothing)
        parent = parentname === nothing ? parent0 :
                 get(transforms, Symbol(parentname), parent0)
        trans = Makie.Transformation(parent)
        transforms[name] = trans
        origin = Vec3f(Tuple(get(pd, "origin", (0.0, 0.0, 0.0))))
        base = any(!=(0), origin) ? origin :
               -Vec3f(Makie.transformation(parent).translation[])
        # THE DESCRIPTION's own words: which way the part turns and where it
        # starts. What it is turned TO is a parameter, not this.
        joints[name] = RigJoint(Vec3f(Tuple(get(pd, "axis", (0.0, 0.0, 1.0)))), base)
        Makie.translate!(trans, base)
        args = map(partmesh, fromspecvalue(get(pd, "args", [])))
        if any(!=(0), origin) && length(args) == 1 && args[1] isa GeometryBasics.Mesh
            m = args[1]
            args = Any[GeometryBasics.mesh(m.position .- Point3f(origin), faces(m);
                                           normal = m.normal)]
        end
        kwargs = fromspecvalue(get(pd, "kwargs", Dict{String, Any}()))
        push!(plots, Makie.PlotSpec(Symbol(pd["type"]), args...; kwargs...,
                                    name = name, transformation = trans))
    end
    lights = Makie.AbstractLight[riglight(l) for l in get(d, "lights", ())]
    cd_ = get(d, "camera", Dict{String, Any}())
    cam = (eye = Vec3f(Tuple(get(cd_, "eye", (3.0, 3.0, 3.0)))),
           lookat = Vec3f(Tuple(get(cd_, "lookat", (0.0, 0.0, 0.0)))),
           up = Vec3f(Tuple(get(cd_, "up", (0.0, 0.0, 1.0)))))
    root = Makie.SpecApi.Scene(; camera = Makie.cam3d!, lights = lights, plots = plots)
    return (root = root, joints = joints, camera = cam)
end


"""
    partmesh(arg) -> arg

A plot argument that names a FILE, loaded. Anything else passes through.

A rig's parts are meshes on disk (`"lego_arm_left.stl"`), and a spec holds what a
plot is given — so the string has to become geometry somewhere. `FileIO.load`
knows what an `.stl` is; there is no table here that would have to learn each
format twice.
"""
function partmesh(x::AbstractString)
    isfile(x) && return Makie.FileIO.load(x)
    # A BARE NAME is one of Makie's own assets — that is where the lego figure's
    # parts come from, and a project written against them carries the name and not
    # a path, which is right: the path is this machine's, the name is the asset's.
    asset = Makie.assetpath(x)
    isfile(asset) && return Makie.FileIO.load(asset)
    error("scene part names a mesh that is nowhere: $(repr(x)) — not a file, and \
           not one of Makie's assets")
end
partmesh(x) = x

"One light of a rig, as Makie's own."
function riglight(l::AbstractDict)
    t = Symbol(get(l, "type", "ambient"))
    col = RGBf(Float32.(Tuple(get(l, "color", (1.0, 1.0, 1.0))))...)
    t === :ambient && return Makie.AmbientLight(col)
    pos = Vec3f(Float32.(Tuple(get(l, "position", (0.0, 0.0, 0.0))))...)
    return Makie.PointLight(col, Point3f(pos), Vec2f(0, 1))
end

# ---------------------------------------------------------- opening an old project
#
# A project written while overlays were a thing of their own.
#
# An overlay was a span on the timeline with an effect on it, drawn over the
# finished canvas. That is a clip on the track above the footage, and this is the
# whole of the conversion: the span becomes the clip's extent, the effect comes
# across as it is (ids, values, curves and all), and what the overlay DREW becomes
# the scene the clip's source renders.
#
# Keys move from TIMELINE frames to SOURCE frames, which for a clip starting at
# the overlay's start and running at rate 1 differ by exactly `start` — so a curve
# is shifted, not reinterpreted.

"""
    overlaybuild(kind, settings, vals, canvas, framerate) -> Dict

The `"build"` recipe for an overlay of this kind: which builder, with which
arguments (see [`buildscene`](@ref)).
"""
function overlaybuild(kind::Symbol, settings::AbstractDict, vals::AbstractDict,
                      canvas::Tuple{Int, Int}, framerate::Real)
    num(k, d) = Float64(get(vals, String(k), d))
    args = Dict{String, Any}("canvas" => [canvas[1], canvas[2]])
    kind === :scene && return Dict{String, Any}("kind" => "rig",
                                                "rig" => get(settings, "spec", Dict{String, Any}()))
    if kind === :text
        args["text"] = String(get(settings, "text", ""))
        args["x"] = num(:x, 0.5); args["y"] = num(:y, 0.85)
        args["size"] = num(:size, 0.08); args["rotation"] = num(:rotation, 0.0)
        args["color"] = specvalue(get(settings, "color", :white))
    elseif kind === :bar
        args["x"] = num(:x, 0.0); args["y"] = num(:y, 0.0)
        args["width"] = num(:width, 1.0); args["height"] = num(:height, 0.18)
        args["opacity"] = num(:opacity, 0.6)
        args["color"] = specvalue(get(settings, "color", :black))
    elseif kind === :timecode
        args["x"] = num(:x, 0.5); args["y"] = num(:y, 0.05)
        args["size"] = num(:size, 0.04); args["framerate"] = Float64(framerate)
        args["color"] = specvalue(get(settings, "color", :white))
    elseif kind === :captions
        args["y"] = num(:y, 0.12); args["size"] = num(:size, 0.05)
    elseif kind === :curve
        args["x"] = num(:x, 0.08); args["y"] = num(:y, 0.12)
        args["width"] = num(:width, 0.84); args["height"] = num(:height, 0.3)
        args["linewidth"] = num(:linewidth, 4.0)
        args["values"] = [Float64(v) for v in get(settings, "values", Float64[])]
    else
        return nothing
    end
    return Dict{String, Any}("kind" => String(kind), "args" => args)
end

"""
    clipfromoverlay(d, seq) -> Union{Nothing, Clip}

One overlay from an old project file, as a clip on a track of its own.

`nothing` for a kind this editor no longer has: a project naming something
unknown opens without it rather than refusing to open, and the rest of the edit
is intact.
"""
function clipfromoverlay(d::AbstractDict, seq::Sequence)
    kind = Symbol(get(d, "kind", ""))
    start = Int(get(d, "start", 0))
    stop = Int(get(d, "stop", start + 90))
    # The canvas the overlay was DRAWN at — which is the sequence's, derived from
    # the clips when the file does not state one. Falling back to a default here
    # gave a migrated scene a 1920x1080 source over a 640x1138 project, and the
    # placement then fitted the whole thing into a corner.
    canvas = canvassize(seq)
    ed = get(d, "effect", nothing)
    settings = Dict{String, Any}(get(d, "settings", Dict{String, Any}()))
    stored = Param[]
    # An overlay's effect held its parameters; a scene's also held the SPEC, as a
    # `Param{SceneSpec}`. There is no such parameter type any more — the scene is
    # the source's — so its entry is taken as the dict it was written as, which is
    # exactly the rig recipe `buildscene` wants. Read from the raw entry rather
    # than through `from_msgpack(Param, …)`, which would look for a type this
    # editor deliberately no longer has.
    for pd in (ed isa AbstractDict ? get(ed, "params", ()) : ())
        if String(get(pd, "name", "")) == "spec"
            v = get(pd, "value", nothing)
            v isa AbstractDict && (settings["spec"] = v)
        else
            push!(stored, MsgPack.from_msgpack(Param, pd))
        end
    end
    vals = Dict{String, Any}(String(p.name) => p.value for p in stored if p.value isa Real)
    build = overlaybuild(kind, settings, vals, canvas, seq.framerate)
    build === nothing && return nothing
    clip = sceneclip(buildscene(build); start, frames = max(stop - start, 1), canvas,
                     framerate = seq.framerate)
    clip.source.build = build          # …so it can be written back out
    clip.track = ntracks(seq) + 1        # over the footage, where it was drawn
    fx = findslot(clip, :scene)
    # EVERY parameter the overlay had comes across, curve and all — pushed onto the
    # scene effect, not matched against it. The effect starts empty: what a scene
    # offers is discovered from the scene when a card is opened, which has not
    # happened yet and must not have to for a saved keyframe to survive. The paths
    # are unchanged (`"torso.angle"`), so when the card is opened these are the
    # parameters it finds.
    #
    # The keys move from TIMELINE frames into the clip's own. An overlay had no
    # source, so it keyed against the timeline; a clip keys against its source, and
    # this clip starts at `start` and runs at rate 1 — so the two differ by exactly
    # that offset. A shift, not a reinterpretation.
    for q in stored
        q.curve === nothing || (q.curve = AnimCurve{typeof(q.value)}(
            [Keyframe{typeof(q.value)}(k.frame - start, k.value, k.ease, k.inhandle, k.outhandle)
             for k in q.curve.keys], q.curve.interp))
        push!(fx.params, q)
    end
    return clip
end



"""
    scenebuild(kind, canvas; kw...) -> Dict

The `"build"` recipe for one of the stock scenes, at this canvas.

What a command hands to [`addsceneclip!`](@ref): the clip is built from the recipe
and SAVES the recipe, so what reopens is what was made. One path — a title created
in the editor and a title read from a project file go through the same
[`buildscene`](@ref).
"""
scenebuild(kind::Symbol, canvas::Tuple{Integer, Integer}; kw...) =
    Dict{String, Any}("kind" => String(kind),
                      "args" => merge(Dict{String, Any}("canvas" => [canvas[1], canvas[2]]),
                                      Dict{String, Any}(String(k) => specvalue(v)
                                                        for (k, v) in pairs(kw))))
