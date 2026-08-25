# Rendering a `SceneSpec` — ONE path, with the backend as a parameter.
#
# There is no per-backend method here and no extension package, because there is
# nothing backend-specific to say. Makie already takes the renderer as a value
# (`display(scene; backend = SomeModule)`) and already reads a backend's own
# options out of the THEME — `ScreenConfig`'s fields for RayMakie are exactly its
# theme attributes (integrator, exposure, tonemap, gamma, device, …). So a scene
# that carries a backend NAME and a theme carries everything the renderer needs,
# and this file just hands both over.
#
# That is why `SceneSpec.backend` is a Symbol: a project file has to name the
# renderer in text, and the module is looked up at render time from what is
# loaded. Loading RayMakie is what makes `backend = :RayMakie` work; nothing here
# has to know it exists.
#
# Why not `Makie.colorbuffer(scene)`: that goes through GLMakie's SINGLETON
# offscreen screen, and every scene-taking `Screen` constructor empties it and
# re-displays itself on it. The editor's own window comes from the same
# constructor, so rendering a scene that way evicts the editor's figure from its
# own window — measured, the renderlist went from 32 plots to 3. See
# `canvasscreen` in overlays.jl, which learned this the hard way.

"""
The renderers a scene may name, by name.

A project file has to name its backend in TEXT, so something has to turn
`:RayMakie` into a module. That something is this, and it is a plain table you
put a module INTO — `usebackend!(RayMakie)` — rather than a search over whatever
happens to be loaded. Explicit both ways: nothing renders through a backend
nobody asked for, and a missing one is named in the error instead of failing
somewhere inside Makie.

GLMakie is here from the start because the editor draws its own window with it.
"""
const BACKENDS = Dict{Symbol, Module}(:GLMakie => GLMakie)

"""
    usebackend!(mod) -> Module

Make `mod` available to scenes that name it, keyed on its own module name.

    usebackend!(RayMakie)     # now `backend = :RayMakie` renders
"""
usebackend!(mod::Module) = (BACKENDS[nameof(mod)] = mod)

"""
    makielights(specs) -> Vector{Makie.AbstractLight}

The scene's lights, built from data.

Ambient carries no position; point and directional do. Anything unknown is
skipped rather than fatal — a scene from a newer editor may name a light type
this one has not got, and losing a lamp beats losing the frame.

A point light falls off with the SQUARE of distance, which is what makes one
intensity number mean the same thing to both renderers. Makie's default is
`Vec2f(0)` — no falloff at all — while a raytracer is physical by construction,
so a lamp written for RayMakie arrived in GLMakie undimmed: measured at the
lego figure, intensity 15000 at 269 units away came back a flat (1.0, 1.0, 1.0)
on every lit pixel. The figure was not missing its legs, it was blue legs blown
to white on white. With `Vec2f(0, 1)` the same scene renders (0.42, 0.03, 0.07)
on the red torso — the hue, at nothing clipped.
"""
const QUADRATIC = Vec2f(0, 1)

function makielights(specs::AbstractVector{LightSpec})
    out = Makie.AbstractLight[]
    for l in specs
        col = RGBf(l.color...)
        if l.type === :ambient
            push!(out, Makie.AmbientLight(col))
        elseif l.type === :point
            push!(out, Makie.PointLight(col, Vec3f(l.position), QUADRATIC))
        elseif l.type === :directional
            push!(out, Makie.DirectionalLight(col, Vec3f(l.position)))
        end
    end
    return out
end

"""
A theme dictionary as Makie `Attributes`, nested dictionaries and all.

`Makie.Theme(; d...)` leaves an inner `Dict` a `Dict`, and Makie then indexes it
as an `Attributes` — `getindex(::Observable{Any}, ::Symbol)`, which is where a
raw theme dict falls over. Converting on the way in means the project file can
hold plain nested JSON and still describe a theme.
"""
themeattributes(d::AbstractDict) =
    Makie.Attributes(; (k => (v isa AbstractDict ? themeattributes(v) : v)
                        for (k, v) in d)...)

"""
Pixels are 1:1 unless the scene says otherwise.

A composited frame is measured in pixels, so a screen that applies HiDPI scaling
silently returns a different size than asked for — measured, a (240, 180) canvas
came back (261, 348). These are ordinary theme entries under the backend's name,
so a spec that wants something else simply says so and wins the merge below.
"""
const PIXELEXACT = Makie.Theme(GLMakie = (px_per_unit = 1.0, scalefactor = 1.0))

"""
    backendscreen(backend, spec, canvas) -> screen

An offscreen screen of `backend`, configured from `spec.theme` — WITHOUT touching
the global theme.

THE GLOBAL THEME IS NOT OURS. This used to build under `Makie.with_theme(...)`,
because that is how a backend's screen options are normally passed. But this runs
from the overlay's draw callback — during a seek, and during PLAYBACK — while
GLMakie's render loop is drawing the editor concurrently. `with_theme` restores
by calling `set_theme!()` first, which resets to Makie's built-in LIGHT default
for an instant, and a render landing in that window redraws the editor with it:
press Play, the whole window turns white. Simon found it by reading, not by
reproducing — "wahrscheinlich veränderst du das globale Theme und machst keine
saubere Trennung", which is exactly what it was.

So the options go straight to the screen constructor instead, which is what
`canvasscreen` in overlays.jl already does for the compositing screen. Nothing
global is written, and there is no window in which anyone else can observe a
theme that is not theirs.

`px_per_unit`/`scalefactor` are pinned to 1: a screen applying its HiDPI scale
silently returns a different size than asked for — measured, a (300, 400) canvas
came back (434, 579). A spec that says otherwise for this backend wins.
"""
function backendscreen(backend::Module, spec::SceneSpec, canvas::NTuple{2, Integer})
    opts = Dict{Symbol, Any}(:visible => false, :px_per_unit => 1.0, :scalefactor => 1.0)
    for (k, v) in get(spec.theme, nameof(backend), Dict{Symbol, Any}())
        opts[Symbol(k)] = v
    end
    # a backend that does not know one of these must not die of it
    fields = fieldnames(backend.ScreenConfig)
    filter!(kv -> kv[1] in fields || kv[1] === :visible, opts)
    return backend.Screen(; opts...)
end

"""
    coverage(screen) -> Union{Nothing, AbstractMatrix{Bool}}

Which pixels the scene actually drew on — `nothing` when the renderer cannot say.

An overlay must be TRANSPARENT where the scene is empty, or it is not an overlay
but a replacement: measured, the lego scene over footage changed all 57600 pixels
of a 320x180 frame and the video was simply gone.

GLMakie cannot answer this from colour. `colorbuffer` stages through a
`Matrix{RGB{N0f8}}` framecache, so alpha is dropped before the caller sees it,
and reading the framebuffer's own colour texture gives alpha 0 on EVERY pixel,
the figure's included. Rendering twice on two background colours does not work
either — the scene's `backgroundcolor` makes no difference to what comes back
(measured: black vs white differ in exactly 0 pixels).

The DEPTH buffer does answer it: whatever the far plane still owns was never
drawn on. Measured against a colour-threshold mask of the same frame the two
agree on 99.6% of pixels, and where they differ it is the antialiased fringe,
which depth correctly leaves out.

Dispatch rather than a branch on backend name: a renderer that carries real alpha
— a raytracer knows a ray that hit nothing — adds its own method and nothing in
the render path above changes.
"""
coverage(::Any) = nothing
coverage(screen::GLMakie.Screen) = GLMakie.depthbuffer(screen) .< 1.0f0

"""
    withalpha(img, cov) -> Matrix{RGBA{N0f8}}

`img` as RGBA, transparent wherever `cov` says nothing was drawn.

`cov === nothing` means the renderer could not tell us, and the honest result is
then fully opaque — an overlay that covers the frame is wrong, but inventing a
mask from colour would key out the figure's own white parts, which is worse and
much harder to see.
"""
function withalpha(img::AbstractMatrix, cov::Nothing)
    return RGBA{N0f8}.(img)
end

function withalpha(img::AbstractMatrix, cov::AbstractMatrix{Bool})
    out = Matrix{RGBA{N0f8}}(undef, size(img))
    @inbounds for i in eachindex(img, cov)
        c = img[i]
        out[i] = RGBA{N0f8}(red(c), green(c), blue(c), cov[i] ? 1 : 0)
    end
    return out
end

"""
    getbackend(name) -> Module

The registered renderer called `name`, or an error naming what is registered.
"""
getbackend(name::Symbol) = get(BACKENDS, name) do
    error("scene backend $(repr(name)) is not registered — `VideoEditor.usebackend!($name)` " *
          "after loading it. Have: $(join(sort!(collect(keys(BACKENDS))), ", "))")
end

# A project file holds strings, so `settings.bakewith` comes back as one — the
# spec's own `backend` is rebuilt as a Symbol by `scenefromdict`, but a plain
# setting is not. Without this a reopened project raised a MethodError on the
# first bake instead of rendering.
getbackend(name::AbstractString) = getbackend(Symbol(name))

"""
    renderspec(spec, canvas) -> Matrix{RGBA{N0f8}}

`spec` rendered at `canvas`, by the backend it names, under the theme it carries.

ONE PATH, and it is the preview's. This used to build the scene a second way of
its own — its own walk over the parts, its own transformation per part — and the
two disagreed: it derived a jointless part's parent compensation from the
parent's ANIMATED translation, so lifting the torso by 20.8 gave the head a
matching -20.8 and the head stayed behind while the body rose. Rendered on
RayMakie the figure came apart: head detached and turned, legs off the belt.
GLMakie, which went through the live path, was correct the whole time — which is
exactly how a second implementation hides, by being the one you look at less.

So there is no second builder. [`livescene!`](@ref) builds, [`liveframe!`](@ref)
draws, and the backend is the parameter it always was. The theme is applied
inside `livescene!`, around the whole build: Makie reads a backend's screen
options from it, so `theme[:RayMakie][:exposure]` reaches the renderer without
this function knowing what an exposure is.
"""
function renderspec(spec::SceneSpec, canvas::NTuple{2, Integer},
                    backend::Module = getbackend(spec.backend))
    live = livescene!(nothing, spec, canvas, backend)
    img = liveframe!(live, spec)
    # GLMakie hands back a screen that is reused; RayMakie's is per-display.
    # Closing what we opened is the caller-neutral thing to do either way.
    applicable(close, live.screen) && close(live.screen)
    return img
end

# ---------------------------------------------------------------- baking

# TWO BACKENDS, ONE SCENE — and that is the whole workflow.
#
# Raytracing a frame costs ~1.2 s at 480x854; rasterising it costs milliseconds.
# Scrubbing a timeline at raytracing speed is not editing, and editing against a
# rasterised preview and then shipping it is not the picture you wanted. So the
# scene carries BOTH: `spec.backend` is what the preview draws with, `bakewith`
# is what the final render uses, and `bake!` walks the overlay's frame range once
# and keeps the results.
#
# A bake is a CACHE, not a state: it is keyed on the frames it covers, and any
# edit that changes what those frames look like drops it (see `bakedframe`). The
# alternative — asking the user to remember to re-bake — ships the wrong picture
# eventually, and silently.

"""
    bakedframes(ov) -> Union{Nothing, Dict{Int, Matrix{RGBA{N0f8}}}}

The overlay's baked frames, or `nothing` when it has never been baked.
"""
bakedframes(ov) = get(ov.settings, :baked, nothing)

"""
    bake!(ov, canvas; framerate = 0.0, progress = nothing) -> Int

Render every frame of `ov`'s range with its bake backend and keep the results.

Returns how many frames were baked. `progress(done, total)` is called as it goes,
because at a second per frame a silent minute reads as a hang.

The bake backend comes from `settings.bakewith`, falling back to the spec's own
— so a scene that names no second backend simply bakes what it previews, which
is the right answer for a scene that is cheap anyway.

ONE scene held across the range, not one per frame. `renderspec` builds and
tears down, which means reloading every part's mesh from disk on every frame —
0.28 s of the lego figure's ten STL files, per frame, on top of the render. A
bake is the one place that cost would be paid hundreds of times over.

Pass `into` — a project path — and each frame is written to that project's
sidecar AS IT FINISHES, not at the end. Minutes of raytracing survive a crash
that way, and the partial directory is directly usable: [`loadbakes!`](@ref)
takes the frames that are there and draws the rest live. The manifest is only
written on a clean finish, so an interrupted bake is not mistaken for a whole
one; until then the frames sit there costing nothing.
"""
function bake!(ov, canvas::NTuple{2, Integer}; framerate::Real = 0.0, progress = nothing,
               into::Union{Nothing, AbstractString} = nothing)
    spec = get(ov.settings, :spec, nothing)
    spec isa SceneSpec || return 0
    backend = get(ov.settings, :bakewith, spec.backend)
    frames = Dict{Int, Matrix{RGBA{N0f8}}}()
    total = max(ov.stop - ov.start, 0)
    # the bake backend, not the spec's: `livescene!` keys its reuse on
    # `spec.backend`, so a scene previewing on GLMakie and baking on RayMakie has
    # to say so here or it would bake with the preview's renderer.
    baking = animatedspec(spec, (;))
    baking.backend = backend
    dir = into === nothing ? nothing : bakeframedir(into, ov.id)
    dir === nothing || isdir(dir) || mkpath(dir)
    live = nothing
    try
        for (k, n) in enumerate(ov.start:(ov.stop - 1))
            st = overlaystate(ov, n; framerate = framerate)
            anim = animatedspec(baking, st)
            live = livescene!(live, anim, canvas, getbackend(backend))
            frames[n] = liveframe!(live, anim)
            dir === nothing || writebakeframe(dir, n, frames[n])
            progress === nothing || progress(k, total)
        end
    finally
        live === nothing || (applicable(close, live.screen) && close(live.screen))
    end
    # the fingerprint is taken HERE, with the frames, and travels with them.
    # Recomputing it at save time instead is the bug this line exists to prevent:
    # edit a keyframe, save, and `savebakes` would stamp the NEW fingerprint onto
    # the OLD frames, so reopening accepted a stale bake as current — measured,
    # it did exactly that.
    ov.settings = merge(ov.settings, (baked = frames, bakedcanvas = canvas,
                                      bakedprint = bakefingerprint(ov, canvas)))
    # the manifest LAST, and only here: it is what marks the bake complete, so an
    # interrupted run leaves frames without one and is not read back as whole
    into === nothing || savebakes(into, (; overlays = [ov]))
    return length(frames)
end

"Throw the bake away — after an edit that changes what the scene looks like."
unbake!(ov) = (ov.settings = Base.structdiff(ov.settings,
                                             NamedTuple{(:baked, :bakedcanvas, :bakedprint)});
               ov)

# ------------------------------------------------------- the bake on disk
#
# A bake costs minutes — the lego walk is 180 raytraced frames — and losing it to
# a crash, or to closing the editor, means paying that again for a picture that
# has not changed. So it goes to a SIDECAR beside the project, the way a matte
# does (`savemattes`, project.jl), and comes back on open.
#
# PNG frames in a directory rather than one raw blob or a video:
#
#   - RAW is unaffordable. 180 frames of 640x1138 RGBA is 524 MB; the same
#     frames as PNG are mostly-transparent and compress to a fraction of it.
#   - A VIDEO would have to carry ALPHA, which rules out the ordinary codecs and
#     buys a container problem in exchange for nothing.
#   - A DIRECTORY is what makes it crash-safe: `bake!` writes each frame as it
#     finishes, so an editor that dies at frame 140 leaves 140 usable frames
#     rather than a truncated file. A missing frame is simply drawn live.
#
# And it is inspectable — the frames are PNGs you can open.

"Directory holding a project's baked scene frames (created on demand)."
bakedir(path::AbstractString) = string(path, ".bakes")
bakeframedir(path::AbstractString, id::Integer) = joinpath(bakedir(path), string(id))
bakemanifest(path::AbstractString, id::Integer) = joinpath(bakedir(path), string(id, ".json"))
bakename(n::Integer) = string(lpad(n, 6, '0'), ".png")

"""
    bakefingerprint(ov, canvas) -> UInt64

What the bake was made FROM, in one number.

THE POINT OF PERSISTING AT ALL is undone without this. A bake that no longer
matches its scene is worse than no bake: it composites a plausible, wrong
picture, and nothing about the result says it is stale. So the fingerprint
covers everything that decides what a frame looks like — the scene's parts,
lights and camera, every curve and keyframe, the frame range, the canvas and the
bake backend — and a mismatch means the sidecar is ignored and the frames are
drawn live until somebody re-bakes.

Deliberately fails SAFE in the other direction too: `hash` is stable within a
Julia version but not promised across them, so an upgrade invalidates old bakes.
That costs a re-bake, which is the cheap mistake to make.
"""
function bakefingerprint(ov, canvas::NTuple{2, Integer})
    spec = get(ov.settings, :spec, nothing)
    spec isa SceneSpec || return UInt64(0)
    h = hash((Int(ov.start), Int(ov.stop), Int(canvas[1]), Int(canvas[2]),
              String(spec.backend), string(get(ov.settings, :bakewith, spec.backend))))
    for p in spec.parts
        h = hash((String(p.name), p.parent === nothing ? "" : String(p.parent),
                  p.origin, p.axis, p.angle, p.offset, String(p.plot.type),
                  string(p.plot.args), string(p.plot.kwargs)), h)
    end
    for l in spec.lights
        h = hash((String(l.type), l.color, l.position), h)
    end
    h = hash((spec.camera.eye, spec.camera.lookat, spec.camera.up), h)
    # sorted: a Dict's order is not part of what the picture looks like
    for key in sort!(collect(keys(ov.animations)); by = String)
        c = ov.animations[key]
        h = hash((String(key), String(c.interp)), h)
        for k in c.keys
            h = hash((Int(k.frame), Float64(k.value), String(k.ease)), h)
        end
    end
    return h
end

"One baked frame, written where `loadbakes!` will look for it."
writebakeframe(dir::AbstractString, n::Integer, img::AbstractMatrix) =
    open(io -> PNGFiles.save(io, PermutedDimsArray(img, (2, 1))),
         joinpath(dir, bakename(n)), "w")

"""
    savebakes(path, seq)

Write every overlay's baked frames beside the project at `path`.

Called from [`saveproject`](@ref) for the same reason `savemattes` is: the frames
are an EDIT's expensive output, not something to recompute on every open. The
project JSON itself never carries them — see `CACHEDSETTINGS` in overlays.jl,
which keeps 18 MB of pixels out of a 13 KB file.
"""
function savebakes(path::AbstractString, seq)
    for ov in seq.overlays
        frames = bakedframes(ov)
        frames === nothing && continue
        canvas = get(ov.settings, :bakedcanvas, nothing)
        canvas isa NTuple{2, Integer} || continue
        # the fingerprint TAKEN AT BAKE TIME, never one computed now: the frames
        # are what they are, and the point of the fingerprint is to say what they
        # were made from. A bake without one cannot be vouched for, so it is not
        # persisted rather than persisted unverifiably.
        print = get(ov.settings, :bakedprint, nothing)
        print isa Integer || continue
        dir = bakeframedir(path, ov.id)
        isdir(dir) || mkpath(dir)
        for (n, img) in frames
            writebakeframe(dir, n, img)
        end
        open(bakemanifest(path, ov.id), "w") do io
            JSON.print(io, Dict{String, Any}(
                "fingerprint" => string(print),
                "canvas" => [Int(canvas[1]), Int(canvas[2])],
                "frames" => sort!(collect(keys(frames)))), 2)
        end
    end
    return nothing
end

"""
    loadbakes!(path, seq) -> seq

Put a project's baked frames back, when they still match what would be rendered.

A frame the sidecar names but does not have is SKIPPED rather than fatal: that is
exactly what a crash mid-bake leaves behind, and the missing ones simply draw
live. Same for a fingerprint that no longer matches — the bake is ignored, not
deleted, so an accidental edit that is undone gets its bake back.
"""
function loadbakes!(path::AbstractString, seq)
    isdir(bakedir(path)) || return seq
    for ov in seq.overlays
        mf = bakemanifest(path, ov.id)
        isfile(mf) || continue
        manifest = JSON.parse(read(mf, String))
        cv = get(manifest, "canvas", nothing)
        cv isa AbstractVector && length(cv) == 2 || continue
        canvas = (Int(cv[1]), Int(cv[2]))
        string(bakefingerprint(ov, canvas)) == get(manifest, "fingerprint", "") || continue
        dir = bakeframedir(path, ov.id)
        frames = Dict{Int, Matrix{RGBA{N0f8}}}()
        for n in get(manifest, "frames", Int[])
            file = joinpath(dir, bakename(Int(n)))
            isfile(file) || continue          # died mid-write: take what is there
            frames[Int(n)] = RGBA{N0f8}.(permutedims(PNGFiles.load(file), (2, 1)))
        end
        isempty(frames) && continue
        ov.settings = merge(ov.settings, (baked = frames, bakedcanvas = canvas,
                                          bakedprint = bakefingerprint(ov, canvas)))
    end
    return seq
end

"""
    bakedframe(ov, n, canvas)

The baked frame for timeline frame `n`, or `nothing` to draw it live.

Refuses a bake taken at another canvas size: a resized project would otherwise
composite a stale, differently-shaped picture, and that is the kind of wrong that
survives all the way into an export.
"""
function bakedframe(ov, n::Integer, canvas::NTuple{2, Integer})
    b = bakedframes(ov)
    b === nothing && return nothing
    get(ov.settings, :bakedcanvas, nothing) == canvas || return nothing
    bakecurrent(ov, canvas) || return nothing
    return get(b, Int(n), nothing)
end

"""
    bakecurrent(ov, canvas) -> Bool

Whether the bake still describes what would be rendered right now.

THE EDIT DOES NOT DESTROY THE BAKE. Moving a keyframe used to call `unbake!` and
throw the frames away, so the preview came back live — and the only way back to
the baked picture was to spend the minutes again, even if you undid the edit
immediately. Instead the frames stay exactly where they are and this asks whether
they still apply: edit, and the fingerprint stops matching, so every frame draws
live on the preview backend; undo, and it matches again and the bake is simply
there. Nothing was rendered twice and nothing was lost.

Same rule `loadbakes!` applies when a project opens, which is the point — a bake
is valid exactly when it was made from what is there now, whether that question
is asked a second or a week later. `unbake!` remains for DISCARDING one on
purpose.
"""
function bakecurrent(ov, canvas::NTuple{2, Integer})
    print = get(ov.settings, :bakedprint, nothing)
    print isa Integer || return false          # no fingerprint, nothing to vouch for it
    return print == bakefingerprint(ov, canvas)
end

# ------------------------------------------------- the scene's keyframable paths

"""
    scenepaths(spec) -> Vector{String}

Every path of `spec` a keyframe can be put on, in reading order.

Enumerated from the SCENE, not declared: which values a scene can animate depends
on which parts it has, so there is no fixed parameter list to register. Each part
contributes its joint angle and its three offset components; the camera
contributes its eye. That is the whole of what moves.
"""
function scenepaths(spec::SceneSpec)
    out = String[]
    for p in spec.parts
        name = String(p.name)
        push!(out, "$name.angle")
        append!(out, ["$name.offset[$i]" for i in 1:3])
    end
    append!(out, ["camera.eye[$i]" for i in 1:3])
    return out
end

"""
Sensible slider bounds for a scene path, by what it IS.

An angle is a turn and lives in ±π. An offset and a camera position are lengths
in the model's own units, and the lego figure is ~30 units tall — so a range that
lets a limb be nudged and the figure be walked across frame, without a slider
whose useful travel is one pixel wide.
"""
function scenebounds(path::AbstractString)
    endswith(path, ".angle") && return (-Float64(π), Float64(π))
    startswith(path, "camera.") && return (-500.0, 500.0)
    return (-200.0, 200.0)
end


# ---------------------------------------------------------------- in the editor

# The 3D scene overlay: a [`SceneSpec`](@ref) rendered over the canvas.
#
# ONE kind, not one per plot type. What it draws is data the overlay carries, and
# every path in that data is keyframeable through the ordinary curve machinery —
# `setoverlaykey!(ov, Symbol("arm_left.angle"), frame, value)` swings a limb, and
# `"camera.eye[1]"` moves the camera. There is nothing here that a Crop or a Text
# kind would need a second copy of; those become presets that build one of these.
#
# The spec rides in `settings` because it is structure, not a number: what changes
# per frame are the paths, and `overlaystate` merges those over it. Rendering goes
# through [`renderspec`](@ref), so the backend is whatever the spec names —
# `:GLMakie` while scrubbing, `:RayMakie` for the final, same scene either way.
"""
A scene held open across frames: the built scene and its screen.

WHY IT IS HELD. The first version rebuilt everything inside the per-frame
callback — a new `Scene`, a new `Screen`, and `partmesh` reloading all ten STL
files from disk, for every frame of a seek AND of playback. Measured after
fixing it, at 480x854 with a ten-part figure: first frame 0.28 s, every frame
after 5.1-5.8 ms on GLMakie. Almost none of the original cost was rendering.

Holding the screen is also what lets a PROGRESSIVE backend work at all: RayMakie
accumulates samples while the playhead stands still, and a screen thrown away
after each frame has nothing to accumulate into.

There is no structure-diffing here on purpose — `plotlist!` takes an observable
of specs and does exactly that, reusing plots whose type is unchanged
(`update_plot!` → `batch_update!`). Writing a second one by hand was a mistake I
made and removed.
"""
mutable struct LiveScene
    scene::Any
    screen::Any
    specs::Observables.Observable{Vector{Makie.PlotSpec}}
    joints::Dict{Symbol, Any}      # part name → its Transformation
    bases::Dict{Symbol, Any}       # …and the translation it was BUILT with
    canvas::NTuple{2, Int}
    backend::Symbol
end

"""
    livescene!(live, spec, canvas, backend) -> LiveScene

Build the scene, or hand back the one already built for this canvas and backend.

Only those two force a rebuild. Everything else — parts added or removed, a plot
changing type, any attribute — goes through the specs observable, which is
`plotlist!`'s job.
"""
function livescene!(live::Union{Nothing, LiveScene}, spec::SceneSpec,
                    canvas::NTuple{2, Integer}, backend::Module)
    if live !== nothing && live.canvas == canvas && live.backend === spec.backend
        return live
    end
    live === nothing || (applicable(close, live.screen) && close(live.screen))
    W, H = Int(canvas[1]), Int(canvas[2])
    scene = Makie.Scene(; size = (W, H), backgroundcolor = RGBAf(0, 0, 0, 0),
                        lights = makielights(spec.lights))
    Makie.cam3d!(scene)
    joints = Dict{Symbol, Any}()
    bases = Dict{Symbol, Any}()
    specs = Observables.Observable(partspecs(spec, joints, bases, scene))
    Makie.plotlist!(scene, specs)
    screen = backendscreen(backend, spec, (W, H))
    display(screen, scene)
    return LiveScene(scene, screen, specs, joints, bases, (W, H), spec.backend)
end

"""
    partspecs(spec, joints, scene) -> Vector{PlotSpec}

The scene's parts as plot specs, each NAMED and carrying its own transformation.

The `name` is what makes a part reachable afterwards — `Makie.findplot(scene,
:arm_left)` — so an animation can write straight to the plot instead of
re-specifying the scene. The transformations are built once and kept in
`joints`, because the parent chain IS the rig: rotating `arm_left` has to carry
`hand_left` with it.
"""
function partspecs(spec::SceneSpec, joints::Dict{Symbol, Any},
                   bases::Dict{Symbol, Any}, scene)
    out = Makie.PlotSpec[]
    for part in spec.parts
        parent = part.parent === nothing ? scene : get(joints, part.parent, scene)
        trans = Makie.Transformation(parent)
        # THE BASE TRANSLATION, set once. A part with a joint sits at its pivot;
        # one without must UNDO its parent's translation, because its mesh is
        # already in world coordinates and would otherwise inherit it twice —
        # that is what made the hands float beside the figure. `liveframe!` adds
        # only the animated offset on top, so re-writing this per frame (which it
        # used to) cannot wipe the compensation out again.
        base = any(!=(0), part.origin) ? Vec3f(part.origin) :
               -Vec3f(Makie.transformation(parent).translation[])
        bases[part.name] = base
        Makie.translate!(trans, base)
        joints[part.name] = trans
        args = map(partmesh, part.plot.args)
        if any(!=(0), part.origin) && length(args) == 1 && args[1] isa GeometryBasics.Mesh
            m = args[1]
            args = Any[GeometryBasics.mesh(m.position .- Point3f(part.origin), faces(m);
                                           normal = m.normal)]
        end
        push!(out, Makie.PlotSpec(part.plot.type, args...; part.plot.kwargs...,
                                  name = part.name, transformation = trans))
    end
    return out
end

"""
    liveframe!(live, spec) -> Matrix{RGBA{N0f8}}

One frame: write this frame's values onto the standing scene, read the picture.

The joints move through their `Transformation`s and any other attribute through
`Makie.findplot` + `update!` — direct observable writes, not a re-spec. The docs
are explicit that specs are the slower path for animation ("it needs to
re-create plots often and needs to go over the whole plot tree"), and a rig only
ever changes VALUES between frames.
"""
function liveframe!(live::LiveScene, spec::SceneSpec)
    for part in spec.parts
        t = get(live.joints, part.name, nothing)
        if t !== nothing
            # base + the animated offset — NOT `origin + offset`, which threw the
            # parent compensation away for every part without a joint.
            Makie.translate!(t, get(live.bases, part.name, Vec3f(0)) .+ Vec3f(part.offset))
            Makie.rotate!(t, Vec3f(part.axis), Float32(part.angle))
        end
        isempty(part.plot.kwargs) && continue
        plot = Makie.findplot(live.scene, part.name)
        plot === nothing || Makie.update!(plot; part.plot.kwargs...)
    end
    c = spec.camera
    Makie.update_cam!(live.scene, Makie.cameracontrols(live.scene),
                      Vec3f(c.eye), Vec3f(c.lookat), Vec3f(c.up))
    img = Makie.colorbuffer(live.screen, Makie.GLNative)
    return withalpha(img, coverage(live.screen))
end

registeroverlay!(:scene, "3D Scene", FxParam[],
    function (scene, canvas, state)
        W, H = canvas
        img = Observables.Observable(zeros(RGBA{N0f8}, W, H))
        # y ASCENDING, unlike the canvas's own video frame. The render hands back
        # GLNative, whose row 1 is the scene's BOTTOM — already flipped relative
        # to a video frame. Drawing it with the descending range the frame uses
        # flips it a second time and the figure comes out upside down.
        Makie.image!(scene, (0, W), (0, H), img; interpolate = false, fxaa = false)
        live = nothing
        Makie.on(state; update = true) do s
            spec = get(s, :spec, nothing)
            spec isa SceneSpec || return
            # A BAKED frame wins: it is the same scene at final quality, already
            # rendered. Falls through to live drawing when the bake does not cover
            # this frame or was taken at another canvas size.
            ov = get(s, :overlay, nothing)
            if ov !== nothing
                b = bakedframe(ov, get(s, :frame, 0), (W, H))
                b === nothing || (img[] = b; return)
            end
            # `animatedspec` writes this frame's paths into a COPY, so a curve
            # that has ended cannot leave its last value in the stored spec.
            anim = animatedspec(spec, s)
            live = livescene!(live, anim, (W, H), getbackend(anim.backend))
            img[] = liveframe!(live, anim)
            return
        end
        return
    end)
