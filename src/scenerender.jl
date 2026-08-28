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

# Lights are a `Scene` keyword, so they are in the spec like everything else:
# `S.Scene(; lights = [AmbientLight(...), PointLight(...)])`. There used to be a
# `LightSpec` here and a converter for it, which is the same parallel description
# the parts had — and it could only express the two light types it knew.

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

An offscreen screen of `backend`, configured from `theme` — WITHOUT touching
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
function backendscreen(backend::Module, canvas::NTuple{2, Integer};
                       theme::Dict{Symbol, Any} = Dict{Symbol, Any}())
    opts = Dict{Symbol, Any}(:visible => false, :px_per_unit => 1.0, :scalefactor => 1.0)
    for (k, v) in get(theme, nameof(backend), Dict{Symbol, Any}())
        opts[Symbol(k)] = v
    end
    # a backend that does not know one of these must not die of it
    fields = fieldnames(backend.ScreenConfig)
    filter!(kv -> kv[1] in fields || kv[1] === :visible, opts)
    # NO RENDERLOOP BEHIND THIS SCREEN. A GLMakie screen starts one by default: an
    # `@async` task — sticky to the thread that made it, which is thread 1 — that
    # loops forever holding `with_context(screen.glscreen)` across its `sleep`.
    # This screen is never shown and never draws by itself; `readfilm` renders it,
    # through `colorbuffer`, which calls `render_frame` itself.
    #
    # Left running, that loop is a THIRD task on thread 1 fighting for the one
    # current GL context, next to the editor's own renderloop and the scene render
    # marshalled over by `onmainthread`. Measured on the lego project: playback of
    # a 60 fps timeline with a scene over it ran at 11.2 / 11.6 / 11.6 fps across
    # three passes, with `dropped == 0` the whole time — the playhead keeps time
    # and the picture does not, which is what "janky" is.
    #
    # `hasmethod(…, (:start_renderloop,))` rather than passing it blind: it is
    # GLMakie's keyword, not every renderer's, and the same shape as `readfilm`
    # asking whether a screen can accumulate.
    hasmethod(backend.Screen, Tuple{}, (:start_renderloop,)) &&
        return backend.Screen(; start_renderloop = false, opts...)
    return backend.Screen(; opts...)
end

"""
    upright(buf) -> view

A `GLNative` framebuffer with its rows the way a picture has them.

OpenGL numbers scanlines from the BOTTOM, and `Makie.GLNative` is the raw buffer:
`(width, height)`, y increasing upwards. `Makie.JuliaNative` is the same buffer
flipped AND transposed — the transpose is what this path does not want, because
the editor works in `(width, height)` throughout. Taking `GLNative` for the axis
order inherits the flip with it.

The whole scene came out UPSIDE DOWN and nothing said so: the lego figure hung
head-down over the birdhouse, and a text preset read "OBEN" as "OBEИ" — O, B and
E survive a top-to-bottom mirror almost unchanged, which is why it takes an N to
see it. No test caught it either, because a flip preserves every pixel COUNT: the
scene still drew on 9238 pixels, the same number as before the rebuild.

The depth buffer `coverage` reads is in the same orientation, so it is flipped
here too — the two have to agree pixel for pixel.
"""
upright(buf::AbstractMatrix) = view(buf, :, size(buf, 2):-1:1)

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

"""
The depth of what was LAST drawn, read without drawing it again.

`GLMakie.depthbuffer` opens with `render_frame` + `glFinish`: a full second
redraw of the scene [`readfilm`](@ref) has just drawn, and a blocking wait for
it — on thread 1, where the editor's own renderloop lives, so nothing else moves
meanwhile. Two renders and two stalls per scene frame.

Measured on the lego project: playback of that timeline ran at 12.1 fps, the same
timeline with the scene clip deleted at 32.6 (the renderloop's own 30 Hz ceiling),
and the scene ALONE at 11.1 — so the scene cost a factor of three all by itself.

Everything below is `depthbuffer`'s own tail, from after the render. Reading the
texture is what was wanted; drawing it twice was not. It has to run AFTER
`readfilm`, which is what fills the buffer — `liveframe!` does them in that order.

…and UPRIGHT, like the film it masks: same bottom-up framebuffer, see
[`upright`](@ref).
"""
function coverage(screen::GLMakie.Screen)
    GL = GLMakie.GLAbstraction
    src = GLMakie.get_depth_buffer(GLMakie.display_framebuffer(screen))
    depth = Matrix{Float32}(undef, size(src))
    GL.bind(src)
    GL.glGetTexImage(src.texturetype, 0, GLMakie.ModernGL.GL_DEPTH_COMPONENT,
                     GLMakie.ModernGL.GL_FLOAT, depth)
    GL.bind(src, 0)
    return upright(depth) .< 1.0f0
end

"""
    withalpha(img, cov) -> Matrix{PlanePixel}

`img` as a plane: RGBA, PREMULTIPLIED, uncovered wherever `cov` says nothing was
drawn.

Premultiplied is the plane convention (see `PlanePixel`), and with a boolean
coverage it costs nothing to honour: an uncovered pixel is zero in every channel.
Leaving the colour standing under a zero alpha is what "straight" alpha means, and
a straight plane read by a compositor that assumes premultiplied puts the
undrawn background back on the screen — measured: 99.99% of a scene's pixels came
out non-black over the footage, because dropping the alpha left the colour.

`cov === nothing` means the renderer could not tell us, and the honest result is
then fully covered — a scene that hides the frame is wrong, but inventing a mask
from colour would key out the figure's own white parts, which is worse and much
harder to see.
"""
withalpha(img::AbstractMatrix, ::Nothing) = PlanePixel.(img)

function withalpha(img::AbstractMatrix, cov::AbstractMatrix{Bool})
    out = Matrix{PlanePixel}(undef, size(img))
    @inbounds for i in eachindex(img, cov)
        c = img[i]
        out[i] = cov[i] ? PlanePixel(red(c), green(c), blue(c), 1) : zero(PlanePixel)
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

# There is no `renderspec` and no `scenepaths` here.
#
# `renderspec` built a scene, drew one frame and closed it — a whole second render
# path that existed because a bake had nowhere else to go. A scene is a clip's
# SOURCE now, so one frame of it is `sceneframe!` like every other frame, holding
# the same screen.
#
# `scenepaths` listed what was animatable by walking our description of the scene.
# `sceneattributes` walks the scene.

# ---------------------------------------------------------------- in the editor

# A scene is a CLIP's source, and this is what stands open for it.
#
# The block that used to be here described the overlay: a spec riding in
# `settings`, `overlaystate` merging per-frame paths over it, `setoverlaykey!`
# swinging a limb, `renderspec` drawing one frame. None of that exists — the
# paragraph above says so — and it stood here in the present tense anyway,
# describing the design the refactor replaced as though it were the code below.
# A comment that survives what it documents is worse than no comment: it is a
# false trail that reads as authoritative.
#
# What is true now: `SceneSource` holds a `LiveScene` across frames, the animated
# numbers are `Param`s on the clip's `:scene` entry keyed by SOURCE frame like
# every other clip, and `sceneframe!` writes them onto the standing plots. The
# backend is whatever the source names — `:GLMakie` while scrubbing, another for
# the bake, and switching is a rebuild because a screen takes its settings at
# construction.
"""
A scene held open across frames: what was built, and what it was built from.

WHY IT IS HELD. The first version rebuilt everything inside the per-frame
callback — a new `Scene`, a new `Screen`, and every mesh re-read from disk, for
every frame of a seek AND of playback. Measured after fixing it, at 480x854 with a
ten-part figure: first frame 0.28 s, every frame after 5.1-5.8 ms on GLMakie.
Almost none of the original cost was rendering.

Holding the screen is also what lets a PROGRESSIVE backend work at all: RayMakie
accumulates samples while the playhead stands still, and a screen thrown away
after each frame has nothing to accumulate into.

There is no spec diffing here and no plot list to keep in step. The structure of a
clip's scene does not change over its life — `visible` is the only structural
control it has — so the scene is realized once and after that only VALUES move.
"""
mutable struct LiveScene
    scene::Any                     # what gets displayed
    screen::Any
    target::Any                    # what the plots went into: the scene, or a block
    canvas::NTuple{2, Int}
    backend::Symbol
    root::Any                      # the spec it was built from
    theme::Dict{Symbol, Any}       # …and the settings its screen was opened with
end

"""
    realize(root, canvas) -> (scene, target)

Build what gets displayed and what the plots went into, from a Makie spec.

Two methods, dispatched on what the spec IS. A `SceneSpec` is a raw scene — no
figure, no layout, and its `camera` keyword is a `Scene` keyword that the scene
calls on itself. Anything else is a layout tree and needs a `Figure` to live in,
and then the block brings its own camera. Neither is a special case of the other
and neither is a branch.
"""
function realize(root::Makie.SceneSpec, canvas::NTuple{2, Int})
    scene = Makie.Scene(root; size = canvas, backgroundcolor = RGBAf(0, 0, 0, 0))
    return (scene, scene)
end

function realize(root, canvas::NTuple{2, Int})
    fig = Makie.Figure(; size = canvas, backgroundcolor = RGBAf(0, 0, 0, 0))
    Makie.plot!(fig, root isa Makie.GridLayoutSpec ? root :
                     Makie.SpecApi.GridLayout([root]))
    return (fig.scene, fig)
end

"""
    targetscene(target) -> Scene

The `Scene` behind whatever the plots went into — a block has one, a figure has
one, a scene IS one. What `cameracontrols` and `findplot` are asked of, so a value
written onto a plot works the same in all three.
"""
targetscene(s::Makie.Scene) = s
targetscene(f::Makie.Figure) = f.scene
targetscene(block) = block.scene

"""
    livescene!(live, root, canvas, backendname, backend) -> LiveScene

Build the scene, or hand back the one already built for this canvas, backend and
root spec. Nothing else forces a rebuild, because nothing else can change: a
clip's scene is realized once and animated by writing values onto it.
"""
function livescene!(live::Union{Nothing, LiveScene}, root, canvas::NTuple{2, Integer},
                    backendname::Symbol, backend::Module;
                    theme::Dict{Symbol, Any} = Dict{Symbol, Any}())
    W, H = Int(canvas[1]), Int(canvas[2])
    # The THEME is in it: a screen takes its settings at construction, so drawing
    # the same scene at another sample count is a different screen. That is what
    # makes switching between the live and the bake settings a rebuild and not a
    # value written onto something already open.
    if live !== nothing && live.canvas == (W, H) && live.backend === backendname &&
       live.root === root && live.theme == theme
        return live
    end
    live === nothing || (applicable(close, live.screen) && close(live.screen))
    scene, target = realize(root, (W, H))
    screen = backendscreen(backend, (W, H); theme = theme)
    display(screen, scene)
    return LiveScene(scene, screen, target, (W, H), backendname, root, theme)
end

"""
    readfilm(screen, clear) -> image

Read the screen's picture, ACCUMULATING into the film it already has when
`clear = false` and the renderer can do that.

A path tracer's frame is a running average of samples: asking it for the picture
without clearing adds more samples to what is there, which is what makes a live
preview converge while the playhead stands still instead of costing its full
budget on every frame. A rasteriser has nothing to accumulate and ignores it.

Asked of the SCREEN rather than branched on a backend name — and asked with
`hasmethod`, not a dependency, because the renderer is registered at runtime
(`usebackend!`) and this file must not know which ones exist.
"""
function readfilm(screen, clear::Bool)
    if hasmethod(Makie.colorbuffer, Tuple{typeof(screen), typeof(Makie.GLNative)}, (:clear,))
        return upright(Makie.colorbuffer(screen, Makie.GLNative; clear = clear))
    end
    return upright(Makie.colorbuffer(screen, Makie.GLNative))
end


"Whether this screen accumulates samples across reads — see [`readfilm`](@ref)."
progressive(screen) =
    hasmethod(Makie.colorbuffer, Tuple{typeof(screen), typeof(Makie.GLNative)}, (:clear,))


"""
    onmainthread(f) -> f()

Run `f` on thread 1, from wherever this is called.

A scene clip is the one source that draws with the UI's own toolkit: GLMakie's
screen belongs to thread 1 and asserts it (`ThreadAssertionError: Code must run
on thread 1`), while the composite runs on whichever thread owns the render
engine's Lava context — the pinned GPU worker in the usual case. So the two
owners meet here, at the one operation that has both.

Already on thread 1: straight through, no channel and no scheduler round trip —
which is the export path, the bake and every headless test.

Safe to call from the worker while thread 1 waits for it: the wait is a `take!`,
which yields, so a task pinned to thread 1 runs while the caller is blocked.
"""
function onmainthread(f::Function)
    Threads.threadid() == 1 && return f()
    done = Channel{Any}(1)
    t = Task() do
        try
            put!(done, (true, f()))
        catch e
            put!(done, (false, e))     # …and rethrown at the caller, below
        end
    end
    t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, 0)   # 0-based: thread 1
    schedule(t)
    # WITH A DEADLINE, and that is not belt-and-braces. A bare `take!` here waits
    # for a task pinned to thread 1 to be scheduled, and thread 1 is also where
    # GLMakie's renderloop lives — holding `with_context` across its own `sleep`.
    # If it never yields at a moment this task can take, the wait never ends and
    # the editor is simply frozen, with no error and nothing on screen to say why.
    # Reported live: "played a few janky frames and then immediately froze… then
    # it unfroze… and now it is frozen forever". A freeze teaches nothing; a
    # message names the thread and the caller.
    t0 = time()
    while !isready(done)
        time() - t0 > SCENEWAIT &&
            error("a scene render waited $(SCENEWAIT)s to reach thread 1 and gave up. " *
                  "Called from thread $(Threads.threadid()); thread 1 is where GLMakie's " *
                  "screen and the editor's renderloop both live. Renders belong BEFORE " *
                  "the graph runs (`prerender!`), where the caller is already on thread 1.")
        sleep(0.001)
    end
    ok, val = take!(done)
    ok || throw(val)
    return val
end

"""
How long [`onmainthread`](@ref) waits to be let onto thread 1 before giving up.

Generous — a cold scene builds its screen and loads its meshes on the first call —
but FINITE, because the alternative is a frozen editor.
"""
const SCENEWAIT = 10.0

"""
    liveframe!(live; clear = true) -> Matrix{PlanePixel}

Read the picture off the standing scene.

There is nothing to write here: this frame's values were written onto the plots
before the render (see `applysceneparams!`), which is where they belong — a plot
attribute is set directly, not re-specified. The Makie docs are explicit that
specs are the slower path for animation ("it needs to re-create plots often and
needs to go over the whole plot tree"), and between frames a scene only ever
changes values.
"""
function liveframe!(live::LiveScene; clear::Bool = true)
    img = readfilm(live.screen, clear)
    return withalpha(img, coverage(live.screen))
end

# A scene is rendered by `SceneSource`'s pass (scenesource.jl), which holds the
# `LiveScene` above across frames and writes this frame's numbers onto it. There
# is no overlay registration here any more: a scene is a CLIP.
