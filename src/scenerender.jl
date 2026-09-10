# Rendering a `SceneSpec`: one path, with the backend as a parameter.
#
# There is no per-backend method here and no extension package, because there is
# nothing backend-specific to say. Makie already takes the renderer as a value
# (`display(scene; backend = SomeModule)`) and already reads a backend's own
# options straight to the screen — `ScreenConfig`'s fields for RayMakie are exactly
# its settings (integrator, exposure, tonemap, gamma, device, …). So a scene that
# carries a backend name and those settings carries everything the renderer needs,
# and this file just hands both over.
#
# That is why `SceneSpec.backend` is a Symbol: a project file has to name the
# renderer in text, and the module is looked up at render time from what is
# loaded. Loading RayMakie is what makes `backend = :RayMakie` work; nothing here
# has to know it exists.
#
# Why not `Makie.colorbuffer(scene)`: that goes through GLMakie's singleton
# offscreen screen, and every scene-taking `Screen` constructor empties it and
# re-displays itself on it. The editor's own window comes from the same
# constructor, so rendering a scene that way evicts the editor's figure from its
# own window — measured, the renderlist went from 32 plots to 3. See
# `canvasscreen` in overlays.jl, which learned this the hard way.

"""
The renderers a scene may name, by name.

A project file names its backend in text, so something has to turn `:RayMakie`
into a module. This is a plain table a module is put into (`usebackend!(RayMakie)`)
rather than a search over whatever happens to be loaded: nothing renders through a
backend nobody asked for, and a missing one is named in the error instead of
failing inside Makie.

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
came back (261, 348). A scene that wants something else says so and wins the
merge below.
"""
const PIXELEXACT = Makie.Theme(GLMakie = (px_per_unit = 1.0, scalefactor = 1.0))

"""
    coerceopt(backend, key, value) -> value

A screen setting as the backend's `ScreenConfig` field wants it.

A dialog text box and a project file both hold text, so a STRING is read against
the field's declared type: a field that takes a `Symbol` gets one (`tonemap`),
one that takes a number gets it parsed, and anything else is a Julia expression
evaluated in the backend's own module — `integrator = "VolPath(samples=256)"`,
`device = "CPU()"`. The expression is the only way a text box can carry an
object, and it is what makes the dialogs reach everything `activate!` accepts
rather than the numbers-and-flags subset. It is also why a backend registered
with [`usebackend!`](@ref) must re-export what its settings name (RayMakie
re-exports `VolPath` for exactly this).
"""
function coerceopt(backend::Module, key::Symbol, v)
    v isa AbstractString || return v
    s = strip(String(v))
    isempty(s) && error("screen setting $key is an empty string — leave it out instead")
    T = fieldtype(backend.ScreenConfig, key)
    T === Any && return Base.eval(backend, Meta.parse(s))
    types = T isa Union ? Base.uniontypes(T) : (T,)
    String <: T && return s
    s == "nothing" && Nothing <: T && return nothing
    any(t -> t === Symbol, types) && return Symbol(s)
    any(t -> t <: Real, types) && return tryparse(Float64, s)
    return Base.eval(backend, Meta.parse(s))
end

"""
    backendscreen(backend, spec, canvas) -> screen

An offscreen screen of `backend`, configured from `opts` — the keywords that
backend's `ScreenConfig` declares — without touching the global theme. Values
that arrived as text (the dialog, a project file) are coerced against the
field's type, see [`coerceopt`](@ref).

Building under `Makie.with_theme(...)` — the usual way to pass a backend's screen
options — is not safe here: this runs from the overlay's draw callback, during a
seek and during playback, while GLMakie's render loop draws the editor
concurrently. `with_theme` restores by calling `set_theme!()` first, which resets
to Makie's built-in light default for an instant, and a render landing in that
window redraws the editor with it:
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

`scene` is there for backends whose screen cannot be built without one:
RayMakie offers `Screen(scene; screen_config…)` only. Where a scene-LESS
constructor exists it is used, because GLMakie's scene-taking one goes through
`singleton_screen` — the one window the editor's own figure is displayed on.
Building a scene's screen that way empties the editor's window, and closing the
scene's screen then closes the editor with it (measured: preview switched to
RayMakie and back, the window was gone).
"""
function backendscreen(backend::Module, canvas::NTuple{2, Integer};
                       opts::Dict{Symbol, Any} = Dict{Symbol, Any}(), scene)
    o = Dict{Symbol, Any}(:visible => false, :px_per_unit => 1.0, :scalefactor => 1.0)
    # A backend that does not know a key must not die of it — the bake dialog's
    # overrides are merged over the preview's settings, and the two renderers
    # rarely name the same ones.
    fields = fieldnames(backend.ScreenConfig)
    for (k, v) in opts
        k = Symbol(k)
        (k in fields || k === :visible) || continue
        o[k] = k in fields ? coerceopt(backend, k, v) : v
    end
    opts = o
    # No renderloop behind this screen. A GLMakie screen starts one by default: an
    # `@async` task — sticky to the thread that made it, which is thread 1 — that
    # loops forever holding `with_context(screen.glscreen)` across its `sleep`.
    # This screen is never shown and never draws by itself; `readfilm` renders it,
    # through `colorbuffer`, which calls `render_frame` itself.
    #
    # Left running, that loop is a third task on thread 1 fighting for the one
    # current GL context, next to the editor's own renderloop and the scene render
    # marshalled over by [`onthread`](@ref). Measured on the lego project: playback of
    # a 60 fps timeline with a scene over it ran at 11.2 / 11.6 / 11.6 fps across
    # three passes, with `dropped == 0` the whole time — the playhead keeps time
    # and the picture does not, which is what "janky" is.
    #
    # `Base.kwarg_decl` rather than `hasmethod(…, (:start_renderloop,))`: a
    # constructor ending in `screen_config...` claims EVERY keyword name to
    # `hasmethod`, and RayMakie's would then swallow `start_renderloop` into its
    # config. The question is whether the constructor declares it, not whether
    # it accepts anything.
    declaresstartrenderloop(types) =
        any(m -> :start_renderloop in Base.kwarg_decl(m), methods(backend.Screen, types))
    if hasmethod(backend.Screen, Tuple{})
        declaresstartrenderloop(Tuple{}) &&
            return backend.Screen(; start_renderloop = false, opts...)
        return backend.Screen(; opts...)
    end
    declaresstartrenderloop(Tuple{Makie.Scene}) &&
        return backend.Screen(scene; start_renderloop = false, opts...)
    return backend.Screen(scene; opts...)
end

"""
    upright(buf) -> view

A `GLNative` framebuffer with its rows the way a picture has them.

OpenGL numbers scanlines from the bottom and `Makie.GLNative` is the raw buffer:
`(width, height)`, y increasing upwards. `Makie.JuliaNative` is the same buffer
flipped and transposed, and the transpose is what this path does not want, since
the editor works in `(width, height)` throughout. Taking `GLNative` for the axis
order inherits the flip with it.

Without the flip the whole scene renders upside down: the lego figure hung
head-down over the birdhouse, and a text preset read "OBEN" as "OBEИ". No test
caught it, because a flip preserves every pixel count — the scene still drew on
9238 pixels.

The depth buffer `coverage` reads is in the same orientation, so it is flipped
here too — the two have to agree pixel for pixel.
"""
upright(buf::AbstractMatrix) = view(buf, :, size(buf, 2):-1:1)

"""
    coverage(screen) -> Union{Nothing, AbstractMatrix{Bool}}

Which pixels the scene actually drew on — `nothing` when the renderer cannot say.

An overlay has to be transparent where the scene is empty, or it replaces the
picture instead of overlaying it: the lego scene over footage changed all 57600
pixels of a 320x180 frame and the video was gone.

GLMakie cannot answer this from colour. `colorbuffer` stages through a
`Matrix{RGB{N0f8}}` framecache, so alpha is dropped before the caller sees it,
and reading the framebuffer's own colour texture gives alpha 0 on every pixel,
the figure's included. Rendering twice on two background colours does not work
either — the scene's `backgroundcolor` makes no difference to what comes back
(measured: black vs white differ in exactly 0 pixels).

The depth buffer does answer it: whatever the far plane still owns was never
drawn on. Measured against a colour-threshold mask of the same frame the two
agree on 99.6% of pixels, and where they differ it is the antialiased fringe,
which depth correctly leaves out.

Dispatch rather than a branch on the backend name: a renderer whose picture
cannot answer it from colour adds its own method, and the render path above does
not change. A renderer whose picture carries REAL alpha — a raytracer knows a
ray that hit nothing — adds nothing at all: `nothing` here lets that alpha
through to the plane, which is the coverage already.
"""
coverage(::Any) = nothing

"""
The depth of what was drawn last, read without drawing it again.

`GLMakie.depthbuffer` opens with `render_frame` + `glFinish`: a full second
redraw of the scene [`readfilm`](@ref) has just drawn, and a blocking wait for
it — on thread 1, where the editor's own renderloop lives, so nothing else moves
meanwhile. Two renders and two stalls per scene frame.

Measured on the lego project: playback of that timeline ran at 12.1 fps, the same
timeline with the scene clip deleted at 32.6 (the renderloop's own 30 Hz ceiling),
and the scene alone at 11.1, so the scene cost a factor of three by itself.

Everything below is `depthbuffer`'s own tail, from after the render: reading the
texture without drawing it again. It has to run after `readfilm`, which fills the
buffer — `liveframe!` does them in that order.

Upright like the film it masks: same bottom-up framebuffer, see [`upright`](@ref).
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
harder to see. An image that carries its own alpha (a raytracer's, where a miss
is transparent) simply keeps it: the conversion passes the channels through.
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
# source now, so one frame of it is `sceneframe!` like every other frame, holding
# the same screen.
#
# `scenepaths` listed what was animatable by walking our description of the scene.
# `sceneattributes` walks the scene.

# ---------------------------------------------------------------- in the editor

# A scene is a clip's source, and this is what stands open for it.
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
# numbers are `Param`s on the clip's `:scene` entry keyed by source frame like
# every other clip, and `sceneframe!` writes them onto the standing plots. The
# backend is whatever the source names — `:GLMakie` while scrubbing, another for
# the bake, and switching is a rebuild because a screen takes its settings at
# construction.
"""
A scene held open across frames: what was built, and what it was built from.

Held across frames because the first version rebuilt everything inside the
per-frame callback: a new `Scene`, a new `Screen` and every mesh re-read from
disk, for every frame of a seek and of playback. Measured after fixing it, at
480x854 with a ten-part figure: first frame 0.28 s, every frame after 5.1-5.8 ms
on GLMakie. Almost none of the original cost was rendering.

Holding the screen is also what lets a progressive backend work at all: RayMakie
accumulates samples while the playhead stands still, and a screen thrown away
after each frame has nothing to accumulate into.

There is no spec diffing and no plot list to keep in step: the structure of a
clip's scene does not change over its life (`visible` is its only structural
control), so the scene is realized once and after that only values move.
"""
mutable struct LiveScene
    scene::Any                     # what gets displayed
    screen::Any
    target::Any                    # what the plots went into: the scene, or a block
    canvas::NTuple{2, Int}
    backend::Symbol
    root::Any                      # the spec it was built from
    opts::Dict{Symbol, Any}        # …and the settings its screen was opened with
    # Screens a backend switch replaced, waiting to be closed between frames —
    # see [`closeretired!`](@ref). `(backendname, screen)`, because which thread
    # may close a screen is a property of the renderer that made it.
    retired::Vector{Any}
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
                    opts::Dict{Symbol, Any} = Dict{Symbol, Any}())
    W, H = Int(canvas[1]), Int(canvas[2])
    # The settings are part of it: a screen takes them at construction, so drawing
    # the same scene at another sample count is a different screen. That is what
    # makes switching between the live and the bake settings a rebuild and not a
    # value written onto something already open.
    if live !== nothing && live.canvas == (W, H) && live.backend === backendname &&
       live.root === root && live.opts == opts
        return live
    end
    # The old screen is RETIRED, not closed here — see [`closeretired!`](@ref).
    # This runs inside the frame the caller is producing, and closing a GLMakie
    # screen destroys GL objects in a context the editor's own screen shares.
    retired = live === nothing ? Any[] : Any[(live.backend, live.screen)]
    scene, target = realize(root, (W, H))
    screen = backendscreen(backend, (W, H); opts = opts, scene)
    display(screen, scene)
    return LiveScene(scene, screen, target, (W, H), backendname, root, opts, retired)
end

"""
    closeretired!(live) -> nothing

Close the screens a backend switch left behind, between frames.

Switching a scene's renderer builds the new screen and hands the old one to
`live.retired` instead of closing it on the spot. Closing it on the spot means
destroying GL objects from inside the frame the caller is in the middle of
producing, and a scene's GLMakie screen shares its context with the editor's own:
measured, switching the preview renderer with the rendering dialog open, while a
frame was being read back, drew the preview as blocks of unrelated memory — and a
second run at the same moment span at 134 % CPU and never returned. Neither
reproduced with the switch alone, the dialog alone, or the readback alone.

Called from [`prerender!`](@ref), which is the defined point between frames: the
caller is on thread 1 and nothing is mid-render.

Each screen closes on the thread ITS renderer owns, not the one this frame renders
on — a GLMakie screen torn down on the GPU worker dies with
`ThreadAssertionError: Code must run on thread 1`.
"""
function closeretired!(live::LiveScene)
    isempty(live.retired) && return nothing
    for (backendname, screen) in live.retired
        onthread(renderthread(getbackend(backendname))) do
            applicable(close, screen) && close(screen)
        end
    end
    empty!(live.retired)
    return nothing
end
closeretired!(::Nothing) = nothing

"""
    readfilm(screen, clear; samples = nothing) -> image

Read the screen's picture, accumulating into the film it already has when
`clear = false` and the renderer can do that.

`samples` is how many samples this read renders: 1 for the live preview — a
playhead move must cost one sample, and standing still adds one at a time — and
`nothing` for finished output (the bake, the export), which renders the
integrator's full configured budget in one go. A renderer that takes `clear`
but not `samples` is asked without it; one that takes neither renders as it
always did.

A path tracer's frame is a running average of samples: asking it for the picture
without clearing adds more samples to what is there, which is what makes a live
preview converge while the playhead stands still instead of costing its full
budget on every frame. A rasteriser has nothing to accumulate and ignores both.

Asked of the screen rather than branched on a backend name — and asked with
`hasmethod`, not a dependency, because the renderer is registered at runtime
(`usebackend!`) and this file must not know which ones exist.
"""
function readfilm(screen, clear::Bool; samples::Union{Nothing, Integer} = nothing)
    T = Tuple{typeof(screen), typeof(Makie.GLNative)}
    if samples !== nothing && hasmethod(Makie.colorbuffer, T, (:clear, :samples))
        return upright(Makie.colorbuffer(screen, Makie.GLNative;
                                         clear = clear, samples = Int(samples)))
    end
    if hasmethod(Makie.colorbuffer, T, (:clear,))
        return upright(Makie.colorbuffer(screen, Makie.GLNative; clear = clear))
    end
    return upright(Makie.colorbuffer(screen, Makie.GLNative))
end


"Whether this screen accumulates samples across reads — see [`readfilm`](@ref)."
progressive(screen) =
    hasmethod(Makie.colorbuffer, Tuple{typeof(screen), typeof(Makie.GLNative)}, (:clear,))


"""
    onthread(f, tid; startwithin = 10.0) -> f()

Run `f` pinned to (0-based) thread `tid`, from wherever this is called.

A scene clip is the one source that draws with resources that belong to a
thread: GLMakie's screen belongs to thread 1 and asserts it
(`ThreadAssertionError: Code must run on thread 1`), a Lava-backed renderer's
Vulkan context belongs to the pinned GPU worker's thread (a `BatchQueue` is
single-writer) — and the composite runs on whichever thread owns the render
engine, so caller and owner routinely differ. They meet here.

Already on the right thread: straight through, no channel and no scheduler round
trip — the export path, the bake and every headless test.

Safe to call while the owning thread waits for the caller: the wait is a
`take!`, which yields, so a pinned task runs while its waiter is blocked.

`startwithin` bounds how long the render may take to START. Nothing bounds how
long it may then run — see the comment on the wait.
"""
function onthread(f::Function, tid::Int; startwithin::Real = 10.0)
    Threads.threadid() == tid + 1 && return f()
    done = Channel{Any}(1)
    started = Threads.Atomic{Bool}(false)
    t = Task() do
        started[] = true               # …which is what the deadline below watches
        try
            put!(done, (true, f()))
        catch e
            put!(done, (false, e))     # …and rethrown at the caller, below
        end
    end
    t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid)
    schedule(t)
    # The deadline covers STARTING, not running. The failure it exists for is the
    # owning thread never yielding — it can be holding a renderloop's
    # `with_context` across its own `sleep`, and then a pinned task scheduled onto
    # it never runs, the wait never ends, and the editor freezes with nothing on
    # screen to say why.
    #
    # Once the render is running there is no deadline at all, because a slow
    # render is not that failure and never was: the first render of a scene builds
    # its screen, compiles the renderer's shaders and loads its meshes, which took
    # well over ten seconds on RayMakie's first frame. Timing that out aborted a
    # render that was working, and the retry after it succeeded — which is what
    # made the error look both alarming and harmless.
    t0 = time()
    while !isready(done)
        (started[] || time() - t0 <= startwithin) ||
            error("a scene render never started on thread $(tid + 1), which owns " *
                  "the renderer's resources; that thread has not yielded in " *
                  "$(startwithin)s. Called from thread $(Threads.threadid()). Renders " *
                  "belong before the graph runs (`prerender!`), where the caller is " *
                  "not waiting on the composite that waits for this.")
        sleep(0.001)
    end
    ok, val = take!(done)
    ok || throw(val)
    return val
end

"Run `f` on thread 1, where GLMakie's screen and the editor's renderloop live."
onmainthread(f::Function) = onthread(f, 0)

"""
    onworkerthread(f) -> f()

Run `f` on the LAST thread — the one the GPU worker pins itself to
(`GPUWorker` in player.jl), and so the one Lava's Vulkan context belongs to.

Called whether or not the worker exists yet: a context belongs to the thread
that first touched it, and routing scene renders here either way keeps that
thread the same — if no worker exists, this render is the first touch, and a
worker created later pins itself here too.
"""
onworkerthread(f::Function) = onthread(f, Threads.nthreads() - 1)

"""
    renderthread(backend::Module) -> 0-based thread id

The thread this renderer's screens may be touched from — building the screen,
writing a frame's values onto its plots, reading the film.

A renderer that draws through Lava — its module binds it (`import Lava`) —
shares the one Vulkan context the GPU worker owns, so it renders on the worker's
thread; anything else (GLMakie) renders on thread 1. Asked of the module with
`isdefined`, not a name comparison: this file must not know which backends exist
(see [`usebackend!`](@ref)).
"""
renderthread(backend::Module) = isdefined(backend, :Lava) ? Threads.nthreads() - 1 : 0

"""
    sharesdevice(clip) -> Bool

Whether this clip draws through the same GPU device the preview's shared texture
lives on — i.e. a scene rendered by a Lava-backed renderer.

The GPU preview tier hands the composited canvas to GLMakie through a Vulkan image
imported as a GL texture. Interleaving a Lava-backed scene render with the blit
into that image leaves the image holding UNRELATED GPU MEMORY: measured by
switching a scene clip's preview to RayMakie with the rendering dialog open, the
preview drew blocks of noise while `player.frame[]` — the same composite, on the
CPU — was pixel-correct, and one `showcpuframe!` put the correct picture back
instantly. So it is neither the composite nor the upload; it is the shared image's
contents.

Asked of the renderer's module, like [`renderthread`](@ref), so this file still
does not know which backends exist. An unregistered name is not a Lava renderer as
far as this is concerned — it cannot render at all, and `sceneframe!` is where
that gets reported.
"""
function sharesdevice(clip::Clip)
    src = clip.source
    src isa SceneSource || return false
    name = renderwith(src)
    haskey(BACKENDS, name) || return false
    return isdefined(BACKENDS[name], :Lava)
end

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
function liveframe!(live::LiveScene; clear::Bool = true,
                    samples::Union{Nothing, Integer} = nothing)
    img = readfilm(live.screen, clear; samples)
    return withalpha(img, coverage(live.screen))
end

# A scene is rendered by `SceneSource`'s pass (scenesource.jl), which holds the
# `LiveScene` above across frames and writes this frame's numbers onto it. There
# is no overlay registration here any more: a scene is a clip.
