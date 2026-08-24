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
    scenefromspec(spec, canvas) -> Makie.Scene

The `SceneSpec` as a live Makie scene at `canvas = (width, height)`.

`plotlist!` is Makie's own way in for `PlotSpec`s (the `PlotList` recipe in
specapi.jl), and it is the reason no plot-type table is needed here: a project
file names a plot `"Mesh"` and Makie resolves it. Building each one by hand
instead needs `plot!(scene, PlotType, Attributes, args...)`, whose argument order
is easy to get wrong — done wrong it passes the type and the attributes as DATA,
and the conversion then fails on a tuple that looks identical to the one it
wanted.
"""
function scenefromspec(spec::SceneSpec, canvas::NTuple{2, Integer})
    W, H = Int(canvas[1]), Int(canvas[2])
    scene = Makie.Scene(; size = (W, H), backgroundcolor = RGBAf(0, 0, 0, 0))
    Makie.cam3d!(scene)
    Makie.plotlist!(scene, spec.plots)
    return scene
end

"""
    updatespec!(plots, spec)

Write `spec`'s attribute values into already-built `plots`, in order.

THE ANIMATION PATH, and deliberately not a spec diff: a frame only ever changes
VALUES, and `update!` on a plot is a store into its observables. Diffing the
whole spec every frame would re-derive that nothing structural changed, sixty
times a second, to arrive at the same stores. A spec diff is the right tool when
the set of plots changes — which is an EDIT, not a frame.
"""
function updatespec!(plots::AbstractVector, spec::SceneSpec)
    for (plot, p) in zip(plots, spec.plots)
        isempty(p.kwargs) && continue
        Makie.update!(plot; p.kwargs...)
    end
    return plots
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
    getbackend(name) -> Module

The registered renderer called `name`, or an error naming what is registered.
"""
getbackend(name::Symbol) = get(BACKENDS, name) do
    error("scene backend $(repr(name)) is not registered — `VideoEditor.usebackend!($name)` " *
          "after loading it. Have: $(join(sort!(collect(keys(BACKENDS))), ", "))")
end

"""
    renderspec(spec, canvas) -> Matrix{RGB{N0f8}}

`spec` rendered at `canvas`, by the backend it names, under the theme it carries.

The theme is applied around the whole build: Makie reads a backend's screen
options from it, so `theme[:RayMakie][:exposure]` reaches the renderer without
this function knowing what an exposure is.
"""
function renderspec(spec::SceneSpec, canvas::NTuple{2, Integer},
                    backend::Module = getbackend(spec.backend))
    theme = merge(themeattributes(spec.theme), PIXELEXACT)   # the spec wins
    return Makie.with_theme(theme) do
        scene = scenefromspec(spec, canvas)
        screen = display(scene; backend = backend, visible = false)
        img = Makie.colorbuffer(screen)
        # GLMakie hands back a screen that is reused; RayMakie's is per-display.
        # Closing what we opened is the caller-neutral thing to do either way.
        applicable(close, screen) && close(screen)
        return RGB{N0f8}.(img)
    end
end
