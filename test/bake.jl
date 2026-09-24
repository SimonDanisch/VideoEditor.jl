# Baking a clip (src/bake.jl): pre-rendering its chain to disk.
#
# Baking used to belong to the 3D scene because the scene was the only thing slow
# enough to need it. It is a property of a CLIP now, and the properties below are
# what that has to mean: the bake is READ instead of the graph running, an edit
# switches it off without deleting it, a changed input file is noticed, and
# exactly one version exists at a path the next open can find.

@testset "a bake is read instead of running the graph" begin
    build = VE.scenebuild(:bar, (64, 48); opacity = 1.0, height = 0.5)
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = (64, 48))
    engine = VE.FxEngine(VE.KA.CPU())
    dir = joinpath(mktempdir(), "b")

    b = VE.bakeclip!(clip, engine; frames = 2:6, dir = dir)
    @test b.frames == 2:6
    @test b.canvas == (64, 48)
    @test b.enabled && !b.dirty
    @test length(readdir(b.dir)) == 5
    @test VE.bakedframe(clip, 4) !== nothing
    @test VE.bakedframe(clip, 99) === nothing        # outside the range

    # THE PROOF that the bake is read and the graph is not: overwrite one baked
    # frame with something the scene would never draw, and look for it.
    VE.PNGFiles.save(VE.bakeframefile(b, 4), fill(VE.PlanePixel(1, 0, 0, 1), 64, 48))
    eng2 = VE.FxEngine(VE.KA.CPU())                  # fresh plans
    o = Ref{Any}(nothing)
    VE.render(eng2, clip.source, clip, 4) do x; o[] = copy(x); end
    @test count(==(VE.RGB{VE.N0f8}(1, 0, 0)), o[]) == length(o[])

    # …and with it off, the graph runs again
    b.enabled = false
    VE.render(eng2, clip.source, clip, 4) do x; o[] = copy(x); end
    @test count(==(VE.RGB{VE.N0f8}(1, 0, 0)), o[]) == 0
    VE.emptyengine!(engine); VE.emptyengine!(eng2)
end

@testset "an edit switches a bake off and deletes nothing" begin
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = (64, 48))
    engine = VE.FxEngine(VE.KA.CPU())
    b = VE.bakeclip!(clip, engine; frames = 0:4, dir = joinpath(mktempdir(), "b"))
    n = length(readdir(b.dir))

    VE.bakedirty!(clip)
    @test !b.enabled && b.dirty
    @test length(readdir(b.dir)) == n                # the frames are still there
    @test VE.bakestale(clip)
    b.enabled = true                                 # …and switching it back on works
    @test VE.bakedframe(clip, 2) !== nothing

    # a file the render READ changing on disk is the one thing an edit cannot say,
    # so it is the one thing that is checked rather than remembered
    b.dirty = false
    @test !VE.bakestale(clip)
    b.inputs["/nonexistent/mesh.stl"] = (0.0, 0)
    @test VE.bakestale(clip)
    VE.emptyengine!(engine)
end

@testset "a bake follows its project and keeps one version" begin
    src = VideoSource(testvideo)
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = (64, 48))
    clip.track = 2
    seq = Sequence([Clip(src; src_in = 0, src_out = 10, start = 0), clip], 30.0)
    engine = VE.FxEngine(VE.KA.CPU())

    # baked BEFORE the project has a path: it goes to a temp directory, and saving
    # is what brings it in — without that it was recorded as if it were beside the
    # file and gone by the next open
    VE.bakeclip!(clip, engine; frames = 0:4)
    proj = joinpath(mktempdir(), "p.videoedit")
    saveproject(proj, seq; checkpoint = false)
    @test clip.bake.dir == VE.bakeclipdir(proj, clip.id)
    @test length(readdir(clip.bake.dir)) == 5

    seq2 = loadproject(proj)
    c2 = seq2.clips[end]
    @test c2.bake !== nothing
    @test c2.bake.frames == 0:4 && c2.bake.canvas == (64, 48)
    @test c2.bake.dir == VE.bakeclipdir(proj, c2.id)
    @test c2.bake.enabled && !VE.bakestale(c2)
    @test VE.bakedframe(c2, 2) !== nothing

    # a second bake lands at the SAME canonical path — one version, and nothing
    # left staged behind
    old = c2.bake.dir
    b2 = VE.bakeclip!(c2, engine; frames = 0:2, dir = old)
    @test b2.dir == old
    @test length(readdir(b2.dir)) == 3
    @test !ispath(old * ".part")
    VE.emptyengine!(engine)
end

@testset "the canvas is the size the clip renders at" begin
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 6, canvas = (64, 48))
    engine = VE.FxEngine(VE.KA.CPU())
    b = VE.bakeclip!(clip, engine; frames = 0:2, canvas = (32, 24),
                     dir = joinpath(mktempdir(), "b"))
    # ONE size, not one per consumer: the bake cannot end up at a resolution the
    # preview will not use, because asking for a canvas resized the source
    @test (clip.source.width, clip.source.height) == (32, 24)
    @test b.canvas == (32, 24)
    @test size(VE.bakedframe(clip, 1)) == (32, 24)
    o = Ref{Any}(nothing)
    VE.render(engine, clip.source, clip, 1) do x; o[] = copy(x); end
    @test size(o[]) == (32, 24)

    # …and a decoder's size is not ours to change
    src = VideoSource(testvideo)
    vclip = Clip(src; src_in = 0, src_out = 4, start = 0)
    VE.resize!(vclip.source, (10, 10))
    @test (vclip.source.width, vclip.source.height) == (src.width, src.height)
    VE.emptyengine!(engine)
end

@testset "screen settings: what a form can offer" begin
    opts = VE.screenoptions(GLMakie)
    names = [o.name for o in opts]
    # everything offered is a real ScreenConfig field — `backendscreen` pins
    # `visible`/`px_per_unit`/`scalefactor` itself, and window-manager fields
    # have no business in an offscreen render
    @test all(n -> n in fieldnames(GLMakie.ScreenConfig), names)
    @test issorted(names)
    for skipped in (:visible, :px_per_unit, :scalefactor, :title, :renderloop,
                    :monitor, :fullscreen, :render_pipeline)
        @test skipped ∉ names
    end
    byname = Dict(o.name => o for o in opts)
    @test byname[:framerate].kind === :number
    @test byname[:vsync].kind === :flag
    # and every backend Makie knows a theme for answers — a renderer loaded
    # later brings its settings with it, there is no list to fall behind
    @test !isempty(opts)
end

# A stand-in renderer: `coerceopt` reads a string against the field's declared
# type, so the test backend's config has one field per kind the real ones have.
const FAKEBACKEND = Module(gensym(:FakeBackend))
Core.eval(FAKEBACKEND, quote
    struct ScreenConfig
        tonemap::Union{Nothing, Symbol}
        samples::Int64
        hook::Any
    end
    double(x) = 2x
end)

@testset "screen settings: text becomes what the field declares" begin
    fb = FAKEBACKEND
    @test VE.coerceopt(fb, :tonemap, "aces") === :aces
    @test VE.coerceopt(fb, :tonemap, :aces) === :aces      # already a value: through
    @test VE.coerceopt(fb, :tonemap, "nothing") === nothing
    @test VE.coerceopt(fb, :samples, "64") === 64.0
    @test VE.coerceopt(fb, :samples, 64) === 64
    # an object a text box cannot carry goes as an expression, evaluated in the
    # BACKEND's module — the names its settings use are the names it exports
    @test VE.coerceopt(fb, :hook, "double") === fb.double
    @test VE.coerceopt(fb, :hook, "double(21)") === 42
end

@testset "the bake's settings are the preview's, overridden" begin
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 6, canvas = (64, 48))
    src = clip.source
    src.screenopts[:framerate] = 30.0
    src.screenopts[:vsync] = false

    # live: the preview's own — a COPY, because the dialogs edit the dict in
    # place and `livescene!` compares settings to decide on a rebuild
    live = VE.renderopts(src)
    @test live == src.screenopts && live !== src.screenopts

    # bake: merged, the bake's winning — touching one bake setting must not
    # drop the preview's others, which is what the old either/or did
    src.mode = :bake
    @test VE.renderopts(src) == src.screenopts          # no overrides: same picture
    src.bakescreenopts[:framerate] = 60.0
    merged = VE.renderopts(src)
    @test merged[:framerate] == 60.0 && merged[:vsync] == false
    @test !haskey(src.screenopts, :integrator)
end

@testset "screen settings survive the project file" begin
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = (64, 48))
    src = clip.source
    src.bakewith = :GLMakie
    src.screenopts[:vsync] = false
    src.screenopts[:framerate] = 24.0
    src.bakescreenopts[:framerate] = 60.0
    src.bakescreenopts[:tonemap] = :aces                    # a Symbol
    src.bakescreenopts[:integrator] = "VolPath(samples=8)"  # an expression
    src.bakescreenopts[:denoise_config] = nothing           # an explicit nothing
    proj = joinpath(mktempdir(), "p.videoedit")
    saveproject(proj, Sequence([clip], 30.0); checkpoint = false)
    c2 = loadproject(proj).clips[1].source
    @test c2.bakewith === :GLMakie
    # numbers, flags and strings come back as themselves; a Symbol as its name
    # and a nothing as the word — `coerceopt` reads both against the field's
    # type at screen-build time
    @test c2.screenopts[:vsync] === false
    @test c2.screenopts[:framerate] == 24.0
    @test c2.bakescreenopts[:framerate] == 60.0
    @test c2.bakescreenopts[:tonemap] == "aces"
    @test c2.bakescreenopts[:integrator] == "VolPath(samples=8)"
    @test c2.bakescreenopts[:denoise_config] == "nothing"
end

# A screen that records how it was asked — the sample-budget contract has no
# real path tracer in the test environment, so the renderer is faked at exactly
# the seam `readfilm` asks through.
mutable struct FakeFilmScreen
    calls::Vector{NamedTuple}
end
function VE.Makie.colorbuffer(s::FakeFilmScreen, ::VE.Makie.ImageStorageFormat;
                              clear = true, samples = nothing)
    push!(s.calls, (; clear, samples))
    return fill(VE.RGB{VE.N0f8}(0.5), 4, 4)
end

@testset "one sample per preview read, the full budget for finished output" begin
    scr = FakeFilmScreen(NamedTuple[])
    VE.readfilm(scr, true; samples = 1)
    @test scr.calls == [(; clear = true, samples = 1)]     # a playhead move: one sample
    VE.readfilm(scr, false; samples = 1)
    @test scr.calls[end] == (; clear = false, samples = 1) # standing still: accumulate one
    VE.readfilm(scr, true)                                 # bake/export: the full budget
    @test scr.calls[end] == (; clear = true, samples = nothing)
end

@testset "the preview keeps refining, with no sample bound" begin
    # It used to stop at a constant 64 — a number that has to be right for every
    # scene, and is not. The bound that matters is what a FINISHED frame costs,
    # which is the renderer's own `samples` setting on the Bake tab; the preview
    # improves the picture you are looking at until you look somewhere else.
    spec = VE.Makie.SpecApi.Scene(; camera = VE.Makie.campixel!)
    clip = VE.sceneclip((root = spec, joints = Dict{Symbol, Any}(), camera = nothing);
                        frames = 10, canvas = (16, 16))
    src = clip.source
    @test !VE.refining(src)                               # nothing built yet
    src.live = VE.LiveScene(nothing, FakeFilmScreen(NamedTuple[]), nothing,
                            (16, 16), :Fake, spec, Dict{Symbol, Any}(), Any[])
    @test VE.progressive(src.live.screen)                 # …or none of this is asked
    for n in (0, 64, 100_000)
        src.samples = n
        @test VE.refining(src)
    end
    # A rasteriser has nothing to add and must not be asked to keep going.
    src.live = VE.LiveScene(nothing, nothing, nothing, (16, 16), :Fake, spec,
                            Dict{Symbol, Any}(), Any[])
    @test !VE.refining(src)
end

# A Mantle-backed renderer (RayMakie) shares the ONE GPU context the pinned
# worker owns — rendering it on thread 1 died with "SubmitChannel is
# single-writer; cross-thread sweep forbidden". The thread a scene renders on is
# the backend's property, not the caller's: asked of the module (it binds Mantle
# or it doesn't), never hard-coded per name.
#
# Asked of the RUNTIME and not of a compiler: naming the latter stopped being
# true when RayMakie was ported to Mantle — the old check went false and every
# scene clip silently went back to thread 1, which throws nothing and renders on
# the wrong thread.
module FakeGPUBackend
    import Mantle
end

@testset "a scene renders on the thread its renderer owns" begin
    @test VE.renderthread(VE.GLMakie) == 0                          # GLMakie: thread 1
    @test VE.renderthread(FakeGPUBackend) == Threads.nthreads() - 1   # GPU: the worker's
    # …and the hop lands there, from wherever it is called
    @test VE.onthread(() -> Threads.threadid(), 0) == 1
    @test VE.onworkerthread(() -> Threads.threadid()) == Threads.nthreads()
end

@testset "a slow render is not a stuck one" begin
    # The deadline is on STARTING, not on running. The first RayMakie frame builds
    # its screen and compiles the renderer's shaders, which took longer than the
    # ten seconds allowed: the render was aborted with "waited 10.0s to reach thread 24",
    # the retry after it succeeded because everything was compiled by then, and the
    # error therefore looked alarming and harmless at once. The failure it is
    # actually for — an owning thread that never yields, so the pinned task never
    # runs — is unchanged.
    tid = Threads.nthreads() - 1                          # a real hop, not the caller
    @test VE.onthread(tid; startwithin = 0.05) do
        sleep(0.3)                                        # running, just not finished
        :rendered
    end === :rendered
end
