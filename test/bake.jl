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
