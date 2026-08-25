# A Makie scene as data (src/scenespec.jl): the tree, the paths, the round trip.
#
# The properties that make the approach work, each easy to lose: a path addresses
# a part BY NAME (so reordering does not re-aim it), animating writes a COPY (so
# it cannot accumulate across frames), and what goes into the project file comes
# back out the same.

# `Makie` is not in `Main` when the suite runs it — reach it through the package,
# the way the other test files reach `VE`.
using VideoEditor.Makie

meshpart(name, parent = nothing; kw...) =
    VE.ScenePart(name, Makie.PlotSpec(:Mesh; color = :red); parent = parent, kw...)

@testset "scenespec: a path addresses a part by name" begin
    spec = VE.SceneSpec([meshpart(:torso),
                         meshpart(:arm_left, :torso; origin = (0.1, 6.2, 5.7),
                                  axis = (0.0, 0.98, 0.18))])

    @test VE.scenepath("arm_left.angle") == (:arm_left, :angle, nothing)
    @test VE.scenepath("torso.offset[2]") == (:torso, :offset, 2)
    # unaddressable is `nothing`, NOT an error: a project from a newer editor may
    # name parts this one has not got, and such a curve is one to ignore
    @test VE.scenepath("nonsense") === nothing
    @test VE.scenepathvalue(spec, "nosuchpart.angle") === nothing
    @test VE.scenepathvalue(spec, "torso.nosuchfield") === nothing

    @test VE.scenepathvalue(spec, "arm_left.angle") == 0.0
    @test VE.scenepathvalue(spec, "arm_left.origin") == (0.1, 6.2, 5.7)
    @test VE.scenepathvalue(spec, "arm_left.origin[2]") == 6.2
    @test VE.setscenepath!(spec, "arm_left.angle", 0.5)
    @test VE.scenepathvalue(spec, "arm_left.angle") == 0.5
    # one component at a time — an offset is three curves, not a new value type
    @test VE.setscenepath!(spec, "torso.offset[3]", 20.0)
    @test VE.scenepathvalue(spec, "torso.offset") == (0.0, 0.0, 20.0)
    @test !VE.setscenepath!(spec, "nosuchpart.angle", 1.0)

    # …and a plot ATTRIBUTE goes through the same path syntax, which is why no
    # part declares in advance what is animatable
    @test VE.setscenepath!(spec, "torso.transparency", true)
    @test VE.scenepathvalue(spec, "torso.transparency") === true
end

@testset "scenespec: reordering parts does not re-aim a curve" begin
    # THE reason paths are by name. With `plots[2].angle` this test is the bug:
    # inserting anything ahead of a part silently points its curve at a neighbour.
    spec = VE.SceneSpec([meshpart(:torso), meshpart(:arm_left, :torso)])
    VE.setscenepath!(spec, "arm_left.angle", 0.5)

    reordered = VE.SceneSpec(reverse(spec.parts); backend = spec.backend)
    @test reordered.parts[1].name === :arm_left          # it really did move
    @test VE.scenepathvalue(reordered, "arm_left.angle") == 0.5
    # …and inserting a part in front changes nothing either
    grown = VE.SceneSpec(vcat([meshpart(:head, :torso)], spec.parts))
    @test VE.scenepathvalue(grown, "arm_left.angle") == 0.5
    @test VE.scenepathvalue(grown, "torso.angle") == 0.0
end

@testset "scenespec: names must be unique" begin
    # A duplicate makes the second part unreachable — `findfirst` takes the first
    # — so it is refused at construction rather than animating the wrong arm.
    @test_throws ErrorException VE.SceneSpec([meshpart(:arm), meshpart(:arm)])
    # `camera` is reserved: "camera.eye[1]" is how a camera move is keyframed
    @test_throws ErrorException VE.SceneSpec([meshpart(:camera)])
end

@testset "scenespec: the camera is keyframed like anything else" begin
    spec = VE.SceneSpec([meshpart(:torso)];
                        camera = VE.CameraSpec(; eye = (100.0, 30.0, 80.0),
                                               lookat = (0.0, 0.0, -10.0)))
    @test VE.scenepathvalue(spec, "camera.eye") == (100.0, 30.0, 80.0)
    @test VE.scenepathvalue(spec, "camera.eye[1]") == 100.0
    @test VE.setscenepath!(spec, "camera.eye[1]", 60.0)
    @test VE.scenepathvalue(spec, "camera.eye") == (60.0, 30.0, 80.0)
    @test VE.scenepathvalue(spec, "camera.nosuchfield") === nothing
end

@testset "scenespec: animating writes a copy" begin
    spec = VE.SceneSpec([meshpart(:torso), meshpart(:arm_left, :torso)])
    a = VE.animatedspec(spec, (; Symbol("arm_left.angle") => 0.9))
    @test VE.scenepathvalue(a, "arm_left.angle") == 0.9
    # the ORIGINAL is untouched: mutating in place would make the animation
    # cumulative — a curve that ended at frame 10 would still be showing its last
    # value at frame 11 instead of the spec's own
    @test VE.scenepathvalue(spec, "arm_left.angle") == 0.0
    # a key that addresses nothing is skipped, not fatal — `overlaystate` merges
    # reserved keys like `frame`/`framerate` into the same namedtuple
    b = VE.animatedspec(spec, (; :frame => 7, Symbol("arm_left.angle") => 0.2))
    @test VE.scenepathvalue(b, "arm_left.angle") == 0.2
end

@testset "scenespec: a point light falls off with distance" begin
    # ONE intensity number, TWO renderers. A raytracer is physical by
    # construction, but Makie's `PointLight` defaults to `attenuation = Vec2f(0)`
    # — no falloff at all — so a lamp written for RayMakie reached GLMakie
    # undimmed. Measured on the lego figure: intensity 15000 at 269 units came
    # back a flat (1.0, 1.0, 1.0) on every lit pixel, and the diagnosis that
    # invites is "the legs are missing" when they are blue legs blown to white.
    lights = VE.makielights([VE.LightSpec(:ambient; color = (0.1, 0.1, 0.1)),
                             VE.LightSpec(:point; color = (15000.0, 15000.0, 15000.0),
                                          position = (150.0, 100.0, 200.0))])
    pt = lights[2]
    @test pt isa Makie.PointLight
    @test pt.attenuation == Makie.Vec2f(0, 1)   # 1/(1 + 0*d + 1*d^2)
    # ambient has no position and so nothing to fall off
    @test lights[1] isa Makie.AmbientLight
end

@testset "scenespec: a scene overlay survives save/load" begin
    # THE GAP THIS CLOSES. `tomlvalue`'s fallback is `string(v)`, so a SceneSpec
    # in an overlay's settings was saved as its `show` form — unreadable — and
    # the project reopened with `spec` a String. `:scene`'s draw opens with
    # `spec isa SceneSpec || return`, so the scene came back SILENTLY ABSENT.
    # The round trip through `scenedict` alone (below) never caught it, because
    # nothing tested the SETTINGS carrying one.
    src = VE.VideoSource(testvideo)
    seq = VE.Sequence(src)
    spec = VE.SceneSpec(
        [VE.ScenePart(:torso, Makie.PlotSpec(:Mesh, "lego_figure_torso.stl")),
         VE.ScenePart(:arm_left, Makie.PlotSpec(:Mesh, "lego_figure_arm_left.stl");
                      parent = :torso, origin = (0.1427, 6.2127, 5.7342),
                      axis = (0.0, 0.9828, 0.1848))];
        backend = :GLMakie)
    # `bakewith` is GLMakie here, not RayMakie: the suite does not load a
    # raytracer, and what is under test is that a SETTING survives as a name the
    # registry accepts — not which renderer it names.
    ov = VE.addoverlay!(seq, :scene; start = 0, stop = 60,
                        spec = spec, bakewith = :GLMakie)
    VE.setoverlaykey!(ov, Symbol("arm_left.angle"), 0, 0.0)
    VE.setoverlaykey!(ov, Symbol("arm_left.angle"), 30, 0.8)

    path = joinpath(mktempdir(), "scene.videoedit.json")
    VE.saveproject(path, seq)
    back = VE.loadproject(path)
    bov = back.overlays[1]
    got = get(bov.settings, :spec, nothing)

    @test got isa VE.SceneSpec                     # NOT a String
    @test length(got.parts) == 2
    @test VE.partbyname(got, :arm_left).parent === :torso
    @test VE.scenepathvalue(got, "arm_left.origin[2]") ≈ 6.2127
    # a setting comes back as a STRING, and `bake!` feeds it straight to
    # `getbackend` — without a String method that was a MethodError on the first
    # bake after reopening
    @test get(bov.settings, :bakewith, nothing) == "GLMakie"
    @test VE.getbackend(get(bov.settings, :bakewith, :GLMakie)) === VE.getbackend(:GLMakie)
    # …and a name nobody registered still says so, rather than rendering nothing
    @test_throws ErrorException VE.getbackend("NoSuchBackend")
    # the curves are the animation — they must survive with their values
    @test length(bov.animations[Symbol("arm_left.angle")].keys) == 2
    @test VE.overlaystate(bov, 30; framerate = 30.0)[Symbol("arm_left.angle")] ≈ 0.8
end

@testset "scenespec: a bake never reaches the project file" begin
    # `bake!` parks rendered frames in `settings`, and `tomlvalue`'s fallback
    # would have written the `show` form of a Dict of images into the JSON —
    # megabytes of pixel repr in place of an edit. The bake is DERIVED from the
    # spec and the curves, both of which are saved.
    seq = VE.Sequence(VE.VideoSource(testvideo))
    spec = VE.SceneSpec([VE.ScenePart(:torso, Makie.PlotSpec(:Mesh, "lego_figure_torso.stl"))])
    ov = VE.addoverlay!(seq, :scene; start = 0, stop = 4, spec = spec)
    # qualified through `VE`: the suite's Main has neither ColorTypes nor
    # FixedPointNumbers, and a testset that only ran in my own session with them
    # imported is a testset that has not run
    frames = Dict{Int, Matrix{VE.RGBA{VE.N0f8}}}(
        n => fill(VE.RGBA{VE.N0f8}(1, 0, 0, 1), 64, 64) for n in 0:3)
    ov.settings = merge(ov.settings, (baked = frames, bakedcanvas = (64, 64)))
    @test VE.bakedframes(ov) !== nothing

    path = joinpath(mktempdir(), "baked.videoedit.json")
    VE.saveproject(path, seq)
    @test filesize(path) < 20_000            # not the pixels
    back = VE.loadproject(path)
    @test VE.bakedframes(back.overlays[1]) === nothing        # cache dropped
    @test get(back.overlays[1].settings, :spec, nothing) isa VE.SceneSpec  # scene kept
end

@testset "scenespec: a bake survives to disk and comes back" begin
    # A bake costs minutes, so it goes beside the project like a matte does and
    # is reloaded on open — including after a crash.
    seq = VE.Sequence(VE.VideoSource(testvideo))
    spec = VE.SceneSpec([VE.ScenePart(:torso, Makie.PlotSpec(:Mesh, "lego_figure_torso.stl"))];
                        backend = :GLMakie)
    ov = VE.addoverlay!(seq, :scene; start = 0, stop = 4, spec = spec, bakewith = :GLMakie)
    VE.setoverlaykey!(ov, Symbol("torso.angle"), 0, 0.0)
    VE.setoverlaykey!(ov, Symbol("torso.angle"), 3, 0.5)

    path = joinpath(mktempdir(), "bake.videoedit.json")
    canvas = (64, 64)
    @test VE.bake!(ov, canvas; framerate = 30.0, into = path) == 4
    VE.saveproject(path, seq)

    # A: it comes back, exactly
    back = VE.loadproject(path)
    got = VE.bakedframes(back.overlays[1])
    @test got !== nothing && length(got) == 4
    @test all(got[k] == VE.bakedframes(ov)[k] for k in keys(got))

    # B: an edit makes it STALE, and stale must lose. This failed the first time:
    # `savebakes` recomputed the fingerprint at save time, stamping the new one
    # onto the old frames, so a changed keyframe reloaded as though current. The
    # fingerprint is taken WITH the frames and travels with them.
    edited = VE.loadproject(path)
    VE.setoverlaykey!(edited.overlays[1], Symbol("torso.angle"), 3, 1.4)
    VE.saveproject(path, edited)
    @test VE.bakedframes(VE.loadproject(path).overlays[1]) === nothing

    # C: a crash mid-bake leaves SOME frames; those still count, the rest go live
    seq2 = VE.Sequence(VE.VideoSource(testvideo))
    ov2 = VE.addoverlay!(seq2, :scene; start = 0, stop = 4, spec = spec, bakewith = :GLMakie)
    p2 = joinpath(mktempdir(), "crash.videoedit.json")
    VE.bake!(ov2, canvas; framerate = 30.0, into = p2)
    VE.saveproject(p2, seq2)
    dir = VE.bakeframedir(p2, ov2.id)
    for f in readdir(dir)[3:end]
        rm(joinpath(dir, f))
    end
    partial = VE.loadproject(p2).overlays[1]
    @test length(VE.bakedframes(partial)) == 2
    @test VE.bakedframe(partial, 0, canvas) !== nothing    # kept
    @test VE.bakedframe(partial, 3, canvas) === nothing    # missing → drawn live
end

@testset "scenespec: survives the project file" begin
    spec = VE.SceneSpec(
        [VE.ScenePart(:torso, Makie.PlotSpec(:Mesh, "lego_figure_torso.stl";
                                             color = :orangered)),
         VE.ScenePart(:arm_left, Makie.PlotSpec(:Mesh, "lego_figure_arm_left.stl");
                      parent = :torso, origin = (0.1427, 6.2127, 5.7342),
                      axis = (0.0, 0.9828, 0.1848), angle = 0.25)];
        lights = [VE.LightSpec(:ambient; color = (0.1, 0.1, 0.1)),
                  VE.LightSpec(:point; color = (15000.0, 15000.0, 15000.0),
                               position = (150.0, 100.0, 200.0))],
        camera = VE.CameraSpec(; eye = (100.0, 30.0, 80.0), lookat = (0.0, 0.0, -10.0)),
        backend = :RayMakie,
        theme = Dict{Symbol, Any}(:RayMakie => Dict{Symbol, Any}(:exposure => 0.8)))

    back = VE.scenefromdict(VE.JSON.parse(VE.JSON.json(VE.scenedict(spec))))

    @test back.backend === :RayMakie
    @test length(back.parts) == 2
    # the TREE survives — without it the animation is a matrix per part
    @test VE.partbyname(back, :arm_left).parent === :torso
    @test VE.partbyname(back, :torso).parent === nothing
    @test VE.scenepathvalue(back, "arm_left.angle") ≈ 0.25
    @test VE.scenepathvalue(back, "arm_left.origin[2]") ≈ 6.2127
    # the mesh is a FILE NAME, not a vertex array: a project file cannot hold one
    @test back.parts[1].plot.args[1] == "lego_figure_torso.stl"
    @test back.parts[1].plot.kwargs[:color] == Makie.to_color(:orangered)
    # lights, camera and the backend's own settings all come back
    @test length(back.lights) == 2 && back.lights[2].type === :point
    @test back.lights[2].color[1] ≈ 15000.0
    @test back.camera.eye == (100.0, 30.0, 80.0)
    @test back.theme[:RayMakie][:exposure] ≈ 0.8
end
