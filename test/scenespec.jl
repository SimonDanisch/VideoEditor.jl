# A scene as a clip's source (src/scenesource.jl): the spec, the live scene, and
# what the panel reads off it.
#
# The properties that make the approach work, each easy to lose: what is
# animatable comes from the SCENE and not from a description beside it, a path
# addresses a plot BY NAME (so nothing re-aims when the scene is rebuilt), a
# joint's axis is authored data while its angle is a parameter, and what goes into
# the project file comes back out the same.

using VideoEditor.Makie
import VideoEditor.Makie.SpecApi as S

@testset "a scene clip draws a spec" begin
    spec = S.Scene(; camera = Makie.campixel!,
                   plots = [Makie.PlotSpec(:Scatter, [Makie.Point2f(10, 10)];
                                           markersize = 8.0, name = :dot)])
    clip = VE.sceneclip((root = spec, joints = Dict{Symbol, Any}(), camera = nothing);
                        frames = 30, canvas = (64, 48), framerate = 30.0)
    @test clip.source isa VE.SceneSource
    @test !VE.decodable(clip.source)          # nothing to decode, nothing to proxy
    @test VE.sourcepath(clip.source) == ""
    @test (clip.source.width, clip.source.height) == (64, 48)
    @test VE.cliplength(clip) == 30
    # the `:scene` entry is DATA — it holds the curves, it is not a pixel pass
    fx = VE.findslot(clip, :scene)
    @test fx !== nothing && !VE.renderable(fx)
    @test isempty(VE.graphof!(clip, (64, 48)).slots)   # …so the chain has no effect pass

    engine = VE.FxEngine(VE.KA.CPU())
    img = Ref{Any}(nothing)
    VE.render(engine, clip.source, clip, 0) do o; img[] = copy(o); end
    @test size(img[]) == (64, 48)
    # it DREW, and it did not cover the frame: a scene is transparent where it
    # drew nothing, which is what lets it sit over a clip
    drawn = count(!=(VE.RGB{VE.N0f8}(0, 0, 0)), img[])
    @test 0 < drawn < length(img[])
    VE.emptyengine!(engine)
end

@testset "what is animatable comes from the scene" begin
    spec = S.Scene(; camera = Makie.campixel!,
                   plots = [Makie.PlotSpec(:Scatter, [Makie.Point2f(10, 10)];
                                           markersize = 8.0, name = :dot)])
    clip = VE.sceneclip((root = spec, joints = Dict{Symbol, Any}(), camera = nothing);
                        frames = 30, canvas = (64, 48))
    src = clip.source
    # BEFORE anything is rendered there is no scene, so there is nothing to offer —
    # and that is why the card is lazy rather than eager
    @test isempty(VE.sceneattributes(src))

    engine = VE.FxEngine(VE.KA.CPU())
    VE.render(engine, src, clip, 0) do _; nothing end
    objs = VE.sceneattributes(src)
    @test !isempty(objs)
    dot = objs[findfirst(o -> o.name === :dot, objs)]
    paths = [String(r.path) for r in dot.rows]
    # a scatter's `markersize` is a `Vec2f`, so it is two rows — the row kind
    # comes from the VALUE, not from a list of attribute names
    @test "dot.markersize[1]" in paths && "dot.markersize[2]" in paths
    @test "dot.strokewidth" in paths                    # …and a scalar is one row
    # a DERIVED attribute is not a row: `eyeposition` and friends are answers
    # Makie computed from the scene, and a slider on an answer gets overwritten
    @test !any(startswith("dot.eyeposition"), paths)
    @test !any(startswith("dot.N_"), paths)

    # writing a path lands on the plot itself
    @test VE.setscenevalue!(src, Symbol("dot.markersize[1]"), 20.0)
    plot = Makie.findplot(VE.targetscene(src.live.target), :dot)
    @test Makie.to_value(plot.attributes[:markersize])[1] ≈ 20.0
    # …and a path that addresses nothing says so instead of throwing
    @test !VE.setscenevalue!(src, Symbol("nosuch.thing"), 1.0)
    VE.emptyengine!(engine)
end

@testset "the screen is held across frames, and a sample accumulates" begin
    # TWO claims of step 7, both cheap to lose and both expensive when lost.
    #
    # The source pass HOLDS the screen and the plots. Rebuilding a GLMakie screen
    # per frame is not a slowdown, it is a different program — every plot is
    # recreated, every attribute forgotten, and a path tracer starts from zero
    # every time. So the identity of the live scene, not just the picture, is what
    # has to be asserted.
    spec = S.Scene(; camera = Makie.campixel!,
                   plots = [Makie.PlotSpec(:Scatter, [Makie.Point2f(10, 10)];
                                           markersize = 8.0, name = :dot)])
    clip = VE.sceneclip((root = spec, joints = Dict{Symbol, Any}(), camera = nothing);
                        frames = 30, canvas = (64, 48))
    src = clip.source
    engine = VE.FxEngine(VE.KA.CPU())
    VE.render(engine, src, clip, 0) do _; nothing end
    live1 = src.live
    @test live1 !== nothing
    for f in (0, 1, 2, 3)
        VE.render(engine, src, clip, f) do _; nothing end
    end
    @test src.live === live1                     # …the same screen, four frames on
    plot1 = Makie.findplot(VE.targetscene(src.live.target), :dot)
    VE.render(engine, src, clip, 4) do _; nothing end
    @test Makie.findplot(VE.targetscene(src.live.target), :dot) === plot1

    # PROGRESSIVE: one sample per read at a position, the count reset when the
    # position moves. `sceneframe!` clears the film on the first read at a
    # position and adds to it after — a rasteriser ignores that, a path tracer is
    # the reason it exists, and both go through the same counter.
    VE.render(engine, src, clip, 10) do _; nothing end
    @test src.at == 10
    n = src.samples
    VE.render(engine, src, clip, 10) do _; nothing end
    @test src.samples == n + 1                   # standing still ADDS
    VE.render(engine, src, clip, 11) do _; nothing end
    @test src.at == 11 && src.samples == 1       # …moving throws the film away

    # a canvas change IS a new screen — the settings go in at construction
    Base.resize!(src, (32, 24))
    VE.render(engine, src, clip, 11) do _; nothing end
    @test src.live !== live1
    VE.emptyengine!(engine)
end

@testset "a missing renderer is reported, not substituted" begin
    # Step 9's rule. A scene names the renderer that drew it; opening the project
    # on a machine without that package must SAY so, not quietly hand the clip to
    # whatever is loaded and show a different picture than the one that was saved.
    spec = S.Scene(; camera = Makie.campixel!,
                   plots = [Makie.PlotSpec(:Scatter, [Makie.Point2f(1, 1)]; name = :dot)])
    clip = VE.sceneclip((root = spec, joints = Dict{Symbol, Any}(), camera = nothing);
                        frames = 10, canvas = (32, 24))
    clip.source.backend = :NoSuchMakie
    clip.source.bakewith = :AlsoMissing
    seq = Sequence([clip], 30.0)
    absent = @test_logs (:warn,) VE.reportbackends(seq)   # not `missing`: that is Base's
    @test absent == [:AlsoMissing, :NoSuchMakie]       # BOTH sets of settings
    @test clip.source.backend === :NoSuchMakie         # …and nothing was replaced

    # …while a project that names only loaded renderers says nothing at all
    clip.source.backend = :GLMakie
    clip.source.bakewith = :auto
    @test isempty(VE.reportbackends(seq))
end

@testset "a rig's axis is data, its angle is a parameter" begin
    rig = Dict{String, Any}("parts" => [
        Dict{String, Any}("name" => "torso", "type" => "Scatter",
                          "args" => [[Makie.Point3f(0, 0, 0)]],
                          "origin" => [0.0, 0.0, 0.0], "axis" => [0.0, 0.0, 1.0]),
        Dict{String, Any}("name" => "arm", "type" => "Scatter", "parent" => "torso",
                          "args" => [[Makie.Point3f(1, 0, 0)]],
                          "origin" => [1.0, 0.0, 0.0], "axis" => [0.0, 1.0, 0.0])])
    built = VE.buildscene(Dict{String, Any}("kind" => "rig", "rig" => rig))
    @test length(built.joints) == 2
    @test built.joints[:arm].axis ≈ VE.Vec3f(0, 1, 0)   # …from the description
    @test built.camera !== nothing

    clip = VE.sceneclip(built; frames = 20, canvas = (64, 48))
    fx = VE.findslot(clip, :scene)
    engine = VE.FxEngine(VE.KA.CPU())
    VE.render(engine, clip.source, clip, 0) do _; nothing end

    objs = VE.sceneattributes(clip.source)
    arm = objs[findfirst(o -> o.name === :arm, objs)]
    paths = [String(r.path) for r in arm.rows]
    # the joint's rows come from the DESCRIPTION — `angle` and `offset` are not
    # Makie attributes, they are how the part is hung
    @test "arm.angle" in paths && "arm.offset[1]" in paths

    # keyframing the angle turns the part, and the parent chain carries the rest
    push!(fx.params, VE.Param(Symbol("arm.angle"), "Angle", 0.0; range = (-3.2, 3.2)))
    p = VE.param(fx, Symbol("arm.angle"))
    p.curve = VE.AnimCurve{Float64}()
    VE.setkey!(p.curve, 0, 0.0); VE.setkey!(p.curve, 19, 1.5)
    a = Ref{Any}(nothing); b = Ref{Any}(nothing)
    VE.render(engine, clip.source, clip, 0) do o; a[] = copy(o); end
    VE.render(engine, clip.source, clip, 19) do o; b[] = copy(o); end
    @test count(a[] .!= b[]) > 0
    VE.emptyengine!(engine)
end

@testset "a scene clip round-trips through a project file" begin
    build = VE.scenebuild(:text, (64, 48); text = "HI")
    seq = Sequence(Clip[], 30.0)
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 20, canvas = (64, 48))
    push!(seq.clips, clip)
    fx = VE.findslot(clip, :scene)
    push!(fx.params, VE.Param(Symbol("title.fontsize"), "Fontsize", 12.0; range = (1.0, 99.0)))
    VE.param(fx, Symbol("title.fontsize")).curve = VE.AnimCurve{Float64}()
    VE.setkey!(VE.param(fx, Symbol("title.fontsize")).curve, 0, 12.0)

    path = tempname() * ".videoedit"
    saveproject(path, seq)
    seq2 = loadproject(path)
    rm(path; force = true)
    c2 = seq2.clips[1]
    @test c2.source isa VE.SceneSource
    @test c2.source.build["kind"] == "text"
    @test c2.source.root isa Makie.SceneSpec
    # …and the CURVE came back, which is the whole point of the parameters living
    # on the effect rather than being discovered fresh from a scene each time
    p2 = VE.param(VE.findslot(c2, :scene), Symbol("title.fontsize"))
    @test p2 !== nothing && VE.isanimated(p2)
end
