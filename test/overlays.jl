# The stock non-footage clips (src/overlays.jl): a title, a bar, a timecode, the
# captions — each a `SceneSpec` preset placed on a track.
#
# There is no `Overlay` any more. What this file used to test was a parallel
# timeline with its own registry, its own state bundle, its own keyframe world and
# its own card builder; every feature of a clip had to be written a second time for
# it. The properties below are what replaced it: a preset builds a scene, a clip
# carries it, and the recipe that made it is what the project file holds.

using VideoEditor.Makie

@testset "presets build scenes, and the recipe is what is saved" begin
    canvas = (320, 180)
    for (kind, kw) in ((:text, (; text = "HELLO")), (:bar, (;)),
                       (:timecode, (; framerate = 25.0)), (:captions, (;)))
        build = VE.scenebuild(kind, canvas; kw...)
        built = VE.buildscene(build)
        @test built.root isa Makie.SceneSpec
        @test length(built.root.plots) == 1
        @test isempty(built.joints)          # a 2-D graphic has no rig
        # pixel space: one unit is one canvas pixel, so the numbers in the card are
        # the numbers on the screen
        @test built.root.kwargs[:camera] === Makie.campixel!
    end

    # a curve preset carries its data
    b = VE.scenebuild(:curve, canvas; values = [1.0, 5.0, 2.0])
    @test length(VE.buildscene(b).root.plots) == 1
end

@testset "a scene lands the right way up" begin
    # THE FLIP NO COUNT CAN SEE. `Makie.GLNative` is the raw OpenGL framebuffer,
    # whose scanlines run bottom-up; `JuliaNative` flips AND transposes, and this
    # path takes the former for its (width, height) axis order — inheriting the
    # flip with it. Every scene came out upside down: the lego figure hung
    # head-down over the birdhouse and a text preset read "OBEN" as "OBEИ".
    #
    # Asserted by WHERE the pixels are, never how many: a mirror preserves every
    # count. The scene still drew on exactly 9238 pixels of the lego canvas, the
    # same number as before the rebuild, which is what made the number look like
    # proof that nothing had changed.
    engine = VE.FxEngine(VE.KA.CPU())
    canvas = (400, 200)
    drawnhalves(img) = (h = size(img, 2);
                        (count(!=(VE.RGB{VE.N0f8}(0, 0, 0)), view(img, :, 1:(h ÷ 2))),
                         count(!=(VE.RGB{VE.N0f8}(0, 0, 0)), view(img, :, (h ÷ 2 + 1):h))))
    function drawn(kind; kw...)
        build = VE.scenebuild(kind, canvas; kw...)
        clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = canvas)
        out = Ref{Any}(nothing)
        VE.render(o -> (out[] = copy(o)), engine, clip.source, clip, 0)
        return drawnhalves(out[])
    end
    # a TITLE sits at the top of the frame …
    top, bottom = drawn(:text; text = "OBEN")
    @test top > 0 && bottom == 0
    # … and a LOWER THIRD at the bottom. Two presets, two directions: one of them
    # alone would pass just as happily on a mirrored frame.
    top, bottom = drawn(:bar)
    @test bottom > 0 && top == 0
    VE.emptyengine!(engine)
end

@testset "an old overlay project opens as a clip on its own track" begin
    # A project written while overlays were a thing of their own. The span becomes
    # the clip's extent, the effect's parameters come across with their curves, and
    # the keys move from TIMELINE frames into the clip's own — a shift by `start`,
    # not a reinterpretation.
    src = VideoSource(testvideo)
    seq = Sequence([Clip(src; src_in = 0, src_out = 60, start = 0)], 30.0)
    path = tempname() * ".videoedit"
    saveproject(path, seq)

    # …hand-write the overlay the old writer produced
    d = VE.decodeblocks(VE.MsgPack.unpack(read(path)))
    d["overlays"] = [Dict{String, Any}(
        "kind" => "bar", "start" => 10, "stop" => 40,
        "settings" => Dict{String, Any}("color" => ":black"),
        "effect" => Dict{String, Any}(
            "kind" => "bar", "enabled" => true,
            "params" => [Dict{String, Any}(
                "T" => "Float64", "name" => "opacity", "label" => "Opacity",
                "value" => 0.6, "visible" => true, "lo" => 0.0, "hi" => 1.0,
                "curve" => Dict{String, Any}("interp" => "linear", "keys" => [
                    Dict{String, Any}("frame" => 10, "value" => 0.0, "ease" => "linear"),
                    Dict{String, Any}("frame" => 30, "value" => 1.0, "ease" => "linear")]))]))]
    open(io -> write(io, VE.MsgPack.pack(d)), path, "w")

    seq2 = loadproject(path)
    rm(path; force = true)
    @test length(seq2.clips) == 2
    c = seq2.clips[end]
    @test c.source isa VE.SceneSource
    @test c.track > 1                       # over the footage, where it was drawn
    @test (c.start, VE.clipend(c)) == (10, 40)
    fx = VE.findslot(c, :scene)
    p = VE.param(fx, :opacity)
    @test p !== nothing && VE.isanimated(p)
    # the keys moved by `start`: 10 → 0 and 30 → 20
    @test [k.frame for k in p.curve.keys] == [0, 20]
end

@testset "a scene clip is a clip: it trims, stacks and composites" begin
    src = VideoSource(testvideo)
    dims = (src.width, src.height)
    base = Clip(src; src_in = 0, src_out = 30, start = 0)
    build = VE.scenebuild(:bar, dims; opacity = 1.0, height = 0.5)
    bar = VE.sceneclip(VE.buildscene(build); build, frames = 30, canvas = dims)
    bar.track = 2
    seq = Sequence([base, bar], 30.0)

    engine = VE.FxEngine(VE.KA.CPU())
    grey = fill(VE.RGB{VE.N0f8}(0.5, 0.5, 0.5), dims...)
    with = Ref{Any}(nothing); without = Ref{Any}(nothing)
    @test VE.composite(engine, [base, bar], 0, (c, sf) -> grey; canvas = dims) do cv
        with[] = copy(cv)
    end
    @test VE.composite(engine, [base], 0, (c, sf) -> grey; canvas = dims) do cv
        without[] = copy(cv)
    end
    # it covers PART of the frame — half by construction — and leaves the rest
    changed = count(with[] .!= without[])
    @test 0 < changed < length(with[])
    # trimming it is trimming a clip
    bar.src_out = 10
    @test VE.cliplength(bar) == 10
    VE.emptyengine!(engine)
end
