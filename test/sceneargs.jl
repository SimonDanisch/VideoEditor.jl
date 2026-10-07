using Test
import VideoEditor as VE
using VideoEditor.Makie, VideoEditor.GeometryBasics

# A recipe with its own animated inputs, a figure layout around its 3D view, and
# no procedural animation: everything that moves comes from the clip's keys.
const ARGRECIPE = """
using Makie
struct Pose
    eye::Vec3f
    lookat::Vec3f
    fov::Float32
end
@enum Mode calm loud
function buildscene(canvas, args)
    fig = Figure(; size = canvas)
    ls = LScene(fig[1, 1]; show_axis = false)
    cam3d!(ls.scene; center = false)
    ax = Axis(fig[1, 2])
    time = Observable(0f0)
    count = Observable(2)
    shown = Observable(true)
    mode = Observable(calm)
    pose = Observable(Pose(Vec3f(3, 0, 1), Vec3f(0), 40f0))
    tint = Observable(RGBAf(1, 0, 0, 1))
    data = Observable(rand(Float32, 4, 4))
    updates = Ref(0)
    on(_ -> (updates[] += 1), time)
    mesh!(ls.scene, Rect3f(Vec3f(-0.5), Vec3f(1)); name = :box, color = tint)
    lines!(ax, lift(t -> [Point2f(x, sin(x + t)) for x in 0:0.1:3], time); name = :curve)
    scatter!(ax, lift(n -> [Point2f(i, 0) for i in 1:n], count); name = :dots, visible = shown)
    return (scene = fig.scene, args = (; time, count, shown, mode, pose, tint, data), updates)
end
"""

@testset "recipe arguments key as numbers, steps and struct fields" begin
    mktempdir() do dir
        file = joinpath(dir, "args.jl")
        write(file, ARGRECIPE)
        build = VE.programscene(file)
        c = VE.sceneclip(VE.buildscene(build); build, frames = 48, canvas = (96, 64), framerate = 24)
        seq = VE.Sequence([c], 24)
        scene, target = VE.realize(c.source.root, (96, 64))
        c.source.live = VE.LiveScene(scene, nothing, target, (96, 64), :GLMakie, c.source.root,
                                     Dict{Symbol, Any}(), Any[])
        # The figure's root has a pixel camera; the LScene's 3D camera is the one keyed.
        @test target.camerascene !== nothing && target.camerascene !== scene
        VE.updatesceneprogram!(target, c.source, c, 0, 24)
        @test haskey(c.source.camera, :fov)

        rows = VE.argrows(target)
        paths = Set(r.path for r in rows)
        for p in ("args.time", "args.count", "args.shown", "args.mode", "args.pose.eye[1]",
                  "args.pose.lookat[3]", "args.pose.fov", "args.tint[4]", "args.data")
            @test Symbol(p) in paths
        end
        row(p) = only(filter(r -> r.path === Symbol(p), rows))
        @test row("args.count").discrete && row("args.mode").discrete && row("args.shown").discrete
        @test !row("args.time").discrete
        @test row("args.data").kind === :data
        @test row("args.pose.eye[1]").label == "Pose · Eye X"

        fx = VE.findslot(c, :scene)
        secs = VE.sceneparamsections(c, fx)
        @test any(s -> s.label == "Arguments", secs)
        @test any(s -> s.label == "Camera" && any(p -> p.name === Symbol("camera.fov"), s.params), secs)
        # A discrete row's parameter steps even when it gains keys later.
        countp = VE.param(fx, Symbol("args.count"))
        @test countp.curve[].interp === :hold

        VE.keyframes!(c, "args.time", [0 => 0.0, 24 => 2.0]; ease = :linear)
        VE.keyframes!(c, "args.count", [0 => 2, 10 => 5, 30 => 1]; discrete = true)
        VE.keyframes!(c, "args.shown", [0 => 1, 20 => 0]; discrete = true)
        VE.keyframes!(c, "args.mode", [0 => 0, 12 => 1]; discrete = true)
        VE.keyframes!(c, "args.pose.eye", [0 => Vec3f(3, 0, 1), 24 => Vec3f(5, 1, 2)])
        VE.keyframes!(c, "args.tint", [0 => RGBAf(1, 0, 0, 1), 24 => RGBAf(0, 0, 1, 1)]; ease = :linear)
        VE.keyframes!(c, "camera.fov", [0 => 40.0, 24 => 20.0])
        VE.keyframes!(c, "dots.visible", [0 => 1, 6 => 0]; discrete = true)

        args = Dict(target.args)
        @test target.update!.callback === VE.staticscene
        VE.applysceneparams!(c.source, c, 12)
        @test args[:time][] ≈ 1f0
        @test args[:count][] == 5                     # held from frame 10 on, no blend
        @test args[:shown][] === true
        @test Int(args[:mode][]) == 1                 # `loud`
        @test args[:pose][].eye ≈ Vec3f(4, 0.5, 1.5)
        @test args[:pose][].lookat == Vec3f(0)        # unkeyed fields keep the recipe value
        @test args[:tint][] ≈ RGBAf(0.5, 0, 0.5, 1)
        @test c.source.camera.fov ≈ 30f0
        @test Makie.cameracontrols(target.camerascene).fov[] ≈ 30f0
        @test Makie.findplot(scene, :dots).visible[] === false

        # One write per argument per frame, and none when nothing changed.
        n = Ref(0)
        on(_ -> (n[] += 1), args[:pose])
        VE.applysceneparams!(c.source, c, 13)
        @test n[] == 1
        VE.applysceneparams!(c.source, c, 13)
        @test n[] == 1

        # Removing keys returns an argument to the recipe's value; so does bypass.
        filter!(p -> p.name !== Symbol("args.time"), fx.params)
        VE.applysceneparams!(c.source, c, 12)
        @test args[:time][] == 0f0
        fx.enabled[] = false
        VE.applysceneparams!(c.source, c, 12)
        @test args[:count][] == 2 && args[:pose][].eye == Vec3f(3, 0, 1)
        fx.enabled[] = true

        # Saved and reopened, the keys and the hold mode come back.
        saved = joinpath(dir, "args.videoedit")
        VE.saveproject(saved, seq)
        restored = VE.loadproject(saved)
        rfx = VE.findslot(first(restored.clips), :scene)
        rc = VE.param(rfx, Symbol("args.count"))
        @test rc.curve[].interp === :hold
        @test VE.valueat(rc, 25) == 5
        @test VE.valueat(VE.param(rfx, Symbol("args.pose.eye[2]")), 12) ≈ 0.5
    end
end

@testset "package recipes resolve through the environment" begin
    build = VE.packagescene(VE.Makie, :Scene; args = Dict("x" => 1))
    @test build["package"] == "Makie" && !haskey(build, "file")
    root = VE.SceneProgram(build)
    @test root.package == "Makie" && root.file == ""
    @test VE.programbuilder!(root) === Makie.Scene
    @test VE.recipemodule("GeometryBasics") === GeometryBasics
end
