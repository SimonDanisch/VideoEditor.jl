using Test
import VideoEditor as VE
using VideoEditor.Makie, VideoEditor.GeometryBasics

const NATIVERECIPE = """
using Makie
function buildscene(canvas, args)
    fig=Figure(;size=canvas)
    first=LScene(fig[1,1];show_axis=false)
    second=LScene(fig[1,2];show_axis=false)
    for ls in (first,second)
        cam3d!(ls.scene;center=false)
        update_cam!(ls.scene,Vec3f(3,-5,2),Vec3f(0),Vec3f(0,0,1))
    end
    mesh!(first.scene,Rect3f(Vec3f(-.5),Vec3f(1));name=:body,color=RGBAf(1,0,0,1))
    mesh!(second.scene,Rect3f(Vec3f(-.5),Vec3f(1));name=:other,color=:blue)
    return (;scene=fig.scene)
end
obsolete(canvas,args)=(scene=Scene(;size=canvas),args=(;phase=Observable(0.0)))
"""

@testset "native plot and camera tracks survive seeking, bypass and save" begin
    mktempdir() do dir
        file=joinpath(dir,"native.jl");write(file,NATIVERECIPE)
        build=VE.programscene(file)
        c=VE.sceneclip(VE.buildscene(build);build,frames=48,canvas=(96,64),framerate=24)
        seq=VE.Sequence([c],24)
        scene,target=VE.realize(c.source.root,(96,64))
        c.source.live=VE.LiveScene(scene,nothing,target,(96,64),:GLMakie,c.source.root,Dict{Symbol,Any}(),Any[])
        try
            @test !hasfield(typeof(target),:args)
            @test !hasfield(typeof(target),:update!)
            @test length(VE.nativecameras(scene))==2
            VE.updatesceneprogram!(target,c.source,c,0,24;reset=true)
            fx=VE.findslot(c,:scene)
            sections=VE.sceneparamsections(c,fx)
            @test any(s->s.name===Symbol("camera[2]"),sections)
            @test all(s->s.name!==:args,sections)
            VE.keyframes!(c,"body.color",[0=>RGBAf(1,0,0,1),24=>RGBAf(0,0,1,.5)];ease=:linear)
            VE.keyframes!(c,"body.visible",[0=>1,20=>0];discrete=true)
            VE.keyframes!(c,"body.translation",[0=>Vec3f(0),24=>Vec3f(2,1,0)];ease=:linear)
            VE.keyframes!(c,"camera.fov",[0=>40.,24=>20.];ease=:linear)
            VE.keyframes!(c,"camera[2].eye",[0=>Vec3f(3,-5,2),24=>Vec3f(5,-3,4)];ease=:linear)
            body=Makie.findplot(scene,:body)
            writes=Ref(0);on(_->(writes[]+=1),body.color)
            for frame in (12,24,0,12)
                VE.updatesceneprogram!(target,c.source,c,frame,24;reset=true)
                n=writes[]
                VE.applysceneparams!(c.source,c,frame)
                @test writes[]<=n+1
                @test body.visible[]==(frame<20)
                @test body.color[]≈RGBAf(1-frame/24,0,frame/24,1-frame/48)
                @test body.transformation.translation[]≈Vec3f(frame/12,frame/24,0)
                @test Makie.cameracontrols(VE.nativecameras(scene)[2]).eyeposition[]≈Vec3f(3+frame/12,-5+frame/12,2+frame/12)
                n=writes[];VE.applysceneparams!(c.source,c,frame)
                @test writes[]==n
            end
            fx.enabled[]=false
            VE.updatesceneprogram!(target,c.source,c,0,24;reset=true)
            VE.applysceneparams!(c.source,c,0)
            @test body.color[]==RGBAf(1,0,0,1)
            @test body.visible[]
            @test Makie.cameracontrols(VE.nativecameras(scene)[2]).eyeposition[]==Vec3f(3,-5,2)
            fx.enabled[]=true
            saved=joinpath(dir,"native.videoedit");VE.saveproject(saved,seq)
            restored=VE.loadproject(saved)
            rfx=VE.findslot(only(restored.clips),:scene)
            @test VE.valueat(VE.param(rfx,Symbol("camera[2].eye[1]")),12)≈4
            @test VE.param(rfx,Symbol("body.visible")).curve[].interp===:hold
            @test all(!startswith(String(p.name),"args.") for p in rfx.params)
            @test_throws ErrorException VE.realize(VE.SceneProgram(VE.programscene(file;entry=:obsolete)),(4,4))
        finally
            Makie.free(scene)
        end
    end
end

@testset "package recipes resolve through the root environment" begin
    build=VE.packagescene(VE.Makie,:Scene;args=Dict("x"=>1))
    root=VE.SceneProgram(build)
    @test root.package=="Makie" && root.file==""
    @test VE.programbuilder!(root)===Makie.Scene
    @test VE.recipemodule("GeometryBasics")===GeometryBasics
end
