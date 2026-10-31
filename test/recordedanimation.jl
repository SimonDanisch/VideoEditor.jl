using Test
import VideoEditor as VE
isdefined(Main,:LAST_RECORDED_PROJECT) || (const LAST_RECORDED_PROJECT=Ref(""))

@testset "streamed animation samples" begin
    mktempdir() do tmp
        buffer = zeros(Float32,3,4)
        tracks = VE.recordanimation(joinpath(tmp,"samples"),0:5;framerate=24) do f,fps
            buffer .= f
            Dict(Symbol("field.color")=>buffer,Symbol("actor.alpha")=>Float32(f)/5,
                 :constant=>fill(2f0,3,4),:points=>[VE.Point3f(f,1,2)],
                 :volume=>fill(ComplexF32(f,1),2,3,4))
        end
        buffer .= -99
        t = tracks[Symbol("field.color")]
        @test t.frames == collect(0:5)
        @test t.framerate == 24
        @test sizeof(t.bytes) == 6*sizeof(buffer)
        @test length(unique(tracks[:constant].offsets)) == 1
        @test !VE.recordedchanged(tracks[:constant],0,5)
        @test VE.recordedchanged(t,0,5)
        @test sizeof(tracks[:constant].bytes) == sizeof(buffer)
        for f in (5,0,3,3,1,4)
            @test VE.valueat(t,f) == fill(Float32(f),3,4)
            @test VE.valueat(tracks[:points],f) == [VE.Point3f(f,1,2)]
            @test VE.valueat(tracks[:volume],f) == fill(ComplexF32(f,1),2,3,4)
        end
        @test_throws Exception VE.valueat(t,2.5)
        @test_throws Exception VE.valueat(t,6)
        @test_throws Exception setindex!(VE.valueat(t,0),12f0,1,1)
        @test VE.valueat(VE.openrecording(t.path),4) == fill(4f0,3,4)
        @test_throws ArgumentError VE.recordanimation((f,fps)->(;v=f),joinpath(tmp,"invalid"),[1,0])
        @test_throws Exception VE.recordanimation((f,fps)->error("simulation failed"),joinpath(tmp,"failed"),0:1)
        @test !ispath(joinpath(tmp,"failed"))
        @test_throws Exception VE.recordanimation((f,fps)->(;v=f),joinpath(tmp,"samples"),0:1)
        variable = VE.recordanimation((f,fps)->(;v=ones(Float32,f+1)),joinpath(tmp,"variable"),0:2)
        @test size(VE.valueat(variable[:v],2)) == (3,)
    end
end

if haskey(VE.BACKENDS,:RayMakie)
@testset "recorded scene, editor and farm seek the same arrays" begin
    mktempdir() do tmp
        out = get(ENV,"VIDEOEDITOR_RECORDING_OUTPUT",nothing)
        if out !== nothing
            mkpath(out);tmp=mktempdir(out;prefix="scene-")
        end
        script = joinpath(tmp,"scene.jl")
        write(script,"""
        using Makie
        const steps=Ref(0)
        function buildscene(canvas,args)
            scene=Scene(;size=canvas,camera=cam3d!,backgroundcolor=:black,
                lights=[AmbientLight(RGBf(.7,.7,.7))])
            texture=Observable(fill(RGBf(.1,.2,.3),4,4))
            subject=mesh!(scene,Rect3f(Vec3f(-.5),Vec3f(1));name=:subject,color=texture)
            mesh!(scene,Rect3f(Vec3f(20),Vec3f(1));name=:default_subject,visible=false)
            update_cam!(scene,Vec3f(3,-5,2),Vec3f(0),Vec3f(0,0,1))
            state=Ref(0)
            update! = (frame,fps)->begin
                steps[]+=1;state[]+=1
                texture[] .= RGBf(state[]/12,.2,.3);notify(texture)
                translate!(subject,state[]/8,0,0)
            end
            (;scene,update!)
        end
        """)
        build=VE.programscene(script)
        clip=VE.sceneclip(VE.buildscene(build);build,frames=12,canvas=(96,64),framerate=24,backend=:RayMakie)
        clip.source.screenopts=Dict{Symbol,Any}(:rasterize=>true,:samples=>1,:device=>"Mantle.defaultbackend()")
        clip.source.bakewith=:RayMakie
        clip.source.bakescreenopts=Dict{Symbol,Any}(:rasterize=>false,:samples=>4,:max_depth=>3,
            :denoise=>false,:device=>"Mantle.defaultbackend()")
        seq=VE.Sequence([clip],24);seq.canvas=(96,64)
        back=nothing;renderer=nothing
        try
        tracks=VE.recordsceneanimation!(clip,joinpath(tmp,"recording"))
        @test !haskey(tracks,Symbol("subject.model"))
        @test !haskey(tracks,Symbol("default_subject.color"))
        @test Base.invokelatest(getproperty,parentmodule(clip.source.root.builder),:steps)[] == 12
        @test VE.valueat(tracks[Symbol("subject.color")],0) == fill(VE.RGBAf(1/12,.2,.3,1),4,4)
        for f in (11,0,6,6,2)
            VE.updatesource!(clip.source,clip,f)
            VE.sceneframe!(clip.source,clip,(96,64))
            plot=VE.Makie.findplot(VE.targetscene(clip.source.live.target),:subject)
            @test plot.transformation.translation[][1] ≈ (f+1)/8
            @test VE.Makie.to_value(plot.attributes[:color]) == fill(VE.RGBAf((f+1)/12,.2,.3,1),4,4)
        end
        @test Base.invokelatest(getproperty,parentmodule(clip.source.root.builder),:steps)[] == 12
        fx=VE.findslot(clip,:scene)
        sections=VE.sceneparamsections(clip,fx)
        data=VE.param(fx,Symbol("subject.color"))
        @test data !== nothing && VE.isanimated(data)
        @test data.name in VE.activeanimationparams(clip,fx,9)
        reader=VE.recordedinput(data); VE.bindinputs!(seq)
        @test VE.recordedinput(data)===reader
        @test VE.valueat(data,9) == fill(VE.RGBAf(10/12,.2,.3,1),4,4)
        path=joinpath(tmp,"movie.videoedit");VE.saveproject(path,seq)
        LAST_RECORDED_PROJECT[]=path
        back=VE.loadproject(path)
        @test VE.valueat(VE.param(VE.findslot(only(back.clips),:scene),data.name),3) == fill(VE.RGBAf(4/12,.2,.3,1),4,4)
        job=VE.renderjob(path,joinpath(tmp,"job"))
        @test all(track.path in keys(job.inputs) for track in values(tracks))
        bundle=VE.bundlefarm(job,joinpath(tmp,"bundle");root=tmp)
        renderer=VE.openfarmbundle(bundle.directory)
            a,b,again=VE.farmframes!(renderer,[11,0,11])
            @test a.png == again.png
            @test a.png != b.png
            src=only(renderer.sequence.clips).source
            @test Base.invokelatest(getproperty,parentmodule(src.root.builder),:steps)[] == 0
            @test startswith(first(values(src.live.target.recordings)).path,bundle.directory)
        finally
            renderer===nothing || VE.closefarm!(renderer)
            close(clip.source)
            back===nothing || foreach(c -> close(c.source),back.clips)
        end
    end
end
end
