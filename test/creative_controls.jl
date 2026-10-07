using Test
import VideoEditor as VE

@testset "native scene lights retain original animation and saved overrides" begin
    mktempdir() do dir
        file=joinpath(dir,"lights.jl")
        write(file,"""
        using Makie
        function buildscene(canvas,args)
            scene=Scene(;size=canvas,camera=cam3d!,lights=[
                AmbientLight(RGBf(.2,.3,.4)),PointLight(RGBf(1,1,1),Point3f(1,2,3))])
            mesh!(scene,Rect3f(Vec3f(-.5),Vec3f(1));name=:actor)
            update! = (frame,fps)->set_light!(scene,1;position=Point3f(frame,2,3))
            return (scene,update! = update!)
        end
        """)
        build=VE.programscene(file)
        c=VE.sceneclip(VE.buildscene(build);build,frames=48,canvas=(96,64),framerate=24)
        seq=VE.Sequence([c],24)
        scene,target=VE.realize(c.source.root,(96,64))
        c.source.live=VE.LiveScene(scene,nothing,target,(96,64),:GLMakie,c.source.root,Dict{Symbol,Any}(),Any[])
        fx=VE.findslot(c,:scene)
        secs=VE.sceneparamsections(c,fx)
        @test any(s->s.label=="Light · Ambient",secs)
        p=VE.param(fx,Symbol("lights[1].position[1]"))
        ambient=VE.param(fx,Symbol("lights[0].color[1]"))
        @test VE.isfollowing(p)
        VE.updatesceneprogram!(target,c.source,c,12,24)
        VE.syncsceneinputs!(c.source,c)
        @test VE.valueat(p,12)==12
        p.input=nothing
        VE.movekey!(p.curve[],1,0,20)
        ambient.input=nothing
        VE.movekey!(ambient.curve[],1,0,.8)
        VE.applysceneparams!(c.source,c,12)
        @test VE.Makie.get_lights(scene)[1].position[1]==20
        @test VE.Makie.Colors.red(scene.compute[:ambient_color][])≈.8f0
        saved=joinpath(dir,"edit.videoedit");VE.saveproject(saved,seq)
        restored=VE.loadproject(saved)
        @test VE.valueat(VE.param(VE.findslot(first(restored.clips),:scene),p.name),12)==20
        # Reset/bypass restore the source, including fields the updater doesn't touch.
        VE.restoresceneprogram!(target)
        @test VE.Makie.Colors.red(scene.compute[:ambient_color][])≈.2f0
        fx.enabled[]=false
        VE.updatesceneprogram!(target,c.source,c,5,24)
        @test VE.Makie.get_lights(scene)[1].position[1]==5
    end
end

@testset "dialogue spans follow cuts, gaps, speed and pending takes" begin
    mktempdir() do dir
        file=joinpath(dir,"scene.jl")
        write(file,"using Makie; buildscene(canvas,args)=(scene=Scene(;size=canvas),update! = (frame,fps)->nothing)")
        build=VE.programscene(file)
        c=VE.sceneclip(VE.buildscene(build);build,frames=48,canvas=(16,16),framerate=24)
        left=VE.Clip(c.source;src_in=0,src_out=24,start=0)
        right=VE.Clip(c.source;src_in=24,src_out=48,start=48)
        seq=VE.Sequence([left,right],24)
        n=VE.Narration("A line crossing a cut",.5;anchor=c.source)
        append!(n.samples,fill(.1f0,48000));n.rate=48000;push!(seq.narration,n)
        @test VE.dialoguespans(seq)==[(index=1,start=.5,stop=1.),(index=1,start=2.,stop=2.5)]
        span=first(VE.dialoguespans(seq))
        @test VE.dialogueclock(seq,n,span)==(0.,1.)
        right.rate=2
        @test last(VE.dialoguespans(seq)).stop==2.25
        pending=VE.narrationcopy(n);seq.narration[1]=pending
        @test !isempty(VE.dialoguespans(seq))
        empty!(seq.narration)
        @test isempty(VE.dialoguespans(seq))
        empty!(seq.clips)
        @test VE.dialogueclock(seq,n,span)===nothing
    end
end
