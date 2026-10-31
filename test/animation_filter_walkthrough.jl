isdefined(Main, :EditingWalkthroughActions) || include("editing_helpers.jl")
module AnimationFilterWalkthrough
using Test
import VideoEditor as VE
const M = VE.Makie
const FI = Main.FakeInteraction
using Main.EditingWalkthroughActions

@testset "animation filter follows source time, selected track, edits and Undo" begin
    mktempdir() do dir
        out = get(ENV, "VIDEOEDITOR_QA_OUTPUT", dir); mkpath(out)
        script = joinpath(dir, "activity.jl")
        write(script, """
        using Makie
        function buildscene(canvas,args)
            scene=Scene(;size=canvas,camera=cam3d!,backgroundcolor=:navy)
            actor=mesh!(scene,Rect3f(Vec3f(-.3),Vec3f(.6));name=:actor,color=:orange)
            values=Ref((sway=0.,height=0.,still=1.))
            original(f,fps)=(sway=min(f,24)/fps,height=max(0,f-24)/fps,still=1.)
            controls=[(name=:actor_performance,label="Actor · acting",sample=original,
                apply! = (edits,f,fps)->(values[]=merge(original(f,fps),NamedTuple(edits))))]
            update! = (f,fps)->begin
                translate!(actor,Vec3f(values[].sway,0,values[].height))
                update_cam!(scene,Vec3f(4,-6,4),Vec3f(0),Vec3f(0,0,1))
            end
            activeparams=(a,b,fps)->Symbol[
                original(a,fps)[k]!=original(b,fps)[k] ? Symbol("actor_performance.",k) : :unused
                for k in (:sway,:height)]
            return (;scene,update!,controls,activeparams)
        end
        """)
        build = VE.programscene(script)
        backend = haskey(VE.BACKENDS,:RayMakie) ? :RayMakie : :GLMakie
        c = VE.sceneclip(VE.buildscene(build);build,frames=48,canvas=(160,120),framerate=24,backend)
        backend === :RayMakie && (c.source.screenopts[:rasterize]=true)
        second = VE.withfields(c;id=VE.freshid(),track=2,
            effects=[copy(fx;id=VE.freshid()) for fx in c.effects])
        keyed = VE.Param(Symbol("actor_performance.still"),"Still",1.;range=(0.,5.))
        VE.setkey!(keyed,0,1.);VE.setkey!(keyed,47,3.)
        push!(VE.findslot(second,:scene).params,keyed)
        seq=VE.Sequence([c,second],24)
        path=joinpath(dir,"activity.videoedit");VE.saveproject(path,seq)
        VE.GLMakie.activate!(visible=false)
        p=VE.Player(path;analysisbackend=VE.Mantle.defaultbackend(),gpupreview=false,audiopreview=false)
        try
            @test timedwait(() -> get(p.fxwidgets,:publishedframe,-1)==0,60)===:ok
            browse=button(p,"Choose file…")
            @test !M.scene_visible(browse.blockscene)
            VE.GLMakie.stop_renderloop!(p.screen;close_after_renderloop=false)
            M.disconnect!(p.screen,M.mouse_position);M.events(p.fig).hasfocus[]=false
            clip()=VE.selectedclip(p)
            fx()=VE.findslot(clip(),:scene)
            sections()=p.fxwidgets[Symbol(:fxsections_,fx().id)]
            prm(name)=VE.param(fx(),Symbol("actor_performance.",name))
            visible(name)=prm(name).view!==nothing && prm(name).view.control.blockscene.visible[]
            function trackpos(frame,track)
                axis=p.timeline.axis;lim=axis.finallimits[];vp=axis.scene.viewport[]
                y=sum(VE.trackband(p.sequence,track,VE.ntracks(p.sequence)))/2
                M.Point2f(vp.origin[1]+(frame/24-lim.origin[1])/lim.widths[1]*vp.widths[1],
                    vp.origin[2]+(y-lim.origin[2])/lim.widths[2]*vp.widths[2])
            end
            actions=[click(()->trackpos(6,1));click(()->center(button(p,"Animation")));
                FI.Lazy(_->begin
                    @test sections().onlyactive[]
                    @test visible(:sway) && !visible(:height) && !visible(:still)
                    @test clip().track==1
                    @test browse.clicks[]==0
                    FI.Wait(0)
                end);
                click(()->timelinepos(p,30));FI.Lazy(_->begin
                    @test !visible(:sway) && visible(:height) && !visible(:still)
                    FI.Wait(0)
                end);
                reveal(p,()->sections().activecheck);click(()->center(sections().activecheck));
                FI.Lazy(_->begin
                    @test !sections().onlyactive[]
                    @test visible(:sway) && visible(:height) && visible(:still)
                    FI.Wait(0)
                end);
                click(()->timelinepos(p,12));
                click(()->center(p.fxwidgets[Symbol(:kfacc_,fx().id,:_,prm(:still).name)][2]));
                typefield(p,()->prm(:still).view.control,"3");
                reveal(p,()->sections().activecheck);click(()->center(sections().activecheck));
                FI.Lazy(_->begin
                    @test sections().onlyactive[] && visible(:still)
                    @test VE.isanimated(prm(:still))
                    FI.Wait(0)
                end);
                click(()->timelinepos(p,6));FI.Lazy(_->begin
                    @test visible(:still)
                    @test VE.valueat(prm(:still),6)≈2
                    FI.Wait(0)
                end);
                click(()->trackpos(6,2));click(()->center(button(p,"Animation")));
                FI.Lazy(_->begin
                    @test clip().track==2
                    @test sections().onlyactive[] && visible(:still)
                    @test VE.valueat(prm(:still),6)≈1+12/47
                    FI.Wait(0)
                end);
                click(()->trackpos(6,1));FI.Lazy(_->begin
                    @test clip().track==1 && visible(:still) && visible(:sway)
                    @test !visible(:height)
                    FI.Wait(0)
                end);
                click(()->trackpos(6,2));FI.Lazy(_->begin
                    @test clip().track==2 && visible(:still) && !visible(:height)
                    FI.Wait(0)
                end)]
            recordactions(p,out,"filter_seek_toggle_tracks",actions)
            scene=clip().source.live
            @test VE.activeanimationparams(first(p.sequence.clips),VE.findslot(first(p.sequence.clips),:scene),60)==Set{Symbol}()
            # A trimmed clip must not advertise keys that ended before its source in-point.
            trimmed=VE.withfields(first(p.sequence.clips);src_in=24,src_out=48,start=0)
            @test !(Symbol("actor_performance.still") in VE.activeanimationparams(trimmed,VE.findslot(trimmed,:scene),6))
            # Undo restores keys in place and updates the filtered form.
            actions=[click(()->trackpos(6,1));click(()->center(button(p,"Animation")));
                chord(M.Keyboard.z);chord(M.Keyboard.z);
                FI.Lazy(_->begin
                    @test !VE.isanimated(prm(:still))
                    @test !visible(:still) && visible(:sway)
                    @test clip().source.live===scene
                    FI.Wait(0)
                end)]
            recordactions(p,out,"filter_undo",actions)
            VE.saveproject(path,p.sequence)
            restored=VE.loadproject(path)
            @test !VE.isanimated(VE.param(VE.findslot(first(restored.clips),:scene),Symbol("actor_performance.still")))
            @test VE.isanimated(VE.param(VE.findslot(last(restored.clips),:scene),Symbol("actor_performance.still")))
        finally
            close(p)
        end
    end
end
end
