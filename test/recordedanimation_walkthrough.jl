isdefined(Main,:EditingWalkthroughActions) || include("editing_helpers.jl")
module RecordedAnimationWalkthrough
using Test
import VideoEditor as VE
const M=VE.Makie
const FI=Main.FakeInteraction
using Main.EditingWalkthroughActions

function run(project,out)
    mkpath(out)
    VE.GLMakie.activate!(visible=false)
    p=VE.Player(project;analysisbackend=VE.Mantle.defaultbackend(),audiopreview=false)
    try
        @test timedwait(() -> get(p.fxwidgets,:publishedframe,-1)==0,120)===:ok
        @test first(p.sequence.clips).source.live.backend===:RayMakie
        @test first(p.sequence.clips).source.live.screen.rasterize
        @test !(first(p.sequence.clips).source.live.screen.config.device isa VE.KA.CPU)
        VE.GLMakie.stop_renderloop!(p.screen;close_after_renderloop=false)
        M.disconnect!(p.screen,M.mouse_position);M.events(p.fig).hasfocus[]=false
        clip()=VE.selectedclip(p)
        fx()=VE.findslot(clip(),:scene)
        sections()=p.fxwidgets[Symbol(:fxsections_,fx().id)]
        data()=VE.param(fx(),Symbol("subject.color"))
        position()=VE.param(fx(),Symbol("subject.translation[1]"))
        actions=[click(()->timelinepos(p,9));click(()->center(button(p,"Scene")));
            reveal(p,()->sections().picker);click(()->center(sections().picker));click(()->menurow(sections().picker,"subject"));
            FI.Lazy(_->begin
                M.save(joinpath(out,"selected.png"),M.colorbuffer(p.screen))
                @test VE.sceneselection(p)[].object===:subject
                @test data().view!==nothing && M.scene_visible(data().view.control.blockscene)
                @test data().view.lane.visible[]
                @test M.to_value(data().view.lane[1]) isa VE.RecordedTrack
                @test isempty(data().view.lane.markkeys[])
                @test VE.valueat(data(),9)==fill(M.RGBAf(10/12,.2,.3,1),4,4)
                FI.Wait(0)
            end);
            click(()->center(p.fxwidgets[Symbol(:scene_reset_,fx().id,:_,data().name)]));
            FI.Lazy(_->begin
                @test VE.followsrecording(clip().source,data())
                @test only(data().curve[].keys).value===nothing
                @test VE.valueat(data(),9)==fill(M.RGBAf(10/12,.2,.3,1),4,4)
                FI.Wait(0)
            end);
            typefield(p,()->position().view.control,"0.4");
            FI.Lazy(_->begin
                @test !VE.isfollowing(position())
                @test VE.valueat(position(),9)≈.4
                @test VE.valueat(data(),9)==fill(M.RGBAf(10/12,.2,.3,1),4,4)
                FI.Wait(0)
            end);
            chord(M.Keyboard.z);
            FI.Lazy(_->begin
                @test VE.isfollowing(position())
                @test VE.valueat(position(),9)≈10/8
                FI.Wait(0)
            end);
            click(()->timelinepos(p,3));
            FI.Lazy(_->begin
                @test VE.valueat(data(),3)==fill(M.RGBAf(4/12,.2,.3,1),4,4)
                FI.Wait(0)
            end);
            FI.KeyPress(M.Keyboard.s);FI.Wait(.4);
            FI.Lazy(_->begin
                @test length(p.sequence.clips)==2
                @test first(p.sequence.clips).source===last(p.sequence.clips).source
                FI.Wait(0)
            end);
            click(()->timelinepos(p,9;scrub=false));
            click(()->timelinepos(p,9));
            click(()->center(button(p,"Scene")));
            FI.Lazy(_->begin
                @test clip()===last(p.sequence.clips)
                @test p.playhead[]==9
                @test position().view.control.stored_string[]=="1.25"
                @test VE.sourceframe(clip(),9)==9
                @test VE.valueat(data(),9)==fill(M.RGBAf(10/12,.2,.3,1),4,4)
                @test Base.invokelatest(getproperty,parentmodule(clip().source.root.builder),:steps)[]==0
                FI.Wait(0)
            end)]
        recordactions(p,out,"recorded_arrays_edit_seek_cut",actions)
        saved=joinpath(out,"edited.videoedit");VE.saveproject(saved,p.sequence)
        back=VE.loadproject(saved)
        @test length(back.clips)==2
        @test first(back.clips).source===last(back.clips).source
        @test VE.valueat(VE.param(VE.findslot(last(back.clips),:scene),Symbol("subject.color")),9)==fill(M.RGBAf(10/12,.2,.3,1),4,4)
        @test previewmatches(p,p.previewaxis,p.frame[])
        foreach(c->close(c.source),back.clips)
    finally
        close(p)
    end
end
end
