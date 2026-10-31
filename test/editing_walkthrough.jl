isdefined(Main, :EditingWalkthroughActions) || include("editing_helpers.jl")

module EditingWalkthroughTests
using Test
import VideoEditor as VE
const M = VE.Makie
const FI = Main.FakeInteraction
const K = M.Keyboard
using Main.EditingWalkthroughActions

@testset "walkthrough editing gestures" begin
    mktempdir() do dir
        out = get(ENV,"VIDEOEDITOR_QA_OUTPUT",dir); mkpath(out)
        script = joinpath(dir,"scene.jl")
        write(script,"""
        using Makie
        function buildscene(canvas,args)
            sc=Scene(;size=canvas,camera=cam3d!,backgroundcolor=:black,lights=[
                AmbientLight(RGBf(.2,.3,.4)),PointLight(RGBf(1,1,1),Point3f(1,2,3))])
            mesh!(sc,Rect3f(Vec3f(-0.4),Vec3f(0.8));color=:orange,name=:actor)
            update! = (frame,fps)->update_cam!(sc,Vec3f(4+frame/100,-6,4),Vec3f(0),Vec3f(0,0,1))
            return (scene=sc,update! = update!)
        end
        """)
        build=VE.programscene(script;objects=[Dict("label"=>"Actor","plots"=>["actor"],
                                                     "attributes"=>["color","alpha"])])
        c=VE.sceneclip(VE.buildscene(build);build,frames=120,canvas=(96,64),framerate=24)
        VE.seteffect!(c,VE.ColorEffect())
        # An identity grade keeps the fixture's colour visible in the recording.
        c.look = [Float32((i,j,k)[channel]-1)
                  for i in 1:2, j in 1:2, k in 1:2, channel in 1:3]
        VE.seteffect!(c,VE.LookEffect())
        seq=VE.Sequence([c],24); seq.canvas=(96,64)
        refs=[joinpath(dir,"voice$i.wav") for i in 1:2]
        for ref in refs
            run(`$(VE.FFMPEG_jll.ffmpeg()) -v error -y -f lavfi -i sine=frequency=220:sample_rate=48000 -t 0.5 $ref`)
        end
        requests=Any[]
        for (model,label) in ((:qa_voice_a,"QA voice A"),(:qa_voice_b,"QA voice B"))
            VE.registerspeechmodel!(model,label,n -> begin
                push!(requests,(n.text,n.voice,n.speech)); (fill(0.05f0,24000),48000)
            end;direction=true,reference=true)
        end
        for i in 1:2
            n=VE.Narration("Original $i",(i-1)*0.5,"actor$i";anchor=c.source,
                speech=VE.SpeechSettings(model=:qa_voice_a,reference=refs[i],reference_text="reference $i"))
            append!(n.samples,fill(0.01f0,24000));n.rate=48000;push!(seq.narration,n)
        end
        path=joinpath(dir,"edit.videoedit");VE.saveproject(path,seq)
        VE.GLMakie.activate!(visible=false)
        p=VE.Player(path;analysisbackend=VE.Mantle.defaultbackend(),gpupreview=false,audiopreview=false)
        try
            @test timedwait(() -> get(p.fxwidgets,:publishedframe,-1)==0,60)===:ok
            VE.GLMakie.stop_renderloop!(p.screen;close_after_renderloop=false)
            M.disconnect!(p.screen,M.mouse_position)
            M.events(p.fig).hasfocus[]=false
            clip()=first(p.sequence.clips)
            fx()=VE.findslot(clip(),:scene)
            camera()=VE.param(fx(),Symbol("camera.eye[1]"))
            position()=VE.param(fx(),Symbol("actor.translation[1]"))
            sections()=p.fxwidgets[Symbol(:fxsections_,fx().id)].cards
            previewmenu()=p.fxwidgets[:previewscale]
            actions=[click(() -> center(previewmenu()));
                     click(() -> menurow(previewmenu(),0.5));
                     FI.Lazy(_ -> begin
                         @test p.previewscale[] == 0.5
                         @test p.sequence.canvas == (96,64)
                         @test size(p.frame[]) == (96,64)
                         FI.Wait(0)
                     end);click(() -> center(previewmenu()));
                     click(() -> menurow(previewmenu(),0.25));
                     FI.Lazy(_ -> begin @test p.previewscale[] == 0.25;FI.Wait(0) end);
                     click(() -> center(previewmenu()));click(() -> menurow(previewmenu(),1.0));
                     FI.Lazy(_ -> begin @test p.previewscale[] == 1;FI.Wait(0) end)]
            recordactions(p,out,"preview_quality",actions)
            actions=[click(() -> center(button(p,"Scene")));
                     reveal(p,()->p.fxwidgets[Symbol(:fxsections_,fx().id)].activecheck);
                     click(()->center(p.fxwidgets[Symbol(:fxsections_,fx().id)].activecheck));
                     FI.Lazy(_ -> begin @test fx().card.open[]; FI.Wait(0) end);
                     click(() -> timelinepos(p,12));
                     FI.Lazy(_ -> begin
                         @test VE.isfollowing(camera())
                         @test camera().view.control.stored_string[] == "4.12"
                         @test clip().source.camera.eye[1] ≈ 4.12
                         FI.Wait(0)
                     end);
                     click(() -> timelinepos(p,36));
                     FI.Lazy(_ -> begin
                         @test camera().view.control.stored_string[] == "4.36"
                         @test clip().source.camera.eye[1] ≈ 4.36
                         FI.Wait(0)
                     end);
                     click(() -> timelinepos(p,0));
                     typefield(p,()->camera().view.control,"8");
                     click(() -> center(p.fxwidgets[Symbol(:kfacc_,fx().id,:_,camera().name)][2]));
                     click(() -> timelinepos(p,24));typefield(p,()->camera().view.control,"6");
                     FI.Lazy(_ -> begin
                         @test VE.isanimated(camera())
                         @test VE.valueat(camera(),24) ≈ 6
                         @test clip().source.camera.eye[1] ≈ 6
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"camera_keys",actions)
            actions=[reveal(p,()->header(first(sections())));
                     FI.Lazy(_ -> FI.MouseTo(header(first(sections())),0.05));
                     FI.Lazy(_ -> first(sections()).open[] ? FI.LeftClick() : FI.Wait(0));FI.Wait(0.25);
                     reveal(p,()->header(sections()[2]));
                     FI.Lazy(_ -> FI.MouseTo(header(sections()[2]),0.05));
                     FI.Lazy(_ -> sections()[2].open[] ? FI.Wait(0) : FI.LeftClick());FI.Wait(0.25);
                     typefield(p,()->position().view.control,"1");
                     click(() -> center(p.fxwidgets[Symbol(:kfacc_,fx().id,:_,position().name)][2]));
                     click(() -> timelinepos(p,48));typefield(p,()->position().view.control,"2");
                     FI.Lazy(_ -> begin
                         @test VE.isanimated(position())
                         @test VE.valueat(position(),48) ≈ 2
                         @test first(VE.sceneobjectplots(clip().source,:actor)).transformation.translation[][1] ≈ 2
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"actor_keys",actions)
            ambient()=VE.param(fx(),Symbol("lights[0].color[1]"))
            lightposition()=VE.param(fx(),Symbol("lights[1].position[1]"))
            lights()=M.get_lights(VE.targetscene(clip().source.live.target))
            beforelight=copy(p.frame[])
            actions=[click(() -> center(button(p,"Light")));
                     typefield(p,()->ambient().view.control,"0.8");
                     FI.Lazy(_ -> begin
                         @test VE.valueat(ambient(),p.playhead[])≈.8
                         @test M.Colors.red(VE.targetscene(clip().source.live.target).compute[:ambient_color][])≈.8f0
                         @test p.frame[] != beforelight
                         FI.Wait(0)
                     end);chord(K.z);
                     FI.Lazy(_ -> begin
                         @test M.Colors.red(VE.targetscene(clip().source.live.target).compute[:ambient_color][])≈.2f0
                         FI.Wait(0)
                     end);
                     typefield(p,()->lightposition().view.control,"3");
                     click(() -> center(p.fxwidgets[Symbol(:kfacc_,fx().id,:_,lightposition().name)][2]));
                     click(() -> timelinepos(p,72));typefield(p,()->lightposition().view.control,"5");
                     FI.Lazy(_ -> begin
                         @test VE.isanimated(lightposition())
                         @test VE.valueat(lightposition(),72)≈5
                         @test lights()[1].position[1]≈5
                         FI.Wait(0)
                     end);click(() -> timelinepos(p,60));
                     FI.Lazy(_ -> begin
                         @test VE.valueat(lightposition(),60)≈4
                         @test lights()[1].position[1]≈4
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"light_keys_undo",actions)
            grade()=VE.param(VE.findslot(clip(),:color),:brightness)
            beforegrade = copy(p.frame[])
            actions=[click(() -> center(button(p,"Colour")));
                     reveal(p,()->grade().view.control);
                     FI.Lazy(_ -> FI.MouseTo(FI.relative_pos(grade().view.control,(0.5,0.5)),0.05));
                     FI.LeftDown();FI.Lazy(_ -> FI.MouseTo(FI.relative_pos(grade().view.control,(0.7,0.5)),0.2));FI.LeftUp();
                     FI.Lazy(_ -> begin
                         @test p.frame[] != beforegrade
                         FI.Wait(0)
                     end);
                     click(() -> timelinepos(p,36));FI.KeyPress(K.s);FI.Wait(0.25);
                     FI.KeyPress(K.t);FI.Wait(0.25);
                     FI.Lazy(_ -> begin
                         @test length(p.sequence.clips)==2
                         @test grade().curve[].keys[1].value > 0
                         @test length(p.sequence.transitions)==1
                         FI.Wait(0)
                     end);chord(K.z);
                     FI.Lazy(_ -> begin @test isempty(p.sequence.transitions);FI.Wait(0) end);
                     FI.KeyPress(K.delete);FI.Wait(0.25);
                     FI.Lazy(_ -> begin @test length(p.sequence.clips)==1;FI.Wait(0) end);
                     chord(K.z);FI.KeyPress(K.c);FI.KeyPress(K.escape);FI.Wait(0.25);
                     FI.Lazy(_ -> begin
                         @test length(p.sequence.clips)==2
                         @test !p.cropmode[] && isempty(p.croprect[])
                         @test p.status[] == "tool cancelled"
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"cuts_grade_undo",actions)
            actions=[click(() -> center(button(p,"Speech")));
                     typefield(p,()->speechfield(p,"Words"),"Hallo, Chef!";guard=true);
                     typefield(p,()->speechfield(p,"Regie · delivery instructions"),"quiet, hesitant");
                     reveal(p,()->speechmenu(p,1));click(() -> center(speechmenu(p,1)));
                     click(() -> menurow(speechmenu(p,1),:qa_voice_b));
                     reveal(p,()->speechmenu(p,2));click(() -> center(speechmenu(p,2)));
                     click(() -> menurow(speechmenu(p,2),3));
                     reveal(p,()->button(p,"Render take"));click(() -> center(button(p,"Render take")));
                     FI.WaitUntil(()->!isempty(first(p.sequence.narration).samples);timeout=30);
                     FI.Lazy(_ -> begin
                         n=first(p.sequence.narration)
                         @test n.text=="Hallo, Chef!"
                         @test n.speech.direction=="quiet, hesitant"
                         @test n.speech.model===:qa_voice_b
                         @test n.voice=="actor2" && n.speech.reference==refs[2]
                         @test length(n.samples)==24000
                         @test last(requests)==(n.text,n.voice,n.speech)
                         FI.Wait(0)
                     end);click(() -> timelinepos(p,24));chord(K.s);
                     FI.Lazy(_ -> begin
                         back=VE.loadproject(path);n=first(back.narration)
                         @test n.speech.model===:qa_voice_b && n.speech.direction=="quiet, hesitant"
                         @test n.samples==first(p.sequence.narration).samples
                         @test n.anchor===first(back.clips).source
                         @test VE.isanimated(VE.param(VE.findslot(first(back.clips),:scene),Symbol("camera.eye[1]")))
                         @test VE.valueat(VE.param(VE.findslot(first(back.clips),:scene),lightposition().name),72)≈5
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"speech_model_regie_save",actions)
            # Opening through a file-manager drop is a real Makie event, without
            # depending on an operating-system native file dialog.
            actions=[FI.DropFiles([path]); FI.Wait(0.5);
                     FI.Lazy(_ -> begin
                         @test p.projectpath == path
                         @test length(p.sequence.clips) == 2
                         @test first(p.sequence.narration).text == "Hallo, Chef!"
                         @test first(p.sequence.narration).samples == first(VE.loadproject(path).narration).samples
                         FI.Wait(0)
                     end)]
            recordactions(p,out,"project_reopen",actions)
        finally
            close(p)
            for name in (:qa_voice_a,:qa_voice_b);delete!(VE.SPEECH_MODELS,name);end
        end
    end
end
end
