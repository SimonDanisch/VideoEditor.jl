isdefined(Main,:EditingWalkthroughActions) || include("editing_helpers.jl")
isdefined(Main,:ThumbnailWalkthroughActions) || include("thumbnail_helpers.jl")
module TrackPreviewWalkthrough
using Test
import VideoEditor as VE
const M=VE.Makie
const FI=Main.FakeInteraction
using Main.EditingWalkthroughActions

@testset "scene and sound previews through zoom, cuts and generated takes" begin
    mktempdir() do dir
        out=get(ENV,"VIDEOEDITOR_QA_OUTPUT",dir);mkpath(out)
        wav=joinpath(dir,"soundbite.wav")
        run(`$(VE.FFMPEG_jll.ffmpeg()) -v error -y -f lavfi -i sine=frequency=440:sample_rate=48000 -t 2 -ac 2 $wav`)
        script=joinpath(dir,"animation.jl")
        write(script,"""
        using Makie
        function buildscene(canvas,args)
            s=Scene(;size=canvas,camera=cam3d!,backgroundcolor=:navy)
            box=mesh!(s,Rect3f(Vec3f(-.4),Vec3f(.8));name=:actor,color=:orange)
            update! = (frame,fps)->begin
                translate!(box,Vec3f(frame/100,0,0))
                update_cam!(s,Vec3f(4,-6,4),Vec3f(0),Vec3f(0,0,1))
            end
            return (scene=s,update! = update!)
        end
        """)
        build=VE.programscene(script)
        backend=haskey(VE.BACKENDS,:RayMakie) ? :RayMakie : :GLMakie
        c=VE.sceneclip(VE.buildscene(build);build,frames=48,canvas=(160,120),framerate=24,backend)
        backend===:RayMakie && (c.source.screenopts[:rasterize]=true)
        c.source.soundtrack=wav
        seq=VE.Sequence([c],24)
        for at in (.1,.3)
            n=VE.Narration("A rendered line",at,"speaker";anchor=c.source)
            append!(n.samples,[Float32(.6sin(i*.07)) for i in 1:24000]);n.rate=48000
            push!(seq.narration,n)
        end
        path=joinpath(dir,"previews.videoedit");VE.saveproject(path,seq)
        VE.registerspeechmodel!(:qa_preview_wave,"QA preview voice",n->(fill(.15f0,12000),48000))
        VE.GLMakie.activate!(visible=false)
        p=VE.Player(path;analysisbackend=VE.Mantle.defaultbackend(),gpupreview=false,audiopreview=false)
        manager=p.fxwidgets[:trackpreviews]
        try
            VE.GLMakie.stop_renderloop!(p.screen;close_after_renderloop=false)
            M.disconnect!(p.screen,M.mouse_position);M.events(p.fig).hasfocus[]=false
            clip()=first(p.sequence.clips)
            lane=p.fxwidgets[:dialogue_timeline]
            @test timedwait(()->!isempty(clip().view.plot.wavepoints[]) && !isempty(lane.waveform[]),30)==:ok
            @test timedwait(()->manager.active!==nothing && lock(()->any(k->first(k)===:scene && k[4]>0 && manager.entries[k]!==nothing,keys(manager.entries)),manager.lock),300)==:ok
            @test length(lane.displayrows[])==2
            backend===:RayMakie && @test manager.active[2].screenopts[:rasterize]===true
            @test manager.active[2].live!==clip().source.live
            thumb=lock(manager.lock) do
                first(values(filter(kv->first(first(kv))===:scene && last(kv)!==nothing,manager.entries)))
            end
            @test size(thumb)==VE.scenethumbdims(clip().source)
            Main.ThumbnailWalkthroughActions.checkwaveformarrival(p,out)
            Main.ThumbnailWalkthroughActions.recordarrival(p,out)
            # A gesture arriving after a scene job was queued must park that
            # job at renderer entry, before constructing or seeking its scene.
            active=manager.active
            recordactions(p,out,"input_defers_queued_preview",[
                FI.MouseTo(center(p.previewaxis),0.05);
                FI.Lazy(_ -> begin
                    @test VE.scenethumbnail!(manager,p,clip(),0.,VE.scenepreviewidentity(clip().source)) isa VE.DeferredScenePreview
                    @test manager.active===active
                    FI.Wait(0)
                end)
            ])
            before=copy(p.frame[]);live=clip().source.live
            scene=VE.targetscene(live.target);frame=live.target.frame
            # Background thumbnails must not seek or modify the monitor's scene.
            sleep(.2)
            @test p.frame[]==before
            @test clip().source.live===live && live.target.frame==frame
            baked=joinpath(dir,"baked");mkpath(baked)
            bake=VE.Bake(baked,0:0,(160,120))
            VE.PNGFiles.save(VE.bakeframefile(bake,0),fill(VE.RGB{VE.N0f8}(1,0,0),160,120))
            clip().bake=bake
            provider=VE.scenethumbsfor(p.timeline,clip())
            provider(0.)
            @test timedwait(()->provider(0.)!==nothing,10)==:ok
            @test all(==(VE.RGB{VE.N0f8}(1,0,0)),provider(0.))
            clip().bake=nothing
            M.save(joinpath(out,"previews_ready.png"),M.colorbuffer(p.screen))
            audio=VE.audioenvelope(p,clip().source)
            actions=[click(()->timelinepos(p,12));FI.KeyPress(M.Keyboard.s);FI.Wait(.3);
                FI.Lazy(_->begin
                    @test length(p.sequence.clips)==2
                    @test all(c->VE.audioenvelope(p,c.source)===audio,p.sequence.clips)
                    @test VE.targetscene(clip().source.live.target)===scene
                    FI.Wait(0)
                end);
                click(()->center(button(p,"Speech")));
                reveal(p,()->speechmenu(p,1));click(()->center(speechmenu(p,1)));
                FI.TypeText("QA preview voice");FI.KeyPress(M.Keyboard.enter);FI.Wait(.3);
                FI.Lazy(_->begin
                    @test first(p.sequence.narration).speech.model===:qa_preview_wave
                    FI.Wait(0)
                end);
                reveal(p,()->button(p,"Render take"));click(()->center(button(p,"Render take")));
                FI.WaitUntil(()->length(first(p.sequence.narration).samples)==12000;timeout=30);
                FI.Wait(.3);
                FI.Lazy(_->begin
                    n=first(p.sequence.narration)
                    @test n.speech.model===:qa_preview_wave
                    @test n.rate==48000 && length(n.samples)==12000
                    @test !isempty(lane.waveform[])
                    FI.Wait(0)
                end)]
            recordactions(p,out,"cut_and_rerender",actions)
            M.xlims!(p.timeline.axis,1,2);sleep(.2)
            @test length(lane.displayrows[])==1
            @test VE.audioenvelope(p,clip().source)===audio
            @test !isempty(first(p.sequence.clips).source.soundtrack)
            @test manager.bytes<=manager.maxbytes
            M.save(joinpath(out,"zoomed_sound.png"),M.colorbuffer(p.screen))
        finally
            close(p);delete!(VE.SPEECH_MODELS,:qa_preview_wave)
        end
        @test istaskdone(manager.task)
        @test manager.active===nothing
    end
end
end
