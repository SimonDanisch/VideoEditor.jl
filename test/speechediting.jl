using Test
import VideoEditor as VE

@testset "speech providers and source-clock editing" begin
    mktempdir() do dir
        script = joinpath(dir,"scene.jl")
        write(script,"using Makie\nbuildscene(canvas,args) = Scene(;size=canvas,backgroundcolor=:black)")
        build = VE.programscene(script)
        c = VE.sceneclip(VE.buildscene(build);build,frames=10,canvas=(32,32),framerate=10)
        seq = VE.Sequence([c],10);seq.canvas=(32,32)
        ref = joinpath(dir,"reference.wav");write(ref,"test reference asset")
        seen = Ref{Any}()
        VE.registerspeechmodel!(:test_directed,"Test directed",n->begin
            seen[]=(n.text,n.voice,n.speech.direction,n.speech.reference,n.speech.seed)
            (Float32[.1,.2,.3,.4,.5,.6,.7,.8,.9,1],10)
        end;direction=true,reference=true)
        n = VE.Narration("Actual words",0,"actor";anchor=c.source,label="actor · line",
            speech=VE.SpeechSettings(model=:test_directed,direction="quiet, scared",reference=ref,reference_text="reference words",seed=42))
        VE.render!(n);push!(seq.narration,n)
        @test seen[] == ("Actual words","actor","quiet, scared",ref,42)
        @test VE.render(n) !== n
        @test VE.render(n).speech == n.speech
        @test VE.render(n).anchor === c.source
        @test length(n.samples)==10
        mix = zeros(Int16,2,10);VE.mixnarration!(mix,seq,0;rate=10)
        @test mix[1,:] ≈ round.(Int16,n.samples*32767) atol=1
        @test mix[1,:] == mix[2,:]
        right = VE.split!(seq,5)
        splitmix = zeros(Int16,2,10);VE.mixnarration!(splitmix,seq,0;rate=10)
        @test splitmix == mix
        # Delete the first half: remaining speech follows the surviving source range.
        VE.deleteclip!(seq,first(seq.clips))
        cutmix = zeros(Int16,2,5);VE.mixnarration!(cutmix,seq,0;rate=10)
        @test cutmix == mix[:,6:10]
        @test n.at == 0
        @test VE.narrationtime(seq,n)==0
        right.start = 2
        moved = zeros(Int16,2,7);VE.mixnarration!(moved,seq,0;rate=10)
        @test moved[:,1:2] == zeros(Int16,2,2)
        @test moved[:,3:7] == cutmix
        @test VE.narrationtime(seq,n) ≈ .2
        # Retiming uses the picture's source clock, including across clip boundaries.
        right.start = 0;right.rate=2
        fast = zeros(Int16,2,3);VE.mixnarration!(fast,seq,0;rate=10)
        @test fast[1,1:2] == mix[1,[6,8]]
        path=joinpath(dir,"speech.videoedit");VE.saveproject(path,seq)
        loaded=VE.loadproject(path);ln=only(loaded.narration)
        @test ln.anchor === only(loaded.clips).source
        @test ln.samples == n.samples
        @test ln.label == "actor · line"
        @test ln.speech.model === :test_directed
        @test ln.speech.direction == "quiet, scared"
        @test ln.speech.reference == ref
        @test ln.speech.seed == 42
        @test ref in VE.farminputs(loaded,[])
        @test VE.narrationtime(loaded,ln)==0
        changed=VE.narrationcopy(n;text="new words")
        @test isempty(changed.samples)
        @test changed.speech == n.speech
        @test changed.anchor === n.anchor
        movedn=VE.narrationcopy(n;at=.5,anchor=nothing,audio=true)
        @test movedn.samples == n.samples
        @test movedn.anchor === nothing
        @test n.at==0
        VE.registerspeechmodel!(:test_plain,"Test plain",_->(Float32[.1],10))
        unsupported=VE.narrationcopy(n;speech=VE.SpeechSettings(model=:test_plain,direction="whisper"))
        @test_throws ErrorException VE.render!(unsupported)
        missing=VE.narrationcopy(n;speech=VE.SpeechSettings(model=:missing_model))
        @test_throws ErrorException VE.render!(missing)
        VE.registerspeechmodel!(:test_invalid,"Test invalid",_->(Float32[NaN],10))
        @test_throws ErrorException VE.render!(VE.narrationcopy(n;speech=VE.SpeechSettings(model=:test_invalid)))
        # A line completely removed by a trim produces no voice samples.
        right.src_in=0;right.src_out=2;n.at=.5
        absent=zeros(Int16,2,5);VE.mixnarration!(absent,seq,0;rate=10)
        @test all(iszero,absent)
        @test VE.narrationtime(seq,n) === nothing
        for name in (:test_directed,:test_plain,:test_invalid)
            delete!(VE.SPEECH_MODELS,name)
        end
    end
end
