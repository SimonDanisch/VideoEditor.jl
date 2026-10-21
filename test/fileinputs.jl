# Run in the application's root environment. Caller supplies a fixture video
# and an output directory; no package/test environment is activated.
function test_fileinputs(video,directory)
    mkpath(directory)
    @testset "lazy file-series inputs" begin
        paths=[joinpath(directory,"input_$i.png") for i in 1:3]
        for (i,path) in enumerate(paths)
            VE.Makie.FileIO.save(path,fill(VE.RGBf(i/3,0,0),2,2))
        end
        ref=VE.FileSeriesRef(paths,[0,2,5])
        reader=VE.FileSeries(ref)
        a=VE.sampleinput(reader,0.5)
        @test a.weight==0.25f0
        @test length(reader.loaded)==2
        @test VE.sampleinput(reader,1).a===a.a
        @test VE.sampleinput(reader,-2).weight==0
        b=VE.sampleinput(reader,3.5)
        @test b.weight==0.5f0
        @test b.a===a.b
        @test sort!(collect(keys(reader.loaded)))==[2,3]
        @test VE.sampleinput(reader,99).weight==1
        @test VE.sampleinput(reader,0.5).weight==0.25f0 # arbitrary seek order
        @test_throws ArgumentError VE.sampleinput(reader,NaN)
        @test_throws ArgumentError VE.FileSeriesRef(paths,[0,0,5])
        packed=VE.inputrefdict(ref)
        restored=VE.inputreffromdict(packed)
        @test restored.paths==ref.paths && restored.times==ref.times
        @test restored.options==ref.options
        clip=VE.Clip(VE.VideoSource(video);src_in=0,src_out=3,start=0)
        fx=VE.Effect(VE.freshid(),:scene,true,VE.Param[])
        clock=VE.Param(:time,"time",1.)
        data=VE.Param{Any}(:volume,"volume",VE.Makie.Observable(VE.AnimCurve{Any}([VE.Keyframe{Any}(0,nothing,:hold)],:hold)),
            VE.Makie.Observable(false),nothing,VE.ParamInput(:sample,ref,VE.ParamRef(:time)),nothing)
        append!(fx.params,[clock,data]);VE.addslot!(clip,fx)
        seq=VE.Sequence([clip],10.)
        VE.bindinputs!(seq)
        @test VE.valueat(data,0).weight==0.5f0
        @test all(p->p in VE.farminputs(seq,String[]),paths)
        @test all(p->haskey(VE.externalinputs(clip),p),paths)
        project=joinpath(directory,"inputs.videoedit")
        VE.saveproject(project,seq)
        loaded=VE.loadproject(project)
        loadeddata=VE.param(VE.findslot(loaded.clips[1],:scene),:volume)
        @test VE.valueat(loadeddata,0).weight==0.5f0
        relocated=VE.farmrebase(packed,Dict(paths[1]=>joinpath(directory,"elsewhere.png")))
        @test relocated["paths"][1]==joinpath(directory,"elsewhere.png")
        package=VE.FileSeriesRef(paths,[0,2,5];package="GRMHDSnapshots",loader=:radiative_volume,
                                options=Dict("resolution"=>8))
        @test VE.inputpackages(package)==["GRMHDSnapshots"]
        @test VE.inputreffromdict(VE.inputrefdict(package)).loader==:radiative_volume
    end
end
