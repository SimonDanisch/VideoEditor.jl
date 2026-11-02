using Test
import VideoEditor as VE

@testset "graded layers and HDR have independent retention" begin
    mktempdir() do dir
        build = VE.scenebuild(:bar, (4, 3))
        clip = VE.sceneclip(VE.buildscene(build); build, frames=1, canvas=(4, 3))
        seq = VE.Sequence([clip], 30)
        VE.enabletrackcache!(seq, joinpath(dir, "worker"))
        plane, raw = VE.trackcache(clip), VE.sourcecache(clip)
        pixels = fill(VE.RGBA{VE.N0f8}(0.25, 0, 0, 0.5), 4, 3)
        bytes = UInt8[1, 2, 3, 4]
        function populate()
            VE.cachewrite!(plane, clip.id, 0, pixels)
            path = VE.cachefile(raw, clip.id, 0)
            mkpath(dirname(path)); write(path, bytes)
            push!(get!(() -> BitSet(), raw.valid, clip.id), 0)
            return path
        end
        path = populate()
        # With HDR disabled, even an unreadable temporary source can be
        # discarded: it must not be read into the returned payload.
        Sys.isunix() && chmod(path, 0o000)
        layer = only(VE.farmlayers!(seq, 0; hdr=false))
        @test !isempty(layer.png) && isempty(layer.hdr)
        @test VE.PNGFiles.load(IOBuffer(layer.png)) == pixels
        @test !isfile(path)
        @test !VE.cachevalid(plane, clip.id, 0)
        @test !VE.cachevalid(raw, clip.id, 0)

        populate()
        full = only(VE.farmlayers!(seq, 0; hdr=true))
        @test full.png == layer.png && full.hdr == bytes
        @test !isfile(VE.cachefile(raw, clip.id, 0))
        populate()
        @test isempty(VE.dropplanes!(seq, 0))
        @test !VE.cachevalid(plane, clip.id, 0) && !VE.cachevalid(raw, clip.id, 0)
        @test_throws ArgumentError VE.renderjob("not-read", "not-created"; planes=false, hdr=true)

        # Old snapshots retain exactly their former policy.
        legacy = joinpath(dir, "legacy")
        mkpath(joinpath(legacy, "job"))
        meta = Dict("size"=>[4, 3], "total"=>1, "fps"=>30.0,
                    "julia"=>string(VERSION), "backends"=>String[],
                    "packages"=>String[], "relocations"=>Dict{String,String}(), "planes"=>true)
        metadata = joinpath(legacy, "job", "job.toml")
        open(io -> VE.TOML.print(io, meta), metadata, "w")
        @test VE.RenderJob(legacy).planes && VE.RenderJob(legacy).hdr
        meta["planes"] = false
        open(io -> VE.TOML.print(io, meta), metadata, "w")
        @test !VE.RenderJob(legacy).planes && !VE.RenderJob(legacy).hdr
        meta["planes"], meta["hdr"] = true, false
        open(io -> VE.TOML.print(io, meta), metadata, "w")
        @test VE.RenderJob(legacy).planes && !VE.RenderJob(legacy).hdr

        fixture(name, planes, hdr) = VE.RenderJob(joinpath(dir,name), name, (4,3), 1, 30.0,
            string(VERSION), String[], String[], Dict{String,String}(), planes, hdr)
        source, pngonly, allbuffers = fixture("source",true,true), fixture("png",true,false), fixture("full",true,true)
        entry = VE.commitlayer!(source.directory, VE.FarmLayer(1,clip.id,0,layer.png,bytes))
        kept = only(VE.carrylayers!(pngonly,source,[entry]))
        @test !haskey(kept,"hdr_sha256") && haskey(entry,"hdr_sha256")
        @test samefile(VE.cachefile(VE.farmplanes(source.directory,1),clip.id,0),
                       VE.cachefile(VE.farmplanes(pngonly.directory,1),clip.id,0))
        @test !isfile(VE.cachefile(VE.farmsources(pngonly.directory,1),clip.id,0))
        keptfull = only(VE.carrylayers!(allbuffers,source,[entry]))
        @test keptfull["hdr_sha256"] == entry["hdr_sha256"]
        @test samefile(VE.cachefile(VE.farmsources(source.directory,1),clip.id,0),
                       VE.cachefile(VE.farmsources(allbuffers.directory,1),clip.id,0))
        flattened = fixture("flattened",false,false)
        @test isempty(VE.carrylayers!(flattened,source,[entry]))
        @test_throws r"composites only" VE.carryframes!(pngonly,flattened)
        @test_throws r"no retained HDR" VE.carryframes!(allbuffers,pngonly)

        # Save As shares immutable data; rendering one language cannot overwrite
        # the other language's linked image.
        VE.cachewrite!(plane,clip.id,0,pixels)
        english, german = joinpath(dir,"english.videoedit"), joinpath(dir,"german.videoedit")
        VE.saveproject(english,seq)
        saved_identity = VE.farmprojectcontent(english)
        roundtrip = VE.loadproject(english)
        VE.saveproject(german,roundtrip)
        @test VE.farmprojectcontent(german) == saved_identity
        first = VE.cachefile(plane,clip.id,0)
        sha = VE.filehash(first)
        VE.saveproject(german,seq)
        second = VE.cachefile(plane,clip.id,0)
        @test samefile(first,second)
        reopened = VE.loadproject(english)
        VE.saveproject(german,reopened)
        @test samefile(first,second)
        VE.cachewrite!(plane,clip.id,0,fill(VE.RGBA{VE.N0f8}(0,0.25,0,0.5),4,3))
        @test VE.filehash(first) == sha
        @test !samefile(first,second)

        # PNG-only reuse compares the finished effect stack. A held scene
        # with animated grading still needs each distinct graded picture.
        recipe = joinpath(dir,"held.jl")
        write(recipe,"using Makie; buildscene(canvas,args)=(scene=Scene(;size=canvas),)")
        heldbuild = VE.programscene(recipe)
        held = VE.sceneclip(VE.buildscene(heldbuild);build=heldbuild,frames=2,canvas=(4,3))
        VE.setrenderconfig!(held;raytrace=true,samples=2)
        grade = VE.Effect(VE.ToneMapEffect())
        VE.addslot!(held,grade)
        heldseq = VE.Sequence([held],30)
        gamma = only(p for p in grade.params if p.name === :gamma)
        gamma.curve[] = VE.AnimCurve{Float64}([
            VE.Keyframe{Float64}(0,1.5,:hold),VE.Keyframe{Float64}(1,2.2,:hold)],:hold)
        @test VE.framekey(held,0) == VE.framekey(held,1)
        @test VE.trackframekey(held,0) != VE.trackframekey(held,1)
        @test VE.distinctframes(heldseq,0:1;hdr=true) == [0]
        @test VE.distinctframes(heldseq,0:1;hdr=false) == [0,1]
        VE.enabletrackcache!(heldseq,joinpath(dir,"held-cache"))
        heldcache = VE.trackcache(held)
        VE.cachewrite!(heldcache,held.id,0,pixels)
        @test VE.twintrackframe(heldcache,held,1) === nothing
        @test VE.farmneededframes(heldseq) == [1]
        gamma.curve[] = VE.AnimCurve{Float64}([VE.Keyframe{Float64}(0,1.5,:hold)],:hold)
        VE.cachewrite!(heldcache,held.id,0,pixels)
        @test VE.twintrackframe(heldcache,held,1) == 0
        @test isempty(VE.farmneededframes(heldseq))
    end
end
