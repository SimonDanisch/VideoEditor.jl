using Test
import VideoEditor as VE

@testset "farm reuses native sources without manually moving cached frames" begin
    mktempdir() do dir
        build=VE.scenebuild(:bar,(4,4))
        picture=VE.sceneclip(VE.buildscene(build);build,frames=3,canvas=(4,4),backend=:RayMakie)
        labels=VE.sceneclip(VE.buildscene(build);build,frames=3,canvas=(4,4),backend=:RayMakie)
        labels.track=2
        VE.setrenderconfig!(picture;raytrace=true,samples=256)
        VE.setrenderconfig!(labels;raytrace=false)
        seq=VE.Sequence([picture,labels],30);seq.canvas=(4,4)
        VE.enabletrackcache!(seq,dir)
        function mark(cache,id,frame)
            file=VE.cachefile(cache,id,frame);mkpath(dirname(file))
            write(file,UInt8[1]) # bookkeeping-only fixture: never decoded/rendered
            push!(get!(() -> BitSet(),cache.valid,UInt64(id)),frame)
            delete!(get!(() -> BitSet(),cache.dirty,UInt64(id)),frame)
        end
        @test VE.farmneededframes(seq)==[0,1,2]
        raw=VE.sourcecache(picture)
        mark(raw,picture.id,0)
        @test VE.farmneededframes(seq)==[1,2] # dirty labels do not retrace a valid master
        clock=VE.Param(:test_clock,"Test clock",0f0)
        clock.input=VE.ParamInput(:identity,VE.ParamRef(:test_clock))
        clock.input.from[1]=picture
        push!(VE.findslot(picture,:scene).params,clock)
        @test !VE.farmneedsframe(seq,0) # a file-series clock may refer to its own clip
        clock.input.from[1]=labels
        @test VE.farmneedsframe(seq,0) # other-clip dependencies stay conservative
        clock.input.from[1]=picture
        scalarinput=clock.input
        clock.input=VE.ParamInput(:identity,VE.ClipRef(picture.id))
        clock.input.from[1]=picture
        @test VE.farmneedsframe(seq,0) # even a self picture can sample other frames
        clock.input=scalarinput
        mark(VE.trackcache(picture),picture.id,1)
        @test VE.farmneededframes(seq)==[2]
        VE.invalidateclip!(picture;frames=0:0,source=false)
        @test VE.farmneededframes(seq)==[2] # grade-only edit keeps raw HDR
        VE.invalidateclip!(picture;frames=0:0)
        @test VE.farmneededframes(seq)==[0,2]
        mark(seq.compositecache,0,2)
        @test VE.farmneededframes(seq)==[0]
        @test VE.farmneededframes(seq;frames=[0,2])==[0]
        @test VE.farmneedsframe(seq,3)
        VE.setrenderconfig!(picture;raytrace=false)
        @test VE.farmneedsframe(seq,0) # no master-only shortcut for uncached raster scenes
        mark(VE.trackcache(picture),picture.id,0)
        mark(VE.trackcache(labels),labels.id,0)
        @test !VE.farmneedsframe(seq,0)
        farmdir=joinpath(dir,"farm");mkpath(farmdir)
        job=VE.RenderJob(farmdir,"cache-reuse-test",(4,4),3,30.0,string(VERSION),String[],String[],Dict{String,String}())
        png=IOBuffer()
        VE.PNGFiles.save(png,fill(VE.RGB{VE.N0f8}(0,0,0),4,4))
        bytes=take!(png);seen=Int[]
        worker=VE.FarmWorker("test",ids->[begin
            push!(seen,n);(frame=n,png=bytes,seconds=0.0,layers=VE.FarmLayer[])
        end for n in ids])
        status=VE.renderfarm!(job,[worker];reuse=seq)
        @test sort(seen)==[1,2]
        @test status["reused"]==1 && status["completed"]==status["total"]==2
        @test !isfile(VE.farmframepath(farmdir,0)) # no fabricated job receipts/frames
        @test VE.cachevalid(VE.trackcache(picture),picture.id,0)
        never=VE.FarmWorker("already cached",_ -> error("must not rerender"))
        @test VE.renderfarm!(job,[never];reuse=seq)["completed"]==2
    end
end
