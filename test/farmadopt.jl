using Test
import VideoEditor as VE

@testset "immutable farm adoption preserves storage and revisions" begin
    mktempdir() do dir
        source = joinpath(dir,"returned-plane")
        write(source, "immutable original")
        cache = VE.TrackCache(joinpath(dir,"cache"))
        id = UInt64(17)
        sha = VE.filehash(source)
        @test VE.adoptplane!(cache,id,0,source,sha)
        target = VE.cachefile(cache,id,0)
        @test read(target)==read(source)
        @test VE.cachevalid(cache,id,0)
        @test VE.adoptplane!(cache,id,0,source,sha) # repeat an already linked adoption
        if Sys.isunix()
            @test stat(target).inode==stat(source).inode
        end
        @test !VE.adoptplane!(cache,id,0,source,"bad hash")
        @test read(target,String)=="immutable original"
        # The ordinary cache writer's rename leaves the job's linked file intact.
        VE.cachewrite!(cache,id,0,fill(VE.RGBA{VE.N0f8}(1,0,0,1),2,2))
        @test read(source,String)=="immutable original"
        @test VE.filehash(source)==sha
        @test VE.adoptplane!(cache,id,0,source,sha)
        @test read(target,String)=="immutable original"
        @test !VE.adoptplane!(cache,id,1,joinpath(dir,"missing"),sha)
    end
end
