using Test
import VideoEditor as VE

@testset "bounded PNG prefetch respects edits and file validity" begin
    mktempdir(dirname(Base.active_project())) do dir
        build = VE.scenebuild(:bar, (4,3))
        clip = VE.sceneclip(VE.buildscene(build); build, frames = 8, canvas = (4,3))
        seq = VE.Sequence([clip], 30)
        VE.enabletrackcache!(seq, dir)
        cache = VE.trackcache(clip)
        for n in 0:7
            VE.cachewrite!(cache, clip.id, n, fill(VE.PlanePixel(n/8,0,0,1),4,3))
        end
        VE.prefetchframeplanes!(seq, 0)
        @test length(cache.pending) <= 4
        @test all(==(VE.PlanePixel(0,0,0,1)), VE.cachedtrackframe(clip,0))
        # An edit between enqueue and consumption must not return the old PNG.
        VE.dirtyframes!(cache, clip.id, [1])
        replacement = fill(VE.PlanePixel(0,1,0,0.5),4,3)
        VE.cachewrite!(cache,clip.id,1,replacement)
        replacement[1] = VE.PlanePixel(1,0,1,1)
        @test all(==(VE.PlanePixel(0,1,0,0.5)), VE.cachedtrackframe(clip,1))
        # A hot decoded plane cannot conceal a missing file.
        rm(VE.cachefile(cache,clip.id,1))
        @test VE.cachedtrackframe(clip,1) === nothing
        VE.finishcacheprefetch!(seq)
        @test isempty(cache.pending)
    end
end
