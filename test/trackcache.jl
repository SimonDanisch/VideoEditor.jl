using Test, VideoEditor, RayMakie, Mantle
import VideoEditor as VE
VE.usebackend!(RayMakie)

@testset "track caches record mutations at frame granularity" begin
    build = VE.scenebuild(:bar, (64, 48))
    base = VE.sceneclip(VE.buildscene(build); build, frames = 4, canvas = (64, 48), backend = :GLMakie)
    label = VE.sceneclip(VE.buildscene(build); build, frames = 4, canvas = (64, 48), backend = :GLMakie)
    label.track = 2
    seq = VE.Sequence([base, label], 30)
    seq.canvas = (64, 48)
    VE.enabletrackcache!(seq, mktempdir())
    engine = VE.FxEngine(Mantle.defaultbackend())
    readers = Dict{String, Any}()
    out = zeros(VE.RGB{VE.N0f8}, 64, 48)
    draw(ns) = foreach(n -> VE.renderframe!(out, seq, n, readers, engine), ns)
    try
        draw(0:3)
        @test [seq.trackcaches[t].renders for t in 1:2] == [4, 4]
        draw(0:3)
        @test [seq.trackcaches[t].renders for t in 1:2] == [4, 4]
        VE.mutateframe!(seq, 2, 2) do
            nothing # an external scene mutation declares its exact scope here
        end
        draw(0:3)
        @test [seq.trackcaches[t].renders for t in 1:2] == [4, 5]
        @test seq.compositecache.renders == 5

        # An ordinary parameter mutation is observed without a UI or history.
        fx = VE.Effect(VE.OpacityEffect(1.0f0))
        VE.addslot!(label, fx)
        draw(0:3)
        p = VE.param(fx, :opacity)
        b = seq.trackcaches[1].renders; l = seq.trackcaches[2].renders
        VE.setframevalue!(p, 2, 0.0f0)
        draw(0:3)
        @test seq.trackcaches[1].renders == b
        @test seq.trackcaches[2].renders == l + 1
        @test VE.valueat(p, 1) == 1.0f0 && VE.valueat(p, 2) == 0.0f0 && VE.valueat(p, 3) == 1.0f0
        VE.setkey!(p, 2, 0.5f0)
        draw(0:3)
        @test seq.trackcaches[1].renders == b
        @test seq.trackcaches[2].renders == l + 2

        before = [seq.trackcaches[t].renders for t in 1:2]
        beforecomposite = seq.compositecache.renders
        for c in seq.clips
            c.source.nframes = 6
            c.src_out = 6
        end
        draw(0:5)
        @test [seq.trackcaches[t].renders for t in 1:2] == before .+ 2
        @test seq.compositecache.renders == beforecomposite + 2

        # Explicit cache metadata survives a project reopen, including dirty bits.
        VE.invalidatetrack!(seq, 2, 1:1)
        project = joinpath(mktempdir(), "test.videoedit")
        VE.saveproject(project, seq; checkpoint = false)
        loaded = VE.loadproject(project)
        @test VE.cachevalid(loaded.trackcaches[1], base.id, 1)
        @test !VE.cachevalid(loaded.trackcaches[2], label.id, 1)
        @test VE.cachevalid(loaded.trackcaches[2], label.id, 0)

        # Workers return the planes a frame was made of, and what they render
        # never lands in the snapshot: the job keeps its key and resumes.
        job = VE.renderjob(project, mktempdir(); portable = false)
        renderer = VE.farmrenderer(job)
        try
            worker = VE.FarmWorker("test", ids -> VE.farmframes!(renderer, ids))
            VE.renderfarm!(job, [worker])
            receipt = VE.JSON.parsefile(VE.farmreceiptpath(job.directory, 1))
            @test length(receipt["layers"]) == 2
            @test VE.snapshotkey(VE.jobsnapshot(job)) == job.key
            # a worker keeps no frames
            @test isempty(filter(isfile, [joinpath(r, f) for (r, _, fs) in walkdir(renderer.planes) for f in fs]))
            VE.adoptfarmcache!(loaded, job)
            @test VE.cachevalid(loaded.trackcaches[2], label.id, 1)
            # already complete: no frame is asked for again
            @test VE.renderfarm!(job, [VE.FarmWorker("none", ids -> error("asked for $ids"))])["completed"] == job.total
        finally
            VE.closefarm!(renderer)
        end

        # Moving a clip reuses its source frames and only recomposes.
        before = seq.trackcaches[1].renders
        base.start = 1
        draw(1:5)
        @test seq.trackcaches[1].renders == before
        revision = get(seq.trackcaches[1].revisions, (base.id, 0), UInt64(0))
        base.track = 3
        @test VE.cachevalid(seq.trackcaches[3], base.id, 0)
        @test get(seq.trackcaches[3].revisions, (base.id, 0), UInt64(0)) == revision
    finally
        VE.emptyengine!(engine)
        foreach(close, values(readers))
    end
end

@testset "mutations during an in-flight render stay dirty" begin
    cache = VE.TrackCache(mktempdir())
    revision = UInt64(0)
    VE.dirtyframes!(cache, 1, 2:2)
    VE.cachewrite!(cache, 1, 2, zeros(VE.PlanePixel, 4, 4); revision)
    @test !VE.cachevalid(cache, 1, 2)
    VE.cachewrite!(cache, 1, 2, zeros(VE.PlanePixel, 4, 4))
    @test VE.cachevalid(cache, 1, 2)
end

@testset "scene renderer defaults" begin
    build = VE.scenebuild(:bar, (64, 48))
    c = VE.sceneclip(VE.buildscene(build); build, canvas = (64, 48))
    @test c.source.backend === :RayMakie
    @test VE.renderopts(c.source)[:rasterize]
    VE.setrenderconfig!(c; raytrace = true, samples = 64)
    c.source.mode = :bake
    @test !VE.renderopts(c.source)[:rasterize]
    @test VE.renderopts(c.source)[:samples] == 64
end

@testset "a transition edit recomposites the frames it blends" begin
    build = VE.scenebuild(:bar, (64, 48))
    a = VE.sceneclip(VE.buildscene(build); build, frames = 8, canvas = (64, 48), backend = :GLMakie)
    b = VE.sceneclip(VE.buildscene(build); build, start = 8, frames = 8, canvas = (64, 48), backend = :GLMakie)
    seq = VE.Sequence([a, b], 30)
    seq.canvas = (64, 48)
    VE.enabletrackcache!(seq, mktempdir())
    engine = VE.FxEngine(Mantle.defaultbackend())
    readers = Dict{String, Any}()
    out = zeros(VE.RGB{VE.N0f8}, 64, 48)
    draw() = foreach(n -> VE.renderframe!(out, seq, n, readers, engine), 0:15)
    composited(ns) = [VE.cachevalid(seq.compositecache, 0, n) for n in ns]
    try
        draw()
        short = VE.transframes(VE.addtransition!(seq, 8; duration = 4))
        @test !any(composited(short)) && all(composited(setdiff(0:15, short)))
        draw()
        long = VE.transframes(VE.addtransition!(seq, 8; duration = 8))   # the old span and the new
        @test !any(composited(long)) && all(composited(setdiff(0:15, long)))
        draw()
        VE.removetransition!(seq, 8)
        @test !any(composited(long)) && all(composited(setdiff(0:15, long)))
    finally
        VE.emptyengine!(engine)
        foreach(close, values(readers))
    end
end

@testset "detaching a following parameter changes every frame" begin
    build = VE.scenebuild(:bar, (64, 48))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 10, canvas = (64, 48), backend = :GLMakie)
    fx = VE.Effect(VE.OpacityEffect(1.0f0))
    VE.addslot!(clip, fx)
    p = VE.param(fx, :opacity)
    VE.setkey!(p, 0, 1.0f0)
    VE.setkey!(p, 9, 0.5f0)
    seen = Any[]
    VE.on(_ -> push!(seen, p.changedframes), p.curve)
    # not following: the segments around the key
    VE.setkey!(p, 4, 0.8f0)
    @test last(seen) == 1:8
    # following its scene, it showed the scene's value on every frame
    p.input = VE.ParamInput(:copy, VE.SceneRef(:opacity))
    @test VE.isfollowing(p)
    VE.setkey!(p, 4, 0.7f0)
    @test last(seen) === nothing && !VE.isfollowing(p)
end
