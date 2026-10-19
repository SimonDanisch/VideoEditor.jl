using Test, VideoEditor, RayMakie, Mantle
import VideoEditor as VE
VE.usebackend!(RayMakie)

@testset "lossless source HDR and guides" begin
    frame = (
        color = fill(VE.RGB{Float32}(4.125f0, -0.125f0, 0.25f0), 4, 3),
        albedo = fill(VE.RGB{Float32}(0.2, 0.4, 0.6), 4, 3),
        normal = fill(VE.Vec3f(0, 0, 1), 4, 3), depth = fill(Inf32, 4, 3),
        coverage = ones(Float32, 4, 3), overlay = zeros(VE.RGBA{Float32}, 4, 3),
        metadata = Dict("color_space" => "linear-sRGB", "albedo_valid" => true),
    )
    cache = VE.TrackCache(mktempdir(); format = :hdr)
    VE.cachewrite!(cache, 1, 0, frame)
    restored = VE.readrenderframe(VE.cachefile(cache, 1, 0))
    @test all(getproperty(restored, k) == getproperty(frame, k) for k in VE.HDR_PLANES)
    @test restored.metadata == frame.metadata
    @test maximum(VE.red.(restored.color)) > 1
    @test minimum(VE.green.(restored.color)) < 0
end

@testset "HDR effects reuse the traced source master" begin
    build = Dict{String, Any}(
        "kind" => "rig", "rig" => Dict(
            "camera" => Dict("eye" => [0.0, -4.0, 1.5], "lookat" => [0.0, 0.0, 0.0]),
            "lights" => [Dict("type" => "ambient", "color" => [5.0, 5.0, 5.0])],
            "parts" => [
                Dict(
                    "name" => "test", "type" => "Mesh", "args" => ["cat.obj"],
                    "kwargs" => Dict("color" => ":red")
                ),
            ]
        )
    )
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 2, canvas = (64, 48))
    VE.setrenderconfig!(clip; raytrace = true, samples = 4, device = "Mantle.defaultbackend()", fxaa = false)
    seq = VE.Sequence([clip], 30);seq.canvas = (64, 48)
    VE.enabletrackcache!(seq, mktempdir())
    grade = VE.Effect(VE.LinearGradeEffect())
    denoise = VE.Effect(VE.HDRDenoiseEffect(strength = 0, iterations = 1))
    VE.addslot!(clip, grade);VE.addslot!(clip, denoise)
    VE.addslot!(clip, VE.Effect(VE.ToneMapEffect()))
    engine = VE.FxEngine(Mantle.defaultbackend()); readers = Dict{String, Any}()
    out = zeros(VE.RGB{VE.N0f8}, 64, 48)
    draw(n = 0) = VE.renderframe!(out, seq, n, readers, engine)
    try
        draw()
        master = VE.cachedsourceframe(clip, 0)
        cache = VE.sourcecache(clip)
        @test cache.renders == 1
        @test size(master.color) == (64, 48)
        @test maximum(VE.red.(master.color)) > 1
        original = read(VE.cachefile(cache, clip.id, 0))
        before = copy(out)
        VE.setvalue!(VE.param(grade, :exposure), -3, 0)
        draw()
        @test out != before
        @test cache.renders == 1
        @test read(VE.cachefile(cache, clip.id, 0)) == original
        VE.setvalue!(VE.param(denoise, :strength), 1, 0)
        draw()
        @test cache.renders == 1
        @test read(VE.cachefile(cache, clip.id, 0)) == original
        path = joinpath(mktempdir(), "hdr.videoedit")
        VE.saveproject(path, seq; checkpoint = false)
        loaded = VE.loadproject(path)
        @test VE.cachevalid(VE.sourcecache(loaded.clips[1]), clip.id, 0)
        @test VE.cachedsourceframe(loaded.clips[1], 0).color == master.color
        job = VE.renderjob(path, mktempdir(); portable = false)
        renderer = VE.farmrenderer(job)
        try
            worker = VE.FarmWorker("hdr-test", ids -> VE.farmframes!(renderer, ids))
            VE.renderfarm!(job, [worker]; frames = [0])
            receipt = VE.JSON.parsefile(VE.farmreceiptpath(job.directory, 0))
            @test haskey(only(receipt["layers"]), "hdr_sha256")
            empty!(VE.sourcecache(loaded.clips[1]).valid)
            # A new grade may invalidate the derived plane while the
            # source master remains eligible for adoption.
            VE.setvalue!(VE.param(VE.findslot(loaded.clips[1], VE.LinearGradeEffect), :exposure), -2, 0)
            VE.adoptfarmcache!(loaded, job)
            @test VE.cachevalid(VE.sourcecache(loaded.clips[1]), clip.id, 0)
            @test !VE.cachevalid(VE.trackcache(loaded.clips[1]), clip.id, 0)
        finally
            VE.closefarm!(renderer)
        end
        draw(1)
        @test cache.renders == 2
        derived = VE.trackcache(clip).renders
        VE.setframevalue!(VE.param(grade, :exposure), 0, -4)
        draw(0);draw(1)
        @test cache.renders == 2
        @test VE.trackcache(clip).renders == derived + 1
        VE.invalidateclip!(clip; frames = 0:0)
        @test !VE.cachevalid(cache, clip.id, 0)
        draw()
        @test cache.renders == 3
        clip.track = 2
        @test VE.cachevalid(VE.sourcecache(clip), clip.id, 0)
        @test VE.cachevalid(VE.sourcecache(clip), clip.id, 1)
    finally
        VE.emptyengine!(engine);foreach(close, values(readers))
        clip.source.live === nothing || close(clip.source.live.screen)
    end
end
