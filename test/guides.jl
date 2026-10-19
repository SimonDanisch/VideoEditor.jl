using Test, VideoEditor, RayMakie, Mantle
import VideoEditor as VE
VE.usebackend!(RayMakie)
@testset "material guides align and skip cutouts" begin
    for hardware in (true, false), cutout in (false, true)
        build = VE.programscene(joinpath(@__DIR__, "guide_scene.jl"); args = Dict("cutout" => cutout))
        clip = VE.sceneclip(VE.buildscene(build); build, frames = 1, canvas = (128, 64))
        seq = VE.Sequence([clip], 30)
        VE.enabletrackcache!(seq, mktempdir())
        VE.setrenderconfig!(clip; raytrace = true, samples = 2, device = "Mantle.defaultbackend()", hw_accel = hardware, fxaa = false)
        clip.source.mode = :bake
        try
            VE.updatesource!(clip.source, clip, 0)
            VE.sceneframe!(clip.source, clip, (128, 64); exact = true)
            frame = VE.cachedsourceframe(clip, 0)
            @test frame.metadata["albedo_valid"]
            @test frame.metadata["normal_valid"]
            hit = map(n -> any(!iszero, n), frame.normal)
            @test count(hit) > 100
            colors = frame.albedo[hit]
            red = filter(a -> a.r > 0.5f0, colors)
            mixed = filter(a -> a.g > 0.5f0, colors)
            @test length(red) > 30 && length(mixed) > 30
            @test all(a -> max(abs(a.r - 0.8f0), abs(a.g - 0.1f0), abs(a.b - 0.05f0)) < 2.0f-6, red)
            @test all(a -> max(abs(a.r - 0.1f0), abs(a.g - 0.625f0), abs(a.b - 0.275f0)) < 2.0f-6, mixed)
            @test all(n -> abs(sum(abs2, n) - 1.0f0) < 2.0f-5, frame.normal[hit])
            @test all(iszero, frame.normal[.!hit])
            # Centre-ray guides are not a sample-filtered beauty coverage mask.
            # Use known asymmetric geometry to test orientation directly.
            redpixels = findall(a -> a.r > 0.5f0, frame.albedo)
            mixedpixels = findall(a -> a.g > 0.5f0, frame.albedo)
            centroid(points, axis) = sum(p[axis] for p in points) / length(points)
            @test centroid(redpixels, 1) < centroid(mixedpixels, 1)
            @test centroid(redpixels, 2) < centroid(mixedpixels, 2)
            # Negative control: flipping Y reverses the expected ordering.
            @test size(frame.albedo, 2) + 1 - centroid(redpixels, 2) > size(frame.albedo, 2) + 1 - centroid(mixedpixels, 2)
        finally
            close(clip.source)
        end
    end
end
