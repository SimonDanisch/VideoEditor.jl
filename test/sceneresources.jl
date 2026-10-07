using Test
import VideoEditor as VE
import RayMakie
VE.usebackend!(RayMakie)

@testset "closed raster sources release resources and can reopen" begin
    mktempdir() do dir
        file = joinpath(dir, "scene.jl")
        write(
            file, """
            using Makie
            function buildscene(canvas, args)
                scene = Scene(; size=canvas, camera=campixel!, backgroundcolor=:black)
                scatter!(scene, [Point2f(32, 24)]; color=:orange, markersize=12)
                update! = (frame, fps) -> nothing
                return (; scene, update!)
            end
            """
        )
        root = VE.programscene(file)
        clip = VE.sceneclip(
            VE.buildscene(root); build = root, frames = 2,
            canvas = (64, 48), framerate = 24
        )
        source = clip.source
        source.backend = :RayMakie
        source.screenopts = Dict(
            :rasterize => true, :shadows => false,
            :device => "Mantle.defaultbackend()"
        )
        atlas = VE.Makie.get_texture_atlas()
        expected = nothing
        for iteration in 1:3
            callbacks = length(atlas.font_render_callback)
            try
                VE.prerender!(source, clip, 0)
                screen = source.live.screen
                @test source.pending !== nothing
                @test screen.gfx_atlas_hook !== nothing
                if expected === nothing
                    expected = copy(source.pending)
                else
                    @test source.pending == expected
                end
                close(source)
                @test source.live === nothing && source.pending === nothing
                @test screen.gfx_atlas_hook === nothing
                @test isempty(screen.frame_plans)
                @test length(atlas.font_render_callback) == callbacks
                close(source) # Closing an already closed source is harmless.
            finally
                close(source)
            end
        end
    end
end
