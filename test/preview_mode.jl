using Test
import VideoEditor as VE

module PreviewModeBackend
    import Makie

    mutable struct Screen
        rasterize::Bool
    end
    isprogressive(screen::Screen) = !screen.rasterize

    # An unrelated screen in a module that also exposes the new capability must
    # retain the older clear-keyword protocol when no method applies to it.
    struct LegacyScreen end
    Makie.colorbuffer(::LegacyScreen, ::typeof(Makie.GLNative); clear = true) = nothing
end

@testset "preview refinement follows runtime renderer mode" begin
    screen = PreviewModeBackend.Screen(true)
    @test !VE.progressive(screen)
    screen.rasterize = false
    @test VE.progressive(screen)
    screen.rasterize = true
    @test !VE.progressive(screen)
    @test VE.progressive(PreviewModeBackend.LegacyScreen())
    @test !VE.progressive(nothing)
end
