using Test, VideoEditor, RayMakie, Mantle, OIDNRunner
import VideoEditor as VE
VE.usebackend!(RayMakie)
@testset "OIDN is a non-destructive editor effect" begin
    build = VE.programscene(joinpath(@__DIR__, "guide_scene.jl"))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 1, canvas = (128, 64))
    seq = VE.Sequence([clip], 30);seq.canvas = (128, 64)
    VE.enabletrackcache!(seq, mktempdir())
    VE.setrenderconfig!(clip; raytrace = true, samples = 2, device = "Mantle.defaultbackend()", fxaa = false)
    effect = VE.Effect(VE.OIDNDenoiseEffect());effect.enabled[] = false
    VE.addslot!(clip, effect);VE.addslot!(clip, VE.Effect(VE.ToneMapEffect()))
    engine = VE.FxEngine(Mantle.defaultbackend());readers = Dict{String, Any}()
    out = zeros(VE.RGB{VE.N0f8}, 128, 64)
    draw() = VE.renderframe!(out, seq, 0, readers, engine)
    try
        draw();before = copy(out)
        raw = VE.sourcecache(clip);file = VE.cachefile(raw, clip.id, 0);original = read(file)
        effect.enabled[] = true;draw()
        @test raw.renders == 1
        @test read(file) == original
        @test before != out
        VE.setvalue!(VE.param(effect, :strength), 0.5, 0);draw()
        @test raw.renders == 1
        @test read(file) == original
        project = joinpath(mktempdir(), "oidn.videoedit")
        VE.saveproject(project, seq; checkpoint = false)
        restored = VE.loadproject(project)
        @test VE.findslot(restored.clips[1], VE.OIDNDenoiseEffect) !== nothing
        # Upgrade an old guide bake by tracing primary rays only. Keep all
        # previously sampled radiance and coverage, even for obsolete guides.
        master = VE.cachedsourceframe(clip, 0)
        old = merge(master, (metadata = merge(master.metadata, Dict("albedo_valid" => false, "normal_valid" => false)),))
        VE.cachewrite!(raw, clip.id, 0, old)
        work = VE.hdrwork!(clip.source, Mantle.defaultbackend(), size(old.color))
        @test_throws ErrorException VE.oidn!(work, old, VE.OIDNDenoiseEffect())
        close(clip.source)
        upgraded = VE.refreshguides!(clip, 0)
        @test upgraded.color == master.color
        @test upgraded.coverage == master.coverage
        @test upgraded.metadata["albedo_valid"]
        @test all(ss -> ss.overlay_only || ss.film.iteration_index[] == 0, clip.source.live.screen.scene_states)
        draw()
        @test VE.cachedsourceframe(clip, 0).color == master.color
    finally
        VE.emptyengine!(engine);foreach(close, values(readers));close(clip.source)
    end
end
