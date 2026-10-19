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

# The HDR stage runs on the device: each effect a kernel, picked by its type. The
# reference here is the host formula it replaced, per pixel.
@testset "the HDR stage is its reference formula, on the device" begin
    w, h = 24, 16
    tone(v, mode, gamma) = (v = max(0.0f0, v);
        t = mode == 1 ? v / (1.0f0 + v) : mode == 2 ? (v * (2.51f0 * v + 0.03f0)) / (v * (2.43f0 * v + 0.59f0) + 0.14f0) : v;
        clamp(t, 0.0f0, 1.0f0)^(1.0f0 / gamma))
    lum(c) = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
    function atrous(img, frame, iterations)
        taps = Float32[1, 4, 6, 4, 1] ./ 16
        result = copy(img)
        for it in 1:iterations
            input = result; result = similar(input); step = 1 << (it - 1)
            for y in 1:h, x in 1:w
                c = input[x, y]; n = frame.normal[x, y]; d0 = frame.depth[x, y]
                tr = tg = tb = cnt = 0.0f0
                for dy in -2:2, dx in -2:2
                    xx = clamp(x + dx * step, 1, w); yy = clamp(y + dy * step, 1, h)
                    d = frame.depth[xx, yy]
                    isfinite(d) == isfinite(d0) || continue
                    abs(frame.coverage[xx, yy] - frame.coverage[x, y]) < 0.01f0 || continue
                    nb = frame.normal[xx, yy]
                    nw = iszero(n) && iszero(nb) ? 1.0f0 : max(0.0f0, n[1] * nb[1] + n[2] * nb[2] + n[3] * nb[3])^64
                    dw = isfinite(d0) ? exp(-abs(d0 - d) / max(0.1f0, 0.02f0 * abs(d0))) : 1.0f0
                    s = input[xx, yy]
                    wt = taps[dx + 3] * taps[dy + 3] * nw * dw * exp(-abs(lum(c) - lum(s)) / max(0.1f0, abs(lum(c))))
                    tr += s.r * wt; tg += s.g * wt; tb += s.b * wt; cnt += wt
                end
                result[x, y] = cnt > 0 ? VE.RGB{Float32}(tr / cnt, tg / cnt, tb / cnt) : c
            end
        end
        return result
    end
    frame = (
        color = [VE.RGB{Float32}(3x / w, 2y / h, (x + y) % 5 / 2) for x in 1:w, y in 1:h],
        albedo = fill(VE.RGB{Float32}(0.5, 0.5, 0.5), w, h),
        normal = [VE.Vec3f(0, x > w ÷ 2 ? 1 : 0, x > w ÷ 2 ? 0 : 1) for x in 1:w, y in 1:h],
        depth = [y > 12 ? Inf32 : Float32(2 + x / w) for x in 1:w, y in 1:h],
        coverage = [y > 12 ? 0.0f0 : 1.0f0 for x in 1:w, y in 1:h],
        overlay = [x == 3 ? VE.RGBA{Float32}(0.4, 0.4, 0.4, 0.5) : VE.RGBA{Float32}(0, 0, 0, 0) for x in 1:w, y in 1:h],
        metadata = Dict{String, Any}(),
    )
    build = VE.scenebuild(:bar, (w, h))
    clip = VE.sceneclip(VE.buildscene(build); build, frames = 1, canvas = (w, h))
    grade = VE.LinearGradeEffect(exposure = 1, saturation = 0.8, redgain = 1.1, bluegain = 0.9)
    VE.addslot!(clip, VE.Effect(grade))
    VE.addslot!(clip, VE.Effect(VE.HDRDenoiseEffect(strength = 0.5, iterations = 2)))
    VE.addslot!(clip, VE.Effect(VE.ToneMapEffect(mode = 2, gamma = 2.2)))
    work = VE.hdrwork!(clip.source, Mantle.defaultbackend(), (w, h))
    out = VE.hdrpicture!(work, VE.uploadmaster!(work, frame), clip, 0)
    @test !(out isa Array)                    # it stays on the device
    scale = exp2(grade.exposure)
    graded = map(frame.color) do c
        l = lum(c)
        VE.RGB{Float32}((l + (c.r - l) * grade.saturation) * scale * grade.redgain,
                        (l + (c.g - l) * grade.saturation) * scale,
                        (l + (c.b - l) * grade.saturation) * scale * grade.bluegain)
    end
    filtered = atrous(graded, frame, 2)
    mixed = map((a, b) -> VE.RGB{Float32}(a.r * 0.5f0 + b.r * 0.5f0, a.g * 0.5f0 + b.g * 0.5f0, a.b * 0.5f0 + b.b * 0.5f0), graded, filtered)
    expected = map(mixed, frame.coverage, frame.overlay) do c, a, o
        VE.RGBA{Float32}(tone(c.r, 2, 2.2f0) * a * (1 - o.alpha) + o.r, tone(c.g, 2, 2.2f0) * a * (1 - o.alpha) + o.g,
                         tone(c.b, 2, 2.2f0) * a * (1 - o.alpha) + o.b, a * (1 - o.alpha) + o.alpha)
    end
    got = Array(out)
    @test maximum(abs(Float32(getfield(g, k)) - getfield(e, k)) for (g, e) in zip(got, expected) for k in (:r, :g, :b, :alpha)) <= 1.5f0 / 255
    # an HDR effect after the display transform is refused
    VE.addslot!(clip, VE.Effect(VE.LinearGradeEffect()))
    @test_throws ErrorException VE.hdrpicture!(work, VE.uploadmaster!(work, frame), clip, 0)
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
