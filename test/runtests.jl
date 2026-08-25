using VideoEditor
using Test
import VideoEditor as VE
import VideoEditor.VideoIO as VideoIO
import FFMPEG_jll
using LinearAlgebra: I        # refactor.jl builds identity MotionTracks

testvideo = joinpath(mktempdir(), "test.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 4 -c:v libx264 -g 30 -pix_fmt yuv420p $testvideo`,
             stdout = devnull, stderr = devnull))
# second source: same framerate, different resolution and content
testvideo2 = joinpath(mktempdir(), "test2.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i smptebars=size=480x270:rate=30 -t 3 -c:v libx264 -g 30 -pix_fmt yuv420p $testvideo2`,
             stdout = devnull, stderr = devnull))
# third source: HALF the rate and a PORTRAIT frame — the mixed-format case (a
# 30 fps phone clip meeting a 60 fps one), which used to be refused at the drop
testvideo15 = joinpath(mktempdir(), "test15.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=180x320:rate=15 -t 4 -c:v libx264 -g 15 -pix_fmt yuv420p $testvideo15`,
             stdout = devnull, stderr = devnull))

include("refactor.jl")   # registry, analyses-as-slots, links, commands

"The curve of `clip`'s `T` effect parameter `name`, created if needed."
function paramcurve!(clip, ::Type{T}, name::Symbol) where {T}
    fx = VE.findslot(clip, T)
    prm = VE.param(fx, name)
    prm.curve === nothing && (prm.curve = VE.AnimCurve())
    return prm.curve
end

# The curve of a clip's opacity, creating the effect and the curve if needed.
# `clip.animations[:opacity]` used to be the way in; a curve now belongs to the
# Param of the effect that renders it, so this is the whole of the change.
function opacitycurve!(clip)
    VE.findslot(clip, VE.OpacityEffect) === nothing &&
        VE.seteffect!(clip, VE.OpacityEffect(1.0f0))
    prm = VE.param(VE.findslot(clip, VE.OpacityEffect), :opacity)
    prm.curve === nothing && (prm.curve = VE.AnimCurve())
    return prm.curve
end
opacitycurve(clip) = (prm = VE.opacityparam(clip); prm === nothing ? nothing : prm.curve)

@testset "VideoSource" begin
    src = VideoSource(testvideo)
    @test src.width == 320 && src.height == 180
    @test src.framerate == 30.0
    @test src.nframes == 120
    @test !isempty(src.keyframe_times)
    @test VE.frameindex(src, 1.0) == 30
    @test VE.frametime(src, 30) == 1.0
end

@testset "FrameRing" begin
    ring = VE.FrameRing(320, 180; capacity = 8)
    buf = VE.RGBFrame(undef, 320, 180)
    @test !VE.fetchframe!(buf, ring, 0)
    slot = VE.claimslot!(ring, 3)
    fill!(slot, VideoEditor.RGB{VideoEditor.N0f8}(1, 0, 0))
    VE.publishslot!(ring, 3)
    @test VE.hasframe(ring, 3)
    @test VE.fetchframe!(buf, ring, 3)
    @test !VE.hasframe(ring, 11)  # same slot, different frame
end

@testset "DecodeWorker" begin
    src = VideoSource(testvideo)
    ring = VE.FrameRing(src.width, src.height; capacity = 16)
    worker = VE.DecodeWorker(src, ring)
    buf = VE.RGBFrame(undef, src.width, src.height)
    for n in (0, 5, 60, 20)  # sequential, forward, jump, backward
        VE.settarget!(worker, n)
        t0 = time()
        while !VE.fetchframe!(buf, ring, n)
            sleep(0.005)
            time() - t0 > 10 && error("decode timeout at frame $n")
        end
        @test true
    end
    VE.stop!(worker)
end

# The worker's loop used to be a bare `for f in jobs; invokelatest(f); end`, so a
# job that threw escaped the loop and killed the task — while the Channel stayed
# open, which meant every later `rungpu` posted into a channel nobody read. The
# next GPU request did not fail, it HUNG forever with nothing in the log.
# `record_loop_demo` sat 15 minutes at full CPU twice before an interrupt showed
# the worker had been dead since the thumbnail probe hit a lost device.
#
# No GPU needed here: the point is that a throwing job leaves the worker serving.
@testset "a throwing GPU job does not kill the worker" begin
    w = VE.GPUWorker()
    VE.rungpu(() -> error("deliberate"), w)
    ran = Ref(false)
    VE.rungpu(() -> (ran[] = true), w)
    t0 = time()
    while !ran[] && time() - t0 < 20
        sleep(0.05)
    end
    @test ran[]                    # the job after the failure still got served
    @test !istaskdone(w.task)      # …because the worker itself survived
end

@testset "Sequence ops" begin
    src = VideoSource(testvideo)  # 120 frames
    seq = Sequence(src)
    split!(seq, 40)
    split!(seq, 80)
    @test length(seq.clips) == 3
    @test VE.seqlength(seq) == 120
    @test VE.locate(seq, 39) == (seq.clips[1], 39)
    @test VE.locate(seq, 40) == (seq.clips[2], 40)

    deleteclip!(seq, 50)  # ripple-delete middle (40 frames)
    @test length(seq.clips) == 2
    @test VE.seqlength(seq) == 80
    @test VE.locate(seq, 40) == (seq.clips[2], 80)

    @test moveclip!(seq, seq.clips[2], 60)          # make a gap
    @test VE.locate(seq, 45) === nothing
    @test !moveclip!(seq, seq.clips[2], 10)         # overlap rejected
    moveclip!(seq, seq.clips[2], 45; snap = 10, snaptargets = [40])
    @test seq.clips[2].start == 40                  # snapped back

    path = joinpath(mktempdir(), "project.toml")
    seq.clips[2].crop = (0.1, 0.2, 0.5, 0.5)
    # stabilization/color tracks are part of the edit and must survive saves
    import VideoEditor.GeometryBasics: Mat3f, Vec3f
    seq.clips[1].motiontrack = VE.MotionTrack(
        [Mat3f(1, 0, 0, 0, 1, 0, 0.5f0 * k, -0.25f0 * k, 1) for k in 1:5], 7)
    seq.clips[1].colortrack = VE.ColorTrack(
        [Vec3f(1.1, 0.9, 1.0), Vec3f(1.0, 1.0, 1.05)],
        [Vec3f(0.01, -0.02, 0.0), Vec3f(0.0, 0.0, 0.0)], 3)
    saveproject(path, seq)
    seq2 = loadproject(path)
    @test [c.start for c in seq2.clips] == [c.start for c in seq.clips]
    @test [c.src_in for c in seq2.clips] == [c.src_in for c in seq.clips]
    @test seq2.clips[2].crop == (0.1, 0.2, 0.5, 0.5)
    @test seq2.clips[1].motiontrack.transforms == seq.clips[1].motiontrack.transforms
    @test seq2.clips[1].motiontrack.src_in == 7
    @test seq2.clips[1].colortrack.gains == seq.clips[1].colortrack.gains
    @test seq2.clips[1].colortrack.offsets == seq.clips[1].colortrack.offsets
    @test seq2.clips[2].motiontrack === nothing   # absent stays absent

    # a moved source file produces a clear error naming the missing path
    broken = replace(read(path, String), testvideo => testvideo * ".gone")
    write(path, broken)
    err = try
        loadproject(path)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("missing video file", err) && occursin(".gone", err)
end

@testset "keyframe animation roundtrip" begin
    src = VideoSource(testvideo)
    clip = VE.Clip(src, 0, 60, 0, (0.0, 0.0, 1.0, 1.0))
    curve = opacitycurve!(clip)
    VE.setkey!(curve, 0, 1.0); VE.setkey!(curve, 30, 0.25); VE.setkey!(curve, 59, 0.8)
    curve.interp = :smooth
    seq = Sequence([clip], src.framerate)

    path = joinpath(mktempdir(), "anim.videoedit.toml")
    saveproject(path, seq)
    c2 = loadproject(path).clips[1]
    a = opacitycurve(c2)
    @test a !== nothing                                         # curve survived the roundtrip
    @test a.interp == :smooth                                   # easing preserved
    @test [(k.frame, k.value) for k in a.keys] == [(0, 1.0), (30, 0.25), (59, 0.8)]
    @test VE.valueat(a, 15) ≈ VE.valueat(curve, 15)            # interpolation identical after reload
    @test VE.isanimated(VE.snapshot(seq)[1])                   # undo snapshot keeps it too
    # per-key eases survive the roundtrip too
    VE.setease!(curve, 2, :hold)
    saveproject(path, seq)
    a2 = opacitycurve(loadproject(path).clips[1])
    @test [k.ease for k in a2.keys] == [:linear, :hold, :linear]
end

@testset "per-key ease math (Premiere temporal interpolation)" begin
    c = VE.AnimCurve()
    VE.setkey!(c, 0, 0.0); VE.setkey!(c, 100, 1.0)
    # linear corners on both ends → constant velocity
    @test VE.valueat(c, 25) ≈ 0.25
    # smooth on BOTH keys = exactly the legacy smoothstep
    VE.setease!(c, 1, :smooth); VE.setease!(c, 2, :smooth)
    @test VE.valueat(c, 25) ≈ 0.25^2 * (3 - 2 * 0.25)
    legacy = VE.AnimCurve()
    VE.setkey!(legacy, 0, 0.0); VE.setkey!(legacy, 100, 1.0)
    legacy.interp = :smooth
    @test VE.valueat(c, 37) ≈ VE.valueat(legacy, 37)
    # smooth ONLY at the far key: hermite H(t) = -t³ + t² + t (slow arrival)
    VE.setease!(c, 1, :linear)
    t = 0.25
    @test VE.valueat(c, 25) ≈ -t^3 + t^2 + t
    # hold freezes until the next key; the endpoints still hit exactly
    VE.setease!(c, 1, :hold)
    @test VE.valueat(c, 99) == 0.0
    @test VE.valueat(c, 100) == 1.0
    # materializing legacy smooth bakes per-key modes and stops overriding
    VE.materializeease!(legacy)
    @test legacy.interp === :linear
    @test all(k -> k.ease === :smooth, legacy.keys)
    VE.setease!(legacy, 2, :linear)                    # now editable per key
    @test VE.valueat(legacy, 25) ≈ (t^3 - 2t^2 + t) * 0 + (-2t^3 + 3t^2) + (t^3 - t^2) * 1
    # moving and re-setting a key keeps its ease; join carries it across halves
    VE.setease!(c, 2, :smooth)
    VE.movekey!(c, 2, 80, 0.9)
    @test c.keys[2].ease === :smooth
    VE.setkey!(c, 80, 0.7)
    @test c.keys[2].ease === :smooth
end

@testset "filmstrip stays put when the head is trimmed" begin
    # Simon, 2026-07-27 (fifth time asking): trimming the left edge must CUT the
    # head — "the thumbnails should NOT move whatsoever". The model always did the
    # right thing; the STRIP was anchored at the clip start, so every head trim
    # re-sliced the whole filmstrip and the picture crept sideways (measured as a
    # 0.31 mean pixel difference over the untouched part of the clip).
    # A thumbnail whose red channel encodes the source second it came from, so the
    # composed strip can be read back and compared time-for-time.
    tint(sec) = fill(VideoEditor.RGB{VideoEditor.N0f8}(clamp(sec / 255, 0, 1), 0, 0), 49, 88)
    fallback = VideoEditor.RGBAf(0, 0, 0, 1)
    compose(t0, s0) = VE.composetiles((t0, 4.0), (0.0, 4.0), 100.0, 72.0, s0,
                                      tint, (49, 88), fallback)
    secat(res, t) = begin
        strip, (x0, x1), _ = res
        i = clamp(round(Int, (t - x0) / (x1 - x0) * size(strip, 1)), 1, size(strip, 1))
        round(Float64(VideoEditor.ColorTypes.red(strip[i, 1])) * 255)
    end
    full = compose(0.0, 0.0)          # clip at 0 s showing the source from 0 s
    trimmed = compose(0.5, 0.5)       # head trimmed by 0.5 s: start AND src_in move
    for t in 1.0:0.2:3.8
        @test secat(full, t) == secat(trimmed, t)     # same frame at the same place
    end
    # …and the strip really starts at the new edge, not before it
    @test trimmed[2][1] >= 0.5 - 1.0e-9
    # trimming the TAIL leaves the head alone just as much
    tailtrimmed = VE.composetiles((0.0, 3.0), (0.0, 4.0), 100.0, 72.0, 0.0,
                                  tint, (49, 88), fallback)
    for t in 0.2:0.2:2.8
        @test secat(full, t) == secat(tailtrimmed, t)
    end
end

@testset "transition roundtrip" begin
    src = VideoSource(testvideo)
    seq = Sequence(src)
    split!(seq, 40)
    VE.addtransition!(seq, 40; duration = 12)
    @test length(seq.transitions) == 1
    path = joinpath(mktempdir(), "trans.videoedit.toml")
    saveproject(path, seq)
    seq2 = loadproject(path)
    @test length(seq2.transitions) == 1                # dissolve survived the roundtrip
    t = seq2.transitions[1]
    @test (t.kind, t.at, t.duration) == (:dissolve, 40, 12)
    @test VE.transitionat(seq2, 40) !== nothing        # resolves at the cut after reload

    # a one-click blend must not swallow the clips it blends (Simon, 2026-07-27:
    # a dissolve drawn across BOTH clips end to end). The hard limit allows twice
    # the shorter clip; the DEFAULT keeps three quarters of each side visible.
    for len in (8, 15, 60, 300)
        l = VE.Clip(src, 0, len, 0, (0.0, 0.0, 1.0, 1.0))
        r = VE.Clip(src, len, 2len, len, (0.0, 0.0, 1.0, 1.0))
        s = Sequence([l, r], 30.0)
        d = VE.clamptransition(l, r, VE.defaultdissolve(s, l, r))
        @test d <= VE.cliplength(l)                    # ≤ half of each side
        @test d <= round(Int, 0.6 * 30.0)              # and never longer than 0.6 s
        @test d >= 2
    end
end

@testset "multi-track edits" begin
    src = VideoSource(testvideo)
    a = VE.Clip(src, 0, 60, 0, (0.0, 0.0, 1.0, 1.0))
    b = VE.Clip(src, 0, 40, 10, (0.0, 0.0, 1.0, 1.0)); b.track = 2
    seq = Sequence([a, b], src.framerate)
    @test VE.ntracks(seq) == 2
    @test length(VE.clipsat(seq, 20)) == 2               # both cover frame 20
    @test seq.clips[VE.clipat(seq, 20)].track == 2       # the top layer wins
    @test VE.clipat(seq, 55) == 1                        # only the base clip covers 55

    # splitting the top clip keeps BOTH halves on layer 2 (regression: the
    # 5-arg Clip constructor used to reset the new half to track 1)
    right = split!(seq, 20)
    @test right.track == 2
    @test count(c -> c.track == 2, seq.clips) == 2

    # the layer survives snapshot/restore and a project roundtrip
    @test sort([c.track for c in VE.snapshot(seq)]) == [1, 2, 2]
    path = joinpath(mktempdir(), "mt.videoedit.toml")
    saveproject(path, seq)
    @test sort([c.track for c in loadproject(path).clips]) == [1, 2, 2]
end

@testset "join clips + effect bypass" begin
    src = VideoSource(testvideo)
    seq = Sequence(src)
    n0 = src.nframes
    right = split!(seq, 40)
    @test length(seq.clips) == 2
    # a key on the right half must survive the join (absolute-frame keyed)
    VE.setkey!(opacitycurve!(right), 50, 0.5)
    joined = VE.joinclips!(seq, 10)
    @test joined !== nothing
    @test length(seq.clips) == 1
    @test seq.clips[1].src_out == n0
    @test opacitycurve(seq.clips[1]) !== nothing
    @test VE.joinclips!(seq, 10) === nothing          # nothing left to join
    # split COPIES curves — the halves must edit independently (Premiere razor)
    VE.setkey!(opacitycurve!(seq.clips[1]), 20, 0.8)
    r3 = split!(seq, 40)
    @test opacitycurve(r3) !== opacitycurve(seq.clips[1])
    VE.setkey!(opacitycurve(seq.clips[1]), 20, 0.1)
    @test VE.valueat(opacitycurve(r3), 20) == 0.8
    VE.joinclips!(seq, 10)
    # a trim that breaks source-contiguity refuses to join (not one cut anymore)
    seq2 = Sequence(VideoSource(testvideo))
    r2 = split!(seq2, 40)
    r2.src_in += 5
    r2.start += 5
    @test VE.joinclips!(seq2, 10) === nothing

    # a switched-off slot keeps its params and its id, and every render path skips it
    c = seq.clips[1]
    VE.seteffect!(c, ColorEffect(saturation = 1.8))
    slot = c.effects[end]
    id = slot.id
    slot.enabled = false
    # the disabled entry is skipped — the stack itself is not empty, because a
    # curve now belongs to an effect and the fade above put an OpacityEffect here
    @test !any(e -> e isa ColorEffect, VE.liveeffects(c))
    @test VE.findeffect(c, ColorEffect).adj.saturation == 1.8f0   # params survive
    @test VE.findslot(c, id) === slot                             # addressable by id
    d = VE.slotdict(slot)                             # project-file roundtrip
    s2 = VE.slotfromdict(d)
    @test s2.id == id && !s2.enabled && VE.op(s2).adj.saturation == 1.8f0
    # files written before ids existed still load — a wrapped effect becomes an off slot
    old = Dict{String, Any}("type" => "bypassed",
                            "inner" => VE.effectdict(ColorEffect(saturation = 1.2)))
    s3 = VE.slotfromdict(old)
    @test !s3.enabled && VE.op(s3).adj.saturation == 1.2f0 && s3.id != 0
    slot.enabled = true
end

@testset "stable identities" begin
    src = VideoSource(testvideo)
    seq = Sequence(src)
    c = seq.clips[1]
    @test c.id != 0
    @test VE.clipbyid(seq, c.id) === c
    right = split!(seq, 40)
    @test right.id != c.id                       # a new clip is a new identity …
    @test VE.clipbyid(seq, right.id) === right
    VE.seteffect!(c, BlurEffect(2.0f0))
    slotid = c.effects[end].id
    snap = VE.snapshot(seq)                      # … and undo preserves identities
    sort!(seq.clips; by = x -> -x.start)         # sorting must not confuse anyone
    VE.restore!(seq, snap)
    @test VE.clipbyid(seq, c.id) !== nothing
    @test VE.clipbyid(seq, c.id).effects[end].id == slotid
end


@testset "a split carries EVERY analysis result, not just the ones on the Clip" begin
    # `colortrack`/`motiontrack`/`mattetrack` are Clip fields, so `split!` copies
    # them by assignment and there is a comment there making sure of it. Restore's
    # cache used to be a module global keyed by clip id, and the right half is
    # minted with `freshid()` — so splitting a restored clip silently dropped the
    # restoration on the right half. It is a Clip field now, and this is the test
    # that says so.
    src = VideoSource(testvideo)
    seq = Sequence(src)
    c = seq.clips[1]
    VE.putrestored!(VE.restorecache!(c), 50, VE.RGBFrame(undef, 4, 4))
    @test VE.hasrestored(c, 50)
    right = split!(seq, 40)                  # frame 50 lands in the RIGHT half
    @test right.id != c.id                   # …which is a different identity …
    @test VE.hasrestored(right, 50)          # … and must still be restored
    @test right.restorecache === c.restorecache          # shared, as the tracks are
end


@testset "blend pairing survives everything" begin
    # Simon, 2026-07-27: "was gibts denn zu suchen? wir markieren 2 clips, und dann
    # merken wir uns die" — the pair is REMEMBERED by clip id, never re-derived from
    # positions, so moves, sorting, undo and a reload all keep it.
    src = VideoSource(testvideo)
    seq = Sequence(src)
    split!(seq, 40)
    a, b = seq.clips[1], seq.clips[2]
    b.blendfrom = a.id
    VE.keyfade!(b, 12, :in)
    @test VE.blends(seq) == [(1, 2, 12)]
    @test VE.findslot(b, VE.OpacityEffect) !== nothing     # the blend IS an effect entry
    slotid = VE.findslot(b, VE.OpacityEffect).id

    b.start = a.start + 20                                  # move it over the other clip
    b.track = 2
    sort!(seq.clips; by = c -> (c.track, c.start))
    @test VE.blends(seq) == [(1, 2, 12)]                    # still paired

    path = joinpath(mktempdir(), "blend.videoedit.toml")
    saveproject(path, seq)
    seq2 = loadproject(path)
    @test seq2.clips[2].blendfrom == seq2.clips[1].id       # ids survive the file
    @test VE.blends(seq2) == [(1, 2, 12)]
    @test VE.findslot(seq2.clips[2], VE.OpacityEffect).id == slotid

    # switching the blend OFF keeps its keys and its id; removing it clears the pair
    slot = VE.findslot(seq2.clips[2], VE.OpacityEffect)
    slot.enabled = false
    @test isempty(collect(VE.liveeffects(seq2.clips[2])))
    @test VE.blends(seq2) == [(1, 2, 12)]                   # still listed, just off
    VE.clearfade!(seq2.clips[2], :in)
    seq2.clips[2].blendfrom = UInt64(0)          # the × action clears the pair too
    @test isempty(VE.blends(seq2))
    @test seq2.clips[2].blendfrom == 0
    @test VE.findslot(seq2.clips[2], VE.OpacityEffect) === nothing
end

"""
Apply `e` to `img` exactly as the graph's `PixelNode` does — the same `applykind!`,
the same `needsfresh` decision, just without a pool. There used to be a second
renderer here (`applyeffect!` → `applykindcpu!`) that these tests reached for; it
had no caller in `src/`, so the suite was the only thing keeping a second
definition of every built-in alive. Test the path that ships.
"""
function applyfx(img, e)
    k = VE.fxkind(e)
    out = VE.needsfresh(k) ? similar(img) : img
    return VE.applykind!(out, img, k)
end

@testset "plugin registry + MCP authoring" begin
    # register a plugin directly (any package can) — it becomes an effect kind
    VE.registerplugin!(:testfx, "Test FX", [VE.FxParam(:k, "k", 0.0, 1.0, 1.0)],
                       p -> VE.Pointwise((c, uv) -> c * Float32(p.k)))
    @test VE.kindbyname(:testfx) !== nothing
    @test any(k -> k.name == :testfx, VE.effectkinds())            # shows in the Add-effect surface

    # …and via the MCP `define_effect` code path (live Base.eval authoring)
    VE.definepluginfromcode!("""
        registerplugin!(:mcpfx, "MCP FX", [FxParam(:gain, "gain", 0.0, 2.0, 1.0)],
                        p -> Pointwise((c, uv) -> c * Float32(p.gain)))
    """)
    @test VE.kindbyname(:mcpfx) !== nothing
    @test any(t -> t["name"] == "effect_mcpfx", VE.tooldefinitions())  # surfaces as an MCP tool

    # a plugin effect applies through the shared kernel + roundtrips through the project dict
    e = VE.plugineffect(:mcpfx; gain = 0.25)
    f = applyfx(fill(VE.RGB{VE.N0f8}(0.8, 0.8, 0.8), 8, 8), e)
    @test all(px -> Float32(px.r) < 0.8, f)                        # gain 0.25 darkens
    @test VE.plugineffectfromdict(VE.effectdict(e)).params.gain == 0.25

    # the stock :soften plugin exercises the OTHER effect kind — Stencil (reads a
    # neighbourhood), applied through the same kernel the GPU graph uses
    soften = VE.plugineffect(:soften; radius = 2)
    @test VE.fxkind(soften) isa VE.Stencil
    edge = fill(VE.RGB{VE.N0f8}(0.0, 0.0, 0.0), 16, 16)
    edge[9:end, :] .= VE.RGB{VE.N0f8}(1.0, 1.0, 1.0)               # sharp black/white seam
    out = applyfx(edge, soften)
    @test out !== edge                                             # a Stencil needs a fresh buffer
    @test any(px -> 0.1 < Float32(px.r) < 0.9, out)               # box blur softened the seam
end

@testset "example plugins load (how-to-hack reference)" begin
    import VideoEditor.GeometryBasics: Vec3f   # precondition: Vec3f already in scope, as in any GLMakie session
    include(joinpath(pkgdir(VideoEditor), "examples", "example_plugins.jl"))  # must not error
    f = fill(VE.RGB{VE.N0f8}(0.4, 0.6, 0.3), 16, 16)
    for name in (:invert, :sepia, :posterize, :levels, :edges, :emboss)
        @test VE.kindbyname(name) !== nothing
        g = applyfx(copy(f), VE.plugineffect(name))
        @test all(px -> isfinite(Float32(px.r)), g)      # every example plugin applies cleanly
    end
end

@testset "Multi-source model" begin
    using Statistics: mean
    src1 = VideoSource(testvideo)    # 320x180, 120 frames
    src2 = VideoSource(testvideo2)   # 480x270, 90 frames
    seq = Sequence(src1)
    push!(seq.clips, VE.Clip(src2, 0, src2.nframes, VE.seqlength(seq), (0.0, 0.0, 1.0, 1.0)))
    @test VE.seqlength(seq) == 210
    @test VE.locate(seq, 119)[1].source === src1
    @test VE.locate(seq, 120) == (seq.clips[2], 0)

    # export walks both sources, scaling the second onto the first's canvas
    out = joinpath(mktempdir(), "multi.mp4")
    exportvideo(out, seq; encoder_options = (crf = 18, preset = "fast"))
    probe = VideoSource(out)
    @test probe.nframes == 210
    @test (probe.width, probe.height) == (320, 180)
    # 4:2:0, not the 4:4:4 ffmpeg picks for RGB input: 4:4:4 lands the file in
    # H.264 High 4:4:4 Predictive, which no phone or TV hardware decoder plays
    # (found the only way these things are found — a file that wouldn't open)
    fmt = readchomp(`$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0
                     -show_entries stream=pix_fmt -of csv=p=0 $out`)
    @test fmt == "yuv420p"
    reader = VideoIO.openvideo(out)
    early = read(reader)                    # clip 1 content (testsrc2)
    seek(reader, 5.0)
    late = read(reader)                     # clip 2 content (smpte bars)
    close(reader)
    reddiff = mean(abs.(Float32.(getfield.(early, :r)) .- Float32.(getfield.(late, :r))))
    @test reddiff > 0.1  # the two sources' content really alternates

    # A source the hardware decoder REFUSES must reach the CPU reader, not throw.
    # `opendecoder` wraps `openstream` in a try/catch for exactly this, but
    # `openstream` only demuxes — `GpuVideoStream` builds its decoder lazily in
    # `startfeed!`, which is where the chroma check lives — so a 4:4:4 file
    # opened cleanly and then threw from the FIRST READ, past the fallback and
    # out through `framereader`. It now probes inside the try, as `graysource`
    # in campath.jl already did.
    v444 = joinpath(mktempdir(), "yuv444.mp4")
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 1
                  -c:v libx264 -g 30 -pix_fmt yuv444p $v444`,
                 stdout = devnull, stderr = devnull))
    @test readchomp(`$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0
                     -show_entries stream=pix_fmt -of csv=p=0 $v444`) == "yuv444p"
    # ON THE PINNED WORKER, not here. A Lava `BatchQueue` belongs to the thread that
    # first builds the Vulkan context, and this is the suite's first touch of Lava —
    # so calling `LavaBackend()` inline made MAIN the owner for the rest of the
    # session, and every later analysis, which the editor runs on its pinned worker
    # by design, died on "BatchQueue is single-writer". That is what took the matte
    # marking beats in `interactions.jl` down (they pass when run on their own,
    # where nothing has claimed the context first). Every `GPUWorker` pins to the
    # same thread, so borrowing one here puts the whole suite on the editor's owner.
    # The assertions stay out here: a testset's state is task-local, so an `@test`
    # inside the worker records nowhere.
    probe = VE.rungpusync(VE.GPUWorker()) do
        gpu = VE.Lava.LavaBackend()
        src444, src420 = VideoSource(v444), VideoSource(testvideo)
        # THE ANCHOR. If 4:2:0 does not take the GPU path on this machine then
        # both cases fall back for unrelated reasons and the assertion below
        # cannot tell a fixed `opendecoder` from a broken one — so say so rather
        # than pass silently.
        d420 = VE.opendecoder(src420, gpu)
        d420 isa VE.SequentialReader && return nothing
        close(d420)
        fellback = VE.opendecoder(src444, gpu) isa VE.SequentialReader
        # and the fallback must actually READ, advancing frames
        r = VE.opendecoder(src444, gpu)
        f = VE.RGBFrame(undef, src444.width, src444.height)
        a = copy(VE.readframe!(f, r, 0))
        b = copy(VE.readframe!(f, r, 10))
        return (fellback = fellback, advanced = a != b)
    end
    if probe === nothing
        @info "GPU decode unavailable here — 4:4:4 fallback test cannot discriminate; skipped"
    else
        @test probe.fellback
        @test probe.advanced
    end

    # projects roundtrip with several sources
    path = joinpath(mktempdir(), "multi.toml")
    saveproject(path, seq)
    seq2 = loadproject(path)
    @test length(seq2.clips) == 2
    @test [c.source.path for c in seq2.clips] == [testvideo, testvideo2]
    @test (seq2.clips[2].source.width, seq2.clips[2].source.height) == (480, 270)
end

@testset "Export" begin
    src = VideoSource(testvideo)  # 320x180, 120 frames @30
    seq = Sequence(src)
    split!(seq, 40)
    deleteclip!(seq, 60)                      # keep first 40 frames
    seq.clips[1].crop = (0.25, 0.25, 0.5, 0.5)  # 160x90 canvas
    seteffect!(seq.clips[1], ColorEffect(saturation = 0.0))  # grayscale
    out = joinpath(mktempdir(), "out.mp4")
    exportvideo(out, seq; encoder_options = (crf = 18, preset = "fast"))
    @test isfile(out)
    probe = VideoSource(out)
    @test probe.nframes == 40
    @test (probe.width, probe.height) == (160, 90)
    reader = VideoIO.openvideo(out)
    frame = read(reader)
    close(reader)
    # grayscale effect must survive encode (allow codec noise)
    @test all(c -> abs(Float32(c.r) - Float32(c.b)) < 0.06, frame)

    # GIF export (palettegen/paletteuse) with looping
    gif = joinpath(mktempdir(), "out.gif")
    exportgif(gif, seq; fps = 10, loop = 0)
    @test isfile(gif)
    @test filesize(gif) > 0
    hdr = read(gif, 6)  # GIF89a magic
    @test String(hdr) in ("GIF89a", "GIF87a")
    # the NETSCAPE loop extension must be present for loop = 0 (loop forever)
    @test occursin("NETSCAPE", String(read(gif)))

    # keyframed parameters bake per-frame into the export (opacity ramp → luma ramp)
    kseq = Sequence(VideoSource(testvideo))
    kcurve = opacitycurve!(kseq.clips[1])
    VE.setkey!(kcurve, 0, 0.1); VE.setkey!(kcurve, 119, 1.0)
    kout = joinpath(mktempdir(), "kf.mp4")
    exportvideo(kout, kseq; audio = false, encoder_options = (crf = 18, preset = "fast"))
    luma(f) = mean(Float32(px.r) + px.g + px.b for px in f) / 3
    kr = VideoIO.openvideo(kout)
    lo = luma(read(kr))                       # frame 0: opacity ~0.1 (dark)
    for _ in 1:100; read(kr); end
    hi = luma(read(kr))                        # frame 101: opacity ~0.86 (bright)
    close(kr)
    @test hi > 2 * lo
end

@testset "Audio export" begin
    dir = mktempdir()
    tone1 = joinpath(dir, "tone440.mp4")
    tone2 = joinpath(dir, "tone880.mp4")
    for (f, freq) in ((tone1, 440), (tone2, 880))
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=160x90:rate=30 -f lavfi -i sine=frequency=$freq:sample_rate=48000 -t 2 -c:v libx264 -g 30 -pix_fmt yuv420p -c:a aac -shortest $f`,
                     stdout = devnull, stderr = devnull))
    end
    @test VE.hasaudio(tone1)
    @test !VE.hasaudio(testvideo)

    # timeline: 1s trimmed 440 Hz · 0.5s gap · 1s 880 Hz · 0.5s mute source
    src1, src2 = VideoSource(tone1), VideoSource(tone2)
    seq = Sequence(src1)
    seq.clips[1].src_in = 15
    seq.clips[1].src_out = 45
    push!(seq.clips, VE.Clip(src2, 0, 30, 45, (0.0, 0.0, 1.0, 1.0)))
    push!(seq.clips, VE.Clip(VideoSource(testvideo), 0, 15, 75, (0.0, 0.0, 1.0, 1.0)))
    out = joinpath(dir, "with_audio.mp4")
    exportvideo(out, seq)
    @test VE.hasaudio(out)

    # per-segment dominant frequency via zero crossings on the decoded PCM
    pcm = joinpath(dir, "out.pcm")
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -i $out -f s16le -ac 1 -ar 8000 $pcm`,
                 stdout = devnull, stderr = devnull))
    samples = reinterpret(Int16, read(pcm))
    sr = 8000
    @test isapprox(length(samples) / sr, 3.0; atol = 0.1)
    seg(t0, t1) = samples[max(1, round(Int, t0 * sr)):min(end, round(Int, t1 * sr))]
    freqof(x) = count(i -> (x[i] < 0) != (x[i + 1] < 0), 1:(length(x) - 1)) / 2 / (length(x) / sr)
    @test isapprox(freqof(seg(0.1, 0.9)), 440; rtol = 0.05)   # trimmed clip 1
    @test maximum(abs.(Int.(seg(1.05, 1.45)))) < 500          # gap is silent
    @test isapprox(freqof(seg(1.6, 2.4)), 880; rtol = 0.05)   # clip 2
    @test maximum(abs.(Int.(seg(2.6, 2.9)))) < 500            # mute source is silent

    # when no source has audio the mux is skipped and the file has no track
    out2 = joinpath(dir, "silent.mp4")
    exportvideo(out2, Sequence(VideoSource(testvideo)))
    @test !VE.hasaudio(out2)
    @test VideoSource(out2).nframes == 120
end

@testset "Audio preview mixer" begin
    dir = mktempdir()
    tone1 = joinpath(dir, "tone440.mp4")
    tone2 = joinpath(dir, "tone880.mp4")
    for (f, freq) in ((tone1, 440), (tone2, 880))
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=160x90:rate=30 -f lavfi -i sine=frequency=$freq:sample_rate=48000 -t 2 -c:v libx264 -g 30 -pix_fmt yuv420p -c:a aac -shortest $f`,
                     stdout = devnull, stderr = devnull))
    end
    src1, src2 = VideoSource(tone1), VideoSource(tone2)
    seq = Sequence(src1)
    seq.clips[1].src_in = 15
    seq.clips[1].src_out = 45                                            # 1s trimmed 440 Hz
    push!(seq.clips, VE.Clip(src2, 0, 30, 45, (0.0, 0.0, 1.0, 1.0)))     # gap, then 1s 880 Hz
    push!(seq.clips, VE.Clip(VideoSource(testvideo), 0, 15, 75, (0.0, 0.0, 1.0, 1.0)))  # mute
    tracks = Dict{String, Union{VE.PCMTrack, Nothing}}(
        src1.path => VE.loadpcm(src1),
        src2.path => VE.loadpcm(src2),
        testvideo => VE.loadpcm(VideoSource(testvideo)))  # → nothing (no audio)
    @test tracks[src1.path] isa VE.PCMTrack
    @test tracks[testvideo] === nothing

    total = ceil(Int, 3.0 * VE.AUDIORATE)
    out = Matrix{Int16}(undef, 2, total)
    VE.fillaudio!(out, seq, tracks, 0)
    mono = Int.(out[1, :])
    seg(t0, t1) = mono[max(1, round(Int, t0 * VE.AUDIORATE)):round(Int, t1 * VE.AUDIORATE)]
    freqof(x) = count(i -> (x[i] < 0) != (x[i + 1] < 0), 1:(length(x) - 1)) / 2 /
                (length(x) / VE.AUDIORATE)
    @test isapprox(freqof(seg(0.1, 0.9)), 440; rtol = 0.05)   # trimmed clip 1
    @test maximum(abs.(seg(1.05, 1.45))) < 500                # gap is silent
    @test isapprox(freqof(seg(1.6, 2.4)), 880; rtol = 0.05)   # clip 2
    @test maximum(abs.(seg(2.6, 2.9))) < 500                  # mute source is silent

    # feeding block-by-block must equal the one-shot fill (feeder correctness)
    out2 = zeros(Int16, 2, total)
    pos = 0
    while pos < total
        n = min(VE.AUDIOBLOCK, total - pos)
        VE.fillaudio!(view(out2, :, (pos + 1):(pos + n)), seq, tracks, pos)
        pos += n
    end
    @test out2 == out
end

@testset "Color stabilization" begin
    using Statistics: std
    import GPUFiltering
    import VideoEditor.ColorTypes: RGB
    import VideoEditor.FixedPointNumbers: N0f8
    # static gradient pattern with known 3Hz multiplicative flicker
    flicker = joinpath(mktempdir(), "flicker.mp4")
    base = [0.25f0 + 0.5f0 * (i + j) / 500 for i in 1:320, j in 1:180]
    VideoIO.open_video_out(flicker, RGB{N0f8}, (180, 320); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        for n in 0:119
            gain = 1.0f0 + 0.2f0 * sin(2.0f0π * 3.0f0 * n / 30)
            frame = map(v -> RGB{N0f8}(clamp(v * gain, 0, 1), clamp(v * gain, 0, 1), clamp(v * gain, 0, 1)), base)
            write(writer, PermutedDimsArray(frame, (2, 1)))
        end
    end
    src = VideoSource(flicker)
    clip = Clip(src)
    track = analyzecolor!(clip)
    @test track !== nothing
    @test length(track.gains) == src.nframes

    sr = VE.SequentialReader(src)
    buf = VE.RGBFrame(undef, src.width, src.height)
    raw = Float32[]
    fixed = Float32[]
    for n in 0:(src.nframes - 1)
        VE.readframe!(buf, sr, n)
        push!(raw, GPUFiltering.channelstats(buf)[1][2])
        VE.applycolortrack!(buf, clip, n)
        push!(fixed, GPUFiltering.channelstats(buf)[1][2])
    end
    close(sr)
    # flicker (temporal std of the channel mean) must drop hard
    @test std(fixed) < std(raw) / 3

    # strength scales the correction at APPLY time: 0 = untouched original
    # (frame 2 sits near the flicker peak — frame 10 would be a sine zero-crossing)
    sr2 = VE.SequentialReader(src)
    VE.readframe!(buf, sr2, 2)
    before = copy(buf)
    track.strength = 0.0f0
    VE.applycolortrack!(buf, clip, 2)
    @test buf == before
    track.strength = 1.0f0
    VE.applycolortrack!(buf, clip, 2)
    @test buf != before
    close(sr2)

    # strength roundtrips through the project file (older files default to 1)
    track.strength = 0.4f0
    seqct = Sequence([clip], src.framerate)
    ctpath = joinpath(mktempdir(), "ct.videoedit.toml")
    saveproject(ctpath, seqct)
    @test loadproject(ctpath).clips[1].colortrack.strength ≈ 0.4f0
    track.strength = 1.0f0
end

@testset "Motion stabilization" begin
    using Statistics: std, mean
    import GPUFiltering
    import VideoEditor.GeometryBasics: Mat3f
    import VideoEditor.ColorTypes: RGB
    import VideoEditor.FixedPointNumbers: N0f8

    pattern = zeros(Float32, 320, 180)
    GPUFiltering.smooth!(pattern, rand(Float32, 320, 180), 3.0)
    base = map(x -> RGB{N0f8}(clamp(x, 0, 1), clamp(x, 0, 1), clamp(x, 0, 1)), pattern)
    frame = similar(base)

    # --- tripod: rotation + translation jitter must lock to the pixel,
    #     verified END TO END through the production apply path -----------
    n = 90
    cx, cy = 160.5f0, 90.5f0
    function jittermatrix(k)  # rotate about the center by θₖ, then shift by tₖ
        θ = 0.8f0 * sin(2.0f0π * 3 * k / 30)
        t = (5.0f0 * sin(2.0f0π * 4 * k / 30), 4.0f0 * cos(2.0f0π * 2.5f0 * k / 30))
        a11, a21 = cosd(θ), sind(θ)
        a12, a22 = -a21, a11
        tx = cx - (a11 * (cx + t[1]) + a12 * (cy + t[2]))
        ty = cy - (a21 * (cx + t[1]) + a22 * (cy + t[2]))
        return Mat3f(a11, a21, 0, a12, a22, 0, tx, ty, 1)
    end
    shaky = joinpath(mktempdir(), "shakyrot.mp4")
    VideoIO.open_video_out(shaky, RGB{N0f8}, (180, 320); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        for k in 1:n
            GPUFiltering.warp!(frame, base, jittermatrix(k))
            write(writer, PermutedDimsArray(frame, (2, 1)))
        end
    end

    src = VideoSource(shaky)
    clip = Clip(src)
    track = analyzemotion!(clip)  # :similarity default — the NCC camera lock
    @test track !== nothing
    @test track.mode === :similarity
    maxrot = maximum(M -> abs(atand(M[2, 1], M[1, 1])), track.transforms)
    @test maxrot > 0.4  # the rotation component was detected

    sr = VE.SequentialReader(src)
    tmp = similar(frame)
    raw1 = similar(frame)
    VE.readframe!(raw1, sr, 0)
    corr1 = copy(raw1)
    VE.applymotiontrack!(corr1, tmp, clip, 0)
    central = (81:240, 46:135)  # away from the warp's replicate borders
    meandiff(a, b) = mean(abs(Float32(a[i, j].g) - Float32(b[i, j].g))
                          for i in central[1], j in central[2])
    rawdiffs = Float32[]
    corrdiffs = Float32[]
    for k in (20, 45, 70)
        rawk = similar(frame)
        VE.readframe!(rawk, sr, k)
        corrk = copy(rawk)
        VE.applymotiontrack!(corrk, tmp, clip, k)
        push!(rawdiffs, meandiff(rawk, raw1))
        push!(corrdiffs, meandiff(corrk, corr1))
        @test corrdiffs[end] < 0.02  # near-locked in absolute terms
    end
    close(sr)
    # aggregated (single frames can coincide with frame 1's jitter phase,
    # making the raw baseline arbitrarily small)
    @test mean(corrdiffs) < mean(rawdiffs) / 4

    # the auto-crop inset must cover the largest jitter excursion
    crop = VE.bordercrop(track, 320, 180)
    @test crop[1] >= 5 / 320 - 0.002
    @test crop[2] >= 4 / 180 - 0.002
    @test isapprox(crop[3], 1 - 2crop[1]; atol = 1e-6)
    @test isapprox(crop[4], 1 - 2crop[2]; atol = 1e-6)

    # composing a user crop with the border crop: full frame takes the
    # border crop, an interior crop passes through EXACTLY (`!=` gates the
    # undo snapshot — float noise would read as a change), an edge-touching
    # crop shrinks only where they disagree, disjoint stays positive
    border = (0.1, 0.05, 0.8, 0.9)
    @test VE.cropintersect((0.0, 0.0, 1.0, 1.0), border) === border
    @test VE.cropintersect((0.2, 0.2, 0.5, 0.5), border) === (0.2, 0.2, 0.5, 0.5)
    edgy = VE.cropintersect((0.5, 0.5, 0.5, 0.5), border)
    @test all(isapprox.(edgy, (0.5, 0.5, 0.4, 0.45); atol = 1e-12))
    tiny = VE.cropintersect((0.0, 0.0, 0.05, 0.05), border)
    @test tiny[3] >= 0.01 && tiny[4] >= 0.01

    # --- perspective: oscillating keystone needs the homography fit -------
    np = 90
    function keystonematrix(k)
        θ = 0.5f0 * sin(2.0f0π * 3 * k / 30)
        a11, a21 = cosd(θ), sind(θ)
        t = (3.0f0 * sin(2.0f0π * 4 * k / 30), 2.5f0 * cos(2.0f0π * 2.5f0 * k / 30))
        tx = cx - (a11 * (cx + t[1]) - a21 * (cy + t[2]))
        ty = cy - (a21 * (cx + t[1]) + a11 * (cy + t[2]))
        px = 8.0f-5 * sin(2.0f0π * 2 * k / 30)
        py = 6.0f-5 * cos(2.0f0π * 1.5f0 * k / 30)
        return Mat3f(a11, a21, px, -a21, a11, py, tx, ty, 1)
    end
    keystone = joinpath(mktempdir(), "keystone.mp4")
    VideoIO.open_video_out(keystone, RGB{N0f8}, (180, 320); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        for k in 1:np
            GPUFiltering.warp!(frame, base, keystonematrix(k))
            write(writer, PermutedDimsArray(frame, (2, 1)))
        end
    end
    srcp = VideoSource(keystone)
    clipp = Clip(srcp)
    clipa = Clip(srcp)
    trackp = analyzemotion!(clipp; mode = :perspective)
    @test trackp !== nothing
    @test maximum(M -> max(abs(M[3, 1]), abs(M[3, 2])), trackp.transforms) > 2e-5
    analyzemotion!(clipa; mode = :tripod)

    # corner regions inset from the replicate borders — exactly where an
    # affine lock leaves keystoned footage swimming
    regions = ((25:85, 25:65), (236:296, 25:65), (25:85, 116:156), (236:296, 116:156))
    regiondiff(a, b) = maximum(r -> mean(abs(Float32(a[i, j].g) - Float32(b[i, j].g))
                                         for i in r[1], j in r[2]), regions)
    srp = VE.SequentialReader(srcp)
    function readcorrected(clip, k)
        buf = similar(frame)
        VE.readframe!(buf, srp, k)
        VE.applymotiontrack!(buf, tmp, clip, k)
        return buf
    end
    refp = readcorrected(clipp, 0)
    refa = readcorrected(clipa, 0)
    pdiffs = Float32[]
    adiffs = Float32[]
    for k in (20, 45, 70)
        push!(pdiffs, regiondiff(readcorrected(clipp, k), refp))
        push!(adiffs, regiondiff(readcorrected(clipa, k), refa))
        @test pdiffs[end] < 0.008  # corners locked (measured ≈0.003)
    end
    close(srp)
    # the affine fit cannot lock keystoned corners (measured ≈4.5× worse)
    @test maximum(adiffs) > 1.5 * maximum(pdiffs)

    # --- smooth mode: translation shake collapses, slow intent survives ---
    n2 = 120
    jitter = [(6.0f0 * sin(2.0f0π * 4 * k / 30), 4.0f0 * cos(2.0f0π * 3 * k / 30)) for k in 0:(n2 - 1)]
    shaky2 = joinpath(mktempdir(), "shaky.mp4")
    VideoIO.open_video_out(shaky2, RGB{N0f8}, (180, 320); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        for k in 1:n2
            GPUFiltering.warp!(frame, base, GPUFiltering.translationmatrix(jitter[k]...))
            write(writer, PermutedDimsArray(frame, (2, 1)))
        end
    end
    clip2 = Clip(VideoSource(shaky2))
    track2 = analyzemotion!(clip2; mode = :smooth)
    @test track2 !== nothing
    corr = [(-M[1, 3], -M[2, 3]) for M in track2.transforms]
    inner = 31:(n2 - 30)  # filtfilt boundary transients excluded
    residual = [hypot(jitter[k][1] + corr[k][1], jitter[k][2] + corr[k][2]) for k in inner]
    shake = [hypot(jitter[k]...) for k in inner]
    @test std(residual) < std(shake) / 10
end

@testset "Proxy" begin
    src = VideoSource(testvideo2)  # 480x270
    proxy = VE.generateproxy(src; height = 90)
    @test (proxy.width, proxy.height) == (160, 90)
    @test proxy.nframes == src.nframes  # frame-index compatible
    @test proxy.path == VE.proxypath(src, 90)
    stamp = mtime(proxy.path)
    proxy2 = VE.generateproxy(src; height = 90)  # cache hit: no re-encode
    @test proxy2.path == proxy.path && mtime(proxy2.path) == stamp
    @test VE.proxypath(src, 180) != proxy.path  # height is part of the key

    # auto-proxy heuristic: pixel count or codec weight (bytes per frame)
    @test !VE.needsproxy(src)
    @test VE.needsproxy(src; maxpixels = 100_000)
    @test VE.needsproxy(src; maxbytesperframe = 1)

    # cache pruning: oldest files go first, newest survive, cap respected
    dir = VE.cachedir("proxies")
    old = joinpath(dir, "prunetest_old.mp4")
    new = joinpath(dir, "prunetest_new.mp4")
    write(old, zeros(UInt8, 1000)); write(new, zeros(UInt8, 1000))
    run(`touch -d "30 days ago" $old`)   # age it: pruning is LRU by mtime
    before = sum(filesize, readdir(dir; join = true); init = 0) +
             sum(filesize, readdir(VE.cachedir("pcm"); join = true); init = 0)
    VE.prunecache!(maxbytes = before - 500)   # force removal of ≥1 file
    @test !isfile(old)                        # oldest went first
    @test isfile(new)
    rm(new; force = true)
end

@testset "Object lock" begin
    using Statistics: mean
    using Random: MersenneTwister
    import GPUFiltering
    import VideoEditor.ColorTypes: RGB
    import VideoEditor.FixedPointNumbers: N0f8

    # textured blob drifting diagonally over a static textured background —
    # tripod would lock the background; object lock must follow the blob.
    # Seed 1 is a regression guard: this draw once bent the global fit into
    # a phantom rotation until subject-overlapping patches were excluded.
    W, H, n = 320, 180, 90
    rng = MersenneTwister(1)
    bg = zeros(Float32, W, H); GPUFiltering.smooth!(bg, rand(rng, Float32, W, H), 4.0)
    blob = zeros(Float32, 40, 40); GPUFiltering.smooth!(blob, rand(rng, Float32, 40, 40), 1.5)
    blobpos(k) = (60.0 + 1.5 * (k - 1), 40.0 + 0.9 * (k - 1))  # top-left, frame k
    moving = joinpath(mktempdir(), "moving.mp4")
    VideoIO.open_video_out(moving, RGB{N0f8}, (H, W); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        for k in 1:n
            f = copy(bg)
            x0, y0 = round.(Int, blobpos(k))
            f[x0:(x0 + 39), y0:(y0 + 39)] .= blob
            rgbf = map(x -> RGB{N0f8}(clamp(x, 0, 1), clamp(x, 0, 1), clamp(x, 0, 1)), f)
            write(writer, PermutedDimsArray(rgbf, (2, 1)))
        end
    end
    clip = Clip(VideoSource(moving))
    track = analyzeobject!(clip, blobpos(1) .+ 20.0)  # click the blob center
    @test track !== nothing
    # translationmatrix(x1 - px, ...) stores the drift: T[1,3] = px - x1
    for k in (30, 60, 90)
        T = track.transforms[k]
        truth = blobpos(k) .- blobpos(1)
        @test isapprox(T[1, 3], truth[1]; atol = 2.0)
        @test isapprox(T[2, 3], truth[2]; atol = 2.0)
    end
    # applying the track pins the subject's pixels to its frame-1 location
    sr = VE.SequentialReader(clip.source)
    f1 = VE.RGBFrame(undef, W, H); VE.readframe!(f1, sr, 0)
    fk = VE.RGBFrame(undef, W, H); VE.readframe!(fk, sr, 89)
    tmp = similar(fk)
    VE.applymotiontrack!(fk, tmp, clip, 89)
    close(sr)
    x0, y0 = round.(Int, blobpos(1))
    region = (x0 .+ (6:33), y0 .+ (6:33))  # blob interior, frame-1 location
    d = mean(abs(Float32(fk[i, j].g) - Float32(f1[i, j].g)) for i in region[1], j in region[2])
    @test d < 0.03

    # camera roll: the whole scene rotates ±4° — the object lock's global
    # stage must undo it so the picked patch stays pixel-identical (a
    # translation-only tracker cannot lock a rotating patch)
    cx, cy = W / 2, H / 2
    rotmat(θ) = Mat3f(cosd(θ), sind(θ), 0, -sind(θ), cosd(θ), 0,
                      cx - cosd(θ) * cx + sind(θ) * cy,
                      cy - sind(θ) * cx - cosd(θ) * cy, 1)
    bgrgb = map(x -> RGB{N0f8}(clamp(x, 0, 1), clamp(x, 0, 1), clamp(x, 0, 1)), bg)
    rolling = joinpath(mktempdir(), "rolling.mp4")
    VideoIO.open_video_out(rolling, RGB{N0f8}, (H, W); framerate = 30,
                           encoder_options = (crf = 15, preset = "fast")) do writer
        rolled = similar(bgrgb)
        for k in 1:n
            GPUFiltering.warp!(rolled, bgrgb, rotmat(4.0 * sin(2π * (k - 1) / 45)))
            write(writer, PermutedDimsArray(rolled, (2, 1)))
        end
    end
    rclip = Clip(VideoSource(rolling))
    rpoint = (cx + 60.0, cy + 30.0)        # off-center: rotation moves it
    @test analyzeobject!(rclip, rpoint) !== nothing
    sr = VE.SequentialReader(rclip.source)
    r1 = VE.RGBFrame(undef, W, H); VE.readframe!(r1, sr, 0)
    rk = VE.RGBFrame(undef, W, H); VE.readframe!(rk, sr, 11)  # ≈ peak roll
    VE.applymotiontrack!(rk, tmp, rclip, 11)
    close(sr)
    px0, py0 = round.(Int, rpoint)
    rr = ((px0 - 14):(px0 + 14), (py0 - 14):(py0 + 14))
    dr = mean(abs(Float32(rk[i, j].g) - Float32(r1[i, j].g)) for i in rr[1], j in rr[2])
    @test dr < 0.04
end

canui = try
    import GLMakie
    GLMakie.activate!(; visible = false)
    screen = GLMakie.Screen(; visible = false)
    close(screen)
    true
catch e
    @warn "skipping UI interaction tests (no GL context)" exception = e
    false
end
include("agentview.jl")   # what an AGENT sees (headless: no window needed)


@testset "Thumbnails" begin
    src = VideoSource(testvideo)
    cache = VE.ThumbnailCache(src; thumbheight = 32)
    VE.request!(cache, [0, 2])
    t0 = time()
    while (VE.getthumb(cache, 0) === nothing || VE.getthumb(cache, 2) === nothing) && time() - t0 < 10
        sleep(0.02)
    end
    thumb = VE.getthumb(cache, 0)
    @test thumb !== nothing
    @test size(thumb, 2) == 32
    @test VE.nearestthumb(cache, 1) !== nothing
    dest = VE.RGBFrame(undef, src.width, src.height)
    VE.blitthumb!(dest, thumb)
    @test dest[1, 1] == thumb[1, 1]
    VE.stop!(cache)

    # downscale must PREFILTER, not decimate: a 1 px checkerboard averages to
    # mid-gray; nearest-neighbor sampling returns full-contrast noise
    checker = [RGB{N0f8}(iseven(i + j), iseven(i + j), iseven(i + j))
               for i in 1:64, j in 1:64]
    small = VE.downscale(checker, 8, 8)
    @test all(c -> 0.4 < Float32(c.g) < 0.6, small)
    # upscale path stays nearest (no zero-count cells → no black pixels)
    big = VE.downscale(checker, 128, 128)
    @test all(c -> Float32(c.g) in (0.0f0, 1.0f0), big)
end

@testset "matte: SAM 2 is the seed by default" begin
    # The editor segments out of the box — no include, no opt-in call.
    # SAM 2 is not optional: `defaultsegmenter` is it, unconditionally, and a
    # missing model is a loud failure rather than a quieter disc-painting mode.
    @test VE.defaultsegmenter() === VE.sam2seed

    # …and it is a PARAMETER, not a global switch: pass another and it is used,
    # with nothing left installed anywhere afterwards
    called = Ref(false)
    mine = (frame, points; key = nothing) -> (called[] = true; fill(0xff, size(frame)))
    src = VideoSource(testvideo)
    clip = Clip(src)
    frame = fill(VE.RGB{VE.N0f8}(0.5, 0.5, 0.5), src.width, src.height)
    m = VideoEditor.seedmask(clip, frame, [(0.5, 0.5, true)]; segmenter = mine)
    @test called[] && size(m) == size(frame) && all(==(0xff), m)
    # no segmenter is an error, not a silent disc: the discs are their own method
    @test_throws ErrorException VideoEditor.seedmask(clip, frame, [(0.5, 0.5, true)];
                                                     segmenter = nothing)
    d = VideoEditor.seedmask(clip, [(0.5, 0.5, true)])
    @test size(d) == size(frame) && any(!=(0x00), d) && !all(==(0xff), d)
end

@testset "matte track, effect and keyframes" begin
    # What this testset asserts is the track/effect/keyframe plumbing, so it pins a
    # trivial propagator (hold the seed across the clip) instead of inheriting
    # whatever is installed: VideoEditor registers the real MatAnyone propagator at
    # load, and then this would (a) assert disc geometry against a segmentation
    # model and (b) drive a GPU model from the main thread, which after
    # `interactions.jl` no longer owns the Lava context — "BatchQueue is
    # single-writer".
    prevprop = VideoEditor.MATTEPROPAGATOR[]
    VideoEditor.registermatte!((frames, seeds; progress = nothing) -> begin
        k0 = minimum(keys(seeds))
        repeat(seeds[k0], 1, 1, length(frames))
    end)
    try
    src = VideoSource(testvideo2)              # smptebars 480x270, strong colour blocks
    clip = Clip(src; src_in = 0, src_out = 12)

    # --- propagation from a marked frame
    rect = (0.1, 0.1, 0.3, 0.3)
    mask = VideoEditor.seedmask(clip, rect)
    @test size(mask) == (src.width, src.height)
    @test any(!=(0x00), mask)

    reader = let sr = VideoEditor.SequentialReader(src),
                 buf = VideoEditor.RGBFrame(undef, src.width, src.height)
        sf -> (VideoEditor.readframe!(buf, sr, Int(sf)); copy(buf))
    end
    track = VideoEditor.analyzematte!(clip, reader, Dict(0 => mask); maxside = 96)
    @test clip.mattetrack === track
    @test size(track.alpha, 3) == 12
    @test track.seeds == [0]
    @test track.src_in == 0
    # the marked region must come out more opaque than the far corner
    mw, mh = VideoEditor.mattesize(track)
    inside = Int(track.alpha[max(1, mw ÷ 5), max(1, mh ÷ 5), 1])
    outside = Int(track.alpha[mw - 1, mh - 1, 1])
    @test inside > outside

    # --- applying: same function the GPU node calls
    frame = reader(0)
    keyed = VideoEditor.applymatte!(copy(frame), clip, 0; strength = 1.0)
    @test size(keyed) == size(frame)
    # strength 0 is an exact no-op, so a keyframe can fade the matte in from nothing
    @test VideoEditor.applymatte!(copy(frame), clip, 0; strength = 0.0) == frame
    # a frame outside the analyzed range is untouched, not blacked out
    @test VideoEditor.applymatte!(copy(frame), clip, 999; strength = 1.0) == frame
    # somewhere the matte actually darkened the background
    @test any(keyed[i] != frame[i] for i in eachindex(frame))

    # --- feather reaches a fraction of the PICTURE, not a count of plane texels
    # Key a white frame against black over a hard vertical edge: the output IS
    # the alpha, so the run of pixels that are neither fully in nor fully out is
    # the softness the user sees. Analyzing the same edge four times finer must
    # not change it — the reach used to be a fixed ±4 texels, so raising
    # `mattereadsize` (which no longer caps at 480) quietly narrowed the band
    # from 34 output pixels to 8.
    featherband = function (mw, mh)
        cl = Clip(src; src_in = 0, src_out = 1)
        a = zeros(UInt8, mw, mh, 1)
        a[1:(mw ÷ 2), :, 1] .= 0xff
        cl.mattetrack = VideoEditor.MatteTrack(a, 0)
        buf = fill(RGB{N0f8}(1, 1, 1), 480, 270)
        VideoEditor.applymatte!(buf, cl, 0; strength = 1.0, feather = 1.0)
        row = [Float32(buf[i, 135].r) for i in 1:480]
        findlast(>(0.02f0), row) - findfirst(<(0.98f0), row) + 1
    end
    @test featherband(480, 270) > 4                      # it softens at all
    @test featherband(480, 270) == featherband(1920, 1080) == featherband(3840, 2160)

    # --- effect: registries, neutrality, serialization
    @test VideoEditor.isneutral(MatteEffect(0.0, 0.0))
    @test !VideoEditor.isneutral(MatteEffect(1.0, 0.0))
    @test VideoEditor.effectfromdict(VideoEditor.effectdict(MatteEffect(0.7, 0.2))) ==
          MatteEffect(0.7f0, 0.2f0)
    k = VideoEditor.kindbyname(:matte)
    @test [pr.name for pr in k.params] == [:strength, :feather]
    @test k.matches(MatteEffect(1.0, 0.0))
    @test k.read(MatteEffect(0.5, 0.25)) == (strength = 0.5, feather = 0.25)
    @test k.make((strength = 0.5, feather = 0.25)) == MatteEffect(0.5f0, 0.25f0)

    # --- keyframes drive it through the normal param path
    push!(clip.effects, VideoEditor.Effect(MatteEffect()))
    mc = paramcurve!(clip, MatteEffect, :strength)
    VideoEditor.setkey!(mc, 0, 0.0)
    VideoEditor.setkey!(mc, 11, 1.0)
    @test VideoEditor.valueat(mc, 0) ≈ 0.0
    e0 = VideoEditor.findeffect(VideoEditor.effectiveclip(clip, 0), MatteEffect)
    e1 = VideoEditor.findeffect(VideoEditor.effectiveclip(clip, 11), MatteEffect)
    @test e0.strength ≈ 0.0f0
    @test e1.strength ≈ 1.0f0
    # feather keyframes must not clobber strength (shared-effect rebuild)
    VideoEditor.setkey!(paramcurve!(clip, MatteEffect, :feather), 11, 0.5)
    e1b = VideoEditor.findeffect(VideoEditor.effectiveclip(clip, 11), MatteEffect)
    @test e1b.strength ≈ 1.0f0 && e1b.feather ≈ 0.5f0

    # --- the graph builds a matte plane node for it, sized by the analysed track
    g = VideoEditor.graphof(VideoEditor.effectiveclip(clip, 11))
    mn = g.nodes[findfirst(n -> n isa VideoEditor.PlaneNode{VideoEditor.MatteOp}, g.nodes)]
    @test mn.shape == VideoEditor.mattesize(clip.mattetrack)

    # --- project round-trip: seeds in the file, alpha in the sidecar
    path = joinpath(mktempdir(), "matte.toml")
    seq = Sequence([clip], 30.0)
    saveproject(path, seq)
    @test isfile(VideoEditor.mattefile(path, clip.id))
    seq2 = loadproject(path)
    t2 = seq2.clips[1].mattetrack
    @test t2 !== nothing
    @test t2.seeds == [0]
    @test t2.alpha == track.alpha
    # a missing sidecar must not lose the edit: seeds survive, alpha comes back zeroed
    rm(VideoEditor.mattefile(path, clip.id))
    seq3 = loadproject(path)
    @test seq3.clips[1].mattetrack.seeds == [0]
    @test all(==(0x00), seq3.clips[1].mattetrack.alpha)

    # --- a registered propagator takes over
    called = Ref(0)
    VideoEditor.registermatte!((frames, seeds; progress = nothing) -> begin
        called[] += 1
        fill(0xff, size(frames[1])..., length(frames))
    end)
    try
        @test VideoEditor.MATTEPROPAGATOR[] !== nothing
        t4 = VideoEditor.analyzematte!(clip, reader, Dict(0 => mask); maxside = 96)
        @test called[] == 1
        @test all(==(0xff), t4.alpha)
    finally
        VideoEditor.MATTEPROPAGATOR[] = nothing
    end
    @test VideoEditor.MATTEPROPAGATOR[] === nothing
    finally
        VideoEditor.MATTEPROPAGATOR[] = prevprop   # put back what was installed
    end
end

@testset "per-frame planes go through the graph" begin
    # A matte's alpha and a restoration's picture are whole images, not a handful
    # of numbers, so they reach the kernel as a `PlaneOp`: a pool-backed buffer
    # the graph writes through `Update` and the node's pass declares a read on.
    # Each one used to be a `KA.allocate` per frame behind a module global —
    # outside the pool, never freed, and invisible to the graph that ordered
    # everything around it. The properties below are what that route has to have.
    src = VideoSource(testvideo)
    frame = fill(VE.RGB{VE.N0f8}(0.2, 0.6, 0.9), src.width, src.height)

    clip = VE.Clip(src)
    alpha = zeros(UInt8, 160, 90, 10)
    alpha[40:120, 20:70, :] .= 0xff
    clip.mattetrack = MatteTrack(alpha, clip.src_in, [clip.src_in])
    push!(clip.effects, VE.Effect(MatteEffect(; strength = 1.0)))
    f0 = clip.src_in

    # the node carries the plane's shape, because that shape sizes a graph
    # resource and so belongs in the plan signature
    node = VE.graphof(clip).nodes[end]
    @test node isa VE.PlaneNode{VE.MatteOp}
    @test node.shape == VE.mattesize(clip.mattetrack)

    engine = VE.FxEngine(VE.KA.CPU())
    cp = VE.runchain!(engine, frame, clip, f0)
    out = copy(VE.chainimage(cp))
    @test out == VE.applymatte!(copy(frame), clip, f0; strength = 1.0)
    @test out[3, 3] == VE.RGB{VE.N0f8}(0, 0, 0)                     # background keyed
    @test out[src.width ÷ 2, src.height ÷ 2] != VE.RGB{VE.N0f8}(0, 0, 0)

    # a changed parameter is a store, not a new plan — the plane did not move
    plans = length(engine.plans)
    clip.effects[end] = VE.Effect(MatteEffect(; strength = 0.5))
    half = copy(VE.chainimage(VE.runchain!(engine, frame, clip, f0)))
    @test length(engine.plans) == plans
    @test half != out

    # the SAME frame of the SAME clip is already in the buffer: nothing to write
    b = only(cp.planes)
    VE.loadplane!(engine.store, b, clip, f0)
    @test (@atomic b.update.pending) === nothing

    # …but a re-propagation is a NEW track under an unchanged clip and frame, and
    # a stamp that could not see that would render the old alpha forever. This is
    # what replaced the eight explicit `freematteplanes!` calls.
    alpha2 = zeros(UInt8, 160, 90, 10); alpha2[10:40, 10:30, :] .= 0xff
    clip.mattetrack = MatteTrack(alpha2, clip.src_in, [clip.src_in])
    clip.effects[end] = VE.Effect(MatteEffect(; strength = 1.0))
    fresh = copy(VE.chainimage(VE.runchain!(engine, frame, clip, f0)))
    @test fresh == VE.applymatte!(copy(frame), clip, f0; strength = 1.0)
    @test fresh != out

    # outside the analysed range the node renders nothing — not the last plane
    @test copy(VE.chainimage(VE.runchain!(engine, frame, clip, f0 + 50))) == frame
    @test !only(cp.planes).active[]

    # the compositor's coverage is the chain's own matte binding, so "is this
    # layer keyed" has ONE answer: where the matte removed the background the
    # track below shows through, rather than the black the keying painted.
    base = VE.Clip(src)
    top = VE.Clip(src); top.track = 2
    top.mattetrack = MatteTrack(alpha, top.src_in, [top.src_in])
    push!(top.effects, VE.Effect(MatteEffect(; strength = 1.0)))
    red  = fill(VE.RGB{VE.N0f8}(1, 0, 0), src.width, src.height)
    blue = fill(VE.RGB{VE.N0f8}(0, 0, 1), src.width, src.height)
    eng2 = VE.FxEngine(VE.KA.CPU())
    canvas = Ref{Any}(nothing)
    @test VE.composite(eng2, [base, top], 0, (c, sf) -> c === base ? red : blue;
                       canvas = (src.width, src.height)) do cv
        canvas[] = copy(cv)
    end
    @test canvas[][src.width ÷ 2, src.height ÷ 2] == VE.RGB{VE.N0f8}(0, 0, 1)
    @test canvas[][3, 3] == VE.RGB{VE.N0f8}(1, 0, 0)

    # every buffer the engine owns comes back to the pool, planes included
    VE.emptyengine!(engine); VE.emptyengine!(eng2)
    @test isempty(engine.store.slots) && isempty(engine.plans)
    @test isempty(eng2.store.slots)
end

@testset "matte reads and propagates in one interleaved pass" begin
    # There used to be two phases — read every frame into a Vector, then
    # propagate — costing ~1:20, and reporting them as equal halves put the bar
    # at 50% after 4% of the wall clock (birds clip: decode+fx 8.8 s against
    # 187.0 s). It read as a hang, and it held the whole clip in RAM.
    #
    # Now frames are FETCHED (see `MatteFrames`), so there is one phase and no
    # share to tune. Asserted structurally, not by timing: the reader must still
    # be being called after propagation has started reporting. Collecting frames
    # up front would satisfy every monotonicity check below and fail this one.
    prevprop = VideoEditor.MATTEPROPAGATOR[]
    # This stand-in has to FETCH each frame as it goes, the way the real
    # propagator does. One that only touched `frames[1]` read a single frame
    # under streaming and could not tell deferred reads from eager ones.
    VideoEditor.registermatte!((frames, seeds; progress = nothing) -> begin
        n = length(frames)
        out = Array{UInt8}(undef, size(frames[1])..., n)
        for j in 1:n
            frames[j]
            out[:, :, j] .= 0xff
            progress === nothing || progress(j, n)
        end
        out
    end)
    try
        src = VideoSource(testvideo2)
        clip = Clip(src; src_in = 0, src_out = 11)
        mask = VideoEditor.seedmask(clip, (0.1, 0.1, 0.4, 0.4))
        nread = Ref(0)
        reader = let sr = VideoEditor.SequentialReader(src),
                     buf = VideoEditor.RGBFrame(undef, src.width, src.height)
            sf -> (nread[] += 1; VideoEditor.readframe!(buf, sr, Int(sf)); copy(buf))
        end
        n = VideoEditor.srclength(clip)

        fr = Float64[]
        readsattick = Int[]
        VideoEditor.analyzematte!(clip, reader, Dict(0 => mask); maxside = 96,
                                  progress = (d, t) -> (push!(fr, d / t);
                                                        push!(readsattick, nread[])))
        @test !isempty(fr)
        @test issorted(fr)                       # never goes backwards
        @test fr[end] ≈ 1.0                      # and lands exactly on full
        @test maximum(fr) <= 1.0 + 1e-9          # never past its own end
        # THE streaming property: reading is not finished when propagation
        # starts reporting. Collecting frames into a Vector first would make
        # every count below equal `n` from the first tick on.
        @test readsattick[1] < n
        @test issorted(readsattick)
        @test readsattick[end] >= n              # and all of them do get read

        # A MID-CLIP seed propagates backward then forward — `head + tail` is
        # `n + 1` steps, and reporting them against `n` used to walk the bar
        # past its own end.
        clipm = Clip(src; src_in = 0, src_out = 11)
        frm = Float64[]
        VideoEditor.analyzematte!(clipm, reader, Dict(5 => mask); maxside = 96,
                                  progress = (d, t) -> push!(frm, d / t))
        @test issorted(frm)
        @test maximum(frm) <= 1.0 + 1e-9
        @test frm[end] ≈ 1.0
    finally
        VideoEditor.MATTEPROPAGATOR[] = prevprop
    end
end

@testset "restore effect and cache" begin
    src = VideoSource(testvideo2)
    clip = Clip(src; src_in = 0, src_out = 6)

    # a stand-in restorer: 2x nearest upscale, so the test needs no model
    called = Ref(0)
    # a `do` block cannot declare keyword arguments, and the contract has one
    function fakerestore(frames; progress = nothing)
        called[] += 1
        map(frames) do f
            w, h = size(f)
            out = Matrix{VideoEditor.RGB{VideoEditor.N0f8}}(undef, 2w, 2h)
            for j in 1:2h, i in 1:2w
                # inverted, so applying it is observable — a plain nearest
                # upscale samples back down to the original pixel exactly and
                # would make the apply test vacuous
                c = f[cld(i, 2), cld(j, 2)]
                out[i, j] = VideoEditor.RGB{VideoEditor.N0f8}(
                    1 - VideoEditor.red(c), 1 - VideoEditor.green(c), 1 - VideoEditor.blue(c))
            end
            out
        end
    end
    VideoEditor.registerrestore!(fakerestore; scale = 2)
    try
        @test VideoEditor.hasrestoremodel()
        @test VideoEditor.restorescale() == 2

        sr = VideoEditor.SequentialReader(src)
        buf = VideoEditor.RGBFrame(undef, src.width, src.height)
        rd = sf -> (VideoEditor.readframe!(buf, sr, Int(sf)); buf)

        n = restorewindow!(clip, rd, 0, 4)
        @test n == 4
        @test called[] == 1
        @test VideoEditor.hasrestored(clip, 0)
        @test !VideoEditor.hasrestored(clip, 5)          # outside the window
        img = clip.restorecache.frames[0]
        @test size(img) == (2 * src.width, 2 * src.height)

        # applying: same function the graph node calls
        frame = copy(rd(0))
        orig = copy(frame)
        applyrestore!(frame, clip, 0; strength = 1.0)
        @test any(frame[i] != orig[i] for i in eachindex(frame))
        # strength 0 and an un-restored frame are both exact no-ops
        f2 = copy(orig); applyrestore!(f2, clip, 0; strength = 0.0)
        @test f2 == orig
        f3 = copy(orig); applyrestore!(f3, clip, 5; strength = 1.0)
        @test f3 == orig

        # the cache is bounded and evicts oldest-first
        c = VideoEditor.RestoreCache(2)
        for k in 1:4
            VideoEditor.putrestored!(c, k, fill(VideoEditor.RGB{VideoEditor.N0f8}(0, 0, 0), 2, 2))
        end
        @test length(c.frames) == 2
        @test haskey(c.frames, 4) && !haskey(c.frames, 1)

        # effect + registries
        @test VideoEditor.isneutral(RestoreEffect(0.0))
        @test !VideoEditor.isneutral(RestoreEffect(1.0))
        @test VideoEditor.effectfromdict(VideoEditor.effectdict(RestoreEffect(0.6))) ==
              RestoreEffect(0.6f0)
        k = VideoEditor.kindbyname(:restore)
        @test [pr.name for pr in k.params] == [:strength]
        @test k.read(RestoreEffect(0.5)) == (strength = 0.5,)

        # keyframable through the normal param path
        push!(clip.effects, VideoEditor.Effect(RestoreEffect()))
        cur = paramcurve!(clip, RestoreEffect, :strength)
        VideoEditor.setkey!(cur, 0, 0.0)
        VideoEditor.setkey!(cur, 5, 1.0)
        @test VideoEditor.findeffect(VideoEditor.effectiveclip(clip, 0), RestoreEffect).strength ≈ 0.0f0
        @test VideoEditor.findeffect(VideoEditor.effectiveclip(clip, 5), RestoreEffect).strength ≈ 1.0f0

        # the graph builds a restore plane node, sized by what the model returned
        g = VideoEditor.graphof(VideoEditor.effectiveclip(clip, 5))
        rn = g.nodes[findfirst(n -> n isa VideoEditor.PlaneNode{VideoEditor.RestoreOp}, g.nodes)]
        @test rn.shape == (2 * src.width, 2 * src.height)
    finally
        VideoEditor.RESTOREMODEL[] = nothing
        VideoEditor.clearrestore!(clip)
    end
    @test !VideoEditor.hasrestoremodel()
end

@testset "conform: a source at another framerate" begin
    fast = VideoSource(testvideo)     # 30 fps, 320×180, 120 frames
    slow = VideoSource(testvideo15)   # 15 fps, 180×320,  60 frames
    @test (fast.framerate, slow.framerate) == (30.0, 15.0)

    # the rate is EXACTLY 1 when the source matches — a native clip must never
    # pick up conform arithmetic (nor its rounding)
    @test VE.conformrate(fast, 30.0) === 1.0
    @test VE.conformrate(slow, 30.0) == 0.5
    @test VE.conformrate(fast, 15.0) == 2.0
    @test VE.conformrate(fast, 30.004) === 1.0    # inside the tolerance

    c = VE.Clip(slow, 0, slow.nframes, 0, (0.0, 0.0, 1.0, 1.0), 0.5)
    @test VE.srclength(c) == 60                    # source frames: unchanged
    @test VE.cliplength(c) == 120                  # timeline frames: twice as many
    # …and the point of it all: the clip lasts as long as its media does
    @test VE.cliplength(c) / 30.0 ≈ slow.nframes / slow.framerate

    # every source frame held twice, and the LAST timeline frame stays in range
    @test [VE.sourceframe(c, n) for n in 0:5] == [0, 0, 1, 1, 2, 2]
    @test VE.sourceframe(c, VE.cliplength(c) - 1) == slow.nframes - 1
    @test VE.timelineframe(c, VE.sourceframe(c, 17)) <= 17   # inverse, no overshoot
    @test VE.conformed(c) && !VE.conformed(VE.Clip(fast, 0, 10, 0, (0.0,0.0,1.0,1.0)))

    seq = VE.Sequence([VE.Clip(fast, 0, 60, 0, (0.0, 0.0, 1.0, 1.0)), c], 30.0)
    c.start = 60
    @test VE.seqlength(seq) == 60 + 120
    @test VE.clipend(seq.clips[1]) == c.start      # gapless

    @testset "split lands on a source frame, and the halves meet" begin
        s = VE.Sequence([VE.Clip(slow, 0, slow.nframes, 0, (0.0,0.0,1.0,1.0), 0.5)], 30.0)
        left = s.clips[1]
        right = VE.split!(s, 41)                   # ODD frame: no source frame starts there
        @test right !== nothing
        @test VE.clipend(left) == right.start      # no one-frame hole
        @test left.src_out == right.src_in         # and no source frame lost
        @test right.rate == 0.5
        @test VE.cliplength(left) + VE.cliplength(right) == 120
        # walking across the seam never lands in a gap
        @test all(VE.locate(s, n) !== nothing for n in 36:46)

        @test VE.joinclips!(s, 20) !== nothing     # …and it joins back up
        @test length(s.clips) == 1 && VE.cliplength(s.clips[1]) == 120
    end

    @testset "trim walks timeline frames, in/out points source frames" begin
        s = VE.Sequence([VE.Clip(slow, 0, slow.nframes, 0, (0.0,0.0,1.0,1.0), 0.5)], 30.0)
        cl = s.clips[1]
        VE.trimclip!(s, cl, 1, :right, 60)
        @test (VE.cliplength(cl), cl.src_out) == (60, 30)
        VE.trimclip!(s, cl, 1, :left, 20)
        @test (cl.start, cl.src_in) == (20, 10)
        @test VE.clipend(cl) == 60                 # the far edge did not move
        # trimming can't run past the media
        VE.trimclip!(s, cl, 1, :right, 10_000)
        @test cl.src_out <= slow.nframes
    end

    @testset "a fade keeps its LENGTH IN SECONDS on a conformed clip" begin
        cl = VE.Clip(slow, 0, slow.nframes, 0, (0.0,0.0,1.0,1.0), 0.5)
        VE.keyfade!(cl, 30, :in)                   # 30 timeline frames = 1 s at 30 fps
        @test VE.fadeinlength(cl) == 30            # reads back in the same unit
        ks = opacitycurve(cl).keys
        @test (ks[end].frame - ks[1].frame + 1) == 15   # …which is 15 SOURCE frames
    end

    @testset "the rate survives undo and the project file" begin
        s = VE.Sequence([VE.Clip(slow, 4, 50, 7, (0.1, 0.1, 0.8, 0.8), 0.5)], 30.0)
        snap = VE.snapshot(s)
        @test [c.rate for c in snap] == [0.5]
        s2 = VE.Sequence(VE.Clip[], 30.0); VE.restore!(s2, snap)
        @test [(c.rate, c.src_in, c.src_out, c.start) for c in s2.clips] ==
              [(0.5, 4, 50, 7)]

        path = joinpath(mktempdir(), "conform.videoedit.toml")
        VE.saveproject(path, s)
        back = VE.loadproject(path)
        @test [(c.rate, c.src_in, c.src_out, c.start) for c in back.clips] ==
              [(0.5, 4, 50, 7)]
        @test VE.cliplength(back.clips[1]) == VE.cliplength(s.clips[1])
        # a project written before conforming existed holds native clips only.
        # Edited through the PARSER, not with a regex over the text: a project
        # file is JSON (`saveproject` uses `JSON.print`; the docstring on
        # project.jl says why), and this dropped `\nrate = [0-9.]+` — TOML
        # syntax, which has not been written since. It matched nothing, `rate`
        # stayed 0.5, and the assertion below was simply false. Nobody saw it
        # because `interactions.jl` is included at the top of this file and its
        # testset throws on its known failures, so execution never reached here.
        d = VE.JSON.parse(read(path, String))
        delete!(d["clips"][1], "rate")
        open(io -> VE.JSON.print(io, d, 2), path, "w")
        @test VE.loadproject(path).clips[1].rate == 1.0
    end
end

@testset "one canvas: mixed resolution composites to the SEQUENCE format" begin
    # portrait 15 fps over landscape 30 fps — different rate AND different shape,
    # on two tracks so the composite path runs
    base = VideoSource(testvideo)      # 320×180
    over = VideoSource(testvideo15)    # 180×320, half rate
    c1 = VE.Clip(base, 0, 90, 0, (0.0, 0.0, 1.0, 1.0)); c1.track = 1
    c2 = VE.Clip(over, 0, 30, 20, (0.0, 0.0, 1.0, 1.0), 0.5); c2.track = 2
    seq = VE.Sequence([c1, c2], 30.0)

    canvas = VE.canvassize(seq)
    @test canvas == (320, 180)                       # the sequence's format
    stack = VE.clipsat(seq, 30)
    @test length(stack) == 2 && stack[end].source.width == 180   # top layer differs

    engine = VE.FxEngine(VE.KA.CPU())
    readers = Dict{String, Any}()
    dest = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(dest, seq, 30, readers, engine)
    @test size(dest) == canvas

    # …and that is the buffer the layer loop actually works on. Sizing it from
    # the top layer was a SECOND definition of the output format: with a
    # different pixel count it could not be copied into the export buffer at
    # all, and with the same count (320×180 vs 180×320 — exactly this pair) it
    # copied linearly and silently scrambled the picture.
    got = Ref((0, 0))
    VE.composite(engine, stack, 30,
                 (clip, _) -> get!(() -> VE.opendecoder(clip.source, engine.backend),
                                   readers, clip.source.path);
                 canvas = canvas, exact = true) do buf
        got[] = size(buf)
    end
    @test got[] == canvas
    @test got[] != (stack[end].source.width, stack[end].source.height)
    @test length(VE.RGBFrame(undef, canvas...)) == length(VE.RGBFrame(undef, 180, 320))
end

@testset "letterbox: material of another shape is fitted, not stretched" begin
    land = VideoSource(testvideo)      # 320×180
    port = VideoSource(testvideo15)    # 180×320, and half the rate
    engine = VE.FxEngine(VE.KA.CPU())
    readers = Dict{String, Any}()
    black = VE.RGB{VE.N0f8}(0, 0, 0)

    # fitting must be a NO-OP on material that already fits: same matrix, bit for
    # bit, so a single-format edit is not silently resampled by this feature
    @test VE.GPUFiltering.fitmatrix((0.0, 0.0, 1.0, 1.0), (320, 180), (320, 180)) ==
          VE.GPUFiltering.cropmatrix((0.0, 0.0, 1.0, 1.0), (320, 180), (320, 180))
    @test VE.GPUFiltering.fitmatrix((0.2, 0.1, 0.5, 0.5), (320, 180), (160, 90)) ==
          VE.GPUFiltering.cropmatrix((0.2, 0.1, 0.5, 0.5), (320, 180), (160, 90))
    # a crop with the canvas' aspect fills it → the visible region IS the crop
    @test VE.canvasrect(VE.Clip(land, 0, 30, 0, (0.25, 0.25, 0.5, 0.5)), (320, 180), (320, 180)) ==
          (0.25, 0.25, 0.5, 0.5)
    # one WITHOUT it is fitted: 160×45 is wider than 16:9, so the width fills and
    # the view opens up vertically around the crop's centre (bars top and bottom)
    r = VE.canvasrect(VE.Clip(land, 0, 30, 0, (0.25, 0.5, 0.5, 0.25)), (320, 180), (320, 180))
    @test (r[1], r[3]) == (0.25, 0.5)                    # width untouched
    @test r[4] ≈ 0.5 && r[2] + r[4] / 2 ≈ 0.5 + 0.25 / 2  # taller, same centre

    # a portrait clip ALONE on a landscape canvas: black bars, picture undistorted
    c1 = VE.Clip(land, 0, 30, 0, (0.0, 0.0, 1.0, 1.0))
    c2 = VE.Clip(port, 0, 30, 30, (0.0, 0.0, 1.0, 1.0), 0.5)   # clip 1 sets the canvas
    seq = VE.Sequence([c1, c2], 30.0)
    canvas = VE.canvassize(seq)
    @test canvas == (320, 180)
    dest = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(dest, seq, 40, readers, engine)   # only the portrait clip is here
    mid = canvas[2] ÷ 2
    @test dest[3, mid] == black && dest[canvas[1] - 2, mid] == black
    @test dest[canvas[1] ÷ 2, mid] != black           # …and the picture in between
    # 180 px fitted into 320×180 is 101 px wide → 109 px of bar each side
    bars = count(x -> dest[x, mid] == black, 1:canvas[1])
    @test bars == 2 * ((canvas[1] - round(Int, 180 * (180 / 320))) ÷ 2)

    # the manual reframe scales about the centre: more picture, fewer bars
    VE.withreframe!(c2, (1.9, 0.0, 0.0))
    zoomed = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(zoomed, seq, 40, readers, engine)
    @test count(x -> zoomed[x, mid] == black, 1:canvas[1]) < bars
    @test !VE.neutralframe(c2)
    # …and shifting moves it: pushed right, the LEFT bar grows
    VE.settransform(c2; scale = 1.9, x = 0.15, y = 0.0)
    shifted = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(shifted, seq, 40, readers, engine)
    leftbar(f) = something(findfirst(x -> f[x, mid] != black, 1:canvas[1]), canvas[1])
    @test leftbar(shifted) > leftbar(zoomed)
    VE.settransform(c2; scale = 1.0, x = 0.0, y = 0.0, rotation = 0.0)

    # a letterboxed layer STACKED over another shows the track below through its
    # bars — painting them black would black out the picture underneath
    c2.start = 0; c2.track = 2
    over = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(over, seq, 5, readers, engine)
    base = VE.RGBFrame(undef, canvas...)
    VE.renderframe!(base, VE.Sequence([c1], 30.0), 5, readers, engine)
    @test over[3, mid] == base[3, mid] && over[3, mid] != black
    @test over[canvas[1] - 2, mid] == base[canvas[1] - 2, mid]
    @test over[canvas[1] ÷ 2, mid] != base[canvas[1] ÷ 2, mid]   # …and the layer itself on top

    # scale/position are ordinary animatable params: keyframes, project, undo
    tfx = VE.findslot(c2, VE.TransformEffect)
    @test VE.param(tfx, :scale) !== nothing
    VE.param(tfx, :scale).value = 1.4
    @test VE.transformof(c2)[1] == 1.4 && VE.param(tfx, :scale).value == 1.4
    VE.param(tfx, :x).value = -0.2
    @test VE.transformof(c2)[2] == -0.2
    cur = paramcurve!(c2, VE.TransformEffect, :scale)
    VE.setkey!(cur, 0, 1.0); VE.setkey!(cur, 20, 2.0)
    @test VE.transformof(VE.effectiveclip(c2, 10))[1] ≈ 1.5   # baked at the frame
    path = joinpath(mktempdir(), "reframe.videoedit.toml")
    VE.saveproject(path, seq)
    back = VE.loadproject(path)
    @test VE.transformof(back.clips[end]) == VE.transformof(c2)
    @test VE.transformof(VE.snapshot(seq)[end]) == VE.transformof(c2)
    # A project written BEFORE the transform became an effect holds a `reframe`
    # tuple on the clip instead — `withreframe!` is what turns it back into one,
    # so feed it exactly that and check it arrives. (This used to strip a TOML
    # `reframe = [...]` line with a regex; project files are JSON, so it matched
    # nothing and the assertion below was reading a field that no longer exists.)
    d = VE.JSON.parse(read(path, String))
    d["clips"][end]["reframe"] = [1.25, 0.1, 0.0]
    open(io -> VE.JSON.print(io, d, 2), path, "w")
    @test VE.transformof(VE.loadproject(path).clips[end])[1] ≈ 1.25
    # …and with NEITHER a `reframe` tuple nor a transform effect, the clip is
    # placed by the plain fit. Both have to go: this clip carries a serialised
    # `TransformEffect` from the `settransform` above, and dropping only the
    # legacy tuple leaves that one answering `transformof`.
    delete!(d["clips"][end], "reframe")
    filter!(e -> get(e, "type", "") != "transform", d["clips"][end]["effects"])
    open(io -> VE.JSON.print(io, d, 2), path, "w")
    @test VE.transformof(VE.loadproject(path).clips[end]) == VE.NEUTRALFRAME
end

include("scenespec.jl")   # a Makie scene as data: paths, animation, round trip
include("overlays.jl")

# LAST, and that is the whole point. `interactions.jl` throws at the end of the
# file on its known failures, which aborts this one — so for as long as it was
# included in the middle, everything after it only ran when GL happened to be
# unavailable and it was skipped. That is not hypothetical: on 2026-08-24 a run
# without a GL context reached this half for the first time and found `conform`
# asserting against TOML syntax in a JSON file, and `letterbox` writing to
# `Clip.reframe`, a field removed when the transform became an effect. Both had
# been dead for as long as they had been passed over.
#
# `fuzz.jl` had NEVER run as part of the suite for the same reason, and the one
# time it did it found two real bugs (Mantle's `nothing`-as-constraint and
# `showframe!`'s no-clip path).
#
# The cost of this order is that the reported totals now cover the whole file
# rather than "everything up to interactions.jl", so the old 259|18|2 and
# 548|6|1 baselines are not comparable to what comes out now.
# BOTH INSIDE ONE OUTER TESTSET, and that is the whole trick. Each of these
# files ends with failing testsets, and a TOP-LEVEL `@testset` throws at its end
# — which aborts the file that included it. Ordering them cannot fix that: with
# two throwers, whichever runs first takes the other with it, which is exactly
# what happened when `fuzz.jl` was moved in front (it ran, found a `MethodError`
# on iteration one, threw, and `interactions.jl` never started).
#
# A NESTED testset reports to its parent instead of throwing, so wrapping both
# makes them run to completion and lets this one throw once, at the end, with
# everything counted.
canui && @testset "GUI" begin
    include("fuzz.jl")            # random edit programs vs the picture (needs a Player)
    include("interactions.jl")
end
