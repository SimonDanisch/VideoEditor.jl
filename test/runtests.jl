using VideoEditor
using Test
import VideoEditor as VE
import VideoEditor.VideoIO as VideoIO
import FFMPEG_jll

testvideo = joinpath(mktempdir(), "test.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 4 -c:v libx264 -g 30 -pix_fmt yuv420p $testvideo`,
             stdout = devnull, stderr = devnull))
# second source: same framerate (required), different resolution and content
testvideo2 = joinpath(mktempdir(), "test2.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i smptebars=size=480x270:rate=30 -t 3 -c:v libx264 -g 30 -pix_fmt yuv420p $testvideo2`,
             stdout = devnull, stderr = devnull))

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
    reader = VideoIO.openvideo(out)
    early = read(reader)                    # clip 1 content (testsrc2)
    seek(reader, 5.0)
    late = read(reader)                     # clip 2 content (smpte bars)
    close(reader)
    reddiff = mean(abs.(Float32.(getfield.(early, :r)) .- Float32.(getfield.(late, :r))))
    @test reddiff > 0.1  # the two sources' content really alternates

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
canui && include("interactions.jl")

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
