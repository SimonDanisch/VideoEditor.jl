# What an AGENT sees (src/agentview.jl): the images must BE the edit, carry a
# manifest that resolves cell → time, and zoom without a scan.

@testset "agent view: sheets, zoom, change search" begin
    src = VideoSource(testvideo)                 # 320×180, 120 frames @30
    seq = Sequence(src)

    @testset "contact sheet is one image of many frames" begin
        sheet, man = VE.contactsheet(seq; cells = 16, size = 320)
        @test length(man) == 16
        @test first(man).time == 0.0
        @test issorted([c.time for c in man])    # ascending: decoders read forward
        @test last(man).frame == VE.seqlength(seq) - 1
        # cells keep the SOURCE aspect (square cells would spend tokens on bars)
        cols = maximum(c.col for c in man); rows = maximum(c.row for c in man)
        cw = size(sheet, 1) ÷ cols; ch = size(sheet, 2) ÷ rows
        @test isapprox(cw / ch, src.width / src.height; rtol = 0.12)
        @test 0.7 * 320 <= maximum(size(sheet)) <= 1.4 * 320   # ~the requested size
        @test any(c -> c != RGB{N0f8}(0, 0, 0), sheet)         # not a black page
    end

    @testset "a cell resolves to a time, and zooming keeps the picture" begin
        _, man = VE.contactsheet(seq; cells = 16, size = 320)
        cell = man[9]
        # the frame the manifest names is the frame the export writes there
        grab = VE.framegrab(seq, cell.time; width = src.width)
        engine = VE.FxEngine(VE.KA.CPU()); readers = Dict{String, Any}()
        want = VE.RGBFrame(undef, src.width, src.height)
        VE.renderframe!(want, seq, cell.frame, readers, engine)
        foreach(close, values(readers)); VE.emptyengine!(engine)
        @test size(grab) == size(want)
        @test maximum(abs.(Float32.(getfield.(grab, :r)) .- Float32.(getfield.(want, :r)))) < 0.02
        # zooming into that cell's neighbourhood covers it
        _, zman = VE.contactsheet(seq, cell.time - 0.2, cell.time + 0.2; cells = 8, size = 240)
        @test zman[1].time <= cell.time <= zman[end].time
    end

    @testset "spatial zoom returns real detail" begin
        whole = VE.framegrab(seq, 1.0; width = 160)
        quarter = VE.regiongrab(seq, 1.0, (0.25, 0.25, 0.25, 0.25); width = 160)
        @test size(quarter, 1) == 160
        @test size(quarter, 2) == 160 * (0.25 * src.height) ÷ (0.25 * src.width) ||
              abs(size(quarter, 2) / size(quarter, 1) - src.height / src.width) < 0.1
        @test quarter != whole                     # it really is a different framing
    end

    @testset "filmstrip reads a short span in order" begin
        strip, man = VE.filmstrip(seq, 1.0, 2.0; count = 5, height = 60)
        @test length(man) == 5
        @test size(strip, 2) == 60
        @test issorted([c.frame for c in man])
    end

    @testset "change search bisects instead of scanning" begin
        # two different sources back to back: the cut is a hard change
        src2 = VideoSource(testvideo2)
        two = Sequence(src)
        push!(two.clips, VE.Clip(src2, 0, src2.nframes, VE.seqlength(two), (0.0, 0.0, 1.0, 1.0)))
        cut = VE.seqlength(Sequence(src)) / two.framerate
        found = VE.findchange(two, cut - 1.0, cut + 1.0; threshold = 0.08)
        @test found !== nothing
        @test abs(found[1] - cut) < 0.1            # lands ON the cut
        # nothing to find inside ONE clip's quiet stretch
        @test VE.findchange(two, 0.1, 0.4; threshold = 0.6) === nothing
    end

    @testset "summary tells an agent what it is looking at" begin
        s = VE.viewsummary(seq)
        @test s.frames == VE.seqlength(seq)
        @test s.canvas == (src.width, src.height)
        @test length(s.clips) == 1
        @test s.clips[1].stabilized == false
    end
end
