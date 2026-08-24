# Overlays (src/overlays.jl): Makie plots composed over the finished canvas.
# Kept in its own file like `interactions.jl`/`fuzz.jl` so the GL-dependent
# half can be skipped and so it runs even when an earlier testset fails.

@testset "overlays: the model" begin
    ov = Overlay(:text; text = "HELLO", color = "cyan", start = 10, stop = 20, x = 0.25)
    # declared scalars are params (keyframable); everything else is a setting
    @test ov.params.x == 0.25 && ov.params.size == 0.08   # kind's default
    @test ov.settings == (text = "HELLO", color = "cyan")
    @test !VE.showsat(ov, 9) && VE.showsat(ov, 10) && VE.showsat(ov, 19) && !VE.showsat(ov, 20)
    st = VE.overlaystate(ov, 10)
    @test st.text == "HELLO" && st.x == 0.25

    # a curve overrides its param — keyed by TIMELINE frame, not source frame
    VE.setoverlaykey!(ov, :x, 10, 0.0); VE.setoverlaykey!(ov, :x, 20, 1.0)
    @test VE.overlaystate(ov, 15).x ≈ 0.5
    @test VE.overlayvalue(ov, :x, 20) ≈ 1.0
    @test VE.overlayvalue(ov, :y, 15) == ov.params.y     # unanimated: the static value

    seq = Sequence(VideoSource(testvideo))
    o2 = addoverlay!(seq, :bar; y = 0.0, height = 0.2)
    @test length(seq.overlays) == 1 && VE.overlaybyid(seq, o2.id) === o2
    @test removeoverlay!(seq, o2.id) && isempty(seq.overlays)
    @test !removeoverlay!(seq, o2.id)
end

@testset "overlays: anything in the state is keyframable, not only declared params" begin
    # A kind's `FxParam` list used to BE the set of animatable things — a curve on
    # any other key was silently dropped. That cannot work for a generic Makie
    # scene, where what is animatable is every attribute of every plot in it and
    # no list can be written down in advance. Such attributes arrive as settings.
    ov = Overlay(:text; text = "HELLO", color = "cyan", start = 0, stop = 100)
    @test haskey(ov.settings, :color) && !haskey(ov.params, :color)

    VE.setoverlaykey!(ov, :color, 0, 0.0)
    VE.setoverlaykey!(ov, :color, 10, 1.0)
    @test VE.overlaystate(ov, 5).color ≈ 0.5        # the curve beats the setting
    @test VE.overlaystate(ov, 0).color ≈ 0.0

    # …and a key that is neither a param nor a setting still lands, which is what
    # a path into a plot spec will be.
    VE.setoverlaykey!(ov, Symbol("plots[1].markersize"), 0, 4.0)
    VE.setoverlaykey!(ov, Symbol("plots[1].markersize"), 20, 24.0)
    @test VE.overlaystate(ov, 10)[Symbol("plots[1].markersize")] ≈ 14.0

    # An unanimated setting is untouched, and the reserved keys still arrive.
    ov2 = Overlay(:text; text = "X", color = "red", start = 0, stop = 10)
    @test VE.overlaystate(ov2, 3; framerate = 25).color == "red"
    @test VE.overlaystate(ov2, 3; framerate = 25).frame == 3
end

@testset "overlays: the reserved frame/framerate keys" begin
    @test VE.timecodestring(0, 60) == "00:00:00:00"
    @test VE.timecodestring(150, 60) == "00:00:02:30"
    @test VE.timecodestring(60 * 61 + 5, 60) == "00:01:01:05"
    @test VE.timecodestring(-3, 25) == "00:00:00:00"       # never a negative timecode
    # every overlay's state carries where the video is, so a kind can follow the
    # playhead with nothing keyframed
    ov = Overlay(:timecode)
    st = VE.overlaystate(ov, 42; framerate = 30)
    @test st.frame == 42 && st.framerate == 30.0
end

@testset "overlays: morphing a series" begin
    @test VE.resampleseries([0.0, 1.0], 3) ≈ [0.0, 0.5, 1.0]
    @test VE.resampleseries([0.0, 1.0, 2.0], 2) ≈ [0.0, 2.0]
    @test VE.resampleseries([5.0], 4) ≈ fill(5.0, 4)
    # different lengths morph through a common resampling — 0 is A, 1 is B
    a, b = [0.0, 0.0, 0.0], [1.0, 1.0]
    @test VE.morphseries(a, b, 0.0) ≈ [0.0, 0.0, 0.0]
    @test VE.morphseries(a, b, 1.0) ≈ [1.0, 1.0, 1.0]
    @test VE.morphseries(a, b, 0.25) ≈ [0.25, 0.25, 0.25]
    @test VE.morphseries(a, Float64[], 1.0) ≈ a          # nothing to morph into
end

@testset "overlays: survive save/load" begin
    seq = Sequence(VideoSource(testvideo))
    ov = addoverlay!(seq, :text; text = "TAKE 42", color = "white", start = 5, stop = 40, size = 0.15)
    VE.setoverlaykey!(ov, :opacity, 5, 0.0)
    VE.setoverlaykey!(ov, :opacity, 15, 1.0)
    path = joinpath(mktempdir(), "overlay.videoedit.toml")
    VE.saveproject(path, seq)
    back = VE.loadproject(path)
    @test length(back.overlays) == 1
    b = back.overlays[1]
    @test b.id == ov.id && b.kind === :text && (b.start, b.stop) == (5, 40)
    @test b.settings.text == "TAKE 42" && b.params.size ≈ 0.15
    @test VE.overlaystate(b, 10).opacity ≈ 0.5           # the curve came back too
    # a project written before overlays existed still loads
    plain = joinpath(mktempdir(), "plain.videoedit.toml")
    VE.saveproject(plain, Sequence(VideoSource(testvideo)))
    @test !occursin("overlays", read(plain, String))
    @test isempty(VE.loadproject(plain).overlays)
end

if canui
    @testset "overlays: Makie composes the canvas, losslessly" begin
        W, H = 320, 200
        base = rand(VE.RGB{VE.N0f8}, W, H)

        # THE property the whole design rests on: a frame that goes through the
        # canvas scene comes back BIT-IDENTICAL. Nothing else here is safe if
        # this isn't — every exported pixel of every clip passes through it.
        dest = copy(base)
        VE.drawoverlays!(dest, VE.Overlay[], 0)
        @test dest == base                                # no overlays at all
        ov = Overlay(:text; text = "HELLO", start = 0, stop = 100, x = 0.5, y = 0.5, size = 0.2)
        VE.drawoverlays!(dest, [ov], 999)
        @test dest == base                                # …and outside its span

        # …and with an overlay ACTUALLY drawn, everything it doesn't cover
        VE.drawoverlays!(dest, [ov], 10)
        changed = findall(dest .!= base)
        @test !isempty(changed)
        # an overlay touches ONLY the pixels it covers — the picture underneath
        # is untouched, not re-encoded
        @test all(90 .<= getindex.(changed, 1) .<= 230)
        @test all(70 .<= getindex.(changed, 2) .<= 130)

        # y points UP and text is upright: an overlay high in the frame lands in
        # the video's TOP rows (small j), because the frame's first row is its top
        top = Overlay(:bar; y = 0.8, height = 0.2, x = 0.0, width = 1.0, opacity = 1.0)
        d2 = copy(base); VE.drawoverlays!(d2, [top], 0)
        rows = getindex.(findall(d2 .!= base), 2)
        @test maximum(rows) <= H ÷ 4

        # a timecode changes with the playhead on its own — no keyframes at all
        tc = Overlay(:timecode; x = 0.5, y = 0.5, size = 0.15)
        shots = map((0, 90)) do n
            d = copy(base); VE.drawoverlays!(d, [tc], n; framerate = 30)
            d
        end
        @test shots[1] != base && shots[1] != shots[2]

        # a curve at progress 0 draws NOTHING — not a two-point stub, which is
        # what rounding the reveal UP to a drawable line used to leave on screen
        vals = collect(range(0, 1, length = 50))
        for p in (0.0, 0.02)
            d = copy(base); VE.drawoverlays!(d, [Overlay(:curve; values = vals, progress = p)], 0)
            @test d == base
        end
        d = copy(base); VE.drawoverlays!(d, [Overlay(:curve; values = vals, progress = 1.0)], 0)
        @test d != base

        # Composing a frame must not disturb the EDITOR's window. GLMakie hands
        # every scene-taking `Screen` constructor its singleton offscreen screen
        # and empties it first — and `display(player.fig)` uses that same
        # constructor, so a canvas rendered the easy way evicted the editor's
        # figure from its own window (measured: 32 plots on that screen, then 3).
        # The canvas must own a screen from the pool instead.
        import GLMakie
        GLMakie.activate!(visible = false)
        efig = VE.Makie.Figure(size = (400, 300))
        VE.Makie.scatter!(VE.Makie.Axis(efig[1, 1]), 1:10, rand(10))
        escreen = display(efig)
        nplots = length(escreen.renderlist)
        @test nplots > 0
        VE.drawoverlays!(copy(base), [ov], 10)
        @test escreen.scene === efig.scene           # still showing the editor's figure
        @test length(escreen.renderlist) == nplots   # …with all of its plots
        @test isopen(escreen)

        # re-registering a kind is advertised as live, so already-built plots
        # must be REBUILT: they belong to the previous `draw` closure and don't
        # just render stale, they raise against the new one
        pspec = [VE.FxParam(:size, "Size"; min = 0.0, max = 1.0, default = 0.2)]
        probe(h) = (scene, canvas, state) ->
            (VE.Makie.poly!(scene, VE.olift(s -> VE.Makie.Rect2f(10, 10, s.size * canvas[1], h), state);
                            color = :red); nothing)
        registeroverlay!(:testprobe, "Probe", pspec, probe(20))
        pr = Overlay(:testprobe; size = 0.2)
        d = copy(base); VE.drawoverlays!(d, [pr], 0); short = count(d .!= base)
        registeroverlay!(:testprobe, "Probe", pspec, probe(60))
        d = copy(base); VE.drawoverlays!(d, [pr], 0)
        @test count(d .!= base) > 2 * short

        # keyframes move it
        VE.setoverlaykey!(ov, :x, 0, 0.2); VE.setoverlaykey!(ov, :x, 50, 0.8)
        centres = map((0, 50)) do n
            d = copy(base); VE.drawoverlays!(d, [ov], n)
            c = findall(d .!= base)
            sum(getindex.(c, 1)) / length(c)
        end
        @test centres[1] < W * 0.35 && centres[2] > W * 0.65
    end

    @testset "overlays: end to end through exportvideo" begin
        using Statistics: mean
        seq = Sequence(VideoSource(testvideo))
        plain = joinpath(mktempdir(), "plain.mp4")
        exportvideo(plain, seq; audio = false, encoder_options = (crf = 18, preset = "ultrafast"))

        addoverlay!(seq, :bar; x = 0.0, y = 0.0, width = 1.0, height = 0.25, opacity = 1.0,
                    color = "black", start = 10, stop = 20)
        titled = joinpath(mktempdir(), "titled.mp4")
        exportvideo(titled, seq; audio = false, encoder_options = (crf = 18, preset = "ultrafast"))

        function grab(path, n)
            r = VideoIO.openvideo(path)
            f = nothing
            for _ in 0:n
                f = VideoIO.read(r)
            end
            close(r)
            return f
        end
        red(m) = Float64.(getfield.(m, :r))
        # the bar sits at the BOTTOM of the frame (y = 0) — VideoIO's LAST rows
        a15, b15 = grab(plain, 15), grab(titled, 15)
        band = size(a15, 1) - 10
        @test mean(red(b15[band, :])) < 0.1                          # blacked out
        @test mean(abs.(red(a15[band, :]) .- red(b15[band, :]))) > 0.1
        # …and a frame before the overlay starts is the same picture
        @test mean(abs.(red(grab(plain, 0)) .- red(grab(titled, 0)))) < 0.02
    end
end
