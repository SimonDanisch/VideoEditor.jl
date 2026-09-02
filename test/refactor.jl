# Model-level tests for the one-registry / one-panel refactor. No window: these
# are about the data — what is registered, what a link resolves to, what the
# palette would rank — so they run fast and cannot go flaky on event timing.
# The GUI half (cards, filtering, the palette's keyboard) lives in interactions.jl.

@testset "one registry" begin
    kinds = VE.effectkinds()
    @test !isempty(kinds)
    # every kind is reachable by name, and names are unique
    @test length(unique(k.name for k in kinds)) == length(kinds)
    for k in kinds
        @test VE.kindbyname(k.name) === k
    end
    # what used to be three registries: a built-in, a plugin, and a tool
    @test VE.kindbyname(:color) !== nothing        # built-in
    @test VE.kindbyname(:vignette) !== nothing     # registered through the plugin API
    @test VE.kindbyname(:stabilize) !== nothing    # was an EditorTool
    @test VE.kindbyname(:nosuchkind) === nothing

    # a kind that has BOTH parameters and a card body is the point of the merge:
    # the matte used to be two entries with the same name in two registries
    m = VE.kindbyname(:matte)
    @test !isempty(m.params) && m.make !== nothing && m.body !== nothing && m.activate !== nothing

    # make/read/matches round-trip for every kind that can build an effect
    for k in kinds
        k.make === nothing && continue
        e = k.make(VE.defaults(k))
        @test k.matches(e)
        @test VE.effectkindfor(e) === k
        got = k.read(e)
        for p in k.params
            @test haskey(got, p.name)
        end
    end

    # Every declared parameter becomes a `Param` on a fresh entry of that kind —
    # no global index to look it up in, and none needed.
    for k in kinds
        k.make === nothing && continue
        fx = VE.Effect(k)
        for pr in k.params
            @test VE.param(fx, pr.name) !== nothing
        end
    end
end

@testset "analyses are stack slots" begin
    seq = Sequence([Clip(VideoSource(testvideo); src_in = 0, src_out = 40, start = 0)], 30.0)
    clip = seq.clips[1]
    @test isempty(clip.effects)

    # attaching an analysis attaches the slot that applies it — one call, both
    track = VE.MotionTrack([VE.Mat3f(1I) for _ in 1:40], clip.src_in, :similarity)
    VE.setmotiontrack!(clip, track)
    @test clip.motiontrack === track
    @test count(s -> VE.op(s) isa VE.StabilizeEffect, clip.effects) == 1

    # …and detaching removes both
    VE.setmotiontrack!(clip, nothing)
    @test clip.motiontrack === nothing
    @test isempty(clip.effects)

    # the colour analysis carries its strength on the EFFECT, so it is tunable
    # and keyframable without re-analysing
    ct = VE.ColorTrack([VE.Vec3f(1, 1, 1) for _ in 1:40], [VE.Vec3f(0, 0, 0) for _ in 1:40],
                       clip.src_in, 0.6f0, 0.3f0)
    VE.setcolortrack!(clip, ct)
    fx = only(filter(s -> VE.op(s) isa VE.FlickerEffect, clip.effects))
    @test VE.op(fx).strength ≈ 0.6f0

    # the render graph takes them from the STACK now, in stack order
    VE.setmotiontrack!(clip, track)
    dims = (clip.source.width, clip.source.height)
    g = VE.graphof!(clip, dims)
    kinds = [nameof(typeof(n)) for n in g.nodes]
    @test :MotionNode in kinds && :ColorTrackNode in kinds
    # …and it is the STRUCTURE: switching a slot off does not change it. Whether a
    # pass does its work is a per-frame flag (`active`), because "is this blur
    # switched on right now" is a value and a value must not decide which graph is
    # compiled — a σ reaching zero mid-drag would otherwise recompile.
    for s in clip.effects
        s.enabled[] = false
    end
    @test VE.graphof!(clip, dims) === g
end

@testset "a parameter can be driven by another" begin
    src = VideoSource(testvideo)
    seq = Sequence([Clip(src; src_in = 0, src_out = 30, start = 0),
                    Clip(src; src_in = 30, src_out = 60, start = 30)], 30.0)
    a, b = seq.clips
    seteffect!(a, VE.OpacityEffect(1.0f0))
    seteffect!(b, VE.OpacityEffect(0.25f0))
    pa = VE.param(a.effects[1], :opacity)
    pb = VE.param(b.effects[1], :opacity)

    # `a`'s opacity is one minus `b`'s: two nodes with an edge between them, which
    # is what a cross-dissolve IS. Ids, not objects — they survive a round trip.
    pa.input = VE.ParamInput(:invert, VE.ParamRef(:opacity; clip = b.id))
    VE.bindinputs!(seq)
    @test VE.isdriven(pa)
    @test VE.valueat(pa, 0) ≈ 0.75f0
    VE.setvalue!(pb, 0.5f0, 0)
    @test VE.valueat(pa, 0) ≈ 0.5f0          # …and it FOLLOWS, there is no copy

    # a `:mix` takes three inputs: two sources and the number between them
    @test VE.inputarity(:mix) == 3
    @test VE.lerp(0.0, 10.0, 0.25) ≈ 2.5

    # nothing drives itself, and nothing drives in a circle
    pa.input = VE.ParamInput(:copy, VE.ParamRef(:opacity))
    VE.bindinputs!(seq)
    @test !VE.isdriven(pa)
    pa.input = VE.ParamInput(:copy, VE.ParamRef(:opacity; clip = b.id))
    pb.input = VE.ParamInput(:copy, VE.ParamRef(:opacity; clip = a.id))
    VE.bindinputs!(seq)
    # ONE edge of the loop is cut, not both: what has to be true is that reading a
    # value terminates, and refusing more than the offending edge would throw away
    # a binding the user made for no reason.
    @test !(VE.isdriven(pa) && VE.isdriven(pb))
    @test VE.valueat(pa, 0) isa Real && VE.valueat(pb, 0) isa Real

    # a target that is deleted leaves the edge DANGLING, not broken: the ids are
    # what the user wrote, and undo has to be able to bring the target back
    pb.input = nothing
    pa.input = VE.ParamInput(:invert, VE.ParamRef(:opacity; clip = b.id))
    VE.bindinputs!(seq)
    @test VE.isdriven(pa)
    VE.removeslot!(b, b.effects[1].id)
    VE.bindinputs!(seq)
    @test !VE.isdriven(pa)
    @test VE.valueat(pa, 0) ≈ pa.curve[].keys[1].value   # …and it reads as its own value

    # …and an edge survives a project round trip, because the file holds its ids
    seteffect!(b, VE.OpacityEffect(0.25f0))
    VE.bindinputs!(seq)
    path = tempname() * ".videoedit"
    saveproject(path, seq)
    seq2 = loadproject(path)
    rm(path; force = true)
    p2 = VE.opacityparam(seq2.clips[1])
    @test p2.input !== nothing && p2.input.op === :invert
    @test VE.isdriven(p2)
end

@testset "a blend pairing is an edge that carries no value" begin
    src = VideoSource(testvideo)
    seq = Sequence([Clip(src; src_in = 0, src_out = 30, start = 0),
                    Clip(src; src_in = 30, src_out = 60, start = 30)], 30.0)
    a, b = seq.clips
    VE.keyfade!(b, 6, :in)
    VE.pairblend!(seq, b, a)
    p = VE.opacityparam(b)
    # it POINTS, it does not drive: `b`'s fade is its own curve, because fading
    # both sides darkens the middle of a dissolve
    @test p.input !== nothing && p.input.op === :pairedwith
    @test !VE.isdriven(p)
    @test VE.isanimated(p)
    @test VE.blendpartner(seq, b) == 1
    @test first(VE.blends(seq)) == (1, 2, 6)
    VE.pairblend!(seq, b, nothing)
    @test VE.blendpartner(seq, b) === nothing
end

@testset "commands" begin
    # every registered command is well formed
    for c in VE.COMMANDS
        @test !isempty(c.label)
        @test c.run !== nothing
    end
    @test length(unique(c.name for c in VE.COMMANDS)) == length(VE.COMMANDS)

    # ranking: a label prefix beats a word-boundary hit beats a keyword
    split = only(filter(c -> c.name === :split, VE.COMMANDS))
    @test VE.matchscore(split, "split") == 4.0
    @test VE.matchscore(split, "at play") == 3.0
    @test VE.matchscore(split, "blade") == 1.5      # a keyword, not in the label
    @test VE.matchscore(split, "zzz") == 0.0
    @test VE.subsequence("split at playhead", "sap")

    # the palette teaches the keyboard: transport commands carry their shortcut
    @test !isempty(only(filter(c -> c.name === :play_pause, VE.COMMANDS)).shortcut)
end
