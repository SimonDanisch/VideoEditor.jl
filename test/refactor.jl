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
    g = VE.graphof(clip)
    kinds = [nameof(typeof(n)) for n in g.nodes]
    @test :MotionNode in kinds && :ColorTrackNode in kinds
    # a slot switched off renders nothing
    for s in clip.effects
        s.enabled = false
    end
    g2 = VE.graphof(clip)
    @test !(:MotionNode in [nameof(typeof(n)) for n in g2.nodes])
end

@testset "effect links" begin
    src = VideoSource(testvideo)
    seq = Sequence([Clip(src; src_in = 0, src_out = 30, start = 0),
                    Clip(src; src_in = 30, src_out = 60, start = 30)], 30.0)
    a, b = seq.clips
    seteffect!(a, VE.OpacityEffect(1.0f0))
    seteffect!(b, VE.OpacityEffect(0.0f0))
    sa, sb = a.effects[1], b.effects[1]

    VE.linkeffects!(seq, a, sa, b, sb; role = :fadesinto, backrole = :fadesfrom)
    @test length(sa.links) == 1 && length(sb.links) == 1

    # a link resolves to the OTHER clip's slot, by id — not by index or identity,
    # so it survives sorting, undo and a project round trip
    got = VE.resolvelink(seq, a, sa.links[1])
    @test got !== nothing && got[1] === b && got[2] === sb

    # the walk visits each slot once: a paired link's first step is the way back
    chain = VE.linkchain(seq, a, sa)
    @test length(chain) == 1 && chain[1][2] === sb

    # what a card inlines is DIRECT links only — the target's own card shows its
    # links, so following the chain here would draw the same sliders twice
    inl = VE.linkedparams(seq, a, sa)
    @test length(inl) == 1 && inl[1][2] === sb && inl[1][4] === :fadesinto

    # deleting one half leaves the other dangling, and pruning says how many
    VE.removeslot!(b, sb.id)
    @test VE.resolvelink(seq, a, sa.links[1]) === nothing
    @test VE.prunelinks!(seq) == 1
    @test isempty(sa.links)

    # links survive a project round trip
    seteffect!(b, VE.OpacityEffect(0.0f0))
    sb2 = b.effects[1]
    VE.linkeffects!(seq, a, sa, b, sb2; role = :fadesinto, backrole = :fadesfrom)
    path = tempname() * ".videoedit.toml"
    saveproject(path, seq)
    seq2 = loadproject(path)
    rm(path; force = true)
    a2 = seq2.clips[1]
    s2 = a2.effects[1]
    @test length(s2.links) == 1
    r = VE.resolvelink(seq2, a2, s2.links[1])
    @test r !== nothing && r[1] === seq2.clips[2]
    @test s2.links[1].role === :fadesinto
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
