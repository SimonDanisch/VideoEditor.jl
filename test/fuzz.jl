# Fuzzing the effect graph with random EDIT PROGRAMS.
#
# The bugs this exists for are not crashes — they are pictures that quietly stop
# matching what the edit says (Simon, 2026-07-27: "we have lots of problems in
# the effects graph … I stabilized a clip, cut it in half and blended the clips
# together and the playback during the blend is completely buggy"). That one was
# a stabilized clip's crop applied TWICE while a blend was on screen, and no
# existing test could see it: they all assert on the MODEL, never on what the
# preview actually shows.
#
# So the fuzzer builds random edits (split, track moves, blends, effects,
# keyframes, crops, stabilization, trims, undo) and after every step checks the
# two things that tie the picture to the edit, at random frames:
#
#   PIXELS  — the preview buffer is EXACTLY what the export writes for that
#             frame (same graph, same layers; both exact on the CPU tier).
#   FRAMING — the visible region (axis limits over the buffer) is EXACTLY the
#             framing the export bakes: a clip's crop when one clip is on
#             screen, the whole canvas when layers are composited (their crops
#             are already baked into it).
#
# A failure prints the seed and the program, so it replays exactly.

using Random
import VideoEditor as VE
using VideoEditor: Clip, FxSlot, MotionTrack, Mat3f, RGBFrame, clipsat, clipend, cliplength,
                   effectiveclip, locate, ntracks, seqlength

"A synthetic stabilization: real transforms (no analysis to run) plus the crop
that hides their borders — the state a stabilized clip is in."
function fakestabilize!(clip::Clip, rng::AbstractRNG)
    n = cliplength(clip)
    amp = 4.0f0 + 8.0f0 * Float32(rand(rng))
    tf = [Mat3f(1, 0, 0, 0, 1, 0, amp * sin(i / 7), amp * cos(i / 11), 1) for i in 1:n]
    track = MotionTrack(tf, clip.src_in, :similarity)
    track.basecrop = clip.crop
    clip.motiontrack = track
    clip.crop = VE.cropintersect(track.basecrop,
                                 VE.bordercrop(track, clip.source.width, clip.source.height))
    return clip
end

"The preview's visible region in buffer-normalized coordinates — what the user SEES."
function visiblerect(player)
    lims = player.previewaxis.finallimits[]
    W, H = size(player.frame[])
    return (lims.origin[1] / W, lims.origin[2] / H, lims.widths[1] / W, lims.widths[2] / H)
end

"Render timeline frame `n` the way the EXPORT does (exact decode, crops baked)."
function exportframe(seq, n, engine, readers)
    cs = clipsat(seq, n)
    isempty(cs) && return nothing
    out = RGBFrame(undef, cs[end].source.width, cs[end].source.height)
    ok = VE.composite(engine, cs, n,
                      (clip, _) -> get!(() -> VE.opendecoder(clip.source, engine.backend),
                                        readers, clip.source.path); exact = true) do canvas
        copyto!(out, canvas)
    end
    return ok ? out : nothing
end

"The single-clip reference: the clip's graph at `n`, crop NOT baked (the preview
leaves that to the axis limits — see the framing check)."
function uncropped(clip::Clip, n::Integer, engine, readers)
    sf = clip.src_in + (n - clip.start)
    ec = effectiveclip(clip, sf)
    flat = Clip(ec.id, ec.source, ec.src_in, ec.src_out, ec.start, (0.0, 0.0, 1.0, 1.0),
                ec.effects, ec.colortrack, ec.motiontrack, ec.mattetrack, ec.animations,
                ec.track, ec.blendfrom)
    dec = get!(() -> VE.opendecoder(clip.source, engine.backend), readers, clip.source.path)
    out = RGBFrame(undef, clip.source.width, clip.source.height)
    VE.render(engine, dec, flat, sf; exact = true) do layer
        copyto!(out, layer)
    end
    return out
end

maxdiff(a, b) = maximum(max.(abs.(Float32.(getfield.(a, :r)) .- Float32.(getfield.(b, :r))),
                             abs.(Float32.(getfield.(a, :g)) .- Float32.(getfield.(b, :g))),
                             abs.(Float32.(getfield.(a, :b)) .- Float32.(getfield.(b, :b)))))

"Settle the preview on `n` the way a parked playhead does (exact frame, not a stand-in)."
function settleon!(player, n; seconds = 3.0)
    t0 = time()
    while time() - t0 < seconds
        VE.showframe!(player, n) && return true
        sleep(0.02)
    end
    return false
end

# ---------------------------------------------------------------- the actions

"Every edit the fuzzer can make, as (name, apply) — each returns a log line."
function fuzzactions(player, rng)
    seq = player.sequence
    randclip() = isempty(seq.clips) ? nothing : seq.clips[rand(rng, 1:length(seq.clips))]
    return [
        ("split", () -> begin
            n = rand(rng, 1:max(seqlength(seq) - 1, 1))
            VE.split!(seq, n) === nothing ? "split $n (no-op)" : "split $n"
        end),
        ("track", () -> begin
            c = randclip(); c === nothing && return "track (no clip)"
            c.track = rand(rng, 1:3)
            "track clip@$(c.start) → V$(c.track)"
        end),
        ("overlap", () -> begin
            length(seq.clips) < 2 && return "overlap (needs 2 clips)"
            i = rand(rng, 2:length(seq.clips))
            c, prev = seq.clips[i], seq.clips[i - 1]
            c.track = prev.track + 1                       # stack it…
            c.start = max(prev.start, clipend(prev) - max(cliplength(c) ÷ 2, 3))
            "overlap clip $i → V$(c.track) at $(c.start)"  # …ONTO the one before it
        end),
        ("blend", () -> begin
            length(seq.clips) < 2 && return "blend (needs 2 clips)"
            i = rand(rng, 1:(length(seq.clips) - 1))
            a, b = seq.clips[i], seq.clips[i + 1]
            fit = rand(rng) < 0.7                          # fit = the overlapping kind
            VE.blendclips!(player, a, b; seconds = 0.1 + 0.3 * rand(rng), fit = fit)
            "blend $(i)↔$(i + 1) fit=$fit"
        end),
        ("effect", () -> begin
            c = randclip(); c === nothing && return "effect (no clip)"
            e = rand(rng, (VE.ColorEffect(; brightness = 0.2f0), VE.BlurEffect(1.5f0),
                           VE.SharpenEffect(1.0f0, 0.6f0), VE.OpacityEffect(0.7f0)))
            push!(c.effects, FxSlot(e))
            "effect $(typeof(e).name.name) on clip@$(c.start)"
        end),
        ("keyframe", () -> begin
            c = randclip(); c === nothing && return "keyframe (no clip)"
            VE.findslot(c, VE.ColorEffect) === nothing && VE.seteffect!(c, VE.ColorEffect())
            curve = get!(VE.AnimCurve, c.animations, :brightness)
            VE.setkey!(curve, c.src_in, -0.3)
            VE.setkey!(curve, max(c.src_out - 1, c.src_in), 0.3)
            "keyframe brightness on clip@$(c.start)"
        end),
        ("crop", () -> begin
            c = randclip(); c === nothing && return "crop (no clip)"
            x = 0.05 + 0.2 * rand(rng); y = 0.05 + 0.2 * rand(rng)
            c.crop = (x, y, 1 - 2x, 1 - 2y)
            "crop clip@$(c.start) → $(round.(c.crop; digits = 2))"
        end),
        ("stabilize", () -> begin
            c = randclip(); c === nothing && return "stabilize (no clip)"
            fakestabilize!(c, rng)
            "stabilize clip@$(c.start) crop=$(round.(c.crop; digits = 2))"
        end),
        ("trim", () -> begin
            c = randclip(); c === nothing && return "trim (no clip)"
            cliplength(c) > 12 || return "trim (too short)"
            if rand(rng, Bool)
                c.src_in += 5; c.start += 5
            else
                c.src_out -= 5
            end
            "trim clip@$(c.start)"
        end),
        ("undo", () -> (VE.undo!(player); "undo")),
    ]
end

@testset "fuzz: random edits keep preview and export the same picture" begin
    engine = VE.FxEngine(VE.KA.CPU())
    readers = Dict{String, Any}()
    player = Player(testvideo; gpupreview = false)
    try
        sleep(1.0)
        seq = player.sequence
        rng = MersenneTwister(20260728)   # seeded: a failure replays exactly
        program = String[]
        checked = 0
        composites = 0
        croppedsingles = 0
        mismatches = String[]
        for step in 1:24
            actions = fuzzactions(player, rng)
            name, apply = actions[rand(rng, 1:length(actions))]
            push!(program, "[$step] " * try
                      apply()
                  catch e
                      "$name THREW $(sprint(showerror, e))"
                  end)
            VE.refreshedit!(player)
            sleep(0.05)
            isempty(seq.clips) && (VE.undo!(player); continue)
            for _ in 1:3
                n = rand(rng, 0:max(seqlength(seq) - 1, 0))
                cs = clipsat(seq, n)
                isempty(cs) && continue          # a gap shows black, nothing to compare
                settleon!(player, n) || continue # decoder not there yet — not a picture bug
                expect = exportframe(seq, n, engine, readers)
                expect === nothing && continue
                checked += 1
                length(cs) > 1 && (composites += 1)
                length(cs) == 1 && cs[1].crop != (0.0, 0.0, 1.0, 1.0) && (croppedsingles += 1)
                # PIXELS: the preview buffer IS what the export renders. A
                # single-clip present leaves the crop to the axis limits, so its
                # reference is the same graph WITHOUT the crop baked in.
                ref = length(cs) == 1 ? uncropped(cs[1], n, engine, readers) : expect
                if size(player.frame[]) == size(ref)
                    d = maxdiff(player.frame[], ref)
                    d > 1.0f-3 && push!(mismatches, "frame $n: pixels differ by $d")
                else
                    push!(mismatches, "frame $n: preview $(size(player.frame[])) vs export $(size(ref))")
                end
                # …and FRAMING: the visible region is the framing the export bakes
                want = length(cs) > 1 ? (0.0, 0.0, 1.0, 1.0) : cs[1].crop
                vis = visiblerect(player)
                all(abs.(vis .- want) .< 5.0e-3) ||
                    push!(mismatches, "frame $n: shows $(round.(vis; digits = 3)), " *
                                      "export bakes $(round.(want; digits = 3)) " *
                                      "($(length(cs)) layer(s))")
            end
        end
        isempty(mismatches) || @info "fuzz program" program mismatches
        @test isempty(mismatches)
        @test checked >= 20            # the run actually looked at pictures…
        # …at BOTH kinds. Without this the fuzzer silently degrades: the first
        # version stacked no clips in time and passed 72 checks without ever
        # rendering a composite — green, and blind to the bug it was written for.
        @info "fuzz coverage" checked composites croppedsingles
        @test composites >= 5
        @test croppedsingles >= 5
    finally
        close(player)
        foreach(close, values(readers))
        VE.emptyengine!(engine)
    end
end
