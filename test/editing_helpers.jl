# Use the very same actions as the recorded Makie walkthroughs. Set
# MAKIE_FAKE_INTERACTION when the Makie docs live outside the sibling checkout.
# VIDEOEDITOR_QA_OUTPUT keeps the recordings/screenshots for visual review.
isdefined(Main, :FakeInteraction) || include(
    get(
        ENV, "MAKIE_FAKE_INTERACTION",
        normpath(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
    )
)

module EditingWalkthroughActions
    using Test
    import VideoEditor as VE
    const M = VE.Makie
    const FI = Main.FakeInteraction
    const K = M.Keyboard

    export center, header, button, speechblocks, speechfield, speechmenu, menurow, timelinepos, click, chord, reveal, typefield, recordactions

    "Compare displayed scene pixels with the rendered image, including orientation."
    function previewmatches(p, axis, picture)
        screen = M.getscreen(p.fig.scene)
        screenshot = M.colorbuffer(screen)
        height, width = size(screenshot)
        dimensions = p.fig.scene.viewport[].widths
        scale = M.Vec2f(width / dimensions[1], height / dimensions[2])
        w, h = size(picture)
        errors = Float64[]
        for x in round.(Int, range(1, w; length = 17)[2:(end - 1)]),
                y in round.(Int, range(1, h; length = 17)[2:(end - 1)])
            expected = M.RGBf(picture[x, y])
            maximum((expected.r, expected.g, expected.b)) > 0.08 || continue
            q = M.project(axis.scene, M.Point3f(x - 0.5, y - 0.5, 0))
            position = (M.Point2f(q[1], q[2]) + M.Point2f(axis.scene.viewport[].origin)) .* scale
            col = clamp(round(Int, position[1]), 1, width)
            row = clamp(height - round(Int, position[2]), 1, height)
            actual = M.RGBf(screenshot[row, col])
            push!(errors, maximum(abs.((actual.r - expected.r, actual.g - expected.g, actual.b - expected.b))))
        end
        return length(errors) > 3 && count(<(0.15), errors) / length(errors) > 0.85
    end
    export previewmatches

    center(b) = FI.relative_pos(b, (0.5, 0.5))
    center(p::M.Point2f) = p
    header(b) = FI.relative_pos(b, (0.5, 1.0)) - M.Point2f(0, 12)
    function button(p, label)
        blocks = Any[]
        function visit(layout)
            for b in M.flatten_layout_content(layout)
                push!(blocks, b)
                b isa M.Block && isdefined(b, :layout) && b.layout !== nothing && visit(b)
            end
            return
        end
        visit(p.fig.layout)
        return only(b for b in blocks if b isa M.Button && b.label[] == label)
    end
    function speechblocks(p)
        return last(filter(c -> c.tool === :narration, p.fxwidgets[:toolcards][3])).blocks
    end
    function speechfield(p, label)
        blocks = speechblocks(p)
        i = findfirst(b -> b isa M.Label && b.text[] == label, blocks)
        i === nothing && error("speech field not found: $label")
        return blocks[i + 1]
    end
    speechmenu(p, i) = filter(b -> b isa M.Menu, speechblocks(p))[i]
    function menurow(menu, value)
        i = findfirst(o -> last(o) == value, M.to_value(menu.options))
        i === nothing && error("menu option not found: $value")
        sc = menu.blockscene.children[end]
        rect = sc.plots[1][1][][i]
        tr = M.translation(sc)[]
        return M.Point2f(sum(extrema(rect)) ./ 2 .+ M.Point2f(tr[1], tr[2]))
    end
    function timelinepos(p, frame; scrub = true)
        ax = p.timeline.axis; lim = ax.finallimits[]; vp = ax.scene.viewport[]
        y = scrub ? sum(VE.SCRUBBAND) / 2 : sum(VE.trackband(1, VE.ntracks(p.sequence))) / 2
        return M.Point2f(
            vp.origin[1] + (frame / p.sequence.framerate - lim.origin[1]) / lim.widths[1] * vp.widths[1],
            vp.origin[2] + (y - lim.origin[2]) / lim.widths[2] * vp.widths[2]
        )
    end
    click(f) = [FI.Lazy(_ -> FI.MouseTo(f(), 0.05)), FI.LeftClick(), FI.Wait(0.25)]
    chord(key) = [FI.KeyDown(K.left_control), FI.KeyPress(key), FI.KeyUp(K.left_control), FI.Wait(0.25)]
    function reveal(p, f)
        actions = Any[FI.Lazy(_ -> FI.MouseTo(center(p.fxpanel.scroll), 0.05))]
        # Wheel delivery and layout settle over several recorded frames. Re-check
        # the target after each gesture instead of assuming one wheel event got it
        # into view. Long, filtered performance sections exercise both directions.
        for _ in 1:4
            push!(
                actions, FI.Lazy(
                    _ -> begin
                        bb = p.fxpanel.scroll.layoutobservables.computedbbox[]
                        y = center(f())[2]
                        step = p.fxpanel.scroll.scroll_speed[]
                        dy = y < bb.origin[2] + 60 ? -(bb.origin[2] + 60 - y) / step :
                            y > bb.origin[2] + bb.widths[2] - 60 ? (y - bb.origin[2] - bb.widths[2] + 60) / step : 0
                        FI.Scroll((0, dy); duration = 0.5)
                    end
                ), FI.Wait(0.25)
            )
        end
        return actions
    end
    function typefield(p, f, text; guard = false)
        shortcuts = guard ? [
                FI.KeyPress(K.c), FI.KeyPress(K.s), FI.KeyPress(K.t),
                FI.Lazy(
                    _ -> begin
                        @test !p.cropmode[] && length(p.sequence.clips) == 2 && isempty(p.sequence.transitions)
                        FI.Wait(0)
                    end
                )
            ] : []
        return [
            reveal(p, f); click(() -> center(f()));
            FI.Lazy(
                _ -> begin
                    @test f().focused[]; FI.Wait(0)
                end
            ); chord(K.a);
            shortcuts;FI.TypeText(text; char_duration = 0.2); FI.KeyPress(K.enter); FI.Wait(0.5)
        ]
    end
    function recordactions(p, out, name, actions)
        FI.interaction_record(
            (_, _) -> sleep(0.01), p.fig,
            joinpath(out, name * ".mp4"), actions; fps = 12, px_per_unit = 1,
            visible = false, pause_renderloop = true, backend = VE.GLMakie
        )
        # Applying recording settings to an existing GL screen can restart its
        # native loop. FakeInteraction owns frame delivery during these tests.
        screen = M.getscreen(p.fig.scene, VE.GLMakie)
        VE.GLMakie.stop_renderloop!(screen; close_after_renderloop = false)
        return M.save(joinpath(out, name * ".png"), M.colorbuffer(screen))
    end

end
