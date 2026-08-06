# A complete third-party GUI tool: "Close gaps" — highlights every gap between
# clips on the thumb track and ripple-closes the one you click. Include this
# file any time (before or after a Player exists — registration is live, the
# Tools dock picks it up immediately), or paste it over MCP.
#
# The recipe every tool follows:
#   registertool!(name, label, description; activate = ctx -> …)
#   • draw hints:   toolplot!(ctx, plot)  — timeline axis (x = seconds,
#                   y = 0..1 track span), or parent = player.previewaxis
#   • wire events:  ontool!(ctx, observable) do … end   (auto-removed)
#   • act:          the same editing API the editor uses (snapshot! for undo,
#                   then mutate clips / addtransition! / effects, refreshedit!)

using VideoEditor
import VideoEditor as VE
import VideoEditor.Makie as Makie
using VideoEditor.Makie: Observable, Rect2f, Mouse, Consume, events, mouseposition

registertool!(:gapcloser, "Close gaps",
    "Highlights every gap between clips on a track — click a red band to " *
    "ripple-close it (later clips slide left).";
    activate = function (ctx)
        player = ctx.player
        seq = player.sequence
        fps = seq.framerate
        ax = player.timeline.axis
        rects = Observable(Rect2f[])
        gaps = Ref(NTuple{3, Int}[])   # (gapstart, gapend, track) in timeline frames
        function refresh()
            gs = NTuple{3, Int}[]
            for tr in 1:VE.ntracks(seq)
                cs = sort([c for c in seq.clips if c.track == tr]; by = c -> c.start)
                for k in 1:(length(cs) - 1)
                    g0, g1 = VE.clipend(cs[k]), cs[k + 1].start
                    g1 > g0 && push!(gs, (g0, g1, tr))
                end
            end
            gaps[] = gs
            rects[] = [begin
                           lo, hi = VE.trackband(tr, VE.ntracks(seq))
                           Rect2f(g0 / fps, lo, (g1 - g0) / fps, hi - lo)
                       end for (g0, g1, tr) in gs]
            return
        end
        plt = Makie.poly!(ax, rects; color = (:red, 0.25), strokecolor = :red, strokewidth = 1)
        Makie.translate!(plt, 0, 0, 6)          # above the thumbs, below the playhead
        toolplot!(ctx, plt)
        refresh()
        VE.setstatus!(player, isempty(gaps[]) ? "Close gaps: the timeline has no gaps" :
                      "Close gaps: click a red band to ripple-close it")
        ontool!(ctx, events(ax.scene).mousebutton) do event
            (event.button == Mouse.left && event.action == Mouse.press &&
             Makie.is_mouseinside(ax.scene)) || return Consume(false)
            t, _ = mouseposition(ax.scene)
            n = VE.timelineframe(player.timeline, t)
            k = findfirst(g -> g[1] <= n < g[2], gaps[])
            k === nothing && return Consume(false)      # elsewhere: scrub as usual
            g0, g1, tr = gaps[][k]
            VE.snapshot!(player)
            for c in seq.clips                          # ripple: slide later clips left
                c.track == tr && c.start >= g1 && (c.start -= g1 - g0)
            end
            VE.prunetransitions!(seq)
            VE.setstatus!(player, "gap closed ($(round((g1 - g0) / fps, digits = 2))s) — Ctrl+Z undoes")
            VE.refreshedit!(player)
            refresh()
            return Consume(true)
        end
        ontool!(_ -> refresh(), ctx, player.playhead)   # edits notify the playhead
    end)
