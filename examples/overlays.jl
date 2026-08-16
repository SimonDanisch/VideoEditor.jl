# ── Plotting into the video ───────────────────────────────────────────────────
#
# Makie plots composited over the finished canvas: titles, lower thirds, and a
# live data chart whose cursor tracks the playhead. Every number here is
# keyframable, and the chart's data is MEASURED FROM THE FOOTAGE — the spikes
# line up with the birds moving because they are the birds moving.
#
#   julia> include("examples/overlays.jl")
#
# What it shows, in order of interest:
#   • a custom overlay kind registered through the public API (`:chart`)
#   • an overlay that reads `frame`/`framerate` off its state, so it follows the
#     video with nothing keyframed at all (the cursor, the timecode)
#   • keyframed parameters with per-key easing (the title card, the panel wipe)
#   • MORPHING one data series into another (`morph`), the plot equivalent of a
#     cross-dissolve

import VideoEditor as VE
using VideoEditor: Overlay, addoverlay!, setoverlaykey!, registeroverlay!, FxParam
using VideoEditor.Makie
using VideoEditor.Makie: Point2f, Rect2f, RGBAf, lift

const FOOTAGE = joinpath(@__DIR__, "..", "..", "..", "media", "demo_source.mp4")
const SECONDS = 8
const ACCENT = "#ffb020"

# ── a custom overlay kind ─────────────────────────────────────────────────────
#
# `draw` builds its plots ONCE and lifts everything else off `state`, which
# carries the overlay's parameters (sampled at this frame), its settings, and the
# reserved `frame`/`framerate`. Registering it is all that's needed: it is now
# placeable, keyframable and saved with the project like any built-in.

registeroverlay!(:chart, "Chart",
    [FxParam(:x, "X"; min = -0.5, max = 1.5, default = 0.06),
     FxParam(:y, "Y"; min = -0.5, max = 1.5, default = 0.60),
     FxParam(:width, "Width"; min = 0.0, max = 1.5, default = 0.88),
     FxParam(:height, "Height"; min = 0.0, max = 1.0, default = 0.24),
     FxParam(:morph, "Morph"; min = 0.0, max = 1.0, default = 0.0),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    function (scene, canvas, state)
        W, H = canvas
        fade(c, a, s) = (col = Makie.to_color(c); RGBAf(col.r, col.g, col.b, col.alpha * a * s.opacity))
        panel = VE.olift(s -> Rect2f(s.x * W, s.y * H, s.width * W, s.height * H), state)
        # the card is a header band (label + readout) over a plot band, so the
        # curve and the numbers can never collide however the data moves
        inner = lift(panel) do r
            p = 0.09 * r.widths[2]
            Rect2f(r.origin[1] + p, r.origin[2] + p, r.widths[1] - 2p, r.widths[2] - 2p)
        end
        plotarea = lift(r -> Rect2f(r.origin, (r.widths[1], 0.66 * r.widths[2])), inner)

        # the series, morphed, and how far along the video we are
        vals = VE.olift(s -> VE.morphseries(get(s, :values, Float64[]),
                                            get(s, :values2, Float64[]), s.morph), state)
        # `frame` is why nothing here needs keyframing: the chart reads the
        # playhead and reveals itself in step with the picture
        at = VE.olift(state) do s
            from, to = Float64(get(s, :from, 0)), Float64(get(s, :to, 1))
            return to > from ? clamp((s.frame - from) / (to - from), 0.0, 1.0) : 1.0
        end

        # the polyline, revealed up to the playhead
        pts = lift(vals, at, plotarea) do v, u, r
            length(v) < 2 && return Point2f[]
            lo, hi = extrema(v); span = hi - lo
            shown = round(Int, u * length(v))
            shown < 2 && return Point2f[]
            return [Point2f(r.origin[1] + r.widths[1] * (i - 1) / (length(v) - 1),
                            r.origin[2] + r.widths[2] * (span == 0 ? 0.5 : (v[i] - lo) / span))
                    for i in 1:shown]
        end

        # panel, then grid, then the filled area, then the line: back to front
        Makie.poly!(scene, panel; color = VE.olift(s -> fade(RGBAf(0.03, 0.04, 0.06, 0.78), 1, s), state),
                    strokewidth = 1, strokecolor = VE.olift(s -> fade(RGBAf(1, 1, 1, 0.16), 1, s), state))
        for f in (0.0, 0.5, 1.0)
            Makie.lines!(scene, lift(plotarea) do r
                             y = r.origin[2] + f * r.widths[2]
                             [Point2f(r.origin[1], y), Point2f(r.origin[1] + r.widths[1], y)]
                         end; linewidth = 1,
                         color = VE.olift(s -> fade(RGBAf(1, 1, 1, 0.12), 1, s), state))
        end
        Makie.poly!(scene, lift(pts, plotarea) do ps, r
                        length(ps) < 2 && return Point2f[]
                        b = r.origin[2]
                        return vcat(ps, [Point2f(ps[end][1], b), Point2f(ps[1][1], b)])
                    end; color = VE.olift(s -> fade(ACCENT, 0.20, s), state))
        Makie.lines!(scene, pts; linewidth = VE.olift(s -> Float32(0.014 * s.height * H), state),
                     color = VE.olift(s -> fade(ACCENT, 1, s), state))

        # the cursor rides the last revealed point
        Makie.scatter!(scene, lift(ps -> isempty(ps) ? Point2f[] : [ps[end]], pts);
                       markersize = VE.olift(s -> Float32(0.09 * s.height * H), state),
                       color = VE.olift(s -> fade(:white, 1, s), state),
                       strokewidth = VE.olift(s -> Float32(0.018 * s.height * H), state),
                       strokecolor = VE.olift(s -> fade(ACCENT, 1, s), state))

        # header: the two labels CROSSFADE with the morph, so the card always
        # says which series it is currently showing
        for (key, alpha) in ((:label, s -> 1 - s.morph), (:label2, s -> s.morph))
            Makie.text!(scene, lift(r -> Point2f(r.origin[1], r.origin[2] + r.widths[2]), inner);
                        text = VE.olift(s -> uppercase(String(get(s, key, ""))), state),
                        fontsize = VE.olift(s -> Float32(0.13 * s.height * H), state), font = :bold,
                        color = VE.olift(s -> fade(RGBAf(1, 1, 1, 0.8), alpha(s), s), state),
                        align = (:left, :top))
        end
        Makie.text!(scene, lift(r -> Point2f(r.origin[1] + r.widths[1], r.origin[2] + r.widths[2]), inner);
                    text = lift(vals, at) do v, u
                        isempty(v) && return ""
                        i = clamp(round(Int, u * length(v)), 1, length(v))
                        lo, hi = extrema(v)
                        return string(round(Int, 100 * (hi == lo ? 0.5 : (v[i] - lo) / (hi - lo))))
                    end,
                    fontsize = VE.olift(s -> Float32(0.22 * s.height * H), state), font = :bold,
                    color = VE.olift(s -> fade(ACCENT, 1, s), state), align = (:right, :top))
        return nothing
    end)

# ── measure the footage ───────────────────────────────────────────────────────
#
# Motion energy (mean absolute frame difference) and brightness, per frame, at a
# small working resolution. This is the data the chart plots.

function measure(source::VE.VideoSource, nframes::Integer)
    reader = VE.SequentialReader(source)
    frame = zeros(VE.RGB{VE.N0f8}, source.width, source.height)
    prev = nothing
    motion, bright = Float64[], Float64[]
    step = max(1, (source.width * source.height) ÷ 40_000)   # subsample; we want a shape, not a metric
    for n in 0:(nframes - 1)
        # `sourceinto!` takes the DECODED frame, not a frame index — decode is a
        # separate step since `1f5332e` moved it out of the source pass, so that
        # `served` is settled before the graph runs. Passing `n` here copied the
        # integer into an RGB buffer ("(2, 2, 2) are integers in the range 0-255").
        VE.sourceinto!(frame, reader, VE.decodesource(reader, n))
        px = @view frame[1:step:end]
        y = [0.299 * Float64(p.r) + 0.587 * Float64(p.g) + 0.114 * Float64(p.b) for p in px]
        push!(bright, sum(y) / length(y))
        push!(motion, prev === nothing ? 0.0 : sum(abs, y .- prev) / length(y))
        prev = y
    end
    close(reader)
    smooth(v, w) = [sum(@view v[max(1, i - w):min(length(v), i + w)]) /
                    length(max(1, i - w):min(length(v), i + w)) for i in eachindex(v)]
    return smooth(motion, 3), smooth(bright, 3)
end

isfile(normpath(FOOTAGE)) ||
    error("no footage at $(normpath(FOOTAGE)) — point FOOTAGE at any clip you have")
source = VE.VideoSource(normpath(FOOTAGE))
fps = source.framerate
LEN = min(round(Int, SECONDS * fps), source.nframes)
@info "measuring $(LEN) frames of $(basename(source.path))…"
motion, bright = measure(source, LEN)

# ── the sequence ──────────────────────────────────────────────────────────────

seq = VE.Sequence([VE.Clip(source, 0, LEN, 0, (0.0, 0.0, 1.0, 1.0))], fps)
sec(t) = round(Int, t * fps)

# 1 · title card: a name that fades up and away, over a full-frame scrim
scrim = addoverlay!(seq, :bar; x = 0.0, y = 0.0, width = 1.0, height = 1.0,
                    color = "black", start = 0, stop = sec(2.6))
setoverlaykey!(scrim, :opacity, 0, 0.75); setoverlaykey!(scrim, :opacity, sec(2.4), 0.0)

title = addoverlay!(seq, :text; text = "HOUSE SPARROWS", color = "white",
                    size = 0.038, x = 0.5, start = 0, stop = sec(2.6))
setoverlaykey!(title, :y, sec(0.2), 0.53); setoverlaykey!(title, :y, sec(2.4), 0.57)
setoverlaykey!(title, :opacity, sec(0.2), 0.0); setoverlaykey!(title, :opacity, sec(0.9), 1.0)
setoverlaykey!(title, :opacity, sec(1.9), 1.0); setoverlaykey!(title, :opacity, sec(2.4), 0.0)

sub = addoverlay!(seq, :text; text = "nest box · 60 fps", color = ACCENT,
                  size = 0.024, x = 0.5, start = 0, stop = sec(2.6))
setoverlaykey!(sub, :y, sec(0.4), 0.47); setoverlaykey!(sub, :y, sec(2.4), 0.50)
setoverlaykey!(sub, :opacity, sec(0.5), 0.0); setoverlaykey!(sub, :opacity, sec(1.2), 1.0)
setoverlaykey!(sub, :opacity, sec(1.9), 1.0); setoverlaykey!(sub, :opacity, sec(2.4), 0.0)

# 2 · the chart: wipes open, then draws itself in step with the playhead
chart = addoverlay!(seq, :chart; values = motion, values2 = bright,
                    label = "motion", label2 = "brightness", from = sec(2.2), to = LEN,
                    x = 0.06, y = 0.60, width = 0.88, height = 0.0,
                    start = sec(2.0), stop = LEN)
setoverlaykey!(chart, :height, sec(2.0), 0.0, :smooth)     # per-key easing
setoverlaykey!(chart, :height, sec(2.8), 0.24, :smooth)
setoverlaykey!(chart, :opacity, sec(2.0), 0.0)
setoverlaykey!(chart, :opacity, sec(2.6), 1.0)
# …and late on, the series MORPHS into the other one
setoverlaykey!(chart, :morph, sec(6.0), 0.0, :smooth)
setoverlaykey!(chart, :morph, sec(7.2), 1.0, :smooth)

# (the card's own two labels crossfade with `morph` — nothing to key here)

# 3 · a lower third and a timecode that need no keyframes to follow the video
addoverlay!(seq, :bar; x = 0.0, y = 0.0, width = 1.0, height = 0.10,
            color = "black", opacity = 0.55, start = sec(2.4), stop = LEN)
addoverlay!(seq, :timecode; x = 0.5, y = 0.045, size = 0.030,
            color = "white", start = sec(2.4), stop = LEN)

# ── render ────────────────────────────────────────────────────────────────────

out = joinpath(tempdir(), "overlay_example.mp4")
@info "rendering $(LEN) frames → $out"
VE.exportvideo(out, seq; audio = false, encoder_options = (crf = 18, preset = "medium"),
               progress = (d, t) -> d % 120 == 0 && @info "  $d/$t")
@info "done" out filesize(out)
out
