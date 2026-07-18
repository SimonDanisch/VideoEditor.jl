"""
    clipview!(ax, timerange; viewrange, pixelspersecond, sourcestart, thumbs, state, …)

Recipe drawing one clip on the timeline: a band with a state-dependent
border and a single composed thumbnail strip. All tile logic lives here and
is reused for every clip — the timeline just creates one `ClipView` per
clip and feeds shared view observables.

Tiles are anchored at the clip start and clamped to its extent (they never
cross a cut). The visible tiles are composed into **one** image (film-strip
style, partial tail tile sliced), so each clip costs two draw calls
regardless of zoom.

`state` is `:idle`, `:hovered` or `:selected`; `thumbs` is a function
`second::Int -> Union{Nothing, Matrix}` (missing seconds render as
placeholder and are expected to be requested by the provider itself).
Bump `refresh` to re-pull thumbnails after asynchronous loads.
"""
@recipe ClipView (timerange,) begin
    "Visible x-range of the parent axis in seconds."
    viewrange = (0.0, 1.0e9)
    "On-screen pixels per second (tile density)."
    pixelspersecond = 100.0
    "On-screen pixel height of the thumbnail band (for aspect-correct tiles)."
    bandheight = 72.0
    "Source time in seconds at the clip's first frame (thumbnail keying)."
    sourcestart = 0.0
    "Interaction state: :idle, :hovered or :selected."
    state = :idle
    "Thumbnail provider `second -> Union{Nothing, Matrix{RGB{N0f8}}}`."
    thumbs = nothing
    "Bump to re-pull thumbnails (async loads)."
    refresh = 0
    "Band fill color."
    color = RGBAf(0.18, 0.19, 0.21, 1)
    "Border color when idle."
    strokecolor_idle = RGBAf(0.35, 0.36, 0.38, 1)
    "Border color when hovered."
    strokecolor_hovered = RGBAf(0.55, 0.4, 0.28, 1)
    "Border color when selected."
    strokecolor_selected = RGBAf(1.0, 0.47, 0.22, 1)
    "Thumbnail width/height in pixels (composition resolution)."
    thumbsize = (49, 88)
end

function Makie.plot!(p::ClipView)
    map!(p, [:timerange], :bandrect) do (t0, t1)
        return Rect2f(t0, 0.03, t1 - t0, 0.94)
    end
    map!(p, [:state, :strokecolor_idle, :strokecolor_hovered, :strokecolor_selected],
         :bandstroke) do state, idle, hovered, selected
        return state === :selected ? selected : state === :hovered ? hovered : idle
    end
    map!(p, [:timerange, :viewrange, :pixelspersecond, :bandheight, :sourcestart,
             :thumbs, :thumbsize, :color, :refresh],
         [:strip, :stripx, :stripvisible]) do trange, vrange, pps, bh, s0, thumbs, tsize, color, _
        return composetiles(trange, vrange, pps, bh, s0, thumbs, tsize, to_color(color))
    end

    poly!(p, p.bandrect; color = p.color, strokecolor = p.bandstroke, strokewidth = 2)
    image!(p, p.stripx, (0.93, 0.07), p.strip; interpolate = true,
           visible = p.stripvisible)
    return p
end

"Compose the clip's visible tiles into one image (see [`ClipView`](@ref))."
function composetiles(trange, vrange, pps, bandheight, s0, thumbs, (tw, th), fallback)
    fill3 = RGB{N0f8}(red(fallback), green(fallback), blue(fallback))
    placeholder() = fill(fill3, tw, th)
    t0, t1 = trange
    lo, hi = max(t0, vrange[1]), min(t1, vrange[2])
    (lo < hi && pps > 0) || return (placeholder(), (t0, t0 + 1.0e-6), false)

    spacing = 2.0^clamp(ceil(Int, log2(tw / pps)), -6, 12)
    # the whole strip is stretched uniformly to its on-screen rectangle, so the
    # tile CANVAS must carry the slot's screen aspect for the thumbnail to keep
    # its own. A full tile's slot is (spacing*pps) px wide × bandheight px tall.
    slotaspect = spacing * pps / max(bandheight, 1.0)
    thumbaspect = tw / th
    Wc, Hc = slotaspect >= thumbaspect ?
             (max(round(Int, th * slotaspect), tw), th) :        # pillarbox
             (tw, max(round(Int, tw / slotaspect), th))          # letterbox
    firsttile = max(floor(Int, (lo - t0) / spacing), 0)
    lasttile = floor(Int, (hi - t0 - 1.0e-9) / spacing)
    parts = Matrix{RGB{N0f8}}[]
    for k in firsttile:lasttile
        second = floor(Int, s0 + k * spacing)
        img = something(thumbs === nothing ? nothing : thumbs(second), placeholder())
        w = Wc
        tilend = t0 + (k + 1) * spacing
        if tilend > t1  # partial tail tile at a cut: slice, don't squeeze
            frac = (t1 - (t0 + k * spacing)) / spacing
            img = img[1:max(round(Int, size(img, 1) * frac), 1), :]
            w = max(round(Int, Wc * frac), 1)
        end
        push!(parts, fittile(img, w, Hc, fill3))
    end
    isempty(parts) && return (placeholder(), (t0, t0 + 1.0e-6), false)
    strip = vcat(parts...)  # (w, h) layout: horizontal concat is along dim 1
    xstart = t0 + firsttile * spacing
    xend = min(t0 + (lasttile + 1) * spacing, t1)
    return (strip, (xstart, xend), true)
end

"Center `img` in a `Wc × Hc` canvas, padding with `fill3` (aspect-preserving)."
function fittile(img::AbstractMatrix{RGB{N0f8}}, Wc::Integer, Hc::Integer, fill3)
    iw, ih = size(img)
    (iw == Wc && ih == Hc) && return collect(img)
    canvas = fill(fill3, Wc, Hc)
    ox = (Wc - iw) ÷ 2
    oy = (Hc - ih) ÷ 2
    for j in 1:ih
        cj = oy + j
        (1 <= cj <= Hc) || continue
        for i in 1:iw
            ci = ox + i
            (1 <= ci <= Wc) && (canvas[ci, cj] = img[i, j])
        end
    end
    return canvas
end
