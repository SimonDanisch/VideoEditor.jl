"""
    clipview!(ax, timerange; viewrange, pixelspersecond, sourcestart, thumbs, state, …)

Recipe drawing one clip on the timeline: a band with a state-dependent
border and a single composed thumbnail strip. All tile logic lives here and
is reused for every clip — the timeline just creates one `ClipView` per
clip and feeds shared view observables.

Tiles are anchored at the clip start and clamped to its extent (they never
cross a cut). The visible tiles are composed into one image (film-strip
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
    "Bottom edge of this clip's band in axis y (0..1) — its track's row."
    bandlo = 0.03
    "Top edge of this clip's band in axis y (0..1)."
    bandhi = 0.97
    "This lane's share of the whole stack's height. The tile pitch follows the lane a
    clip is actually drawn in, so a track made taller shows BIGGER frames rather than
    the same ones stretched — which is the entire point of making it taller."
    bandshare = 1.0
end

function Makie.plot!(p::ClipView)
    map!(p, [:timerange, :bandlo, :bandhi], :bandrect) do (t0, t1), lo, hi
        return Rect2f(t0, lo, t1 - t0, hi - lo)
    end
    map!(p, [:state, :strokecolor_idle, :strokecolor_hovered, :strokecolor_selected],
         :bandstroke) do state, idle, hovered, selected
        return state === :selected ? selected : state === :hovered ? hovered : idle
    end
    map!(p, [:timerange, :viewrange, :pixelspersecond, :bandheight, :sourcestart,
             :thumbs, :thumbsize, :color, :refresh, :bandshare],
         [:strip, :stripx, :stripvisible]) do trange, vrange, pps, bh, s0, thumbs, tsize, color, _, share
        return composetiles(trange, vrange, pps, bh * share, s0, thumbs, tsize, to_color(color))
    end
    # thumbnails inset inside the band (same 0.04/0.94 proportion as the full-height band)
    map!(p, [:bandlo, :bandhi], :stripy) do lo, hi
        inset = 0.04 / 0.94 * (hi - lo)
        return (hi - inset, lo + inset)
    end

    poly!(p, p.bandrect; color = p.color, strokecolor = p.bandstroke, strokewidth = 2)
    image!(p, p.stripx, p.stripy, p.strip; interpolate = true, visible = p.stripvisible)
    return p
end

"Compose the clip's visible tiles into one image (see [`ClipView`](@ref))."
function composetiles(trange, vrange, pps, bandheight, s0, thumbs, (tw, th), fallback)
    fill3 = RGB{N0f8}(red(fallback), green(fallback), blue(fallback))
    placeholder() = fill(fill3, tw, th)
    t0, t1 = trange
    lo, hi = max(t0, vrange[1]), min(t1, vrange[2])
    (lo < hi && pps > 0) || return (placeholder(), (t0, t0 + 1.0e-6), false)

    # a gapless, UNDISTORTED filmstrip: every tile keeps the frame's aspect at
    # the lane height, so the tile pitch (in seconds) is exactly the width an
    # aspect-correct lane-height tile covers on screen — zooming in repeats
    # frames, zooming out skips them, tiles always butt against each other
    pitch = max(bandheight, 8.0) * (tw / th) / pps
    # …and the grid is anchored to the MEDIA, not to the clip: `origin` is where
    # this source's time 0 would sit on the timeline. Trimming the head moves the
    # clip start and `s0` by the SAME amount, so the grid does not move and the
    # picture stays exactly where it was — you just see less of it. Anchored at the
    # clip start instead, every head trim re-sliced the whole strip, which is what
    # made trimming the left edge look like the far end was being cut.
    origin = t0 - s0
    firsttile = floor(Int, (lo - origin) / pitch)
    lasttile = floor(Int, (hi - origin - 1.0e-9) / pitch)
    lasttile = min(lasttile, firsttile + 600)   # runaway guard at absurd zoom
    parts = Matrix{RGB{N0f8}}[]
    xstart = origin + firsttile * pitch         # exact edges of what we KEEP, so the
    xend = origin + (lasttile + 1) * pitch      # image maps back onto the same pixels
    for k in firsttile:lasttile
        second = floor(Int, k * pitch)          # source second under this tile
        img = something(thumbs === nothing ? nothing : thumbs(second), placeholder())
        tilestart = origin + k * pitch
        tileend = tilestart + pitch
        if tilestart < t0 || tileend > t1       # partial tile at either cut: slice
            w = size(img, 1)
            a = clamp((max(tilestart, t0) - tilestart) / pitch, 0.0, 1.0)
            b = clamp((min(tileend, t1) - tilestart) / pitch, 0.0, 1.0)
            # round the cuts INWARD so the strip never bleeds past the band's edge
            i0 = clamp(ceil(Int, w * a) + 1, 1, w)
            i1 = clamp(floor(Int, w * b), 1, w)
            i0 > i1 && (i0 = i1)
            img = img[i0:i1, :]
            # the strip's outer edges follow the ROUNDED cut, not the ideal one —
            # otherwise the image is stretched by a fraction of a tile and the whole
            # filmstrip creeps sideways as a trim rounds differently
            k == firsttile && (xstart = tilestart + (i0 - 1) / w * pitch)
            k == lasttile && (xend = tilestart + i1 / w * pitch)
        end
        push!(parts, collect(img))
    end
    isempty(parts) && return (placeholder(), (t0, t0 + 1.0e-6), false)
    strip = vcat(parts...)  # (w, h) layout: horizontal concat is along dim 1
    return (strip, (xstart, xend), true)
end

"Fill a `Wc × Hc` canvas with `img` scaled-to-cover and center-cropped
(aspect-preserving, no padding) — filmstrip tiles butt against each other."
function covertile(img::AbstractMatrix{RGB{N0f8}}, Wc::Integer, Hc::Integer)
    iw, ih = size(img)
    (iw == Wc && ih == Hc) && return collect(img)
    s = max(Wc / iw, Hc / ih)
    x0 = (iw - Wc / s) / 2
    y0 = (ih - Hc / s) / 2
    canvas = Matrix{RGB{N0f8}}(undef, Wc, Hc)
    for j in 1:Hc
        sj = clamp(ceil(Int, y0 + j / s), 1, ih)
        for i in 1:Wc
            si = clamp(ceil(Int, x0 + i / s), 1, iw)
            canvas[i, j] = img[si, sj]
        end
    end
    return canvas
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
