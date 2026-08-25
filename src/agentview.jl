# Seeing an edit as an AGENT — images that carry the most information per token.
#
# An LLM understands video through pictures, but a picture per frame drowns its
# context. So this is a ZOOM, not a dump: three calls, each one image plus a
# manifest that says exactly what is in it, and each answer tells you what to
# call next.
#
#   1. [`contactsheet`] — one square image, N frames of a time range in a grid,
#      each cell numbered. 64 cells at 64 px is a 512×512 PNG (~350 vision
#      tokens) and shows a whole 10-minute edit at 10 s granularity.
#   2. [`contactsheet`] again on the interval a cell points at — every level
#      divides the time span by N (64 cells: 3 levels take 10 minutes down to
#      ~15 ms). That IS the binary search, only 64-ary and driven by what the
#      agent sees. [`findchange`] does the same automatically for a measurable
#      property (a cut, a flash) in O(log n) decodes, without the model in the
#      loop.
#   3. [`framegrab`] / [`regiongrab`] — one frame, or one normalized rectangle
#      of it at full detail, when the answer is spatial rather than temporal.
#
# Every image comes from [`renderframe!`](@ref) — the same function the encoder
# writes — so what the agent sees IS what the export produces, including
# effects, keyframes, stabilization, crop, transitions and track composites.

"""
    sheetgrid(n, aspect, size) -> (cols, rows, cellw, cellh)

Lay `n` cells of the source's `aspect` (w/h) into a sheet of about `size` × `size`
pixels. The cells keep the video's shape — square cells letterbox portrait
footage into 44% black, which is 44% of the tokens spent on nothing.
"""
function sheetgrid(n::Integer, aspect::Real, size::Integer)
    n <= 0 && return (0, 0, 0, 0)
    cols = max(round(Int, sqrt(n / max(aspect, 1.0e-3))), 1)
    rows = max(ceil(Int, n / cols), 1)
    cellh = max(fld(size, rows), 16)
    cellw = max(round(Int, cellh * aspect), 16)
    return (cols, rows, cellw, cellh)
end

# A 3×5 bitmap digit font: at 50–64 px cells, antialiased text turns to mush,
# and the cell INDEX is what an agent quotes back ("zoom into cell 27").
const SHEETDIGITS = (
    0b111_101_101_101_111, 0b010_110_010_010_111, 0b111_001_111_100_111,
    0b111_001_111_001_111, 0b101_101_111_001_001, 0b111_100_111_001_111,
    0b111_100_111_101_111, 0b111_001_001_001_001, 0b111_101_111_101_111,
    0b111_101_111_001_111,
)

"Draw `text` (digits only) into `img` at (x, y), `scale`×, light on a dark box."
function drawdigits!(img::AbstractMatrix{RGB{N0f8}}, text::AbstractString,
                     x::Integer, y::Integer, scale::Integer)
    W, H = size(img)
    fg = RGB{N0f8}(1, 1, 1)
    bg = RGB{N0f8}(0, 0, 0)
    w = (3 * length(text) + length(text) - 1) * scale
    for j in (y - scale):(y + 5 * scale), i in (x - scale):(x + w)      # backdrop
        (1 <= i <= W && 1 <= j <= H) && (img[i, j] = bg)
    end
    for (k, ch) in enumerate(text)
        d = ch - '0'
        (0 <= d <= 9) || continue
        bits = SHEETDIGITS[d + 1]
        for row in 0:4, col in 0:2
            ((bits >> (14 - (row * 3 + col))) & 1) == 1 || continue
            for sy in 0:(scale - 1), sx in 0:(scale - 1)
                i = x + (k - 1) * 4 * scale + col * scale + sx
                j = y + row * scale + sy
                (1 <= i <= W && 1 <= j <= H) && (img[i, j] = fg)
            end
        end
    end
    return img
end

"""
    sheetframes(seq, t0, t1, n) -> Vector{Int}

The `n` timeline frames a sheet of `[t0, t1]` shows: evenly spaced, ascending,
inside the sequence — the decoders read forward instead of seeking around.
"""
function sheetframes(seq::Sequence, t0::Real, t1::Real, n::Integer)
    last = max(seqlength(seq) - 1, 0)
    a = clamp(round(Int, t0 * seq.framerate), 0, last)
    b = clamp(round(Int, t1 * seq.framerate), a, last)
    n <= 1 && return [a]
    b == a && return fill(a, n)
    return unique(clamp.(round.(Int, range(a, b; length = n)), a, b))
end

"""
    contactsheet(seq, t0 = 0, t1 = seqduration(seq); cells = 64, size = 512,
                 labels = true) -> (image, manifest)

`cells` frames of the time range `[t0, t1]`, rendered as the export renders
them, tiled into ONE image of about `size` × `size` pixels — the knob that
matters is `size`, because vision tokens go by pixels (512² ≈ 350 tokens for 64
frames). Cells keep the source's aspect, so nothing is spent on black bars.
Cell 1 is top-left, filled row by row; with `labels` each carries its number,
which is how an agent points at it.

`manifest` is one entry per cell — `(index, time, frame, clip, col, row)` — so
"cell 27" resolves to a time, and `contactsheet(seq, t, t')` on the interval
around it is the next zoom level. Three levels of 64 cells cover 10 minutes at
frame resolution; see [`findchange`](@ref) for the automatic version.

    sheet, manifest = contactsheet(player.sequence, 0, 12; cells = 36, size = 640)
"""
function contactsheet(seq::Sequence, t0::Real = 0.0, t1::Real = seqduration(seq);
                      cells::Integer = 64, size::Integer = 512, labels::Bool = true,
                      backend = KA.CPU())
    isempty(seq.clips) && return (fill(RGB{N0f8}(0, 0, 0), size, size), NamedTuple[])
    frames = sheetframes(seq, t0, t1, cells)
    canvas = canvassize(seq)
    cols, rows, cw, ch = sheetgrid(length(frames), canvas[1] / canvas[2], size)
    sheet = fill(RGB{N0f8}(0.08, 0.08, 0.09), cols * cw, rows * ch)
    engine = FxEngine(backend)
    readers = Dict{String, Any}()
    manifest = NamedTuple[]
    frame = RGBFrame(undef, canvas[1], canvas[2])
    try
        for (k, n) in enumerate(frames)
            renderframe!(frame, seq, n, readers, engine)
            tile = fitbox(frame, cw, ch, RGB{N0f8}(0.08, 0.08, 0.09))
            col = (k - 1) % cols
            row = (k - 1) ÷ cols
            sheet[(col * cw + 1):(col * cw + cw), (row * ch + 1):(row * ch + ch)] = tile
            scale = max(min(cw, ch) ÷ 32, 1)
            labels && drawdigits!(sheet, string(k), col * cw + 2 + scale,
                                  row * ch + ch - 2 - 5 * scale, scale)
            i = clipat(seq, n)
            push!(manifest, (index = k, time = round(n / seq.framerate, digits = 3),
                             frame = n, clip = something(i, 0), col = col + 1, row = row + 1))
        end
    finally
        foreach(close, values(readers))
        emptyengine!(engine)
    end
    return (sheet, manifest)
end

contactsheet(player::Player, args...; kwargs...) =
    contactsheet(player.sequence, args...; kwargs...)

"""
    filmstrip(seq, t0, t1; count = 8, height = 120) -> (image, manifest)

The same as [`contactsheet`](@ref) but in ONE row and bigger — for reading a
short span (a gesture, a cut, a transition) in order, left to right.
"""
function filmstrip(seq::Sequence, t0::Real, t1::Real; count::Integer = 8,
                   height::Integer = 120, labels::Bool = true, backend = KA.CPU())
    isempty(seq.clips) && return (fill(RGB{N0f8}(0, 0, 0), height, height), NamedTuple[])
    frames = sheetframes(seq, t0, t1, count)
    canvas = canvassize(seq)
    cw = max(round(Int, height * canvas[1] / canvas[2]), 16)
    strip = fill(RGB{N0f8}(0.08, 0.08, 0.09), cw * length(frames), height)
    engine = FxEngine(backend)
    readers = Dict{String, Any}()
    manifest = NamedTuple[]
    frame = RGBFrame(undef, canvas[1], canvas[2])
    try
        for (k, n) in enumerate(frames)
            renderframe!(frame, seq, n, readers, engine)
            strip[((k - 1) * cw + 1):(k * cw), :] = fitbox(frame, cw, height,
                                                           RGB{N0f8}(0.08, 0.08, 0.09))
            labels && drawdigits!(strip, string(k), (k - 1) * cw + 4, height - 12, 2)
            i = clipat(seq, n)
            push!(manifest, (index = k, time = round(n / seq.framerate, digits = 3),
                             frame = n, clip = something(i, 0)))
        end
    finally
        foreach(close, values(readers))
        emptyengine!(engine)
    end
    return (strip, manifest)
end

filmstrip(player::Player, args...; kwargs...) = filmstrip(player.sequence, args...; kwargs...)

"""
    framegrab(seq, t; width = 480) -> image

One finished frame at timeline time `t`, scaled to `width` — what the export
writes there. The spatial counterpart of a sheet cell.
"""
function framegrab(seq::Sequence, t::Real; width::Integer = 480, backend = KA.CPU())
    return regiongrab(seq, t, (0.0, 0.0, 1.0, 1.0); width = width, backend = backend)
end

framegrab(player::Player, args...; kwargs...) = framegrab(player.sequence, args...; kwargs...)

"""
    regiongrab(seq, t, (x, y, w, h); width = 480) -> image

The normalized rectangle `(x, y, w, h)` of the finished frame at time `t`, at
`width` pixels — spatial zoom. `(0, 0, 1, 1)` is the whole frame; a sheet cell
that shows something small becomes readable with e.g. `(0.4, 0.3, 0.2, 0.2)`.
Cheap: the crop is taken from the rendered frame, so the detail is real pixels,
not an upscaled thumbnail (as long as `width` stays under `w × source width`).
"""
function regiongrab(seq::Sequence, t::Real, rect::NTuple{4, Real};
                    width::Integer = 480, backend = KA.CPU())
    canvas = canvassize(seq)
    isempty(seq.clips) && return fill(RGB{N0f8}(0, 0, 0), width, width)
    n = clamp(round(Int, t * seq.framerate), 0, max(seqlength(seq) - 1, 0))
    engine = FxEngine(backend)
    readers = Dict{String, Any}()
    frame = RGBFrame(undef, canvas[1], canvas[2])
    try
        renderframe!(frame, seq, n, readers, engine)
    finally
        foreach(close, values(readers))
        emptyengine!(engine)
    end
    x, y, w, h = clamp.(Float64.(rect), 0.0, 1.0)
    i0 = clamp(round(Int, x * canvas[1]) + 1, 1, canvas[1])
    j0 = clamp(round(Int, y * canvas[2]) + 1, 1, canvas[2])
    i1 = clamp(round(Int, (x + w) * canvas[1]), i0, canvas[1])
    j1 = clamp(round(Int, (y + h) * canvas[2]), j0, canvas[2])
    sub = collect(frame[i0:i1, j0:j1])
    ow = clamp(width, 16, 1920)
    # asked for the pixels as they are: hand them over untouched. `downscale`
    # RESAMPLES even at 1:1 (nearest sampling with a half-pixel offset), which
    # shifts sharp edges by a pixel — an agent comparing this against a render
    # would see a difference that isn't in the edit
    ow == Base.size(sub, 1) && return sub
    oh = max(round(Int, Base.size(sub, 2) / Base.size(sub, 1) * ow), 16)
    return downscale(sub, ow, oh)
end

regiongrab(player::Player, args...; kwargs...) = regiongrab(player.sequence, args...; kwargs...)

"""
    framestat(seq, n, readers, engine, buf) -> (mean, meanabsdiff-ready gray)

Mean brightness of finished frame `n`, plus the small gray image the change
search compares. Small on purpose: 64×36 is enough to see a cut and costs
nothing to compare.
"""
function framestat!(gray::Matrix{Float32}, seq::Sequence, n::Integer,
                    readers::Dict{String, Any}, engine::FxEngine, buf::RGBFrame)
    renderframe!(buf, seq, n, readers, engine)
    small = downscale(buf, size(gray, 1), size(gray, 2))
    @inbounds for j in axes(gray, 2), i in axes(gray, 1)
        c = small[i, j]
        gray[i, j] = Float32(0.299 * c.r + 0.587 * c.g + 0.114 * c.b)
    end
    return gray
end

"""
    findchange(seq, t0, t1; threshold = 0.08, backend) -> (time, difference) | nothing

Bisect `[t0, t1]` for the moment the picture CHANGES — a cut, a flash, a light
switching on: the first time the finished frame stops resembling the frame at
`t0` by more than `threshold` (mean absolute difference of a 64×36 gray
thumbnail, 0..1). O(log n) frame renders instead of a scan: 10 minutes at 30 fps
is ~15 decodes.

Returns `nothing` when the end frame still resembles the start — nothing to
find in that range. This is the automatic half of the zoom: use it when the
property is measurable, [`contactsheet`](@ref) when only a model can judge it.
"""
function findchange(seq::Sequence, t0::Real = 0.0, t1::Real = seqduration(seq);
                    threshold::Real = 0.08, backend = KA.CPU())
    isempty(seq.clips) && return nothing
    last = max(seqlength(seq) - 1, 0)
    lo = clamp(round(Int, t0 * seq.framerate), 0, last)
    hi = clamp(round(Int, t1 * seq.framerate), lo, last)
    lo == hi && return nothing
    canvas = canvassize(seq)
    engine = FxEngine(backend)
    readers = Dict{String, Any}()
    buf = RGBFrame(undef, canvas[1], canvas[2])
    ref = zeros(Float32, 64, 36)
    probe = zeros(Float32, 64, 36)
    try
        framestat!(ref, seq, lo, readers, engine, buf)
        diff(n) = (framestat!(probe, seq, n, readers, engine, buf);
                   sum(abs, probe .- ref) / length(ref))
        d = diff(hi)
        d <= threshold && return nothing          # the range never changes
        a, b = lo, hi                             # invariant: diff(a) <= thr < diff(b)
        while b - a > 1
            m = (a + b) ÷ 2
            diff(m) > threshold ? (b = m) : (a = m)
        end
        return (round(b / seq.framerate, digits = 3), round(d, digits = 4))
    finally
        foreach(close, values(readers))
        emptyengine!(engine)
    end
end

findchange(player::Player, args...; kwargs...) = findchange(player.sequence, args...; kwargs...)

"""
    viewsummary(seq) -> NamedTuple

The numbers an agent needs before it looks at anything: duration, frame rate,
canvas, per-clip ranges (timeline AND source), and what is on each clip
(effects, stabilization, keyframed params). The text half of "see the edit".
"""
function viewsummary(seq::Sequence)
    fps = seq.framerate
    clips = map(enumerate(seq.clips)) do (i, c)
        (index = i, track = c.track,
         start_time = round(c.start / fps, digits = 3),
         end_time = round(clipend(c) / fps, digits = 3),
         source = basename(c.source.path),
         # in the SOURCE's own seconds: `src_in` counts source frames, and on a
         # conformed clip those don't tick at the sequence rate
         source_in = round(c.src_in / c.source.framerate, digits = 3),
         source_fps = c.source.framerate,
         conformed = conformed(c),
         crop = round.(c.crop, digits = 3),
         effects = [string(nameof(typeof(op(s)))) for s in c.effects if s.enabled],
         stabilized = c.motiontrack !== nothing,
         flicker_fixed = c.colortrack !== nothing,
         animated = sort!([string(fx.kind, ".", prm.name) for fx in c.effects
                           for prm in fx.params if isanimated(prm)]))
    end
    return (duration = round(seqduration(seq), digits = 3), framerate = fps,
            frames = seqlength(seq), canvas = canvassize(seq),
            tracks = ntracks(seq), clips = clips)
end

viewsummary(player::Player) = viewsummary(player.sequence)
