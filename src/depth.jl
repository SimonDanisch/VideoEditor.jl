"""
Estimated per-frame depth, and the one effect that reads it.

Depth is the cheapest subject separation this editor has: no marking, no seeds,
no propagation — you press a button and the background is behind the subject.
It is also the *worst* separation, because a monocular estimate has soft, wrong
edges wherever the model is unsure. So it is deliberately not offered as a matte.
What it is offered as is **defocus**, where soft and slightly wrong is exactly
what a lens does anyway, and where the matte's hard edge is the thing that looks
fake.

The model is pluggable, like the restoration one and for the same reason: the
editor should run with it absent. [`registerdepth!`](@ref) installs it, and
`DepthAnythingRunner` is what `installdnns!` puts there.
"""

"""
    registerdepth!(f)

Install the depth model. `f(img) -> AbstractMatrix{<:Real}` takes one host RGB
frame and returns a depth map of any size; larger values mean nearer.

Any size, because the model has its own input resolution and resampling it to the
frame is this file's business, not the caller's. Any element type, because
normalization happens here — see [`depthbytes`](@ref) for why it has to.
"""
const DEPTHMODEL = Ref{Any}(nothing)

registerdepth!(f) = (DEPTHMODEL[] = f; nothing)
hasdepthmodel() = DEPTHMODEL[] !== nothing

"""
The built-in depth model: DepthAnything V2, from `DepthAnythingRunner`.

Built on FIRST USE and kept, exactly as the SAM 2 segmenter is: constructing it
compiles a graph, and doing that at load time would cost every session the price
whether or not it ever estimates depth. `Array(...)` brings the result back to
the host because `depthbytes` normalizes over the whole frame, and a device
reduction per frame would be a synchronize per frame for a number that then
travels to the host anyway.
"""
const DEPTHANYTHING = Ref{Any}(nothing)

"""
    dropsingletons(a) -> AbstractArray

`a` without its size-1 axes.

The runner hands back the model's raw tensor — a depth map with a batch axis and
a channel axis, neither of which a depth map has. Dropping them here, in the
adapter, keeps the editor's contract as "a depth map is a matrix" rather than
widening every consumer to accept a 4-D array that only ever has two real axes.
"""
dropsingletons(a::AbstractArray) = dropdims(a; dims = Tuple(findall(==(1), size(a))))

function depthanythingdepth(img)
    if DEPTHANYTHING[] === nothing
        DEPTHANYTHING[] = DepthAnythingRunner.depthanything(; backend = Lava.LavaBackend())
    end
    d = dropsingletons(Array(DepthAnythingRunner.depthmap!(DEPTHANYTHING[], img)))
    # Loud, not silent: a model whose output stops being one plane is a change the
    # editor must not paper over by guessing which axis to keep.
    ndims(d) == 2 || error("depth model returned $(ndims(d)) non-singleton axes, expected 2")
    return d
end

"""
    installdepth!()

Point [`registerdepth!`](@ref) at the built-in model. Called by `Player`, so an
editor has depth without being asked; the indirection stays because the editor
must still run with the model absent or replaced.
"""
installdepth!() = registerdepth!(depthanythingdepth)

"""
    depthbytes(d) -> Matrix{UInt8}

One depth map, normalized to `0x00` (farthest) … `0xff` (nearest).

Per frame, necessarily: a monocular model has no scale, so its
output is an ordering, and the numbers behind that ordering drift between frames
of the same shot. Normalizing per frame at least makes "nearest thing visible"
mean the same thing everywhere; not normalizing would make an effect's threshold
mean a different distance on every frame.

A frame with no range at all — a flat wall, a fade to black — would divide by
zero, and is mapped to a constant mid-grey instead. Nothing is nearer than
anything else there, which is the truth.
"""
function depthbytes(d::AbstractMatrix{<:Real})
    lo, hi = extrema(d)
    span = hi - lo
    out = Matrix{UInt8}(undef, size(d, 1), size(d, 2))
    if !(span > 0) || !isfinite(span)
        fill!(out, 0x80)
        return out
    end
    @inbounds for i in eachindex(out, d)
        out[i] = round(UInt8, clamp(255 * (d[i] - lo) / span, 0, 255))
    end
    return out
end

"""
    analyzedepth!(clip, readframe; maxside = 384, progress = nothing) -> DepthTrack

Estimate depth for every frame of `clip` and hang it on the clip.

`readframe(srcframe) -> host RGB frame` is the same reader `analyzematte!` takes,
so the two analyses see the same picture — post-effect and cropped, which is what
the user is looking at and therefore what an effect reading the result must line
up with.

`maxside` caps the stored resolution, not the model's: the model resamples to its
own input size regardless, so a bigger track costs memory and buys nothing but a
softer resample on the way out. Depth is smooth by nature, which is exactly the
signal that survives being stored small.
"""
function analyzedepth!(clip::Clip, readframe; maxside::Integer = 384, progress = nothing)
    hasdepthmodel() || error("no depth model installed — see registerdepth!")
    n = srclength(clip)
    n > 0 || error("cannot estimate depth for an empty clip")
    f0 = readframe(clip.src_in)
    dw, dh = mattereadsize(size(f0), maxside)
    track = DepthTrack(Array{UInt8, 3}(undef, dw, dh, n), clip.src_in)
    for k in 1:n
        img = k == 1 ? f0 : readframe(clip.src_in + k - 1)
        d = depthbytes(DEPTHMODEL[](img))
        view(track.depth, :, :, k) .= depthscale(d, dw, dh)
        progress === nothing || progress(k, n)
    end
    clip.depthtrack = track
    return track
end

"""
    depthframe(clip, srcframe) -> AbstractMatrix{UInt8} | nothing

One frame of the clip's depth, as a view. `nothing` when there is no track or the
frame is outside it — which is a miss, not an error: a clip renders as decoded
until its depth has been estimated.
"""
function depthframe(clip::Clip, srcframe::Integer)
    t = clip.depthtrack
    t === nothing && return nothing
    k = Int(srcframe) - t.src_in + 1
    1 <= k <= size(t.depth, 3) || return nothing
    return view(t.depth, :, :, k)
end

"""
    depthscale(src, w, h) -> Matrix{UInt8}

Resize a depth map to `w`×`h`, keeping its values.

Not [`mattemaskscale`](@ref), which this used and which ends
`v > 0 ? 0xff : 0x00` — correct for a binary seed mask, catastrophic for a depth
map: every non-zero depth became 0xff, so the track was a uniform "everything is
nearest" plane. Depth blur still changed the picture (it defocused everything
equally), so it passed a does-the-effect-do-something check; it had simply never
done anything depth-related. The card's thumbnail, which is solid white when this
is wrong, is what showed it.

Bilinear, unlike the mask version's nearest neighbour. A mask has two values and
nearest is the only honest choice; a depth map is continuous and drives a
per-pixel blur radius, so a stepped depth map becomes visible banding in the
defocus — concentric rings where the radius jumps. Interpolating costs three
extra lerps per pixel, once per frame, at analysis time.
"""
function depthscale(src::AbstractMatrix{UInt8}, w::Int, h::Int)
    sw, sh = size(src)
    out = Matrix{UInt8}(undef, w, h)
    @inbounds for j in 1:h, i in 1:w
        # sample at pixel CENTRES, so the map is not shifted half a pixel
        x = clamp((i - 0.5) * sw / w + 0.5, 1.0, Float64(sw))
        y = clamp((j - 0.5) * sh / h + 0.5, 1.0, Float64(sh))
        x0 = floor(Int, x); y0 = floor(Int, y)
        x1 = min(x0 + 1, sw); y1 = min(y0 + 1, sh)
        fx = x - x0; fy = y - y0
        a = Float32(src[x0, y0]); b = Float32(src[x1, y0])
        c = Float32(src[x0, y1]); d = Float32(src[x1, y1])
        top = a + (b - a) * fx
        bot = c + (d - c) * fx
        out[i, j] = round(UInt8, clamp(top + (bot - top) * fy, 0.0f0, 255.0f0))
    end
    return out
end

"""
    depthimage(d) -> Matrix{RGB{N0f8}}

One depth plane as a grayscale picture, bright = near.

The card shows this because a monocular depth estimate is a guess, and the effect
built on it hides how good a guess it was: defocus turns a wrong depth into a
soft halo rather than a visible error, so a shot the model misread looks merely
mediocre instead of wrong. The map shows whether it separated subject from
background at all — which is the one question worth asking before touching
either slider.

It is also what makes `Focus` legible. That slider is a number in depth units
with nothing on screen carrying units, so without the map the only way to find a
value is to drag until it looks right.
"""
depthimage(d::AbstractMatrix{UInt8}) =
    map(v -> (g = reinterpret(N0f8, v); RGB{N0f8}(g, g, g)), d)

# ---------------------------------------------------------------- the plane

planeeltype(::DepthBlurOp) = UInt8
passname(::DepthBlurOp) = "depthblur"

planeshape(::DepthBlurOp, clip::Clip) =
    (t = clip.depthtrack; t === nothing ? nothing : depthsize(t))

"""
The object a depth plane's bytes come from: the track itself. Compared by
identity, so re-estimating depth — which replaces the array under an unchanged
clip and frame — writes the plane again instead of trusting a stale upload.
"""
planesource(::DepthBlurOp, clip::Clip, ::Integer) = clip.depthtrack

planedata(::DepthBlurOp, clip::Clip, srcframe::Integer) =
    (d = depthframe(clip, srcframe); d === nothing ? nothing : vec(d))

"""
    depthblur!(out, img, plane, op)

Defocus `img` into `out` by each pixel's distance from `op.focus` in depth.

One gather pass, not two separable ones: a per-pixel radius is not separable
— the two passes disagree wherever the radius changes — and separating it would
buy a second scratch buffer for the privilege. At this radius the gather is ~169
taps at the very worst and only where the picture is defocused.

`out` is the node's own transient (see `DepthBlurNode`), never `img`: a gather
reads neighbours of the pixel it writes, so writing in place would sample pixels
this pass had already blurred.
"""
function depthblur!(out, img, plane, op::DepthBlurOp)
    s = clamp(op.strength, 0.0f0, 1.0f0)
    w, h = size(img, 1), size(img, 2)
    if s <= 0.001f0
        copyto!(out, img)
        return out
    end
    # Capped in pixels, not by the fraction alone: the tap count is the square of
    # this, so a fraction of a 4K frame would be a 40-tap radius and 6561 samples
    # a pixel. Six is where defocus reads as defocus and the cost stays flat.
    maxr = Int32(clamp(round(Int, s * 0.02 * min(w, h)), 1, 6))
    # NO synchronize. This runs inside a graph pass body, where Mantle orders the
    # passes from the `use` declarations in `chainpass!` — and the kernels that
    # already run there (`coloradjust!`, `gaussianblur!`, `unsharpmask!`) do not
    # synchronize either. Draining the pipeline mid-graph is a measured -7%
    # elsewhere in this project for exactly this mistake.
    depthblur_kernel!(KA.get_backend(out))(out, img, plane,
                                           Int32(size(plane, 1)), Int32(size(plane, 2)),
                                           op.focus, maxr; ndrange = (w, h))
    return out
end

"""
Blur `img` into `out`, radius from `depth`'s distance to `focus`.

The depth plane is sampled at ITS resolution — depth is stored small on purpose —
so the lookup rescales. Nearest, not bilinear: the value only chooses a radius,
and a radius is an integer by the time anything uses it.
"""
@kernel function depthblur_kernel!(out, @Const(img), @Const(depth),
                                   dw::Int32, dh::Int32, focus::Float32, maxr::Int32)
    I = @index(Global, Cartesian)
    x, y = Int32(I[1]), Int32(I[2])
    w, h = Int32(size(img, 1)), Int32(size(img, 2))
    @inbounds begin
        dx = clamp(div((x - Int32(1)) * dw, w) + Int32(1), Int32(1), dw)
        dy = clamp(div((y - Int32(1)) * dh, h) + Int32(1), Int32(1), dh)
        z = Float32(depth[dx + (dy - Int32(1)) * dw]) / 255.0f0
        # Distance from the focus plane, so the sharp band sits at `focus` and
        # both nearer and farther go soft — which is what a lens does, and what
        # makes focusing on a mid-ground subject possible at all.
        rad = clamp(Int32(round(abs(z - focus) * Float32(maxr))), Int32(0), maxr)
        r = 0.0f0; g = 0.0f0; b = 0.0f0; a = 0.0f0; n = 0.0f0
        for j in -rad:rad, i in -rad:rad
            xx = clamp(x + i, Int32(1), w)
            yy = clamp(y + j, Int32(1), h)
            c = img[xx, yy]
            r += Float32(red(c)); g += Float32(green(c)); b += Float32(blue(c))
            a += alphaof(c)
            n += 1.0f0
        end
        # `unitn0f8`, NOT `RGB{N0f8}(::Float32, …)` — the latter validates and calls
        # `throw_colorerror`, which drags string building into the kernel's IR and
        # makes Lava reject the whole thing. See the note on `unitn0f8`: the clamp
        # IS the check. This kernel had the validating form and so never compiled,
        # which is why depth blur could not render even once depth existed.
        # Coverage is averaged with the colour: this is a box blur, both are
        # weighted sums of the same taps, and the plane is premultiplied.
        out[x, y] = topixel(eltype(out), r / n, g / n, b / n, a / n)
    end
end
