# Linear HDR source masters are immutable; effects only produce derived planes.
sourcecache(clip::Clip) = (c = trackcache(clip); c === nothing ? nothing : c.source)
function ensuresourcecache!(clip::Clip)
    clip.sequence isa Sequence || error("attach the scene clip to a Sequence before baking HDR")
    if trackcache(clip) === nothing
        enabletrackcache!(clip.sequence, mktempdir(; cleanup = false))
    end
    c = trackcache(clip)
    c.source === nothing && (c.source = TrackCache(joinpath(c.directory, "source"); format = :hdr))
    return c.source
end
function cachedsourceframe(clip::Clip, sf::Integer)
    c = sourcecache(clip)
    c === nothing && return nothing
    cachevalid(c, clip.id, sf) || return nothing
    c.hits += 1
    return readrenderframe(cachefile(c, clip.id, sf))
end

const HDR_PLANES = (:color, :albedo, :normal, :depth, :coverage, :overlay)
function writerenderframe(path, frame)
    ENDIAN_BOM == 0x04030201 || error("VEHDR requires a little-endian host")
    dims = size(frame.color)
    all(size(getproperty(frame, k)) == dims for k in HDR_PLANES) || error("buffer size mismatch")
    meta = JSON.json(Dict("version" => 1, "dims" => collect(dims), "metadata" => frame.metadata))
    open(path, "w") do io
        write(io, "VEHDR1\n"); write(io, UInt32(ncodeunits(meta))); write(io, meta)
        for k in HDR_PLANES
            write(io, reinterpret(UInt8, vec(hostplane(getproperty(frame, k)))))
        end
    end
    return path
end
# A plane of a frame being written to disk: a captured frame's are on the
# device and come down once, here, for the master.
hostplane(a::Array) = a
hostplane(a) = Array(a)

function readrenderframe(path)
    ENDIAN_BOM == 0x04030201 || error("VEHDR requires a little-endian host")
    return open(path, "r") do io
        String(read(io, 7)) == "VEHDR1\n" || error("invalid HDR header")
        bytes = read(io, UInt32); bytes <= 1024 * 1024 || error("invalid metadata size")
        d = JSON.parse(String(read(io, bytes)))
        d["version"] == 1 || error("unsupported HDR version")
        dims = Tuple(Int.(d["dims"]))
        length(dims) == 2 && all(>(0), dims) || error("invalid dimensions")
        types = (RGB{Float32}, RGB{Float32}, Vec3f, Float32, Float32, RGBA{Float32})
        planes = map(types) do T
            image = Matrix{T}(undef, dims)
            read!(io, reinterpret(UInt8, vec(image))); image
        end
        eof(io) || error("trailing HDR data")
        merge(NamedTuple{HDR_PLANES}(planes), (metadata = d["metadata"],))
    end
end

abstract type HDRFxOp <: FxOp end
struct LinearGradeEffect <: HDRFxOp
    exposure::Float32
    saturation::Float32
    redgain::Float32
    bluegain::Float32
end
LinearGradeEffect(; exposure = 0, saturation = 1, redgain = 1, bluegain = 1) =
    LinearGradeEffect(Float32(exposure), Float32(saturation), Float32(redgain), Float32(bluegain))
struct HDRDenoiseEffect <: HDRFxOp
    strength::Float32
    iterations::Int
end
HDRDenoiseEffect(; strength = 1, iterations = 3) = HDRDenoiseEffect(Float32(strength), Int(iterations))
struct ToneMapEffect <: HDRFxOp
    mode::Int
    gamma::Float32
end
ToneMapEffect(; mode = 2, gamma = 2.2) = ToneMapEffect(Int(mode), Float32(gamma))
# Executed on floating-point source data, not twice in the display GPU graph.
isneutral(::HDRFxOp) = true
nodefor(::HDRFxOp, input, clip) = nothing
for (name, label, T, params, make, read) in (
        (
            :lineargrade, "Linear HDR grade", LinearGradeEffect,
            [
                FxParam(:exposure, "Exposure (stops)"; min = -10, max = 10, default = 0),
                FxParam(:saturation, "Saturation"; min = 0, max = 3, default = 1),
                FxParam(:redgain, "Red gain"; min = 0, max = 3, default = 1),
                FxParam(:bluegain, "Blue gain"; min = 0, max = 3, default = 1),
            ],
            nt -> LinearGradeEffect(; nt...), e -> (
                exposure = Float64(e.exposure), saturation = Float64(e.saturation),
                redgain = Float64(e.redgain), bluegain = Float64(e.bluegain),
            ),
        ),
        (
            :hdrdenoise, "HDR denoise (spatial)", HDRDenoiseEffect,
            [
                FxParam(:strength, "Strength"; min = 0, max = 1, default = 1),
                FxParam(:iterations, "Iterations"; min = 1, max = 5, default = 3),
            ],
            nt -> HDRDenoiseEffect(strength = nt.strength, iterations = round(Int, nt.iterations)),
            e -> (strength = Float64(e.strength), iterations = Float64(e.iterations)),
        ),
        (
            :tonemap, "HDR display transform", ToneMapEffect,
            [
                FxParam(:mode, "Curve: 0 clip / 1 Reinhard / 2 ACES"; min = 0, max = 2, default = 2),
                FxParam(:gamma, "Display gamma"; min = 1, max = 3, default = 2.2),
            ],
            nt -> ToneMapEffect(mode = round(Int, nt.mode), gamma = nt.gamma),
            e -> (mode = Float64(e.mode), gamma = Float64(e.gamma)),
        ),
    )
    registereffect!(EffectKind(name, label; params, make, matches = e -> e isa T, read))
end

"Native OIDN with optional albedo/normal prefiltering; never changes the source bake."
struct OIDNDenoiseEffect <: HDRFxOp
    strength::Float32
    prefilter::Bool
end
OIDNDenoiseEffect(; strength = 1, prefilter = true) = OIDNDenoiseEffect(Float32(strength), Bool(prefilter))
effectpackages(::OIDNDenoiseEffect) = ["OIDNRunner"]
const HDR_DENOISERS = Dict{Symbol, Any}()
"""
    registerhdrdenoiser!(name, callback)

Install the HDR denoiser `name` (`:oidn` is OIDNRunner's): `callback(out, frame,
image, effect)` writes `image` denoised with `frame`'s albedo and normal guides
into `out`, all device matrices on one device, and leaves them there.
"""
registerhdrdenoiser!(name::Symbol, callback) = (HDR_DENOISERS[name] = callback; nothing)
registerhdrdenoiser!(callback::Function, name::Symbol) = registerhdrdenoiser!(name, callback)
registereffect!(
    EffectKind(
        :oidndenoise, "OIDN HDR denoise";
        params = [
            FxParam(:strength, "Strength"; min = 0, max = 1, default = 1),
            FxParam(:prefilter, "Prefilter guides"; min = 0, max = 1, default = 1),
        ],
        make = nt -> OIDNDenoiseEffect(strength = nt.strength, prefilter = nt.prefilter >= 0.5),
        matches = e -> e isa OIDNDenoiseEffect,
        read = e -> (strength = Float64(e.strength), prefilter = Float64(e.prefilter))
    )
)
# ---------------------------------------------------------------- the HDR stage, on the device
#
# A traced frame's effects run where its planes are. RayMakie captures them on
# the device (`renderbuffers`), a cached master is uploaded into the clip's
# workspace once, and every effect is a kernel or a model there, chosen by its
# type. The display transform writes the plane the clip's graph reads, which
# the scene pass copies in on the device too; the master's download for the
# cache is the only time the frame comes to the host.

"""
    HDRWork(backend, w, h)

What one scene clip's HDR stage works in, kept between frames: the radiance it
grades and denoises (a copy, the master stays as captured), two scratch images
(a denoiser's output, a filter's ping-pong), the display plane it writes, and
the planes a cached master is uploaded into.
"""
struct HDRWork{C, P, M}
    image::C
    scratch1::C
    scratch2::C
    out::P
    master::M
end
function HDRWork(backend, w::Int, h::Int)
    plane(T) = KA.allocate(backend, T, w, h)
    master = (color = plane(RGB{Float32}), albedo = plane(RGB{Float32}), normal = plane(Vec3f),
              depth = plane(Float32), coverage = plane(Float32), overlay = plane(RGBA{Float32}))
    return HDRWork(plane(RGB{Float32}), plane(RGB{Float32}), plane(RGB{Float32}), plane(PlanePixel), master)
end
freework!(w::HDRWork) = foreach(KA.unsafe_free!, (w.image, w.scratch1, w.scratch2, w.out, w.master...))

"""The HDR workspace of `src` for w×h frames on `backend`, made on first use and when either changes."""
function hdrwork!(src::SceneSource, backend, dims::Tuple{Int, Int})
    w = src.hdr
    w !== nothing && size(w.image) == dims && KA.get_backend(w.image) == backend && return w
    w === nothing || freework!(w)
    src.hdr = HDRWork(backend, dims...)
    return src.hdr
end

"""A cached master's planes uploaded into the workspace's, one copy each; the frame the stage reads."""
function uploadmaster!(work::HDRWork, frame)
    for k in HDR_PLANES
        copyto!(getproperty(work.master, k), getproperty(frame, k))
    end
    return merge(work.master, (metadata = frame.metadata,))
end

"""
    hdrpicture!(work, frame, clip, sf) -> device matrix of PlanePixel

The display plane of a traced source frame, in `work.out`: the clip's HDR
effects in order on a copy of the linear radiance, then its display transform
(the default one when it has none), the coverage and the raster overlay over
it. `frame`'s planes are on the device; nothing here touches the host.
"""
function hdrpicture!(work::HDRWork, frame, clip, sf)
    copyto!(work.image, frame.color)    # never modify the source master or its guides
    stage = HDRStage(work, nothing)
    for fx in clip.effects
        renderable(fx) && fx.enabled[] || continue
        stage = hdrstep(stage, frame, op(fx, sf))
    end
    display = displaytransform(stage)
    displayplane_kernel!(KA.get_backend(work.out))(work.out, work.image, frame.coverage, frame.overlay,
        Int32(display.mode), 1.0f0 / display.gamma; ndrange = size(work.out))
    return work.out
end

"""
The HDR stage partway through a clip's effects: its workspace, and `nothing`
until the display transform is reached, then that `ToneMapEffect`. Each effect
is one `hdrstep` method; an op that is not an HDR effect is the display graph's
and passes through.
"""
struct HDRStage{W, D}
    work::W
    display::D
end
hdrstep(s::HDRStage, frame, op) = s
hdrstep(::HDRStage{<:Any, ToneMapEffect}, frame, ::HDRFxOp) =
    error("HDR effects must precede the display transform")
hdrstep(s::HDRStage{<:Any, Nothing}, frame, e::ToneMapEffect) = HDRStage(s.work, e)
hdrstep(s::HDRStage{<:Any, Nothing}, frame, e::LinearGradeEffect) = (lineargrade!(s.work.image, e); s)
hdrstep(s::HDRStage{<:Any, Nothing}, frame, e::HDRDenoiseEffect) = (hdrdenoise!(s.work, frame, e); s)
hdrstep(s::HDRStage{<:Any, Nothing}, frame, e::OIDNDenoiseEffect) = (oidn!(s.work, frame, e); s)

displaytransform(::HDRStage{<:Any, Nothing}) = ToneMapEffect()
displaytransform(s::HDRStage{<:Any, ToneMapEffect}) = s.display

lineargrade!(img, e::LinearGradeEffect) =
    lineargrade_kernel!(KA.get_backend(img))(img, exp2(e.exposure), e.saturation, e.redgain, e.bluegain;
                                             ndrange = size(img))

@kernel function lineargrade_kernel!(img, scale::Float32, saturation::Float32, redgain::Float32, bluegain::Float32)
    I = @index(Global, Cartesian)
    @inbounds c = img[I]
    lum = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
    @inbounds img[I] = RGB{Float32}((lum + (c.r - lum) * saturation) * scale * redgain,
                                    (lum + (c.g - lum) * saturation) * scale,
                                    (lum + (c.b - lum) * saturation) * scale * bluegain)
end

"""`img` moved towards `other` by `s`: `img (1 - s) + other s`, per channel."""
hdrmix!(img, other, s::Float32) = hdrmix_kernel!(KA.get_backend(img))(img, other, s; ndrange = size(img))
@kernel function hdrmix_kernel!(img, @Const(other), s::Float32)
    I = @index(Global, Cartesian)
    @inbounds a = img[I]
    @inbounds b = other[I]
    @inbounds img[I] = RGB{Float32}(a.r * (1 - s) + b.r * s, a.g * (1 - s) + b.g * s, a.b * (1 - s) + b.b * s)
end

"""
OIDN through the denoiser OIDNRunner registers: radiance and guides on the
device, its result into the workspace's scratch, mixed in by the strength.
"""
function oidn!(work::HDRWork, frame, e::OIDNDenoiseEffect)
    e.strength <= 0 && return work
    get(frame.metadata, "albedo_valid", false) && get(frame.metadata, "normal_valid", false) ||
        error("OIDN requires valid cached material albedo and normals; refresh old guides first")
    callback = get(HDR_DENOISERS, :oidn, nothing)
    callback === nothing && error("load OIDNRunner and call OIDNRunner.install!(denoiser)")
    callback(work.scratch1, frame, work.image, e)
    hdrmix!(work.image, work.scratch1, e.strength)
    return work
end

# The B3-spline taps of the à-trous filter, one row of the 5×5 kernel.
const ATROUS_TAPS = (0.0625f0, 0.25f0, 0.375f0, 0.25f0, 0.0625f0)

"""
A conservative spatial reference filter: `e.iterations` à-trous passes, the
taps spreading by a power of two each pass, stopping at depth, coverage, normal
and luminance edges. No temporal neighbours, so its invalidation stays
frame-local. Ping-pongs between the workspace's scratch images and mixes the
result into the radiance by the strength.
"""
function hdrdenoise!(work::HDRWork, frame, e::HDRDenoiseEffect)
    e.strength <= 0 && return work
    1 <= e.iterations <= 5 || error("denoiser iterations must be 1:5")
    backend = KA.get_backend(work.image)
    src, dst = work.image, work.scratch1
    for iteration in 1:e.iterations
        atrous_kernel!(backend)(dst, src, frame.normal, frame.depth, frame.coverage, 1 << (iteration - 1);
                                ndrange = size(dst))
        src, dst = dst, (dst === work.scratch1 ? work.scratch2 : work.scratch1)
    end
    hdrmix!(work.image, src, e.strength)
    return work
end

@inline hdrluminance(c) = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b

@kernel function atrous_kernel!(dst, @Const(src), @Const(normal), @Const(depth), @Const(coverage), step::Int)
    x, y = @index(Global, NTuple)
    w, h = size(src)
    @inbounds begin
        center = src[x, y]
        n = normal[x, y]
        d0 = depth[x, y]
        c0 = coverage[x, y]
        lc = hdrluminance(center)
        tr = 0.0f0; tg = 0.0f0; tb = 0.0f0; count = 0.0f0
        for dy in -2:2, dx in -2:2
            xx = clamp(x + dx * step, 1, w)
            yy = clamp(y + dy * step, 1, h)
            d = depth[xx, yy]
            isfinite(d) == isfinite(d0) || continue
            abs(coverage[xx, yy] - c0) < 0.01f0 || continue
            nb = normal[xx, yy]
            nd = max(0.0f0, n[1] * nb[1] + n[2] * nb[2] + n[3] * nb[3])
            nw = iszero(n) && iszero(nb) ? 1.0f0 : nd^64
            dw = isfinite(d0) ? exp(-abs(d0 - d) / max(0.1f0, 0.02f0 * abs(d0))) : 1.0f0
            s = src[xx, yy]
            cw = exp(-abs(lc - hdrluminance(s)) / max(0.1f0, abs(lc)))
            wt = ATROUS_TAPS[dx + 3] * ATROUS_TAPS[dy + 3] * nw * dw * cw
            tr += s.r * wt; tg += s.g * wt; tb += s.b * wt; count += wt
        end
        dst[x, y] = count > 0 ? RGB{Float32}(tr / count, tg / count, tb / count) : center
    end
end

"""One channel through the display transform: Reinhard (mode 1), the ACES fit (mode 2) or none, then the gamma."""
@inline function tonemapchannel(v::Float32, mode::Int32, invgamma::Float32)
    v = max(0.0f0, v)
    t = mode == Int32(1) ? v / (1.0f0 + v) :
        mode == Int32(2) ? (v * (2.51f0 * v + 0.03f0)) / (v * (2.43f0 * v + 0.59f0) + 0.14f0) : v
    return clamp(t, 0.0f0, 1.0f0)^invgamma
end

"""A 0–1 value as 8-bit unsigned normalized, rounded to nearest."""
@inline unorm8(v::Float32) = reinterpret(N0f8, unsafe_trunc(UInt8, round(clamp(v, 0.0f0, 1.0f0) * 255.0f0)))

"""The tone-mapped radiance where it covers the frame, under the raster overlay, premultiplied."""
@kernel function displayplane_kernel!(out, @Const(img), @Const(coverage), @Const(overlay), mode::Int32, invgamma::Float32)
    I = @index(Global, Cartesian)
    @inbounds begin
        c = img[I]; a = coverage[I]; o = overlay[I]
        out[I] = PlanePixel(unorm8(tonemapchannel(c.r, mode, invgamma) * a * (1 - o.alpha) + o.r),
                            unorm8(tonemapchannel(c.g, mode, invgamma) * a * (1 - o.alpha) + o.g),
                            unorm8(tonemapchannel(c.b, mode, invgamma) * a * (1 - o.alpha) + o.b),
                            unorm8(a * (1 - o.alpha) + o.alpha))
    end
end
