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
            write(io, reinterpret(UInt8, vec(getproperty(frame, k))))
        end
    end
    return path
end
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

# A conservative spatial reference filter. No temporal neighbours, so its
# invalidation remains frame-local. OIDN can replace this stage independently.
function hdrdenoise(frame, image, e::HDRDenoiseEffect)
    e.strength <= 0 && return image
    1 <= e.iterations <= 5 || error("denoiser iterations must be 1:5")
    result = copy(image); weights = Float32[1, 4, 6, 4, 1] ./ 16
    lum(c) = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
    for iteration in 1:e.iterations
        input = result; result = similar(input); step = 1 << (iteration - 1)
        for y in axes(input, 2), x in axes(input, 1)
            center = input[x, y]; normal = frame.normal[x, y]; depth = frame.depth[x, y]
            total = zero(RGB{Float32}); count = 0.0f0
            for dy in -2:2, dx in -2:2
                xx = clamp(x + dx * step, 1, size(input, 1)); yy = clamp(y + dy * step, 1, size(input, 2))
                d = frame.depth[xx, yy]
                isfinite(d) == isfinite(depth) || continue
                abs(frame.coverage[xx, yy] - frame.coverage[x, y]) < 0.01f0 || continue
                nd = max(0.0f0, sum(normal .* frame.normal[xx, yy]))
                nw = all(iszero, normal) && all(iszero, frame.normal[xx, yy]) ? 1.0f0 : nd^64
                dw = isfinite(depth) ? exp(-abs(depth - d) / max(0.1f0, 0.02f0 * abs(depth))) : 1.0f0
                cw = exp(-abs(lum(center) - lum(input[xx, yy])) / max(0.1f0, abs(lum(center))))
                w = weights[dx + 3] * weights[dy + 3] * nw * dw * cw
                total += input[xx, yy] * w; count += w
            end
            result[x, y] = count > 0 ? total / count : center
        end
    end
    return image .* (1 - e.strength) .+ result .* e.strength
end

"Native OIDN with optional albedo/normal prefiltering; never changes the source bake."
struct OIDNDenoiseEffect <: HDRFxOp
    strength::Float32
    prefilter::Bool
end
OIDNDenoiseEffect(; strength = 1, prefilter = true) = OIDNDenoiseEffect(Float32(strength), Bool(prefilter))
const HDR_DENOISERS = Dict{Symbol, Any}()
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
function oidn_effect(frame, image, e::OIDNDenoiseEffect)
    e.strength <= 0 && return image
    get(frame.metadata, "albedo_valid", false) && get(frame.metadata, "normal_valid", false) ||
        error("OIDN requires valid cached material albedo and normals; refresh old guides first")
    callback = get(HDR_DENOISERS, :oidn, nothing)
    callback === nothing && error("load OIDNRunner and call OIDNRunner.install!(denoiser)")
    output = callback(frame, image, e)
    size(output) == size(image) || error("OIDN returned the wrong dimensions")
    return image .* (1 - e.strength) .+ output .* e.strength
end

function hdrpicture(frame, clip, sf)
    image = copy(frame.color) # never modify the source master or its guides
    display = ToneMapEffect(); seen_display = false
    for fx in clip.effects
        renderable(fx) && fx.enabled[] || continue
        e = op(fx, sf); e isa HDRFxOp || continue
        seen_display && error("HDR effects must precede the display transform")
        if e isa LinearGradeEffect
            scale = exp2(e.exposure)
            image = map(image) do c
                lum = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
                RGB{Float32}(
                    (lum + (c.r - lum) * e.saturation) * scale * e.redgain,
                    (lum + (c.g - lum) * e.saturation) * scale,
                    (lum + (c.b - lum) * e.saturation) * scale * e.bluegain
                )
            end
        elseif e isa HDRDenoiseEffect
            image = hdrdenoise(frame, image, e)
        elseif e isa OIDNDenoiseEffect
            image = oidn_effect(frame, image, e)
        elseif e isa ToneMapEffect
            display = e; seen_display = true
        end
    end
    channel(v) = begin
        v = max(0.0f0, v)
        t = display.mode == 1 ? v / (1.0f0 + v) :
            display.mode == 2 ? (v * (2.51f0 * v + 0.03f0)) / (v * (2.43f0 * v + 0.59f0) + 0.14f0) : v
        clamp(t, 0.0f0, 1.0f0)^(1.0f0 / display.gamma)
    end
    out = Matrix{PlanePixel}(undef, size(image))
    for i in eachindex(image)
        c = image[i]; a = frame.coverage[i]; o = frame.overlay[i]
        out[i] = PlanePixel(
            channel(c.r) * a * (1 - o.alpha) + o.r,
            channel(c.g) * a * (1 - o.alpha) + o.g, channel(c.b) * a * (1 - o.alpha) + o.b,
            a * (1 - o.alpha) + o.alpha
        )
    end
    return out
end
