# Raytraced special effects ON a video: RayMakie objects composited over frames,
# lit BY the frame they sit on.
#
# The point is the environment light. Each frame of the clip becomes the scene's
# environment map, so the chrome sphere reflects the actual wall, window and
# nest box it is floating in front of, and the glass sphere refracts them — and
# both change as the shot changes. That is the thing a raytracer buys you here
# and a compositing filter cannot fake.
#
# Three things that were NOT obvious, each found by rendering it wrong first:
#
#  * `Hikari.Silver(roughness=0.02)` is not a mirror — it came out milky. The
#    perfect mirror is `Conductor(eta=0.2, k=3.0, roughness=0)`, as in RayMakie's
#    own env_light_debug example.
#  * There is no usable alpha: `colorbuffer` returns RGBA but alpha is 1.0
#    everywhere, measured across the whole image. The matte comes from a second
#    SILHOUETTE pass instead — pitch-black material on the default light
#    background — which is also why it can run at 2 spp.
#  * Exposure has to come down hard (0.35, not 1.0). A video frame used as an
#    env map is bright enough that the default blows every highlight to white.
#
# Cost, measured: 0.5 s beauty (540x960, 24 spp) + 0.2 s matte per frame. The
# first render of a session costs ~23 s — that is shader compilation, not
# rendering, and it does not repeat.
#
# Run on :1.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using RayMakie, Hikari, Makie, GLMakie, GeometryBasics, Colors, FileIO
using FixedPointNumbers: N0f8
import VideoIO

source  = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
outpath = "/sim/Programmieren/VideoEdit/media/rt_vfx.mp4"
skip    = 90       # frames to run into the clip before starting
nframes = 150      # frames to render
outsize = (960, 540)   # (H, W) of the output
spp     = 32
exposure = 0.35f0
envres  = 768      # square env map, see `envsquare`

mirror = Hikari.Conductor(eta = Hikari.RGBSpectrum(0.2f0), k = Hikari.RGBSpectrum(3.0f0),
                          roughness = 0.0f0)
glass  = Hikari.Dielectric(Kt = (1, 1, 1), index = 1.5)
gold   = Hikari.Conductor(eta = Hikari.RGBSpectrum(0.15f0), k = Hikari.RGBSpectrum(3.5f0),
                          roughness = 0.08f0)

lum(px) = Float32(0.2126 * red(px) + 0.7152 * green(px) + 0.0722 * blue(px))

"Nearest-neighbour resample — no extra package, and the matte wants hard edges."
function resample(img, (H, W))
    sh, sw = size(img)
    return [img[clamp(1 + round(Int, (i - 1) * (sh - 1) / max(H - 1, 1)), 1, sh),
                clamp(1 + round(Int, (j - 1) * (sw - 1) / max(W - 1, 1)), 1, sw)]
            for i in 1:H, j in 1:W]
end

"""
A video frame as a square environment map.

pbrt-v4 — and Hikari with it — maps equal-area and expects a SQUARE image; it
warns on anything else. Feeding the 1080x1920 frame straight in squashes the
axes differently and the reflection smears.
"""
envsquare(frame; n = envres) = resample(frame, (n, n))

"Where the two spheres are at timeline fraction `u` — a slow orbit plus a bob."
function spheres(u)
    θ = 2π * u
    a = Point3f(1.15cos(θ), 1.15sin(θ), 0.25sin(2θ))
    b = Point3f(-1.15cos(θ), -1.15sin(θ), -0.25sin(2θ))
    return [(GeometryBasics.normal_mesh(Sphere(a, 0.5f0)), mirror),
            (GeometryBasics.normal_mesh(Sphere(b, 0.42f0)), glass),
            (GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0.9sin(θ + 1)), 0.3f0)), gold)]
end

const CAM = (Vec3f(0, -6, 1.2), Vec3f(0, 0, 0), Vec3f(0, 0, 1))

function beautypass(envimg, gm; size, spp, exposure)
    s = Scene(; size = size, lights = Makie.AbstractLight[])
    cam3d!(s)
    update_cam!(s, CAM...)
    for (g, m) in gm
        mesh!(s, g; material = m)
    end
    push_light!(s, Makie.EnvironmentLight(1.0f0, envimg))
    return colorbuffer(display(s; backend = RayMakie, visible = false, exposure = exposure,
                               integrator = RayMakie.VolPath(samples = spp, max_depth = 6)))
end

"The silhouette, as a matte: black material on the lit default background."
function maskpass(gm; size)
    s = Scene(; size = size, lights = [Makie.AmbientLight(RGBf(1, 1, 1))])
    cam3d!(s)
    update_cam!(s, CAM...)
    for (g, _) in gm
        mesh!(s, g; material = Hikari.Diffuse(Kd = (0, 0, 0)))
    end
    return colorbuffer(display(s; backend = RayMakie, visible = false,
                               integrator = RayMakie.VolPath(samples = 2, max_depth = 1)))
end

function composite(videoframe, beauty, matte)
    H, W = size(videoframe)
    b = resample(beauty, (H, W))
    m = resample(matte, (H, W))
    bg = maximum(lum, m)
    # EXPLIZIT N0f8, nicht `similar`: der Frame ist RGB{Float32}, und ein
    # `similar` davon nimmt jedes zurückkonvertierte Pixel wieder als Float32 an —
    # VideoIO encodiert RGB{Float32} nicht ("not yet supported").
    out = Matrix{RGB{N0f8}}(undef, H, W)
    @inbounds for i in eachindex(out)
        α = clamp(1 - lum(m[i]) / max(bg, 1.0f-6), 0, 1)
        v, r = videoframe[i], b[i]
        out[i] = RGB{N0f8}(clamp((1 - α) * red(v) + α * red(r), 0, 1),
                           clamp((1 - α) * green(v) + α * green(r), 0, 1),
                           clamp((1 - α) * blue(v) + α * blue(r), 0, 1))
    end
    return out
end

vr = VideoIO.openvideo(source)
for _ in 1:skip
    VideoIO.read(vr)
end

writer = nothing
t0 = time()
for k in 1:nframes
    raw = VideoIO.read(vr)
    frame = RGB{Float32}.(resample(raw, outsize))
    gm = spheres((k - 1) / nframes)
    beauty = RGB{Float32}.(beautypass(envsquare(frame), gm;
                                      size = (outsize[2], outsize[1]), spp, exposure))
    matte = RGB{Float32}.(maskpass(gm; size = (outsize[2], outsize[1])))
    out = composite(frame, beauty, matte)
    global writer
    writer === nothing && (writer = VideoIO.open_video_out(outpath, out;
                                                           framerate = 30, target_pix_fmt = VideoIO.AV_PIX_FMT_YUV420P))
    VideoIO.write(writer, out)
    k % 25 == 0 && println("  ", k, "/", nframes, "  ", round(time() - t0; digits = 1), " s")
end
VideoIO.close_video_out!(writer)   # `close` gibt es für den Writer nicht
close(vr)
println("fertig: ", outpath, "  ", round(time() - t0; digits = 1), " s für ", nframes, " Frames")
