# The LEGO world, raytraced to a video, for the walkthrough's background track.
#
# Rendered STANDALONE rather than as a second scene clip. Two scene clips in one
# project means two RayMakie screens rendering inside one editor frame on one
# submit channel, and that races: a dispatch gets emitted into a recording that
# was already handed over (`hold!: no recording is open on this channel`), and
# the integrator's plans get nulled under a render (`runsample!(::Nothing)`).
# Both are real Mantle/Hikari bugs and both are open.
#
# The walkthrough wants a background and a figure on two tracks, which this
# gives: a video clip underneath, the figure's scene clip over it. That is the
# shape the original project had — only the footage changes.

using Makie, RayMakie, GeometryBasics, FileIO, Colors, Printf
import FFMPEG_jll
import Makie: Point3f, Vec3f, RGBf

const WD  = "/sim/Programmieren/VideoEdit/media/lego_world"
const OUT = "/sim/Programmieren/VideoEdit/media/lego_world_bg.mp4"
const W, H, N, FPS = 1920, 1080, 180, 60

# The camera BOTH layers use. Closer than the figure clip's original
# (260, 80, 150): that was framed for a 640x1138 portrait preview holding
# nothing but the minifig, and against a 256 mm baseplate at 1920x1080 he came
# out a few pixels tall. The project's figure clip carries the same numbers.
const CAM_EYE = (118.0, 38.0, 66.0)
const CAM_AT  = (0.0, 6.0, 20.0)

part(n) = FileIO.load(joinpath(WD, "world_$(n).stl"))

"""The world, drawn through the SAME camera the figure clip uses, so the two
layers agree — eye and lookat are `lego.videoedit`'s."""
function worldfigure()
    fig = Figure(size = (W, H))
    ls = LScene(fig[1, 1]; show_axis = false)
    for (name, col) in (("plate",  RGBf(0.29, 0.61, 0.27)),
                        ("red",    RGBf(0.79, 0.11, 0.13)),
                        ("yellow", RGBf(0.97, 0.76, 0.11)),
                        ("blue",   RGBf(0.00, 0.33, 0.61)),
                        ("white",  RGBf(0.95, 0.95, 0.94)),
                        ("trunk",  RGBf(0.35, 0.22, 0.10)),
                        ("leaves", RGBf(0.13, 0.44, 0.23)))
        mesh!(ls, part(name); color = col)
    end
    Makie.update_cam!(ls.scene, Vec3f(CAM_EYE...), Vec3f(CAM_AT...), Vec3f(0, 0, 1))
    return fig, ls
end

function renderbg(; out = OUT, n = N, fps = FPS, drift = 14.0)
    fig, ls = worldfigure()
    dir = mktempdir()
    for i in 1:n
        # A slow orbit, so the background is not a frozen still under a walking
        # figure. `drift` degrees over the whole clip.
        θ = deg2rad(drift * (i - 1) / max(n - 1, 1) - drift / 2)
        ex, ey = CAM_EYE[1], CAM_EYE[2]
        e = Vec3f(ex*cos(θ) - ey*sin(θ), ex*sin(θ) + ey*cos(θ), CAM_EYE[3])
        Makie.update_cam!(ls.scene, e, Vec3f(CAM_AT...), Vec3f(0, 0, 1))
        img = Makie.colorbuffer(fig; backend = RayMakie, update = false)
        FileIO.save(joinpath(dir, @sprintf("%04d.png", i)), img)
        i % 30 == 0 && @info "background" frame = i of = n
    end
    run(`$(FFMPEG_jll.ffmpeg()) -y -framerate $fps -i $(joinpath(dir, "%04d.png"))
         -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p $out`)
    @info "background rendered" out size = round(filesize(out) / 1024^2, digits = 2)
    return out
end
