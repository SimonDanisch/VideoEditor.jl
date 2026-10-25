using Makie
import Hikari
function buildscene(canvas, args)
    scene = Scene(;
        size = Tuple(Int.(canvas)), backgroundcolor = :black,
        lights = [PointLight(RGBf(100, 100, 100), Vec3f(2, 3, 4))]
    )
    cam3d!(scene; center = false)
    mesh!(
        scene, Sphere(Point3f(0), 1.0f0); color = :red,
        material = Hikari.Emissive(Le = (20, 5, 2), two_sided = true)
    )
    update_cam!(scene, Vec3f(0, -4, 1.5), Vec3f(0, 0, 0))
    return (scene = scene,)
end
