using Makie
import Hikari
function buildscene(canvas, args)
    sc = Scene(;
        size = Tuple(Int.(canvas)), backgroundcolor = :black,
        lights = [PointLight(RGBf(100), Vec3f(2, -3, 4))]
    )
    cam3d!(sc; center = false)
    if get(args, "cutout", false)
        mesh!(
            sc, Sphere(Point3f(0, -1, 0), 1.7f0);
            material = Hikari.Diffuse(Kd = Hikari.RGBSpectrum(1.0f0, 1.0f0, 1.0f0, 0.0f0))
        )
    end
    texture = Hikari.Texture(fill(Hikari.RGBSpectrum(0.8f0, 0.1f0, 0.05f0), 2, 2))
    mesh!(
        sc, Sphere(Point3f(-0.8, 0, 0.25), 0.65f0);
        material = Hikari.Diffuse(Kd = texture)
    )
    mixed = Hikari.MixMaterialSpec(
        Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.1f0, 0.8f0, 0.1f0)),
        Hikari.CoatedDiffuse(reflectance = Hikari.RGBSpectrum(0.1f0, 0.1f0, 0.8f0)), 0.25f0
    )
    mesh!(sc, Sphere(Point3f(0.8, 0, -0.1), 0.65f0); material = mixed)
    update_cam!(sc, Vec3f(0, -4, 1.5), Vec3f(0))
    return (scene = sc,)
end
