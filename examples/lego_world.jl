# A LEGO world for the walkthrough's BACKGROUND scene, as meshes on disk.
#
# The foreground clip is the minifig rig that Makie ships as assets. Nothing
# ships a brick, so this generates them — and generating is the right answer
# rather than a workaround: a brick IS a box plus a grid of studs, which is
# eleven lines, and a spec file can only name a part by path or asset name
# (`partmesh` in `VideoEditor/src/overlays.jl`).
#
# UNITS ARE MILLIMETRES, because the figure's are. Measured, not assumed: the
# torso asset is 8.0 x 15.4 x 18.6 and the whole figure spans about 40 from the
# soles to the top of the head, which is a real minifig. So the real brick
# dimensions apply and the world lands at the figure's scale with no fudge
# factor — a 2x4 brick next to him is a 2x4 brick next to him.
#
#   julia> include("lego_world.jl"); legoworld()

using GeometryBasics, FileIO, LinearAlgebra
using GeometryBasics: Point3f, Vec3f, Rect3f, Cylinder, normal_mesh

"The System: LEGO's own dimensions, in millimetres."
const PITCH   = 8.0f0     # stud to stud
const BRICK_H = 9.6f0     # a brick
const PLATE_H = 3.2f0     # a plate — a third of a brick
const STUD_D  = 4.8f0
const STUD_H  = 1.8f0

"""
    brick(nx, ny; height = BRICK_H, studs = true) -> Mesh

A `nx` x `ny` brick, centred on its footprint, sitting on z = 0.

Centred rather than corner-anchored because a rig part rotates about its own
origin, and a brick a keyframe spins should spin about itself.
"""
function brick(nx::Integer, ny::Integer; height = BRICK_H, studs::Bool = true)
    w, d = nx * PITCH, ny * PITCH
    parts = [normal_mesh(Rect3f(Point3f(-w / 2, -d / 2, 0), Vec3f(w, d, height)))]
    studs && for i in 1:nx, j in 1:ny
        # Stud centres sit at the middle of each stud cell, not on the corners.
        cx = -w / 2 + (i - 0.5f0) * PITCH
        cy = -d / 2 + (j - 0.5f0) * PITCH
        push!(parts, normal_mesh(Cylinder(Point3f(cx, cy, height),
                                          Point3f(cx, cy, height + STUD_H),
                                          STUD_D / 2)))
    end
    # FLATTENED. `merge` leaves `normal` as a `FaceView` carrying its own face
    # indices, and STL's writer indexes the normals with the POSITION faces —
    # which is an out-of-bounds read the moment the two disagree, as they do for
    # every mesh with more than one primitive in it.
    return GeometryBasics.expand_faceviews(merge(parts))
end

"A baseplate: one plate-thin slab, studded, big enough to stand a scene on."
baseplate(nx::Integer, ny::Integer) = brick(nx, ny; height = PLATE_H)

"""
    legoworld(dir) -> Dict{String, String}

Write the world's parts as STL and return `name => path`.

One file per DISTINCT shape, not per placed object: placement is the rig's job
(each part carries its own transform and colour), so a wall of six 2x4s is six
parts naming one file.
"""
function legoworld(dir::AbstractString = joinpath(@__DIR__, "..", "..", "..",
                                                  "media", "lego_world"))
    dir = normpath(dir)
    mkpath(dir)
    shapes = Dict("baseplate" => baseplate(24, 24),
                  "brick_2x4" => brick(2, 4),
                  "brick_2x2" => brick(2, 2),
                  "brick_1x1" => brick(1, 1),
                  "plate_4x4" => brick(4, 4; height = PLATE_H),
                  "tile_2x2"  => brick(2, 2; height = PLATE_H, studs = false))
    out = Dict{String, String}()
    for (name, m) in shapes
        path = joinpath(dir, name * ".stl")
        # BINARY, named explicitly: FileIO reads `.stl` as ASCII by default, and
        # a 24x24 baseplate is 576 studs — tens of megabytes of text for geometry
        # that is 50 bytes a triangle in binary.
        save(File{format"STL_BINARY"}(path), m)
        out[name] = path
    end
    return out
end

# ── the background scene ────────────────────────────────────────────────────
#
# Placement is BAKED INTO THE GEOMETRY, one mesh per colour.
#
# A rig part's `origin` is its rotation PIVOT, not its position — `rigscene`
# translates by it and then subtracts it from the vertices again, so the mesh
# stays where it was. What does move a part is its `offset`, which is an
# animatable parameter; using it for static placement would mint three curves per
# brick and bury the figure's own seven in a list of hundreds. So the world is
# seven parts, one per colour, each a merged mesh of every brick of that colour.
# Each is still one keyframable object — "lift all the red bricks" is one curve.

"`m` translated by `p`, as a new mesh."
at(m, p) = GeometryBasics.Mesh(GeometryBasics.coordinates(m) .+ Ref(Point3f(p)),
                               GeometryBasics.faces(m); normal = m.normal)

"""
    worldmeshes() -> Dict{String, Mesh}

The background world, grouped by colour.

Laid out around the figure's walk, which runs the y axis from -60 to +60 at
x = 0 (`torso.offset[2]` in the project). Nothing is placed within ~25 of that
line, so he walks through open ground rather than through a wall.
"""
function worldmeshes()
    plate = baseplate(32, 26)
    b24, b22, b11 = brick(2, 4), brick(2, 2), brick(1, 1)
    groups = Dict("plate"  => [at(plate, (0, 0, 0))],
                  "red"    => Any[], "yellow" => Any[], "blue" => Any[],
                  "white"  => Any[], "trunk"  => Any[], "leaves" => Any[])
    # A wall, off to the figure's left, built of alternating courses.
    for (k, c) in enumerate(("red", "yellow", "red", "blue"))
        push!(groups[c], at(b24, (-58, -18 + 16 * ((k - 1) % 2), 5 + (k - 1) * BRICK_H)))
    end
    for (k, c) in enumerate(("blue", "white", "yellow"))
        push!(groups[c], at(b24, (-58, 14 - 16 * ((k - 1) % 2), 5 + (k - 1) * BRICK_H)))
    end
    # Loose bricks on the right, at a couple of heights, to catch the light.
    push!(groups["red"],   at(b22, (52, -34, 5)))
    push!(groups["blue"],  at(b22, (52, -34, 5 + BRICK_H)))
    push!(groups["white"], at(b24, (46, 28, 5)))
    push!(groups["yellow"], at(b22, (62, 6, 5)))
    # Two trees, well clear of the walk.
    for (tx, ty) in ((70, 46), (-72, 52))
        for k in 1:3
            push!(groups["trunk"], at(b11, (tx, ty, 5 + (k - 1) * BRICK_H)))
        end
        push!(groups["leaves"], at(b22, (tx - 4, ty - 4, 5 + 3 * BRICK_H)))
        push!(groups["leaves"], at(b22, (tx - 4, ty - 4, 5 + 4 * BRICK_H)))
    end
    # `[x for x in v]` and not `v`: the groups are built as `Any[]` so a colour
    # can take any brick, and `merge` dispatches on the vector's ELEMENT type.
    return Dict(k => GeometryBasics.expand_faceviews(merge([x for x in v]))
                for (k, v) in groups if !isempty(v))
end

"LEGO's own colours, as the spec writes them."
const WORLD_COLORS = Dict("plate"  => "#4A9B45FF", "red"    => "#C91C21FF",
                          "yellow" => "#F7C11CFF", "blue"   => "#00549CFF",
                          "white"  => "#F2F2F0FF", "trunk"  => "#5A3819FF",
                          "leaves" => "#22703AFF")

"""
    worldspec(dir; camera, lights) -> Dict

The background as a scene spec: write the meshes, then name them as rig parts.

`camera` and `lights` default to the figure clip's, because the two layers are
composited and a background lit from somewhere else reads as a cut-out.
"""
function worldspec(dir::AbstractString = joinpath(@__DIR__, "..", "..", "..",
                                                  "media", "lego_world");
                   camera = Dict("eye" => [260.0, 80.0, 150.0],
                                 "lookat" => [0.0, 0.0, 30.0],
                                 "up" => [0.0, 0.0, 1.0]),
                   lights = Any[Dict("type" => "ambient", "color" => [0.4, 0.4, 0.4],
                                     "position" => [0.0, 0.0, 0.0]),
                                Dict("type" => "point", "color" => [15000.0, 15000.0, 15000.0],
                                     "position" => [150.0, 100.0, 200.0])])
    dir = normpath(dir); mkpath(dir)
    parts = Any[]
    # `by = first`: sorting the pairs themselves compares the MESHES when two
    # names tie, and a mesh has no `isless`.
    for (name, m) in sort(collect(worldmeshes()); by = first)
        path = joinpath(dir, "world_" * name * ".stl")
        save(File{format"STL_BINARY"}(path), m)
        push!(parts, Dict("name" => name, "type" => "Mesh", "args" => [path],
                          "kwargs" => Dict("color" => WORLD_COLORS[name])))
    end
    return Dict("kind" => "rig",
                "rig" => Dict("parts" => parts, "lights" => lights,
                              "camera" => camera, "backend" => "RayMakie"))
end

