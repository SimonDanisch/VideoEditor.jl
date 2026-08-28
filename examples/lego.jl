# The lego figure as a VideoEditor project: a Makie scene as DATA, animated by
# keyframes on named paths, previewed on GLMakie and baked on RayMakie.
#
# Run the blocks one at a time (VS Code: cursor inside a `begin ... end`,
# Shift+Enter). Every parameter below is a plain binding you can re-assign and
# re-run — nothing here is const and nothing reads ARGS.

# ---------------------------------------------------------------- 1. open it
begin
    using VideoEditor, GLMakie
    import VideoEditor as VE

    project = "/sim/Programmieren/VideoEdit/media/lego.videoedit"
    player = Player(project)            # 640x1138 @ 60 fps, cut at frame 90
end

# ---------------------------------------------------------------- 2. the scene
# The overlay carries the scene; the curves carry the animation. Both came out
# of the project file — `spec` is a SceneSpec, not a String.
begin
    overlay = player.sequence.overlays[1]
    spec = overlay.settings.spec

    println("parts:  ", join((String(p.name) for p in spec.parts), ", "))
    println("curves: ", join(sort!(String.(collect(keys(overlay.animations)))), ", "))
    # every path is addressable BY NAME — reordering the parts cannot re-aim a curve
    println("arm_left at frame 10: ", VE.overlaystate(overlay, 10;
                                        framerate = player.sequence.framerate)[Symbol("arm_left.angle")])
end

# ---------------------------------------------------------------- 3. edit it
# A keyframe is one call. The preview redraws on the next seek, because
# `publishframe!` runs the overlay pass on every present path.
begin
    VE.setoverlaykey!(overlay, Symbol("arm_left.angle"), 10, 1.2)
    VE.unbake!(overlay)                 # the bake described the OLD curve
    VE.seek!(player, 10)
end

# Any path in the scene works the same way — a limb, the whole figure's yaw,
# or the camera. These are the ones this project already animates:
#
#   "arm_left.angle" "arm_right.angle" "leg_left.angle" "leg_right.angle"
#   "torso.angle"            — yaw of the WHOLE figure (its axis is z)
#   "torso.offset[2]"        — walks him across frame
#   "torso.offset[3]"        — lift + the bob, twice per stride
#   "camera.eye[1]"          — …and the camera is keyframed like anything else

# ---------------------------------------------------------------- 4. bake it
# RayMakie has to be loaded and registered before a scene may name it, and it
# needs one method VideoEditor cannot provide itself: which pixels the render
# actually covered. VideoEditor does not depend on RayMakie, so the method lives
# here — `coverage` dispatches exactly so a renderer can bring its own.
begin
    using RayMakie
    VE.usebackend!(RayMakie)

    function VE.coverage(screen::RayMakie.Screen)
        isempty(screen.scene_states) && return nothing
        film = first(values(screen.scene_states)).film
        # misses carry 1f30 — a large FINITE value, not Inf — and the film is
        # stored (H, W) while the editor works in (W, H)
        return permutedims(Array(film.depth) .< 1.0f29, (2, 1))
    end
end

# The bake writes each frame to `<project>.bakes/` as it finishes, so minutes of
# raytracing survive a crash; a partial bake reloads and the missing frames draw
# live. ~2 s/frame at this canvas, so ~6 minutes for the 180.
begin
    n = VE.bake!(overlay, VE.canvassize(player.sequence);
                 framerate = player.sequence.framerate,
                 into = project,
                 progress = (done, total) -> done % 20 == 0 && @info "baked $done/$total")
    saveproject(project, player.sequence)
    @info "baked $n frames — reopening the project restores them"
end

# ---------------------------------------------------------------- 5. export it
begin
    VE.exportvideo("/sim/Programmieren/VideoEdit/media/lego_over_footage.mp4",
                   player.sequence)
end
