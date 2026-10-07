isdefined(Main, :EditingWalkthroughActions) || include("editing_helpers.jl")
module SceneEditingWalkthroughTests
    using Test, RayMakie
    import VideoEditor as VE
    const M = VE.Makie
    const FI = Main.FakeInteraction
    using Main.EditingWalkthroughActions

    function axispoint(axis, point)
        q = M.project(axis.scene, M.Point3f(point[1], point[2], 0))
        return M.Point2f(q[1], q[2]) + M.Point2f(axis.scene.viewport[].origin)
    end

    function objectpixel(clip, object)
        frame, data = VE.onthread(VE.renderthread(RayMakie)) do
            lock(VE.SCENELOCK) do
                frame = RayMakie.raster_pick_data(clip.source.live.screen)
                frame, reshape(Array(VE.Mantle.storage(frame.pixels)), frame.size...)
            end
        end
        ids = Set(i for (i, p) in enumerate(frame.plots) if VE.sceneobjectowner(clip.source, p) === object)
        found = findall(v -> Int(v[1] >> 1) in ids, data)
        isempty(found) && error("fixture object $object isn't visible")
        centre = sum(M.Vec2f(Tuple(i)) for i in found) / length(found)
        i = found[argmin([sum(abs2, M.Vec2f(Tuple(i)) - centre) for i in found])]
        x, y = Tuple(i);w, h = frame.size
        return M.Point2f((x - 0.5) / frame.ppu + 0.5, clip.source.height - (h - y + 0.5) / frame.ppu + 0.5)
    end

    @testset "pick actors and edit the shared raster scene with real gestures" begin
        mktempdir() do dir
            out = get(ENV, "VIDEOEDITOR_QA_OUTPUT", dir);mkpath(out)
            file = joinpath(dir, "scene.jl")
            write(
                file, """
                using Makie
                function buildscene(canvas,args)
                    sc=Scene(;size=canvas,camera=cam3d!,backgroundcolor=:black)
                    body=mesh!(sc,Rect3f(Vec3f(-.5),Vec3f(1));name=:actor,color=:orange)
                    part=mesh!(sc,Rect3f(Vec3f(.5,-.2,-.2),Vec3f(.4));name=:arm,color=:orange)
                    other=mesh!(sc,Rect3f(Vec3f(-.5),Vec3f(1));name=:other,color=:dodgerblue)
                    translate!(other,2.5,0,0)
                    camera=(frame,fps)->(;eye=Vec3f(5+sin(frame/20),-8,5),lookat=Vec3f(1,0,0),up=Vec3f(0,0,1))
                    update! = (frame,fps)->begin
                        c=camera(frame,fps);update_cam!(sc,c.eye,c.lookat,c.up)
                    end
                    controls=[(;name=:acting,object=:actor,label="Actor · acting",
                        sections=[(label="Actor · face",fields=[:smile])],
                        sample=(frame,fps)->(;smile=.5),apply! = (changes,frame,fps)->nothing)]
                    return (;scene=sc,update!,controls,camera,activeparams=(lo,hi,fps)->Symbol[])
                end
                """
            )
            root = VE.programscene(
                file; objects = [
                    Dict(
                        "label" => "Actor", "plots" => ["actor", "arm"],
                        "attributes" => ["color"]
                    ),
                ]
            )
            fixture = VE.sceneclip(VE.buildscene(root); build = root, frames = 96, canvas = (320, 240), framerate = 24)
            fixture.source.backend = :RayMakie
            fixture.source.screenopts = Dict(:rasterize => true, :shadows => false, :device => "Mantle.defaultbackend()")
            sequence = VE.Sequence([fixture], 24);sequence.canvas = (320, 240)
            speech = VE.Narration("Linked speech survives transform Undo", 0.25; anchor = fixture.source)
            append!(speech.samples, fill(0.05f0, 48000));speech.rate = 48000
            push!(sequence.narration, speech)
            project = joinpath(dir, "scene.videoedit");VE.saveproject(project, sequence)
            VE.usebackend!(RayMakie);VE.GLMakie.activate!(visible = false)
            p = VE.Player(
                project; analysisbackend = VE.Mantle.defaultbackend(), gpupreview = false,
                audiopreview = false, previewscale = 0.5
            )
            try
                # A fresh runtime compiles the first raster scene through normal
                # package use; warm interaction timings do not include this wait.
                ready = timedwait(
                    () -> begin
                        source = first(p.sequence.clips).source
                        source.live !== nothing && RayMakie.raster_pick_data(source.live.screen) !== nothing &&
                            get(p.fxwidgets, :publishedframe, -1) == 0
                    end, 300; pollint = 0.05
                )
                @test ready == :ok
                ready == :ok || error("initial raster preview did not become ready")
                VE.GLMakie.stop_renderloop!(p.screen; close_after_renderloop = false)
                M.disconnect!(p.screen, M.mouse_position);M.events(p.fig).hasfocus[] = false
                sleep(0.5)
                clip() = first(p.sequence.clips)
                fx() = VE.findslot(clip(), :scene)
                sections() = p.fxwidgets[Symbol(:fxsections_, fx().id)]
                live = clip().source.live
                originalcamera = copy(VE.targetscene(live.target).camera.projectionview[])
                originalplots = VE.sceneobjectplots(clip().source, :actor)
                actions = [
                    click(() -> axispoint(p.previewaxis, objectpixel(clip(), :actor)));
                    FI.Lazy(
                        _ -> begin
                            @test VE.sceneselection(p)[].object === :actor
                            @test VE.selectedscenesections(p, clip()) == Set([:actor, :acting])
                            @test Set(sec.name for (sec, card) in zip(sections().sections, sections().cards) if card.visible[]) == Set([:actor, :acting])
                            @test any(sec -> sec.label == "Actor · face", sections().sections)
                            @test sections().onlyactive[]
                            @test all(card.open[] for (sec, card) in zip(sections().sections, sections().cards) if card.visible[])
                            FI.Wait(0)
                        end
                    );click(() -> center(p.fxwidgets[:sceneviewbutton]));FI.Wait(1.0)
                ]
                recordactions(p, out, "pick_actor", actions)
                view = p.fxwidgets[:sceneview]
                screen = M.getscreen(view.fig.scene)
                VE.GLMakie.stop_renderloop!(screen; close_after_renderloop = false)
                M.disconnect!(screen, M.mouse_position);M.events(view.fig).hasfocus[] = false
                while view.busy
                    sleep(0.01)
                end
                @test view.frame !== nothing
                @test previewmatches(p, p.previewaxis, p.frame[])
                @test view.clip.source.live === live
                @test VE.sceneobjectplots(clip().source, :actor) == originalplots
                @test VE.targetscene(live.target).camera.projectionview[] ≈ originalcamera
                @test length(view.handles[]) == 3
                # Selection outlines and handles deliberately cover the fixture's
                # small coloured objects. Compare the image without that overlay,
                # then check projection/picking through actual object clicks.
                recordactions(
                    view, out, "show_all_objects", [
                        click(() -> center(p.fxwidgets[:sceneclearbutton]));
                    ]
                )
                @test VE.sceneselection(p)[] === nothing
                @test previewmatches(view, view.axis, view.image[])
                recordactions(p, out, "pick_static_objects_and_face_from_inspector", [
                    click(() -> center(sections().picker));
                    click(() -> menurow(sections().picker, "other"));
                    FI.Lazy(_ -> begin
                        @test sections().onlyactive[]
                        @test VE.sceneselection(p)[].object === :other
                        @test any(sec.name === :other && card.visible[] for (sec,card) in zip(sections().sections,sections().cards))
                        @test VE.param(fx(), Symbol("other.translation[1]")).view !== nothing
                        FI.Wait(0)
                    end);
                    click(() -> center(sections().picker));
                    click(() -> menurow(sections().picker, "actor · face"));
                    FI.Lazy(_ -> begin
                        @test VE.sceneselection(p)[].object === :actor
                        @test [sec.label for (sec,card) in zip(sections().sections,sections().cards) if card.visible[]] == ["Actor · face"]
                        @test VE.param(fx(), Symbol("acting.smile")).view !== nothing
                        FI.Wait(0)
                    end)
                ])
                recordactions(
                    view, out, "pick_objects_in_scene_view", [
                        click(() -> axispoint(view.axis, VE.viewproject(view, M.Point3f(2.5, 0, 0))));
                        FI.Lazy(
                            _ -> begin
                                @test VE.sceneselection(p)[].object === :other
                                @test VE.selectedscenesections(p, clip()) == Set([:other])
                                FI.Wait(0)
                            end
                        );
                        click(() -> axispoint(view.axis, VE.viewproject(view, M.Point3f(0))));
                        FI.Lazy(
                            _ -> begin
                                @test VE.sceneselection(p)[].object === :actor
                                FI.Wait(0)
                            end
                        )
                    ]
                )
                before = VE.scenevalue(clip().source, Symbol("actor.translation[1]"))
                beforearm = originalplots[2].transformation.translation[][1]
                undo = length(p.undostack)
                point() = axispoint(view.axis, view.handles[][1])
                recordactions(
                    view, out, "move_actor_xyz", [
                        FI.Lazy(_ -> FI.MouseTo(point(), 0.05)), FI.LeftDown(),
                        FI.Lazy(_ -> FI.MouseTo(point() + M.Point2f(40, 0), 0.4)), FI.LeftUp(), FI.Wait(0.75),
                    ]
                )
                @test VE.scenevalue(clip().source, Symbol("actor.translation[1]")) != before
                @test originalplots[2].transformation.translation[][1] != beforearm
                @test length(p.undostack) == undo + 1
                @test VE.targetscene(live.target).camera.projectionview[] ≈ originalcamera
                recordactions(view, out, "undo_actor_move", chord(M.Keyboard.z))
                @test VE.scenevalue(clip().source, Symbol("actor.translation[1]")) ≈ before
                @test VE.sceneobjectplots(clip().source, :actor) == originalplots
                for (label, field) in ((:rotation, :rotation), (:scale, :scale))
                    mode() = p.fxwidgets[:sceneviewmode]
                    path = Symbol("actor.", field, "[1]")
                    initialvalue = VE.scenevalue(clip().source, path)
                    recordactions(
                        view, out, "$(field)_actor_xyz", [
                            click(() -> center(mode()));click(() -> menurow(mode(), label));
                            FI.Lazy(_ -> FI.MouseTo(point(), 0.05));FI.LeftDown();
                            FI.Lazy(_ -> FI.MouseTo(point() + M.Point2f(35, 0), 0.3));FI.LeftUp();FI.Wait(0.5)
                        ]
                    )
                    @test view.mode === field
                    @test VE.scenevalue(clip().source, path) != initialvalue
                    recordactions(view, out, "undo_$(field)", chord(M.Keyboard.z))
                    @test VE.scenevalue(clip().source, path) ≈ initialvalue
                end
                workingcamera = copy(view.camera.camera.projectionview[])
                orbitpoint = () -> center(view.axis)
                recordactions(
                    view, out, "orbit_scene_camera", [
                        FI.Lazy(_ -> FI.MouseTo(orbitpoint(), 0.05)), FI.MouseDown(M.Mouse.right),
                        FI.Lazy(_ -> FI.MouseTo(orbitpoint() + M.Point2f(50, 25), 0.3)), FI.MouseUp(M.Mouse.right), FI.Wait(0.5),
                    ]
                )
                @test view.camera.camera.projectionview[] != workingcamera
                @test VE.targetscene(live.target).camera.projectionview[] ≈ originalcamera
                workingcamera = copy(view.camera.camera.projectionview[])
                recordactions(
                    view, out, "pan_and_zoom_scene_view", [
                        FI.Lazy(_ -> FI.MouseTo(orbitpoint(), 0.05));FI.MouseDown(M.Mouse.middle);
                        FI.Lazy(_ -> FI.MouseTo(orbitpoint() + M.Point2f(30, 15), 0.3));FI.MouseUp(M.Mouse.middle);
                        FI.Scroll((0, 1); duration = 0.25);FI.Wait(0.5)
                    ]
                )
                @test view.camera.camera.projectionview[] != workingcamera
                @test VE.targetscene(live.target).camera.projectionview[] ≈ originalcamera
                recordactions(view, out, "match_film_camera", click(() -> center(button(view, "Match film view"))))
                @test M.cameracontrols(view.camera).eyeposition[] ≈ clip().source.camera.eye
                recordactions(
                    view, out, "camera_path_keys", [
                        click(() -> center(button((; fig = view.fig), "Camera path")));
                        FI.Lazy(_ -> begin
                            VE.scrollinspectorto!(p, VE.param(fx(), Symbol("camera.eye[1]")).view.kf)
                            FI.Wait(0.2)
                        end);
                        click(() -> center(VE.param(fx(), Symbol("camera.eye[1]")).view.kf));FI.Wait(0.75)
                    ]
                )
                @test VE.sceneselection(p)[].object === :camera
                @test length(view.pathframes) >= 2
                eye = VE.param(fx(), Symbol("camera.eye[1]"))
                @test maximum(abs(VE.valueat(eye, f) - (5 + sin(f / 20))) for f in 0:95) < 0.001
                keypoint() = axispoint(view.axis, first(view.pathhandles[]))
                initial = VE.valueat(VE.param(fx(), Symbol("camera.eye[1]")), first(view.pathframes))
                recordactions(
                    view, out, "drag_camera_path", [
                        FI.Lazy(_ -> FI.MouseTo(keypoint(), 0.05)), FI.LeftDown(),
                        FI.Lazy(_ -> FI.MouseTo(keypoint() + M.Point2f(35, 20), 0.4)), FI.LeftUp(), FI.Wait(0.75),
                    ]
                )
                @test VE.valueat(VE.param(fx(), Symbol("camera.eye[1]")), first(view.pathframes)) != initial
                finalframe = first(view.pathframes)
                recordactions(view, out, "undo_camera_path", chord(M.Keyboard.z))
                @test VE.valueat(VE.param(fx(), Symbol("camera.eye[1]")), finalframe) ≈ initial
                recordactions(
                    view, out, "redo_camera_path", [
                        FI.KeyDown(M.Keyboard.left_control), FI.KeyDown(M.Keyboard.left_shift),
                        FI.KeyPress(M.Keyboard.z), FI.KeyUp(M.Keyboard.left_shift), FI.KeyUp(M.Keyboard.left_control), FI.Wait(0.25),
                    ]
                )
                @test VE.valueat(VE.param(fx(), Symbol("camera.eye[1]")), finalframe) != initial
                saved = joinpath(out, "sceneediting_saved.videoedit");VE.saveproject(saved, p.sequence)
                reopened = VE.loadproject(saved)
                @test VE.valueat(VE.param(VE.findslot(first(reopened.clips), :scene), Symbol("camera.eye[1]")), first(view.pathframes)) ≈
                    VE.valueat(VE.param(fx(), Symbol("camera.eye[1]")), first(view.pathframes))
                # Empty curve bands reveal controls; Ctrl/Shift still address clips.
                VE.clearselectedkey!(p)
                function emptybandpoint()
                    lo, hi = eye.view.lane.band[]
                    for f in 10:85, fraction in (.5, .25, .75)
                        t = VE.timelineframe(clip(), f) / p.sequence.framerate
                        y = lo + fraction * (hi - lo)
                        VE.nearestmarker(p, t, y) === nothing || continue
                        VE.trackedgeat(p.sequence, y, VE.ntracks(p.sequence); grab=VE.grabzone(p.timeline)) === nothing || continue
                        return axispoint(p.timeline.axis, M.Point2f(t, y))
                    end
                    error("no empty parameter band")
                end
                beforestart = clip().start
                target = emptybandpoint()
                recordactions(p, out, "ctrl_move_clip_over_curve", [
                    FI.KeyDown(M.Keyboard.left_control), FI.MouseTo(target, .05), FI.LeftDown(),
                    FI.Lazy(_ -> begin
                        @test p.timeline.dragclip !== nothing
                        FI.MouseTo(target + M.Point2f(65, 0), .3)
                    end), FI.LeftUp(), FI.KeyUp(M.Keyboard.left_control), FI.Wait(.5)
                ])
                @test clip().start > beforestart
                recordactions(p, out, "undo_ctrl_move_clip", chord(M.Keyboard.z))
                @test clip().start == beforestart
                sleep(.5) # A separate click, not the double-click solo gesture.
                recordactions(p, out, "shift_select_clip_over_curve", [
                    FI.KeyDown(M.Keyboard.left_shift); click(emptybandpoint); FI.KeyUp(M.Keyboard.left_shift)
                ])
                @test clip().id in p.timeline.selection[]
                recordactions(p, out, "clear_scene_selection", click(() -> center(p.fxwidgets[:sceneclearbutton])))
                @test VE.sceneselection(p)[] === nothing
                @test view.fig === p.fig
                for mode in (:scene, :film, :both)
                    recordactions(p, out, "preview_layout_$(mode)", [
                        click(() -> center(p.fxwidgets[:previewlayout]));
                        click(() -> menurow(p.fxwidgets[:previewlayout], mode));
                    ])
                    @test p.fxwidgets[:sceneview] === view && !view.closed
                    @test view.camera.visible[] == (mode !== :film)
                    @test p.previewaxis.blockscene.visible[] == (mode !== :scene)
                    @test view.clip.source.live === live
                end
            finally
                sources = unique(c.source for c in p.sequence.clips if c.source isa VE.SceneSource)
                screens = [source.live.screen for source in sources if source.live !== nothing]
                close(p)
                @test all(source -> source.live === nothing && source.pending === nothing, sources)
                @test all(screen -> isempty(screen.frame_plans) && screen.gfx_atlas_hook === nothing, screens)
                @test isempty(p.engine.layers) && isempty(p.engine.compositions)
            end
        end
    end
end
