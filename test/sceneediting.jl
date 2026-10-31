using Test
import VideoEditor as VE

@testset "recipe preview detail and exact output" begin
    mktempdir() do dir
        script = joinpath(dir, "detail.jl")
        write(script, """
        using Makie
        function buildscene(canvas,args)
            sc=Scene(;size=canvas,camera=campixel!)
            point=scatter!(sc,[Point2f(10,10)];name=:point,color=:red)
            scale=Ref(1.0);calls=Float64[]
            preview! = value -> (scale[]=value;push!(calls,value))
            update! = (frame,fps) -> (point[1]=[Point2f(10+frame,10*scale[])])
            return (scene=sc,update! = update!,preview! = preview!)
        end
        """)
        build=VE.programscene(script)
        c=VE.sceneclip(VE.buildscene(build);build,frames=12,canvas=(48,32),framerate=24)
        src=c.source;src.bakewith=:GLMakie
        try
            VE.updatesource!(src,c,4)
            VE.sceneframe!(src,c,(48,32);pixel_scale=0.5)
            target=src.live.target
            point=VE.Makie.findplot(VE.targetscene(target),:point)
            @test point[1][] == [VE.Point2f(14,5)]
            # Changing quality at the SAME source frame must re-evaluate the
            # animation after preparing detail, even during refinement.
            VE.sceneframe!(src,c,(48,32);pixel_scale=0.25)
            @test src.live.target === target
            @test point[1][] == [VE.Point2f(14,2.5)]
            @test target.previewscale == 0.25
            VE.sceneframe!(src,c,(48,32);exact=true,pixel_scale=0.25)
            @test point[1][] == [VE.Point2f(14,10)]
            @test target.previewscale == 1
            @test isempty(src.root.args)
            @test (src.width,src.height) == (48,32)
        finally
            src.live === nothing || close(src.live.screen)
        end
    end
end

@testset "procedural scene editing" begin
    mktempdir() do dir
        script = joinpath(dir, "scene.jl")
        write(script, """
        using Makie
        function buildscene(canvas, args)
            sc = Scene(; size=canvas, camera=cam3d!, backgroundcolor=:black)
            body = mesh!(sc, Rect3f(Vec3f(-0.4), Vec3f(0.8)); color=:orange, name=:body)
            face = mesh!(sc, Rect3f(Vec3f(-0.15), Vec3f(0.3)); color=:white, name=:face)
            update! = (frame, fps) -> begin
                translate!(body, frame / 10, 0, 0)
                translate!(face, frame / 10, 0, 1)
                rotate!(body, qrotation(Vec3f(0,0,1), 0))
                rotate!(face, qrotation(Vec3f(0,0,1), 0))
                scale!(body, 1,1,1); scale!(face, 1,1,1)
                update_cam!(sc, Vec3f(4 + frame / 10, -6, 4), Vec3f(0), Vec3f(0,0,1))
            end
            return (scene=sc, update! = update!)
        end
        """)
        objects = [Dict("label" => "Actor", "plots" => ["body", "face"],
                        "attributes" => ["color", "alpha"])]
        build = VE.programscene(script; objects,
            markers = [Dict("frame"=>0,"label"=>"Opening"), Dict("frame"=>4,"label"=>"Close-up")])
        recipe = VE.SceneProgram(build)
        _, initial = VE.realize(recipe, (96, 64))
        builder = recipe.builder
        _, resized = VE.realize(recipe, (48, 32))
        @test recipe.builder === builder
        @test typeof(initial.update!.callback) === typeof(resized.update!.callback)
        @test initial.scene !== resized.scene
        clip = VE.sceneclip(VE.buildscene(build); build, frames=8,
                             canvas=(96,64), framerate=24)
        VE.seteffect!(clip, VE.ColorEffect())
        clip.look = ones(Float32, 2,2,2,3)
        VE.seteffect!(clip, VE.LookEffect())
        seq = VE.Sequence([clip], 24); seq.canvas = (96,64)
        path = joinpath(dir, "edit.videoedit")
        VE.saveproject(path, seq)
        renderer = VE.farmrenderer(VE.renderjob(path, joinpath(dir, "job")))
        try
            original = only(VE.farmframes!(renderer, [4])).png
            c = only(renderer.sequence.clips); src = c.source
            fx = VE.findslot(c, :scene)
            secs = VE.sceneparamsections(c, fx)
            @test getproperty.(secs, :label) == ["Camera", "Actor", "Light · Ambient", "Light · Directional 1"]
            @test all(s -> s.detail == "Lighting", secs[3:4])
            @test all(VE.isfollowing, fx.params)
            @test only(VE.farmframes!(renderer, [4])).png == original
            @test src.camera.eye[1] ≈ 4.4
            VE.farmframes!(renderer, [7])
            @test src.camera.eye[1] ≈ 4.7
            camera = VE.param(fx, Symbol("camera.eye[1]"))
            @test VE.valueat(camera, 7) ≈ 4.7 atol=1e-6
            position = VE.param(fx, Symbol("body.translation[1]"))
            VE.setvalue!(position, 2.0, 6)
            @test !VE.isfollowing(position)
            VE.farmframes!(renderer, [7])
            body, face = VE.sceneobjectplots(src, :body)
            @test body.transformation.translation[][1] ≈ 2
            @test face.transformation.translation[] - body.transformation.translation[] ≈ VE.Vec3f(0,0,1)
            rotation = VE.param(fx, Symbol("body.rotation[2]"))
            VE.setvalue!(rotation, 90.0, 6)
            VE.farmframes!(renderer, [7])
            @test face.transformation.translation[] - body.transformation.translation[] ≈ VE.Vec3f(1,0,0) atol=1e-5
            VE.setvalue!(camera, 8.0, 6)
            VE.farmframes!(renderer, [7])
            @test src.camera.eye[1] ≈ 8
            colour = VE.param(fx, Symbol("body.color[1]"))
            basecolour = VE.valueat(colour, 7)
            VE.setvalue!(colour, 0.2, 7)
            VE.farmframes!(renderer, [7])
            @test VE.scenevalue(src, colour.name) ≈ 0.2 atol=1e-6
            colour.input = VE.ParamInput(:copy, VE.SceneRef(colour.name))
            VE.bindinputs!(renderer.sequence)
            VE.farmframes!(renderer, [7])
            @test VE.scenevalue(src, colour.name) ≈ basecolour atol=1e-6
            VE.setkey!(position, 0, 1.0); VE.setkey!(position, 7, 3.0)
            VE.saveproject(path, renderer.sequence)
            reopened = VE.loadproject(path)
            reopenedfx = VE.findslot(only(reopened.clips), :scene)
            @test !VE.isfollowing(VE.param(reopenedfx, position.name))
            @test VE.isanimated(VE.param(reopenedfx, position.name))
            @test VE.isfollowing(VE.param(reopenedfx, Symbol("camera.lookat[1]")))
            saved = VE.farmrenderer(VE.renderjob(path, joinpath(dir, "edited-job")))
            try
                @test only(VE.farmframes!(saved, [7])).png == only(VE.farmframes!(renderer, [7])).png
            finally
                VE.closefarm!(saved)
            end
        finally
            VE.closefarm!(renderer)
        end

        # Open with a cold source: the first preview populates the inspector.
        player = VE.Player(path; analysisbackend=VE.Mantle.defaultbackend(), gpupreview=false,
                            audiopreview=false)
        try
            c = only(player.sequence.clips)
            VE.showplayhead!(player)
            VE.opendock!(player, :effects)
            fx = VE.findslot(c, :scene)
            @test length(fx.params) > 9
            @test fx.card.layoutobservables.autosize[][2] !== nothing
            @test VE.Makie.GridLayoutBase.determinedirsize(player.fxpanel.scroll.layout,
                                                         VE.Makie.GridLayoutBase.Row()) !== nothing
            look = VE.findslot(c, :look)
            @test all(b -> !(b isa VE.Makie.Button) ||
                           b.layoutobservables.computedbbox[].widths[1] > 200, look.tool.controls)
            fx.card.open = true
            nextcard = VE.findslot(c, :color).card
            a, b = fx.card.layoutobservables.computedbbox[], nextcard.layoutobservables.computedbbox[]
            @test b.origin[2] + b.widths[2] <= a.origin[2] + 1
            camera = VE.param(fx, Symbol("camera.eye[1]"))
            @test camera.view.control isa VE.Makie.Textbox
            @test camera.view.control.layoutobservables.computedbbox[].widths[1] >= 80
            sections = player.fxwidgets[Symbol(:fxsections_, fx.id)]
            @test sections.box.layoutobservables.computedbbox[].widths[1] > 200
            @test first(sections.cards).layoutobservables.computedbbox[].widths[1] > 200
            camera.view.control.stored_string[] = "9.25"
            @test VE.valueat(camera, 0) ≈ 9.25
            @test c.source.camera.eye[1] ≈ 9.25
            # Preview quality is session state, outside the saved document and Undo.
            undo_count = length(player.undostack)
            VE.setpreviewscale!(player, 0.5)
            @test player.previewscale[] == 0.5
            @test player.sequence.canvas == (96,64)
            @test length(player.undostack) == undo_count
            @test (c.source.width, c.source.height) == (96,64)
            @test_throws ArgumentError VE.setpreviewscale!(player, 0)
            VE.setpreviewscale!(player, 1)
            @test camera.view.lane.valuerange[][2] >= 9.25
            VE.followscene!(player, c, camera)
            @test VE.isfollowing(camera)
            @test !VE.isanimated(camera)
            @test c.source.camera.eye[1] ≈ 4
            VE.seek!(player, 3)
            @test c.source.camera.eye[1] ≈ 4.3
            @test VE.valueat(camera, 3) ≈ 4.3 atol=1e-6
            VE.togglekey!(player, c, camera)
            @test !VE.isfollowing(camera)
            @test VE.isanimated(camera)
            VE.saveproject(path, player.sequence)
            back = VE.loadproject(path)
            @test VE.isanimated(VE.param(VE.findslot(only(back.clips), :scene), camera.name))
            VE.followscene!(player, c, camera)
            VE.Makie.xlims!(player.timeline.axis, -0.1, 1.0)
            VE.undo!(player)
            @test player.timeline.viewrange[] == (-0.1, 1.0)
            @test !VE.isfollowing(VE.param(VE.findslot(only(player.sequence.clips), :scene), camera.name))
            VE.split!(player.sequence, 4)
            source, target = player.sequence.clips
            @test VE.clipname(source) == "Opening"
            @test VE.clipname(target) == "Close-up"
            VE.seek!(player,4)
            VE.toggletransition!(player)
            @test length(player.sequence.transitions) == 1
            VE.undo!(player)
            @test isempty(player.sequence.transitions)
            VE.redo!(player)
            @test length(player.sequence.transitions) == 1
            VE.toggletransition!(player)
            @test isempty(player.sequence.transitions)
            VE.undo!(player)
            @test length(player.sequence.transitions) == 1
            VE.undo!(player)
            @test isempty(player.sequence.transitions)
            source,target=player.sequence.clips
            colourfx = VE.findslot(source, :color)
            VE.setvalue!(VE.param(colourfx, :saturation), 0.7, 0)
            VE.copygrading!(player, source)
            targetfx = VE.findslot(target, :color)
            @test targetfx.id != colourfx.id
            @test VE.valueat(VE.param(targetfx, :saturation), 4) ≈ 0.7
            @test VE.param(targetfx, :saturation) !== VE.param(colourfx, :saturation)
            VE.undo!(player)
            @test VE.valueat(VE.param(VE.findslot(player.sequence.clips[2], :color), :saturation), 4) ≈ 1
            # Opening another document replaces its sequence-level edits too.
            checkpoint = joinpath(dir, "checkpoint.videoedit")
            savedvoice = VE.Narration("saved words",0,"actor";anchor=first(player.sequence.clips).source,
                speech=VE.SpeechSettings(model=:voxcpm2,direction="quiet"),label="saved line")
            append!(savedvoice.samples,Float32[.1,.2]); savedvoice.rate=48000
            push!(player.sequence.narration,savedvoice)
            player.sequence.canvas=(80,48); player.sequence.framerate=12
            push!(player.sequence.trackheights,2.0)
            VE.saveproject(checkpoint,player.sequence)
            empty!(player.sequence.narration); player.sequence.canvas=(96,64)
            player.sequence.framerate=24; empty!(player.sequence.trackheights)
            VE.restoreproject!(player,checkpoint;adopt=true)
            reopenedvoice=only(player.sequence.narration)
            @test reopenedvoice.text == "saved words"
            @test reopenedvoice.speech.direction == "quiet"
            @test reopenedvoice.samples == Float32[.1,.2]
            @test reopenedvoice.anchor === first(player.sequence.clips).source
            @test player.sequence.canvas == (80,48)
            @test player.sequence.framerate == 12
            @test player.sequence.trackheights == [2.0]
            @test player.projectpath == checkpoint
            @test player.timeline.viewrange[][2] >= VE.seqlength(player.sequence)/12
            VE.undo!(player)
            @test isempty(player.sequence.narration)
            @test player.sequence.framerate == 24
            @test player.sequence.canvas == (96,64)
            @test isempty(player.sequence.trackheights)
            VE.redo!(player)
            @test only(player.sequence.narration).speech.direction == "quiet"
            before_source = first(player.sequence.clips).source
            VE.restoreproject!(player,checkpoint)
            @test first(player.sequence.clips).source !== before_source
            VE.undo!(player)
            @test first(player.sequence.clips).source === before_source
            @test only(player.sequence.narration).anchor === before_source
            player.tool[] = :crop
            @test !isempty(player.croprect[])
            player.tool[] = :none
            @test isempty(player.croprect[])
            # A document resize can arrive while a scene render yields. The
            # completed old frame must not download into, or publish over, the
            # replacement canvas. Trigger the resize inside composition, after
            # it has captured its output size and before it downloads the pixels.
            replacement = Ref{Any}()
            canvas = player.sequence.canvas
            resizecanvas = (clip,frame) -> begin
                player.sequence.canvas = (32,24)
                VE.ensureframesize!(player,(32,24))
                replacement[] = player.frame[]
                fill!(replacement[],VE.RGB{VE.N0f8}(1,0,1))
                nothing
            end
            try
                n = player.playhead[]
                @test !VE.compositepreview!(player,VE.clipsat(player.sequence,n),n,
                                           (_,_) -> nothing;alphafor=resizecanvas)
                @test player.frame[] === replacement[]
                @test all(==(VE.RGB{VE.N0f8}(1,0,1)),player.frame[])
            finally
                player.sequence.canvas = canvas
            end
        finally
            close(player)
        end
    end
end

@testset "recipe performance controls" begin
    mktempdir() do dir
        script = joinpath(dir, "performance.jl")
        write(script, """
        using Makie
        function buildscene(canvas, args)
            sc = Scene(; size=canvas, camera=cam3d!, backgroundcolor=:black)
            body = mesh!(sc, Rect3f(Vec3f(-0.4), Vec3f(0.8)); color=:orange, name=:body)
            position = Ref(0.0)
            control = (name=:acting, label="Acting",
                sample=(frame,fps)->(phase=frame/10, amplitude=1.0, offset=Vec3f(0)),
                apply! = (edits,frame,fps)->(position[] = get(edits,:phase,frame/10) *
                    get(edits,:amplitude,1.0) + get(edits,:offset,Vec3f(0))[1]))
            update! = (frame,fps)->begin
                translate!(body, position[], 0, 0)
                update_cam!(sc, Vec3f(4,-6,4), Vec3f(0), Vec3f(0,0,1))
            end
            return (scene=sc, update! = update!, controls=[control])
        end
        """)
        build = VE.programscene(script)
        c = VE.sceneclip(VE.buildscene(build); build, frames=8,canvas=(96,64),framerate=24)
        seq = VE.Sequence([c],24);seq.canvas=(96,64)
        path=joinpath(dir,"performance.videoedit");VE.saveproject(path,seq)
        renderer=VE.farmrenderer(VE.renderjob(path,joinpath(dir,"original")))
        try
            VE.farmframes!(renderer,[4])
            c=only(renderer.sequence.clips);src=c.source;fx=VE.findslot(c,:scene)
            VE.sceneparamsections(c,fx)
            phase=VE.param(fx,Symbol("acting.phase"))
            @test VE.isfollowing(phase)
            @test VE.scenevalue(src,phase.name) ≈ 0.4
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 0.4
            VE.setvalue!(phase,2,4);VE.farmframes!(renderer,[6])
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 2
            offset=VE.param(fx,Symbol("acting.offset[1]"));VE.setvalue!(offset,1,6)
            VE.farmframes!(renderer,[6])
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 3
            # World-space actor transforms run after recipe performance edits.
            translation=VE.param(fx,Symbol("body.translation[1]"))
            VE.setvalue!(translation,5,6);VE.farmframes!(renderer,[6])
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 5
            translation.input=VE.ParamInput(:copy,VE.SceneRef(translation.name));VE.bindinputs!(renderer.sequence)
            VE.setkey!(phase,0,0.0);VE.setkey!(phase,7,3.0)
            VE.farmframes!(renderer,[7]);VE.saveproject(path,renderer.sequence)
            back=VE.loadproject(path)
            @test VE.isanimated(VE.param(VE.findslot(only(back.clips),:scene),phase.name))
            saved=VE.farmrenderer(VE.renderjob(path,joinpath(dir,"edited")))
            try
                @test only(VE.farmframes!(saved,[7])).png == only(VE.farmframes!(renderer,[7])).png
            finally
                VE.closefarm!(saved)
            end
            fx.enabled[]=false;VE.farmframes!(renderer,[6])
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 0.6
            fx.enabled[]=true
            phase.input=VE.ParamInput(:copy,VE.SceneRef(phase.name))
            offset.input=VE.ParamInput(:copy,VE.SceneRef(offset.name));VE.bindinputs!(renderer.sequence)
            VE.farmframes!(renderer,[3])
            @test VE.Makie.findplot(VE.targetscene(src.live.target),:body).transformation.translation[][1] ≈ 0.3
        finally
            VE.closefarm!(renderer)
        end
    end
end
