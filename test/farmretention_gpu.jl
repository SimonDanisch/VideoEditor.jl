# Run with RayMakie loaded, on the GPU's owner thread, in the application root.
using Test, RayMakie
import VideoEditor as VE

@testset "German labels reuse PNG pictures with no retained HDR" begin
    VE.usebackend!(RayMakie)
    mktempdir(dirname(Base.active_project())) do dir
        recipe = joinpath(dir, "layers.jl")
        write(recipe, """
        using Makie, GeometryBasics
        function buildscene(canvas,args)
            if args["layer"] == "picture"
                scene = Scene(;size=canvas,backgroundcolor=:black)
                cam3d!(scene;center=false)
                mesh!(scene,Sphere(Point3f(0),0.7f0);color=:steelblue,name=:sphere)
                update_cam!(scene,Vec3f(2,2,2),Vec3f(0))
            else
                scene = Scene(;size=canvas,camera=campixel!,backgroundcolor=RGBAf(0,0,0,0))
                text!(scene,Point2f(4,8);text=args["text"],fontsize=15,color=:white,name=:label)
            end
            return (;scene)
        end
        """)
        function layer(name)
            build = VE.programscene(recipe;args=Dict("layer"=>name,"text"=>"English"))
            VE.sceneclip(VE.buildscene(build);build,frames=2,canvas=(96,64),backend=:RayMakie)
        end
        picture, labels = layer("picture"), layer("labels")
        labels.track = 2
        VE.setrenderconfig!(picture;raytrace=true,samples=2,max_depth=2)
        VE.addslot!(picture,VE.Effect(VE.ToneMapEffect()))
        VE.setrenderconfig!(labels;raytrace=false,fxaa=false)
        english = VE.Sequence([picture,labels],30)
        english.canvas = (96,64)
        enfile, defile = joinpath(dir,"english.videoedit"), joinpath(dir,"german.videoedit")
        VE.enabletrackcache!(english,enfile*".frames")
        VE.saveproject(enfile,english)
        job = VE.renderjob(enfile,joinpath(dir,"farm");portable=false,hdr=false)
        @test job.planes && !job.hdr
        renderer = VE.farmrenderer(job)
        initial_traces = 0
        try
            @test renderer.keepplanes && !renderer.keephdr
            worker = VE.FarmWorker("retention test GPU",ids -> VE.farmframes!(renderer,ids))
            @test VE.renderfarm!(job,[worker];reuse=english)["completed"] == 1
            initial_traces = VE.sourcecache(VE.clipbyid(renderer.sequence,picture.id)).renders
            @test initial_traces > 0
        finally
            VE.closefarm!(renderer)
        end
        receipt = VE.JSON.parsefile(VE.farmreceiptpath(job.directory,0))
        @test length(receipt["layers"]) == 2
        @test all(!haskey(l,"hdr_sha256") for l in receipt["layers"])
        @test all(!endswith(f,".vehdr") for (_,_,files) in walkdir(job.directory) for f in files)
        VE.adoptfarmcache!(english,job)
        VE.saveproject(enfile,english)
        # Saving newly adopted planes must not turn a resume into a new edit.
        @test VE.renderjob(enfile,job.directory;portable=false,hdr=false).key == job.key
        carried = VE.renderjob(enfile,joinpath(dir,"carried");portable=false,hdr=false)
        @test VE.carryframes!(carried,job) == 1
        @test length(VE.JSON.parsefile(VE.farmreceiptpath(carried.directory,0))["layers"]) == 2
        @test samefile(VE.cachefile(VE.farmplanes(carried.directory,1),picture.id,0),
                       VE.cachefile(VE.farmplanes(job.directory,1),picture.id,0))
        @test VE.cachevalid(VE.trackcache(picture),picture.id,0)
        @test VE.cachevalid(VE.trackcache(labels),labels.id,0)
        @test !VE.cachevalid(VE.sourcecache(picture),picture.id,0)
        @test !VE.cachevalid(VE.trackcache(picture),picture.id,1)
        @test isempty(VE.farmneededframes(english)) # the held picture has an equivalent PNG
        original_picture = VE.cachefile(VE.trackcache(picture),picture.id,0)
        original_labels = VE.cachefile(VE.trackcache(labels),labels.id,0)
        picture_hash, label_hash = VE.filehash(original_picture), VE.filehash(original_labels)

        german = VE.loadproject(enfile)
        translated = VE.clipbyid(german,labels.id)
        VE.setsceneargs!(translated,Dict("text"=>"Deutsch"))
        VE.saveproject(defile,german)
        @test_throws r"changed" VE.renderjob(defile,job.directory;portable=false,hdr=false)
        cached_picture = VE.clipbyid(german,picture.id)
        shared = samefile(original_picture,VE.cachefile(VE.trackcache(cached_picture),picture.id,0))
        @test shared
        @test isempty(VE.farmneededframes(german))
        @test VE.sourcecache(cached_picture).renders == 0
        engine = VE.FxEngine(VE.Mantle.defaultbackend())
        readers = Dict{String,Any}()
        output = zeros(VE.RGB{VE.N0f8},96,64)
        black = zero(output)
        try
            VE.withfinalscenes(german) do
                VE.renderframe!(output,german,0,readers,engine;black)
            end
            @test VE.sourcecache(cached_picture).renders == 0
            @test VE.trackcache(translated).renders == 1
            @test VE.filehash(original_picture) == picture_hash
            @test VE.filehash(original_labels) == label_hash
            @test VE.filehash(VE.cachefile(VE.trackcache(translated),labels.id,0)) != label_hash
            @test output != permutedims(VE.PNGFiles.load(VE.farmframepath(job.directory,0)),(2,1))
            german_frame = copy(output)
            VE.withfinalscenes(german) do
                VE.renderframe!(output,german,1,readers,engine;black)
            end
            @test output == german_frame
            @test VE.sourcecache(cached_picture).renders == 0
            @test samefile(original_picture,VE.cachefile(VE.trackcache(cached_picture),picture.id,1))
            for c in german.clips
                c.start = 3
            end
            VE.withfinalscenes(german) do
                VE.renderframe!(output,german,3,readers,engine;black)
            end
            @test output == german_frame
            @test VE.sourcecache(cached_picture).renders == 0
            VE.saveproject(defile,german)
            reopened = VE.loadproject(defile)
            @test isempty(VE.farmneededframes(reopened;frames=[3,4]))
            @test VE.clipbyid(reopened,labels.id).source.build["args"]["text"] == "Deutsch"
            @test samefile(original_picture,VE.cachefile(VE.trackcache(VE.clipbyid(reopened,picture.id)),picture.id,0))
            if haskey(ENV,"VIDEOEDITOR_RETENTION_REVIEW")
                review = ENV["VIDEOEDITOR_RETENTION_REVIEW"]
                mkpath(review)
                cp(VE.farmframepath(job.directory,0),joinpath(review,"english.png");force=true)
                VE.PNGFiles.save(joinpath(review,"german.png"),permutedims(output,(2,1)))
                VE.farmwritejson(joinpath(review,"verification.json"),Dict(
                    "initial_hdr_renders"=>initial_traces,"german_hdr_renders"=>0,
                    "retimed_hdr_renders"=>0,"hdr_files_retained"=>0,"picture_storage_shared"=>shared))
            end
        finally
            foreach(close,values(readers))
            VE.emptyengine!(engine)
            foreach(close,unique(c.source for c in german.clips))
        end
    end
end
