using Test
import VideoEditor as VE

struct OwnedTestFrames{F}
    render::F
    closed::Base.RefValue{Bool}
end
(r::OwnedTestFrames)(ids) = r.render(ids)
Base.close(r::OwnedTestFrames) = (r.closed[] = true; nothing)

@testset "procedural scene document and farm" begin
    mktempdir() do dir
        script = joinpath(dir, "scene.jl")
        write(script, """
        using Makie
        function buildscene(canvas, args)
            scene = Scene(; size=canvas, camera=campixel!)
            dot = scatter!(scene, [Point2f(8, 12)]; color=:red, markersize=10, name=:dot)
            update! = (frame, fps) -> (dot[1] = [Point2f(8 + 4frame, 12)])
            return (scene=scene, update! = update!)
        end
        """)
        wav = joinpath(dir, "tone.wav")
        run(`$(VE.FFMPEG_jll.ffmpeg()) -v error -y -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 $wav`)
        build = VE.programscene(script)
        clip = VE.sceneclip(VE.buildscene(build); build, frames=6, canvas=(64, 40), framerate=24)
        clip.source.soundtrack = wav
        clip.source.bakewith = :GLMakie
        clip.look = ones(Float32, 2, 2, 2, 3)
        seq = VE.Sequence([clip], 24)
        seq.canvas = (64, 40)
        narration = VE.Narration("approved voice", 0.0)
        append!(narration.samples, Float32[0.01sin(2pi*220i/48000) for i in 1:12000])
        narration.rate = 48000
        push!(seq.narration, narration)
        VE.split!(seq, 3)
        path = joinpath(dir, "movie.videoedit")
        VE.saveproject(path, seq)
        opened = VE.loadproject(path)
        @test opened.clips[1].source === opened.clips[2].source
        @test opened.clips[1].look === opened.clips[2].look
        @test opened.clips[1].source.soundtrack == wav
        @test opened.clips[2].src_in == 3
        @test only(opened.narration).samples == narration.samples
        @test only(opened.narration).rate == 48000
        doc = VE.decodeblocks(VE.MsgPack.unpack(read(path)))
        @test length(doc["sources"]) == length(doc["looks"]) == 1

        # Rebase a saved script/audio path to a different machine's root,
        # without modifying the saved project or its content identity.
        rebased = joinpath(dir, "copy")
        mkdir(rebased); cp(script, joinpath(rebased, "scene.jl")); cp(wav, joinpath(rebased, "tone.wav"))
        mapped = VE.loadproject(path; pathmap=Dict(dir => rebased))
        @test mapped.clips[1].source.root.file == joinpath(rebased, "scene.jl")
        @test VE.audiopath(mapped.clips[2].source) == joinpath(rebased, "tone.wav")

        # Audio attached to a scene uses the scene's source clock after a cut.
        empty!(opened.narration)
        pcm = VE.loadpcm(opened.clips[1].source)
        tracks = Dict{String, Union{VE.PCMTrack, Nothing}}(wav => pcm)
        block = zeros(Int16, 2, 1000)
        VE.fillaudio!(block, opened, tracks, 6500)
        @test block == pcm.samples[:, 6501:7500]

        job = VE.renderjob(path, joinpath(dir, "job"))
        renderer = VE.farmrenderer(job)
        try
            a, b, again = VE.farmframes!(renderer, [3, 1, 3])
            @test a.png == again.png
            @test a.png != b.png
            # RPCs execute on arbitrary threads; graph uploads must still run
            # on the device's owning thread, as scene drawing already does.
            @test fetch(Threads.@spawn VE.farmframes!(renderer, [3]))[1].png == a.png
            @test size(VE.PNGFiles.load(IOBuffer(a.png))) == (40, 64)
            @test renderer.sequence.clips[1].source.mode === :live
            held = renderer.sequence.clips[1].source.live
            VE.farmframes!(renderer, [4])
            @test renderer.sequence.clips[2].source.live === held

            bundle = VE.bundlefarm(job, joinpath(dir, "bundle"); root=dir)
            moved = joinpath(dir, "relocated-bundle")
            cp(bundle.directory, moved)
            bundled = VE.openfarmbundle(moved)
            try
                @test only(VE.farmframes!(bundled, [3])).png == a.png
                @test bundled.sequence.clips[1].source.root.file == joinpath(moved, "data", "scene.jl")
            finally
                VE.closefarm!(bundled)
            end
            write(joinpath(moved, "data", "scene.jl"), "changed")
            @test_throws ErrorException VE.openfarmbundle(moved)

            attempts = Ref(0)
            failing = VE.FarmWorker("fails once", ids -> begin
                attempts[] += 1
                attempts[] == 1 && error("simulated transport failure")
                VE.farmframes!(renderer, ids)
            end)
            worker = VE.FarmWorker("steady", ids -> VE.farmframes!(renderer, ids))
            status = VE.renderfarm!(job, [failing, worker])
            @test status["completed"] == 6
            @test isempty(status["errors"])
            never = VE.FarmWorker("cached", _ -> error("completed frames must be retained"))
            @test VE.renderfarm!(job, [never])["completed"] == 6
            # A broken transfer cannot masquerade as a completed frame on resume.
            write(VE.farmframepath(job.directory, 2), "corrupt")
            @test VE.renderfarm!(job, [worker])["workers"]["steady"] == 1
            mp4 = VE.encodefarm!(job.directory, joinpath(dir, "movie.mp4"); preset="fast")
            meta = VE.JSON.parse(read(`$(VE.FFMPEG_jll.ffprobe()) -v error -show_streams -of json $mp4`, String))
            video = only(filter(s -> s["codec_type"] == "video", meta["streams"]))
            @test parse(Int, video["nb_frames"]) == 6
            @test VE.hasaudio(mp4)
            @test isfile(VE.encodefarm!(job.directory, joinpath(dir, "movie.mkv"); preset="fast"))

            ownedjob = VE.renderjob(path, joinpath(dir, "ownedjob"))
            closed = Ref(false)
            owned = VE.FarmWorker("owned", OwnedTestFrames(ids -> VE.farmframes!(renderer, ids), closed))
            @test VE.renderfarm!(ownedjob, [owned];frames=0:1)["completed"] == 2
            @test closed[]

            reportjob = VE.renderjob(path, joinpath(dir, "reportjob"))
            @test_throws Exception VE.renderfarm!(reportjob, [worker];
                progress=(done,total) -> done == 1 && error("progress listener failed"))
            @test count(n -> VE.farmcomplete(reportjob.directory,n,reportjob.identity), 0:5) == 1
            @test VE.renderfarm!(reportjob,[worker])["completed"] == 6

            pausejob = VE.renderjob(path, joinpath(dir, "pausejob"))
            pauseworker = VE.FarmWorker("pause", ids -> begin
                VE.pausefarm!(pausejob.directory)
                VE.farmframes!(renderer, ids)
            end)
            status = VE.renderfarm!(pausejob, [pauseworker])
            @test status["paused"] && status["completed"] == 1
            rm(joinpath(pausejob.directory, "pause.requested"))
            @test VE.renderfarm!(pausejob, [worker])["completed"] == 6

            wrongjob = VE.renderjob(path, joinpath(dir, "wrongjob"))
            wrong = VE.FarmWorker("wrong", ids -> [merge(only(VE.farmframes!(renderer, ids)), (identity="other",))])
            @test_throws ErrorException VE.renderfarm!(wrongjob, [wrong]; retries=1)
            @test !isfile(VE.farmreceiptpath(wrongjob.directory, 0))
            mkpath(path * ".mattes")
            write(joinpath(path * ".mattes", "test.bin"), "sidecar changed")
            @test_throws ErrorException VE.renderjob(path,job.directory)
            rm(path * ".mattes"; recursive=true)
            write(script, read(script, String) * "\n# changed input\n")
            @test_throws ErrorException VE.farmrenderer(job)
            @test_throws ErrorException VE.renderjob(path, job.directory)
        finally
            VE.closefarm!(renderer)
        end
        @test renderer.closed
        @test_throws ErrorException VE.farmframes!(renderer, [0])
    end
end
