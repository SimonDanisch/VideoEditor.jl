using Test
import VideoEditor as VE

@testset "finished farm frames survive a cut without rendering" begin
    mktempdir() do dir
        build = VE.scenebuild(:bar, (4, 3))
        clip = VE.sceneclip(VE.buildscene(build); build, frames = 6, canvas = (4, 3))
        seq = VE.Sequence([clip], 30)
        seq.canvas = (4, 3)
        function snapshot(name, sequence)
            root = joinpath(dir, name)
            mkpath(joinpath(root, "job"))
            VE.saveproject(joinpath(root, "job", "project.videoedit"), sequence)
            VE.RenderJob(root, name, (4, 3), VE.seqlength(sequence), 30.0, string(VERSION),
                         String[], String[], Dict{String, String}(), false)
        end
        original = snapshot("original", seq)
        for n in 0:5
            # Bookkeeping fixture: copying/validation must never decode or render it.
            bytes = Vector{UInt8}(codeunits("finished frame $n"))
            path = VE.farmframepath(original.directory, n)
            sha = VE.commitbytes!(path, bytes)
            VE.farmwritejson(VE.farmreceiptpath(original.directory, n),
                Dict("job" => original.key, "frame" => n, "sha256" => sha, "layers" => Any[]))
        end
        clip.src_in = 2
        revised = snapshot("trimmed", seq)
        mapping = [n => n + 2 for n in 0:3]
        @test VE.carryframes!(revised, original; frames = mapping) == 4
        @test VE.carryframes!(revised, original; frames = mapping) == 0
        for (n, old) in mapping
            @test VE.farmcomplete(revised.directory, n, revised.key)
            @test read(VE.farmframepath(revised.directory, n)) == read(VE.farmframepath(original.directory, old))
            receipt = VE.JSON.parsefile(VE.farmreceiptpath(revised.directory, n))
            @test receipt["frame"] == n && receipt["carried_frame"] == old
            @test receipt["carried_from"] == original.key
            Sys.isunix() && @test samefile(VE.farmframepath(revised.directory, n), VE.farmframepath(original.directory, old))
        end
        @test_throws r"differs" VE.carryframes!(revised, original; frames = [0 => 1])
        @test_throws r"duplicate" VE.carryframes!(revised, original; frames = [0 => 2, 0 => 3])
        @test_throws r"outside" VE.carryframes!(revised, original; frames = [4 => 6])
        clip.crop = (0.1, 0.0, 1.0, 1.0)
        changed = snapshot("changed", seq)
        @test_throws r"differs" VE.carryframes!(changed, original; frames = mapping)
        @test !isdir(joinpath(changed.directory, "frames"))
        # A bad pair later in the map cannot publish an earlier valid one.
        clip.crop = (0.0, 0.0, 1.0, 1.0)
        atomic = snapshot("atomic", seq)
        @test_throws r"differs" VE.carryframes!(atomic, original; frames = [0 => 2, 1 => 1])
        @test !isdir(joinpath(atomic.directory, "frames"))
        write(joinpath(atomic.directory, "job", "changed-input"), "other pixels")
        @test_throws r"unchanged input" VE.carryframes!(atomic, original; frames = mapping)
        @test all(n -> VE.farmcomplete(original.directory, n, original.key), 0:5)
        @test isempty(seq.trackcaches)
        VE.farmcaches!(seq, joinpath(dir, "worker-planes"))
        @test VE.trackcache(clip) !== nothing
        @test startswith(VE.sourcecache(clip).directory, joinpath(dir, "worker-planes"))
        @test seq.compositecache === nothing
    end
end
