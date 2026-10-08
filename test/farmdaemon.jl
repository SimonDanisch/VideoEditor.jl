# The farm across processes: a farm daemon (`farm/farmd.jl`) on this machine,
# a slot per GPU, a job with a `portable = false` environment, every slot opened
# over TCP and the job rendered by the daemon's own renderer processes. The
# frames must be the ones this process renders from the same job.
#
# Runs with `VIDEOEDITOR_FARM_TEST=1` and RayMakie loaded: the first run
# instantiates the job's environment and compiles VideoEditor in it, which takes
# minutes. RayMakie, not GLMakie: three processes opening hidden GL windows at
# once raced XWayland's monitor list (an X `BadRRCrtc` from GLFW, which ends the
# process), and a farm renders off-screen anyway.

using Test, Sockets
import VideoEditor as VE
import RayMakie

"A free TCP port on this machine."
freeport() = (s = listen(ip"127.0.0.1", 0); p = getsockname(s)[2]; close(s); Int(p))

@testset "farm daemon renders a job on its slots" begin
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
        VE.usebackend!(RayMakie)
        build = VE.programscene(script)
        clip = VE.sceneclip(VE.buildscene(build); build, frames=8, canvas=(64, 40), framerate=24, backend = :RayMakie)
        clip.source.screenopts = Dict{Symbol, Any}(:rasterize => true, :samples => 1, :device => "Mantle.defaultbackend()")
        clip.source.bakewith = :RayMakie
        clip.source.bakescreenopts = Dict{Symbol, Any}(:rasterize => true, :samples => 1)
        seq = VE.Sequence([clip], 24)
        seq.canvas = (64, 40)
        path = joinpath(dir, "movie.videoedit")
        VE.saveproject(path, seq)
        job = VE.renderjob(path, joinpath(dir, "job"); portable = false)

        port = freeport()
        token = "test-token"
        config = joinpath(dir, "farmd.toml")
        write(config, """
        token = "$token"
        port = $port
        bind = "127.0.0.1"
        root = "$(joinpath(dir, "farmd"))"
        [[slot]]
        name = "first"
        [[slot]]
        name = "second"
        """)
        daemon = run(pipeline(`$(Base.julia_cmd()) --startup-file=no $(joinpath(pkgdir(VE), "farm", "farmd.jl")) $config`;
                              stdout = joinpath(dir, "farmd.log"), stderr = joinpath(dir, "farmd.log")); wait = false)
        try
            machine = VE.FarmMachine("127.0.0.1"; port, token)
            @test timedwait(() -> success(`$(Base.julia_cmd()) --startup-file=no -e "using Sockets; close(connect(ip\"127.0.0.1\", $port))"`),
                            120; pollint = 1) === :ok
            @test [s["name"] for s in VE.farmslots(machine)] == ["first", "second"]
            @test_throws r"wrong token" VE.farmslots(VE.FarmMachine("127.0.0.1"; port, token = "other"))
            workers = VE.farmworkers(job, [machine])
            @test sort([w.name for w in workers]) == sort(["$(gethostname()) · first", "$(gethostname()) · second"])
            @test all(s -> s["busy"], VE.farmslots(machine))
            status = VE.renderfarm!(job, workers)
            @test status["completed"] == 8
            @test isempty(status["errors"])
            # the slots are free again once the job closed them
            @test timedwait(() -> !any(s -> s["busy"], VE.farmslots(machine)), 60) === :ok
            local_ = VE.farmrenderer(job)
            try
                for n in 0:7
                    @test read(VE.farmframepath(job.directory, n)) == only(VE.farmframes!(local_, [n])).png
                end
            finally
                VE.closefarm!(local_)
            end
        finally
            kill(daemon)
        end
    end
end
