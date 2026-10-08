# The farm across processes: this process's farm server, a farm client offering
# every GPU of this machine, and a job with a `portable = false` environment that
# the client's renderers render, one per GPU. The frames must be the ones this
# process renders from the same job.
#
# Runs with `VIDEOEDITOR_FARM_TEST=1` and RayMakie loaded: the first run
# instantiates the job's environment and compiles VideoEditor in it, which takes
# minutes. RayMakie, not GLMakie: several processes opening hidden GL windows at
# once raced XWayland's monitor list, and a farm renders off-screen anyway.

using Test, Sockets, TOML
import VideoEditor as VE
import RayMakie

const FARMCLIENT = joinpath(pkgdir(VE), "FarmClient")

@testset "farm client: the command line" begin
    out = IOBuffer()
    p = run(pipeline(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --project=$FARMCLIENT -m FarmClient`);
                     stdout = devnull, stderr = out))
    @test p.exitcode == 1
    @test occursin("usage: julia -m FarmClient <gpus> <host>[:port]", String(take!(out)))
end

@testset "farm clients render a job on their GPUs" begin
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

        server = VE.FarmServer(; port = 0, bind = ip"127.0.0.1")
        @test_throws r"no farm client is connected" VE.farmworkers(job, server)
        root = joinpath(dir, "client")
        code = "using FarmClient; FarmClient.serve(FarmClient.Client(\"all\", \"127.0.0.1:$(server.port)\"; root = \"$root\"))"
        log = joinpath(dir, "client.log")
        client = run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$FARMCLIENT -e $code`;
                              stdout = log, stderr = log); wait = false)
        try
            @test timedwait(() -> !isempty(VE.farmmachines(server)), 120; pollint = 1) === :ok
            machine = only(VE.farmmachines(server)).name
            @test startswith(machine, gethostname())
            listing = joinpath(dir, "gpus.toml")
            VE.farmgpus(listing)
            gpus = [g["name"] for g in TOML.parsefile(listing)["gpu"]
                    if g["kind"] in ("discrete", "integrated") && isempty(g["problem"])]
            status = VE.renderfarm!(job, VE.farmworkers(job, server))
            @test status["completed"] == 8
            @test isempty(status["errors"])
            @test sort(collect(keys(status["workers"]))) == sort(["$machine · $g" for g in gpus])
            # every GPU is free again once the job closed it
            @test timedwait(() -> isempty(readdir(joinpath(root, "locks"))), 60) === :ok
            # a finished job ends at once; GPUs that get ready afterwards are closed and freed
            @test VE.renderfarm!(job, VE.farmworkers(job, server))["completed"] == 8
            @test timedwait(() -> isempty(server.jobs), 300) === :ok
            @test timedwait(() -> isempty(readdir(joinpath(root, "locks"))), 60) === :ok
            local_ = VE.farmrenderer(job)
            try
                for n in 0:7
                    @test read(VE.farmframepath(job.directory, n)) == only(VE.farmframes!(local_, [n])).png
                end
            finally
                VE.closefarm!(local_)
            end
        finally
            close(server)
            kill(client)
        end
    end
end
