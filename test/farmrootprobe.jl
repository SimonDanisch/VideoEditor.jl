module FarmRootProbeTests
using Test
include(joinpath(@__DIR__,"..","FarmClient","src","FarmClient.jl"))

@testset "root-mode Julia version probes retain the application project" begin
    project=dirname(Base.active_project())
    julia=`$(joinpath(Sys.BINDIR,Base.julia_exename()))`
    command=FarmClient.versioncommand(julia,project)
    @test "--project=$project" in command.exec
    @test "--startup-file=no" in command.exec
    @test read(command,String)==string(VERSION)
    # Inspect the general-purpose client mode without executing in another env.
    portable=FarmClient.versioncommand(julia,"")
    @test !any(startswith("--project"),portable.exec)
end

if Sys.isunix()
    @testset "renderer shutdown waits for stubborn processes to exit" begin
        renderer = open(`sh -c 'trap "" TERM; printf "ready\n"; while :; do read line; done'`, "r+")
        @test readline(renderer) == "ready"
        FarmClient.stop!(renderer; grace = 0.1, terminate = 0.1)
        @test !process_running(renderer)
        @test renderer.termsignal == Base.SIGKILL
    end
end
end
