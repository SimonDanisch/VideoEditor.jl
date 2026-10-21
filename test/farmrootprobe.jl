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
end
