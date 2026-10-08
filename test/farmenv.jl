# A farm job's code is an environment pinned to what this machine runs (see
# `farmenvironment`), and its messages are TOML headers with byte payloads. Both
# are checked here without a GPU: the environment against a throwaway git
# repository with a local remote, the messages over a real TCP connection.

using Test, Sockets
import VideoEditor as VE

"Run git in `dir` with a fixed identity, as the test's commits need one."
gitin(dir, args...) = run(pipeline(`git -C $dir -c user.email=farm@test -c user.name=farm $args`;
                                    stdout = devnull, stderr = devnull))

@testset "a job's environment pins what this machine runs" begin
    mktempdir() do dir
        remote = joinpath(dir, "remote.git")
        run(pipeline(`git init --bare -q $remote`; stdout = devnull))
        checkout = joinpath(dir, "monorepo")
        mkpath(checkout)
        gitin(checkout, "init", "-q", "-b", "main")
        gitin(checkout, "remote", "add", "origin", remote)
        # a package in a subfolder of a monorepo
        pkg = joinpath(checkout, "Probe")
        mkpath(joinpath(pkg, "src"))
        uuid = "0b9a3c1e-4d5f-4a6b-9c7d-8e9f0a1b2c3d"
        write(joinpath(pkg, "Project.toml"), "name = \"Probe\"\nuuid = \"$uuid\"\nversion = \"0.1.0\"\n")
        write(joinpath(pkg, "src", "Probe.jl"), "module Probe end\n")
        gitin(checkout, "add", ".")
        gitin(checkout, "commit", "-q", "-m", "probe")
        gitin(checkout, "push", "-q", "origin", "main")
        head = readchomp(`git -C $checkout rev-parse HEAD`)
        # an environment developing it, with a registered package and a stdlib
        env = joinpath(dir, "env")
        mkpath(env)
        write(joinpath(env, "Project.toml"), """
        [deps]
        Probe = "$uuid"
        [preferences.Probe]
        quality = "high"
        """)
        write(joinpath(env, "Manifest.toml"), """
        julia_version = "$VERSION"
        manifest_format = "2.0"
        [[deps.Probe]]
        deps = ["Rexported", "Dates"]
        path = "../monorepo/Probe"
        uuid = "$uuid"
        version = "0.1.0"
        [[deps.Rexported]]
        git-tree-sha1 = "0000000000000000000000000000000000000000"
        uuid = "1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f"
        version = "2.3.4+1"
        [[deps.Dates]]
        uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
        version = "1.11.0"
        """)
        project = joinpath(env, "Project.toml")
        pinned = VE.farmenvironment(["Probe"]; project)
        @test pinned["sources"]["Probe"] == Dict("url" => remote, "rev" => head, "subdir" => "Probe")
        @test pinned["compat"]["Rexported"] == "=2.3.4"          # a build number cannot be a compat entry
        @test pinned["compat"]["julia"] == "=$(VERSION.major).$(VERSION.minor).$(VERSION.patch)"
        @test !haskey(pinned["deps"], "Dates")                   # standard libraries come with Julia
        @test pinned["preferences"]["Probe"]["quality"] == "high"
        @test VE.farmenvironment(["Probe"]; project, portable = false)["sources"]["Probe"] ==
              Dict("path" => pkg)
        # what a worker could not fetch is refused, by name
        write(joinpath(pkg, "src", "draft.jl"), "# not committed\n")
        @test_throws r"Probe has uncommitted changes" VE.farmenvironment(["Probe"]; project)
        gitin(checkout, "add", ".")
        gitin(checkout, "commit", "-q", "-m", "draft")
        @test_throws r"is on no remote branch: push it" VE.farmenvironment(["Probe"]; project)
        gitin(checkout, "push", "-q", "origin", "main")
        @test VE.farmenvironment(["Probe"]; project)["sources"]["Probe"]["rev"] ==
              readchomp(`git -C $checkout rev-parse HEAD`)
    end
end

@testset "farm messages over TCP" begin
    server = listen(ip"127.0.0.1", 0)
    port = getsockname(server)[2]
    # `peer`, not `sock`: inside a testset the task would share the client's variable
    echo = errormonitor(@async begin
        peer = accept(server)
        while !eof(peer)
            header, payload = VE.receivemessage(peer)
            VE.sendmessage(peer, merge(header, Dict("echoed" => true)), reverse(payload))
        end
        close(peer)
    end)
    sock = connect(ip"127.0.0.1", port)
    big = rand(UInt8, 3_000_000)                          # a frame's PNG is megabytes
    VE.sendmessage(sock, Dict{String, Any}("command" => "frames", "frames" => [3, 1]), big)
    header, payload = VE.receivemessage(sock)
    @test header["frames"] == [3, 1] && header["echoed"]
    @test payload == reverse(big)
    VE.sendmessage(sock, Dict{String, Any}("error" => "frame 3: no such clip"))
    header, payload = VE.receivemessage(sock)
    @test VE.farmerror(header) == "frame 3: no such clip" && isempty(payload)
    close(sock)
    wait(echo)
    close(server)
end
