# VideoEditor's render farm daemon: run one on every machine that renders.
#
#     julia farmd.jl [config.toml]          (default ~/.videoeditor/farmd.toml)
#
# It offers the machine's GPUs as slots (`farmd.example.toml`). A coordinator
# (`VideoEditor.farmworkers`) opens a slot for a job over TCP and sends the job's
# snapshot; the daemon instantiates the job's environment (its
# `environment/Project.toml`, every package pinned) with the job's Julia version
# and starts a renderer in it on that slot's GPU (`VideoEditor.farmchild`, the
# GPU chosen by `MANTLE_DEVICE`), then relays frames until the coordinator
# closes the slot or its connection drops. A second GPU is a second slot.
#
# Standard library only: the daemon loads no VideoEditor, so one daemon serves
# jobs of every VideoEditor and package version. Environments and job snapshots
# are kept under `root` and reused; the packages and compiled code they need live
# in the ordinary Julia depot, shared by every job with the same versions.

using Sockets, TOML, Tar, SHA
using FileWatching.Pidfile: mkpidlock

include(joinpath(@__DIR__, "..", "src", "farmwire.jl"))

"""A machine's farm daemon: its config, where it keeps jobs, and which slots are taken."""
struct Daemon
    config::Dict{String, Any}
    root::String
    slots::Vector{Dict{String, Any}}
    busy::Set{String}
    locks::Dict{String, ReentrantLock}       # per job and per environment: one unpacks or installs it
    juliacmds::Dict{String, Cmd}             # Julia version => the command starting it
    guard::ReentrantLock
end

function Daemon(config::Dict{String, Any})
    haskey(config, "token") && !isempty(config["token"]) ||
        error("set a `token` in the farm daemon's config: coordinators send it to use this machine")
    slots = get(config, "slot", [Dict{String, Any}("name" => gethostname())])
    length(unique(s["name"] for s in slots)) == length(slots) || error("slot names must be unique")
    root = expanduser(get(config, "root", joinpath(homedir(), ".videoeditor", "farm")))
    foreach(d -> mkpath(joinpath(root, d)), ("jobs", "envs", "logs", "locks"))
    return Daemon(config, root, slots, Set{String}(), Dict{String, ReentrantLock}(), Dict{String, Cmd}(),
                  ReentrantLock())
end

namedlock(d::Daemon, name) = lock(() -> get!(ReentrantLock, d.locks, name), d.guard)

"""
Serve until killed: every connection is one request (`hello`) or one slot
session (`open`), each on its own task.
"""
function serve(d::Daemon)
    host = parse(IPAddr, get(d.config, "bind", "::"))
    port = get(d.config, "port", FARM_PORT)
    server = listen(host, port)
    @info "farm daemon listening" host port slots = [s["name"] for s in d.slots] root = d.root
    while true
        sock = accept(server)
        errormonitor(@async session(d, sock))
    end
end

function session(d::Daemon, sock)
    try
        header, payload = receivemessage(sock)
        get(header, "token", "") == d.config["token"] || return refuse(sock, "wrong token")
        get(header, "protocol", 0) == FARM_PROTOCOL ||
            return refuse(sock, "farm protocol $(get(header, "protocol", 0)), this daemon speaks $FARM_PROTOCOL")
        command = header["command"]
        if command == "hello"
            sendmessage(sock, Dict{String, Any}("machine" => gethostname(),
                "slots" => [Dict{String, Any}("name" => s["name"], "busy" => s["name"] in d.busy) for s in d.slots]))
        elseif command == "open"
            slotsession(d, sock, header, payload)
        else
            refuse(sock, "unknown request $command")
        end
    catch e
        # The coordinator learns why, and so does the log; then the connection closes.
        @error "farm session failed" exception = (e, catch_backtrace())
        isopen(sock) && sendmessage(sock, Dict{String, Any}("error" => sprint(showerror, e)))
    finally
        close(sock)
    end
end

refuse(sock, why) = sendmessage(sock, Dict{String, Any}("error" => why))

"""
One job on one slot: claim the GPU, unpack the snapshot, instantiate its
environment, start the renderer, relay frames, and on any end (closed, dropped,
failed) stop the renderer and free the GPU.
"""
function slotsession(d::Daemon, sock, header, archive)
    slot = only(filter(s -> s["name"] == header["slot"], d.slots))
    lease = claim!(d, slot)
    try
        job = unpack!(d, header["job"], archive)
        julia = juliacmd!(d, header["julia"])
        logpath = joinpath(d.root, "logs", string(first(header["job"], 12), "-", fileslug(slot["name"]), ".log"))
        log = open(logpath, "a")
        try
            env = environment!(d, job, julia, log, logpath)
            renderer = startrenderer(d, slot, julia, env, job, header["backends"], header["packages"], log)
            try
                eof(renderer) && error("the renderer on $(slot["name"]) stopped while starting:\n" * logtail(logpath))
                ready, _ = receivemessage(renderer)
                e = farmerror(ready)
                e === nothing || error(e)
                sendmessage(sock, Dict{String, Any}("ready" => true, "machine" => gethostname()))
                relay(sock, renderer)
            finally
                stop!(renderer)
            end
        finally
            close(log)
        end
    finally
        release!(d, slot, lease)
    end
end

"""Take the slot's GPU: in this daemon, and on the machine (a lock file another daemon would also find)."""
function claim!(d::Daemon, slot)
    name = slot["name"]
    lock(d.guard) do
        name in d.busy && error("slot $name is busy")
        push!(d.busy, name)
    end
    try
        return mkpidlock(joinpath(d.root, "locks", fileslug(name) * ".pid"); wait = false)
    catch
        lock(() -> delete!(d.busy, name), d.guard)
        rethrow()
    end
end

function release!(d::Daemon, slot, lease)
    close(lease)
    lock(() -> delete!(d.busy, slot["name"]), d.guard)
end

fileslug(name) = replace(name, r"[^A-Za-z0-9_.-]" => "_")

"""The last `n` lines of a log, for an error message."""
logtail(path, n = 15) = join(last(readlines(path), n), "\n")

"""The job's snapshot under `root/jobs/<job>`: unpacked once, whichever slot gets it first."""
function unpack!(d::Daemon, key, archive)
    job = joinpath(d.root, "jobs", key)
    lock(namedlock(d, "job:" * key)) do
        isfile(joinpath(job, "job", "job.toml")) && return
        mkpath(job)
        mktempdir(job) do temp
            Tar.extract(IOBuffer(archive), joinpath(temp, "job"))
            mv(joinpath(temp, "job"), joinpath(job, "job"); force = true)
        end
    end
    return job
end

"""
The job's environment, instantiated: one per distinct Project.toml, kept under
`root/envs` and reused by every job with the same code. Pkg's automatic
precompilation is off; the renderer compiles what it loads.
"""
function environment!(d::Daemon, job, julia::Cmd, log, logpath)
    project = joinpath(job, "job", "environment", "Project.toml")
    env = joinpath(d.root, "envs", first(bytes2hex(open(sha256, project)), 16))
    lock(namedlock(d, "env:" * env)) do
        isfile(joinpath(env, "instantiated")) && return
        mkpath(env)
        cp(project, joinpath(env, "Project.toml"); force = true)
        println(log, "instantiating ", env)
        flush(log)
        vars = merge(ENV, Dict("JULIA_PKG_PRECOMPILE_AUTO" => "0"))
        cmd = setenv(`$julia --startup-file=no --project=$env -e "import Pkg; Pkg.instantiate()"`, vars)
        success(pipeline(cmd; stdout = log, stderr = log)) ||
            error("instantiating the job's environment failed on $(gethostname()):\n" * logtail(logpath))
        touch(joinpath(env, "instantiated"))
    end
    return env
end

"""
The command running Julia `version`: `julia +version` where juliaup is installed,
or the configured `julia` if it is that version. Asked once per version.
"""
function juliacmd!(d::Daemon, version)
    lock(d.guard) do
        haskey(d.juliacmds, version) && return d.juliacmds[version]
        base = Cmd(String.(split(get(d.config, "julia", "julia"))))
        for cmd in (`$base +$version`, base)
            out = IOBuffer()
            ok = success(pipeline(`$cmd --startup-file=no -e "print(VERSION)"`; stdout = out, stderr = devnull))
            ok && String(take!(out)) == version && return (d.juliacmds[version] = cmd)
        end
        error("Julia $version is not installed on $(gethostname()): `juliaup add $version`")
    end
end

"""
Start the renderer for `slot` in `env`: Julia with the job's environment,
loading VideoEditor, the job's backends and the packages its scenes come from
(all before rendering starts, whose threads could not call code loaded later),
its GPU chosen by the slot's `device` (`MANTLE_DEVICE`), talking over its stdin
and stdout, logging to `log`.
"""
function startrenderer(d::Daemon, slot, julia::Cmd, env, job, backends, packages, log)
    code = join(["using VideoEditor"; ["using $b; VideoEditor.usebackend!($b)" for b in backends];
                 ["using $p" for p in packages]; "VideoEditor.farmchild(ARGS[1])"], "; ")
    vars = copy(ENV)
    haskey(slot, "device") && (vars["MANTLE_DEVICE"] = string(slot["device"]))
    threads = get(d.config, "threads", "auto")
    cmd = `$julia --startup-file=no --project=$env --threads=$threads -e $code $job`
    println(log, "starting the renderer on ", slot["name"])
    flush(log)
    return open(pipeline(setenv(cmd, vars); stderr = log), "r+")
end

"""Frames requested on `sock` go to the renderer, its answers back, until close or a dropped connection."""
function relay(sock, renderer)
    while !eof(sock)
        header, payload = receivemessage(sock)
        sendmessage(renderer, header, payload)
        header["command"] == "close" && return
        for _ in header["frames"]
            answer, png = receivemessage(renderer)
            sendmessage(sock, answer, png)
        end
    end
end

"""End a renderer: asked to close, then given half a minute, then killed."""
function stop!(renderer)
    process_running(renderer) || return
    isopen(renderer.in) && sendmessage(renderer, Dict{String, Any}("command" => "close"))
    close(renderer.in)
    timedwait(() -> !process_running(renderer), 30) === :ok || kill(renderer)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    serve(Daemon(TOML.parsefile(get(ARGS, 1, joinpath(homedir(), ".videoeditor", "farmd.toml")))))
end
