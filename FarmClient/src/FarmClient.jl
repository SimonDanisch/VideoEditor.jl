"""
Offer this machine's GPUs to a VideoEditor render farm:

    julia -m FarmClient all 192.168.178.92:7600
    julia -m FarmClient 7900,NVIDIA 192.168.178.92:7600

The first argument picks the GPUs: `all` of them, or a comma-separated list of
GPU indices or parts of their names. The second is the farm server the editor
prints when it starts. The client connects to it, waits for jobs, and
reconnects whenever the editor restarts.

For a job, the editor sends its snapshot. The client unpacks it, instantiates
its environment (`environment/Project.toml`, every package pinned) with the
job's Julia version, and starts a renderer per GPU in it
(`VideoEditor.farmchild`), each serving frames over its own connection to the
editor. Standard library only, so one client serves jobs of every VideoEditor
and package version. Jobs, environments and logs are kept under
`~/.videoeditor/farm`; packages and compiled code live in the ordinary depot.
"""
module FarmClient

using Sockets, TOML, Tar, SHA
using FileWatching.Pidfile: mkpidlock, PidlockedError

include("wire.jl")

const USAGE = """
    usage: julia -m FarmClient <gpus> <host>[:port]
      gpus  `all`, or a comma-separated list of GPU indices or parts of their names
      host  the farm server VideoEditor prints when it starts (port $FARM_PORT by default)"""

"""A farm client: the server it offers GPUs to, which GPUs, and where it keeps jobs."""
struct Client
    host::String
    port::Int
    gpus::String
    root::String
    locks::Dict{String, ReentrantLock}   # per job and per environment: one task unpacks or installs it
    juliacmds::Dict{String, Cmd}         # Julia version => the command starting it
    guard::ReentrantLock
end

function Client(gpus::AbstractString, address::AbstractString;
                root::AbstractString = joinpath(homedir(), ".videoeditor", "farm"))
    host, port = splitaddress(address)
    foreach(d -> mkpath(joinpath(root, d)), ("jobs", "envs", "logs", "locks"))
    return Client(host, port, String(gpus), root, Dict{String, ReentrantLock}(), Dict{String, Cmd}(),
                  ReentrantLock())
end

"`host:port`, `host` for the default port, or `[v6 address]:port`."
function splitaddress(address::AbstractString)
    m = match(r"^(?:\[(?<v6>[^\]]+)\]|(?<host>[^:\[\]]+))(?::(?<port>\d+))?$", address)
    m === nothing && throw(ArgumentError("not a farm server address: $address\n$USAGE"))
    return String(something(m[:v6], m[:host])), m[:port] === nothing ? FARM_PORT : parse(Int, m[:port])
end

function (@main)(args)
    length(args) == 2 || (println(stderr, USAGE); return 1)
    serve(Client(args[1], args[2]))
    return 0
end

"""The farm server turned this client away, so trying again would not help."""
struct Refused <: Exception
    why::String
end
Base.showerror(io::IO, e::Refused) = print(io, "the farm server refused this client: ", e.why)

"""Offer the GPUs until killed, reconnecting whenever the server is unreachable or the connection drops."""
function serve(c::Client)
    waiting = false
    while true
        sock = try
            connect(c.host, c.port)
        catch e
            e isa Base.IOError || e isa Sockets.DNSError || rethrow()
            waiting || @info "waiting for the farm server at $(c.host):$(c.port)" reason = sprint(showerror, e)
            waiting = true
            sleep(5)
            continue
        end
        waiting = false
        try
            session(c, sock)
            @info "the farm server closed the connection; reconnecting"
        catch e
            e isa Refused && rethrow()
            @error "farm connection failed; reconnecting" exception = (e, catch_backtrace())
        finally
            close(sock)
        end
        sleep(5)
    end
end

"""
One connection to the server: introduce this machine, then open every job it
sends, each on its own task, reporting back on the same connection.
"""
function session(c::Client, sock)
    sendmessage(sock, Dict{String, Any}("protocol" => FARM_PROTOCOL, "role" => "machine",
                                        "machine" => gethostname(), "gpus" => c.gpus))
    answer, _ = receivemessage(sock)
    e = farmerror(answer)
    e === nothing || throw(Refused(e))
    id = answer["id"]
    @info "offering GPUs ($(c.gpus)) to the farm server at $(c.host):$(c.port)"
    writes = ReentrantLock()
    report(header) = lock(() -> sendmessage(sock, header), writes)
    while !eof(sock)
        header, archive = receivemessage(sock)
        header["command"] == "open" || error("unknown farm request $(header["command"])")
        errormonitor(@async openjob(c, report, id, header, archive))
    end
end

"""
One job: unpack it, instantiate its environment, then for each GPU a renderer
and a connection to the server. What fails before a GPU's connection exists is
reported on the client's connection, so the editor knows not to wait for it.
"""
function openjob(c::Client, report, id, header, archive)
    key = header["job"]
    @info "farm job $(first(key, 12)) arrived"
    try
        job = unpack!(c, key, archive)
        julia = juliacmd!(c, header["julia"])
        logpath = joinpath(c.root, "logs", first(key, 12) * ".log")
        env = open(log -> environment!(c, job, julia, log, logpath), logpath, "a")
        gpus = open(log -> selectgpus(c, julia, env, log, logpath), logpath, "a")
        report(Dict{String, Any}("job" => key, "gpus" => [g.name for g in gpus]))
        @sync for gpu in gpus
            @async try
                slotsession(c, id, key, gpu, julia, env, job, header["backends"], header["packages"])
            catch e
                @error "farm GPU $(gpu.name) could not start" exception = (e, catch_backtrace())
                report(Dict{String, Any}("job" => key, "gpu" => gpu.name, "error" => sprint(showerror, e)))
            end
        end
    catch e
        @error "farm job $(first(key, 12)) failed" exception = (e, catch_backtrace())
        report(Dict{String, Any}("job" => key, "error" => sprint(showerror, e)))
    end
end

namedlock(c::Client, name) = lock(() -> get!(ReentrantLock, c.locks, name), c.guard)

"""The last `n` lines of a log, for an error message."""
logtail(path, n = 15) = join(last(readlines(path), n), "\n")

"""The job's snapshot under `root/jobs/<job>`: unpacked once."""
function unpack!(c::Client, key, archive)
    job = joinpath(c.root, "jobs", key)
    lock(namedlock(c, "job:" * key)) do
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
precompilation is off; what is loaded compiles when it is loaded.
"""
function environment!(c::Client, job, julia::Cmd, log, logpath)
    project = joinpath(job, "job", "environment", "Project.toml")
    env = joinpath(c.root, "envs", first(bytes2hex(open(sha256, project)), 16))
    lock(namedlock(c, "env:" * env)) do
        isfile(joinpath(env, "instantiated")) && return
        mkpath(env)
        cp(project, joinpath(env, "Project.toml"); force = true)
        @info "instantiating the job's environment; the first job on a machine installs its packages" env
        println(log, "instantiating ", env)
        flush(log)
        cmd = addenv(`$julia --startup-file=no --project=$env -e "import Pkg; Pkg.instantiate()"`,
                     "JULIA_PKG_PRECOMPILE_AUTO" => "0")
        success(pipeline(cmd; stdout = log, stderr = log)) ||
            error("instantiating the job's environment failed on $(gethostname()):\n" * logtail(logpath))
        touch(joinpath(env, "instantiated"))
    end
    return env
end

"""
The command running Julia `version`: `julia +version` where juliaup is
installed, or this client's own Julia if it is that version. Asked once per version.
"""
function juliacmd!(c::Client, version)
    lock(c.guard) do
        haskey(c.juliacmds, version) && return c.juliacmds[version]
        for cmd in (`julia +$version`, `$(joinpath(Sys.BINDIR, Base.julia_exename()))`)
            out = IOBuffer()
            ran = try
                success(pipeline(`$cmd --startup-file=no -e "print(VERSION)"`; stdout = out, stderr = devnull))
            catch e
                e isa Base.IOError || rethrow()     # no `julia` launcher on the PATH: no juliaup
                false
            end
            ran && String(take!(out)) == version && return (c.juliacmds[version] = cmd)
        end
        error("Julia $version is not installed on $(gethostname()): `juliaup add $version`")
    end
end

"""
The GPUs this client offers, as the job's VideoEditor finds them
(`VideoEditor.farmgpus`, asked once per environment): `all` usable discrete and
integrated ones, or those the comma-separated selectors name, each an index or
part of a name. Each comes with a `name` unique on this machine.
"""
function selectgpus(c::Client, julia::Cmd, env, log, logpath)
    listing = joinpath(env, "gpus.toml")
    lock(namedlock(c, "env:" * env)) do
        isfile(listing) && return
        println(log, "listing the GPUs")
        flush(log)
        cmd = `$julia --startup-file=no --project=$env -e "using VideoEditor; VideoEditor.farmgpus(ARGS[1])" $listing`
        success(pipeline(cmd; stdout = log, stderr = log)) ||
            error("listing the GPUs failed on $(gethostname()):\n" * logtail(logpath))
    end
    devices = TOML.parsefile(listing)["gpu"]
    describe(d) = "  $(d["index"]): $(d["name"]) ($(d["kind"]))" * (isempty(d["problem"]) ? "" : ": $(d["problem"])")
    chosen = if c.gpus == "all"
        filter(d -> d["kind"] in ("discrete", "integrated") && isempty(d["problem"]), devices)
    else
        named = unique(d for s in split(c.gpus, ',') for d in selected(strip(s), devices))
        unusable = filter(d -> !isempty(d["problem"]), named)
        isempty(unusable) || error("cannot render on\n" * join(describe.(unusable), "\n"))
        named
    end
    isempty(chosen) && error("no usable GPU of $(gethostname()) matches `$(c.gpus)`; it has:\n" *
                             join(describe.(devices), "\n"))
    return map(chosen) do d
        twins = count(o -> o["name"] == d["name"], chosen)
        (index = d["index"], name = twins == 1 ? d["name"] : "$(d["name"]) #$(d["index"])")
    end
end

"""
The GPUs a selector names: a number that is a GPU's index is that GPU, anything
else (`7900`, `nvidia`) is part of the names it picks.
"""
function selected(selector::AbstractString, devices)
    index = tryparse(Int, selector)
    any(d -> d["index"] == index, devices) && return filter(d -> d["index"] == index, devices)
    return filter(d -> occursin(lowercase(selector), lowercase(d["name"])), devices)
end

"""
One GPU for one job: claim it, start the renderer on it, and once it is ready
open a connection to the server for it and relay frames until the server
closes it or the connection drops. A failure before that connection exists is
the caller's to report; afterwards the server sees the connection end.
"""
function slotsession(c::Client, id, key, gpu, julia, env, job, backends, packages)
    lease = try
        mkpidlock(joinpath(c.root, "locks", "gpu$(gpu.index).pid"); wait = false)
    catch e
        e isa PidlockedError || rethrow()
        error("$(gpu.name) is busy with another job")
    end
    logpath = joinpath(c.root, "logs", "$(first(key, 12))-gpu$(gpu.index).log")
    log = open(logpath, "a")
    try
        renderer = startrenderer(julia, env, job, gpu, backends, packages, log)
        try
            eof(renderer) && error("the renderer on $(gpu.name) stopped while starting:\n" * logtail(logpath))
            ready, _ = receivemessage(renderer)
            e = farmerror(ready)
            e === nothing || error(e)
            slot = connect(c.host, c.port)
            try
                sendmessage(slot, Dict{String, Any}("protocol" => FARM_PROTOCOL, "role" => "slot",
                                                    "machine" => id, "job" => key, "gpu" => gpu.name))
                @info "rendering on $(gpu.name)"
                relay(slot, renderer)
            catch e
                @error "farm GPU $(gpu.name) stopped" exception = (e, catch_backtrace())
            finally
                close(slot)
            end
        finally
            stop!(renderer)
        end
    finally
        close(log)
        close(lease)
    end
end

"""
Start the renderer for `gpu` in `env`: Julia with the job's environment,
loading VideoEditor, the job's backends and the packages its scenes come from
(all before rendering starts, whose threads could not call code loaded later),
talking over its stdin and stdout, logging to `log`.
"""
function startrenderer(julia::Cmd, env, job, gpu, backends, packages, log)
    code = join(["using VideoEditor"; ["using $b; VideoEditor.usebackend!($b)" for b in backends];
                 ["using $p" for p in packages]; "VideoEditor.farmchild(ARGS[1]; device = $(gpu.index))"], "; ")
    println(log, "starting the renderer on ", gpu.name)
    flush(log)
    cmd = `$julia --startup-file=no --project=$env --threads=auto -e $code $job`
    return open(pipeline(cmd; stderr = log), "r+")
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

end # module
