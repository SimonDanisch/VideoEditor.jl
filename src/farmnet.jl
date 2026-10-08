# Rendering on other machines, and on more than one GPU of this one. Every
# machine that renders runs a farm daemon (`farm/farmd.jl`), which offers its
# GPUs as slots. The coordinator sends a slot the job's snapshot; the daemon
# instantiates the job's environment and starts a renderer for that GPU in it
# (`farmchild`), then relays frames over the slot's TCP connection. Nothing else
# is shared: no filesystem, no remote shell, no agent connection.

"""
    FarmMachine(host; port = FARM_PORT, token)

A machine running the farm daemon: where it listens, and the token it was
configured with.
"""
struct FarmMachine
    host::String
    port::Int
    token::String
end
FarmMachine(host::AbstractString; port::Integer = FARM_PORT, token::AbstractString) =
    FarmMachine(String(host), Int(port), String(token))

"""
    farmmachines(config = ~/.videoeditor/farm.toml) -> Vector{FarmMachine}

The machines a farm config lists, each a `[[machine]]` table with `host`, and
optionally `port`, and `token` (or one top-level `token` for all of them).
"""
function farmmachines(config::AbstractString = joinpath(homedir(), ".videoeditor", "farm.toml"))
    d = TOML.parsefile(config)
    return [FarmMachine(m["host"]; port = get(m, "port", FARM_PORT), token = get(m, "token", get(d, "token", "")))
            for m in d["machine"]]
end

"""One request to a daemon on its own connection, and its answer (an error if it reports one)."""
function farmrequest(m::FarmMachine, header::AbstractDict, payload = UInt8[])
    sock = connect(m.host, m.port)
    try
        sendmessage(sock, merge(Dict{String, Any}("token" => m.token, "protocol" => FARM_PROTOCOL), header), payload)
        answer, _ = receivemessage(sock)
        e = farmerror(answer)
        e === nothing || error("$(m.host): $e")
        return answer
    finally
        close(sock)
    end
end

"""The slots a farm daemon offers: its GPUs, by name, and whether a job has one now."""
farmslots(m::FarmMachine) = farmrequest(m, Dict{String, Any}("command" => "hello"))["slots"]

"""
A GPU of a farm daemon, opened for one job: renders the frames it is asked for
over its connection. Closing it ends the renderer and frees the GPU.
"""
mutable struct FarmSlot
    name::String
    socket::TCPSocket
end

function (s::FarmSlot)(frames)
    sendmessage(s.socket, Dict{String, Any}("command" => "frames", "frames" => collect(Int, frames)))
    return map(frames) do _
        header, png = receivemessage(s.socket)
        e = farmerror(header)
        e === nothing || error("$(s.name): $e")
        (frame = Int(header["frame"]), png = png, seconds = Float64(header["seconds"]))
    end
end

function Base.close(s::FarmSlot)
    isopen(s.socket) && sendmessage(s.socket, Dict{String, Any}("command" => "close"))
    close(s.socket)
    return nothing
end

"""
    openslot(machine, slot, job, archive) -> FarmSlot

Open `slot` of a machine's daemon for `job`, sending its snapshot (`archive`, a
tar of it). Returns once the renderer there is ready: after the daemon has
instantiated the job's environment and the renderer has loaded it, which the
first time on a machine includes installing and compiling the packages.
"""
function openslot(m::FarmMachine, slot::AbstractString, job::RenderJob, archive::Vector{UInt8})
    sock = connect(m.host, m.port)
    header = Dict{String, Any}("token" => m.token, "protocol" => FARM_PROTOCOL, "command" => "open", "slot" => slot,
                               "job" => job.key, "julia" => job.julia, "backends" => job.backends,
                               "packages" => job.packages)
    sendmessage(sock, header, archive)
    answer, _ = receivemessage(sock)
    e = farmerror(answer)
    if e !== nothing
        close(sock)
        error("$(m.host) · $slot: $e")
    end
    return FarmSlot("$(answer["machine"]) · $slot", sock)
end

"""
    farmworkers(job, machines) -> Vector{FarmWorker}

A `FarmWorker` for every free GPU slot of the farm daemons on `machines`, each
opened for `job` (see `openslot`); slots open in parallel. A machine or slot
that cannot be used is reported and left out; an error only if none can.
"""
function farmworkers(job::RenderJob, machines)
    archive = Tar.create(jobsnapshot(job), IOBuffer()) |> take!
    workers = FarmWorker[]
    problems = String[]
    guard = ReentrantLock()
    @sync for m in machines
        @async begin
            slots = try
                [s["name"] for s in farmslots(m) if !s["busy"]]
            catch e
                lock(() -> push!(problems, "$(m.host): $(sprint(showerror, e))"), guard)
                String[]
            end
            @sync for slot in slots
                @async try
                    s = openslot(m, slot, job, archive)
                    lock(() -> push!(workers, FarmWorker(s.name, s)), guard)
                catch e
                    lock(() -> push!(problems, "$(m.host) · $slot: $(sprint(showerror, e))"), guard)
                end
            end
        end
    end
    isempty(problems) || @warn "farm slots left out" problems
    isempty(workers) && error("no farm slot could be opened:\n  " * join(problems, "\n  "))
    return workers
end

"""
    farmchild(directory; input = stdin, output = stdout)

A farm renderer serving the job in `directory` over a pipe: what a farm daemon
runs for each GPU, in the job's environment, its GPU chosen by `MANTLE_DEVICE`.
Answers each `frames` request with one message per frame (the PNG as payload),
or an error message for a frame that fails, until `close` or the end of input.
"""
function farmchild(directory::AbstractString; input::IO = stdin, output::IO = stdout)
    # what scene code prints goes to the daemon's log, not into the messages
    redirect_stdout(stderr)
    renderer = farmrenderer(directory)
    try
        sendmessage(output, Dict{String, Any}("ready" => true))
        while !eof(input)
            header, _ = receivemessage(input)
            header["command"] == "close" && break
            header["command"] == "frames" || error("unknown farm request $(header["command"])")
            for n in header["frames"]
                result = try
                    # opening the job may have loaded the scenes' packages: newer methods than this call's
                    only(Base.invokelatest(farmframes!, renderer, [n]))
                catch e
                    @error "farm frame $n failed" exception = (e, catch_backtrace())
                    sendmessage(output, Dict{String, Any}("error" => "frame $n: " * sprint(showerror, e)))
                    continue
                end
                sendmessage(output, Dict{String, Any}("frame" => result.frame, "seconds" => result.seconds), result.png)
            end
        end
    finally
        closefarm!(renderer)
    end
    return nothing
end
