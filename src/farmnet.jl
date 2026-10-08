# Rendering on other machines, and on every GPU of this one. The editor runs a
# farm server; every machine that renders runs a farm client (the `FarmClient`
# package in this repository, `julia -m FarmClient all <server>`) that connects
# to it and offers its GPUs. For a job the server sends each client the job's
# snapshot; the client instantiates the job's environment, starts a renderer per
# GPU in it (`farmchild`) and connects back once per GPU, and frames travel over
# that connection. Nothing else is shared: no filesystem, no remote shell.

"""A farm client connected to the server: one machine, and which of its GPUs it offers."""
struct FarmMachine
    id::Int
    name::String
    gpus::String
    socket::TCPSocket
    writes::ReentrantLock
end

"""
    FarmServer(; port = FARM_PORT, bind = ip"::")

Where farm clients offer their GPUs: started with the editor (see
[`farmserver`](@ref)), and the address every machine that renders runs
`julia -m FarmClient all <address>` with. `port = 0` takes any free port.
"""
mutable struct FarmServer
    listener::Sockets.TCPServer
    port::Int
    machines::Dict{Int, FarmMachine}
    jobs::Dict{String, Channel{Any}}   # what clients report for a job being opened, by its key
    lastid::Int
    guard::ReentrantLock
end

function FarmServer(; port::Integer = FARM_PORT, bind::IPAddr = ip"::")
    listener = listen(bind, port)
    server = FarmServer(listener, Int(getsockname(listener)[2]), Dict{Int, FarmMachine}(),
                        Dict{String, Channel{Any}}(), 0, ReentrantLock())
    errormonitor(@async acceptclients(server))
    @info "render farm: on each machine that renders, run\n    $(clientcommand(server))"
    return server
end

"""Stop listening and drop every client; they keep trying to reconnect."""
function Base.close(server::FarmServer)
    close(server.listener)
    foreach(m -> close(m.socket), farmmachines(server))
    return nothing
end

"""The command a machine runs to offer all its GPUs to `server`."""
clientcommand(server::FarmServer) = "julia -m FarmClient all $(getipaddr()):$(server.port)"

"""The machines connected to `server` now."""
farmmachines(server::FarmServer) = lock(() -> collect(values(server.machines)), server.guard)

const FARM_SERVER = Ref{Union{Nothing, FarmServer}}(nothing)

"""
    farmserver(; port = FARM_PORT) -> FarmServer

This session's farm server, started on first use.
"""
function farmserver(; port::Integer = FARM_PORT)
    FARM_SERVER[] === nothing && (FARM_SERVER[] = FarmServer(; port))
    return FARM_SERVER[]
end

function acceptclients(server::FarmServer)
    while true
        sock = try
            accept(server.listener)
        catch e
            isopen(server.listener) && rethrow()
            return      # closed: the server stopped
        end
        errormonitor(@async welcome(server, sock))
    end
end

"""A new connection: a client introducing its machine, or one GPU of it ready for a job."""
function welcome(server::FarmServer, sock)
    header, _ = receivemessage(sock)
    protocol = get(header, "protocol", 0)
    if protocol != FARM_PROTOCOL
        sendmessage(sock, Dict{String, Any}("error" =>
            "this client speaks farm protocol $protocol and the editor $FARM_PROTOCOL: update FarmClient"))
        close(sock)
    elseif header["role"] == "machine"
        serveclient(server, sock, header)
    elseif header["role"] == "slot"
        machine = lock(() -> get(server.machines, header["machine"], nothing), server.guard)
        machine === nothing && return close(sock)      # its client has disconnected since
        post!(server, header["job"], machine.id,
              Dict{String, Any}("slot" => FarmSlot("$(machine.name) · $(header["gpu"])", sock)))
    else
        sendmessage(sock, Dict{String, Any}("error" => "unknown farm role $(header["role"])"))
        close(sock)
    end
end

"""A client's connection: register its machine, then pass on what it reports until it disconnects."""
function serveclient(server::FarmServer, sock, header)
    machine = lock(server.guard) do
        # a hostname that does not tell machines apart is completed by the address
        name = header["machine"]
        if name == "localhost" || any(m -> m.name == name, values(server.machines))
            name = "$name ($(getpeername(sock)[1]))"
        end
        server.lastid += 1
        server.machines[server.lastid] = FarmMachine(server.lastid, name, header["gpus"], sock, ReentrantLock())
    end
    sendmessage(sock, Dict{String, Any}("id" => machine.id))
    @info "farm client connected" machine = machine.name gpus = machine.gpus
    try
        while !eof(sock)
            report, _ = receivemessage(sock)
            post!(server, report["job"], machine.id, report)
        end
    finally
        jobs = lock(server.guard) do
            delete!(server.machines, machine.id)
            collect(keys(server.jobs))
        end
        foreach(key -> post!(server, key, machine.id, Dict{String, Any}("error" => "disconnected")), jobs)
        close(sock)
        @info "farm client disconnected" machine = machine.name
    end
end

"""Hand a client's report to the job it is about; a GPU arriving for a job no longer opening is closed."""
function post!(server::FarmServer, key, machine, report)
    events = lock(() -> get(server.jobs, key, nothing), server.guard)
    if events === nothing
        haskey(report, "slot") && close(report["slot"])
    else
        put!(events, (machine, report))
    end
    return nothing
end

"""
A GPU of a farm client, opened for one job: renders the frames it is asked for
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
    farmworkers(job, server = farmserver()) -> Channel{FarmWorker}

Every GPU the connected farm clients offer, opened for `job`: each client gets
the job's snapshot, instantiates its environment and starts a renderer per GPU.
A GPU joins the channel as soon as its renderer is ready, so a render starts on
the first one while slower machines still install or compile. The channel
closes once every client has opened or failed all its GPUs; what failed is
reported, and is the channel's error if no GPU opened at all.
"""
function farmworkers(job::RenderJob, server::FarmServer = farmserver())
    machines = farmmachines(server)
    isempty(machines) &&
        error("no farm client is connected: on each machine that renders, run\n    $(clientcommand(server))")
    archive = take!(Tar.create(jobsnapshot(job), IOBuffer()))
    events = Channel{Any}(Inf)
    lock(() -> server.jobs[job.key] = events, server.guard)
    header = Dict{String, Any}("command" => "open", "job" => job.key, "julia" => job.julia,
                               "backends" => job.backends, "packages" => job.packages)
    for m in machines
        errormonitor(@async try
            lock(() -> sendmessage(m.socket, header, archive), m.writes)
        catch e
            put!(events, (m.id, Dict{String, Any}("error" => sprint(showerror, e))))
        end)
    end
    workers = Channel{FarmWorker}(Inf)
    errormonitor(@async collectslots!(workers, server, job.key, events, machines))
    return workers
end

"""
Turn what the clients report for a job into workers, until each machine has
named its GPUs and every one of them has opened or failed, or the machine failed
or disconnected as a whole.
"""
function collectslots!(workers::Channel{FarmWorker}, server::FarmServer, key, events, machines)
    names = Dict(m.id => m.name for m in machines)
    expected = Dict{Int, Union{Nothing, Int}}(m.id => nothing for m in machines)
    arrived = Dict(m.id => 0 for m in machines)
    opened = 0
    problems = String[]
    try
        while any(id -> expected[id] === nothing || arrived[id] < expected[id], keys(expected))
            id, report = take!(events)
            if !haskey(expected, id)       # a machine that connected after the job was sent
                haskey(report, "slot") && close(report["slot"])
            elseif haskey(report, "gpus")
                expected[id] = length(report["gpus"])
            elseif haskey(report, "slot")
                arrived[id] += 1
                slot = report["slot"]
                try
                    put!(workers, FarmWorker(slot.name, slot))
                    opened += 1
                catch e
                    e isa InvalidStateException || rethrow()
                    close(slot)            # the render ended before this GPU was ready
                end
            elseif haskey(report, "gpu")
                arrived[id] += 1
                push!(problems, "$(names[id]) · $(report["gpu"]): $(report["error"])")
            elseif expected[id] === nothing || arrived[id] < expected[id]
                push!(problems, "$(names[id]): $(report["error"])")
                expected[id] = arrived[id]
            end
        end
    finally
        lock(() -> delete!(server.jobs, key), server.guard)
    end
    isempty(problems) || @warn "farm GPUs left out" problems
    if opened == 0
        close(workers, ErrorException("no farm GPU could be opened:\n  " * join(problems, "\n  ")))
    else
        close(workers)
    end
end

"""The API whose devices farm renderers choose from on this platform."""
farmapi() = Sys.isapple() ? Mantle.MetalAPI() : Mantle.VulkanAPI()

"""
    farmgpus(path)

Write the GPUs a farm renderer could use here to `path`, as TOML: what a farm
client offers, by the index `farmchild`'s `device` takes. Each is tried: a GPU
that cannot make a device (an integrated one lacking a feature the renderer
needs, say) carries the reason as its `problem`.
"""
function farmgpus(path::AbstractString)
    api = farmapi()
    gpus = map(Mantle.devices(api)) do d
        problem = try
            Mantle.Device(api; select = d.index)
            ""
        catch e
            # any failure makes the GPU unusable; the reason goes to whoever offers it
            first(split(sprint(showerror, e), '\n'))
        end
        Dict{String, Any}("index" => d.index, "name" => d.name, "kind" => string(d.kind), "problem" => problem)
    end
    open(io -> TOML.print(io, Dict("gpu" => gpus)), path, "w")
    return nothing
end

"""
    farmchild(directory; device = nothing, input = stdin, output = stdout)

A farm renderer serving the job in `directory` over a pipe: what a farm client
runs for each GPU, in the job's environment, on the GPU with index `device`
(see `farmgpus`; the default GPU without one). Answers each `frames` request
with one message per frame (the PNG as payload), or an error message for a
frame that fails, until `close` or the end of input.
"""
function farmchild(directory::AbstractString; device = nothing, input::IO = stdin, output::IO = stdout)
    # what scene code prints goes to the client's log, not into the messages
    redirect_stdout(stderr)
    device === nothing || Mantle.defaultdevice!(Mantle.Device(farmapi(); select = device))
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
