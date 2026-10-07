# Include in a BonitoAgents Julia session. The editor itself stays independent
# of BonitoAgents: it accepts the same frame callback from any transport.
using VideoEditor
import Tar

struct RemoteFarmFrames
    session::Any
end
(c::RemoteFarmFrames)(indices) = c.session(ids ->
    Main.VideoEditor.onthread(Main.VideoEditor.mantlethread(Main.VideoEditor.Mantle)) do
        Base.invokelatest(Main.VideoEditor.farmframes!, Main.videoeditor_farm_renderer, ids)
    end, indices)
function Base.close(c::RemoteFarmFrames)
    c.session(() -> begin
        try
            Base.invokelatest(Main.VideoEditor.closefarm!, Main.videoeditor_farm_renderer)
        finally
            close(Main.videoeditor_farm_lease)
        end
        nothing
    end)
end

"""
    bonitofarm(job; root, slots, sessionfor=nothing)

Connect a portable job to BonitoAgents workers. Each slot has `name`, `worker`,
`directory` (job storage), `env_path` (an ordinary, prepared Julia environment),
`selector` (a Mantle device selector), and `resource` (a stable GPU name shared by
all projects). Prepare/instantiate that environment once using Pkg.
Each slot must have its own directory/session, including two GPUs on one machine.

A per-GPU PID lock prevents another project from claiming the same slot. Busy or
unavailable slots are skipped; the remaining slots can finish the job. Worker
resources/locks close automatically at the end of renderfarm!, even on failure.

`sessionfor(worker; env_path=nothing)` supplies the connected RPC session. The
default uses BonitoAgents' injected `Main.remote_session` in the chat's primary
Julia session, not merely `using VideoEditor` in a REPL. Remote eval hosts answer
RPCs; they do not initiate further worker connections through this relay.
No `bt_julia_eval` tool calls are made during dispatch. An application with its
own BonitoAgents connection can pass its session factory explicitly.
"""
function bonitofarm(job; root, slots, sessionfor=nothing)
    if sessionfor === nothing
        isdefined(Main, :remote_session) || error(
            "no BonitoAgents connection: run the editor in the chat's primary " *
            "Julia session, or pass sessionfor from your application's connection")
        sessionfor = Main.remote_session
    end
    bundle = job.directory * ".bundle"
    isdir(bundle) || bundlefarm(job, bundle; root)
    io = IOBuffer(); Tar.create(bundle, io); archive = take!(io)
    workers = VideoEditor.FarmWorker[]
    unavailable = String[]
    for slot in slots
        destination = joinpath(slot.directory, job.identity)
        try
            # The bootstrap session needs only stdlibs, before the job's env exists.
            bootstrap = sessionfor(slot.worker)
            # Reads reconnect safely after a worker restart; mutating RPC calls
            # deliberately never retry, since they may already have executed.
            bootstrap[:VERSION]
            bootstrap((destination, archive) -> begin
                isfile(joinpath(destination, "bundle.json")) && return nothing
                Core.eval(Main, :(import Tar))
                mkpath(dirname(destination))
                mktempdir(dirname(destination)) do temp
                    Base.invokelatest(Main.Tar.extract, IOBuffer(archive), temp)
                    mv(temp, destination)
                end
                nothing
            end, destination, archive)
            session = sessionfor(slot.worker; env_path = slot.env_path)
            session[:VERSION]
            session((destination, selector, resource) -> begin
                # RPC callbacks may arrive on any thread. GLFW is imported by
                # VideoEditor and must initialize on thread 1, even headlessly.
                # Pin the cold bootstrap before the editor's routing API exists.
                bootstrap = Task() do
                    Core.eval(Main, :(using FileWatching, VideoEditor, RayMakie))
                    Base.invokelatest() do
                        leasepath = joinpath(homedir(), ".videoeditor", "farm-locks", resource * ".pid")
                        mkpath(dirname(leasepath))
                        lease = Main.FileWatching.Pidfile.mkpidlock(leasepath; wait = false)
                        try
                            Main.VideoEditor.usebackend!(Main.RayMakie)
                            device = Main.VideoEditor.Mantle.device(selector)
                            Main.VideoEditor.Mantle.defaultdevice!(device)
                            renderer = Main.VideoEditor.openfarmbundle(destination)
                            Core.eval(Main, :(videoeditor_farm_renderer = $renderer))
                            Core.eval(Main, :(videoeditor_farm_lease = $lease))
                        catch
                            close(lease)
                            rethrow()
                        end
                    end
                    nothing
                end
                bootstrap.sticky = true
                ccall(:jl_set_task_tid, Cint, (Any, Cint), bootstrap, 0)
                schedule(bootstrap)
                fetch(bootstrap)
            end, destination, slot.selector, slot.resource)
            push!(workers, FarmWorker(slot.name, RemoteFarmFrames(session)))
        catch e
            push!(unavailable, "$(slot.name): $(sprint(showerror,e))")
            @warn "farm slot unavailable" slot = slot.name exception = e
        end
    end
    isempty(workers) && error("no available farm slots" *
        (isempty(unavailable) ? "" : " — " * join(unavailable, "; ")))
    return workers
end

# Example (use the same resource names in both movie projects):
# slots = [(name="Bosgame GPU", worker="Bosgame",
#           directory="/home/sim/Programmieren/VideoEditor-farm/bosgame",
#           env_path="/home/sim/Programmieren/VideoEditor-renderfarm-20261004/demo/env",
#           selector=nothing, resource="bosgame-gpu")]
# registerfarm!(:shared, "shared GPU farm";
#     directory=joinpath(projectroot, "renders", "farm"), inputs=scene_inputs,
#     connect=job -> bonitofarm(job; root=projectroot, slots))
# Now choose Export → Render on farm… or Ctrl+P → Render on shared GPU farm.
