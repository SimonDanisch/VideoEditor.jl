# Live scenes and render farms

A film is an ordinary saved `.videoedit` sequence. Scene clips, footage, effects,
cuts, transitions and audio all go through `renderframe!`, the editor's existing
finished-output compositor. The farm distributes timeline frames; it does not
implement another movie renderer.

Preview quality is an editor-session preference. Farm frames always use the
project canvas, final sample budget and full procedural detail. Recipes with an
optional `preview!(pixel_scale)` detail callback receive `1.0` before final
animation evaluation. Reduced previews do not change saved artistic controls.

## A procedural scene clip

The animation file defines a factory with an absolute source-frame updater:

```julia
using Makie
function buildscene(canvas, args)
    scene = Scene(; size=canvas, camera=cam3d!)
    subject = mesh!(scene, Sphere(Point3f(0), 1f0); name=:subject, color=:orange)
    update! = (frame, fps) -> translate!(subject, sin(frame / fps), 0, 0)
    return (scene=scene, update! = update!)
end
```

The updater must support seeking backwards and frames arriving in any order.
Procedural animation runs before the clip's editor keyframes. Named plots expose
translation, rotation, scale and editable attributes; camera and effect curves can override
the script. The scene is built lazily and retained across cuts. Saved sources and
colour grades are shared resources rather than copies in every shot.

The [scene editing guide](scene-editing.md) describes grading, numeric camera
controls, grouped objects, keyframes and restoring the original animation.

```julia
using VideoEditor, RayMakie
VideoEditor.usebackend!(RayMakie)
build = programscene("scene.jl")
clip = sceneclip(VideoEditor.buildscene(build); build, frames=240,
                 canvas=(1920,1080), framerate=24)
clip.source.bakewith = :RayMakie
clip.source.bakescreenopts = Dict{Symbol,Any}(:rasterize=>false, :samples=>100,
    :denoise=>false, :device=>"Mantle.defaultbackend()")
clip.source.soundtrack = abspath("audio.wav")   # optional; follows source trims
seq = Sequence([clip],24)
saveproject("movie.videoedit",seq)
```

The script and its indirect includes/assets must be declared as render inputs.
Scene code is executable Julia, just like the animation project's other code.
The procedural choreography remains code; this adapter does not convert every
motion or expression into editor keyframes.

## Jobs and workers

```julia
job = renderjob("movie.videoedit", "renders/job-001";
                inputs=["included.jl", "model.glb"])
renderer = farmrenderer(job)
try
    workers = [FarmWorker("local GPU", ids -> farmframes!(renderer,ids))]
    renderfarm!(job,workers)
    encodefarm!(job.directory,"movie.mp4"; crf=17, audio=true)
finally
    closefarm!(renderer)
end
```

Prepare the same ordinary Julia Pkg environment on each worker, with VideoEditor,
RayMakie and the scene's imports. `Pkg.instantiate()` installs the Manifest's
versions; Julia manages compilation caches. Cold startup includes compilation.
Keep this environment across jobs, and pin/release a shared runtime when its
versions are stable. The farm does not resolve package dependencies, manage
compiled caches, or install models.

`bundlefarm(job, "renders/job-001.bundle"; root=projectroot)` copies only the job
and declared animation inputs, preserving relative paths. Sync that directory;
in the worker's prepared environment use `openfarmbundle(worker_bundle)`.
Packages remain in the worker's environment. Select the GPU and register RayMakie
before opening the renderer. Each GPU needs a separate process/session.

Transport callbacks accept indices and return `farmframes!` results: PNG bytes,
job identity and timings. Scenes, decoders and GPU objects stay in the worker.
RPCs can run on arbitrary threads; rendering is routed to the GPU's owning thread.
A callable worker may implement `Base.close`, which the coordinator invokes at
completion, pause or failure. Plain functions leave resource ownership to callers.

## BonitoAgents and the editor button

Include `examples/bonito_farm.jl` in a BonitoAgents session. Its adapter transfers
job inputs through `remote_session` and uses existing worker environments:

`remote_session` is injected into the chat's primary BonitoAgents Julia session.
It opens Julia RPC over the existing worker relay and server connection; it
starts the target environment if necessary. The coordinator calls it directly,
so rendering needs no `bt_julia_eval` tool calls or agent turns. Both Julia
sessions must have the same Julia version. The project's remote-Julia setting
must permit the connection.
Remote eval hosts answer these RPCs; they cannot initiate another worker
connection through the same chat relay.

A plain Julia REPL does not get this connection from `using VideoEditor`.
Standalone editor connection setup is not implemented yet. An application that
already has a BonitoAgents connection can provide its equivalent session factory
as `bonitofarm(...; sessionfor=connected_session_factory)`. The farm renderer and
job scheduler are independent of the transport.

```julia
slots = [(name="Bosgame GPU", worker="Bosgame",
    directory="/home/sim/Programmieren/VideoEditor-farm/jobs",
    env_path="/home/sim/Programmieren/VideoEditor-renderfarm-20261004/demo/editor_env",
    selector=nothing, resource="bosgame-gpu")]
registerfarm!(:shared, "shared GPU farm";
    directory=joinpath(projectroot,"renders","farm"),
    inputs=scene_inputs,
    connect=job -> bonitofarm(job; root=projectroot,slots))
```

`inputs` is a file list or a function of the saved project path. Both projects
should use the same slot definitions and GPU `resource` names. Standard PID locks
in `~/.videoeditor/farm-locks` prevent simultaneous ownership of a GPU. Busy/offline
slots are skipped. Separate directories and environments identify different GPUs
on the same machine. The adapter closes renderers and releases locks after dispatch.

**Export → Render on farm…** and **Ctrl+P → Render on shared GPU farm** invoke the
same registered command. It snapshots the edit, shows progress, then uses the
Export panel's chosen file, H.264/H.265, quality, preset and audio settings. Farm
output supports MP4/MKV/MOV. GIF still uses the local export action.

## Resume and verification

Faster workers take more frames. A failed worker retries unfinished requests;
completed PNGs have atomic commits and checksummed receipts. Run `renderfarm!`
again on the same job to resume. Different documents, settings, declared inputs,
sidecar contents or Julia versions cannot be mixed into an existing job.

`pausefarm!(job.directory)` stops new dispatch after current frames finish.
Remove `pause.requested` before resuming. `farmstatus` reads progress/errors;
`frames/*.json` records worker and render time per frame. A directory lock prevents
two coordinators from writing one job. After a coordinator process is killed,
remove its stale `coordinator.lock` only after confirming it is no longer running.

Encoding requires every frame receipt, checks output dimensions and decoded frame
count, checks audio presence when requested, and fully decodes the result before
publishing it. Scene soundtracks mix on their source clocks, including cuts and
speed changes. Rendered narration is saved as binary PCM; synthesize it before
preparing a job so workers need no speech model.
