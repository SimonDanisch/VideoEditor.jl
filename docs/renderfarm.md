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

## Recording a simulation as animation tracks

A history-dependent simulation can run once, in source-frame order, before
editing or distributing its frames:

```julia
tracks = recordsceneanimation!(clip, joinpath(projectroot,"recordings","run-001"))
saveproject("movie.videoedit", seq)
```

The saved recipe must construct fresh simulation state. Recording uses a fresh
scene, leaving editor overrides separate. It discovers named plots' writable
numeric/array inputs, transforms, the scene's 3D camera and lights. Derived Makie
transformation matrices are excluded. On playback the scene is constructed once,
its simulation updater is bypassed, and the requested samples are restored before
editor overrides. The same recording is shared across cuts and uses their source
clock, including trims and speed changes. Farm bundles include and rebase the
recording files automatically.

Array tracks contain exact recorded samples, not scalar Bézier anchors. They
appear in the existing object inspector with their current shape and sample count,
and can be shown as timeline bands, cut and retimed. Scalar properties retain
the usual editable Bézier controls. Per-element array editing and interpolation
between simulation samples are not implemented. A missing source-frame sample
is an error; it is never replaced with a nearby frame.

For simulation outputs outside a scene, use the same recorder directly:

```julia
tracks = recordanimation(joinpath(projectroot,"recordings","wave-001"),0:239;
                         framerate=24) do frame,fps
    advance_simulation!(frame/fps)
    Dict(Symbol("wave.arg3")=>wave_height, Symbol("sensor.color")=>sensor_pixels)
end
```

`RecordedRef(track.path)` supplies the track through the existing parameter-input
graph. Numeric, point, colour and complex arrays are supported, including changing
shapes. Samples stream to binary files during recording; repeated consecutive
values share storage. Playback memory-maps the data and reads only the requested
sample. Completed recordings are immutable; use a new directory when simulation
inputs change. Keep those files with the project.

Recording captures native supported inputs, rather than arbitrary Julia state or
topology changes. Unnamed plots, changing plot topology, nested-camera discovery
and procedural control callbacks that recompute geometry are not converted by
this operation. Such controls need a new recording when their simulation inputs
change; editing a recorded native transform or attribute needs no resimulation.

## Jobs and workers

A job is a snapshot of everything its frames are made of, so a worker renders
exactly what the coordinator would, with nothing to compare afterwards:

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

`renderjob` writes `renders/job-001/job/`: the project and its sidecars, the
files it reads (soundtracks, recordings, scene scripts, and declared `inputs`;
files under `root`, the project's folder by default, keep their relative paths
so `include` works) and `environment/Project.toml`, the code. That Project.toml
pins every package the job loads (VideoEditor, the scenes' final backends, the
packages scene recipes come from, and everything they need) to what this
machine runs: a package developed from a git checkout is a `[sources]` entry
with its remote's `url`, the commit as `rev` and its `subdir` in a monorepo;
a registered package an exact `[compat]` version; Julia its exact version; and
the environment's package preferences come along. A checkout with uncommitted
changes or a commit on no remote branch cannot be fetched by a worker:
`renderjob` names every such package and stops. `portable = false` pins
developed packages by `path` instead, for farm clients on this machine only.

The same edit asked for again resumes its job; a changed edit, file or package
is refused for that directory (make a new job), so frames never mix.

Transport callbacks accept indices and return `farmframes!` results: PNG bytes
and timings. Scenes, decoders and GPU objects stay in the worker. RPCs can run
on arbitrary threads; rendering is routed to the GPU's owning thread. A callable
worker may implement `Base.close`, which the coordinator invokes at completion,
pause or failure. Plain functions leave resource ownership to callers.

## Farm clients: other machines and every GPU

The editor runs a farm server; every `Player` starts one and logs the command
that offers a machine's GPUs to it:

```sh
julia -m FarmClient all 192.168.178.92:7600
julia -m FarmClient 7900,NVIDIA 192.168.178.92:7600   # only these GPUs
```

`FarmClient` is a standard-library-only package in this repository. Install it
once per machine that renders, into the default environment:

```julia
import Pkg
Pkg.add(url = "https://github.com/SimonDanisch/VideoEditor.jl", subdir = "FarmClient")
```

The first argument picks GPUs: `all` discrete and integrated ones, or a
comma-separated list of indices or parts of their names. Run a client on the
editor's own machine as well for its GPUs. A client waits for the server and
reconnects when the editor restarts, so it can keep running.

```julia
workers = farmworkers(job)       # this session's server: farmserver()
renderfarm!(job, workers)
encodefarm!(job.directory, "movie.mp4")
```

`farmworkers` sends every connected client the job's snapshot. The client
unpacks it, instantiates `environment/Project.toml` with the job's Julia
version (`julia +<version>` through juliaup, or its own Julia if it is that
version; automatic precompilation off), asks the job's VideoEditor which GPUs
there are (`farmgpus`) and starts a renderer per GPU in it (`farmchild`, the
GPU chosen by index). Each renderer, once ready, connects back to the server and
joins the running render, so frames start on the first GPU while slower
machines still install or compile. Environments are kept by their Project.toml
under `~/.videoeditor/farm` and reused; packages and compiled code live in the
ordinary Julia depot. The first job on a machine installs and compiles, which
takes minutes. A GPU that cannot be opened is reported and left out. Closing a
GPU's worker, or a dropped connection, ends its renderer and frees the GPU; a
lock file per GPU keeps two jobs from sharing one.

Messages are TOML headers with byte payloads, not Julia serialization, so the
editor, client and renderer may run different Julia or VideoEditor versions.
The connections are plain TCP without authentication: a client renders what
the server it was pointed at sends, and the server hands every connected client
the job's files. Run the farm on a network you trust.

Every `Player` offers **Export → Render on farm…** and **Ctrl+P → Render on LAN
farm** (`registerlanfarm!`), enabled while a client is connected. It snapshots
the edit, renders on every offered GPU, shows progress, then uses the Export
panel's chosen file, H.264/H.265, quality, preset and audio settings. Farm
output supports MP4/MKV/MOV. GIF still uses the local export action.
`registerfarm!` registers any other transport the same way.

## Resume and verification

Faster workers take more frames. A failed worker retries unfinished requests;
completed PNGs have atomic commits and receipts naming the job and the PNG's
hash. Run `renderfarm!` again on the same job to resume.

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

## Backend selection and current cache limits

Farm rendering uses the scene's saved final backend and screen options. For
RayMakie movies, use GPU rasterization for editor previews and
`bakewith = :RayMakie` with `rasterize = false` for final frames. Loading GLMakie is not proof
that it renders the scene: VideoEditor currently imports it for its editor
window. The package's small `test/renderfarm.jl` fixture explicitly chooses
GLMakie to test scheduling, receipts, transfer and encoding; it does not verify
GPU ray tracing.

Resume currently reuses verified frames from the **same immutable job**. It does
not automatically reuse frames between edited documents. A job's snapshot holds
the complete project and its audio sidecars, so even an audio-only edit makes a
different job. An editor curve change invalidates its entire clip-level bake;
the scheduler does not calculate which curve segments changed. Supplying a frame
subset to `renderfarm!` is possible, but is not automatic edit invalidation.

`encodefarm!` currently encodes all verified PNGs again, then mixes/muxes audio.
There is no cached video-stream remux path for an audio-only edit. Independent
scene-frame, composite/video and speech caches are needed before an edited job
can safely reuse unaffected frames or replace only the soundtrack. Changes to
speech timing can also affect visuals when a scene follows narration cues.
