# The transport is supplied by the caller. A BonitoAgents remote_session,
# Distributed worker, or a local renderer all expose the same frame callback.
# No GPU object, decoder, or live Makie scene crosses that boundary.

struct FarmWorker{F}
    name::String
    render::F
end
FarmWorker(name::AbstractString, render) = FarmWorker{typeof(render)}(String(name), render)
closeworker!(worker::FarmWorker) = applicable(close, worker.render) ? close(worker.render) : nothing

"A saved edit and its verified render settings, independent of any machine or transport."
struct RenderJob
    directory::String
    identity::String
    canvas::Tuple{Int, Int}
    total::Int
    fps::Float64
    inputs::Dict{String, String}
end

function RenderJob(directory::AbstractString)
    d = JSON.parsefile(joinpath(directory, "job.json"))
    return RenderJob(abspath(directory), String(d["identity"]), Tuple(Int.(d["size"])),
                     Int(d["total"]), Float64(d["fps"]), Dict{String, String}(d["inputs"]))
end
jobproject(job::RenderJob) = joinpath(job.directory, "project.videoedit")

"""
    renderjob(project, directory; size=nothing, inputs=[]) -> RenderJob

Snapshot the saved project and its sidecars, and checksum the render inputs.
Declare indirectly loaded scripts/assets in `inputs`. Sync the job and those
files to workers, then open `farmrenderer(job_directory; pathmap=...)` there.
The job identity rejects differing code/assets/settings and unsafe resumes.
"""
function renderjob(project::AbstractString, directory::AbstractString;
                   size = nothing, inputs = String[])
    seq = loadproject(project)
    any(n -> !isempty(strip(n.text)) && isempty(n.samples), seq.narration) &&
        error("render narration before preparing the farm job")
    total = seqlength(seq)
    total > 0 || error("empty sequence")
    canvas = something(size, canvassize(seq))
    files = farminputs(seq, inputs)
    identity, hashes = farmidentity(project, canvas, files)
    mkpath(directory)
    path = joinpath(directory, "job.json")
    if isfile(path)
        job = RenderJob(directory)
        job.identity == identity || error("render inputs changed; use a new job directory")
        return job
    end
    snapshot = joinpath(directory, "project.videoedit")
    cp(project, snapshot; force = true)
    for suffix in (".mattes", ".bakes")
        ispath(project * suffix) && cp(project * suffix, snapshot * suffix; force = true)
    end
    farmwritejson(path, Dict("identity" => identity, "size" => collect(canvas), "total" => total,
                            "fps" => seq.framerate, "julia_version" => string(VERSION), "inputs" => hashes))
    return RenderJob(directory)
end

mutable struct FarmRenderer
    sequence::Sequence
    canvas::RGBFrame
    black::RGBFrame
    readers::Dict{String, Any}
    engine::FxEngine
    identity::String
    lock::ReentrantLock
    closed::Bool
end
function farmthread(renderer::FarmRenderer)
    queue = Mantle.batchqueue(enginedevice(renderer.engine))
    return hasproperty(queue, :thread) ? queue.thread - 1 : 0
end

"Rebase project paths while loading the same saved document on another machine."
function farmpath(path::AbstractString, pathmap::AbstractDict)
    p = replace(String(path), '\\' => '/')
    # Longest first: a mapping for an asset folder overrides one for the project.
    for (src, dst) in sort!(collect(pathmap); by = kv -> -length(first(kv)))
        prefix = rstrip(replace(String(src), '\\' => '/'), '/')
        p == prefix && return String(dst)
        startswith(p, prefix * "/") &&
            return joinpath(String(dst), split(p[length(prefix)+2:end], '/')...)
    end
    return String(path)
end

function farmrebase(x::AbstractDict, pathmap)
    return Dict(k => farmrebase(v, pathmap) for (k, v) in x)
end
farmrebase(x::Vector{Any}, pathmap) = [farmrebase(v, pathmap) for v in x]
farmrebase(x::AbstractString, pathmap) = farmpath(x, pathmap)
farmrebase(x, pathmap) = x

"Explicit file dependencies of a saved sequence, plus caller-declared includes and assets."
function farminputs(seq::Sequence, extra)
    files = String[String(p) for p in extra]
    for n in seq.narration
        isempty(n.speech.reference) || push!(files,n.speech.reference)
    end
    for c in seq.clips
        for fx in c.effects, p in fx.params
            p.input === nothing && continue
            for ref in p.input.inputs
                ref isa RecordedRef && push!(files,ref.path)
            end
        end
        for p in (sourcepath(c.source), audiopath(c.source))
            isempty(p) || push!(files, p)
        end
        c.source isa SceneSource || continue
        c.source.root isa SceneProgram && isempty(c.source.root.package) && push!(files, c.source.root.file)
        c.source.build isa AbstractDict && append!(files,String.(collect(values(get(c.source.build,"recordings",Dict())))))
    end
    return sort!(unique(files))
end

filehash(path) = open(SHA.sha256, path) |> bytes2hex

function farmidentity(project, canvas, files; pathmap = Dict())
    hashes = Dict(p => filehash(farmpath(p, pathmap)) for p in files)
    sidecars = Tuple{String, String}[]
    for suffix in (".mattes", ".bakes")
        directory = project * suffix
        isdir(directory) || continue
        for (parent, _, names) in walkdir(directory), name in names
            file = joinpath(parent, name)
            push!(sidecars, (suffix * "/" * replace(relpath(file, directory), '\\' => '/'), filehash(file)))
        end
    end
    document = isempty(sidecars) ? filehash(project) :
               bytes2hex(SHA.sha256(MsgPack.pack((filehash(project), sort!(sidecars)))))
    # Package recipes travel as code in each worker's environment, not as job
    # files: their content is part of what a frame is, so it is part of the job.
    packages = [(name, packagehash(name)) for name in projectpackages(project)]
    # Sort all input pairs: Dict iteration order is not a transport protocol.
    value = (document, canvas, string(VERSION), [(p, hashes[p]) for p in sort!(collect(keys(hashes)))])
    isempty(packages) || (value = (value..., packages))
    return bytes2hex(SHA.sha256(MsgPack.pack(value))), hashes
end

"The packages a saved project's scene recipes are loaded from, sorted."
function projectpackages(project::AbstractString)
    dict = MsgPack.unpack(read(project))
    names = String[]
    for sd in get(dict, "sources", Any[])
        scene = get(sd, "scene", nothing)
        scene isa AbstractDict || continue
        build = get(scene, "build", nothing)
        build isa AbstractDict && haskey(build, "package") && push!(names, String(build["package"]))
    end
    return sort!(unique(names))
end

"""
    packagehash(name) -> String

The content of a recipe package as a worker loads it: its `Project.toml` and
every file under `src/` and `ext/`, by relative path. Equal on two machines
exactly when they would build the same scenes.
"""
function packagehash(name::AbstractString)
    dir = pkgdir(recipemodule(name))
    files = [joinpath(dir, "Project.toml")]
    for sub in ("src", "ext")
        root = joinpath(dir, sub)
        isdir(root) || continue
        for (parent, _, names) in walkdir(root), file in names
            push!(files, joinpath(parent, file))
        end
    end
    entries = sort!([(replace(relpath(f, dir), '\\' => '/'), filehash(f)) for f in files])
    return bytes2hex(SHA.sha256(MsgPack.pack(entries)))
end

"""
    farmrenderer(job_directory; pathmap=Dict(), backend=Mantle.defaultbackend())

Open one persistent renderer for a saved VideoEditor project. Register the
scene's final backend (`usebackend!(RayMakie)`) and select the GPU beforehand.
The job declares included code and assets, with `pathmap` translating
coordinator roots to the worker's local roots.
The compositor uses `backend`; scene rendering uses the saved final settings.
"""
function farmrenderer(job::RenderJob; pathmap = Dict(), backend = Mantle.defaultbackend())
    project, canvas = jobproject(job), job.canvas
    id, _ = farmidentity(project, canvas, sort!(collect(keys(job.inputs))); pathmap)
    id == job.identity || error("worker's project, Julia version, or input files differ from the job")
    seq = loadproject(project; pathmap)
    return FarmRenderer(seq, zeros(RGB{N0f8}, canvas...), zeros(RGB{N0f8}, canvas...),
                        Dict{String, Any}(), FxEngine(backend), id, ReentrantLock(), false)
end
farmrenderer(directory::AbstractString; kw...) = farmrenderer(RenderJob(directory); kw...)

"Return only PNG bytes and measurements; all GPU state stays in the renderer's process."
function farmframes!(renderer::FarmRenderer, indices)
    return lock(renderer.lock) do
        renderer.closed && error("farm renderer is closed")
        # Remote RPCs may arrive on any Julia thread. Graph construction also
        # uploads GPU parameters, so route the whole frame, not only the scene.
        onthread(farmthread(renderer)) do
            withfinalscenes(renderer.sequence) do
                map(indices) do n
                    0 <= n < seqlength(renderer.sequence) || error("invalid farm frame $n")
                    started = time_ns()
                    renderframe!(renderer.canvas, renderer.sequence, n, renderer.readers,
                                 renderer.engine; black = renderer.black)
                    io = IOBuffer()
                    # The editor's canvas is x×y, PNG is row×column. This is also
                    # writeframe!'s orientation conversion for VideoIO.
                    PNGFiles.save(io, permutedims(renderer.canvas, (2, 1)))
                    (frame = Int(n), png = take!(io), seconds = (time_ns()-started)/1e9,
                     identity = renderer.identity)
                end
            end
        end
    end
end

function closefarm!(r::FarmRenderer)
    lock(r.lock) do
        r.closed && return nothing
        onthread(farmthread(r)) do
            foreach(close, values(r.readers))
            emptyengine!(r.engine)
            for s in unique(c.source for c in r.sequence.clips if c.source isa SceneSource)
                s.live === nothing && continue
                closeretired!(s.live)
                onthread(renderthread(getbackend(s.live.backend))) do
                    applicable(close, s.live.screen) && close(s.live.screen)
                end
                s.live = nothing
            end
            r.closed = true
        end
    end
    return nothing
end

function farmwritejson(path, value)
    tmp = path * ".part"
    open(io -> JSON.print(io, value, 2), tmp, "w")
    mv(tmp, path; force = true)
end
farmframepath(dir, n) = joinpath(dir, "frames", string(n; pad = 6) * ".png")
farmreceiptpath(dir, n) = joinpath(dir, "frames", string(n; pad = 6) * ".json")

function farmcomplete(dir, n, identity)
    path, receipt = farmframepath(dir, n), farmreceiptpath(dir, n)
    isfile(path) && isfile(receipt) || return false
    try
        d = JSON.parsefile(receipt)
        return d["identity"] == identity && d["sha256"] == filehash(path)
    catch
        return false
    end
end

farmstatus(dir::AbstractString) = JSON.parsefile(joinpath(dir, "status.json"))
pausefarm!(dir::AbstractString) = write(joinpath(dir, "pause.requested"), "pause after in-flight frames\n")

"""
    renderfarm!(job, workers; frames=nothing, progress=nothing)

Distribute timeline frames dynamically, one request at a time per worker.
Each `FarmWorker(name, callback)` accepts frame indices and returns the result
of `farmframes!`. Faster workers take more work; a failed worker retries up to
`retries` times and returns unfinished frames to the queue. Workers must each
own a separate process/session when using GPUs.
The callback may be a callable resource implementing `close`; it is closed when
dispatch finishes, including on pause or failure. Plain functions have no cleanup.

PNG commits and checksummed receipts are atomic. Resume the same call and
directory to keep completed frames. Changed projects/settings/declared inputs
are rejected rather than mixed with older output. `pausefarm!` stops dispatch
after in-flight requests finish; remove `pause.requested` to resume.
This renders frames only; `encodefarm!` performs the final encode and audio mix.
"""
function renderfarm!(job::RenderJob, workers; frames = nothing,
                     retries::Integer = 3, progress = nothing)
    isempty(workers) && error("no farm workers supplied")
    retries > 0 || error("retries must be positive")
    names = [w.name for w in workers]
    length(unique(names)) == length(names) || error("farm worker names must be unique")
    dir, total, canvas, identity = job.directory, job.total, job.canvas, job.identity
    wanted = frames === nothing ? collect(0:total-1) : sort!(unique(Int.(collect(frames))))
    all(n -> 0 <= n < total, wanted) || error("invalid farm frame range")
    mkpath(dir)
    # An atomic mkdir is a process-wide coordinator lock, including other chats.
    lockdir = joinpath(dir, "coordinator.lock")
    try
        mkdir(lockdir)
    catch
        error("farm directory already has a coordinator: $lockdir")
    end
    try
        actual, _ = farmidentity(jobproject(job), canvas, sort!(collect(keys(job.inputs))))
        actual == identity || error("job input files changed; prepare a new job")
        mkpath(joinpath(dir, "frames"))
        pending = reverse([n for n in wanted if !farmcomplete(dir, n, identity)])
        completed = length(wanted) - length(pending)
        counts = Dict(name => 0 for name in names)
        errors = Dict{String, String}()
        active = Set{String}()
        guard = ReentrantLock()
        started = time_ns()
        function report()
            data = Dict("identity" => identity, "completed" => completed, "total" => length(wanted),
                        "timeline_frames" => total, "workers" => counts, "active" => collect(active),
                        "errors" => copy(errors), "elapsed_seconds" => (time_ns()-started)/1e9,
                        "paused" => isfile(joinpath(dir, "pause.requested")))
            farmwritejson(joinpath(dir, "status.json"), data)
            progress === nothing || progress(completed, length(wanted))
        end
        report()
        @sync for worker in workers
            @async begin
                failures = 0
                lock(guard) do; push!(active, worker.name); report(); end
                try
                    while true
                        ids = lock(guard) do
                            isempty(pending) && return Int[]
                            isfile(joinpath(dir, "pause.requested")) && return Int[]
                            [pop!(pending)]
                        end
                        isempty(ids) && break
                        n = only(ids)
                        committed = false
                        try
                            result = worker.render(ids)
                            length(result) == 1 || error("worker returned an incomplete request")
                            r = only(result)
                            r.frame == n || error("worker returned the wrong frame")
                            r.identity == identity || error("worker loaded a different project or inputs")
                            img = PNGFiles.load(IOBuffer(r.png))
                            Base.size(img) == reverse(canvas) || error("worker returned the wrong dimensions")
                            path = farmframepath(dir, n)
                            write(path * ".part", r.png); mv(path * ".part", path; force = true)
                            farmwritejson(farmreceiptpath(dir, n), Dict("identity" => identity,
                                "sha256" => bytes2hex(SHA.sha256(r.png)), "worker" => worker.name,
                                "frame" => n, "seconds" => r.seconds, "completed_utc" => string(now(UTC))))
                            committed = true
                        catch e
                            lock(guard) do
                                push!(pending, n); failures += 1
                                errors[worker.name] = sprint(showerror, e); report()
                            end
                            failures >= retries && break
                        end
                        # Reporting failures must never retry a frame that has
                        # already been committed and counted.
                        if committed
                            lock(guard) do
                                counts[worker.name] += 1; completed += 1; failures = 0
                                delete!(errors, worker.name); report()
                            end
                        end
                        yield()
                    end
                finally
                    lock(guard) do; delete!(active, worker.name); report(); end
                end
            end
        end
        if !isempty(pending) && !isfile(joinpath(dir, "pause.requested"))
            error("farm incomplete: $(length(pending)) frames remain; inspect status.json and resume")
        end
        return farmstatus(dir)
    finally
        for worker in workers
            try
                closeworker!(worker)
            catch e
                @warn "farm worker cleanup failed" worker = worker.name exception = e
            end
        end
        rm(lockdir; recursive = true)
    end
end

"Encode all committed timeline frames, mix the saved edit's audio, and verify the full decode."
function encodefarm!(dir::AbstractString, output::AbstractString; crf::Integer = 17,
                     preset::AbstractString = "slow", audio::Bool = true,
                     codec_name::AbstractString = "libx264")
    extension = lowercase(splitext(output)[2])
    extension in (".mp4", ".mkv", ".mov") || error("farm export supports MP4, MKV and MOV")
    job = JSON.parsefile(joinpath(dir, "job.json"))
    total = job["total"]
    all(n -> farmcomplete(dir, n, job["identity"]), 0:total-1) ||
        error("farm is missing committed frames")
    seq = loadproject(joinpath(dir, "project.videoedit"))
    files = sort!(collect(keys(job["inputs"])))
    identity, _ = farmidentity(joinpath(dir, "project.videoedit"), Tuple(job["size"]), files)
    identity == job["identity"] || error("farm input files changed before encoding")
    mkpath(dirname(abspath(output)))
    mktempdir(dirname(abspath(output))) do tmpdir
        silent, muxed = joinpath(tmpdir, "video" * extension), joinpath(tmpdir, "muxed" * extension)
        ff, probe = FFMPEG_jll.ffmpeg(), FFMPEG_jll.ffprobe()
        rate = rationalize(job["fps"]; tol = 1e-6)
        fps = string(numerator(rate), '/', denominator(rate))
        flags = extension == ".mkv" ? `` : `-movflags +faststart`
        run(`$ff -v error -y -framerate $fps -start_number 0
             -i $(joinpath(dir, "frames", "%06d.png")) -frames:v $total
             -c:v $codec_name -preset $preset -crf $crf -pix_fmt yuv420p $flags $silent`)
        wantaudio = audio && (any(hasaudio, unique(audiopath(c.source) for c in seq.clips)) ||
                             any(n -> !isempty(n.samples), seq.narration))
        result = wantaudio ? muxaudio(silent, seq, muxed) : silent
        metadata = JSON.parse(read(`$probe -v error -show_streams -show_format -of json $result`, String))
        video = only(filter(s -> s["codec_type"] == "video", metadata["streams"]))
        # Matroska does not populate nb_frames; counting decoded frames works
        # for all supported containers and catches truncated streams.
        count = read(`$probe -v error -count_frames -select_streams v:0
                      -show_entries stream=nb_read_frames -of csv=p=0 $result`, String)
        parse(Int, strip(count)) == total || error("encoded frame count mismatch")
        (video["width"], video["height"]) == Tuple(job["size"]) || error("encoded size mismatch")
        wantaudio && !any(s -> s["codec_type"] == "audio", metadata["streams"]) && error("audio missing")
        run(`$ff -v error -xerror -i $result -f null -`)
        mv(result, output; force = true)
        farmwritejson(joinpath(dir, "complete.json"), Dict("movie" => abspath(output),
                      "identity" => identity, "metadata" => metadata, "full_decode_passed" => true))
    end
    return output
end
