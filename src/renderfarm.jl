# A saved edit rendered frame by frame on any number of GPUs. A job is a
# snapshot of everything a frame is made of: the project, the files it reads and
# the environment of its code (see `farmenvironment`). Workers render that
# snapshot, so nothing on a worker has to be compared with the coordinator. The
# coordinator hands out frames, keeps what comes back, resumes, and encodes.
#
# The transport is the caller's: `FarmWorker`s are frame callbacks, local
# (`farmframes!`) or over the network (`farmworkers`, farm daemons on other
# machines). No GPU object, decoder or live Makie scene crosses that boundary.

struct FarmWorker{F}
    name::String
    render::F
end
FarmWorker(name::AbstractString, render) = FarmWorker{typeof(render)}(String(name), render)
closeworker!(worker::FarmWorker) = applicable(close, worker.render) ? close(worker.render) : nothing

"""The folder of a job directory that workers render: the snapshot."""
const FARM_SNAPSHOT = "job"

"""
    RenderJob(directory)

A saved edit made into a farm job (see [`renderjob`](@ref)). `directory/job` is
the snapshot every worker renders: the project and its sidecars, the files it
reads (`data/`, `inputs/`), `environment/Project.toml` (the code) and
`job.toml`. `key` names the snapshot's content: frames committed for another
snapshot are not this job's, so a changed edit can never mix with old frames.
"""
struct RenderJob
    directory::String
    key::String
    canvas::Tuple{Int, Int}
    total::Int
    fps::Float64
    julia::String
    backends::Vector{String}
    packages::Vector{String}          # the scene recipes': a renderer loads them before it renders
    relocations::Dict{String, String}
end

function RenderJob(directory::AbstractString)
    snapshot = joinpath(directory, FARM_SNAPSHOT)
    meta = TOML.parsefile(joinpath(snapshot, "job.toml"))
    return RenderJob(abspath(directory), snapshotkey(snapshot), Tuple(Int.(meta["size"])), Int(meta["total"]),
                     Float64(meta["fps"]), String(meta["julia"]), String.(meta["backends"]),
                     String.(meta["packages"]), Dict{String, String}(meta["relocations"]))
end

jobsnapshot(job::RenderJob) = joinpath(job.directory, FARM_SNAPSHOT)
jobproject(job::RenderJob) = joinpath(jobsnapshot(job), "project.videoedit")

"""What a snapshot holds: every file by relative path and content, hashed."""
function snapshotkey(snapshot::AbstractString)
    entries = Tuple{String, String}[]
    for (parent, _, names) in walkdir(snapshot), name in names
        file = joinpath(parent, name)
        push!(entries, (replace(relpath(file, snapshot), '\\' => '/'), filehash(file)))
    end
    return bytes2hex(SHA.sha256(MsgPack.pack(sort!(entries))))
end

filehash(path) = open(SHA.sha256, path) |> bytes2hex

"""
    renderjob(project, directory; inputs = [], root = dirname(project), size = nothing, portable = true) -> RenderJob

Snapshot the saved `project` into `directory/job`: the project and its
sidecars, the files it reads (soundtracks, recordings, scene scripts, and the
includes and assets declared in `inputs`), and the environment of its code
(`farmenvironment` of the packages it loads; `portable = false` for farm
daemons on this machine only). Files under `root` keep their relative paths,
so a script's `include`s still work; others are kept by content.

Narration must be rendered first, so workers need no speech model. Asked again
for the same directory, the same snapshot resumes the job; anything changed
(the edit, a file, a package) is refused: make a new job for it.
"""
function renderjob(project::AbstractString, directory::AbstractString; inputs = String[],
                   root::AbstractString = dirname(abspath(project)), size = nothing, portable::Bool = true)
    seq = loadproject(project)
    any(n -> !isempty(strip(n.text)) && isempty(n.samples), seq.narration) &&
        error("render narration before preparing the farm job")
    total = seqlength(seq)
    total > 0 || error("empty sequence")
    canvas = something(size, canvassize(seq))
    directory = abspath(directory)
    mkpath(directory)
    snapshot = joinpath(directory, FARM_SNAPSHOT)
    root = rstrip(abspath(root), ['/', '\\'])
    mktempdir(directory) do temp
        relocations = Dict{String, String}()
        for file in farminputs(seq, inputs)
            # outside the root, a file is kept by its content, a folder by its path
            kept = isfile(file) ? filehash(file) : bytes2hex(SHA.sha256(file))
            relative = startswith(file, root * "/") || startswith(file, root * "\\") ?
                       joinpath("data", relpath(file, root)) : joinpath("inputs", kept, basename(file))
            relocations[file] = relative
            mkpath(dirname(joinpath(temp, relative)))
            cp(file, joinpath(temp, relative); force = true)
        end
        # the project's files, under the root as a whole, are found there too
        relocations[root] = "data"
        cp(project, joinpath(temp, "project.videoedit"))
        for suffix in (".mattes", ".bakes")
            ispath(project * suffix) && cp(project * suffix, joinpath(temp, "project.videoedit" * suffix))
        end
        mkpath(joinpath(temp, "environment"))
        open(io -> TOML.print(io, farmenvironment(farmpackages(seq); portable)),
             joinpath(temp, "environment", "Project.toml"), "w")
        open(joinpath(temp, "job.toml"), "w") do io
            TOML.print(io, Dict{String, Any}("size" => collect(canvas), "total" => total, "fps" => seq.framerate,
                                             "julia" => string(VERSION), "backends" => farmbackends(seq),
                                             "packages" => recipepackages(seq), "relocations" => relocations))
        end
        if isdir(snapshot)
            snapshotkey(temp) == snapshotkey(snapshot) ||
                error("the edit, its files or its code changed since $directory was made: use a new job directory")
        else
            mv(temp, snapshot)
        end
    end
    return RenderJob(directory)
end

mutable struct FarmRenderer
    sequence::Sequence
    canvas::RGBFrame
    black::RGBFrame
    readers::Dict{String, Any}
    engine::FxEngine
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
    files = String[abspath(p) for p in extra]
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

"""
    farmrenderer(job; backend = Mantle.defaultbackend())

Open one persistent renderer for a job's snapshot: the edit loaded from it,
its files found in it. Register the scenes' final backend
(`usebackend!(RayMakie)`) and select the GPU beforehand. The compositor uses
`backend`; scenes render with the saved final settings.
"""
function farmrenderer(job::RenderJob; backend = Mantle.defaultbackend())
    snapshot = jobsnapshot(job)
    # A recording is a folder: its files map under it, longest first.
    pathmap = Dict(source => joinpath(snapshot, relative) for (source, relative) in job.relocations)
    seq = loadproject(jobproject(job); pathmap)
    return FarmRenderer(seq, zeros(RGB{N0f8}, job.canvas...), zeros(RGB{N0f8}, job.canvas...),
                        Dict{String, Any}(), FxEngine(backend), ReentrantLock(), false)
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
                    (frame = Int(n), png = take!(io), seconds = (time_ns()-started)/1e9)
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

"""Whether frame `n` is committed for the job `key`: its PNG, and a receipt (written atomically) naming the job and the PNG's hash."""
function farmcomplete(dir, n, key)
    path, receipt = farmframepath(dir, n), farmreceiptpath(dir, n)
    isfile(path) && isfile(receipt) || return false
    d = JSON.parsefile(receipt)
    return d["job"] == key && d["sha256"] == filehash(path)
end

farmstatus(dir::AbstractString) = JSON.parsefile(joinpath(dir, "status.json"))
pausefarm!(dir::AbstractString) = write(joinpath(dir, "pause.requested"), "pause after in-flight frames\n")

"""
    renderfarm!(job, workers; frames=nothing, progress=nothing)

Distribute timeline frames dynamically, one request at a time per worker.
Each `FarmWorker(name, callback)` accepts frame indices and returns the result
of `farmframes!`. Faster workers take more work; a failed worker retries up to
`retries` times and returns unfinished frames to the queue. Workers must each
own a separate process when using GPUs (see `farmworkers`).
The callback may be a callable resource implementing `close`; it is closed when
dispatch finishes, including on pause or failure. Plain functions have no cleanup.

PNG commits and receipts are atomic. Resume the same call and directory to keep
completed frames. `pausefarm!` stops dispatch after in-flight requests finish;
remove `pause.requested` to resume. This renders frames only; `encodefarm!`
performs the final encode and audio mix.
"""
function renderfarm!(job::RenderJob, workers; frames = nothing,
                     retries::Integer = 3, progress = nothing)
    isempty(workers) && error("no farm workers supplied")
    retries > 0 || error("retries must be positive")
    names = [w.name for w in workers]
    length(unique(names)) == length(names) || error("farm worker names must be unique")
    dir, total, canvas, key = job.directory, job.total, job.canvas, job.key
    wanted = frames === nothing ? collect(0:total-1) : sort!(unique(Int.(collect(frames))))
    all(n -> 0 <= n < total, wanted) || error("invalid farm frame range")
    # An atomic mkdir is a process-wide coordinator lock, including other editors.
    lockdir = joinpath(dir, "coordinator.lock")
    ispath(lockdir) && error("farm directory already has a coordinator: $lockdir")
    mkdir(lockdir)
    try
        mkpath(joinpath(dir, "frames"))
        pending = reverse([n for n in wanted if !farmcomplete(dir, n, key)])
        completed = length(wanted) - length(pending)
        counts = Dict(name => 0 for name in names)
        errors = Dict{String, String}()
        active = Set{String}()
        guard = ReentrantLock()
        started = time_ns()
        function report()
            data = Dict("job" => key, "completed" => completed, "total" => length(wanted),
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
                            r.frame == n || error("worker returned frame $(r.frame) for frame $n")
                            img = PNGFiles.load(IOBuffer(r.png))
                            Base.size(img) == reverse(canvas) || error("worker returned the wrong dimensions")
                            path = farmframepath(dir, n)
                            write(path * ".part", r.png); mv(path * ".part", path; force = true)
                            farmwritejson(farmreceiptpath(dir, n), Dict("job" => key,
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
    job = RenderJob(dir)
    total = job.total
    all(n -> farmcomplete(dir, n, job.key), 0:total-1) || error("farm is missing committed frames")
    # the snapshot's own copies of the soundtracks, as the frames were rendered from them
    snapshot = jobsnapshot(job)
    seq = loadproject(jobproject(job); pathmap = Dict(s => joinpath(snapshot, r) for (s, r) in job.relocations))
    mkpath(dirname(abspath(output)))
    mktempdir(dirname(abspath(output))) do tmpdir
        silent, muxed = joinpath(tmpdir, "video" * extension), joinpath(tmpdir, "muxed" * extension)
        ff, probe = FFMPEG_jll.ffmpeg(), FFMPEG_jll.ffprobe()
        rate = rationalize(job.fps; tol = 1e-6)
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
        (video["width"], video["height"]) == job.canvas || error("encoded size mismatch")
        wantaudio && !any(s -> s["codec_type"] == "audio", metadata["streams"]) && error("audio missing")
        run(`$ff -v error -xerror -i $result -f null -`)
        mv(result, output; force = true)
        farmwritejson(joinpath(dir, "complete.json"), Dict("movie" => abspath(output),
                      "job" => job.key, "metadata" => metadata, "full_decode_passed" => true))
    end
    return output
end
