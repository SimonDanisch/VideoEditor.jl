# A saved edit rendered frame by frame on any number of GPUs. A job is a
# snapshot of everything a frame is made of: the project, the files it reads and
# the environment of its code (see `farmenvironment`). Workers render that
# snapshot, so nothing on a worker has to be compared with the coordinator. The
# coordinator hands out frames, keeps what comes back, resumes, and encodes.
#
# The transport is the caller's: `FarmWorker`s are frame callbacks, local
# (`farmframes!`) or over the network (`farmworkers`, farm clients on other
# machines). No GPU object, decoder or live Makie scene crosses that boundary.

struct FarmWorker{F}
    name::String
    render::F
end
FarmWorker(name::AbstractString, render) = FarmWorker{typeof(render)}(String(name), render)
closeworker!(worker::FarmWorker) = applicable(close, worker.render) ? close(worker.render) : nothing

"Whether this worker can render a timeline frame; specialize on its callable resource."
acceptsfarmframe(resource, job, frame::Integer) = true
function nextfarmframes!(pending::Vector{Int}, worker::FarmWorker, job, batchsize::Integer = 16)
    i=findlast(n->acceptsfarmframe(worker.render,job,n),pending)
    i===nothing && return Int[]
    frames = [splice!(pending,i)]
    # Adjacent work preserves scene locality. Never lease a cached, already
    # leased or incompatible frame just to fill a batch.
    while length(frames) < batchsize && i > 1
        i -= 1
        pending[i] == last(frames)+1 || break
        acceptsfarmframe(worker.render,job,pending[i]) || break
        push!(frames,splice!(pending,i))
    end
    return frames
end

"""Consume a leased batch incrementally. Transports may stream instead of retaining a batch in RAM."""
foreachfarmresult(accept, resource, frames) = foreach(accept, resource(frames))

"""The folder of a job directory that workers render: the snapshot."""
const FARM_SNAPSHOT = "job"

"""
    RenderJob(directory)

A saved edit made into a farm job (see [`renderjob`](@ref)). `directory/job` is
the snapshot every worker renders: the project and its sidecars, the files it
reads (`data/`, `inputs/`), `environment/Project.toml` (the code) and
`job.toml`. `key` names the snapshot's content: frames committed for another
snapshot are not this job's, so a changed edit can never mix with old frames.
`planes`: whether workers return each frame's planes beside it (see
[`renderjob`](@ref)).
"""
struct RenderJob
    directory::String
    key::String
    canvas::Tuple{Int, Int}
    total::Int
    fps::Float64
    julia::String
    backends::Vector{String}
    packages::Vector{String}          # the recipes' and effects' packages: a renderer loads them before it renders (`loadpackages`)
    relocations::Dict{String, String}
    planes::Bool
end

function RenderJob(directory::AbstractString)
    snapshot = joinpath(directory, FARM_SNAPSHOT)
    meta = TOML.parsefile(joinpath(snapshot, "job.toml"))
    return RenderJob(abspath(directory), snapshotkey(snapshot), Tuple(Int.(meta["size"])), Int(meta["total"]),
                     Float64(meta["fps"]), String(meta["julia"]), String.(meta["backends"]),
                     String.(meta["packages"]), Dict{String, String}(meta["relocations"]),
                     Bool(get(meta, "planes", true)))
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
    renderjob(project, directory; inputs = [], root = dirname(project), size = nothing, portable = true,
              planes = true) -> RenderJob

Snapshot the saved `project` into `directory/job`: the project and its
sidecars, the files it reads (soundtracks, recordings, scene scripts, and the
includes and assets declared in `inputs`), and the environment of its code
(`farmenvironment` of the packages it loads; `portable = false` for farm
clients on this machine only). Files under `root` keep their relative paths,
so a script's `include`s still work; others are kept by content.

With `planes`, workers return each frame's track planes and a traced clip's
linear HDR source beside it, for the edit to adopt (`adoptfarmcache!`): a
traced 1080p frame's source is over 100 MB. `planes = false` returns finished
frames only, for a film rendered to be encoded (`encodefarm!`).

Narration must be rendered first, so workers need no speech model. Asked again
for the same directory, the same snapshot resumes the job; anything changed
(the edit, a file, a package) is refused: make a new job for it.
"""
function renderjob(project::AbstractString, directory::AbstractString; inputs = String[],
                   root::AbstractString = dirname(abspath(project)), size = nothing, portable::Bool = true,
                   planes::Bool = true)
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
                                             "packages" => loadpackages(seq), "relocations" => relocations,
                                             "planes" => planes))
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
    # where the sequence's track caches write the planes of the frame being
    # rendered, until `farmlayers!` takes them (see `farmcaches!`)
    planes::String
    # whether they are returned with the frame or only cleared (`renderjob`'s `planes`)
    keepplanes::Bool
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
farmrebase(x::AbstractVector{<:AbstractString}, pathmap) = [farmpath(v,pathmap) for v in x]
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
                append!(files,inputfiles(ref))
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
    return farmrenderer(seq;backend,directory=job.directory,canvas=job.canvas,keepplanes=job.planes)
end

"""
    farmrenderer(sequence; backend=Mantle.defaultbackend(), directory)

Open the same persistent timeline renderer directly on an existing sequence.
File inputs keep their saved addresses; this does not copy large volume files.
Use `renderjob` when a self-contained immutable farm snapshot is needed.
The caller owns the sequence and its input files for the renderer's lifetime.
"""
function farmrenderer(seq::Sequence; backend = Mantle.defaultbackend(),
                      directory::AbstractString = mktempdir(;cleanup=false),canvas=canvassize(seq),
                      keepplanes::Bool = true)
    mkpath(directory)
    planes = mktempdir(directory; prefix = "planes-", cleanup = false)
    farmcaches!(seq, planes)
    return FarmRenderer(seq, zeros(RGB{N0f8}, canvas...), zeros(RGB{N0f8}, canvas...),
                        Dict{String, Any}(), FxEngine(backend), ReentrantLock(), false, planes, keepplanes)
end

"""
    farmcaches!(seq, directory) -> seq

Give a worker's sequence track caches of its own, empty, in `directory`. The
project's caches name the coordinator's folder, which a worker does not have;
and a plane written into the snapshot would change the job's key, so a frame
rendered would make the job another one. What a worker renders into them it
hands back with the frame (`farmlayers!`); it composites every frame once, so it
keeps no composite cache.
"""
function farmcaches!(seq::Sequence, directory::AbstractString)
    isempty(seq.trackcaches) && return seq
    empty!(seq.trackcaches)
    enabletrackcache!(seq, directory)
    seq.compositecache = nothing
    return seq
end

"""
    FarmLayer(track, id, frame, png, hdr)

A plane a farm frame was composited from, as a worker returns it: the clip's
track and id, its source frame, the track plane as PNG bytes and, for a traced
clip, its linear HDR source as `.vehdr` bytes (empty for others). The
coordinator keeps them (`commitlayer!`) for the edit to take into its caches
(`adoptfarmcache!`), so neither the tracing nor the grading has to be done again.
"""
struct FarmLayer
    track::Int
    id::UInt64
    frame::Int
    png::Vector{UInt8}
    hdr::Vector{UInt8}
end

"""
    farmlayers!(seq, n) -> Vector{FarmLayer}

The planes timeline frame `n` was just composited from, taken out of the
worker's caches: a worker keeps no frames, and a traced clip's HDR source is
tens of megabytes a frame.
"""
function farmlayers!(seq::Sequence, n::Integer)
    layers = FarmLayer[]
    for clip in frameclips(seq, n)
        cache = trackcache(clip)
        cache === nothing && continue
        sf = sourceframe(clip, n)
        cachevalid(cache, clip.id, sf) || continue
        raw = cache.source
        hdr = raw !== nothing && cachevalid(raw, clip.id, sf) ? takeplane!(raw, clip.id, sf) : UInt8[]
        push!(layers, FarmLayer(clip.track, clip.id, sf, takeplane!(cache, clip.id, sf), hdr))
    end
    return layers
end

"""A cached plane's bytes, its file and its validity taken out of the cache."""
function takeplane!(c::TrackCache, id::UInt64, sf::Int)
    bytes = read(cachefile(c, id, sf))
    dropplane!(c, id, sf)
    return bytes
end

"""A cached plane's file and validity removed from the cache."""
function dropplane!(c::TrackCache, id::UInt64, sf::Int)
    rm(cachefile(c, id, sf))
    delete!(c.valid[id], sf)
    return c
end

"""
    dropplanes!(seq, n) -> FarmLayer[]

The planes timeline frame `n` was composited from, cleared from the worker's
caches and not returned: a job without planes keeps only its frames.
"""
function dropplanes!(seq::Sequence, n::Integer)
    for clip in frameclips(seq, n)
        cache = trackcache(clip)
        cache === nothing && continue
        sf = sourceframe(clip, n)
        cachevalid(cache, clip.id, sf) && dropplane!(cache, clip.id, sf)
        raw = cache.source
        raw !== nothing && cachevalid(raw, clip.id, sf) && dropplane!(raw, clip.id, sf)
    end
    return FarmLayer[]
end
farmrenderer(directory::AbstractString; kw...) = farmrenderer(RenderJob(directory); kw...)

"Return only PNG bytes, the planes they were made of, and measurements; all GPU state stays in the renderer's process."
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
                     layers = renderer.keepplanes ? farmlayers!(renderer.sequence, n) :
                                                   dropplanes!(renderer.sequence, n))
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
        rm(r.planes; recursive = true, force = true)
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

"""Where the coordinator keeps the planes workers return for `track`, laid out as a track cache and its HDR sources."""
farmplanes(dir, track) = TrackCache(joinpath(dir, "trackframes", "track-$track"))
farmsources(dir, track) = TrackCache(joinpath(dir, "trackframes", "track-$track", "source"); format = :hdr)

"""Write `bytes` to `path` atomically; their SHA-256."""
function commitbytes!(path::AbstractString, bytes::AbstractVector{UInt8})
    mkpath(dirname(path))
    write(path * ".part", bytes)
    mv(path * ".part", path; force = true)
    return bytes2hex(SHA.sha256(bytes))
end

"""Keep a returned plane in the job directory; its entry in the frame's receipt."""
function commitlayer!(dir, l::FarmLayer)
    entry = Dict{String, Any}("track" => l.track, "id" => string(l.id), "frame" => l.frame,
                              "sha256" => commitbytes!(cachefile(farmplanes(dir, l.track), l.id, l.frame), l.png))
    isempty(l.hdr) ||
        (entry["hdr_sha256"] = commitbytes!(cachefile(farmsources(dir, l.track), l.id, l.frame), l.hdr))
    return entry
end

"""
    adoptfarmcache!(seq, job) -> seq

Take the planes a farm job's workers returned into the track caches of `seq`,
the edit the job was made from: a plane is taken where its clip is in `seq` and
its frame has not been edited since the job was made (its cache revision is the
one the snapshot saved), and its file is the one the receipt names. A traced
clip's HDR source is taken by the same rule, so a grade changed after the job
keeps the traced source and renders only the derived plane. Adopt before
editing further: an edit made since is never overwritten, but the planes it
makes stale are not taken either.
"""
function adoptfarmcache!(seq::Sequence, job::RenderJob)
    saved = savedframecaches(jobproject(job))
    for n in 0:(job.total - 1)
        farmcomplete(job.directory, n, job.key) || continue
        for layer in get(JSON.parsefile(farmreceiptpath(job.directory, n)), "layers", ())
            id, sf, track = parse(UInt64, layer["id"]), Int(layer["frame"]), Int(layer["track"])
            clip = clipbyid(seq, id)
            clip === nothing && continue
            cache, then = trackcache(clip), get(saved, track, nothing)
            (cache === nothing || then === nothing) && continue
            if haskey(layer, "hdr_sha256") && cache.source !== nothing && then.source !== nothing &&
               samerevision(cache.source, then.source, id, sf)
                adoptplane!(cache.source, id, sf, cachefile(farmsources(job.directory, track), id, sf), layer["hdr_sha256"])
            end
            samerevision(cache, then, id, sf) &&
                adoptplane!(cache, id, sf, cachefile(farmplanes(job.directory, track), id, sf), layer["sha256"])
        end
    end
    return seq
end

"""The track caches a saved project describes, by track: only their bookkeeping, for the revisions it saved."""
function savedframecaches(path::AbstractString)
    d = get(decodeblocks(MsgPack.unpack(read(path))), "framecaches", nothing)
    d === nothing && return Dict{Int, TrackCache}()
    return Dict(parse(Int, t) => restorecache(meta, "") for (t, meta) in d["tracks"])
end

samerevision(a::TrackCache, b::TrackCache, id::UInt64, sf::Int) =
    get(a.revisions, (id, sf), UInt64(0)) == get(b.revisions, (id, sf), UInt64(0))

"""Adopt a verified immutable plane, sharing its storage when the filesystem permits."""
function adoptplane!(cache::TrackCache, id::UInt64, sf::Int, file::AbstractString, sha::AbstractString)
    isfile(file) && filehash(file) == sha || return false
    linkplane!(cache, id, sf, file)
    return true
end

"""Publish `file` as `cache`'s valid frame `sf`, sharing its storage when the filesystem permits."""
function linkplane!(cache::TrackCache, id::UInt64, sf::Int, file::AbstractString)
    target = cachefile(cache, id, sf)
    mkpath(dirname(target))
    # Cache writers publish new files by rename, never by modifying an existing
    # plane. A hard link therefore preserves both job receipts and later edits,
    # without doubling hundreds of GB of HDR masters. Stage the replacement so
    # an interrupted copy cannot destroy a previously valid cached frame.
    if !isfile(target) || !samefile(file, target)
        mktempdir(dirname(target)) do dir
            staged = joinpath(dir, basename(target))
            try
                hardlink(file, staged)
            catch err
                err isa Base.IOError || err isa SystemError || rethrow()
                cp(file, staged)
            end
            mv(staged, target; force = true)
        end
    end
    push!(get!(() -> BitSet(), cache.valid, id), sf)
    delete!(get!(() -> BitSet(), cache.dirty, id), sf)
    return target
end

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
of `farmframes!`. `workers` is a collection, or a `Channel` they arrive on while
the render runs (as `farmworkers` delivers them). Faster workers take more work;
a failed worker retries up to `retries` times and returns unfinished frames to
the queue. Workers must each own a separate process when using GPUs.
The callback may be a callable resource implementing `close`; it is closed when
dispatch finishes, including on pause or failure. Plain functions have no cleanup.

PNG commits and receipts are atomic. Resume the same call and directory to keep
completed frames. `pausefarm!` stops dispatch after in-flight requests finish;
remove `pause.requested` to resume. This renders frames only; `encodefarm!`
performs the final encode and audio mix.

With `reuse=seq`, native cached pictures/HDR masters are left in that project's
caches instead of being retraced in a new snapshot. `status["reused"]` counts
these frames, and the frames left out because another frame traces the same
pictures (see [`distinctframes`](@ref)). Adopt the returned planes and finish
with `exportvideo(output, seq)`;
`encodefarm!` alone cannot encode frames intentionally kept only in `seq`.
"""
function renderfarm!(job::RenderJob, workers; frames = nothing, reuse = nothing,
                     retries::Integer = 3, batchsize::Integer = 16, progress = nothing)
    workers isa Channel || isempty(workers) && error("no farm workers supplied")
    retries > 0 || error("retries must be positive")
    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    pool = workerchannel(workers)
    dir, total, canvas, key = job.directory, job.total, job.canvas, job.key
    wanted = frames === nothing ? collect(0:total-1) : sort!(unique(Int.(collect(frames))))
    all(n -> 0 <= n < total, wanted) || error("invalid farm frame range")
    requested=length(wanted)
    reuse===nothing || (wanted = distinctframes(reuse, filter!(n->farmneedsframe(reuse,n),wanted)))
    # An atomic mkdir is a process-wide coordinator lock, including other editors.
    lockdir = joinpath(dir, "coordinator.lock")
    ispath(lockdir) && error("farm directory already has a coordinator: $lockdir")
    mkdir(lockdir)
    joined = FarmWorker[]                    # every worker that arrived: closed at the end
    try
        mkpath(joinpath(dir, "frames"))
        pending = reverse([n for n in wanted if !farmcomplete(dir, n, key)])
        completed = length(wanted) - length(pending)
        counts = Dict{String, Int}()
        batches = Dict{String, Any}()
        errors = Dict{String, String}()
        incidents = Dict{String,Any}[]
        active = Set{String}()
        guard = ReentrantLock()
        started = time_ns()
        function report()
            data = Dict("job" => key, "completed" => completed, "total" => length(wanted),
                        "timeline_frames" => total, "workers" => counts, "active" => collect(active),
                        "reused" => requested-length(wanted),
                        "errors" => copy(errors), "incidents" => copy(incidents),
                        "elapsed_seconds" => (time_ns()-started)/1e9,
                        "batches" => copy(batches),
                        "paused" => isfile(joinpath(dir, "pause.requested")), "batchsize" => batchsize)
            farmwritejson(joinpath(dir, "status.json"), data)
            progress === nothing || progress(completed, length(wanted))
        end
        paused() = isfile(joinpath(dir, "pause.requested"))
        # No worker left with anything to do: stop waiting for more to arrive.
        finished() = isempty(active) && (isempty(pending) || paused()) && close(pool)
        report()
        finished()
        @sync for worker in pool
            fresh = lock(guard) do
                if haskey(counts, worker.name)
                    errors[worker.name] = "a second worker of this name joined and was left out"
                    return false
                end
                counts[worker.name] = 0
                push!(joined, worker)
                push!(active, worker.name)
                report()
                return true
            end
            fresh || (closeworker!(worker); continue)
            @async begin
                failures = 0
                try
                    while true
                        ids = lock(guard) do
                            isempty(pending) && return Int[]
                            paused() && return Int[]
                            nextfarmframes!(pending,worker,job,batchsize)
                        end
                        isempty(ids) && break
                        remaining = Set(ids)
                        reporting = false
                        batch_started = time_ns()
                        render_seconds = 0.0
                        first_result_seconds = nothing
                        try
                          foreachfarmresult(worker.render,ids) do r
                            n = r.frame
                            n in remaining || error("worker returned duplicate or unleased frame $n")
                            first_result_seconds === nothing && (first_result_seconds=(time_ns()-batch_started)/1e9)
                            render_seconds += r.seconds
                            img = PNGFiles.load(IOBuffer(r.png))
                            Base.size(img) == reverse(canvas) || error("worker returned the wrong dimensions")
                            # the planes first: a receipt names only planes that are there
                            layers = [commitlayer!(dir, l) for l in r.layers]
                            sha = commitbytes!(farmframepath(dir, n), r.png)
                            farmwritejson(farmreceiptpath(dir, n), Dict("job" => key,
                                "sha256" => sha, "worker" => worker.name, "layers" => layers,
                                "frame" => n, "seconds" => r.seconds, "completed_utc" => string(now(UTC))))
                            # Keep successful prefixes even if a connection fails
                            # halfway through the batch or a progress observer throws.
                            delete!(remaining,n)
                            reporting = true
                            lock(guard) do
                                counts[worker.name] += 1; completed += 1; failures = 0
                                delete!(errors, worker.name)
                                completed==length(wanted) && empty!(errors)
                                report()
                            end
                            reporting = false
                          end
                          isempty(remaining) || error("worker returned an incomplete batch: $(sort!(collect(remaining)))")
                        catch e
                            reporting && rethrow()
                            isempty(remaining) && rethrow() # observer failed after a durable commit
                            lock(guard) do
                                append!(pending,remaining); sort!(pending;rev=true); failures += 1
                                errors[worker.name] = sprint(showerror, e)
                                push!(incidents,Dict("worker"=>worker.name,"frames"=>sort!(collect(remaining)),
                                    "error"=>errors[worker.name],"utc"=>string(now(UTC))))
                                report()
                            end
                            failures >= retries && break
                        end
                        lock(guard) do
                            batches[worker.name] = Dict("frames"=>ids,"committed"=>length(ids)-length(remaining),
                                "elapsed_seconds"=>(time_ns()-batch_started)/1e9,
                                "render_seconds"=>render_seconds,"first_result_seconds"=>first_result_seconds)
                            report()
                        end
                        # yield() alone can keep a CPU-bound ready queue running
                        # indefinitely without polling network I/O.
                        sleep(0.001)
                    end
                finally
                    lock(guard) do; delete!(active, worker.name); report(); finished(); end
                end
            end
        end
        if !isempty(pending) && !paused()
            error("farm incomplete: $(length(pending)) frames remain; inspect status.json and resume")
        end
        return farmstatus(dir)
    finally
        close(pool)
        while isready(pool)                  # arrived, but never started
            push!(joined, take!(pool))
        end
        for worker in joined
            try
                closeworker!(worker)
            catch e
                @warn "farm worker cleanup failed" worker = worker.name exception = e
            end
        end
        rm(lockdir; recursive = true)
    end
end

"""The workers of a render as the channel it takes them from: a collection's, all there from the start."""
workerchannel(workers::Channel) = workers
function workerchannel(workers)
    pool = Channel{FarmWorker}(Inf)
    foreach(w -> put!(pool, w), workers)
    close(pool)
    return pool
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
