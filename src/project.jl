"""
    saveproject(path, seq; checkpoint = true) -> path
    loadproject(path) -> Sequence

A project file is the edit metadata — source paths, in/out points, timeline
positions, crops, effects, analyses — as MessagePack. Sources are re-probed on
load, so the file holds no pixels; matte alpha, the one thing per-frame enough to
matter, sits next to it as raw planes.

MessagePack rather than JSON because most of a real project is not the edit but
per-frame analysis. One hour of stabilization and colour tracking on one clip is
108 000 `Mat3f` and 216 000 `Vec3f`, and measured on exactly that:

    JSON                             52.4 MB   write 1.87 s   read 0.30 s
    MessagePack, numbers as numbers  14.9 MB   write 0.08 s   read 0.19 s
    …with the analysis tracks raw     6.5 MB   write 0.00 s   read 0.05 s

Nothing about the tree changed — see pack.jl for how a type says what it is. The
saving is that a number stops being decimal text: `0.20000000298023224` was
nineteen bytes for a `Float32` that JSON could not represent in the first place.

Writing is atomic (write a temp, rename) and keeps the previous version as a
checkpoint, so an interrupted save cannot destroy the last good one.
"""
function saveproject(path::AbstractString, seq::Sequence; checkpoint::Bool = true)
    dict = Dict{String, Any}(
        "framerate" => seq.framerate,
        "clips" => [clipdict(clip) for clip in seq.clips],
    )
    # transitions (cross-dissolves) are part of the edit — losing them on reopen
    # would silently drop every dissolve
    isempty(seq.transitions) || (dict["transitions"] =
        [Dict{String, Any}("kind" => String(t.kind), "at" => t.at, "duration" => t.duration)
         for t in seq.transitions])
    # overlays are edits like any other — a lost title is a lost edit
    # …and the transcript, for the same reason: a corrected caption is an edit,
    # and re-running Whisper to get it back costs minutes.
    # …and the narration's WORDS. The samples are a cache of them and would be
    # megabytes of JSON; `render!` puts them back.
    # The canvas, when it has been set — the crop tool's output size.
    seq.canvas === nothing ||
        (dict["canvas"] = [seq.canvas[1], seq.canvas[2]])
    # …and how tall each track was made. Only written once somebody has resized
    # one, so a project nobody laid out by hand reads and writes exactly as before.
    isempty(seq.trackheights) || (dict["trackheights"] = copy(seq.trackheights))
    isempty(seq.narration) ||
        (dict["narration"] = [Dict{String, Any}("text" => n.text, "at" => n.at,
                                                "voice" => n.voice) for n in seq.narration])
    isempty(seq.captions) ||
        (dict["captions"] = [Dict{String, Any}("start" => c.start, "stop" => c.stop,
                                               "text" => c.text) for c in seq.captions])
    checkpoint && checkpointproject(path)
    tmp = path * ".part"
    open(io -> write(io, MsgPack.pack(dict)), tmp, "w")
    mv(tmp, path; force = true)          # atomic: a half-written file never replaces the good one
    savemattes(path, seq)
    # …and the baked frames, for the same reason: minutes of raytracing is an
    # edit's output, not something to recompute on every open. Defined in bake.jl,
    # which loads after this file — resolved at call time.
    savebakes(path, seq)
    return path
end

"""
Keep the file that is about to be overwritten.

Checkpoints go to `<dir>/.<name>.checkpoints/` with a sortable timestamp, newest
`CHECKPOINTS` kept. Undo dies with the session; a project you saved over an hour
ago does not have to.
"""
const CHECKPOINTS = 20

function checkpointproject(path::AbstractString)
    isfile(path) || return nothing
    dir = joinpath(dirname(path), "." * basename(path) * ".checkpoints")
    isdir(dir) || mkpath(dir)
    stamp = Dates.format(Dates.unix2datetime(mtime(path)), "yyyymmdd-HHMMSS")
    try
        cp(path, joinpath(dir, stamp * "-" * basename(path)); force = true)
    catch e
        @warn "could not write a project checkpoint" exception = e
        return nothing
    end
    keep = sort!(readdir(dir))
    for old in keep[1:max(0, length(keep) - CHECKPOINTS)]
        # a checkpoint that will not go is worth one line: otherwise the directory
        # grows without bound and nothing says why
        try
            rm(joinpath(dir, old); force = true)
        catch e
            @warn "could not remove an old project checkpoint" file = old exception = (e, catch_backtrace())
        end
    end
    return dir
end

"""
    projectcheckpoints(path) -> Vector{String}

Every kept version of `path`, oldest first — what a "revert to…" list reads.
"""
function projectcheckpoints(path::AbstractString)
    dir = joinpath(dirname(path), "." * basename(path) * ".checkpoints")
    isdir(dir) || return String[]
    return [joinpath(dir, f) for f in sort!(readdir(dir))]
end

"""
How a clip's source is written: a path for a file, the format for a scene.

A scene clip has no media on disk — what it draws is a `SceneSpec` on its `:scene`
effect, which goes out through the ordinary effect writer like every other
parameter. All the file needs is how big its frames are and how many.
"""
sourcedict(s::VideoSource) = Dict{String, Any}("source" => s.path)
sourcedict(s::SceneSource) = Dict{String, Any}(
    "source" => "", "scene" => Dict{String, Any}(
        "width" => s.width, "height" => s.height,
        "framerate" => s.framerate, "nframes" => s.nframes,
        "backend" => String(s.backend), "bakewith" => String(s.bakewith),
        "build" => s.build))

"The inverse: a source from what the file says about it."
function sourcefromdict(cd::AbstractDict, sources::Dict{String, VideoSource})
    sd = get(cd, "scene", nothing)
    if sd !== nothing
        build = get(sd, "build", nothing)
        built = build === nothing ?
                plainscene(Makie.SpecApi.Scene(; camera = Makie.cam3d!)) :
                buildscene(build)
        return SceneSource(built.root; joints = built.joints, camera = built.camera,
                           build = build,
                           backend = Symbol(get(sd, "backend", "GLMakie")),
                           bakewith = Symbol(get(sd, "bakewith", "auto")),
                           width = Int(get(sd, "width", 1920)),
                           height = Int(get(sd, "height", 1080)),
                           framerate = Float64(get(sd, "framerate", 30.0)),
                           nframes = Int(get(sd, "nframes", 90)))
    end
    path = String(cd["source"])
    return get!(() -> VideoSource(path), sources, path)
end

function clipdict(clip::Clip)
    cd = merge(sourcedict(clip.source), Dict{String, Any}(
        "src_in" => clip.src_in,
        "src_out" => clip.src_out,
        "start" => clip.start,
        "track" => clip.track,
        "crop" => collect(clip.crop),
        "rate" => clip.rate,                            # conform factor; 1.0 on native-rate clips
        # …and how it fills the frames between: an edit, not a cache
        "timeinterp" => String(clip.timeinterp),
        # ids go as themselves. JSON has one number type and it is `Float64`, so
        # a `UInt64` id had to be written as its decimal STRING to survive the
        # round trip; MessagePack has unsigned integers.
        "id" => clip.id,
        # …and the effects go as themselves too — see pack.jl.
        "effects" => clip.effects,
    ))
    # stabilization tracks are part of the edit — losing an analysis on
    # save/reopen would be silently destructive. `packvec` writes the whole track
    # as one Float32 block: this is the field that decides what a project of any
    # length costs to open.
    if clip.motiontrack !== nothing
        mt = clip.motiontrack
        cd["motiontrack"] = Dict{String, Any}(
            "src_in" => mt.src_in,
            "mode" => String(mt.mode),
            "transforms" => packvec(mt.transforms))
        mt.basecrop === nothing || (cd["motiontrack"]["basecrop"] = collect(mt.basecrop))
    end
    if clip.colortrack !== nothing
        ct = clip.colortrack
        cd["colortrack"] = Dict{String, Any}(
            "src_in" => ct.src_in,
            "strength" => Float64(ct.strength),
            "gains" => packvec(ct.gains),
            "offsets" => packvec(ct.offsets))
    end
    # The matte's seeds are the edit and go in the project file; the propagated
    # alpha is a cache and goes to a sidecar, because a clip's worth of per-frame
    # mattes has no business inside the project file and losing it costs a
    # recompute rather than an edit.
    if clip.mattetrack !== nothing
        t = clip.mattetrack
        cd["matte"] = Dict{String, Any}(
            "src_in" => t.src_in, "seeds" => collect(Int64, t.seeds),
            "size" => collect(Int64, size(t.alpha)))
    end
    # The learned look, on the same terms as the stabilization track above: it is
    # an analysis but not a cache. Re-learning is not the same operation:
    # `runlook!` fits from whatever frame the playhead is parked on, so a dropped
    # LUT would come back as a different grade rather than as nothing. It is a `lookdim`³ table, small enough to sit in the file.
    if clip.look !== nothing
        cd["look"] = Dict{String, Any}(
            "dim" => size(clip.look, 1),
            # stays Float32 — it IS a Float32 table — and goes raw: `dim`³×3
            "table" => Block(vec(clip.look)))
    end
    # The bake: where it is is derived from the project path on load, so only
    # what it covers and whether it is on go in the file.
    clip.bake === nothing || (cd["bake"] = bakedict(clip.bake))
    # NO clip-level `animations`: a curve is written inside the effect whose
    # parameter it animates, which is where it lives in memory.
    return cd
end

function loadproject(path::AbstractString)
    # `decodeblocks` first, so nothing below this line has to know that a numeric
    # array may have been written as raw bytes — see pack.jl.
    dict = decodeblocks(MsgPack.unpack(read(path)))
    # A scene clip has no file to be missing (its `"source"` is empty).
    missing_sources = unique(String[cd["source"] for cd in dict["clips"]
                                    if !isempty(get(cd, "source", "")) && !isfile(cd["source"])])
    isempty(missing_sources) ||
        error("project references missing video file(s):\n  " * join(missing_sources, "\n  ") *
              "\nMove them back (or edit the paths in $path) and reload.")
    sources = Dict{String, VideoSource}()
    # `saveproject` writes every field below, but the reader takes a default for
    # each one it can. A project file is meant to be writable by a script — an
    # agent placing clips does not want to spell out `timeinterp` to say nothing
    # — so the minimum is a source and a range, and everything else means what
    # a freshly dropped clip means.
    clips = map(dict["clips"]) do cd
        source = sourcefromdict(cd, sources)
        clip = Clip(source, cd["src_in"], cd["src_out"], cd["start"],
                    NTuple{4, Float64}(get(cd, "crop", (0.0, 0.0, 1.0, 1.0))),
                    Float64(get(cd, "rate", 1.0)))
        clip.track = Int(get(cd, "track", 1))
        # Validated rather than trusted: the value picks a source node, and an
        # unrecognised one would silently mean `:sample` forever with no way to
        # notice.
        ti = Symbol(get(cd, "timeinterp", "sample"))
        clip.timeinterp = ti in (:sample, :flow) ? ti : :sample
        # ids are part of the edit: a blend points at its partner by id, and the
        # inspector at a stack entry. A file that names none keeps the fresh one
        # the constructor handed out.
        haskey(cd, "id") && (clip.id = UInt64(cd["id"]))
        # A project written when the pairing was a clip field: it becomes an edge
        # on the opacity parameter once the effects are in (below).
        oldpair = UInt64(get(cd, "blendfrom", 0))
        for ed in get(cd, "effects", ())
            addslot!(clip, MsgPack.from_msgpack(Effect, ed))
        end
        # …and now the old clip-level pairing has somewhere to go.
        if oldpair != 0
            prm = opacityparam(clip)
            prm === nothing ||
                (prm.input = ParamInput(:pairedwith, ParamRef(:opacity; clip = oldpair)))
        end
        # An analysis and the stack slot that applies it are attached together
        # (`setmotiontrack!`), which is also how a project written before analyses
        # had slots gets its cards: the track is there, so the slot is made.
        if haskey(cd, "motiontrack")
            mt = cd["motiontrack"]
            setmotiontrack!(clip, MotionTrack(
                unpackvec(Mat3f, mt["transforms"]), Int(mt["src_in"]),
                Symbol(mt["mode"]),
                haskey(mt, "basecrop") ?
                    NTuple{4, Float64}(mt["basecrop"]) : nothing))
        end
        if haskey(cd, "colortrack")
            ct = cd["colortrack"]
            setcolortrack!(clip, ColorTrack(
                unpackvec(Vec3f, ct["gains"]),
                unpackvec(Vec3f, ct["offsets"]),
                Int(ct["src_in"]), Float32(ct["strength"])))
        end
        if haskey(cd, "matte")
            mt = cd["matte"]
            sz = NTuple{3, Int}(mt["size"])
            clip.mattetrack = MatteTrack(loadmatte(path, clip.id, sz), Int(mt["src_in"]),
                                         Int[Int(x) for x in mt["seeds"]])
        end
        haskey(cd, "bake") &&
            (clip.bake = bakefromdict(cd["bake"], bakeclipdir(path, clip.id)))
        if haskey(cd, "look")
            lk = cd["look"]
            d = Int(lk["dim"])
            clip.look = reshape(collect(Float32, lk["table"]), d, d, d, 3)
        end
        clip
    end
    seq = Sequence(collect(Clip, clips), Float64(dict["framerate"]))
    for td in get(dict, "transitions", [])
        push!(seq.transitions, Transition(Symbol(td["kind"]), Int(td["at"]), Int(td["duration"])))
    end
    # A project written when overlays were a thing of their own: each becomes a
    # clip on a track above the footage, which is where it was drawn anyway.
    for od in get(dict, "overlays", [])
        c = clipfromoverlay(od, seq)
        c === nothing || addclip!(seq, c)
    end
    if haskey(dict, "trackheights")
        empty!(seq.trackheights)
        append!(seq.trackheights, Float64.(dict["trackheights"]))
    end
    if haskey(dict, "canvas")
        cv = dict["canvas"]
        seq.canvas = (Int(cv[1]), Int(cv[2]))
    end
    for nd in get(dict, "narration", [])
        push!(seq.narration, Narration(String(nd["text"]), Float64(get(nd, "at", 0.0)),
                                       String(get(nd, "voice", "af_heart"))))
    end
    for cd in get(dict, "captions", [])
        push!(seq.captions, Caption(Float64(cd["start"]), Float64(cd["stop"]),
                                    String(cd["text"])))
    end
    # A parameter driven by another one is written as the ids of what drives it;
    # this is where those become the objects `valueat` follows. Last, because it
    # needs every clip and every effect in place — an edge may point across a clip
    # boundary in either direction.
    bindinputs!(seq)
    # …and the bakes: their directory follows the project, and one whose inputs
    # moved on disk is switched off rather than shown as current.
    adoptbakes!(path, seq)
    reportbackends(seq)
    return seq
end

"""
    reportbackends(seq) -> Vector{Symbol}

Warn about every renderer this project names that is not loaded, and return them.

Reported, not substituted: a scene says which renderer draws it, and falling back
to another one shows a different picture than the one that was saved, with nothing
on screen to say so. Loading the package that provides it (and `usebackend!`) is
something the user can do.

At load rather than at the first render, so it is not found by scrubbing onto the
clip and getting an exception.
"""
function reportbackends(seq::Sequence)
    want = Symbol[]
    for clip in seq.clips
        src = clip.source
        src isa SceneSource || continue
        push!(want, src.backend)
        src.bakewith === :auto || push!(want, src.bakewith)
    end
    missing = sort!(unique!(filter(n -> !haskey(BACKENDS, n), want)))
    isempty(missing) ||
        @warn "this project names scene renderers that are not loaded — those clips " *
              "cannot render until they are" missing = missing loaded = sort!(collect(keys(BACKENDS)))
    return missing
end


"Directory holding a project's matte sidecars (created on demand)."
mattedir(path::AbstractString) = string(path, ".mattes")
mattefile(path::AbstractString, id::Integer) = joinpath(mattedir(path), string(id, ".bin"))

"Write every clip's propagated alpha next to the project file."
function savemattes(path::AbstractString, seq::Sequence)
    any(c -> c.mattetrack !== nothing, seq.clips) || return
    dir = mattedir(path)
    isdir(dir) || mkpath(dir)
    for clip in seq.clips
        clip.mattetrack === nothing && continue
        open(io -> write(io, clip.mattetrack.alpha), mattefile(path, clip.id), "w")
    end
    return
end

"""
    masksidecar(path, id, kind) -> String

Where a clip's per-frame masks live, beside its alpha. `kind` is `"repairs"` or
`"marks"`.

Two stores, one format, because they are the same thing shaped differently: a
frame number and a mask. Both are edits — a mark is what the propagator runs from,
a repair is what overrules the result — and neither fits in a JSON file at a
clip's worth of frames.
"""
masksidecar(path::AbstractString, id::Integer, kind::AbstractString) =
    joinpath(mattedir(path), string(id, ".", kind, ".bin"))

"Backwards-compatible alias: the repairs sidecar."
repairfile(path::AbstractString, id::Integer) = masksidecar(path, id, "repairs")

"""
    savemasks(path, store, kind)

Write a frame -> mask store next to the alphas.

One record is `frame::Int32, w::Int32, h::Int32` then `w*h` bytes, so a file
reads without consulting the alpha's shape.

Losing either store is invisible at first: the repaired pixels are baked into the
alpha sidecar, so a reopened project looks correct while the repair cards are gone
and the next full run discards every fix. And `runmatte!` propagates from the
marks, so without them a reopened matte cannot be re-run at all.
"""
function savemasks(path::AbstractString, store, kind::AbstractString)
    any(!isempty, values(store)) || return
    dir = mattedir(path)
    isdir(dir) || mkpath(dir)
    for (id, masks) in store
        isempty(masks) && continue
        open(masksidecar(path, id, kind), "w") do io
            for (frame, mask) in masks
                write(io, Int32(frame), Int32(size(mask, 1)), Int32(size(mask, 2)))
                write(io, mask)
            end
        end
    end
    return
end

"""
    loadmasks!(store, clips, path, kind)

Put a frame -> mask store back for every clip that has a sidecar.

Silent on a missing or truncated file: a sidecar is not the edit, and half a mask
is better dropped than drawn.
"""
function loadmasks!(store, clips, path::AbstractString, kind::AbstractString)
    isdir(mattedir(path)) || return store
    for clip in clips
        f = masksidecar(path, clip.id, kind)
        isfile(f) || continue
        masks = Dict{Int, Matrix{UInt8}}()
        open(f, "r") do io
            while !eof(io)
                bytesavailable(io) >= 12 || break
                frame = read(io, Int32); w = read(io, Int32); h = read(io, Int32)
                (w <= 0 || h <= 0) && break
                n = Int(w) * Int(h)
                bytesavailable(io) >= n || break
                masks[Int(frame)] = reshape(read(io, n), Int(w), Int(h))
            end
        end
        isempty(masks) || (store[clip.id] = masks)
    end
    return store
end

"Write the matte MARKS and REPAIRS that live on the player, not the sequence."
function saverepairs(path::AbstractString, player)
    savemasks(path, player.matterepairs, "repairs")
    savemasks(path, player.mattemarks, "marks")
    return
end

"Read them back — see [`savemasks`](@ref) for what losing either one cost."
function loadrepairs!(player, path::AbstractString)
    loadmasks!(player.matterepairs, player.sequence.clips, path, "repairs")
    loadmasks!(player.mattemarks, player.sequence.clips, path, "marks")
    return player
end

"""
Read a clip's matte sidecar, or zeros of the recorded size when it is missing or
the wrong length. A missing cache must not be an error: the seeds are still in
the project, so the matte is one re-run away, and `applymatte!` treats an
all-zero track as "not analyzed here" rather than blacking the frame out.
"""
function loadmatte(path::AbstractString, id::Integer, sz::NTuple{3, Int})
    f = mattefile(path, id)
    n = prod(sz)
    if isfile(f) && filesize(f) == n
        return reshape(read(f), sz)
    end
    isfile(f) && @warn "matte sidecar for clip $id is $(filesize(f)) bytes, expected $n — ignoring"
    return zeros(UInt8, sz)
end

