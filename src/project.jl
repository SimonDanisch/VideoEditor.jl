"""
    saveproject(path, seq; checkpoint = true) -> path
    loadproject(path) -> Sequence

A project file is JSON of the edit metadata — source paths, in/out points,
timeline positions, crops, effects, analyses. Sources are re-probed on load, so
the file stays small and readable; matte alpha, the one thing too big for text,
sits next to it as raw planes.

JSON rather than TOML because a project is a tree — clips holding effects holding
parameters — and TOML says that in table-array syntax nobody reads twice. A
project file is something you may have to open in an editor at 2am.

Writing is ATOMIC (write a temp, rename) and keeps the previous version as a
checkpoint: an interrupted save must not be able to destroy the last good one.
TOML projects still LOAD — the format is sniffed, not assumed — so older files
keep working.
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
    isempty(seq.overlays) || (dict["overlays"] = [overlaydict(ov) for ov in seq.overlays])
    # …and the transcript, for the same reason: a corrected caption is an edit,
    # and re-running Whisper to get it back costs minutes.
    # …and the narration's WORDS. The samples are a cache of them and would be
    # megabytes of JSON; `render!` puts them back.
    # The canvas, when it has been set — the crop tool's output size.
    seq.canvas === nothing ||
        (dict["canvas"] = [seq.canvas[1], seq.canvas[2]])
    isempty(seq.narration) ||
        (dict["narration"] = [Dict{String, Any}("text" => n.text, "at" => n.at,
                                                "voice" => n.voice) for n in seq.narration])
    isempty(seq.captions) ||
        (dict["captions"] = [Dict{String, Any}("start" => c.start, "stop" => c.stop,
                                               "text" => c.text) for c in seq.captions])
    checkpoint && checkpointproject(path)
    tmp = path * ".part"
    open(io -> JSON.print(io, dict, 2), tmp, "w")
    mv(tmp, path; force = true)          # atomic: a half-written file never replaces the good one
    savemattes(path, seq)
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
        try; rm(joinpath(dir, old); force = true); catch; end
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

function clipdict(clip::Clip)
    cd = Dict{String, Any}(
        "source" => clip.source.path,
        "src_in" => clip.src_in,
        "src_out" => clip.src_out,
        "start" => clip.start,
        "track" => clip.track,
        "crop" => collect(clip.crop),
        "rate" => clip.rate,
        # …and how it fills the frames between: an edit, not a cache
        "timeinterp" => String(clip.timeinterp),        # conform factor; 1.0 on native-rate clips
        "id" => string(clip.id),
        "blendfrom" => string(clip.blendfrom),
        "effects" => [slotdict(s) for s in clip.effects],
    )
    # stabilization tracks are part of the edit — losing an analysis on
    # save/reopen would be silently destructive
    if clip.motiontrack !== nothing
        cd["motiontrack"] = Dict{String, Any}(
            "src_in" => clip.motiontrack.src_in,
            "mode" => String(clip.motiontrack.mode),
            "transforms" => [vec(Float64.(collect(M))) for M in clip.motiontrack.transforms])
        clip.motiontrack.basecrop === nothing ||
            (cd["motiontrack"]["basecrop"] = collect(clip.motiontrack.basecrop))
    end
    if clip.colortrack !== nothing
        cd["colortrack"] = Dict{String, Any}(
            "src_in" => clip.colortrack.src_in,
            "strength" => Float64(clip.colortrack.strength),
            "gains" => [Float64.(collect(g)) for g in clip.colortrack.gains],
            "offsets" => [Float64.(collect(o)) for o in clip.colortrack.offsets])
    end
    # The matte's SEEDS are the edit and go in the project file; the propagated
    # alpha is a cache and goes to a sidecar, because a clip's worth of per-frame
    # mattes has no business inside a TOML and losing it costs a recompute rather
    # than an edit.
    if clip.mattetrack !== nothing
        t = clip.mattetrack
        cd["matte"] = Dict{String, Any}(
            "src_in" => t.src_in, "seeds" => t.seeds,
            "size" => collect(Int.(size(t.alpha))))
    end
    # The learned look, on the same terms as the stabilization track above: it is
    # an ANALYSIS but not a cache. Re-learning it is not the same operation —
    # `runlook!` fits from whatever frame the playhead is parked on, so a dropped
    # LUT would come back as a different grade, which is worse than coming back
    # empty. It is a `lookdim`³ table, small enough to sit in the file.
    if clip.look !== nothing
        cd["look"] = Dict{String, Any}(
            "dim" => size(clip.look, 1),
            "table" => Float64.(vec(clip.look)))
    end
    # keyframed parameters — losing them on reopen would silently drop an animation
    anims = Dict{String, Any}(
        String(key) => Dict{String, Any}(
            "interp" => String(curve.interp),
            "frames" => [k.frame for k in curve.keys],
            "values" => [Float64(k.value) for k in curve.keys],
            "eases" => [String(k.ease) for k in curve.keys])
        for (key, curve) in clip.animations if !isempty(curve))
    isempty(anims) || (cd["animations"] = anims)
    return cd
end

"""
Parse a project file, whatever it was written as.

Sniffed, not assumed: JSON starts with `{`. Projects written before the format
moved to JSON are TOML and still open — a file format change must not strand the
edits somebody already saved.
"""
function readprojectdict(path::AbstractString)
    txt = read(path, String)
    return startswith(lstrip(txt), "{") ? JSON.parse(txt) : TOML.parse(txt)
end

function loadproject(path::AbstractString)
    dict = readprojectdict(path)
    missing_sources = unique(String[cd["source"] for cd in dict["clips"] if !isfile(cd["source"])])
    isempty(missing_sources) ||
        error("project references missing video file(s):\n  " * join(missing_sources, "\n  ") *
              "\nMove them back (or edit the paths in $path) and reload.")
    sources = Dict{String, VideoSource}()
    clips = map(dict["clips"]) do cd
        source = get!(() -> VideoSource(cd["source"]), sources, cd["source"])
        # files written before conforming existed hold only native-rate clips
        clip = Clip(source, cd["src_in"], cd["src_out"], cd["start"],
                    Tuple(Float64.(cd["crop"])), Float64(get(cd, "rate", 1.0)),
                    # a 3-tuple here is a project saved before rotation existed
                    reframe4(Tuple(Float64.(get(cd, "reframe", collect(NEUTRALFRAME))))))
        clip.track = Int(get(cd, "track", 1))
        # Files written before optical flow existed hold no mode and mean `:sample`,
        # which is what every editor does without a model. Validated rather than
        # trusted: the value picks a source node, and an unrecognised one would
        # silently mean `:sample` forever with no way to notice.
        ti = Symbol(get(cd, "timeinterp", "sample"))
        clip.timeinterp = ti in (:sample, :flow) ? ti : :sample
        # ids are part of the edit: a blend points at its partner by id, and the
        # inspector at a stack entry. Files written before ids existed simply keep
        # the fresh ones the constructor handed out.
        haskey(cd, "id") && (clip.id = parse(UInt64, cd["id"]))
        clip.blendfrom = haskey(cd, "blendfrom") ? parse(UInt64, cd["blendfrom"]) : UInt64(0)
        for ed in get(cd, "effects", [])
            push!(clip.effects, slotfromdict(ed))
        end
        # An analysis and the stack slot that applies it are attached together
        # (`setmotiontrack!`), which is also how a project written before analyses
        # had slots gets its cards: the track is there, so the slot is made.
        if haskey(cd, "motiontrack")
            mt = cd["motiontrack"]
            setmotiontrack!(clip, MotionTrack(
                [Mat3f(Float32.(v)...) for v in mt["transforms"]], Int(mt["src_in"]),
                Symbol(get(mt, "mode", "unknown")),
                haskey(mt, "basecrop") ? NTuple{4, Float64}(mt["basecrop"]) : nothing))
        end
        if haskey(cd, "colortrack")
            ct = cd["colortrack"]
            setcolortrack!(clip, ColorTrack(
                [Vec3f(Float32.(v)...) for v in ct["gains"]],
                [Vec3f(Float32.(v)...) for v in ct["offsets"]], Int(ct["src_in"]),
                Float32(get(ct, "strength", 1.0))))   # absent in older project files
        end
        if haskey(cd, "matte")
            mt = cd["matte"]
            sz = NTuple{3, Int}(Int.(mt["size"]))
            clip.mattetrack = MatteTrack(loadmatte(path, clip.id, sz), Int(mt["src_in"]),
                                         Int.(mt["seeds"]))
        end
        if haskey(cd, "look")
            lk = cd["look"]
            d = Int(lk["dim"])
            clip.look = reshape(Float32.(lk["table"]), d, d, d, 3)
        end
        for (key, ad) in get(cd, "animations", Dict{String, Any}())
            eases = get(ad, "eases", fill("linear", length(ad["frames"])))  # older files
            clip.animations[Symbol(key)] = AnimCurve(
                [Keyframe(Int(f), Float64(v), Symbol(e))
                 for (f, v, e) in zip(ad["frames"], ad["values"], eases)],
                Symbol(get(ad, "interp", "linear")))
        end
        clip
    end
    seq = Sequence(collect(Clip, clips), Float64(dict["framerate"]))
    for td in get(dict, "transitions", [])
        push!(seq.transitions, Transition(Symbol(td["kind"]), Int(td["at"]), Int(td["duration"])))
    end
    for od in get(dict, "overlays", [])
        push!(seq.overlays, overlayfromdict(od))
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
    return seq
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
frame number and a mask. Both are EDITS — a mark is what the propagator runs
from, a repair is what overrules the result — and neither fits in a JSON file at
a clip's worth of frames.
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

Losing either store was INVISIBLE, which is what made it worth chasing. The
repaired pixels are baked into the alpha sidecar, so a reopened project looked
correct — while the repair cards were gone and the next full run discarded every
fix. And `runmatte!` propagates from the MARKS, so without them a reopened matte
could not be re-run at all: "Apply matte to clip" met an empty store and refused.
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
