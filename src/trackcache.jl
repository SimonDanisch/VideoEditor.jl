"Enable persistent frame caches, one owner per track, and a composite cache."
function enabletrackcache!(seq::Sequence, directory::AbstractString)
    mkpath(directory)
    for track in unique(c.track for c in seq.clips)
        get!(() -> TrackCache(joinpath(directory, "track-$track")), seq.trackcaches, track)
        c = seq.trackcaches[track]
        c.source === nothing && (c.source = TrackCache(joinpath(c.directory, "source"); format = :hdr))
    end
    seq.compositecache === nothing &&
        (seq.compositecache = TrackCache(joinpath(directory, "composite")))
    return seq
end

trackcache(clip::Clip) = clip.sequence isa Sequence ?
    get(clip.sequence.trackcaches, clip.track, nothing) : nothing

function cachedtrackframe(clip::Clip, sf::Integer)
    c = trackcache(clip)
    c === nothing && return nothing
    cachevalid(c, clip.id, sf) || return nothing
    c.hits += 1
    return PlanePixel.(PNGFiles.load(cachefile(c, clip.id, sf)))
end

function invalidatecomposite!(seq::Sequence, frames)
    seq.compositecache === nothing || dirtyframes!(seq.compositecache, 0, frames)
    return seq
end

"Invalidate exactly these SOURCE frames and their timeline dependents."
function invalidateclip!(clip::Clip; frames = clip.src_in:(clip.src_out - 1), source = true, visited = Set{UInt64}())
    clip.id in visited && return clip
    push!(visited, clip.id)
    if clip.bake !== nothing
        clip.bake.dirty = true
        clip.bake.enabled = false
    end
    fs = BitSet(filter(>=(0), frames))
    cache = trackcache(clip)
    cache === nothing || dirtyframes!(cache, clip.id, fs)
    if source && cache !== nothing && cache.source !== nothing
        dirtyframes!(cache.source, clip.id, fs)
    end
    seq = clip.sequence
    if seq isa Sequence
        # Repeated source frames dirty all the timeline frames that show them.
        affected = [n for n in clip.start:(clipend(clip) - 1) if sourceframe(clip, n) in fs]
        invalidatecomposite!(seq, affected)
        # A driven effect can depend on another clip's data. Its operator may
        # resample time, so conservatively invalidate the dependent clip rather
        # than assume that the two source frame indices mean the same thing.
        for dependent in seq.clips
            any(
                p.input !== nothing && any(x -> x === clip, p.input.from)
                    for fx in dependent.effects for p in fx.params
            ) || continue
            sceneinput = any(
                fx.kind === :scene && any(
                    p.input !== nothing && any(x -> x === clip, p.input.from) for p in fx.params
                ) for fx in dependent.effects
            )
            invalidateclip!(dependent; source = sceneinput, visited)
        end
    end
    if clip.source isa SceneSource
        clip.source.samples = 0
        clip.source.pending = nothing
        clip.source.pendingat = -1
    end
    return clip
end

"Mark timeline frames on one track dirty; unrelated tracks keep their planes."
function invalidatetrack!(seq::Sequence, track::Integer, frames)
    fs = BitSet(filter(>=(0), frames))
    for clip in seq.clips
        clip.track == track || continue
        ns = intersect(fs, BitSet(clip.start:(clipend(clip) - 1)))
        isempty(ns) || invalidateclip!(clip; frames = unique(sourceframe(clip, n) for n in ns))
    end
    invalidatecomposite!(seq, fs)
    return seq
end

"A frame-local mutation: validity is revoked BEFORE the mutation executes."
function mutateframes!(f::Function, seq::Sequence, track::Integer, frames)
    invalidatetrack!(seq, track, frames)
    return f()
end
mutateframe!(f::Function, seq::Sequence, track::Integer, frame::Integer) =
    mutateframes!(f, seq, track, frame:frame)

function placementchanged!(clip, oldtrack, oldspan, field)
    seq = clip.sequence
    seq isa Sequence || return nothing
    invalidatecomposite!(seq, oldspan)
    invalidatecomposite!(seq, clip.start:(clipend(clip) - 1))
    if field === :track && oldtrack != clip.track
        old = get(seq.trackcaches, oldtrack, nothing)
        if old !== nothing
            new = get!(
                () -> TrackCache(joinpath(dirname(old.directory), "track-$(clip.track)")),
                seq.trackcaches, clip.track
            )
            # Placement changes move ownership, not pixels. Both the valid and
            # dirty frame sets travel with the clip.
            for sf in get(old.valid, clip.id, BitSet())
                file = cachefile(old, clip.id, sf)
                isfile(file) || continue
                target = cachefile(new, clip.id, sf)
                mkpath(dirname(target)); cp(file, target; force = true)
            end
            new.valid[clip.id] = pop!(old.valid, clip.id, BitSet())
            new.dirty[clip.id] = pop!(old.dirty, clip.id, BitSet())
            if old.source !== nothing
                new.source === nothing && (new.source = TrackCache(joinpath(new.directory, "source"); format = :hdr))
                for sf in get(old.source.valid, clip.id, BitSet())
                    file = cachefile(old.source, clip.id, sf)
                    isfile(file) || continue
                    target = cachefile(new.source, clip.id, sf)
                    mkpath(dirname(target)); cp(file, target; force = true)
                end
                new.source.valid[clip.id] = pop!(old.source.valid, clip.id, BitSet())
                new.source.dirty[clip.id] = pop!(old.source.dirty, clip.id, BitSet())
                for (key, revision) in collect(old.source.revisions)
                    key[1] == clip.id || continue
                    new.source.revisions[key] = revision; delete!(old.source.revisions, key)
                end
            end
            for (key, revision) in collect(old.revisions)
                key[1] == clip.id || continue
                new.revisions[key] = revision
                delete!(old.revisions, key)
            end
        end
    end
    return nothing
end

function bindcacheparams!(clip::Clip, fx::Effect)
    on(fx.enabled) do _
        invalidateclip!(clip; source = fx.kind === :scene)
    end
    for p in fx.params
        bindcacheparam!(clip, p)
    end
    return fx
end
function bindcacheparam!(clip::Clip, p::Param)
    p.owner === clip && return p
    p.owner = clip
    on(p.curve) do _
        owner = p.owner
        owner isa Clip || return
        sceneparam = any(fx.kind === :scene && any(x -> x === p, fx.params) for fx in owner.effects)
        invalidateclip!(owner; frames = something(p.changedframes, owner.src_in:(owner.src_out - 1)), source = sceneparam)
    end
    return p
end

"Explicit renderer choice; raster is the default, tracing is opt-in."
function setrenderconfig!(clip::Clip; backend = :RayMakie, raytrace = false, samples = 128, options...)
    clip.source isa SceneSource || error("render settings require a scene clip")
    invalidateclip!(clip)
    src = clip.source
    src.backend = Symbol(backend)
    src.bakewith = Symbol(backend)
    src.screenopts = Dict{Symbol, Any}(:rasterize => true, :samples => 1, options...)
    src.bakescreenopts = Dict{Symbol, Any}(:rasterize => !raytrace, :samples => raytrace ? samples : 1, options...)
    src.live = nothing
    return clip
end

"Mutate a procedural recipe and explicitly declare which source frames it affects."
function setsceneargs!(clip::Clip, changes::AbstractDict; frames = clip.src_in:(clip.src_out - 1))
    src = clip.source
    src isa SceneSource && src.root isa SceneProgram || error("not a program scene")
    invalidateclip!(clip; frames)
    merge!(src.build["args"], changes)
    src.root = SceneProgram(src.build)
    src.live = nothing
    return clip
end

function cachetrackframe!(clip::Clip, sf::Int, readers, engine)
    cache = trackcache(clip)
    cache === nothing && return
    cachevalid(cache, clip.id, sf) && return
    clip.timeinterp === :flow && return # flow depends on the fractional phase
    token = (clip.id, sf)
    revision = get(cache.revisions, token, UInt64(0))
    push!(cache.rendering, token)
    src = clip.source
    mode = src isa SceneSource ? src.mode : :live
    try
        src isa SceneSource && (src.mode = :bake)
        src isa SceneSource && (src.pending = nothing; src.pendingat = -1)
        source = decodable(src) ? get!(() -> opendecoder(src, engine.backend), readers, sourcepath(src)) : src
        renderplane(engine, source, clip, sf; exact = true) do image
            cachewrite!(cache, clip.id, sf, image; revision)
        end
    finally
        src isa SceneSource && (src.mode = mode)
        delete!(cache.rendering, token)
    end
    return nothing
end

"The clips timeline frame `n` is composited from: those at it, or a transition's two."
function frameclips(seq::Sequence, n::Integer)
    tr = transitionat(seq, n)
    sample = tr === nothing ? nothing : transitionsample(seq, tr, n)
    return sample === nothing ? clipsat(seq, n) : [sample[1], sample[3]]
end

"Finished output: render dirty track frames, then composite the cached planes."
function renderframe!(
        dest::AnyRGBFrame, seq::Sequence, n::Integer, readers::Dict{String, Any}, engine::FxEngine;
        black::Union{Nothing, AbstractMatrix} = nothing
    )
    comp = seq.compositecache
    revision = comp === nothing ? UInt64(0) : get(comp.revisions, (UInt64(0), Int(n)), UInt64(0))
    if comp !== nothing && cachevalid(comp, 0, n)
        comp.hits += 1
        copyto!(dest, RGB{N0f8}.(PNGFiles.load(cachefile(comp, 0, n))))
        return dest
    end
    if !isempty(seq.trackcaches)
        for clip in frameclips(seq, n)
            cachetrackframe!(clip, sourceframe(clip, n), readers, engine)
        end
    end
    renderuncachedframe!(dest, seq, n, readers, engine; black)
    comp === nothing || cachewrite!(comp, 0, n, dest; revision)
    return dest
end

cachemetadata(c::TrackCache) = Dict(
    "format" => String(c.format),
    "source" => c.source === nothing ? nothing : cachemetadata(c.source),
    "valid" => Dict(string(id) => collect(fs) for (id, fs) in c.valid),
    "dirty" => Dict(string(id) => collect(fs) for (id, fs) in c.dirty),
    "revisions" => [[string(id), sf, rev] for ((id, sf), rev) in c.revisions]
)
function restorecache(d, directory)
    c = TrackCache(directory; format = Symbol(get(d, "format", "png")))
    if c.format === :png
        source = get(d, "source", nothing)
        c.source = source === nothing ? TrackCache(joinpath(directory, "source"); format = :hdr) :
            restorecache(source, joinpath(directory, "source"))
    end
    for field in ("valid", "dirty"), (id, fs) in get(d, field, Dict())
        getfield(c, Symbol(field))[parse(UInt64, id)] = BitSet(Int.(fs))
    end
    for (id, sf, revision) in get(d, "revisions", ())
        c.revisions[(parse(UInt64, id), Int(sf))] = UInt64(revision)
    end
    return c
end

function saveframecaches!(path, seq)
    isempty(seq.trackcaches) && return nothing
    root = path * ".frames"
    stores = [("track-$t", c) for (t, c) in seq.trackcaches]
    append!(stores, [("track-$t/source", c.source) for (t, c) in seq.trackcaches if c.source !== nothing])
    seq.compositecache === nothing || push!(stores, ("composite", seq.compositecache))
    for (name, cache) in stores
        target = abspath(joinpath(root, name))
        abspath(cache.directory) == target && continue
        # Saving As preserves the original project's cached footage.
        for (id, fs) in cache.valid, sf in fs
            source = cachefile(cache, id, sf)
            isfile(source) || continue
            dest = joinpath(target, string(id), basename(source))
            mkpath(dirname(dest)); cp(source, dest; force = true)
        end
        cache.directory = target
    end
    return Dict(
        "tracks" => Dict(string(t) => cachemetadata(c) for (t, c) in seq.trackcaches),
        "composite" => seq.compositecache === nothing ? nothing : cachemetadata(seq.compositecache)
    )
end
function loadframecaches!(path, seq, d)
    d === nothing && return seq
    for (track, meta) in d["tracks"]
        seq.trackcaches[parse(Int, track)] = restorecache(meta, joinpath(path * ".frames", "track-$track"))
    end
    get(d, "composite", nothing) === nothing ||
        (seq.compositecache = restorecache(d["composite"], joinpath(path * ".frames", "composite")))
    return seq
end
