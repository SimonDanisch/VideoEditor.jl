# Track-owned, source-frame caches. Validity is recorded by mutation hooks;
# rendering never compares a document with an earlier version of itself.
mutable struct TrackCache
    directory::String
    valid::Dict{UInt64, BitSet}
    dirty::Dict{UInt64, BitSet}
    rendering::Set{Tuple{UInt64, Int}}
    revisions::Dict{Tuple{UInt64, Int}, UInt64}
    hits::Int
    renders::Int
    format::Symbol
    source::Any
end
TrackCache(directory::AbstractString; format = :png) = TrackCache(
    String(directory),
    Dict{UInt64, BitSet}(), Dict{UInt64, BitSet}(), Set{Tuple{UInt64, Int}}(), Dict{Tuple{UInt64, Int}, UInt64}(), 0, 0, format, nothing
)

cachefile(c::TrackCache, id::Integer, frame::Integer) =
    joinpath(c.directory, string(id), string(frame; pad = 6) * (c.format === :hdr ? ".vehdr" : ".png"))
cachevalid(c::TrackCache, id::Integer, frame::Integer) =
    !((UInt64(id), Int(frame)) in c.rendering) &&
    Int(frame) in get(c.valid, UInt64(id), BitSet()) &&
    !(Int(frame) in get(c.dirty, UInt64(id), BitSet())) && isfile(cachefile(c, id, frame))

function dirtyframes!(c::TrackCache, id::Integer, frames)
    fs = filter(>=(0), frames)
    union!(get!(() -> BitSet(), c.dirty, UInt64(id)), fs)
    for frame in fs
        key = (UInt64(id), Int(frame))
        c.revisions[key] = get(c.revisions, key, UInt64(0)) + UInt64(1)
    end
    return c
end

function cachewrite!(
        c::TrackCache, id::Integer, frame::Integer, image;
        revision = get(c.revisions, (UInt64(id), Int(frame)), UInt64(0))
    )
    file = cachefile(c, id, frame)
    mkpath(dirname(file))
    # PNG writes one complete frame before publishing it. Keep the previous
    # image until rename, including when an export is interrupted.
    tmp = file * ".part"
    if c.format === :hdr
        writerenderframe(tmp, image)
    else
        PNGFiles.save(tmp, collect(image); filters = Int(PNGFiles.PNG_FILTER_SUB))
    end
    mv(tmp, file; force = true)
    # An edit during an in-flight render must remain dirty. These counters
    # record mutations; they do not compare render histories or document data.
    get(c.revisions, (UInt64(id), Int(frame)), UInt64(0)) == revision || return image
    push!(get!(() -> BitSet(), c.valid, UInt64(id)), Int(frame))
    delete!(get!(() -> BitSet(), c.dirty, UInt64(id)), Int(frame))
    c.renders += 1
    return image
end
