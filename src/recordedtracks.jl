# Recorded simulation values are inputs to ordinary parameters. The file holds
# raw samples and a small seek index; opening it does not load every array.
const RECORDEDTYPES = Dict{String,Type}(
    "Bool"=>Bool,"Int8"=>Int8,"UInt8"=>UInt8,"Int16"=>Int16,"UInt16"=>UInt16,
    "Int32"=>Int32,"UInt32"=>UInt32,"Int64"=>Int64,"UInt64"=>UInt64,
    "Float16"=>Float16,"Float32"=>Float32,"Float64"=>Float64,
    "ComplexF32"=>ComplexF32,"ComplexF64"=>ComplexF64,
    "Point2f"=>Point2f,"Point3f"=>Point3f,"Vec2f"=>Vec2f,"Vec3f"=>Vec3f,"RGBf"=>RGBf,"RGBAf"=>RGBAf)
const RECORDMAGIC = codeunits("VEANIM01")

"Immutable, disk-backed samples keyed by absolute source frame."
struct RecordedTrack
    path::String
    frames::Vector{Int}
    offsets::Vector{Int}
    shapes::Vector{Vector{Int}}
    element::Type
    scalar::Bool
    framerate::Float64
    varying::Bool
    bytes::Vector{UInt8}
end

"A sample view cannot mutate the saved recording or another frame sharing it."
struct RecordedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end
Base.size(a::RecordedArray) = size(a.data)
Base.getindex(a::RecordedArray, i::Int...) = getindex(a.data,i...)
Base.IndexStyle(::Type{<:RecordedArray{T,N,A}}) where {T,N,A} = IndexStyle(A)

"A saved recording file, resolved through the ordinary parameter-input graph."
struct RecordedRef <: InputRef
    path::String
end
RecordedRef(path::AbstractString) = RecordedRef(abspath(path))

mutable struct RecordingWriter
    path::String
    io::IOStream
    element::Type
    scalar::Bool
    frames::Vector{Int}
    offsets::Vector{Int}
    shapes::Vector{Vector{Int}}
    previous::Vector{UInt8}
end

function RecordingWriter(path, value)
    scalar = !(value isa AbstractArray)
    T = scalar ? typeof(value) : eltype(value)
    T in values(RECORDEDTYPES) || throw(ArgumentError("unsupported recorded element type $T"))
    return RecordingWriter(path, open(path, "w"), T, scalar, Int[], Int[], Vector{Int}[], UInt8[])
end

function recordsample!(w::RecordingWriter, frame, value)
    scalar = !(value isa AbstractArray)
    T = scalar ? typeof(value) : eltype(value)
    (scalar == w.scalar && T == w.element) || throw(ArgumentError("recorded value type changed at frame $frame"))
    # Snapshot before advancing the simulation again: it may reuse this buffer.
    host = scalar ? [value] : vec(Array(value))
    bytes = reinterpret(UInt8, host)
    shape = scalar ? Int[] : collect(size(value))
    offset = position(w.io)
    if !isempty(w.frames) && shape == w.shapes[end] && bytes == w.previous
        offset = w.offsets[end]
    else
        write(w.io, bytes)
        resize!(w.previous, length(bytes)); copyto!(w.previous, bytes)
    end
    push!(w.frames, Int(frame)); push!(w.offsets, offset); push!(w.shapes, shape)
    return nothing
end

function finishrecording!(w::RecordingWriter, fps)
    index = MsgPack.pack(Dict("element" => findfirst(==(w.element),RECORDEDTYPES), "scalar" => w.scalar,
        "framerate" => Float64(fps), "endian" => ENDIAN_BOM,
        "samples" => Any[Any[f, o, Any[s...]] for (f,o,s) in zip(w.frames,w.offsets,w.shapes)]))
    write(w.io, index); write(w.io, htol(UInt64(length(index)))); write(w.io, RECORDMAGIC)
    close(w.io)
    return nothing
end

"Open a recording, validating its index and memory-mapping its sample bytes."
function openrecording(path::AbstractString)
    path = abspath(path)
    return open(path, "r") do io
        n = filesize(path)
        n >= 16 || error("invalid animation recording: $path")
        seek(io, n - 16); indexsize = ltoh(read(io, UInt64))
        read(io, 8) == RECORDMAGIC || error("invalid animation recording: $path")
        indexsize <= n - 16 || error("invalid recording index: $path")
        databytes = n - 16 - Int(indexsize)
        seek(io, databytes)
        d = decodeblocks(MsgPack.unpack(read(io, Int(indexsize))))
        d["endian"] == ENDIAN_BOM || error("recording byte order differs: $path")
        T = get(RECORDEDTYPES, d["element"], nothing)
        T === nothing && error("unknown recording element type: $(d["element"])")
        scalar = Bool(d["scalar"])
        rows = d["samples"]
        isempty(rows) && error("empty animation recording: $path")
        frames = Int[r[1] for r in rows]; offsets = Int[r[2] for r in rows]
        shapes = [Int.(r[3]) for r in rows]
        issorted(frames) && allunique(frames) || error("recording frames must be strictly increasing")
        for (offset, shape) in zip(offsets, shapes)
            all(>=(0), shape) || error("invalid recording shape")
            scalar && !isempty(shape) && error("invalid scalar recording shape")
            count = scalar ? 1 : prod(big.(shape); init=big(1))
            0 <= offset && offset + count * sizeof(T) <= databytes || error("invalid recording sample bounds")
        end
        seekstart(io)
        bytes = databytes == 0 ? UInt8[] : Mmap.mmap(io, Vector{UInt8}, databytes; shared=false)
        varying = any(i -> offsets[i] != offsets[1] || shapes[i] != shapes[1],eachindex(frames))
        RecordedTrack(path, frames, offsets, shapes, T, scalar, Float64(d["framerate"]), varying, bytes)
    end
end

function recordedsampleindex(track::RecordedTrack,frame::Real)
    i = searchsortedfirst(track.frames, frame)
    i <= length(track.frames) && track.frames[i] == frame ||
        error("no recorded sample for source frame $frame in $(track.path)")
    return i
end

function recordedchanged(track::RecordedTrack,a,b)
    i,j = recordedsampleindex(track,a),recordedsampleindex(track,b)
    return track.offsets[i] != track.offsets[j] || track.shapes[i] != track.shapes[j]
end

"Read exactly the recorded source frame; missing samples are never substituted."
function valueat(track::RecordedTrack, frame::Real)
    i = recordedsampleindex(track,frame)
    shape = track.shapes[i]
    count = track.scalar ? 1 : prod(shape)
    offset = track.offsets[i]
    values = reinterpret(track.element, @view track.bytes[offset+1:offset+count*sizeof(track.element)])
    return track.scalar ? only(values) : RecordedArray(reshape(values, Tuple(shape)))
end
readinput(track::RecordedTrack, frame::Integer) = valueat(track, frame)
Base.isempty(track::RecordedTrack) = isempty(track.frames)

"""
    recordanimation(sample!, directory, frames; framerate=30, progress=nothing)

Run a simulation once in increasing source-frame order. `sample!(frame, fps)`
returns a dictionary/NamedTuple of scene-property paths to scalars or arrays.
Samples stream to immutable binary tracks; mutable buffers are snapshotted and
consecutive identical values share storage. The completed directory is published
atomically. Returned tracks can feed ordinary parameters via `RecordedRef`.
"""
function recordanimation(sample!, directory::AbstractString, frames;
                         framerate::Real=30, progress=nothing)
    fs = Int.(collect(frames))
    !isempty(fs) && issorted(fs) && allunique(fs) ||
        throw(ArgumentError("recording frames must be non-empty and strictly increasing"))
    isfinite(framerate) && framerate > 0 || throw(ArgumentError("invalid recording framerate"))
    directory = abspath(directory)
    ispath(directory) && error("recording already exists: $directory")
    mkpath(dirname(directory))
    names = Symbol[]
    mktempdir(dirname(directory)) do tmp
        writers = Dict{Symbol,RecordingWriter}()
        try
            for (i,f) in enumerate(fs)
                samples = Dict(Symbol(k)=>v for (k,v) in pairs(sample!(f,framerate)))
                if i == 1
                    append!(names,sort!(collect(keys(samples))))
                    isempty(names) && error("no animation outputs to record")
                    for (j,name) in enumerate(names)
                        writers[name] = RecordingWriter(joinpath(tmp,"$j.veanim"),samples[name])
                    end
                end
                Set(keys(samples)) == Set(names) || error("animation outputs changed at frame $f")
                for name in names; recordsample!(writers[name],f,samples[name]); end
                progress === nothing || progress(i,length(fs))
            end
            for writer in values(writers); finishrecording!(writer,framerate); end
            manifest = Dict(String(name)=>basename(writers[name].path) for name in names)
            write(joinpath(tmp,"animation.msgpack"),MsgPack.pack(manifest))
            mv(tmp,directory)
        finally
            for writer in values(writers); isopen(writer.io) && close(writer.io); end
        end
    end
    return Dict(name=>openrecording(joinpath(directory,"$i.veanim")) for (i,name) in enumerate(names))
end

"The recording supplying this parameter, if any."
function recordedinput(p::Param)
    n = p.input
    n === nothing || n.op !== :copy || length(n.resolved) != 1 ? nothing :
        only(n.resolved) isa RecordedTrack ? only(n.resolved) : nothing
end
inputanimated(track::RecordedTrack) = track.varying
