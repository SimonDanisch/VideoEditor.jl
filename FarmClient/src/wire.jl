# The render farm's wire format: a message is a TOML header and a byte payload,
# each length-prefixed. Plain TOML and bytes rather than Julia serialization, so
# the editor, a machine's farm client and the renderers it starts (each possibly
# another Julia version or VideoEditor commit) still read each other.
#
# Standard library only, and shared: FarmClient includes it, and so does
# VideoEditor (`include`d from this repository's `FarmClient/src`).

"""The version of the farm's messages; the farm server refuses a client speaking another."""
const FARM_PROTOCOL = 2

"""The port the farm server listens on unless told otherwise."""
const FARM_PORT = 7600

"""
    sendmessage(io, header, payload = UInt8[])

Write one message: `header`, a dictionary TOML can hold, and `payload`, bytes
(a PNG, a job archive).
"""
function sendmessage(io::IO, header::AbstractDict, payload::AbstractVector{UInt8} = UInt8[])
    text = sprint(TOML.print, header)
    write(io, hton(UInt32(sizeof(text))), text, hton(UInt64(length(payload))), payload)
    flush(io)
    return nothing
end

"Write an existing stream without materializing a multi-gigabyte archive."
function sendmessage(io::IO, header::AbstractDict, payload::IO, count::Integer)
    text = sprint(TOML.print, header)
    write(io, hton(UInt32(sizeof(text))), text, hton(UInt64(count)))
    copypayload!(io, payload, count)
    flush(io)
    return nothing
end

"Copy exactly one payload, using at most one MiB of working memory."
function copypayload!(destination::IO, source::IO, count::Integer)
    count >= 0 || error("invalid farm payload length $count")
    buffer = Vector{UInt8}(undef, min(count, 1024 * 1024))
    remaining = count
    while remaining > 0
        wanted = min(remaining, length(buffer))
        wanted < length(buffer) && resize!(buffer, wanted)
        got = readbytes!(source, buffer, wanted)
        got == wanted || error("farm connection closed inside a message ($got of $wanted bytes)")
        write(destination, buffer)
        remaining -= got
    end
    return destination
end

"""
    receivemessage(io) -> (header, payload)

Read one message. A connection that closes inside a message is an error; one
that closes between messages is `eof(io)`, for the caller to check first.
"""
function receivemessage(io::IO; sink::Union{IO,Nothing}=nothing)
    count=Int(ntoh(read(io,UInt32)))
    0<count<=1024*1024 || error("invalid farm header length $count; the peer may have printed to the protocol stream")
    header = TOML.parse(String(readexactly(io,count)))
    size = Int(ntoh(read(io, UInt64)))
    payload = sink === nothing ? readexactly(io, size) : copypayload!(sink, io, size)
    return header, payload
end

"""`n` bytes of `io`, or an error if it closes before."""
function readexactly(io::IO, n::Integer)
    bytes = read(io, n)
    length(bytes) == n || error("farm connection closed inside a message ($(length(bytes)) of $n bytes)")
    return bytes
end

"""A message's error, if it reports one: what a client, server or renderer sends instead of a result."""
farmerror(header::AbstractDict) = get(header, "error", nothing)
