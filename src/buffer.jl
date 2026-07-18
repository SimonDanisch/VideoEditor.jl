"""
    FrameRing(width, height; capacity=64)

Preallocated ring of decoded frames, single producer (decode worker) /
single consumer (UI). Frame `n` lives in slot `mod(n, capacity) + 1`.

Frames are stored in parent orientation (width × height, width-contiguous) —
the exact memory layout VideoIO decodes into and GLMakie's `image!` displays,
so neither side ever transposes.

Protocol: the producer invalidates a slot under the lock, decodes into it
unlocked, then validates it under the lock. The consumer copies out valid
slots under the lock, so it can never observe a half-written frame.
"""
struct FrameRing
    slots::Vector{RGBFrame}
    indices::Vector{Int}
    lock::ReentrantLock
end

function FrameRing(width::Integer, height::Integer; capacity::Integer = 64)
    slots = [RGBFrame(undef, width, height) for _ in 1:capacity]
    return FrameRing(slots, fill(-1, capacity), ReentrantLock())
end

capacity(ring::FrameRing) = length(ring.slots)
slotindex(ring::FrameRing, n::Integer) = mod(n, capacity(ring)) + 1

function hasframe(ring::FrameRing, n::Integer)
    lock(ring.lock) do
        ring.indices[slotindex(ring, n)] == n
    end
end

"""
    fetchframe!(dest, ring, n) -> Bool

Copy frame `n` into `dest` if present. Returns whether the frame was found.
"""
function fetchframe!(dest::RGBFrame, ring::FrameRing, n::Integer)
    slot = slotindex(ring, n)
    lock(ring.lock) do
        ring.indices[slot] == n || return false
        copyto!(dest, ring.slots[slot])
        return true
    end
end

"Producer side: claim the slot for frame `n`. Returns the slot's buffer."
function claimslot!(ring::FrameRing, n::Integer)
    slot = slotindex(ring, n)
    lock(ring.lock) do
        ring.indices[slot] = -1
    end
    return ring.slots[slot]
end

"Producer side: mark frame `n` as complete after decoding into its claimed slot."
function publishslot!(ring::FrameRing, n::Integer)
    lock(ring.lock) do
        ring.indices[slotindex(ring, n)] = n
    end
    return nothing
end

function Base.empty!(ring::FrameRing)
    lock(ring.lock) do
        fill!(ring.indices, -1)
    end
    return ring
end
