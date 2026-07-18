"""
    DecodeWorker(source, ring; lookbehind=8, seek_threshold=48)

Background decoder that owns a `VideoReader` on its own thread and keeps
`ring` filled ahead of the playhead.

The UI communicates through `settarget!(worker, n)` — the frame index the
playhead wants next. The worker decodes sequentially towards
`target + lookahead`; it only seeks when the target jumps backwards or
further ahead than `seek_threshold` frames (sequential decode is cheaper
than seek-to-keyframe + trim for short forward hops).
"""
mutable struct DecodeWorker
    const source::VideoSource
    const ring::FrameRing
    const target::Threads.Atomic{Int}
    const protectlo::Threads.Atomic{Int}  # frames still needed by playback:
    const protecthi::Threads.Atomic{Int}  # their ring slots must not be reused
    const running::Threads.Atomic{Bool}
    const lookbehind::Int
    const seek_threshold::Int
    task::Task

    function DecodeWorker(source::VideoSource, ring::FrameRing;
                          lookbehind::Integer = 8, seek_threshold::Integer = 48)
        worker = new(source, ring, Threads.Atomic{Int}(0),
                     Threads.Atomic{Int}(1), Threads.Atomic{Int}(0),
                     Threads.Atomic{Bool}(true), lookbehind, seek_threshold)
        worker.task = Threads.@spawn decodeloop(worker)
        return worker
    end
end

"""
    settarget!(worker, n; protect = 1:0)

Aim the decoder at source frame `n`. `protect` marks frames whose ring
slots must not be overwritten — used by cut prefetching, where the worker
jumps ahead to the next clip while the tail of the current one still plays.
"""
function settarget!(worker::DecodeWorker, n::Integer; protect::UnitRange{<:Integer} = 1:0)
    worker.protectlo[] = first(protect)
    worker.protecthi[] = last(protect)
    worker.target[] = n
    return nothing
end

function stop!(worker::DecodeWorker)
    worker.running[] = false
    wait(worker.task)
    return nothing
end

function decodeloop(worker::DecodeWorker)
    source, ring = worker.source, worker.ring
    lookahead = capacity(ring) - worker.lookbehind - 2
    reader = VideoIO.openvideo(source.path, target_format = VideoIO.AV_PIX_FMT_RGB24)
    position = 0  # next frame index the reader will produce
    try
        while worker.running[]
            target = worker.target[]
            hi = min(target + lookahead, source.nframes - 1)
            if target - position > worker.seek_threshold
                # target jumped far ahead: keyframe seek + trim beats decoding forward
                seek(reader, frametime(source, target))
                position = target
            elseif position > hi
                if hasframe(ring, target)
                    sleep(0.002)  # window is filled (or a back-scrub hit cached frames)
                    continue
                else
                    # target jumped backwards past what the ring still holds
                    seek(reader, frametime(source, target))
                    position = target
                end
            end
            # never overwrite a slot holding a frame playback still needs
            lo, hi = worker.protectlo[], worker.protecthi[]
            if lo <= hi
                held = lock(() -> ring.indices[slotindex(ring, position)], ring.lock)
                if lo <= held <= hi && held != position
                    sleep(0.002)
                    continue
                end
            end
            buf = claimslot!(ring, position)
            read!(reader, PermutedDimsArray(buf, (2, 1)))
            publishslot!(ring, position)
            position += 1
        end
    catch e
        e isa EOFError || @error "decode worker died" exception = (e, catch_backtrace())
    finally
        close(reader)
    end
    return nothing
end
