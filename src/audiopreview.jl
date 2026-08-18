# Audio preview during playback — zero extra dependencies.
#
# Each source's audio is extracted once (ffmpeg → interleaved s16le stereo
# 48 kHz) into the package scratch cache and memory-mapped, so even long
# sources cost no RAM. During playback a feeder task walks the cut list
# with the pure mixer `fillaudio!` (clips → source samples, gaps and
# audio-less sources → silence) and pipes blocks into PipeWire's `pw-cat`.
# Pipe buffering gives roughly ~0.1–0.2 s of latency — fine for preview;
# export audio (audio.jl) stays sample-exact.
#
# Everything is guarded: no `pw-cat`, no audio streams, or extraction still
# running simply means silent playback (with a one-time status note).

const AUDIORATE = 48_000
const AUDIOBLOCK = 4096  # samples per pipe write (~85 ms)

"Interleaved stereo Int16 PCM of `source`, memory-mapped from the scratch cache."
mutable struct PCMTrack
    samples::Matrix{Int16}  # (2, nsamples) view of the mmap
    nsamples::Int
end

"Per-player audio preview state."
mutable struct AudioPreview
    tracks::Dict{String, Union{PCMTrack, Nothing}}  # nothing = no audio / pending
    proc::Any            # the pw-cat process (nothing when stopped)
    feeder::Any          # feeder task
    warned::Bool
    enabled::Bool        # the Sound/Muted toggle
end
AudioPreview() = AudioPreview(Dict{String, Union{PCMTrack, Nothing}}(), nothing, nothing, false, true)

"Scratch-cached PCM path for `source` (same identity key scheme as proxies)."
function pcmpath(source::VideoSource)
    st = stat(source.path)
    key = string(hash((abspath(source.path), st.size, st.mtime, :pcm48)), base = 16)
    return joinpath(cachedir("pcm"), key * ".s16")
end

"""
Extract and mmap `source`'s audio (blocking; run on a background thread).
Returns `nothing` when the source has no audio stream.
"""
function loadpcm(source::VideoSource)
    hasaudio(source.path) || return nothing
    path = pcmpath(source)
    if !isfile(path)
        tmp = path * ".part"
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -i $(source.path)
                      -f s16le -ac 2 -ar $(AUDIORATE) $tmp`,
                     stdout = devnull, stderr = devnull))
        mv(tmp, path; force = true)
    end
    n = filesize(path) ÷ 4  # 2 channels × 2 bytes
    io = open(path, "r")
    samples = Mmap.mmap(io, Matrix{Int16}, (2, n))
    close(io)
    return PCMTrack(samples, n)
end

"""
    fillaudio!(out, seq, tracks, startsample) -> out

The pure cut-list mixer: fill the interleaved stereo block `out`
(`(2, blocklen)` Int16) with the sequence's audio starting at timeline
sample `startsample` (48 kHz). Walks clips span-by-span — source audio for
clips whose source has a loaded `PCMTrack`, silence for gaps, audio-less
or still-extracting sources, and past the sequence end.
"""
function fillaudio!(out::AbstractMatrix{Int16}, seq::Sequence,
                    tracks::Dict{String, Union{PCMTrack, Nothing}}, startsample::Integer)
    blocklen = size(out, 2)
    fps = seq.framerate
    pos = 0
    while pos < blocklen
        tsample = startsample + pos
        frame = floor(Int, tsample * fps / AUDIORATE)
        loc = locate(seq, frame)
        if loc === nothing
            # gap (or past the end): silent until the next clip starts
            nxt = minimum((c.start for c in seq.clips if c.start > frame); init = typemax(Int))
            span = nxt == typemax(Int) ? blocklen - pos :
                   min(blocklen - pos, ceil(Int, nxt / fps * AUDIORATE) - tsample)
            fill!(view(out, :, (pos + 1):(pos + max(span, 1))), Int16(0))
            pos += max(span, 1)
            continue
        end
        clip, srcframe = loc
        # samples left inside this clip on the timeline
        clipendsample = ceil(Int, clipend(clip) / fps * AUDIORATE)
        span = min(blocklen - pos, max(clipendsample - tsample, 1))
        track = get(tracks, clip.source.path, nothing)
        if track === nothing
            fill!(view(out, :, (pos + 1):(pos + span)), Int16(0))
        else
            # timeline sample → source sample: through the clip's source frame, and
            # from there at the SOURCE's rate. Dividing by the sequence rate was the
            # same number only while every source ran at it; conformed clips would
            # otherwise drift their audio against their picture.
            tframe = tsample / AUDIORATE * fps - clip.start
            srcframe = clip.src_in + tframe * clip.rate
            srcsample = round(Int, srcframe / clip.source.framerate * AUDIORATE)
            avail = clamp(track.nsamples - srcsample, 0, span)
            if avail > 0
                copyto!(view(out, :, (pos + 1):(pos + avail)),
                        view(track.samples, :, (srcsample + 1):(srcsample + avail)))
            end
            avail < span && fill!(view(out, :, (pos + avail + 1):(pos + span)), Int16(0))
        end
        pos += span
    end
    # Narration plays OVER the cut list, so it is a second pass that ADDS rather
    # than another branch in the loop above, which writes.
    mixnarration!(out, seq, startsample; rate = AUDIORATE)
    return out
end

"Ensure `source`'s PCM is loaded (kicks off background extraction once)."
function ensurepcm!(player::Player, source::VideoSource)
    ap = player.audio
    haskey(ap.tracks, source.path) && return nothing
    ap.tracks[source.path] = nothing  # pending / no audio
    Threads.@spawn try
        track = loadpcm(source)
        track === nothing || (ap.tracks[source.path] = track)
    catch e
        @error "audio extraction failed" source = source.path exception = e
    end
    return nothing
end

"Start feeding the cut list's audio to PipeWire (called by `play!`)."
function startaudio!(player::Player)
    ap = player.audio
    ap isa AudioPreview || return nothing
    ap.enabled || return nothing
    stopaudio!(player)
    if Sys.which("pw-cat") === nothing
        if !ap.warned
            ap.warned = true
            setstatus!(player, "audio preview off — pw-cat (PipeWire) not found")
        end
        return nothing
    end
    foreach(c -> ensurepcm!(player, c.source), player.sequence.clips)
    proc = open(pipeline(`pw-cat -p --raw --rate $(AUDIORATE) --channels 2 --format s16 -`,
                         stderr = devnull), "w")
    ap.proc = proc
    ap.feeder = Threads.@spawn try
        block = Matrix{Int16}(undef, 2, AUDIOBLOCK)
        cursor = round(Int, player.playhead[] / player.sequence.framerate * AUDIORATE)
        while player.playing[] && process_running(proc)
            expected = round(Int, player.playhead[] / player.sequence.framerate * AUDIORATE)
            abs(cursor - expected) > AUDIORATE ÷ 4 && (cursor = expected)  # resync on scrubs
            fillaudio!(block, player.sequence, ap.tracks, cursor)
            write(proc, block)   # pipe backpressure paces the feed
            cursor += AUDIOBLOCK
        end
    catch e
        e isa Union{EOFError, Base.IOError, InvalidStateException} ||
            @error "audio feeder died" exception = (e, catch_backtrace())
    finally
        stopaudio!(player)
    end
    return nothing
end

"Stop the audio feed and its PipeWire process (called by `pause!`/`close`)."
function stopaudio!(player::Player)
    ap = player.audio
    ap isa AudioPreview || return nothing
    proc = ap.proc
    ap.proc = nothing
    if proc !== nothing && process_running(proc)
        try
            kill(proc)     # first — a blocked feeder may hold the pipe open
        catch
        end
        try
            close(proc.in)
        catch
        end
    end
    return nothing
end
