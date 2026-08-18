"""
Speech-to-text captions, from `WhisperRunner`.

The fifth JuliaVision model in the editor, and the only one that produces TEXT
rather than pixels — so it lands as an overlay on the sequence rather than an
effect on a clip. That is also where it belongs semantically: a caption tracks
what is being said across the whole timeline, and a cut underneath it changes
nothing about when a word is spoken.

The transcript is stored once on the sequence and the overlay picks the line for
the frame being rendered. It is NOT baked into per-frame text params, because
those would have to be keyframed — a thousand keys on a `text` field, unreadable
and unfixable — and because the transcript is the thing a user edits when the
model mishears a word.
"""

"""
    registertranscribe!(f)

Install the speech model. `f(samples, rate) -> Vector{Caption}` takes MONO
Float32 samples and their sample rate.

Mono and a stated rate rather than the editor's stereo Int16: every model wants
something different, and the conversion belongs on this side of the boundary
where the editor's audio layout is known.
"""
const SPEECHMODEL = Ref{Any}(nothing)

registertranscribe!(f) = (SPEECHMODEL[] = f; nothing)
hasspeechmodel() = SPEECHMODEL[] !== nothing

"The built-in speech model: Whisper, from `WhisperRunner`. Built on first use."
const WHISPER = Ref{Any}(nothing)

function whispertranscribe(samples::AbstractVector{Float32}, rate::Real)
    if WHISPER[] === nothing
        WHISPER[] = WhisperRunner.whisper(; backend = Lava.LavaBackend())
    end
    # The runner defines its own sample rate and resamples nothing, so handing it
    # audio at another rate transcribes a pitch-shifted signal — which comes back
    # as plausible words rather than as an error.
    r = Int(WhisperRunner.SAMPLERATE)
    audio = rate == r ? samples : resampleaudio(samples, rate, r)
    # `transcribe` returns `(text, segments)` — the whole transcript AND the timed
    # pieces. Iterating the tuple walks the String first, so every run died on
    # `s.text` with "type String has no field text". The editor wants the SEGMENTS:
    # a caption needs a start and a stop, which the joined text does not have.
    _, segs = WhisperRunner.transcribe(WHISPER[], audio)
    return [Caption(s.start, s.stop, strip(s.text)) for s in segs if !isempty(strip(s.text))]
end

installtranscribe!() = registertranscribe!(whispertranscribe)

"""
    resampleaudio(x, from, to) -> Vector{Float32}

Linear resample. Good enough here and nowhere else: speech recognition is
band-limited well below where linear interpolation's aliasing lives, and the
alternative is a filter design nobody would tune. Do not reuse this for audio
that will be HEARD.
"""
function resampleaudio(x::AbstractVector{Float32}, from::Real, to::Real)
    from == to && return collect(x)
    n = max(1, floor(Int, length(x) * to / from))
    out = Vector{Float32}(undef, n)
    step = from / to
    @inbounds for i in 1:n
        t = (i - 1) * step + 1
        j = floor(Int, t)
        f = Float32(t - j)
        a = x[clamp(j, 1, length(x))]
        b = x[clamp(j + 1, 1, length(x))]
        out[i] = (1 - f) * a + f * b
    end
    return out
end

"""
    sequenceaudio(seq) -> (Vector{Float32}, rate)

The timeline's audio as mono Float32, and its rate.

Mixed through [`fillaudio!`](@ref) rather than read from a source, so what gets
transcribed is what the edit PLAYS: cuts, gaps and clip order included. A caption
for audio the edit removed is worse than no caption at all.

Extraction is synchronous here, unlike the preview's — `ensurepcm!` spawns and
returns because playback cannot wait, and a transcript can and must.
"""
function sequenceaudio(seq::Sequence)
    total = seqlength(seq)
    total > 0 || return (Float32[], AUDIORATE)
    tracks = Dict{String, Union{PCMTrack, Nothing}}()
    for c in seq.clips
        haskey(tracks, c.source.path) && continue
        tracks[c.source.path] = loadpcm(c.source)
    end
    all(isnothing, values(tracks)) && return (Float32[], AUDIORATE)
    nsamp = ceil(Int, total / seq.framerate * AUDIORATE)
    buf = zeros(Int16, 2, nsamp)
    fillaudio!(buf, seq, tracks, 0)
    mono = Vector{Float32}(undef, nsamp)
    @inbounds for i in 1:nsamp
        mono[i] = (Float32(buf[1, i]) + Float32(buf[2, i])) / (2 * 32768f0)
    end
    return (mono, AUDIORATE)
end

"""
    captionat(caps, seconds) -> String

What is being said at `seconds`, or `""`.

The LAST match wins when segments overlap, which they do at window boundaries:
the later one is the more recent guess and reads correctly when a word is split
across two windows.
"""
function captionat(caps, t::Real)
    out = ""
    for c in caps
        c.start <= t <= c.stop && (out = c.text)
    end
    return out
end

"""
    transcribe!(seq; progress = nothing) -> Vector{Caption}

Transcribe the timeline's audio and store the result on `seq`.
"""
function transcribe!(seq::Sequence)
    hasspeechmodel() || error("no speech model installed — see registertranscribe!")
    audio, rate = sequenceaudio(seq)
    isempty(audio) && error("this timeline has no audio to transcribe")
    caps = SPEECHMODEL[](audio, rate)
    empty!(seq.captions)                  # the field is `const`: replace contents
    append!(seq.captions, caps)
    return seq.captions
end

# The caption overlay. `frame`, `framerate` and `captions` are all in every
# overlay's state (see `overlaystate`), so it finds its own line rather than being
# told — which is what keeps the transcript ONE object on the sequence instead of
# a keyframed text param with a key per spoken word.
registeroverlay!(:captions, "Captions",
    [FxParam(:y, "Y"; min = 0.0, max = 1.0, default = 0.12),
     FxParam(:size, "Size"; min = 0.01, max = 0.2, default = 0.05),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    function (scene, canvas, state)
        W, H = canvas
        Makie.text!(scene, olift(s -> Makie.Point2f(0.5 * W, s.y * H), state);
                    text = olift(state) do s
                        fr = s.framerate > 0 ? s.framerate : 25.0
                        captionat(s.captions, s.frame / fr)
                    end,
                    fontsize = olift(s -> Float32(s.size * H), state),
                    color = olift(s -> fadedcolor(:white, s.opacity), state),
                    strokecolor = olift(s -> fadedcolor(:black, s.opacity), state),
                    strokewidth = olift(s -> Float32(0.12 * s.size * H), state),
                    align = (:center, :center))
        return nothing
    end)


"""
    captionindexat(seq, frame) -> Int

Which caption covers timeline `frame`, or 0.

Frames here, seconds in the [`Caption`](@ref): the UI asks with a playhead and
the transcript is stored in the model's units, so the conversion happens at the
boundary rather than in either of them.
"""
function captionindexat(seq::Sequence, frame::Integer)
    fps = seq.framerate > 0 ? seq.framerate : 25.0
    t = frame / fps
    idx = 0
    for (i, c) in enumerate(seq.captions)
        c.start <= t <= c.stop && (idx = i)     # last match, like `captionat`
    end
    return idx
end

"""
    editcaption!(player, text) -> Bool

Replace the text of the caption under the playhead.

A transcript is a *guess*, and the one thing a user reliably wants to do with a
guess is correct it. Without this the Whisper integration is read-only: you can
generate captions, see them, move them and restyle them, and the moment the model
mishears a name your only recourse is to re-run it and get the same answer.

Undoable, because a correction is an edit — and because the next full transcribe
replaces the whole transcript, which is exactly when you want the old wording
back.
"""
function editcaption!(player::Player, text::AbstractString)
    seq = player.sequence
    i = captionindexat(seq, player.playhead[])
    i == 0 && (setstatus!(player, "captions: none under the playhead"); return false)
    snapshot!(player)
    old = seq.captions[i]
    seq.captions[i] = Caption(old.start, old.stop, String(text))
    notify(player.playhead)
    setstatus!(player, "caption updated")
    return true
end
