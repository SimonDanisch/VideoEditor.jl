"""
Spoken narration, from `KokoroRunner`.

The seventh JuliaVision model in the editor, and the second that produces sound
rather than pixels — but unlike the captions it produces something the timeline
has to play and export, which is the difficulty. This editor has no
audio-only clip: audio arrives with a video source and is mixed from the clip
list. Narration is therefore a sequence-level track, mixed over that list on both
paths — [`fillaudio!`](@ref) for the preview and `muxaudio` for the export.

Doing only one of the two would be worse than doing neither: a voiceover you can
hear while editing and that vanishes from the render is a bug that surfaces at
the end of the job.
"""

"""
    registerspeak!(f)

Install the speech synthesizer. `f(text, voice) -> (samples::Vector{Float32},
rate)`, mono.
"""

registerspeak!(f; voices = nothing) = (INSTALLED.speak = f; INSTALLED.voices = voices; nothing)
hasspeakmodel() = INSTALLED.speak !== nothing

"""
    INSTALLED.voices

`() -> Vector{String}`, or `nothing`: which voices the installed synthesizer can
use. Separate from the synthesizer itself because it has to be cheap — the panel
asks on every rebuild, and a list that built a model to answer would freeze the
UI to draw a menu.
"""

"The installed synthesizer's voices, or empty if it has none to offer yet."
speakvoices() = INSTALLED.voices === nothing ? String[] : INSTALLED.voices()

"The built-in synthesizer: Kokoro, from `KokoroRunner`. Built on first use."

function kokorospeak(text::AbstractString, voice::AbstractString)
    if INSTALLED.kokoro === nothing
        INSTALLED.kokoro = KokoroRunner.Kokoro(; backend = Mantle.defaultbackend())
    end
    samples = KokoroRunner.speak(INSTALLED.kokoro, String(text); voice = String(voice))
    return (Vector{Float32}(vec(Array(samples))), Int(KokoroRunner.SAMPLERATE))
end

"""
Kokoro's usable voices, or empty until the model is loaded.

Filtered to `af_`/`am_`/`bf_`/`bm_`: Kokoro ships 54, but the rest are for
languages this package has no G2P for — offering them would be offering 40-odd
ways to produce noise.

Empty before the first render on purpose: the names live in the loaded voicepack,
and building a model to populate a menu would cost seconds on a panel rebuild.
The picker appears once you have spoken one line, which is also when you first
have a reason to want it.
"""
kokorovoices() = INSTALLED.kokoro === nothing ? String[] :
    filter(v -> startswith(v, "af_") || startswith(v, "am_") ||
                startswith(v, "bf_") || startswith(v, "bm_"),
           KokoroRunner.voices(INSTALLED.kokoro))

installspeak!() = registerspeak!(kokorospeak; voices = kokorovoices)

"""
    render!(nar) -> nar

Synthesize `nar`'s text into its sample buffer, replacing whatever was there.

Separate from construction because the text is the EDIT and the samples are a
cache of it: a project file carries the words and re-renders on demand, which is
what keeps a minute of speech from becoming five megabytes of JSON.
"""
function render!(nar::Narration)
    hasspeakmodel() || error("no speech synthesizer installed — see registerspeak!")
    isempty(strip(nar.text)) && error("nothing to say")
    samples, rate = INSTALLED.speak(nar.text, nar.voice)
    empty!(nar.samples); append!(nar.samples, samples)
    nar.rate = rate
    return nar
end

"""
    render(nar) -> Narration

A FRESH `Narration` with the same words, spoken again — the non-mutating twin of
[`render!`](@ref).

Which one a caller wants is decided by undo, not by taste. `docsnapshot` shares
the `Narration` objects it snapshots rather than copying their samples, so a
re-render that wrote through the shared one would silently rewrite the audio
inside every undo step that held it. Re-rendering a line already in the sequence
therefore REPLACES the element; `render!` stays for the line being built, which
nothing else can be holding yet.
"""
render(nar::Narration) = render!(Narration(nar.text, nar.at, nar.voice))

"""
    narrate!(seq, text; at = 0.0, voice = "af_heart") -> Narration

Add a spoken line to the sequence at `at` seconds and render it.
"""
function narrate!(seq::Sequence, text::AbstractString; at::Real = 0.0,
                  voice::AbstractString = "af_heart")
    nar = Narration(String(text), Float64(at), String(voice))
    render!(nar)
    push!(seq.narration, nar)
    return nar
end

"""
    mixnarration!(out, seq, startsample; rate)

Add every rendered narration over `out`, which already holds the clips' audio.

Adds rather than writes, which is why narration is a second pass
over the block instead of another branch inside [`fillaudio!`](@ref)'s clip loop:
a voiceover plays *over* the timeline, not instead of it.

Clipped, not normalized. Normalizing would make the mix quieter the moment a
narration is added, which reads as the edit changing when only the monitoring
did; clipping is audible and localized, and the fix is the user's volume.
"""
function mixnarration!(out::AbstractMatrix{Int16}, seq::Sequence, startsample::Integer;
                       rate::Integer = AUDIORATE)
    isempty(seq.narration) && return out
    blocklen = size(out, 2)
    for nar in seq.narration
        isempty(nar.samples) && continue          # never rendered
        # Narration samples run at the model's rate; the block is at the mixer's.
        step = nar.rate / rate
        base = nar.at * rate                      # where it starts, in block samples
        for i in 1:blocklen
            t = (startsample + i - 1) - base
            t < 0 && continue
            j = t * step + 1
            j >= length(nar.samples) && continue
            k = floor(Int, j)
            f = Float32(j - k)
            a = nar.samples[k]
            b = nar.samples[min(k + 1, length(nar.samples))]
            v = round(Int, ((1 - f) * a + f * b) * 32767)
            @inbounds for ch in 1:size(out, 1)
                out[ch, i] = Int16(clamp(Int(out[ch, i]) + v, -32768, 32767))
            end
        end
    end
    return out
end

"""
    narrationwav(seq, dir) -> path | nothing

Every rendered narration mixed onto one silent bed as a WAV, for the exporter to
hand ffmpeg. `nothing` when there is nothing to say.

A file rather than a filter chain because `muxaudio` already speaks in inputs and
branches, and one more input is a smaller change to it than teaching it to
synthesize.
"""
function narrationwav(seq::Sequence, dir::AbstractString; rate::Integer = AUDIORATE)
    any(n -> !isempty(n.samples), seq.narration) || return nothing
    total = ceil(Int, seqlength(seq) / seq.framerate * rate)
    total > 0 || return nothing
    bed = zeros(Int16, 2, total)
    mixnarration!(bed, seq, 0; rate)
    path = joinpath(dir, "narration.wav")
    raw = joinpath(dir, "narration.pcm")
    write(raw, reinterpret(UInt8, vec(bed)))
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f s16le -ac 2 -ar $rate -i $raw $path`,
                 stdout = devnull, stderr = devnull))
    return path
end
