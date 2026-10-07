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
hasspeakmodel() = INSTALLED.speak !== nothing || !isempty(SPEECH_MODELS)

struct SpeechModel
    label::String
    synthesize::Any
    voices::Any
    direction::Bool
    reference::Bool
end
const SPEECH_MODELS = Dict{Symbol,SpeechModel}()

"Register a lazy speech provider: `synthesize(narration) -> (mono_samples, rate)`."
function registerspeechmodel!(name::Symbol, label, synthesize;
                              voices=() -> String[], direction=false, reference=false)
    SPEECH_MODELS[name] = SpeechModel(String(label), synthesize, voices, direction, reference)
    return nothing
end
speechmodels() = sort!(collect(SPEECH_MODELS); by=p -> p.second.label)
speechmodel(n::Narration) = get(SPEECH_MODELS, n.speech.model, nothing)

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

function installspeak!()
    registerspeak!(kokorospeak; voices = kokorovoices)
    registerspeechmodel!(:kokoro, "Kokoro · English", n -> kokorospeak(n.text, n.voice);
                         voices=kokorovoices)
end

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
    settings = nar.speech
    model = speechmodel(nar)
    if settings.model === :default
        INSTALLED.speak === nothing && error("the project's default speech model is not installed")
        isempty(settings.direction) && isempty(settings.reference) ||
            error("the default speech provider does not support delivery instructions or reference audio")
        samples, rate = INSTALLED.speak(nar.text, nar.voice)
    else
        model === nothing && error("speech model $(settings.model) is not registered in this session")
        model.direction || isempty(settings.direction) || error("$(model.label) does not support delivery instructions")
        model.reference || isempty(settings.reference) || error("$(model.label) does not support reference audio")
        isempty(settings.reference) || isfile(settings.reference) || error("voice reference is missing: $(settings.reference)")
        samples, rate = Base.invokelatest(model.synthesize, nar)
    end
    rate > 0 && !isempty(samples) && all(isfinite, samples) || error("speech provider returned invalid audio")
    samples = filtervoice(samples, rate, nar.mix.filter)
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
render(nar::Narration) = render!(narrationcopy(nar))

function filtervoice(samples, rate, filter)
    isempty(filter) && return Float32.(samples)
    raw = reinterpret(UInt8, Float32.(samples))
    bytes = read(pipeline(`$(FFMPEG_jll.ffmpeg()) -v error -f f32le -ar $rate -ac 1 -i -
                           -af $filter -f f32le -ar $rate -ac 1 -`; stdin=IOBuffer(raw)))
    return collect(reinterpret(Float32, bytes))
end

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

"Timeline time of a source-anchored line after cuts, or `nothing` when cut away."
function narrationtime(seq::Sequence,n::Narration)
    n.anchor === nothing && return n.at
    stop = n.rate > 0 ? n.at + min(length(n.samples)/n.rate,n.mix.stop) : n.at
    for c in sort(seq.clips;by=c->c.start)
        c.source === n.anchor || continue
        a,b = c.src_in/c.source.framerate,c.src_out/c.source.framerate
        (a <= n.at < b || n.at < a < stop) || continue
        speed = c.rate*seq.framerate/c.source.framerate
        return c.start/seq.framerate + (max(n.at,a)-a)/speed
    end
    return nothing
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
    for nar in seq.narration
        nar.anchor === nothing || continue
        mixvoiceblock!(out,nar,startsample,rate,0.0,1.0,0.0,Inf)
    end
    # Only the picture clips intersecting this audio block can contribute
    # anchored speech. No per-sample clip lookup, and no separate cut clock.
    blockend = (startsample+size(out,2))/rate
    for c in seq.clips
        a,b = c.start/seq.framerate,clipend(c)/seq.framerate
        (a < blockend && b > startsample/rate) || continue
        speed = c.rate*seq.framerate/c.source.framerate
        offset = c.src_in/c.source.framerate - a*speed
        for nar in seq.narration
            nar.anchor === c.source || continue
            mixvoiceblock!(out,nar,startsample,rate,offset,speed,a,b)
        end
    end
    return out
end

function mixvoiceblock!(out,nar,startsample,rate,offset,speed,beginat,endat)
    isempty(nar.samples) && return out
    # Restrict work to the audible intersection of take, picture and block.
    firstsample = max(startsample,ceil(Int,beginat*rate),ceil(Int,(nar.at-offset)/speed*rate))
    stop = nar.at + min(length(nar.samples)/nar.rate,nar.mix.stop)
    lastsample = min(startsample+size(out,2)-1,(isfinite(endat) ? ceil(Int,endat*rate)-1 : typemax(Int)),
                     ceil(Int,(stop-offset)/speed*rate)-1)
    firstsample <= lastsample || return out
    left = sqrt(2)*cos((clamp(nar.mix.pan,-1,1)+1)*pi/4)
    right = sqrt(2)*sin((clamp(nar.mix.pan,-1,1)+1)*pi/4)
    for absolute in firstsample:lastsample
        seconds = offset + absolute/rate*speed - nar.at
        j = max(1.0,seconds*nar.rate+1)
        k = min(floor(Int,j),length(nar.samples)); f = Float32(j-k)
        a,b = nar.samples[k],nar.samples[min(k+1,length(nar.samples))]
        gain = nar.mix.gain * (1+(nar.mix.ramp-1)*Float32(j/length(nar.samples)))
        isfinite(nar.mix.stop) && (gain *= clamp((nar.mix.stop-seconds)*rate/480,0,1))
        v = ((1-f)*a+f*b)*gain*32767
        i = absolute-startsample+1
        for ch in axes(out,1)
            out[ch,i] = Int16(clamp(Int(out[ch,i])+round(Int,v*(ch==1 ? left : right)),-32768,32767))
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
