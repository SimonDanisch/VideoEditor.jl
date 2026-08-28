"""
    hasaudio(path) -> Bool

Whether the media file contains at least one audio stream (ffprobe).
"""
# A clip that renders its frames has no file and therefore no soundtrack.
hasaudio(path::AbstractString) = isempty(path) ? false : hasaudiofile(path)

function hasaudiofile(path::AbstractString)
    out = read(`$(FFMPEG_jll.ffprobe()) -v error -select_streams a -show_entries stream=index -of csv=p=0 $path`,
               String)
    return !isempty(strip(out))
end

"""
    muxaudio(rendered, seq, outpath; samplerate=48000) -> outpath

Assemble the sequence's audio track from the cut list — one `atrim` branch
per clip in timeline order, generated silence over gaps and clips whose
source has no audio — and mux it with the rendered video (video stream
copied, audio encoded as AAC). Because the edit model is frame-exact, the
audio segments are simply the clips' source time ranges: cuts, trims and
moves stay sample-aligned with the picture.
"""
function muxaudio(rendered::AbstractString, seq::Sequence, outpath::AbstractString;
                  samplerate::Integer = 48000)
    fps = seq.framerate
    clips = sort(seq.clips; by = c -> c.start)

    inputs = `-i $rendered`
    inputidx = Dict{String, Int}()  # source path → ffmpeg input index
    for clip in clips
        path = sourcepath(clip.source)
        if !haskey(inputidx, path) && hasaudio(path)
            inputidx[path] = length(inputidx) + 1
            inputs = `$inputs -i $path`
        end
    end

    norm = "aresample=$samplerate,aformat=channel_layouts=stereo"
    branches = String[]
    silence(frames) = "aevalsrc=0:d=$(frames / fps):s=$samplerate,$norm[s$(length(branches) + 1)]"
    cursor = 0
    for clip in clips
        clip.start > cursor && push!(branches, silence(clip.start - cursor))
        idx = get(inputidx, sourcepath(clip.source), nothing)
        if idx === nothing
            push!(branches, silence(cliplength(clip)))
        else
            t0 = clip.src_in / clip.source.framerate
            t1 = clip.src_out / clip.source.framerate
            push!(branches,
                  "[$idx:a]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS,$norm[s$(length(branches) + 1)]")
        end
        cursor = clipend(clip)
    end

    pads = join(("[s$i]" for i in eachindex(branches)))
    graph = join(branches, ";") * ";$(pads)concat=n=$(length(branches)):v=0:a=1[acat]"
    # Narration is mixed OVER the cut list, not concatenated into it — the same
    # relationship the preview's `mixnarration!` has, and the reason it is a
    # separate input rather than another branch. `duration=first` keeps the
    # timeline's length authoritative when the speech runs past the end.
    nar = narrationwav(seq, mktempdir())
    if nar === nothing
        graph *= ";[acat]anull[aout]"
    else
        inputs = `$inputs -i $nar`
        graph *= ";[$(length(inputidx) + 1):a]$norm[nar];[acat][nar]amix=inputs=2:" *
                 "duration=first:dropout_transition=0,volume=2[aout]"
    end
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y $inputs -filter_complex $graph
                  -map 0:v -map "[aout]" -c:v copy -c:a aac -shortest $outpath`,
                 stdout = devnull, stderr = devnull))
    return outpath
end
