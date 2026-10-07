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
    total = seqlength(seq)
    duration = total / fps
    inputs = `-i $rendered`
    inputidx = Dict{String, Int}()
    for clip in seq.clips
        path = audiopath(clip.source)
        if !haskey(inputidx, path) && hasaudio(path)
            inputidx[path] = length(inputidx) + 1
            inputs = `$inputs -i $path`
        end
    end
    norm = "aresample=$samplerate,aformat=channel_layouts=stereo"
    # A silent bed fixes the duration; branches are placed rather than
    # concatenated, so overlapping tracks and gaps both keep their timing.
    branches = ["anullsrc=r=$samplerate:cl=stereo,atrim=duration=$duration[bed]"]
    pads = ["[bed]"]
    for (i, clip) in enumerate(seq.clips)
        idx = get(inputidx, audiopath(clip.source), nothing)
        idx === nothing && continue
        t0 = clip.src_in / clip.source.framerate
        t1 = clip.src_out / clip.source.framerate
        speed = clip.rate * fps / clip.source.framerate
        tempo = String[]
        while speed > 2
            push!(tempo, "atempo=2"); speed /= 2
        end
        while speed < 0.5
            push!(tempo, "atempo=0.5"); speed *= 2
        end
        push!(tempo, "atempo=$speed")
        delay = round(Int, clip.start / fps * samplerate)
        push!(branches, "[$idx:a:0]atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS," *
              "$norm,$(join(tempo, ',')),atrim=duration=$(cliplength(clip)/fps)," *
              "adelay=$(delay)S:all=1[c$i]")
        push!(pads, "[c$i]")
    end
    mktempdir() do dir
        nar = narrationwav(seq, dir)
        if nar !== nothing
            inputs = `$inputs -i $nar`
            push!(branches, "[$(length(inputidx) + 1):a:0]$norm[nar]")
            push!(pads, "[nar]")
        end
        graph = join(branches, ";") * ";$(join(pads))amix=inputs=$(length(pads)):" *
                "duration=first:normalize=0:dropout_transition=0[aout]"
        run(`$(FFMPEG_jll.ffmpeg()) -v error -y $inputs -filter_complex $graph
             -map 0:v:0 -map "[aout]" -frames:v $total -c:v copy -c:a aac $outpath`)
    end
    return outpath
end
