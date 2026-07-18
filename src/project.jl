"""
    saveproject(path, seq)
    loadproject(path) -> Sequence

Project files are plain TOML of the edit metadata — source paths, in/out
points, timeline positions and crops. Sources are re-probed on load.
"""
function saveproject(path::AbstractString, seq::Sequence)
    dict = Dict{String, Any}(
        "framerate" => seq.framerate,
        "clips" => [clipdict(clip) for clip in seq.clips],
    )
    open(io -> TOML.print(io, dict), path, "w")
    return path
end

function clipdict(clip::Clip)
    cd = Dict{String, Any}(
        "source" => clip.source.path,
        "src_in" => clip.src_in,
        "src_out" => clip.src_out,
        "start" => clip.start,
        "crop" => collect(clip.crop),
        "effects" => [effectdict(e) for e in clip.effects],
    )
    # stabilization tracks are part of the edit — losing an analysis on
    # save/reopen would be silently destructive
    if clip.motiontrack !== nothing
        cd["motiontrack"] = Dict{String, Any}(
            "src_in" => clip.motiontrack.src_in,
            "mode" => String(clip.motiontrack.mode),
            "transforms" => [vec(Float64.(collect(M))) for M in clip.motiontrack.transforms])
        clip.motiontrack.basecrop === nothing ||
            (cd["motiontrack"]["basecrop"] = collect(clip.motiontrack.basecrop))
    end
    if clip.colortrack !== nothing
        cd["colortrack"] = Dict{String, Any}(
            "src_in" => clip.colortrack.src_in,
            "gains" => [Float64.(collect(g)) for g in clip.colortrack.gains],
            "offsets" => [Float64.(collect(o)) for o in clip.colortrack.offsets])
    end
    return cd
end

function loadproject(path::AbstractString)
    dict = TOML.parsefile(path)
    missing_sources = unique(String[cd["source"] for cd in dict["clips"] if !isfile(cd["source"])])
    isempty(missing_sources) ||
        error("project references missing video file(s):\n  " * join(missing_sources, "\n  ") *
              "\nMove them back (or edit the paths in $path) and reload.")
    sources = Dict{String, VideoSource}()
    clips = map(dict["clips"]) do cd
        source = get!(() -> VideoSource(cd["source"]), sources, cd["source"])
        clip = Clip(source, cd["src_in"], cd["src_out"], cd["start"],
                    Tuple(Float64.(cd["crop"])))
        for ed in get(cd, "effects", [])
            push!(clip.effects, effectfromdict(ed))
        end
        if haskey(cd, "motiontrack")
            mt = cd["motiontrack"]
            clip.motiontrack = MotionTrack(
                [Mat3f(Float32.(v)...) for v in mt["transforms"]], Int(mt["src_in"]),
                Symbol(get(mt, "mode", "unknown")),
                haskey(mt, "basecrop") ? NTuple{4, Float64}(mt["basecrop"]) : nothing)
        end
        if haskey(cd, "colortrack")
            ct = cd["colortrack"]
            clip.colortrack = ColorTrack(
                [Vec3f(Float32.(v)...) for v in ct["gains"]],
                [Vec3f(Float32.(v)...) for v in ct["offsets"]], Int(ct["src_in"]))
        end
        clip
    end
    return Sequence(collect(Clip, clips), Float64(dict["framerate"]))
end
