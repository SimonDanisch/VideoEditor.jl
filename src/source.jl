"""
What a [`Clip`](@ref) shows frames of.

A file on disk is one kind ([`VideoSource`](@ref)); a Makie scene rendered on
demand is another ([`SceneSource`](@ref)). Everything a clip does — trimming,
keyframes, the effect stack, the placement, compositing — is the same either way,
because a clip only ever asks its source for a frame of a given size at a given
index.

The four numbers below are the interface, and every source has them as fields
rather than behind accessors: `width`, `height`, `framerate`, `nframes`. What
differs is answered by dispatch — [`decodable`](@ref) says whether frames come
from a decoder at all, and the render graph's source pass is chosen by
[`sourcenode`](@ref).
"""
abstract type ClipSource end

"""
    decodable(source) -> Bool

Whether frames of this source come out of a video decoder. False for a source
that renders its frames, which therefore needs no decode pool, no proxy, no
thumbnail scan and no reader.
"""
decodable(::ClipSource) = true

"Where this source's media lives, or `\"\"` for one that has no file."
sourcepath(s::ClipSource) = s.path

"""
A clip whose frames are rendered rather than decoded.

Holds the format — how big the frames are, how fast, and how many — and the live
render state. It does not hold the scene: that is a
[`SceneSpec`](@ref) on the clip's `:scene` effect, where it is document data like
every other parameter, gets saved with the project, and contributes the animatable
numbers through [`dataparams`](@ref). A source that also held a spec would be a
second home for it.

`live` is the standing scene and its screen, kept across frames. That is not an
optimisation detail: rebuilding it per frame re-read ten STL files from disk for
every frame of a seek (0.28 s per frame, against 5.1 ms once held), and a
progressive renderer cannot work at all without it — RayMakie
accumulates samples into a screen, and a screen thrown away after each frame has
nothing to accumulate into.
"""
mutable struct SceneSource <: ClipSource
    # What it draws: a Makie spec — `S.Scene(; camera = cam3d!, plots = [...])` for
    # a raw scene, a layout spec when the picture wants an axis. The scene itself,
    # not a description of it beside one: what is animatable is read off the scene
    # that gets built (see `sceneattributes`), so there is nothing here that could
    # fall out of step with what is drawn.
    root::Any
    # The rig, when the scene has one: a joint per named plot. A joint is `angle`
    # and `offset` applied through the plot's `Transformation`, and applying them
    # needs the axis to turn about and the base translation to add to.
    #
    # Not on the plot, though that is where it belongs: Makie validates plot
    # attributes against a closed set, so a `jointaxis` on a `Mesh` is refused.
    # Owned by whoever built the rig, which is the next best place: it describes
    # the rig rather than the scene, and nothing else reads it.
    joints::Dict{Symbol, Any}
    # Where a 3-D scene looks from, once it exists. `cam3d!` places the eye through
    # `update_cam!` after the scene is realized, and a spec has nowhere to put that.
    camera::Any
    # How `root` was made, as data a project file can hold: `{"kind" => "text",
    # "args" => …}`. A realized spec holds live objects — a rig's
    # `Transformation`s, loaded meshes — so it cannot be written out; the recipe
    # can. A placeholder until scenes serialise themselves; see `buildscene`.
    build::Any
    backend::Symbol              # which renderer draws it live
    # The renderer's own settings, as the keywords its screen takes — `px_per_unit`,
    # `ssao`, RayMakie's `integrator`/`exposure`/`samples`. Passed straight to the
    # screen constructor (and to `colorbuffer`/`record`), not keyed by backend name
    # inside a theme: the settings belong to the call that uses them, and that call
    # already knows which backend it is making. A Makie THEME — fonts, colours — is
    # set where the scene is built, which is a different thing.
    screenopts::Dict{Symbol, Any}
    width::Int
    height::Int
    framerate::Float64
    nframes::Int
    live::Any                    # LiveScene, built on the first render
    # What the standing scene was last drawn for. A progressive renderer keeps
    # refining while this is unchanged and starts over when it is not — see
    # [`sceneframe!`](@ref).
    at::Int
    # Two backends, one scene. Raytracing a frame costs ~1.2 s at 480x854 and
    # rasterising it costs milliseconds: scrubbing at raytracing speed is not
    # editing, and shipping the rasterised preview is not the picture that was
    # edited for. The scene's own `backend` draws the preview; this is what a bake
    # uses.
    # `:auto` means "the same one" — a scene that is fast enough live has nothing
    # to switch to.
    bakewith::Symbol
    # …and its settings. A second set, not a second scene: the point of two
    # backends is that the preview is cheap and the final render is not, and "how
    # many samples" is exactly the number that differs between them. Empty means
    # the bake draws with the live settings.
    bakescreenopts::Dict{Symbol, Any}
    # Which of the two is being asked for right now. A field rather than an
    # argument because the render happens inside a graph pass, several calls below
    # whoever decided — `bakeclip!` sets it around its loop.
    mode::Symbol                 # :live | :bake
    # How many samples a progressive renderer has accumulated at `at`. Reset when
    # the position moves; grown by `refinescene!` while it holds.
    samples::Int
    # The frame, already drawn, and which frame it is. A scene draws with GLMakie,
    # whose screen belongs to thread 1; the composite runs on whichever thread owns
    # the Lava context, which is the pinned GPU worker. Drawing inside the pass
    # body therefore goes thread 1 → worker → thread 1 → worker for every frame,
    # and the hop back waits for the editor's own renderloop to reach a yield:
    # measured on the lego project at 22.5 ms of waiting against 7.3 ms of
    # drawing, per frame. `prerender!` fills these in while the caller is still on
    # thread 1; the pass body takes what is here. See [`takepending!`](@ref).
    pending::Any
    pendingat::Int
end
SceneSource(root; joints::Dict{Symbol, Any} = Dict{Symbol, Any}(), camera = nothing,
            build = nothing, backend::Symbol = :GLMakie,
            screenopts::Dict{Symbol, Any} = Dict{Symbol, Any}(),
            width::Integer = 1920, height::Integer = 1080,
            framerate::Real = 30.0, nframes::Integer = 90,
            bakewith::Symbol = :auto,
            bakescreenopts::Dict{Symbol, Any} = Dict{Symbol, Any}()) =
    SceneSource(root, joints, camera, build, backend, screenopts, Int(width), Int(height),
                Float64(framerate),
                Int(nframes), nothing, -1, bakewith, bakescreenopts, :live, 0, nothing, -1)

decodable(::SceneSource) = false
sourcepath(::SceneSource) = ""

function Base.resize!(s::SceneSource, wh::Tuple{Integer, Integer})
    w, h = max(Int(wh[1]), 2), max(Int(wh[2]), 2)
    (s.width, s.height) == (w, h) && return s
    s.width, s.height = w, h
    s.live = nothing        # the standing screen is that size; it has to go
    s.samples = 0
    return s
end

"""
    VideoSource(path)

Probed metadata for a video file: dimensions, framerate, duration, frame count,
the keyframe index and the timestamp of every frame (scanned from packet flags,
no decoding).

Frame indices are 0-based throughout. `n / framerate` is only the display time of
frame `n` on constant-rate material — phone clips drop frames (measured on a
"60 fps" clip: median period 0.0167 s, longest gap 0.2 s), and then the n-th
decoded frame and the frame at time `n/60` are different pictures. Analysis reads
sequentially while the preview seeks by time, so every per-frame track (flicker,
stabilization) landed on the wrong frame — 11 frames off at frame 100 on that
clip. `frametimes` makes the mapping exact for both.
"""
struct VideoSource <: ClipSource
    path::String
    width::Int
    height::Int
    framerate::Float64
    duration::Float64
    nframes::Int
    keyframe_times::Vector{Float64}
    frametimes::Vector{Float64}   # display time of every frame, in display order
end

function VideoSource(path::AbstractString)
    isfile(path) || error("no such file: $path")
    reader = VideoIO.openvideo(path)
    width, height = VideoIO.out_frame_size(reader)
    fps = Float64(VideoIO.framerate(reader))
    close(reader)
    duration = VideoIO.get_duration(path)
    keyframes, frametimes = scan_packets(path)
    npackets = length(frametimes)
    counted = VideoIO.get_number_frames(path)
    nframes = something(counted, npackets > 0 ? npackets : round(Int, duration * fps))
    # containers may claim a track rate the stream doesn't deliver (YouTube mkv
    # remuxes report 29.97 while frames actually arrive at 23.976) — when the
    # true frame count disagrees with duration × claimed rate, the effective rate
    # is what every frame↔time mapping (and the proxy check) has to use.
    # Snap it to the nearest standard video rate: count/duration carries the
    # container's rounding noise, and a raw float like 23.975288… later
    # explodes into a 2^47 denominator when converted to ffmpeg's Int32
    # AVRational at export time.
    if nframes > 0 && duration > 0 && abs(nframes / duration - fps) / fps > 0.01
        eff = nframes / duration
        std = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0,
               60000 / 1001, 60.0, 120000 / 1001, 120.0)
        near = findfirst(r -> abs(eff - r) / r < 0.002, std)
        fps = near === nothing ? eff : std[near]
    end
    return VideoSource(String(path), width, height, fps, duration, nframes, keyframes,
                       length(frametimes) == nframes ? frametimes : Float64[])
end

"""
    scan_packets(path) -> (keyframe_times, frame_times)

Timestamps of every frame in display order, plus the subset that are keyframes —
read from packet flags via ffprobe. Demux only, no decoding, fast even for long
files. The frame count implied here is authoritative where containers (mkv!)
carry no `nb_frames`, and the per-frame times are what makes index↔time exact on
variable-rate material.
"""
function scan_packets(path::AbstractString)
    cmd = `$(FFMPEG_jll.ffprobe()) -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $path`
    keys = Float64[]
    times = Float64[]
    for line in eachline(cmd)
        parts = split(line, ',')
        length(parts) >= 2 || continue
        t = tryparse(Float64, parts[1])
        t === nothing && continue
        push!(times, t)
        occursin('K', parts[2]) && push!(keys, t)
    end
    # packets arrive in decode order (B-frames); display order is by timestamp
    sort!(times)
    sort!(keys)
    return keys, times
end

"Display time of frame `n` — the scanned timestamp when we have one, else the
constant-rate assumption."
frametime(src::VideoSource, n::Integer) =
    1 <= n + 1 <= length(src.frametimes) ? src.frametimes[n + 1] : n / src.framerate

"Frame displayed at time `t` — the inverse of [`frametime`](@ref)."
function frameindex(src::VideoSource, t::Real)
    isempty(src.frametimes) && return clamp(round(Int, t * src.framerate), 0, src.nframes - 1)
    i = searchsortedlast(src.frametimes, Float64(t) + 1.0e-6)
    return clamp(i - 1, 0, src.nframes - 1)
end

"Time of the last keyframe at or before `t` (falls back to 0.0)."
function nearest_keyframe(src::VideoSource, t::Real)
    isempty(src.keyframe_times) && return 0.0
    i = searchsortedlast(src.keyframe_times, t)
    return i < 1 ? 0.0 : src.keyframe_times[i]
end
