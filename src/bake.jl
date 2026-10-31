# PRE-RENDERING A CLIP.
#
# Baking used to belong to the 3D scene, because the scene was the only thing slow
# enough to need it: raytracing a frame costs ~1.2 s at 480x854 and scrubbing a
# timeline at that rate is not editing. But a heavy effect stack on ordinary
# footage is slow for the same reason and wants the same thing, and the mechanism
# — on or off, one kept version, invalidation that DISABLES rather than deletes —
# has nothing scene-shaped about it.
#
# So it is a property of a clip. `get_frame!` reads the bake when it is on and
# covers the frame, and runs the graph otherwise. What a SOURCE contributes is
# render settings: a video contributes none, a scene contributes which backend and
# at what quality, one set for the live preview and one for the bake.

"""
A clip's pre-rendered frames.

`dir` is where they are; `frames` is the source-frame range they cover; `canvas`
is what they were rendered at. `enabled` is the switch, and it is what
INVALIDATION touches — a bake that no longer matches what the clip would render is
switched off and kept, not deleted, because the user may have wanted exactly that
picture and re-rendering it costs minutes.

`dirty` is set at the EDIT, not derived by comparison. There is no fingerprint
here: nothing walks the effect stack per frame to discover something the edit
already knew. The one thing that cannot be caught at an edit site is a file
changing under us — a source video replaced, a mesh re-exported — so the external
inputs are `stat`ed once on load, and that is the whole of what is checked.
"""
mutable struct Bake
    dir::String
    frames::UnitRange{Int}
    canvas::Tuple{Int, Int}
    enabled::Bool
    dirty::Bool
    # mtime+size of every file the render read, as of the bake. Only files: an
    # edit inside the editor sets `dirty` where it happens.
    inputs::Dict{String, Tuple{Float64, Int}}
end
Bake(dir::AbstractString, frames::UnitRange{Int}, canvas::Tuple{Int, Int}) =
    Bake(String(dir), frames, canvas, true, false, Dict{String, Tuple{Float64, Int}}())

"""
    bakedirty!(clip) -> clip

Say that what `clip` renders has changed, so its bake no longer describes it.

Called from [`dirtygraph!`](@ref) — the same edits that change the structure change
the picture — and from the value edits that do not (a slider, a keyframe).

It switches the bake OFF and DELETES NOTHING. Off, because a bake that no longer
describes the clip is a wrong picture, and shipping one silently is the failure
this whole mechanism exists to avoid. Kept, because the frames may be exactly what
the user wanted and re-rendering them costs minutes — `dirty` stays set, the panel
says so, and switching it back on is one click.
"""
function bakedirty!(clip::Clip)
    b = clip.bake
    b === nothing && return clip
    b.dirty = true
    b.enabled = false
    return clip
end

"""
    externalinputs(clip) -> Dict{String, Tuple{Float64, Int}}

Every FILE this clip's render reads, with its mtime and size: the source video,
and any mesh a scene loads.

The only thing a bake cannot know from the edit history. Everything else that
decides the picture is edited in this process and says so at the edit.
"""
function externalinputs(clip::Clip)
    out = Dict{String, Tuple{Float64, Int}}()
    stamp!(p::AbstractString) = (f = String(p);
                                 isfile(f) && (out[f] = (mtime(f), filesize(f))))
    stamp!(::Any) = nothing
    stamp!(sourcepath(clip.source))
    # A scene's meshes are named in its BUILD RECIPE, not in the realized spec —
    # by the time the spec exists the strings have become geometry. Walking the
    # recipe for anything that names a file finds them without knowing which key
    # a given builder puts them under.
    walk(d::AbstractDict) = foreach(walk, values(d))
    walk(v::AbstractVector) = foreach(walk, v)
    walk(x::AbstractString) = (stamp!(x); stamp!(assetfile(x)))
    walk(::Any) = nothing
    clip.source isa SceneSource && clip.source.build !== nothing && walk(clip.source.build)
    return out
end

"A bare name resolved against Makie's assets, or `nothing` — see `partmesh`."
function assetfile(name::AbstractString)
    isfile(name) && return nothing
    p = Makie.assetpath(String(name))
    return isfile(p) ? p : nothing
end

"""
    bakestale(clip) -> Bool

Whether the clip's bake no longer describes what it would render: an edit said so,
or a file it read has changed on disk.
"""
function bakestale(clip::Clip)
    b = clip.bake
    b === nothing && return false
    b.dirty && return true
    now = externalinputs(clip)
    return any(get(now, k, nothing) != v for (k, v) in b.inputs) ||
           any(!haskey(b.inputs, k) for k in keys(now))
end

bakeframefile(b::Bake, sf::Integer) = joinpath(b.dir, string(lpad(sf, 6, '0'), ".png"))

"""
    bakedframe(clip, sf) -> Union{Nothing, Matrix{RGBA{N0f8}}}

The baked picture for source frame `sf`, or `nothing` when the bake is off, does
not cover it, or its file is gone.
"""
function bakedframe(clip::Clip, sf::Integer)
    b = clip.bake
    (b === nothing || !b.enabled) && return nothing
    Int(sf) in b.frames || return nothing
    f = bakeframefile(b, sf)
    isfile(f) || return nothing
    # In the PLANE's format whatever the file turned out to be, because the source
    # pass copies this straight into a device buffer and an element type that only
    # usually matches is one that fails on somebody else's PNG.
    return PlanePixel.(PNGFiles.load(f))
end

"""
    bakeclip!(clip, engine; frames, canvas, dir, progress) -> Bake

Render every frame of `frames` through the clip's own chain and write it out.

A NEW DIRECTORY, always, and the old one goes only once this has finished: a bake
interrupted half way must not be able to leave the clip with a version that is
neither the old one nor a whole new one. Exactly one version is kept — a history
of bakes is disk nobody asked for, and the thing you actually want back is the
edit, which undo has.
"""
function bakeclip!(clip::Clip, engine::FxEngine;
                   frames::UnitRange{Int} = clip.src_in:(clip.src_out - 1),
                   canvas::Union{Nothing, Tuple{Integer, Integer}} = nothing,
                   dir::AbstractString = mktempdir(; cleanup = false),
                   sourcefor = (c, sf) -> c.source,
                   progress = nothing)
    # A CANVAS RESIZES THE SOURCE, it does not give the bake a size of its own.
    # There is one size a clip renders at, so the bake and the preview cannot end
    # up disagreeing about it — and for a decoder there is nothing to change.
    canvas === nothing || resize!(clip.source, canvas)
    can = (clip.source.width, clip.source.height)
    # It is the CLIP's picture, not its placement: the crop, the fit, the reframe
    # and the rotation happen where the layer meets the sequence canvas, and baking
    # those in would have the compositor apply them twice.
    # STAGE, then swap into place. Not "write to a fresh name and keep it": the
    # bake's home is derived from the project path and the clip id on every load
    # (`adoptbakes!`), so a bake that ended up living under some other name is one
    # the next open cannot find — it would fall silently back to rendering, which
    # is the failure a bake exists to prevent, arriving as "it got slow again".
    staging = dir * ".part"
    ispath(staging) && rm(staging; force = true, recursive = true)
    mkpath(staging)
    fresh = Bake(staging, frames, can)
    n = 0
    for sf in frames
        src = decodable(clip.source) ? sourcefor(clip, sf) : clip.source
        src === nothing && error("bake: frame $sf of the source is not available")
        # THE PLANE, coverage included: a baked scene that came back opaque would
        # hide the clip underneath it.
        renderplane(engine, src, clip, sf; exact = true) do img
            PNGFiles.save(bakeframefile(fresh, sf), collect(img))
        end
        n += 1
        progress === nothing || progress(n, length(frames))
    end
    fresh.inputs = externalinputs(clip)
    # …and only NOW is the old one replaceable: a bake interrupted half way has
    # written nothing but its staging directory.
    old = clip.bake
    ispath(dir) && rm(dir; force = true, recursive = true)
    mv(staging, dir)
    fresh.dir = dir
    clip.bake = fresh
    old === nothing || old.dir == dir || !ispath(old.dir) ||
        rm(old.dir; force = true, recursive = true)
    return fresh
end

# ---------------------------------------------------------------- in a project

"Where a project keeps its bakes."
bakedir(path::AbstractString) = string(path, ".bakes")
bakeclipdir(path::AbstractString, id::Integer) = joinpath(bakedir(path), string(id))

"""
A bake as the project file holds it: where, what it covers, whether it is on, and
what it read. The FRAMES are not in the file — they are next to it, one PNG each,
under a directory named for the clip.
"""
bakedict(b::Bake) = Dict{String, Any}(
    "first" => first(b.frames), "last" => last(b.frames),
    "canvas" => [b.canvas[1], b.canvas[2]],
    "enabled" => b.enabled, "dirty" => b.dirty,
    "inputs" => Dict{String, Any}(k => [v[1], v[2]] for (k, v) in b.inputs))

function bakefromdict(d::AbstractDict, dir::AbstractString)
    b = Bake(dir, Int(d["first"]):Int(d["last"]),
             (Int(d["canvas"][1]), Int(d["canvas"][2])))
    b.enabled = Bool(get(d, "enabled", true))
    b.dirty = Bool(get(d, "dirty", false))
    for (k, v) in get(d, "inputs", Dict{String, Any}())
        b.inputs[String(k)] = (Float64(v[1]), Int(v[2]))
    end
    return b
end

"""
    savebakes(path, seq) -> seq

Move every bake that is not already beside this project into place.

An unsaved edit still gets to bake — it goes to a temp directory — and this is
where it becomes part of the project. Without it a bake made before the first
Ctrl+S was written to `/tmp`, recorded in the file as if it were beside it, and
was gone by the next open: `adoptbakes!` derives the directory from the project
path, so it would have looked somewhere that had never held anything.

MOVED, not copied: there is exactly one version of a bake, and leaving a second
copy in `/tmp` is a copy nobody will ever delete.
"""
function savebakes(path::AbstractString, seq::Sequence)
    for clip in seq.clips
        b = clip.bake
        b === nothing && continue
        home = bakeclipdir(path, clip.id)
        b.dir == home && continue
        ispath(b.dir) || (b.dir = home; continue)
        mkpath(dirname(home))
        ispath(home) && rm(home; force = true, recursive = true)
        mv(b.dir, home)
        b.dir = home
    end
    return seq
end

"""
    adoptbakes!(path, seq) -> seq

Point every clip's bake at this project's bake directory, and check the files it
read.

Called on load. A bake whose inputs moved is switched OFF and kept: the frames are
still there, still openable, and the user decides whether the picture they show is
the one they want.
"""
function adoptbakes!(path::AbstractString, seq::Sequence)
    for clip in seq.clips
        b = clip.bake
        b === nothing && continue
        b.dir = bakeclipdir(path, clip.id)
        bakestale(clip) && (b.enabled = false)
    end
    return seq
end

# ---------------------------------------------------------------- the modal
#
# What a bake needs settling before it runs: how big, which frames, and — from the
# SOURCE — anything about HOW it renders. A video source contributes nothing; a
# scene contributes which renderer draws the bake, which is the whole point of
# baking a scene (rasterise while you edit, raytrace what you ship).
#
# `bakesettings!` is the hook. It draws into a grid and returns a function that
# applies what it collected, so a new kind of source adds its own settings by
# adding one method and nothing here changes.

"""
    bakesettings!(source, gridpos, uicolors) -> apply

Draw the settings this SOURCE contributes to a bake, and return `apply()`.

Nothing for a file: what a decoder produces is not a choice.
"""
bakesettings!(::ClipSource, gridpos, uicolors) = () -> nothing

function bakesettings!(src::SceneSource, gridpos, uicolors)
    gl = GridLayout(gridpos)
    Label(gl[1, 1], "Render the bake with"; halign = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = false)
    names = sort!(collect(keys(BACKENDS)))
    opts = vcat([("same as the preview", :auto)], [(String(n), n) for n in names])
    i = findfirst(o -> o[2] === src.bakewith, opts)
    menu = Menu(gl[1, 2]; options = opts, default = opts[something(i, 1)][1],
                width = 180, tellwidth = false)
    # A path tracer is minutes per frame and a rasteriser is milliseconds, which is
    # the whole reason a scene has two: this says which one the FINAL picture uses.
    return () -> (src.bakewith = menu.selection[]; nothing)
end

"""
    openbakemodal!(player) -> nothing

The bake dialog for the clip at the playhead: canvas, frame range, whatever the
source contributes, and the button that runs it.

One modal for every clip, because baking is one operation. A complex effect stack
on ordinary footage is worth pre-rendering for the same reason a raytraced scene
is, and the only thing that differs between them is what the source has to say.
"""
function openbakemodal!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "bake: no clip at the playhead")
    clip = loc[1]
    uicolors = player.fxwidgets[:uicolors]
    modal = Modal(player.fig; title = "Bake this clip", min_size = (420, 220))
    body = GridLayout(modal[1, 1])
    b = clip.bake
    have = b === nothing ? "" :
           "\nIt has one already: frames $(first(b.frames))–$(last(b.frames)) at " *
           "$(b.canvas[1])×$(b.canvas[2]), $(b.enabled ? "in use" : "switched off")" *
           (bakestale(clip) ? " · OUT OF DATE — the clip changed since" : "")
    Label(body[1, 1], "Pre-render this clip's effect graph to disk. While the bake is on, " *
                      "the clip shows those frames instead of running the graph." * have;
          halign = :left, justification = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = false, word_wrap_width = 400)

    form = GridLayout(body[2, 1])
    Label(form[1, 1], "Frames"; halign = :left, fontsize = 11, tellwidth = false)
    fromb = Textbox(form[1, 2]; stored_string = string(clip.src_in), width = 80, validator = Int)
    tob = Textbox(form[1, 3]; stored_string = string(clip.src_out - 1), width = 80, validator = Int)
    Label(form[2, 1], "Canvas"; halign = :left, fontsize = 11, tellwidth = false)
    W0, H0 = clip.source.width, clip.source.height
    wbox = Textbox(form[2, 2]; stored_string = string(W0), width = 80, validator = Int)
    hbox = Textbox(form[2, 3]; stored_string = string(H0), width = 80, validator = Int)
    Label(form[3, 1], decodable(clip.source) ?
                      "fixed at $(W0)×$(H0) — a decoder delivers the frames it has" :
                      "this clip renders, so this is the size it renders at — the " *
                      "preview follows it; the crop and the placement stay live";
          halign = :left, fontsize = 10, color = uicolors.text_muted, tellwidth = false)
    apply = bakesettings!(clip.source, body[3, 1], uicolors)

    status = Label(body[4, 1], ""; halign = :left, fontsize = 11,
                   color = uicolors.text_muted, tellwidth = false)
    row = GridLayout(body[5, 1])
    go = Button(row[1, 1]; label = "Bake", width = 120)
    cancel = Button(row[1, 2]; label = "Close", width = 100)
    on(_ -> close!(modal), cancel.clicks)

    num(box, dflt) = (v = tryparse(Int, something(box.stored_string[], "")); v === nothing ? dflt : v)
    on(go.clicks) do _
        apply()
        lo = clamp(num(fromb, clip.src_in), clip.src_in, clip.src_out - 1)
        hi = clamp(num(tob, clip.src_out - 1), lo, clip.src_out - 1)
        canvas = (max(num(wbox, W0), 2), max(num(hbox, H0), 2))
        dir = bakedirfor(player, clip)
        status.text[] = "baking $(hi - lo + 1) frame(s)…"
        # ON THE ENGINE'S THREAD, like every other render: the plan's context has
        # one owning thread and a bake is the same graph the preview runs.
        runanalysis(player) do
            src = clip.source
            src isa SceneSource && (src.mode = :bake)
            try
                bakeclip!(clip, player.engine; frames = lo:hi, canvas, dir,
                          sourcefor = (c, sf) -> bakesource(player, c, sf),
                          progress = (i, n) -> put!(player.uiqueue,
                                                    () -> (status.text[] = "baking $i / $n…")))
            finally
                src isa SceneSource && (src.mode = :live)
            end
            put!(player.uiqueue, () -> begin
                status.text[] = "done — the clip is showing its bake"
                notify(player.playhead)
            end)
        end
    end
    open!(modal)
    return nothing
end

"""
Where this clip's bake goes: beside the project when there is one, a temp
directory otherwise — an unsaved edit still gets to bake, it just does not survive
the session.

A FRESH directory every time (`.new`, swapped in by `bakeclip!` once it finishes),
because a bake interrupted half way must not leave the clip with a version that is
neither the old one nor a whole new one.
"""
function bakedirfor(player::Player, clip::Clip)
    player.projectpath === nothing && return mktempdir(; cleanup = false)
    return bakeclipdir(player.projectpath, clip.id)
end

"What a bake reads a frame from: the clip's own source pool, exactly as the
preview does."
function bakesource(player::Player, clip::Clip, sf::Integer)
    sp = pool(player, clip)
    settarget!(sp.worker, Int(sf))
    buf = RGBFrame(undef, clip.source.width, clip.source.height)
    deadline = time() + 10.0
    while !fetchframe!(buf, sp.ring, Int(sf))
        time() > deadline && return nothing
        sleep(0.004)
    end
    return buf
end
