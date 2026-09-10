# Pre-rendering a clip.
#
# Baking used to belong to the 3D scene, because the scene was the only thing slow
# enough to need it: raytracing a frame costs ~1.2 s at 480x854 and scrubbing a
# timeline at that rate is not editing. But a heavy effect stack on ordinary
# footage is slow for the same reason and wants the same thing, and the mechanism
# — on or off, one kept version, invalidation that disables rather than deletes —
# has nothing scene-shaped about it.
#
# So it is a property of a clip. `get_frame!` reads the bake when it is on and
# covers the frame, and runs the graph otherwise. What a source contributes is
# render settings: a video contributes none, a scene contributes which backend and
# at what quality, one set for the live preview and one for the bake.

# `Bake` is declared in clips.jl, where `Clip` has a field of one.

"""
    bakedirty!(clip) -> clip

Say that what `clip` renders has changed, so its bake no longer describes it.

Called from [`dirtygraph!`](@ref) — the same edits that change the structure change
the picture — and from the value edits that do not (a slider, a keyframe).

Switches the bake off and deletes nothing. Off, because a bake that no longer
describes the clip is a wrong picture. Kept, because the frames may be exactly
what was wanted and re-rendering costs minutes — `dirty` stays set, the panel says
so, and switching it back on is one click.
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

Every file this clip's render reads, with its mtime and size: the source video,
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
    # In the plane's format whatever the file turned out to be, because the source
    # pass copies this straight into a device buffer and an element type that only
    # usually matches is one that fails on somebody else's PNG.
    return PlanePixel.(PNGFiles.load(f))
end

"""
    bakeclip!(clip, engine; frames, canvas, dir, progress) -> Bake

Render every frame of `frames` through the clip's own chain and write it out.

Always a new directory, and the old one goes only once this has finished, so a
bake interrupted half way cannot leave the clip with a version that is neither
the old one nor a complete new one. Exactly one version is kept: a history of
bakes is disk nobody asked for, and what one wants back is the edit, which undo
has.
"""
function bakeclip!(clip::Clip, engine::FxEngine;
                   frames::UnitRange{Int} = clip.src_in:(clip.src_out - 1),
                   canvas::Union{Nothing, Tuple{Integer, Integer}} = nothing,
                   dir::AbstractString = mktempdir(; cleanup = false),
                   sourcefor = (c, sf) -> c.source,
                   progress = nothing)
    # A canvas resizes the source rather than giving the bake a size of its own,
    # so the bake and the preview cannot disagree about the one size a clip renders
    # at. For a decoder there is nothing to change.
    canvas === nothing || resize!(clip.source, canvas)
    can = (clip.source.width, clip.source.height)
    # It is the clip's picture, not its placement: the crop, the fit, the reframe
    # and the rotation happen where the layer meets the sequence canvas, and baking
    # those in would have the compositor apply them twice.
    # Stage, then swap into place, rather than writing to a fresh name and keeping
    # it: the bake's home is derived from the project path and the clip id on every
    # load (`adoptbakes!`), so a bake living under another name is one the next open
    # cannot find, and it falls back to rendering.
    staging = dir * ".part"
    ispath(staging) && rm(staging; force = true, recursive = true)
    mkpath(staging)
    fresh = Bake(staging, frames, can)
    n = 0
    for sf in frames
        src = decodable(clip.source) ? sourcefor(clip, sf) : clip.source
        src === nothing && error("bake: frame $sf of the source is not available")
        # The plane, coverage included: a baked scene that came back opaque would
        # hide the clip underneath it.
        renderplane(engine, src, clip, sf; exact = true) do img
            PNGFiles.save(bakeframefile(fresh, sf), collect(img))
        end
        n += 1
        progress === nothing || progress(n, length(frames))
    end
    fresh.inputs = externalinputs(clip)
    # …and only now is the old one replaceable: a bake interrupted half way has
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
what it read. The frames are not in the file but next to it, one PNG each,
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

Moved, not copied: there is one version of a bake, and a second copy left in
`/tmp` is one nobody deletes.
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

Called on load. A bake whose inputs moved is switched off and kept: the frames are
still there and still openable, and the user decides whether to use them.
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
# What a bake needs settled before it runs: how big, which frames, and from the
# source anything about how it renders. A video source contributes nothing; a
# scene contributes which renderer draws the bake, which is the whole point of
# baking a scene (rasterise while you edit, raytrace what you ship).
#
# `bakesettings!` is the hook. It draws into a grid and returns a function that
# applies what it collected, so a new kind of source adds its own settings by
# adding one method and nothing here changes.

"""
    screenoptions(backend) -> Vector{NamedTuple{(:name, :default, :kind)}}

The settings of `backend`'s screen that a form can offer, with the values Makie
would use if nobody said otherwise and the kind of widget each wants:
`:flag` (checkbox), `:number`, `:symbol`, or `:expr` — a Julia expression the
screen build evaluates in the backend's module, for the settings a text box
cannot carry as a plain value (`integrator = VolPath(samples=256)`,
`device = CPU()`).

`ScreenConfig`'s fields ARE what `activate!` and `Screen(...)` accept, and Makie
keeps their defaults in the default theme under the backend's name — so this needs
no per-backend list and a renderer loaded later brings its own settings with it.

Left out: the three `backendscreen` pins (`visible`, `px_per_unit`, `scalefactor`)
and the window-manager fields an offscreen bake has no use for.
"""
const SCREENOPTS_SKIP = (:visible, :px_per_unit, :scalefactor, :renderloop,
                         :render_pipeline, :monitor, :title, :fullscreen,
                         :decorated, :focus_on_show, :float, :pause_renderloop)

function screenoptions(backend::Module)
    fields = fieldnames(backend.ScreenConfig)
    theme = Makie.current_default_theme()
    key = nameof(backend)
    opt = NamedTuple{(:name, :default, :kind), Tuple{Symbol, Any, Symbol}}
    haskey(theme, key) || return opt[]
    out = opt[]
    for (k, v) in pairs(theme[key])
        (k in fields && !(k in SCREENOPTS_SKIP)) || continue
        val = v[]
        types = (T = fieldtype(backend.ScreenConfig, k)) isa Union ?
                Base.uniontypes(T) : (T,)
        # From the DEFAULT's value where it has one, from the field's type where
        # it does not (`nothing` could be a symbol or a number; an `Automatic()`
        # stands for an object only the backend can name).
        kind = val isa Bool ? :flag :
               val isa Real ? :number :
               val isa Symbol ? :symbol :
               val === nothing && any(t -> t === Symbol, types) ? :symbol :
               val === nothing && any(t -> t <: Real, types) ? :number :
               :expr
        push!(out, (name = k, default = val, kind = kind))
    end
    return sort!(out; by = o -> o.name)
end

"""
    screenoptsform!(gl, backend, store, uicolors; base, onchange) -> Vector

One row per setting of `backend`: a checkbox for a flag, a number box, a symbol
box, or an expression box, by the setting's kind (see [`screenoptions`](@ref)).
Every widget writes into `store` — there is no apply step and nothing to read
back out. A row shows `base`'s value where `store` has none of its own, which is
how the bake dialog shows what the bake inherits from the preview; clearing a
field removes the key, handing the default (or `base`) back.

`onchange` runs after every edit — the preview dialog uses it to redraw, the
bake dialog leaves it, since a bake reads the settings when it runs.

Returns the blocks it made so the caller can take them off again when the backend
changes, which is the one thing that changes WHICH settings exist. `widgets` is
filled with the same blocks keyed by the setting they edit, for the callers that
need a NAMED one — a test, and the walkthrough that types a sample count into it.
"""
function screenoptsform!(gl, backend::Module, store::Dict{Symbol, Any}, uicolors;
                         base::Dict{Symbol, Any} = store, onchange = nothing,
                         widgets::Dict{Symbol, Any} = Dict{Symbol, Any}())
    made = Any[]
    empty!(widgets)
    changed() = (onchange === nothing || onchange(); nothing)
    for (r, opt) in enumerate(screenoptions(backend))
        k, dflt, kind = opt.name, opt.default, opt.kind
        # `tellwidth = true`: the label column has to be as wide as the longest
        # name, or it collapses to nothing and every caption is drawn underneath
        # the box it belongs to.
        lbl = Label(gl[r, 1], String(k); halign = :left, fontsize = 11,
                    color = uicolors.text_muted, tellwidth = true)
        cur = get(store, k, get(base, k, dflt))
        w = if kind === :flag
            cb = Checkbox(gl[r, 2]; checked = cur === true)
            on(cb.checked) do v
                store[k] = v
                changed()
            end
            cb
        elseif kind === :number
            # Empty is a VALUE here — it means "leave this to the default", which is
            # what clearing the field does below. `validator = Float64` called that
            # invalid, so a setting with no default of its own (RayMakie's
            # `samples`) opened blank and turned red the moment it was clicked into,
            # before anything had been typed. The placeholder says what blank means
            # instead of leaving an empty box to be read as a broken one.
            box = Textbox(gl[r, 2]; stored_string = cur === nothing ? "" : string(cur),
                          width = 90,
                          placeholder = dflt === nothing ? "unset" : string(dflt),
                          validator = s -> isempty(strip(s)) ||
                                           tryparse(Float64, s) !== nothing)
            on(box.stored_string) do str
                s = strip(something(str, ""))
                if isempty(s)
                    delete!(store, k)      # back to the default / `base`
                else
                    v = tryparse(Float64, s)
                    v === nothing && return nothing
                    store[k] = dflt isa Integer ? round(Int, v) : v
                end
                changed()
                return nothing
            end
            box
        elseif kind === :symbol
            box = Textbox(gl[r, 2]; stored_string = cur === nothing ? "" : string(cur),
                          width = 90)
            on(box.stored_string) do str
                s = strip(something(str, ""))
                if isempty(s)
                    delete!(store, k)
                elseif s == "nothing"
                    store[k] = nothing     # a symbol field's "off", e.g. no tonemap
                else
                    store[k] = Symbol(s)
                end
                changed()
                return nothing
            end
            box
        else # :expr — evaluated in the backend's module when the screen is built
            okexpr(s) = (e = Meta.parse(s); !(e isa Expr && e.head in (:error, :incomplete)))
            box = Textbox(gl[r, 2];
                          stored_string = cur isa AbstractString ? String(cur) : "",
                          width = 160, placeholder = repr(dflt),
                          validator = s -> isempty(strip(s)) || okexpr(s))
            on(box.stored_string) do str
                s = strip(something(str, ""))
                isempty(s) ? delete!(store, k) : (store[k] = s)
                changed()
                return nothing
            end
            box
        end
        append!(made, (lbl, w))
        widgets[k] = w
    end
    return made
end

"""
    backendrow!(gl, r, caption, options, current, uicolors) -> Menu

A "render this with" menu. `options` are `(label, value)` pairs.
"""
function backendrow!(gl, r::Integer, caption::AbstractString, options, current, uicolors)
    # In its own single-column cell. A row's height is determined only from
    # content that spans ONE column (`determinedirsize`), so laying the label and
    # the menu out as two columns of the OUTER grid — next to a settings block
    # spanning both — left that block contributing nothing: it measured 1 px tall
    # and drew over the rest of the dialog.
    row = GridLayout(gl[r, 1])
    Label(row[1, 1], caption; halign = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = true)
    i = findfirst(o -> o[2] === current, options)
    # `tellwidth = true`: the menu has a fixed width, and a column told nothing
    # collapses — the menu was then drawn from x = 0 of that column, on top of the
    # caption beside it.
    return Menu(row[1, 2]; options = options, default = options[something(i, 1)][1],
                width = 180, tellwidth = true)
end

"""
    bakesettings!(source, gridpos, uicolors; menu, widgets) -> apply

Draw the settings this source contributes to a bake, and return `apply()`.

Nothing for a file: what a decoder produces is not a choice.

`menu` and `widgets` are `Ref`/`Dict` the caller owns, filled with the renderer
menu and each setting's widget by name — a test and the walkthrough type a sample
count into one of them, and nothing else can reach a block this deep.
"""
bakesettings!(::ClipSource, gridpos, uicolors; menu = Ref{Any}(nothing),
              widgets::Dict{Symbol, Any} = Dict{Symbol, Any}()) = () -> nothing

function bakesettings!(src::SceneSource, gridpos, uicolors;
                       menu = Ref{Any}(nothing),
                       widgets::Dict{Symbol, Any} = Dict{Symbol, Any}())
    gl = GridLayout(gridpos)
    names = sort!(collect(keys(BACKENDS)))
    # A path tracer is minutes per frame and a rasteriser is milliseconds, which is
    # the whole reason a scene has two: this says which one the final picture uses.
    # The PREVIEW renderer is not here — it is not a bake setting, it is how the
    # clip draws all the time, and it lives on the scene's own card.
    bake = backendrow!(gl, 1, "Bake with",
                       vcat([("same as the preview", :auto)], [(String(n), n) for n in names]),
                       src.bakewith, uicolors)
    menu[] = bake
    # `measurable!`: a layout whose rows all left has no determinable height, and
    # one such row makes the whole block indeterminate — it measured 1 px and drew
    # over the rest of the dialog. Same rule as the effects panel's slots.
    sub = measurable!(GridLayout(gl[2, 1]))
    made = Any[]
    # The settings of the renderer the bake WILL use — "same as the preview"
    # resolves to the preview's renderer. Seeded from the preview's own settings
    # (`base`), because that is what the bake inherits: `renderopts` merges these
    # over them. Editing a row pins that one setting for the bake; clearing it
    # hands the preview's value back.
    function fill!(name)
        foreach(Makie.delete!, made); empty!(made)
        resolved = name === :auto ? src.backend : name
        append!(made, screenoptsform!(sub, getbackend(resolved), src.bakescreenopts,
                                      uicolors; base = src.screenopts, widgets))
        return nothing
    end
    fill!(src.bakewith)
    on(bake.selection) do v
        v === nothing && return nothing
        src.bakewith = v
        empty!(src.bakescreenopts)   # …they belonged to the renderer that left
        fill!(v)
        return nothing
    end
    # Nothing to apply: every widget writes the clip as it is used.
    return () -> nothing
end

"""
    previewpane!(player, parent, clip) -> nothing

The Preview tab of the rendering dialog: which renderer draws this clip while
you work, and that renderer's own settings.

A non-scene clip has nothing to choose — what you see is the effect graph's
output — and the tab says so instead of offering a menu that does nothing.
"""
function previewpane!(player::Player, parent, clip::Clip)
    src = clip.source
    uicolors = player.fxwidgets[:uicolors]
    body = GridLayout(parent)
    if !(src isa SceneSource)
        Label(body[1, 1], wraptext("This clip does not render a scene — the preview is " *
                                   "the effect graph's output, and there is nothing to " *
                                   "choose here.", 64);
              halign = :left, justification = :left, fontsize = 11,
              color = uicolors.text_muted, tellwidth = true)
        return nothing
    end
    # `tellwidth = true`, unlike a caption sitting beside a control: this text is
    # the widest thing in the pane, and the dialog takes its width from what the
    # pane reports (see [`fittabs!`](@ref)). Told nothing, the dialog sized itself
    # to the settings form and the sentence ran off its right edge. Pre-wrapped, so
    # the width it reports is a real one.
    Label(body[1, 1], wraptext("Which renderer draws this clip in the preview, and what " *
                               "that renderer is set to. The bake can use another one — " *
                               "see the Bake tab.", 64);
          halign = :left, justification = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = true)
    gl = GridLayout(body[2, 1])
    names = sort!(collect(keys(BACKENDS)))
    prev = backendrow!(gl, 1, "Preview with", [(String(n), n) for n in names],
                       src.backend, uicolors)
    sub = measurable!(GridLayout(gl[2, 1]))
    made = Any[]
    # Published like every other panel widget: the renderer menu and each setting
    # by name, so a test and the walkthrough can reach them. `previewopts` is
    # refilled in place on every rebuild, so a holder keeps seeing the live one.
    opts = get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :previewopts)
    player.fxwidgets[:previewmenu] = prev
    # `onchange`: a screen takes its settings at construction, so an edit only
    # shows once the frame is drawn again — ask for it. (`renderopts` hands
    # `livescene!` a copy, so the edit is seen as a change and the screen is
    # rebuilt with it.)
    fill!(name) = (foreach(Makie.delete!, made); empty!(made);
                   append!(made, screenoptsform!(sub, getbackend(name),
                                                 src.screenopts, uicolors;
                                                 onchange = () -> showplayhead!(player),
                                                 widgets = opts));
                   nothing)
    fill!(src.backend)
    # No bound to set for the preview, and that is deliberate — see [`refining`](@ref).
    # `samples` in the form above is a different number and worth saying so: it is
    # what ONE finished frame costs, which the preview never pays because it asks
    # for one sample at a time.
    Label(gl[3, 1], wraptext("A progressive renderer keeps adding samples to the " *
                             "frame for as long as the playhead holds still, so there " *
                             "is no limit to set here. A renderer's own `samples` is " *
                             "what ONE finished frame costs: the export renders at it, " *
                             "and the Bake tab can override it.", 64);
          halign = :left, justification = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = true)
    on(prev.selection) do v
        v === nothing && return nothing
        src.backend = v
        empty!(src.screenopts)
        fill!(v)
        # The standing scene belongs to the renderer that made it: `livescene!`
        # rebuilds when the backend changes, and the preview has to be asked for
        # again to see it.
        showplayhead!(player)
        return nothing
    end
    return nothing
end

"""
    bakepane!(player, parent, clip, modal) -> nothing

The Bake tab of the rendering dialog: canvas, frame range, whatever the source
contributes (for a scene: which renderer the bake uses and its settings), and
the button that runs it.

Baking is offered for every clip, because baking is one operation. A complex
effect stack on ordinary footage is worth pre-rendering for the same reason a
raytraced scene is, and the only thing that differs between them is what the
source has to say.
"""
function bakepane!(player::Player, parent, clip::Clip, modal)
    uicolors = player.fxwidgets[:uicolors]
    body = GridLayout(parent)
    # Published like every other panel's widgets: a dialog a test cannot reach is
    # a dialog nothing checks.
    player.fxwidgets[:bakemodal] = (; modal, body)
    player.fxwidgets[:bakego] = nothing        # …replaced once the button exists
    b = clip.bake
    have = b === nothing ? "" :
           "\nIt has one already: frames $(first(b.frames))–$(last(b.frames)) at " *
           "$(b.canvas[1])×$(b.canvas[2]), $(b.enabled ? "in use" : "switched off")" *
           (bakestale(clip) ? " · out of date — the clip changed since" : "")
    # Pre-wrapped, not `word_wrap`: a wrapping Label derives its height from a
    # width the layout only knows after it has sized the row — see `wraptext`.
    # `word_wrap_width` is not a Label attribute at all, so this threw and the
    # bake dialog could not be opened.
    Label(body[1, 1], wraptext("Pre-render this clip's effect graph to disk. While the bake " *
                               "is on, the clip shows those frames instead of running the " *
                               "graph.", 64) * have;
          halign = :left, justification = :left, fontsize = 11,
          color = uicolors.text_muted, tellwidth = true)

    form = GridLayout(body[2, 1])
    Label(form[1, 1], "Frames"; halign = :left, fontsize = 11, tellwidth = false)
    fromb = Textbox(form[1, 2]; stored_string = string(clip.src_in), width = 80, validator = Int)
    tob = Textbox(form[1, 3]; stored_string = string(clip.src_out - 1), width = 80, validator = Int)
    Label(form[2, 1], "Canvas"; halign = :left, fontsize = 11, tellwidth = false)
    W0, H0 = clip.source.width, clip.source.height
    wbox = Textbox(form[2, 2]; stored_string = string(W0), width = 80, validator = Int)
    hbox = Textbox(form[2, 3]; stored_string = string(H0), width = 80, validator = Int)
    # In `body`, not in `form`: `form` is three columns wide and this note spans
    # one of them, so telling its width there would have made the label column as
    # wide as the sentence and pushed the boxes off to the right. A row of its own
    # can report the width the dialog needs.
    Label(body[3, 1], wraptext(decodable(clip.source) ?
                               "fixed at $(W0)×$(H0) — a decoder delivers the frames it has" :
                               "this clip renders, so this is the size it renders at — the " *
                               "preview follows it; the crop and the placement stay live", 58);
          halign = :left, justification = :left, fontsize = 10,
          color = uicolors.text_muted, tellwidth = true)
    bakemenu = Ref{Any}(nothing)
    bakeopts = get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :bakeopts)
    apply = bakesettings!(clip.source, body[4, 1], uicolors;
                          menu = bakemenu, widgets = bakeopts)
    player.fxwidgets[:bakemenu] = bakemenu

    status = Label(body[5, 1], ""; halign = :left, fontsize = 11,
                   color = uicolors.text_muted, tellwidth = false)
    row = GridLayout(body[6, 1])
    go = Button(row[1, 1]; label = "Bake", width = 120)
    cancel = Button(row[1, 2]; label = "Close", width = 100)
    player.fxwidgets[:bakego] = go
    player.fxwidgets[:bakestatus] = status
    # The OBSERVABLE, not the block: a bake outlives the dialog it was started
    # from, and reopening the dialog deletes this one (see `openrendermodal!`).
    # A deleted block has no attributes left, so `status.text` would then throw
    # from inside the render's progress callback — writing the observable it used
    # to be drawn from is simply unread.
    statustext = status.text
    on(_ -> close!(modal), cancel.clicks)

    num(box, dflt) = (v = tryparse(Int, something(box.stored_string[], "")); v === nothing ? dflt : v)
    on(go.clicks) do _
        apply()
        lo = clamp(num(fromb, clip.src_in), clip.src_in, clip.src_out - 1)
        hi = clamp(num(tob, clip.src_out - 1), lo, clip.src_out - 1)
        canvas = (max(num(wbox, W0), 2), max(num(hbox, H0), 2))
        dir = bakedirfor(player, clip)
        nframes = hi - lo + 1
        statustext[] = "baking $nframes frame(s)…"
        # …and in the editor's own status line and footer spinner as well, because
        # a bake outlives this dialog: close it mid-render and the label goes with
        # it, while the render keeps going with nothing on screen saying so.
        # `jobprogress` is what every other long job here writes — the footer
        # animates the spinner and the bar off it, and NaN means idle.
        setstatus!(player, "baking $nframes frame(s)…")
        player.jobprogress[] = 0.0
        # On the engine's thread, like every other render: the plan's context has
        # one owning thread and a bake is the same graph the preview runs.
        runanalysis(player) do
            src = clip.source
            src isa SceneSource && (src.mode = :bake)
            # Throttled: the frame counter is 180 `put!`s on a 32-slot queue on a
            # 180-frame bake, and a blocked queue blocks the RENDER. The spinner
            # polls `jobprogress` at its own rate and carries the animation.
            lastui = Ref(0.0)
            ok = try
                bakeclip!(clip, player.engine; frames = lo:hi, canvas, dir,
                          sourcefor = (c, sf) -> bakesource(player, c, sf),
                          progress = function (i, n)
                              player.jobprogress[] = i / max(n, 1)
                              time() - lastui[] < 0.1 && return nothing
                              lastui[] = time()
                              put!(player.uiqueue, () -> (statustext[] = "baking $i / $n…"))
                              return nothing
                          end)
                true
            catch e
                # A bake that dies used to leave the dialog reading "baking N
                # frame(s)…" for good: the worker logged it and nothing on screen
                # ever changed. It is the same "nothing happened" as a button that
                # does not fire.
                @error "bake failed" exception = (e, catch_backtrace())
                msg = "bake failed: $(sprint(showerror, e))"
                setstatus!(player, msg)
                put!(player.uiqueue, () -> (statustext[] = msg))
                false
            finally
                src isa SceneSource && (src.mode = :live)
                player.jobprogress[] = NaN
            end
            ok && put!(player.uiqueue, () -> begin
                statustext[] = "done — the clip is showing its bake"
                setstatus!(player, "bake done — the clip is showing its pre-rendered frames")
                showplayhead!(player)
            end)
        end
    end
    return nothing
end

"""
    fittabs!(tabs) -> nothing

Size a `Tabs` block to the tab that is showing, so the dialog around it can size
itself to what is actually in it.

The tab that is SHOWING rather than the largest of them: the two forms differ by
a third of the dialog's height, and sizing to the larger leaves the other one
sitting in a field of empty dialog. Switching tabs then resizes the dialog, which
is what a dialog that fits its content does.

Called once when the dialog is built AND on every later change, because the two
are not the same event: the panes are filled before this is wired up, so their
`contentsize` already holds its final value and firing on change alone never
fires at all — the dialog then opened at `min_size` with the settings form
scrolled out of sight. `refresh_contentsize!` recomputes rather than reads,
because a layout that has not been through a `computedbbox` yet has published
nothing.
"""
function fittabs!(tabs)
    w, h = Makie.refresh_contentsize!(tabs[tabs.active[]])
    (w > 0 && h > 0) || return nothing
    # Rounded UP: a content width the block matches to the pixel leaves the tab's
    # own scroll view a fraction short, and it answers with a scrollbar along the
    # bottom of a dialog that fits.
    tabs.width = ceil(w) + 1
    tabs.height = ceil(h) + 1 + tabs.headerheight[]
    return nothing
end

"""
    openrendermodal!(player) -> nothing

How the clip at the playhead is rendered: one dialog, two tabs. **Preview** —
which renderer draws the clip while you work, and that renderer's settings.
**Bake** — which renderer and settings the pre-rendered frames are made with,
and the button that makes them.

One dialog because the two are the same question at two timescales, and the
settings form is the same one: the bake tab's is the preview tab's, seeded with
the preview's values and overriding them.
"""
function openrendermodal!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "no clip at the playhead")
    clip = loc[1]
    # The dialog is built per open, because its two forms describe THIS clip and
    # this clip's renderer. So the previous one has to go: `close!` only hides a
    # Modal, and a hidden dialog is still a dialog — its blocks stay in the
    # figure with their event handlers live. Measured before this line existed:
    # a second open put a working dialog on screen whose tab header could not be
    # clicked, because the first dialog's `Tabs` — invisible, at the same place —
    # took the click and switched its own tab.
    prev = get(player.fxwidgets, :rendermodal, nothing)
    prev === nothing || Makie.delete!(prev.modal)
    modal = Modal(player.fig; title = "Rendering", min_size = (440, 240))
    # `tellwidth`/`tellheight`: a `Tabs` defaults to neither, because it is built to
    # fill the space it is given and scroll what does not fit. Here it is the other
    # way round — the dialog has no size of its own and takes the tabs' — and a
    # block that tells its parent nothing leaves the modal at `min_size` with the
    # settings form scrolled out of sight. See [`fittabs!`](@ref) for the numbers.
    tabs = Tabs(modal[1, 1], ["Preview", "Bake"]; closable = false,
                tellwidth = true, tellheight = true)
    # Published like every other panel's widgets: a dialog a test cannot reach is
    # a dialog nothing checks.
    player.fxwidgets[:rendermodal] = (; modal, tabs)
    previewpane!(player, tabs[1][1, 1], clip)
    bakepane!(player, tabs[2][1, 1], clip, modal)
    # Tabs is a fixed-space container by design — it scrolls what overflows —
    # while the Modal sizes itself from its content. So nobody was saying how
    # big the content IS, and the dialog clipped it to min_size. Bridge the two:
    # the tabs block gets the bigger tab's size, and the modal's own
    # content-sizing takes it from there. Live: switching the renderer rebuilds
    # the settings form, and the dialog grows with it.
    fittabs!(tabs)
    onany((_...) -> fittabs!(tabs),
          tabs.active, tabs[1].contentsize, tabs[2].contentsize, tabs.headerheight)
    open!(modal)
    return nothing
end
"""
Where this clip's bake goes: beside the project when there is one, a temp
directory otherwise — an unsaved edit still gets to bake, it just does not survive
the session.

A fresh directory every time (`.new`, swapped in by `bakeclip!` once it finishes),
so a bake interrupted half way cannot leave the clip with a version that is
neither the old one nor a complete new one.
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
