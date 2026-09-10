# The Effects panel: one list of cards for the selected clip.
#
# What this replaces: an Inspector that listed "effects" and a Tools dock that
# listed "tools", with two card builders, two sets of widget helpers and a line
# between them nobody could draw. Blur was an effect; Stabilize was a tool; both
# are things you put on a clip and see in the render.
#
# Nothing here rebuilds. A card belongs to its effect (`Effect.card`) and a row's
# widgets belong to their parameter (`Param.view`), so adding an effect builds one
# card, removing one deletes it, and switching clips hides one set and shows
# another. What it replaces: a signature over the whole stack, compared on every
# playhead move, driving a teardown-and-rebuild that a per-clip cache then had to
# make affordable.

"""
What a card body is built against: the player, the slot the card belongs to, and
the bookkeeping that lets everything the body added be removed again when the
card goes.

`state` is the kind's own scratch. The plots and handlers registered through
[`toolplot!`](@ref)/[`ontool!`](@ref) are dropped together by
[`cleartoolcontext!`](@ref).
"""
const EffectContext = ToolContext

"""
    fxfilter(player) -> Observable{String}

The Effects panel's filter query. On the panel because the palette and MCP can set
it too ("show me the color effects").
"""
fxfilter(player::Player) = player.fxpanel.query

"Does `kind` match the filter `q`? Its label, its description or any parameter label."
function kindmatches(kind::EffectKind, q::AbstractString)
    isempty(q) && return true
    occursin(q, lowercase(kind.label)) && return true
    any(p -> occursin(q, lowercase(p.label)), kind.params) && return true
    return occursin(q, lowercase(kind.description))
end

# A card with no `EffectKind` behind it. It has nothing to match on but its name,
# so an empty query keeps it and anything else rules it out. Without this the
# filter threw the moment such a card appeared: `kindmatches` is typed to
# `EffectKind`.
kindmatches(::Nothing, q::AbstractString) = isempty(q)

"""
Build the Effects panel into `gridpos`.

Layout, top to bottom: the title with the keyframe overview, "+ Add effect…", the
filter box, which clip is being edited, the bake row, then the card stack, then
the tool-only cards. Everything below the title scrolls.
"""
function buildfxpanel!(player::Player, gridpos, uicolors)
    fxscroll = Subfigure(gridpos; scroll_speed = 70, scrollbar_size = 9,
                         scrollbar_color = (uicolors.background, 0.55),
                         scrollbar_thumb_color = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                                                  RGBf(1, 1, 1), 0.34),
                         scrollbar_thumb_color_active = uicolors.accent)
    player.fxwidgets[:uicolors] = uicolors   # tool card bodies draw in the editor's palette
    wiretoolcards!(player)
    panel = GridLayout(fxscroll[1, 1]; valign = :top)
    head = GridLayout(panel[1, 1])
    Label(head[1, 1], "Effects"; font = :bold, halign = :left, tellwidth = false)
    # The eye bypasses the whole stack: the compare-with-the-original toggle,
    # the per-card eye one level up.
    headflat = (buttoncolor = (:transparent, 0.0), strokewidth = 0, cornerradius = 4,
                buttoncolor_hover = uicolors.accent_subtle,
                buttoncolor_active = uicolors.accent, width = 26, height = 22,
                halign = :right)
    eyeb = Button(head[1, 2]; label = map(a -> a ? "◉" : "○", player.applytracks),
                  headflat...)
    on(_ -> (player.applytracks[] = !player.applytracks[]; showplayhead!(player)),
       eyeb.clicks)
    on(player.applytracks; update = true) do on_
        eyeb.buttoncolor[] = on_ ? uicolors.surface : uicolors.accent
        eyeb.labelcolor[] = on_ ? uicolors.text : uicolors.text_on_accent
    end
    tips = get(player.fxwidgets, :tips, nothing)
    tips === nothing || (tips[eyeb] = "Bypass every effect on this clip")
    # ◆ opens the animated-parameter overview rather than toggling all lanes.
    kfb = Button(head[1, 3]; label = "◆", headflat...)
    tips === nothing || (tips[kfb] = "Animated parameters on this clip")
    on(_ -> openkeyframes!(player), kfb.clicks)

    # ---------------------------------------------------------------- add menu
    # Directly under the title, before anything describing the current stack.
    # Lists effects and the tool-only kinds: the stack shows only what the user
    # put there, so a tool that is not a clip effect is reachable only here.
    # Selecting one opens its card instead of adding an effect.
    menuopts() = vcat([(k.label, k.name) for k in addablekinds()],
                      [(k.label, k.name) for k in toolonlykinds()])
    # The menu is the only way to reach a tool-only kind, so its options are
    # exposed for the tests to assert on rather than read off the widget.
    player.fxwidgets[:fxmenuopts] = menuopts
    addmenu = Menu(panel[2, 1]; prompt = "+  Add effect…", default = nothing,
                   searchable = true, search_placeholder = "type to filter…",
                   options = menuopts(), tellwidth = false)
    # A kind registered at run time changes what can be ADDED, and nothing else:
    # the cards on screen describe effects that are already there. This is the one
    # thing that still listens to the registry — it used to drive a rebuild of the
    # whole stack. `EFFECTS` is module-level, so the listener outlives the player
    # unless taken off again; `close` does that (see `Player.globallisteners`).
    push!(player.globallisteners,
          on(_ -> (addmenu.options[] = menuopts()), EFFECTS.version))
    on(addmenu.selection) do sel
        sel === nothing && return
        addmenu.i_selected[] = 0     # back to the prompt; re-fires with nothing
        k = kindbyname(sel)
        if k !== nothing && k.make === nothing
            opentool!(player, sel)   # a tool: open its card, do not touch the clip
        else
            addeffect!(player, sel)
        end
    end

    # The primary way to put an effect on a clip, and until now the only panel
    # control without a handle — so a walkthrough could drive a card's parameters
    # but not the step that creates the card.
    player.fxwidgets[:addeffect] = addmenu

    # ------------------------------------------------------------------ filter
    query = Observable("")
    filterrow = GridLayout(panel[3, 1])
    filterbox = Textbox(filterrow[1, 1]; placeholder = "filter effects…", width = Makie.Relative(1.0),
                        tellwidth = false, reset_on_defocus = false)
    on(filterbox.stored_string) do s
        query[] = lowercase(s === nothing ? "" : s)
    end
    # A filtered panel must never read as an empty one — say what is hidden.
    countlabel = Label(filterrow[2, 1], ""; halign = :left, fontsize = 10,
                       color = uicolors.text_muted, tellwidth = false)
    player.fxwidgets[:fxfilterbox] = filterbox

    # …and which clip this all applies to labels the stack, right above it. Its
    # text names the clip's SPAN, so it follows three different facts: which clip
    # is selected, where the playhead is, and the clip's own in/out. The third one
    # is not an observable, so the write says it — see `retitle!`, called from
    # `Clip`'s `setproperty!`. Deriving from the first two alone left the header
    # reading the pre-split length after every cut.
    title = Observable(cliptitle(player))
    player.fxwidgets[:cliptitle] = title
    onany((_...) -> retitle!(player), player.playhead, player.timeline.selected)
    Label(panel[4, 1], title; halign = :left, fontsize = 11, color = uicolors.accent,
          tellwidth = false)

    # One bake row per clip, for scenes and ordinary footage alike: the state
    # ("no bake" / "in use" / "out of date"), the on/off switch for it, and the
    # rendering dialog — preview and bake settings are the same question at two
    # timescales, so they share one dialog with two tabs.
    bakerow = GridLayout(panel[5, 1])
    bakelabel = Label(bakerow[1, 1], ""; halign = :left, fontsize = 10,
                      color = uicolors.text_muted, tellwidth = false)
    bakeuse = Button(bakerow[1, 2]; label = "use", width = 42, height = 18, fontsize = 10)
    bakebtn = Button(bakerow[1, 3]; label = "Rendering", width = 74, height = 18, fontsize = 10)
    player.fxwidgets[:bakebutton] = bakebtn
    player.fxwidgets[:bakeuse] = bakeuse
    on(_ -> openrendermodal!(player), bakebtn.clicks)
    on(bakeuse.clicks) do _
        runcommand!(player, :bake_toggle)
    end
    player.fxwidgets[:bakelabel] = bakelabel
    player.fxwidgets[:bakerefresh] = () -> showbake!(player)
    on(_ -> showbake!(player), player.playhead)
    showbake!(player)

    # `measurable!`: a row with no content has no determinable height, and one such
    # row makes the whole panel indeterminable. That stretches it to fill the dock
    # (rows share the slack, so a short panel floats in the middle) and leaves
    # `Subfigure.contentsize` at zero, so a long list gets no scrollbar.
    #
    # Row 0 as well as row 1: that is where `showemptystate!` puts its labels, and
    # taking them away again left the row behind with nothing in it.
    stackgl = measurable!(GridLayout(panel[6, 1]; valign = :top, default_rowgap = 0), 0, 1)
    colsize!(stackgl, 1, Makie.Relative(1.0))
    # …and below the clip's own effects, the tools that are not clip effects. Its
    # own layout, because those cards belong to the panel rather than to any clip:
    # the crop scope and the transcript stay put when the playhead crosses a cut.
    toolgl = measurable!(GridLayout(panel[7, 1]; valign = :top, default_rowgap = 0))
    colsize!(toolgl, 1, Makie.Relative(1.0))

    player.fxpanel = FxPanel(fxscroll, stackgl, toolgl, query, countlabel, uicolors)
    on(_ -> applyfilter!(player), query)
    showclip!(player)

    # Esc clears the filter — the one key everyone tries first.
    on(events(player.fig).keyboardbutton; priority = 26) do ev
        (ev.key == Keyboard.escape && ev.action == Keyboard.press) || return Consume(false)
        isempty(query[]) && return Consume(false)
        player.dockopen[] === :effects || return Consume(false)
        query[] = ""
        filterbox.stored_string[] = ""
        return Consume(true)
    end

    merge!(player.fxwidgets, Dict{Symbol, Any}(
        :addeffect => addmenu, :compare => eyeb, :bypassall => eyeb,
        :fxapplyfilter => () -> applyfilter!(player),
        # "take me to this effect" — the Stabilize controls used to live in a
        # dock of their own, and this is what replaced knowing that
        :showkind => (name::Symbol) -> showkind!(player, name),
        :stabopen => () -> showkind!(player, :stabilize)))
    rowgap!(panel, 10)
    colsize!(panel, 1, Makie.Relative(1.0))
    return panel
end

# ---------------------------------------------------------------- card ownership

"""
    showclip!(player[, clip]) -> nothing

Show `clip`'s cards and hide whatever was up before.

The whole of "the selection changed": a clip's cards live as long as the clip, so
this is a visibility flip, plus a build for effects that have no card yet — one
just added, or one an undo brought back. Nothing is compared against the screen.

`clip` defaults to [`editclip`](@ref)'s, which is what the panel is FOR: the
selected clip, else the one under the playhead.
"""
function showclip!(player::Player, clip::Union{Nothing, Clip} = selectedclip(player))
    panel = player.fxpanel
    panel === nothing && return nothing
    old = player.shownclip
    old === clip && return nothing
    # A solo is a temporary view of the clip being edited, so leaving it ends the
    # solo — while `old` is still the shown one, or the restore would be applied to
    # the wrong clip's parameters and `old`'s lanes would stay hidden for good.
    unsolo!(player)
    old === nothing || setcardsvisible!(player, old, false)
    player.shownclip = clip
    keep = panel.scroll.scroll[]
    if clip !== nothing
        # the panel always says which stabilization this clip carries
        player.stabinfo[] = stabdescription(clip.motiontrack)
        buildcards!(player, clip)
        setcardsvisible!(player, clip, true)
    end
    showemptystate!(player, clip)
    applyfilter!(player)
    # Filling or emptying the layout moves `contentsize`, and the Subfigure clamps
    # the scroll to fit — so a card appearing threw the reader back to the top of
    # the list. Put it back; the Subfigure re-clamps if the list really did get
    # shorter. Twice, because `contentsize` is recomputed from the layout's bbox
    # and that can land after this call returns.
    panel.scroll.scroll[] = keep
    put!(player.uiqueue, () -> (panel.scroll.scroll[] = keep))
    return nothing
end

"""
    stackrow!(panel, clip) -> Int

The row of the card stack `clip`'s cards live in.

A clip keeps its row for as long as the panel does, so an undo that brings a
deleted clip back builds into the row it had. One row per clip that has ever been
selected, which is bounded by the clips in the project.

The row is anchored as it is minted: a clip that leaves the document takes its
sub-layout out of the stack ([`dropcards!`](@ref)) and would otherwise leave an
empty row behind — see [`measurable!`](@ref) for what that costs.
"""
stackrow!(panel::FxPanel, clip::Clip) =
    get!(panel.rows, clip.id) do
        r = (panel.lastrow += 1)
        measurable!(panel.stack, (1:r)...)
        return r
    end

"""
    buildcards!(player, clip) -> nothing

Give every effect of `clip` a card, for the ones that have none.

Usually nothing to do: cards are built once and kept. An effect added by
[`addeffect!`](@ref) gets its card there; this covers the ones that arrive another
way — a project loaded, an undo restoring an effect that was removed, a scene clip
whose parameters appear only once it has rendered.
"""
function buildcards!(player::Player, clip::Clip)
    panel = player.fxpanel
    if clip.cardlayout === nothing
        # `measurable!` from the start: a clip with no effects has an empty
        # sub-layout, and an empty row is what takes the panel's height away.
        clip.cardlayout = measurable!(GridLayout(panel.stack[stackrow!(panel, clip), 1];
                                                 valign = :top, default_rowgap = 0))
        colsize!(clip.cardlayout, 1, Makie.Relative(1.0))
    end
    any(fx -> fx.card === nothing, clip.effects) || return nothing
    # One layout pass for the whole stack instead of one per block. `cardlayout`
    # hangs in the stack, so blocking the parent stops the chain below it.
    # Measured on the lego scene: 3062 → 2633 ms, 1450 → 1225 MB.
    Makie.GridLayoutBase.with_updates_suspended(panel.stack) do
        for fx in clip.effects
            fx.card === nothing && buildcard!(player, clip, fx)
        end
    end
    placelanes!(player.timeline, clip)   # the lanes those rows just created
    return nothing
end

"""
    buildcard!(player, clip, fx) -> Card

Build `fx`'s card into `clip`'s row of the stack, and hang it on the effect.

Its row in that layout is the effect's position in the stack, so reordering the
stack is a row write rather than a rebuild.
"""
function buildcard!(player::Player, clip::Clip, fx::Effect)
    panel = player.fxpanel
    # An effect has ONE card. Overwriting `fx.card` would leave the old one
    # undeletable and still drawing — and still taking clicks, at the very
    # rectangle its replacement occupies.
    fx.card === nothing || dropcard!(fx)
    clip.cardlayout === nothing && buildcards!(player, clip)
    row = something(findfirst(s -> s === fx, clip.effects), length(clip.effects))
    # …and this row can be emptied again, by `dropcard!` — anchor it, so removing
    # the effect above another one does not cost the panel its scrollbar.
    measurable!(clip.cardlayout, (1:row)...)
    card, ctx = fxcard!(player, clip.cardlayout, row, clip, fx, panel.uicolors)
    fx.card = card
    # …and the tool's state hangs on the effect beside its card, so `dropcard!`
    # takes it down with the card rather than leaving it listening.
    fx.tool = ctx
    ctx === nothing ||
        (get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :toolpanels)[ctx.tool] = ctx)
    return card
end

"""
    derive(f, obs) -> (derived, registration)

An Observable computed from `obs`, together with the registration that feeds it.

`map(f, obs)` hands back only the Observable, so nothing can ever unhook it: the
widget it feeds is deleted and the listener stays on `obs` for the rest of the
session, recomputing a label for a button that is gone. Every derived widget
attribute over an observable that outlives the widget goes through this instead,
and the registration goes wherever that widget's lifetime is written down —
`blockscene.deregister_callbacks` for a card, [`ParamView`](@ref) for a lane.

`map` is right where the source dies with the widget (a local Observable, a
button's own state); it is only the ones reaching up to the player, the effect or
the parameter that leak.
"""
function derive(f, obs::Observable)
    out = Observable(f(obs[]))
    return out, on(o -> (out[] = f(o)), obs)
end

"""
    cliptitle(player) -> String

What the Effects panel says it is aimed at: which clip, from which file, and the
span it covers.
"""
function cliptitle(player::Player)
    c = selectedclip(player)
    c === nothing && return "▸ no clip selected"
    i = something(findfirst(x -> x === c, player.sequence.clips), 0)
    fps = player.sequence.framerate
    return "▸ clip $i · $(basename(sourcepath(c.source))) " *
           "($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
end

"""
    retitle!(player) -> nothing
    retitle!(clip) -> nothing

Say again what the panel is aimed at.

The clip form is what a write to a clip's in/out calls, and only when that clip is
the one on screen: a span is not an observable, so nothing would recompute the
header otherwise, and it went on reading the pre-split length after every cut.
"""
function retitle!(player::Player)
    t = get(player.fxwidgets, :cliptitle, nothing)
    t === nothing && return nothing
    s = cliptitle(player)
    t[] == s || (t[] = s)
    return nothing
end

function retitle!(clip::Clip)
    player = editorof(clip)
    (player === nothing || selectedclip(player) !== clip) && return nothing
    return retitle!(player)
end

"""
    showbake!(player) -> nothing
    showbake!(clip) -> nothing

Put the clip's bake state into the panel's bake row: what there is, whether it is
in use, and what the switch beside it would do.

Driven from `Clip`'s `setproperty!` when `:bake` is written — the bake finishing,
an undo restoring the one a clip had, a project load — rather than from those
places one by one. The row was refreshed only by the playhead listener, so a
finished bake left it reading "no bake — the graph renders every frame" until the
playhead moved: no feedback at exactly the moment someone is watching that row for
a result. The clip's own picture is separate and already follows (`showplayhead!`).
"""
function showbake!(player::Player)
    bakelabel = get(player.fxwidgets, :bakelabel, nothing)
    bakeuse = get(player.fxwidgets, :bakeuse, nothing)
    (bakelabel === nothing || bakeuse === nothing) && return nothing
    # `clip.bake` is written by `bakeclip!`, which runs on the pinned worker, and
    # a Label's text reaches GLMakie's screen — thread 1's. The queue is the
    # editor's one way across, and it is closed with the player.
    if Threads.threadid() != 1
        isopen(player.uiqueue) && put!(player.uiqueue, () -> showbake!(player))
        return nothing
    end
    loc = editclip(player)
    clip = loc === nothing ? nothing : loc[1]
    b = clip === nothing ? nothing : clip.bake
    # Both buttons stay put: hiding one would move the row's layout while it
    # is being read. The label says what there is to do, and the command
    # refuses with a reason when there is nothing.
    bakelabel.text[] =
        clip === nothing ? "" :
        b === nothing ? "no bake — the graph renders every frame" :
        !b.enabled && bakestale(clip) ?
            "bake switched off: the clip changed since · $(length(b.frames)) frames kept" :
        !b.enabled ? "bake off · $(length(b.frames)) frames on disk" :
        "bake in use · frames $(first(b.frames))–$(last(b.frames))"
    bakeuse.label[] = b !== nothing && b.enabled ? "off" : "use"
    return nothing
end

function showbake!(clip::Clip)
    player = editorof(clip)
    player === nothing && return nothing
    return showbake!(player)
end

"""
    buildcardfor!(clip, fx) -> nothing

Give `fx` a card, if `clip` is open in an editor and is the one on screen.

Called from [`addslot!`](@ref), where an effect joins the document. A clip that is
in no editor — one being built, one in an undo snapshot, one on the clipboard —
has no panel to build into and no card to build; a clip that is not the one being
shown gets its cards when it becomes so ([`showclip!`](@ref)).
"""
function buildcardfor!(clip::Clip, fx::Effect)
    # One that already has a card is being re-seated, not added — a restore
    # rebuilds `clip.effects` in the snapshot's order and hands every entry back
    # through `addslot!`. Rebuilding here would throw away the cards of everything
    # the undo did not touch, along with their fold state.
    fx.card === nothing || return nothing
    player = editorof(clip)
    (player === nothing || player.shownclip !== clip) && return nothing
    buildcard!(player, clip, fx)
    placelanes!(player.timeline, clip)
    showemptystate!(player, clip)
    applyfilter!(player)
    return nothing
end

"""
    rebuildcard!(player, clip, fx) -> nothing

Build `fx`'s card again — for the two edits that change what a row IS rather than
what it shows: binding a parameter to an input takes its slider away, and cutting
the edge gives it back.
"""
function rebuildcard!(player::Player, clip::Clip, fx::Effect)
    dropcard!(fx)
    buildcard!(player, clip, fx)
    placelanes!(player.timeline, clip)
    applyfilter!(player)
    return nothing
end

"""
    dropcard!(fx) -> nothing

Delete `fx`'s card and everything its rows drew: the widgets, and every
parameter's lane on the timeline.

Called where an effect leaves the document — taken off the stack, or dropped by an
undo — because that is the only moment at which its drawing becomes meaningless.
"""
function dropcard!(fx::Effect)
    for p in fx.params
        v = p.view
        v === nothing && continue
        # from wherever it is drawn — the plot knows, which is what lets this be
        # said from `clips.jl`, where an effect leaves the document and no panel
        # is in reach
        Makie.delete!(Makie.parent_scene(v.lane), v.lane)
        foreach(Observables.off, v.regs)   # …and what the lane derived, see `derive`
        p.view = nothing
    end
    # Before the card: the context's rows and controls are blocks inside it, and
    # its listeners fire on the playhead and the selection. Left behind, one of
    # them redraws its list into a layout whose scene has just been freed.
    cleartoolcontext!(fx.tool)
    fx.tool = nothing
    fx.card === nothing || Makie.delete!(fx.card)
    fx.card = nothing
    return nothing
end

"""
    dropcards!(clip) -> nothing

Delete every card of `clip` and take its sub-layout out of the stack.

Called where the clip leaves the document — deleted, or dropped by an undo — for
the reason [`dropcard!`](@ref) is: that is the moment its drawing stops meaning
anything. Takes no player, so `clips.jl` can say it at the two places that know.
"""
function dropcards!(clip::Clip)
    foreach(dropcard!, clip.effects)
    clip.cardlayout === nothing && return nothing
    gc = Makie.GridLayoutBase.gridcontent(clip.cardlayout)
    gc === nothing || Makie.GridLayoutBase.remove_from_gridlayout!(gc)
    clip.cardlayout = nothing
    return nothing
end

"Show or hide every card of `clip`, in one relayout."
setcardsvisible!(player::Player, clip::Clip, on::Bool) =
    (clip.cardlayout === nothing ||
         filter_cards!(_ -> on, clip.cardlayout, cardsof(clip));
     nothing)

"The cards of `clip`'s effects that have been built."
cardsof(clip::Clip) = Card[fx.card for fx in clip.effects if fx.card !== nothing]

"""
    applyfilter!(player) -> nothing

Hide the cards the query rules out, in one relayout per stack. Never rebuilds.
"""
function applyfilter!(player::Player)
    panel = player.fxpanel
    panel === nothing && return nothing
    q = panel.query[]
    clip = player.shownclip
    shown = 0
    total = 0
    if clip !== nothing && clip.cardlayout !== nothing
        fxs = [fx for fx in clip.effects if fx.card !== nothing]
        total += length(fxs)
        filter_cards!(clip.cardlayout, Card[fx.card for fx in fxs]) do card
            i = findfirst(fx -> fx.card === card, fxs)::Int
            keep = kindmatches(kindofslot(fxs[i]), q)
            keep && (shown += 1)
            keep
        end
    end
    if !isempty(panel.toolcards)
        names = collect(keys(panel.toolcards))
        total += length(names)
        filter_cards!(panel.tools, Card[panel.toolcards[n] for n in names]) do card
            i = findfirst(n -> panel.toolcards[n] === card, names)::Int
            keep = kindmatches(kindbyname(names[i]), q)
            keep && (shown += 1)
            keep
        end
    end
    panel.countlabel.text[] = isempty(q) ? "" : "$shown of $total shown · Esc clears"
    return nothing
end

"""
    showemptystate!(player, clip) -> nothing

Say what there is instead of a stack: no clip at the playhead, or a clip with no
effects on it.

Built and thrown away rather than kept, because it is two `Label`s and it says one
of exactly three things — one of which is nothing at all, when there are cards to
look at. The symbol it last said is what stops it being rebuilt on every playhead
move.
"""
function showemptystate!(player::Player, clip::Union{Nothing, Clip})
    panel = player.fxpanel
    want = clip === nothing ?
           (:noclip, "No clip at the playhead.",
            "Move the playhead onto a clip to give it effects.") :
           isempty(clip.effects) ?
           (:noeffects, "No effects on this clip.", "Add one above, or press Ctrl+P.") :
           nothing
    said = get(player.fxwidgets, :emptystate, nothing)
    said === (want === nothing ? nothing : want[1]) && return nothing
    for b in panel.empty
        b isa GridLayout ?
            (gc = Makie.GridLayoutBase.gridcontent(b);
             gc === nothing || Makie.GridLayoutBase.remove_from_gridlayout!(gc)) :
            Makie.delete!(b)
    end
    empty!(panel.empty)
    player.fxwidgets[:emptystate] = want === nothing ? nothing : want[1]
    want === nothing && return nothing
    # Row 0: above the clips' rows, whichever of them exist. The box goes into
    # `empty` with its labels — leaving it behind would stack one dead layout per
    # trip through here.
    box = GridLayout(panel.stack[0, 1]; alignmode = Makie.Outside(4, 4, 10, 10))
    push!(panel.empty, box,
          Label(box[1, 1], want[2]; halign = :left, fontsize = 12,
                color = panel.uicolors.text, tellwidth = false),
          Label(box[2, 1], want[3]; halign = :left, fontsize = 11,
                color = panel.uicolors.text_muted, tellwidth = false))
    return nothing
end

"The scene a card's image plots are drawn into — the Effects panel's own."
contentscene(player::Player) = player.fxpanel.scroll.scene

"""
Make the tool cards clickable.

A card built by `tooladdcard!` is a picture in this panel's scroll scene plus a
frame Block, not a Button, so it takes no click by itself. The Tools dock used to
dispatch that; without it every loop-reference card was inert and `onclick` was
never called.

Below the default priority, so the × Button in a card's own header still wins its
press. The hit test is the frame's computed bbox, topmost card first.
"""
function wiretoolcards!(player::Player)
    on(events(player.fig).mousebutton; priority = -1) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        player.dockopen[] === :effects || return Consume(false)
        tc = get(player.fxwidgets, :toolcards, nothing)
        tc === nothing && return Consume(false)
        pos = events(player.fig).mouseposition[]
        for c in Iterators.reverse(tc[3])
            c.onclick === nothing && continue
            bb = c.frame.layoutobservables.computedbbox[]
            (all(isfinite, bb.origin) && all(isfinite, bb.widths)) || continue
            bb.origin[1] <= pos[1] <= bb.origin[1] + bb.widths[1] || continue
            bb.origin[2] <= pos[2] <= bb.origin[2] + bb.widths[2] || continue
            c.onclick(c.id)
            return Consume(true)
        end
        return Consume(false)
    end
    return nothing
end

"""
    withtoolslots!(build, player, ctx, gridpos)

Run a kind's `body` with its card slots pointed at `gridpos`.

A body says `toolaction!(ctx, ...)`, `toolrows!(ctx, ...)`, `tooladdcard!(ctx, ...)`
and each lands in a named slot. The slots were cells of the Tools dock and are now
cells of this card, so the same body code builds into the Effects panel.

The slots are made `measurable!` because a GridLayout with no content has no
determinable height, and one such slot hides the height of everything above it —
including the scroll panel's content size.
"""
function withtoolslots!(build::Function, player::Player, ctx::EffectContext, gridpos)
    slots = get!(() -> (Dict{Symbol, Any}(), Dict{Symbol, Any}(), Dict{Symbol, Any}()),
                 player.fxwidgets, :toolslots)
    cards = get!(() -> (Dict{Symbol, Any}(), contentscene(player), Any[]),
                 player.fxwidgets, :toolcards)
    # Every one of these SPANS the card. A GridLayout reports its content's width
    # and is centred in its cell, so a slot holding rows that do not report one —
    # a status label, a list row, anything `tellwidth = false` — shrank to the
    # widest child that did: the matte panel drew its list in a 20 px column with
    # "frame 0" clipped to "n". `fill!` is the whole statement, made where the
    # slot is created rather than in each tool that builds into one.
    # …and the COLUMN gets the width too. A tool's controls are built as
    # `Button(slot[r, 1]; tellwidth = false, width = Relative(1.0))` — "as wide as
    # my column", by a widget that tells the column nothing. In an `Auto` column
    # that is circular and resolves to one pixel, which is what every action
    # button in a tool card was.
    span!(g) = (g.width[] = Makie.Relative(1.0); g.halign[] = :left;
                colsize!(g, 1, Makie.Relative(1.0)); g)
    gl = span!(GridLayout(gridpos))
    slots[1][ctx.tool] = span!(measurable!(GridLayout(gl[1, 1])))
    slots[2][ctx.tool] = span!(measurable!(GridLayout(gl[2, 1])))
    cards[1][ctx.tool] = span!(measurable!(GridLayout(gl[3, 1])))
    # …and one below the cards, for the action that acts on the whole list: "Apply
    # matte to clip" above the marked frames reads as a control for something
    # further up.
    slots[3][ctx.tool] = span!(measurable!(GridLayout(gl[4, 1])))
    # The four slots sit nearly flush. GridLayout's default rowgap of 16 is ~48 px
    # of empty band under every tool card's header, paid whether or not the slots
    # below hold anything. The content separates itself (a card has padding,
    # stacked actions have their own gap); a few pixels still marks the areas.
    rowgap!(gl, 4)
    build()
    return gl
end

"""
    addeffect!(player, name) -> Bool

Put effect kind `name` on the clip under the playhead, at its defaults, and build
its card. The menu, the palette and MCP all come through here.
"""
function addeffect!(player::Player, name::Symbol)
    loc = editclip(player)
    if loc === nothing
        setstatus!(player, "no clip at the playhead — move it onto a clip first")
        return false
    end
    k = kindbyname(name)
    if k === nothing || k.make === nothing
        setstatus!(player, "no effect named “$(name)”")
        return false
    end
    clip = loc[1]
    snapshot!(player)
    fx = addslot!(clip, Effect(k.make(defaults(k))))
    # …and this clip's cards are the ones on screen, the new entry's included: an
    # effect without a card is exactly what `showclip!` builds. Calling
    # `buildcard!` here as well built a SECOND card for the same effect and
    # overwrote `fx.card` with it — the first was then unreachable and undeletable,
    # while its widgets kept their mouse handlers and kept claiming clicks at the
    # rectangle the live card now occupies. A `Menu` so orphaned swallowed the
    # release that would have opened the live one.
    showclip!(player, clip)
    showemptystate!(player, clip)
    applyfilter!(player)
    selectfxcard!(player, (:fx, fx.id))
    setstatus!(player, "added $(k.label) — tune it below (Ctrl+Z removes)")
    showplayhead!(player)
    return true
end

"""
    removeeffect!(player, clip, fx) -> nothing

Take `fx` off `clip` and delete its card — the inverse of [`addeffect!`](@ref).
"""
function removeeffect!(player::Player, clip::Clip, fx::Effect)
    removeslot!(clip, fx.id)
    dropcard!(fx)
    showemptystate!(player, clip)
    applyfilter!(player)
    showplayhead!(player)
    return nothing
end

"""
    showkind!(player, name) -> Bool

Bring effect kind `name` into view on the clip at the playhead: open the Effects
panel, add the effect if it is not there yet, unfold its card and select it.

The single entry point for "put me where this effect is": the palette, MCP and
the tests use it instead of knowing which dock something lives in.
"""
function showkind!(player::Player, name::Symbol)
    opendock!(player, :effects)
    loc = editclip(player)
    if loc === nothing
        setstatus!(player, "no clip at the playhead — move it onto a clip first")
        return false
    end
    clip = loc[1]
    k = kindbyname(name)
    k === nothing && return false
    if findeffect(clip, k) === nothing
        addeffect!(player, name) || return false
    end
    slot = findfirst(s -> renderable(s) && k.matches !== nothing && k.matches(op(s)),
                     clip.effects)
    slot === nothing && return false
    fx = clip.effects[slot]
    # …and its card is up. "Bring this into view" has to mean the card exists: the
    # effect may have been on the clip all along with its card never built (the
    # dock was closed) or dropped and not yet replaced (an undo). Its body's
    # widgets are what the caller is about to reach for, and a card that is not
    # there leaves them pointing at the deleted one.
    showclip!(player, clip)
    fx.card === nothing && return false
    selectfxcard!(player, (:fx, fx.id))
    fx.card.open = true
    return true
end

"""
    opentools(player) -> Vector{EffectKind}

The tool-only cards the user has actually opened, in registration order.

Rendering all of them put four cards on every clip. The panel shows what was put
there and nothing else; the searchable menu is how a tool that is not a clip
effect is found.
"""
opentools(player::Player) =
    player.fxpanel === nothing ? EffectKind[] :
    filter(k -> haskey(player.fxpanel.toolcards, k.name), toolonlykinds())

"""
    opentool!(player, name) -> nothing

Put a tool-only card on the panel — the menu's "add" for a kind that is not a clip
effect. Already open, it stays as it is.
"""
function opentool!(player::Player, name::Symbol)
    panel = player.fxpanel
    k = kindbyname(name)
    if k === nothing || haskey(panel.toolcards, name)
        setstatus!(player, k === nothing ? "no tool named “$(name)”" :
                           "$(k.label) is already open")
        return nothing
    end
    placetoolcard!(player, k)
    setstatus!(player, "$(k.label) — its × closes it again")
    return nothing
end

"""
    placetoolcard!(player, kind) -> Card

Build `kind`'s card into the panel's tool row and record it.

A fixed row per kind — its place in the registry — so opening or closing one does
not move the others.
"""
function placetoolcard!(player::Player, kind::EffectKind)
    panel = player.fxpanel
    row = something(findfirst(x -> x.name === kind.name, toolonlykinds()), 1)
    # …its × empties this row again, and the rows of the kinds before it in the
    # registry were never filled in the first place.
    measurable!(panel.tools, (1:row)...)
    card, ctx = toolonlycard!(player, panel.tools, row, kind, panel.uicolors)
    panel.toolcards[kind.name] = card
    get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :toolpanels)[ctx.tool] = ctx
    applyfilter!(player)
    return card
end

"Delete a tool-only card and everything its body drew. `false` if it was not open."
function droptoolcard!(player::Player, name::Symbol)
    panel = player.fxpanel
    panel === nothing && return false
    card = get(panel.toolcards, name, nothing)
    card === nothing && return false
    delete!(panel.toolcards, name)
    ctx = get(get(player.fxwidgets, :toolpanels, Dict{Symbol, Any}()), name, nothing)
    ctx === nothing || cleartoolcontext!(ctx)
    Makie.delete!(card)
    # The row is empty now, and the layout has to be told: deleting a block takes
    # it out of the grid but leaves the layout reporting the height it had, so the
    # panel kept 34 px of nothing under the stack for every card ever closed.
    Makie.GridLayoutBase.update!(panel.tools)
    return true
end

"Take a tool-only card off the panel again, with everything its body drew."
closetool!(player::Player, name::Symbol) =
    (droptoolcard!(player, name) && applyfilter!(player); nothing)

"""
    rebuildtoolcard!(player, name) -> nothing

Build tool `name`'s card again, for an edit that changes what the card IS rather
than what it shows — the crop card's scope toggle and its size readout, and a
tool being switched on or off ([`activatetool!`](@ref)).

Either kind of card: a tool-only one on the panel, or the card of the tool's own
slot on the shown clip (the loop finder's, the matte's). A no-op when neither is
up.

The state change really is a different card and not a different label: the loop
finder's one action reads "Find similar frames" in both states, but off it turns
the tool ON and running it adds another reference frame — which of the two the
button carries is decided where the body is built. Cards are built once and kept,
so nothing else would ever replace it, and the card built while the tool was off
went on offering to turn it on: the second "Find" switched the tool back off.
"""
function rebuildtoolcard!(player::Player, name::Symbol)
    if droptoolcard!(player, name)
        k = kindbyname(name)
        k === nothing || placetoolcard!(player, k)
        return nothing
    end
    clip = player.shownclip
    clip === nothing && return nothing
    fx = findslot(clip, name)
    fx === nothing && return nothing
    rebuildcard!(player, clip, fx)
    return nothing
end

"""
    toolcardopen(kind, player) -> Bool

Whether a tool-only card starts unfolded.

Open when the tool has something to show — narration lines, a transcript, a
retimed clip — and folded otherwise, which still shows the header. All four
expanded on every clip fills the panel with cards that are mostly empty.

Crop is never open by default: it is reached for, not read.
"""
function toolcardopen(kind, player::Player)
    seq = player.sequence
    kind.name === :narration  && return !isempty(seq.narration)
    kind.name === :transcript && return !isempty(seq.captions)
    if kind.name === :timeinterp
        loc = editclip(player)
        return loc !== nothing && loc[1].rate != 1.0
    end
    return false
end

"""
    toolonlycard!(player, toolgl, row, kind, uicolors) -> (card, ctx)

A card for a kind that is not a clip effect: a body and no `make`.

The card stack is built from `clip.effects`, which covers Blur, Matte and
Stabilize — their effect sits on the clip. A kind with no `make` never gets there,
so `registertool!` — whose whole job is registering exactly that shape — produced
panels that could not appear. It went unused after the Tools dock was removed,
which is why nothing noticed until four of them were written against it.

These are the project- and sequence-level tools (the crop scope, the transcript,
the narration, the retime mode). They belong under the clip's effects, not inside
them, and they have no slot, no bypass eye and no keyframes — which is why this is
its own builder rather than another branch through `fxcard!`.
"""
function toolonlycard!(player::Player, toolgl, row::Integer, kind, uicolors)
    card = Card(toolgl[row, 1]; title = kind.label,
                selected = false,
                open = toolcardopen(kind, player),
                backgroundcolor = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                                   RGBf(1, 1, 1), 0.045),
                headercolor = uicolors.surface,
                headercolor_selected = uicolors.select_subtle,
                strokecolor = uicolors.border,
                selectioncolor = uicolors.select,
                titlecolor = uicolors.text)
    acc = GridLayout(card_accessory(card))
    flat = (buttoncolor = (:transparent, 0.0), strokewidth = 0, cornerradius = 3,
            height = 20, buttoncolor_hover = uicolors.accent_subtle)
    if !isempty(kind.description)
        help = Button(acc[1, 1]; label = "?", width = 18, fontsize = 11,
                      labelcolor = uicolors.text_muted, flat...)
        tips = get(player.fxwidgets, :tips, nothing)
        tips === nothing || (tips[help] = wraptext(kind.description, 46))
    end
    # …and a × like every other card has. A card the menu can open has to be
    # closable from the card, not only by finding the menu entry again.
    rm = Button(acc[1, 2]; label = "×", width = 20, fontsize = 13,
                labelcolor = uicolors.text_muted, flat...)
    on(_ -> closetool!(player, kind.name), rm.clicks)
    ctx = EffectContext(player, kind.name)
    ctx.state = nothing
    withtoolslots!(player, ctx, card[1, 1]) do
        # guarded exactly as `fxcard!` guards a body: one tool that throws must
        # not take the rest of the stack with it
        try
            kind.body(ctx)
        catch e
            @error "tool card body failed" kind = kind.name exception = (e, catch_backtrace())
        end
    end
    return card, ctx
end

"""
Kinds the Effects panel must render on their own — a card with no home on a
clip, opened from the menu.

Declared (`tool = true`), not inferred: "a body and no `make`" also describes the
`:scene` kind, whose body belongs INSIDE the clip's own card — inferring the
shape put a "Scene" entry in the Add-effect menu whose card had no clip to
describe.
"""
toolonlykinds() = filter(k -> k.tool, effectkinds())

"""
The kind behind a stack entry: from its payload where it has one, by name where it
does not.

A data entry (the `:scene`) has no payload — see [`renderable`](@ref) — so
`effectkindfor(op(slot))` would try to build one. Its registered name is the
answer.
"""
kindofslot(slot::Effect) =
    renderable(slot) ? effectkindfor(op(slot)) : kindbyname(slot.kind)

"""
One card: the effect's name in the header, its enable toggle and remove × in the
accessory, its parameters and its own body inside.

What the header shows is derived — the selection highlight from
`player.fxselection`, the eye from `fx.enabled` — so selecting a card or bypassing
an effect is a write, not a rebuild of the stack.
"""
function fxcard!(player::Player, stackgl, row::Integer, clip::Clip, slot::Effect, uicolors)
    kind = kindofslot(slot)
    title = kind === nothing ? String(slot.kind) : kind.label
    key = (:fx, slot.id)
    # Derived, and unhooked with the card: `player.fxselection` and `slot.enabled`
    # both outlive it, and `map` would leave a listener on each per card ever
    # built. Measured before this: 20 rebuilds of one card left 20 listeners on
    # `fxselection` and 40 on `enabled`.
    selected, selreg = derive(s -> s == key, player.fxselection)
    card = Card(stackgl[row, 1]; title, selected,
                backgroundcolor = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                                   RGBf(1, 1, 1), 0.075),
                headercolor = uicolors.surface,
                headercolor_selected = uicolors.select_subtle,
                strokecolor = uicolors.border,
                selectioncolor = uicolors.select,
                titlecolor = uicolors.text)
    push!(card.blockscene.deregister_callbacks, selreg)
    on(_ -> selectfxcard!(player, key), card.headerclicks)

    acc = GridLayout(card_accessory(card))
    # Flat: no fill, no stroke, the glyph with a hover tint. The header already
    # has a background and, when selected, an outline; a button drawing a third
    # box inside those reads as clutter.
    flat = (buttoncolor = (:transparent, 0.0), strokewidth = 0, cornerradius = 3,
            buttoncolor_hover = uicolors.accent_subtle,
            buttoncolor_active = uicolors.accent, height = 20)
    # The description on hover: the tool descriptions are instructions, and printed
    # in the card they cost a third of its height and push the controls out of
    # sight. A ? in the title bar keeps them one hover away.
    if kind !== nothing && !isempty(kind.description)
        help = Button(acc[1, 2]; label = "?", width = 18, fontsize = 11,
                      labelcolor = uicolors.text_muted, flat...)
        tips = get(player.fxwidgets, :tips, nothing)
        tips === nothing || (tips[help] = wraptext(kind.description, 46))
    end
    # The same glyph and gesture as the panel's bypass-everything eye, so "is this
    # applied" reads the same at both levels. Glyph and colour are both derived
    # from the flag the RENDER reads, so there is one answer to "is this on".
    eyeglyph, eyereg = derive(e -> e ? "◉" : "○", slot.enabled)
    eyetint, tintreg = derive(e -> e ? uicolors.text : uicolors.text_muted, slot.enabled)
    append!(card.blockscene.deregister_callbacks, (eyereg, tintreg))
    eye = Button(acc[1, 3]; label = eyeglyph, labelcolor = eyetint,
                 width = 22, fontsize = 12, flat...)
    on(eye.clicks) do _           # off keeps the parameters; every render path skips it
        snapshot!(player)
        slot.enabled[] = !slot.enabled[]
        showplayhead!(player)
    end
    player.fxwidgets[Symbol(:fxeye_, slot.id)] = eye
    rm = Button(acc[1, 4]; label = "×", width = 20, fontsize = 13,
                labelcolor = uicolors.text_muted, flat...)
    colgap!(acc, 2)
    on(rm.clicks) do _
        snapshot!(player)
        removeeffect!(player, clip, slot)
        setstatus!(player, "removed $title (Ctrl+Z restores)")
    end
    player.fxwidgets[Symbol(:fxremove_, slot.id)] = rm

    # A kind the registry no longer knows has no parameters to mint, so its ∿ can
    # be built here — everything below only applies to a kind that IS known.
    kind === nothing &&
        (player.fxwidgets[Symbol(:fxlane_, slot.id)] =
             laneeye!(player, acc[1, 1], slot.params, uicolors,
                      card.blockscene.deregister_callbacks;
                      width = 20, fontsize = 12, flat...);
         return card, nothing)
    # A card whose action is running says so — the highlight the Tools dock used
    # to put on its header.
    on(activetoolname(player); update = true) do a
        card.headercolor = a === kind.name ? uicolors.accent : uicolors.surface
        card.titlecolor = a === kind.name ? uicolors.text_on_accent : uicolors.text
    end
    r = 0
    # `sectionform!`, not `paramform!`: an effect whose parameters come with a
    # grouping gets one section per group and a filter box over them. For the ten
    # kinds that declare a flat list of scalars it is exactly the rows it always
    # drew.
    # Ask `paramsections` what there is to show rather than reading `slot.params`.
    # For the ten kinds that declare scalars it hands back exactly those; for a
    # scene the rows come from the realized scene and the parameters are made from
    # them, so a freshly inserted scene clip has empty `params` and a full card.
    # Testing `isempty(slot.params)` skipped the build that creates them.
    secs = paramsections(clip, slot)
    isempty(secs) ||
        (r += 1; sectionform!(player, card[r, 1], clip, slot, uicolors; sections = secs))
    # The whole card's lanes, in the header's leftmost accessory cell. Not an ◉/○:
    # that pair means "is this effect applied" two buttons to the right, and in the
    # panel's own header — see `laneeye!`.
    #
    # AFTER `paramsections`, which is what MINTS a scene clip's parameters: built
    # before it, this hung its listeners on an empty list, and the ∿ then reported
    # the state from before the last click.
    player.fxwidgets[Symbol(:fxlane_, slot.id)] =
        laneeye!(player, acc[1, 1], slot.params, uicolors,
                 card.blockscene.deregister_callbacks;
                 width = 20, fontsize = 12, flat...)
    ctx = nothing
    # Slots for every kind on the stack, body or no body: a tool can also build
    # cards while it runs — the loop finder pushes a reference card per Find, from
    # an analysis that ends long after the panel was drawn. Without slots
    # `toolcard!` has nowhere to build and returns 0.
    if kind.body !== nothing || kind.activate !== nothing
        ctx = EffectContext(player, kind.name)
        r += 1
        # the body builds into the card, next to the parameters it belongs with
        ctx.state = nothing
        withtoolslots!(player, ctx, card[r, 1]) do
            kind.body === nothing && return
            # guarded: one body that throws must not take the whole stack with it
            try
                kind.body(ctx)
            catch e
                @error "effect card body failed" kind = kind.name exception = (e, catch_backtrace())
            end
        end
    end
    # An action button only for a kind with no body of its own. A body decides what
    # its actions are and when to offer them (`mattepanel!` shows "Mark subject"
    # only while nothing is marked), and `activatetool!` toggles, so a generic
    # button underneath ends a marking session and drops its points.
    #
    # Testing "unless the body registered callbacks" instead misses `mattecard!`,
    # which builds plain Buttons rather than `toolaction!`s and so registers none.
    if kind.activate !== nothing && kind.body === nothing
        r += 1
        act = Button(card[r, 1]; label = actionlabel(kind), tellwidth = false,
                     width = Makie.Relative(1.0))
        on(_ -> activatetool!(player, kind.name), act.clicks)
        player.fxwidgets[Symbol(:fxaction_, kind.name)] = act
    end
    return card, ctx
end

# There is no `overlaycard!` and no `overlaysat`.
#
# An overlay's card was this file's second card builder: same header, same eye,
# same ×, same parameter sections, and a span readout of its own — kept in step
# with the clip's by hand. A scene, a title and a subtitle are clips now, so
# `fxcard!` draws them, and what used to be overlay-specific (which part of the
# timeline it covers) is the clip's own extent.


"What the card's action button says — the kind's own verb, not a generic 'Run'."
actionlabel(kind::EffectKind) =
    kind.name === :stabilize ? "Stabilize clip" :
    kind.name === :flicker   ? "Analyze + fix flicker" :
    kind.name === :matte     ? "Mark subject" :
    kind.name === :restore   ? "Restore around the playhead" :
    kind.name === :blend     ? "Blend the marked clips" :
    kind.name === :loopfinder ? "Find similar frames" : "Run $(kind.label)"

"""
    showvalue!(control, value) -> nothing

Put `value` on the widget that shows it. A no-op where a parameter has no widget
— a driven one shows the edge's value and owns nothing.

Dispatch rather than a type test at the call site, and no cast: each method takes
the parameter's own value and converts what its widget needs.
"""
showvalue!(::Nothing, value) = nothing
showvalue!(sl::Makie.Slider, value) = (Makie.set_close_to!(sl, value); nothing)
showvalue!(cb::Makie.Checkbox, value) = (cb.checked[] = Bool(value); nothing)
function showvalue!(m::Makie.Menu, value)
    i = findfirst(o -> o === value || o == value, Makie.to_value(m.options))
    i === nothing || (m.i_selected[] = i)
    return nothing
end

"""
    wireedit!(player, target, p, control, derived) -> nothing

Let `control` edit `p` — while the user has hold of it.

The gesture is the DRAG, not the value: moving a slider from code changes its
value too, and a row that edited on that would write its own display back into
the curve it came from. Registrations go into `derived`, so they come off with
the card.
"""
function wireedit!(player::Player, target, p::Param, sl::Makie.Slider, derived)
    push!(derived, on(sl.value) do v
        sl.dragging[] || return nothing        # …not a gesture, just the display
        isdriven(p) && return nothing          # its value comes down an edge
        editparam!(player, target, p, v; frame = playheadframe(player, target))
    end)
    return nothing
end

function wireedit!(player::Player, target, p::Param, cb::Makie.Checkbox, derived)
    push!(derived, on(cb.checked) do v
        isdriven(p) || editparam!(player, target, p, v;
                                  frame = playheadframe(player, target))
    end)
    return nothing
end

wireedit!(::Player, target, ::Param, ::Any, derived) = nothing

"""
    laneeye!(player, gridpos, p::Param, uicolors, sink; kw...) -> Button
    laneeye!(player, gridpos, params::Vector{Param}, uicolors, sink; kw...) -> Button

The ∿ that draws or hides lanes on the timeline: click toggles, alt-click solos —
see [`sololanes!`](@ref).

One glyph for one question, asked at three levels: a row, a section of a card, a
whole card. ◉/○ is deliberately not reused for it, although the overview modal
did: that pair already means "is this effect applied", in the card's header and in
the panel's, and a second eye meaning something else in the same header is how a
header stops being readable.

A row's ∿ takes the parameter's own colour while its lane is drawn — what the lane
and the ◆ draw in — so the row says which of the curves down there is its own. A
group's ∿ is accent while it is the solo, the text colour while any of its lanes
are drawn, and muted when none is: pressing it has to say whether it did anything.

`sink` is where the registration goes, i.e. wherever the button's lifetime is
written down: a card's `deregister_callbacks`, or the form's `derived`.
"""
function laneeye!(player::Player, gridpos, p::Param, uicolors, sink; kw...)
    tint, reg = derive(v -> v ? paramcolor(p) : uicolors.text_muted, p.visible)
    push!(sink, reg)
    return buildlaneeye!(player, gridpos, p, tint,
                         "$(p.label) on the timeline · alt-click: only this lane"; kw...)
end

function laneeye!(player::Player, gridpos, params::Vector{Param}, uicolors, sink; kw...)
    want = Set{Param}(params)
    tint = Observable{Any}(uicolors.text_muted)
    function restate()
        s = player.lanesolo[]
        tint[] = s !== nothing && s.on == want ? uicolors.accent :
                 any(q -> q.visible[], params) ? uicolors.text : uicolors.text_muted
        return nothing
    end
    # One registration per parameter, rather than a single `onany` over all of
    # them: a scene card's group is two hundred parameters wide, and `onany` would
    # specialise a closure on a two-hundred-element tuple.
    push!(sink, on(_ -> restate(), player.lanesolo))
    for q in params
        push!(sink, on(_ -> restate(), q.visible))
    end
    restate()
    n = length(curvesof(params))
    return buildlaneeye!(player, gridpos, params, tint,
                         "$n curve$(n == 1 ? "" : "s") of $(length(params)) on the timeline · " *
                         "alt-click: only these"; kw...)
end

"""
    buildlaneeye!(player, gridpos, target, tint, tip; kw...) -> Button

The ∿ button itself, once its colour has been worked out. `target` is a `Param`
for a row and a `Vector{Param}` for a group, and that is what decides — by
dispatch, in [`togglelanes!`](@ref) and [`sololanes!`](@ref) — whether the gesture
is filtered to the curves.
"""
function buildlaneeye!(player::Player, gridpos, target, tint::Observable,
                       tip::AbstractString; kw...)
    eye = Button(gridpos; label = "∿", labelcolor = tint, kw...)
    tips = get(player.fxwidgets, :tips, nothing)
    tips === nothing || (tips[eye] = tip)
    # Alt is read at the click rather than carried by it: a Button reports that it
    # was pressed and nothing about the keyboard, and the modifier is the whole
    # difference between "hide this" and "show only this".
    on(_ -> ispressed(player.fig, Keyboard.left_alt | Keyboard.right_alt) ?
            sololanes!(player, target) : togglelanes!(player, target),
       eye.clicks)
    return eye
end

"""
The parameter rows: a `ParamForm` with the Premiere ◀ ◆ ▶ trio per parameter, and
each parameter's curve on the timeline.

Everything a row SHOWS is derived from `p.curve` and the playhead: the ◆'s glyph
and colour, the slider's position, and the lane. Nothing is pushed into a row, so
there is no refresh pass over the rows on screen.

The other direction is the form's own handler, which edits the curve. A slider
notification is a gesture exactly when the slider DISAGREES with the document at
this frame — so a sync, which moves it to that value, reports nothing by
construction. There is no flag saying "this write was not a gesture", nothing to
silence, and no baseline to keep in step.
"""
function paramform!(player::Player, pos, target, fx::Effect, uicolors;
                    params::Vector{Param} = fx.params, labels = p -> p.label,
                    labelcolor = uicolors.text, widgetwidth = 104, labelwidth = 80)
    # This entry's own parameters: `fx.params` are the objects themselves, not a
    # kind's declared list resolved through an index, so a row writes to the
    # parameter it is drawn for and two entries of one kind cannot collide.
    # The row label is not always the parameter's own: inside a section headed
    # `torso`, a row reading `torso · Angle` repeats it. The full label is what a
    # timeline lane and the ◆ overview show, where there is no heading.
    fieldsym(p) = Symbol(labels(p))
    frame() = playheadframe(player, target)
    spec = NamedTuple(fieldsym(p) => (Float64(valueat(p, frame())),
                                      Makie.Between(p.range[1], p.range[2]))
                      for p in params)
    kfbuttons = Dict{Symbol, Any}()
    # What a row derives from the PLAYHEAD, which outlives every card on it. The
    # handles are kept so they can come off with the form's scene below — `map`
    # returns the Observable and not the registration, so a derived label built
    # that way holds its row alive for the rest of the session.
    derived = Observables.ObserverFunction[]
    accessory = (field, gp) -> begin
        p = params[findfirst(q -> fieldsym(q) === field, params)]
        acc = GridLayout(gp)
        # Whether this row's curve is on the timeline at all — first in the trio's
        # row because it decides whether there is anything down there to walk.
        laneb = laneeye!(player, acc[1, 1], p, uicolors, derived;
                         width = 16, height = 22, fontsize = 11)
        # Fixed width and height: a Button sizes itself from its label, so ◇ → ◆
        # reported a new size and relaid out the panel — 85 ms and 34 MB per
        # character change. Pinned, the write is the glyph and nothing else.
        prevb = Button(acc[1, 2]; label = "◀", width = 16, height = 22, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        # What the ◆ says is a view of (this curve, this frame): a key sitting
        # here, a curve with none here, or a value arriving down an edge — in
        # which case it offers the only edit there is, cutting the edge.
        # …in the parameter's OWN colour, the one its lane draws in, so a curve on
        # the timeline can be traced back to the row that owns it.
        own = paramcolor(p)
        function kfglyph(c, n)
            isdriven(p) && return ("⇥", own)
            animated = length(c.keys) > 1
            here = animated && any(k -> k.frame == sourceframe(target, n), c.keys)
            return (here ? "◆" : "◇",
                    here ? own :
                    animated ? uicolors.text : uicolors.text_muted)
        end
        kfstate = Observable(kfglyph(p.curve[], player.playhead[]))
        append!(derived, onany((c, n) -> (kfstate[] = kfglyph(c, n)),
                               p.curve, player.playhead))
        kf = Button(acc[1, 3]; label = map(first, kfstate),
                    labelcolor = map(last, kfstate), width = 22, height = 22)
        nextb = Button(acc[1, 4]; label = "▶", width = 16, height = 22, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        colgap!(acc, 1)
        # ◀ ▶ walk this parameter's keys and ◆ keys it, except when it is driven
        # from elsewhere: its value is not this row's to author, so the middle
        # button cuts the edge and keeps the value it was showing.
        on(_ -> gotokey!(player, target, p, -1), prevb.clicks)
        on(kf.clicks) do _
            isdriven(p) ? unbindinput!(player, target, fx, p) :
                          togglekey!(player, target, p)
        end
        on(_ -> gotokey!(player, target, p, 1), nextb.clicks)
        kfbuttons[fieldsym(p)] = kf
        player.fxwidgets[Symbol(:kfacc_, fx.id, :_, p.name)] = (prevb, kf, nextb)
        player.fxwidgets[Symbol(:kflane_, fx.id, :_, p.name)] = laneb
        acc
    end
    # `width = nothing` plus a flexible widget column, so the row follows the
    # panel. The row has to fit the card's scene: 80 + 104 + 73 + 2 gaps = 273
    # inside the 356 px it draws into. At 88 + 132 + 62 = 298 the row kept its
    # width, hung over the edge and the scene cut it off mid-widget — the ◆ sliced
    # in half, the ▶ missing. Nothing in the layout reports an overflow.
    #
    # A flexible widget column alone does not fix it: the form then fills its
    # cell, which is ten pixels wider than the scene that draws it (the card hands
    # out more than it paints — see the note in `Subfigure`). 73 is what the ∿ and
    # the trio measure: 16 + 16 + 22 + 16 plus three 1 px gaps.
    pf = Makie.ParamForm(pos, spec, accessory; labelwidth = labelwidth,
                         widgetwidth = widgetwidth,
                         accessorywidth = 73, rowgap = 4, halign = :left,
                         labelcolor = labelcolor)
    for p in params
        w = get(pf.widgets, fieldsym(p), nothing)
        bindparamview!(player, target, p, w isa Makie.Block ? w : nothing,
                       kfbuttons[fieldsym(p)])
    end
    # What a row SHOWS, per row: its own slider follows its own curve and the
    # playhead. Two named inputs, one derivation, no pass over the card and no
    # listener whose arity comes from the data.
    #
    # And what a row WRITES hangs on the DRAG, not on the value: a sync moves the
    # widget and notifies, which is harmless because nothing edits on a value
    # change. That is what replaced `display_value!`, `update_silent!` and the
    # baseline they were measured against — there is nothing to silence when a
    # notification was never mistaken for an intent.
    for p in params
        ctrl = p.view === nothing ? nothing : p.view.control
        ctrl === nothing && continue
        append!(derived, onany(p.curve, player.playhead) do c, _
            showvalue!(ctrl, valueat(c, frame()))
        end)
        wireedit!(player, target, p, ctrl, derived)
    end
    append!(pf.blockscene.deregister_callbacks, derived)
    return pf
end

"""
    bindparamview!(player, clip, p, control, kf) -> ParamView

Hang `p`'s widgets on it, and put its curve on the timeline.

The lane is created here, next to the row it belongs with, so that showing a
parameter and drawing its curve are one act. `visible` is the parameter's own
flag, `selectedkey` is derived from the editor's single selection, the colour is
[`paramcolor`](@ref) — the same one its ◆ draws in — and the placement comes from
[`placelanes!`](@ref): the clip's own, pushed where the timeline places everything
else it draws.
"""
function bindparamview!(player::Player, clip, p::Param, control, kf::Makie.Button)
    # …through `derive`, not `map`: `player.selectedkey` outlives every lane ever
    # drawn on it, and the registration has to come off with this one.
    selkey, reg = derive(s -> s !== nothing && s[1] === p ? s[2] : 0,
                         player.selectedkey)
    lane = lanecurve!(player.timeline.axis, p.curve;
                      valuerange = p.range,
                      viewrange = player.timeline.viewrange,
                      pixelspersecond = player.timeline.pps,
                      visible = p.visible,
                      selectedkey = selkey,
                      color = paramcolor(p))
    translate!(lane, 0, 0, 3)          # over the filmstrip, under the playhead
    p.view = ParamView(control, kf, lane, Observables.ObserverFunction[reg])
    return p.view
end

# --------------------------------------------------- parameters, in sections

"""
    paramsections(target, fx) -> Vector{NamedTuple{(:label, :detail, :params)}}

How a card groups its parameters.

One unnamed group by default: a Blur has one parameter and a heading over it is
noise. A scene clip is the other case — its groups come from the scene that was
built (one per named plot), and its parameters are minted on first use.

Asked lazily: what a scene offers depends on the scene, so the panel asks when the
card is opened rather than keeping a description of it
around to consult. A parameter that already exists — read from the project file,
or made when the card was last open — is reused, so its curve is never lost.
"""
function paramsections(target, fx::Effect)
    secs = sceneparamsections(target, fx)
    secs === nothing || return secs
    # The ones the KIND declares, not everything the entry happens to carry. A
    # Blur on a scene clip used to be handed the scene's two hundred parameters —
    # and `sceneparam!` does not only read them, it MINTS them onto the effect, so
    # projects saved since carry them. Filtering here is what stops such an entry
    # from drawing a scene's object list in a Blur card.
    k = kindbyname(fx.kind)
    declared = k === nothing ? nothing : Set{Symbol}(p.name for p in k.params)
    rest = Param[q for q in fx.params
                 if q.range !== nothing && (declared === nothing || q.name in declared)]
    return isempty(rest) ? NamedTuple[] :
           NamedTuple[(label = "", detail = "", params = rest)]
end

"""
    sceneparamsections(target, fx) -> Vector{NamedTuple} | nothing

The sections a scene clip's card shows, or `nothing` when the target is not one.

`nothing` rather than an empty list, so "not a scene" and "a scene not built yet"
stay different answers; the second is what a clip returns before it has rendered
once.

Both halves are asked: the clip has to BE a scene, and the entry has to be the one
that renders it. Asking only the clip gave every effect on a scene clip the whole
object list — a Blur card headed "229 parameters · 11 objects" — and because
[`sceneparam!`](@ref) mints what it does not find, it wrote all of them onto that
Blur as well.
"""
sceneparamsections(::Any, ::Effect) = nothing

function sceneparamsections(clip::Clip, fx::Effect)
    (fx.kind === :scene && clip.source isa SceneSource) || return nothing
    out = NamedTuple[]
    for obj in sceneattributes(clip.source)
        ps = Param[sceneparam!(fx, r) for r in obj.rows if r.kind === :number]
        isempty(ps) || push!(out, (label = obj.label, detail = obj.detail, params = ps))
    end
    return out
end

"""
    sceneparam!(fx, row) -> Param

The parameter for one scene attribute, made if it is not there yet.

Kept on the effect, so a keyframe outlives the scene: a project is loaded long
before anything is rendered and its curves have to be waiting when the scene is
built. An existing parameter is returned untouched — its curve is the edit.
"""
function sceneparam!(fx::Effect, row)
    v = Float64(row.value)
    span = (min(0.0, 2v), max(1.0, 2v))
    i = findfirst(q -> q.name === row.path, fx.params)
    i === nothing || return withspan!(fx, i, span, row.label)
    fresh = Param(row.path, row.label, v; range = span)
    push!(fx.params, fresh)
    return fresh
end

"""
    withspan!(fx, i, span, label) -> Param

The `i`-th parameter, guaranteed to have a range and the scene's own label. Both
fields are `const`, so it is replaced when it has neither.

A project written while scenes were overlays stored a parameter's curve but no
span, there being no slider to size — all seven of the lego project's animated
scene parameters load that way. A row without a range has no widget to build:
`paramform!` reads `p.range[1]` and throws `getindex(::Nothing, ::Int64)` inside
the card builder.

The label comes across for the same reason: the file stores the path
(`torso.offset[2]`) where the scene knows the name (`torso · Offset Y`).

The curve and any edge come across untouched — the span and the name are the only
things missing, and the built scene knows both.
"""
function withspan!(fx::Effect, i::Integer, span, label::AbstractString)
    p = fx.params[i]
    p.range === nothing || return p
    fresh = Param(p.name, label, valueat(p, 0); curve = p.curve[], visible = p.visible[],
                  range = span, input = p.input)
    fx.params[i] = fresh
    return fresh
end

"""
    sectionform!(player, gridpos, target, fx, uicolors) -> blocks

The parameter area of a card: plain rows for one group, a filter box over a list
of collapsible sections for more.

The sections are `Card`s with their chrome turned off — no border, no fill, a
short header — because that is what a section is, and because it means the filter
is `filter_cards!`: one relayout that hides what does not match, the same
mechanism and the same speed as the panel's own filter over the effect cards.
Rebuilding the rows per keystroke instead was measured at 433 ms against 14 ms.
"""
function sectionform!(player::Player, gridpos, target, fx::Effect, uicolors;
                      sections = nothing, lazyabove = 60)
    secs = sections === nothing ? paramsections(target, fx) : sections
    isempty(secs) && return Any[]
    if length(secs) == 1 && isempty(secs[1].label)
        return Any[paramform!(player, gridpos, target, fx, uicolors;
                              params = secs[1].params)]
    end
    gl = GridLayout(gridpos)
    filterrow = GridLayout(gl[1, 1])
    box = Textbox(filterrow[1, 1]; placeholder = "filter objects…",
                  width = Makie.Relative(1.0), tellwidth = false,
                  reset_on_defocus = false)
    count = Label(filterrow[2, 1], ""; halign = :left, fontsize = 10,
                  color = uicolors.text_muted, tellwidth = false)
    stack = measurable!(GridLayout(gl[2, 1]; valign = :top, default_rowgap = 0))
    nparams = sum(length(sec.params) for sec in secs)

    cards = Card[]
    eyes = Makie.Button[]
    for (i, sec) in enumerate(secs)
        title = isempty(sec.detail) ? sec.label : "$(sec.label)   ·   $(sec.detail)"
        card = Card(stack[i, 1]; title, open = false,
                    backgroundcolor = (:transparent, 0.0),
                    headercolor = uicolors.surface_subtle,
                    headercolor_selected = uicolors.surface_subtle,
                    strokewidth = 0, cornerradius = 3, headerheight = 20,
                    titlefont = :regular, titlesize = 11, titleoffset = 6,
                    titlecolor = uicolors.text_muted,
                    bodypadding = (10, 2, 4, 2), spacing = 2)
        # One object's lanes as a group: on a scene clip a section IS a plot, which
        # is the unit you want off the timeline while you work on another one.
        push!(eyes,
              laneeye!(player, card_accessory(card), sec.params, uicolors,
                       card.blockscene.deregister_callbacks;
                       width = 18, height = 16, fontsize = 10,
                       buttoncolor = (:transparent, 0.0), strokewidth = 0,
                       cornerradius = 3,
                       buttoncolor_hover = uicolors.accent_subtle,
                       buttoncolor_active = uicolors.accent))
        # Above `lazyabove` a folded section builds its rows the first time it is
        # opened rather than when the panel is drawn. A scene's card carries one
        # section per object — 229 parameters on the lego project, all folded —
        # and building them anyway was 7 of the 9 seconds a rebuild took. A row
        # that does not exist draws nothing and derives nothing.
        #
        # Below that threshold the whole card is built at once, so every row
        # exists as soon as the card does (a graphic's rows are asserted on
        # directly).
        built = Ref(false)
        function buildrows!()
            built[] && return nothing
            built[] = true
            paramform!(player, card[1, 1], target, fx, uicolors;
                       params = sec.params, labels = q -> chopprefix(q.label, sec.label * " · "),
                       labelwidth = 84, widgetwidth = 124)
            # …and the lanes those rows just created get the clip's placement
            target isa Clip && placelanes!(player.timeline, target)
            return nothing
        end
        on(o -> o && buildrows!(), card.open)
        # animated sections open themselves: a scene has more objects than fit on
        # screen and all start folded, so the ones being worked on would have to
        # be hunted for
        any(isanimated, sec.params) && (card.open = true)
        (nparams <= lazyabove || card.open[]) && buildrows!()
        push!(cards, card)
    end
    colsize!(stack, 1, Makie.Relative(1.0))

    function apply()
        q = lowercase(something(box.stored_string[], ""))
        shown = 0
        filter_cards!(stack, cards) do card
            sec = secs[findfirst(c -> c === card, cards)::Int]
            keep = isempty(q) || occursin(q, lowercase(sec.label)) ||
                   occursin(q, lowercase(sec.detail)) ||
                   any(p -> occursin(q, lowercase(p.label)), sec.params)
            # a section the query singled out opens, so filtering to one object is
            # one gesture rather than two
            keep && !isempty(q) && (card.open = true)
            keep && (shown += length(sec.params))
            keep
        end
        n = isempty(q) ? length(secs) : count_shown(cards)
        count.text[] = (isempty(q) ? "$nparams parameters" : "$shown of $nparams parameters") *
                       " · $n object" * (n == 1 ? "" : "s")
        return
    end
    on(_ -> apply(), box.stored_string)
    apply()
    player.fxwidgets[Symbol(:fxsections_, fx.id)] = (; box, cards, eyes, sections = secs, apply)
    return Any[gl]
end

"How many of `cards` are currently shown — what the section count label reports."
count_shown(cards) = count(c -> c.visible[], cards)

# ------------------------------------------------------- the keyframe overview

"""
    buildkeyframemodal!(player, uicolors)

The animated-parameter overview: one entry per parameter of the selected effect,
each with its own eye for whether its lane is drawn.

A modal rather than a strip in the panel: it answers an occasional question and
needs a list's worth of room.

Rebuilt on open, so it always describes the effect you are looking at.
"""
function buildkeyframemodal!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Animated parameters", min_size = (300, 200))
    body = GridLayout(modal[1, 1])
    hint = Label(body[1, 1], ""; halign = :left, fontsize = 10,
                 color = uicolors.text_muted, tellwidth = false)
    holder = GridLayout(body[2, 1])

    function refresh()
        foreach(Makie.delete!, get!(() -> Any[], player.fxwidgets, :kflanerows))
        empty!(player.fxwidgets[:kflanerows])
        # …and what those rows derived from `Param.visible`, which outlives them.
        # The rows used to be built with `map`, so every open left one listener per
        # parameter on the clip behind.
        regs = get!(() -> Observables.ObserverFunction[], player.fxwidgets, :kflaneregs)
        foreach(Observables.off, regs)
        empty!(regs)
        sel = selectedeffect(player)
        if sel === nothing
            hint.text[] = "Select an effect card to see its parameter lanes."
            return
        end
        _, fx = sel
        nanim = count(isanimated, fx.params)
        nopen = count(p -> p.visible[], fx.params)
        hint.text[] = "$(length(fx.params)) parameter$(length(fx.params) == 1 ? "" : "s") · " *
                      "$nanim animated · $nopen lane$(nopen == 1 ? "" : "s") shown\n" *
                      "a lane can be shown before it has keyframes — that is where the first one goes"
        # one row per parameter, each with its own eye: showing a lane and
        # animating it are separate questions
        for (i, p) in enumerate(fx.params)
            row = GridLayout(holder[i, 1])
            # The same ∿ and the same gesture as the row in the card, so that
            # "show me only this one" is one thing to learn and works wherever a
            # parameter is listed.
            laneeye!(player, row[1, 1], p, uicolors, regs;
                     width = 24, buttoncolor = (:transparent, 0.0), strokewidth = 0)
            Label(row[1, 2], p.label; halign = :left, tellwidth = false,
                  color = isanimated(p) ? uicolors.text : uicolors.text_muted)
            Label(row[1, 3], isanimated(p) ? "$(length(p.curve[].keys)) keys" : "static";
                  halign = :right, fontsize = 10, color = uicolors.text_muted)
            push!(player.fxwidgets[:kflanerows], row)
        end
        return
    end

    open = () -> (refresh(); open!(modal))
    merge!(player.fxwidgets, Dict{Symbol, Any}(
        :kfmodal => modal, :kfmodalopen => open, :kfmodalrefresh => refresh))
    return open
end

"Open the animated-parameter overview (the ◆ in the Effects panel head)."
openkeyframes!(player::Player) =
    (f = get(player.fxwidgets, :kfmodalopen, nothing); f === nothing || f(); nothing)
