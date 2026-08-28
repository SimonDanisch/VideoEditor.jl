# The Effects panel — ONE list of cards for the selected clip.
#
# What this replaces: an Inspector that listed "effects" and a Tools dock that
# listed "tools", with two card builders, two sets of widget helpers and a line
# between them nobody could draw. Blur was an effect; Stabilize was a tool; both
# are things you put on a clip and see in the render.
#
# Cards are `Makie.Card`s, so folding, selecting and FILTERING never rebuild
# anything: they set an attribute and the layout closes up. The stack is rebuilt
# only when the clip's set of effects changes — which is also when a rebuild is
# the honest answer, because there is a different card in it.

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

The Effects panel's filter query. Lives on the player because the palette and
MCP can set it too ("show me the color effects").
"""
fxfilter(player::Player) = player.fxwidgets[:fxquery]::Observable{String}

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
    CardSet

A CLIP'S OWN CARDS, kept so that going back to it does not build them again.

Building is Makie `Block` construction — a parameter row is a label, a slider, a
grid and three buttons, about 5 ms each — so the lego project's scene clip cost
**3.2 to 3.6 seconds every time it was selected**, measured, not once. Nothing
about those widgets goes stale in between: they are bound to a clip and an
effect, and `refreshfxrows!` derives what they show from the playhead.

So each clip gets a sub-layout of the card stack that is ITS cards, built once.
Leaving the clip hides them; coming back shows them again and swaps in the
registries the rest of the panel reads. `sig` is what the cards were built from —
[`effsig`](@ref) and [`docsig`](@ref) — so adding an effect still rebuilds, into
the same sub-layout.

**Not a cache.** There is no eviction policy and no bound, because there is
nothing to bound: a clip's cards exist exactly as long as the clip does, and the
sequence says which clips those are. Anything keyed to a clip that is gone is
dropped when the next stack is put away.

**A stack per sub-layout, never shared cells.** The first attempt let two stacks
share `stackgl`'s cells, on the reasoning that a hidden `Card` reports zero height
so only one of them has a size. True of the LAYOUT, and it says nothing about the
render objects, so it was not worth relying on.
"""
mutable struct CardSet
    const layout::GridLayout
    const slot::Int
    sig::Any
    const cards::Vector{Makie.Card}
    const kinds::Vector{Union{Nothing, EffectKind}}
    const rows::Vector{ParamRow}
    const sliders::Dict{Tuple{UInt64, Symbol}, Makie.Slider}
    const tips::Vector{Pair{Any, String}}
end

"""
Off switch for [`CardSet`](@ref): with it off the cards go straight into the
shared stack and are deleted on every change, the way they were before.

Not a tuning knob — a CONTROL. It is how "the editor segfaults when you switch
clips" was shown to be nothing to do with keeping cards: three runs each way,
three crashes each way. An off switch that still changes the structure would have
proved nothing, so this one really does take the old path.
"""
const KEEPCARDS = Ref(true)

"""
Build the Effects panel into `gridpos`.

Layout, top to bottom: the title with the keyframe-curve toggle, which clip is
being edited, "+ Add effect…", the filter box, then the card stack, then the
global before/after button. Everything below the title scrolls.
"""
function buildfxpanel!(player::Player, gridpos, uicolors)
    fxscroll = Subfigure(gridpos; scroll_speed = 70, scrollbar_size = 9,
                         scrollbar_color = (uicolors.background, 0.55),
                         scrollbar_thumb_color = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                                                  RGBf(1, 1, 1), 0.34),
                         scrollbar_thumb_color_active = uicolors.accent)
    player.fxwidgets[:fxscroll] = fxscroll
    player.fxwidgets[:uicolors] = uicolors   # tool card bodies draw in the editor's palette
    wiretoolcards!(player)
    panel = GridLayout(fxscroll[1, 1]; valign = :top)
    head = GridLayout(panel[1, 1])
    Label(head[1, 1], "Effects"; font = :bold, halign = :left, tellwidth = false)
    # The EYE bypasses the whole stack — the compare-with-the-original button
    # that used to sit at the bottom of the panel, as a 24px toggle instead of a
    # full-width bar. Same idea as the per-card eye, one level up.
    headflat = (buttoncolor = (:transparent, 0.0), strokewidth = 0, cornerradius = 4,
                buttoncolor_hover = uicolors.accent_subtle,
                buttoncolor_active = uicolors.accent, width = 26, height = 22,
                halign = :right)
    eyeb = Button(head[1, 2]; label = map(a -> a ? "◉" : "○", player.applytracks),
                  headflat...)
    on(_ -> (player.applytracks[] = !player.applytracks[]; notify(player.playhead)),
       eyeb.clicks)
    on(player.applytracks; update = true) do on_
        eyeb.buttoncolor[] = on_ ? uicolors.surface : uicolors.accent
        eyeb.labelcolor[] = on_ ? uicolors.text : uicolors.text_on_accent
    end
    tips = get(player.fxwidgets, :tips, nothing)
    tips === nothing || (tips[eyeb] = "Bypass every effect on this clip")
    # ◆ OPENS the animated-parameter overview rather than being an opaque
    # all-or-nothing toggle (Simon, 2026-07-31: "what is this button doing?").
    kfb = Button(head[1, 3]; label = "◆", headflat...)
    tips === nothing || (tips[kfb] = "Animated parameters on this clip")
    on(_ -> openkeyframes!(player), kfb.clicks)
    # the keyframe-curve overlay toggle configures the timeline overlay that the
    # cards' ◆ accessories feed, so it belongs to this panel's head

    # ---------------------------------------------------------------- add menu
    # Directly under the title: adding and finding an effect are what the panel
    # is FOR, so they come before anything describing what is already there.
    # Effects AND the tool-only kinds. The panel shows what the user PUT there and
    # nothing else, so a clip with no effects has an empty stack; a tool that is
    # not a clip effect still has to be findable, and this searchable menu is
    # where you look. Selecting one opens its card instead of adding an effect.
    menuopts() = vcat([(k.label, k.name) for k in addablekinds()],
                      [(k.label, k.name) for k in toolonlykinds()])
    # The menu is the ONLY way to reach a tool-only kind now, so what it offers is
    # a fact worth asserting on rather than reading off the widget's internals.
    player.fxwidgets[:fxmenuopts] = menuopts
    addmenu = Menu(panel[2, 1]; prompt = "+  Add effect…", default = nothing,
                   searchable = true, search_placeholder = "type to filter…",
                   options = menuopts(), tellwidth = false)
    # EFFECTS is a GLOBAL registry, so this listener outlives the player unless it
    # is taken off again — see `:fxglobalobs` below.
    menuobs = on(EFFECTS.version) do _
        addmenu.options[] = menuopts()
    end
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
    player.fxwidgets[:fxquery] = query
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

    # …and which clip this all applies to labels the stack, right above it
    target = map(player.playhead, player.timeline.selected) do n, _
        loc = editclip(player)
        loc === nothing && return "▸ no clip at the playhead"
        c = loc[1]; i = something(findfirst(x -> x === c, player.sequence.clips), 0)
        fps = player.sequence.framerate
        "▸ clip $i · $(basename(sourcepath(c.source))) ($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
    end
    Label(panel[4, 1], target; halign = :left, fontsize = 11, color = uicolors.accent,
          tellwidth = false)

    # THE BAKE, one row, for every clip. Not a scene feature: an expensive stack on
    # ordinary footage is worth pre-rendering for the same reason, so the state
    # ("no bake" / "in use" / "out of date") and the way to change it belong where
    # the clip's effects are, not in a dialog you have to know exists.
    bakerow = GridLayout(panel[5, 1])
    bakelabel = Label(bakerow[1, 1], ""; halign = :left, fontsize = 10,
                      color = uicolors.text_muted, tellwidth = false)
    bakeuse = Button(bakerow[1, 2]; label = "use", width = 42, height = 18, fontsize = 10)
    bakebtn = Button(bakerow[1, 3]; label = "Bake…", width = 58, height = 18, fontsize = 10)
    player.fxwidgets[:bakebutton] = bakebtn
    player.fxwidgets[:bakeuse] = bakeuse
    on(_ -> openbakemodal!(player), bakebtn.clicks)
    on(bakeuse.clicks) do _
        runcommand!(player, :bake_toggle)
    end
    function refreshbake()
        loc = editclip(player)
        clip = loc === nothing ? nothing : loc[1]
        b = clip === nothing ? nothing : clip.bake
        # Both buttons stay put. A `Button` has no `visible` here, and hiding one
        # would move the row's layout under the reader anyway; the LABEL says what
        # there is to do, and the command refuses with a reason when there is none.
        bakelabel.text[] =
            clip === nothing ? "" :
            b === nothing ? "no bake — the graph renders every frame" :
            !b.enabled && bakestale(clip) ?
                "bake switched off: the clip changed since · $(length(b.frames)) frames kept" :
            !b.enabled ? "bake off · $(length(b.frames)) frames on disk" :
            "bake in use · frames $(first(b.frames))–$(last(b.frames))"
        bakeuse.label[] = b === nothing ? "—" : b.enabled ? "off" : "use"
        return
    end
    player.fxwidgets[:bakerefresh] = refreshbake
    on(_ -> refreshbake(), player.playhead)
    refreshbake()

    # `measurable!`: a GridLayout with no content has no determinable height, and
    # ONE such row makes the whole panel indeterminable — which stretches it to
    # fill the dock (rows then share the slack, so a short panel floats in the
    # middle) AND leaves `Subfigure.contentsize` at zero, so a long list never
    # gets a scrollbar. Both symptoms, one cause.
    stackgl = measurable!(GridLayout(panel[6, 1]; valign = :top, default_rowgap = 0))

    # The live cards, in stack order, and the contexts their bodies built into.
    cards = Card[]
    # `nothing` is allowed: a card may have no `EffectKind` behind it.
    # `kindmatches` has a method for that; this vector being narrowed to
    # `EffectKind` is what threw when the first such card appeared.
    cardkinds = Union{Nothing, EffectKind}[]
    bodyctxs = EffectContext[]
    strays = Any[]        # blocks that are not cards (the empty state), to delete on rebuild
    player.fxwidgets[:fxcards] = cards
    player.fxwidgets[:fxcardkinds] = cardkinds

    # ------------------------------------------------------------ kept stacks
    # See [`CardSet`](@ref): one sub-layout of `stackgl` per CLIP, kept and toggled
    # rather than rebuilt. Rows come from a free list, so a dropped clip gives its
    # row back and the outer grid never grows past the clips being kept.
    stacks = Dict{Any, CardSet}()       # clip id (or `:noclip`) → its cards
    livekey = Ref{Any}(nothing)         # which clip the cards on screen belong to
    livesig = Ref{Any}(nothing)         # …and what they were built from
    livegl = Ref{Any}(nothing)          # …and the sub-layout they are built into
    liveslot = Ref(0)
    freeslots = Int[]
    lastslot = Ref(0)
    takeslot!() = isempty(freeslots) ? (lastslot[] += 1) : pop!(freeslots)
    stackkey(clip) = clip === nothing ? :noclip : clip.id

    "Hide the cards the query rules out, in one relayout. Never rebuilds."
    function applyfilter()
        q = query[]
        shown = 0
        gl = livegl[] === nothing ? stackgl : livegl[]
        filter_cards!(gl, cards) do card
            i = findfirst(c -> c === card, cards)::Int
            keep = kindmatches(cardkinds[i], q)
            keep && (shown += 1)
            keep
        end
        countlabel.text[] = isempty(q) ? "" :
            "$shown of $(length(cards)) shown · Esc clears"
        return
    end
    on(_ -> applyfilter(), query)

    "Take the tooltips of the stack that is leaving, so nothing hovers a hidden button."
    function taketips!()
        fxtips = get(player.fxwidgets, :fxtips, nothing)
        alltips = get(player.fxwidgets, :tips, nothing)
        (fxtips === nothing || alltips === nothing) && return Pair{Any, String}[]
        kept = Pair{Any, String}[b => alltips[b] for b in fxtips if haskey(alltips, b)]
        foreach(b -> delete!(alltips, b), fxtips)
        empty!(fxtips)
        return kept
    end

    "Give the live stack its own row of `stackgl` and a layout to build into."
    function newstack!()
        liveslot[] = takeslot!()
        livegl[] = GridLayout(stackgl[liveslot[], 1])
        return livegl[]
    end

    "Throw the live stack away for good, and give its row back."
    function dropstack!()
        foreach(Makie.delete!, cards)
        foreach(Makie.delete!, strays)
        gl = livegl[]
        if gl !== nothing
            gc = Makie.GridLayoutBase.gridcontent(gl)
            gc === nothing || Makie.GridLayoutBase.remove_from_gridlayout!(gc)
            push!(freeslots, liveslot[])
        end
        livegl[] = nothing; liveslot[] = 0
        return nothing
    end

    "Delete a kept clip's cards and give its row back."
    function dropset!(set::CardSet)
        foreach(Makie.delete!, set.cards)
        gc = Makie.GridLayoutBase.gridcontent(set.layout)
        gc === nothing || Makie.GridLayoutBase.remove_from_gridlayout!(gc)
        push!(freeslots, set.slot)
        return nothing
    end

    "Hide the live stack and keep it as `key`'s, so going back is a visibility flip."
    function keepstack!(key, sig)
        set = CardSet(livegl[], liveslot[], sig, copy(cards), copy(cardkinds),
                      copy(player.fxrows), copy(player.fxsliders), taketips!())
        filter_cards!(_ -> false, set.layout, cards)
        stacks[key] = set
        livegl[] = nothing; liveslot[] = 0
        # A CLIP'S CARDS LIVE AS LONG AS THE CLIP. No eviction policy, no bound:
        # the sequence says which clips exist, so the ones that no longer do are
        # simply gone, and nothing else can accumulate.
        alive = Set{Any}(c.id for c in player.sequence.clips)
        push!(alive, :noclip)
        for (k, s) in stacks
            k in alive && continue
            delete!(stacks, k)
            dropset!(s)
        end
        return nothing
    end

    "Put a filed stack back on screen."
    function restorestack!(set::CardSet)
        livegl[] = set.layout; liveslot[] = set.slot
        append!(cards, set.cards); append!(cardkinds, set.kinds)
        append!(player.fxrows, set.rows); merge!(player.fxsliders, set.sliders)
        fxtips = get(player.fxwidgets, :fxtips, nothing)
        alltips = get(player.fxwidgets, :tips, nothing)
        if fxtips !== nothing && alltips !== nothing
            for (b, txt) in set.tips
                alltips[b] = txt
                push!(fxtips, b)
            end
        end
        # …only undoing the blanket hide. Which cards the FILTER wants is
        # `applyfilter`'s call, and it runs in `finishstack!` right after.
        filter_cards!(_ -> true, set.layout, cards)
        return nothing
    end

    """
    What both routes end with: a stack is on screen, so say what it is bound to,
    redraw the lanes from that, honour the filter, and derive what the rows show.
    Shared because a RESTORED stack owes the reader exactly what a built one does.
    """
    function finishstack!(clip, keepscroll)
        # WHAT THE CARDS ARE BOUND TO — the one answer to "which clip and which
        # effects are being edited". The timeline lanes read this instead of
        # asking `editclip` themselves: a card's sliders and its keyframe lane
        # belong to the same (clip, effect), so they must not be able to disagree
        # about which one that is.
        player.fxwidgets[:fxbound] =
            clip === nothing ? nothing : (clip, Effect[fx for fx in clip.effects])
        # …and redraw the lanes from it, HERE, rather than leaving them to the
        # playhead handler: both run on a playhead move and the order between
        # them is not fixed, so the lanes would show the previous card set.
        let f = get(player.fxwidgets, :kfrefresh, nothing)
            f === nothing || f()
        end
        colsize!(stackgl, 1, Makie.Relative(1.0))
        livegl[] === nothing || colsize!(livegl[], 1, Makie.Relative(1.0))
        applyfilter()          # a rebuild must honour the filter that is showing
        # The rows show whatever value they were BUILT (or last refreshed) with.
        # Deriving them here rather than waiting for the next playhead move is what
        # makes a freshly drawn card show the frame you are on.
        refreshfxrows!(player)
        fxscroll.scroll[] = keepscroll
        # …and again once the layout has settled: `contentsize` is recomputed from
        # the layout's bbox, which may land after this call returns, and the clamp
        # that runs with it would undo the line above.
        put!(player.uiqueue, () -> (fxscroll.scroll[] = keepscroll))
        return nothing
    end

    lastsig = Ref{Any}(:init)
    function rebuildstack(; force::Bool = false)
        force && (lastsig[] = :force)
        # NOBODY IS LOOKING: skip the whole teardown and rebuild. The stack is
        # rebuilt on every clip boundary, which during playback is every cut — and
        # paying ~30 ms for cards behind a hidden dock is a stutter bought for
        # nothing. `opendock!` forces a rebuild when the panel comes back, so this
        # cannot leave stale cards on screen; `lastsig` is deliberately NOT updated
        # here, so the catch-up rebuild still sees a changed signature.
        if player.dockopen[] !== :effects && !force
            return
        end
        loc = editclip(player)
        clip = loc === nothing ? nothing : loc[1]
        sig = (effsig(clip), docsig(player.sequence, clip))
        sig == lastsig[] && return
        lastsig[] = sig

        # Emptying the layout drops `contentsize` to zero for an instant, and the
        # Subfigure clamps the scroll to fit — so adding an object, or a card
        # appearing, threw the reader back to the top of the list. Put it back
        # after; the Subfigure re-clamps if the list really did get shorter.
        keepscroll = fxscroll.scroll[]
        key = stackkey(clip)

        # THE CARDS ON SCREEN: keep them for their clip, or throw them away.
        # Keepable means the stack owns nothing outside its own cards — a tool
        # context draws into the panel's SCENE, and an empty state is not worth
        # keeping — see [`CardSet`](@ref).
        if KEEPCARDS[] && livekey[] !== nothing && !isempty(cards) && isempty(bodyctxs) && isempty(strays)
            keepstack!(livekey[], livesig[])
        else
            foreach(cleartoolcontext!, bodyctxs); empty!(bodyctxs)
            haskey(player.fxwidgets, :toolpanels) && empty!(player.fxwidgets[:toolpanels])
            # the ? tips point at buttons this rebuild is about to delete, and the
            # hover loop walks every entry — a stale one is a phantom hover target
            taketips!()
            # the image cards a body added (loop references, matte marks) are plots in
            # the panel's SCENE, not in its layout — rebuilding the layout leaves them
            # drawn, floating over whatever replaced them
            cleartoolcards!(player)
            dropstack!()
            # the shared stack only needs reclaiming when the cards went into it
            KEEPCARDS[] || Makie.trim!(stackgl)
        end
        empty!(cards); empty!(cardkinds); empty!(strays)
        # …and the rows, which are the bindings between the widgets and their
        # parameters. `paramform!` fills this again as the cards go back up; a kept
        # stack carries its own copy and puts it back.
        empty!(player.fxrows); empty!(player.fxsliders)
        livekey[] = key; livesig[] = sig

        # ALREADY BUILT FOR THIS CLIP, and still describing it? Then showing it is
        # the whole of the work.
        kept = get(stacks, key, nothing)
        if kept !== nothing
            delete!(stacks, key)
            if kept.sig == sig
                restorestack!(kept)
                finishstack!(clip, keepscroll)
                return
            end
            # its clip gained or lost an effect: the cards are wrong, the row is free
            dropset!(kept)
        end
        # OFF MEANS OFF: with the feature disabled the cards go straight into the
        # shared stack the way they always did, so the two paths can be compared on
        # the same session. An off switch that still changes the structure is not a
        # control, it is a second experiment.
        mine = KEEPCARDS[] ? newstack!() : stackgl

        # rows the empty state consumed, so the tool cards below start clear of it
        skip = 0
        # The empty state is only empty if NOTHING else is going in the stack. It
        # used to key on the clip alone, so a 3D scene's card appeared directly
        # under the words "No effects on this clip."
        if clip === nothing
            append!(strays, emptystate!(mine, 1, "No clip at the playhead.",
                        "Move the playhead onto a clip to give it effects.", uicolors))
            skip = 1
        elseif isempty(clip.effects)
            append!(strays, emptystate!(mine, 1, "No effects on this clip.",
                        "Add one above, or press Ctrl+P.", uicolors))
            skip = 1
        else
            for slot in clip.effects
                card, ctx = fxcard!(player, mine, length(cards) + 1, clip, slot, uicolors)
                push!(cards, card)
                push!(cardkinds, kindofslot(slot))
                if ctx !== nothing
                    push!(bodyctxs, ctx)
                    get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :toolpanels)[ctx.tool] = ctx
                end
            end
        end
        # …and UNDER them, the tools that are not clip effects at all — see
        # `toolonlycard!`. Outside the `isempty(clip.effects)` branch on purpose:
        # the crop and the transcript are reachable on a clip with no effects on
        # it, which is exactly when somebody is most likely to want the crop.
        if clip !== nothing
            for kind in opentools(player)
                card, ctx = toolonlycard!(player, mine, length(cards) + 1 + skip,
                                          kind, uicolors)
                push!(cards, card)
                push!(cardkinds, kind)
                push!(bodyctxs, ctx)
                get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :toolpanels)[ctx.tool] = ctx
            end
        end
        finishstack!(clip, keepscroll)
        return
    end
    player.fxwidgets[:fxlistrefresh] = rebuildstack
    on(_ -> rebuildstack(), player.playhead)
    # A kind whose card body has live content (the matte's marks, its object bars,
    # its preview thumbnails) asks for a rebuild by bumping the registry version.
    # That used to be the Tools dock's cue; it is this panel's now, and losing it
    # is why the matte stopped showing anything after the first mark.
    stackobs = on(_ -> rebuildstack(force = true), EFFECTS.version)
    # Both of these hang off a global registry, so a closed player keeps reacting
    # to every effect-registry bump for the rest of the process. That is not a
    # slow leak, it is a live fault: the dead player's handler `put!`s onto its
    # closed `uiqueue` and throws, and `notify` abandons the remaining listeners —
    # so the LIVE player's panel silently stops rebuilding, and its Matte card
    # loses the button that was about to be clicked. `close` takes them off again.
    player.fxwidgets[:fxglobalobs] = Any[menuobs, stackobs]
    rebuildstack()

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
        :fxapplyfilter => applyfilter,
        # "take me to this effect" — the Stabilize controls used to live in a
        # dock of their own, and this is what replaced knowing that
        :showkind => (name::Symbol) -> showkind!(player, name),
        :stabopen => () -> showkind!(player, :stabilize)))
    rowgap!(panel, 10)
    colsize!(panel, 1, Makie.Relative(1.0))
    return panel
end

"The scene a card's image plots are drawn into — the Effects panel's own."
contentscene(player::Player) = player.fxwidgets[:fxscroll].scene

"""
Make the tool cards clickable.

A card built by `tooladdcard!` is a picture in this panel's scroll scene plus a
frame Block — not a Button — so nothing about it takes a click by itself. The
Tools dock used to dispatch that, and it went with the dock: every loop reference
card was inert, `onclick` was registered and never called once, and the only way
to select a reference was to make a new one.

BELOW the default priority, so the × Button inside a card's own header still wins
its press; the hit test is the frame's computed bbox, topmost card first.
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
and each lands in a named slot. Those slots used to be cells of the Tools dock;
now they are cells of this card, which is the whole point — the same body code
builds into the Effects panel with nothing changed but where it points.

The slots are made `measurable!` because a GridLayout with no content has no
determinable height, and one such slot hides the height of everything above it —
including the scroll panel's content size.
"""
function withtoolslots!(build::Function, player::Player, ctx::EffectContext, gridpos)
    slots = get!(() -> (Dict{Symbol, Any}(), Dict{Symbol, Any}(), Dict{Symbol, Any}()),
                 player.fxwidgets, :toolslots)
    cards = get!(() -> (Dict{Symbol, Any}(), contentscene(player), Any[]),
                 player.fxwidgets, :toolcards)
    gl = GridLayout(gridpos)
    slots[1][ctx.tool] = measurable!(GridLayout(gl[1, 1]))
    slots[2][ctx.tool] = measurable!(GridLayout(gl[2, 1]))
    cards[1][ctx.tool] = measurable!(GridLayout(gl[3, 1]))
    # …and one BELOW the cards, for the action that acts on the whole list. "Apply
    # matte to clip" printed above the list of marked frames read as a control for
    # something further up; an action goes under what it consumes.
    slots[3][ctx.tool] = measurable!(GridLayout(gl[4, 1]))
    # The four slots sit flush. GridLayout's default rowgap is 16, and three gaps
    # between four slots is ~48 px of empty band under every tool card's header —
    # paid whether or not the slot below it holds anything, which for most tools
    # is three times out of four. The CONTENT provides its own separation (a card
    # has padding, stacked actions have their own gap), so the slots do not need
    # to. Not zero: a few pixels still reads as "these are different areas".
    rowgap!(gl, 4)
    build()
    return gl
end

"The panel's resting state: what is here, and what to do about it."
function emptystate!(gl, row::Integer, title::AbstractString, hint::AbstractString, uicolors)
    # a ROW, not a hardcoded 1: the tool-only cards share this stack, and an empty
    # state pinned to row 1 sat underneath the first of them
    box = GridLayout(gl[row, 1]; alignmode = Makie.Outside(4, 4, 10, 10))
    return [Label(box[1, 1], title; halign = :left, fontsize = 12, color = uicolors.text,
                  tellwidth = false),
            Label(box[2, 1], hint; halign = :left, fontsize = 11, color = uicolors.text_muted,
                  tellwidth = false)]
end

"""
    docsig(seq, clip) -> Tuple

The part of the rebuild signature that is NOT on the clip's effect stack.

`effsig` reads the clip and only the clip, which was right while the panel showed
clip effects and nothing else. It now also hosts the tool-only cards — Narration,
Transcript, Crop, Time interpolation — and those read the SEQUENCE. Without this
the signature never changed when they did, `rebuildstack` returned early, and a
narration line you had just typed did not appear on the card that added it.

Deliberately cheap: this runs on every playhead move. Captions hash by value
(small immutable structs), narration by its fields rather than by `hash(nar)` —
a `Narration` carries its rendered SAMPLES, and hashing a minute of audio on
every frame change would be a real cost for a summary that never needed it.
"""
docsig(seq, clip) =
    (length(seq.captions), hash(seq.captions), seq.canvas,
     Tuple((n.text, n.at, n.voice, isempty(n.samples)) for n in seq.narration),
     clip === nothing ? nothing : (clip.timeinterp, clip.rate, clip.crop),
     # …and WHETHER THE SCENE HAS BEEN BUILT. What a scene clip offers comes from
     # the realized scene (`sceneattributes`), which does not exist until the clip
     # has rendered once — and the panel is built when the Player is, which is
     # before that. Without this the card was drawn empty at startup and never
     # again, because nothing else in the signature changes when the scene
     # appears: opening the lego project showed a "Scene" card with none of its
     # seven animated parameters on it.
     scenebuilt(clip))

"""
Whether this clip's scene is realized yet — `false` for anything that is not a
scene clip, so it is a constant for every other card and cannot cause a rebuild.

`objectid` of the live scene rather than a bare `true`: switching backend or
canvas replaces it, and the rows come from the new one.
"""
scenebuilt(::Nothing) = false
scenebuilt(clip::Clip) = scenebuilt(clip.source)
scenebuilt(::ClipSource) = false
scenebuilt(src::SceneSource) = src.live === nothing ? UInt(0) : objectid(src.live)

"""
The signature that decides whether the card stack still describes the clip: which
slots, in which order, enabled or not, and which parameters are animated. Folding,
selecting and filtering are NOT in it — they are card attributes now, and changing
one must not throw the panel away.
"""
effsig(clip) = clip === nothing ? nothing :
    (clip.id,
     # `renderable` FIRST, and short-circuiting: a data entry (the `:scene`) has
     # no payload, so asking `op` for one calls a `make` that is deliberately
     # `nothing`. Its kind is its identity.
     Tuple((s.id, s.enabled,
            renderable(s) ? nameof(typeof(op(s))) : s.kind,
            renderable(s) && op(s) isa PluginEffect ? op(s).name : :_) for s in clip.effects),
     # which parameters are animated, which are DRIVEN, and which lanes are open —
     # all three change what the cards show, and none is on the effect's identity
     # above. A driven parameter has no slider, so binding one is a rebuild.
     Tuple((fx.id, Tuple(p.name for p in fx.params if isanimated(p)),
            Tuple(p.name for p in fx.params if p.input !== nothing),
            Tuple(p.name for p in fx.params if p.visible)) for fx in clip.effects))

"""
    addeffect!(player, name) -> Bool

Put effect kind `name` on the clip under the playhead, at its defaults. The one
place that adds an effect — the menu, the palette and MCP all come through here,
so they cannot drift apart.
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
    addslot!(clip, Effect(k.make(defaults(k))))
    selectfxcard!(player, (:fx, clip.effects[end].id))
    setstatus!(player, "added $(k.label) — tune it below (Ctrl+Z removes)")
    notify(player.playhead)      # rebuilds the stack + re-presents
    return true
end

"""
    showkind!(player, name) -> Bool

Bring effect kind `name` into view on the clip at the playhead: open the Effects
panel, add the effect if it is not there yet, unfold its card and select it.

The one way to say "put me where this effect is" — the palette uses it, MCP uses
it, and the tests use it instead of knowing which dock something lives in.
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
    id = clip.effects[slot].id
    selectfxcard!(player, (:fx, id))
    i = findfirst(c -> c.title[] == k.label, player.fxwidgets[:fxcards])
    i === nothing || (player.fxwidgets[:fxcards][i].open = true)
    return true
end

"""
    opentools(player) -> Vector{EffectKind}

The tool-only cards the user has actually opened, in registration order.

They used to ALL render, always. That put four cards on every clip whether or
not anyone wanted them — Simon: "for a clip without effects that should be
empty, discovery works via the searchable menu." So the panel now shows what was
put there and nothing else, and the menu is how a tool that is not a clip effect
gets found.
"""
function opentools(player::Player)
    open = get!(() -> Set{Symbol}(), player.fxwidgets, :opentools)
    return filter(k -> k.name in open, toolonlykinds())
end

"""
    opentool!(player, name) -> nothing

Put a tool-only card on the panel (the menu's answer to "add" for a kind that is
not a clip effect), and select it so it is where the eye already is.
"""
function opentool!(player::Player, name::Symbol)
    push!(get!(() -> Set{Symbol}(), player.fxwidgets, :opentools), name)
    k = kindbyname(name)
    setstatus!(player, k === nothing ? "opened $(name)" :
                       "$(k.label) — its × closes it again")
    r = get(player.fxwidgets, :fxlistrefresh, nothing)
    r === nothing || r(force = true)
    return nothing
end

"Take a tool-only card off the panel again."
function closetool!(player::Player, name::Symbol)
    delete!(get!(() -> Set{Symbol}(), player.fxwidgets, :opentools), name)
    r = get(player.fxwidgets, :fxlistrefresh, nothing)
    r === nothing || r(force = true)
    return nothing
end

"""
    toolcardopen(kind, player) -> Bool

Whether a tool-only card starts unfolded.

Open when the tool HAS something to show — narration lines, a transcript, a clip
that is actually retimed — and folded otherwise. Making these four render at all
was the fix for them being invisible; leaving all four permanently expanded on
every clip was the over-correction, and turned the panel into a scroll of things
you are mostly not using. Folded still shows the header, so nothing is hidden.

Crop is never open by default: it is a tool you reach for, not a thing you read.
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
    toolonlycard!(player, stackgl, row, kind, uicolors) -> (card, ctx)

A card for a kind that is NOT a clip effect — it has a body and no `make`.

`rebuildstack` builds its cards from `clip.effects`, which is right for
everything that IS one: Blur, Matte, Stabilize all reach the panel because their
effect sits on the clip. A kind with no `make` can never get there, so
`registertool!` — whose whole job is registering exactly that shape — produced
panels that could not appear. It went unused after the Tools dock was removed,
which is why nothing noticed until four of them were written against it.

These are the project- and sequence-level tools (the crop scope, the transcript,
the narration, the retime mode). They belong under the clip's effects, not
inside them, and they have no slot, no bypass eye and no keyframes — which is
why this is its own builder rather than another branch through `fxcard!`.
"""
function toolonlycard!(player::Player, stackgl, row::Integer, kind, uicolors)
    card = Card(stackgl[row, 1]; title = kind.label,
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
        push!(get!(() -> Any[], player.fxwidgets, :fxtips), help)
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

"Kinds the Effects panel must render on their own — a body, but no effect to hang it on."
toolonlykinds() = filter(k -> k.body !== nothing && k.make === nothing, effectkinds())

"""
The kind behind a stack entry: recognised from its PAYLOAD where it has one, and
by name where it does not.

A data entry (the `:scene`) has no payload to recognise — that is what
[`renderable`](@ref) says — so asking `effectkindfor(op(slot))` would try to build
one. Its name is the answer, and its name is what it was registered under.
"""
kindofslot(slot::Effect) =
    renderable(slot) ? effectkindfor(op(slot)) : kindbyname(slot.kind)

"""
One card: the effect's name in the header, its enable toggle and remove ×
in the accessory, its parameters and its own body inside.
"""
function fxcard!(player::Player, stackgl, row::Integer, clip::Clip, slot::Effect, uicolors)
    kind = kindofslot(slot)
    title = kind === nothing ? String(slot.kind) : kind.label
    key = (:fx, slot.id)
    card = Card(stackgl[row, 1]; title,
                selected = fxselected(player) == key,
                backgroundcolor = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                                                   RGBf(1, 1, 1), 0.075),
                headercolor = uicolors.surface,
                headercolor_selected = uicolors.select_subtle,
                strokecolor = uicolors.border,
                selectioncolor = uicolors.select,
                titlecolor = uicolors.text)
    on(_ -> selectfxcard!(player, key), card.headerclicks)

    acc = GridLayout(card_accessory(card))
    # FLAT: no fill, no stroke, just the glyph with a hover tint. The header
    # already has a background and, when selected, an outline — a button drawing
    # a third box inside that reads as clutter (Simon: "this looks bad").
    flat = (buttoncolor = (:transparent, 0.0), strokewidth = 0, cornerradius = 3,
            buttoncolor_hover = uicolors.accent_subtle,
            buttoncolor_active = uicolors.accent, height = 20)
    # WHAT IT DOES, on hover. The tool descriptions are instructions ("CLICK the
    # subject, right-click marks what is NOT it, then Enter") — real information,
    # but printed in the card they cost a third of its height and pushed the
    # controls out of sight. A ? in the title bar keeps them one hover away, on
    # every effect card that has one.
    if kind !== nothing && !isempty(kind.description)
        help = Button(acc[1, 1]; label = "?", width = 18, fontsize = 11,
                      labelcolor = uicolors.text_muted, flat...)
        tips = get(player.fxwidgets, :tips, nothing)
        tips === nothing || (tips[help] = wraptext(kind.description, 46))
        push!(get!(() -> Any[], player.fxwidgets, :fxtips), help)
    end
    # An EYE, not a toggle: the same gesture and the same glyph as the panel's
    # bypass-everything eye, so "is this applied" reads identically at both levels.
    eye = Button(acc[1, 2]; label = slot.enabled ? "◉" : "○", width = 22,
                 fontsize = 12, flat...)
    on(eye.clicks) do _           # off keeps the parameters; every render path skips it
        snapshot!(player)
        slot.enabled = !slot.enabled
        eye.label[] = slot.enabled ? "◉" : "○"
        eye.labelcolor[] = slot.enabled ? uicolors.text : uicolors.text_muted
        notify(player.playhead)
    end
    eye.labelcolor[] = slot.enabled ? uicolors.text : uicolors.text_muted
    player.fxwidgets[Symbol(:fxeye_, slot.id)] = eye
    rm = Button(acc[1, 3]; label = "×", width = 20, fontsize = 13,
                labelcolor = uicolors.text_muted, flat...)
    colgap!(acc, 2)
    on(rm.clicks) do _
        snapshot!(player)
        removeslot!(clip, slot.id)
        setstatus!(player, "removed $title (Ctrl+Z restores)")
        notify(player.playhead)
    end
    player.fxwidgets[Symbol(:fxremove_, slot.id)] = rm

    kind === nothing && return card, nothing
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
    # ASK WHAT THERE IS TO SHOW, do not assume it is already on the entry. For the
    # ten kinds that declare scalars, `paramsections` hands back exactly those. For
    # a SCENE the rows come from the realized scene and the parameters are MADE
    # from them — so a freshly inserted scene clip has an empty `params` and a card
    # full of rows, and `isempty(slot.params)` skipped the build that would have
    # created them. "Add a bar" put a clip on the timeline whose card was blank.
    secs = paramsections(clip, slot)
    isempty(secs) ||
        (r += 1; sectionform!(player, card[r, 1], clip, slot, uicolors; sections = secs))
    ctx = nothing
    # SLOTS FOR EVERY KIND ON THE STACK, body or no body. A body is one way to
    # fill them; the other is a tool that builds its cards while it runs — the
    # loop finder pushes a reference card per Find, from an analysis that ends
    # long after the panel was drawn. Without slots `toolcard!` has nowhere to
    # build and returns 0, which is a tool that reports a result nobody can see.
    if kind.body !== nothing || kind.activate !== nothing
        ctx = EffectContext(player, kind.name)
        r += 1
        # the body builds into the CARD, next to the parameters it belongs with —
        # not into a slot in some other panel
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
    # An action button ONLY for a kind with NO body of its own. A body decides
    # what its actions are and when to offer them — `mattepanel!` shows "Mark
    # subject" only while there is nothing marked yet — and a generic button
    # underneath it is worse than redundant: `activatetool!` TOGGLES, so pressing
    # it during a marking session ended the session and dropped the points.
    #
    # (The earlier rule, "unless the body registered callbacks", missed this:
    #  `mattecard!` builds plain Buttons rather than `toolaction!`s, so it
    #  registers none and looked to this test like a body with no actions.)
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
The parameter rows: a `ParamForm` with the Premiere ◀ ◆ ▶ trio per parameter.

Nothing here listens to the playhead. Each row is registered as a
[`ParamRow`](@ref) and [`refreshfxrows!`](@ref) derives what it shows — the
slider's position, the ◆'s state — from the parameter and the playhead, from the
single listener that owns that job. A row per listener is what this used to do,
and none of them were ever taken off again.

The value handler receives the WHOLE form tuple on every fire, so only parameters
whose own slider moved since the last one may act — anything else is a synced echo
(scrubbing keeps animated sliders on their curves, quantized to the slider step),
and treating those as edits stamped keyframes on parameters nobody touched.
"""
function paramform!(player::Player, pos, target, fx::Effect, uicolors;
                    params::Vector{Param} = fx.params, labels = p -> p.label,
                    labelcolor = uicolors.text, widgetwidth = 104, labelwidth = 80)
    # THE PARAMETERS OF THIS ENTRY. Not a kind's declared list resolved through a
    # global index — `fx.params` are the objects themselves, so a row writes to
    # the parameter it is drawn for and two entries of one kind cannot collide.
    # The ROW LABEL, which is not always the parameter's own: inside a section
    # that already says `torso`, a row reading `torso · Angle` spends a third of
    # the card's width saying it again. The full label is what a timeline lane
    # and the ◆ overview show, where there is no heading above it to say which
    # object it belongs to.
    fieldsym(p) = Symbol(labels(p))
    frame() = playheadframe(player, target)
    spec = NamedTuple(fieldsym(p) => (Float64(valueat(p, frame())),
                                      Makie.Between(p.range[1], p.range[2]))
                      for p in params)
    kfbuttons = Dict{Symbol, Any}()
    accessory = (field, gp) -> begin
        p = params[findfirst(q -> fieldsym(q) === field, params)]
        acc = GridLayout(gp)
        prevb = Button(acc[1, 1]; label = "◀", width = 16, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        kf = Button(acc[1, 2]; label = "◇", width = 22, labelcolor = uicolors.text_muted)
        nextb = Button(acc[1, 3]; label = "▶", width = 16, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        colgap!(acc, 1)
        # ◀ ▶ walk this parameter's keys; ◆ keys it — EXCEPT when it is driven from
        # somewhere else, where its value is not this parameter's to author. Then
        # the middle button is the one thing there is to do about that: cut the
        # edge and keep the value it was showing.
        on(_ -> gotokey!(player, target, p, -1), prevb.clicks)
        on(kf.clicks) do _
            isdriven(p) ? unbindinput!(player, target, p) : togglekey!(player, target, p)
        end
        on(_ -> gotokey!(player, target, p, 1), nextb.clicks)
        kfbuttons[fieldsym(p)] = kf
        player.fxwidgets[Symbol(:kfacc_, fx.id, :_, p.name)] = (prevb, kf, nextb)
        acc
    end
    # `width = nothing` + a flexible widget column: the row FOLLOWS THE PANEL.
    # With three fixed columns it was 25 px wider than the card that holds it, so
    # the card's scene cut the ◆ in half and the ▶ never appeared at all — the
    # keyframe trio was a duo, and nothing in the layout said so.
    # THE ROW HAS TO FIT THE CARD'S SCENE, and 80 + 104 + 56 + 2 gaps = 256 does,
    # inside the 268 px it draws into. It used to ask for 88 + 132 + 62 = 298: the
    # row kept its width, hung over the edge, and the scene cut it off mid-widget —
    # the ◆ was sliced in half and the ▶ of the keyframe trio never appeared at all.
    # Nothing in the layout reports an overflow, which is why it survived this long.
    #
    # A flexible widget column does NOT fix it: the form then fills its CELL, and
    # the cell is ten pixels wider than the scene that draws it (the card hands out
    # more than it paints — see the note in `Subfigure`). 56 is what the trio
    # measures: 16 + 22 + 16 plus two 1 px gaps.
    pf = Makie.ParamForm(pos, spec, accessory; labelwidth = labelwidth,
                         widgetwidth = widgetwidth,
                         accessorywidth = 56, rowgap = 4, halign = :left,
                         labelcolor = labelcolor)
    for p in params                             # scrub-sync registry
        w = get(pf.widgets, fieldsym(p), nothing)
        w isa Slider && (player.fxsliders[(fx.id, p.name)] = w)
        # THE BINDING. One row, registered once, refreshed from one listener.
        push!(player.fxrows,
              ParamRow(target, p, w isa Slider ? w : nothing,
                       get(kfbuttons, fieldsym(p), nothing),
                       (uicolors.accent, uicolors.text, uicolors.text_muted)))
    end
    # SEEDED, not `nothing`. A slider has 100 steps across its range, so building
    # one for a value that does not land on a step reports the nearest one — and
    # with an empty baseline the handler below read that as a gesture and wrote it
    # back. Opening a card silently edited it: a camera at 80.0 in a scene 520
    # units across came back 83.2, every parameter at once, before anyone touched
    # anything. What the form reports at construction is where the widgets START;
    # only a change from there is an edit.
    lastvals = Ref{Any}(pf.graph[:values][])
    on(pf.graph[:values]) do vals
        prevvals = lastvals[]; lastvals[] = vals
        player.fxsyncing[] && return           # sync: just refresh the baseline
        for p in params
            v = Float64(vals[fieldsym(p)])
            moved = v != Float64(prevvals[fieldsym(p)])
            moved || continue
            # A DRIVEN parameter's value is not this row's to set — it comes down
            # an edge. The refresh puts the slider straight back; dragging it is
            # how you find out the parameter is bound, and ◆ is how you unbind it.
            isdriven(p) && continue
            if time() - player.lastslidersnap > 1.5    # one undo entry per gesture
                snapshot!(player); player.lastslidersnap = time()
            end
            # animated: the slider writes a KEY at the playhead; otherwise it
            # moves the static value. Same object either way.
            isanimated(p) ? setkey!(p.curve, frame(), v) : (p.value = v)
            # …and the clip's bake no longer describes what it renders. Said at
            # the EDIT, which is the only place that knows.
            target isa Clip && bakedirty!(target)
        end
        player.playing[] || notify(player.playhead)
    end
    return pf
end

# --------------------------------------------------- parameters, in sections

"""
    paramsections(target, fx) -> Vector{NamedTuple{(:label, :detail, :params)}}

How a card GROUPS its parameters.

ONE unnamed group by default — a Blur has one parameter and a heading over it is
noise. A SCENE clip is the other case: its groups come from the scene that was
built (one per named plot), and its parameters are minted the first time they are
asked for.

LAZY, and that is the point: what a scene offers depends on the scene, so the
panel asks it when the card is opened rather than keeping a description of it
around to consult. A parameter that already exists — read from the project file,
or made when the card was last open — is reused, so its curve is never lost.
"""
function paramsections(target, fx::Effect)
    secs = sceneparamsections(target, fx)
    secs === nothing || return secs
    rest = Param[q for q in fx.params if q.range !== nothing]
    return isempty(rest) ? NamedTuple[] :
           NamedTuple[(label = "", detail = "", params = rest)]
end

"""
    sceneparamsections(target, fx) -> Vector{NamedTuple} | nothing

The sections a SCENE clip's card shows, or `nothing` when this is not one.

`nothing` rather than an empty list, so "not a scene" and "a scene with nothing in
it yet" stay different answers — the second is what you get before the clip has
rendered once, and it should read as "not built yet", not as "has no parameters".
"""
sceneparamsections(::Any, ::Effect) = nothing

function sceneparamsections(clip::Clip, fx::Effect)
    clip.source isa SceneSource || return nothing
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

MADE HERE and kept on the effect, which is what gives a keyframe somewhere to live
that outlasts the scene: a project is loaded long before anything is rendered, and
its curves have to be waiting when the scene is finally built. A parameter that
already exists is returned untouched — its curve is the edit.
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

The `i`-th parameter, guaranteed to have a range and the scene's own label —
REPLACED if it had neither, because both fields are `const`.

A project written while scenes were overlays stored a parameter's curve but no
span: there was no slider to size. Every one of the lego project's seven animated
scene parameters comes back that way, and a row without a range has no widget to
build — `paramform!` reads `p.range[1]` and dies with
`getindex(::Nothing, ::Int64)` from inside the card builder, which is how opening
that project met a "Scene" card with nothing on it.

The LABEL comes across too, for the same reason: the file stored the path
(`torso.offset[2]`) where the scene knows the name (`torso · Offset Y`), so the
seven parameters someone actually animated were the seven reading like machine
output, right next to two hundred reading like a UI.

The curve, the value and any edge come across untouched: the span and the name
are the only things missing, and the built scene is what knows both.
"""
function withspan!(fx::Effect, i::Integer, span, label::AbstractString)
    p = fx.params[i]
    p.range === nothing || return p
    fresh = Param(p.name, label, p.value; curve = p.curve, visible = p.visible,
                  range = span, input = p.input)
    fx.params[i] = fresh
    return fresh
end

"""
    sectionform!(player, gridpos, target, fx, uicolors) -> blocks

The parameter area of a card: plain rows when there is one group, and a FILTER
BOX over a list of collapsible sections when there is more than one.

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
        # A FOLDED SECTION COSTS NOTHING once the card is big enough to care. Its
        # rows are then built the first time it is opened rather than when the
        # panel is drawn: a scene's card carries one section per object and the
        # lego project has 229 parameters across them, all folded, and building
        # every widget anyway was 7 of the 9 seconds a panel rebuild took. Nothing
        # reads a row that is not on screen — `fxrows` is only walked by
        # `refreshfxrows!`, and `fxsliders` is looked up by key.
        #
        # Below `lazyabove` the whole card is built at once, because there the
        # deferral buys nothing and costs a real property: every row exists as soon
        # as the card does, which is what an ordinary effect's card has always
        # promised (a graphic's rows are asserted on directly).
        built = Ref(false)
        function buildrows!()
            built[] && return nothing
            built[] = true
            paramform!(player, card[1, 1], target, fx, uicolors;
                       params = sec.params, labels = q -> chopprefix(q.label, sec.label * " · "),
                       labelwidth = 84, widgetwidth = 124)
            # …and they show the frame the playhead is ON, same as a rebuild does
            refreshfxrows!(player)
            return nothing
        end
        on(o -> o && buildrows!(), card.open)
        # ANIMATED SECTIONS OPEN THEMSELVES. A scene has more objects than fit on
        # a screen and all of them start folded, so the ones you are actually
        # working on would be the ones you have to go and find.
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
            # A section the query singled out OPENS: filtering to one object and
            # then having to unfold it is two gestures for one intent.
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
    player.fxwidgets[Symbol(:fxsections_, fx.id)] = (; box, cards, sections = secs, apply)
    return Any[gl]
end

"How many of `cards` are currently shown — what the section count label reports."
count_shown(cards) = count(c -> c.visible[], cards)

# ------------------------------------------------------- the keyframe overview

"""
    buildkeyframemodal!(player, uicolors)

The animated-parameter overview: one entry per keyframed parameter of the clip
under the playhead, in the colour its curve is drawn in on the timeline.

A modal rather than a strip in the panel, because it answers a question you ask
occasionally ("what is animated here?") and it needs a list's worth of room —
inline it just sat between the clip line and the cards looking like neither.

The interactions are Makie's `Legend`, not ours:

  * left-click an entry — hide THAT parameter's curve and its ◆ markers
  * right-click — hide every one
  * middle-click — bring them all back

Rebuilt on open, so it always describes the clip you are looking at.
"""
function buildkeyframemodal!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Animated parameters", min_size = (300, 200))
    body = GridLayout(modal[1, 1])
    hint = Label(body[1, 1], ""; halign = :left, fontsize = 10,
                 color = uicolors.text_muted, tellwidth = false)
    holder = GridLayout(body[2, 1])
    legend = Ref{Any}(nothing)

    function refresh()
        legend[] === nothing || (Makie.delete!(legend[]); legend[] = nothing)
        foreach(Makie.delete!, get!(() -> Any[], player.fxwidgets, :kflanerows))
        empty!(player.fxwidgets[:kflanerows])
        sel = selectedeffect(player)
        if sel === nothing
            hint.text[] = "Select an effect card to see its parameter lanes."
            return
        end
        clip, fx = sel
        nanim = count(isanimated, fx.params)
        nopen = count(p -> p.visible, fx.params)
        hint.text[] = "$(length(fx.params)) parameter$(length(fx.params) == 1 ? "" : "s") · " *
                      "$nanim animated · $nopen lane$(nopen == 1 ? "" : "s") shown\n" *
                      "a lane can be shown BEFORE it has keyframes — that is where you put the first"
        # ONE ROW PER PARAMETER, each with its own eye. Showing a lane is not the
        # same question as animating it, so the two are separate controls.
        for (i, p) in enumerate(fx.params)
            row = GridLayout(holder[i, 1])
            eye = Button(row[1, 1]; label = p.visible ? "◉" : "○", width = 24,
                         buttoncolor = (:transparent, 0.0), strokewidth = 0)
            Label(row[1, 2], p.label; halign = :left, tellwidth = false,
                  color = isanimated(p) ? uicolors.text : uicolors.text_muted)
            Label(row[1, 3], isanimated(p) ? "$(length(p.curve.keys)) keys" : "static";
                  halign = :right, fontsize = 10, color = uicolors.text_muted)
            on(eye.clicks) do _
                p.visible = !p.visible
                eye.label[] = p.visible ? "◉" : "○"
                f = get(player.fxwidgets, :kfrefresh, nothing); f === nothing || f()
                notify(player.playhead)
            end
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
