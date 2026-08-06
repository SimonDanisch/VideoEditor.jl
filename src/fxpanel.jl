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
    menuopts() = [(k.label, k.name) for k in addablekinds()]
    addmenu = Menu(panel[2, 1]; prompt = "+  Add effect…", default = nothing,
                   searchable = true, search_placeholder = "type to filter…",
                   options = menuopts(), tellwidth = false)
    on(EFFECTS.version) do _
        addmenu.options[] = menuopts()
    end
    on(addmenu.selection) do sel
        sel === nothing && return
        addmenu.i_selected[] = 0     # back to the prompt; re-fires with nothing
        addeffect!(player, sel)
    end

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
        "▸ clip $i · $(basename(c.source.path)) ($(timestring(c.start / fps))–$(timestring(clipend(c) / fps)))"
    end
    Label(panel[4, 1], target; halign = :left, fontsize = 11, color = uicolors.accent,
          tellwidth = false)

    # `measurable!`: a GridLayout with no content has no determinable height, and
    # ONE such row makes the whole panel indeterminable — which stretches it to
    # fill the dock (rows then share the slack, so a short panel floats in the
    # middle) AND leaves `Subfigure.contentsize` at zero, so a long list never
    # gets a scrollbar. Both symptoms, one cause.
    stackgl = measurable!(GridLayout(panel[5, 1]; valign = :top, default_rowgap = 0))

    # ◀◆▶ accessory state: ONE persistent (label, color) pair per parameter,
    # shared by every rebuild of its button and driven by a single playhead
    # listener — a `map(playhead)` inside the card builder would leak a listener
    # per rebuild.
    kfaccstate = Dict{Symbol, NamedTuple}()
    function updatekfaccs()
        loc = editclip(player)
        clip = loc === nothing ? nothing : loc[1]
        for (key, st) in kfaccstate
            anim = clip !== nothing && clipanimated(clip, key)
            onkey = anim && any(k -> k.frame == playheadframe(player, clip),
                                clip.animations[key].keys)
            lbl = onkey ? "◆" : "◇"
            col = anim ? paramcolor(key) : uicolors.text_muted
            st.label[] == lbl || (st.label[] = lbl)
            st.color[] == col || (st.color[] = col)
        end
        return
    end
    on(_ -> updatekfaccs(), player.playhead)

    # The live cards, in stack order, and the contexts their bodies built into.
    cards = Card[]
    cardkinds = EffectKind[]
    bodyctxs = EffectContext[]
    strays = Any[]        # blocks that are not cards (the empty state), to delete on rebuild
    player.fxwidgets[:fxcards] = cards
    player.fxwidgets[:fxcardkinds] = cardkinds

    "Hide the cards the query rules out, in one relayout. Never rebuilds."
    function applyfilter()
        q = query[]
        shown = 0
        filter_cards!(stackgl, cards) do card
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

    lastsig = Ref{Any}(:init)
    function rebuildstack(; force::Bool = false)
        force && (lastsig[] = :force)
        loc = editclip(player)
        clip = loc === nothing ? nothing : loc[1]
        sig = effsig(clip)
        sig == lastsig[] && return
        lastsig[] = sig

        # Emptying the layout drops `contentsize` to zero for an instant, and the
        # Subfigure clamps the scroll to fit — so adding an object, or a card
        # appearing, threw the reader back to the top of the list. Put it back
        # after; the Subfigure re-clamps if the list really did get shorter.
        keepscroll = fxscroll.scroll[]

        foreach(cleartoolcontext!, bodyctxs); empty!(bodyctxs)
        haskey(player.fxwidgets, :toolpanels) && empty!(player.fxwidgets[:toolpanels])
        # the ? tips point at buttons this rebuild is about to delete, and the
        # hover loop walks every entry — a stale one is a phantom hover target
        fxtips = get(player.fxwidgets, :fxtips, nothing)
        alltips = get(player.fxwidgets, :tips, nothing)
        if fxtips !== nothing && alltips !== nothing
            foreach(b -> delete!(alltips, b), fxtips); empty!(fxtips)
        end
        # the image cards a body added (loop references, matte marks) are plots in
        # the panel's SCENE, not in its layout — rebuilding the layout leaves them
        # drawn, floating over whatever replaced them
        cleartoolcards!(player)
        foreach(Makie.delete!, cards)
        # the empty state is blocks too — `trim!` only drops empty ROWS, so an
        # untracked Label kept drawing under the first card that replaced it
        foreach(Makie.delete!, strays)
        empty!(cards); empty!(cardkinds); empty!(strays); empty!(player.fxsliders)
        Makie.trim!(stackgl)

        if clip === nothing
            append!(strays, emptystate!(stackgl, "No clip at the playhead.",
                        "Move the playhead onto a clip to give it effects.", uicolors))
        elseif isempty(clip.effects)
            append!(strays, emptystate!(stackgl, "No effects on this clip.",
                        "Add one above, or press Ctrl+P.", uicolors))
        else
            for slot in clip.effects
                card, ctx = fxcard!(player, stackgl, length(cards) + 1, clip, slot,
                                    uicolors, kfaccstate, updatekfaccs)
                push!(cards, card)
                push!(cardkinds, effectkindfor(slot.effect))
                if ctx !== nothing
            push!(bodyctxs, ctx)
            get!(() -> Dict{Symbol, Any}(), player.fxwidgets, :toolpanels)[ctx.tool] = ctx
        end
            end
        end
        colsize!(stackgl, 1, Makie.Relative(1.0))
        applyfilter()          # a rebuild must honour the filter that is showing
        fxscroll.scroll[] = keepscroll
        # …and again once the layout has settled: `contentsize` is recomputed from
        # the layout's bbox, which may land after this call returns, and the clamp
        # that runs with it would undo the line above.
        put!(player.uiqueue, () -> (fxscroll.scroll[] = keepscroll))
        return
    end
    player.fxwidgets[:fxlistrefresh] = rebuildstack
    on(_ -> rebuildstack(), player.playhead)
    # A kind whose card body has live content (the matte's marks, its object bars,
    # its preview thumbnails) asks for a rebuild by bumping the registry version.
    # That used to be the Tools dock's cue; it is this panel's now, and losing it
    # is why the matte stopped showing anything after the first mark.
    on(_ -> rebuildstack(force = true), EFFECTS.version)
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
    build()
    return gl
end

"The panel's resting state: what is here, and what to do about it."
function emptystate!(gl, title::AbstractString, hint::AbstractString, uicolors)
    box = GridLayout(gl[1, 1]; alignmode = Makie.Outside(4, 4, 10, 10))
    return [Label(box[1, 1], title; halign = :left, fontsize = 12, color = uicolors.text,
                  tellwidth = false),
            Label(box[2, 1], hint; halign = :left, fontsize = 11, color = uicolors.text_muted,
                  tellwidth = false)]
end

"""
The signature that decides whether the card stack still describes the clip: which
slots, in which order, enabled or not, and which parameters are animated. Folding,
selecting and filtering are NOT in it — they are card attributes now, and changing
one must not throw the panel away.
"""
effsig(clip) = clip === nothing ? nothing :
    (clip.id,
     Tuple((s.id, s.enabled, nameof(typeof(s.effect)),
            s.effect isa PluginEffect ? s.effect.name : :_,
            Tuple((l.clip, l.slot, l.role) for l in s.links)) for s in clip.effects),
     Tuple(sort!(collect(keys(clip.animations)))))

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
    push!(clip.effects, FxSlot(k.make(defaults(k))))
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
    slot = findfirst(s -> k.matches !== nothing && k.matches(s.effect), clip.effects)
    slot === nothing && return false
    id = clip.effects[slot].id
    selectfxcard!(player, (:fx, id))
    i = findfirst(c -> c.title[] == k.label, player.fxwidgets[:fxcards])
    i === nothing || (player.fxwidgets[:fxcards][i].open = true)
    return true
end

"""
One card: the effect's name in the header, its enable toggle and remove ×
in the accessory, its parameters and its own body inside.
"""
function fxcard!(player::Player, stackgl, row::Integer, clip::Clip, slot::FxSlot,
                 uicolors, kfaccstate, updatekfaccs)
    kind = effectkindfor(slot.effect)
    title = kind === nothing ? string(nameof(typeof(slot.effect))) : kind.label
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
    isempty(kind.params) || (r += 1; paramform!(player, card[r, 1], clip, slot, kind,
                                                uicolors, kfaccstate, updatekfaccs))
    # Linked effects come next: the target's parameters, inline and flat, behind a
    # caption that names the card they belong to. Between the card's own
    # parameters and its action, because that is the reading order — this is what
    # I do, this is what I am tied to, this is the button.
    linkblocks = linkedgroups!(player, card[r + 1, 1], clip, slot, uicolors,
                               kfaccstate, updatekfaccs)
    isempty(linkblocks) || (r += 1)
    ctx = nothing
    if kind.body !== nothing
        ctx = EffectContext(player, kind.name)
        r += 1
        # the body builds into the CARD, next to the parameters it belongs with —
        # not into a slot in some other panel
        ctx.state = nothing
        withtoolslots!(player, ctx, card[r, 1]) do
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

The value handler receives the WHOLE form tuple on every fire, so only parameters
whose own slider moved since the last one may act — anything else is a synced echo
(scrubbing keeps animated sliders on their curves, quantized to the slider step),
and treating those as edits stamped keyframes on parameters nobody touched.
"""
function paramform!(player::Player, pos, clip::Clip, slot::FxSlot, kind::EffectKind,
                    uicolors, kfaccstate, updatekfaccs;
                    labelcolor = uicolors.text, widgetwidth = 132)
    cur0 = kind.read(slot.effect)
    fieldsym(pr) = Symbol(pr.label)
    spec = NamedTuple(fieldsym(pr) => (Float64(cur0[pr.name]), Makie.Between(pr.min, pr.max))
                      for pr in kind.params)
    accessory = (field, gp) -> begin
        key = kind.kfkeys[findfirst(pr -> fieldsym(pr) === field, kind.params)]
        st = get!(() -> (label = Observable("◇"), color = Observable{Any}(uicolors.text_muted)),
                  kfaccstate, key)
        acc = GridLayout(gp)
        prevb = Button(acc[1, 1]; label = "◀", width = 16, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        kf = Button(acc[1, 2]; label = st.label, width = 22, labelcolor = st.color)
        nextb = Button(acc[1, 3]; label = "▶", width = 16, fontsize = 8,
                       labelcolor = uicolors.text_muted)
        colgap!(acc, 1)
        on(_ -> gotokey!(player, key, -1), prevb.clicks)
        on(_ -> togglekey!(player, key), kf.clicks)
        on(_ -> gotokey!(player, key, 1), nextb.clicks)
        player.fxwidgets[Symbol(:kfacc_, key)] = (prevb, kf, nextb)
        acc
    end
    pf = Makie.ParamForm(pos, spec, accessory; labelwidth = 88, widgetwidth = widgetwidth,
                         accessorywidth = 62, rowgap = 4, halign = :left,
                         labelcolor = labelcolor)
    updatekfaccs()
    for (j, pr) in enumerate(kind.params)      # scrub-sync registry
        w = get(pf.widgets, fieldsym(pr), nothing)
        w isa Slider && (player.fxsliders[kind.kfkeys[j]] = w)
    end
    lastvals = Ref{Any}(nothing)
    on(pf.graph[:values]) do vals
        prevvals = lastvals[]; lastvals[] = vals
        player.fxsyncing[] && return           # sync: just refresh the baseline
        cur = findslot(clip, slot.id)
        cur === nothing && return
        kind.matches(cur.effect) || return
        prev = kind.read(cur.effect)
        changedstatic = NamedTuple()
        for (j, pr) in enumerate(kind.params)
            v = Float64(vals[fieldsym(pr)])
            key = kind.kfkeys[j]
            moved = prevvals === nothing ?
                abs(v - (clipanimated(clip, key) ?
                         paramvalue(clip, key, playheadframe(player, clip)) :
                         Float64(prev[pr.name]))) > 1.0e-9 :
                v != Float64(prevvals[fieldsym(pr)])
            moved || continue
            if time() - player.lastslidersnap > 1.5    # one undo entry per gesture
                snapshot!(player); player.lastslidersnap = time()
            end
            if clipanimated(clip, key)                 # animated: the slider writes a key
                setkey!(clip.animations[key], playheadframe(player, clip), v)
            else
                changedstatic = merge(changedstatic, NamedTuple{(pr.name,)}((v,)))
            end
        end
        isempty(changedstatic) || (cur.effect = kind.make(merge(prev, changedstatic)))
        player.playing[] || notify(player.playhead)
    end
    return pf
end

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
        loc = editclip(player)
        clip = loc === nothing ? nothing : loc[1]
        curves = kfcurves(player)
        keys_ = clip === nothing ? Symbol[] :
                [k for k in sort!(collect(keys(clip.animations))) if haskey(curves, k)]
        if isempty(keys_)
            hint.text[] = clip === nothing ? "No clip at the playhead." :
                "Nothing is animated on this clip yet — press a ◆ next to a parameter."
            return
        end
        nkeys = sum(length(clip.animations[k]) for k in keys_)
        hint.text[] = "$(length(keys_)) parameter$(length(keys_) == 1 ? "" : "s") · " *
                      "$nkeys keyframe$(nkeys == 1 ? "" : "s")\n" *
                      "click to hide one · right-click hides all · middle-click shows all"
        # `plot => element` keeps the plot connected (that is what the click
        # toggles) while drawing the swatch as a ◆ — the same marker, colour and
        # white stroke the keyframes have on the clip, so the list reads as the
        # thing it controls rather than as a chart legend.
        entries = [curves[k].plot => Makie.MarkerElement(marker = :diamond,
                                                         color = paramcolor(k),
                                                         markersize = 11,
                                                         strokecolor = :white,
                                                         strokewidth = 1.0)
                   for k in keys_]
        legend[] = Makie.Legend(holder[1, 1], entries,
                                [paramspec(k).label for k in keys_];
                                framevisible = false, patchsize = (18, 14), labelsize = 11,
                                rowgap = 4, padding = (4, 4, 4, 4),
                                halign = :left, valign = :top, tellwidth = false)
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
