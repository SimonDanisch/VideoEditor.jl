# GUI tools — premade editing operations with timeline presence. A tool draws
# HINTS on the thumb track (arbitrary plots on the timeline axis), reads clicks
# there, and acts through the same editing API the rest of the editor uses
# (snapshot!, addtransition!, trims, effects, …). Tools register live exactly
# like effect plugins, so any package or MCP session can add its own.

"""
    ToolContext

What an active tool works with: the `player`, plus bookkeeping so everything
the tool adds — overlay plots, event handlers — is removed on deactivation.
Timeline overlays draw in TIMELINE coordinates: x is seconds
([`tooltime`](@ref)), y is the 0..1 track span ([`toolband`](@ref) for a
clip's band); preview overlays draw on `player.previewaxis` in source pixels.
Record plots with [`toolplot!`](@ref) and wire observables with
[`ontool!`](@ref); `state` is tool-private scratch.
"""
mutable struct ToolContext
    const player::Player
    const tool::Symbol          # the card this context fills — its slots, its widgets
    const plots::Vector{Any}    # (parent, plot) pairs for removal
    const handlers::Vector{Any}
    const controls::Vector{Any} # menus/buttons this context put in the card body
    const callbacks::Vector{Any}# their callbacks in creation order (tests, MCP)
    const rows::Vector{Any}     # the card's LIST rows
    const tips::Vector{Any}     # blocks this context gave a hover tooltip
    state::Any
end
ToolContext(player::Player, tool::Symbol) =
    ToolContext(player, tool, Any[], Any[], Any[], Any[], Any[], Any[], nothing)

"""
Give `block` a hover tooltip while this context lives (the shared registry the
figure's hover handler reads — see `buildui`).
"""
function tooltip!(ctx::ToolContext, block, text::AbstractString)
    tips = get(ctx.player.fxwidgets, :tips, nothing)
    tips === nothing && return block
    tips[block] = String(text)
    push!(ctx.tips, block)
    return block
end

"Remove everything a context put into its card and drop its listeners."
function cleartoolcontext!(ctx::ToolContext)
    tips = get(ctx.player.fxwidgets, :tips, nothing)
    tips === nothing || foreach(b -> delete!(tips, b), ctx.tips)
    empty!(ctx.tips)
    foreach(Observables.off, ctx.handlers)
    empty!(ctx.handlers)
    for w in vcat(ctx.controls, ctx.rows)
        try
            Makie.delete!(w)
        catch
        end
    end
    empty!(ctx.controls); empty!(ctx.callbacks); empty!(ctx.rows)
    for (parent, plt) in ctx.plots
        try
            Makie.delete!(parent, plt)
        catch
        end
    end
    empty!(ctx.plots)
    return nothing
end

"""
    toolplot!(ctx, plt; parent = ctx.player.timeline.axis) -> plt

Record a plot the tool drew, so deactivation removes it — from the timeline
axis by default, or from any other `parent` (e.g. `ctx.player.previewaxis`
for hints over the video itself)."""
toolplot!(ctx::ToolContext, plt; parent = ctx.player.timeline.axis) =
    (push!(ctx.plots, (parent, plt)); plt)

"`Makie.on` with auto-`off` at tool deactivation."
function ontool!(f::Function, ctx::ToolContext, obs; priority::Integer = 25)
    h = on(f, obs; priority = priority)
    push!(ctx.handlers, h)
    return h
end

"Timeline-axis x-coordinate (seconds) of timeline frame `n`."
tooltime(player::Player, n::Integer) = n / player.sequence.framerate

"(lo, hi) y-band of `clip`'s track on the timeline axis."
toolband(player::Player, clip::Clip) = trackband(clip.track, ntracks(player.sequence))

"""
    registertool!(name, label, description; activate, deactivate = ctx -> nothing, panel)

Register a premade editing operation: `panel(ctx)` fills its card body (always,
active or not — it shows project state), `activate(ctx)` is what asking for the
operation does, and `deactivate(ctx)` is extra teardown beyond the plots and
handlers the context already tracks.

The short form of [`registereffect!`](@ref) for a kind whose card is its own body
and its own action rather than a row of sliders. There is no separate tool
registry any more: a tool IS an effect kind, and the Effects panel lists it next
to Blur.
"""
registertool!(name::Symbol, label::AbstractString, description::AbstractString;
              activate, deactivate = ctx -> nothing, panel = ctx -> nothing) =
    registereffect!(EffectKind(name, label; description, body = panel,
                               activate, deactivate, analysis = true))

"Every kind that has a card body or an action of its own. NOT a list of what is on
 screen — the Tools dock it once described is gone, and the Effects panel renders
 a kind's body inside its effect's card, or via `toolonlykinds` when it has none."
toolkinds(registry::EffectRegistry = EFFECTS) =
    filter(k -> k.activate !== nothing || k.body !== nothing, registry.kinds)

"""
Hard-wrap `text` at `cols` characters. A Makie `Label` with `word_wrap = true`
derives its HEIGHT from its wrapped width — which the layout only knows after it
has already sized the row, so a long description overflows its cell and draws over
the card header. Pre-wrapped text has a known size in both directions.
"""
function wraptext(text::AbstractString, cols::Integer = 44)
    lines = String[]
    line = ""
    for word in split(text)
        if isempty(line)
            line = word
        elseif length(line) + 1 + length(word) <= cols
            line *= " " * word
        else
            push!(lines, line); line = word
        end
    end
    isempty(line) || push!(lines, line)
    return join(lines, "\n")
end

"The `(tool, ctx)` of the active tool, or `nothing`."
activetool(player::Player) = get(player.fxwidgets, :activetool, nothing)

"Observable naming the active tool (`:none`) — the Tools panel highlights from it."
activetoolname(player::Player) =
    get!(() -> Observables.Observable{Symbol}(:none), player.fxwidgets, :activetoolname)

function deactivatetool!(player::Player)
    cur = activetool(player)
    cur === nothing && return nothing
    tool, ctx = cur
    player.fxwidgets[:activetool] = nothing
    try
        tool.deactivate(ctx)
    catch e
        @warn "tool deactivate failed" tool = tool.name exception = e
    end
    cleartoolcontext!(ctx)
    cleartoolcards!(player)
    activetoolname(player)[] = :none
    return nothing
end

"""
    activatetool!(player, name) -> ctx

Run what a card's action does. Kinds that stay ON (the loop finder draws hints
and reads clicks) hold the active slot until asked again; kinds that just DO
something (blend) act and hand it straight back — a card's body does not depend
on this either way, it is built by the kind's `body`.
"""
function activatetool!(player::Player, name::Symbol)
    cur = activetool(player)
    if cur !== nothing && cur[1].name === name
        deactivatetool!(player)
        setstatus!(player, "$(cur[1].label) off")
        return nothing
    end
    deactivatetool!(player)
    tool = kindbyname(name)
    tool === nothing && return nothing
    ctx = ToolContext(player, name)
    player.fxwidgets[:activetool] = (tool, ctx)
    activetoolname(player)[] = name
    tool.activate === nothing || tool.activate(ctx)
    return ctx
end

# The Tools dock is gone. `buildtoolspanel!` built a second card list, with its
# own fold state and its own copy of the panel machinery, into a dock beside the
# Inspector — and since both wrote `player.fxwidgets[:toolslots]`, whichever
# rebuilt last owned the slots every tool body then built into. The card bodies
# below are unchanged; the Effects panel builds them (`withtoolslots!`).

"""
Make an (initially empty) layout report a height: GridLayoutBase treats a layout
with no measurable content as indeterminate, and one such slot is enough to hide
the height of everything above it — including the dock's scrollable content size.
"""
function measurable!(gl)
    Box(gl[1, 2]; width = 0, height = 0, color = :transparent, strokewidth = 0)
    return gl
end

"""
The card slot this context fills, or `nothing`: `:controls` above the card list,
`:rows` the list itself, `:footer` below it.
"""
function toolslot(ctx::ToolContext, which::Symbol)
    slots = get(ctx.player.fxwidgets, :toolslots, nothing)
    slots === nothing && return nothing
    d = which === :rows ? slots[2] : which === :footer ? slots[3] : slots[1]
    return get(d, ctx.tool, nothing)
end

"""
    ghostbutton(colors) -> NamedTuple

The look of a SECONDARY action: outlined, transparent fill, full-strength label.

Splat it into a `Button`. It exists because `labelcolor = colors.text_muted` on a
transparent fill reads as DISABLED, and that combination was written four
separate times in this panel — "+ object", the brush's ±, a narration's render,
a repair's "go to frame" — and was wrong all four times. Outlined-not-filled is
what makes an action secondary; dimming its TEXT just makes it look broken.
"""
ghostbutton(colors) = (fontsize = 11, height = 24, tellwidth = false,
                       width = Makie.Relative(1.0), cornerradius = PILLRADIUS,
                       buttoncolor = (:transparent, 0.0), strokewidth = 1,
                       strokecolor = (colors.text, 0.45), labelcolor = colors.text,
                       buttoncolor_hover = colors.surface)

"""
    toolaction!(ctx, label, callback; footer = false)

Add an action button to this tool's card. Repeated calls STACK, in call order;
`ctx.callbacks` addresses them for tests and MCP.

`footer = true` puts it BELOW the tool's cards instead of above them — for the
action that consumes the whole list ("Apply matte to clip" under the frames that
were marked), which above the list reads as a control for something else.
"""
function toolaction!(ctx::ToolContext, label::AbstractString, callback;
                     footer::Bool = false)
    slot = toolslot(ctx, footer ? :footer : :controls)
    slot === nothing && return nothing
    push!(ctx.callbacks, callback)
    i = length(ctx.callbacks)
    # Controls stack by `ctx.controls`; the footer counts its own slot, because a
    # footer button in `ctx.controls` would advance the CONTROL row too and leave
    # a gap — and one empty row makes a layout indeterminate, which collapses the
    # whole card.
    r = footer ? count(c -> c.content isa Button, slot.content) + 1 :
                 length(ctx.controls) + 1
    b = Button(slot[r, 1]; label = String(label), tellwidth = false,
               width = Makie.Relative(1.0))
    on(_ -> (f = ctx.callbacks[i]; f === nothing || f()), b.clicks)
    # cleanup goes through `ctx.rows` for a footer so the control row count stays
    # exactly the number of controls
    push!(footer ? ctx.rows : ctx.controls, b)
    return b
end

"""
    toollabel!(ctx, text) -> Label

Add a line of text to this tool's card — `text` may be an Observable, e.g. the
result of an analysis.
"""
function toollabel!(ctx::ToolContext, text)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    # WRAPPED, like every other line in a card. An analysis result is a sentence
    # ("marking — left: subject, right: not subject, Enter: propagate"), it does
    # not fit a 300 px dock on one line, and a Label neither wraps nor truncates
    # on its own — it just draws past the card and the panel clips it mid-word.
    wrapped = text isa Observables.AbstractObservable ?
              lift(t -> wraptext(string(t)), text) : wraptext(string(text))
    l = Label(slot[length(ctx.controls) + 1, 1], wrapped; halign = :left,
              justification = :left, fontsize = 11,
              tellwidth = false, color = ctx.player.timeline.colors.text)
    push!(ctx.controls, l)
    return l
end

"""
    toolmenu!(ctx, label, options, callback; default = nothing) -> Menu

Add a labelled option menu to this tool's card — `options` are `(label, value)`
tuples (Makie's Menu needs tuples, not pairs) and `callback(value)` fires on every
pick.
"""
function toolmenu!(ctx::ToolContext, label::AbstractString, options::Vector,
                   callback; default = nothing)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    row = GridLayout(slot[length(ctx.controls) + 1, 1])
    Label(row[1, 1], String(label); halign = :left, fontsize = 11, tellwidth = false)
    m = Menu(row[1, 2]; options = options, default = default, tellwidth = false,
             width = Makie.Relative(1.0))
    on(v -> callback(v), m.selection)
    push!(ctx.controls, row)
    return m
end

"""
    toolcheckbox!(ctx, label, callback; checked = false) -> Checkbox

Add a labelled checkbox to this tool's card — an option the user sets BEFORE the
action, not another button that does something.
"""
function toolcheckbox!(ctx::ToolContext, label::AbstractString, callback;
                       checked::Bool = false)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    row = GridLayout(slot[length(ctx.controls) + 1, 1])
    cb = Checkbox(row[1, 1]; checked = checked)
    Label(row[1, 2], String(label); halign = :left, fontsize = 11, tellwidth = false)
    on(v -> callback(v), cb.checked)
    colsize!(row, 2, Makie.Auto(false, 1.0))
    push!(ctx.controls, row)
    return cb
end

"""
    toolrows!(ctx, entries)

Show this tool's LIST — one row per thing it manages (each blend, each reference),
replacing the previous list. An entry is `(caption, [(label, callback), …],
selected::Bool)`: the caption names the item, the small buttons act on THAT item,
and a selected row is drawn in the accent colour. Declarative on purpose — a tool
recomputes its entries after every change and calls this again.
"""
function toolrows!(ctx::ToolContext, entries::Vector)
    slot = toolslot(ctx, :rows)
    slot === nothing && return nothing
    for r in ctx.rows
        try
            Makie.delete!(r)
        catch
        end
    end
    empty!(ctx.rows)
    colors = ctx.player.timeline.colors
    for (k, entry) in enumerate(entries)
        caption, actions = entry[1], entry[2]
        selected = length(entry) > 2 && entry[3]
        row = GridLayout(slot[k, 1])
        Label(row[1, 1], String(caption); fontsize = 11, halign = :left, tellwidth = false,
              color = selected ? colors.accent : colors.text)
        for (j, act) in enumerate(actions)
            b = Button(row[1, 1 + j]; label = String(act[1]), width = 26, height = 22)
            on(_ -> act[2](), b.clicks)
            length(act) > 2 && tooltip!(ctx, b, act[3])   # icons need a name on hover
        end
        colsize!(row, 1, Makie.Auto(false, 1.0))   # caption takes what the buttons leave
        push!(ctx.rows, row)
    end
    return nothing
end

"""
    toolcard!(build, ctx; caption, onremove) -> card

ONE card holding a whole feature: a header row `[caption … ×]` and, under it,
whatever `build(body, blocks)` puts in the `body` layout — labels, buttons,
checkboxes, a picture. Everything the callback creates goes into `blocks` so the
next rebuild can delete it.

The card is the unit the user acts on, so the × belongs to the card and not to a
picture inside it: closing it takes the whole feature off the clip.
"""
function toolcard!(build::Function, ctx::ToolContext; caption::AbstractString = "",
                   onremove = nothing)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return 0
    cardslots, scene, cards = tc
    cardrows = get(cardslots, ctx.tool, nothing)
    cardrows === nothing && return 0
    colors = ctx.player.timeline.colors
    cardbg = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)), RGBf(1, 1, 1), 0.075)
    # `id` is GLOBAL (it addresses `cards`, which every tool shares) but the ROW
    # is per-tool, and the two are not the same number. They coincided only while
    # a single tool body rendered at a time; with the tool-only cards the panel now
    # builds four at once, so the second tool's first card landed in row 2 of its
    # own grid and left row 1 empty — and one empty row makes a nested layout
    # indeterminate, which collapses the card that should have held it.
    id = length(cards) + 1
    row = count(c -> c.tool === ctx.tool, cards) + 1
    cell = cardrows[row, 1]
    frame = Box(cell; color = cardbg, strokecolor = colors.border, strokewidth = 1,
                cornerradius = 6, tellwidth = false, tellheight = false)
    # The header spans the card EDGE TO EDGE — the outer layout carries no padding
    # and the two rows bring their own. Inset by the body's margin it read as a
    # label floating inside the card rather than as the card's own title bar, which
    # is what the effect cards above it look like.
    g = GridLayout(cell)
    blocks = Any[]
    push!(blocks, Box(g[1, 1]; color = colors.surface, strokewidth = 0, cornerradius = 5,
                      tellwidth = false, tellheight = false))
    head = GridLayout(g[1, 1]; alignmode = Makie.Outside(8, 6, 4, 4))
    lbl = Label(head[1, 1], String(caption); fontsize = 11, halign = :left,
                color = (colors.text, 0.75), tellwidth = false)
    rm = nothing
    if onremove !== nothing
        rm = Button(head[1, 2]; label = "×", width = 22, height = 20)
        on(_ -> onremove(id), rm.clicks)
    end
    # Registered BEFORE the content is built, and `blocks` is the same vector the
    # builder appends to: a build that throws (or one superseded while an async
    # preview lands) then still leaves everything it made in the teardown list.
    # Orphaned scene plots do not just leak — they keep drawing, stacked over the
    # card that replaced them.
    # `tool` LAST. The note above says this list is read by field — and it mostly
    # is, but not everywhere: the suite indexes it positionally (`c[5](c[1])`), so
    # inserting a field in the middle turned `onclick` into a `Label` and every
    # click on a tool card threw. Appending is the only safe place to grow it.
    entry = (; id, frame, im = nothing, lbl, onclick = nothing, rm, box = nothing,
             blocks, tool = ctx.tool)
    push!(cards, entry)
    build(GridLayout(g[2, 1]; alignmode = Makie.Outside(8, 8, 6, 8)), blocks)
    return entry
end

"""
    tooladdcard!(ctx, img; caption = "", onclick = nothing, onremove = nothing) -> id

Append a preview card (image + caption) to the active tool's card area in its
Tools-panel section — e.g. one loop reference frame. `onclick(id)` fires when
the card is clicked; passing `onremove` adds an × next to the caption that
fires `onremove(id)`. Highlight the selected card with
[`toolhighlight!`](@ref); all cards are removed at deactivation.
"""
function tooladdcard!(ctx::ToolContext, img::AbstractMatrix{RGB{N0f8}};
                      caption::AbstractString = "", onclick = nothing,
                      onremove = nothing)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return 0
    cardslots, scene, cards = tc
    cardrows = get(cardslots, ctx.tool, nothing)   # THIS tool's card area
    cardrows === nothing && return 0
    colors = ctx.player.timeline.colors
    id = length(cards) + 1
    # PER TOOL, like `toolcard!` — `id` addresses the shared `cards` list, not a
    # row in this tool's grid. Two cards per entry here (header + picture).
    k = 2 * count(c -> c.tool === ctx.tool, cards) + 1
    # A CARD, visibly: one framed container holding a header row [caption … ×] and
    # the picture under it — the same section look the tool cards themselves use, so
    # a reference reads as one object instead of a loose label next to an image.
    cardbg = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)), RGBf(1, 1, 1), 0.075)
    frame = Box(cardrows[k:(k + 1), 1]; color = cardbg, strokecolor = colors.border,
                strokewidth = 1, cornerradius = 6, tellwidth = false, tellheight = false)
    Box(cardrows[k, 1]; color = colors.surface, strokewidth = 0, cornerradius = 5,
        tellwidth = false, tellheight = false)
    head = GridLayout(cardrows[k, 1]; alignmode = Makie.Outside(8, 6, 4, 4))
    lbl = Label(head[1, 1], String(caption); fontsize = 11, halign = :left,
                color = (colors.text, 0.75), tellwidth = false)
    rm = nothing
    if onremove !== nothing
        rm = Button(head[1, 2]; label = "×", width = 22, height = 20)
        on(_ -> onremove(id), rm.clicks)
    end
    box = Box(cardrows[k + 1, 1]; height = 96, tellwidth = false,
              width = Makie.Relative(1.0), color = colors.surface,
              strokecolor = (:black, 0.0), strokewidth = 0, cornerradius = 3,
              alignmode = Makie.Outside(6, 6, 2, 6))
    imgobs = Observable(reverse(collect(img), dims = 2))   # dock scene is y-up
    xy = lift(box.layoutobservables.computedbbox, scene.viewport) do bb, vp
        all(isfinite, bb.origin) && all(isfinite, bb.widths) || return (0.0, 1.0, 0.0, 1.0)
        (bb.origin[1] - vp.origin[1] + 2, bb.origin[1] + bb.widths[1] - vp.origin[1] - 2,
         bb.origin[2] - vp.origin[2] + 2, bb.origin[2] + bb.widths[2] - vp.origin[2] - 2)
    end
    im = image!(scene, lift(v -> (v[1], v[2]), xy), lift(v -> (v[3], v[4]), xy), imgobs;
                space = :pixel, interpolate = true,
                visible = lift(d -> d === :effects, ctx.player.dockopen))
    translate!(im, 0, 0, 20)
    # NAMED, like the entry `toolcard!` registers: everything that reads this list
    # reads it by field (`c.onclick`, `c.frame`, `c.blocks`), so a positional tuple
    # here made every click on the tools panel throw a `FieldError` instead.
    # `blocks` is empty because the teardown loop already names every block this
    # card made (frame, box, lbl, im, rm) — but the field has to BE there: the
    # list is read by field, so a positional tuple made every click on the tools
    # panel throw a `FieldError` instead of selecting a card.
    push!(cards, (; id, frame, im, lbl, onclick, rm, box, blocks = Any[], tool = ctx.tool))
    return id
end

"Accent-outline card `id`, resetting the others (the selected loop reference)."
function toolhighlight!(ctx::ToolContext, id::Integer)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return nothing
    colors = ctx.player.timeline.colors
    for (cid, frame, _, _, _, _, _) in tc[3]
        frame.strokecolor[] = cid == id ? colors.accent : colors.border
        frame.strokewidth[] = cid == id ? 2 : 1
    end
    return nothing
end

"Remove all tool cards (kept separate from the action button so a tool can
re-render its card list in place)."
function cleartoolcards!(player::Player)
    tc = get(player.fxwidgets, :toolcards, nothing)
    tc === nothing && return nothing
    _, scene, cards = tc
    for c in cards
        for b in (c.frame, c.box, c.lbl, c.im, c.rm, c.blocks...)
            b === nothing && continue
            try
                b isa Makie.AbstractPlot ? Makie.delete!(scene, b) : Makie.delete!(b)
            catch
            end
        end
    end
    # …and a SWEEP. Every card picture is a plot in this scene and every one of
    # them is recreated by the rebuild that follows, so anything still here is an
    # orphan: a card built but never registered, or one whose blocks list this
    # loop did not know about. An orphan does not merely leak — it keeps drawing,
    # at its old rectangle, stacked over the card that replaced it.
    for pl in copy(scene.plots)
        pl isa Makie.Image && (try Makie.delete!(scene, pl) catch end)
    end
    empty!(cards)
    return nothing
end

"Remove all tool cards and the action button (deactivation cleanup)."
function cleartoolpanel!(player::Player)
    cur = activetool(player)
    cur === nothing || cleartoolcontext!(cur[2])
    cleartoolcards!(player)
    return nothing
end

# ----------------------------------------------------------------- Loop tool

"Screen-space nearest hint marker to axis position (t, y) within ~14 px, else 0."
function nearesttoolpoint(player::Player, pts::Vector{Point2f}, t, y)
    isempty(pts) && return 0
    ax = player.timeline.axis
    vp = ax.scene.viewport[]
    (x0, x1) = player.timeline.viewrange[]
    sx = (x1 - x0) / max(vp.widths[1], 1)
    sy = 1.0 / max(vp.widths[2], 1)
    best = 0
    bestd = 14.0
    for (i, pt) in enumerate(pts)
        d = hypot((t - pt[1]) / sx, (y - pt[2]) / sy)
        d < bestd && ((best, bestd) = (i, d))
    end
    return best
end

"Draw the ACTIVE reference card's markers on the thumb track."
function showloopmarkers!(ctx::ToolContext)
    st = ctx.state
    player = ctx.player
    a = st[:active][]
    if a == 0 || a > length(st[:refs])
        st[:hintpts][] = Point2f[]; st[:refpt][] = Point2f[]
        return nothing
    end
    clip, ref, hints = st[:refs][a]
    lo, hi = toolband(player, clip)
    yh = hi - 0.12 * (hi - lo)
    accent = RGBAf(Makie.to_color(player.timeline.colors.accent))
    st[:hintpts][] = [Point2f(tooltime(player, clip.start + h.frame - 1), yh) for h in hints]
    st[:hintcols][] = [RGBAf(accent.r, accent.g, accent.b, k == 1 ? 1.0 : 0.55)
                       for k in eachindex(hints)]
    st[:refpt][] = [Point2f(tooltime(player, clip.start + ref - 1), yh)]
    return nothing
end

"▼ click: CUT the timeline at that hint (undoable) — the tool stays active."
function loopcutat!(ctx::ToolContext, k::Integer)
    player = ctx.player
    st = ctx.state
    a = st[:active][]
    (a == 0 || a > length(st[:refs])) && return nothing
    clip, _, hints = st[:refs][a]
    1 <= k <= length(hints) || return nothing
    tframe = clip.start + hints[k].frame - 1
    snapshot!(player)
    if split!(player.sequence, tframe) === nothing
        pop!(player.undostack)
        setstatus!(player, "Loop finder: already a cut at that frame")
    else
        seek!(player, tframe)
        setstatus!(player, "Loop finder: cut at the hint — Ctrl+Z undoes, Esc puts the tool away")
        refreshedit!(player)
    end
    return nothing
end

"One reference card in the Tools panel (image from the thumb track, click
selects, × removes)."
function loopcard!(ctx::ToolContext, clip::Clip, ref::Integer)
    player = ctx.player
    st = ctx.state
    cache = cachefor(player.timeline, clip.source)   # same imagery as the thumb track
    sec = round(Int, (clip.src_in + ref - 1) / clip.source.framerate)
    th = nearestthumb(cache, sec)
    fill3 = RGB{N0f8}(player.timeline.colors.surface)
    img = th === nothing ? fill(fill3, 16, 9) : fitbox(th, 240, 92, fill3)
    return tooladdcard!(ctx, img;
        caption = "ref " * timecode(player.sequence, clip.start + ref - 1),
        onclick = cid -> begin      # clicking a card shows ITS similarity markers
            st[:active][] = cid
            toolhighlight!(ctx, cid)
            showloopmarkers!(ctx)
        end,
        onremove = cid -> removeloopref!(ctx, cid))
end

"Re-render every reference card from `refs` (after a removal) and select `select`."
function refreshloopcards!(ctx::ToolContext; select::Integer = length(ctx.state[:refs]))
    st = ctx.state
    cleartoolcards!(ctx.player)
    for (clip, ref, _) in st[:refs]
        loopcard!(ctx, clip, ref)
    end
    st[:active][] = clamp(select, 0, length(st[:refs]))
    st[:active][] == 0 || toolhighlight!(ctx, st[:active][])
    showloopmarkers!(ctx)
    return nothing
end

"× on a reference card: drop that reference and re-render the remaining cards."
function removeloopref!(ctx::ToolContext, id::Integer)
    st = ctx.state
    1 <= id <= length(st[:refs]) || return nothing
    deleteat!(st[:refs], id)
    refreshloopcards!(ctx)
    setstatus!(ctx.player, "Loop finder: reference removed" *
                           (isempty(st[:refs]) ? " — Find adds a new one" : ""))
    return nothing
end

"Card + hints for one reference frame (cheap — the clip's signatures exist)."
function addloopref!(ctx::ToolContext, clip::Clip, ref::Integer, sig)
    player = ctx.player
    st = ctx.state
    fps = player.sequence.framerate
    hints = similarframes(sig, clamp(ref, 1, size(sig, 3));
                          n = st[:nhints], exclude = max(round(Int, fps), 2))
    push!(st[:refs], (clip, Int(ref), hints))
    id = loopcard!(ctx, clip, ref)
    st[:active][] = id
    toolhighlight!(ctx, id)
    showloopmarkers!(ctx)
    setstatus!(player, "Loop finder: ▼ marks frames similar to the selected card — " *
                       "click one to cut there, Find adds another reference")
    return nothing
end

"The Find action: a NEW reference = the frame under the playhead right now.
Signatures are computed once per clip; further references rescore instantly."
function loopfind!(ctx::ToolContext)
    player = ctx.player
    st = ctx.state
    loc = locate(player.sequence, player.playhead[])
    if loc === nothing
        setstatus!(player, "Loop finder: put the playhead on a clip first")
        return nothing
    end
    clip, srcframe = loc
    ref = clamp(srcframe - clip.src_in + 1, 1, srclength(clip))
    sigs = st[:sigs]
    if haskey(sigs, clip)
        addloopref!(ctx, clip, ref, sigs[clip])
        return nothing
    end
    setstatus!(player, "Loop finder: analyzing $(srclength(clip)) frames…")
    player.jobprogress[] = 0.0
    job = () -> try
        sig = loopsignatures(clip; backend = player.analysisbackend,
                             progress = (d, t) -> (player.jobprogress[] = d / max(t, 1)))
        put!(player.uiqueue, () -> begin
            cur = activetool(player)                   # put away / switched while analyzing?
            (cur === nothing || cur[2] !== ctx) && return
            sigs[clip] = sig
            addloopref!(ctx, clip, ref, sig)
        end)
    catch e
        setstatus!(player, "Loop finder failed: $(sprint(showerror, e))")
        @error "loop signature analysis failed" exception = (e, catch_backtrace())
    finally
        player.jobprogress[] = NaN
    end
    runanalysis(job, player)
    return nothing
end

function activateloopfinder!(ctx::ToolContext)
    player = ctx.player
    ax = player.timeline.axis
    st = Dict{Symbol, Any}(:nhints => 6,
                           :sigs => IdDict{Clip, Any}(),
                           :refs => Any[],            # (clip, ref, hints) per card
                           :active => Ref(0),
                           :hintpts => Observable(Point2f[]),
                           :hintcols => Observable(RGBAf[]),
                           :refpt => Observable(Point2f[]))
    ctx.state = st
    hp = scatter!(ax, st[:hintpts]; color = st[:hintcols], marker = :dtriangle,
                  markersize = 16, strokecolor = :black, strokewidth = 1)
    rp = scatter!(ax, st[:refpt]; color = :white, marker = :utriangle,
                  markersize = 13, strokecolor = :black, strokewidth = 1)
    foreach(p -> translate!(p, 0, 0, 6), (hp, rp))   # above thumbs, below playhead
    toolplot!(ctx, hp)
    toolplot!(ctx, rp)
    toolaction!(ctx, "Find similar frames", () -> loopfind!(ctx))
    ontool!(ctx, events(ax.scene).mousebutton) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        Makie.is_mouseinside(ax.scene) || return Consume(false)
        t, y = mouseposition(ax.scene)
        k = nearesttoolpoint(player, st[:hintpts][], t, y)
        k == 0 && return Consume(false)   # elsewhere: scrub/select as usual
        loopcutat!(ctx, k)
        return Consume(true)
    end
    loopfind!(ctx)   # first reference: the frame selected when pressing the button
    return nothing
end

# ---------------------------------------------------------------- Blend tool

"""
    opacityparam(clip) -> Union{Nothing, Param}

The `Param` behind the clip's opacity, or `nothing` when it has no opacity
effect. The fade helpers below reach for THIS rather than a clip-level curve
keyed `:opacity`: the curve belongs to the parameter of the effect that renders
it, so there is nothing to look up and nothing that can point at the wrong entry.
"""
function opacityparam(clip::Clip)
    fx = findslot(clip, OpacityEffect)
    return fx === nothing ? nothing : param(fx, :opacity)
end

"""
    keyfade!(clip, frames, dir) -> clip

Key `clip`'s opacity as a fade over `frames` at its START (`dir = :in`, 0 → 1) or
its END (`dir = :out`, 1 → 0), replacing whatever fade sat there before. The
matching `OpacityEffect` is added when missing, so the fade shows up in the
inspector's effect list like every other effect — its slider is the static value
the curve animates, and the ◆ editor reshapes the curve.
"""
function keyfade!(clip::Clip, frames::Integer, dir::Symbol)
    clearfade!(clip, dir)
    findslot(clip, OpacityEffect) === nothing && seteffect!(clip, OpacityEffect(1.0f0))
    prm = opacityparam(clip)
    prm.curve === nothing && (prm.curve = AnimCurve{typeof(prm.value)}())
    prm.visible = true
    curve = prm.curve
    # callers count the fade in TIMELINE frames (a length in seconds off the
    # sequence rate); keys live on SOURCE frames. On a conformed clip those are
    # different counts, and a 0.6 s blend would otherwise last 0.3 s or 1.2 s.
    sf = max(round(Int, Int(frames) * clip.rate), 1)
    if dir === :out
        setkey!(curve, clip.src_out - sf, 1.0)
        setkey!(curve, clip.src_out - 1, 0.0)
    else
        setkey!(curve, clip.src_in, 0.0)
        setkey!(curve, clip.src_in + sf - 1, 1.0)
    end
    return clip
end

"""
    clearfade!(clip, dir) -> clip

Drop the opacity keys in `clip`'s fade zone — the half at its start (`:in`) or at
its end (`:out`). Keys the user placed elsewhere on the clip survive; when none
are left the curve and its (now pointless) `OpacityEffect` go too.
"""
function clearfade!(clip::Clip, dir::Symbol)
    prm = opacityparam(clip)
    prm === nothing && return clip
    c = prm.curve
    c === nothing && return clip
    half = max(srclength(clip) ÷ 2, 1)   # the fade zone is keyed in SOURCE frames
    zone = dir === :out ? ((clip.src_out - half):clip.src_out) :
           (clip.src_in:(clip.src_in + half))
    filter!(k -> !(k.frame in zone), c.keys)
    if isempty(c.keys)
        prm.curve = nothing
        slot = findslot(clip, OpacityEffect)
        slot === nothing || removeslot!(clip, slot.id)
    end
    # NB: the pairing is NOT cleared here — `keyfade!` clears before it re-keys, and
    # changing a blend's length must not forget which clip it blends from. Removing
    # a blend clears it explicitly (the × action).
    return clip
end

"""
    fadeinlength(clip) -> Int

Length in frames of the fade-IN keyed at `clip`'s start (0 at the first frame,
rising to full), or 0 when there is none. This — nothing else — is what makes a
clip the incoming side of a blend, so the blend list is DERIVED from the keys
instead of tracked beside them.
"""
function fadeinlength(clip::Clip)
    prm = opacityparam(clip)
    c = prm === nothing ? nothing : prm.curve
    c === nothing && return 0
    ks = c.keys
    length(ks) >= 2 && ks[1].frame == clip.src_in && ks[1].value < 0.02 || return 0
    j = findfirst(k -> k.value > 0.98, ks)
    j === nothing && return 0
    # in TIMELINE frames — the same unit [`keyfade!`](@ref) took, so a blend
    # reads back the length it was given
    return timelineframes(clip, ks[j].frame - clip.src_in + 1)
end

"""
    blends(seq) -> Vector{Tuple{Int, Int, Int}}

Every blend in the sequence as `(from, into, frames)` clip indices: a clip that
fades in at its start, paired with the clip it blends away from — the one that was
MARKED when the blend was made, remembered by id (`Clip.blendfrom`), `0` when that
clip is gone. Nothing is inferred from positions, so moving, sorting, undo and a
project reload all keep the pair.
"""
function blends(seq::Sequence)
    out = Tuple{Int, Int, Int}[]
    for (j, c) in enumerate(seq.clips)
        d = fadeinlength(c)
        d == 0 && continue
        from = c.blendfrom == 0 ? nothing : findfirst(o -> o.id == c.blendfrom, seq.clips)
        push!(out, (something(from, 0), j, d))
    end
    return out
end

"""
    blendclips!(player, a, b; seconds = 0.6) -> Bool

Blend two clips by keying opacity on the LATER one: it fades in over its first
`seconds` while the earlier clip stays fully opaque underneath. One clip, one
curve — fading BOTH would darken the middle of the blend, because the outgoing
clip fades against black (nothing is below it) while the incoming one only
partly covers it (measured: about half the brightness mid-blend).

Nothing else is created: no transition object, no special render path. Overlap
the clips ([`movetooverlap!`](@ref)) and the fade-in IS the cross-dissolve; end
to end it is a soft cut-in. Either way it stays an ordinary animated parameter —
◆ on its clip, an Opacity effect in the inspector.
"""
function blendclips!(player::Player, a::Clip, b::Clip; seconds::Real = 0.6,
                     fit::Bool = false)
    seq = player.sequence
    fps = seq.framerate
    left, right = a.start <= b.start ? (a, b) : (b, a)
    d = max(min(round(Int, seconds * fps), cliplength(right) ÷ 2), 2)
    snapshot!(player)
    right.blendfrom = left.id      # the pair YOU marked, remembered by id
    keyfade!(right, d, :in)
    fit && fitoverlap!(player, left, right, d)   # slide them together in ONE step
    refreshedit!(player)
    seek!(player, clamp(right.start + d ÷ 2, 0, max(seqlength(seq) - 1, 0)))
    overlapping = right.start < clipend(left)
    setstatus!(player, "blend keyed: $(round(d / fps, digits = 1))s fade-in on the later clip " *
                       (overlapping ? "— the clips overlap, so it cross-dissolves" :
                        "— tick “move clips to fit” or ⇄ to make it cross-dissolve") *
                       " · ◆ shapes it, Ctrl+Z undoes")
    return true
end

"""
    movetotrack!(player, clip, track) -> Bool

Lift `clip` onto `track` (clips on different tracks may overlap in time). The
sequence grows a lane when needed; nothing moves in time.
"""
function movetotrack!(player::Player, clip::Clip, track::Integer)
    track >= 1 || return false
    snapshot!(player)
    clip.track = track
    refreshedit!(player)
    setstatus!(player, "clip moved to V$track (Ctrl+Z undoes)")
    return true
end

"""
    fitoverlap!(player, from, into, frames)

Place `into` so it overlaps `from` by exactly `frames` — the arrangement in which
its fade-in IS a cross-dissolve. Lifts a lane when the two share one (a lane holds
one clip at a time). No snapshot of its own: callers own the undo step.
"""
function fitoverlap!(player::Player, from::Clip, into::Clip, frames::Integer)
    seq = player.sequence
    newstart = max(clipend(from) - frames, from.start)
    canplace(seq, into, newstart) || (into.track = maximum(c.track for c in seq.clips) + 1)
    into.start = newstart
    sort!(seq.clips, by = c -> c.start)
    return nothing
end

"""
    movetooverlap!(player, a, b, frames) -> Bool

Slide the later clip back so it overlaps the earlier one by `frames` — what turns
its fade-in into a cross-dissolve. Same-track neighbours cannot overlap, so the
later clip is lifted one lane first; everything after it keeps its place (the
move is a slide, not a ripple).
"""
function movetooverlap!(player::Player, a::Clip, b::Clip, frames::Integer)
    seq = player.sequence
    left, right = a.start <= b.start ? (a, b) : (b, a)
    snapshot!(player)
    fitoverlap!(player, left, right, frames)
    refreshedit!(player)
    setstatus!(player, "clips overlap by $(round(frames / seq.framerate, digits = 1))s " *
                       "on V$(right.track) — the fade now cross-dissolves (Ctrl+Z undoes)")
    return true
end

"""
    toolslider!(ctx, label, range, callback; startvalue) -> Slider

Add a labelled slider to this tool's card, with a live readout of its value.
`callback(value)` fires while dragging — a slider (not a menu of canned values)
so a length is whatever the cut needs, and so the control can SHOW the value the
project actually has.
"""
function toolslider!(ctx::ToolContext, label::AbstractString, range, callback;
                     startvalue = first(range), format = string)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    row = GridLayout(slot[length(ctx.controls) + 1, 1])
    Label(row[1, 1], String(label); halign = :left, fontsize = 11, tellwidth = false)
    sl = Slider(row[1, 2]; range = range, startvalue = startvalue, tellwidth = false,
                width = Makie.Relative(1.0))
    Label(row[1, 3], lift(format, sl.value); halign = :right, fontsize = 11, width = 42)
    on(v -> callback(v), sl.value)
    colsize!(row, 1, Makie.Auto(false, 1.0))
    push!(ctx.controls, row)
    return sl
end

"""
The blend length the card currently offers, in seconds (its length slider).
"""
function blendseconds(player::Player)
    panels = get(player.fxwidgets, :toolpanels, nothing)
    panels === nothing && return 0.6
    ctx = get(panels, :blend, nothing)
    (ctx === nothing || !(ctx.state isa Dict)) && return 0.6
    return get(ctx.state, :seconds, 0.6)
end

"Whether the card's \"move clips to fit\" option is ticked."
function blendfit(player::Player)
    panels = get(player.fxwidgets, :toolpanels, nothing)
    panels === nothing && return false
    ctx = get(panels, :blend, nothing)
    (ctx === nothing || !(ctx.state isa Dict)) && return false
    return get(ctx.state, :fit, false)
end

"The two marked clips as `(index, index)`, or `nothing` when the marks aren't a pair."
function markedpair(player::Player)
    sel = player.timeline.selection[]
    n = length(player.sequence.clips)
    length(sel) == 2 && all(i -> 1 <= i <= n, sel) || return nothing
    a, b = sel[1], sel[2]
    return player.sequence.clips[a].start <= player.sequence.clips[b].start ? (a, b) : (b, a)
end

"Mark `i` and `j` — the blend they form becomes the selected row, and both clips
light up on the timeline."
markpair!(player::Player, i::Integer, j::Integer) =
    (player.timeline.selection[] = Int[i, j]; player.timeline.selected[] = j; nothing)

"""
Fill the Blend card: the length control plus ONE ROW PER BLEND in the sequence.
The card shows PROJECT state, so it is built whether or not the tool is on and
refreshes itself on every edit; the row whose two clips are marked is the selected
one, and the length control reads and edits exactly that blend.
"""
function blendpanel!(ctx::ToolContext)
    player = ctx.player
    ctx.state = Dict{Symbol, Any}(:seconds => 0.6, :syncing => false, :fit => true)
    toolcheckbox!(ctx, "move clips to fit the blend",
                  v -> (ctx.state[:fit] = v;
                        setstatus!(player, v ? "blends will slide the clips together" :
                                               "blends leave the clips where they are"));
                  checked = true)
    ctx.state[:slider] = toolslider!(ctx, "Length", 0.1:0.1:3.0,
                                     v -> ctx.state[:syncing] || setblendlength!(ctx, v);
                                     startvalue = 0.6,
                                     format = v -> "$(round(v, digits = 1)) s")
    ontool!(ctx, player.playhead; priority = 0) do _        # every edit notifies it
        refreshblendlist!(ctx)
    end
    ontool!(ctx, player.timeline.selection; priority = 0) do _
        refreshblendlist!(ctx)
    end
    refreshblendlist!(ctx)
    return nothing
end

"""
Re-key the SELECTED blend to `seconds` — live while dragging the slider — or, when
nothing is selected, remember the length for the next blend. With "move clips to
fit" ticked the overlap follows the new length, so the cross-dissolve stays exactly
as long as the fade.
"""
function setblendlength!(ctx::ToolContext, seconds::Real)
    player = ctx.player
    seq = player.sequence
    ctx.state[:seconds] = seconds
    b = selectedblend(player)
    if b === nothing
        setstatus!(player, "blend length: $(round(seconds, digits = 1))s — the next blend uses it")
        return nothing
    end
    i, j, _ = b
    into = seq.clips[j]
    frames = max(min(round(Int, seconds * seq.framerate), cliplength(into) ÷ 2), 2)
    snapshot!(player)
    keyfade!(into, frames, :in)
    i == 0 || !ctx.state[:fit] || fitoverlap!(player, seq.clips[i], into, frames)
    refreshedit!(player)
    setstatus!(player, "blend: $(round(frames / seq.framerate, digits = 1))s")
    refreshblendlist!(ctx)
    return nothing
end

"""
    selectedblend(player) -> (from, into, frames) | nothing

The blend the card's controls act on: the one whose two clips are marked, or — for
the common "click the clip, change its length" — the blend on the single selected
clip.
"""
function selectedblend(player::Player)
    seq = player.sequence
    bs = blends(seq)
    marked = markedpair(player)
    if marked !== nothing
        i = findfirst(b -> (b[1], b[2]) == marked, bs)
        i === nothing || return bs[i]
    end
    sel = player.timeline.selected[]
    1 <= sel <= length(seq.clips) || return nothing
    i = findfirst(b -> b[2] == sel, bs)
    return i === nothing ? nothing : bs[i]
end

"Rebuild the Blend card's list from the sequence's fade keys."
function refreshblendlist!(ctx::ToolContext)
    player = ctx.player
    seq = player.sequence
    fps = seq.framerate
    marked = markedpair(player)
    bs = blends(seq)
    # rebuild ONLY when the list really changed: `toolrows!` deletes and recreates
    # Blocks, which relayouts the dock — doing that on every playhead tick moved the
    # panel (and with it the preview axis) mid-gesture, so a crop drag that started
    # before the tick finished somewhere else
    sig = (bs, marked)
    get(ctx.state, :listsig, nothing) == sig && return nothing
    ctx.state[:listsig] = sig
    # the length control shows what the SELECTED blend actually is, so the card
    # never claims a value the timeline doesn't have
    sel = marked === nothing ? nothing : findfirst(b -> (b[1], b[2]) == marked, bs)
    sl = get(ctx.state, :slider, nothing)
    if sl !== nothing && sel !== nothing
        secs = round(bs[sel][3] / fps, digits = 1)
        if abs(sl.value[] - secs) > 0.05 && secs in 0.1:0.1:3.0
            ctx.state[:syncing] = true
            Makie.set_close_to!(sl, secs)
            ctx.state[:syncing] = false
        end
    end
    entries = Any[]
    for (i, j, d) in bs
        into = seq.clips[j]
        from = i == 0 ? nothing : seq.clips[i]
        slot = findslot(into, OpacityEffect)      # the blend IS this effect entry
        on = slot === nothing || slot.enabled
        crossing = from !== nothing && into.start < clipend(from)
        caption = "$(timestring(into.start / fps))  ·  $(round(d / fps, digits = 1)) s" *
                  (on ? "" : "  (off)") * (crossing ? "  ⇄" : from === nothing ? "  ↑" : "")
        actions = Any[("▸", () -> (i == 0 ? (player.timeline.selected[] = j) :
                                   markpair!(player, i, j);
                                   seek!(player, clamp(into.start + d ÷ 2, 0,
                                                       max(seqlength(seq) - 1, 0)))),
                       "go to this blend")]
        # ● on / ○ off — a glyph the UI font actually HAS (⏻ has no glyph and takes
        # Makie's text layout down with it)
        slot === nothing || push!(actions,
            (on ? "●" : "○", () -> (snapshot!(player); slot.enabled = !slot.enabled;
                         refreshedit!(player);
                         setstatus!(player, "blend $(slot.enabled ? "on" : "off") " *
                                            "(its keys stay — Ctrl+Z undoes)")),
             on ? "switch this blend off" : "switch it back on"))
        # ⇄ only while it would DO something: once the clips cross, the caption says so
        (crossing || from === nothing) || push!(actions,
            ("⇄", () -> (markpair!(player, i, j); movetooverlap!(player, from, into, d)),
             "overlap the clips"))
        push!(actions, ("×", () -> (snapshot!(player); clearfade!(into, :in);
                                    into.blendfrom = UInt64(0);   # the pair goes with it
                                    refreshedit!(player);
                                    setstatus!(player, "blend removed (Ctrl+Z undoes)")),
                        "remove"))
        push!(entries, (caption, actions,
                        marked !== nothing && i != 0 && marked == (i, j)))
    end
    isempty(entries) &&
        push!(entries, ("no blends yet — mark two clips, then click Blend clips", Any[]))
    toolrows!(ctx, entries)
    return nothing
end

"""
Clicking the card header blends the two marked clips — that IS the action, there
is no mode to enter and nothing that captures timeline clicks (moving, trimming
and scrubbing keep working while the card is open). The new blend ends up marked,
so the length control edits it right away.
"""
function activateblend!(ctx::ToolContext)
    player = ctx.player
    seq = player.sequence
    p = markedpair(player)
    if p === nothing
        setstatus!(player, "Blend: mark two clips (Shift+click), then click Blend clips")
    else
        i, j = p
        blendclips!(player, seq.clips[i], seq.clips[j]; seconds = blendseconds(player),
                    fit = blendfit(player))
        markpair!(player, i, j)
    end
    deactivatetool!(player)   # nothing stays active — the card keeps showing the blends
    return nothing
end

# ------------------------------------------------------------- Stabilize tool

"The stabilization modes the tool offers, as Makie Menu `(label, value)` options."
# Short on purpose: a Menu shares its row with the "Mode" label inside a 300 px
# dock and, unlike a Label, neither wraps nor shrinks — it just draws past its box
# and gets clipped mid-word. What each mode DOES belongs in the card's
# description, which wraps.
const STABMODES = [("Camera lock", :similarity),
                   ("Object lock", :objectlock),
                   ("Tripod (legacy)", :tripod),
                   ("Perspective (legacy)", :perspective),
                   ("Smooth", :smooth)]

"""
    stabmode(player) -> Ref{Symbol}

The stabilization mode the user picked, surviving a panel rebuild.

It lived in `ctx.state`, which is a FRESH Dict on every rebuild, and the menu's
default was hardcoded to the first entry. So choosing "Object lock" and then
doing anything that rebuilds the panel — Esc, which also deactivates the tool —
silently put you back on Camera lock, and the next press of "Stabilize clip" ran
plain stabilization instead of starting the object pick. Nothing said so.

On `fxwidgets` for the same reason [`mattecardview`](@ref) is: the dock rebuilds
every panel from scratch, so a control that keeps its state in the context has it
discarded by the very rebuild it triggers.
"""
stabmode(player::Player) = get!(() -> Ref(:similarity), player.fxwidgets, :stabmode)

"""
Fill the Stabilize card: the mode, the action that runs the analysis, what the
selected clip currently carries, and a way to take it off again. Stabilizing is
an ANALYSIS of a clip — like the loop finder's search — so it lives with the
tools, not in the effect stack (Simon, 2026-07-27); what it produces (the motion
track) still renders as part of the clip like any other effect.
"""
function stabilizepanel!(ctx::ToolContext)
    player = ctx.player
    mode = stabmode(player)
    ctx.state = Dict{Symbol, Any}(:mode => mode[])
    # the menu opens on the mode that is actually in force, not on the first entry
    i = something(findfirst(m -> m[2] === mode[], STABMODES), 1)
    menu = toolmenu!(ctx, "Mode", STABMODES,
                     v -> (v === nothing || (ctx.state[:mode] = v; mode[] = v));
                     default = STABMODES[i][1])
    analyze = toolaction!(ctx, "Stabilize clip", () -> runstabilize!(ctx))
    toollabel!(ctx, player.stabinfo)
    remove = toolaction!(ctx, "Remove stabilization", () -> begin
        loc = editclip(player)
        (loc === nothing || loc[1].motiontrack === nothing) &&
            return setstatus!(player, "no stabilization on this clip")
        removestabilization!(player)
    end)
    # the widget names the tests and MCP address stay the same, they just live here now
    merge!(player.fxwidgets, Dict{Symbol, Any}(:modemenu => menu, :analyze => analyze,
                                               :remove => remove))
    return nothing
end

"Run the analysis the Stabilize card is set to (object lock waits for a click first)."
function runstabilize!(ctx::ToolContext)
    player = ctx.player
    player.stabinfo[] == "analyzing…" &&
        return setstatus!(player, "analysis already running — progress in the bottom right")
    mode = get(ctx.state, :mode, :similarity)
    if mode === :objectlock
        startobjectpick!(player)      # the next preview click picks the subject to lock on
    else
        analyzeat!(player, (c; kwargs...) ->
                       analyzemotion!(c; mode, backend = player.analysisbackend, kwargs...),
                   "motion stabilization")
    end
    return nothing
end

registereffect!(EffectKind(:stabilize, "Stabilize";
    description = "Locks the SELECTED clip. Camera lock holds the framing like a tripod. " *
        "Object lock keeps one subject still — click it in the preview afterwards. " *
        "Smooth keeps the camera moves and only takes the shake out.",
    make = _ -> StabilizeEffect(),
    matches = e -> e isa StabilizeEffect,
    read = _ -> NamedTuple(),
    body = stabilizepanel!,
    activate = ctx -> (runstabilize!(ctx); deactivatetool!(ctx.player))))

# --------------------------------------------------------------- Flicker tool

"""
Fill the Flicker card: run the analysis, see WHAT IT BOUGHT, dial it back live,
or take it off. The old flow only staged a section from the effect palette and
then waited for a second button nobody found — so the honest answer to "did it do
anything?" was silence (Simon, 2026-07-27).
"""
function flickerpanel!(ctx::ToolContext)
    player = ctx.player
    ctx.state = Dict{Symbol, Any}(:cutoff => 0.5)
    toolslider!(ctx, "Cutoff", 0.1:0.05:2.0, v -> (ctx.state[:cutoff] = v);
                startvalue = 0.5, format = v -> "$(round(v, digits = 2)) Hz")
    analyze = toolaction!(ctx, "Analyze + fix flicker", () -> runflickerfix!(ctx))
    # live strength: the one control that shows the fix working, frame by frame
    toolslider!(ctx, "Strength", 0.0:0.05:1.0, v -> begin
        loc = editclip(player)
        loc === nothing && return
        ct = loc[1].colortrack
        ct === nothing && return
        ct.strength = Float32(v)
        player.playing[] || notify(player.playhead)
    end; startvalue = 1.0, format = v -> string(round(v, digits = 2)))
    # NB: plain Observable + `ontool!`, never a bare `lift` on a player observable:
    # a lift is not registered for cleanup, so every panel rebuild leaves another
    # one writing into a DELETED label — which throws inside the observable chain
    # and takes every listener after it down with it (playback and the crop tool
    # stopped reacting three testsets later).
    status = Observable("not analyzed yet")
    refresh!() = (loc = editclip(player);
                  status[] = loc === nothing ? "no clip selected" :
                             loc[1].colortrack === nothing ? "not analyzed yet" :
                             stabdescription(loc[1].colortrack))
    toollabel!(ctx, status)
    ontool!(ctx, player.playhead; priority = 0) do _
        refresh!()
    end
    refresh!()
    remove = toolaction!(ctx, "Remove flicker fix", () -> begin
        loc = editclip(player)
        (loc === nothing || loc[1].colortrack === nothing) &&
            return setstatus!(player, "no flicker fix on this clip")
        snapshot!(player)
        setcolortrack!(loc[1], nothing)
        notify(player.playhead)
        setstatus!(player, "flicker fix removed (Ctrl+Z restores)")
    end)
    merge!(player.fxwidgets, Dict{Symbol, Any}(:color => analyze, :flickerremove => remove))
    return nothing
end

"""
Clicking the Flicker card's header runs the analysis — the header IS the action,
same as the other tool cards; nothing stays switched on afterwards.
"""
function runflickerfix!(ctx::ToolContext)
    player = ctx.player
    if player.stabinfo[] == "analyzing…"
        setstatus!(player, "analysis already running — progress in the bottom right")
    else
        cut = ctx.state isa Dict ? get(ctx.state, :cutoff, 0.5) : 0.5
        analyzeat!(player, (c; kw...) -> analyzecolor!(c; cutoff = cut,
                                                       backend = player.analysisbackend, kw...),
                   "color stabilization")
    end
    deactivatetool!(player)
    return nothing
end

registereffect!(EffectKind(:flicker, "Fix flicker";
    description = "Evens out exposure/colour jitter on the SELECTED clip: everything faster " *
        "than the cutoff is treated as flicker, slower changes survive.",
    params = [FxParam(:strength, "Strength"; min = 0.0, max = 1.0, default = 1.0)],
    kfkeys = [:flicker_strength],
    make = nt -> FlickerEffect(Float32(nt.strength)),
    matches = e -> e isa FlickerEffect,
    read = e -> (strength = Float64(e.strength),),
    body = flickerpanel!, activate = runflickerfix!))

registereffect!(EffectKind(:loopfinder, "Loop finder";
    description = "Each reference card holds one frame; ▼ hints on the thumb track mark the " *
        "frames most similar to the SELECTED card — click a ▼ to cut there. " *
        "Find adds the playhead frame as another reference.",
    make = _ -> LoopFinderEffect(),
    matches = e -> e isa LoopFinderEffect,
    read = _ -> NamedTuple(),
    activate = activateloopfinder!, analysis = true))

registereffect!(EffectKind(:blend, "Blend clips";
    description = "Mark two clips (Shift+click), then press the button — the later clip " *
        "fades in. The two halves link to each other, so each card shows the other's " *
        "fade next to its own.",
    params = [FxParam(:seconds, "Length (s)"; min = 0.1, max = 3.0, default = 0.6)],
    kfkeys = [:blend_seconds],
    make = nt -> BlendEffect(Float64(nt.seconds)),
    matches = e -> e isa BlendEffect,
    read = e -> (seconds = e.seconds,),
    body = blendpanel!, activate = activateblend!, analysis = true))

# ----------------------------------------------------------------- Matte tool

"""
Mark a subject on one frame, propagate it across the clip.

The marked frames are *matte keyframes*: each one is a place the user says "the
subject is here", and propagation fills the frames between them. Adding a second
mark where the propagation drifts is the whole correction workflow, which is why
the card list is the seed list — one row per marked frame, removable, click to
jump there ([[feedback-ui-shape-lists-not-buttons]] applies: the managed things
are the marks, not the buttons).

Marking is a click in the preview: it seeds a box around the click, at source
resolution. `MatteEffect` is added to the clip on the first successful run, so
the result is visible immediately rather than requiring a separate "add effect"
step, and its `Matte`/`Feather` sliders are keyframable like any other param.
"""
function mattepanel!(ctx::ToolContext)
    player = ctx.player
    loc = editclip(player)
    clip = loc === nothing ? nothing : loc[1]
    toollabel!(ctx, player.matteinfo)
    col = mattecollect(player)
    marking = col !== nothing && clip !== nothing && col.clip === clip
    marks = clip === nothing ? Dict{Int, Matrix{UInt8}}() : mattemarks(player, clip)
    if clip === nothing || (!marking && isempty(marks) && clip.mattetrack === nothing)
        # no card on this clip: drop the published one too, or "is there a card?"
        # keeps answering yes with a handle to blocks that no longer exist
        delete!(player.fxwidgets, :mattecard)
        toolaction!(ctx, "Mark subject", () -> startmattepick!(ctx))
        warmmattepanel!(ctx)
        return nothing
    end
    matteviewrow!(ctx)
    # Painting is Alt+drag and always has been; what was missing was anything on
    # screen saying so, and any way to size the brush without knowing about `[`
    # and `]`. Next to the view toggle because both are "how am I looking at and
    # touching this matte", as opposed to the cards below, which are the matte.
    clip.mattetrack === nothing || brushrow!(ctx)
    # one card per marked frame, plus the frame being marked — its points are not
    # a mark yet, they become one when the session is committed
    frames = collect(keys(marks))
    marking && !(col.srcframe in frames) && push!(frames, col.srcframe)
    sort!(frames)
    # ONE CARD for the frame being marked — it holds the object pills and the live
    # preview — and one thin ROW for each of the others. A full card per marked
    # frame turned ten marks into a screen and a half of scrolling, when all a
    # non-live mark needs is its number, a way back to it, and a way to drop it.
    # `toolrows!` is the codebase's own "list of items, not buttons" shape.
    live = nothing
    others = Any[]
    for f in frames
        if marking && col.srcframe == f
            w = matteseedcard!(ctx, clip, f, true)
            w.live && (live = w)
        else
            push!(others, ("frame $f",
                           [("go", () -> gotomatteseed!(ctx, clip, f)),
                            ("×",  () -> removematteseed!(player, clip, f))],
                           false))
        end
    end
    # …and one per REPAIRED frame. Without these a repair is invisible: you can
    # fix a frame and have no way to see which frames you fixed, no way to get
    # back to one, and no way to drop a single repair short of undoing every edit
    # since. A repair is an edit and needs the same handle a mark has.
    # …and the repairs join the same list, marked as such: they are the same kind
    # of thing (a frame you can return to and undo), so two separate lists of
    # near-identical rows would only ask the reader which is which.
    for (f, _) in sort!(collect(matterepairs(player, clip)); by = first)
        push!(others, ("frame $f · repaired",
                       [("go", () -> gotomatteseed!(ctx, clip, f)),
                        ("×",  () -> dropmatterepair!(player, clip, f))],
                       false))
    end
    isempty(others) || toolrows!(ctx, others)
    # published so a click during marking can repaint the live card's pills in
    # place — a full rebuild per click loses the one arriving mid-rebuild
    live === nothing ? delete!(player.fxwidgets, :mattecard) :
                       (player.fxwidgets[:mattecard] = live)
    # The two ways to finish a marking session, as two buttons, because they were
    # two keystrokes one Shift apart and the difference is the whole decision:
    # propagate through the shot, or fix the one frame that came out broken.
    # Shift+Enter still does this — but a repair gesture that exists only as a
    # modifier on a key, hinted in a status line that the next status overwrites,
    # is a feature nobody finds.
    if marking && col.prevtrack !== nothing
        toolaction!(ctx, "Fix this frame only", () -> repairmattecollect!(col); footer = true)
    elseif !marking
        # THE WAY BACK IN. "Mark subject" is gated on there being no matte yet, so
        # once one was applied there was no button, no shortcut and no palette
        # entry that started marking again — and "Fix this frame only" above needs
        # a marking session to exist before it appears. The whole repair flow was
        # a door that locked behind you: the one case it was built for, a bad
        # frame in a finished matte, was the one case you could not reach.
        toolaction!(ctx, "Mark this frame", () -> startmattepick!(ctx); footer = true)
    end
    # Say what it will COST when that is worth knowing. The alpha is linear in the
    # shot's length — 3.5 GB a minute at 1080p — so on a long clip this is the
    # number that decides whether to matte the whole thing or trim it first. Shown
    # only past a gigabyte: on the short clips that are most of them it is noise.
    need = mattebytes(clip)
    toolaction!(ctx, need > 2^30 ?
                     "Apply matte to clip (≈$(round(need / 2^30; digits = 1)) GB)" :
                     "Apply matte to clip",
                () -> applymattenow!(ctx, clip); footer = true)
    warmmattepanel!(ctx)
    return nothing
end

"""
    gotocaption!(player, step) -> nothing

Move the playhead to the caption `step` lines away from the one under it.

Correcting a transcript is a walk through it, line by line — and the panel only
ever shows the line under the playhead, so without this you scrub blindly hunting
for the next one. From nowhere (`i == 0`) it goes to the first line forward and
the last line back, so the buttons do something useful from a gap too.
"""
function gotocaption!(player::Player, step::Integer)
    seq = player.sequence
    isempty(seq.captions) && return nothing
    i = captionindexat(seq, player.playhead[])
    j = i == 0 ? (step > 0 ? 1 : length(seq.captions)) :
        clamp(i + Int(step), 1, length(seq.captions))
    c = seq.captions[j]
    seek!(player, clamp(round(Int, c.start * seq.framerate), 0,
                        max(seqlength(seq) - 1, 0)))
    setstatus!(player, "line $j of $(length(seq.captions)): $(c.text)")
    return nothing
end

"""
The transcript panel: how many lines there are, and the one under the playhead —
editable.

A transcript is a GUESS, and the one thing anybody reliably wants to do with a
guess is correct it. Without this the speech integration is read-only: you can
generate captions, see them, move them and restyle them, and the moment the model
mishears a name your only recourse is to run it again and get the same answer.

Shows the line under the playhead rather than a list of every line, because the
playhead is already how you navigate to the one you disagree with — you hear it,
you stop, you fix it. A scrolling list would be a second way to move through the
timeline that does not move the timeline.
"""
function transcriptpanel!(ctx::ToolContext)
    player = ctx.player
    seq = player.sequence
    if isempty(seq.captions)
        toolaction!(ctx, "Transcribe", () -> runtranscribe!(player))
        return nothing
    end
    i = captionindexat(seq, player.playhead[])
    toolcard!(ctx; caption = "$(length(seq.captions)) line(s)") do g, blocks
        if i == 0
            lab = Label(g[1, 1], "no line under the playhead"; fontsize = 11,
                        halign = :left, tellwidth = false)
            push!(blocks, lab)
        else
            c = seq.captions[i]
            box = Textbox(g[1, 1]; stored_string = c.text, width = Makie.Relative(1.0),
                          tellwidth = false, reset_on_defocus = false, fontsize = 11)
            # `stored_string` fires on ENTER, not per keystroke, so one correction
            # is one undo step rather than one per letter.
            on(box.stored_string) do str
                str === nothing && return
                str == seq.captions[i].text && return      # focus loss, not an edit
                editcaption!(player, str)
            end
            push!(blocks, box)
            t = Label(g[2, 1], "line $i of $(length(seq.captions))  ·  " *
                                "$(round(c.start; digits = 2))s – $(round(c.stop; digits = 2))s";
                      fontsize = 10, halign = :left, tellwidth = false)
            push!(blocks, t)
        end
        nothing
    end
    # Walk the transcript. The panel shows ONE line — the one under the playhead —
    # so without a way to step between them, fixing what the model misheard means
    # scrubbing at random until a line appears.
    slot = toolslot(ctx, :controls)
    if slot !== nothing
        colors = player.fxwidgets[:uicolors]
        nav = GridLayout(slot[length(ctx.controls) + 1, 1])
        prev = Button(nav[1, 1]; label = "◀ line", fontsize = 11, height = 22,
                      tellwidth = false, width = Makie.Relative(1.0),
                      buttoncolor = colors.surface)
        nxt = Button(nav[1, 2]; label = "line ▶", fontsize = 11, height = 22,
                     tellwidth = false, width = Makie.Relative(1.0),
                     buttoncolor = colors.surface)
        on(_ -> gotocaption!(player, -1), prev.clicks)
        on(_ -> gotocaption!(player, 1), nxt.clicks)
        push!(ctx.controls, nav)          # ONE entry — see `matteviewrow!`
    end
    toolaction!(ctx, "Re-transcribe", () -> runtranscribe!(player); footer = true)
    return nothing
end

"""
The narration panel: what has been said, and a box to say something new.

Placed at the PLAYHEAD, because that is where you are when you decide a line
belongs — the alternative is a time field to type into, which means reading the
timecode off the screen and copying it by hand.

Each existing line is a card with its own × so one can go without touching the
others, and re-rendering is per line: a voice or wording change should not cost
the whole track.
"""
function narrationpanel!(ctx::ToolContext)
    player = ctx.player
    seq = player.sequence
    colors = player.fxwidgets[:uicolors]
    toolcard!(ctx; caption = "new line") do g, blocks
        box = Textbox(g[1, 1]; placeholder = "what should be said…",
                      width = Makie.Relative(1.0), tellwidth = false,
                      reset_on_defocus = false, fontsize = 11)
        # ENTER commits, so a line is one edit rather than one per keystroke.
        on(box.stored_string) do str
            (str === nothing || isempty(strip(str))) && return
            addnarration!(player, str)
            box.stored_string[] = nothing
        end
        push!(blocks, box)
        nothing
    end
    for (i, nar) in enumerate(seq.narration)
        toolcard!(ctx; caption = "at $(round(nar.at; digits = 2))s",
                  onremove = _ -> dropnarration!(player, i)) do g, blocks
            # A Textbox, not a Label: the words are the EDIT, and a line you can
            # only delete and retype is not an editable line.
            box = Textbox(g[1, 1]; stored_string = nar.text, fontsize = 11,
                          width = Makie.Relative(1.0), tellwidth = false,
                          reset_on_defocus = false)
            on(str -> str === nothing || setnarrationtext!(player, i, str), box.stored_string)
            push!(blocks, box)
            # Side by side, not stacked. Each line is a card in a scrolling panel
            # and a documentary has dozens of them — three full-width controls
            # apiece made thirty lines a scroll nobody would use.
            brow = GridLayout(g[2, 1])
            flat = ghostbutton(colors)
            b  = Button(brow[1, 1]; label = isempty(nar.samples) ? "render" : "re-render", flat...)
            mv = Button(brow[1, 2]; label = "move here", flat...)
            on(_ -> rendernarration!(player, i), b.clicks)
            on(_ -> movenarration!(player, i), mv.clicks)
            push!(blocks, brow)
            # The voice, once the synthesizer can say what it has. No menu at all
            # rather than an empty one: a control listing nothing is worse than a
            # control that is not there yet.
            vs = speakvoices()
            if !isempty(vs)
                menu = Menu(g[3, 1]; options = vs, fontsize = 11,
                            default = nar.voice in vs ? nar.voice : nothing,
                            prompt = "voice: $(nar.voice)",
                            width = Makie.Relative(1.0), tellwidth = false)
                on(v -> v === nothing || setnarrationvoice!(player, i, v), menu.selection)
                push!(blocks, menu)
            end
            nothing
        end
    end
    return nothing
end

"""
    addnarration!(player, text) -> nothing

Add a spoken line at the playhead and synthesize it.

On the analysis executor: Kokoro is ~0.5 s of model time for a sentence warm and
far more cold, and the UI thread must stay answerable — the same reason the matte
and the depth run there.
"""
function addnarration!(player::Player, text::AbstractString)
    hasspeakmodel() || return setstatus!(player, "narration: no synthesizer installed")
    seq = player.sequence
    at = player.playhead[] / (seq.framerate > 0 ? seq.framerate : 25.0)
    snapshot!(player)
    nar = Narration(String(text), at)
    push!(seq.narration, nar)
    refreshedit!(player)
    setstatus!(player, "narration: speaking…")
    runanalysis(player) do
        try
            render!(nar)
            put!(player.uiqueue, () -> begin
                refreshedit!(player)
                setstatus!(player, "narration added at $(round(at; digits = 2))s")
            end)
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                setstatus!(player, "narration failed: $(briefly(e))")
                @error "narration failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"""
    setnarrationtext!(player, i, text) -> nothing

Reword line `i`, which un-renders it.

The samples are a cache OF THE WORDS. Keeping them after an edit would leave a
line whose card says one thing and whose audio says another — so the replacement
starts unrendered and the button goes back to saying "render". REPLACE rather
than mutate: the undo stack shares these objects.
"""
function setnarrationtext!(player::Player, i::Integer, text::AbstractString)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    old = seq.narration[i]
    strip(String(text)) == strip(old.text) && return nothing
    snapshot!(player)
    seq.narration[i] = Narration(String(text), old.at, old.voice)
    refreshedit!(player)
    setstatus!(player, "narration reworded — press render to speak it")
    return nothing
end

"""
    setnarrationvoice!(player, i, voice) -> nothing

Speak line `i` in a different voice, which un-renders it.

Same rule as [`setnarrationtext!`](@ref) and for the same reason: the samples are
a cache of the words AND the voice, so keeping them would leave a card claiming
one voice over audio in another. Replaces rather than mutates — the undo stack
shares these objects.
"""
function setnarrationvoice!(player::Player, i::Integer, voice::AbstractString)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    old = seq.narration[i]
    String(voice) == old.voice && return nothing
    snapshot!(player)
    seq.narration[i] = Narration(old.text, old.at, String(voice))
    refreshedit!(player)
    setstatus!(player, "narration voice $(voice) — press render to hear it")
    return nothing
end

"""
    movenarration!(player, i) -> nothing

Move line `i` to the playhead, keeping its audio.

WHEN a voiceover lands is the thing you adjust most, and it was the one property
with no control at all: a line added a second early had to be deleted and typed
again with the playhead moved. The words have not changed, so the samples are
still valid and come along — re-synthesizing to move a line would cost seconds
and produce the same sound.
"""
function movenarration!(player::Player, i::Integer)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    old = seq.narration[i]
    at = player.playhead[] / (seq.framerate > 0 ? seq.framerate : 25.0)
    snapshot!(player)
    fresh = Narration(old.text, at, old.voice)
    append!(fresh.samples, old.samples)
    fresh.rate = old.rate
    seq.narration[i] = fresh
    refreshedit!(player)
    setstatus!(player, "narration moved to $(round(at; digits = 2))s")
    return nothing
end

"Re-synthesize one line — after editing its words or changing its voice."
function rendernarration!(player::Player, i::Integer)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    nar = seq.narration[i]
    setstatus!(player, "narration: speaking…")
    runanalysis(player) do
        try
            # REPLACE, never `render!` in place — see `render`. The undo stack
            # shares this object.
            fresh = render(nar)
            put!(player.uiqueue, () -> (i <= length(seq.narration) &&
                                        (seq.narration[i] = fresh)))
            put!(player.uiqueue, () -> (refreshedit!(player);
                                        setstatus!(player, "narration rendered")))
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                setstatus!(player, "narration failed: $(briefly(e))")
                @error "narration render failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"Drop one spoken line."
function dropnarration!(player::Player, i::Integer)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    snapshot!(player)
    deleteat!(seq.narration, i)
    refreshedit!(player)
    setstatus!(player, "narration line removed")
    return nothing
end

"""
    lookbody!(ctx)

The look card's body: learn the grade, or learn it again somewhere else.

Re-learning matters more than it sounds. The LUT is fitted from ONE frame, so
which frame you were parked on when you pressed it is the whole result — and the
only way to find that out is to try another one. Offering "Learn from this frame"
again on a clip that already has a look is the difference between a dial you can
work with and a one-shot you have to undo.
"""
function lookbody!(ctx)
    player = ctx.player
    loc = editclip(player)
    loc === nothing && return nothing
    label = loc[1].look === nothing ? "Learn look from this frame" : "Re-learn from this frame"
    toolaction!(ctx, label, () -> runlook!(player))
    return nothing
end

"""
    depthbody!(ctx)

The depth-blur card's body: estimate the depth, or pick the focus point.

The card's sliders come from its `EffectKind` params; this adds the two ACTIONS,
so one card answers "make the background soft" end to end instead of sending the
user to a second card for the analysis and a third for the picker.
"""
function depthbody!(ctx)
    player = ctx.player
    loc = editclip(player)
    loc === nothing && return nothing
    clip, srcframe = loc
    if clip.depthtrack === nothing
        toolaction!(ctx, "Estimate depth", () -> rundepth!(player))
        return nothing
    end
    toolaction!(ctx, "Pick focus point", () -> pickfocus!(player))
    toolaction!(ctx, "Re-estimate depth", () -> rundepth!(player))
    # The map itself, under the actions. A card in the tool list rather than an
    # overlay on the preview: an overlay has to be mapped through the clip's crop
    # and the canvas letterbox to line up, and a depth map that is subtly
    # misaligned is worse than none — it would be read as the model being wrong.
    d = depthframe(clip, srcframe)
    d === nothing ||
        tooladdcard!(ctx, depthimage(d); caption = "depth here — bright is near")
    return nothing
end

"""
    pickfocus!(player) -> nothing

Arm a one-shot preview click that sets the depth-blur focus to whatever you click
ON.

The parameter is a depth in `0..1` and nothing on screen is labelled with one, so
setting it by slider is guesswork — you drag until the thing you care about looks
sharp, which is a search, not an adjustment. Clicking the subject is the question
the user actually has: *make this sharp*.

Reads the clip's own depth track at the clicked pixel, so it agrees with what the
effect will do by construction rather than by a second estimate.
"""
function pickfocus!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "focus: no clip under the playhead")
    clip, srcframe = loc
    clip.depthtrack === nothing &&
        return setstatus!(player, "focus: estimate depth first")
    setstatus!(player, "focus: click what should be sharp")
    player.onpick = function (pt)
        d = depthframe(clip, srcframe)
        if d === nothing
            setstatus!(player, "focus: this frame has no depth")
            return nothing
        end
        # Preview point → matte-space fraction: `previewtomatte` is the same
        # conversion the matte's clicks use, so a click means the same pixel in
        # both tools rather than two nearly-identical mappings.
        nx, ny = previewtomatte(player, clip, srcframe, pt)
        w, h = size(d)
        ix = clamp(floor(Int, nx * w) + 1, 1, w)
        iy = clamp(floor(Int, ny * h) + 1, 1, h)
        z = Float32(d[ix, iy]) / 255.0f0
        snapshot!(player)
        e = findeffect(clip, DepthBlurEffect)
        seteffect!(clip, DepthBlurEffect(z, e === nothing ? 0.6f0 : e.strength))
        refreshedit!(player)
        notify(player.playhead)
        setstatus!(player, "focus set to $(round(z; digits = 2)) — what you clicked is sharp")
        return nothing
    end
    return nothing
end

"""
The time-interpolation panel: which mode this clip uses, and what its rate is.

A card for the same reason `StabilizeEffect` is one — its docstring records what
the alternative cost: "there was no card to fold, no toggle to compare with".
Optical flow was reachable only from the command palette, so a clip either
juddered or did not and nothing on screen said which, or why.

Shows the RATE too, because the mode does nothing at all above 1× and a control
that is correctly idle looks broken. A user who turns it on and sees no change
needs to be told the clip is not slowed, not left to conclude the feature is.
"""
function timeinterppanel!(ctx::ToolContext)
    player = ctx.player
    loc = editclip(player)
    loc === nothing && return nothing
    clip = loc[1]
    colors = player.fxwidgets[:uicolors]
    flow = clip.timeinterp === :flow
    toolcard!(ctx; caption = "$(round(clip.rate; digits = 2))× source rate") do g, blocks
        off = Makie.RGBf(0.22, 0.23, 0.26)
        row = GridLayout(g[1, 1])
        b1 = Button(row[1, 1]; label = "Frame sampling", fontsize = 11, height = 22,
                    tellwidth = false, width = Makie.Relative(1.0),
                    cornerradius = PILLRADIUS, buttoncolor = flow ? off : colors.accent)
        b2 = Button(row[1, 2]; label = "Optical flow", fontsize = 11, height = 22,
                    tellwidth = false, width = Makie.Relative(1.0),
                    cornerradius = PILLRADIUS, buttoncolor = flow ? colors.accent : off)
        on(_ -> setinterp!(player, clip, :sample), b1.clicks)
        on(_ -> setinterp!(player, clip, :flow), b2.clicks)
        push!(blocks, b1, b2)
        if clip.rate >= 1.0
            lab = Label(g[2, 1], wraptext("This clip is not slowed, so no frame " *
                                          "falls between two source frames yet.", 34);
                        fontsize = 10, halign = :left, tellwidth = false,
                        color = colors.text_muted)
            push!(blocks, lab)
        end
        nothing
    end
    return nothing
end

"Set a clip's time interpolation from the card, undoably."
function setinterp!(player::Player, clip::Clip, mode::Symbol)
    clip.timeinterp === mode && return nothing
    snapshot!(player)
    settimeinterp!(clip, mode)
    refreshedit!(player)
    notify(player.playhead)
    setstatus!(player, mode === :flow ?
        "optical flow on — in-between frames are synthesized" :
        "frame sampling — the nearest source frame repeats")
    return nothing
end

registertool!(:timeinterp, "Time interpolation",
    "How a SLOWED clip fills the frames its source does not have: repeat the " *
    "nearest (frame sampling) or synthesize it with RIFE (optical flow).";
    activate = ctx -> smoothslowmo!(ctx.player),
    panel = timeinterppanel!)

registertool!(:narration, "Narration",
    "Type a line, press Enter, and Kokoro speaks it at the playhead. Mixed over " *
    "the timeline in the preview AND in the export.";
    activate = ctx -> setstatus!(ctx.player, "narration: type a line in the card and press Enter"),
    panel = narrationpanel!)

registertool!(:transcript, "Transcript",
    "Speech to captions with Whisper. The line under the playhead is editable — " *
    "a transcript is a guess, and correcting it must not mean running it again.";
    activate = ctx -> runtranscribe!(ctx.player),
    panel = transcriptpanel!)

"""
Forget one frame's repair. The pixels stay until the matte is re-run — see
[`repairmatteat!`](@ref) for why they cannot simply be put back.
"""
function dropmatterepair!(player::Player, clip::Clip, srcframe::Integer)
    snapshot!(player)
    delete!(matterepairs(player, clip), Int(srcframe))
    refreshmattepanel!(player; structure = true)
    setstatus!(player, "matte: repair on frame $srcframe forgotten — re-run the " *
                       "matte to put the propagated frame back")
    return nothing
end

"""
The preview's picture while a matte is being made: the finished matte, or SAM 2's
segmentation with each object outlined in its colour.

On the PANEL rather than in a card, because it is one state for the whole clip —
with one card per marked frame there is no card that owns it.
"""
function matteviewrow!(ctx::ToolContext)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    player = ctx.player
    v = mattecardview(player)[]
    row = GridLayout(slot[length(ctx.controls) + 1, 1])
    off = Makie.RGBf(0.22, 0.23, 0.26)
    b1 = Button(row[1, 1]; label = "Matte", fontsize = 11, height = 22, tellwidth = false,
                width = Makie.Relative(1.0), buttoncolor = v === :matte ? MATTECOLORS[2] : off)
    b2 = Button(row[1, 2]; label = "SAM 2", fontsize = 11, height = 22, tellwidth = false,
                width = Makie.Relative(1.0), buttoncolor = v === :sam2 ? MATTECOLORS[2] : off)
    on(_ -> setmatteview!(player, :matte), b1.clicks)
    on(_ -> setmatteview!(player, :sam2), b2.clicks)
    # EXACTLY ONE entry, like every other control helper: the next control's row
    # is `length(ctx.controls) + 1`, so pushing the two buttons as well left rows
    # 3 and 4 empty — and an empty row makes the whole layout indeterminate, which
    # collapsed the effect card to a 64 px sliver. `delete!` on a GridLayout
    # recurses into its content, so the buttons still get cleaned up with it.
    push!(ctx.controls, row)
    return (b1, b2)
end

"""
    brushrow!(ctx) -> (minus, plus)

The paint gesture, said out loud, with a brush size you can set by clicking.

Alt+drag has painted into the mask for as long as the brush has existed, and
right-drag has erased — but neither appeared anywhere on screen, so the repair
half of the matte tool was reachable only by already knowing it was there. The
size readout doubles as the label: it is the one brush property you change often
enough to want a number for, and `[`/`]` still move it.
"""
function brushrow!(ctx::ToolContext)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    player = ctx.player
    row = GridLayout(slot[length(ctx.controls) + 1, 1])
    # the FULL palette, for `text_muted` — see `matteseedcard!`
    colors = player.fxwidgets[:uicolors]
    Label(row[1, 1], "Alt+drag paints · right erases"; fontsize = 10,
          color = colors.text_muted, halign = :left, tellwidth = false)
    pct = round(Int, 100 * player.brushradius)
    # The HINT stays muted — it is a sentence you read once. The ± are controls
    # and are drawn as controls: at `text_muted` on a transparent fill they were
    # as quiet as the sentence next to them and read as decoration, the same way
    # `+ object` did before it got its contrast back.
    minus = Button(row[1, 2]; label = "−", fontsize = 12, width = 22, height = 22,
                   buttoncolor = (:transparent, 0.0), strokewidth = 1,
                   strokecolor = (colors.text, 0.45), labelcolor = colors.text)
    Label(row[1, 3], "$(pct)%"; fontsize = 10, color = colors.text, tellwidth = true)
    plus = Button(row[1, 4]; label = "+", fontsize = 12, width = 22, height = 22,
                  buttoncolor = (:transparent, 0.0), strokewidth = 1,
                  strokecolor = (colors.text, 0.45), labelcolor = colors.text)
    on(_ -> setbrushradius!(player, player.brushradius / 1.25), minus.clicks)
    on(_ -> setbrushradius!(player, player.brushradius * 1.25), plus.clicks)
    # ONE entry, like `matteviewrow!` — see the note there about empty rows.
    push!(ctx.controls, row)
    return (minus, plus)
end

"""
Apply: commit whatever is being marked, then propagate across the clip.

A marking session's points are not marks yet — they become one when the session
ends — so applying mid-marking used to find an empty mark dict and answer "mark
the subject on a frame first" at somebody who had just marked two birds.
Committing IS what Apply means here.
"""
function applymattenow!(ctx::ToolContext, clip::Clip)
    c = mattecollect(ctx.player)
    if c !== nothing && c.clip === clip && !isempty(c.points)
        finishmattecollect!(c)
    else
        runmatte!(ctx; clip = clip)
    end
    return nothing
end

"""
Which view the matte card and the preview are showing: `:matte`, the finished
result, or `:sam2`, the segmentation with each object outlined in its colour. One
state, because they are two ways of looking at the same thing and never both.

On the PLAYER, not on the tool context: the dock rebuilds every panel context from
scratch on each refresh, so a control that wrote its state there would have it
discarded by the very rebuild it triggered — the toggle flipped back on screen and
nothing ever changed.
"""
mattecardview(player::Player) = get!(() -> Ref(:matte), player.fxwidgets, :matteview)

"""
Every frame the user marked on this clip, keyed by source frame.

Held next to the player rather than on the track because a mark is an input to
propagation and the track is its output: re-running must see every mark, not the
seed frames the last run happened to record. Keyed by clip id so it survives
sorting and undo.
"""
mattemarks(player::Player, clip::Clip) =
    get!(() -> Dict{Int, Matrix{UInt8}}(), player.mattemarks, clip.id)

"""
    rundepth!(player) -> nothing

Estimate depth for the clip under the playhead and defocus its background.

One action, not two, because "estimate depth" on its own shows the user nothing —
the track is invisible until something reads it, and a button whose effect is
invisible reads as broken. So this adds the [`DepthBlurEffect`](@ref) as well, and
the picture changes the moment the analysis lands. The effect's parameters are
then the dial: focus is keyframable, so a rack focus is a curve on it.

Runs on the analysis executor like the matte does, and for the same reason — it
is one model call per frame and the UI thread must stay answerable.
"""
function rundepth!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "depth: no clip under the playhead")
    clip = loc[1]
    hasdepthmodel() || return setstatus!(player, "depth: no model installed")
    player.matteinfo[] == "matting…" &&
        return setstatus!(player, "depth: an analysis is already running")
    setstatus!(player, "depth: estimating $(srclength(clip)) frames…")
    player.jobprogress[] = 0.0
    runanalysis(player) do
        try
            reader = framereader(clip, player.engine)
            analyzedepth!(clip, reader; progress = (d, t) -> begin
                player.jobprogress[] = d / max(t, 1)
            end)
            put!(player.uiqueue, () -> begin
                # The effect too — see the docstring on why the analysis alone is
                # not a usable result.
                findeffect(clip, DepthBlurEffect) === nothing &&
                    push!(clip.effects, Effect(DepthBlurEffect()))
                player.jobprogress[] = NaN
                refreshedit!(player)
                notify(player.playhead)
                setstatus!(player, "depth ready — Focus and Defocus are keyframable in the inspector")
            end)
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                player.jobprogress[] = NaN
                setstatus!(player, "depth failed: $(briefly(e))")
                @error "depth analysis failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"""
    runlook!(player) -> nothing

Grade the clip under the playhead from the frame under the playhead.

**From the frame you are looking at**, not the clip's first: a shot often opens
on black or mid-whip, and a look predicted from that is a look for a frame nobody
sees. Choosing the frame is the entire user input to this feature, which is why
it is the playhead's and not a hidden default.

One model call, so it runs inline rather than through the analysis executor —
~1.8 ms against the ~0.6 s a segmenter takes. The frame read costs more than the
prediction.
"""
function runlook!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "look: no clip under the playhead")
    clip, srcframe = loc
    haslookmodel() || return setstatus!(player, "look: no model installed")
    setstatus!(player, "look: grading from frame $srcframe…")
    runanalysis(player) do
        try
            img = framereader(clip, player.engine)(srcframe)
            analyzelook!(clip, img)
            put!(player.uiqueue, () -> begin
                findeffect(clip, LookEffect) === nothing &&
                    push!(clip.effects, Effect(LookEffect()))
                refreshedit!(player)
                notify(player.playhead)
                setstatus!(player, "look applied — Look is keyframable in the inspector")
            end)
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                setstatus!(player, "look failed: $(briefly(e))")
                @error "look analysis failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"""
    runtranscribe!(player) -> nothing

Transcribe the timeline and put a caption overlay on it.

The overlay is added too, for the reason `rundepth!` adds its effect: a
transcript nothing draws is invisible, and an invisible result reads as a broken
button. If one is already there the transcript simply replaces what it shows.

Whisper decodes the whole timeline, so this is minutes rather than milliseconds —
it runs on the analysis executor with the progress bar, like the matte.
"""
function runtranscribe!(player::Player)
    seq = player.sequence
    isempty(seq.clips) && return setstatus!(player, "captions: the timeline is empty")
    hasspeechmodel() || return setstatus!(player, "captions: no speech model installed")
    setstatus!(player, "captions: transcribing…")
    snapshot!(player)      # a re-run replaces corrections; undo has to reach them
    player.jobprogress[] = 0.0
    runanalysis(player) do
        try
            caps = transcribe!(seq)
            put!(player.uiqueue, () -> begin
                player.jobprogress[] = NaN
                any(o -> o.kind === :captions, seq.overlays) ||
                    addoverlay!(seq, :captions)
                refreshedit!(player)
                notify(player.playhead)
                setstatus!(player, "captions: $(length(caps)) line(s) — the overlay's " *
                                   "Y/Size/Opacity are in the inspector")
            end)
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                player.jobprogress[] = NaN
                setstatus!(player, "captions failed: $(briefly(e))")
                @error "transcription failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"""
    matterepairs(player, clip) -> Dict{Int, Matrix{UInt8}}

Single frames whose matte was fixed by hand, by source frame.

Kept beside the marks rather than inside the track for the reason the marks are:
a track is the propagator's OUTPUT and gets rebuilt wholesale, so anything that
must survive a rebuild has to live outside it. `runmatte!` re-applies these after
every full analysis.
"""
matterepairs(player::Player, clip::Clip) =
    get!(() -> Dict{Int, Matrix{UInt8}}(), player.matterepairs, clip.id)

"""
    beginmattebrush!(player, foreground) -> Bool
    mattebrushto!(player, p) -> Bool
    endmattebrush!(player) -> Bool

Paint into a finished matte, one stroke at a time.

The three exist separately because a stroke is three events and only the last one
should reach the document. `begin` takes a copy of the frame's alpha,
`to` dabs into that copy and shows it, and `end` commits the whole stroke through
[`repairmatteat!`](@ref) — so one stroke is one undo step, and letting go outside
the picture or pressing Escape leaves the matte exactly as it was.

Committing per dab instead would put a hundred entries on the undo stack for one
stroke and re-upload the track a hundred times.
"""
function beginmattebrush!(player::Player, foreground::Bool)
    loc = editclip(player)
    loc === nothing && return false
    clip, srcframe = loc
    m = matteframe(clip, srcframe)
    m === nothing && (setstatus!(player, "matte: nothing to paint into — run the matte first"); return false)
    # `foreground` is fixed for the stroke's whole length. Re-reading the mouse on
    # every move would flip add to erase mid-stroke on a stray second button.
    player.mattebrush = (clip, Int(srcframe), m, foreground)
    return true
end

function mattebrushto!(player::Player, p; radius::Real = 0.04)
    br = player.mattebrush
    br === nothing && return false
    clip, srcframe, mask, foreground = br
    nx, ny = previewtomatte(player, clip, srcframe, p)
    brushmatte!(mask, nx, ny, foreground; radius)
    # Shown by writing the live track, not by committing: the picture has to
    # follow the brush, and the document must not.
    repairframe!(clip, srcframe, mask)
    notify(player.playhead)
    return true
end

function endmattebrush!(player::Player)
    br = player.mattebrush
    br === nothing && return false
    mask = br[3]
    player.mattebrush = nothing
    return repairmatteat!(player, mask)
end

"""
    repairmatteat!(player, mask) -> Bool

Fix the matte on the frame under the playhead, and remember the fix.

The repair a propagated matte needs is almost never "run it all again" — it is
one frame in the middle that came out wrong while the frames either side are
fine. Re-analysing rebuilds the clip and costs what the first run cost; this
writes one frame and costs nothing.
"""
function repairmatteat!(player::Player, mask::AbstractMatrix)
    loc = editclip(player)
    loc === nothing && (setstatus!(player, "matte: no clip under the playhead"); return false)
    clip, srcframe = loc
    clip.mattetrack === nothing &&
        (setstatus!(player, "matte: nothing to repair — run the matte first"); return false)
    snapshot!(player)
    # COPY the track before writing into it. `snapshot` shares `mattetrack`
    # between the undo record and the live clip, so mutating the alpha in place
    # would edit the snapshot too and the repair would survive its own undo.
    # One clip's alpha, once per explicit repair — the alternative is an undo
    # that silently does nothing.
    old = clip.mattetrack
    clip.mattetrack = MatteTrack(copy(old.alpha), old.src_in, copy(old.seeds))
    if !repairframe!(clip, srcframe, mask)
        clip.mattetrack = old
        setstatus!(player, "matte: frame $srcframe is outside this clip's matte")
        return false
    end
    matterepairs(player, clip)[Int(srcframe)] = Matrix{UInt8}(mask)
    refreshmattepanel!(player)
    notify(player.playhead)
    setstatus!(player, "matte: frame $srcframe repaired — kept through the next full run")
    return true
end

"""
Propagate the marks across the clip, on the analysis backend, off the UI thread.

The reader is the clip's own post-fx frame stream, so the model tracks the
subject through the stabilised, cropped picture the user is looking at.
"""
function runmatte!(ctx::ToolContext; clip = nothing, seeds = nothing)
    player = ctx.player
    if clip === nothing
        loc = editclip(player)
        loc === nothing && return setstatus!(player, "matte: no clip under the playhead")
        clip = loc[1]
    end
    marks = seeds === nothing ? mattemarks(player, clip) : seeds
    isempty(marks) &&
        return setstatus!(player, "matte: mark the subject on a frame first")
    player.matteinfo[] == "matting…" &&
        return setstatus!(player, "matte: already running — progress in the bottom right")
    player.matteinfo[] = "matting…"
    setstatus!(player, "matte: propagating from $(length(marks)) marked frame(s)")
    runanalysis(player) do
        try
            # No cap: the matte is drawn into the layer, so computing it smaller
            # is an edge rebuilt from fewer samples than the layer can show. It
            # is not a quality dial either — the propagator is causal, so a
            # different input resolution tracks the subject differently. Cost is
            # linear in matte area (607 ms/megapixel measured), which is what a
            # `maxside` cap buys back if a clip is ever too slow to matte.
            reader = framereader(clip, player.engine)
            track = analyzematte!(clip, reader, marks; progress = (d, t) -> begin
                player.jobprogress[] = d / max(t, 1)
            end)
            # A full run rebuilds every frame from the seeds, which would silently
            # throw away single-frame repairs. They are put back rather than
            # re-seeded: a repair says "this frame should look like this", not
            # "propagate from here", and the propagator has no way to be told the
            # first. Whoever repaired a frame did so because the propagation was
            # wrong there, so propagation does not get to overrule it.
            reps = matterepairs(player, clip)
            for (sf, m) in reps
                repairframe!(clip, sf, m)
            end
            put!(player.uiqueue, () -> begin
                findeffect(clip, MatteEffect) === nothing &&
                    push!(clip.effects, Effect(MatteEffect()))
                cov = mattecoverage(track)
                player.matteinfo[] = "matte: $(length(track.seeds)) marked frame(s), " *
                                     "$(size(track.alpha, 3)) frames, " *
                                     "$(round(Int, 100cov))% kept"
                player.jobprogress[] = NaN
                refreshmattepanel!(player)
                notify(player.playhead)
                # A matte that keeps everything (or nothing) renders as no visible
                # change, which reads as a broken tool — so report the number.
                setstatus!(player, if cov > 0.97 || cov < 0.02
                        "matte covers $(round(Int, 100cov))% of the frame — mark more of " *
                        "the subject, or a background point where it spills"
                    else
                        "matte ready — Matte/Feather are keyframable in the inspector"
                    end)
            end)
        catch e
            put!(player.uiqueue, () -> begin
                player.matteinfo[] = "matte failed"
                player.jobprogress[] = NaN
                setstatus!(player, "matte failed: $(sprint(showerror, e))")
            end)
        end
    end
    return nothing
end

"""
The card's ×: take the matte off this clip completely — a marking session in
progress, the marks it collected, the propagated track and the effect slot. The
card exists from the first click on, so this is also how you abandon one.
"""
function removematte!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "matte: no clip under the playhead")
    clip = loc[1]
    snapshot!(player)          # track, marks and the effect all come back together
    col = mattecollect(player)
    col === nothing || col.clip !== clip || cancelmattecollect!(col)
    clip.mattetrack = nothing
    delete!(player.mattemarks, clip.id)
    i = findfirst(s -> op(s) isa MatteEffect, clip.effects)
    i === nothing || deleteat!(clip.effects, i)
    player.matteinfo[] = "no matte"
    refreshmattepanel!(player; structure = true)
    notify(player.playhead)
    setstatus!(player, "matte removed")
    return nothing
end

"""
Colours for matte objects, cycled. The SAME colour identifies an object's points
in the preview, its bar in the card and its outline in the SAM 2 view — that is
the whole feedback that `+ object` did something.
"""
const MATTECOLORS = (RGBf(0.35, 0.86, 0.36), RGBf(0.30, 0.66, 1.00), RGBf(1.00, 0.78, 0.25),
                     RGBf(0.98, 0.45, 0.85), RGBf(0.40, 0.95, 0.90), RGBf(1.00, 0.55, 0.30))

mattecolor(obj::Integer) = MATTECOLORS[mod1(Int(obj), length(MATTECOLORS))]

"""
The object ids present in `points`, in the order the objects were started.
[`mattegroups`](@ref) orders its groups the same way, so group *i* is object
`matteobjectids(points)[i]` — the card used to recover the id with `unique`,
which is first-appearance order and disagrees the moment an object is deleted.
"""
matteobjectids(points::AbstractVector{<:Tuple{<:Real, <:Real, Bool, Int}}) =
    sort!(unique(q[4] for q in points))

"""
Which objects the card shows for a live marking session: every object with a
point, plus the one `+ object` just started. That last one has nothing in
`points` yet and would otherwise be invisible — pressing `+ object` would look
like it did nothing until the next click landed.
"""
function mattecardobjects(col)
    ids = matteobjectids(col.points)
    col.object > 0 && !(col.object in ids) && push!(ids, col.object)
    return ids
end

"""
Corner radius of the matte card's object pills.

**5, not 12.** At `height = 24` a radius of 12 is exactly half the height, i.e. a
full stadium — the roundest a rounded rectangle can be — and it read as a
different design language from every other control in the editor, which sits at
3–6. It also renders badly: at half-height the two arcs meet with no straight
edge between them, so the stroke has no flat run to sit on and the outline looks
lumpy where the curves join.
"""
const PILLRADIUS = 5

"""
A pill's height, and the gap between two of them.

They are constants together because the row that holds the pills is a NESTED
grid, and a nested grid reports one row's height however many it holds — so the
outer row has to be sized by hand, from exactly these two numbers. When that
arithmetic assumed a gap the grid did not actually have, each extra object
pushed the pills further past the bottom of their row and over the control
below: the "+ object" button drifting out of line with the pills above it is
what that looks like on screen.
"""
const PILLHEIGHT = 24
const PILLGAP = 4

"The exact height of a nested pill grid holding `n` rows of [`PILLHEIGHT`](@ref)."
pillrowsize(n::Integer) = Makie.Fixed((PILLHEIGHT + PILLGAP) * n - PILLGAP)

"""
ONE card for ONE marked frame: which objects were marked there, and the × that
un-marks that frame.

Per FRAME because a mark is per frame. Propagation drifts, you go back, you mark
another frame — the list of marks is the list of corrections, and each one has to
be removable on its own. This × used to take the entire matte off the clip
instead, so "drop this correction" deleted all the work; taking the whole effect
off is the effect card's own ×, one level up.
"""
function matteseedcard!(ctx::ToolContext, clip::Clip, seedframe::Integer, live::Bool)
    player = ctx.player
    # the FULL palette (the timeline's is a six-colour subset with no `text_muted`)
    colors = player.fxwidgets[:uicolors]
    col    = live ? mattecollect(player) : nothing
    ids    = col === nothing ? Int[] : mattecardobjects(col)
    pillwidgets = Any[]
    newobjbtn = Ref{Any}(nothing)

    card = toolcard!(ctx; caption = "frame $seedframe",
                     onremove = _ -> removematteseed!(player, clip, seedframe)) do g, blocks
        r = 0
        if col !== nothing
            # One PILL per object, in that object's own colour — the same colour
            # its points have in the preview and its outline has in the SAM 2
            # view. Selected is a brighter fill plus its own outline, so "which
            # object do my clicks land in" is answerable without reading text.
            # `measurable!`: an empty nested layout has no determinable height, and
            # one such cell makes the whole card indeterminate.
            pills = measurable!(GridLayout(g[r += 1, 1]))
            for (k, id) in enumerate(ids)
                n = count(q -> q[4] == id, col.points)
                # lit = WHERE THE NEXT CLICK LANDS (`object`), not `selected`,
                # which is the isolate-its-dots toggle. A fresh session and a
                # fresh `+ object` both light their pill at once, so pressing it
                # visibly did something before any point exists.
                sel = col.object == id
                base = mattecolor(id)
                fill = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)), base,
                                        sel ? 0.55 : 0.16)
                pill = Button(pills[k, 1]; label = "$n point$(n == 1 ? "" : "s")",
                              fontsize = 11, height = PILLHEIGHT, tellwidth = false,
                              width = Makie.Relative(1.0), cornerradius = PILLRADIUS,
                              buttoncolor = fill,
                              buttoncolor_hover = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)),
                                                                   base, 0.35),
                              buttoncolor_active = base,
                              labelcolor = sel ? RGBf(0.09, 0.09, 0.10) : colors.text,
                              strokewidth = sel ? 2 : 1,
                              strokecolor = sel ? base : (base, 0.4))
                on(_ -> selectmatteobject!(player, id), pill.clicks)
                bx = Button(pills[k, 2]; label = "×", fontsize = 11, width = 22,
                            height = PILLHEIGHT,
                            buttoncolor = (:transparent, 0.0), strokewidth = 0,
                            labelcolor = colors.text_muted,
                            buttoncolor_hover = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)),
                                                                 RGBf(1, 0.4, 0.35), 0.35))
                on(_ -> deletematteobject!(player, id), bx.clicks)
                push!(blocks, pill, bx)
                push!(pillwidgets, (; object = id, pill, remove = bx))
            end
            colsize!(pills, 1, Makie.Auto(false, 1.0))
            # SET the gap rather than assume one: `rowsize!` below computes the
            # outer row from it, and the default is not 4.
            rowgap!(pills, PILLGAP)

            # UNDER the pills, because that is where the thing it makes appears.
            # Further clicks on a subject REFINE it — that is what SAM 2 does with
            # several positive points — so "this is a different thing" has to be
            # said, not guessed.
            #
            # In the pills' OWN grid, column 1, not the card's outer grid. Outside
            # it, the button stretched the full card width while every pill above
            # stopped short by the `×` column, so the one control that makes a new
            # pill was the one control that did not line up with them.
            newobj = newobjbtn[] = Button(pills[length(ids) + 1, 1]; label = "+ object", fontsize = 11,
                            height = PILLHEIGHT, tellwidth = false, width = Makie.Relative(1.0),
                            cornerradius = PILLRADIUS, buttoncolor = (:transparent, 0.0),
                            # Outlined, to read as "makes a new one" rather than as
                            # another object — but the LABEL is full-strength text.
                            # At `text_muted` on a transparent fill it looked
                            # switched off, which is a bad way to draw the one
                            # control on this card that creates a pill.
                            strokewidth = 1, strokecolor = (colors.text, 0.45),
                            labelcolor = colors.text,
                            buttoncolor_hover = colors.surface)
            # …which makes the row one taller than the object count. Without an
            # explicit height the nested layout reported one row's worth however
            # many it held, and everything below was laid over what did not fit.
            rowsize!(g, r, pillrowsize(length(ids) + 1))
            on(newobj.clicks) do _
                c = mattecollect(player)
                c === nothing && return setstatus!(player, "matte: mark a subject first")
                c.object = (isempty(c.points) ? 0 : maximum(q[4] for q in c.points)) + 1
                c.selected = c.object
                refreshmattepanel!(player; structure = true)
                setstatus!(player, "matte: object $(c.object) — click the next subject")
            end
            push!(blocks, newobj)
        else
            # Not the frame being marked: the way back to it. `gotomatteseed!`
            # moves the playhead AND starts marking, which is the whole reason to
            # press it — so there is no separate "edit points" button.
            jump = Button(g[r += 1, 1]; label = "go to frame $seedframe", fontsize = 11,
                          height = 24, tellwidth = false, width = Makie.Relative(1.0))
            on(_ -> gotomatteseed!(ctx, clip, seedframe), jump.clicks)
            push!(blocks, jump)
        end
        nothing
    end
    return (; clip, srcframe = Int(seedframe), live,
            nobj = length(ids), pills = pillwidgets,
            # `newobj` is published for the same reason `pills` is — everything on
            # this card should be addressable by a test, a walkthrough or MCP. It
            # was the one control that MAKES a pill and the one that could not be
            # pressed except by a human with a mouse, which is precisely the
            # control a rendering check of the pills needs.
            newobj = newobjbtn[],
            removebtn = card === 0 ? nothing : card.rm)
end

"""
Un-mark ONE frame: the selection made there goes, the rest of the matte stays.

The effect, the other marks and the propagated track are untouched — removing a
correction must not delete the work it was correcting. When the LAST mark goes
the track is dropped too, because nothing the user asked for derives it any more,
but the effect slot stays: its × is the one that means "take this off the clip".
"""
function removematteseed!(player::Player, clip::Clip, frame::Integer)
    snapshot!(player)
    col = mattecollect(player)
    col === nothing || col.clip !== clip || col.srcframe != Int(frame) || cancelmattecollect!(col)
    marks = mattemarks(player, clip)
    delete!(marks, Int(frame))
    if isempty(marks)
        clip.mattetrack = nothing
        player.matteinfo[] = "no matte"
        setstatus!(player, "matte: last mark removed — mark a frame to key again")
    else
        setstatus!(player, "matte: frame $frame un-marked — Apply to propagate from the rest")
    end
    refreshmattepanel!(player; structure = true)
    notify(player.playhead)
    return nothing
end

"Switch the matte view; the preview and the card both follow it."
function setmatteview!(player::Player, v::Symbol)
    mattecardview(player)[] = v
    col = mattecollect(player)
    col === nothing || showmatteview!(col)
    refreshmattepanel!(player; structure = true)
    setstatus!(player, v === :sam2 ? "matte view: SAM 2 segmentation" : "matte view: matte")
    return nothing
end

"Select an object: its points come forward, the others dim."
function selectmatteobject!(player::Player, obj::Integer)
    col = mattecollect(player)
    col === nothing && return setstatus!(player, "matte: nothing being marked")
    col.selected = col.selected == obj ? 0 : Int(obj)
    col.object = col.selected == 0 ? col.object : Int(obj)   # further clicks refine THIS one
    refreshmattedots!(col)
    refreshmattepanel!(player; structure = true)
    setstatus!(player, col.selected == 0 ? "matte: no object selected" :
                       "matte: object $obj selected — clicks refine it")
    return nothing
end

"Drop an object and everything marked for it."
function deletematteobject!(player::Player, obj::Integer)
    col = mattecollect(player)
    col === nothing && return setstatus!(player, "matte: nothing being marked")
    filter!(q -> q[4] != obj, col.points)
    col.selected == obj && (col.selected = 0)
    # …and move the clicks somewhere that still exists: the card lights the pill
    # for `object`, so leaving it on the deleted one shows a phantom empty group
    # and the next click resurrects it.
    if col.object == obj
        rest = matteobjectids(col.points)
        col.object = isempty(rest) ? 1 : last(rest)
    end
    refreshmattedots!(col)
    isempty(col.points) ? clearlivematte!(col) : livematte!(col)
    refreshmattepanel!(player; structure = true)
    setstatus!(player, "matte: object $obj removed")
    return nothing
end

"""
Back to the frame this seed belongs to, ready to edit its points.

The card's picture is a FROZEN seed frame, not a live preview: moving the playhead
leaves it alone (its points come off the screen, since they describe that frame
and no other), and this is the way back to it.
"""
function gotomatteseed!(ctx::ToolContext, clip::Clip, seedframe::Integer)
    player = ctx.player
    n = clip.start + (Int(seedframe) - clip.src_in)
    player.playhead[] = clamp(n, clip.start, clipend(clip) - 1)
    mattecollect(player) === nothing && startmattepick!(ctx)
    setstatus!(player, "matte: back at frame $seedframe — its points are editable again")
    return nothing
end

"""
Repaint the card's picture and caption where they stand.

A checkbox must not rebuild the dock: the rebuild deletes and recreates every
tool's blocks, and a second click arriving mid-rebuild lands on a block that is
being replaced — one toggle in four was simply lost. Only the image and the
caption depend on the toggles, and both are observables.
"""
function repaintmattecard!(player::Player)
    w = get(player.fxwidgets, :mattecard, nothing)
    w === nothing && return false
    loc = editclip(player)
    # a different clip needs a different CARD, not a repaint
    (loc === nothing || loc[1] !== w.clip) && return false
    col = mattecollect(player)
    (col === nothing || col.clip !== w.clip || col.srcframe != w.srcframe) && return false
    # …and so does a new object: it brings its own pill, which a repaint cannot add
    ids = mattecardobjects(col)
    length(ids) == w.nobj || return false
    for p in w.pills
        n = count(q -> q[4] == p.object, col.points)
        p.pill.label[] = "$n point$(n == 1 ? "" : "s")"
    end
    return true
end

"""
Warm the propagation model once, while the user is still choosing where to click.

The size has to be the size the tool will really ask for: the cooperative-matrix
GEMM specializes per tile shape, so warming at a different resolution leaves the
specialization to be paid again on the first click — which is the click the user
is watching. That is `mattereadsize(clip, nothing)`, the same call `framereader`
and `analyzematte!` make, not a resolution of this function's own choosing.

With no clip selected there is no size to know; 480x270 is then a guess that at
least loads the model, and the first click pays for its own shape.
"""
function warmmattepanel!(ctx::ToolContext)
    player = ctx.player
    MATTEWARMED[] && return nothing
    loc0 = editclip(player)
    mw, mh = loc0 === nothing ? (480, 270) : mattereadsize(loc0[1], nothing)
    # The longest wait in the editor — MEASURED 67 s for the first matte of a
    # session (~49 s building the model, ~18 s compiling kernels for this clip's
    # shape) and ~18 s again for each new resolution. It showed only a line of
    # text in the card, which for a minute of silence reads as a hang. The footer
    # spinner is what the segmenting and propagation steps already use.
    player.matteinfo[] = "warming up the model… (up to a minute, first time)"
    player.jobprogress[] = 0.0
    setstatus!(player, "matte: warming up the model — the first one takes a minute")
    runanalysis(player) do
        try
            warmmatte!(mw, mh)
            put!(player.uiqueue, () -> (player.matteinfo[] = "ready — mark the subject";
                                        player.jobprogress[] = NaN;
                                        setstatus!(player, "matte: ready — mark the subject")))
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                player.matteinfo[] = "model warm-up failed: $(briefly(e))"
                player.jobprogress[] = NaN      # …or the spinner runs for ever
                @error "matte warm-up failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"Seed size as a fraction of frame width — a box the user can hit by clicking."
const MATTESEED = Ref{Float64}(0.25)

"""
A selection being marked.

The interaction is a **scene laid over the preview**, not a mode the rest of the
editor has to know about. `scene` is a child of the preview axis with
`captures_mouse = true`: while it is visible it claims the pointer, so
`Makie.receives_events` is false for every other handler over that viewport, and
false for its OWN handlers the moment it is hidden. "Am I marking?" is therefore
not a flag anyone can consult and get wrong — it is whether the scene is up.

Everything the interaction owns hangs off it: the two dot plots live in that
scene, the listeners are registered on it, and ending the marking hides the scene
and empties it in one step. Nothing to forget to delete, and no second feature
can be handed the same click.
"""
mutable struct MatteCollect
    ctx::ToolContext
    clip::Clip
    srcframe::Int
    scene::Makie.Scene                              # the overlay; owns plots + events
    points::Vector{Tuple{Float64, Float64, Bool, Int}}  # matte coords, foreground?, object
    fg::Observable{Vector{Point2f}}                 # canvas coords, for the dots
    bg::Observable{Vector{Point2f}}
    listeners::Vector{Any}
    prevtrack::Any                                  # restored on Esc
    prevmatte::Any                                  # the user's MatteEffect, dimmed while marking
    painting::Any                                   # true/false = a stroke of fg/bg, nothing = idle
    lastpaint::Any                                  # where the stroke last laid a point
    hadeffect::Bool
    busy::Bool                                      # a preview run is in flight
    pending::Bool                                   # …and the points changed while it ran
    # The seed the last preview was computed from, and the points it came from.
    # Kept so that pressing Enter propagates *the mask the user was just looking
    # at* rather than recomputing one — with a segmenter installed that is not a
    # cheap repeat, and worse, a recomputation is a chance for what propagates to
    # differ from what was previewed.
    lastseed::Any
    lastseedpoints::Vector{Tuple{Float64, Float64, Bool, Int}}
    object::Int      # the object being marked; `+ object` starts the next one
    selected::Int    # object whose bar is selected in the card (0 = none)
    lastmasks::Any   # per-object masks from the last preview, for the SAM 2 view
    # The post-fx frame the seed was computed from, kept so the card can draw a
    # picture without re-rendering: the render belongs to the worker that owns the
    # Lava context, and the card is built on the UI thread.
    lastframe::Any
end

"""
    mattecollect(player) -> MatteCollect | nothing

The selection being marked on `player`, if any.

Kept in `player.fxwidgets` — the slot this editor already uses for transient UI
state (`:activetool`, `:kfmenu`) — rather than a module-level `Ref`. State that
belongs to one player must not be reachable without one: a global would make two
players share a selection, and it would outlive the window that owns the dots.
"""
mattecollect(player::Player) = get(player.fxwidgets, :mattecollect, nothing)

"""
    matteoverlay(player) -> (scene, fg, bg)

The marking overlay: a child scene of the preview axis, plus the two point
vectors drawn in it. Built on first use and then **reused** — one per player, for
the player's life.

Not rebuilt per marking: a `Scene` adds itself to its parent's `children` and
there is no unregistering it by hiding it, so a scene per interaction is a leak
that grows with every click of "Mark subject". Idle it is invisible, which
`receives_events` reads as "not listening", and empty vectors draw nothing.

It shares the axis' camera so a dot sits in the same data coordinates the frame
is drawn in; a positive z plus `captures_mouse` put it on top for drawing *and*
for pointer routing.
"""
function matteoverlay(player::Player)
    return get!(player.fxwidgets, :matteoverlay) do
        ax = player.previewaxis
        # A SIBLING of the axis, not a child of it. `receives_events` lets a
        # covering scene through to anything on its own root-to-leaf path, so a
        # child would claim the pointer and still leave the axis' own handlers
        # (crop, pick, scrub) firing — the exact conflict this is here to end.
        # Same viewport and camera, so a dot still lands in frame coordinates.
        sc = Makie.Scene(player.fig.scene; viewport = ax.scene.viewport,
                         camera = ax.scene.camera, clear = false, visible = false)
        Makie.translate!(sc, 0, 0, 10)
        fg, bg = Observable(Point2f[]), Observable(Point2f[])
        # per-point colour vectors, and they stay vectors: Makie type-locks a
        # scalar-vs-vector attribute at creation, so starting scalar would refuse
        # the per-object colours later
        fgcolor = Observable(RGBAf[]); bgcolor = Observable(RGBAf[])
        scatter!(sc, fg; color = fgcolor, marker = :circle,
                 markersize = 10, strokewidth = 1, strokecolor = (:black, 0.6))
        scatter!(sc, bg; color = bgcolor, marker = :xcross,
                 markersize = 10, strokewidth = 1, strokecolor = (:black, 0.6))
        # one contour per palette colour, reused: the object count changes with
        # every click, and plots created per change in a scene that outlives them
        # are a leak that keeps drawing
        contours = map(1:length(MATTECOLORS)) do k
            ct = contour!(sc, Float32[0, 1], Float32[0, 1], zeros(Float32, 2, 2);
                          levels = [0.5f0], color = MATTECOLORS[k], linewidth = 2,
                          visible = false)
            Makie.translate!(ct, 0, 0, 5)
            ct
        end
        (scene = sc, fg = fg, bg = bg, fgcolor = fgcolor, bgcolor = bgcolor, contours = contours)
    end
end

"""
Start marking. Clicks land in the preview until the user is done:

  * left click marks the subject, right click marks what is *not* the subject
  * dragging paints a run of points
  * every point re-mattes THIS frame only, so the picture updates while marking
  * Enter propagates across the clip, Esc restores what was there before

The live matte goes through the ordinary render path — a one-frame `MatteTrack`
on the clip — rather than a bespoke overlay, so what is shown while marking is
literally what the finished key looks like, at the strength and feather the
inspector is set to.
"""
function startmattepick!(ctx::ToolContext)
    player = ctx.player
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "matte: move the playhead onto a clip first")
    mattecollect(player) === nothing || return finishmattecollect!(mattecollect(player))
    clip, srcframe = loc
    ax = player.previewaxis
    ov = matteoverlay(player)
    sc = ov.scene
    paintstep = 12   # canvas units between the points a drag lays down: dense
                     # enough that the discs overlap into a stroke, sparse enough
                     # that a long drag stays a few dozen points
    sc.visible[] = true
    sc.captures_mouse = true
    col = MatteCollect(ctx, clip, Int(srcframe), sc, Tuple{Float64, Float64, Bool, Int}[],
                       ov.fg, ov.bg, Any[],
                       clip.mattetrack, nothing, nothing, nothing,
                       findeffect(clip, MatteEffect) !== nothing, false, false,
                       nothing, Tuple{Float64, Float64, Bool, Int}[], 1, 0, nothing, nothing)
    push!(col.listeners, on(events(sc).mousebutton) do event
        Makie.receives_events(sc) || return Consume(false)
        event.button in (Mouse.left, Mouse.right) || return Consume(false)
        if event.action == Mouse.press && is_mouseinside(sc)
            col.painting = (event.button == Mouse.left)
            col.lastpaint = Point2f(mouseposition(ax.scene))
            addmattepoint!(col, col.lastpaint, col.painting; live = false)
            return Consume(true)
        elseif event.action == Mouse.release && col.painting !== nothing
            col.painting = nothing
            livematte!(col)          # ONE model run per stroke, not per point
            return Consume(true)
        end
        return Consume(false)
    end)
    # Dragging paints. The seed MatAnyone wants is a rough mask of the subject,
    # not a hint: measured on this clip, one disc gives a matte covering 0.3% of
    # the frame and the 25% box gives 4.9% — the model propagates what it is
    # given and grows nothing. So the gesture that produces a usable seed has to
    # be able to cover the subject, and that is a stroke.
    push!(col.listeners, on(events(sc).mouseposition) do _
        col.painting === nothing && return Consume(false)
        Makie.receives_events(sc) || return Consume(false)
        p = Point2f(mouseposition(ax.scene))
        last = col.lastpaint
        (last !== nothing && sum(abs2, p .- last) < paintstep^2) && return Consume(false)
        col.lastpaint = p
        addmattepoint!(col, p, col.painting; live = false, remove = false)
        return Consume(true)
    end)
    player.fxwidgets[:mattecollect] = col
    # The gesture depends on what makes the seed. A segmenter turns ONE click
    # into the object's boundary, so telling the user to drag would have them
    # painting a stroke the model does not need; without one the seed IS the
    # painted shape and a stroke is the only way to cover a subject.
    clicks = player.segmenter !== nothing
    player.matteinfo[] = clicks ? "marking: click the subject, right-click to exclude" :
                                  "painting: drag over the subject, right-drag to exclude"
    setstatus!(player, (clicks ? "matte: CLICK the subject (right-click excludes) · " :
                                 "matte: DRAG over the subject (right-drag excludes) · ") *
                       "Ctrl+Z or Backspace undoes · Enter propagates · Esc cancels")
    # deferred: this runs inside the button's own callback, and the refresh
    # rebuilds that button
    # The points describe ONE frame. Move the playhead and they come off the
    # screen (the card keeps the frozen picture, and its "go to frame" brings both
    # back), so the next frame can be seeded on its own instead of collecting
    # clicks that belong to somewhere else.
    push!(col.listeners, on(player.playhead) do n
        # NO `receives_events` guard here: hiding the scene is exactly what this
        # does, and a hidden scene receives nothing — the guard would make the
        # points impossible to bring back.
        onseed = clip.start + (col.srcframe - clip.src_in) == n
        sc.visible[] == onseed && return nothing
        sc.visible[] = onseed
        onseed || setstatus!(player, "matte: left frame $(col.srcframe) — its points are " *
                                     "kept; press the card's picture to go back")
        return nothing
    end)
    put!(player.uiqueue, () -> refreshmattepanel!(player; structure = true))
    return nothing
end


"""
Add one marked point and re-matte this frame — or take one away.

Clicking a point you already placed **removes** it, whichever button you use:
the gesture that put a point down takes it back, with no mode to enter and no
modifier to remember. Backspace drops the most recent one, for the "no, not
there" immediately after clicking.

The target is the **dot you can see**, `hit` canvas units wide — deliberately not
the much larger disc that dot paints into the seed. Matching the paint radius
made two neighbouring points impossible to place: on a cropped, stabilized clip
the preview is scaled up, so a second click aimed at a subject right next to the
first landed inside the first one's disc and deleted it instead of adding.
"""
function addmattepoint!(col::MatteCollect, p, foreground::Bool;
                       hit::Real = 14, remove::Bool = true, live::Bool = true)
    player = col.ctx.player
    # Matte space (post-fx, cropped), not source: `framereader` renders what the
    # user sees, so only the crop fit has to come off.
    nx, ny = previewtomatte(player, col.clip, col.srcframe, p)
    # compared where the user is pointing — on the canvas — not in source pixels,
    # so the hit area is the dot's size however far the preview is zoomed.
    # `remove = false` while a stroke is being painted: a drag lays points close
    # together on purpose and would otherwise erase the ones it just made.
    ondot = remove ? findlast(eachindex(col.points)) do i
            q = mattetopreview(player, col.clip, col.srcframe, col.points[i])
            (q[1] - p[1])^2 + (q[2] - p[2])^2 <= hit^2
        end : nothing
    if ondot === nothing
        push!(col.points, (nx, ny, foreground, col.object))
    else
        deleteat!(col.points, ondot)
    end
    refreshmattedots!(col)
    # `live = false` while painting: the model runs once when the stroke ends,
    # not once per point laid down.
    live && (isempty(col.points) ? clearlivematte!(col) : livematte!(col))
    return nothing
end

"Drop the most recently marked point."
function dropmattepoint!(col::MatteCollect)
    isempty(col.points) && return nothing
    pop!(col.points)
    refreshmattedots!(col)
    isempty(col.points) ? clearlivematte!(col) : livematte!(col)
    return nothing
end

"""
Draw the SAM 2 view: each object's boundary, in its object's colour, over the
plain frame — and take the live matte off while it is up, since the two are two
ways of looking at the same thing.

The outlines are `contour!` plots over each object's mask, placed at the clip's
crop rect in preview coordinates (the masks live in matte space). A fixed pool,
one per palette colour, because the number of objects changes with every click
and creating plots per change in a scene that outlives them is a leak.
"""
function showmatteview!(col::MatteCollect)
    player = col.ctx.player
    ov = matteoverlay(player)
    sam2 = mattecardview(player)[] === :sam2
    masks = col.lastmasks
    lay = (col.clip.source.width, col.clip.source.height)
    cr = col.clip.crop
    for (k, ct) in enumerate(ov.contours)
        m = (sam2 && masks !== nothing && k <= length(masks)) ? masks[k] : nothing
        if m === nothing
            ct.visible[] = false
            continue
        end
        mw, mh = size(m)
        ct[1][] = range(cr[1] * lay[1], (cr[1] + cr[3]) * lay[1], length = mw)
        ct[2][] = range(cr[2] * lay[2], (cr[2] + cr[4]) * lay[2], length = mh)
        ct[3][] = Float32.(m)
        ct.color[] = mattecolor(k)
        ct.visible[] = true
    end
    # the matte is the OTHER view of the same thing: SAM 2 up means the plain
    # frame with outlines, and switching back has to put the live matte BACK —
    # leaving it off was "the toggle does nothing" in the other direction
    if sam2
        col.clip.mattetrack === col.prevtrack || clearlivematte!(col)
    elseif col.lastseed !== nothing && col.clip.mattetrack === col.prevtrack
        livematte!(col)
    end
    matteinfo!(col)
    notify(player.playhead)
    return nothing
end

"""
The selection as the segmenter takes it: one vector of `(x, y, foreground)` per
object, in the order the objects were started.
"""
function mattegroups(points::AbstractVector{<:Tuple{<:Real, <:Real, Bool, Int}})
    ids = sort!(unique(q[4] for q in points))
    return [[(q[1], q[2], q[3]) for q in points if q[4] == id] for id in ids]
end

"""
Redraw the dots from `col.points`.

Rebuilt from the source of truth rather than pushed/popped alongside it: the two
lists disagreeing is the classic way a removed point keeps its dot. Canvas
positions come back out of the same mapping that put them in, so a dot sits where
its point is even after the playhead moved or the clip was reframed.
"""
function refreshmattedots!(col::MatteCollect)
    player = col.ctx.player
    ov = matteoverlay(player)
    fg, bg = Point2f[], Point2f[]
    fgc, bgc = RGBAf[], RGBAf[]
    for (nx, ny, isfg, obj) in col.points
        q = Point2f(mattetopreview(player, col.clip, col.srcframe, (nx, ny)))
        # dimmed unless it belongs to the selected object — that is what selecting
        # a bar in the card SHOWS
        a = (col.selected == 0 || col.selected == obj) ? 0.95f0 : 0.30f0
        c = RGBAf(mattecolor(obj), a)
        push!(isfg ? fg : bg, q)
        push!(isfg ? fgc : bgc, c)
    end
    ov.fgcolor[] = fgc
    ov.bgcolor[] = bgc
    col.fg[] = fg
    col.bg[] = bg
    return nothing
end

"Back to the picture the clip had before this marking, with no points left."
function clearlivematte!(col::MatteCollect)
    col.clip.mattetrack = col.prevtrack
    restorematteeffect!(col)
    matteinfo!(col)
    notify(col.ctx.player.playhead)
    return nothing
end

"The one place that says what is being marked, so no path can leave it stale."
function matteinfo!(col::MatteCollect)
    n = length(col.points)
    col.ctx.player.matteinfo[] = n == 0 ? "marking — no points" :
        "marking — $n point(s) on $(length(mattegroups(col.points))) object(s), " *
        "frame $(col.srcframe)"
    return nothing
end

"""
Re-matte the marked frame with the points so far.

Coalescing rather than queueing: while a run is in flight further clicks only set
`pending`, and one more run happens when it lands. Marking is faster than the
model, and a queue would spend the whole interaction showing states the user has
already clicked past.
"""
function livematte!(col::MatteCollect)
    player = col.ctx.player
    if col.busy
        col.pending = true
        return nothing
    end
    isempty(col.points) && return nothing
    col.busy = true
    # The points are copied, and the seed is built on the analysis executor
    # rather than here: a segmenter needs the frame, and the frame is decoded
    # there. Copying is what makes the cached seed trustworthy — `col.points` is
    # mutated by every click, so keeping a reference would let `lastseedpoints`
    # silently agree with points the seed was never computed from.
    points = copy(col.points)
    # SAY something before the model runs. Discs were instant, so a silent path
    # was honest; a segmenter takes ~0.6 s warm, and the first click of a session
    # also builds and compiles the model — measured at 58 s. A dot appearing and
    # then nothing at all for a minute reads as a tool that does not work.
    setstatus!(player, segmenterready(player.segmenter) ?
                       "matte: segmenting…" :
                       "matte: building the SAM 2 model — " *
                       "first click of the session, this one takes a while")
    player.jobprogress[] = 0.0   # footer spinner, so the wait is visible too
    runanalysis(player) do
        try
            frame = framereader(col.clip, player.engine)(col.srcframe)
            mask, per = seedmasks(col.clip, frame, mattegroups(points);
                                  segmenter = player.segmenter,
                                  key = (col.clip.id, col.srcframe))
            alpha = previewmatte(col.clip, frame, mask)
            put!(player.uiqueue, () -> begin
                if mattecollect(player) === col         # not cancelled meanwhile
                    col.lastseed = mask
                    col.lastseedpoints = points
                    col.lastframe = frame
                    col.lastmasks = per
                    showlivematte!(col, alpha)
                    showmatteview!(col)
                    refreshmattepanel!(player)      # the card gains the seed outline
                    setstatus!(player, "matte: $(length(points)) point$(length(points) == 1 ? "" : "s") — " *
                               "selection covers $(round(100 * count(!=(0x00), mask) / length(mask); digits = 1))%" *
                               " · Enter to propagate" *
                               (col.prevtrack === nothing ? "" : " · Shift+Enter for this frame only"))
                end
                player.jobprogress[] = NaN
                col.busy = false
                if col.pending
                    col.pending = false
                    livematte!(col)
                end
            end)
        catch e
            bt = catch_backtrace()
            put!(player.uiqueue, () -> begin
                player.jobprogress[] = NaN
                col.busy = false
                setstatus!(player, "matte: $(briefly(e))")
                @error "matte preview failed" exception = (e, bt)
            end)
        end
    end
    return nothing
end

"""
    showlivematte!(col, alpha; strength = 0.85f0)

Install a one-frame matte track so the ordinary render path shows the selection.

`strength` is how much of the background is keyed away *while marking*: keying it
to black says what is IN the selection and nothing about what is next to it, and
what you judge while marking is exactly the edge — i.e. what got left out. At
0.85 the background stays readable at ~15%. The user's own strength is put back
the moment marking ends, and this value reaches the matte kernel as a plain
argument, so it must not come from anywhere mutable.
"""
function showlivematte!(col::MatteCollect, alpha::Matrix{UInt8}; strength = 0.85f0)
    player = col.ctx.player
    track = MatteTrack(reshape(alpha, size(alpha, 1), size(alpha, 2), 1),
                       col.srcframe, [col.srcframe])
    col.clip.mattetrack = track
    if col.prevmatte === nothing                       # first preview of this marking
        # `findeffect` hands back the EFFECT (`findslot` is the one that returns
        # the slot) — reaching for `.effect` here asked a `MatteEffect` for a
        # field it has never had, on the very first marked point.
        col.prevmatte = something(findeffect(col.clip, MatteEffect), MatteEffect())
        seteffect!(col.clip, MatteEffect(; strength, feather = col.prevmatte.feather))
    end
    nobj = isempty(col.points) ? 1 : length(unique(q[4] for q in col.points))
    player.matteinfo[] = "marking — $(length(col.points)) point(s) on $nobj object(s), " *
                         "this frame only"
    notify(player.playhead)
    return nothing
end

"Put the user's own matte strength back after the dimmed marking preview."
function restorematteeffect!(col::MatteCollect)
    col.prevmatte === nothing && return nothing
    seteffect!(col.clip, col.prevmatte)
    col.prevmatte = nothing
    return nothing
end

"Propagate the marked points across the clip."
function finishmattecollect!(col::MatteCollect)
    endmattecollect!(col)
    restorematteeffect!(col)
    isempty(col.points) &&
        return setstatus!(col.ctx.player, "matte: nothing marked")
    # the live one-frame track was never the clip's matte; propagation replaces it
    col.clip.mattetrack = col.prevtrack
    if col.lastseed !== nothing && col.lastseedpoints == col.points
        # exactly the mask the preview was showing a moment ago
        addmatteseed!(col.ctx, col.clip, col.srcframe, col.lastseed)
        return nothing
    end
    # Enter before a preview landed for these points. Build the seed on the
    # analysis executor, never here: this runs on the UI thread, and with a
    # segmenter installed the seed is a model call — half a second of a frozen
    # window, on the frame the user is looking at.
    ctx, clip, srcframe = col.ctx, col.clip, col.srcframe
    points = copy(col.points)
    player = ctx.player
    setstatus!(player, "matte: reading the marked frame…")
    runanalysis(player) do
        try
            frame = framereader(clip, player.engine)(srcframe)
            mask = seedmask(clip, frame, mattegroups(points);
                            segmenter = player.segmenter, key = (clip.id, Int(srcframe)))
            put!(player.uiqueue, () -> addmatteseed!(ctx, clip, srcframe, mask))
        catch e
            put!(player.uiqueue,
                 () -> setstatus!(player, "matte: seeding failed: $(sprint(showerror, e))"))
        end
    end
    return nothing
end

"""
    repairmattecollect!(col) -> Bool

End the marking session by writing THIS FRAME only, instead of propagating.

The other exit from a marking session, and the one a finished matte usually
wants. `finishmattecollect!` adds the marks as a seed and re-propagates the whole
clip, which is right when the tracking went wrong and everything after it
drifted, and wrong for the ordinary case: one frame in the middle came out broken
while the frames either side are fine. Re-propagating then costs what the first
run cost and risks changing frames that were already correct.

The mask is the one the preview is already showing — the same `lastseed` the
propagating exit uses — so what lands is what you were looking at when you
pressed the key. No preview yet means no repair: the alternative is running the
model on the UI thread, and that is half a second of frozen window.
"""
function repairmattecollect!(col::MatteCollect)
    player = col.ctx.player
    endmattecollect!(col)
    restorematteeffect!(col)
    col.clip.mattetrack = col.prevtrack          # drop the live one-frame track
    if col.prevtrack === nothing
        setstatus!(player, "matte: nothing to repair yet — Enter propagates the first run")
        return false
    end
    mask = col.lastseed
    if mask === nothing || col.lastseedpoints != col.points
        setstatus!(player, "matte: wait for the selection to appear, then repair")
        return false
    end
    ok = repairmatteat!(player, mask)
    ok && notify(player.playhead)
    return ok
end

"Discard the marking and put back what the preview showed before it."
function cancelmattecollect!(col::MatteCollect)
    endmattecollect!(col)
    restorematteeffect!(col)
    col.clip.mattetrack = col.prevtrack
    if !col.hadeffect                      # we added it; take it back off again
        i = findfirst(s -> op(s) isa MatteEffect, col.clip.effects)
        i === nothing || deleteat!(col.clip.effects, i)
    end
    col.ctx.player.matteinfo[] = "marking cancelled"
    notify(col.ctx.player.playhead)
    setstatus!(col.ctx.player, "matte: marking cancelled")
    return nothing
end

"""
Leave marking mode.

Taking the scene down takes the dots, the listeners and the pointer claim with
it — which is the reason they live there rather than side by side in the player,
each needing its own line here and its own way to be forgotten.
"""
function endmattecollect!(col::MatteCollect)
    foreach(Observables.off, col.listeners)
    empty!(col.listeners)
    col.scene.captures_mouse = false      # stops claiming the pointer…
    col.scene.visible[] = false           # …and `receives_events` goes false with it
    col.fg[] = Point2f[]                  # the plots stay, their contents do not
    col.bg[] = Point2f[]
    delete!(col.ctx.player.fxwidgets, :mattecollect)
    return nothing
end

"Record a mark at `srcframe` and re-propagate the clip."
function addmatteseed!(ctx::ToolContext, clip::Clip, srcframe::Integer, mask)
    player = ctx.player
    snapshot!(player)          # a committed mark is an edit, undoable like any other
    seeds = mattemarks(player, clip)
    seeds[Int(srcframe)] = mask
    runmatte!(ctx; clip = clip, seeds = seeds)
    return nothing
end


"""
Bring the matte UI up to date.

`structure = true` when the CARD itself must appear or disappear (a matte was
removed, marking was active, another clip was selected) — that is the only case
worth a dock rebuild. Everything else is a repaint of the card that is already
there: a rebuild recreates every tool's blocks and its scene plots, and it throws
the user's scroll position away, which turns "click a checkbox" into "scroll back
down to the card again".
"""
function refreshmattepanel!(player::Player; structure::Bool = false)
    r = get(player.fxwidgets, :fxlistrefresh, nothing)
    r === nothing || r()                      # inspector: the MatteEffect card
    (structure || !repaintmattecard!(player)) && (EFFECTS.version[] += 1)
    return nothing
end

"""
The crop card: what the next drag will change, and how big the project is now.

The crop tool had no card at all — it was a toolbar button and a status line, and
the status line is where its one irreversible act (resizing the project) was
reported and then overwritten by the next message. Two things needed a home on
screen: the SCOPE, because "crop this clip" and "resize the project" are
different intents that were one gesture, and the CANVAS SIZE, because a project
size you cannot see is one you cannot check before exporting.

The scope is a two-button toggle rather than a modifier on the drag, for the
reason the matte brush needed a row of its own: a gesture nobody can see is a
gesture only its author knows about.
"""
function croppanel!(ctx::ToolContext)
    player = ctx.player
    seq = player.sequence
    colors = player.fxwidgets[:uicolors]
    scope = cropscope(player)
    w, h = canvassize(seq)
    toollabel!(ctx, "project canvas $(w)×$(h)" *
                    (seq.canvas === nothing ? " (from the first clip)" : ""))

    slot = toolslot(ctx, :controls)
    if slot !== nothing
        row = GridLayout(slot[length(ctx.controls) + 1, 1])
        off = Makie.RGBf(0.22, 0.23, 0.26)
        b1 = Button(row[1, 1]; label = "Whole project", fontsize = 11, height = 22,
                    tellwidth = false, width = Makie.Relative(1.0),
                    buttoncolor = scope[] === :canvas ? colors.accent : off)
        b2 = Button(row[1, 2]; label = "This clip", fontsize = 11, height = 22,
                    tellwidth = false, width = Makie.Relative(1.0),
                    buttoncolor = scope[] === :clip ? colors.accent : off)
        on(_ -> setcropscope!(player, :canvas), b1.clicks)
        on(_ -> setcropscope!(player, :clip), b2.clicks)
        # ONE entry — see `matteviewrow!` on what an empty row does to the layout.
        push!(ctx.controls, row)
    end

    # The ratio row, under the scope row: first WHAT the drag changes, then WHAT
    # SHAPE it comes out — the order the two decisions are actually made in.
    slot2 = toolslot(ctx, :controls)
    if slot2 !== nothing
        arow = GridLayout(slot2[length(ctx.controls) + 1, 1])
        off = Makie.RGBf(0.22, 0.23, 0.26)
        lock = cropaspect(player)
        for (j, (lbl, val)) in enumerate(CROPRATIOS)
            on_ = val === nothing ? lock[] === nothing :
                  lock[] !== nothing && isapprox(lock[], val; rtol = 1e-3)
            b = Button(arow[1, j]; label = lbl, fontsize = 10, height = 22,
                       tellwidth = false, width = Makie.Relative(1.0),
                       buttoncolor = on_ ? colors.accent : off)
            on(_ -> setcropaspect!(player, val), b.clicks)
        end
        push!(ctx.controls, arow)
    end

    toolaction!(ctx, player.tool[] === :crop ? "Crop tool (active)" : "Crop tool",
                () -> usetool!(player, :crop))
    seq.canvas === nothing ||
        toolaction!(ctx, "Reset canvas to the first clip", () -> resetcanvas!(player);
                    footer = true)
    return nothing
end

"""
The shapes the crop tool can be locked to, and what they are for.

`nothing` is free-drag. The other three are the deliveries a shot actually gets
cut for — a wide timeline, a phone, and a square post — rather than a list of
every ratio that exists, which is a menu nobody reads.
"""
const CROPRATIOS = (("Free", nothing), ("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0))

"Name a locked ratio — one of [`CROPRATIOS`](@ref) if it is one, else the number.
 `first` over a filtered generator would throw on a ratio nobody listed, which is
 a status line taking down the click that produced it."
ratiolabel(a::Real) =
    (i = findfirst(r -> r[2] !== nothing && isapprox(r[2], a; rtol = 1e-3), CROPRATIOS);
     i === nothing ? string(round(Float64(a); digits = 3), ":1") : CROPRATIOS[i][1])

"""
    setcropaspect!(player, a) -> nothing

Lock the crop tool to a shape (or to `nothing`, for free-drag).

Locking RESHAPES the framing already in force rather than waiting for the next
drag. Choosing 9:16 and seeing nothing happen reads as a control that did not
work — and the whole reason to pick a ratio is to see the shot in it.
"""
function setcropaspect!(player::Player, a)
    cropaspect(player)[] = a === nothing ? nothing : Float64(a)
    player.tool[] === :crop || usetool!(player, :crop)
    loc = editclip(player)
    if a !== nothing && loc !== nothing
        clip = loc[1]
        snapshot!(player)
        clip.crop = lockaspect(clip.crop, clip.source, a)
        cropscope(player)[] === :canvas && (player.sequence.canvas =
            (max(2 * (round(Int, clip.crop[3] * clip.source.width) ÷ 2), 2),
             max(2 * (round(Int, clip.crop[4] * clip.source.height) ÷ 2), 2)))
        applycrop!(player, clip)
        refreshedit!(player)
    end
    showcurrentcrop!(player)
    sz = canvassize(player.sequence)
    setstatus!(player, a === nothing ? "crop: drag any shape" :
                       "crop locked to $(ratiolabel(a)) — $(sz[1])×$(sz[2])")
    refreshcroppanel!(player)
    return nothing
end

"""
    setcropscope!(player, s) -> Symbol

Point the crop tool at the project or at one clip, and say which on the preview.

Switching scope also brings the tool up: choosing what a crop will change is
something you do because you are about to crop, and making that a second click
was one click of ceremony on every use.
"""
function setcropscope!(player::Player, s::Symbol)
    cropscope(player)[] = s
    # NOT `usetool!`, which TOGGLES: changing scope while the crop tool is already
    # up would have put it away, i.e. the control would cancel the thing it
    # configures.
    player.tool[] === :crop || usetool!(player, :crop)
    showcurrentcrop!(player)
    setstatus!(player, s === :canvas ?
        "crop resizes the WHOLE PROJECT — drag past the edge to make it bigger" :
        "crop reframes THIS CLIP only — the project keeps its size")
    refreshcroppanel!(player)
    return s
end

registereffect!(EffectKind(:crop, "Crop";
    description = "Drag a rectangle on the preview. The rectangle may reach OUTSIDE the " *
        "picture — that is how the canvas grows, and the new area comes in empty, " *
        "exactly as it exports. Choose whether the drag resizes the whole project " *
        "or reframes only the clip you dragged on.",
    body = croppanel!,
    # No `make`/`matches`: crop is not an effect in the stack. `clip.crop` is a
    # field, and the canvas is the sequence's — so this kind is a card and a
    # gesture, and `addablekinds` correctly leaves it out of the Add-effect menu.
    activate = ctx -> usetool!(ctx.player, :crop)))

# ONE kind: the parameters that tune the matte AND the card that produces it.
# Split across two registries these were two entries with the same name — a
# "Matte" row of sliders in the Inspector that could not make a matte, and a
# "Matte" card in Tools that could not tune one.
registereffect!(EffectKind(:matte, "Matte";
    description = "Isolates a subject on the SELECTED clip. CLICK the subject (right-click " *
        "marks what is NOT it), then Enter: SAM 2 turns each click into an object " *
        "boundary, and that seed is propagated across the clip. Mark another frame " *
        "wherever it drifts. Both models see the clip AFTER its effects — cropped and " *
        "stabilized — so a tighter crop is also a faster, easier matte.",
    params = [FxParam(:strength, "Matte"; min = 0.0, max = 1.0, default = 1.0),
              FxParam(:feather, "Feather"; min = 0.0, max = 1.0, default = 0.0)],
    kfkeys = [:matte_strength, :matte_feather],
    make = nt -> MatteEffect(Float32(nt.strength), Float32(nt.feather)),
    matches = e -> e isa MatteEffect,
    read = e -> (strength = Float64(e.strength), feather = Float64(e.feather)),
    body = mattepanel!, activate = startmattepick!, analysis = true))


# --------------------------------------------------------------- Restore tool

"""
Run a restoration model over frames around the playhead.

Temporal models want a window, not a frame, and the exported graph pins how long
that window is — so the tool restores `restorewindowlength()` frames centred on
the playhead rather than letting the user pick. Frames outside the cache render
as decoded, so this is additive: restore the stretch you are working on, leave
the rest.

`RestoreEffect` goes on the clip at the first successful run, the same way the
matte tool adds its effect, so the result is visible without a second step.
"""
function restorepanel!(ctx::ToolContext)
    player = ctx.player
    toollabel!(ctx, player.restoreinfo)
    toolaction!(ctx, "Restore around playhead", () -> runrestore!(ctx))
    toolaction!(ctx, "Clear restored frames", () -> begin
        loc = editclip(player)
        loc === nothing && return setstatus!(player, "restore: no clip under the playhead")
        clearrestore!(loc[1])
        i = findfirst(s -> op(s) isa RestoreEffect, loc[1].effects)
        i === nothing || deleteat!(loc[1].effects, i)
        player.restoreinfo[] = "no restoration"
        refreshmattepanel!(player)
        notify(player.playhead)
        setstatus!(player, "restored frames cleared")
    end)
    hasrestoremodel() ||
        toollabel!(ctx, "no model installed — see examples/basicvsrpp.jl")
    return nothing
end

function runrestore!(ctx::ToolContext)
    player = ctx.player
    hasrestoremodel() ||
        return setstatus!(player, "restore: no model installed (examples/basicvsrpp.jl)")
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "restore: move the playhead onto a clip")
    clip, srcframe = loc
    player.restoreinfo[] == "restoring…" &&
        return setstatus!(player, "restore: already running")
    n = restorewindowlength()
    first = clamp(srcframe - n ÷ 2, clip.src_in, max(clip.src_in, clip.src_out - n))
    player.restoreinfo[] = "restoring…"
    setstatus!(player, "restore: $n frames from $(first)")
    runanalysis(player) do
        try
            reader = framereader(clip, player.engine)
            got = restorewindow!(clip, reader, first, n;
                                 progress = (d, t) -> (player.jobprogress[] = d / max(t, 1)))
            put!(player.uiqueue, () -> begin
                findeffect(clip, RestoreEffect) === nothing &&
                    push!(clip.effects, Effect(RestoreEffect()))
                player.restoreinfo[] = "$got frames restored from $(first) (x$(restorescale()))"
                player.jobprogress[] = NaN
                refreshmattepanel!(player)
                notify(player.playhead)
                setstatus!(player, "restored $got frames — Restore is keyframable in the inspector")
            end)
        catch e
            put!(player.uiqueue, () -> begin
                player.restoreinfo[] = "restore failed"
                player.jobprogress[] = NaN
                setstatus!(player, "restore failed: $(sprint(showerror, e))")
            end)
        end
    end
    return nothing
end

registereffect!(EffectKind(:restore, "Restore";
    description = "Runs an upscaling/restoration model over frames around the playhead on the " *
        "SELECTED clip. Needs a model installed (examples/basicvsrpp.jl).",
    params = [FxParam(:strength, "Restore"; min = 0.0, max = 1.0, default = 1.0)],
    kfkeys = [:restore_strength],
    make = nt -> RestoreEffect(Float32(nt.strength)),
    matches = e -> e isa RestoreEffect,
    read = e -> (strength = Float64(e.strength),),
    body = restorepanel!, activate = runrestore!, analysis = true))
