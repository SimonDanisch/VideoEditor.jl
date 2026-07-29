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
A registered GUI tool: `activate(ctx::ToolContext)` draws its hints and wires
its clicks; `deactivate(ctx)` is extra teardown (plots and handlers recorded
through the context are removed automatically). Register with
[`registertool!`](@ref); the Tools dock lists it live.
"""
struct EditorTool
    name::Symbol
    label::String
    description::String
    panel::Any       # fills the card body — ALWAYS, armed or not (project state)
    activate::Any    # what clicking the card header DOES
    deactivate::Any
end

const TOOLS = EditorTool[]
const TOOLBYNAME = Dict{Symbol, EditorTool}()
const TOOLSVERSION = Observables.Observable(0)   # bumped on every (re)registration

"""
    registertool!(name, label, description; activate, deactivate = ctx -> nothing)

Register a GUI tool — a premade editing operation that shows hints on the
thumb track and acts on the user's clicks. Live: callable any time, from any
package or the MCP agent, and the Tools panel picks it up immediately.
"""
function registertool!(name::Symbol, label::AbstractString, description::AbstractString;
                       activate, deactivate = ctx -> nothing, panel = ctx -> nothing)
    t = EditorTool(name, String(label), String(description), panel, activate, deactivate)
    TOOLBYNAME[name] = t
    i = findfirst(q -> q.name == name, TOOLS)
    i === nothing ? push!(TOOLS, t) : (TOOLS[i] = t)
    TOOLSVERSION[] = TOOLSVERSION[] + 1
    return t
end

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
    activatetool!(player, name) -> ToolContext | nothing

Run what clicking tool `name`'s header does. Tools that stay ON (the loop finder
draws hints and reads clicks) hold the armed slot until clicked again; tools that
just DO something (blend) act and hand the slot straight back — a card's body
content does not depend on this either way, it is built by the tool's `panel`.
"""
function activatetool!(player::Player, name::Symbol)
    cur = activetool(player)
    if cur !== nothing && cur[1].name === name
        deactivatetool!(player)
        setstatus!(player, "$(cur[1].label) off")
        return nothing
    end
    deactivatetool!(player)
    tool = TOOLBYNAME[name]
    ctx = ToolContext(player, name)
    player.fxwidgets[:activetool] = (tool, ctx)
    activetoolname(player)[] = name
    tool.activate(ctx)
    return ctx
end

# --------------------------------------------------------------- Tools panel

"The Tools dock: one CARD per registered tool — the same section visual as the
inspector's effect stack. The header row [▾ · Tool name] arms the tool (click
again puts it away); the body holds the tool's description, its OWN action slot
and its OWN preview cards, so nothing of one tool renders under another's.
Rebuilt live on [`registertool!`](@ref)."
function buildtoolspanel!(player::Player, gridpos, uicolors)
    # tellheight = TRUE on purpose: the dock is a scrollable Subfigure and derives
    # its content size from the layout's determined height. With the height hidden
    # the Subfigure thought the content was 0 tall, so nothing scrolled and long
    # cards were simply cut off at the bottom.
    panel = GridLayout(gridpos; valign = :top)
    Label(panel[1, 1], "Tools"; font = :bold, halign = :left, tellwidth = false)
    rows = GridLayout(panel[2, 1])
    colsize!(panel, 1, Makie.Relative(1.0))
    active = activetoolname(player)
    built = Any[]
    folded = Dict{Symbol, Bool}()
    actionslots = Dict{Symbol, Any}()   # tool name → its body control slot
    rowslots = Dict{Symbol, Any}()      # tool name → its body LIST rows
    cardslots = Dict{Symbol, Any}()     # tool name → its body image-card rows
    panelctxs = Dict{Symbol, ToolContext}()   # tool name → its always-on card content
    cardbg = Makie.lerp_oklab(RGBf(Makie.to_color(uicolors.background)),
                              RGBf(1, 1, 1), 0.075)
    scene = player.dockpanels[:tools].sf.scene
    cards = Any[]   # (id, box, im, label, onclick, removebtn) — image cards
    player.fxwidgets[:toolslots] = (actionslots, rowslots)
    player.fxwidgets[:toolcards] = (cardslots, scene, cards)
    player.fxwidgets[:toolpanels] = panelctxs
    function rebuild()
        for ctx in values(panelctxs)     # drop the previous cards' content + listeners
            cleartoolcontext!(ctx)
        end
        # …and the cards themselves. Their image plots live in the DOCK SCENE, not
        # in the panel layout, so rebuilding the panels leaves them drawn: after
        # "Remove matte" a mark's thumbnail kept floating between two other tools'
        # cards, belonging to nothing.
        cleartoolcards!(player)
        empty!(panelctxs)
        foreach(Makie.delete!, built)
        empty!(built)
        empty!(actionslots); empty!(rowslots); empty!(cardslots)
        for (k, tool) in enumerate(TOOLS)
            open = !get(folded, tool.name, false)
            card = GridLayout(rows[k, 1])
            push!(built, card)
            Box(card[1:(open ? 2 : 1), 1]; color = cardbg,
                strokecolor = uicolors.border, strokewidth = 1, cornerradius = 6,
                tellwidth = false, tellheight = false)
            Box(card[1, 1]; color = uicolors.surface, strokewidth = 0,
                cornerradius = 5, tellwidth = false, tellheight = false)
            hgl = GridLayout(card[1, 1]; alignmode = Makie.Outside(8, 8, 5, 5))
            fold = Button(hgl[1, 1]; label = open ? "▾" : "▸", width = 24)
            on(fold.clicks) do _
                folded[tool.name] = open
                activetoolname(player)[] === tool.name && deactivatetool!(player)
                rebuild()
            end
            arm = Button(hgl[1, 2]; label = tool.label, tellwidth = false,
                         width = Makie.Relative(1.0), halign = :left, font = :bold)
            on(_ -> activatetool!(player, tool.name), arm.clicks)
            on(active; update = true) do a
                armed = a === tool.name
                arm.buttoncolor[] = armed ? uicolors.accent : uicolors.surface
                arm.labelcolor[] = armed ? uicolors.text_on_accent : uicolors.text
            end
            colsize!(card, 1, Makie.Relative(1.0))
            open || continue
            body = GridLayout(card[2, 1]; alignmode = Makie.Outside(10, 10, 10, 6))
            # `halign` places the BLOCK; `justification` sets the lines inside it,
            # and its default centres them — which is why a wrapped description
            # read as ragged centred text floating in the card instead of a
            # left-aligned paragraph under the header.
            Label(body[1, 1], wraptext(tool.description); fontsize = 11, halign = :left,
                  justification = :left, color = uicolors.text_muted, tellwidth = false)
            # A slot starts EMPTY, and an empty GridLayout has no determinable
            # height — which makes the whole card, the panel and finally the dock's
            # content size indeterminate, so the Subfigure never scrolls and long
            # cards are simply cut off. A zero-size spacer in a side column keeps
            # every slot measurable while claiming nothing.
            actionslots[tool.name] = measurable!(GridLayout(body[2, 1]))
            rowslots[tool.name] = measurable!(GridLayout(body[3, 1]))
            cardslots[tool.name] = measurable!(GridLayout(body[4, 1]))
            # the card shows the PROJECT's state, not the tool's mode: its content
            # is built now and stays, armed or not
            ctx = ToolContext(player, tool.name)
            panelctxs[tool.name] = ctx
            try
                tool.panel(ctx)
            catch e
                @error "tool panel failed" tool = tool.name exception = (e, catch_backtrace())
            end
        end
        return
    end
    on(_ -> rebuild(), TOOLSVERSION)
    rebuild()
    on(active) do a   # activating a folded tool unfolds its card first
        (a !== :none && get(folded, a, false)) || return
        folded[a] = false
        rebuild()
    end
    # one shared click handler: hit-test the image cards in figure pixels
    on(events(player.fig).mousebutton; priority = 30) do event
        (event.button == Mouse.left && event.action == Mouse.press &&
         player.dockopen[] === :tools && !isempty(cards)) || return Consume(false)
        # cards are hit-tested by bbox, which cannot see an overlay (modal,
        # dropdown) drawn on top of them — Makie's event routing can
        Makie.receives_events(player.dockpanels[:tools].sf.scene) || return Consume(false)
        mp = events(player.fig).mouseposition[]
        for (id, frame, _, _, onclick, _, _) in cards
            bb = frame.layoutobservables.computedbbox[]
            if bb.origin[1] <= mp[1] <= bb.origin[1] + bb.widths[1] &&
               bb.origin[2] <= mp[2] <= bb.origin[2] + bb.widths[2]
                onclick === nothing || onclick(id)
                return Consume(true)
            end
        end
        return Consume(false)
    end
    return panel
end

"""
Make an (initially empty) layout report a height: GridLayoutBase treats a layout
with no measurable content as indeterminate, and one such slot is enough to hide
the height of everything above it — including the dock's scrollable content size.
"""
function measurable!(gl)
    Box(gl[1, 2]; width = 0, height = 0, color = :transparent, strokewidth = 0)
    return gl
end

"The card slot (`:controls` or `:rows`) this context fills, or `nothing`."
function toolslot(ctx::ToolContext, which::Symbol)
    slots = get(ctx.player.fxwidgets, :toolslots, nothing)
    slots === nothing && return nothing
    return get(which === :rows ? slots[2] : slots[1], ctx.tool, nothing)
end

"""
    toolaction!(ctx, label, callback)

Add an action button to this tool's card. Repeated calls STACK, in call order;
`ctx.callbacks` addresses them for tests and MCP.
"""
function toolaction!(ctx::ToolContext, label::AbstractString, callback)
    slot = toolslot(ctx, :controls)
    slot === nothing && return nothing
    push!(ctx.callbacks, callback)
    i = length(ctx.callbacks)
    b = Button(slot[length(ctx.controls) + 1, 1]; label = String(label), tellwidth = false,
               width = Makie.Relative(1.0))
    on(_ -> (f = ctx.callbacks[i]; f === nothing || f()), b.clicks)
    push!(ctx.controls, b)
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
    k = 2id - 1
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
                visible = lift(d -> d === :tools, ctx.player.dockopen))
    translate!(im, 0, 0, 20)
    push!(cards, (id, frame, im, lbl, onclick, rm, box))
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
    for (_, frame, im, lbl, _, rm, box) in cards
        try
            Makie.delete!(frame); Makie.delete!(box); Makie.delete!(lbl)
            Makie.delete!(scene, im)
            rm === nothing || Makie.delete!(rm)
        catch
        end
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

"▼ click: CUT the timeline at that hint (undoable) — the tool stays armed."
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
    curve = get!(AnimCurve, clip.animations, :opacity)
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
    c = get(clip.animations, :opacity, nothing)
    c === nothing && return clip
    half = max(srclength(clip) ÷ 2, 1)   # the fade zone is keyed in SOURCE frames
    zone = dir === :out ? ((clip.src_out - half):clip.src_out) :
           (clip.src_in:(clip.src_in + half))
    filter!(k -> !(k.frame in zone), c.keys)
    if isempty(c.keys)
        delete!(clip.animations, :opacity)
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
    c = get(clip.animations, :opacity, nothing)
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
    deactivatetool!(player)   # nothing stays armed — the card keeps showing the blends
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
Fill the Stabilize card: the mode, the action that runs the analysis, what the
selected clip currently carries, and a way to take it off again. Stabilizing is
an ANALYSIS of a clip — like the loop finder's search — so it lives with the
tools, not in the effect stack (Simon, 2026-07-27); what it produces (the motion
track) still renders as part of the clip like any other effect.
"""
function stabilizepanel!(ctx::ToolContext)
    player = ctx.player
    ctx.state = Dict{Symbol, Any}(:mode => :similarity)
    menu = toolmenu!(ctx, "Mode", STABMODES, v -> (v === nothing || (ctx.state[:mode] = v));
                     default = STABMODES[1][1])
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

"Run (or arm) the analysis the Stabilize card is set to."
function runstabilize!(ctx::ToolContext)
    player = ctx.player
    player.stabinfo[] == "analyzing…" &&
        return setstatus!(player, "analysis already running — progress in the bottom right")
    mode = get(ctx.state, :mode, :similarity)
    if mode === :objectlock
        armpick!(player)      # the next preview click picks the subject to lock on
    else
        analyzeat!(player, (c; kwargs...) ->
                       analyzemotion!(c; mode, backend = player.analysisbackend, kwargs...),
                   "motion stabilization")
    end
    return nothing
end

registertool!(:stabilize, "Stabilize",
    "Locks the SELECTED clip. Camera lock holds the framing like a tripod. " *
    "Object lock keeps one subject still — click it in the preview afterwards. " *
    "Smooth keeps the camera moves and only takes the shake out.";
    panel = stabilizepanel!, activate = ctx -> (runstabilize!(ctx);
                                                deactivatetool!(ctx.player)))

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
        loc[1].colortrack = nothing
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

registertool!(:flicker, "Fix flicker",
    "Evens out exposure/colour jitter on the SELECTED clip: everything faster " *
    "than the cutoff is treated as flicker, slower changes survive.";
    panel = flickerpanel!, activate = runflickerfix!)

registertool!(:loopfinder, "Loop finder",
    "Each reference card holds one frame; ▼ hints on the thumb track mark the " *
    "frames most similar to the SELECTED card — click a ▼ to cut there. " *
    "Find adds the playhead frame as another reference.";
    activate = activateloopfinder!)

registertool!(:blend, "Blend clips",
    "Mark two clips (Shift+click), then click this header — the later clip fades in.";
    panel = blendpanel!, activate = activateblend!)

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
    toollabel!(ctx, player.matteinfo)
    # the button says what it DOES; how to work it belongs in the description,
    # which has room to wrap — a Button label has none and simply overflows
    toolaction!(ctx, "Mark subject", () -> armmattepick!(ctx))
    toolaction!(ctx, "Re-propagate", () -> runmatte!(ctx))
    toolaction!(ctx, "Remove matte", () -> removematte!(player))
    refreshmattecards!(ctx)
    # Opening the panel is the earliest honest signal that a matte is coming, and
    # the user is still deciding where to click — much better than paying the
    # model's one-time specialization cost on their first mark.
    if hasmattemodel() && !MATTEWARMED[]
        # at the resolution `analyzematte!` will actually use for this clip
        # Must match `analyzematte!`'s own sizing exactly, `min` clamp included:
        # warming the wrong tile shapes buys nothing, because the GEMM
        # specializes per tile and the real clip then pays the stall anyway.
        loc0 = editclip(player)
        mw, mh = if loc0 === nothing
            480, 270
        else
            sw, sh = loc0[1].source.width, loc0[1].source.height
            w = min(480, sw)
            w, max(1, round(Int, sh * w / sw))
        end
        player.matteinfo[] = "warming up the model…"
        runanalysis(player) do
            try
                warmmatte!(mw, mh)
                put!(player.uiqueue, () -> (player.matteinfo[] = "ready — mark the subject"))
            catch e
                put!(player.uiqueue, () -> (player.matteinfo[] = "model warm-up failed"))
            end
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
    points::Vector{Tuple{Float64, Float64, Bool}}   # normalized source coords, foreground?
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
    lastseedpoints::Vector{Tuple{Float64, Float64, Bool}}
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
        scatter!(sc, fg; color = (:limegreen, 0.75), marker = :circle,
                 markersize = 9, strokewidth = 1, strokecolor = (:black, 0.6))
        scatter!(sc, bg; color = (:orangered, 0.75), marker = :xcross,
                 markersize = 9, strokewidth = 1, strokecolor = (:black, 0.6))
        (scene = sc, fg = fg, bg = bg)
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
function armmattepick!(ctx::ToolContext)
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
    col = MatteCollect(ctx, clip, Int(srcframe), sc, Tuple{Float64, Float64, Bool}[],
                       ov.fg, ov.bg, Any[],
                       clip.mattetrack, nothing, nothing, nothing,
                       findeffect(clip, MatteEffect) !== nothing, false, false,
                       nothing, Tuple{Float64, Float64, Bool}[])
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
    player.matteinfo[] = "painting: drag over the subject, right-drag to exclude"
    setstatus!(player, "matte: DRAG over the subject (right-drag excludes) · " *
                       "Ctrl+Z or Backspace undoes · Enter propagates · Esc cancels")
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
    sx, sy = previewtosource(player, col.clip, col.srcframe, p)
    sw, sh = col.clip.source.width, col.clip.source.height
    nx, ny = clamp(sx / sw, 0.0, 1.0), clamp(sy / sh, 0.0, 1.0)
    # compared where the user is pointing — on the canvas — not in source pixels,
    # so the hit area is the dot's size however far the preview is zoomed.
    # `remove = false` while a stroke is being painted: a drag lays points close
    # together on purpose and would otherwise erase the ones it just made.
    ondot = remove ? findlast(eachindex(col.points)) do i
            q = sourcetopreview(player, col.clip, col.srcframe,
                                (col.points[i][1] * sw, col.points[i][2] * sh))
            (q[1] - p[1])^2 + (q[2] - p[2])^2 <= hit^2
        end : nothing
    if ondot === nothing
        push!(col.points, (nx, ny, foreground))
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
Redraw the dots from `col.points`.

Rebuilt from the source of truth rather than pushed/popped alongside it: the two
lists disagreeing is the classic way a removed point keeps its dot. Canvas
positions come back out of the same mapping that put them in, so a dot sits where
its point is even after the playhead moved or the clip was reframed.
"""
function refreshmattedots!(col::MatteCollect)
    player = col.ctx.player
    sw, sh = col.clip.source.width, col.clip.source.height
    fg, bg = Point2f[], Point2f[]
    for (nx, ny, isfg) in col.points
        q = sourcetopreview(player, col.clip, col.srcframe, (nx * sw, ny * sh))
        push!(isfg ? fg : bg, Point2f(q))
    end
    col.fg[] = fg
    col.bg[] = bg
    return nothing
end

"Back to the picture the clip had before this marking, with no points left."
function clearlivematte!(col::MatteCollect)
    col.clip.mattetrack === nothing || freematteplanes!(col.clip.mattetrack)
    col.clip.mattetrack = col.prevtrack
    restorematteeffect!(col)
    col.ctx.player.matteinfo[] = "marking — no points"
    notify(col.ctx.player.playhead)
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
    runanalysis(player) do
        try
            frame = framereader(player, col.clip)(col.srcframe)
            mask = seedmask(col.clip, frame, points; key = (col.clip.id, col.srcframe))
            alpha = previewmatte(col.clip, frame, mask)
            put!(player.uiqueue, () -> begin
                if mattecollect(player) === col         # not cancelled meanwhile
                    col.lastseed = mask
                    col.lastseedpoints = points
                    showlivematte!(col, alpha)
                end
                col.busy = false
                if col.pending
                    col.pending = false
                    livematte!(col)
                end
            end)
        catch e
            put!(player.uiqueue, () -> begin
                col.busy = false
                setstatus!(player, "matte preview failed: $(sprint(showerror, e))")
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
    col.clip.mattetrack === nothing || freematteplanes!(col.clip.mattetrack)
    col.clip.mattetrack = track
    if col.prevmatte === nothing                       # first preview of this marking
        # `findeffect` hands back the EFFECT (`findslot` is the one that returns
        # the slot) — reaching for `.effect` here asked a `MatteEffect` for a
        # field it has never had, on the very first marked point.
        col.prevmatte = something(findeffect(col.clip, MatteEffect), MatteEffect())
        seteffect!(col.clip, MatteEffect(; strength, feather = col.prevmatte.feather))
    end
    player.matteinfo[] = "marking — $(length(col.points)) point(s), this frame only"
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
            frame = framereader(player, clip)(srcframe)
            mask = seedmask(clip, frame, points; key = (clip.id, Int(srcframe)))
            put!(player.uiqueue, () -> addmatteseed!(ctx, clip, srcframe, mask))
        catch e
            put!(player.uiqueue,
                 () -> setstatus!(player, "matte: seeding failed: $(sprint(showerror, e))"))
        end
    end
    return nothing
end

"Discard the marking and put back what the preview showed before it."
function cancelmattecollect!(col::MatteCollect)
    endmattecollect!(col)
    restorematteeffect!(col)
    col.clip.mattetrack === nothing || freematteplanes!(col.clip.mattetrack)
    col.clip.mattetrack = col.prevtrack
    if !col.hadeffect                      # we added it; take it back off again
        i = findfirst(s -> s.effect isa MatteEffect, col.clip.effects)
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
    seeds = mattemarks(player, clip)
    seeds[Int(srcframe)] = mask
    runmatte!(ctx; clip = clip, seeds = seeds)
    return nothing
end

"""
The marks for `clip`, live.

Held next to the player rather than on the track because a mark is an input to
propagation and the track is its output: re-running must see every mark, not the
seed frames the last run happened to record. Keyed by clip id so it survives
sorting and undo.
"""
mattemarks(player::Player, clip::Clip) =
    get!(() -> Dict{Int, Matrix{UInt8}}(), player.mattemarks, clip.id)

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
    setstatus!(player, "matte: propagating from $(length(marks)) marked frame(s) " *
                       "($(mattebackendname()))")
    runanalysis(player) do
        try
            reader = framereader(player, clip)
            track = analyzematte!(clip, reader, marks; progress = (d, t) -> begin
                player.jobprogress[] = d / max(t, 1)
            end)
            put!(player.uiqueue, () -> begin
                freematteplanes!(track)
                findeffect(clip, MatteEffect) === nothing &&
                    push!(clip.effects, FxSlot(MatteEffect()))
                cov = mattecoverage(track)
                player.matteinfo[] = "matte: $(length(track.seeds)) marked frame(s), " *
                                     "$(size(track.alpha, 3)) frames, " *
                                     "$(round(Int, 100cov))% kept ($(mattebackendname()))"
                player.jobprogress[] = NaN
                refreshmattepanel!(player)
                notify(player.playhead)
                # A matte that keeps everything (or nothing) renders as no visible
                # change at all, and the user reads that as "the tool is broken".
                # The built-in propagator does exactly that on most footage — it is
                # a stand-in, not a matting model — so say which one ran and what
                # it produced instead of reporting "ready" either way.
                setstatus!(player, if cov > 0.97 || cov < 0.02
                        "matte covers $(round(Int, 100cov))% of the frame — " *
                        (hasmattemodel() ? "mark more of the subject, or a background point where it spills" :
                                           "no matting model is installed, so this is the built-in stand-in " *
                                           "(run `usematanyone!()` for the real one)")
                    else
                        "matte ready ($(mattebackendname())) — Matte/Feather are keyframable in the inspector"
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

function removematte!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "matte: no clip under the playhead")
    clip = loc[1]
    clip.mattetrack === nothing && return setstatus!(player, "no matte on this clip")
    freematteplanes!(clip.mattetrack)
    clip.mattetrack = nothing
    delete!(player.mattemarks, clip.id)
    i = findfirst(s -> s.effect isa MatteEffect, clip.effects)
    i === nothing || deleteat!(clip.effects, i)
    player.matteinfo[] = "no matte"
    refreshmattepanel!(player)
    notify(player.playhead)
    setstatus!(player, "matte removed")
    return nothing
end

"One card per marked frame: jump to it, or drop the mark and re-propagate."
function refreshmattecards!(ctx::ToolContext)
    player = ctx.player
    loc = editclip(player)
    loc === nothing && return nothing
    clip = loc[1]
    marks = mattemarks(player, clip)
    for f in sort!(collect(keys(marks)))
        thumb = mattecardimage(marks[f])
        tl = clip.start + (f - clip.src_in)
        # `onclick`/`onremove` are handed the CARD id (see `tooladdcard!`); taking
        # no argument made every click on a mark — and every × — a MethodError in
        # the render loop, which is why the button looked dead.
        tooladdcard!(ctx, thumb; caption = "frame $tl",
                     onclick = _ -> (player.playhead[] = tl),
                     onremove = _ -> begin
                         delete!(marks, f)
                         isempty(marks) ? removematte!(player) : runmatte!(ctx; clip = clip)
                     end)
    end
    return nothing
end

"A small preview of one mark, so a card shows WHICH region was marked."
function mattecardimage(mask::AbstractMatrix{UInt8})
    w, h = 48, 27
    img = Matrix{RGB{N0f8}}(undef, w, h)
    sw, sh = size(mask)
    @inbounds for j in 1:h, i in 1:w
        v = mask[clamp(round(Int, (i - 0.5) * sw / w + 0.5), 1, sw),
                 clamp(round(Int, (j - 0.5) * sh / h + 0.5), 1, sh)]
        img[i, j] = v > 0 ? RGB{N0f8}(1, 1, 1) : RGB{N0f8}(0.15, 0.15, 0.18)
    end
    return img
end

"Rebuild the Tools dock and the inspector after the matte changed either."
function refreshmattepanel!(player::Player)
    TOOLSVERSION[] += 1                       # tool cards
    r = get(player.fxwidgets, :fxlistrefresh, nothing)
    r === nothing || r()                      # inspector: the MatteEffect card
    return nothing
end

registertool!(:matte, "Matte",
    "Isolates a subject on the SELECTED clip. DRAG over the subject to paint it " *
    "roughly (right-drag paints what is NOT it), then Enter. The model propagates " *
    "the painted mask across the clip — it does not grow a dot, so cover the " *
    "subject. Mark another frame wherever it drifts.";
    panel = mattepanel!, activate = ctx -> armmattepick!(ctx))


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
        i = findfirst(s -> s.effect isa RestoreEffect, loc[1].effects)
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
            reader = framereader(player, clip)
            got = restorewindow!(clip, reader, first, n;
                                 progress = (d, t) -> (player.jobprogress[] = d / max(t, 1)))
            put!(player.uiqueue, () -> begin
                findeffect(clip, RestoreEffect) === nothing &&
                    push!(clip.effects, FxSlot(RestoreEffect()))
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

registertool!(:restore, "Restore",
    "Runs an upscaling/restoration model over frames around the playhead on the " *
    "SELECTED clip. Needs a model installed (examples/basicvsrpp.jl).";
    panel = restorepanel!, activate = ctx -> runrestore!(ctx))
