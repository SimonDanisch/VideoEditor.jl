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
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
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
            Label(body[1, 1], wraptext(tool.description); fontsize = 11, halign = :left,
                  color = uicolors.text_muted, tellwidth = false)
            actionslots[tool.name] = GridLayout(body[2, 1])
            rowslots[tool.name] = GridLayout(body[3, 1])
            cardslots[tool.name] = GridLayout(body[4, 1])
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
        mp = events(player.fig).mouseposition[]
        for (id, box, _, _, onclick) in cards
            bb = box.layoutobservables.computedbbox[]
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
    return nothing
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
    box = Box(cardrows[k, 1]; height = 96, tellwidth = false, width = Makie.Relative(1.0),
              color = colors.surface, strokecolor = colors.border, strokewidth = 1,
              cornerradius = 3)
    lbl = Label(cardrows[k + 1, 1], String(caption); fontsize = 11, halign = :left,
                color = (colors.text, 0.65), tellwidth = false)
    rm = nothing
    if onremove !== nothing   # × shares the caption row, clear of the image
        rm = Button(cardrows[k + 1, 1]; label = "×", width = 22, halign = :right,
                    tellwidth = false)
        on(_ -> onremove(id), rm.clicks)
    end
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
    push!(cards, (id, box, im, lbl, onclick, rm))
    return id
end

"Accent-outline card `id`, resetting the others (the selected loop reference)."
function toolhighlight!(ctx::ToolContext, id::Integer)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return nothing
    colors = ctx.player.timeline.colors
    for (cid, box, _, _, _, _) in tc[3]
        box.strokecolor[] = cid == id ? colors.accent : colors.border
        box.strokewidth[] = cid == id ? 2 : 1
    end
    return nothing
end

"Remove all tool cards (kept separate from the action button so a tool can
re-render its card list in place)."
function cleartoolcards!(player::Player)
    tc = get(player.fxwidgets, :toolcards, nothing)
    tc === nothing && return nothing
    _, scene, cards = tc
    for (_, box, im, lbl, _, rm) in cards
        try
            Makie.delete!(box); Makie.delete!(lbl); Makie.delete!(scene, im)
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
    ref = clamp(srcframe - clip.src_in + 1, 1, cliplength(clip))
    sigs = st[:sigs]
    if haskey(sigs, clip)
        addloopref!(ctx, clip, ref, sigs[clip])
        return nothing
    end
    setstatus!(player, "Loop finder: analyzing $(cliplength(clip)) frames…")
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
    toolaction!(ctx, "Find similar to the playhead frame", () -> loopfind!(ctx))
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
    if dir === :out
        setkey!(curve, clip.src_out - frames, 1.0)
        setkey!(curve, clip.src_out - 1, 0.0)
    else
        setkey!(curve, clip.src_in, 0.0)
        setkey!(curve, clip.src_in + frames - 1, 1.0)
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
    half = max(cliplength(clip) ÷ 2, 1)
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
    return ks[j].frame - clip.src_in + 1
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

registertool!(:loopfinder, "Loop finder",
    "Each reference card holds one frame; ▼ hints on the thumb track mark the " *
    "frames most similar to the SELECTED card — click a ▼ to cut there. " *
    "Find adds the playhead frame as another reference.";
    activate = activateloopfinder!)

registertool!(:blend, "Blend clips",
    "Mark two clips (Shift+click), then click this header — the later clip fades in.";
    panel = blendpanel!, activate = activateblend!)
