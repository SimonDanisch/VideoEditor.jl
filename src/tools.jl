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
    const plots::Vector{Any}    # (parent, plot) pairs for removal
    const handlers::Vector{Any}
    state::Any
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
    activate::Any
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
                       activate, deactivate = ctx -> nothing)
    t = EditorTool(name, String(label), String(description), activate, deactivate)
    TOOLBYNAME[name] = t
    i = findfirst(q -> q.name == name, TOOLS)
    i === nothing ? push!(TOOLS, t) : (TOOLS[i] = t)
    TOOLSVERSION[] = TOOLSVERSION[] + 1
    return t
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
    foreach(Observables.off, ctx.handlers)
    empty!(ctx.handlers)
    for (parent, p) in ctx.plots
        try
            Makie.delete!(parent, p)
        catch
        end
    end
    empty!(ctx.plots)
    cleartoolpanel!(player)
    activetoolname(player)[] = :none
    return nothing
end

"""
    activatetool!(player, name) -> ToolContext | nothing

Activate tool `name` (deactivating any current one); activating the active
tool toggles it off.
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
    ctx = ToolContext(player, Any[], Any[], nothing)
    player.fxwidgets[:activetool] = (tool, ctx)
    activetoolname(player)[] = name
    tool.activate(ctx)
    return ctx
end

# --------------------------------------------------------------- Tools panel

"The Tools dock: one row per registered tool — click arms it, click again puts
it away — plus a PREVIEW CARD the active tool can fill (e.g. the loop finder's
reference frame, via [`toolpreview!`](@ref)). Rebuilt live on [`registertool!`](@ref)."
function buildtoolspanel!(player::Player, gridpos, uicolors)
    panel = GridLayout(gridpos; tellheight = false, valign = :top)
    Label(panel[1, 1], "Tools"; font = :bold, halign = :left, tellwidth = false)
    rows = GridLayout(panel[2, 1])
    colsize!(panel, 1, Makie.Relative(1.0))
    active = activetoolname(player)
    built = Any[]
    function rebuild()
        foreach(Makie.delete!, built)
        empty!(built)
        for (k, tool) in enumerate(TOOLS)
            btn = Button(rows[2k - 1, 1]; label = tool.label, tellwidth = false,
                         width = Makie.Relative(1.0), halign = :left)
            desc = Label(rows[2k, 1], tool.description; fontsize = 11, halign = :left,
                         color = uicolors.text_muted, tellwidth = false, word_wrap = true)
            on(_ -> activatetool!(player, tool.name), btn.clicks)
            on(active; update = true) do a
                armed = a === tool.name
                btn.buttoncolor[] = armed ? uicolors.accent : uicolors.surface
                btn.labelcolor[] = armed ? uicolors.text_on_accent : uicolors.text
            end
            push!(built, btn, desc)
        end
        return
    end
    on(_ -> rebuild(), TOOLSVERSION)
    rebuild()
    # ACTION slot: the active tool can offer one panel action ("Find …") —
    # the Button is created/deleted on demand (Buttons have no `visible`)
    actionslot = GridLayout(panel[3, 1])
    actionbtn = Ref{Any}(nothing)
    actioncb = Ref{Any}(nothing)
    # CARD list: the active tool adds preview cards (image + caption, clickable,
    # highlightable). Images draw on the dock scene over layout Boxes — an Axis
    # inside a Subfigure won't render (same pattern as the media bin thumbs).
    cardrows = GridLayout(panel[4, 1])
    scene = player.dockpanels[:tools].sf.scene
    cards = Any[]   # (id, box, im, label, onclick)
    player.fxwidgets[:toolaction] = (actionslot, actionbtn, actioncb)
    player.fxwidgets[:toolcards] = (cardrows, scene, cards)
    # one shared click handler: hit-test the card boxes in figure pixels
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

"""
    toolaction!(ctx, label, callback)

Offer one action button in the Tools panel while this tool is active (e.g. the
loop finder's "Find similar frames"). Hidden again at deactivation.
"""
function toolaction!(ctx::ToolContext, label::AbstractString, callback)
    ta = get(ctx.player.fxwidgets, :toolaction, nothing)
    ta === nothing && return nothing
    slot, btn, cb = ta
    btn[] === nothing || Makie.delete!(btn[])
    cb[] = callback
    b = Button(slot[1, 1]; label = String(label), tellwidth = false,
               width = Makie.Relative(1.0))
    on(_ -> (f = cb[]; f === nothing || f()), b.clicks)
    btn[] = b
    return nothing
end

"""
    tooladdcard!(ctx, img; caption = "", onclick = nothing) -> id

Append a preview card (image + caption) to the Tools panel — e.g. one loop
reference frame. `onclick(id)` fires when the card is clicked. Highlight the
selected card with [`toolhighlight!`](@ref); all cards are removed at
deactivation.
"""
function tooladdcard!(ctx::ToolContext, img::AbstractMatrix{RGB{N0f8}};
                      caption::AbstractString = "", onclick = nothing)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return 0
    cardrows, scene, cards = tc
    colors = ctx.player.timeline.colors
    id = length(cards) + 1
    k = 2id - 1
    box = Box(cardrows[k, 1]; height = 96, tellwidth = false, width = Makie.Relative(1.0),
              color = colors.surface, strokecolor = colors.border, strokewidth = 1,
              cornerradius = 3)
    lbl = Label(cardrows[k + 1, 1], String(caption); fontsize = 11, halign = :left,
                color = (colors.text, 0.65), tellwidth = false)
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
    push!(cards, (id, box, im, lbl, onclick))
    return id
end

"Accent-outline card `id`, resetting the others (the selected loop reference)."
function toolhighlight!(ctx::ToolContext, id::Integer)
    tc = get(ctx.player.fxwidgets, :toolcards, nothing)
    tc === nothing && return nothing
    colors = ctx.player.timeline.colors
    for (cid, box, _, _, _) in tc[3]
        box.strokecolor[] = cid == id ? colors.accent : colors.border
        box.strokewidth[] = cid == id ? 2 : 1
    end
    return nothing
end

"Remove all tool cards and the action button (deactivation cleanup)."
function cleartoolpanel!(player::Player)
    ta = get(player.fxwidgets, :toolaction, nothing)
    if ta !== nothing
        _, btn, cb = ta
        btn[] === nothing || (try; Makie.delete!(btn[]); catch; end)
        btn[] = nothing
        cb[] = nothing
    end
    tc = get(player.fxwidgets, :toolcards, nothing)
    if tc !== nothing
        _, scene, cards = tc
        for (_, box, im, lbl, _) in cards
            try
                Makie.delete!(box); Makie.delete!(lbl); Makie.delete!(scene, im)
            catch
            end
        end
        empty!(cards)
    end
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

"Card + hints for one reference frame (cheap — the clip's signatures exist)."
function addloopref!(ctx::ToolContext, clip::Clip, ref::Integer, sig)
    player = ctx.player
    st = ctx.state
    fps = player.sequence.framerate
    hints = similarframes(sig, clamp(ref, 1, size(sig, 3));
                          n = st[:nhints], exclude = max(round(Int, fps), 2))
    push!(st[:refs], (clip, Int(ref), hints))
    cache = cachefor(player.timeline, clip.source)   # same imagery as the thumb track
    sec = round(Int, (clip.src_in + ref - 1) / clip.source.framerate)
    th = nearestthumb(cache, sec)
    fill3 = RGB{N0f8}(player.timeline.colors.surface)
    img = th === nothing ? fill(fill3, 16, 9) : fitbox(th, 240, 92, fill3)
    id = tooladdcard!(ctx, img;
        caption = "ref " * timecode(player.sequence, clip.start + ref - 1),
        onclick = cid -> begin      # clicking a card shows ITS similarity markers
            st[:active][] = cid
            toolhighlight!(ctx, cid)
            showloopmarkers!(ctx)
        end)
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

function activateblend!(ctx::ToolContext)
    player = ctx.player
    ax = player.timeline.axis
    seq = player.sequence
    fps = seq.framerate
    outline = Observable(Point2f[])
    ol = lines!(ax, outline; color = player.timeline.colors.accent, linewidth = 2)
    translate!(ol, 0, 0, 6)
    toolplot!(ctx, ol)
    picked = Ref(0)   # index of the first picked clip, 0 = none
    setstatus!(player, "Blend: click the first clip")
    ontool!(ctx, events(ax.scene).mousebutton) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        Makie.is_mouseinside(ax.scene) || return Consume(false)
        t, _ = mouseposition(ax.scene)
        i = clipat(seq, timelineframe(player.timeline, t))
        i === nothing && return Consume(false)
        if picked[] == 0 || picked[] == i
            picked[] = i
            clip = seq.clips[i]
            lo, hi = toolband(player, clip)
            t0, t1 = clip.start / fps, clipend(clip) / fps
            outline[] = Point2f[(t0, lo), (t1, lo), (t1, hi), (t0, hi), (t0, lo)]
            setstatus!(player, "Blend: now click the adjacent clip to dissolve into")
            return Consume(true)
        end
        a, b = seq.clips[picked[]], seq.clips[i]
        left, right = a.start <= b.start ? (a, b) : (b, a)
        at = clipend(left)
        if right.start != at || left.track != right.track
            setstatus!(player, "Blend: those clips don't share a cut — pick two adjacent clips")
        else
            snapshot!(player)
            tr = addtransition!(seq, at; duration = max(round(Int, 0.6 * fps), 2))
            if tr === nothing
                pop!(player.undostack)
                setstatus!(player, "Blend: no room for a dissolve at that cut " *
                                   "(both clips need trim handles)")
            else
                setstatus!(player, "cross-dissolve added ($(round(tr.duration / fps, digits = 1))s) " *
                                   "— T at the cut removes it, Ctrl+Z undoes")
                notify(player.playhead)
            end
        end
        picked[] = 0
        outline[] = Point2f[]
        return Consume(true)
    end
    return nothing
end

registertool!(:loopfinder, "Loop finder",
    "Each reference card holds one frame; ▼ hints on the thumb track mark the " *
    "frames most similar to the SELECTED card — click a ▼ to cut there. " *
    "Find adds the playhead frame as another reference.";
    activate = activateloopfinder!)

registertool!(:blend, "Blend clips",
    "Click two adjacent clips to add a cross-dissolve at their cut. " *
    "T at the cut removes it again.";
    activate = activateblend!)
