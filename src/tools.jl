# GUI tools — premade editing operations with timeline presence. A tool draws
# HINTS on the thumb track (arbitrary plots on the timeline axis), reads clicks
# there, and acts through the same editing API the rest of the editor uses
# (snapshot!, addtransition!, trims, effects, …). Tools register live exactly
# like effect plugins, so any package or MCP session can add its own.

"""
    ToolContext

What an active tool works with: the `player`, plus bookkeeping so everything
the tool adds — plots on the timeline axis, event handlers — is removed on
deactivation. Overlays draw in TIMELINE coordinates: x is seconds
([`tooltime`](@ref)), y is the 0..1 track span ([`toolband`](@ref) for a
clip's band). Record plots with [`toolplot!`](@ref) and wire observables with
[`ontool!`](@ref); `state` is tool-private scratch.
"""
mutable struct ToolContext
    const player::Player
    const plots::Vector{Any}
    const handlers::Vector{Any}
    state::Any
end

"Record a plot the tool drew, so deactivation removes it."
toolplot!(ctx::ToolContext, plt) = (push!(ctx.plots, plt); plt)

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
    for p in ctx.plots
        try
            Makie.delete!(player.timeline.axis, p)
        catch
        end
    end
    empty!(ctx.plots)
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
it away. Rebuilt live on [`registertool!`](@ref)."
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
    return panel
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

function refreshloophints!(ctx::ToolContext)
    st = ctx.state
    st === nothing && return nothing
    haskey(st, :sig) || return nothing
    player = ctx.player
    clip = st[:clip]
    fps = player.sequence.framerate
    # reference = the playhead frame while it is on the analyzed clip; leaving
    # the clip keeps the last hints (they stay clickable)
    loc = locate(player.sequence, player.playhead[])
    (loc === nothing || loc[1] !== clip) && return nothing
    ref = clamp(loc[2] - clip.src_in + 1, 1, size(st[:sig], 3))
    st[:ref] = ref
    hints = similarframes(st[:sig], ref; n = st[:nhints],
                          exclude = max(round(Int, fps), 2))
    st[:hints] = hints
    lo, hi = toolband(player, clip)
    yh = hi - 0.12 * (hi - lo)
    accent = RGBAf(Makie.to_color(player.timeline.colors.accent))
    st[:hintpts][] = [Point2f(tooltime(player, clip.start + h.frame - 1), yh) for h in hints]
    st[:hintcols][] = [RGBAf(accent.r, accent.g, accent.b, k == 1 ? 1.0 : 0.55)
                       for k in eachindex(hints)]
    st[:refpt][] = [Point2f(tooltime(player, clip.start + ref - 1), yh)]
    return nothing
end

"Trim the timeline to the loop between the current reference frame and hint `k`."
function looptrimto!(ctx::ToolContext, k::Integer)
    player = ctx.player
    st = ctx.state
    clip = st[:clip]
    hints = st[:hints]
    (1 <= k <= length(hints) && haskey(st, :ref)) || return nothing
    a, b = minmax(st[:ref], hints[k].frame)
    b - a >= 2 && begin
        fps = player.sequence.framerate
        snapshot!(player)
        clip.src_out = clip.src_in + b - 1   # ORDER: shrink the out edge first,
        clip.src_in = clip.src_in + a - 1    # src_in shifts both bounds' base
        clip.start = 0
        filter!(c -> c === clip, player.sequence.clips)
        prunetransitions!(player.sequence)
        seek!(player, 0)
        setstatus!(player, "trimmed to a $(round((b - a) / fps, digits = 1))s loop " *
                           "(seam $(round(hints[k].score, digits = 4))) — Ctrl+Z undoes")
        refreshedit!(player)
    end
    deactivatetool!(player)   # hints are stale after the trim — tool is done
    return nothing
end

function activateloopfinder!(ctx::ToolContext)
    player = ctx.player
    loc = locate(player.sequence, player.playhead[])
    if loc === nothing
        setstatus!(player, "Loop finder: put the playhead on a clip first")
        deactivatetool!(player)
        return nothing
    end
    clip = loc[1]
    ax = player.timeline.axis
    st = Dict{Symbol, Any}(:clip => clip, :nhints => 6,
                           :hintpts => Observable(Point2f[]),
                           :hintcols => Observable(RGBAf[]),
                           :refpt => Observable(Point2f[]),
                           :hints => NamedTuple{(:frame, :score), Tuple{Int, Float32}}[])
    ctx.state = st
    hp = scatter!(ax, st[:hintpts]; color = st[:hintcols], marker = :dtriangle,
                  markersize = 16, strokecolor = :black, strokewidth = 1)
    rp = scatter!(ax, st[:refpt]; color = :white, marker = :utriangle,
                  markersize = 13, strokecolor = :black, strokewidth = 1)
    foreach(p -> translate!(p, 0, 0, 6), (hp, rp))   # above thumbs, below playhead
    toolplot!(ctx, hp)
    toolplot!(ctx, rp)
    setstatus!(player, "Loop finder: analyzing $(cliplength(clip)) frames…")
    player.jobprogress[] = 0.0
    job = () -> try
        sig = loopsignatures(clip; backend = player.analysisbackend,
                             progress = (d, t) -> (player.jobprogress[] = d / max(t, 1)))
        put!(player.uiqueue, () -> begin
            cur = activetool(player)                   # put away / switched while analyzing?
            (cur === nothing || cur[2] !== ctx) && return
            st[:sig] = sig
            refreshloophints!(ctx)
            setstatus!(player, "Loop finder: ▲ marks the playhead frame, ▼ its best " *
                               "matches — move the playhead to rescore, click a ▼ to loop")
        end)
    catch e
        setstatus!(player, "Loop finder failed: $(sprint(showerror, e))")
        @error "loop signature analysis failed" exception = (e, catch_backtrace())
    finally
        player.jobprogress[] = NaN
    end
    runanalysis(job, player)
    ontool!(_ -> refreshloophints!(ctx), ctx, player.playhead)
    ontool!(ctx, events(ax.scene).mousebutton) do event
        (event.button == Mouse.left && event.action == Mouse.press) || return Consume(false)
        Makie.is_mouseinside(ax.scene) || return Consume(false)
        t, y = mouseposition(ax.scene)
        k = nearesttoolpoint(player, st[:hintpts][], t, y)
        k == 0 && return Consume(false)   # elsewhere: scrub/select as usual
        looptrimto!(ctx, k)
        return Consume(true)
    end
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
    "Scores every frame of the clip against the frame under the playhead and marks " *
    "the best loop points on the thumb track — click a ▼ hint to trim to that loop.";
    activate = activateloopfinder!)

registertool!(:blend, "Blend clips",
    "Click two adjacent clips to add a cross-dissolve at their cut. " *
    "T at the cut removes it again.";
    activate = activateblend!)
