# Everything the editor can do, as data.
#
# The palette used to search a hand-built list of effects, the right-click menu
# had its own list of clip actions, the toolbar had buttons, and MCP had a third
# list — four places to add a feature to, and three of them got forgotten. A
# command is declared once here and reachable from all of them.

"""
One thing the editor can do.

`enabled(player)` returns `true`, or a STRING saying why not — the palette shows
the reason instead of hiding the row, because a command you cannot find is
indistinguishable from one that does not exist. `run(player)` does it.

`keywords` are extra search terms that are not in the label: someone looking for
"trim" should find "Split at playhead".
"""
struct Command
    name::Symbol
    label::String
    category::Symbol
    keywords::Vector{String}
    shortcut::String
    enabled::Any     # (player) -> Bool | String
    run::Any         # (player) -> anything
end

function Command(name::Symbol, label::AbstractString; category::Symbol = :general,
                 keywords = String[], shortcut::AbstractString = "",
                 enabled = _ -> true, run)
    return Command(name, String(label), category, String[String(k) for k in keywords],
                   String(shortcut), enabled, run)
end

"""
Every registered command.

One module-level list, like [`EFFECTS`](@ref) and for the same reason: commands
are declared at load time, long before a window exists, and they take the player
they act on as an argument rather than closing over one. A player's own list —
[`commands`](@ref) — is this plus the ones derived from the effects it offers.
"""
const COMMANDS = Command[]

"""
    registercommand!(cmd) -> Command

Add or replace a command by name. Live, like effect registration: a package or an
MCP session can add one and it is in the palette immediately.
"""
function registercommand!(cmd::Command)
    i = findfirst(c -> c.name === cmd.name, COMMANDS)
    i === nothing ? push!(COMMANDS, cmd) : (COMMANDS[i] = cmd)
    return cmd
end

registercommand!(name::Symbol, label::AbstractString; kwargs...) =
    registercommand!(Command(name, label; kwargs...))

"""
    commands(player) -> Vector{Command}

What this player can do: the registered commands, plus one "Add <effect>" per
addable effect kind. The derived half is generated rather than registered so that
an effect registered at runtime is in the palette without a second step.
"""
function commands(player::Player)
    out = copy(COMMANDS)
    for k in addablekinds(player.effects)
        push!(out, Command(Symbol(:add_, k.name), "Add $(k.label)";
                           category = :effect,
                           keywords = vcat(["effect"], [p.label for p in k.params]),
                           enabled = p -> editclip(p) === nothing ?
                                          "needs a clip at the playhead" : true,
                           run = p -> addeffect!(p, k.name)))
    end
    return out
end

"Why `cmd` cannot run right now, or `nothing` when it can."
function disabledreason(cmd::Command, player::Player)
    v = cmd.enabled(player)
    v === true && return nothing
    v === false && return "not available right now"
    return String(v)
end

"""
    matchscore(cmd, query) -> Float64

How well `cmd` answers `query` (already lowercased); 0 means no match. A prefix of
the label beats a word boundary inside it, which beats a keyword hit, which beats
the category — so typing "sp" puts "Split at playhead" above "Fix flicker"
(whose *description* happens to contain "faster than").
"""
function matchscore(cmd::Command, query::AbstractString)
    isempty(query) && return 1.0
    label = lowercase(cmd.label)
    startswith(label, query) && return 4.0
    occursin(" " * query, label) && return 3.0
    occursin(query, label) && return 2.0
    any(k -> occursin(query, lowercase(k)), cmd.keywords) && return 1.5
    occursin(query, String(cmd.category)) && return 1.0
    # subsequence match last: "sap" finds "Split At Playhead", but only when
    # nothing better did
    return subsequence(label, query) ? 0.5 : 0.0
end

"Whether every character of `q` appears in `s`, in order."
function subsequence(s::AbstractString, q::AbstractString)
    i = firstindex(s)
    for c in q
        j = findnext(isequal(c), s, i)
        j === nothing && return false
        i = nextind(s, j)
    end
    return true
end

"""
    rankcommands(player, query; limit = 12) -> Vector{Tuple{Command, Union{Nothing,String}}}

The palette's list: the best matches first, each with its disabled reason (or
`nothing`). Disabled commands rank below enabled ones with the same score — they
are shown, greyed, because "why can't I?" is a better answer than silence.
"""
function rankcommands(player::Player, query::AbstractString; limit::Integer = 12)
    q = lowercase(strip(query))
    scored = Tuple{Float64, Command, Any}[]
    for c in commands(player)
        s = matchscore(c, q)
        s > 0 || continue
        why = disabledreason(c, player)
        push!(scored, (why === nothing ? s : s - 0.25, c, why))
    end
    sort!(scored; by = t -> (-t[1], t[2].label))
    return [(c, why) for (_, c, why) in first(scored, limit)]
end

# ------------------------------------------------------------ what ships here

"True, or the reason there is no clip to act on."
needsclip(player::Player) = editclip(player) === nothing ? "needs a clip at the playhead" : true
"True, or the reason the timeline is empty."
needsclips(player::Player) = isempty(player.sequence.clips) ? "the timeline is empty" : true

registercommand!(:play_pause, "Play / pause"; category = :transport, shortcut = "Space",
    keywords = ["stop", "start"], enabled = needsclips,
    run = p -> p.playing[] ? pause!(p) : play!(p))
registercommand!(:step_forward, "Step one frame forward"; category = :transport,
    shortcut = "→", enabled = needsclips, run = p -> step!(p, 1))
registercommand!(:step_back, "Step one frame back"; category = :transport,
    shortcut = "←", enabled = needsclips, run = p -> step!(p, -1))
registercommand!(:go_start, "Go to the start"; category = :transport, shortcut = "Home",
    enabled = needsclips, run = p -> (p.playhead[] = 0))
registercommand!(:go_end, "Go to the end"; category = :transport, shortcut = "End",
    enabled = needsclips,
    run = p -> seek!(p, max(seqlength(p.sequence) - 1, 0)))

registercommand!(:depth_blur, "Blur background (depth)"; category = :effect,
    keywords = ["depth", "defocus", "bokeh", "background", "portrait"],
    enabled = needsclip,
    run = p -> rundepth!(p))

registercommand!(:split, "Split at playhead"; category = :edit, shortcut = "S",
    keywords = ["cut", "blade", "trim"], enabled = needsclip,
    run = split!)
registercommand!(:delete_clip, "Ripple-delete the clip at the playhead"; category = :edit,
    shortcut = "X", keywords = ["remove"], enabled = needsclip,
    run = deleteat!)
registercommand!(:undo, "Undo"; category = :edit, shortcut = "Ctrl+Z",
    enabled = p -> isempty(p.undostack) ? "nothing to undo" : true, run = undo!)
registercommand!(:redo, "Redo"; category = :edit, shortcut = "Ctrl+Shift+Z",
    enabled = p -> isempty(p.redostack) ? "nothing to redo" : true, run = redo!)

registercommand!(:reset_crop, "Reset crop"; category = :edit, shortcut = "R",
    keywords = ["uncrop", "full frame"], enabled = needsclip, run = p -> resetcrop!(p))

registercommand!(:save_project, "Save project"; category = :file, shortcut = "Ctrl+S",
    enabled = needsclips, run = p -> saveproject!(p))
registercommand!(:import_media, "Import media…"; category = :file,
    keywords = ["open", "add", "source", "file"], run = p -> p.fxwidgets[:browse]())
registercommand!(:export_video, "Export video…"; category = :file,
    keywords = ["render", "write", "mp4"], enabled = needsclips,
    run = p -> opendock!(p, :export))

registercommand!(:show_effects, "Show the Effects panel"; category = :view,
    keywords = ["fx", "inspector"], run = p -> opendock!(p, :effects))
registercommand!(:show_bin, "Show the media bin"; category = :view,
    keywords = ["sources", "files"], run = p -> opendock!(p, :media))
registercommand!(:bypass_all, "Bypass all effects (compare with the original)";
    category = :view, keywords = ["compare", "original", "before", "after", "eye"],
    enabled = needsclip,
    run = p -> (p.applytracks[] = !p.applytracks[]; notify(p.playhead)))
registercommand!(:show_keyframes, "Animated parameters on this clip…"; category = :view,
    keywords = ["keyframes", "curves", "animation", "overview", "legend"],
    # a closure, not a bare reference: commands.jl is included before fxpanel.jl,
    # so the name must resolve at CALL time, not at registration time
    enabled = needsclip, run = p -> openkeyframes!(p))
registercommand!(:toggle_curves, "Show / hide all keyframe curves"; category = :view,
    keywords = ["animation", "graph", "legend"],
    run = p -> showcurves!(p, !anycurvevisible(p)))

registercommand!(:crop, "Crop the picture"; category = :edit, shortcut = "C",
    keywords = ["reframe", "zoom", "trim edges"], enabled = needsclip,
    run = p -> usetool!(p, :crop))
registercommand!(:blade, "Blade — cut where you click"; category = :edit,
    keywords = ["razor", "split", "cut"], enabled = needsclips,
    run = p -> usetool!(p, :split))
registercommand!(:transition, "Add / remove a cross-dissolve"; category = :edit,
    shortcut = "T", keywords = ["fade", "dissolve", "blend"], enabled = needsclip,
    run = toggletransition!)
registercommand!(:next_edit, "Go to the next edit point"; category = :transport,
    shortcut = "↓", enabled = needsclips, run = p -> jumpedit!(p, 1))
registercommand!(:prev_edit, "Go to the previous edit point"; category = :transport,
    shortcut = "↑", enabled = needsclips, run = p -> jumpedit!(p, -1))
