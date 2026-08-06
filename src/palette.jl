# Ctrl+P — everything the editor can do, one search box.
#
# The old palette searched effects only, and applied whatever it found to the
# clip under the playhead. This one searches [`commands`](@ref): effects are in
# it because "Add Blur" is a command, but so are Split, Export, Undo and the
# dock toggles. Nothing is reachable only by knowing a keyboard shortcut.

"""
Build the command palette and return the function that opens it.

Keyboard, while open: type to filter, ↑/↓ to move, ⏎ to run the highlighted
command, Esc to close. The palette owns the keyboard for as long as it is up, so
nothing behind it reacts to the typing.
"""
function buildpalette!(player::Player, uicolors)
    modal = Modal(player.fig; title = "Run a command — type to search", min_size = (460, 340))
    query = Observable("")
    selected = Observable(1)
    hits = Observable(Tuple{Command, Any}[])

    Label(modal[1, 1], map(q -> isempty(q) ? "▏ type to search…" : q * "▏", query);
          halign = :left, font = :bold, tellwidth = false)
    rowsgl = GridLayout(modal[2, 1])
    listref = Ref{Any}(nothing)

    function refresh()
        listref[] === nothing || Makie.clear!(listref[])
        gl = GridLayout(rowsgl[1, 1]); listref[] = gl
        hs = hits[]
        if isempty(hs)
            Label(gl[1, 1], "no command matches “$(query[])”"; fontsize = 11,
                  color = uicolors.text_muted, tellwidth = false)
            return
        end
        for (r, (cmd, why)) in enumerate(hs)
            picked = r == selected[]
            row = GridLayout(gl[r, 1])
            Box(row[1, 1:3]; color = picked ? uicolors.select_subtle : (:transparent, 0.0),
                strokecolor = picked ? uicolors.select : (:transparent, 0.0),
                strokewidth = picked ? 1 : 0, cornerradius = 4,
                tellwidth = false, tellheight = false)
            b = Button(row[1, 1]; label = cmd.label, tellwidth = false,
                       width = Makie.Relative(1.0),
                       buttoncolor = (:transparent, 0.0), strokewidth = 0,
                       labelcolor = why === nothing ? uicolors.text : uicolors.text_muted)
            on(_ -> runpalette!(player, modal, cmd, why), b.clicks)
            # the category, so two commands with similar names are told apart,
            # and the shortcut, so the palette TEACHES the keyboard
            Label(row[1, 2], String(cmd.category); fontsize = 10, halign = :right,
                  color = uicolors.text_muted, tellwidth = false, width = 74)
            Label(row[1, 3], why === nothing ? cmd.shortcut : "— $why";
                  fontsize = 10, halign = :right, tellwidth = false, width = 150,
                  color = why === nothing ? uicolors.text_muted : uicolors.accent)
        end
        return
    end

    function recompute()
        hits[] = rankcommands(player, query[])
        selected[] = 1
        refresh()
        return
    end
    on(_ -> recompute(), query)
    on(_ -> refresh(), selected)

    on(events(player.fig).unicode_input) do chars
        modal.open[] || return Consume(false)
        s = chars isa AbstractVector ? String(collect(chars)) : string(chars)
        isempty(s) && return Consume(false)
        query[] = query[] * s
        return Consume(true)
    end
    on(events(player.fig).keyboardbutton; priority = 30) do ev
        modal.open[] || return Consume(false)
        ev.action in (Keyboard.press, Keyboard.repeat) || return Consume(false)
        if ev.key == Keyboard.backspace
            isempty(query[]) || (query[] = String(chop(query[])))
        elseif ev.key == Keyboard.down
            selected[] = min(selected[] + 1, max(length(hits[]), 1))
        elseif ev.key == Keyboard.up
            selected[] = max(selected[] - 1, 1)
        elseif ev.key == Keyboard.enter
            hs = hits[]
            isempty(hs) ? setstatus!(player, "no command matches “$(query[])”") :
                          runpalette!(player, modal, hs[selected[]]...)
        elseif ev.key == Keyboard.escape
            close!(modal)
        end
        return Consume(true)   # the palette owns the keyboard while it is open
    end

    open = () -> (query[] = ""; recompute(); open!(modal))
    merge!(player.fxwidgets, Dict{Symbol, Any}(
        :palettemodal => modal, :palettequery => query, :paletteopen => open,
        :paletterun => (name::Symbol) -> runcommand!(player, name),
        :palettehits => hits, :paletteselected => selected))
    return open
end

"Run `cmd` from the palette (or say why it cannot run) and close it."
function runpalette!(player::Player, modal, cmd::Command, why)
    if why !== nothing
        setstatus!(player, "$(cmd.label) — $why")
        return nothing
    end
    close!(modal)
    cmd.run(player)
    return nothing
end

"""
    runcommand!(player, name) -> Bool

Run the command called `name`. The one entry point MCP, tests and scripts use, so
"can the agent do what the user can do" has one answer.
"""
function runcommand!(player::Player, name::Symbol)
    i = findfirst(c -> c.name === name, commands(player))
    if i === nothing
        setstatus!(player, "no command named “$(name)”")
        return false
    end
    cmd = commands(player)[i]
    why = disabledreason(cmd, player)
    if why !== nothing
        setstatus!(player, "$(cmd.label) — $why")
        return false
    end
    cmd.run(player)
    return true
end
