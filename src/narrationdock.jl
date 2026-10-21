# The narration workflow as a dock of its own, the toolbar's "Nar": add a spoken
# line at the playhead, step through the lines, and edit the one in hand.
# Selecting a line on the dialogue lane opens it here, so this dock is that
# selection's inspector the way the effects dock is a clip's. A line and a clip
# are never selected at the same time.

"""
What the narration dock fills, the listeners its current build added, and its
text fields by role, so a rebuild can give focus back to the one that had it.
"""
struct NarrationDock
    layout::GridLayout
    uicolors::Any
    handlers::Vector{Any}
    fields::Dict{Symbol, Any}
end

"""
    buildnarrationdock!(player, gridpos, uicolors)

The narration dock, filled from the sequence by [`refreshnarrationdock!`](@ref)
and again after every narration edit.
"""
function buildnarrationdock!(player::Player, gridpos, uicolors)
    layout = GridLayout(gridpos; valign = :top)
    colsize!(layout, 1, Makie.Relative(1.0))
    player.fxwidgets[:narrationdock] = NarrationDock(layout, uicolors, Any[], Dict{Symbol, Any}())
    # the take being auditioned: one at a time, `nothing` when silent
    player.fxwidgets[:listening] = Observable{Any}(nothing)
    # Selecting a clip ends the line's selection, and its inspector takes the dock.
    on(player.timeline.selected) do id
        id == 0 && return
        selected = get(player.fxwidgets, :dialogue_selected, nothing)
        selected === nothing || selected[] == 0 || (selected[] = 0)
        player.dockopen[] === :narration && opendock!(player, :effects)
    end
    refreshnarrationdock!(player)
    return layout
end

"The line the dock is editing: the selected one, or the last one selected."
currentline(player::Player) =
    clamp(get(player.fxwidgets, :narration_index, 1), 1, max(length(player.sequence.narration), 1))

"""
    selectnarration!(player, i; seek = true)

Make line `i` the selection and open it in the narration dock; with `seek`, the
playhead goes to where it is spoken.
"""
function selectnarration!(player::Player, i::Integer; seek = true)
    seq = player.sequence
    1 <= i <= length(seq.narration) || return nothing
    player.fxwidgets[:narration_index] = Int(i)
    opendock!(player, :narration)
    if seek
        n = seq.narration[i]
        seek!(player, round(Int, something(narrationtime(seq, n), n.at) * seq.framerate))
    end
    return nothing
end

"""
The line in hand becomes the selection: highlighted on the dialogue lane, no clip
selected, its editor rebuilt. What opening the narration dock means.
"""
function marknarration!(player::Player)
    selected = get(player.fxwidgets, :dialogue_selected, nothing)
    if !isempty(player.sequence.narration)
        player.timeline.selection[] = UInt64[]
        player.timeline.selected[] = UInt64(0)
        selected === nothing || (selected[] = currentline(player))
    end
    # deferred: this runs inside the callbacks of the widgets the rebuild deletes
    put!(player.uiqueue, () -> refreshnarrationdock!(player))
    return nothing
end

"Take down everything in `gl`: its blocks and the layouts nested in it."
function clearlayout!(gl::GridLayout)
    for c in reverse(copy(gl.content))
        if c.content isa GridLayout
            clearlayout!(c.content)
            Makie.GridLayoutBase.remove_from_gridlayout!(c)
        else
            Makie.delete!(c.content)
        end
    end
    Makie.trim!(gl)
    return gl
end

"""
    refreshnarrationdock!(player)

Build the narration dock again from the sequence: the new-line box, the line
list, and the editor of the current line.
"""
function refreshnarrationdock!(player::Player)
    dock = get(player.fxwidgets, :narrationdock, nothing)
    dock === nothing && return nothing
    foreach(Observables.off, dock.handlers)
    empty!(dock.handlers)
    # Committing one field rebuilds the dock, often from the very click that went
    # into the next field: that field comes back focused, or the click is lost.
    # A rebuild while typing (a take finished) keeps what was typed so far.
    focused = findfirst(box -> box.focused[], dock.fields)
    typed = focused === nothing ? nothing : dock.fields[focused].editor.arg1[]
    empty!(dock.fields)
    clearlayout!(dock.layout)
    seq = player.sequence
    # a take that is no longer in the document has nothing left to say
    playing = player.fxwidgets[:listening][]
    playing === nothing || any(n -> n === playing.line, seq.narration) || stoplistening!(player)
    layout, colors = dock.layout, dock.uicolors
    Label(layout[1, 1], "Narration"; font = :bold, halign = :left, tellwidth = false)
    newlinecard!(player, dockcard!(layout, 2, "New line at the playhead", colors))
    if isempty(seq.narration)
        Label(layout[3, 1], "No spoken lines yet."; fontsize = 11, halign = :left,
              color = colors.text_muted, tellwidth = false)
    else
        i = currentline(player)
        linelistcard!(player, dockcard!(layout, 3, "Lines · $(length(seq.narration))", colors), i)
        nar = seq.narration[i]
        when = round(something(narrationtime(seq, nar), nar.at); digits = 2)
        linecard!(player, dockcard!(layout, 4, "Line $i · $(isempty(nar.label) ? nar.voice : nar.label) · at $(when)s",
                                    colors; onremove = () -> dropnarration!(player, i)), i, dock)
    end
    rowgap!(layout, 10)
    if focused !== nothing && haskey(dock.fields, focused)
        box = dock.fields[focused]
        box.editor.arg1[] == typed || (box.editor.arg1 = typed)
        Makie.focus!(box)
    end
    return nothing
end

"""
A card of the dock: a title bar, with a × when `onremove` is given, over the body
layout this returns.
"""
function dockcard!(layout::GridLayout, row::Integer, caption::AbstractString, colors; onremove = nothing)
    cell = layout[row, 1]
    Box(cell; color = Makie.lerp_oklab(RGBf(Makie.to_color(colors.background)), RGBf(1, 1, 1), 0.075),
        strokecolor = colors.border, strokewidth = 1, cornerradius = 6, tellwidth = false, tellheight = false)
    card = GridLayout(cell)
    Box(card[1, 1]; color = colors.surface, strokewidth = 0, cornerradius = 5, tellwidth = false, tellheight = false)
    head = GridLayout(card[1, 1]; alignmode = Makie.Outside(8, 6, 4, 4))
    Label(head[1, 1], caption; fontsize = 11, halign = :left, color = (colors.text, 0.75), tellwidth = false)
    if onremove !== nothing
        remove = Button(head[1, 2]; label = "×", width = 22, height = 20)
        on(_ -> onremove(), remove.clicks)
    end
    rowgap!(card, 0)
    body = GridLayout(card[2, 1]; alignmode = Makie.Outside(8, 8, 6, 8))
    rowgap!(body, 6)
    foreach(g -> colsize!(g, 1, Makie.Relative(1.0)), (card, body))
    # the title takes what the × leaves; a whole width for it pushed it out on the left
    colsize!(head, 1, Makie.Auto(false, 1.0))
    return body
end

"Words as typed into a wrapped field: line breaks are only layout, a space to the voice."
spokenwords(text::AbstractString) = strip(replace(text, r"\s*\n\s*" => " "))

"""
A text field for words, wrapped at `columns` characters (what fits the dock):
Enter breaks the line, Ctrl+Enter commits, and so does leaving the field when
`commitonleave`. `commit(words)` gets them with the line breaks undone.
"""
function wordsfield!(commit, gridpos, text::AbstractString; placeholder = "", commitonleave = true,
                     columns = 52)
    box = Textbox(gridpos; stored_string = wraptext(text, columns), placeholder, multiline = true,
                  fontsize = 12, width = Makie.Relative(1.0), tellwidth = false, reset_on_defocus = false)
    on(s -> s === nothing || commit(spokenwords(s)), box.stored_string)
    commitonleave && on(focused -> focused || commit(spokenwords(box.editor.arg1[])), box.focused)
    return box
end

"""
The box for a new line: the words, and the button that places them at the
playhead (Ctrl+Enter in the box does the same).
"""
function newlinecard!(player::Player, body)
    add(words) = isempty(words) || addnarration!(player, words)
    box = wordsfield!(add, body[1, 1], ""; placeholder = "words to speak…", commitonleave = false)
    player.fxwidgets[:narrationdock].fields[:newline] = box
    button = Button(body[2, 1]; label = "Add at playhead", fontsize = 11, tellwidth = false,
                    width = Makie.Relative(1.0))
    on(_ -> add(spokenwords(box.editor.arg1[])), button.clicks)
    return nothing
end

"""
The list of lines and the steps through it: choosing a line, or the previous or
next one, selects it and moves the playhead to it.
"""
function linelistcard!(player::Player, body, i::Integer)
    seq = player.sequence
    options = [("$j. $(round(something(narrationtime(seq, n), n.at); digits = 1))s · $(first(n.text, 40))", j)
               for (j, n) in enumerate(seq.narration)]
    menu = Menu(body[1, 1]; options, default = options[i][1], fontsize = 11, searchable = true,
                search_placeholder = "find a line…", width = Makie.Relative(1.0), tellwidth = false)
    on(j -> j === nothing || j == i || selectnarration!(player, j), menu.selection)
    player.fxwidgets[:narrationlist] = menu
    steps = GridLayout(body[2, 1])
    colgap!(steps, 4)
    for (k, label, j) in ((1, "◀ previous", i - 1), (2, "next ▶", i + 1))
        step = Button(steps[1, k]; label, fontsize = 11, tellwidth = false, width = Makie.Relative(1.0))
        on(_ -> selectnarration!(player, clamp(j, 1, length(seq.narration))), step.clicks)
    end
    return nothing
end

"""
The editor of line `i`: its words, the take (render, listen, move to the
playhead), and the voice that speaks it.
"""
function linecard!(player::Player, body, i::Integer, dock::NarrationDock)
    seq = player.sequence
    nar = seq.narration[i]
    colors = dock.uicolors
    rendering = haskey(get(player.fxwidgets, :speechjobs, IdDict()), nar)
    caption(row, text) = Label(body[row, 1], text; fontsize = 10, halign = :left, tellwidth = false,
                               color = colors.text_muted)
    function field(row, role, label, value, commit; placeholder = "")
        caption(row, label)
        box = Textbox(body[row + 1, 1]; stored_string = value, placeholder, fontsize = 11,
                      width = Makie.Relative(1.0), tellwidth = false, reset_on_defocus = false)
        on(v -> v === nothing || commit(v), box.stored_string)
        dock.fields[role] = box
        return box
    end

    player.fxwidgets[:narrationline] = body       # the line's editor, for walkthroughs and agents
    caption(1, "Words")
    dock.fields[:words] = wordsfield!(body[2, 1], nar.text) do words
        words == spokenwords(nar.text) || setnarrationtext!(player, i, words)
    end
    caption(3, "Enter breaks the line · Ctrl+Enter or clicking elsewhere keeps the change")

    take = GridLayout(body[4, 1])
    colgap!(take, 4)
    render = Button(take[1, 1]; label = rendering ? "Rendering…" : isempty(nar.samples) ? "Render take" : "Re-render take",
                    fontsize = 11, tellwidth = false, width = Makie.Relative(1.0))
    on(_ -> rendernarration!(player, i), render.clicks)
    listening = player.fxwidgets[:listening]
    listenlabel(now) = now !== nothing && now.line === nar ? "■ Stop" : "▶ Listen"
    listen = Button(take[1, 2]; label = listenlabel(listening[]), fontsize = 11, tellwidth = false,
                    width = Makie.Relative(1.0))
    on(_ -> togglelisten!(player, nar), listen.clicks)
    push!(dock.handlers, on(now -> (listen.label[] = listenlabel(now)), listening))
    move = Button(take[2, 1]; label = "Move to playhead", fontsize = 11, tellwidth = false,
                  width = Makie.Relative(1.0))
    on(_ -> movenarration!(player, i), move.clicks)
    reference = Button(take[2, 2]; label = "Reference audio…", fontsize = 11, tellwidth = false,
                       width = Makie.Relative(1.0))
    on(_ -> choosenarrationreference!(player, i), reference.clicks)
    caption(5, rendering ? "Generating this take · you can keep editing." :
               isempty(nar.samples) ? "Not rendered · render to hear these settings." :
               "Take ready · $(round(length(nar.samples) / nar.rate; digits = 2))s · in the preview and the export.")

    caption(6, "Voice model")
    models = [("Session default", :default); [(m.label, k) for (k, m) in speechmodels()]]
    any(o -> o[2] === nar.speech.model, models) ||
        push!(models, ("$(nar.speech.model) · unavailable", nar.speech.model))
    modelmenu = Menu(body[7, 1]; options = models, default = first(only(filter(o -> o[2] === nar.speech.model, models))),
                     fontsize = 11, width = Makie.Relative(1.0), tellwidth = false, searchable = true,
                     search_placeholder = "find a voice model…")
    on(k -> k === nothing || setnarrationspeech!(player, i; model = k), modelmenu.selection)
    references = Pair{String, Tuple{String, String, String}}["No reference voice" => ("", "", nar.voice)]
    for n in seq.narration
        isempty(n.speech.reference) && continue
        any(r -> r.second[1] == n.speech.reference, references) ||
            push!(references, "$(n.voice) · $(basename(n.speech.reference))" =>
                              (n.speech.reference, n.speech.reference_text, n.voice))
    end
    current = something(findfirst(r -> r.second[1] == nar.speech.reference, references), 1)
    refmenu = Menu(body[8, 1]; options = [(r.first, j) for (j, r) in enumerate(references)],
                   default = references[current].first, fontsize = 11, searchable = true,
                   search_placeholder = "find a reference voice…", width = Makie.Relative(1.0), tellwidth = false)
    on(refmenu.selection) do j
        j === nothing && return
        reference, reference_text, voice = references[j].second
        setnarrationspeech!(player, i; reference, reference_text, voice)
    end
    model = speechmodel(nar)
    if model !== nothing && model.reference
        caption(9, "The voice comes from the reference audio.")
        field(10, :reference_text, "Reference transcript", nar.speech.reference_text,
              v -> setnarrationspeech!(player, i; reference_text = v))
    else
        field(9, :voice, "Voice ID", nar.voice, v -> setnarrationvoice!(player, i, v))
    end
    if model === nothing || model.direction
        field(12, :direction, "Regie · delivery instructions", nar.speech.direction,
              v -> setnarrationspeech!(player, i; direction = v); placeholder = "quiet, hesitant, then increasingly angry…")
    else
        caption(12, "This voice model takes no delivery instructions; VoxCPM2 or Fish do.")
    end
    measurable!(body, 1:first(size(body))...)
    return nothing
end

"""
    togglelisten!(player, line)

Play `line`'s take, or stop it when it is the one playing. One take plays at a
time: starting another stops the first, so repeated clicks never stack voices.
"""
function togglelisten!(player::Player, nar::Narration)
    playing = player.fxwidgets[:listening][]
    stoplistening!(player)
    playing !== nothing && playing.line === nar && return setstatus!(player, "stopped")
    isempty(nar.samples) && return setstatus!(player, "render this take before listening")
    Sys.which("pw-cat") === nothing && return setstatus!(player, "audio preview needs pw-cat")
    cmd = audiocommand(`pw-cat --playback --raw --rate $(nar.rate) --channels 1 --format f32 -`)
    process = open(pipeline(cmd; stdout = devnull, stderr = devnull), "w")
    player.fxwidgets[:listening][] = (; line = nar, process)
    setstatus!(player, "playing this take…")
    errormonitor(Threads.@spawn begin
        try
            write(process, reinterpret(UInt8, nar.samples))
            close(process.in)
        catch e
            e isa Base.IOError || rethrow()     # the pipe closed: pw-cat ended, or was stopped
        end
        wait(process)
        put!(player.uiqueue, () -> listened!(player, process))
    end)
    return nothing
end

"A take finished playing. Says so, unless it was stopped or replaced first."
function listened!(player::Player, process)
    listening = player.fxwidgets[:listening]
    listening[] !== nothing && listening[].process === process || return nothing
    listening[] = nothing
    setstatus!(player, success(process) ? "played this take" :
                       "could not play this take: pw-cat exited with $(process.exitcode)")
    return nothing
end

"Stop the take being auditioned, if any."
function stoplistening!(player::Player)
    listening = player.fxwidgets[:listening]
    playing = listening[]
    playing === nothing && return nothing
    listening[] = nothing
    process_running(playing.process) && kill(playing.process)
    return nothing
end
