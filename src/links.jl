# Effects that reference other effects.
#
# A blend is two halves of one idea: the outgoing clip fades out while the
# incoming one fades in, and the length is shared. Before this, that was
# expressed as `Clip.blendfrom` — a clip-level field only the blend code knew
# about, invisible in the panel, and unavailable to anything else that wants to
# say "this effect is tied to that one".
#
# A LINK is a reference from one slot to another. The parent's card shows the
# target's parameters INLINE — flat, at full width, not as a card inside a card,
# so a chain of links does not shrink each step into a strip. The target keeps
# its own card in the list; editing either edits the same slot.

# `FxLink` itself is declared in clips.jl, next to the `Effect` that holds one.
"""
    FxLink(slot; clip = 0, role = :linked)

A reference to slot `slot` on clip `clip` (`0` = the same clip). Both are the
stable ids that survive sorting, undo and a project round trip — a link by index
or by object identity would not. `role` says what the target IS to the parent, so
a card can label the link ("fades into") instead of showing a bare id.
"""
FxLink(slot::Integer; clip::Integer = 0, role::Symbol = :linked) =
    FxLink(UInt64(clip), UInt64(slot), role)

linkdict(l::FxLink) = Dict{String, Any}("clip" => string(l.clip), "slot" => string(l.slot),
                                        "role" => String(l.role))
linkfromdict(d::AbstractDict) = FxLink(parse(UInt64, d["clip"]), parse(UInt64, d["slot"]),
                                       Symbol(get(d, "role", "linked")))

"""
    resolvelink(seq, clip, link) -> (clip, slot) | nothing

Follow `link` from `clip`. Returns `nothing` when the target is gone — a link
whose target was deleted is DANGLING, and every reader has to be able to say so
rather than fail; [`prunelinks!`](@ref) is what removes them.
"""
function resolvelink(seq::Sequence, clip::Clip, link::FxLink)
    target = link.clip == 0 ? clip : clipbyid(seq, link.clip)
    target === nothing && return nothing
    slot = findslot(target, link.slot)
    slot === nothing && return nothing
    return (target, slot)
end

"""
    linkeffects!(seq, a, sa, b, sb; role, backrole = role) -> nothing

Link slot `sa` on clip `a` to slot `sb` on clip `b`, and back again. Links are
made in PAIRS because that is what "these two are the same edit" means: removing
either side has to be able to find the other.
"""
function linkeffects!(seq::Sequence, a::Clip, sa::Effect, b::Clip, sb::Effect;
                      role::Symbol = :linked, backrole::Symbol = role)
    push!(sa.links, FxLink(sb.id; clip = b.id, role))
    push!(sb.links, FxLink(sa.id; clip = a.id, role = backrole))
    return nothing
end

"""
    prunelinks!(seq) -> Int

Drop every link whose target no longer exists, and return how many went.

A dangling link is not an error to hide: it means the user deleted one half of a
paired edit, and the other half now stands alone. Callers say so in the status
bar rather than leaving a card claiming a partner it does not have.
"""
function prunelinks!(seq::Sequence)
    n = 0
    for clip in seq.clips, slot in clip.effects
        before = length(slot.links)
        filter!(l -> resolvelink(seq, clip, l) !== nothing, slot.links)
        n += before - length(slot.links)
    end
    return n
end

"""
    linkchain(seq, clip, slot; seen = Set()) -> Vector{Tuple{Clip, Effect, Symbol}}

Everything `slot` links to, transitively, each with the role it was reached by —
and each visited at most once. The guard is not paranoia: a blend links both ways
by construction, so the first thing any walk hits is the way back.
"""
function linkchain(seq::Sequence, clip::Clip, slot::Effect;
                   seen::Set{UInt64} = Set{UInt64}([slot.id]))
    out = Tuple{Clip, Effect, Symbol}[]
    for l in slot.links
        r = resolvelink(seq, clip, l)
        r === nothing && continue
        tclip, tslot = r
        tslot.id in seen && continue
        push!(seen, tslot.id)
        push!(out, (tclip, tslot, l.role))
        append!(out, linkchain(seq, tclip, tslot; seen))
    end
    return out
end

"""
    linkedparams(seq, clip, slot) -> Vector{Tuple{Clip, Effect, EffectKind, Symbol}}

What a card should show inline: the DIRECT links of `slot` that have parameters
to show, with the kind needed to render them.

Direct only, and flat: the target's own card shows ITS links, so following the
chain here would draw the same parameters twice and nest a little further each
time. That is the rule Simon set — link the parameters in the parent card
instead of inlining a smaller card.
"""
function linkedparams(seq::Sequence, clip::Clip, slot::Effect)
    out = Tuple{Clip, Effect, EffectKind, Symbol}[]
    for l in slot.links
        r = resolvelink(seq, clip, l)
        r === nothing && continue
        tclip, tslot = r
        k = effectkindfor(op(tslot))
        k === nothing && continue
        push!(out, (tclip, tslot, k, l.role))
    end
    return out
end

"""
How a link reads in a card header: the role in the user's words, not the symbol.
"""
linkrolelabel(role::Symbol) =
    role === :fadesinto ? "fades into" :
    role === :fadesfrom ? "fades from" :
    role === :drives    ? "drives" :
    role === :drivenby  ? "driven by" : "linked to"

"""
    linkcaption(seq, clip, slot, target, role) -> String

The line that goes above an inlined parameter group: which effect, on which clip,
and how it is related. A link the user cannot trace back to a card in the list is
just a mystery slider.
"""
function linkcaption(seq::Sequence, clip::Clip, target::Clip, tslot::Effect, role::Symbol)
    k = effectkindfor(op(tslot))
    name = k === nothing ? string(nameof(typeof(op(tslot)))) : k.label
    if target === clip
        return "↗ $(linkrolelabel(role)) $name — same clip"
    end
    i = something(findfirst(c -> c === target, seq.clips), 0)
    return "↗ $(linkrolelabel(role)) $name on clip $i"
end

# ------------------------------------------------------------------- the card

"""
    linkedgroups!(player, pos, clip, slot, uicolors, kfaccstate, updatekfaccs)

Draw one inlined parameter group per DIRECT link of `slot`, into `pos`.

Each group is the target's own parameters — the same [`paramform!`](@ref) the
target's card uses, writing to the same slot — under a caption that names it and
a button that takes you to its card. Flat and full width: a card inside a card
would give every step of a chain less room than the last.

Returns the blocks it made, so the card can drop them on a rebuild.
"""
function linkedgroups!(player::Player, pos, clip::Clip, slot::Effect, uicolors,
                       kfaccstate, updatekfaccs)
    made = Any[]
    groups = linkedparams(player.sequence, clip, slot)
    isempty(groups) && return made
    gl = GridLayout(pos)
    push!(made, gl)
    for (r, (tclip, tslot, kind, role)) in enumerate(groups)
        box = GridLayout(gl[r, 1])
        # The tint wraps the caption AND the parameters — it is one borrowed group,
        # and a box around the caption alone reads as a header for the sliders
        # below it rather than as "these sliders are not mine".
        Box(box[1:2, 1]; color = uicolors.select_subtle, strokecolor = uicolors.select,
            strokewidth = 1, cornerradius = 4, tellwidth = false, tellheight = false)
        head = GridLayout(box[1, 1]; alignmode = Makie.Outside(8, 6, 2, 5))
        # WRAPPED and narrow: a Label neither truncates nor ellipsizes, so a caption
        # wider than its column is simply drawn under the button next to it.
        Label(head[1, 1], wraptext(linkcaption(player.sequence, clip, tclip, tslot, role), 30);
              halign = :left, justification = :left, fontsize = 10,
              color = uicolors.select, tellwidth = false)
        go = Button(head[1, 2]; label = "show", width = 46, height = 18, fontsize = 10,
                    halign = :right)
        colsize!(head, 1, Makie.Auto(true, 1.0f0))
        on(_ -> revealslot!(player, tclip, tslot), go.clicks)
        # Indented, so the borrowed parameters sit visibly inside the group — and
        # narrower by the same amount, or the indent pushes the ◀◆▶ trio off the
        # card's right edge.
        body = GridLayout(box[2, 1]; alignmode = Makie.Outside(12, 6, 8, 2))
        push!(made, paramform!(player, body[1, 1], tclip, tslot, kind, uicolors,
                               kfaccstate, updatekfaccs;
                               labelcolor = uicolors.select, widgetwidth = 114))
    end
    return made
end

"""
    revealslot!(player, clip, slot)

Take the user to a linked effect's own card: put the playhead on its clip if it
is elsewhere, select the card, and say so.

Selection is the feedback — the card draws its accent outline — which is why the
link button does not merely scroll. "It jumped somewhere" and "it jumped HERE"
are different messages.
"""
function revealslot!(player::Player, clip::Clip, slot::Effect)
    cur = editclip(player)
    if cur === nothing || cur[1] !== clip
        seek!(player, clamp(clip.start, 0, max(seqlength(player.sequence) - 1, 0)))
        # `selected` is the clip's INDEX in the sequence, not its id
        i = findfirst(c -> c === clip, player.sequence.clips)
        i === nothing || (player.timeline.selected[] = i)
    end
    selectfxcard!(player, (:fx, slot.id))
    k = effectkindfor(op(slot))
    setstatus!(player, "showing $(k === nothing ? "the linked effect" : k.label)")
    notify(player.playhead)
    return nothing
end
