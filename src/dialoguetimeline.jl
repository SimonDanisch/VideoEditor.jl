"Visible intersections of spoken takes with the cut list, in timeline seconds."
function dialoguespans(seq::Sequence)
    spans = NamedTuple[]
    for (i,n) in enumerate(seq.narration)
        duration = n.rate > 0 ? min(length(n.samples)/n.rate,n.mix.stop) :
            isfinite(n.mix.stop) ? n.mix.stop : max(0.5,0.3length(split(n.text)))
        duration > 0 || continue
        if n.anchor === nothing
            push!(spans,(index=i,start=n.at,stop=n.at+duration))
            continue
        end
        for c in seq.clips
            c.source === n.anchor || continue
            lo = max(n.at,c.src_in/c.source.framerate)
            hi = min(n.at+duration,c.src_out/c.source.framerate)
            lo < hi || continue
            speed = c.rate*seq.framerate/c.source.framerate
            at = c.start/seq.framerate
            offset = c.src_in/c.source.framerate
            push!(spans,(index=i,start=at+(lo-offset)/speed,stop=at+(hi-offset)/speed))
        end
    end
    return sort!(spans;by=s->s.start)
end

"Pack overlapping takes into separate rows so neither text nor waveforms hide."
function dialoguerows(spans)
    ends=Float64[];rows=Int[]
    for span in spans
        row=findfirst(t->t<=span.start,ends)
        if row===nothing
            push!(ends,span.stop);row=length(ends)
        else
            ends[row]=span.stop
        end
        push!(rows,row)
    end
    return rows
end

function dialogueclock(seq,n,span)
    n.anchor===nothing && return (span.start-n.at,1.)
    i=findfirst(c -> c.source===n.anchor &&
        c.start/seq.framerate<=span.start<clipend(c)/seq.framerate,seq.clips)
    i===nothing && return nothing
    c=seq.clips[i]
    speed=c.rate*seq.framerate/c.source.framerate
    return (c.src_in/c.source.framerate+(span.start-c.start/seq.framerate)*speed-n.at,speed)
end

"A linked speech lane backed directly by Sequence.narration and its existing editor."
function dialoguetimeline!(player::Player, gridpos)
    seq,tl = player.sequence,player.timeline
    axis = Axis(gridpos; title="Dialogue · click a line to edit its words, voice and delivery",
        titlealign=:left,titlesize=11,yzoomlock=true,ypanlock=true,
        xgridvisible=false,ygridvisible=false,backgroundcolor=tl.colors.background)
    hidedecorations!(axis);hidespines!(axis)
    deregister_interaction!(axis,:rectanglezoom)
    limits!(axis,0,max(seqduration(seq),1),0,1)
    linkxaxes!(tl.axis,axis)
    rects = Observable(Rect2f[])
    positions = Observable(Point2f[])
    labels = Observable(String[])
    colors = Observable(RGBAf[])
    selected = Observable(get(player.fxwidgets,:narration_index,1))
    player.fxwidgets[:dialogue_selected] = selected
    boxes = poly!(axis,rects;color=colors,strokecolor=tl.colors.border,strokewidth=1)
    words = text!(axis,positions;text=labels,fontsize=11,color=tl.colors.text,
                  align=(:left,:center),offset=(5,0))
    translate!(words,0,0,2)
    waveform = Observable(Point2f[])
    waves = linesegments!(axis,waveform;color=tl.colors.text,linewidth=1)
    translate!(waves,0,0,1)
    vlines!(axis,map(n->[n/seq.framerate],player.playhead);color=tl.colors.accent,linewidth=2)
    spans = Ref(NamedTuple[])
    clocks = Ref(Union{Nothing,Tuple{Float64,Float64}}[])
    rows = Ref(Int[])
    displayrows = Ref(Int[1])
    signature = Ref{Any}(nothing)
    visibility = Ref{Any}(nothing)
    function refresh(; document=false)
        if document
            current = (copy(seq.narration),[(c.id,c.start,c.src_in,c.src_out,c.rate,c.source) for c in seq.clips])
            current == signature[] && return nothing
            signature[] = current
            spans[] = dialoguespans(seq)
            # Keep waveform clocks with their span snapshot. Undo can change
            # clips while layout callbacks still draw the previous snapshot.
            clocks[] = [dialogueclock(seq,seq.narration[s.index],s) for s in spans[]]
            rows[] = dialoguerows(spans[])
        end
        shown = !isempty(seq.narration)
        lo,hi = tl.viewrange[]
        displayrows[]=sort!(unique([rows[][j] for (j,s) in enumerate(spans[]) if max(lo,s.start)<min(hi,s.stop)]))
        isempty(displayrows[]) && push!(displayrows[],1)
        nrows=length(displayrows[])
        player.fxwidgets[:speechrowcount]=nrows
        if visibility[] != (shown,nrows)
            visibility[] = (shown,nrows)
            axis.blockscene.visible[] = shown
            ylims!(axis,0,nrows)
            rowsize!(player.fxwidgets[:timelinegrid],1,Makie.Fixed(shown ? 68*nrows : 0))
            fittimelinerow!(player)
        end
        pps = axis.scene.viewport[].widths[1]/max(hi-lo,eps())
        rs,ps,ls,cs,ws = Rect2f[],Point2f[],String[],RGBAf[],Point2f[]
        for (j,span) in enumerate(spans[])
            span.index<=length(seq.narration) && clocks[][j]!==nothing || continue
            a,b = max(lo,span.start),min(hi,span.stop)
            a < b || continue
            n = seq.narration[span.index]
            row=findfirst(==(rows[][j]),displayrows[])-1
            push!(rs,Rect2f(a,row+.08,b-a,.84))
            push!(cs,RGBAf(span.index == selected[] ? tl.colors.accent : tl.colors.accent_subtle))
            # Elide to this clip's pixel width. Zooming reveals the spoken text.
            budget = max(0,floor(Int,((b-a)*pps-10)/6))
            text = isempty(n.text) ? n.label : "$(n.voice): $(n.text)"
            label = budget < 3 ? "" : length(text) > budget ? first(text,budget-1)*"…" : text
            push!(ps,Point2f(a,row+.78));push!(ls,label)
            envelope=audioenvelope(player,n)
            if envelope!==nothing
                source0,speed=clocks[][j]
                wavepoints!(ws,envelope,span.start,span.stop,source0,speed,lo,hi,pps,
                    row+.14,row+.62;gain=n.mix.gain)
            end
        end
        rects[]=rs;positions[]=ps;labels[]=ls;colors[]=cs
        waveform[]=ws
        return nothing
    end
    on(_->refresh(document=true),player.edited)
    onany((_...)->refresh(),tl.viewrange,axis.scene.viewport,selected)
    on(_->refresh(),tl.refresh)
    on(events(axis.scene).mousebutton;priority=40) do event
        event.button == Mouse.left && event.action == Mouse.press || return Consume(false)
        is_mouseinside(axis.scene) || return Consume(false)
        time,y = mouseposition(axis.scene)
        rowindex=floor(Int,y)+1
        1<=rowindex<=length(displayrows[]) || return Consume(false)
        row=displayrows[][rowindex]
        i = findlast(j->rows[][j]==row && spans[][j].start<=time<spans[][j].stop,eachindex(spans[]))
        if i === nothing
            seek!(player,round(Int,time*seq.framerate))
        else
            selectnarration!(player,spans[][i].index;seek=false)
            seek!(player,round(Int,time*seq.framerate))
        end
        return Consume(true)
    end
    player.fxwidgets[:dialogue_timeline] = (;axis,rects,positions,labels,spans,rows,displayrows,waveform,refresh)
    refresh(document=true)
    return nothing
end
