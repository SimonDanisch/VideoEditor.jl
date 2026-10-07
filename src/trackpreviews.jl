# Derived timeline media, owned by the editor rather than the document.
# CPU audio analysis and small scene renders share a bounded work queue. Workers
# publish cache entries only; Makie observables are refreshed on the UI task.
struct WaveEnvelope{A <: AbstractMatrix}
    samples::A                  # original PCM/mmap, never a second full audio copy
    rate::Int
    block::Int
    levels::Vector{Tuple{Matrix{Float32},Matrix{Float32}}}
end

function waveenvelope(samples, rate::Integer; block=64, running=()->true)
    data = samples isa AbstractVector ? reshape(samples,1,:) : samples
    channels,n = size(data)
    levels = Tuple{Matrix{Float32},Matrix{Float32}}[]
    n == 0 && return WaveEnvelope(data,Int(rate),block,levels)
    scale = eltype(data) <: Integer ? Float32(typemax(eltype(data))) : 1f0
    lo,hi = zeros(Float32,channels,cld(n,block)),zeros(Float32,channels,cld(n,block))
    for b in axes(lo,2), ch in 1:channels
        a,z = (b-1)*block+1,min(b*block,n)
        mn,mx = Inf32,-Inf32
        @inbounds for i in a:z
            v = Float32(data[ch,i])/scale
            isfinite(v) || (v=0f0)
            mn=min(mn,v);mx=max(mx,v)
        end
        lo[ch,b]=mn;hi[ch,b]=mx
        if b % 4096 == 0
            running() || return nothing
            yield()
        end
    end
    push!(levels,(lo,hi))
    while size(lo,2)>1
        lower,upper = similar(lo,channels,cld(size(lo,2),2)),similar(hi,channels,cld(size(hi,2),2))
        for b in axes(lower,2),ch in 1:channels
            a,z=2b-1,min(2b,size(lo,2))
            lower[ch,b]=min(lo[ch,a],lo[ch,z]);upper[ch,b]=max(hi[ch,a],hi[ch,z])
        end
        lo,hi=lower,upper;push!(levels,(lo,hi))
    end
    return WaveEnvelope(data,Int(rate),block,levels)
end

"Exact extrema in a half-open sample interval, using cached powers of two."
function waveextrema(e::WaveEnvelope,ch::Int,a::Int,b::Int)
    a,b=clamp(a,0,size(e.samples,2)),clamp(b,0,size(e.samples,2))
    a<b || return (0f0,0f0)
    mn,mx=Inf32,-Inf32
    scale=eltype(e.samples)<:Integer ? Float32(typemax(eltype(e.samples))) : 1f0
    # Partial end blocks are read exactly, so a transient outside a cut cannot
    # leak into the visible waveform. Whole blocks use the envelope pyramid.
    while a<b
        if a%e.block!=0 || b-a<e.block
            z=min(b,(fld(a,e.block)+1)*e.block)
            @inbounds for i in a+1:z
                v=Float32(e.samples[ch,i])/scale
                isfinite(v)||(v=0f0)
                mn=min(mn,v);mx=max(mx,v)
            end
            a=z
        else
            bin=fld(a,e.block)
            level=1
            while level<length(e.levels) && bin%(1<<level)==0 && e.block*(1<<level)<=b-a
                level+=1
            end
            lo,hi=e.levels[level];i=fld(bin,1<<(level-1))+1
            mn=min(mn,lo[ch,i]);mx=max(mx,hi[ch,i]);a+=e.block*(1<<(level-1))
        end
    end
    return (mn,mx)
end

"At most one extrema segment per screen pixel and channel, in timeline time."
function wavepoints!(points,e::WaveEnvelope,t0,t1,source0,speed,lo,hi,pps,y0,y1;gain=1)
    a,b=max(t0,lo),min(t1,hi)
    (a<b && speed>0 && pps>0 && !isempty(e.levels)) || return points
    count=clamp(ceil(Int,(b-a)*pps),1,4096)
    channels=size(e.samples,1)
    lower,upper=last(e.levels)
    peak=max(maximum(abs,lower),maximum(abs,upper))
    displaygain=peak>0 ? gain/peak : gain
    for i in 1:count
        x0=a+(i-1)*(b-a)/count;x1=a+i*(b-a)/count
        firstsample=floor(Int,(source0+(x0-t0)*speed)*e.rate)
        lastsample=ceil(Int,(source0+(x1-t0)*speed)*e.rate)
        x=(x0+x1)/2
        for ch in 1:channels
            mn,mx=waveextrema(e,ch,firstsample,lastsample)
            center=y0+(ch-.5)*(y1-y0)/channels
            amplitude=.46*(y1-y0)/channels
            push!(points,Point2f(x,center+clamp(mn*displaygain,-1,1)*amplitude),
                         Point2f(x,center+clamp(mx*displaygain,-1,1)*amplitude))
        end
    end
    return points
end

struct PreviewRequest
    kind::Symbol
    build::Function
    position::Float64          # sequence time, not source time or GUI callback order
end
previewpriority(request::PreviewRequest) = (request.kind === :audio ? 0 : 1, request.position)

mutable struct TimelinePreviews
    lock::ReentrantLock
    running::Threads.Atomic{Bool}
    dirty::Threads.Atomic{Bool}
    quiet_until::Threads.Atomic{Float64}
    entries::Dict{Any,Any}
    stamps::Dict{Any,Int}
    pending::Dict{Any,PreviewRequest}
    order::Vector{Any}
    active::Any
    bytes::Int
    counter::Int
    maxbytes::Int
    task::Task
end
previewbytes(e::WaveEnvelope)=sum(sizeof(a)+sizeof(b) for (a,b) in e.levels;init=0)
previewbytes(e::RGBFrame)=sizeof(e)
previewbytes(::Nothing)=0

function TimelinePreviews(player;maxbytes=64*2^20)
    p=TimelinePreviews(ReentrantLock(),Threads.Atomic{Bool}(true),Threads.Atomic{Bool}(false),Threads.Atomic{Float64}(time()+.5),
        Dict{Any,Any}(),Dict{Any,Int}(),Dict{Any,PreviewRequest}(),Any[],nothing,0,0,maxbytes,Task(()->nothing))
    for event in (player.playhead,player.edited)
        push!(player.globallisteners,on(_ -> (p.quiet_until[]=time()+.2),event))
    end
    append!(player.globallisteners,onany(player.timeline.viewrange,player.edited) do _...
        p.quiet_until[]=time()+.2
        lock(p.lock) do
            for key in copy(p.order)
                p.pending[key].kind in (:scene, :image) || continue
                delete!(p.pending,key);filter!(!=(key),p.order)
            end
        end
        p.dirty[]=true
    end)
    # Navigation and typing also need the UI/GPU, even without document edits.
    # Observe before consuming handlers, so a camera drag parks scene thumbnails.
    if hasproperty(player, :fig)
        events = Makie.events(player.fig)
        for event in (events.mouseposition, events.mousebutton, events.scroll,
                      events.keyboardbutton, events.unicode_input)
            push!(player.globallisteners, on(event; priority = typemax(Int)) do _
                p.quiet_until[] = time() + .2
                Makie.Consume(false)
            end)
        end
    end
    p.task=Threads.@spawn previewloop(p,player)
    return p
end
trackpreviews(player)=get!(() -> TimelinePreviews(player),player.fxwidgets,:trackpreviews)

"A queued scene job was overtaken by input; request it again when idle."
struct DeferredScenePreview end

function scenepreviewbusy(p,player)
    return !p.running[] || player.screen===nothing || player.playing[] ||
        player.timeline.scrubbing[] || time()<p.quiet_until[]
end

function requestpreview!(build::Function,p::TimelinePreviews,key,kind::Symbol;position=Inf)
    lock(p.lock) do
        if haskey(p.entries,key)
            p.stamps[key]=(p.counter+=1)
            return p.entries[key]
        end
        p.running[] || return nothing
        request = PreviewRequest(kind, build, Float64(position))
        if haskey(p.pending,key)
            old = p.pending[key]
            # A shared source frame can appear in several cuts. Its earliest
            # visible occurrence determines when to build it, once.
            p.pending[key] = PreviewRequest(old.kind, old.build, min(old.position, request.position))
        else
            if length(p.order) >= 256
                latest = argmax(k -> previewpriority(p.pending[k]), p.order)
                previewpriority(request) < previewpriority(p.pending[latest]) || return nothing
                delete!(p.pending, latest); filter!(!=(latest), p.order)
            end
            p.pending[key]=request;push!(p.order,key)
        end
        return nothing
    end
end

function storepreview!(p::TimelinePreviews,key,value)
    lock(p.lock) do
        delete!(p.pending,key)
        filter!(!=(key), p.order)
        p.running[] || return
        p.bytes -= previewbytes(get(p.entries, key, nothing))
        previewbytes(value)>p.maxbytes && (value=nothing)
        p.entries[key]=value;p.stamps[key]=(p.counter+=1);p.bytes+=previewbytes(value)
        while (p.bytes>p.maxbytes && length(p.entries)>1) || length(p.entries)>1024
            oldest=argmin(p.stamps)
            p.bytes-=previewbytes(pop!(p.entries,oldest));delete!(p.stamps,oldest)
        end
        p.dirty[]=true
    end
    return nothing
end

function deferpreview!(p,key)
    lock(p.lock) do
        delete!(p.pending,key)
    end
    p.dirty[]=true
    return nothing
end

function previewloop(p,player)
    try
        while p.running[]
            # Audio is CPU-only. Scene jobs wait for a parked editor and yield
            # between tiny frames instead of monopolising its GPU while editing.
            busy=scenepreviewbusy(p,player)
            job=lock(p.lock) do
                # GUI callbacks can request clips in any order. Build exact
                # visible thumbnails left to right, with stable FIFO ties.
                # CPU waveform work remains available while scene rendering waits.
                i = nothing
                for j in eachindex(p.order)
                    request = p.pending[p.order[j]]
                    if i === nothing || previewpriority(request) < previewpriority(p.pending[p.order[i]])
                        i = j
                    end
                end
                i===nothing && return nothing
                busy && p.pending[p.order[i]].kind === :scene && return nothing
                key=splice!(p.order,i)
                return (key,p.pending[key].build)
            end
            if job===nothing
                sleep(.08);continue
            end
            key,build=job
            try
                value=Base.invokelatest(build)
                if value isa DeferredScenePreview
                    # Do not cache a deferral as a missing/unsupported frame.
                    # Refresh visible tracks so only still-needed jobs return.
                    deferpreview!(p,key)
                else
                    storepreview!(p,key,value)
                end
            catch e
                if e isa RenderThreadTimeout
                    # Initial foreground compilation may occupy the renderer
                    # for longer than its startup deadline. Retry when idle;
                    # this is not an unsupported source or a missing frame.
                    deferpreview!(p,key)
                else
                    @warn "track preview unavailable" kind=first(key) exception=(e,catch_backtrace())
                    storepreview!(p,key,nothing) # no retry storm for an unsupported source
                end
            end
            sleep(.03)
        end
    finally
        closepreviewscene!(p)
    end
end

function stop!(p::TimelinePreviews)
    p.running[]=false
    lock(p.lock) do;empty!(p.pending);empty!(p.order);end
    wait(p.task)
    empty!(p.entries);empty!(p.stamps)
    return nothing
end

function previewtick!(tl)
    player=editorof(tl.sequence)
    player===nothing && return false
    p=get(player.fxwidgets,:trackpreviews,nothing)
    if p!==nothing && p.dirty[]
        p.dirty[]=false
        return true
    end
    return false
end

function audioenvelope(player,n::Narration)
    n.rate>0 && !isempty(n.samples) || return nothing
    p=trackpreviews(player);key=(:voice,objectid(n.samples),length(n.samples),n.rate)
    return requestpreview!(p,key,:audio) do
        waveenvelope(n.samples,n.rate;running=()->p.running[])
    end
end

function audioenvelope(player,source::ClipSource)
    path=audiopath(source)
    isempty(path) && return nothing
    isfile(path) || return nothing
    st=stat(path);key=(:audio,abspath(path),st.size,st.mtime)
    p=trackpreviews(player)
    return requestpreview!(p,key,:audio) do
        pcm=loadpcm(source)
        block=pcm===nothing ? 64 : max(64,nextpow(2,cld(pcm.nsamples*2*16,p.maxbytes÷2)))
        pcm===nothing ? nothing : waveenvelope(pcm.samples,AUDIORATE;block,running=()->p.running[])
    end
end

function clipwaveform(tl,clip)
    # Presence is document data; availability of the analysed PCM is not layout.
    isempty(audiopath(clip.source)) && return nothing
    return (range,view,pps,lo,hi)->begin
        player=editorof(tl.sequence)
        player===nothing && return Point2f[]
        max(range[1],view[1])<min(range[2],view[2]) || return Point2f[]
        e=audioenvelope(player,clip.source)
        e===nothing && return Point2f[]
        speed=clip.rate*tl.sequence.framerate/clip.source.framerate
        wavepoints!(Point2f[],e,range...,clip.src_in/clip.source.framerate,speed,
                    view...,pps,lo,hi)
    end
end

function scenepreviewidentity(src::SceneSource)
    root=src.root
    recipe=root isa SceneProgram ? (root.file,root.package,root.entry,root.args) : root
    return hash((recipe,src.build,src.backend,src.screenopts,src.width/src.height,src.framerate))
end
function scenethumbdims(src::SceneSource)
    height=max(8,min(88,round(Int,320*src.height/src.width)))
    return (max(8,min(320,round(Int,height*src.width/src.height))),height)
end
function scenepreviewrevision(c::Clip)
    hash((c.crop,objectid(c.look),[(fx.kind,fx.enabled[],
        [(p.name,p.curve[].keys) for p in fx.params if !isfollowing(p) && !followsrecording(c.source,p)]) for fx in c.effects]))
end

function closethumbsource!(src,engine)
    src===nothing && return
    close(src)
    engine===nothing || onthread(mantlethread(Mantle)) do; emptyengine!(engine); end
end
function closepreviewscene!(p)
    p.active===nothing && return
    _,src,engine=p.active
    closethumbsource!(src,engine);p.active=nothing
end

function scenethumbnail!(p,player,c,second,identity)
    # The queued clip carries frozen edit curves; the background render owns a
    # separate tiny scene and never seeks/resizes the editor's standing scene.
    snapshot=let
        sf=clamp(round(Int,second*c.source.framerate),c.src_in,c.src_out-1)
        effects=Effect[Effect(fx.id,fx.kind,fx.enabled[],
            Param[Param(q.name,q.label,valueat(q,sf);range=q.range) for q in fx.params if !isfollowing(q) && !followsrecording(c.source,q)]) for fx in c.effects]
        (sf,effects,c.crop,c.look)
    end
    original=c.source;backend=getbackend(original.backend)
    rendered=onthread(renderthread(backend)) do;lock(SCENELOCK) do
        # This task may have waited in the renderer's queue since the idle check
        # in previewloop. New input must take precedence over an obsolete job.
        scenepreviewbusy(p,player) && return DeferredScenePreview()
        if p.active===nothing || first(p.active)!=identity
            closepreviewscene!(p)
            opts=copy(original.screenopts)
            original.backend===:RayMakie && (opts[:rasterize]=true)
            width,height=scenethumbdims(original)
            src=SceneSource(original.root;joints=deepcopy(original.joints),camera=deepcopy(original.camera),
                build=deepcopy(original.build),
                backend=original.backend,screenopts=opts,width,height,
                framerate=original.framerate,nframes=original.nframes)
            engine=onthread(mantlethread(Mantle)) do; FxEngine(player.analysisbackend); end
            p.active=(identity,src,engine)
        end
        _,src,engine=p.active
        sf,effects,crop,look=snapshot
        tiny=withfields(c;source=src,src_in=sf,src_out=sf+1,start=0,rate=1.,effects,crop,look,bake=nothing)
        scale=min(1,src.height/original.height)
        prerender!(src,tiny,sf;detail_scale=scale)
        return tiny,engine
    end;end
    rendered isa DeferredScenePreview && return rendered
    tiny,engine=rendered
    return onthread(mantlethread(Mantle)) do
        scenepreviewbusy(p,player) && return DeferredScenePreview()
        result=Ref{RGBFrame}()
        composite(engine,[tiny],0,(_,_) -> nothing;canvas=(tiny.source.width,tiny.source.height)) do frame
            result[]=copy(frame)
        end
        result[]
    end
end

function scenethumbsfor(tl,c)
    player=editorof(tl.sequence);p=trackpreviews(player)
    revision=Ref(-1);signature=Ref(UInt(0))
    frozen=Ref(c)
    return second->begin
        if revision[]!=player.edited[]
            signature[]=scenepreviewrevision(c);revision[]=player.edited[]
            # Freeze only actual overrides, once per edit revision. A queued
            # frame must keep the values its cache key describes even if the
            # user edits again before the background worker reaches it.
            effects=Effect[Effect(fx.id,fx.kind,fx.enabled[],
                Param[copyparam(q) for q in fx.params if !isfollowing(q) && !followsrecording(c.source,q)]) for fx in c.effects]
            frozen[]=withfields(c;effects)
        end
        identity=scenepreviewidentity(c.source)
        sf=clamp(round(Int,second*c.source.framerate),c.src_in,c.src_out-1)
        position = timelineframe(c,sf) / tl.sequence.framerate
        if hasbakedframe(c,sf)
            file=bakeframefile(c.bake,sf);st=stat(file)
            key=(:bake,file,st.size,st.mtime,scenethumbdims(c.source))
            return requestpreview!(p,key,:image;position) do
                image=RGB{N0f8}.(PNGFiles.load(file))
                downscale(image,scenethumbdims(c.source)...)
            end
        end
        key = (:scene, identity, signature[], sf)
        snapshot = frozen[]
        return requestpreview!(p,key,:scene;position) do
            scenethumbnail!(p,player,snapshot,sf/snapshot.source.framerate,identity)
        end
    end
end
