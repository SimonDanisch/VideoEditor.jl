using Test, Random
import VideoEditor as VE

@testset "Undo keeps complete effects available to preview observers" begin
    source = VE.SceneSource(VE.Makie.SpecApi.Scene(); width = 160, height = 120,
                            framerate = 24, nframes = 48)
    c = VE.Clip(source)
    VE.addslot!(c, VE.Effect(VE.ColorEffect(brightness = 0.1)))
    VE.addslot!(c, VE.Effect(VE.LookEffect(0.7f0)))
    saved = VE.withfields(c; effects = copy.(c.effects))
    color = VE.findslot(c, :color)
    brightness = VE.param(color, :brightness)
    VE.setvalue!(brightness, 0.3, 0)
    observed = Int[]
    listener = VE.Observables.on(brightness.curve) do _
        # The background thumbnail path freezes and renders these entries.
        push!(observed, length(c.effects))
        for fx in c.effects
            @test length(fx.params) == length(VE.kindbyname(fx.kind).params)
            @test VE.op(fx) isa VE.FxOp
        end
    end
    try
        VE.restoreinto!(c, saved)
        @test observed == [length(saved.effects)]
        @test VE.findslot(c, :color) === color
        @test VE.param(color, :brightness) === brightness
        @test VE.valueat(brightness, 0) ≈ 0.1
    finally
        VE.Observables.off(listener)
    end
end

@testset "wave envelopes preserve transients, channels and exact cut boundaries" begin
    rng=MersenneTwister(87)
    samples=randn(rng,Float32,2,2003).*0.2f0
    samples[1,17]=1;samples[2,17]=-1
    e=VE.waveenvelope(samples,48000)
    for (a,b) in ((0,2003),(16,17),(17,66),(63,128),(512,1536),(1990,2010))
        for ch in 1:2
            expected=extrema(view(samples,ch,a+1:min(b,2003)))
            @test VE.waveextrema(e,ch,a,b)==expected
        end
    end
    @test VE.waveextrema(e,1,2003,2100)==(0f0,0f0)
    # Opposite stereo channels must not cancel as they would in a mono average.
    @test VE.waveextrema(e,1,16,17)==(1f0,1f0)
    @test VE.waveextrema(e,2,16,17)==(-1f0,-1f0)
    @test e.samples===samples
    points=VE.wavepoints!(VE.Point2f[],e,10.,11.,0.,2.,10.,11.,20.,0.,1.)
    @test length(points)==80
    @test all(p->10<=p[1]<=11,points)
    pcm=reshape(Int16[-32767,0,32767],1,:)
    @test VE.waveextrema(VE.waveenvelope(pcm,48000),1,0,3)==(-1f0,1f0)
    @test VE.dialoguerows([(start=0.,stop=1.),(start=.2,stop=.8),(start=1.,stop=2.)])==[1,2,1]
end

@testset "preview queue deduplicates, publishes asynchronously, evicts and stops" begin
    M=VE.Makie
    dummy=(screen=nothing,playing=M.Observable(false),playhead=M.Observable(0),
        edited=M.Observable(0),timeline=(scrubbing=Ref(false),viewrange=M.Observable((0.,1.))),
        globallisteners=VE.Observables.ObserverFunction[])
    p=VE.TimelinePreviews(dummy;maxbytes=600)
    gate=Channel{Nothing}(1);calls=Threads.Atomic{Int}(0);task=Ref{Task}()
    build=()->begin
        task[]=current_task();Threads.atomic_add!(calls,1);take!(gate)
        fill(VE.RGB{VE.N0f8}(1,0,0),10,10)
    end
    try
        for _ in 1:40
            @test VE.requestpreview!(build,p,(:test,1),:audio)===nothing
        end
        @test timedwait(()->calls[]==1,5)==:ok
        @test task[]!==current_task()
        put!(gate,nothing)
        @test timedwait(()->p.dirty[],5)==:ok
        @test calls[]==1
        @test VE.requestpreview!(build,p,(:test,1),:audio)!==nothing
        for i in 2:5
            VE.requestpreview!(()->fill(VE.RGB{VE.N0f8}(0,1,0),10,10),p,(:test,i),:audio)
        end
        @test timedwait(()->lock(()->isempty(p.pending),p.lock),5)==:ok
        @test p.bytes<=p.maxbytes
        @test length(p.entries)<=2
        dummy.playing[]=true
        VE.requestpreview!(() -> error("scene jobs must remain parked"),p,(:scene,:waiting),:scene)
        sleep(.15)
        @test haskey(p.pending,(:scene,:waiting))
    finally
        foreach(VE.Observables.off,dummy.globallisteners)
        VE.stop!(p)
    end
    @test istaskdone(p.task)
    @test isempty(p.pending) && isempty(p.entries)
end

@testset "timeline thumbnails stop repeating at frame resolution" begin
    tile = fill(VE.RGB{VE.N0f8}(1, 0, 0), 49, 88)
    times = Float64[]
    thumbs(t) = (push!(times, t); tile)
    # A subframe viewport reproduces the reported 32.955–32.960s zoom.
    for speed in (0.5, 1.0, 2.0), fps in (24, 30, 60)
        empty!(times)
        lo, hi = 32.955, 32.960
        strip, span, shown = VE.composetiles((0., 100.), (lo, hi), 200000., 72., 0.,
            thumbs, (49, 88), VE.RGBAf(0, 0, 0, 1);
            speed, interval = 1 / fps, frameinterval = 1 / fps)
        frames = ceil(Int, hi * fps * speed) - floor(Int, lo * fps * speed)
        @test shown
        @test length(times) <= frames
        @test length(unique(times)) == length(times)
        @test span[1] <= lo && span[2] >= hi
        @test count(c -> c == VE.RGB{VE.N0f8}(1, 0, 0), strip) <= frames * length(tile)
        @test size(strip, 1) <= ceil(Int, (hi-lo) * 200000 * 88 / 72)
    end
end

@testset "deferred previews remain retryable rather than cached as missing" begin
    M=VE.Makie
    dummy=(screen=nothing,playing=M.Observable(false),playhead=M.Observable(0),
        edited=M.Observable(0),timeline=(scrubbing=Ref(false),viewrange=M.Observable((0.,1.))),
        globallisteners=VE.Observables.ObserverFunction[])
    p=VE.TimelinePreviews(dummy)
    try
        for (i,build) in enumerate((() -> VE.DeferredScenePreview(),
                                   () -> throw(VE.RenderThreadTimeout("busy renderer"))))
            key=(:test,:deferred,i)
            p.dirty[]=false
            VE.requestpreview!(build,p,key,:audio)
            @test timedwait(()->p.dirty[],5)==:ok
            @test !haskey(p.entries,key)
            @test !haskey(p.pending,key)
            value=fill(VE.RGB{VE.N0f8}(0,1,0),10,10)
            VE.requestpreview!(() -> value,p,key,:audio)
            @test timedwait(()->haskey(p.entries,key),5)==:ok
            @test p.entries[key]===value
        end
    finally
        foreach(VE.Observables.off,dummy.globallisteners)
        VE.stop!(p)
    end
end

@testset "exact thumbnails build in timeline order and yield to editing" begin
    M=VE.Makie
    dummy=(screen=:window,playing=M.Observable(true),playhead=M.Observable(0),
        edited=M.Observable(0),timeline=(scrubbing=Ref(false),viewrange=M.Observable((0.,10.))),
        globallisteners=VE.Observables.ObserverFunction[])
    p=VE.TimelinePreviews(dummy)
    order=Symbol[]
    tile=fill(VE.RGB{VE.N0f8}(1,0,0),10,10)
    try
        # Reversed clips, mixed baked/live frames, and unrelated source-frame
        # numbers must still follow their positions in the edited movie.
        for (key,kind,position) in ((:last,:scene,9.),(:baked,:image,4.),
                                   (:first,:scene,1.),(:second,:scene,2.))
            VE.requestpreview!(() -> (push!(order,key);tile),p,(:test,key),kind;position)
        end
        VE.requestpreview!(() -> (push!(order,:audio);tile),p,(:audio,),:audio)
        @test timedwait(()->order==[:audio],5)==:ok
        @test all(haskey(p.pending,(:test,k)) for k in (:last,:baked,:first,:second))
        p.quiet_until[]=0;dummy.playing[]=false
        @test timedwait(()->length(order)==5,5)==:ok
        @test order==[:audio,:first,:second,:baked,:last]

        empty!(order);dummy.playing[]=true
        # Admit the batch atomically so even the CPU tile waits for its ordering.
        lock(p.lock) do
            for (key,position) in ((:last2,8.),(:baked2,4.),(:shared,7.),(:first2,1.))
                VE.requestpreview!(() -> (push!(order,key);tile),p,(:test,key),
                                   key===:baked2 ? :image : :scene;position)
            end
            VE.requestpreview!(() -> error("duplicate must not rebuild"),p,(:test,:shared),:scene;position=2.)
            @test p.pending[(:test,:shared)].position==2.
            p.quiet_until[]=0;dummy.playing[]=false
        end
        @test timedwait(()->length(order)==4,5)==:ok
        @test order==[:first2,:shared,:baked2,:last2]

        dummy.playing[]=true
        VE.requestpreview!(() -> error("obsolete viewport"),p,(:scene,:obsolete),:scene;position=0.)
        dummy.timeline.viewrange[]=(20.,30.)
        @test !haskey(p.pending,(:scene,:obsolete))
        @test isempty(p.order)
        # Earlier visible tiles must not be rejected behind a full later queue.
        for i in 1:256
            VE.requestpreview!(() -> tile,p,(:scene,:full,i),:scene;position=100+i)
        end
        VE.requestpreview!(() -> tile,p,(:scene,:earliest),:scene;position=20.)
        @test length(p.order)==256
        @test haskey(p.pending,(:scene,:earliest))
        @test !haskey(p.pending,(:scene,:full,256))
        dummy.edited[]+=1
        @test isempty(p.pending) && isempty(p.order)

        # Cache replacement counts memory once and leaves no stale queue item.
        key=(:scene,:queued)
        VE.requestpreview!(() -> error("fulfilled frame must not render again"),p,key,:scene)
        before=p.bytes
        VE.storepreview!(p,key,tile)
        VE.storepreview!(p,key,copy(tile))
        @test p.bytes==before+sizeof(tile)
        @test !haskey(p.pending,key) && !(key in p.order)
        @test !istaskdone(p.task)
    finally
        foreach(VE.Observables.off,dummy.globallisteners);VE.stop!(p)
    end
end

@testset "filmstrip slots show only their requested frames" begin
    red=fill(VE.RGB{VE.N0f8}(1,0,0),49,88)
    green=fill(VE.RGB{VE.N0f8}(0,1,0),49,88)
    ready=Dict(0.0 => red)
    times=Float64[]
    provider=t->(push!(times,t);get(ready,t,nothing))
    strip,_,shown=VE.composetiles((0.,2.),(0.,2.),100.,72.,0.,provider,
        (49,88),VE.RGBAf(0,0,0,1);interval=.25,frameinterval=1/24)
    @test shown && issorted(times)
    @test all(==(first(red)),strip[1:49,:])
    @test all(==(VE.RGB{VE.N0f8}(0,0,0)),strip[50:end,:])
    ready[times[2]]=green
    strip,_,_=VE.composetiles((0.,2.),(0.,2.),100.,72.,0.,provider,
        (49,88),VE.RGBAf(0,0,0,1);interval=.25,frameinterval=1/24)
    @test all(==(first(red)),strip[1:49,:])
    @test all(==(first(green)),strip[50:98,:])
    @test all(==(VE.RGB{VE.N0f8}(0,0,0)),strip[99:end,:])
end
