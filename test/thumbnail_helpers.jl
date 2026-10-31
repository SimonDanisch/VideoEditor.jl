# Observe the real background renderer; do not replace tiles or scene builders.
module ThumbnailWalkthroughActions
using Test
import VideoEditor as VE
const M=VE.Makie
const FI=Main.FakeInteraction
using Main.EditingWalkthroughActions

"A late waveform in one clip must not replace its already requested time grid."
function checkwaveformarrival(p,out;clip=first(p.sequence.clips))
    manager=p.fxwidgets[:trackpreviews]
    plot=clip.view.plot
    wave=plot.waveform[];thumbs=plot.thumbs[]
    @test wave!==nothing
    arrived=Ref(false);times=Float64[]
    try
        plot.waveform=(args...)->arrived[] ? wave(args...) : VE.Point2f[]
        plot.thumbs=t->(push!(times,t);thumbs(t))
        # First let the real worker finish the clip's pre-waveform thumbnails.
        recordactions(p,out,"before_waveform_arrival",[
            FI.WaitUntil(()->lock(()->isempty(manager.pending),manager.lock);timeout=300),
            FI.Wait(.3)])
        manager.quiet_until[]=Inf
        @test timedwait(()->lock(()->length(manager.pending)==length(manager.order),manager.lock),300)==:ok
        empty!(times);plot.refresh=plot.refresh[]+1
        M.colorbuffer(p.screen)
        before=copy(times);picture=copy(plot.strip[])
        geometry=(plot.stripx[],plot.stripy[])
        @test isempty(plot.wavepoints[])
        @test !isempty(before) && issorted(before)
        @test lock(()->isempty(manager.pending),manager.lock)
        empty!(times)
        # This event models the asynchronous envelope becoming available. It is
        # the same refresh path used by the real worker, with no camera/zoom/edit.
        recordactions(p,out,"one_clip_waveform_arrival",[
            FI.Lazy(_->begin
                arrived[]=true;plot.refresh=plot.refresh[]+1
                FI.Wait(.3)
            end)])
        @test !isempty(plot.wavepoints[])
        @test times==before
        @test (plot.stripx[],plot.stripy[])==geometry
        @test plot.strip[]==picture
        @test lock(()->isempty(manager.pending),manager.lock)
    finally
        plot.waveform=wave;plot.thumbs=thumbs
        manager.quiet_until[]=0
    end
    return nothing
end

function recordarrival(p,out)
    manager=p.fxwidgets[:trackpreviews]
    recordactions(p,out,"thumbnail_layout",[FI.Wait(.2)])
    manager.quiet_until[]=Inf
    @test timedwait(()->lock(()->length(manager.pending)==length(manager.order),manager.lock),300)==:ok
    arrivals=Float64[]
    expected=Float64[]
    lock(manager.lock) do
        # Keep CPU waveform results, and simulate a cold thumbnail cache at the
        # current settled zoom/lane layout. The provider requests exact tiles.
        for key in collect(keys(manager.entries))
            first(key) in (:scene,:bake) || continue
            manager.bytes-=VE.previewbytes(pop!(manager.entries,key))
            delete!(manager.stamps,key)
        end
        empty!(manager.pending);empty!(manager.order)
        p.timeline.refresh[]+=1
    end
    # Makie's compute graph pulls the refreshed strip at draw time.
    M.save(joinpath(out,"thumbnails_waiting.png"),M.colorbuffer(p.screen))
    lock(manager.lock) do
        for key in manager.order
            request=manager.pending[key]
            request.kind in (:scene,:image) || continue
            push!(expected,request.position)
            build=request.build
            manager.pending[key]=VE.PreviewRequest(request.kind,()->begin
                result=build()
                result isa VE.RGBFrame && lock(()->push!(arrivals,request.position),manager.lock)
                result
            end,request.position)
        end
    end
    @test !isempty(expected)
    isempty(expected) && error("No visible thumbnail requests after redraw")
    manager.dirty[]=false
    manager.quiet_until[]=0
    recordactions(p,out,"thumbnails_left_to_right",[
        FI.WaitUntil(()->lock(()->length(arrivals)==length(expected),manager.lock);timeout=300),
        FI.Wait(.3)])
    @test arrivals==sort(expected)
    @test lock(()->isempty(manager.pending),manager.lock)
    return (positions=arrivals,tiles=length(arrivals))
end
end
