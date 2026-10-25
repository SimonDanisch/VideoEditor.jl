module FarmBatchTests
using Test, Sockets
import VideoEditor as VE

struct PartialFrames
    png::Vector{UInt8}
    seen::Vector{Int}
    attempts::Base.RefValue{Int}
end
function VE.foreachfarmresult(accept,w::PartialFrames,frames)
    w.attempts[]+=1
    for (i,n) in enumerate(frames)
        w.attempts[]==1 && i==3 && error("connection lost after a durable prefix")
        push!(w.seen,n)
        accept((frame=n,png=w.png,seconds=0.01,layers=VE.FarmLayer[]))
    end
end

@testset "bounded adjacent batches are the default" begin
  mktempdir() do dir
    job=VE.RenderJob(dir,"batches",(4,4),40,30.,string(VERSION),String[],String[],Dict{String,String}())
    io=IOBuffer();VE.PNGFiles.save(io,fill(VE.RGB{VE.N0f8}(0,0,0),4,4));png=take!(io)
    requests=Vector{Int}[]
    render=frames->begin
        push!(requests,copy(frames))
        [(frame=n,png=png,seconds=0.01,layers=VE.FarmLayer[]) for n in frames]
    end
    s=VE.renderfarm!(job,[VE.FarmWorker("GPU process",render)])
    @test requests==[collect(0:15),collect(16:31),collect(32:39)]
    @test s["batchsize"]==16
    @test s["completed"]==40 && isempty(s["errors"])
    @test s["batches"]["GPU process"]["committed"]==8
    @test VE.renderfarm!(job,[VE.FarmWorker("resume",_->error("must reuse"))])["completed"]==40
    @test_throws ArgumentError VE.renderfarm!(job,[VE.FarmWorker("invalid",render)];batchsize=0)
  end
end

@testset "partial streamed batches retain committed prefixes" begin
  mktempdir() do dir
    job=VE.RenderJob(dir,"partial",(4,4),20,30.,string(VERSION),String[],String[],Dict{String,String}())
    io=IOBuffer();VE.PNGFiles.save(io,fill(VE.RGB{VE.N0f8}(0,0,0),4,4));png=take!(io)
    resource=PartialFrames(png,Int[],Ref(0))
    s=VE.renderfarm!(job,[VE.FarmWorker("stream",resource)])
    @test s["completed"]==20 && isempty(s["errors"])
    @test length(s["incidents"])==1
    @test sort(resource.seen)==collect(0:19) # no retrace of the committed prefix
    @test all(n->VE.farmcomplete(dir,n,job.key),0:19)
  end
end

@testset "another slot can recover a failed lease without stale active errors" begin
  mktempdir() do dir
    job=VE.RenderJob(dir,"recovery",(4,4),6,30.,string(VERSION),String[],String[],Dict{String,String}())
    io=IOBuffer();VE.PNGFiles.save(io,fill(VE.RGB{VE.N0f8}(0,0,0),4,4));png=take!(io)
    steady=VE.FarmWorker("steady",frames->[(frame=n,png=png,seconds=0.,layers=VE.FarmLayer[]) for n in frames])
    failed=VE.FarmWorker("failed",_->error("injected failed lease"))
    s=VE.renderfarm!(job,[failed,steady])
    @test s["completed"]==6 && isempty(s["errors"])
    @test !isempty(s["incidents"])
  end
end

@testset "pause drains a lease; resume keeps it" begin
  mktempdir() do dir
    job=VE.RenderJob(dir,"pausebatch",(4,4),20,30.,string(VERSION),String[],String[],Dict{String,String}())
    io=IOBuffer();VE.PNGFiles.save(io,fill(VE.RGB{VE.N0f8}(0,0,0),4,4));png=take!(io)
    render=frames->begin
        VE.pausefarm!(dir)
        [(frame=n,png=png,seconds=0.01,layers=VE.FarmLayer[]) for n in frames]
    end
    s=VE.renderfarm!(job,[VE.FarmWorker("pause",render)])
    @test s["paused"] && s["completed"]==16
    rm(joinpath(dir,"pause.requested"))
    resource=PartialFrames(png,Int[],Ref(1))
    @test VE.renderfarm!(job,[VE.FarmWorker("resume",resource)])["completed"]==20
    @test resource.seen==collect(16:19)
  end
end

@testset "native farm slot commits before the entire batch arrives" begin
    listener=listen(ip"127.0.0.1",0)
    delivered=Channel{Int}(1);allowsecond=Channel{Nothing}(1)
    client=@async begin
        sock=connect(ip"127.0.0.1",getsockname(listener)[2])
        try
            h,_=VE.receivemessage(sock)
            @test h["frames"]==[0,1]
            for n in 0:1
                n==1 && take!(allowsecond)
                VE.sendmessage(sock,VE.farmmessage((frame=n,png=UInt8[],seconds=0.,layers=VE.FarmLayer[]))...)
            end
        finally
            close(sock)
        end
    end
    socket=accept(listener)
    slot=VE.FarmSlot("stream",socket)
    receiver=@async VE.foreachfarmresult(slot,[0,1]) do result
        result.frame==0 && put!(delivered,0)
    end
    @test timedwait(()->isready(delivered),5)==:ok
    @test take!(delivered)==0
    @test !istaskdone(receiver)
    put!(allowsecond,nothing);wait(receiver);wait(client)
    close(socket);close(listener)
end
end
