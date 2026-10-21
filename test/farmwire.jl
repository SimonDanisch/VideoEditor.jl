module FarmWireTests
using Test, TOML, Sockets
include(joinpath(@__DIR__, "..", "FarmClient", "src", "wire.jl"))

@testset "farm wire: bounded archive streams, unchanged protocol" begin
    bytes=repeat(UInt8[0,1,255,42],800_003)
    header=Dict("command"=>"open","job"=>"test")
    vector_wire=IOBuffer()
    sendmessage(vector_wire,header,bytes)
    stream_wire=IOBuffer()
    sendmessage(stream_wire,header,IOBuffer(bytes),length(bytes))
    @test take!(vector_wire)==take!(copy(stream_wire))
    seekstart(stream_wire)
    output=IOBuffer()
    received,payload=receivemessage(stream_wire;sink=output)
    @test received==header
    @test payload===output
    @test take!(output)==bytes
    @test eof(stream_wire)
    empty_wire=IOBuffer()
    sendmessage(empty_wire,header,IOBuffer(),0)
    seekstart(empty_wire)
    @test receivemessage(empty_wire)==(header,UInt8[])
    @test_throws ErrorException sendmessage(IOBuffer(),header,IOBuffer(UInt8[1]),2)
    @test_throws ErrorException copypayload!(IOBuffer(),IOBuffer(UInt8[1]),2)
    mktempdir() do dir
        input=joinpath(dir,"archive.bin")
        open(io->write(io,bytes),input,"w")
        wire=IOBuffer()
        open(io->sendmessage(wire,header,io,filesize(input)),input)
        seekstart(wire)
        target=joinpath(dir,"received.bin")
        open(target,"w") do sink
            @test first(receivemessage(wire;sink))==header
        end
        @test read(target)==bytes
    end
    listener=listen(ip"127.0.0.1",0)
    port=Int(getsockname(listener)[2])
    sender=@async begin
        socket=connect(ip"127.0.0.1",port)
        try
            sendmessage(socket,header,IOBuffer(bytes),length(bytes))
        finally
            close(socket)
        end
    end
    receiver=accept(listener)
    try
        sink=IOBuffer()
        @test first(receivemessage(receiver;sink))==header
        @test take!(sink)==bytes
        wait(sender)
    finally
        close(receiver)
        close(listener)
    end
end
end
