# Stdlib-only checks; run in the application's existing root environment.
using Test, Sockets
function test_farmstartup(FC)
    @testset "renderer honors the configured CPU thread budget" begin
        root=dirname(Base.active_project())
        gpu=(index=1,name="fixture")
        withenv("JULIA_NUM_THREADS"=>"2,1") do
            args=collect(FC.renderercommand(Base.julia_cmd(),root,"fixture",gpu,String[],String[]))
            @test "--threads=2,1" in args
            @test "--project=$root" in args
            code=findfirst(==("-e"),args)+1
            stub="module FarmThreadStub; farmchild(args...;device,output)=print(output,Threads.nthreads(:default),\",\",Threads.nthreads(:interactive),\"|\",Base.active_project()); end"
            args[code]=replace(args[code],"using VideoEditor"=>stub,
                "VideoEditor.farmchild"=>"FarmThreadStub.farmchild")
            @test read(Cmd(args),String)=="2,1|$(Base.active_project())"
        end
        @test "--threads=auto" in collect(FC.renderercommand(Base.julia_cmd(),root,
            "fixture",gpu,String[],String[];threads="auto"))
    end
    @testset "farm startup wire is protected" begin
        bad=IOBuffer();write(bad,hton(UInt32(2_000_000)));seekstart(bad)
        @test_throws ErrorException FC.receivemessage(bad)
        code=FC.renderercode(1,String[],String[])
        stub="module FarmStartupStub; farmchild(args...;device,output)=print(output,\"READY\"); end; println(\"IMPORT NOISE\")"
        code=replace(code,"using VideoEditor"=>stub,"VideoEditor.farmchild"=>"FarmStartupStub.farmchild")
        out=IOBuffer();err=IOBuffer()
        run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $code fixture`;
                     stdout=out,stderr=err))
        @test String(take!(out))=="READY"
        @test occursin("IMPORT NOISE",String(take!(err)))
        client=FC.Client("all","127.0.0.1:7600";project=dirname(Base.active_project()))
        gate=FC.namedlock(client,"running:startup-test")
        lock(gate)
        try
            reports=Ref(0)
            task=@async FC.openjob(client,_ -> (reports[]+=1),0,Dict("job"=>"startup-test"),UInt8[])
            @test fetch(task)===nothing
            @test reports[]==0
        finally
            unlock(gate)
        end
    end
end
