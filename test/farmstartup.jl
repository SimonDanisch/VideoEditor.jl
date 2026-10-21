# Stdlib-only checks; run in the application's existing root environment.
using Test, Sockets
function test_farmstartup(FC)
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
