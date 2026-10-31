using Test
import VideoEditor as VE

@testset "renderer handoff wakes its caller and propagates errors" begin
    if Threads.nthreads() > 1
        worker = Threads.nthreads() - 1
        @test VE.onthread(() -> Threads.threadid(), worker) == worker + 1
        @test_throws ErrorException VE.onthread(() -> error("handoff failure"), worker)
        @test VE.onthread(worker) do
            VE.onmainthread(() -> :nested)
        end === :nested
        @test VE.onthread(worker; startwithin = .05) do
            sleep(.1)
            :completed
        end === :completed

        # Occupy the owner without yielding. A request that expires before it
        # starts must report failure and must never apply a late edit.
        entered = Threads.Atomic{Bool}(false)
        release = Threads.Atomic{Bool}(false)
        ran = Threads.Atomic{Bool}(false)
        blocker = Task() do
            entered[] = true
            # Keep the owner occupied without yielding, but allow another
            # thread to collect while compiling a first-use timeout request.
            while !release[]
                GC.safepoint()
            end
        end
        blocker.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), blocker, worker)
        schedule(blocker)
        try
            @test timedwait(() -> entered[], 5) === :ok
            @test_throws VE.RenderThreadTimeout VE.onthread(() -> (ran[] = true), worker; startwithin = .02)
        finally
            release[] = true
            wait(blocker)
        end
        @test VE.onthread(() -> :available, worker) === :available
        @test !ran[]
    else
        @test VE.onthread(() -> :direct, 0) === :direct
    end
end
