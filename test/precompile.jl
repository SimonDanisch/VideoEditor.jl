"""
Does the frozen cache in `src/precompile.jl` actually hold?

Same shape as the model runners' tests, and for the same reason: compile time is
per-process, so the claim can only be checked in a fresh one — which is why this
is a standalone script (`julia --project test/precompile.jl`) rather than a
testset inside `runtests.jl`, which would pay for a second Julia and a GPU on
every run of the suite.

What the editor adds is that the frame has to be a *picture* — a render path that
returns black fast is not a pass.
"""

using Test, VideoEditor

const SUBPROCESS = """
using VideoEditor, Mantle, KernelAbstractions
const KA = KernelAbstractions; const VE = VideoEditor
asset = VE.editorassets()
backend = Mantle.defaultbackend()
src = VE.VideoSource(asset); seq = VE.Sequence(src)
engine = VE.FxEngine(backend); readers = Dict{String,Any}()
dest = VE.RGBFrame(undef, src.width, src.height)
Mantle.resetkernelcompiles!(Mantle.todevice(backend))
c0 = Base.cumulative_compile_time_ns()
t = @elapsed begin
    VE.runeditorframe(seq, engine, readers, dest, 1)
    KA.synchronize(backend)
end
c1 = Base.cumulative_compile_time_ns()
s = Mantle.kernelcompiles(Mantle.todevice(backend))
nonblack = count(c -> c != VE.RGB{VE.N0f8}(0,0,0), dest) / length(dest)
println("RESULT ", (; wall = t, compile = (c1[1]-c0[1])/1e9, hits = s.hits,
                     misses = s.misses, nonblack = nonblack))
"""

@testset "VideoEditor: first frame does not compile" begin
    asset = VideoEditor.editorassets()
    if !isfile(asset)
        @info "no asset video; skipping"
    else
        script = tempname() * ".jl"; write(script, SUBPROCESS)
        out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`, String)
        i = findfirst(l -> startswith(l, "RESULT "), split(out, '\n'))
        @test i !== nothing
        r = eval(Meta.parse(split(out, '\n')[i][8:end]))
        @info "editor first frame in a fresh process" r
        @test r.nonblack > 0.5                      # a picture, not a black canvas
        @test r.misses == 0
        @test r.compile < 5.0
        @test r.wall < 20.0
        @test r.version == VideoEditor.KERNELS_VERSION
    end
end
