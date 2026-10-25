using Test, Makie
import VideoEditor as VE
using VideoEditor.Performance

@testset "Performed recipes bind construction values to native property tracks" begin
    P = VE.Performance
    function builder(canvas)
        fig = Figure(; size = canvas)
        ax = Axis(fig[1, 1])
        a = (; fade = Observable(0f0), phase = Observable(0f0), size = Observable(0f0),
              caption = Observable(0f0))
        # A colour fade becomes the line's own alpha; a nonlinear parameter a named input.
        lines!(ax, 1:10, lift(φ -> sin.((1:10) .+ φ), a.phase); color = lift(x -> (:orange, x), a.fade))
        dots = scatter!(ax, [Point2f(0)]; markersize = lift(s -> 18s, a.size))
        connect!(a.size, lift(m -> Float32(m[1] / 18), dots.markersize))
        propertypath!(a.size, dots, "markersize[1]"; scale = 18, also = ["markersize[2]"])
        ov = labelscene!(Scene(fig.scene; camera = campixel!, clear = false))
        filmtext!(ov, 10, 10; text = "caption", color = lift(x -> (:white, x), a.caption))
        @test labels_only() == (P.CONTEXT[].layer === :labels)
        return fig, a
    end
    combined = buildperformed(builder, (320, 180))
    @test endswith(combined.paths["fade"], ".alpha")
    @test endswith(combined.paths["phase"], ".phase")
    @test endswith(combined.paths["size"], ".markersize[1]")
    @test combined.transforms[combined.paths["size"]] == (18.0, 0.0)
    @test only(combined.aliases[combined.paths["size"]]) == replace(combined.paths["size"], "[1]" => "[2]")
    performed = (; picture = ["fade" => ramp(1, 0.5), "size" => [(0f0, 1f0, :linear)]],
                 labels = ["caption" => inout(0.5, 2)])
    tracks = performedtracks(performed, combined, 30)
    byname = Dict(first(t) => t for t in tracks)
    @test byname[combined.paths["fade"]][2] == [(30, 0.0, :linear), (45, 1.0, :linear)]
    @test byname[combined.paths["size"]][2] == [(0, 18.0, :linear)]
    @test haskey(byname, replace(combined.paths["size"], "[1]" => "[2]"))
    @test haskey(byname, combined.paths["caption"])
    # The picture layer has no text; its tracks leave the labels' keys out.
    picture = buildperformed(builder, (320, 180); layer = :picture)
    @test !haskey(picture.paths, "caption")
    @test all(t -> first(t) != combined.paths["caption"], performedtracks(performed, picture, 30; layer = :picture))
    labels = buildperformed(builder, (320, 180); layer = :labels, labels = Dict("caption" => "replaced"))
    @test haskey(labels.paths, "caption") && !haskey(labels.paths, "phase")
    texts(sc) = [reduce(vcat, (vcat(Makie.to_value(p.text)) for p in sc.plots if p isa Makie.Text); init = Any[]);
                 reduce(vcat, texts.(sc.children); init = Any[])]
    @test "replaced" in texts(labels.scene) && !("caption" in texts(labels.scene))
    @test performancevalue([(0f0, 0f0, :linear), (1f0, 1f0, :hold), (2f0, 3f0, :linear)], 1.5f0) == 1f0
    @test_throws ArgumentError buildperformed(builder, (320, 180); layer = :overlay)
    foreach(b -> Makie.free(b.scene), (combined, picture, labels))
end
