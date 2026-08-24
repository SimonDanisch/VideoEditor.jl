# A Makie scene as data (src/scenespec.jl): paths, animation, round trip.
#
# The three properties that make the whole approach work, and each is easy to
# lose: a path addresses ONE attribute of ONE plot, animating writes a COPY (so
# it cannot accumulate across frames), and what goes into the project file comes
# back out the same.

@testset "scenespec: a path addresses one attribute" begin
    spec = VE.SceneSpec([Makie.PlotSpec(:Scatter; markersize = 4.0, color = :red),
                         Makie.PlotSpec(:Lines; linewidth = 1.5)])

    @test VE.scenepath("plots[1].markersize") == (1, :markersize)
    @test VE.scenepath("plots[2].linewidth") == (2, :linewidth)
    # Anything this cannot address is `nothing`, NOT an error: a project written
    # by a newer editor may name plots this one has not got, and a curve
    # pointing nowhere must be ignored rather than fail the open.
    @test VE.scenepath("nonsense") === nothing
    @test VE.scenepath("plots[1]") === nothing
    @test VE.scenepathvalue(spec, "plots[9].markersize") === nothing
    @test VE.scenepathvalue(spec, "plots[1].nosuchattr") === nothing

    @test VE.scenepathvalue(spec, "plots[1].markersize") == 4.0
    # NOT `:red`: `PlotSpec` normalises its keywords as it is built, so a colour
    # is already an `RGBA` here. That is deliberate on Makie's side (two spellings,
    # one spec) and it is why `specvalue` needs a colour method — see there.
    @test VE.scenepathvalue(spec, "plots[1].color") == Makie.RGBAf(1, 0, 0, 1)
    @test VE.setscenepath!(spec, "plots[1].markersize", 12.0)
    @test VE.scenepathvalue(spec, "plots[1].markersize") == 12.0
    # false, not a throw — the caller can tell "did not apply" from "applied false"
    @test !VE.setscenepath!(spec, "plots[9].markersize", 1.0)
end

@testset "scenespec: animating writes a copy" begin
    spec = VE.SceneSpec([Makie.PlotSpec(:Scatter; markersize = 4.0)])
    a = VE.animatedspec(spec, (; Symbol("plots[1].markersize") => 20.0))
    @test VE.scenepathvalue(a, "plots[1].markersize") == 20.0
    # …and the ORIGINAL is untouched. Mutating in place would make the animation
    # cumulative: a curve that ended at frame 10 would still be showing its last
    # value at frame 11 instead of the spec's own.
    @test VE.scenepathvalue(spec, "plots[1].markersize") == 4.0

    # a key that addresses nothing is skipped, not fatal — `overlaystate` merges
    # reserved keys like `frame`/`framerate` into the same namedtuple
    b = VE.animatedspec(spec, (; :frame => 7, Symbol("plots[1].markersize") => 9.0))
    @test VE.scenepathvalue(b, "plots[1].markersize") == 9.0
end

@testset "scenespec: survives the project file" begin
    spec = VE.SceneSpec([Makie.PlotSpec(:Mesh; color = :orangered, transparency = true),
                         Makie.PlotSpec(:Scatter; markersize = 3.5,
                                        positions = [1.0, 2.0, 3.0])];
                        backend = :RayMakie,
                        theme = Dict{Symbol, Any}(:RayMakie => Dict{Symbol, Any}(
                            :samples => 32, :exposure => 0.35, :integrator => :VolPath)))

    # through JSON, because that is what a project file is (see project.jl)
    back = VE.scenefromdict(VE.JSON.parse(VE.JSON.json(VE.scenedict(spec))))

    @test back.backend === :RayMakie
    @test length(back.plots) == 2
    @test back.plots[1].type === :Mesh
    # the colour survives EXACTLY, as `#rrggbbaa` in the file — `:orangered` was
    # already an RGBA before it ever got there
    @test back.plots[1].kwargs[:color] == Makie.to_color(:orangered)
    @test VE.scenedict(spec)["plots"][1]["kwargs"]["color"] isa AbstractString
    @test back.plots[1].kwargs[:transparency] === true
    @test back.plots[2].kwargs[:markersize] == 3.5
    @test back.plots[2].kwargs[:positions] == [1.0, 2.0, 3.0]
    # …and the backend's settings ride in the THEME, not in a second bag of
    # options beside it
    @test back.theme[:RayMakie][:samples] == 32
    @test back.theme[:RayMakie][:integrator] === :VolPath
    @test back.theme[:RayMakie][:exposure] ≈ 0.35

    # a plain string must NOT come back as a Symbol
    s2 = VE.SceneSpec([Makie.PlotSpec(:Text; text = "HELLO")])
    b2 = VE.scenefromdict(VE.JSON.parse(VE.JSON.json(VE.scenedict(s2))))
    @test b2.plots[1].kwargs[:text] == "HELLO"
    @test b2.plots[1].kwargs[:text] isa AbstractString
end
