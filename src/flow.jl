"""
Optical-flow time interpolation, from `RIFERunner`.

The sixth JuliaVision model in the editor, and the one that is not an effect at
all. Every other model decorates a frame; this one *invents* frames that the
source does not have — which is what a slowed clip needs, and why it lives at the
source pass rather than in the effect stack.

The gesture is Premiere's: a clip has a time-interpolation mode, `:sample`
(repeat the nearest source frame) or `:flow` (synthesize the one in between).
`sourcephase` says where between, and it is exactly the fraction `sourceframe`
rounds away — so a clip at `rate = 0.5` alternates 0, ½, 0, ½, and the halves are
the frames that judder without this.
"""

"""
    registerinterpolate!(f)

Install the frame interpolator. `f(out, a, b, t)` writes the frame `t` of the way
from `a` to `b` into `out`; all three are same-size RGB matrices and may live on
the device.

Pluggable like the other models so the editor runs with it absent — a clip set to
`:flow` then falls back to showing the frame it has, which is what `:sample`
would have shown anyway.
"""
const INTERPOLATOR = Ref{Any}(nothing)

registerinterpolate!(f) = (INTERPOLATOR[] = f; nothing)
hasinterpolator() = INTERPOLATOR[] !== nothing

"""
The built-in interpolator: RIFE, from `RIFERunner`. Built on first use, and
REBUILT when the frame size changes.

The export pins the padded frame size, so one model serves one resolution — a
different one throws rather than producing a wrong picture, which is why the size
is part of what is cached rather than a property assumed constant.
"""
const RIFEMODEL = Ref{Any}(nothing)
const RIFESIZE = Ref{Tuple{Int, Int}}((0, 0))

function rifeinterpolate!(out, a, b, t::Real)
    sz = size(a)
    if RIFEMODEL[] === nothing || RIFESIZE[] != sz
        RIFEMODEL[] = RIFERunner.rife(; backend = Lava.LavaBackend())
        RIFESIZE[] = sz
    end
    RIFERunner.interpolate!(out, RIFEMODEL[], a, b; t = Float64(t))
    return out
end

"""
    installinterpolate!()

Point [`registerinterpolate!`](@ref) at the built-in model. Called by `Player`;
builds nothing until a clip is actually set to `:flow`.
"""
installinterpolate!() = registerinterpolate!(rifeinterpolate!)

"""
    settimeinterp!(clip, mode) -> Symbol

Set a clip's time interpolation to `:sample` or `:flow`.

Rejects anything else rather than storing it: the value reaches `graphof`, which
picks a different source node from it, and an unknown mode would silently mean
`:sample` forever.
"""
function settimeinterp!(clip::Clip, mode::Symbol)
    mode in (:sample, :flow) ||
        throw(ArgumentError("time interpolation is :sample or :flow, got :$mode"))
    return clip.timeinterp = mode
end

"""
    smoothslowmo!(player) -> nothing

Turn optical-flow interpolation on for the clip under the playhead, or off again.

A toggle rather than two commands because it is one question with two answers,
and because the answer is visible immediately — the preview re-renders through a
different source node, so the judder either goes away or it does not.

Says so when the clip is not slowed at all: at `rate >= 1` no timeline frame
falls between two source frames, so there is nothing to synthesize and the
setting would look broken rather than idle.
"""
function smoothslowmo!(player::Player)
    loc = editclip(player)
    loc === nothing && return setstatus!(player, "smooth: no clip under the playhead")
    clip = loc[1]
    hasinterpolator() || return setstatus!(player, "smooth: no interpolator installed")
    snapshot!(player)
    on = clip.timeinterp !== :flow
    settimeinterp!(clip, on ? :flow : :sample)
    redraw!(player)
    showplayhead!(player)
    setstatus!(player, if !on
            "smooth slow motion off — frames repeat again"
        elseif clip.rate >= 1.0
            "smooth slow motion on — but this clip is not slowed ($(round(clip.rate; digits = 2))×), " *
            "so nothing is between its frames yet"
        else
            "smooth slow motion on — in-between frames are synthesized"
        end)
    return nothing
end
