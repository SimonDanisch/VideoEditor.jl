"""
Overlays: Makie plots drawn ON TOP of the finished canvas — titles, lower
thirds, data plots, annotations.

The mechanism is deliberately not "rasterize a plot, then alpha-blend it in":
the finished video frame goes INTO a Makie scene as an `image!`, the plots are
drawn over it, and what comes back out of that scene IS the frame. Makie does
the compositing, so there is no alpha channel to invent, no second rasterizer
to keep in sync with the first, and no per-pixel blend kernel.

That only works because the round trip is lossless. Measured on this hardware,
a 1920×1080 frame pushed through a canvas-sized scene with no overlays comes
back BIT-IDENTICAL, and an overlay changes only the pixels it actually covers
(the tests assert both). So the stage can sit in the render path unconditionally
without touching a single pixel of anyone's footage.

Two conventions make an overlay mean the same thing everywhere:

  - **Canvas pixels, y UP, origin bottom-left.** The frame is drawn with a
    reversed y interval (`image!(scene, (0, W), (H, 0), frame)`) and read back
    with a vertical flip. A video frame's first row is its TOP, and Makie's y
    axis points up; without the reversal the picture round-trips fine but every
    glyph comes out mirrored.
  - **Sizes are fractions of the canvas.** A title authored at `size = 0.08` is
    8 % of the frame height whether the sequence is 720p or 4K, so an overlay
    survives a change of output format.
"""

# ---------------------------------------------------------------- registry

"""
A registered overlay kind: a name, display label, keyframable scalar `params`,
and `draw(scene, canvas, state)` which builds its plots ONCE.

`state` is an `Observable{NamedTuple}` carrying the overlay's parameters
(sampled at the current frame) merged with its non-numeric settings — text,
colours, data. `draw` must `lift` everything it needs from it and must not
rebuild plots per frame: the compositor renders thousands of frames through the
same scene, and a plot rebuilt each time would dominate the cost.
"""
struct OverlayKind
    name::Symbol
    label::String
    params::Vector{FxParam}
    draw::Any        # (scene, canvas::NTuple{2,Int}, state::Observable{<:NamedTuple})
end

const OVERLAYKINDS = OverlayKind[]
const OVERLAYBYNAME = Dict{Symbol, OverlayKind}()
const OVERLAYSVERSION = Observables.Observable(0)   # bumped on every (re)registration

"The declared defaults of a parameter list, as a `NamedTuple`."
defaults(params::Vector{FxParam}) =
    NamedTuple{Tuple(p.name for p in params)}(Tuple(p.default for p in params))
defaults(k::OverlayKind) = defaults(k.params)

"""
    registeroverlay!(name, label, params, draw) -> OverlayKind

Register an overlay kind so it can be placed on a sequence, keyframed and saved.
`params::Vector{FxParam}` are its animatable scalars; `draw(scene, canvas, state)`
builds the plots (see [`OverlayKind`](@ref)). Live — call it from any package or
from a running session, no restart.
"""
function registeroverlay!(name::Symbol, label::AbstractString,
                          params::Vector{FxParam}, draw)
    k = OverlayKind(name, String(label), params, draw)
    OVERLAYBYNAME[name] = k
    i = findfirst(q -> q.name == name, OVERLAYKINDS)
    i === nothing ? push!(OVERLAYKINDS, k) : (OVERLAYKINDS[i] = k)
    OVERLAYSVERSION[] = OVERLAYSVERSION[] + 1
    return k
end

"The registered [`OverlayKind`](@ref) named `name` (throws if unknown)."
overlaykind(name::Symbol) = OVERLAYBYNAME[name]

# ---------------------------------------------------------------- the instance

"""
One overlay placed on a sequence: which kind, its parameter values, and the
span of TIMELINE frames it is drawn on.

Keyframes are keyed by timeline frame here, unlike a clip's (which key by
absolute SOURCE frame so a curve survives a split). An overlay is not attached
to any clip — it sits over the finished canvas — so the timeline is the only
frame of reference it has.

`params` are the registry-declared scalars, the ones a curve can animate.
`settings` is everything else — the text of a title, a colour, a data series —
which the editor edits but the keyframe engine has no opinion about.
"""
mutable struct Overlay
    id::UInt64
    kind::Symbol
    params::NamedTuple
    settings::NamedTuple
    start::Int                  # first timeline frame it is drawn on
    stop::Int                   # one PAST the last (half-open, like a clip)
    const animations::Dict{Symbol, AnimCurve}
end

"""
    Overlay(kind; start, stop, settings..., params...)

An overlay of registered `kind`, spanning timeline frames `start:stop-1`.
Keyword arguments matching the kind's declared parameters override their
defaults; everything else is carried as a setting.
"""
function Overlay(kind::Symbol; start::Integer = 0, stop::Integer = typemax(Int), kw...)
    k = overlaykind(kind)
    names = Tuple(p.name for p in k.params)
    given = values(kw)
    params = merge(defaults(k), NamedTuple{filter(in(names), keys(given))}(given))
    settings = NamedTuple{filter(!in(names), keys(given))}(given)
    return Overlay(freshid(), kind, params, settings, Int(start), Int(stop),
                   Dict{Symbol, AnimCurve}())
end

"Whether `ov` is drawn on timeline frame `n`."
showsat(ov::Overlay, n::Integer) = ov.start <= n < ov.stop

"""
    overlaystate(ov, n; framerate) -> NamedTuple

What `ov`'s `draw` sees at timeline frame `n`: its settings merged with its
parameters, each animated parameter replaced by its curve's value. The
`effectiveclip` of the overlay world — and the only place a keyframe becomes a
number.

Two keys are RESERVED and always present: `frame` (the timeline frame) and
`framerate`. They are what a timecode is made of, and what lets a plot track the
playhead without anyone keyframing a cursor — the overlay reads where the video
is instead of being told twice.
"""
function overlaystate(ov::Overlay, n::Integer; framerate::Real = 0.0)
    p = ov.params
    for (key, curve) in ov.animations
        haskey(p, key) || continue
        v = valueat(curve, n)
        v === nothing || (p = merge(p, NamedTuple{(key,)}((Float64(v),))))
    end
    return merge(ov.settings, p, (frame = Int(n), framerate = Float64(framerate)))
end

"Value of `ov`'s parameter `key` at timeline frame `n` (its curve, or the static value)."
function overlayvalue(ov::Overlay, key::Symbol, n::Integer)
    curve = get(ov.animations, key, nothing)
    if curve !== nothing
        v = valueat(curve, n)
        v === nothing || return Float64(v)
    end
    return Float64(get(ov.params, key, 0.0))
end

"""
    setoverlaykey!(ov, key, n, value, ease = :linear)

Key `ov`'s parameter `key` to `value` at TIMELINE frame `n`. `ease` is the same
per-key interpolation a clip's keyframes use — `:linear`, `:smooth`, `:hold`.
"""
function setoverlaykey!(ov::Overlay, key::Symbol, n::Integer, v::Real,
                        ease::Symbol = :linear)
    curve = get!(() -> AnimCurve(), ov.animations, key)
    setkey!(curve, Int(n), Float64(v), ease)
    return curve
end

"The overlay with `id` on `seq`, or `nothing`."
function overlaybyid(seq, id::Integer)
    i = findfirst(o -> o.id == id, seq.overlays)
    return i === nothing ? nothing : seq.overlays[i]
end

"""
    addoverlay!(seq, kind; start, stop, kw...) -> Overlay

Place an overlay of registered `kind` on `seq`. Keywords set its parameters and
settings (see [`Overlay`](@ref)); the default span is the whole sequence.
"""
function addoverlay!(seq, kind::Symbol; kw...)
    ov = Overlay(kind; kw...)
    push!(seq.overlays, ov)
    return ov
end

"Drop the overlay with `id` from `seq` (returns whether one went)."
function removeoverlay!(seq, id::Integer)
    i = findfirst(o -> o.id == id, seq.overlays)
    i === nothing && return false
    deleteat!(seq.overlays, i)
    return true
end

# ---------------------------------------------------------------- the compose stage

"""
The scene the finished canvas is composed in: one `image!` holding the frame,
one child scene per overlay stacked over it, rendered offscreen — on GLMakie's
headless screen, never the editor window's, so an export composes while the
editor is up.

Built once per canvas size and reused for every frame — the plots are created
when an overlay first appears and only their `state` observable changes after
that, so a render is a texture upload plus a draw, not a scene rebuild.
"""
mutable struct CanvasScene
    canvas::NTuple{2, Int}
    scene::Makie.Scene
    frame::Observables.Observable{RGBFrame}
    host::RGBFrame                     # readback staging (canvas-sized, reused)
    entries::Dict{UInt64, Tuple{Makie.Scene, Observables.Observable{NamedTuple}}}
    # child scenes of overlays that went, emptied and parked for the next one.
    # Makie has no public "remove this child scene", and an overlay deleted and
    # re-added (every undo does that) would otherwise grow the scene tree forever.
    spare::Vector{Makie.Scene}
    # the registry version these plots were built from. Re-registering a kind is
    # advertised as live, but plots built by the OLD `draw` keep running against
    # the old closure — they don't pick the new one up, they FAIL against it.
    version::Int
    # our OWN offscreen screen — see `canvasscreen`. Not optional.
    screen::Any
end

function CanvasScene(canvas::Tuple{Integer, Integer})
    W, H = Int(canvas[1]), Int(canvas[2])
    frame = Observables.Observable(zeros(RGB{N0f8}, W, H))
    scene = Makie.Scene(size = (W, H), camera = Makie.campixel!,
                        backgroundcolor = RGBf(0, 0, 0))
    # reversed y interval: the frame's first row is the video's TOP, Makie's y
    # points up — flipping HERE keeps overlay glyphs upright (see the header)
    Makie.image!(scene, (0, W), (H, 0), frame; interpolate = false, fxaa = false)
    return CanvasScene((W, H), scene, frame, zeros(RGB{N0f8}, W, H),
                       Dict{UInt64, Tuple{Makie.Scene, Observables.Observable{NamedTuple}}}(),
                       Makie.Scene[], OVERLAYSVERSION[], nothing)
end

"""
This canvas's own offscreen GL screen, created on first use.

It must be its own, and `colorbuffer(scene)` will not do: that route goes
through GLMakie's SINGLETON offscreen screen, and every scene-taking `Screen`
constructor EMPTIES that screen and re-displays itself on it. The editor's
window comes from the same constructor (`display(player.fig)`), so composing one
overlaid frame would evict the editor's own figure from its own window —
measured, its renderlist went from 32 plots to 3. Asking for a screen WITHOUT a
scene takes one from the reuse pool instead, which is nobody else's.
"""
function canvasscreen(cs::CanvasScene)
    if cs.screen === nothing || !isopen(cs.screen)
        screen = GLMakie.Screen(; visible = false, start_renderloop = false,
                                px_per_unit = 1.0, scalefactor = 1.0, focus_on_show = false)
        display(screen, cs.scene)
        cs.screen = screen
    end
    return cs.screen
end

"""
    rastercanvas(cs) -> Matrix{RGB{N0f8}}

The canvas scene rendered on its own screen ([`canvasscreen`](@ref)) at exactly
its own size, in GL layout: `(width, height)`, column 1 the scene's BOTTOM.

GLNative rather than the Julia orientation: it comes back as `(width, height)`
like a frame, so putting it back is a column flip instead of a transpose of two
million pixels.
"""
rastercanvas(cs::CanvasScene) = Makie.colorbuffer(canvasscreen(cs), Makie.GLNative)

"""
Copy `gl` (the GL readback, bottom row first) into `dest` the right way up.

Column by column, because each column of both is contiguous memory and a
`copyto!` over a reversed `view` is not: it walks two million elements through
generic strided indexing instead of doing 1080 memcpys (measured on a 1080p
frame: 5.8 ms that way, 0.75 ms this way).

One CPU-side flip is unavoidable — a video frame's first row is its top, the GL
framebuffer's first row is its bottom, and no amount of flipping the image on
upload moves that: whichever way the picture is drawn, "upright in the scene"
puts the video's top at the GL top.
"""
function flipinto!(dest::AbstractMatrix, gl::AbstractMatrix)
    h = Base.size(gl, 2)
    @inbounds for j in 1:h
        copyto!(view(dest, :, j), view(gl, :, h + 1 - j))
    end
    return dest
end

# A host frame takes the flip directly; a device buffer wants contiguous host
# memory to upload from, so it goes through the canvas's staging frame.
stageback!(dest::RGBFrame, host::RGBFrame, gl) = flipinto!(dest, gl)
stageback!(dest, host::RGBFrame, gl) = (flipinto!(host, gl); copyto!(dest, host))

# One canvas scene per output format, for the life of the process. A GL context
# is process state, not sequence state — a `Sequence` is plain serializable data
# and has no business owning one — and it is deliberately never destroyed:
# creating a window after the last one closed walks into a GLFW monitor-lookup
# crash on this setup, and there are only ever a handful of canvas sizes.
const CANVASSCENES = Dict{NTuple{2, Int}, CanvasScene}()
const CANVASLOCK = ReentrantLock()

"The shared [`CanvasScene`](@ref) for a `(width, height)` output format."
canvasscenefor(canvas::Tuple{Integer, Integer}) =
    get!(() -> CanvasScene(canvas), CANVASSCENES, (Int(canvas[1]), Int(canvas[2])))

"""
    syncoverlays!(cs, overlays, n)

Bring `cs`'s scenes in line with `overlays` at timeline frame `n`: build the
plots for one that appeared, drop the scenes of one that went, and hand every
live overlay its sampled [`overlaystate`](@ref). Overlays outside their span are
hidden rather than rebuilt, so a title entering and leaving costs a boolean.
"""
function syncoverlays!(cs::CanvasScene, overlays, n::Integer; framerate::Real = 0.0)
    # a kind was (re)registered since these plots were built: they belong to the
    # previous `draw` closure and will not run against the new one, so drop them
    # all and let the loop below rebuild. This is what makes live authoring —
    # tweak an overlay, re-register, look — actually work.
    if cs.version != OVERLAYSVERSION[]
        recycle!(cs, collect(keys(cs.entries)))
        cs.version = OVERLAYSVERSION[]
    end
    live = Set{UInt64}()
    for ov in overlays
        push!(live, ov.id)
        entry = get(cs.entries, ov.id, nothing)
        if entry === nothing
            child = isempty(cs.spare) ?
                Makie.Scene(cs.scene; camera = Makie.campixel!, clear = false) :
                pop!(cs.spare)
            state = Observables.Observable{NamedTuple}(overlaystate(ov, n; framerate))
            overlaykind(ov.kind).draw(child, cs.canvas, state)
            entry = (child, state)
            cs.entries[ov.id] = entry
        end
        child, state = entry
        visible = showsat(ov, n)
        child.visible[] = visible
        # the state always differs frame to frame — it carries `frame` — so
        # there is nothing to compare against. What keeps a static overlay from
        # re-computing anything is `olift`'s `ignore_equal_values` on each
        # derived attribute, one level down.
        visible && (state[] = overlaystate(ov, n; framerate))
    end
    recycle!(cs, [id for id in keys(cs.entries) if !(id in live)])   # deleted since last frame
    return cs
end

"Empty the child scenes of `ids` and park them for reuse (see `CanvasScene.spare`)."
function recycle!(cs::CanvasScene, ids)
    for id in ids
        child, _ = cs.entries[id]
        foreach(p -> delete!(child, p), copy(child.plots))
        child.visible[] = false
        push!(cs.spare, child)
        delete!(cs.entries, id)
    end
    return cs
end

"""
    drawoverlays!(dest, overlays, n) -> dest

Draw `overlays` onto the finished canvas `dest` at timeline frame `n`, in place —
the last stage of a rendered frame, wherever that frame is going.

Takes no scene: it looks one up by `dest`'s size, so every caller composes on the
same context and no call site can accidentally be the one that renders WITHOUT
overlays. Returns immediately when nothing is drawn at `n` — not as an
optimization but because the pass would otherwise be a pure tax: the round trip
is bit-exact (the suite asserts it), so skipping it is provably the same picture,
and a 1080p pass costs ~5.5 ms.

`dest` may live on a device; it is staged through the canvas's host buffers. The
lock is the GL context's: one export job and one preview can ask at once.
"""
function drawoverlays!(dest::AnyRGBFrame, overlays, n::Integer; framerate::Real = 0.0)
    (isempty(overlays) || !any(ov -> showsat(ov, n), overlays)) && return dest
    return lock(CANVASLOCK) do
        cs = canvasscenefor(Base.size(dest))
        copyto!(cs.frame[], dest)                # device→host when dest is a LavaArray
        Observables.notify(cs.frame)             # same array object: notify explicitly
        syncoverlays!(cs, overlays, n; framerate)
        gl = rastercanvas(cs)                    # (W, H), column 1 = the scene's BOTTOM
        stageback!(dest, cs.host, gl)
        return dest
    end
end

# ---------------------------------------------------------------- serialization

overlaydict(ov::Overlay) = Dict{String, Any}(
    "id" => string(ov.id), "kind" => String(ov.kind),
    "start" => ov.start, "stop" => ov.stop,
    "params" => Dict{String, Any}(String(k) => Float64(v) for (k, v) in pairs(ov.params)),
    "settings" => Dict{String, Any}(String(k) => tomlvalue(v) for (k, v) in pairs(ov.settings)),
    "animations" => Dict{String, Any}(
        String(key) => Dict{String, Any}(
            "interp" => String(curve.interp),
            "frames" => [k.frame for k in curve.keys],
            "values" => [Float64(k.value) for k in curve.keys],
            "eases" => [String(k.ease) for k in curve.keys])
        for (key, curve) in ov.animations if !isempty(curve)))

# TOML holds strings, numbers and arrays of them — a colour or a symbol goes as
# its string form and comes back as one, which every `draw` already accepts
# (Makie's `to_color` takes "white" as happily as `:white`).
tomlvalue(v::Union{Real, AbstractString}) = v isa Real ? Float64(v) : String(v)
tomlvalue(v::Symbol) = String(v)
tomlvalue(v::AbstractVector) = [tomlvalue(x) for x in v]
tomlvalue(v) = string(v)

function overlayfromdict(d::AbstractDict)
    kind = Symbol(d["kind"])
    params = NamedTuple(Symbol(k) => Float64(v) for (k, v) in get(d, "params", Dict()))
    settings = NamedTuple(Symbol(k) => v for (k, v) in get(d, "settings", Dict()))
    ov = Overlay(freshid(), kind, params, settings, Int(d["start"]), Int(d["stop"]),
                 Dict{Symbol, AnimCurve}())
    haskey(d, "id") && (ov.id = parse(UInt64, d["id"]))
    for (key, ad) in get(d, "animations", Dict{String, Any}())
        eases = get(ad, "eases", fill("linear", length(ad["frames"])))
        ov.animations[Symbol(key)] = AnimCurve(
            [Keyframe(Int(f), Float64(v), Symbol(e))
             for (f, v, e) in zip(ad["frames"], ad["values"], eases)],
            Symbol(get(ad, "interp", "linear")))
    end
    return ov
end

# ---------------------------------------------------------------- stock overlays

"Colour from a setting, faded by an opacity parameter (settings carry strings)."
fadedcolor(c, α::Real) = (col = Makie.to_color(c);
                          Makie.RGBAf(col.r, col.g, col.b, col.alpha * Float32(α)))

"""
One derived plot attribute of an overlay, from its `state`.

`ignore_equal_values` is the point: the state is one bundle, so animating a
single parameter re-fires EVERY attribute that reads it, and a text plot whose
`text` was "recomputed" to the same string still re-shapes its glyphs. This
stops the recomputation at the value that did not change. Overlay kinds should
build their attributes with it rather than a bare `lift`.
"""
olift(f, state) = Makie.lift(f, state; ignore_equal_values = true)

registeroverlay!(:text, "Text",
    [FxParam(:x, "X"; min = -0.5, max = 1.5, default = 0.5),
     FxParam(:y, "Y"; min = -0.5, max = 1.5, default = 0.85),
     FxParam(:size, "Size"; min = 0.01, max = 0.5, default = 0.08),
     FxParam(:rotation, "Rotation"; min = -180.0, max = 180.0, default = 0.0),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    function (scene, canvas, state)
        W, H = canvas
        Makie.text!(scene, olift(s -> Makie.Point2f(s.x * W, s.y * H), state);
                    text = olift(s -> String(get(s, :text, "")), state),
                    fontsize = olift(s -> Float32(s.size * H), state),
                    color = olift(s -> fadedcolor(get(s, :color, :white), s.opacity), state),
                    rotation = olift(s -> Float32(deg2rad(s.rotation)), state),
                    align = (:center, :center))
        return nothing
    end)

registeroverlay!(:bar, "Bar",
    [FxParam(:x, "X"; min = -0.5, max = 1.5, default = 0.0),
     FxParam(:y, "Y"; min = -0.5, max = 1.5, default = 0.0),
     FxParam(:width, "Width"; min = 0.0, max = 2.0, default = 1.0),
     FxParam(:height, "Height"; min = 0.0, max = 1.0, default = 0.18),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 0.6)],
    function (scene, canvas, state)
        W, H = canvas
        Makie.poly!(scene,
                    olift(s -> Makie.Rect2f(s.x * W, s.y * H, s.width * W, s.height * H), state);
                    color = olift(s -> fadedcolor(get(s, :color, :black), s.opacity), state))
        return nothing
    end)

"Frame `n` at `fps` as `HH:MM:SS:FF` (SMPTE-style, non-drop)."
function timecodestring(n::Integer, fps::Real)
    f = fps > 0 ? fps : 25.0
    total = max(Int(n), 0)
    frames = round(Int, f)
    s, ff = divrem(total, frames)
    m, ss = divrem(s, 60)
    h, mm = divrem(m, 60)
    return string(lpad(h, 2, '0'), ':', lpad(mm, 2, '0'), ':',
                  lpad(ss, 2, '0'), ':', lpad(ff, 2, '0'))
end

registeroverlay!(:timecode, "Timecode",
    [FxParam(:x, "X"; min = -0.5, max = 1.5, default = 0.5),
     FxParam(:y, "Y"; min = -0.5, max = 1.5, default = 0.05),
     FxParam(:size, "Size"; min = 0.01, max = 0.3, default = 0.04),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    function (scene, canvas, state)
        W, H = canvas
        # reads `frame`/`framerate` off the state — nothing to keyframe, it
        # simply knows where the video is
        Makie.text!(scene, olift(s -> Makie.Point2f(s.x * W, s.y * H), state);
                    text = olift(s -> timecodestring(s.frame, s.framerate), state),
                    fontsize = olift(s -> Float32(s.size * H), state),
                    font = :bold,
                    color = olift(s -> fadedcolor(get(s, :color, :white), s.opacity), state),
                    align = (:center, :center))
        return nothing
    end)

"""
`values` resampled to `n` points by linear interpolation — the correspondence
rule that lets two series of different length morph into one another.
"""
function resampleseries(v::AbstractVector{<:Real}, n::Integer)
    length(v) == n && return Float64.(v)
    length(v) == 1 && return fill(Float64(v[1]), n)
    n == 1 && return [Float64(v[1])]
    return [(t = (i - 1) * (length(v) - 1) / (n - 1) + 1;
             lo = floor(Int, t); hi = min(lo + 1, length(v));
             Float64(v[lo]) + (t - lo) * Float64(v[hi] - v[lo])) for i in 1:n]
end

"""
The two series interpolated at `morph`, both resampled to the longer length
first. `morph = 0` is `values`, `1` is `values2`; a keyframed `morph` is a plot
growing into another plot, which is all "morphing" needs to be as long as the
two shapes are sampled onto a common parameterization.
"""
function morphseries(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, morph::Real)
    isempty(b) && return Float64.(a)
    isempty(a) && return Float64.(b)
    n = max(length(a), length(b))
    ra, rb = resampleseries(a, n), resampleseries(b, n)
    t = clamp(Float64(morph), 0.0, 1.0)
    return (1 - t) .* ra .+ t .* rb
end

registeroverlay!(:curve, "Curve",
    [FxParam(:x, "X"; min = -0.5, max = 1.5, default = 0.08),
     FxParam(:y, "Y"; min = -0.5, max = 1.5, default = 0.12),
     FxParam(:width, "Width"; min = 0.0, max = 2.0, default = 0.84),
     FxParam(:height, "Height"; min = 0.0, max = 1.0, default = 0.3),
     FxParam(:progress, "Draw", min = 0.0, max = 1.0, default = 1.0),
     FxParam(:morph, "Morph"; min = 0.0, max = 1.0, default = 0.0),
     FxParam(:linewidth, "Line width"; min = 0.5, max = 20.0, default = 4.0),
     FxParam(:opacity, "Opacity"; min = 0.0, max = 1.0, default = 1.0)],
    function (scene, canvas, state)
        W, H = canvas
        pts = olift(state) do s
            vals = morphseries(get(s, :values, Float64[]), get(s, :values2, Float64[]), s.morph)
            isempty(vals) && return Makie.Point2f[]
            lo, hi = extrema(vals)
            span = hi - lo
            # `progress` reveals the curve left to right — a keyframed draw-on.
            # Below two points there is no line to draw, and rounding UP to two
            # left a stub visible at progress = 0 (caught on a demo frame, not
            # by a test: the tests only ever looked at a finished curve).
            shown = round(Int, clamp(s.progress, 0, 1) * length(vals))
            shown < 2 && return Makie.Point2f[]
            return [Makie.Point2f(
                        (s.x + s.width * (length(vals) == 1 ? 0.0 : (i - 1) / (length(vals) - 1))) * W,
                        (s.y + s.height * (span == 0 ? 0.5 : (vals[i] - lo) / span)) * H)
                    for i in 1:min(shown, length(vals))]
        end
        Makie.lines!(scene, pts;
                     linewidth = olift(s -> Float32(s.linewidth), state),
                     color = olift(s -> fadedcolor(get(s, :color, :orangered), s.opacity), state))
        return nothing
    end)
