"""
Performed scene recipes for narrated films.

A builder constructs its figure once around construction Observables (a fade, a
phase, a focus) and returns them by name. `buildperformed` connects each one to
the native plot property that consumes it, so the saved project keys ordinary
Makie properties. A film authors a beat's animation as `(seconds, value, ease)`
keys from its narration `Timing` (`ramp`, `inout`, `dial`, …), and
`performedtracks` turns them into the property tracks `keyframes!` stores.
A recipe may split into a picture layer and an independent labels layer.
"""
module Performance

using Makie, GeometryBasics
using Makie: Colors
using Base.ScopedValues: ScopedValue, with

export Timing, ease, ramp, inout, dial, eased, sampledkeys, performancevalue,
       labels_only, filmtext, filmtext!, labelscene!, plotattribute!, propertypath!,
       buildperformed, performedtracks

"""When each narration line starts and ends, in seconds into the beat, and how long the beat runs."""
struct Timing
    starts::Vector{Float32}
    ends::Vector{Float32}
    duration::Float32
end

# ── performance keys ────────────────────────────────────────────────────────
# A key is `(seconds, value, ease)`: `:linear` ramps, `:smooth` eases, `:hold` steps.

"""Smooth ease-in-out on 0..1."""
ease(t) = (t = clamp(Float32(t), 0f0, 1f0); t * t * (3f0 - 2f0 * t))

"0 to 1 over `d` seconds from `t`, at a constant rate."
ramp(t, d = 0.6f0) = [(Float32(t), 0f0, :linear), (Float32(t + d), 1f0, :linear)]

"In over `d_in` seconds from `t_in`, out over `d_out` from `t_out`."
inout(t_in, t_out; d_in = 0.6f0, d_out = 0.6f0) =
    [(Float32(t_in), 0f0, :linear), (Float32(t_in + d_in), 1f0, :linear),
     (Float32(t_out), 1f0, :linear), (Float32(t_out + d_out), 0f0, :linear)]

"0 to 1 eased in and out over `d` seconds from `t`."
dial(t, d) = [(Float32(t), 0f0, :smooth), (Float32(t + d), 1f0, :smooth)]

"Eased between successive `seconds => value` pairs."
eased(pairs) = [(Float32(t), Float32(v), :smooth) for (t, v) in pairs]

"Sample a nonlinear performance into editable value keys at 60 Hz, including its endpoints."
function sampledkeys(f, start, stop)
    times = sort!(unique!([Float32.(collect(start:1/60:stop)); Float32(stop)]))
    return [(t, Float32(f(t)), :linear) for t in times]
end

"Evaluate authored value keys (only while constructing native property tracks)."
function performancevalue(keys, t)
    t <= first(keys)[1] && return first(keys)[2]
    for (a, b) in zip(keys, Iterators.drop(keys, 1))
        if t <= b[1]
            a[3] === :hold && return t < b[1] ? a[2] : b[2]
            u = (t - a[1]) / (b[1] - a[1]); a[3] === :smooth && (u = ease(u))
            return (1 - u) * a[2] + u * b[2]
        end
    end
    return last(keys)[2]
end

# ── the build context ───────────────────────────────────────────────────────

"A construction value stored on `plot.property` as `scale * value + offset`, and on the `also` properties too."
struct NativeProperty
    plot::Any
    property::String
    scale::Float64
    offset::Float64
    also::Vector{String}
end

"""
One recipe build: its layer, label text replacements, and which construction
Observables are already bound to a plot attribute or a scaled native property.
"""
struct BuildContext
    layer::Symbol
    labels::Dict{String,String}
    attributes::IdDict{Observable,Tuple{Any,Symbol}}
    properties::IdDict{Observable,NativeProperty}
    annotations::IdDict{Scene,Nothing}
end
BuildContext(layer::Symbol, labels) = BuildContext(layer, Dict{String,String}(labels),
    IdDict{Observable,Tuple{Any,Symbol}}(), IdDict{Observable,NativeProperty}(), IdDict{Scene,Nothing}())

"The build in progress; builders run only inside `buildperformed`."
const CONTEXT = ScopedValue{BuildContext}()

"Whether this build draws only the labels layer."
labels_only() = CONTEXT[].layer === :labels

"`text` with this build's label replacements applied (text may be an Observable)."
function filmtext(text)
    replacements = copy(CONTEXT[].labels)
    replaceone(x::AbstractString) = get(replacements, String(x), String(x))
    replaceone(x::AbstractVector) = replaceone.(x)
    replaceone(x) = x
    return text isa Observable ? lift(replaceone, text) : replaceone(text)
end
filmtext!(args...; text, kw...) = Makie.text!(args...; text = filmtext(text), kw...)

"Mark `scene` as annotation: all its plots belong to the labels layer."
labelscene!(scene::Scene) = (CONTEXT[].annotations[scene] = nothing; scene)

"""
    splitlayer!(scene, layer)

Keep only `layer`'s plots: text and annotation scenes for `:labels`, everything
else for `:picture`. Invisible native-property owners named in `keep_owners`
stay in the labels layer so projections and reveals still follow them.
"""
function splitlayer!(scene::Scene, layer::Symbol; annotation = false, keep_owners = name -> false)
    annotation = annotation || haskey(CONTEXT[].annotations, scene)
    layer === :labels && (scene.backgroundcolor[] = RGBAf(0, 0, 0, 0))
    for plot in copy(scene.plots)
        text = plot isa Makie.Text || plot isa Makie.TextLabel
        keep = layer === :labels ? (text || annotation) : (!text && !annotation)
        name = Makie.to_value(get(plot.attributes, :name, nothing))
        if !keep && layer === :labels && name isa Symbol && keep_owners(name)
            Makie.update!(plot; visible = false)
        else
            keep || delete!(scene, plot)
        end
    end
    for child in scene.children
        splitlayer!(child, layer; annotation, keep_owners)
    end
    return scene
end

# ── binding construction values to native properties ────────────────────────

"""
    plotattribute!(plot, key, value::Observable)

Make `value` the plot's native attribute `key` (adding it as an input if the
plot has none of that name): the attribute owns the value and the existing
derivations consume its changes.
"""
function plotattribute!(plot, key::Symbol, value::Observable)
    bound = CONTEXT[].attributes
    haskey(bound, value) && return plot
    if haskey(plot.attributes.inputs, key)
        Makie.update!(plot; NamedTuple{(key,)}((value[],))...)
    else
        # Preserve the declared element type, including Any for file-backed data
        # initially set to nothing; the bridge must not narrow it to Nothing.
        Makie.add_input!(plot.attributes, key, Ref{eltype(value)}(value[]))
    end
    input = Makie.ComputePipeline.get_observable!(plot.attributes, key;
        use_deepcopy = !(value[] === nothing || value[] isa AbstractArray))
    connect!(value, input)
    bound[value] = (plot, key)
    return plot
end

"""
    propertypath!(value::Observable, plot, property; scale = 1, offset = 0, also = String[])

Declare that `value` is the native `plot.property` (e.g. `"rotation[3]"`), as
`property = scale * value + offset`: keys authored for `value` are stored on it,
and on the plot's `also` properties (e.g. both components of a marker size).
"""
propertypath!(value::Observable, plot, property::AbstractString; scale = 1.0, offset = 0.0, also = String[]) =
    (CONTEXT[].properties[value] = NativeProperty(plot, String(property), Float64(scale), Float64(offset),
                                                  String.(also)); value)

function observation_distance(value, source, seen = IdDict{Any,Nothing}())
    value === source && return 0
    value isa Observable || return typemax(Int) ÷ 2
    haskey(seen, value) && return typemax(Int) ÷ 2
    seen[value] = nothing
    best = typemax(Int) ÷ 2
    for input in value.inputs
        best = min(best, 1 + observation_distance(input.observable, source, seen))
    end
    return best
end

# Recognise a simple colour fade so it becomes the primitive's ordinary alpha
# attribute, rather than adding a second control for opacity.
function unaryvalue(value, source, x)
    value === source && return x
    value isa Observable || return value
    isempty(value.inputs) && throw(ArgumentError("another input"))
    callback = first(value.inputs).f
    hasfield(typeof(callback), :args) && hasfield(typeof(callback), :f) || throw(ArgumentError("not a map"))
    args = map(arg -> unaryvalue(arg, source, x), callback.args)
    return Base.invokelatest(callback.f, args...)
end

function nativefade!(plot, key, observed, value)
    key === :color && value[] isa AbstractFloat || return false
    Makie.to_value(get(plot.attributes, :strokewidth, 0)) == 0 || return false
    try
        c0 = Makie.to_color(unaryvalue(observed, value, 0f0))
        c1 = Makie.to_color(unaryvalue(observed, value, 1f0))
        RGBf(c0) == RGBf(c1) && Colors.alpha(c0) == 0 && Colors.alpha(c1) > 0 || return false
        haskey(plot.attributes.inputs, :alpha) || return false
        scale = Float64(Colors.alpha(c1))
        base = RGBAf(RGBf(c1), Float32(Makie.to_value(plot.alpha)))
        # The colour is now fixed; alpha alone is animated. Disconnect just the
        # superseded Observable input, keeping all other geometry connections.
        filter!(plot.attributes.observerfunctions) do observer
            f = observer.f
            if hasfield(typeof(f), :fullkey) && getfield(f, :fullkey) === :color
                off(observer)
                return false
            end
            true
        end
        Makie.update!(plot; color = base, alpha = scale * value[])
        connect!(value, lift(a -> convert(typeof(value[]), a / scale), plot.alpha))
        CONTEXT[].properties[value] = NativeProperty(plot, "alpha", scale, 0.0, String[])
        return true
    catch err
        err isa Union{ArgumentError,MethodError,TypeError} || rethrow()
        return false
    end
end

"All plots of `scene` and its children, each with a unique name (kind_index unless named)."
function namedplots(scene)
    plots = Any[]
    visit(sc) = (append!(plots, sc.plots); foreach(visit, sc.children))
    visit(scene)
    counts = Dict{String,Int}()
    used = Set{Symbol}()
    for plot in plots
        kind = String(Makie.plotkey(typeof(plot)))
        k = get(counts, kind, 0) + 1; counts[kind] = k
        current = Makie.to_value(get(plot.attributes, :name, :automatic))
        if current isa Symbol && current ∉ (:automatic, Symbol(kind)) && current ∉ used
            push!(used, current)
            continue
        end
        haskey(plot.attributes.inputs, :name) ? Makie.update!(plot; name = Symbol(kind, "_", k)) :
            Makie.add_input!(plot.attributes, :name, Symbol(kind, "_", k))
        push!(used, Symbol(kind, "_", k))
    end
    return plots
end

"""
    sceneattributes!(scene, values) -> (; paths, transforms, aliases)

Native property paths (`"plot.attribute"`) for the named construction `values`,
on the plots of `scene` that consume them, each path's `(scale, offset)`, and
the further paths keyed together with it. A value this layer does not draw has
no path.
"""
function sceneattributes!(scene, values)
    ctx = CONTEXT[]
    plots = namedplots(scene)
    paths = Dict{String,String}()
    transforms = Dict{String,Tuple{Float64,Float64}}()
    aliases = Dict{String,Vector{String}}()
    function property!(name, value)
        p = ctx.properties[value]
        p.plot in plots || return
        owner = String(Makie.to_value(p.plot.name))
        path = owner * "." * p.property
        paths[String(name)] = path
        transforms[path] = (p.scale, p.offset)
        isempty(p.also) || (aliases[path] = [owner * "." * a for a in p.also])
    end
    for (name, value) in pairs(values)
        if haskey(ctx.properties, value)
            property!(name, value)
            continue
        end
        haskey(ctx.attributes, value) && first(ctx.attributes[value]) ∉ plots && delete!(ctx.attributes, value)
        if !haskey(ctx.attributes, value)
            candidates = Tuple{Int,Any,Symbol,Any}[]
            for plot in plots, observer in plot.attributes.observerfunctions
                f = observer.f
                field = hasfield(typeof(f), :fullkey) ? :fullkey : hasfield(typeof(f), :k) ? :k : nothing
                field === nothing && continue
                key = getfield(f, field)
                key isa Symbol || continue
                haskey(plot.attributes.inputs, key) || continue
                key in (:model, :transform_func, :transformation) && continue
                distance = observation_distance(observer.observable, value)
                distance < typemax(Int) ÷ 2 && push!(candidates, (distance, plot, key, observer.observable))
            end
            isempty(candidates) && continue # this layer does not draw this property
            sort!(candidates; by = first)
            distance, plot, key, observed = first(candidates)
            if distance > 0 && nativefade!(plot, key, observed, value)
                property!(name, value)
                continue
            end
            # A direct alpha/visibility/data input already is the native property.
            # A nonlinear geometric parameter (focus, phase, …) is a named input on
            # the actual image/line/mesh that derives from it.
            plotattribute!(plot, distance == 0 ? key : Symbol(name), value)
        end
        plot, key = ctx.attributes[value]
        plot in plots || continue
        owner = String(Makie.to_value(plot.name))
        v = value[]
        if v isa Union{GeometryBasics.Vec,GeometryBasics.Point}
            for i in eachindex(v)
                paths["$name[$i]"] = "$owner.$key[$i]"
            end
        else
            paths[String(name)] = "$owner.$key"
        end
    end
    return (; paths, transforms, aliases)
end

"""
    buildperformed(builder, canvas; layer = :combined, labels = Dict(), keep_owners, extra = (;), options...)
        -> (; scene, paths, transforms, aliases)

Run `builder(canvas; options...)`, which returns `(figure, keyed::NamedTuple)`,
keep only `layer`'s plots, and bind the keyed construction values (plus `extra`
ones, such as file-series inputs) to the native properties that consume them.
"""
function buildperformed(builder, canvas; layer::Symbol = :combined, labels = Dict{String,String}(),
                        keep_owners = name -> false, extra = (;), options...)
    layer in (:combined, :picture, :labels) || throw(ArgumentError("invalid film layer $layer"))
    with(CONTEXT => BuildContext(layer, labels)) do
        fig, keyed = builder(Tuple(Int.(canvas)); options...)
        layer === :combined || splitlayer!(fig.scene, layer; keep_owners)
        (; scene = fig.scene, sceneattributes!(fig.scene, merge(keyed, extra))...)
    end
end

"""
    performedtracks(performed, native, fps; layer) -> Vector{(path, keys, discrete)}

The property tracks for a beat performed as `(; picture, labels)` lists of
`name => keys`: frame-quantised, on the native paths (`buildperformed`'s
`paths`, `transforms`, `aliases`) this layer draws, camera paths always. The
picture layer gets only picture keys, so frames that differ in their labels
alone are the same picture.
"""
function performedtracks(performed, native, fps; layer::Symbol = :combined)
    paths, transforms = native.paths, native.transforms
    keyed = layer === :picture ? performed.picture : [performed.picture; performed.labels]
    tracks = map(filter(entry -> haskey(paths, first(entry)) || startswith(first(entry), "camera"), keyed)) do (name, keys)
        # A step holds from the first frame at or after its time, a key otherwise
        # sits on the nearest frame.
        frame(t, ease) = ease === :hold ? ceil(Int, t * fps - 1f-3) : round(Int, t * fps)
        path = get(paths, name, name)
        scale, offset = get(transforms, path, (1.0, 0.0))
        frames = [(frame(t, ease), scale * Float64(v) + offset, ease) for (t, v, ease) in keys]
        # Keys closer than a frame land on one: keep the later one's value.
        merged = Tuple{Int,Float64,Symbol}[]
        for k in sort(frames; by = first)
            !isempty(merged) && first(last(merged)) == first(k) ? (merged[end] = k) : push!(merged, k)
        end
        (path, merged, all(k -> k[2] isa Integer, keys))
    end
    aliased = [(alias, keys, discrete) for (path, keys, discrete) in tracks for alias in get(native.aliases, path, String[])]
    return [tracks; aliased]
end

end
