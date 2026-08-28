# How the editor's own types go into a project file.
#
# ONE registration per type, and MsgPack does the rest — a `Param` knows how to
# be a map of its fields, an `RGBAf` knows how to be four numbers.
# What this replaces is a hand-written mirror of the data model: `effectdict`,
# `paramdict` and a `t == "blur" && return BlurEffect(...)` chain, where every
# field appeared three times (in the type, in the writer, in the reader) and
# could disagree with itself in two of them.
#
# `Clip` and `Sequence` are still written by hand in project.jl, and should be:
# a clip holds an open `VideoSource` and four caches, so what goes in the file is
# a PROJECTION of it and not its fields.
#
# The format is MessagePack rather than JSON because the bulk of a real project
# is per-frame ANALYSIS, not the edit — see `saveproject` for what an hour of it
# costs in each of the three encodings. The short version: 52.4 MB of JSON is
# 6.5 MB here, and the last factor of two is [`Block`](@ref), which writes a
# track as its own bytes and reads it back by pointing at them.

# ---------------------------------------------------------------- primitives
#
# MsgPack's `msgpack_type` is ONE GLOBAL dispatch table shared by every loaded
# package, so a registration here is a registration for the whole session. Two
# consequences, both learned the hard way:
#
#   - Registering a type somebody else already did OVERWRITES their method. A
#     `Symbol` registration here — which MsgPack handles perfectly well itself —
#     made the package refuse to precompile.
#   - Bonito registers every `Vector{Float32}`, `Vector{Float64}`, … as a raw
#     extension, and Bonito is in the editor's tree. That is not something this
#     file can opt out of, so it goes WITH it: `PACKEDELTYPES` below uses
#     Bonito's tag numbering, and the reader normalises whatever it finds.
#
# `UInt64` ids: MessagePack has unsigned integers, so they go as themselves
# rather than as the decimal strings JSON needed to avoid Float64 rounding.
MsgPack.msgpack_type(::Type{UInt64}) = MsgPack.IntegerType()

# ------------------------------------------------------- geometry and colour

# A fixed-size vector is a short array — three numbers, not worth an extension.
MsgPack.msgpack_type(::Type{<:GeometryBasics.Vec}) = MsgPack.ArrayType()
MsgPack.to_msgpack(::MsgPack.ArrayType, v::GeometryBasics.Vec) = collect(v)
MsgPack.from_msgpack(::Type{V}, x::AbstractVector) where {V <: GeometryBasics.Vec} = V(x...)

MsgPack.msgpack_type(::Type{<:GeometryBasics.Mat}) = MsgPack.ArrayType()
MsgPack.to_msgpack(::MsgPack.ArrayType, m::GeometryBasics.Mat) = vec(collect(m))
MsgPack.from_msgpack(::Type{M}, x::AbstractVector) where {M <: GeometryBasics.Mat} = M(x...)

# Colours carry EVERY component. The JSON writer had a `Colorant` method that
# wrote three, so a `Param{RGBAf}` lost its alpha on save without a word.
MsgPack.msgpack_type(::Type{<:Colorant}) = MsgPack.ArrayType()
MsgPack.to_msgpack(::MsgPack.ArrayType, c::Color3) = Float64[red(c), green(c), blue(c)]
MsgPack.to_msgpack(::MsgPack.ArrayType, c::TransparentColor) =
    Float64[red(c), green(c), blue(c), alpha(c)]
MsgPack.from_msgpack(::Type{C}, x::AbstractVector) where {C <: Colorant} = C(x...)

# ------------------------------------------------------------- bulk arrays

"""
The element type per extension tag, at `tag - BLOCKTAG0`.

POSITION IS THE FORMAT, and the positions are not ours to choose: this is
Bonito's assignment, which it registers into MsgPack's global table for every
`Vector{Float32}` and friends. Using the same numbering means a project file has
ONE block encoding and does not change shape depending on which packages a
session happens to have loaded.
"""
const PACKEDELTYPES = (Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64)
const BLOCKTAG0 = Int8(0x10)

"The element type a raw block was written as, or `nothing` for a tag we did not write."
function blocktype(tag::Integer)
    i = Int(tag) - Int(BLOCKTAG0)
    return 1 <= i <= length(PACKEDELTYPES) ? PACKEDELTYPES[i] : nothing
end

"""
    Block(v)

A numeric vector to be written as its own BYTES, whatever else is loaded.

Bonito's registration already does this for a plain `Vector{Float32}` — but
VideoEditor does not depend on Bonito, so a session without it would write the
same track as decimal numbers and a six-times-larger file. The fields where that
matters are few and known — a stabilization track, a colour track, a LUT — so
they say so explicitly and the format stops being a function of the environment.
"""
struct Block{T, V <: AbstractVector{T}}
    values::V
end

MsgPack.msgpack_type(::Type{<:Block}) = MsgPack.ExtensionType()
function MsgPack.pack_type(io, ::MsgPack.ExtensionType, b::Block{T}) where {T}
    i = findfirst(==(T), PACKEDELTYPES)
    i === nothing && error("no raw block format for $T — it is not one of $PACKEDELTYPES")
    a = b.values isa Vector{T} ? b.values : collect(b.values)   # a view or a `reinterpret`
    MsgPack.write_extension_header(io, sizeof(a), Int8(i) + BLOCKTAG0)
    GC.@preserve a Base.unsafe_write(io, Ptr{UInt8}(pointer(a)), UInt(sizeof(a)))
    return nothing
end

"""
    decodeblocks(x) -> x

Every raw numeric block in a just-unpacked document, back as an ordinary vector.

ONE pass over the whole document rather than a conversion at each read site. How
a numeric array was encoded is the format's business: a reader asking for
`d["crop"]` wants four numbers, and whether they arrived as an array or as bytes
is not a question it should have to answer. Getting that wrong is not
theoretical — the scene overlay's settings broke on a `[x, y, z]` that came back
as something with no `length`, because Bonito had written it raw.

A tag nobody here wrote is left ALONE rather than guessed at.
"""
function decodeblocks(e::MsgPack.Extension)
    T = blocktype(e.type)
    return T === nothing ? e : collect(reinterpret(T, e.data))
end
decodeblocks(d::AbstractDict) = Dict(k => decodeblocks(v) for (k, v) in d)
decodeblocks(v::AbstractVector) = Any[decodeblocks(x) for x in v]
decodeblocks(v::Vector{UInt8}) = v          # msgpack binary: bytes, not a list of numbers
decodeblocks(x) = x

"""
    packvec(v) / unpackvec(T, x)

A vector of fixed-size values (`Vec3f`, `Mat3f`) as one flat raw block, and back.

A `Vector{Mat3f}` is nine numbers per entry and a stabilization track has one
entry per frame; written as nested arrays that is an allocation per frame on both
sides. Flattened it is a `reinterpret` in each direction.
"""
packvec(v::AbstractVector{T}) where {T} = Block(reinterpret(eltype(T), v))
unpackvec(::Type{T}, x::AbstractVector) where {T} =
    collect(reinterpret(T, x isa Vector{eltype(T)} ? x : collect(eltype(T), x)))

# ------------------------------------------------------------- editor types

# `MapType`, though `StructType` would write these fields by itself: MsgPack's
# struct READER constructs from the declared field types, which only works where
# the target type is known at the unpack site. A project file is a tree walked
# from the top, so most of it arrives generically unpacked — and `MapType` routes
# both directions through `from_msgpack`, so ONE method per type serves the typed
# read and the hand-walked one.

MsgPack.msgpack_type(::Type{<:Keyframe}) = MsgPack.MapType()
function MsgPack.to_msgpack(::MsgPack.MapType, k::Keyframe)
    d = Dict{String, Any}("frame" => k.frame, "value" => k.value, "ease" => String(k.ease))
    # Only a Bézier anchor has handles, and only then are they written: an older
    # reader sees exactly the file it has always seen, and a `:linear` key does not
    # grow four numbers it will never use.
    if isbezier(k)
        hashandle(k.inhandle) && (d["in"] = [k.inhandle[1], k.inhandle[2]])
        hashandle(k.outhandle) && (d["out"] = [k.outhandle[1], k.outhandle[2]])
    end
    return d
end
readhandle(d, name) = (v = get(d, name, nothing);
                       v === nothing ? NOHANDLE : Handle(Float32(v[1]), Float32(v[2])))
MsgPack.from_msgpack(::Type{Keyframe{T}}, d::AbstractDict) where {T} =
    Keyframe{T}(Int(d["frame"]), MsgPack.from_msgpack(T, d["value"]),
                Symbol(get(d, "ease", "linear")),
                readhandle(d, "in"), readhandle(d, "out"))

MsgPack.msgpack_type(::Type{<:AnimCurve}) = MsgPack.MapType()
MsgPack.to_msgpack(::MsgPack.MapType, c::AnimCurve) =
    Dict{String, Any}("interp" => String(c.interp), "keys" => c.keys)
MsgPack.from_msgpack(::Type{AnimCurve{T}}, d::AbstractDict) where {T} =
    AnimCurve{T}([MsgPack.from_msgpack(Keyframe{T}, k) for k in d["keys"]],
                 Symbol(get(d, "interp", "linear")))

# An input node is written as its OP and its ADDRESSES. What `bindinputs!`
# resolved them to are objects in this process and never go in the file; that same
# call puts them back after the load, as it does after every structural edit.
inputrefdict(r::ParamRef) = Dict{String, Any}("kind" => "param", "clip" => r.clip,
                                              "effect" => r.effect,
                                              "param" => String(r.param))
inputrefdict(r::FileRef) = Dict{String, Any}("kind" => "file", "path" => r.path)
inputrefdict(r::ClipRef) = Dict{String, Any}("kind" => "clip", "clip" => r.clip)

function inputreffromdict(d::AbstractDict)
    k = String(get(d, "kind", "param"))
    k == "file" && return FileRef(String(d["path"]))
    k == "clip" && return ClipRef(UInt64(d["clip"]))
    k == "param" || error("project names an unknown input kind: $(repr(k))")
    return ParamRef(Symbol(d["param"]); clip = UInt64(get(d, "clip", 0)),
                    effect = UInt64(get(d, "effect", 0)))
end

MsgPack.msgpack_type(::Type{<:ParamInput}) = MsgPack.MapType()
MsgPack.to_msgpack(::MsgPack.MapType, n::ParamInput) =
    Dict{String, Any}("op" => String(n.op),
                      "inputs" => [inputrefdict(r) for r in n.inputs])

function MsgPack.from_msgpack(::Type{ParamInput}, d::AbstractDict)
    op = Symbol(get(d, "op", "copy"))
    refs = InputRef[inputreffromdict(r) for r in get(d, "inputs", ())]
    # CHECKED, because the arity is what the evaluation indexes by: a `:mix` with
    # two inputs would reach past the end of the list at the first frame that read
    # it, deep inside a render, rather than here where the file can be named.
    length(refs) == inputarity(op) ||
        error("project has a $(repr(op)) input with $(length(refs)) source(s); it takes " *
              "$(inputarity(op))")
    return ParamInput(op, refs...)
end

# An effect is its KIND plus its parameters, and nothing about the kind itself.
# What it replaces is one writer per effect type and a reader that was a chain of
# `t == "blur" && return BlurEffect(...)`: adding an effect meant editing three
# places, and a plugin effect — a kind that exists only at runtime — could not be
# saved at all, because there was no branch to add.
MsgPack.msgpack_type(::Type{<:Effect}) = MsgPack.MapType()
MsgPack.to_msgpack(::MsgPack.MapType, fx::Effect) =
    Dict{String, Any}("id" => fx.id, "kind" => String(fx.kind), "enabled" => fx.enabled,
                      "params" => fx.params)

"""
The kind supplies the parameter LIST, the file supplies each parameter's STATE.

Not simply the parameters as written: a kind that has gained a parameter since
the file was saved must still get its default, and one it has since dropped must
not come back. Label and range come from the kind for the same reason — they are
what the effect IS, and the file is only what somebody set.

That is also what makes the format writable by a script: `{"kind": "blur",
"params": [{"T": "Float64", "name": "blur", "value": 4.0}]}` is a complete
effect, because everything omitted is something the kind already knows.
"""
function MsgPack.from_msgpack(::Type{Effect}, d::AbstractDict)
    kind = anykind(Symbol(d["kind"]))
    kind === nothing && error("project uses an effect kind that is not registered: " *
                              repr(d["kind"]))
    fx = Effect(UInt64(get(d, "id", freshid())), Symbol(d["kind"]),
                Bool(get(d, "enabled", true)), paramsfor(kind))
    stored = Param[MsgPack.from_msgpack(Param, pd) for pd in get(d, "params", ())]
    for st in stored
        cur = param(fx, st.name)
        if cur === nothing
            # A parameter the KIND does not declare. For an ordinary effect that
            # means the kind has dropped it since the file was written, and it must
            # not come back. For a DATA entry — the `:scene`, whose kind declares
            # nothing because what is animatable depends on the scene — the stored
            # parameters ARE the content, and dropping them would lose every curve
            # in the project.
            kind.make === nothing && push!(fx.params, st)
            continue
        end
        cur.value = st.value
        cur.visible = st.visible
        cur.curve = st.curve
        cur.input = st.input     # unresolved — `loadproject` binds once the clips exist
    end
    return fx
end

# `Param` is the one type that has to record its own, because `Effect.params` is
# a `Vector{Param}` — heterogeneous by design, since one effect may have a
# `Param{Float64}` beside a `Param{Vec3f}`. Reading one back therefore cannot be
# inferred from the field's declared type the way every other field can; the
# packed map says which it is.
#
# A NAME, resolved against a table, not an `eval` of whatever the file says: a
# project file is data, and data must not be able to name a type into existence.
# Written out rather than derived from `nameof`: `nameof(Vec3f)` is `:Vec`, so
# any scheme that patches the length back on gives `Vec3d` the same name as
# `Vec3f` — two entries, one key, and whichever lost would read back as the
# other's element type.
const PARAMTYPES = Dict{String, Type}(
    "Float64" => Float64, "Float32" => Float32, "Int64" => Int64, "Bool" => Bool,
    "Vec2f" => Vec2f, "Vec3f" => Vec3f, "RGBf" => RGBf, "RGBAf" => RGBAf)

paramtypename(::Type{T}) where {T} = findfirst(==(T), PARAMTYPES)

MsgPack.msgpack_type(::Type{<:Param}) = MsgPack.MapType()

function MsgPack.to_msgpack(::MsgPack.MapType, p::Param{T}) where {T}
    name = paramtypename(T)
    name === nothing && error("Param{$T} cannot be written — add $T to `PARAMTYPES`")
    d = Dict{String, Any}("T" => name, "name" => String(p.name), "label" => p.label,
                          "value" => p.value, "visible" => p.visible)
    # two scalars, not a vector: a range is always a pair, and as an array it
    # went out through the raw-block path for no gain — and came back as
    # something the reader could not broadcast `Float64` over
    p.range === nothing || (d["lo"] = Float64(p.range[1]); d["hi"] = Float64(p.range[2]))
    p.curve === nothing || (d["curve"] = p.curve)
    p.input === nothing || (d["input"] = p.input)
    return d
end

function MsgPack.from_msgpack(::Type{<:Param}, d::AbstractDict)
    T = get(PARAMTYPES, d["T"]) do
        error("project names an unknown parameter type: $(repr(d["T"]))")
    end
    value = MsgPack.from_msgpack(T, d["value"])
    # the curve arrives as a plain map, and `T` — which the file just told us —
    # is what says how to read a key's value. That is the whole reason this
    # method is written out: every other field could be inferred from the type.
    curve = haskey(d, "curve") ? MsgPack.from_msgpack(AnimCurve{T}, d["curve"]) : nothing
    range = haskey(d, "lo") ? (Float64(d["lo"]), Float64(d["hi"])) : nothing
    input = haskey(d, "input") ? MsgPack.from_msgpack(ParamInput, d["input"]) : nothing
    return Param{T}(Symbol(d["name"]), String(get(d, "label", d["name"])), value, curve,
                    Bool(get(d, "visible", false)), range, input)
end

# No `from_msgpack` for the scalars: MsgPack converts a number to the asked-for
# `Real` itself, and adding methods here made `Bool` ambiguous with `Real`
# (a `Bool` IS a `Real`) — which shows up only when unpacking, not when writing.
