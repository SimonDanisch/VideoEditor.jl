"""
    framekey(clip, sf)

What source frame `sf` of `clip` is drawn from: frames with equal keys are the
same picture. A scene program (see [`programscene`](@ref)) is drawn from its
scene parameters alone, so its key is their values at `sf`; any other source is
drawn from its frame, so its key is `sf`.
"""
framekey(clip::Clip, sf::Integer) = framekey(clip.source, clip, Int(sf))
framekey(src::SceneSource, clip::Clip, sf::Int) = parameterizedscene(src.root) ?
    (get(src.build, "recordings", nothing) === nothing ? paramskey(clip, sf) : (sf, paramskey(clip, sf))) : sf
framekey(::ClipSource, ::Clip, sf::Int) = sf

parameterizedscene(::SceneProgram) = true
parameterizedscene(::Any) = false

function paramskey(clip::Clip, sf::Int)
    fx = findslot(clip, :scene)
    # A bypassed scene effect draws the recipe's own arguments on every frame.
    (fx === nothing || !fx.enabled[]) && return Any[]
    return Any[valuekey(p, sf) for p in fx.params]
end

"""
    valuekey(p, frame)

What `p` sets at `frame`: its value, or for a driven parameter what its node
reads there. A parameter read gives its own key at the frame read, a recorded
track its value; a file or file series is the same object on every frame, and
another clip's picture is only known to be equal at the same frame. A volume
sampled from a file series is so compared by the time it is sampled at, never
by its data.
"""
function valuekey(p::Param, frame::Integer)
    n = p.input
    (n === nothing || !drives(n.op) || any(isnothing, n.resolved)) && return valueat(p.curve[], frame)
    return (n.op, ntuple(i -> inputkey(n.resolved[i], inputframe(n, i, frame)), length(n.resolved)))
end
inputkey(p::Param, frame::Integer) = valuekey(p, frame)
inputkey(t::RecordedTrack, frame::Integer) = valueat(t, frame)
inputkey(c::Clip, frame::Integer) = (c.id, frame)
inputkey(x, ::Integer) = objectid(x)

"""
    twinframe(cache, clip, sf) -> Int or nothing

Another source frame of `clip` with `sf`'s key (see [`framekey`](@ref)) whose
entry in `cache` is valid.
"""
function twinframe(cache::TrackCache, clip::Clip, sf::Integer)
    key = framekey(clip, sf)
    for f in get(cache.valid, clip.id, BitSet())
        f != sf && isequal(framekey(clip, f), key) && cachevalid(cache, clip.id, f) && return f
    end
    return nothing
end

"Whether a clip's finished pictures are traced, so worth a farm."
traced(src::SceneSource) = !get(src.bakescreenopts, :rasterize, true)
traced(::ClipSource) = false

"""
    distinctframes(seq, frames) -> Vector{Int}

`frames` without those whose traced pictures others of them already trace: a
frame is left out when every traced clip in it shows a source frame with the key
(see [`framekey`](@ref)) of one in a frame kept before it. The export links the
left-out frames' masters from those (see [`twinsourceframe!`](@ref)).
"""
function distinctframes(seq::Sequence, frames)
    seen = Set{Any}()
    kept = Int[]
    for n in frames
        keys = [(c.id, framekey(c, sourceframe(c, n))) for c in frameclips(seq, n) if traced(c.source)]
        (isempty(keys) || !all(in(seen), keys)) && push!(kept, n)
        union!(seen, keys)
    end
    return kept
end

"Whether a timeline frame still needs farm work rather than the project's native caches."
function farmneedsframe(seq::Sequence, n::Integer)
    0 <= n < seqlength(seq) || return true
    comp=seq.compositecache
    comp!==nothing && cachevalid(comp,0,n) && return false
    transitionat(seq,n)===nothing || return true
    clips=frameclips(seq,n)
    isempty(clips) && return true
    # A linked clip can request data outside this timeline frame. Be conservative
    # rather than infer that its dependency is cached from the visible clips.
    any(p.input!==nothing &&
        (any(x->x isa ClipRef || x isa SceneRef,p.input.inputs) ||
         any(x->x isa Clip && x!==c,p.input.from))
        for c in clips for fx in c.effects for p in fx.params) && return true
    allcached=true
    traced=false
    for c in clips
        cache=trackcache(c)
        sf=sourceframe(c,n)
        plane=cache!==nothing && cachevalid(cache,c.id,sf)
        allcached &= plane
        if c.source isa SceneSource
            if !get(c.source.bakescreenopts,:rasterize,true)
                traced=true
                raw=sourcecache(c)
                plane || (raw!==nothing && (cachevalid(raw,c.id,sf) || twinframe(raw,c,sf)!==nothing)) || return true
            end
        elseif !plane
            return true
        end
    end
    # A cached traced picture can be graded/composited with dirty raster labels
    # locally. Entirely uncached raster scenes still go to the farm normally.
    return !(allcached || traced)
end

"""
    farmneededframes(seq; frames=0:seqlength(seq)-1)

Frames whose expensive sources are missing from the native project caches, a
traced frame counting as cached when a frame with its key is (see [`framekey`](@ref)).
Used by `renderfarm!(...; reuse=seq)`: returned HDR masters survive a new job
snapshot or a label/grade edit. Finish with `adoptfarmcache!` and ordinary
`exportvideo`; that export resolves the remaining cached pictures/raster labels.
No receipts are fabricated and no cache files are moved or deleted.
"""
farmneededframes(seq::Sequence;frames=0:(seqlength(seq)-1)) =
    [Int(n) for n in frames if farmneedsframe(seq,n)]
