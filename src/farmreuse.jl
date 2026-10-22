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
                plane || (raw!==nothing && cachevalid(raw,c.id,sf)) || return true
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

Frames whose expensive sources are missing from the native project caches.
Used by `renderfarm!(...; reuse=seq)`: returned HDR masters survive a new job
snapshot or a label/grade edit. Finish with `adoptfarmcache!` and ordinary
`exportvideo`; that export resolves the remaining cached pictures/raster labels.
No receipts are fabricated and no cache files are moved or deleted.
"""
farmneededframes(seq::Sequence;frames=0:(seqlength(seq)-1)) =
    [Int(n) for n in frames if farmneedsframe(seq,n)]
