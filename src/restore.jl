"""
Upscaling and restoration: replace a clip's frames with a model's better ones.

Shaped like the matte (`matte.jl`) rather than like a per-pixel effect, and for
the same reason: a restoration model is temporal — BasicVSR++ propagates
information along a whole clip — so it cannot be a function of one frame, and
running it inside the render path would put recurrent state where scrubbing,
transitions and export all jump around.

Unlike a matte, though, the output is *the picture*, at output resolution. A
clip's worth of 4x frames is far too much to keep (a 480x270 source restores to
1920x1080, ~6 MB a frame), so this does not precompute a whole track. It keeps a
bounded cache of restored frames and fills it a window at a time. A frame that
is not in the cache renders unchanged rather than stalling the player — the
restoration shows up when it is ready, which is the same bargain proxies make.

The model is pluggable via [`registerrestore!`](@ref); with none installed the
effect is inert, so the UI, the project format and the tests all work without
weights on the machine.
"""

"""
    registerrestore!(f)

Install the restoration model. `f(frames; progress) -> Vector{Matrix{RGB{N0f8}}}`
takes a window of consecutive source frames and returns the same number of
restored frames, each `scale` times larger.

Temporal models need the window, not a frame: that is the whole reason this is
an analysis. The returned frames must line up one-to-one with the input.
"""
const RESTOREMODEL = Ref{Any}(nothing)
const RESTORESCALE = Ref{Int}(4)
# How many consecutive frames the model wants per call. Exported graphs pin the
# frame count, so the UI has to ask rather than choose.
const RESTOREWINDOW = Ref{Int}(5)

function registerrestore!(f; scale::Integer = 4, window::Integer = 5)
    RESTOREMODEL[] = f
    RESTORESCALE[] = Int(scale)
    RESTOREWINDOW[] = Int(window)
    nothing
end

hasrestoremodel() = RESTOREMODEL[] !== nothing
restorescale() = RESTORESCALE[]
restorewindowlength() = RESTOREWINDOW[]

"""
Restored frames for one clip, bounded.

Keyed by absolute source frame so it survives splits and trims like the tracks
do. `order` is an insertion queue used to evict the oldest window when `limit`
is exceeded — a plain LRU would be better if playback ever ran backwards, which
it does not.
"""
mutable struct RestoreCache
    const frames::Dict{Int, Matrix{RGB{N0f8}}}
    const order::Vector{Int}
    limit::Int
end
RestoreCache(limit::Integer = 96) = RestoreCache(Dict{Int, Matrix{RGB{N0f8}}}(), Int[], Int(limit))

const RESTORECACHES = Dict{UInt64, RestoreCache}()

restorecache(clip::Clip) = get!(() -> RestoreCache(), RESTORECACHES, clip.id)
hasrestored(clip::Clip, srcframe::Integer) = haskey(restorecache(clip).frames, Int(srcframe))
clearrestore!(clip::Clip) = (delete!(RESTORECACHES, clip.id); nothing)
clearrestore!() = (empty!(RESTORECACHES); nothing)

function putrestored!(c::RestoreCache, f::Integer, img)
    f = Int(f)
    haskey(c.frames, f) || push!(c.order, f)
    c.frames[f] = img
    while length(c.order) > c.limit
        delete!(c.frames, popfirst!(c.order))
    end
    return img
end

"""
    restorewindow!(clip, readframe, first, n; progress) -> Int

Restore `n` consecutive source frames starting at `first` into the clip's cache.
Returns how many frames were produced.

`readframe(srcframe) -> Matrix{RGB{N0f8}}` supplies source frames; the caller
decides the decoder, exactly as `analyzematte!` does.
"""
function restorewindow!(clip::Clip, readframe, first::Integer, n::Integer;
                        progress = nothing)
    hasrestoremodel() || error("no restoration model installed — see registerrestore!")
    lo = max(clip.src_in, Int(first))
    hi = min(clip.src_out - 1, lo + Int(n) - 1)
    hi >= lo || return 0
    frames = [copy(readframe(f)) for f in lo:hi]
    out = RESTOREMODEL[](frames; progress = progress)
    length(out) == length(frames) ||
        error("restoration model returned $(length(out)) frames for $(length(frames))")
    c = restorecache(clip)
    for (k, f) in enumerate(lo:hi)
        putrestored!(c, f, out[k])
    end
    return length(out)
end

"""
    applyrestore!(buf, clip, srcframe; strength)

Blend the restored frame for `srcframe` into `buf`, in place.

A miss is a no-op, not an error: the frame renders as decoded until its window
has been restored. `strength` cross-fades against the original so the effect can
be keyframed in, and because a restoration is a judgement call the user may want
half of.

`buf` is at the clip's render size; the restored frame is `restorescale()` times
the source, so it is sampled rather than blitted — which also means the effect
does something visible even when the render target is not 4x.
"""
@kernel function restore_kernel!(buf, @Const(hi), sw::Int32, sh::Int32, strength::Float32)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w, h = size(buf, 1), size(buf, 2)
        u = clamp(round(Int32, (Float32(i) - 0.5f0) / Float32(w) * Float32(sw) + 0.5f0),
                  Int32(1), sw)
        v = clamp(round(Int32, (Float32(j) - 0.5f0) / Float32(h) * Float32(sh) + 0.5f0),
                  Int32(1), sh)
        r = hi[u, v]
        c = buf[i, j]
        buf[i, j] = RGB{N0f8}(
            clamp(Float32(red(c)) + strength * (Float32(red(r)) - Float32(red(c))), 0.0f0, 1.0f0),
            clamp(Float32(green(c)) + strength * (Float32(green(r)) - Float32(green(c))), 0.0f0, 1.0f0),
            clamp(Float32(blue(c)) + strength * (Float32(blue(r)) - Float32(blue(c))), 0.0f0, 1.0f0))
    end
end

const RESTOREPLANES = IdDict{Any, Dict{Any, Tuple{Int, Any}}}()

function restoreplane!(clip::Clip, f::Int, img, backend)
    per = get!(() -> Dict{Any, Tuple{Int, Any}}(), RESTOREPLANES, clip.id)
    hit = get(per, backend, nothing)
    hit === nothing || hit[1] == f && return hit[2]
    dev = KA.allocate(backend, RGB{N0f8}, size(img)...)   # one path, every backend
    copyto!(dev, img)
    per[backend] = (f, dev)
    return dev
end

function applyrestore!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer;
                       strength::Real = 1.0)
    s = Float32(clamp(strength, 0.0, 1.0))
    s <= 0.0f0 && return buf
    c = get(RESTORECACHES, clip.id, nothing)
    c === nothing && return buf
    img = get(c.frames, Int(srcframe), nothing)
    img === nothing && return buf
    backend = KA.get_backend(buf)
    dev = restoreplane!(clip, Int(srcframe), img, backend)
    restore_kernel!(backend)(buf, dev, Int32(size(img, 1)), Int32(size(img, 2)), s;
                             ndrange = size(buf))
    KA.synchronize(backend)
    return buf
end
