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
    restorecache!(clip) -> RestoreCache

The clip's restored frames, created on first use. [`RestoreCache`](@ref) is
declared in `clips.jl` beside the tracks, because it is a field of `Clip` — it
used to be a module global keyed by clip id, and [`split!`](@ref) mints the right
half's id fresh, so a split silently dropped the restoration while the line above
it took explicit care of the matte.
"""
restorecache!(clip::Clip) =
    clip.restorecache === nothing ? (clip.restorecache = RestoreCache()) : clip.restorecache
hasrestored(clip::Clip, srcframe::Integer) =
    clip.restorecache !== nothing && haskey(clip.restorecache.frames, Int(srcframe))
clearrestore!(clip::Clip) = (clip.restorecache = nothing)
clearrestore!(seq::Sequence) = (foreach(clearrestore!, seq.clips); nothing)

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
    c = restorecache!(clip)
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
        # `unitn0f8`, not `RGB{N0f8}(::Float32, …)` — see its docstring in
        # matte.jl: the checked constructor's error path builds a message with
        # `repr`, and Lava rejects the whole kernel for the string allocation.
        # This kernel had the checked one, so the restoration compiled on the
        # CPU and could never have run on the GPU tier at all; nothing noticed
        # because nothing had rendered a restored clip on a device.
        src = Vec3f(red(c), green(c), blue(c))
        out = src .+ strength .* (Vec3f(red(r), green(r), blue(r)) .- src)
        buf[i, j] = RGB{N0f8}(unitn0f8(out[1]), unitn0f8(out[2]), unitn0f8(out[3]))
    end
end

# ── the restoration as a plane op (see `PlaneOp` in gpugraph.jl) ──────────────
#
# The restored picture is the node's second input, so it travels the one route
# every per-frame plane travels: a pool-backed buffer the graph writes and the
# kernel reads. It used to be a `KA.allocate` per frame behind a module global —
# ~6 MB of device memory allocated and dropped every time the frame changed,
# outside the pool and never freed.

planeeltype(::RestoreOp) = RGB{N0f8}
passname(::RestoreOp) = "restore"

"How big the restored picture is: `restorescale()` times the source, whatever the
model actually returned. `nothing` until a window has been restored."
function planeshape(::RestoreOp, clip::Clip)
    c = clip.restorecache
    (c === nothing || isempty(c.frames)) && return nothing
    return size(first(values(c.frames)))
end

"""
The object whose bytes a restore plane holds: the cached frame itself. Compared
by identity, so restoring a window a second time — which replaces the image under
an unchanged clip and frame — writes the plane again.
"""
function planesource(::RestoreOp, clip::Clip, srcframe::Integer)
    c = clip.restorecache
    return c === nothing ? nothing : get(c.frames, Int(srcframe), nothing)
end

planedata(op::RestoreOp, clip::Clip, srcframe::Integer) =
    (img = planesource(op, clip, srcframe);
     img === nothing ? nothing : vec(img))   # a reshape of the cached frame, not a copy

"""
    applyplane!(buf, plane, op::RestoreOp, clip)

Blend the restored frame into `buf`, in place. `strength` cross-fades against the
original so the effect can be keyframed in, and because a restoration is a
judgement call the user may want half of.

`buf` is at the clip's render size and the plane is `restorescale()` times the
source, so it is sampled rather than blitted — which also means the effect does
something visible when the render target is not 4x.
"""
function applyplane!(buf::AnyRGBFrame, plane, op::RestoreOp, ::Clip)
    s = clamp(op.strength, 0.0f0, 1.0f0)
    s <= 0.0f0 && return buf
    backend = KA.get_backend(buf)
    restore_kernel!(backend)(buf, plane, Int32(size(plane, 1)), Int32(size(plane, 2)), s;
                             ndrange = size(buf))
    return buf
end

"""
    applyrestore!(buf, clip, srcframe; strength)

The host-side form: restore a HOST buffer straight from the clip's cache. A miss
is a no-op, not an error — the frame renders as decoded until its window has been
restored.

In the render path the plane is a graph resource and the node calls
[`applyplane!`](@ref); this exists for the same reason `applymatte!` does, so a
tool or a test can key one frame without building a graph for it.
"""
function applyrestore!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer;
                       strength::Real = 1.0)
    op = RestoreOp(Float32(clamp(strength, 0.0, 1.0)))
    d = planedata(op, clip, srcframe)
    d === nothing && return buf
    return applyplane!(buf, reshape(d, planeshape(op, clip)), op, clip)
end
