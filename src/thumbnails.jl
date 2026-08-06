"""
    ThumbnailCache(source; thumbheight=88, maxthumbs=2000)

Thumbnails keyed by integer video-second, filled by a dedicated background
decoder. All timeline zoom levels share this store: a tile spanning `2^k`
seconds shows the thumbnail of its starting second, so only one strip of
thumbnails ever needs decoding.

The UI pushes wanted seconds via `request!` (visible range, ascending);
the worker decodes them (forward-decode when close, seek when far — same
policy as the playback worker), downscales, and sets `dirty` so the UI
knows to refresh. Eviction is LRU above `maxthumbs`.
"""
mutable struct ThumbnailCache
    const source::VideoSource
    const thumbwidth::Int
    const thumbheight::Int
    const thumbs::Dict{Int, RGBFrame}
    const stamps::Dict{Int, Int}
    const wishlist::Vector{Int}
    const lock::ReentrantLock
    const running::Threads.Atomic{Bool}
    const dirty::Threads.Atomic{Bool}
    counter::Int
    maxthumbs::Int
    gpurun::Any   # synchronous GPU-worker runner (f -> f's result) — nothing = CPU decode
    task::Task

    function ThumbnailCache(source::VideoSource; thumbheight::Integer = 88,
                            maxthumbs::Integer = 2000, gpurun = nothing)
        thumbwidth = max(round(Int, thumbheight * source.width / source.height), 8)
        cache = new(source, thumbwidth, thumbheight, Dict{Int, RGBFrame}(),
                    Dict{Int, Int}(), Int[], ReentrantLock(),
                    Threads.Atomic{Bool}(true), Threads.Atomic{Bool}(false),
                    0, maxthumbs, gpurun)
        cache.task = Threads.@spawn (gpurun === nothing ? thumbloop(cache) : gputhumbloop(cache))
        return cache
    end
end

function stop!(cache::ThumbnailCache)
    cache.running[] = false
    wait(cache.task)
    return nothing
end

"Queue a single second for decoding (used by the on-demand thumbnail provider)."
function requestone!(cache::ThumbnailCache, s::Integer)
    lock(cache.lock) do
        haskey(cache.thumbs, s) && return
        s in cache.wishlist || push!(cache.wishlist, Int(s))
    end
    return nothing
end

"Replace the wishlist with the seconds wanted now (ascending = decode order)."
function request!(cache::ThumbnailCache, seconds::Vector{Int})
    lock(cache.lock) do
        empty!(cache.wishlist)
        for s in seconds
            haskey(cache.thumbs, s) || push!(cache.wishlist, s)
        end
    end
    return nothing
end

"Thumbnail for exactly second `s`, or `nothing`. Touches the LRU stamp."
function getthumb(cache::ThumbnailCache, s::Integer)
    lock(cache.lock) do
        thumb = get(cache.thumbs, s, nothing)
        thumb === nothing || (cache.stamps[s] = (cache.counter += 1))
        return thumb
    end
end

"Nearest cached thumbnail to second `s` within `maxdist`, or `nothing`."
function nearestthumb(cache::ThumbnailCache, s::Integer; maxdist::Integer = 30)
    lock(cache.lock) do
        for d in 0:maxdist
            for cand in (s + d, s - d)
                thumb = get(cache.thumbs, cand, nothing)
                if thumb !== nothing
                    cache.stamps[cand] = (cache.counter += 1)
                    return thumb
                end
            end
        end
        return nothing
    end
end

function storethumb!(cache::ThumbnailCache, s::Integer, thumb::RGBFrame)
    lock(cache.lock) do
        cache.thumbs[s] = thumb
        cache.stamps[s] = (cache.counter += 1)
        while length(cache.thumbs) > cache.maxthumbs
            oldest = argmin(cache.stamps)
            delete!(cache.thumbs, oldest)
            delete!(cache.stamps, oldest)
        end
    end
    cache.dirty[] = true
    return nothing
end

"""
GPU thumbnail worker: decodes through its OWN small [`GpuVideoStream`](@ref)
(never evicting the playback ring) and area-averages on-device — only the tiny
finished thumb crosses to the host. Every GPU touch goes through `cache.gpurun`
(the player's pinned worker — Lava is single-writer). Falls back to the CPU
loop when the stream won't open.
"""
function gputhumbloop(cache::ThumbnailCache)
    source = cache.source
    stream = dev = thumbdev = nothing
    ok = try
        # `long`: cold this compiles decode + convert + downscale — warm them all
        # HERE so the per-thumb jobs below stay bounded and presents interleave
        cache.gpurun(; long = true) do
            stream = openstream(LavaBackend(), source.path, source.width, source.height;
                                vrambudget = 2^30)
            dev = KA.allocate(LavaBackend(), RGB{N0f8}, (source.width, source.height))
            thumbdev = KA.allocate(LavaBackend(), RGB{N0f8}, (cache.thumbwidth, cache.thumbheight))
            f = exactframeat!(stream, 0)
            nv12torgb!(dev, f.y, f.uv; bt601 = stream.bt601)
            areadownscale!(thumbdev, dev)
            nothing
        end
        true
    catch e
        @warn "GPU thumbnails unavailable — CPU decode" source = source.path exception = e
        false
    end
    ok || return thumbloop(cache)
    host = RGBFrame(undef, cache.thumbwidth, cache.thumbheight)
    try
        while cache.running[]
            s = lock(() -> isempty(cache.wishlist) ? nothing : popfirst!(cache.wishlist), cache.lock)
            if s === nothing
                sleep(0.03)
                continue
            end
            n = clamp(frameindex(source, Float64(s)), 0, source.nframes - 1)
            # decode in LATENCY-BOUNDED jobs (≤ ~90 ms each) so playback presents
            # interleave with the thumbnail's GOP decode on the shared worker
            while cache.running[] && !cache.gpurun(() -> (frameat!(stream, n); hasframe(stream, n)))
            end
            cache.running[] || break
            cache.gpurun() do
                f = frameat!(stream, n)              # resident now — pure ring hit
                nv12torgb!(dev, f.y, f.uv; bt601 = stream.bt601)
                areadownscale!(thumbdev, dev)
                copyto!(host, thumbdev)
                nothing
            end
            storethumb!(cache, Int(s), copy(host))
        end
    catch e
        @error "GPU thumbnail worker died" exception = (e, catch_backtrace())
    finally
        try
            cache.gpurun() do; close(stream); nothing; end
        catch
        end
    end
    return nothing
end

function thumbloop(cache::ThumbnailCache)
    source = cache.source
    reader = VideoIO.openvideo(source.path, target_format = VideoIO.AV_PIX_FMT_RGB24)
    scratch = RGBFrame(undef, source.width, source.height)
    scratch_hw = PermutedDimsArray(scratch, (2, 1))
    position = -1  # frame index the reader will produce next, -1 = unknown
    try
        while cache.running[]
            s = lock(() -> isempty(cache.wishlist) ? nothing : popfirst!(cache.wishlist), cache.lock)
            if s === nothing
                sleep(0.03)
                continue
            end
            n = frameindex(source, Float64(s))
            if position < 0 || n < position || n - position > 4 * round(Int, source.framerate)
                seek(reader, frametime(source, n))
                position = n
            end
            while position <= n && cache.running[]
                read!(reader, scratch_hw)
                position += 1
            end
            storethumb!(cache, Int(s), downscale(scratch, cache.thumbwidth, cache.thumbheight))
        end
    catch e
        e isa EOFError || @error "thumbnail worker died" exception = (e, catch_backtrace())
    finally
        close(reader)
    end
    return nothing
end

"""
Area-average downscale of a (w, h) frame to (tw, th): every output pixel is
the mean of its full source cell, so decimation cannot alias (nearest-neighbor
sampling every ~12th pixel is what made the timeline thumbnails look mangled).
Falls back to nearest-neighbor when the target is not actually smaller.
"""
function downscale(frame::RGBFrame, tw::Integer, th::Integer)
    w, h = size(frame)
    thumb = RGBFrame(undef, tw, th)
    if tw >= w || th >= h   # upscale: no cells to average
        for j in 1:th, i in 1:tw
            thumb[i, j] = frame[clamp(round(Int, (i - 0.5) * w / tw), 1, w),
                                clamp(round(Int, (j - 0.5) * h / th), 1, h)]
        end
        return thumb
    end
    acc = zeros(Float32, 3, tw, th)
    cnt = zeros(Float32, tw, th)
    xmap = [min(floor(Int, (i - 1) * tw / w) + 1, tw) for i in 1:w]
    ymap = [min(floor(Int, (j - 1) * th / h) + 1, th) for j in 1:h]
    @inbounds for j in 1:h
        tj = ymap[j]
        for i in 1:w
            ti = xmap[i]
            c = frame[i, j]
            acc[1, ti, tj] += Float32(c.r)
            acc[2, ti, tj] += Float32(c.g)
            acc[3, ti, tj] += Float32(c.b)
            cnt[ti, tj] += 1.0f0
        end
    end
    @inbounds for j in 1:th, i in 1:tw
        n = cnt[i, j]
        thumb[i, j] = RGB{N0f8}(clamp(acc[1, i, j] / n, 0.0f0, 1.0f0),
                                clamp(acc[2, i, j] / n, 0.0f0, 1.0f0),
                                clamp(acc[3, i, j] / n, 0.0f0, 1.0f0))
    end
    return thumb
end

"Nearest-neighbor upscale blit of `thumb` into the full-size `dest` frame."
function blitthumb!(dest::RGBFrame, thumb::RGBFrame)
    w, h = size(dest)
    tw, th = size(thumb)
    for j in 1:h
        ty = clamp(round(Int, (j - 0.5) * th / h), 1, th)
        for i in 1:w
            dest[i, j] = thumb[clamp(round(Int, (i - 0.5) * tw / w), 1, tw), ty]
        end
    end
    return dest
end
