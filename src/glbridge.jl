# Opt-in GPU-resident preview via the Vulkan↔GL bridge.
#
# With `Player(path; gpupreview = true)`, presentation runs on the GPU:
# the decoded frame is uploaded once, motion/color tracks and the effect
# stack run as Lava kernels (the same backend-generic GPUFiltering calls
# the CPU path uses), the result is blitted into a `Lava.ExternalImage`,
# and GLMakie samples that memory directly through a swapped texture —
# no CPU effect cost, no display-path copy. Measured 2.1× the CPU preview
# at 1080p (`gpu_preview_prototype.jl`), with the CPU freed for decoding.
#
# Threading contract (see GPUWorker): every Lava call runs on the pinned
# GPU worker thread; every GL call runs on the main thread with GLMakie's
# context current. `presentgpu!` does a synchronous round-trip per frame —
# ~7 ms at 1080p, strictly less than the CPU path it replaces (~15 ms).
#
# Any error disables the bridge permanently for the session and playback
# falls back to the CPU path (with a status message) — `gpupreview` can
# never make the editor worse than the default.

"Per-resolution GPU presentation state (see `presentgpu!`). The shared image is
DOUBLE-BUFFERED: GLMakie's render loop runs as a concurrent task, so blitting
into the texture it is currently sampling tears — visible as a glitched frame
whenever consecutive frames differ strongly (a stabilization warp!). Each
present blits into the texture GLMakie is NOT showing, then swaps."
mutable struct GPUPreview
    width::Int
    height::Int
    failed::Bool
    inline::Bool   # Lava context is owned by the main thread → run jobs inline
    # worker-owned (Lava)
    packed::Any          # LavaArray{UInt32,1} — RGBA pack scratch for the blit
    eimages::Vector{Any} # 2 Lava.ExternalImage back/front buffers
    engine::Any          # FxEngine — owns the effect graph's reusable device buffers
    # main-thread-owned (GL)
    gltex::Vector{Any}   # the 2 imported GL texture wrappers
    cur::Int             # index (1/2) of the buffer GLMakie currently samples
    doublebuffer::Bool   # live A/B switch: false = pre-2026-07-22 single-buffer blit
    origtex::Any         # the plot's OWN texture — restored whenever the CPU `frame`
                         # path presents (scrub thumbs, gap fills, CPU fallback):
                         # GLMakie's notify-upload writes into the texture the render
                         # object holds, and uploading linear CPU pixels into the
                         # imported optimal-tiled external texture shreds the image
end
GPUPreview() = GPUPreview(0, 0, false, false, nothing, Any[], nothing, Any[], 1, true, nothing)

"""
Run `f` on the player's pinned GPU worker and wait for its result. `long = true`
marks jobs that hold the worker for more than a frame period (stream warmup,
cold kernel compiles): while any is queued or running, `gpuready` turns false
and presents take the CPU lane instead of stalling behind it in the job queue.
"""
function rungpusync(f::Function, player::Player; long::Bool = false)
    long && Threads.atomic_add!(player.gpubusy, 1)
    try
        done = Channel{Any}(1)
        rungpu(player) do
            try
                put!(done, (true, f()))
            catch e
                put!(done, (false, e))
            end
        end
        ok, val = take!(done)
        ok || throw(val)
        return val
    finally
        long && Threads.atomic_sub!(player.gpubusy, 1)
    end
end

"""
Run `f` on whichever thread owns the Lava context. Normally that is the
player's pinned GPU worker (the context is created there on first use);
when Lava was already used on the main thread before this player existed,
the single-writer BatchQueue belongs to main — detected once via the
worker's assertion, after which jobs run inline (presentation already
happens on the main thread, so inline is both legal and lower-latency).
"""
function rungpuowned(f::Function, player::Player, gp::GPUPreview)
    gp.inline && return f()
    try
        return rungpusync(f, player)
    catch e
        if e isa AssertionError && occursin("single-writer", e.msg)
            gp.inline = true
            return f()
        end
        rethrow()
    end
end

"""
Prepare (or re-prepare after a resolution change) the GPU presentation
chain: device buffers + exportable image on the worker, then GL import and
texture swap on the main thread. Assumes `player.frame[]` already has the
target size and the preview plot has processed it (so the plot geometry is
current before its texture is replaced).
"""
function setupgpupreview!(player::Player, gp::GPUPreview, W::Integer, H::Integer)
    # VideoEditor doesn't depend on Lava — reach it through the backend's module
    lavamod = parentmodule(typeof(player.analysisbackend))
    fds = rungpuowned(player, gp) do
        backend = player.analysisbackend
        gp.engine !== nothing && emptyengine!(gp.engine)   # drop old-resolution graph buffers
        gp.engine = nothing
        gp.packed = KA.allocate(backend, UInt32, Int(W) * Int(H))   # RGBA pack scratch
        empty!(gp.eimages)
        map(1:2) do _   # double buffer: blit into one while GL samples the other
            img = lavamod.ExternalImage(W, H)
            push!(gp.eimages, img)
            (lavamod.memoryfd(img), img.allocation_size)
        end
    end

    # GL side: import both fds and swap the preview plot's texture (main thread)
    screen = player.screen
    GLMakie.GLFW.MakeContextCurrent(screen.glscreen)
    getfn(n) = GLMakie.GLFW.GetProcAddress(n)
    GL = GLMakie.ModernGL
    robj = screen.cache[objectid(player.previewplot)]
    old = robj.uniforms[:image]
    gp.origtex === nothing && (gp.origtex = old)   # re-setups see OUR texture here
    empty!(gp.gltex)
    for (fd, allocsize) in fds
        mo = Ref{UInt32}(0)
        ccall(getfn("glCreateMemoryObjectsEXT"), Cvoid, (Int32, Ptr{UInt32}), 1, mo)
        ccall(getfn("glMemoryObjectParameterivEXT"), Cvoid, (UInt32, UInt32, Ptr{Int32}),
              mo[], 0x9581, Ref{Int32}(1))                       # DEDICATED = TRUE
        ccall(getfn("glImportMemoryFdEXT"), Cvoid, (UInt32, UInt64, UInt32, Int32),
              mo[], UInt64(allocsize), 0x9586, Int32(fd))        # consumes the fd
        texid = Ref{UInt32}(0)
        GL.glGenTextures(1, texid)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texid[])
        GL.glTexParameteri(GL.GL_TEXTURE_2D, 0x9580, Int32(0x9584))  # OPTIMAL tiling
        ccall(getfn("glTexStorageMem2DEXT"), Cvoid,
              (UInt32, Int32, UInt32, Int32, Int32, UInt32, UInt64),
              GL.GL_TEXTURE_2D, 1, GL.GL_RGBA8, W, H, mo[], UInt64(0))
        # single mip level: the default MIN filter would leave the texture incomplete
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glGetError() == 0 || error("GL import of the shared texture failed")
        push!(gp.gltex, GLMakie.GLAbstraction.Texture{Makie.RGBAf, 2}(
            old.context, texid[], UInt32(GL.GL_TEXTURE_2D), UInt32(GL.GL_UNSIGNED_BYTE),
            UInt32(GL.GL_RGBA8), UInt32(GL.GL_RGBA), old.parameters, (Int(W), Int(H))))
    end
    gp.cur = 1
    robj.uniforms[:image] = gp.gltex[1]
    gp.width, gp.height = Int(W), Int(H)
    return nothing
end

# RGBA8 packing for the external image (bytes land as R,G,B,A little-endian)
packrgba(c::RGB{N0f8}) = UInt32(reinterpret(UInt8, c.r)) |
                         UInt32(reinterpret(UInt8, c.g)) << 8 |
                         UInt32(reinterpret(UInt8, c.b)) << 16 | 0xff000000

"""
Present a frame through the GPU chain: the source pixels are put into the working
buffer, motion/color tracks + the effect stack run as Lava kernels, and the result
is blitted into the shared image GLMakie samples. The source is a `GpuVideoStream`
when given — `frameat!` decodes the owning GOP device-resident and `nv12torgb!`
converts it in place (fully-GPU, disk→VRAM path, no CPU frame) — otherwise the
CPU frame `player.frame[]` is uploaded once. Returns `true` when on screen; `false`
(after flagging `failed`) hands the job back to the CPU path.
"""
function presentgpu!(player::Player, clip::Clip, srcframe::Integer; stream = nothing)
    gp = player.gpupreview
    try
        # source = the streaming decoder (disk→VRAM) or the CPU frame (one upload)
        source = stream === nothing ? player.frame[] : stream
        W, H = framesize(source)
        if (gp.width, gp.height) != (W, H)
            notify(player.frame)  # settle plot geometry for the new size first
            setupgpupreview!(player, gp, W, H)
        end
        nxt = gp.doublebuffer ? 3 - gp.cur : gp.cur   # blit target (see doublebuffer)
        rungpuowned(player, gp) do
            gp.engine === nothing && (gp.engine = FxEngine(player.analysisbackend))
            render(gp.engine, source, clip, Int(srcframe); applytracks = player.applytracks[],
                   playing = player.playing[]) do out
                gp.packed .= packrgba.(reshape(out, gp.width * gp.height))
                copyto!(gp.eimages[nxt], gp.packed)        # device blit + wait
                nothing
            end
        end
        # main thread again: swap the shown texture (the render loop is a sibling
        # task on this thread, so the swap can't interleave with a GL draw)
        player.screen.cache[objectid(player.previewplot)].uniforms[:image] = gp.gltex[nxt]
        gp.cur = nxt
        player.screen.requires_update = true
        return true
    catch e
        gp.failed = true
        setstatus!(player, "GPU preview failed — falling back to CPU preview")
        @error "GPU preview disabled" exception = (e, catch_backtrace())
        return false
    end
end

"""
Composite a stack of clips (bottom track → top) entirely on the GPU: each layer's
effect graph runs device-resident, its crop is baked with `warp!`, and layers are
alpha-blended by opacity — the multi-track analogue of [`presentgpu!`]. Every layer's
source must have a live `GpuVideoStream`; returns `false` otherwise so the caller
falls back to the CPU composite. The composited canvas is blitted to the shared image.
"""
function presentgpucomposite!(player::Player, clips::Vector{Clip}, n::Integer)
    gp = player.gpupreview
    all(haskey(player.gpucache, c.source) for c in clips) || return false
    try
        top = clips[end]
        W, H = top.source.width, top.source.height
        if (gp.width, gp.height) != (W, H)
            notify(player.frame)
            setupgpupreview!(player, gp, W, H)
        end
        nxt = gp.doublebuffer ? 3 - gp.cur : gp.cur   # same switch as presentgpu!
        rungpuowned(player, gp) do
            gp.engine === nothing && (gp.engine = FxEngine(player.analysisbackend))
            pool = gp.engine.pool
            accum = acquire!(pool, (W, H))
            first = true
            for clip in clips
                srcframe = clip.src_in + (n - clip.start)
                stream = player.gpucache[clip.source]
                ec = effectiveclip(clip, srcframe)
                # graph WITHOUT opacity — it becomes the layer alpha, not a fade to black
                gclip = Clip(ec.source, ec.src_in, ec.src_out, ec.start, ec.crop,
                             filter(e -> !(e isa OpacityEffect), ec.effects),
                             ec.colortrack, ec.motiontrack, ec.animations, ec.track)
                layer = execute!(graphof(gclip; applytracks = player.applytracks[]), pool,
                                 FxContext(stream, gclip, srcframe, player.playing[]))
                α = Float32(clamp(paramvalue(clip, :opacity, srcframe), 0.0, 1.0))
                if first
                    warp!(accum, layer, ec.crop)                  # bake crop into the canvas
                    α < 0.999f0 && channellinear!(accum, Vec3f(α), Vec3f(0))
                    first = false
                else
                    warpbuf = acquire!(pool, (W, H))
                    warp!(warpbuf, layer, ec.crop)
                    blend!(accum, accum, warpbuf, α)              # (1-α)·below + α·layer
                    release!(pool, warpbuf)
                end
                release!(pool, layer)
            end
            KA.synchronize(pool.backend)
            gp.packed .= packrgba.(reshape(accum, W * H))
            copyto!(gp.eimages[nxt], gp.packed)
            release!(pool, accum)
            nothing
        end
        player.screen.cache[objectid(player.previewplot)].uniforms[:image] = gp.gltex[nxt]
        gp.cur = nxt
        player.screen.requires_update = true
        return true
    catch e
        gp.failed = true
        @error "GPU composite failed — falling back" exception = (e, catch_backtrace())
        return false
    end
end

"""
Point the preview plot back at its OWN texture before a CPU `frame`-notify present
(scrub thumbnails, gap fills, the CPU fallback lane). GLMakie's notify-upload writes
into the texture the render object currently holds — uploading linear CPU pixels
into the imported optimal-tiled EXTERNAL texture shreds the on-screen image. The
next GPU present swaps the external texture back in. No-op on CPU-only players.
"""
function showcpuframe!(player::Player)
    gp = player.gpupreview
    (gp isa GPUPreview && gp.origtex !== nothing) || return nothing
    player.screen.cache[objectid(player.previewplot)].uniforms[:image] = gp.origtex
    player.screen.requires_update = true
    return nothing
end

"Frames kept VRAM-resident per source stream (a bounded ring; ~a few seconds)."
const GPU_STREAM_CAPACITY = 120

"""
Open a streaming GPU decoder ([`GpuVideoStream`]) over `source` so its clips play
back purely on the GPU — Vulkan-Video decode into a bounded VRAM ring, no CPU
decode or per-frame upload. A no-op (playback stays on the CPU decode path) unless
the GPU preview is live and the stream is hardware-decodable at the display size.
Cheap — demux + `mmap` + GOP index only; frames decode on demand. Call it off the
UI thread; playback uses the CPU path until the stream is ready.
"""
function preloadgpu!(player::Player, source::VideoSource)
    gp = player.gpupreview
    (gp isa GPUPreview && !gp.failed) || return nothing
    haskey(player.gpucache, source) && return nothing
    stream = nothing
    Threads.atomic_add!(player.gpubusy, 1)   # presents stay on the CPU tier during warmup
    try
        stream = openstream(player.analysisbackend, source.path, source.width, source.height;
                            capacity = GPU_STREAM_CAPACITY)
        # probe: decode GOP 0, confirm it's supported at the display size
        # (cold this also compiles the decode session + kernels)
        ok = rungpusync(player) do
            f = frameat!(stream, 0)
            size(f.y) == (source.width, source.height)
        end
        if ok
            player.gpucache[source] = stream
            setstatus!(player, "$(basename(source.path)) — streaming decode on the GPU")
        else
            close(stream)
        end
    catch e
        stream === nothing || (try; close(stream); catch; end)
        @warn "GPU stream unavailable; staying on CPU decode" exception = e
    finally
        Threads.atomic_sub!(player.gpubusy, 1)
    end
    return nothing
end

"""
Enable GPU playback if the device supports hardware video decode. The capability is
probed with `vk_context().video_decode_available` ON THE GPU WORKER, so the worker
creates and owns the (process-global, single-writer) Vulkan context — keeping async
analysis on the same thread. On success it attaches a [`GPUPreview`] and opens a
streaming decoder per source; otherwise it is a silent no-op and playback stays on
the CPU. Called in the background from the default `Player` constructor.
"""
function autodetectgpu!(player::Player)
    player.gpupreview isa GPUPreview && return nothing   # already enabled (explicit gpupreview)
    capable = try
        rungpusync(player) do
            Lava.vk_context().video_decode_available
        end
    catch
        false
    end
    capable || return nothing
    player.analysisbackend = LavaBackend()               # wraps the worker-owned context
    player.gpupreview = GPUPreview()
    setgpurun!(player.timeline, (f; long = false) -> rungpusync(f, player; long))  # GPU thumbnails from here on
    for src in unique(c.source for c in player.sequence.clips)
        Threads.@spawn preloadgpu!(player, src)
    end
    setstatus!(player, "GPU playback on — hardware decode + effects on the GPU")
    # re-present on the MAIN thread (GL context is main-thread-owned; this runs off it)
    player.playing[] || put!(player.uiqueue, () -> notify(player.playhead))
    return nothing
end

"Close every source's GPU stream (frees its VRAM ring + unmaps its bitstream) and the
effect graph's buffer pool."
function freegpucache!(player::Player)
    gp = player.gpupreview
    if gp isa GPUPreview && gp.engine !== nothing
        try; rungpusync(player) do; emptyengine!(gp.engine); end; catch; end
        gp.engine = nothing
    end
    isempty(player.gpucache) && return nothing
    for s in values(player.gpucache)
        try
            rungpusync(player) do; close(s); end
        catch
            try; close(s); catch; end
        end
    end
    empty!(player.gpucache)
    return nothing
end
