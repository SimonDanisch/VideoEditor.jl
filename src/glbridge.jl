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
# Error policy: a GPU render error is a BUG, not a mode. Playback pauses
# LOUDLY (status + logged backtrace) and the render backend stays what the
# config declared — no silent CPU continuation, no error-driven switching.
# There is no CPU tier. A present during a long job queues behind it on the
# single-writer worker; it does not take a second path.

"Per-resolution GPU presentation state (see `presentgpu!`). The shared image is
DOUBLE-BUFFERED: GLMakie's render loop runs as a concurrent task, so blitting
into the texture it is currently sampling tears — visible as a glitched frame
whenever consecutive frames differ strongly (a stabilization warp!). Each
present blits into the texture GLMakie is NOT showing, then swaps."
mutable struct GPUPreview
    width::Int
    height::Int
    # worker-owned (Lava)
    packed::Any          # LavaArray{UInt32,1} — RGBA pack scratch for the blit
    eimages::Vector{Any} # 2 Lava.ExternalImage back/front buffers
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
GPUPreview() = GPUPreview(0, 0, nothing, Any[], Any[], 1, true, nothing)

"""
A GPU render error is a BUG, not a mode: playback pauses LOUDLY (status + log)
and the render backend stays what the config declared — no silent CPU
continuation, no error-driven lane switching. The frame that errored still
presents through the CPU tier so the screen isn't stale.
"""
function gpurendererror!(player::Player, err)
    pause!(player)
    setstatus!(player, "GPU render ERROR — playback paused (details in the log)")
    @error "GPU render failed" exception = (err, catch_backtrace())
    return nothing
end

"""
    previewrobj(player) -> RenderObject | nothing

The preview image's GL render object, or `nothing` when there is none to talk to.

A CLOSED window keeps its plots but empties its render cache, so the lookup that
swaps the preview texture threw a bare `KeyError` — which surfaced as "GPU render
ERROR, playback paused" for what is simply a window that is gone. Presenting into
a closed screen is a no-op, not a failure.
"""
function previewrobj(player::Player)
    scr = player.screen
    (scr === nothing || !isopen(scr)) && return nothing
    return get(scr.cache, objectid(player.previewplot), nothing)
end

"""
    gpupreviewlive(player) -> Bool

Whether the Vulkan images are actually IMPORTED as GL textures — i.e. whether a
present can reach the screen through the GPU tier at all.

**Not the same question as [`gpuready`](@ref)**, which says the decode and effect
side is device-resident. The import additionally needs the GL context and the
Vulkan device to be the same card. Under a software GL — Xvfb with `llvmpipe`,
which is what a headless test run gets — an NVIDIA memory handle cannot be
imported however many extensions Mesa advertises, `setupgpupreview!` throws on
every attempt, and every present falls back to the CPU tier while `gpuready`
stays true throughout.

That combination is why three preview tests in `test_gpu.jl` read as a playback
regression and are nothing of the kind. Anything asking "will the GPU lane put a
picture on the screen" — a test, or a UI deciding what to label the lane — has to
ask this and not `gpuready`.
"""
gpupreviewlive(player::Player) =
    (gp = player.gpupreview;
     gp !== nothing && !isempty(gp.gltex) && previewrobj(player) !== nothing)

"""
Run `f` on a pinned GPU worker and wait for its result — given either the worker
itself or the `Player` that owns one.

Exceptions are carried back and rethrown here rather than escaping on the worker's
task, where nothing would see them.
"""
function rungpusync(f::Function, w::GPUWorker)
    done = Channel{Any}(1)
    rungpu(w) do
        try
            put!(done, (true, f()))
        catch e
            put!(done, (false, e))
        end
    end
    ok, val = take!(done)
    ok || throw(val)
    return val
end

function rungpusync(f::Function, player::Player)
    player.gpuworker === nothing && (player.gpuworker = GPUWorker())
    return rungpusync(f, player.gpuworker)
end

"""
Run `f` on whichever thread owns the render engine's context.

Every call into `player.engine` goes through this, presentation and card previews
alike: a Lava `BatchQueue` binds to the thread that first touched it, and which
thread that was depends on what ran first — the pinned worker (analysis, the
usual case) or main (Lava already used before this player existed). So the owner
is *discovered*, once, from the worker's own assertion, and never configured.
"""
function runowned(f::Function, player::Player)
    player.engineinline && return f()
    # already ON the worker (analysis jobs render too): posting to the queue only
    # this task drains would wait for itself
    w = player.gpuworker
    w === nothing || current_task() !== w.task || return f()
    try
        return rungpusync(f, player)
    catch e
        if e isa AssertionError && occursin("single-writer", e.msg)
            player.engineinline = true
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
    fds = runowned(player) do
        backend = player.analysisbackend
        emptyengine!(player.engine)        # drop old-resolution graph buffers
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
    robj = previewrobj(player)
    robj === nothing && error("the preview window is gone — cannot set up the GPU chain")
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
function presentgpu!(player::Player, clip::Clip, srcframe::Integer;
                     stream = nothing, source = nothing)
    gp = player.gpupreview
    try
        # source = the streaming decoder (disk→VRAM) or a decoded CPU frame (one upload)
        src = stream !== nothing ? stream : source !== nothing ? source : player.frame[]
        # the CANVAS, not the layer: a clip's crop, reframe and rotation are baked
        # in by `placelayer!` here exactly as the export bakes them
        W, H = canvassize(player.sequence)
        if (gp.width, gp.height) != (W, H)
            notify(player.frame)  # settle plot geometry for the new size first
            setupgpupreview!(player, gp, W, H)
        end
        nxt = gp.doublebuffer ? 3 - gp.cur : gp.cur   # blit target (see doublebuffer)
        ok = runowned(player) do
            composite(player.engine, [clip], n_of(player, clip, srcframe), (_, _) -> src;
                      canvas = (W, H), applytracks = player.applytracks[],
                      playing = player.playing[]) do canvas
                gp.packed .= packrgba.(reshape(canvas, W * H))
                copyto!(gp.eimages[nxt], gp.packed)        # device blit + wait
                nothing
            end
        end
        ok === true || return false
        # main thread again: swap the shown texture (the render loop is a sibling
        # task on this thread, so the swap can't interleave with a GL draw)
        robj = previewrobj(player)
        robj === nothing && return false     # the window is gone: nothing to show it on
        robj.uniforms[:image] = gp.gltex[nxt]
        gp.cur = nxt
        player.screen.requires_update = true
        return true
    catch e
        gpurendererror!(player, e)
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
    all(haskey(player.gpucache, readerkey(player.sequence, c)) for c in clips) || return false
    try
        W, H = canvassize(player.sequence)   # the SEQUENCE's format, not the top layer's
        if (gp.width, gp.height) != (W, H)
            notify(player.frame)
            setupgpupreview!(player, gp, W, H)
        end
        nxt = gp.doublebuffer ? 3 - gp.cur : gp.cur   # same switch as presentgpu!
        # the GPU tier of ONE composite (see `composite`): its only job is to
        # name each layer's stream and to blit the finished canvas
        ok = runowned(player) do
            composite(player.engine, clips, n,
                      (clip, _) -> player.gpucache[readerkey(player.sequence, clip)];
                      canvas = (W, H), applytracks = player.applytracks[],
                      playing = player.playing[]) do canvas
                gp.packed .= packrgba.(reshape(canvas, W * H))
                copyto!(gp.eimages[nxt], gp.packed)
            end
        end
        ok === true || return false
        robj = previewrobj(player)
        robj === nothing && return false     # the window is gone: nothing to show it on
        robj.uniforms[:image] = gp.gltex[nxt]
        gp.cur = nxt
        player.screen.requires_update = true
        return true
    catch e
        gpurendererror!(player, e)
        return false
    end
end

"""
Advance `stream`'s decode toward frame `n` WITHOUT rendering or presenting: the
settle path (a parked playhead) refines a cold seek across a few of these calls
while the last exact image stays on screen. Returns `false` when the decode
errored — the caller then falls through to the CPU lane, which decodes it exactly.
"""
function primeframe!(player::Player, stream::GpuVideoStream, n::Integer)
    gp = player.gpupreview
    try
        runowned(player) do
            frameat!(stream, n)
            nothing
        end
        return true
    catch e
        gpurendererror!(player, e)
        return false
    end
end

"""
Advance every layer of a multi-track composite toward timeline frame `n`; `true`
once they are ALL exactly decoded. A composite blends the layers into one image,
so a single stand-in among them dates the whole frame.
"""
function primecomposite!(player::Player, clips::Vector{Clip}, n::Integer)
    ready = true
    for clip in clips
        stream = player.gpucache[readerkey(player.sequence, clip)]
        srcframe = sourceframe(clip, n)
        hasframe(stream, srcframe) && continue
        # a failed decode is loud and hands the frame to the CPU composite —
        # waiting for exactness on a stream that just errored is pointless
        primeframe!(player, stream, srcframe) || return true
        ready = false
    end
    return ready
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
    robj = previewrobj(player)
    robj === nothing && return nothing
    robj.uniforms[:image] = gp.origtex
    player.screen.requires_update = true
    return nothing
end

"Frames kept VRAM-resident per source stream (a bounded ring; ~a few seconds)."
const GPU_STREAM_CAPACITY = 120

"""
Open a stream for one CLIP under its [`readerkey`](@ref) — its source when clips
can share a read head, its own id when they cannot. Two clips of one file that
overlap further apart than the ring each get a stream, so neither has to seek.
"""
preloadgpu!(player::Player, clip::Clip) =
    preloadgpu!(player, clip.source; key = readerkey(player.sequence, clip))

"""
Open a streaming GPU decoder ([`GpuVideoStream`]) over `source` so its clips play
back purely on the GPU — Vulkan-Video decode into a bounded VRAM ring, no CPU
decode or per-frame upload. A no-op (playback stays on the CPU decode path) unless
the GPU preview is live and the stream is hardware-decodable at the display size.
Cheap — demux + `mmap` + GOP index only; frames decode on demand. Call it off the
UI thread; playback uses the CPU path until the stream is ready.
"""
function preloadgpu!(player::Player, source::VideoSource; key = source)
    gp = player.gpupreview
    gp isa GPUPreview || return nothing
    haskey(player.gpucache, key) && return nothing
    # One probe per source, not one per present. `ensurestreams!` runs on every
    # frame that is shown, so a source that is not directly streamable used to be
    # re-probed forever: each attempt failed, logged, and overwrote the
    # mezzanine's own status line. The retry that matters is the
    # one `startmezzanine!` makes itself once the transcode has landed.
    # ONE probe per source per session. The retry that matters is the single
    # explicit one `startmezzanine!` makes when its transcode lands; everything
    # else is `ensurestreams!` running per present, and per analysis, and per
    # matte click. The old escape hatch ("…unless a mezzanine file exists") made
    # that memo useless for exactly the sources that need it, and the transcode's
    # own retry then re-entered this on failure — transcode → probe → fail →
    # transcode, two sources interleaving, forever, at one warning per turn.
    jobs = get!(() -> Set{String}(), player.fxwidgets, :mezzjobs)
    source.path in jobs && return nothing        # a transcode is running; it retries itself
    probed = get!(() -> Set{String}(), player.fxwidgets, :gpuprobed)
    source.path in probed && return nothing
    push!(probed, source.path)
    stream = nothing
    # a mezzanine transcoded earlier for this source IS the streamable version —
    # a later run opens on it directly instead of failing the probe again
    mezz = mezzaninepath(source)
    path = isfile(mezz) ? mezz : source.path
    try
        stream = openstream(player.analysisbackend, path, source.width, source.height;
                            capacity = GPU_STREAM_CAPACITY)
        # probe: decode GOP 0, confirm it's supported at the display size
        # (cold this also compiles the decode session + kernels)
        ok = rungpusync(player) do
            f = frameat!(stream, 0)
            size(f.y) == (source.width, source.height)
        end
        if ok
            player.gpucache[key] = stream
            delete!(probed, source.path)
            setstatus!(player, "$(basename(source.path)) — streaming decode on the GPU")
        else
            close(stream)
        end
    catch e
        stream === nothing || (try; close(stream); catch; end)
        # silent degradation hid a whole codec gap once — say it where the user
        # looks, and where the source is merely not DIRECTLY streamable (codec,
        # open GOPs, profile) start the one-time mezzanine transcode instead of
        # settling for the CPU tier
        msg = sprint(showerror, e)
        if occursin("VRAM budget", msg)
            setstatus!(player, "$(basename(source.path)): CPU decode — $msg")
        elseif isfile(mezzaninepath(source))
            # the mezzanine is already there and STILL will not open: transcoding
            # it again cannot change that, so this is terminal for the GPU tier
            setstatus!(player, "$(basename(source.path)): CPU decode — its mezzanine " *
                               "will not open on the GPU either")
        else
            setstatus!(player, "$(basename(source.path)): not GPU-streamable, transcoding once")
            startmezzanine!(player, source)
        end
        @warn "GPU stream unavailable; staying on CPU decode" exception = e
    finally
    end
    return nothing
end

"""
Transcode `source` into the editing mezzanine in the background (footer
progress) and, when done, open the GPU stream ON the mezzanine file — keyed by
the ORIGINAL source, so playback flips from the CPU tier to pure-GPU streaming
transparently. Export keeps reading the original (no generation loss).
[`preloadgpu!`](@ref) picks the finished mezzanine up from the cache path by
itself, so the open lives in ONE place.
"""
function startmezzanine!(player::Player, source::VideoSource)
    jobs = get!(() -> Set{String}(), player.fxwidgets, :mezzjobs)
    source.path in jobs && return nothing
    push!(jobs, source.path)
    Threads.@spawn try
        setstatus!(player, "$(basename(source.path)): preparing editing mezzanine…")
        player.jobprogress[] = 0.0
        generatemezzanine(source;
            progress = (d, t) -> (player.jobprogress[] = d / max(t, 1)))
        player.jobprogress[] = NaN
        gp = player.gpupreview
        gp isa GPUPreview || return nothing
        delete!(jobs, source.path)          # the file is there now; retries may re-run
        delete!(get!(() -> Set{String}(), player.fxwidgets, :gpuprobed), source.path)
        preloadgpu!(player, source)         # opens ON the mezzanine (cache path)
        player.playing[] || put!(player.uiqueue, () -> notify(player.playhead))
    catch e
        player.jobprogress[] = NaN
        setstatus!(player, "mezzanine transcode failed: $(sprint(showerror, e))")
        @error "mezzanine transcode failed" exception = (e, catch_backtrace())
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
    emptyengine!(player.engine)                          # its plans hold pool regions now
    player.engine = FxEngine(player.analysisbackend)     # the engine follows the backend
    player.gpupreview = GPUPreview()
    haskey(player.fxwidgets, :lanechip) && (player.fxwidgets[:lanechip][] = "GPU")
    setgpurun!(player.timeline, f -> rungpusync(f, player))   # GPU thumbnails from here on
    for clip in player.sequence.clips
        Threads.@spawn preloadgpu!(player, clip)
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
    if gp isa GPUPreview
        try; rungpusync(player) do; emptyengine!(player.engine); end; catch; end
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
