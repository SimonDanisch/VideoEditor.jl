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

"Per-resolution GPU presentation state (see `presentgpu!`)."
mutable struct GPUPreview
    width::Int
    height::Int
    failed::Bool
    inline::Bool   # Lava context is owned by the main thread → run jobs inline
    # worker-owned (Lava)
    gpuframe::Any   # LavaArray{RGB{N0f8},2}
    gputmp1::Any
    gputmp2::Any
    packed::Any     # LavaArray{UInt32,1}
    eimage::Any     # Lava.ExternalImage
    # main-thread-owned (GL)
    texid::UInt32
end
GPUPreview() = GPUPreview(0, 0, false, false, nothing, nothing, nothing, nothing, nothing, UInt32(0))

"Run `f` on the player's pinned GPU worker and wait for its result."
function rungpusync(f::Function, player::Player)
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
    fd, allocsize = rungpuowned(player, gp) do
        backend = player.analysisbackend
        gp.gpuframe = KA.allocate(backend, RGB{N0f8}, (Int(W), Int(H)))
        gp.gputmp1 = KA.allocate(backend, RGB{N0f8}, (Int(W), Int(H)))
        gp.gputmp2 = KA.allocate(backend, RGB{N0f8}, (Int(W), Int(H)))
        gp.packed = KA.allocate(backend, UInt32, Int(W) * Int(H))
        gp.eimage = lavamod.ExternalImage(W, H)
        (lavamod.memoryfd(gp.eimage), gp.eimage.allocation_size)
    end

    # GL side: import the fd and swap the preview plot's texture (main thread)
    screen = player.screen
    GLMakie.GLFW.MakeContextCurrent(screen.glscreen)
    getfn(n) = GLMakie.GLFW.GetProcAddress(n)
    GL = GLMakie.ModernGL
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

    robj = screen.cache[objectid(player.previewplot)]
    old = robj.uniforms[:image]
    robj.uniforms[:image] = GLMakie.GLAbstraction.Texture{Makie.RGBAf, 2}(
        old.context, texid[], UInt32(GL.GL_TEXTURE_2D), UInt32(GL.GL_UNSIGNED_BYTE),
        UInt32(GL.GL_RGBA8), UInt32(GL.GL_RGBA), old.parameters, (Int(W), Int(H)))
    gp.texid = texid[]
    gp.width, gp.height = Int(W), Int(H)
    return nothing
end

# RGBA8 packing for the external image (bytes land as R,G,B,A little-endian)
packrgba(c::RGB{N0f8}) = UInt32(reinterpret(UInt8, c.r)) |
                         UInt32(reinterpret(UInt8, c.g)) << 8 |
                         UInt32(reinterpret(UInt8, c.b)) << 16 | 0xff000000

"""
Present the already-fetched CPU frame through the GPU chain. Returns `true`
when the frame is on screen; `false` (after flagging `failed`) hands the
job back to the CPU path.
"""
function presentgpu!(player::Player, clip::Clip, srcframe::Integer)
    gp = player.gpupreview
    try
        W, H = size(player.frame[])
        if (gp.width, gp.height) != (W, H)
            notify(player.frame)  # settle plot geometry for the new size first
            setupgpupreview!(player, gp, W, H)
        end
        applytracks = player.applytracks[]
        rungpuowned(player, gp) do
            copyto!(gp.gpuframe, player.frame[])           # the one PCIe upload
            if applytracks
                applymotiontrack!(gp.gpuframe, gp.gputmp1, clip, srcframe)
                applycolortrack!(gp.gpuframe, clip, srcframe)
            end
            applyeffects!(gp.gpuframe, gp.gputmp1, gp.gputmp2, clip)
            gp.packed .= packrgba.(reshape(gp.gpuframe, gp.width * gp.height))
            copyto!(gp.eimage, gp.packed)                  # device blit + wait
            nothing
        end
        player.screen.requires_update = true
        return true
    catch e
        gp.failed = true
        setstatus!(player, "GPU preview failed — falling back to CPU preview")
        @error "GPU preview disabled" exception = (e, catch_backtrace())
        return false
    end
end
