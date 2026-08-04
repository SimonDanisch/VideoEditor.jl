"""
The editor's kernels, frozen.

Same mechanism as `SAM2Runner` and `MatAnyoneRunner`, pointed at the editor
instead of a network. The editor compiles a lot of kernels — decode, the effect
graph, matte compositing, thumbnails, transitions — and it compiles them at the
worst possible moment, which is the first time the user asks for a picture.

The workload drives [`renderframe!`](@ref), deliberately: since the render
unification that is the *one* orchestration behind preview, scrub and export, so
freezing its kernels covers all three. Around it the workload adds the effects
whose kernels would otherwise compile the first time somebody drags a slider.

This was a separate package (`VideoEditorRunner`) downstream of the editor, and
the reason was invalidation: loading `VideoEditor` invalidates the code
`SAM2Runner` precompiled, so a workload sitting *inside* a model package is
thrown away, and the caching has to happen on the far side of that
invalidation. `VideoEditor` depends on both model packages directly, which makes
it itself the far side — the workload lives here and the wrapper package is
gone. It is `include`d last for the other half of that rule: a workload has to
come after everything it calls is defined.

Sharing `DNNKernels.KERNELS_VERSION` with the networks is the point rather than an
accident: the editor and the models both broadcast over `LavaArray`s, and one
frozen entry serves both.
"""

# Only the workload below needs the matte propagator — the editor itself takes
# whatever `registermatte!` was handed (see matte.jl).
import MatAnyoneRunner
using Lava: @setup_workload, @compile_workload

"""`DNNKernels.KERNELS_VERSION` — one generation for the whole runtime. Read
through `SAM2Runner` (already a dependency, and it defines the constant as
exactly that) so the editor shares the generation without depending on the
kernel package directly."""
const KERNELS_VERSION = SAM2Runner.KERNELS_VERSION

"""
    editorassets() -> String

A video to render during the workload. Any short clip will do; what is being
frozen is the code path, not the content.
"""
function editorassets()
    p = get(ENV, "JULIA_VIDEOEDITOR_ASSET", "")
    isempty(p) || return p
    # Walked up from here rather than a fixed `../../..`: the depth of the media
    # directory above this package is a property of the checkout, not of the
    # package, and encoding it here is what broke the model runners when they
    # moved into a monorepo.
    #
    # This is the editor's *own* walk because `DNNKernels.findasset` was deleted
    # — a model's assets come from its artifact and from nowhere else, and a
    # filesystem search was the fallback that kept a broken download invisible.
    # A demo clip is not a model asset: it is optional, the workload skips when
    # it is absent, and there is nothing to download.
    for c in ("media/demo_source.mp4", "media/birds_export.mp4", "media/demo_loop.mp4")
        dir = @__DIR__
        for _ in 1:6
            f = joinpath(dir, c)
            isfile(f) && return f
            parent = dirname(dir)
            parent == dir && break
            dir = parent
        end
    end
    return ""
end

"""
    runeditorframe(seq, engine, readers, dest, n)

One frame through the editor's render graph — the same call preview, scrub and
export all make.
"""
function runeditorframe(seq, engine, readers, dest, n::Integer)
    renderframe!(dest, seq, n, readers, engine)
    return dest
end

# The editor's only `__init__`, and this is what it exists for: point Lava at the
# frozen cache the workload below fills, before anything asks for a kernel.
function __init__()
    Lava.use_frozen_kernels(KERNELS_VERSION)
    # Install the matte propagator HERE, and never from the workload below. The
    # workload's closure has already run, so its `modelref` holds a built model —
    # and a built model holds `LavaArray`s whose `VkContext` belongs to the
    # *precompilation* process. `registermatte!` writes a module global, so that
    # gets serialised into the package image and every matte call at runtime then
    # drives a dead device: `sync_access!` sees a `BatchQueue` from another
    # context and refuses, and before that guard existed it was a segfault inside
    # vkQueueSubmit2 during `warmmatte!`. `matanyonepropagator` builds its model
    # on first use, so registering a fresh one costs nothing at load.
    MATTEPROPAGATOR[] = nothing     # drop anything an older image baked in
    # `assetdir()` can reach `ensure_artifact_installed`, so this could in
    # principle download at load. It does not in practice: the workload below
    # resolves the same asset during precompilation, and `sam2ready()` does the
    # equivalent at every `Player` construction — by the time this runs the
    # artifact is on disk and the call is a TOML read. The `catch` is what makes
    # the remaining case (no assets, no network) a no-op rather than a package
    # that will not load.
    try
        if isdir(MatAnyoneRunner.assetdir()) && isfile(MatAnyoneRunner.weightpath())
            registermatte!(MatAnyoneRunner.matanyonepropagator())
        end
    catch err
        @debug "VideoEditor: no matte propagator registered" exception = err
    end
    return nothing
end

@setup_workload begin
    asset = editorassets()
    if isfile(asset)
        try
            backend = LavaBackend()
            src = VideoSource(asset)
            seq = Sequence(src)
            engine = FxEngine(backend)
            readers = Dict{String, Any}()
            dest = RGBFrame(undef, src.width, src.height)
            clip = seq.clips[1]

            # The matte tool's segmenter path. Covered here rather than in
            # `SAM2Runner` because it is the *editor* side that is expensive:
            # with every kernel already frozen, the first click still cost 41 s,
            # 97% of it Julia inferring `seedmask` and the segmenter it calls.
            samready = isfile(joinpath(SAM2Runner.assetdir(), "weights.safetensors"))
            matready = isdir(MatAnyoneRunner.assetdir()) && isfile(MatAnyoneRunner.weightpath())
            markframe = framereader(clip, engine)(1)
            marks = [(0.5, 0.5, true), (0.2, 0.2, false)]

            @compile_workload KERNELS_VERSION begin
                # Building the model is INSIDE the traced block, and only here:
                # in `@setup_workload` its inference is not captured, and reading
                # the graph plus 900 MB of weights is ~3 s of Julia on the first
                # click. One model, not two — a second would be 900 MB of VRAM
                # during precompilation for nothing.
                segmodel = samready ? SAM2Runner.sam2model(; backend) : nothing
                seg = segmodel === nothing ? nothing : SAM2Runner.sam2segmenter(segmodel)
                # Plain: decode + canvas warp, the common case.
                runeditorframe(seq, engine, readers, dest, 1)
                # With the effects a user reaches for first. Each is its own
                # kernel, and each would otherwise compile on the first drag.
                seteffect!(clip, ColorEffect(saturation = 1.4))
                runeditorframe(seq, engine, readers, dest, 2)
                seteffect!(clip, BlurEffect(2.0f0))
                runeditorframe(seq, engine, readers, dest, 3)
                seteffect!(clip, OpacityEffect(0.7f0))
                runeditorframe(seq, engine, readers, dest, 4)
                # …and the matte seam, both with a model and without it, since
                # the editor runs the disc fallback whenever none is installed.
                if seg !== nothing
                    seedmask(clip, markframe, marks; segmenter = seg, key = (clip.id, 1))
                    seedmask(clip, markframe, marks; segmenter = seg, key = (clip.id, 1))  # cached
                end
                # …and the disc seed, which is the rect/box path, not a fallback
                seedmask(clip, [(0.5, 0.5, true), (0.2, 0.2, false)])

                # …and the OTHER half of the matte: propagation. Measured at
                # 94.3 s on the first run, 74.4 s of it Julia, and it has to be
                # traced here rather than in `MatAnyoneRunner` for the same
                # reason the segmenter is — loading `VideoEditor` invalidates
                # what a model package cached. Two frames and `warmup = 1`: the
                # shapes and the frame count do not change which methods get
                # inferred, and a 10-step warmup would only make precompilation
                # slower.
                if matready
                    prop = MatAnyoneRunner.matanyonepropagator(; backend, warmup = 1)
                    seed = zeros(UInt8, size(markframe)...)
                    seed[(size(markframe,1)÷3):(2size(markframe,1)÷3),
                         (size(markframe,2)÷3):(2size(markframe,2)÷3)] .= 0xff
                    prop([markframe, markframe], Dict(1 => seed))
                    # Deliberately NOT `registermatte!(prop)`: running `prop`
                    # built its model, and installing it would serialise those
                    # device buffers — and the context they belong to — into the
                    # package image. `__init__` registers a fresh lazy one.
                end
                KA.synchronize(backend)
            end

            foreach(close, values(readers))
            emptyengine!(engine)
        catch err
            @warn "VideoEditor: precompile workload skipped; first render will compile" exception = err
        end
    else
        @info "VideoEditor: no asset video — nothing precompiled"
    end
end
