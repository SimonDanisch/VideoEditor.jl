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

The clip the workload renders, out of this package's own artifact. Any short clip
will do; what is being frozen is the code path, not the content.

Neither an environment variable nor a walk up the filesystem, which is what was
here — and not a reach into `DNNKernels` for either, which is what it used. Both
mechanisms are the ones `DNNKernels/src/assets.jl` records as deleted, for the
reason it gives: a walk answers on the machine that happens to have the tree and
nowhere else, so this froze the editor's kernels for one checkout and silently
covered nothing for every other. A package owns its assets through its own
`Artifacts.toml`, exactly as each model runner owns its weights, and nothing
outside it constructs a path into them.

`""` until `videoeditor-demo` is bound, and the workload skips — which is what it
already did on every machine without the media tree, now said out loud instead of
looking like coverage. Binding it is `create_artifact` over a short clip and
`bind_artifact!` with the release URL; `dev/JuliaVision/tools/make_artifacts.jl`
documents that flow and why the upload is a separate, deliberate step.
"""
function editorassets()
    toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
    isfile(toml) || return ""
    meta = Artifacts.artifact_meta("videoeditor-demo", toml)
    meta === nothing && return ""
    # Resolved, not installed: a lazy artifact nobody has fetched leaves the
    # workload with nothing to render, and downloading a video inside someone's
    # precompilation is not this file's decision to make.
    dir = Artifacts.artifact_path(Base.SHA1(meta["git-tree-sha1"]))
    isdir(dir) || return ""
    clip = joinpath(dir, "demo.mp4")
    isfile(clip) ? clip : ""
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
            markframe = framereader(nothing, clip)(1)
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
                # …and the disc fallback, which is what a Player without SAM 2
                # weights runs (`segmenter = nothing`, not a global left unset)
                seedmask(clip, markframe, marks; segmenter = nothing, key = (clip.id, 1))

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
                    registermatte!(prop)
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
