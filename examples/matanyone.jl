"""
Drive the editor's matte tool with MatAnyone running on Lava.

Opt-in glue, deliberately not a dependency in either direction: VideoEditor owns
the track, the UI and the render path and asks a propagator to fill in the
frames; LavaDNN knows nothing about editors. Including this file is what joins
them.

    include("examples/matanyone.jl")
    usematanyone!(; backend = LavaBackend())     # once, before marking a subject

After that the Matte tool's status line reads "model" instead of "built-in" and
every propagation runs the network.

Two shape rules the model imposes, both handled here rather than pushed onto the
caller:

  * frames are padded up to a multiple of 16 (the encoder downsamples by 16 and
    `initstate` sizes the memory bank from `W ÷ 16`), and the alpha is cropped
    back, so any matte resolution works;
  * propagation is forward-only and stateful, so the frames are fed in order and
    a marked frame re-injects its mask via `step!(...; mask = ...)` — the first
    mark also resets the memory bank with `firstframe = true`.

The seed mask is **0..255**, not 0..1. A normalized mask is 255x too faint and
produces an all-zero matte with no error anywhere — no exception, correct shape,
empty result. The image operand *is* 0..1, so the two use different scales.
"""

using VideoEditor, MatAnyoneRunner
using Lava: LavaBackend

"""
    usematanyone!(; backend = LavaBackend(), graphdir, weights, warmup = 10)

Install the MatAnyone propagator into the editor's matte tool.

# Load `VideoEditorRunner` first, or the first propagation costs 94 seconds

The propagator lives in `MatAnyoneRunner`, not in this file, and this file is a
dozen lines because of a measurement rather than a preference. Through the
editor's seam, with every kernel already frozen:

    first propagation (6 frames)   94.30 s   (74.44 s compiling)
    second propagation              9.20 s   ( 0.00 s compiling)

Code in a script cannot be in a package image, so a propagator defined here is
one that always pays the 74 s. And moving it into a package is only half: loading
`VideoEditor` invalidates the inference a model package cached, so what actually
has to run the workload is `VideoEditorRunner`, which depends on both and caches
on the far side of that. Same story as `sam2.jl`, same fix, and its docstring has
the numbers for the segmenter half.
"""
function usematanyone!(; backend = LavaBackend(), kwargs...)
    propref = Ref{Any}(nothing)
    # Built on FIRST USE, not here: a Vulkan `BatchQueue` is single-writer and
    # belongs to whichever thread first touches the context, while the editor
    # calls a propagator from `runanalysis` — its pinned GPU worker. Building at
    # registration binds the queue to whoever called `usematanyone!` (the REPL)
    # and every later call dies on "cross-thread sweep forbidden".
    VideoEditor.registermatte!(function (frames, seeds; progress = nothing)
        propref[] === nothing &&
            (propref[] = MatAnyoneRunner.matanyonepropagator(; backend, kwargs...))
        return propref[](frames, seeds; progress)
    end)
    isdefined(Main, :VideoEditorRunner) ||
        @warn "load VideoEditorRunner for a fast first matte — without it it costs ~74 s of Julia compilation"
    @info "matte: MatAnyone propagator installed"
    return nothing
end
