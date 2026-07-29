"""
Drive the editor's matte tool with SAM 2.1 running on Lava.

Opt-in glue, deliberately not a dependency in either direction, exactly like
`matanyone.jl`: VideoEditor owns the tool, the track and the render path and asks
a *segmenter* to turn clicks into a mask; LavaDNN knows nothing about editors.

    using VideoEditorRunner            # <- for a fast first click; see below
    include("examples/sam2.jl")
    usesam2!()                         # once, before marking a subject

After that the Matte tool turns each click into an actual object boundary
instead of a disc, and the status line reads "model" for the seed.

This is the other half of the matte, and the half that was missing. MatAnyone
*propagates* a mask; measured on both this implementation and PyTorch's, its
matte tracks whatever seed it is given (IoU against the seed 0.95 falling to 0.82
across a clip) rather than segmenting the subject. So the quality of the whole
matte was decided by a seed painted as discs around clicks. SAM 2.1 replaces that
seed; MatAnyone still carries it through the clip.

# Load `VideoEditorRunner` first, or the first click costs 42 seconds

The segmenter itself lives in `SAM2Runner`, not here, and this file is a dozen
lines because of a measurement rather than a preference:

    SAM2Runner alone                first click 1.50 s   (0.08 s compiling)
    with VideoEditor also loaded    first click 41.84 s  (41.04 s compiling)

Every kernel is frozen and every method is already in a package image in both
rows. **Loading VideoEditor invalidates SAM2Runner's cached inference**, and the
41 s is Julia re-deriving it. The fix is not to avoid the invalidation but to
cache on the other side of it: `VideoEditorRunner` depends on both packages and
runs `seedmask` in its own workload, so its image holds the inference that
survives having the editor loaded. Through it the same click costs **0.75 s**,
0.06 s of that compilation.

Which is also why the segmenter is not defined in this file any more. Code in a
script cannot be in a package image, so a copy here would be a copy that always
pays the 41 s — and a second copy of `maskatframe` to drift away from the first.

Two things the segmenter has to get right, both about interaction rather than
accuracy, and both handled in `SAM2Runner.sam2segmenter`:

  * **The embedding is cached per frame.** Embedding a 1024-square frame costs
    ~0.6 s; answering a click against a cached embedding costs ~0.1 s. Marking a
    subject is a dozen clicks on one frame, so the cache is the difference
    between a live preview and a progress bar. `seedmask` passes a `key`
    identifying the frame for exactly this.
  * **The model is built on first use, not at registration.** A Vulkan
    `BatchQueue` is single-writer and belongs to whichever thread first touches
    the context, and the editor calls a segmenter from `runanalysis` — its
    pinned GPU worker. Constructing at registration time binds the queue to
    whoever called `usesam2!` (the REPL) and every later call dies on
    "BatchQueue is single-writer; cross-thread sweep forbidden". Same reasoning,
    same fix, as `matanyone.jl`.
"""

using VideoEditor, SAM2Runner, KernelAbstractions
using Lava: LavaBackend

"""
    usesam2!(; backend = LavaBackend(), dir = SAM2Runner.assetdir(), pick = :confident)

Install the SAM 2.1 segmenter into the editor's matte tool.

`pick` chooses among SAM's three proposals per click. The default `:confident`
takes the highest predicted IoU but breaks a near-tie by logit magnitude, which
on measured clicks cuts the seed's boundary fragmentation from 13.6x a compact
blob to 3.6x; `:best` is SAM's own argmax and is what PyTorch does.
`LavaDNN.segment` carries the measurement. A click is genuinely ambiguous (a
windowpane, the window, the wall) and the model says so by returning all three,
so this is the one place a UI control would belong later.

The model is constructed on the first click rather than here — see the module
docstring for why that is a threading requirement and not laziness.
"""
function usesam2!(; backend = LavaBackend(), dir::AbstractString = SAM2Runner.assetdir(),
                  pick = :confident)
    isdir(dir) || error("no SAM 2 graphs at $dir — run tools/export_sam2.py")
    isfile(joinpath(dir, "weights.safetensors")) || error("no weights in $dir")
    segref = Ref{Any}(nothing)
    VideoEditor.registersegmenter!(function (frame, points; key = nothing)
        segref[] === nothing &&
            (segref[] = SAM2Runner.sam2segmenter(SAM2Runner.sam2model(; backend, dir); pick))
        return segref[](frame, points; key)
    end)
    isdefined(Main, :VideoEditorRunner) ||
        @warn "load VideoEditorRunner for a fast first click — without it it costs ~42 s of Julia compilation"
    @info "matte: SAM 2.1 segmenter installed — clicks now produce object masks"
    return nothing
end
