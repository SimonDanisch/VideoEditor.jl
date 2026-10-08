# Walkthrough regression tests

`test/editing_walkthrough.jl` uses Makie's `docs/fake_interaction.jl`, the same
`FakeInteraction` actions as the recorded editor examples. Gestures target widget
bounding boxes and timeline coordinates. Assertions read the resulting document
and scene; they do not set the parameters that the gesture is supposed to edit.

Run this on a machine where GPU/window tests will not disturb interactive work.
In CrawCity, the user reserved the local PC again on 2026-10-05. Use **Bosgame**
for Julia evaluations and window tests until the user releases the local PC.
Its prepared ordinary Julia environment is
`/home/sim/Programmieren/VideoEditor-renderfarm-20261004/demo/ui_env`.

In that persistent worker session:

```julia
using VideoEditor
ENV["MAKIE_FAKE_INTERACTION"] = "/path/to/Makie/docs/fake_interaction.jl"
ENV["VIDEOEDITOR_QA_OUTPUT"] = "/path/to/qa/results"
include(joinpath(pkgdir(VideoEditor), "test", "editing_walkthrough.jl"))
```

The helper defaults to the sibling Makie checkout used by the examples. It is
documentation code, so an installed Makie package need not contain it. Set the
environment variable to an existing copy when using a prepared worker runtime.
The GUI test group includes these tests when the helper is available and reports
a skipped test otherwise. The standalone file requires it explicitly.

`test/recordedanimation.jl` verifies streamed scalar/array/complex samples, mutable
simulation buffers, exact seeking, variable shapes, immutable sample views,
save/reopen and rebased farm bundles. Register RayMakie first to include the GPU
scene test: a stateful updater runs once during recording, then raster previews
and traced frames restore its samples in arbitrary order without calling it again.
`test/recordedanimation_walkthrough.jl` uses the same mouse/keyboard helpers for
object selection, array lanes, restoring recorded data, numeric edits, Undo,
backward seeking, cuts and save/reopen. After including the core test with
`ENV["VIDEOEDITOR_RECORDING_OUTPUT"]` set, call
`RecordedAnimationWalkthrough.run(LAST_RECORDED_PROJECT[], output_directory)`.
It saves an MP4 and screenshots and checks GPU raster configuration and preview
pixels; this small fixture does not establish full telescope-movie performance.

The telescope film lives in Makie4Science's `Telescope` package, every part a
scene built once and animated by keyframes. Its tests build every part and check
that every key names a plot, an argument or the camera of what was built (the
editor skips a key that names nothing); they need a GPU and the film's
artifacts. `Telescope/tools/make_project.jl` writes the whole film as a project
with its approved narration, which a render farm uses without a speech model.
Unset Makie cycled colours are excluded from recording so replay cannot turn
palette defaults into explicit material overrides.

The recordings cover preview resolution selection, camera keys, actor transform keys, grading, cuts,
cross-dissolve Undo, ripple deletion, crop cancellation, speech text, reference
voice, model selection, Regie, asynchronous generation Ctrl+S and reopening a saved project through a file-manager drop. PNGs show the
final state of each recording. Grading is also checked against the rendered
pixels, using an identity curve so the fixture retains its colour. Without
`VIDEOEDITOR_QA_OUTPUT`, artifacts live in the test's temporary directory and are
removed afterwards.

Speech regression providers return deterministic PCM so UI tests need no model
downloads. Test real providers separately with actual takes; these assertions do
not judge voice quality. Native file dialogs and device audio playback also need
separate integration checks.

`test/creative_controls.jl` checks native light animation, saved overrides, reset
and bypass, and dialogue spans across cuts, gaps, changed speed and pending takes.
The CrawCity `tools/odyssee_creative_walkthrough.jl` records mouse edits on the
complete movie: selecting a SunSky light and changing its rendered intensity,
selecting an actor's performance and adding pose keys, and clicking a spoken-text
block to change its voice model and Regie and render a replacement take. It also
checks saving and reopening those edits. Its three recordings are `movie_light`,
`movie_animation` and `movie_dialogue_model_regie`. Its voice providers are QA
stubs; the desktop launcher registers the real VoxCPM2 and Fish providers.
The walkthrough also asserts that selecting a spoken line immediately exposes
its words and voice model, and that actor timing and pose precede facial fields.
Scene navigation brings the object picker and selected form into view without
requiring an initial scroll through the clip header.
Dropdown choices retain the Camera/Animation/Light category while preview picking
opens the selected object's full inspector. The movie creative walkthrough checks
switching from Light to Animation, selecting the intended actor, changing a key
without freezing its original timing curve, and regenerating a dialogue take.

`test/trackpreviews.jl` checks transient preservation, stereo separation, exact
cut boundaries, asynchronous request deduplication, memory eviction and shutdown.
It also checks that Undo observers can still read complete effect stacks and
parameters while curves are restored, so background thumbnails never freeze
an empty colour or grading effect.
Thumbnail requests carry their sequence time, including source trim and playback
speed. The bounded queue builds exact visible tiles from left to right, regardless
of GUI callback order; equal positions keep request order. Audio analysis can run
while scene work is parked for input, but a later baked image does not overtake an
earlier parked scene tile. View changes and edits discard obsolete queued image
requests. Valid cached frames remain immediately available; missing slots stay
empty until their own requested frame is ready.
The waveform band is reserved from the presence of its provider, before PCM
analysis is available. Arriving waveform points do not resize the thumbnail band
or change its sampled source times. Sources with no audio path have no waveform
provider and retain the full thumbnail band.
`checkwaveformarrival` records a single clip before/after late waveform publication
with its real thumbnail worker and exact cache. It checks unchanged requested
times, strip geometry/pixels and an empty job queue after the waveform appears.
This transition is separate from the settled-layout queue-order test: an ordered
worker alone cannot prevent holes if a loading result changes the requested grid.
The queue tests cover scrambled requests, shared frames, cache replacement,
viewport cancellation and an earlier request arriving at capacity. The preview
walkthrough clears only its test thumbnail cache after layout settles, then records
and checks actual background-render completion order with FakeInteraction.
`tools/odyssee_thumbnail_walkthrough.jl` repeats that check on the movie's reported
0–16 second viewport with Watcher Round's facial lanes open. It retains
`thumbnails_left_to_right.mp4`, waiting/ready PNGs and `thumbnail_order.json`.
`tools/odyssee_thumbnail_trace.jl` logs exact frame lookups, cache availability,
render completion and grid geometry for the card shot. Its cold-cache run checks
loading through a seek and actor selection. With `cachedhistory=true`, it first
renders genuine frames on two earlier grids, then changes the viewport while
only the test worker is parked. It checks an internal missing slot surrounded by
exact cached frames and waits for that slot to fill. `cached_history_hole.png`
and `cache_trace.json` retain the evidence. This controlled replay uses a 0–17 s
viewport at the worker's window geometry; it demonstrates the cache mechanism,
not the history of the user's original 0–16 s screenshot.
`test/trackpreview_walkthrough.jl` records cutting a scene with a file soundtrack
and regenerating a take through the voice-model menu. It checks that sound
envelopes survive cuts/zoom, overlapping takes stay selectable, thumbnails use
an independent raster scene, and closing the editor releases the preview worker.
Load/register RayMakie before running it to exercise the raster backend; otherwise
the small fixture uses its ordinary GLMakie scene backend. These new preview
tests do not synthesize real voices. `VIDEOEDITOR_QA_OUTPUT` retains PNGs/MP4s.

`test/animation_filter_walkthrough.jl` records the default active-only filter,
seeking between different recipe activities, revealing static fields, creating
keys, changing tracks and Undo. It checks source trim boundaries and saved edits.
Makie's `test/SceneLike/paramform_filter.jl` checks that row filtering collapses
space, preserves values/widgets/accessories and survives a parent being unfolded.
Existing editing walkthroughs explicitly uncheck the filter before exercising
static camera/light/pose fields. These tests reuse the same FakeInteraction actions.
The CrawCity `tools/odyssee_animation_filter_walkthrough.jl` checks the original
movie recipe's activity metadata, retained scenes and visible rows, and records
warm seeking and filter-toggle latency at Half raster preview resolution.

Use the worker's normal Pkg environment and Julia compilation caches. First
compilation can be slow. Load the packages needed for the test with `using`;
never call `Pkg.precompile()` or run a separate precompile pass. Keep the environment
for later runs, and pin its versions when the runtime stabilizes. Each test closes
its editor window in `finally`.

`test/farmdaemon.jl` checks the farm across processes: it starts a farm daemon
(`farm/farmd.jl`) with a slot per GPU of this machine, makes a job with a
`portable = false` environment, opens every slot over TCP and renders through
the daemons' own renderer processes. Set `VIDEOEDITOR_FARM_TEST=1` to run it:
the first run instantiates the job environment and compiles VideoEditor in it.

`test/sceneediting.jl` also covers opening projects with different canvas sizes
and discarding a render when its destination changes during composition.
This prevents an old preview from overwriting the newly opened document.

The project-specific `tools/odyssee_editor_walkthrough.jl` in CrawCity reuses
`test/editing_helpers.jl` and runs on a portable preview copy of the complete
74-shot, 3351-frame, 34-voice movie. It visits every shot by mouse and records
camera keys, actor timing/local pose/facial edits, reset/Undo, grading across
all scene clips, ship motion, cuts, Regie, searching the last spoken line and
save/reopen. Keep the approved project intact;
create a separate review project with a smaller canvas for interactive QA.
The test requires the movie's real scripts, meshes, recordings and mouth cues.
It is intentionally separate from the small package regression fixture.
`OdysseeEditingWalkthrough.prepare(original, review; pathmap, canvas=(180,320))`
creates that preview copy. Load RayMakie and call `usebackend!(RayMakie)` before
running `OdysseeEditingWalkthrough.run(review, output)`. Its default visits every
shot; `visitshots=false` repeats only the editing gestures when the shot checks
already passed on the same scene recipe.
The full movie also checks that closing the editor stops progressive preview
tasks, so successive walkthroughs do not leave raytracers running in the background.
`test/sceneresources.jl` repeatedly renders, closes and reopens a raster source,
checking identical output and removal of global font-atlas listeners. The scene
editing walkthrough also checks that closing the player releases its live sources
and the effect engine's plans, including when GPU video decoding is disabled.
First compilation can increase process memory; these lifetime checks do not
measure how much of a full movie process is geometry, driver memory or compiled code.

`tools/odyssee_selection_walkthrough.jl` reproduces selection after seeking out
of an earlier selected shot. It picks Watcher Round in the displayed frame and
checks that the inspector and 3D view address that shot, expose the original
performance curves without editing them, and show highlighted lanes.
It checks sparse Bézier fitting against every original frame, clicking a parameter
band/anchor to reveal its existing inspector control, dragging a Bézier handle
and Undo, and absence of duplicated timeline labels. It also checks the preview
pane allocation, full-height inspector, readable lane heights,
retention of the original effect header on scene realization, absence of a second object selector,
face edits through the existing numeric field, framing the actor with the working
camera, actor and camera dragging into the same inspector/curves, one Undo per drag,
save/reopen, and warm selection timings with scene identity preserved. Run it on the prepared
movie review project on Bosgame; `run(project, out)` keeps MP4/PNG evidence.
The recording helper pauses the native render loop so FakeInteraction controls
rendering, and saves screenshots from the framebuffer without restarting that
loop. Timings flush the setup frame before the measured event and distinguish
the mouse handler from the updated GUI framebuffer; waits between independent
clicks avoid measuring the toolkit's double-click gesture. The standalone scene
walkthrough allows up to five minutes for its first raster preview in a fresh
runtime, and stops immediately if it fails to become ready. This compilation
wait is separate from interaction latency measurements.
RayMakie's `test_preview_density.jl` additionally checks raster-first meshes and
surfaces: colour edits leave the unused trace texture empty, then switching to
tracing converts the current colour. Makie's `test/pipeline.jl` checks shared
RGBA texture storage at identity alpha and separate storage for changed alpha.
`test/sceneediting_walkthrough.jl` records object picking in both previews,
Move/Rotate/Scale, working-camera orbit/pan/zoom, matching the film camera,
camera-curve capture, path dragging, Undo/Redo and project
reopen through the existing FakeInteraction actions. A nonlinear camera fixture
checks that creating keys preserves the recipe's original motion. Pixel comparisons
check image orientation; selection overlays are cleared through the All objects button
for that comparison, then clicks check the projected picking coordinates. The CrawCity
`tools/odyssee_sceneediting_walkthrough.jl` checks the same picking, transform Undo
and camera-path navigation on the full movie and measures warm selection latency.
The walkthrough also guards textbox focus and derives both timeline coordinates
from the current axis limits. Nested controls must respect ancestor clipping;
otherwise a scrolled-out card header can steal a seek meant for the timeline.

`test/sceneediting.jl` also checks recipe-supplied performance controls: original
sampling, per-component overrides, ordering before world transforms, bypass,
reset, keyframes, project reopen and farm output. The controls are evaluated once
per group per frame; parameters are collected in one pass.
The editing walkthrough changes ambient colour and keys a point light's position
through the Light inspector, checks the rendered change, Undo, interpolation
after seeking and saved overrides.

`tools/odyssee_editor_performance.jl` measures the complete movie preview at
1080×1920 with Full, Half and Quarter raytracing resolution. It includes
animation, GPU upscale, effects and readback; it excludes GUI and network time.
It records allocation counts and checks that resolution switching keeps the
same scene and screen, and that exact output restores full resolution. The
movie walkthrough also selects all three settings through real menu gestures.
RayMakie's `test_preview_density.jl` checks resized output, overlay placement and
texture-atlas UV changes that preserve an unchanged material.
The movie additionally checks its sea grid at all three settings and restores
720,000 triangles for exact output. The small scene fixture checks that a recipe
detail callback runs before animation even when quality changes at the same
source frame, and that exact output requests full detail.

`tools/odyssee_editor_interaction_performance.jl` measures editing latency through
the same FakeInteraction widget events: grading drags, camera and animation
textbox commits, timeline clicks and preview quality selection. Run
`OdysseeInteractionPerformance.run(review, output)` with a portable 1080×1920
review project in the remote UI environment. It times the first published scene
preview and the updated GL UI framebuffer separately, excluding typing, scripted
waits, recording and physical display latency. Six warmed samples per operation
are recorded alongside the first sample, JSON results and screenshots. Refinement
follows the renderer's current mode; raster previews are checked to remain idle.
Timeline assertions check that the live scene, screen and plot objects are reused
and that animated inspector values still follow the original scene after each
click. Fixed-width numeric fields update their glyphs without a forced panel
layout pass. Changes to which rows are visible still require layout work.

Backends that expose `isprogressive(screen)` determine whether the editor keeps
refining a parked frame. RayMakie reports false in raster mode and true in traced
mode. Accepting a `clear` keyword alone cannot distinguish those modes. The
RayMakie density/atlas regression also toggles modes and checks both the rendered
colour and this capability; movie checks keep final rasterization off at 100 spp.

`tools/odyssee_pointer_performance.jl` extends these measurements to the reported
shot at frame 791: window clicks, pointer motion, timeline wheel zoom, subframe
zoom, seeking, object picking, numeric typing and working-camera orbit/pan/zoom.
It uses the walkthrough event helpers and verifies that picks select the intended
object. The selected filmstrip clip must cover the playhead. Each operation saves
first-use latency, warmed handler/UI-readback samples, allocations and a separate
CPU profile. Changed-object picks and clicks on the existing selection are
separate measurements; clearing the selection is preparation outside the timed
pick. This keeps compilation out of the warm profiles without hiding the
first interaction. The normal preview queue remains enabled during these tests.
The first background thumbnail build is measured separately from warmed input.
Subframe screenshots wait for the actual filmstrip tile to arrive; an empty
placeholder is not evidence of a usable thumbnail view. Preparation is drawn
before timing the next action. Saving a PNG restarts GLMakie's render loop, so
the benchmark stops it again before resuming manual event/readback samples.
Bosgame's HiDPI UI framebuffer is 3000×1900 for a 1500×950 editor figure.
Run `OdysseePointerPerformance.run(project, output)` on Bosgame. The scene-view
walkthrough also exercises film-only, scene-only and side-by-side layout changes
and checks that they preserve the same scene, controls and view.

Makie's `test/SceneLike/subfigure.jl` checks `replace_content!` across positions:
keyed buttons, sliders and focused textboxes retain identity and editing state
when reordered, old callbacks are removed even at elevated priorities, deleted
keys are released and changing a keyed control's type replaces it. Unkeyed
controls can move too, but stateful forms should use stable keys. It also guards
against fixed-width typing notifying the parent layout on every character.
Dock transitions republish visibility after their controls have been constructed;
the animation walkthrough asserts that the hidden export file chooser never
receives its clicks. Container and label tests distinguish their own visibility
from temporary scroll culling. Recordings keep the test windows hidden to avoid
native window events changing HiDPI framebuffer dimensions during encoding.
Walkthroughs wait for the first published frame before
interacting with controls built during asynchronous scene initialization.

Forwarded container geometry reports size and padding together. Unchanged bounds
and fixed intrinsic dimensions do not trigger a parent layout solve. GLMakie
keeps clipped or ancestor-hidden plots dirty until visible, skipping both their
uploads and CPU draw preparation. Its clipping tests check the restored pixels,
draw hooks, changing depth order and removal of a child scene.

Background scene thumbnails recheck input priority on the renderer thread before
building or seeking their independent source. Deferred work and renderer startup
timeouts remain retryable rather than becoming cached missing thumbnails.
`test/renderthread.jl` checks thread ownership, nested calls, exceptions, long
running work and expiry of a queued request without late side effects. Its blocked
worker includes GC safepoints so first compilation cannot deadlock the test.

Initialize a fresh runtime's Mantle queue on `onworkerthread` before opening the
Player, as CrawCity's `tools/odyssee_editor_start.jl` does. RayMakie raster and the
effect engine then share the existing renderer owner, while editor input stays
on the UI thread. An already-owned GPU queue cannot be moved to another thread.
Use the prepared ordinary Julia environment and its normal compile caches;
first compilation through the required `using` statements is expected.

`filter_cards!(...; compact=true)` keeps visible inspector cards together and
places hidden cards in a shared collapsed row. The compact-filter tests verify
widget identity, values, restored order and deletion of hidden controls. No
controls are detached from their owner's layout.

`tools/odyssee_rebuild_performance.jl` benchmarks reorderings of 30, 150 and 300
keyed buttons/textboxes/sliders, asserting identity on each move. Call
`OdysseeRebuildPerformance.run(output)` in the same prepared environment.

The 2026-10-06 Bosgame review stores its latest raw JSON, CPU profiles and reviewed
screenshots in CrawCity's `renders/odyssee_videoeditor/interaction_20261006/final/`.
At Half raster preview of the 1080×1920 movie, warmed handler / complete GUI
readback medians were:

| Interaction | Handler, ms | GUI readback, ms |
| --- | ---: | ---: |
| Window click | 2.25 | 22.02 |
| Pointer motion | 0.99 | 22.26 |
| Numeric typing | 1.17 | 43.83 |
| Timeline wheel zoom | 8.13 | 46.15 |
| Seek in the reported shot | 50.87 | 75.79 |
| All objects → actor | 22.35 | 77.84 |
| Existing selection | 4.29 | 37.34 |
| Actor ↔ ship | 17.91 | 67.98 |
| Working-camera orbit | 1.49 | 51.51 |
| Working-camera pan | 1.47 | 52.49 |
| Working-camera zoom | 0.19 | 48.13 |

These are eight samples each, with first use, allocation counts and GC time
retained in `pointer_performance.json`. The first background thumbnail measured
321 ms separately. CPU profiles still attribute changed-object work to inspector
layout and GUI updates; scene identity assertions rule out full reconstruction
during seeking. Raw warmed GUI samples include outliers up to 227 ms, so these
medians do not establish a constant frame rate. The worker also had another active
Julia workload; physical display latency is excluded.

At Half / Quarter / Full raster preview, grading drags measured 40.52 / 41.69 /
62.13 ms, camera commits 76.95 / 56.67 / 82.24 ms, animation commits 54.92 / 52.51 /
82.72 ms, and first-shot seeks 68.39 / 71.00 / 82.14 ms including rendering and GUI
readback. Six warmed samples per operation are retained in
`interaction_performance.json`; all qualities use the same GUI pixel density.
Reordering 30 / 150 / 300 keyed controls measured 0.68 / 6.71 / 18.73 ms in
`rebuild_performance.json`, with identity asserted on every move. Earlier results
remain in the parent directory; preparation and renderer ownership changed, so
they are not a controlled comparison of just one optimization.

The relevant suites contain 859 passing assertions and one existing failure:
`themeable axislegend` expects margin `(1,2,3,4)` but gets `(6,6,6,6)` from the
unchanged legend defaults. This is not an all-green Makie test run. The five
editing suites saved 32 mouse/keyboard recordings under
`qa/regression_clean_20261006` on the prepared worker. Voice/model/Regie tests
use deterministic PCM providers to check requests and regeneration; they do not
assess real speech-model quality or native audio-device playback. All test Players
are closed afterwards; the QA runtime measured about 16 GB RSS and no process swap
after the movie benchmarks. That footprint also includes compiled code and does
not establish a memory bound for every scene.
