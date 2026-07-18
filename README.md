# VideoEditor.jl

A GPU-accelerated non-destructive video editor in Julia: GLMakie UI,
KernelAbstractions image processing (via GPUFiltering.jl, CPU or Lava/Vulkan),
VideoIO/FFmpeg decode + export.

```julia
using VideoEditor
p = Player("footage.mp4")           # opens the editor window
addsource!(p, "more.mp4")           # append another video (or drop the file onto the window)

saveproject("edit.toml", p.sequence)
exportvideo("out.mp4", p.sequence)  # bakes crop/effects/stabilization + muxes
                                    # the sources' audio along the cut list

srv = mcpserve!(p)                  # expose the live editor to AI agents (see below)
```

Sequences can mix any number of sources (each decodes through its own
background worker); resolutions may differ, framerates must match.

Heavy sources — above full HD, or heavyweight codecs like DNxHR/ProRes —
automatically get a background **preview proxy** (720p h264, cached across
sessions); scrubbing and playback switch to it when ready while analysis,
export and the project file keep the original. Tune or trigger manually
with `Player(...; proxyheight, proxythreshold)` / `startproxy!(p, source)`.

With a Lava GPU, `Player(path; analysisbackend = LavaBackend(),
gpupreview = true)` runs the whole presentation on the GPU: one frame
upload, motion/color tracks and effects as Vulkan kernels, and the result
displayed through a zero-copy shared texture (Vulkan↔GL external memory) —
2.1× the CPU preview throughput at 1080p with the CPU freed for decoding.
Renders identically to the CPU path (WYSIWYG-verified) and falls back to
it automatically on any error.

## Keys

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | step one frame (Shift: ±10) |
| S | split at playhead |
| Ctrl+S | save the project next to the first source (reopen with `Player(path)`) |
| X / Del | ripple-delete clip at playhead |
| C | crop mode — drag a rect on the preview (Esc cancels, R resets) |
| A | analyze color/exposure stabilization for clip at playhead (background) |
| M | analyze camera-shake stabilization for clip at playhead (background) |

Stabilization lives on the effect panel: pick a mode — *Camera lock*
(default: hold the scene like a tripod — reference patches from the first
frame, 4-DOF similarity fit, the same approach as DaVinci's
Similarity + Camera Lock), *Object lock* (follow a **moving** subject and
pin it in place — click "Stabilize clip", then click the subject in the
preview; Esc cancels), the legacy *Tripod*/*Tripod + perspective* flow
fits, or *Smooth* (keep intentional camera moves, remove shake) — then
"Stabilize clip". The panel names the mode already on the clip, the result
label shows the measured correction, "Hold to compare original" shows the
unstabilized frames while pressed, and the lock modes auto-crop the warp
borders. Re-analyzing replaces the previous stabilization (the crop is
re-derived from your original framing, not stacked), and "Remove
stabilization" (button or right-click modal) restores that framing —
both undoable. The status line next to Export shows analysis progress.

## Layout

A vertical toolbar runs down the far left; next to it is one docked panel slot
that the **FX** (effects & stabilization), **Bin** (media) and **Out** (export)
panels share — a panel opens there instead of overlaying the preview. The tool
icons below **arm**: split (✂) and crop (▢) change the cursor to a crosshair
and act where you click — split cuts the timeline at the click position (not
the playhead), crop drags a rectangle on the preview; ✕/↶/↷ (delete/undo/redo)
act at the playhead. The timeline spans the full window width. The preview
fills the whole available width when a crop/zoom makes it wider than the source.

- **Bin** — imported sources, each with a first-frame thumbnail. *Import clip…*
  opens a native file dialog; press a row and drag it onto the timeline (a
  translucent drop-region preview shows where it lands) to place a clip.
  Dropping video files on the window imports them too.
- **Out** — output path (native save dialog), format (mp4/mkv/mov or animated
  **gif**), codec, quality (crf), preset, audio toggle, and GIF fps/loop; then
  *Export video* / *Export GIF* renders in the background.

Timeline: left-drag scrubs (thumbnail preview while dragging, exact frame on
release), drag a clip *edge* to trim it (the handle appears right on the edge
and just past it), Ctrl+left-drag moves a clip (snaps to edges/playhead),
scroll zooms, right-drag pans. The FX panel edits the clip under the playhead.

## Architecture

Edits are metadata (`Sequence` of `Clip`s: source ranges, timeline placement,
crop rect, effect stack, stabilization tracks). Frames are produced on demand:

```
decode (worker thread, ring buffer) → motion warp → color track →
effect stack (GPUFiltering kernels) → preview (crop via axis limits)
                                    → export  (crop baked via warp!)
```

Preview and export share the same resolution machinery, so what you see is
what renders. Stabilization tracks are keyed by absolute source frame and
survive splits.

## MCP server (AI editing)

`mcpserve!(player; port=8765)` starts an MCP endpoint inside the editor
process. Attach Claude Code:

```
claude mcp add --transport http videoedit http://localhost:8765
```

The agent can then inspect state, look at rendered frames, cut, move, crop,
grade, stabilize and export — e.g. "look through this video, cut out the
boring parts and make it warmer" works against the live editor window.

## Demo

`examples/record_demo.jl` records a full feature walkthrough
(`media/editor_demo.mp4`, ~135 s) by driving the real UI with synthetic
mouse, keyboard and file-drop events (FakeInteraction from Makie/docs) —
playback, stepping, scrubbing, timeline zoom, split/delete, right-click
modal, edge trim + undo, Ctrl+drag clip moves, dropping a second video
onto the window, crop, grading, "Tripod + perspective" stabilization
(analyzed on the GPU) with the A/B hold-compare, object lock (the
birdhouse is clicked in the preview and pinned), and export. Everything
on screen is a real interaction; re-record after UI changes with
`include(...)`.

Playback has **audio**: each source's track is extracted once into the
scratch cache (memory-mapped, no RAM cost) and a feeder mixes the cut list
into PipeWire (`pw-cat`) while playing — cuts, trims, gaps and mute sources
all follow the edit. Zero extra dependencies; silently off when PipeWire
isn't available (`Player(...; audiopreview = false)` disables it).

## Known limitations

- preview audio has ~0.1–0.2 s of pipe latency (export stays sample-exact);
  no audio effects/crossfades
- sources must share the sequence framerate (frame-exact model, no resampling)
