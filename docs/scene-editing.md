# Editing a procedural movie

Open the saved `.videoedit` project, select a shot on the timeline, and open the
Effects panel. Scene controls populate when the first preview frame is ready.
The Camera, Animation and Light buttons open their controls directly.
The selected shot is named in the inspector; its source markers follow trims and cuts.
Scene/Colour/Speech navigation and effect search stay visible above the scrolling
controls, including when an actor's acting section is long.

For CrawCity's complete Odyssee movie, run
`include("tools/odyssee_editor_start.jl")` from its normal Julia environment.
This opens `renders/odyssee_videoeditor/odyssee_editable.videoedit`: all 74 shots,
34 approved spoken takes and the effects bed. Use **Camera**, **Animation** or
**Light**, then choose the actor or light in the object menu inside
**Camera, animation & light**. Its menu lists the selected category so you do not
have to search through unrelated scene objects. Use **Colour** for grading. The dialogue lane above
the picture clips shows spoken words: click a block to edit that line, or use
**Speech** for the searchable spoken-line picker. Zoom in to read shorter takes.
The launcher starts with the camera inspector and the first twelve seconds visible.
Save with **Ctrl+S**. Keep a separate project copy when experimenting with takes.

Odyssee uses **RayMakie's raster mode** for editing and raytracing for finished
and farm output. Both use the same scene recipe and saved edits. In
**Rendering → Preview**, `rasterize` controls the preview mode; the Bake settings
keep `rasterize=false` and the finished sample count. Raster previews draw once
per edit and remain idle when nothing changes.

The footer's **Preview** menu offers Full, Half and Quarter resolution for
RayMakie scenes in either mode. Half halves each dimension; Quarter quarters each dimension.
The GPU scales the picture back to the project canvas before applying effects,
so grading, crop coordinates and effect sizes keep their usual meaning. This is
a session preference: it does not change the saved project, Undo history or final
export resolution. Other scene backends keep their native resolution. The
Odyssee launcher starts at Half; choose Full to inspect fine details.

Recipes may also reduce expensive geometry through their optional `preview!`
callback. Odyssee uses the same waves and whirlpool with a coarser sea grid:
720,000 triangles at Full, 180,000 at Half and 45,000 at Quarter. Full preview
and finished output restore the original grid. Reduced previews are approximate
and should be checked at Full when judging small surface details.

Resolution changes reuse the live scene and renderer. Rebuilding a scene at a
different canvas size also reuses its loaded recipe builder within that open
project. Reopen the project to load edited Julia recipe definitions. Package
loading and compilation still use normal Julia Pkg and compilation caches;
the first render in a fresh process can take substantially longer.

## Colour

**Color grade** holds the saved colour curve. **Strength** blends it with the
original picture. **Color** supplies brightness, contrast, saturation and
temperature adjustments after that curve. Both use the normal keyframe controls.

**Replace curve with AI grade** is a separate, explicit action. It predicts a new
curve from the frame under the playhead and replaces the selected clip's saved
curve. It is not needed to adjust brightness or the strength of an existing grade.

For a movie split into shots sharing one scene, **Apply grading to all scene
clips** copies the selected shot's grade and Color effects to those clips. This
includes their enable states and curves. Other effects and scene edits remain
per shot. The action is undoable. Curves stay in absolute source-frame time.

## Camera and animation

In a RayMakie raster preview, click an actor or prop to select it. The inspector
shows that object's transforms and associated acting controls. Static transforms
remain visible for an explicitly selected object even with the animation filter
enabled. **All objects** restores the full scene inspector. Selecting Camera,
Animation or Light also leaves the object selection so their controls are accessible.

Open **Scene view** in the preview footer for an independent 3D working camera.
Click objects there or use its searchable object menu. Select **Move**, **Rotate**
or **Scale**, then drag the red, green or blue handle for the corresponding axis.
The working view reuses the scene's GPU geometry. Right-drag orbits, middle-drag
pans and the mouse wheel zooms; these gestures do not change the film camera.
**Match film view** restores the working camera to the current film view.

**Camera path** shows the selected shot's authored camera motion. **Add camera key**
captures that motion as ordinary editable parameter curves while retaining the
current frame as a key. Drag the gold key points to change camera positions in 3D.
Recipes provide a pure `camera(frame, fps)` sampler so path inspection requires
no intermediate scene renders. Each transform or path drag makes one Undo step;
**Ctrl+Z** and **Ctrl+Shift+Z** work in the Scene view too. Save normally to retain
the same edits for export and the render farm.

**Animated at playhead only** is checked by default. It shows editor curves whose
key span includes the playhead and original recipe properties active around that
frame, within the selected clip's source range. Cuts, retiming and track selection
use the selected clip's source clock. Key spans remain editable even when two
keys currently have equal values. Uncheck it to reveal static/unkeyed parameters
and start a new animation. The choice is shared across scene cards for this
editor session; it does not change the project or its keyframes.

The object picker excludes groups with no active parameters. An empty filter
explains how to show all controls. Seeking and Undo update the row masks while
retaining the existing widgets and renderer. Odyssee declares original activity
with a pure CPU `activeparams(firstframe,lastframe,fps)` recipe callback, covering
its actor performances, ship motion and camera. Recipes without that callback
still expose editor-keyed/driven parameters by default; uncheck the filter for
their other controls. Activity queries must not render, seek or mutate the scene.

Press **Camera** or **Animation**, then choose an object in the menu. Type positions,
targets, rotations in degrees, and scales directly into the numeric fields. The
object filter finds actors and their properties. The movie's actor groups move
their body, face and attached parts together.

An untouched property follows the original procedural animation. Editing a
property replaces that component with a constant for the selected shot. Press
**◇** to animate it, scrub, and change its value to add keys. **◀ / ▶** walk the
keys; **∿** shows the curve on the timeline. **↺** removes the override and restores
that property's original animation. Edits and reset actions support Undo.

The recipe supplies the original animation. Inspector edits and keys override
individual properties while untouched properties continue to follow it. Save
normally to retain overrides, keyframes and grading. The render farm reads the
same saved edit.

Odyssee also exposes an **acting** section for each actor. **Animation time** is
the actor's source clock in seconds: a constant freezes its performance, while
keys retime it independently of the camera and other actors. **Local position**,
**Yaw/Lean/Roll degrees**, **Grow** and **Squash** change the performance inside
the actor's ship/parent coordinates. The object's ordinary transform controls
then apply in world coordinates. **Ship · acting** moves the ship together with
its crew and onboard cameras; **Rock strength** controls its rocking motion. Painted-face actors also expose eyes, gaze,
brows, mouth, smile, tears and reaction expressions. Edited expressions are
painted exactly into the small face texture; the movie's existing face atlas is
reused when no expression is overridden. Reset restores following the original.

## Light

Press **Light** and choose Ambient, SunSky or a numbered light in the object menu.
The inspector reads the scene's native Makie lights. It exposes their numeric
fields, including intensity, colour components, direction and position where
supported by that light type. These use the same overrides, keyframes, reset,
Undo and saved-project path as camera and animation edits. Ambient light has
index zero; other indices follow the scene's native light order. There is no
separate editor lighting rig.

## Timeline previews

Video clips keep their existing background filmstrip decoder. Makie scene clips
now build small filmstrips in the background as well, including saved camera,
animation and effect edits. RayMakie thumbnails use raster mode. Their renderer
is separate from the programme monitor and never seeks or resizes its scene.
Visible requests take priority; panning or editing discards queued old scene
requests. Scene work pauses during playback/scrubbing and briefly after edits.
If an enabled bake covers the requested frame, its PNG supplies the thumbnail
on the CPU instead of rendering that frame again.

File soundtracks and already rendered speech show classic peak waveforms. Stereo
channels remain separate; overlapping spoken takes occupy separate visible rows.
These rows collapse when the overlap is outside the visible time range. Waveforms
use source time through cuts, trims and changed clip speed. A regenerated take
gets a new envelope; pending text has no invented audio waveform. Display amplitude
is normalized per source for readability and does not change the audio mix.

Audio analysis runs on the CPU and shares the existing memory-mapped PCM scratch
cache with playback. Peak pyramids preserve short transients when zoomed out and
query exact sample boundaries at cuts. The scene/audio preview cache retains at most 64 MiB of derived
peaks/images, 1024 entries and 256 pending requests; waveforms for very long files
use coarser base blocks to fit that budget. Full PCM is not duplicated. UI refreshes
are batched at 10 Hz rather than drawing once per completed background request.

## Exposing useful objects from another scene

By default, a procedural scene exposes its named plots. A recipe can supply
editing groups to keep multi-part objects together and avoid listing renderer
settings as creative controls:

```julia
build = programscene("animation.jl"; objects = [
    Dict("label" => "Telescope",
         "plots" => ["telescope_body", "telescope_lens"],
         "attributes" => ["color", "alpha"]),
])
```

The first plot defines the group's position, rotation and scale. Transform edits
move its members together around that plot's origin. Named groups and property
values are ordinary project data; the live scene remains the source of truth.
Plot names must be stable and group members should not also belong to another
editing group. Camera controls are offered automatically for a 3D camera.

A recipe can expose creative controls that aren't Makie plot attributes by
returning `controls` alongside `scene` and `update!`:

```julia
clock = Ref(0.0)
controls = [(name=:performance, label="Performance",
    sample=(frame, fps)->(time=frame/fps,),
    apply! = (edits, frame, fps)->(clock[]=get(edits, :time, frame/fps)))]
return (scene=scene, update! = (frame,fps)->animate!(scene, clock[]), controls)
```

`sample` reports the original values without mutating the scene. `apply!` gets
only edited fields, once per frame, before the updater. It also gets an empty
dictionary when the scene effect is bypassed or a property is reset. Values can
be numbers, vectors or colours. These controls reuse ordinary scene parameters;
callbacks stay in the Julia recipe and are never serialized into the project.

A builder may also return `preview! = pixel_scale -> ...` to adjust procedural
detail for the preview. The editor calls it once per quality change, before
`update!`, including when the playhead stays still. A scale of `1.0` must restore
the original scene detail; exact output always requests it. Keep this callback
separate from saved artistic parameters and do not change recipe arguments.

Speech editing
--------------

Click a spoken-text block in the dialogue lane to open its line inspector and
seek to that time. The lane follows cuts, gaps and retiming of anchored takes.
The Effects panel's **Speech** button offers the same inspector with a searchable
line picker. Edit **Words**, select **Voice model** and a reference voice, add
**Regie** separately from the spoken text, then use the **Render take** and
**Listen** buttons at the top of the card. A model without delivery support reports
that explicitly. Reference models take voice identity from the selected recording;
a named-voice model uses its Voice ID.

Changing words or synthesis settings invalidates that take. Rendering runs off
the UI thread. Undo restores both settings and rendered audio; a result arriving
after the line was edited/deleted is discarded rather than overwriting that edit.
Projects keep approved PCM and synthesis settings, so reopening and farm rendering
do not require loading a speech model.

Applications register lazy providers with `registerspeechmodel!(:name, label,
synthesize; direction=true, reference=true)`. `synthesize(narration)` returns
finite mono samples and a positive sample rate. Julia Pkg environments and model
installation remain separate from a render job. The Odyssee launcher reuses its
existing VoxCPM2 and Fish workers; switching models releases the previous worker.

A `Narration` can anchor to a picture `ClipSource`. Its time is then source time:
trimming, splitting, moving, ripple deleting and retiming the picture also cut and
retime that voice. **Move here** detaches a line into a timeline voiceover.

`odyssee_editable.videoedit` has 34 approved spoken takes plus an effects bed,
using the original take gains, ramps and loudspeaker filters. This is the
pre-master mix; the approved MP4 and original mixed-soundtrack project are retained.
The scene recipe still owns captions and Rhubarb mouth cues. Changing spoken words
or duration does not yet regenerate those visual cues automatically.
Actor/camera transforms, local poses, actor clocks and facial-expression keys
are editable in the UI. The recipe still defines the original choreography,
animation layers, world construction and relationships between actors; the
editor does not translate arbitrary Julia functions into original timeline keys.
