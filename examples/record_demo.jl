# Record the editor walkthrough at /sim/Programmieren/VideoEdit/media/editor_demo.mp4.
#
# =============================================================================
# RULES for this script (same discipline as GeoChrono's record_demo.jl):
#
#   1. **Everything visible in the video must be driven by mouse, keyboard or
#      file-drop events** (`MouseTo`, `LeftDown/Up/Click`, `KeyPress/Down/Up`,
#      `Scroll`, `DropFiles`). Never mutate observables, call editor
#      functions, or seek from the events list. If a feature needs a
#      script-side mutation to demo, add UI for it instead.
#   2. `Lazy(_ -> MouseTo(...))` is fine — a mouse target computed at runtime.
#      `Lazy` with a side-effecting body is NOT fine.
#   3. Off-camera setup (before `interaction_record`) may prepare media and
#      warm caches. Nothing there appears in the video.
#   4. Held modifier keys render as the on-screen badge so viewers see why an
#      interaction behaves differently.
# =============================================================================
#
# Coverage — every editor feature is exercised on-screen:
#   [ 1] playback: Play button, pause via Space
#   [ 2] frame stepping: →, Shift+→ (badge)
#   [ 3] timeline scrubbing: drag = thumbnail preview, release = exact frame
#   [ 4] timeline zoom: scroll in/out (thumbnail density adapts, film-strip)
#   [ 5] cutting: S splits at playhead (×2), X ripple-deletes the middle
#   [5c] edge trim on a clip border + Ctrl+Z undo
#   [ 6] clip move: Ctrl+drag (badge), snap back to the cut edge
#   [6b] multi-source: DROP a second video file onto the window — appended
#        as a new clip with its own thumbnails/decoder, then played
#   [ 7] crop: C, drag a rect on the preview (Photoshop-style)
#   [ 8] grading: temperature/saturation/contrast sliders on the live frame
#   [ 9] camera stabilization: added via Ctrl+P (typed, Enter takes the first
#        hit), then its card's mode menu opened on screen and
#        "Camera lock" selected, analysis progress (running on the GPU via
#        the GPU — the CPU is busy rendering the recording itself),
#        locked playback + A/B
#   [9b] object lock: "Object lock" mode, Stabilize starts a pick, the
#        birdhouse is CLICKED in the preview and pinned in place
#   [10] export: Export button, status shows progress and the output path
#        (sources with sound get their audio muxed along the cut list)
#
#   (not shown: preview proxies — they kick in automatically for heavy
#   sources and are visually transparent by design)

using VideoEditor, GLMakie, Makie, Mantle
import VideoEditor as VE
import KernelAbstractions as KA
import FFMPEG_jll

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, RightClick,
                        Lazy, KeyPress, KeyDown, KeyUp, Scroll, DropFiles, TypeText

# ---------------------------------------------------------------- off-camera

# Demo footage: 16 s, 640 px proxy of the handheld bird clip — small enough
# that the on-camera stabilization analysis finishes in a few seconds.
demosource = joinpath(@__DIR__, "..", "..", "..", "media", "demo_source.mp4")
if !isfile(demosource)
    bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -ss 8 -t 16 -i $bird -vf scale=640:-2 -c:v libx264 -g 30 -crf 20 -preset fast -an $demosource`,
                 stdout = devnull, stderr = devnull))
end
# Second source for the drag & drop beat (same 60 fps, different aspect).
demosource2 = joinpath(@__DIR__, "..", "..", "..", "media", "demo_source2.mp4")
if !isfile(demosource2)
    clip2 = "/windows/Users/sdani/Cloudi/giffers/a34Dyj7_460svvp9.mp4"
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -t 6 -i $clip2 -vf scale=640:-2 -c:v libx264 -g 30 -crf 20 -preset fast -an $demosource2`,
                 stdout = devnull, stderr = devnull))
end

# Hidden screen: visible=true would let GLFW poll the real mouse and clobber
# the synthetic positions.
GLMakie.activate!(; visible = false, framerate = 30)

# NOT `analysisbackend = Mantle.defaultbackend()`. the GPU context belongs to whichever
# thread touches it FIRST, and constructing the backend here makes that MAIN —
# after which every analysis on the pinned worker dies with "BatchQueue is
# single-writer; cross-thread sweep forbidden". `autodetectgpu!` (spawned by the
# constructor) gets the order right: `vk_context()` through `rungpusync` first,
# the backend built around the worker-owned context second.
player = Player(demosource)
let t0 = time()
    while !(player.analysisbackend isa KA.GPU) && time() - t0 < 90
        sleep(0.2)
    end
    player.analysisbackend isa KA.GPU ||
        @warn "GPU autodetect did not enable the GPU tier — the walkthrough will run on the CPU tier"
end
# DO NOT set `player.gpupreview = nothing` here — see record_loop_demo.jl. The
# GL shared-texture import fails on this machine and the editor says so in the
# status bar, which is ugly; clearing the field after `autodetectgpu!` has wired
# the preview is worse, and renders green/black stripes instead of video.
fig = player.fig
resize!(fig, 1440, 900)
sleep(4.0)  # let the thumbnail strip fill before the video starts
Makie.disconnect!(player.screen, Makie.mouse_position)
fig.scene.events.hasfocus[] = false

# Warm the GPU analysis (Vulkan init + kernel compile take ~30 s on first
# use) on the player's pinned GPU worker so the on-camera stabilization
# only pays the actual analysis time.
#
# `rungpusync`, not `rungpu` + `while !gpuwarm[]`: fire-and-forget leaves the
# caller no way to hear that the job failed, so a worker that died on a lost
# device turned this into a silent forever-spin. The sync form rethrows here.
VE.rungpusync(player) do
    warmclip = VE.Clip(VideoSource(demosource2), 0, 48, 0, (0.0, 0.0, 1.0, 1.0))
    analyzemotion!(warmclip; backend = Mantle.defaultbackend())  # camera-lock patch kernels
end

# ------------------------------------------------------------------ helpers

block_center(block) = FakeInteraction.relative_pos(block, (0.5, 0.5))

"Pixel position of timeline time `t` (uses the CURRENT zoom — always Lazy this)."
function timeline_pos(t::Real; yfrac = 0.5)
    ax = player.timeline.axis
    lims = ax.finallimits[]
    vp = ax.scene.viewport[]
    frac = (Float64(t) - lims.origin[1]) / lims.widths[1]
    return Point2f(vp.origin[1] + frac * vp.widths[1],
                   vp.origin[2] + yfrac * vp.widths[2])
end

"Pixel position at a fraction of the preview axis."
function preview_pos(fx::Real, fy::Real)
    vp = player.previewaxis.scene.viewport[]
    return Point2f(vp.origin[1] + fx * vp.widths[1],
                   vp.origin[2] + fy * vp.widths[2])
end

# viewport position of a SOURCE-normalized point (nx, ny), crop/zoom aware:
# maps through the preview axis' current data limits (data = displayed buffer
# pixels; image y runs top-down, the viewport bottom-up)
function preview_data_pos(nx::Real, ny::Real)
    W, H = size(player.frame[])
    lims = player.previewaxis.finallimits[]
    x0, y0 = minimum(lims)
    x1, y1 = maximum(lims)
    fx = (nx * W - x0) / (x1 - x0)
    fy = 1 - (ny * H - y0) / (y1 - y0)
    return preview_pos(fx, fy)
end

slider_pos(key::Symbol, frac::Real) =
    FakeInteraction.relative_pos(player.fxsliders[key], (frac, 0.5))

buttons = [c for c in fig.content if c isa Makie.Button]
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
export_btn = first(b for b in buttons if b.label[] == "Export")
# The dock starts on FX, but dropping a file switches it to Bin and nothing
# switches back — so the grading and stabilization beats used to play with their
# own panel hidden. `toggledock!` only closes the dock when the clicked tab is
# ALREADY open, so clicking FX while Bin shows just switches to FX.
fx_btn = first(b for b in buttons if b.label[] == "FX")
# Panel widgets live in the left dock (not in fig.content). Stabilize is an
# effect KIND like any other now, so the mode menu, the analyze button and the
# A/B compare button are built by ITS CARD — they do not exist until the clip
# carries the effect, and they are rebuilt whenever the stack is. Hence
# accessors, read at event time, not bindings read once at script level.
stabilize_btn() = player.fxwidgets[:analyze]
compare_btn() = player.fxwidgets[:compare]
mode_menu() = player.fxwidgets[:modemenu]

"""
Add an effect the way a user does: Ctrl+P, type enough of its name to filter,
Enter takes the first hit.

Rule 1 — the card that carries the sliders (and the mode menu, and the analyze
button) only exists once the clip has the effect, and a script-side `seteffect!`
would put it there with no visible cause.

Ctrl+P rather than the "+ Add effect…" menu, which is what this first used and
which silently added nothing here: the palette "owns the keyboard while it is
open" (`palette.jl`), whereas plain typing into a menu that did not open reaches
the editor as single-letter shortcuts — and the letters in "color" include `c`,
the crop tool that the step right before this one uses. `record_keyframe_walkthrough`
drives the palette the same way and passes.
"""
addeffect_events(query) = [
    KeyDown(Makie.Keyboard.left_control), KeyPress(Makie.Keyboard.p),
    KeyUp(Makie.Keyboard.left_control), Wait(0.9),
    TypeText(query), Wait(0.9),
    KeyPress(Makie.Keyboard.enter), Wait(1.3),
]

"Pixel position on the mode menu: rely=0.5 is the button, -(i-0.5) is open option i."
function menu_pos(rely::Real)
    bb = mode_menu().layoutobservables.computedbbox[]
    return Point2f(bb.origin[1] + bb.widths[1] / 2, bb.origin[2] + rely * bb.widths[2])
end

"Click-drag a slider handle to `frac` of its track."
slider_set_events(key, frac) = [
    Lazy(_ -> MouseTo(slider_pos(key, max(frac - 0.15, 0.0)))),
    LeftDown(), Wait(0.1),
    Lazy(_ -> MouseTo(slider_pos(key, frac))),
    Wait(0.1), LeftUp(), Wait(0.7),
]

# Held-key badge (same pattern as the GeoChrono demo).
const KEY_LABELS = Dict(
    Makie.Keyboard.left_control => "Ctrl", Makie.Keyboard.right_control => "Ctrl",
    Makie.Keyboard.left_shift => "Shift", Makie.Keyboard.right_shift => "Shift",
)
key_badge = Observable(" ")
on(Makie.events(fig).keyboardbutton) do _
    held = unique(String[KEY_LABELS[k] for k in Makie.events(fig).keyboardstate if haskey(KEY_LABELS, k)])
    key_badge[] = isempty(held) ? " " : join(held, " + ")
end
Makie.text!(fig.scene, key_badge;
            position = Point2f(720, 880), space = :pixel, align = (:center, :top),
            fontsize = 40, font = :bold, color = RGBAf(0.85, 0.15, 0.15, 1.0),
            strokecolor = :white, strokewidth = 3, overdraw = true)

# -------------------------------------------------------------------- events

events = [
    Wait(1.2),

    # [1] Playback: Play button, pause with Space
    MouseTo(block_center(play_btn)), LeftClick(), Wait(2.2),
    KeyPress(Makie.Keyboard.space), Wait(0.8),

    # [2] Frame stepping: →, →, then Shift+→ (±10) with the badge visible
    KeyPress(Makie.Keyboard.right), Wait(0.35),
    KeyPress(Makie.Keyboard.right), Wait(0.35),
    KeyDown(Makie.Keyboard.left_shift), Wait(0.2),
    KeyPress(Makie.Keyboard.right), Wait(0.4),
    KeyPress(Makie.Keyboard.right), Wait(0.4),
    KeyUp(Makie.Keyboard.left_shift), Wait(0.5),

    # [3] Scrubbing: drag across the timeline — thumbnails while moving,
    # the exact frame snaps in on release
    Lazy(_ -> MouseTo(timeline_pos(2.0))), LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(timeline_pos(6.0))), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(11.0))), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(8.0))), Wait(0.3),
    LeftUp(), Wait(1.0),

    # [4] Timeline zoom: scroll in (denser film-strip tiles), then back out
    Lazy(_ -> MouseTo(timeline_pos(8.0))),
    Scroll((0, 4); duration = 1.0), Wait(1.2),
    Scroll((0, -5); duration = 1.0), Wait(0.8),

    # [5] Cutting: split at 4 s and 10 s, ripple-delete the middle clip —
    # watch the clip boundaries and the total duration in the timecode
    Lazy(_ -> MouseTo(timeline_pos(4.0))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.s), Wait(0.8),
    Lazy(_ -> MouseTo(timeline_pos(10.0))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.s), Wait(0.8),
    Lazy(_ -> MouseTo(timeline_pos(7.0))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.x), Wait(1.2),

    # [5b] Right-click on a clip: the actions modal lists everything
    # (with shortcuts), dismissed by clicking the backdrop
    Lazy(_ -> MouseTo(timeline_pos(2.0))), RightClick(), Wait(2.0),
    Lazy(_ -> MouseTo(preview_pos(0.05, 0.5))), LeftClick(), Wait(0.6),

    # [5c] Edge trim: drag clip 1's right edge inwards — the strip follows
    # live — then Ctrl+Z restores it (undo covers every edit)
    Lazy(_ -> MouseTo(timeline_pos(VE.clipend(player.sequence.clips[1]) / player.sequence.framerate))),
    LeftDown(), Wait(0.3),
    Lazy(_ -> MouseTo(timeline_pos(VE.clipend(player.sequence.clips[1]) / player.sequence.framerate - 1.2))),
    Wait(0.3), LeftUp(), Wait(0.9),
    KeyDown(Makie.Keyboard.left_control), Wait(0.2),
    KeyPress(Makie.Keyboard.z), Wait(0.9),
    KeyUp(Makie.Keyboard.left_control), Wait(0.5),

    # [6] Clip move: Ctrl+drag the second clip right, then back until it
    # snaps against the cut
    KeyDown(Makie.Keyboard.left_control), Wait(0.2),
    Lazy(_ -> MouseTo(timeline_pos((player.sequence.clips[2].start + 30) / player.sequence.framerate))),
    LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(timeline_pos((player.sequence.clips[2].start + 30) / player.sequence.framerate + 2.5))),
    Wait(0.5),
    Lazy(_ -> MouseTo(timeline_pos(VE.clipend(player.sequence.clips[1]) / player.sequence.framerate + 0.6))),
    Wait(0.4), LeftUp(),
    KeyUp(Makie.Keyboard.left_control), Wait(1.0),

    # [6b] Multi-source: DROP a second video file onto the window — the Bin
    # opens itself and the file lands there as a row (thumbnail, duration);
    # zoom out for room past the last clip, then drag that row onto the
    # timeline, where it becomes a clip with its own decoder and aspect
    DropFiles(demosource2), Wait(2.5),   # opening a source probes it off-thread
    Lazy(_ -> MouseTo(timeline_pos(VE.seqduration(player.sequence) * 0.6))),
    Scroll((0, -3); duration = 0.8), Wait(0.8),
    Lazy(_ -> MouseTo(block_center(player.binrows[end][2]))),   # the row's thumbnail
    LeftDown(), Wait(0.6),
    Lazy(_ -> MouseTo(timeline_pos(VE.seqduration(player.sequence) * 0.75))), Wait(0.4),
    Lazy(_ -> MouseTo(timeline_pos(VE.seqduration(player.sequence)))), Wait(0.5),
    LeftUp(), Wait(1.4),
    Lazy(_ -> MouseTo(timeline_pos(VE.seqduration(player.sequence) - 4.0))),
    LeftClick(), Wait(1.2),
    KeyPress(Makie.Keyboard.space), Wait(2.2),
    KeyPress(Makie.Keyboard.space), Wait(0.6),

    # [7] Crop: C, then drag a rect on the preview — the view crops to it.
    # On the DROPPED clip: the bird clip stays full-frame so the
    # stabilization beat can show its auto-crop zooming in.
    Lazy(_ -> MouseTo(timeline_pos(VE.seqduration(player.sequence) - 3.0))),
    LeftClick(), Wait(0.5),
    KeyPress(Makie.Keyboard.c), Wait(0.4),
    Lazy(_ -> MouseTo(preview_pos(0.22, 0.18))), LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(preview_pos(0.55, 0.5))), Wait(0.25),
    Lazy(_ -> MouseTo(preview_pos(0.8, 0.78))), Wait(0.25),
    LeftUp(), Wait(1.2),

    # [8] Grading: back on the bird clip — add the Color effect, then warm
    # temperature, more saturation, a bit of contrast; the frame updates live.
    # The card is what registers `player.fxsliders[:temperature]` &c., so adding
    # it is a step, not setup — without it the drag hits a KeyError.
    MouseTo(block_center(fx_btn)), LeftClick(), Wait(0.6),   # back to FX — the drop left the Bin open
    Lazy(_ -> MouseTo(timeline_pos(2.0))), LeftClick(), Wait(0.6),
    addeffect_events("color")...,
    slider_set_events(:temperature, 0.78)...,
    slider_set_events(:saturation, 0.68)...,
    slider_set_events(:contrast, 0.55)...,
    Wait(0.6),

    # [9] Camera stabilization: play the handheld footage, open the mode
    # menu so all modes are on screen, pick "Camera lock" (the NCC patch
    # tracker), click "Stabilize clip" — the status line counts the
    # analysis, the result label reports the shake, and the view GLIDES
    # into the auto-crop (outline flash marks the new framing) — then play
    # the SAME range locked off, and hold "Hold to compare original" for A/B
    Lazy(_ -> MouseTo(timeline_pos(0.5))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.space), Wait(3.0),
    KeyPress(Makie.Keyboard.space), Wait(0.5),
    addeffect_events("stabil")...,                               # the card, on camera
    Lazy(_ -> MouseTo(menu_pos(0.5))), LeftClick(), Wait(1.8),   # dropdown open
    Lazy(_ -> MouseTo(menu_pos(-0.5))), LeftClick(), Wait(0.8),  # Camera lock
    Lazy(_ -> MouseTo(block_center(stabilize_btn()))), LeftClick(), Wait(14.0),
    Lazy(_ -> MouseTo(timeline_pos(0.5))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.space), Wait(2.2),
    Lazy(_ -> MouseTo(block_center(compare_btn()))), LeftDown(), Wait(1.6),  # original, shaky
    LeftUp(), Wait(1.6),                                        # stabilized again
    KeyPress(Makie.Keyboard.space), Wait(0.6),

    # [9b] Object lock: keep the SUBJECT still — "Stabilize clip" starts a
    # pick, then the birdhouse BOX is clicked in the preview (the static
    # structure at source px (140, 760) — NOT the sparrows hopping on top,
    # a tracker told to pin a live bird will chase it) and the clip is
    # re-analyzed to pin it in place
    Lazy(_ -> MouseTo(menu_pos(0.5))), LeftClick(), Wait(1.4),
    Lazy(_ -> MouseTo(menu_pos(-1.5))), LeftClick(), Wait(0.8),  # Object lock
    Lazy(_ -> MouseTo(block_center(stabilize_btn()))), LeftClick(), Wait(1.4),
    Lazy(_ -> MouseTo(preview_data_pos(140 / 640, 760 / 1138))), LeftClick(), Wait(14.0),
    Lazy(_ -> MouseTo(timeline_pos(0.5))), LeftClick(), Wait(0.6),
    KeyPress(Makie.Keyboard.space), Wait(2.5),
    KeyPress(Makie.Keyboard.space), Wait(0.6),

    # [10] Export: the button opens the export dock (path/format/quality),
    # "Export video" renders through the same pipeline — both sources, and
    # the cut list's audio when the footage has any (status shows progress
    # and the output path)
    MouseTo(block_center(export_btn)), LeftClick(), Wait(1.2),
    Lazy(_ -> MouseTo(block_center(player.fxwidgets[:exportgo]))), LeftClick(), Wait(18.0),
    Wait(3.0),
]

video_path = joinpath(@__DIR__, "..", "..", "..", "media", "editor_demo.mp4")
FakeInteraction.interaction_record(fig, video_path, events; fps = 30, px_per_unit = 1)
println("Saved: ", video_path)
close(player)
