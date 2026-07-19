# Mouse-driven walkthrough: turn a handheld bird clip into a STABILIZED, seamless,
# full-cycle loop and export it as a GIF — every step driven by the on-screen mouse
# (FakeInteraction renders a real cursor + click state), the way a user actually does it.
#
#   play the raw clip (nest box drifts) → Camera-lock stabilize the whole clip (GPU) →
#   cut off the shaky intro with the split tool → "Make seamless loop" (finds the full
#   ~13 s behavioural cycle where the bird goes around and comes back) → play the loop →
#   export as a looping GIF (export dock, GIF format).
#
# RULES (same as record_demo.jl): everything VISIBLE is a mouse/keyboard event — never
# mutate observables or call editor functions from the events list. The only script-side
# state pokes are INSTRUMENTATION (STABILIZING flag) so we can time-lapse the GPU-analysis
# wait in post; they don't drive any on-screen interaction.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
ENV["XAUTHORITY"] = get(ENV, "XAUTHORITY", "/run/user/1000/xauth_hqQZRv")
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE
import FFMPEG_jll

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, WaitUntil, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, KeyPress

const DEMOLOOP = joinpath(@__DIR__, "..", "..", "..", "media", "demo_loop.mp4")
const RAW_MP4  = joinpath(tempdir(), "loop_walkthrough_raw.mp4")
const OUT_MP4  = joinpath(@__DIR__, "..", "..", "..", "media", "loop_walkthrough.mp4")
const OUT_GIF  = joinpath(tempdir(), "bird_fullcycle.gif")

GLMakie.activate!(; visible = false, framerate = 30)
player = Player(DEMOLOOP; analysisbackend = LavaBackend())
fig = player.fig
resize!(fig, 1440, 900)
sleep(4.0)                                   # let the thumbnail strip fill
Makie.disconnect!(player.screen, Makie.mouse_position)   # keep the real OS mouse out
fig.scene.events.hasfocus[] = false
player.fxwidgets[:exportpath][] = OUT_GIF    # off-camera: preset the export path

# warm the GPU kernels off-camera (first Lava analysis compiles them)
let done = Ref(false)
    VE.rungpu(() -> (VE.analyzemotion!(VE.Clip(VideoSource(DEMOLOOP), 0, 48, 0, (0.,0.,1.,1.));
                                       backend = LavaBackend()); done[] = true), player)
    while !done[]; sleep(0.3); end
end

# ------------------------------------------------------------------ helpers
block_center(b) = FakeInteraction.relative_pos(b, (0.5, 0.5))
function timeline_pos(t; yfrac = 0.5)
    ax = player.timeline.axis; lims = ax.finallimits[]; vp = ax.scene.viewport[]
    frac = (Float64(t) - lims.origin[1]) / lims.widths[1]
    Point2f(vp.origin[1] + frac * vp.widths[1], vp.origin[2] + yfrac * vp.widths[2])
end
function menu_pos(menu, rely)  # rely 0.5 = the closed button; -(i-0.5) = open option i
    bb = menu.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + bb.widths[1] / 2, bb.origin[2] + rely * bb.widths[2])
end

buttons = [c for c in fig.content if c isa Makie.Button]
play_btn = first(b for b in buttons if b.label[] in ("Play", "Pause"))
fx_btn   = first(b for b in buttons if b.label[] == "FX")
out_btn  = first(b for b in buttons if b.label[] == "Out")
split_btn = first(b for b in buttons if b.label[] == "✂")
mode_menu = player.fxwidgets[:modemenu]
stabilize_btn = player.fxwidgets[:analyze]
loop_btn = player.fxwidgets[:loop]
fmt_menu = player.fxwidgets[:exportformat]
export_go = player.fxwidgets[:exportgo]

# held-key badge (so viewers see Esc etc.)
key_badge = Observable(" ")
on(Makie.events(fig).keyboardbutton) do e
    e.action == Makie.Keyboard.press && e.key == Makie.Keyboard.escape ?
        (key_badge[] = "Esc") : (key_badge[] = " ")
end
Makie.text!(fig.scene, key_badge; position = Point2f(720, 880), space = :pixel,
            align = (:center, :top), fontsize = 34, font = :bold,
            color = RGBAf(0.85, 0.15, 0.15, 1), strokecolor = :white, strokewidth = 3,
            overdraw = true)

# instrumentation: during every async wait (GPU analysis, loop search, GIF export) throttle
# the record loop to ~real time (else a WaitUntil spins at max encode speed and records tens
# of thousands of frames) AND collect those frame indices so post can time-lapse each span.
const TIMELAPSE = Ref(false)
const SLOW_FRAMES = Int[]
const MAXFRAME = Ref(0)
recfunc = (i, t) -> (MAXFRAME[] = i; TIMELAPSE[] && (push!(SLOW_FRAMES, i); sleep(1 / 30)); nothing)

# condition predicates (recording load makes GPU analysis time vary wildly — wait on the
# RESULT, not a guessed duration, so make-loop never runs on a half-stabilized clip)
hasmotion() = !isempty(player.sequence.clips) && player.sequence.clips[1].motiontrack !== nothing
looptrimmed() = VE.seqduration(player.sequence) < 15.0
exported() = occursin("exported", player.status[])

# -------------------------------------------------------------------- events
K = Makie.Keyboard
events = [
    Wait(1.2),

    # [1] the raw handheld clip — play it, then scrub to the busy nest box; it DRIFTS
    MouseTo(block_center(play_btn)), LeftClick(), Wait(3.0),
    KeyPress(K.space), Wait(0.5),
    Lazy(_ -> MouseTo(timeline_pos(2.0))), LeftDown(), Wait(0.2),
    Lazy(_ -> MouseTo(timeline_pos(13.0))), Wait(0.5), LeftUp(), Wait(0.8),

    # [2] Camera-lock stabilize the WHOLE clip (GPU): open the mode menu to show the
    # modes, pick "Camera lock", click "Stabilize clip". The status counts the frames;
    # the view glides into the auto-crop when it's done. (analysis wait time-lapsed in post)
    Lazy(_ -> MouseTo(menu_pos(mode_menu, 0.5))), LeftClick(), Wait(1.6),
    Lazy(_ -> MouseTo(menu_pos(mode_menu, -0.5))), LeftClick(), Wait(0.8),
    Lazy(_ -> (TIMELAPSE[] = true; MouseTo(block_center(stabilize_btn)))), LeftClick(),
    WaitUntil(hasmotion; timeout = 500.0),          # wait for the GPU analysis to finish

    # [3] the auto-crop glides in (normal speed = a nice reveal), then play the stabilized
    # clip: background nailed, only the birds move
    Lazy(_ -> (TIMELAPSE[] = false; MouseTo(block_center(play_btn)))), Wait(2.5),
    LeftClick(), Wait(3.0),
    KeyPress(K.space), Wait(0.6),

    # [4] cut off the shaky zoom-in intro with the SPLIT TOOL (crosshair cursor): click
    # the timeline at 5 s to cut, Esc to put the tool away, click the intro clip, X deletes it
    MouseTo(block_center(split_btn)), LeftClick(), Wait(0.8),
    Lazy(_ -> MouseTo(timeline_pos(5.0))), LeftClick(), Wait(1.1),
    KeyPress(K.escape), Wait(0.5),
    Lazy(_ -> MouseTo(timeline_pos(2.5))), LeftClick(), Wait(0.6),
    KeyPress(K.x), Wait(1.3),

    # [5] "Make seamless loop" — finds the full ~13 s cycle (bird goes around and returns)
    # and trims the timeline to it
    Lazy(_ -> MouseTo(timeline_pos(6.0))), LeftClick(), Wait(0.5),
    Lazy(_ -> (TIMELAPSE[] = true; MouseTo(block_center(loop_btn)))), LeftClick(),
    WaitUntil(looptrimmed; timeout = 120.0),
    Lazy(_ -> (TIMELAPSE[] = false; MouseTo(block_center(play_btn)))), Wait(1.0),

    # [6] play the finished loop — it goes around completely and comes back seamlessly
    MouseTo(block_center(play_btn)), LeftClick(), Wait(15.0),
    KeyPress(K.space), Wait(0.6),

    # [7] export as a looping GIF: open the export dock, set format = gif, Export GIF
    MouseTo(block_center(out_btn)), LeftClick(), Wait(1.2),
    Lazy(_ -> MouseTo(menu_pos(fmt_menu, 0.5))), LeftClick(), Wait(1.0),
    Lazy(_ -> MouseTo(menu_pos(fmt_menu, -3.5))), LeftClick(), Wait(0.8),   # 4th option = gif
    Lazy(_ -> (TIMELAPSE[] = true; MouseTo(block_center(export_go)))), LeftClick(),
    WaitUntil(exported; timeout = 150.0),
    Lazy(_ -> (TIMELAPSE[] = false; MouseTo(block_center(export_go)))), Wait(2.5),   # show "exported …"
]

FakeInteraction.interaction_record(recfunc, fig, RAW_MP4, events; fps = 30, px_per_unit = 1)
close(player)

# ---- collapse the dead time ------------------------------------------------
# Recording the editor while it does heavy GPU/CPU work (analysis, loop search,
# GIF export) starves the record loop, so the raw is minutes of a near-static
# frame between the real beats. mpdecimate drops the duplicate frames and we
# re-time to a constant 30 fps — the mouse moves, playback and edits survive,
# the dead waits collapse to a beat each.
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -vf "mpdecimate=hi=64*10:lo=64*4:frac=0.05,setpts=N/30/TB" -r 30 -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
@info "saved walkthrough" OUT_MP4 gif = OUT_GIF
