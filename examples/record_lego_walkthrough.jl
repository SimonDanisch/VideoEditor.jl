# Mouse-driven walkthrough of the RAYTRACED-SCENE workflow, end to end:
#
#   keyframe the figure and the camera → preview it with RayMakie → put an
#   ordinary effect on it → bake it at N samples per frame → cut it → export
#
# Every step is a real on-screen mouse or keyboard event (FakeInteraction draws a
# live cursor). Only three things are script-side and off camera, each for a
# reason given where it happens: the short working clip, the shader warm-up, and
# the export path (a native file dialog no synthetic event can reach).
#
# ## What is measured rather than guessed
#
#   * The scene card's rows are looked up by NAME through `player.fxwidgets`
#     (`kfacc_<slot>_<param>` is the ◆, `fxsections_<slot>.box` is the object
#     filter). Every lookup happens per use: filtering the card REBUILDS it, and a
#     captured Button is a deleted block whose `computedbbox` still answers.
#   * The Bake tab only offers `samples` once the renderer it resolves to is
#     RayMakie — with `Bake with = same as the preview`, that is after the preview
#     has been switched. So the beats run in that order, which is also the order a
#     person would work in.
#   * RayMakie's first frame in a process compiles its shaders (~20 s). That is
#     warmed off camera, or the take is a fifth of a minute of a frozen window.
#
# Run on :1.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, Lava, RayMakie, GLMakie, Makie
import VideoEditor as VE
import FFMPEG_jll
import Printf

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, WaitUntil, MouseTo, LeftClick, LeftDown, LeftUp,
                        Lazy, KeyPress, KeyDown, KeyUp, TypeText, Scroll

const MEDIA    = normpath(joinpath(@__DIR__, "..", "..", "..", "media"))
const PROJECT  = joinpath(MEDIA, "lego.videoedit")
const RAW_MP4  = joinpath(tempdir(), "lego_raw.mp4")
const OUT_MP4  = joinpath(MEDIA, "lego_walkthrough.mp4")
const EXPORTED = joinpath(MEDIA, "lego_export.mp4")
const ASS      = joinpath(tempdir(), "lego_captions.ass")
# Absent files are an honest failure signal: `Makie.record` only writes its file
# when the block completes, so a take that throws would otherwise leave the last
# take sitting there with a plausible mtime, to be watched as if it were this one.
rm(RAW_MP4; force = true)
rm(OUT_MP4; force = true)
rm(EXPORTED; force = true)

GLMakie.activate!(; visible = false, framerate = 30)
VE.usebackend!(RayMakie)      # a scene names its renderer in text; the name has to resolve
player = VE.Player(PROJECT)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
# `visible = false` hides the window but GLMakie keeps polling GLFW, so anything
# done on the machine while a take runs is delivered into the figure alongside the
# synthetic events. A stray scroll is the worst of them: it zooms an axis and
# silently invalidates every coordinate computed afterwards.
for chan in (Makie.mouse_position, Makie.scroll, Makie.mouse_buttons,
             Makie.keyboard_buttons, Makie.unicode_input, Makie.dropped_files,
             Makie.hasfocus, Makie.entered_window)
    Makie.disconnect!(player.screen, chan)
end
fig.scene.events.hasfocus[] = false

scene_clip() = first(c for c in seq.clips if c.source isa VE.SceneSource)
const SRC = scene_clip().source

# --------------------------------------------------------------- off camera (1)
# A short working clip. The bake renders every frame of it with a path tracer, so
# the full 180 would be five minutes of watching a progress bar.
let c = scene_clip()
    c.src_out = c.src_in + 72
end
VE.redraw!(player)
player.playhead[] = 0
VE.pause!(player)

# --------------------------------------------------------------- off camera (2)
# RayMakie's first frame in a process builds its screen and compiles the
# integrator's shaders. Measured cold: over twenty seconds, all of it a frozen
# window. Pay it here.
let t0 = time()
    SRC.backend = :RayMakie
    VE.showplayhead!(player)
    while SRC.samples < 2 && time() - t0 < 180
        sleep(0.2)
    end
    SRC.backend = :GLMakie       # …and start the take where a person would start
    VE.showplayhead!(player)
    sleep(1.0)
    @info "raytracer warmed off camera" seconds = round(time() - t0, digits = 1)
end

K = Makie.Keyboard
fps = seq.framerate
CHECKS = Tuple{String, Bool, String}[]
"""
Record a beat's outcome. Loud on failure, but the take runs to the end: throwing
here aborts `Makie.record`'s block, and then there is no video of the run that
failed — which is the one worth having.
"""
function beat!(name, ok, detail = "")
    push!(CHECKS, (name, ok, detail))
    ok ? @info("beat ok", name, detail) : @error("BEAT FAILED", name, detail)
    ok
end

# ------------------------------------------------------------------- geometry
"Pixel point for axis time `t` (seconds) and axis-y `ay` (0..1)."
function tlpos(t; ay = 0.5)
    a = player.timeline.axis; lims = a.finallimits[]; vp = a.scene.viewport[]
    Point2f(vp.origin[1] + (Float64(t) - lims.origin[1]) / lims.widths[1] * vp.widths[1],
            vp.origin[2] + (Float64(ay) - lims.origin[2]) / lims.widths[2] * vp.widths[2])
end
"Centre of a clip, from ITS OWN lane — never a guessed fraction of the axis."
function clippos(c; tfrac = 0.5)
    lo, hi = VE.trackband(c.track, VE.ntracks(seq))
    tlpos((c.start + tfrac * (VE.clipend(c) - c.start)) / fps; ay = (lo + hi) / 2)
end
"A point in the SCRUB strip above the lanes — this moves the playhead, a lane click picks."
scrubpos(t) = (vp = player.timeline.axis.scene.viewport[];
               Point2f(tlpos(t)[1],
                       vp.origin[2] + (VE.SCRUBBAND[1] + VE.SCRUBBAND[2]) / 2 / VE.AXISTOP * vp.widths[2]))
bcenter(b) = (bb = b.layoutobservables.computedbbox[]; Point2f(bb.origin .+ bb.widths ./ 2))
"…and for the things that are a rect rather than a block — a Tabs header."
bcenter_rect(r) = Point2f(r.origin .+ r.widths ./ 2)
"The middle of the scrolling effects dock — where a wheel gesture has to happen."
dockmid() = (v = player.fxpanel.scroll.scene.viewport[]; Point2f(v.origin .+ v.widths ./ 2))
button(label) = first(b for b in fig.content if b isa Makie.Button && b.label[] == label)

# The scene card's widgets, per use. Filtering rebuilds the card, so a captured
# Button is a block that no longer exists and whose bbox still answers plausibly.
slotkey() = first(String(k)[length("fxsections_") + 1:end]
                  for k in keys(player.fxwidgets) if startswith(String(k), "fxsections_"))
sections() = player.fxwidgets[Symbol("fxsections_", slotkey())]
objfilter() = sections().box
# `kfacc_*` is the Premiere-style ◀ ◆ ▶ trio, not one button: the middle one is
# the ◆ that arms the parameter, the outer two step between its keys.
kfbtn(name) = player.fxwidgets[Symbol("kfacc_", slotkey(), "_", name)][2]
sceneparam(name) = VE.param(VE.findslot(scene_clip(), :scene), Symbol(name))
sceneslider(name) = sceneparam(name).view.control
nkeys(name) = length(sceneparam(name).curve[].keys)

sliderfrac(sl) = (r = sl.range[]; (sl.value[] - first(r)) / (last(r) - first(r)))
val2frac(sl, v) = (r = sl.range[]; (Float64(v) - first(r)) / (last(r) - first(r)))
function slider_x(sl, frac)
    bb = sl.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + clamp(frac, 0, 1) * bb.widths[1], bb.origin[2] + bb.widths[2] / 2)
end
"""
Press the handle where it is, drag to `v`, release — the preview follows the drag.

`v` may be a number or `sl -> number`, and either way it is read INSIDE a `Lazy`:
everything about the card exists only once the clip is selected, which is several
beats after this array is built.
"""
dragparam(name, v) = let target(sl) = v isa Function ? v(sl) : v
    [Lazy(_ -> (sl = sceneslider(name); MouseTo(slider_x(sl, sliderfrac(sl))))),
     LeftDown(), Wait(0.15),
     Lazy(_ -> (sl = sceneslider(name); MouseTo(slider_x(sl, val2frac(sl, target(sl)))))), Wait(0.15),
     Lazy(_ -> (sl = sceneslider(name); MouseTo(slider_x(sl, val2frac(sl, target(sl)))))),
     LeftUp(), Wait(0.5)]
end

"""
Type into a Textbox that has to be focused first, and commit it. `getbox` is a
thunk for the same reason `dragparam` takes one.
"""
typeinto(getbox, text) = [Lazy(_ -> MouseTo(bcenter(getbox()))), LeftClick(), Wait(0.35),
                          TypeText(text), Wait(0.5), KeyPress(K.enter), Wait(0.8)]

# The rendering dialog, per use for the same reason.
rmodal() = player.fxwidgets[:rendermodal]
"Where a Menu's row lands once its dropdown is open — off the popup scene itself."
function menurow(m, label)
    i = findfirst(o -> first(o) == label, m.options[])
    sc = m.blockscene.children[end]
    rects = sc.plots[1][1][]; tr = Makie.translation(sc)[]
    Point2f(sum(extrema(rects[i])) ./ 2 .+ Point2f(tr[1], tr[2]))
end
pickmenu(get_menu, label) = [
    Lazy(_ -> MouseTo(bcenter(get_menu()))), LeftClick(),
    WaitUntil(() -> get_menu().is_open[]; timeout = 8.0), Wait(0.3),
    Lazy(_ -> MouseTo(menurow(get_menu(), label))), LeftClick(), Wait(0.8)]

# The captions are a SUBTITLE TRACK burned in afterwards, not text drawn into the
# figure. Drawn into the figure they are a plot like any other, and the preview
# panel wins over them: measured across two takes, "Bake it: render every frame to
# disk, once" ended at x = 778 — the preview's left edge — mid-word, and moving the
# text into a scene of its own changed nothing, because what puts the picture there
# is a texture swapped into the preview plot's render object (`presentgpu!`), not a
# draw Makie orders against the caption.
#
# `now_t` is the video time of the frame being recorded, from `interaction_record`'s
# per-frame callback, so a caption's start is the frame it was set on.
caption = Observable("")
CAPTIONS = Tuple{Float64, String}[]
BUSY = Tuple{Float64, Float64}[]   # spans where a long job ran and nothing moved
now_t = Ref(0.0)
on(caption) do txt
    push!(CAPTIONS, (now_t[], txt))
end
busy_since = Ref(NaN)
"""
Called once per recorded frame: records the video time, and whether a long job is
running right now.

`jobprogress` is an `Atomic`, not an `Observable` — the footer spinner polls it and
so does this, which is also the only way to get a job's span in VIDEO time rather
than wall-clock. NaN means idle.
"""
function tick!(t)
    now_t[] = t
    running = !isnan(player.jobprogress[])
    if running && isnan(busy_since[])
        busy_since[] = t
    elseif !running && !isnan(busy_since[])
        push!(BUSY, (busy_since[], t))
        busy_since[] = NaN
    end
    return nothing
end

"ASS timestamp: `h:mm:ss.cc`."
asstime(t) = (h = floor(Int, t / 3600); m = floor(Int, (t - 3600h) / 60);
              Printf.@sprintf("%d:%02d:%05.2f", h, m, t - 3600h - 60m))

"""
Write the caption timeline as an ASS subtitle file.

ASS rather than SRT because the style travels in the file, so the ffmpeg filter
argument stays a bare path — `force_style` would need every comma escaped inside
the filter graph. `PlayResX/Y` are the video's own size, so the font size below is
in the pixels of the recording.
"""
function writecaptions(path, caps, tend)
    open(path, "w") do io
        print(io, """
        [Script Info]
        ScriptType: v4.00+
        PlayResX: 1500
        PlayResY: 950
        WrapStyle: 0
        ScaledBorderAndShadow: yes

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Cap,DejaVu Sans,30,&H00FFFFFF,&H000000FF,&HB4000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,8,20,20,16,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        """)
        for (k, (t0, txt)) in enumerate(caps)
            isempty(strip(txt)) && continue
            t1 = k < length(caps) ? caps[k + 1][1] : tend
            t1 - t0 > 0.3 || continue
            println(io, "Dialogue: 0,", asstime(t0), ",", asstime(t1), ",Cap,,0,0,0,,", txt)
        end
    end
    return path
end

"""
An ffmpeg `-vf` chain: captions burned in, then the stretches where a long job ran
played at `speed`×.

Those stretches are a progress bar and nothing else. Each one longer than
`longerthan` is thinned to about `target` seconds — the bake keeps enough frames
that "baking 28 / 72…" and the footer bar still read, the export does not sit
there for half a minute. `select` keeps every `k`-th frame inside a span and every
frame outside, `setpts` renumbers what is left, so it is one pass and no
concatenation. The captions are burned BEFORE the drop, so their timing is still
the timing they were recorded at.
"""
function speedupfilter(assfile, spans; target = 5.0, longerthan = 7.0)
    long = [(a, b, max(2, round(Int, (b - a) / target))) for (a, b) in spans if b - a > longerthan]
    isempty(long) && return "ass=$assfile"
    # Nested, innermost last: each span keeps every `k`-th frame, everything else
    # keeps all of them. One `select` expression, so it stays one pass.
    expr = "1"
    for (a, b, k) in reverse(long)
        expr = "if(between(t\\,$(round(a, digits = 2))\\,$(round(b, digits = 2)))\\," *
               "not(mod(n\\,$k))\\,$expr)"
    end
    return "ass=$assfile,select=$expr,setpts=N/FRAME_RATE/TB"
end

T = (scene_clip().src_out - scene_clip().src_in) / fps      # working clip length (s)
armed = Ref(0)

events = [
    Wait(1.0),

    # ------------------------------------------------------- the piece
    Lazy(_ -> (caption[] = "A raytraced LEGO figure on its own track, over the footage";
               MouseTo(bcenter(button("Play"))))),
    LeftClick(), Wait(2.5), KeyPress(K.space), Wait(0.6),

    # ------------------------------------------------------- keyframe the figure
    Lazy(_ -> (caption[] = "Pick the scene clip"; MouseTo(clippos(scene_clip())))),
    LeftClick(), Wait(1.2),
    Lazy(_ -> (beat!("scene clip selected", VE.selectedclip(player) === scene_clip());
               caption[] = "Every part of it is a parameter — filter down to the torso";
               MouseTo(bcenter(objfilter())))),
    typeinto(objfilter, "torso")...,

    Lazy(_ -> (caption[] = "Park the playhead…"; MouseTo(scrubpos(0.15T)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (armed[] = nkeys("torso.angle");
               caption[] = "◆ arms Angle — the first key lands at the playhead";
               MouseTo(bcenter(kfbtn("torso.angle"))))),
    LeftClick(), Wait(1.3),

    Lazy(_ -> (caption[] = "Scrub ahead…"; MouseTo(scrubpos(0.8T)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (caption[] = "…and turn it: a second key, and the curve lands on the clip";
               Wait(0.1))),
    dragparam("torso.angle", 1.2)...,
    Lazy(_ -> (beat!("figure keyframed", nkeys("torso.angle") > armed[],
                     "keys $(armed[]) → $(nkeys("torso.angle"))"); Wait(0.1))),

    # ------------------------------------------------------- keyframe the camera
    Lazy(_ -> (caption[] = "The camera is a parameter too"; MouseTo(bcenter(objfilter())))),
    LeftClick(), Wait(0.3),
    KeyDown(K.left_control), KeyPress(K.a), KeyUp(K.left_control), Wait(0.2),
    TypeText("camera"), Wait(0.5), KeyPress(K.enter), Wait(1.2),
    Lazy(_ -> (armed[] = nkeys("camera.eye[2]");
               caption[] = "◆ arms the eye position"; MouseTo(bcenter(kfbtn("camera.eye[2]"))))),
    LeftClick(), Wait(1.2),
    Lazy(_ -> (caption[] = "Back to the start…"; MouseTo(scrubpos(0.1T)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (caption[] = "…push the camera in — the move is keyed"; Wait(0.1))),
    dragparam("camera.eye[2]",
              sl -> sl.value[] + 0.25 * (last(sl.range[]) - first(sl.range[])))...,
    Lazy(_ -> (beat!("camera keyframed", nkeys("camera.eye[2]") > armed[],
                     "keys $(armed[]) → $(nkeys("camera.eye[2]"))"); Wait(0.1))),

    # ------------------------------------------------------- RayMakie preview
    Lazy(_ -> (caption[] = "Which renderer draws it live is a setting on the clip";
               MouseTo(bcenter(player.fxwidgets[:bakebutton])))),
    LeftClick(), WaitUntil(() -> haskey(player.fxwidgets, :rendermodal); timeout = 8.0),
    Wait(0.8),
    Lazy(_ -> (caption[] = "Preview with — RayMakie"; Wait(0.1))),
    pickmenu(() -> player.fxwidgets[:previewmenu], "RayMakie")...,
    Lazy(_ -> (beat!("preview switched", SRC.backend === :RayMakie, "backend = $(SRC.backend)");
               caption[] = "It refines while the playhead stands still — no sample limit";
               MouseTo(Point2f(1180, 500)))),
    # Short, because there is nothing to WATCH here. Measured on the recorded
    # frames: from sample 1 to sample 200 the picture keeps changing every frame,
    # but by a mean of 0.02 of one 8-bit level — this scene is already clean after
    # one sample, so nine seconds of it is nine seconds of a still image. The
    # refinement is real (live: samples 34 → 432 in eight seconds, 4 800–6 700
    # pixels changing per two seconds); it is simply not a thing this scene shows.
    WaitUntil(() -> SRC.samples > 60; timeout = 90.0), Wait(1.2),

    # ------------------------------------------------------- close, then an effect
    # The dialog is modal and dismisses on a backdrop click, which is the gesture
    # a person uses; a click inside the body would land on a setting.
    Lazy(_ -> (caption[] = ""; MouseTo(Point2f(210, 470)))),
    LeftClick(), WaitUntil(() -> !rmodal().modal.open[]; timeout = 8.0), Wait(0.6),
    Lazy(_ -> (beat!("dialog closed", !rmodal().modal.open[]);
               caption[] = "It is still an ordinary clip — put a Blur on it";
               MouseTo(dockmid()))),
    # The menu scrolls WITH the card list, so it can sit above the dock's top edge
    # where its `computedbbox` still reads plausibly and a click reaches nothing.
    # Wheel back up first; the Subfigure clamps, so an over-large delta is safe.
    Scroll((0.0, 40.0); duration = 0.5), Wait(0.5),
    pickmenu(() -> player.fxwidgets[:addeffect], "Blur")...,
    WaitUntil(() -> VE.findslot(scene_clip(), :blur) !== nothing; timeout = 20.0), Wait(1.0),
    Lazy(_ -> (beat!("blur added", VE.findslot(scene_clip(), :blur) !== nothing);
               caption[] = "…and it blurs the raytraced frame like any other picture";
               Wait(0.1))),
    Lazy(_ -> (sl = VE.param(VE.findslot(scene_clip(), :blur), :blur).view.control;
               MouseTo(slider_x(sl, sliderfrac(sl))))),
    LeftDown(), Wait(0.2),
    Lazy(_ -> (sl = VE.param(VE.findslot(scene_clip(), :blur), :blur).view.control;
               MouseTo(slider_x(sl, 0.45)))), Wait(0.4),
    LeftUp(), Wait(1.6),

    # ------------------------------------------------------- bake at N samples
    Lazy(_ -> (caption[] = "Bake it: render every frame to disk, once";
               MouseTo(bcenter(player.fxwidgets[:bakebutton])))),
    LeftClick(), WaitUntil(() -> rmodal().modal.open[]; timeout = 8.0), Wait(0.8),
    Lazy(_ -> (caption[] = "The Bake tab"; MouseTo(bcenter_rect(rmodal().tabs.tabs[2].rect[])))),
    LeftClick(), WaitUntil(() -> rmodal().tabs.active[] == 2; timeout = 8.0), Wait(1.2),
    Lazy(_ -> (beat!("bake tab", rmodal().tabs.active[] == 2,
                     "active = $(rmodal().tabs.active[])"); Wait(0.1))),
    Lazy(_ -> (beat!("samples field offered", haskey(player.fxwidgets[:bakeopts], :samples),
                     "bake settings: $(sort!(String.(collect(keys(player.fxwidgets[:bakeopts])))))");
               caption[] = "8 samples per frame — that is what one finished frame costs";
               Wait(0.1))),
    Lazy(_ -> MouseTo(bcenter(player.fxwidgets[:bakeopts][:samples]))), LeftClick(), Wait(0.4),
    TypeText("8"), Wait(0.6), KeyPress(K.enter), Wait(0.8),
    Lazy(_ -> (beat!("samples set", get(SRC.bakescreenopts, :samples, nothing) == 8,
                     "bakescreenopts = $(SRC.bakescreenopts)");
               caption[] = "Bake"; MouseTo(bcenter(player.fxwidgets[:bakego])))),
    LeftClick(), Wait(1.2),
    # Wait on the JOB. `jobprogress` is NaN when idle and 0..1 while one runs, so a
    # NaN AFTER a non-NaN is the completion edge — the bake's own files appear all
    # at once at the end, so watching the directory would report "not done"
    # throughout and then fail a moment before the frames land.
    WaitUntil(let ran = Ref(false)
        () -> (p = player.jobprogress[]; isnan(p) || (ran[] = true); ran[] && isnan(p))
    end; timeout = 240.0),
    Wait(2.0),
    Lazy(_ -> (beat!("baked", scene_clip().bake !== nothing,
                     scene_clip().bake === nothing ? "no bake" :
                     "$(length(scene_clip().bake.frames)) frames");
               caption[] = ""; MouseTo(Point2f(210, 470)))),
    LeftClick(), WaitUntil(() -> !rmodal().modal.open[]; timeout = 8.0), Wait(1.0),

    # ------------------------------------------------------- cut
    Lazy(_ -> (caption[] = "Cut it — S splits at the playhead"; MouseTo(scrubpos(0.5T)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (armed[] = length(seq.clips); MouseTo(clippos(scene_clip())))),
    LeftClick(), Wait(0.6),
    KeyPress(K.s), Wait(1.4),
    Lazy(_ -> (beat!("split", length(seq.clips) > armed[],
                     "clips $(armed[]) → $(length(seq.clips))"); Wait(0.1))),

    # ------------------------------------------------------- export
    Lazy(_ -> (caption[] = "Export — the animation, the blur, the bake and the cut";
               MouseTo(bcenter(button("Out"))))),
    LeftClick(), Wait(1.5),
    # Off camera, and the last thing that is: the path comes from a native file
    # dialog, which no synthetic event can reach. The button below is clicked.
    Lazy(_ -> (player.fxwidgets[:exportpath][] = EXPORTED;
               MouseTo(bcenter(player.fxwidgets[:exportgo])))),
    Wait(0.6), LeftClick(),
    WaitUntil(let ran = Ref(false)
        () -> (p = player.jobprogress[]; isnan(p) || (ran[] = true); ran[] && isnan(p))
    end; timeout = 240.0),
    Wait(2.0),
    Lazy(_ -> (beat!("export", isfile(EXPORTED) && filesize(EXPORTED) > 10_000,
                     isfile(EXPORTED) ? "$(filesize(EXPORTED)) bytes" : "no file");
               caption[] = "Done"; MouseTo(Point2f(1180, 500)))),
    Wait(1.5),
    Lazy(_ -> (caption[] = ""; Wait(0.1))), Wait(0.6),
]

FakeInteraction.interaction_record((i, t) -> tick!(t), fig, RAW_MP4, events;
                                   fps = 30, px_per_unit = 1)
result = (clips = length(seq.clips),
          baked = scene_clip() === nothing ? 0 : 0,
          exported = isfile(EXPORTED) ? filesize(EXPORTED) : 0)
close(player)
writecaptions(ASS, CAPTIONS, now_t[] + 2)
@info "caption track" file = ASS captions = length(CAPTIONS) sped_up = BUSY take_seconds = now_t[]
run(`$(FFMPEG_jll.ffmpeg()) -y -i $RAW_MP4 -vf $(speedupfilter(ASS, BUSY))
     -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
failed = [c for c in CHECKS if !c[2]]
@info "saved lego walkthrough" OUT_MP4 result checks = CHECKS
isempty(failed) || error("beats failed: $(join(first.(failed), ", "))")
