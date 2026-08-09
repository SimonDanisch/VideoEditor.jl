# The full walkthrough: one clip becomes two crows.
#
#   grade it → cut it in half → stack the halves → key the crow out of its
#   background → place it beside itself → export
#
# Every step on screen is a real mouse or keyboard event (FakeInteraction draws a
# live cursor). Off camera we do exactly two things, both noted at their line: cut
# a short proxy out of the 110 s source, and set the export path, which is a
# native file dialog and therefore not reachable by synthetic events.
#
# ## Why this script asserts on pixels
#
# An earlier version recorded seventy seconds in which the captions narrated an
# editor that was idle, and its checks all passed, because they read Julia state
# after the fact: "a ColorEffect exists with saturation 1.76" was true while the
# slider on screen had never moved and the picture had never changed colour. State
# is downstream of far too much to prove a gesture landed.
#
# So the beats here check the thing the viewer is being told to look at:
#
#   * grade   — the rendered preview's mean saturation and warmth, before against
#               after. Not the effect's parameters.
#   * matte   — foreground pixels in the propagated alpha.
#   * place   — the preview changes when the keyed copy moves.
#   * export  — a file on disk, with frames in it.
#
# `beat!` throws the moment one does not hold, so a take either shows the edit or
# stops; it can no longer run on to a pretty lie.
#
# ## The geometry, all measured rather than guessed
#
#   * the effects dock is **already open** on `:effects`. Clicking `FX` CLOSES it.
#   * a clip is clicked at `clippos(i)`, which asks `trackband` where lane `i`
#     actually is. The previous version used a hand-picked `yfrac = 0.9`, which is
#     a lane with one track and the **"+ new track" drop zone** with two — so
#     after stacking it selected nothing, `editclip` fell back to the bottom clip,
#     and the Matte card was built for the wrong clip. That one constant is why
#     the matte never happened.
#   * the crow sits at (0.38, 0.74) of the frame, measured off the proxy. The
#     previous (0.50, 0.34) landed on the wall beside it.
#   * `brightness` is an OFFSET in -0.5..0.5, not a multiplier.
#   * keying a subject is three clicks — "Mark subject", the crow, "Apply matte to
#     clip" — with a SCROLL in between, because marking grows the card and pushes
#     the Apply button out of the dock's visible area. `computedbbox` cannot tell
#     you that: it reports a layout position, and a click outside the Subfigure's
#     scene is swallowed. `ondock` is the check, and every dock button is put
#     through it.
#   * the third click starts a ~55 s propagation, and the clip is only keyed when
#     it finishes. Clicking the crow already puts a ONE-FRAME preview matte on the
#     clip, so "does a mattetrack exist" is true long before any of this is.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
if !haskey(ENV, "XAUTHORITY")
    xs = filter(f -> startswith(basename(f), "xauth_"), readdir("/run/user/1000"; join = true))
    isempty(xs) || (ENV["XAUTHORITY"] = last(sort(xs; by = mtime)))
end
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Makie, Statistics, Colors
import VideoEditor as VE
import FFMPEG_jll

isdefined(Main, :FakeInteraction) ||
    include(joinpath(@__DIR__, "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, WaitUntil, QuietUntil, MouseTo, LeftClick, LeftDown, LeftUp,
                        Lazy, KeyPress, KeyDown, KeyUp, Scroll

const SOURCE   = "/home/simon/Videos/test-jay-gpu.mp4"   # named jay; it is a hooded crow
const MEDIA    = normpath(joinpath(@__DIR__, "..", "..", "..", "media"))
const PROXY    = joinpath(MEDIA, "crow_proxy.mp4")
const RAW_MP4  = joinpath(tempdir(), "crow_raw.mp4")
const OUT_MP4  = joinpath(MEDIA, "crow_walkthrough.mp4")
const EXPORTED = joinpath(MEDIA, "crow_export.mp4")       # what the editor itself writes

# ------------------------------------------------------------------ off-camera
# 6 s at 720 px. The source is 110 s of 1080p60 and the matte propagates frame by
# frame, so the full clip would spend the demo waiting instead of showing.
mkpath(MEDIA)
isfile(PROXY) || run(pipeline(
    `$(FFMPEG_jll.ffmpeg()) -y -ss 2 -t 6 -i $SOURCE -vf scale=720:-2 -c:v libx264
     -g 30 -crf 20 -preset fast -an $PROXY`, stdout = devnull, stderr = devnull))
rm(EXPORTED; force = true)
# Both outputs go BEFORE the take, and this is not tidiness. `Makie.record` only
# copies its stream to `RAW_MP4` when the block completes, so a take that throws
# leaves the previous take's file sitting there with a plausible mtime — and the
# next reader converts it, watches it, and reports an hour-old run as this one's
# result. That happened twice. Absent files are an honest failure signal.
rm(RAW_MP4; force = true)
rm(OUT_MP4; force = true)

GLMakie.activate!(; visible = false, framerate = 30)
player = VE.Player(PROXY)
seq = player.sequence
fig = player.fig
resize!(fig, 1500, 950)
sleep(4.0)
# Hidden screen, but GLFW would still poll the real mouse and fight the synthetic
# cursor, and a focused window would eat real keystrokes.
# Cut the window off from the real device entirely. `visible = false` hides it but
# GLMakie keeps polling GLFW, so anything the machine's owner does while a take is
# running is delivered straight into the figure alongside the synthetic events —
# and a stray scroll is the worst of them, because it zooms an axis and silently
# invalidates every coordinate computed afterwards. One take was lost to exactly
# that: "when the window opened, i just zoomed in by accident, so this recording is
# borked". Disconnecting only `mouse_position` (which is all this used to do) stops
# the cursor fighting but leaves scroll, clicks and keys wide open.
for chan in (Makie.mouse_position, Makie.scroll, Makie.mouse_buttons,
             Makie.keyboard_buttons, Makie.unicode_input, Makie.dropped_files,
             Makie.hasfocus, Makie.entered_window)
    Makie.disconnect!(player.screen, chan)
end
fig.scene.events.hasfocus[] = false
@assert player.dockopen[] === :effects "the dock is expected open; do not click FX"

# ------------------------------------------------------------------ warm-up
# Off camera, and the third thing that is. Without it the take is 277 s long and
# **163 s of that is one frozen frame**: the first propagation in a process pays
# Julia specializing the propagator's graph (~99 s cold, and 163 s here because
# the CPU is encoding the recording at the same time), and SAM 2's first click
# builds a ~5 GB model. `freezedetect` put the still at 58.3 s → 221.6 s, 59% of
# the runtime, with not even the status bar repainting — the propagation blocks
# the UI thread outright.
#
# The editor already warms on "Mark subject" (`warmmattepanel!`), but it does it
# ASYNCHRONOUSLY so the user can keep choosing where to click — so the take's
# Apply simply queued behind the warm-up and paid the whole cost on camera.
# Paying it here, and WAITING, is the difference.
#
# Both calls must run on the analysis worker: the model builds a `BatchQueue`,
# which is single-writer and belongs to whichever thread first touches the
# context. Calling `sam2seed` from this thread dies with "BatchQueue is
# single-writer; cross-thread sweep forbidden". Warm at the resolution the tool
# will actually ask for — the cooperative-matrix GEMM specializes per tile shape.
let warmframe = fill(VE.RGB{VE.N0f8}(0.4, 0.5, 0.6), 720, 406), state = Ref{Any}(:pending)
    warmframe[300:420, 150:260] .= VE.RGB{VE.N0f8}(0.9, 0.3, 0.2)
    t0 = time()
    VE.runanalysis(player) do
        try
            VE.warmmatte!(480, 271)
            VE.sam2seed(warmframe, [(0.5, 0.5, true, 1)])
            state[] = :ok
        catch e
            state[] = (:err, sprint(showerror, e))
        end
    end
    while state[] === :pending && time() - t0 < 600
        sleep(1.0)
    end
    state[] === :ok || error("model warm-up did not finish: $(state[])")
    @info "models warmed off camera" seconds = round(time() - t0, digits = 1)
end
# Nothing may be decoding into the preview while the take runs: a playing clip
# competes with the matte for the GPU worker and puts stand-in frames on screen.
VE.pause!(player)

K = Makie.Keyboard
fps = seq.framerate
half = max(VE.seqlength(seq) ÷ 2, 30)

# ------------------------------------------------------------------- geometry
"Pixel point for axis time `t` (seconds) and axis-y `ay` (0..1)."
function tlpos(t; ay = 0.5)
    a = player.timeline.axis; lims = a.finallimits[]; vp = a.scene.viewport[]
    Point2f(vp.origin[1] + (Float64(t) - lims.origin[1]) / lims.widths[1] * vp.widths[1],
            vp.origin[2] + (Float64(ay) - lims.origin[2]) / lims.widths[2] * vp.widths[2])
end
"Centre of clip `i`, from its own lane — never a guessed fraction of the axis."
function clippos(i; tfrac = 0.5)
    c = seq.clips[i]; lo, hi = VE.trackband(c.track, VE.ntracks(seq))
    tlpos((c.start + tfrac * (VE.clipend(c) - c.start)) / seq.framerate; ay = (lo + hi) / 2)
end
"A point on the preview from frame fractions measured from the TOP left."
pvtop(fx, ft) = let vp = player.previewaxis.scene.viewport[]
    Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + (1 - ft) * vp.widths[2]) end
bcenter(b) = (bb = b.layoutobservables.computedbbox[]; Point2f(bb.origin .+ bb.widths ./ 2))
button(label) = first(b for b in fig.content if b isa Makie.Button && b.label[] == label)

# Looked up per use, never captured: the dock rebuilds its panel, and a captured
# Menu is a deleted block whose `computedbbox` still answers perfectly plausibly.
addfx() = player.fxwidgets[:addeffect]
addfx_center() = bcenter(addfx())
"Where `name`'s row lands once the dropdown is open — measured off its own scene."
function addfx_row(name)
    m = addfx()
    i = findfirst(o -> first(o) == name, m.options[])
    sc = m.blockscene.children[end]
    rects = sc.plots[1][1][]; tr = Makie.translation(sc)[]
    Point2f(sum(extrema(rects[i])) ./ 2 .+ Point2f(tr[1], tr[2]))
end
"""
Open the `+ Add effect…` menu and pick `name`, checking both halves land.

Two clicks that can each miss for different reasons — the menu is a fixed block
but its ROWS live in a popup scene that only exists while it is open — so the
open is asserted before the row position is computed off a scene that might be
something else entirely.
"""
function addfx_events(name)
    # How many cards the stack had before the pick — the card list growing is the
    # signal that the effect landed and its body is built. Waiting on THAT instead
    # of a fixed sleep is what keeps the video tight: a flat `Wait(1.8)` recorded
    # 2.5 s of an unchanging panel under "Add a Color grade" and 3.8 s under "the
    # second half needs its background gone", and would still have been too short
    # on a busier machine. A state change is both faster and safer than a guess.
    ncards = Ref(0)
    [
        # The menu scrolls WITH the card list; it is not a fixed header. By the
        # time the Transform is added the dock has been wheeled to the bottom to
        # reach the matte's Apply button, and the menu sits about a hundred px
        # ABOVE the panel's top edge — where its `computedbbox` still reads
        # perfectly plausibly and a click on it reaches nothing. Wheel back up
        # first; the Subfigure clamps, so an over-large delta is safe.
        Lazy(_ -> MouseTo(dockmid())),
        Scroll((0.0, 40.0); duration = 0.6),
        WaitUntil(() -> ondock(addfx()); timeout = 8.0), Wait(0.2),
        Lazy(_ -> begin
            v = FXSCROLL.scene.viewport[]
            beat!("$name menu reachable", ondock(addfx()),
                  "menu y = $(round(addfx().layoutobservables.computedbbox[].origin[2], digits = 1)), " *
                  "dock $(v.origin[2])..$(v.origin[2] + v.widths[2])")
            ncards[] = length(player.fxwidgets[:fxcards])
            MouseTo(addfx_center())
        end),
        LeftClick(),
        WaitUntil(() -> addfx().is_open[]; timeout = 8.0), Wait(0.25),
        Lazy(_ -> begin
            beat!("$name menu", addfx().is_open[], "is_open = $(addfx().is_open[])")
            MouseTo(addfx_row(name))
        end),
        LeftClick(),
        WaitUntil(() -> length(player.fxwidgets[:fxcards]) > ncards[]; timeout = 20.0),
        Wait(0.4),
    ]
end

# `paramform!` registers each card's sliders into `player.fxsliders` as it builds
# them and empties the registry on every rebuild, so this is always the live
# widget for the selected clip — looked up per drag, never captured.
formslider(name) = player.fxsliders[name]
sliderfrac(sl) = (r = sl.range[]; (sl.value[] - first(r)) / (last(r) - first(r)))
val2frac(sl, v) = (r = sl.range[]; (Float64(v) - first(r)) / (last(r) - first(r)))
function slider_x(sl, frac)
    bb = sl.layoutobservables.computedbbox[]
    Point2f(bb.origin[1] + clamp(frac, 0, 1) * bb.widths[1], bb.origin[2] + bb.widths[2] / 2)
end
"Press on the handle where it is, drag it to `v`, release. The preview follows."
drag(key, v) = [Lazy(_ -> (sl = formslider(key); MouseTo(slider_x(sl, sliderfrac(sl))))),
                LeftDown(), Wait(0.15),
                Lazy(_ -> (sl = formslider(key); MouseTo(slider_x(sl, val2frac(sl, v))))),
                Wait(0.1),
                Lazy(_ -> (sl = formslider(key); MouseTo(slider_x(sl, val2frac(sl, v))))),
                LeftUp(), Wait(0.4)]

const FXSCROLL = player.fxwidgets[:fxscroll]
dockmid() = (v = FXSCROLL.scene.viewport[]; Point2f(v.origin .+ v.widths ./ 2))
"""
Is `b` actually on screen inside the scrolling dock?

`computedbbox` reports where the LAYOUT puts a block whether or not the scroll has
it in view, and a click outside the Subfigure's scene reaches nothing at all — so
a button that has been pushed past the bottom edge looks perfectly clickable to
every coordinate this script computes, and swallows the click in silence.
"""
function ondock(b)
    b === nothing && return false
    v = FXSCROLL.scene.viewport[]; bb = b.layoutobservables.computedbbox[]
    bb.origin[2] >= v.origin[2] && bb.origin[2] + bb.widths[2] <= v.origin[2] + v.widths[2]
end

"The named Button inside the Matte card's tool slot, or `nothing`."
function toolbutton(label)
    slots = get(player.fxwidgets, :toolslots, nothing); slots === nothing && return nothing
    for sl in slots, (k, gl) in sl
        k === :matte || continue
        for gc in gl.content
            gc.content isa Makie.Button && gc.content.label[] == label && return gc.content
        end
    end
    nothing
end

# -------------------------------------------------------------------- checking
const CROW = (0.38, 0.74)          # measured off the proxy's own frame

"The rendered preview, cropped out of the figure's own colour buffer."
function previewshot()
    img = Makie.colorbuffer(player.screen)
    s = size(img, 2) / size(fig.scene)[1]          # the screen's px_per_unit
    figh = size(fig.scene)[2]
    vp = player.previewaxis.scene.viewport[]
    x0 = round(Int, vp.origin[1] * s) + 1; x1 = round(Int, (vp.origin[1] + vp.widths[1]) * s)
    ytop = round(Int, (figh - (vp.origin[2] + vp.widths[2])) * s) + 1
    ybot = round(Int, (figh - vp.origin[2]) * s)
    img[ytop:ybot, x0:x1]
end
"Mean HSV saturation and mean (R-B) warmth — what a grade is supposed to move."
function colorstats(a)
    rgb = RGB{Float32}.(a)
    # Rec.601 luma, and its spread. `contrast` as a standard deviation is what
    # separates a grade from a wash: brightness alone moves `lum` and leaves
    # `contrast` flat, which is how the first take looked bright and dead.
    lum = 0.299f0 .* getfield.(rgb, :r) .+ 0.587f0 .* getfield.(rgb, :g) .+
          0.114f0 .* getfield.(rgb, :b)
    (sat = mean(getfield.(HSV.(rgb), :s)),
     warm = mean(getfield.(rgb, :r) .- getfield.(rgb, :b)),
     lum = mean(lum), contrast = std(lum))
end
framediff(a, b) = size(a) == size(b) ?
    mean(abs.(Float32.(getfield.(RGB{Float32}.(a), :r)) .-
              Float32.(getfield.(RGB{Float32}.(b), :r)))) : 1.0f0

"""
Is there a real picture on the preview, or is the stream still catching up?

The GPU stream decodes a GOP at a time and presents black for one it has not
reached, so any playhead jump can land on an empty frame. Every measurement taken
against the preview has to gate on this: a take that did not measured its grade
twice against black (luminance 0.031 → 0.027) and failed its own beat while the
editor was working perfectly.
"""
haspicture() = mean(Float32.(getfield.(RGB{Float32}.(previewshot()), :r))) > 0.15

const CHECKS = Tuple{String, Bool, String}[]
"""
Record a beat's outcome. Loud on failure, but the take runs on to the end.

Throwing here aborts the `record` block, and `Makie.record` only writes the file
when its block completes — so a thrown beat produced NO video of the run that
failed, which is the one recording worth having. The run still fails: the script
throws after the mp4 is on disk, listing every beat that did not happen.
"""
function beat!(name, ok, detail = "")
    push!(CHECKS, (name, ok, detail))
    ok ? @info("beat ok", name, detail) : @error("BEAT FAILED", name, detail)
    ok
end
matteclip() = findfirst(c -> c.mattetrack !== nothing, seq.clips)

before_grade = Ref{Any}(nothing)
before_place = Ref{Any}(nothing)

caption = Observable("")
Makie.text!(fig.scene, caption; position = Point2f(750, 928), space = :pixel,
            align = (:center, :top), fontsize = 23, font = :bold, color = :white,
            strokecolor = RGBAf(0, 0, 0, 0.85), strokewidth = 2.5, overdraw = true)

events = [
    Wait(0.8),
    Lazy(_ -> (caption[] = "One clip, straight off the camera: a hooded crow";
               MouseTo(bcenter(button("Play"))))),
    LeftClick(), Wait(3.0), KeyPress(K.space), Wait(0.8),

    # -------------------------------------------------------------- grade it
    Lazy(_ -> (caption[] = "Pick the clip"; MouseTo(clippos(1)))),
    LeftClick(),
    # Selecting a clip moves the playhead into it, and the stream may need a moment
    # to present that frame. The grade is measured against this picture, so wait
    # for one — the alternative is a baseline of black, which is what failed.
    WaitUntil(haspicture; timeout = 30.0), Wait(0.5),
    Lazy(_ -> (before_grade[] = previewshot(); caption[] = "Add a Color grade"; Wait(0.2))),
    addfx_events("Color")...,
    # A grade, not a wash. The first take lifted BRIGHTNESS and left contrast
    # alone, which is the one combination that makes a picture worse: measured on
    # the preview it moved luminance +0.094 and contrast -0.002 — flatter and
    # paler, exactly the "washed out" look. Pulling brightness DOWN and pushing
    # contrast up gives luminance -0.068 and contrast +0.064 on the same frame,
    # so the greens deepen and the crow's black reads as black.
    # Both sliders stop at 55% of their travel (1.1 on a 0..2 range). An earlier
    # grade pushed saturation to 77% and contrast to 72% and read as garish: the
    # greens went electric and the warmth turned orange. At 55% the measured move
    # is saturation +0.104 and contrast +0.020, which is a grade you notice
    # without one you object to.
    Lazy(_ -> (caption[] = "Lift the saturation — the picture follows the drag"; Wait(0.2))),
    drag(:saturation, 1.1)..., Wait(0.8),
    Lazy(_ -> (caption[] = "…warm it up…"; Wait(0.2))),
    drag(:temperature, 0.20)..., Wait(0.8),
    Lazy(_ -> (caption[] = "…pull the brightness back…"; Wait(0.2))),
    drag(:brightness, -0.04)..., Wait(0.8),
    Lazy(_ -> (caption[] = "…and let the contrast carry it"; Wait(0.2))),
    drag(:contrast, 1.1)..., Wait(1.0),
    Lazy(_ -> begin
        a, b = colorstats(before_grade[]), colorstats(previewshot())
        # Thresholds sit just under what a 55% grade measures (saturation +0.104,
        # warmth +0.067, contrast +0.020, luminance -0.036) — tight enough that a
        # slider which did not move fails, loose enough to survive the frame the
        # playhead happens to land on. The contrast bound came DOWN from 0.03 when
        # the grade was softened: left as it was it would have failed a take that
        # is doing exactly what it is meant to.
        beat!("grade",
              b.sat - a.sat > 0.05 && b.warm - a.warm > 0.04 &&
              b.contrast - a.contrast > 0.01 && b.lum < a.lum,
              "saturation $(round(a.sat, digits=3)) → $(round(b.sat, digits=3)), " *
              "warmth $(round(a.warm, digits=3)) → $(round(b.warm, digits=3)), " *
              "contrast $(round(a.contrast, digits=3)) → $(round(b.contrast, digits=3)), " *
              "luminance $(round(a.lum, digits=3)) → $(round(b.lum, digits=3))")
        MouseTo(tlpos(half / fps; ay = 0.45))
    end), Wait(0.4),

    # ------------------------------------------------------------ cut in half
    Lazy(_ -> (caption[] = "Park the playhead halfway…"; MouseTo(tlpos(half / fps; ay = 0.45)))),
    LeftClick(), Wait(0.8),
    Lazy(_ -> (caption[] = "…and S cuts the clip in two"; Wait(0.2))),
    KeyPress(K.s), Wait(1.2),
    Lazy(_ -> (beat!("split", length(seq.clips) == 2, "clips = $(length(seq.clips))");
               MouseTo(clippos(2)))),

    # -------------------------------------------------------- key the crow
    # KEY BEFORE STACKING, and the order is the whole point.
    #
    # `showlivematte!` IS the marking feedback: it puts a one-frame matte on the
    # clip at strength 0.85 and re-presents, so the background drops out and what
    # is left is the subject. With the second half still on track 1 there is
    # nothing behind it, so the background goes dark and the selection is
    # unmistakable — measured as a 0.398 mean change across the preview.
    #
    # Key it AFTER stacking and the same code reveals the clip underneath, which
    # is the same footage a few seconds earlier: the preview barely changes (0.104,
    # and all of that is the second crow appearing), so the recording shows a green
    # dot and nothing else, and the tool looks like it did nothing. That is what
    # the first take looked like.
    #
    # The Matte card is built for the clip `editclip` returns — the SELECTED one
    # when the playhead is inside it — so select the second half first.
    Lazy(_ -> (caption[] = "The second half needs its background gone"; MouseTo(clippos(2)))),
    LeftClick(), Wait(1.0),
    addfx_events("Matte")...,
    Lazy(_ -> begin
        b = toolbutton("Mark subject")
        beat!("matte card", ondock(b),
              b === nothing ? "no Mark subject button" :
              ondock(b) ? "found, on screen" : "found but scrolled out of the dock")
        caption[] = "Mark the subject…"
        MouseTo(bcenter(b))
    end),
    LeftClick(), Wait(1.5),
    Lazy(_ -> begin
        col = try VE.mattecollect(player) catch; nothing end
        beat!("marking on", col !== nothing, col === nothing ? "no marking session" : "collector up")
        caption[] = "…click the crow. SAM 2.1 finds its edge on the GPU"
        MouseTo(pvtop(CROW...))
    end),
    Wait(0.8), LeftClick(),
    # The click only lays a green dot; the SELECTION — background dimmed, subject
    # held bright, the object's outline over it — arrives when SAM 2 comes back,
    # measured at 6.4 s warm. A flat `Wait(3.0)` moved the take on before any of
    # it appeared, so the recording showed a dot and then a scroll, and the whole
    # point of the tool was invisible. Wait for the seed, THEN hold on it: this is
    # what the viewer should SEE — so wait quietly and then hold on the result.
    # Left rolling it recorded 24.5 s of a frozen "matte: segmenting… 0%" footer:
    # 6.4 s warm in an idle session, four times that under the recording, and the
    # UI does not animate through any of it. Cut across the wait, then sit on the
    # selection long enough to read it.
    QuietUntil(() -> (c = try VE.mattecollect(player) catch; nothing end;
                      c !== nothing && c.lastseed !== nothing && !c.busy);
               timeout = 120.0),
    Wait(3.0),
    Lazy(_ -> begin
        col = try VE.mattecollect(player) catch; nothing end
        n = col === nothing ? 0 : length(col.points)
        seeded = col !== nothing && col.lastseed !== nothing
        beat!("point marked", n == 1 && seeded,
              "points = $n at $(CROW), seed = " *
              (seeded ? "$(round(100 * count(!=(0x00), col.lastseed) / length(col.lastseed), digits=1))% of frame" :
               "NONE — the live selection never appeared"))
        caption[] = "Scroll down to the Apply button"
        MouseTo(dockmid())
    end), Wait(0.4),
    # Marking GROWS the Matte card — a row per marked frame, plus the live one —
    # and pushes its footer button below the dock's visible area. The take that
    # skipped this sat for 240 s with the marking session still open and the
    # status stuck on "1 point(s) on 1 object(s), frame 270", because every click
    # went to a layout position that was off screen. Scroll to the bottom (the
    # Subfigure clamps, so an over-large delta is safe) and then check.
    Scroll((0.0, -40.0); duration = 1.0), Wait(0.8),
    Lazy(_ -> begin
        b = toolbutton("Apply matte to clip")
        beat!("apply reachable", ondock(b),
              b === nothing ? "no Apply button" :
              "button y = $(round(b.layoutobservables.computedbbox[].origin[2], digits = 1)), " *
              "dock starts at y = $(FXSCROLL.scene.viewport[].origin[2])")
        caption[] = "Apply it — MatAnyone carries the matte across the clip, on the GPU"
        MouseTo(bcenter(b))
    end),
    Wait(0.6), LeftClick(),
    # One press is enough: `applymattenow!` commits the marking session, and
    # committing runs `addmatteseed!`, which propagates. What is NOT enough is
    # waiting for `mattetrack !== nothing` — the live preview mask puts a
    # ONE-FRAME track on the clip the moment the crow is clicked, so that
    # predicate is already true before Apply is pressed. The first take of this
    # script waited on it, moved on ~20 s into a ~55 s propagation, and exported
    # a composite in which the top copy was a full rectangle on every frame but
    # the marked one, under a caption claiming the matte had been carried across.
    # Wait for the track to span the clip; nothing weaker distinguishes the two.
    #
    # Forward from the seed and backward over what comes before, one frame at a
    # time. Condition-based, so the take neither cuts it short nor sits on a fixed
    # sleep after it is done.
    #
    # QUIET, so the propagation costs the viewer nothing. It pins the UI thread
    # outright — not even the status bar repaints — so recording through it adds
    # one unchanging frame per tick and nothing else. Left rolling it was 20 s of
    # still image here, and 163 s before the models were warmed off camera.
    QuietUntil(() -> (t = seq.clips[2].mattetrack;
                      t !== nothing && size(t.alpha, 3) >= VE.srclength(seq.clips[2]));
               timeout = 300.0),
    Wait(1.5),
    Lazy(_ -> begin
        c = seq.clips[2]; tr = c.mattetrack
        nfr = tr === nothing ? 0 : size(tr.alpha, 3)
        want = VE.srclength(c)
        # Foreground on the FIRST frame, which is ~90 frames from the seed: a
        # count over the whole array cannot tell a propagated track from a single
        # seeded frame, and that is precisely the bug this beat missed once.
        fg1 = tr === nothing ? 0 : count(>(0x80), view(tr.alpha, :, :, 1))
        beat!("matte", nfr == want && fg1 > 200,
              tr === nothing ? "no mattetrack" :
              "$nfr of $want frames, foreground on frame 1 = $fg1 px")
        caption[] = "Now the second half is only the crow"
        MouseTo(clippos(2))
    end), Wait(0.8),

    # ----------------------------------------------------- stack the halves
    # Ctrl-drag the keyed half up past the top lane, into the strip that means
    # "a new track above". `liftthreshold` wants the gesture to clear the lane, so
    # this walks all the way to axis-y 0.985 rather than jumping.
    Lazy(_ -> (caption[] = "Ctrl-drag it…"; MouseTo(clippos(2)))),
    KeyDown(K.left_control), LeftDown(), Wait(0.5),
    Lazy(_ -> (caption[] = "…onto a track of its own, over the first";
               MouseTo(tlpos(half / fps; ay = 0.70)))), Wait(0.4),
    Lazy(_ -> MouseTo(tlpos(half / 2 / fps; ay = 0.985))), Wait(0.5),
    Lazy(_ -> MouseTo(tlpos(half / 2 / fps; ay = 0.985))), Wait(0.3),
    LeftUp(), KeyUp(K.left_control), Wait(1.0),
    Lazy(_ -> begin
        beat!("stack", VE.ntracks(seq) == 2 && seq.clips[2].track == 2,
              "tracks = $(VE.ntracks(seq)), clip tracks = $([c.track for c in seq.clips])")
        # No caption yet. The dragged clip is off the timeline during the gesture
        # and the two-track composite is not presented until the clip is selected
        # again, so the frame here is still the keyed crow on black. Claiming
        # "two crows" over it put the payoff line on an empty picture.
        caption[] = ""
        MouseTo(clippos(2))
    end),
    LeftClick(),
    # THIS is the state change worth waiting on, and it is a picture, not a flag:
    # selecting the stacked clip is what makes the composite present, and the
    # preview jumps from 0.114 (keyed crow on black) to 0.358 (the lower track
    # filling the frame) when it does. 0.25 sits between the two with room either
    # side. An earlier attempt waited on this BEFORE the click — the thing it was
    # waiting for could not happen yet, so it spun its whole 20 s timeout with the
    # camera rolling and made the take 19 s longer instead of shorter.
    WaitUntil(() -> mean(Float32.(getfield.(RGB{Float32}.(previewshot()), :r))) > 0.25;
              timeout = 20.0),
    Wait(0.4),
    Lazy(_ -> begin
        # Baseline for the place beat, taken once the composite is on screen, so
        # the beat measures the drag and not the track move.
        before_place[] = previewshot()
        caption[] = "Two crows now — one keyed over the other"
        Wait(0.2)
    end), Wait(1.2),

    # ------------------------------------------ place it beside the other
    # The dock is still scrolled to the bottom from the Apply above, and the menu
    # went off the top with it — `addfx_events` wheels back up before it clicks.
    # The take that assumed the menu was a fixed header died right here, with
    # `is_open = false` and every later beat unreachable.
    addfx_events("Transform")...,
    Lazy(_ -> (caption[] = "Drag it: a second crow, perched above the first";
               MouseTo(pvtop(CROW...)))),
    LeftDown(), Wait(0.4),
    Lazy(_ -> MouseTo(pvtop(CROW[1] + 0.17, CROW[2] - 0.15))), Wait(0.4),
    Lazy(_ -> MouseTo(pvtop(CROW[1] + 0.34, CROW[2] - 0.30))), Wait(0.4),
    LeftUp(), Wait(1.2),
    Lazy(_ -> begin
        c = seq.clips[2]
        t = findfirst(e -> e.effect isa VE.TransformEffect, c.effects)
        moved = framediff(before_place[], previewshot())
        beat!("place", t !== nothing && moved > 0.005,
              t === nothing ? "no TransformEffect" :
              "position $(player.fxsliders[:pos_x].value[]), $(player.fxsliders[:pos_y].value[]); " *
              "preview moved by $(round(moved, digits=4))")
        caption[] = "Two crows, one clip"
        MouseTo(bcenter(button("Play")))
    end),
    LeftClick(), Wait(4.0), KeyPress(K.space), Wait(1.0),

    # ------------------------------------------------------------------ export
    Lazy(_ -> (caption[] = "Export — the cut, the grade and the matte, muxed out";
               MouseTo(bcenter(button("Out"))))),
    LeftClick(), Wait(1.5),
    # Off camera, and the only other thing that is: the path comes from a native
    # file dialog, which no synthetic event can reach. The button below is clicked.
    Lazy(_ -> (player.fxwidgets[:exportpath][] = EXPORTED;
               MouseTo(bcenter(player.fxwidgets[:exportgo])))),
    Wait(0.6), LeftClick(),
    # Wait on the JOB, not on the file. `exportvideo` hands ffmpeg a pipe and the
    # muxer flushes at the end, so the file sits at 48 bytes for the whole encode
    # and then jumps to its full size — a `filesize > 10_000` predicate reports
    # "not done" throughout and then fails the beat two seconds before the bytes
    # land. Seen exactly that: the beat said "48 bytes", the result tuple built
    # moments later said 769828. `jobprogress` is NaN when idle, 0..1 while a job
    # runs, so a NaN AFTER a non-NaN is the completion edge.
    # Quiet too: the encode is a progress bar and a still preview, and it ran 14 s.
    QuietUntil(let ran = Ref(false)
        () -> begin
            p = player.jobprogress[]
            isnan(p) || (ran[] = true)
            ran[] && isnan(p)
        end
    end; timeout = 240.0),
    Wait(2.0),
    Lazy(_ -> begin
        beat!("export", isfile(EXPORTED) && filesize(EXPORTED) > 10_000,
              isfile(EXPORTED) ? "$(filesize(EXPORTED)) bytes" : "no file")
        caption[] = ""
        MouseTo(pvtop(0.5, 0.5))
    end), Wait(0.8),
]

"Per-frame sample of what the matte is doing, so a stall has a record."
const TRACE = Tuple{Float64, String, String}[]
function trace!(i, t)
    i % 30 == 0 || return nothing
    tr = length(seq.clips) >= 2 ? seq.clips[2].mattetrack : nothing
    push!(TRACE, (round(t, digits = 1), string(player.matteinfo[]),
                  tr === nothing ? "none" : string(size(tr.alpha, 3))))
    return nothing
end

# Settle the preview before the camera rolls. The warm-up above owns the analysis
# worker for its whole run, and the streaming GPU decode is fed from there — so
# the take opened on a BLACK preview: the parked first frame was fine, and the
# moment Play was clicked the stream started a feed it had had no chance to prime
# and presented black for ~2 s under the caption "One clip, straight off the
# camera". Play it here instead, off camera, so the GOP is decoded and cached,
# then park at the head and wait for real pixels.
let t0 = time()
    # Walk the WHOLE clip, not just the first few seconds. The stream decodes a
    # GOP at a time and presents black for one it has not reached, so priming only
    # the head leaves every later jump — clicking a clip, parking the playhead at
    # the split — landing on an undecoded region. That is not hypothetical: a take
    # primed with a 4 s play measured the grade against a BLACK preview twice over
    # (luminance 0.031 → 0.027) and failed its own beat, because both the before
    # and the after frame were empty.
    n = VE.seqlength(seq)
    for f in round.(Int, range(0, n - 1; length = 16))
        player.playhead[] = f
        sleep(0.35)
    end
    player.playhead[] = 0
    while time() - t0 < 120
        sleep(0.5)
        mean(Float32.(getfield.(RGB{Float32}.(previewshot()), :r))) > 0.15 && break
    end
    @info "preview settled" seconds = round(time() - t0, digits = 1)
end

try
    FakeInteraction.interaction_record(trace!, fig, RAW_MP4, events;
                                       fps = 30, px_per_unit = 1)
finally
    close(player)
    # `RAW_MP4` was deleted before the take, so it exists here only if `record`
    # completed and wrote it. Converting whatever happened to be at that path is
    # how an hour-old take got reviewed as this one's.
    isfile(RAW_MP4) && run(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error -i $RAW_MP4
                            -c:v libx264 -crf 22 -pix_fmt yuv420p $OUT_MP4`)
end
result = (checks = CHECKS,
          ntracks = VE.ntracks(seq), nclips = length(seq.clips),
          matted = count(c -> c.mattetrack !== nothing, seq.clips),
          effects = [length(c.effects) for c in seq.clips],
          exported = isfile(EXPORTED) ? filesize(EXPORTED) : 0)
@info "saved the crow walkthrough" OUT_MP4 result
foreach(c -> println(c[2] ? "  ok    " : "  FAILED", "  ", c[1], "   ", c[3]), CHECKS)

# The take is on disk by now, so a failure here still leaves the recording that
# shows what went wrong — but the run does fail, and names every beat, so nobody
# has to read a beat log to notice.
failed = [c for c in CHECKS if !c[2]]
isempty(failed) || error("$(length(failed)) of $(length(CHECKS)) beats did not happen: " *
                         join(["$(n) ($(d))" for (n, _, d) in failed], "; "))
result
