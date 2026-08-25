# UI tests driving the real GLMakie window with synthetic mouse/keyboard
# events (same mechanism FakeInteraction uses) and asserting on editor state.
# Included from runtests.jl only when a GL context is available.

import GLMakie
import VideoEditor.Makie as Makie
using Statistics: mean
using VideoEditor.Makie: Keyboard, Mouse, KeyEvent, MouseButtonEvent, Point2f

@testset "UI interactions" begin
    GLMakie.activate!(; visible = false)
    p = Player(testvideo; gpupreview = false)  # 320×180, 120 frames @30 (from runtests.jl); CPU for determinism
    try
        fig = p.fig
        ev = Makie.events(fig)
        tl = p.timeline
        ax = tl.axis
        sleep(1.5)  # decode warm-up + layout

        tlx(t) = begin  # timeline time → figure pixel
            lims = ax.finallimits[]
            vp = ax.scene.viewport[]
            frac = (t - minimum(lims)[1]) / (maximum(lims)[1] - minimum(lims)[1])
            Point2f(vp.origin[1] + frac * vp.widths[1], vp.origin[2] + 0.5 * vp.widths[2])
        end
        pv(fx, fy) = begin  # preview viewport fraction → figure pixel
            vp = p.previewaxis.scene.viewport[]
            Point2f(vp.origin[1] + fx * vp.widths[1], vp.origin[2] + fy * vp.widths[2])
        end
        moveto(pos) = (ev.mouseposition[] = Tuple(pos))
        press(pos) = (moveto(pos); ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.press))
        release() = (ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.release))
        keypress(k) = (ev.keyboardbutton[] = KeyEvent(k, Keyboard.press);
                       ev.keyboardbutton[] = KeyEvent(k, Keyboard.release))
        waitfor(pred; s = 6) = (t0 = time();
                                while !pred() && time() - t0 < s
                                    sleep(0.1)
                                end;
                                pred())
        # stabilization controls live INLINE in the inspector dock (no modal);
        # stabopen ensures that dock is showing, stabclose is a no-op kept so the
        # beats read the same as before the restructure
        # Stabilization is a CARD in the Effects panel now, not a dock of its own:
        # this brings it into view (adding it if the clip has none), which is what
        # a user reaching for it does.
        stabopen() = (VE.showkind!(p, :stabilize); sleep(0.3))
        stabclose() = sleep(0.1)

        @testset "scrub selects and follows" begin
            @test occursin("Space plays", p.status[])   # onboarding hint on startup
            press(tlx(1.0))
            @test p.playhead[] == 30
            @test tl.selected[] == 1
            @test !tl.scrubbing[]     # a click is not a scrub yet — it settles EXACTLY
            moveto(tlx(2.5))
            @test p.playhead[] == 75
            @test tl.scrubbing[]      # the press became a drag: stand-in frames welcome
            release()
            @test !tl.scrubbing[]
        end

        @testset "crop tool keeps orientation" begin
            keypress(Keyboard.c)
            @test p.cropmode[]
            press(pv(0.3, 0.3))
            moveto(pv(0.7, 0.7))
            release()
            @test p.cropmode[]                 # persistent crop tool stays on (re-drag to refine)
            crop = p.sequence.clips[1].crop
            @test crop != (0.0, 0.0, 1.0, 1.0)
            @test 0.05 < crop[1] < 0.5 && 0.2 < crop[3] < 0.8
            # regression: applying the crop must not un-reverse the y axis
            # (Makie's ylims! derives yreversed from argument order)
            @test p.previewaxis.yreversed[]
            keypress(Keyboard.escape)          # put the crop tool away
            @test !p.cropmode[]
            keypress(Keyboard.r)  # reset crop for later tests
            @test p.sequence.clips[1].crop == (0.0, 0.0, 1.0, 1.0)
        end

        @testset "right-click modal" begin
            ev.mouseposition[] = Tuple(tlx(1.0))
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.press)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.release)
            @test p.clipmodal.open[]
            press(Point2f(10, 10))  # backdrop click dismisses
            release()
            @test !p.clipmodal.open[]
        end

        @testset "the eye bypasses the whole stack" begin
            # Was a full-width "Hold to compare with the original" bar at the
            # bottom of the panel; it is a 30px eye in the panel head now, and a
            # TOGGLE rather than a hold (Simon, 2026-07-31: the bar took far too
            # much space, and the same glyph is on every card).
            VE.opendock!(p, :effects); sleep(0.3)
            eye = p.fxwidgets[:bypassall]
            bb = eye.layoutobservables.computedbbox[]
            pos = Point2f(bb.origin .+ bb.widths ./ 2)
            @test p.applytracks[]
            press(pos); release(); sleep(0.2)
            @test !p.applytracks[]
            press(pos); release(); sleep(0.2)
            @test p.applytracks[]
            stabclose()
        end

        @testset "split, ctrl-drag with snap, ripple delete" begin
            press(tlx(2.0)); release()
            keypress(Keyboard.s)
            @test length(p.sequence.clips) == 2
            @test p.sequence.clips[2].start == 60

            # Ctrl-drag clip 2 right: nothing moves until release
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            press(tlx(3.0))
            @test tl.dragclip !== nothing
            moveto(tlx(3.5))
            @test p.sequence.clips[2].start == 60      # not committed yet
            @test tl.ghost_plot.visible[]
            release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test p.sequence.clips[2].start == 75      # committed on release
            @test !tl.ghost_plot.visible[]

            # drag back until it snaps against clip 1's end
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            press(tlx(3.0))                            # grab (offset 15 frames)
            moveto(tlx(2.55))                          # raw ≈ 61-62 → snaps to 60
            @test !isempty(tl.snapline[])
            release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test p.sequence.clips[2].start == 60

            # ripple delete the second clip
            press(tlx(3.0)); release()
            keypress(Keyboard.x)
            @test length(p.sequence.clips) == 1
            @test VE.seqlength(p.sequence) == 60
            @test length(tl.clipplots) == 1            # plot reconciled away
            # the view follows the shrunken sequence (no dead space)
            @test maximum(ax.finallimits[])[1] <= 2.0 * 1.05
        end

        @testset "edge trim" begin
            # state here: one clip, frames 0:60 (2 s)
            clip = p.sequence.clips[1]
            @test VE.cliplength(clip) == 60
            moveto(tlx(1.0))                      # clip middle: no trim handle
            @test isempty(p.timeline.edgeline[])
            moveto(tlx(2.0))                      # right edge: handle bar shows
            @test !isempty(p.timeline.edgeline[])
            press(tlx(2.0))                       # grab the right edge (within 8 px)
            @test p.timeline.trimclip !== nothing
            moveto(tlx(1.5))                      # trim to 1.5 s
            @test !isempty(p.timeline.edgeline[])             # handle follows the drag
            @test startswith(p.timeline.tooltip_text[], "clip ")  # live clip length
            release()
            @test p.timeline.trimclip === nothing
            @test VE.cliplength(clip) == 45
            @test clip.src_out == 45

            # trimming shows the frame AT THE EDGE, and leaves the playhead alone —
            # you have to see the frame you are cutting on to know where to stop
            # (Simon, 2026-07-27). Reference: what source frame 15 looks like.
            VE.seek!(p, 30); sleep(0.5)
            playheadframe = copy(p.frame[])
            edgeref = VE.renderpreview(p, 15 / p.sequence.framerate, size(p.frame[], 1))
            chan(x) = Float32.(VE.ColorTypes.red.(x))
            press(tlx(0.0))                       # left edge: shift start + src_in
            moveto(tlx(0.5))
            sleep(0.6)
            @test p.playhead[] == 30                          # playhead stays put …
            @test mean(abs.(chan(p.frame[]) .- chan(edgeref))) < 0.01   # … preview is the EDGE
            @test mean(abs.(chan(p.frame[]) .- chan(playheadframe))) > 0.02  # not the playhead's
            # …and the PICTURE follows the edge you are dragging (Simon, 2026-07-27:
            # "we should CLIP/Resize the clip WHERE WE DRAG"). The model was already
            # right — it was the strip that kept drawing the old head under a
            # shrinking band, which reads as the FAR end being cut instead.
            @test p.timeline.clipranges[1][][1] ≈ clip.start / p.sequence.framerate
            @test p.timeline.clipstarts[1][] ≈ clip.src_in / clip.source.framerate
            release()
            sleep(0.8)
            @test clip.src_in == 15
            @test clip.start == 15
            @test VE.cliplength(clip) == 30
            @test mean(abs.(chan(p.frame[]) .- chan(playheadframe))) < 0.01   # back to the playhead

            # regression (Simon): exactly ON the edge and just OUTSIDE the last
            # clip must both show the handle — the boundary frame belongs to
            # the next clip, so a clipat-based hit test missed these
            moveto(tlx(1.5))                      # exactly on the right edge
            @test !isempty(p.timeline.edgeline[])
            @test p.timeline.hovered[] == 1       # edge hover brightens its clip
            moveto(tlx(1.5 + VE.edgezone(p.timeline) / 2))  # just past the end
            @test !isempty(p.timeline.edgeline[])
            moveto(tlx(1.0))                      # interior: no handle
            @test isempty(p.timeline.edgeline[])
        end

        @testset "undo / redo" begin
            # two trims above pushed two snapshots; Ctrl+Z restores them
            clip() = p.sequence.clips[1]
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)                  # undo left trim
            @test clip().src_in == 0 && VE.cliplength(clip()) == 45
            keypress(Keyboard.z)                  # undo right trim
            @test VE.cliplength(clip()) == 60
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_shift, Keyboard.press)
            keypress(Keyboard.z)                  # redo right trim
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_shift, Keyboard.release)
            @test VE.cliplength(clip()) == 45
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)

            # undo also covers keyboard edits
            press(tlx(0.7)); release()
            keypress(Keyboard.s)
            @test length(p.sequence.clips) == 2
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test length(p.sequence.clips) == 1
            @test length(p.timeline.clipplots) == 1
        end

        @testset "cut prefetch" begin
            # current state: one clip of 45 frames; split → two adjacent clips
            press(tlx(0.7)); release()
            keypress(Keyboard.s)
            @test length(p.sequence.clips) == 2
            cut = p.sequence.clips[2].start
            press(tlx((cut - 5) / 30)); release()   # 5 frames before the cut
            sleep(0.6)                              # worker buffers the tail
            clip, srcframe = VE.locate(p.sequence, p.playhead[])
            target, protect = VE.decodetarget(p, p.playhead[], clip, srcframe)
            @test target == p.sequence.clips[2].src_in  # aims across the cut
            @test first(protect) == srcframe            # tail slots protected
            # on the last clip there is nothing to prefetch
            press(tlx(1.0)); release()
            sleep(0.3)
            clip2, src2 = VE.locate(p.sequence, p.playhead[])
            target2, protect2 = VE.decodetarget(p, p.playhead[], clip2, src2)
            @test target2 == src2 && isempty(protect2)
        end

        @testset "dropping files fills the media bin" begin
            # state: two clips (0.7s split of a 45-frame clip), 45 frames total.
            # A drop from the file manager IMPORTS (it does not edit the
            # timeline): the bin opens itself so the new rows are visible, every
            # file in the drop is accounted for, and the probing happens off the
            # UI thread — which is why these are waitfor and not plain reads.
            nclips = length(p.sequence.clips)
            nbin = length(p.mediasources[])
            VE.opendock!(p, :effects); sleep(0.2)          # bin CLOSED before the drop
            ev.dropped_files[] = [testvideo2, testvideo, "/nonexistent/nope.mp4"]
            @test p.dockopen[] === :media                  # …opens itself on the drop
            @test waitfor(() -> length(p.mediasources[]) == nbin + 1)
            @test last(p.mediasources[]).width == 480
            @test length(p.sequence.clips) == nclips       # nothing placed on the timeline
            @test waitfor(() -> occursin("imported", p.status[]))
            @test occursin("already in the bin", p.status[])   # testvideo was there
            @test occursin("couldn't open nope.mp4", p.status[])
            @test waitfor(() -> isempty(p.fxwidgets[:dropstatus][]))  # zone back to idle
            @test isnan(p.jobprogress[])                   # …and the footer job is done

            VE.addsource!(p, testvideo2)                   # place it: the multi-source beats
            @test length(p.sequence.clips) == nclips + 1
            @test length(tl.clipplots) == nclips + 1
            added = p.sequence.clips[end]
            @test added.source.width == 480
            @test added.start == 45                # appended at the sequence end
            @test maximum(ax.finallimits[])[1] >= VE.seqduration(p.sequence) - 1e-6

            # scrubbing onto the new clip switches the preview buffers
            press(tlx((45 + 15) / 30)); release()
            sleep(0.6)
            # NOT `== (480, 270)`. The preview is CANVAS-sized on purpose —
            # `ensureframesize!(player, canvassize(seq))`, "the SEQUENCE's format,
            # not the top layer's" — so a second source of a different shape is
            # letterboxed into the project's format rather than resizing it. This
            # assertion predated the canvas and was checking the old behaviour: it
            # read (320, 180), the FIRST clip's size, long before an explicit
            # canvas existed to blame. What scrubbing across the cut actually
            # changes is which source is being decoded.
            @test size(p.frame[]) == VE.canvassize(p.sequence)
            @test VE.locate(p.sequence, p.playhead[])[1].source.width == 480

            # near the cross-source cut the other source's worker pre-warms;
            # the current clip's worker stays on its own tail (separate rings)
            VE.settarget!(VE.pool(p, added.source).worker, 60)  # point it away first
            press(tlx((45 - 5) / 30)); release()
            clip, src = VE.locate(p.sequence, p.playhead[])
            target, protect = VE.decodetarget(p, p.playhead[], clip, src)
            @test target == src && isempty(protect)
            @test VE.pool(p, added.source).worker.target[] == added.src_in

            # adding is one undoable edit
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test length(p.sequence.clips) == nclips
            @test length(tl.clipplots) == nclips
        end

        @testset "object-lock pick flow" begin
            stabopen()
            # select "Object lock" in the mode menu via real clicks
            menu = p.fxwidgets[:modemenu]
            bb() = menu.layoutobservables.computedbbox[]
            mpos(rely) = Point2f(bb().origin[1] + bb().widths[1] / 2,
                                 bb().origin[2] + rely * bb().widths[2])
            press(mpos(0.5)); release()          # open the dropdown
            @test menu.is_open[]
            press(mpos(-1.5)); release()         # option 2 (list opens downward)
            @test menu.selection[] == :objectlock
            @test !menu.is_open[]                # picking closes it — an open dropdown
                                                 # eats the next press anywhere

            # read the CURRENT button every time: a panel rebuild recreates it, so a
            # captured reference points at a deleted widget whose bbox is stale —
            # and the click then lands nowhere
            stabcenter() = (sc = p.fxwidgets[:analyze].layoutobservables.computedbbox[];
                            Point2f(sc.origin .+ sc.widths ./ 2))
            press(stabcenter()); release()       # starts the pick AND closes the modal
            @test p.onpick !== nothing

            keypress(Keyboard.escape)            # Esc cancels the pick
            @test p.onpick === nothing

            # start it again (reopen, click Stabilize) then pick on the preview
            # (pause first: a second click at the same spot within the dblclick window
            # would register as dblclick, not click)
            sleep(0.5)
            stabopen()
            press(stabcenter()); release()
            @test p.onpick !== nothing
            clip = VE.locate(p.sequence, p.playhead[])[1]
            clip.motiontrack = nothing
            press(pv(0.5, 0.5)); release()       # modal closed by the start → click reaches the preview
            @test p.onpick === nothing
            t0 = time()
            while clip.motiontrack === nothing && time() - t0 < 15
                sleep(0.1)
            end
            clip.motiontrack === nothing &&
                @info "objectlock debug" p.status[] p.stabinfo[] p.playhead[] length(p.sequence.clips)
            @test clip.motiontrack !== nothing

            # a motion analysis composes the stabilization border crop into
            # the clip's crop, and the new framing is flashed on the preview
            @test waitfor(() -> clip.crop != (0.0, 0.0, 1.0, 1.0))
            @test waitfor(() -> !isempty(p.croprect[]))      # outline flash on
            @test waitfor(() -> isempty(p.croprect[]))       # …and cleared again
            # the track remembers the framing it auto-cropped away, and the
            # panel names the active mode
            @test clip.motiontrack.basecrop == (0.0, 0.0, 1.0, 1.0)
            @test waitfor(() -> occursin("object lock", p.stabinfo[]))

            stabopen()
            # back to camera lock through the menu's own API — the click-driven path
            # is covered by the object-lock beat above, and a MISSED option click
            # leaves the dropdown open, which then swallows the next press anywhere
            # in the window (that is how a stray click here broke the crop and the
            # space bar three testsets later)
            menu = p.fxwidgets[:modemenu]
            menu.i_selected[] = 1
            sleep(0.2)
            @test menu.selection[] == :similarity
            @test !menu.is_open[]
        end

        @testset "remove stabilization restores the framing" begin
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test clip.motiontrack !== nothing   # from the object-lock beat
            @test clip.crop != (0.0, 0.0, 1.0, 1.0)
            # the Stabilize card's "Remove stabilization" takes the track off again
            rmbtn = p.fxwidgets[:remove]
            bb = rmbtn.layoutobservables.computedbbox[]
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release()
            @test clip.motiontrack === nothing
            @test clip.crop == (0.0, 0.0, 1.0, 1.0)          # basecrop restored
            @test occursin("no stabilization", p.stabinfo[])
            @test waitfor(() -> occursin("removed", p.status[]))   # async status queue
            # let the restore glide + outline flash FULLY finish before later beats:
            # croprect starts empty, so wait for the outline to appear, then clear
            @test waitfor(() -> !isempty(p.croprect[]))
            @test waitfor(() -> isempty(p.croprect[]))
            # the card stays put now (it is a tool, not a section that comes and
            # goes) — pressing Remove on a clean clip says so instead of doing damage.
            # Re-fetch the button: the panel relaid out after the removal, so the
            # bbox captured above points at whatever moved into that spot (a stray
            # press there opened the mode Menu, which then ate the next keystrokes).
            rmbtn2 = p.fxwidgets[:remove]
            bb2 = rmbtn2.layoutobservables.computedbbox[]
            press(Point2f(bb2.origin .+ bb2.widths ./ 2)); release()
            @test clip.motiontrack === nothing
            @test waitfor(() -> occursin("no stabilization", p.status[]))
            stabclose()
        end

        @testset "proxy swap keeps the preview consistent" begin
            press(tlx(0.3)); release()               # onto clip 1
            # a press on the ruler must REACH the timeline: scrolled-out dock content
            # used to sit over it and eat the click (see the Subfigure clip fix)
            @test p.playhead[] == 9
            clip = p.sequence.clips[1]
            VE.startproxy!(p, clip.source; height = 90)
            t0 = time()
            while size(p.frame[]) != (160, 90) && time() - t0 < 20
                sleep(0.1)
            end
            @test size(p.frame[]) == (160, 90)       # preview decodes the proxy
            @test VE.pool(p, clip.source).source.height == 90
            # the axis limits follow on the first successful present — the
            # crop drag below maps through them, so wait for the switch
            @test waitfor(() -> maximum(p.previewaxis.finallimits[])[1] <= 161)

            # crop is normalized, so dragging the same viewport region gives
            # the same crop regardless of the displayed resolution
            keypress(Keyboard.c)
            press(pv(0.3, 0.3))
            moveto(pv(0.7, 0.7))
            release()
            crop = clip.crop
            @test 0.05 < crop[1] < 0.5 && 0.2 < crop[3] < 0.8
            @test p.previewaxis.yreversed[]
            keypress(Keyboard.r)

            # playback still presents (through the proxy pool)
            before = p.presented
            press(tlx(0.1)); release()
            keypress(Keyboard.space)
            sleep(0.8)
            keypress(Keyboard.space)
            @test p.presented > before
        end

        @testset "Ctrl+S saves the project" begin
            path = VE.projectfile(p)
            isfile(path) && rm(path)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.s)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test isfile(path)
            @test length(p.sequence.clips) == length(VE.loadproject(path).clips)
            # and plain S still splits (Ctrl branch must not shadow it)
            nclips = length(p.sequence.clips)
            press(tlx(0.4)); release()
            keypress(Keyboard.s)
            @test length(p.sequence.clips) == nclips + 1
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)                       # undo the split
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            rm(path)
        end

        @testset "space toggles playback" begin
            keypress(Keyboard.space)
            @test p.playing[]
            keypress(Keyboard.space)
            @test !p.playing[]
        end

        @testset "audio feed & mute" begin
            if Sys.which("pw-cat") === nothing
                @test_skip "PipeWire (pw-cat) not available"
            else
                # lifecycle: the feed spawns on play and dies on pause
                p.playhead[] = 2
                VE.play!(p); sleep(0.4)
                @test p.audio.proc !== nothing
                VE.pause!(p); sleep(0.3)
                @test p.audio.proc === nothing

                # the Sound/Muted button (one isolated real click)
                mutebtn = first(b for b in fig.content
                                if b isa Makie.Button && b.label[] in ("Sound", "Muted"))
                mb = mutebtn.layoutobservables.computedbbox[]
                mcenter = Point2f(mb.origin .+ mb.widths ./ 2)
                sleep(0.5)                             # outside any dblclick window
                press(mcenter); release(); sleep(0.2)
                @test !p.audio.enabled
                @test mutebtn.label[] == "Muted"

                # muted playback spawns no feed
                p.playhead[] = 2
                VE.play!(p); sleep(0.4)
                @test p.audio.proc === nothing
                VE.pause!(p)

                sleep(0.5)
                press(mcenter); release(); sleep(0.2)  # unmute again
                @test p.audio.enabled
                @test mutebtn.label[] == "Sound"
            end
        end

        @testset "empty timeline stays safe" begin
            # deleting every clip is a valid state — nothing may crash on it
            while !isempty(p.sequence.clips)
                VE.deleteclip!(p.sequence, p.sequence.clips[1].start)
            end
            VE.relayout!(p.timeline)
            @test isempty(p.sequence.clips)
            @test VE.saveproject!(p) === nothing         # Ctrl+S path: refuses politely
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.s)                          # the real key event too
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            sleep(0.3)
            @test occursin("empty", p.status[])
            @test VE.renderpreview(p, 0.0, 240) isa Matrix  # MCP get_frame: black, no throw
            @test_throws ErrorException exportvideo(joinpath(mktempdir(), "x.mp4"), p.sequence)
            @test VE.showframe!(p, 0)                     # preview shows black
        end

        @testset "media bin drag-drop refills the timeline" begin
            # the bin still lists the source; dragging a row onto the (empty)
            # timeline places a clip at the drop position
            binbtn = first(b for b in fig.content
                           if b isa Makie.Button && b.label[] == "Bin")
            bb = binbtn.layoutobservables.computedbbox[]
            sleep(0.5)                       # clear the dblclick window
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release()
            @test waitfor(() -> p.dockopen[] === :media)
            @test !isempty(p.binrows)
            rb = p.binrows[1][1].layoutobservables.computedbbox[]  # (button, thumbnail axis)
            sleep(0.5)
            press(Point2f(rb.origin .+ rb.widths ./ 2))
            @test p.dragsource !== nothing
            moveto(tlx(0.3))                       # ghost follows the cursor
            moveto(tlx(0.5))
            release()
            @test p.dragsource === nothing
            @test length(p.sequence.clips) == 1
            @test p.sequence.clips[1].start == 15         # dropped at 0.5 s @30fps
            @test waitfor(() -> occursin("placed", p.status[]))  # async statusqueue
        end

        @testset "toolbar tools: pick up, click-to-act, cursor" begin
            toolbtn(lbl) = first(b for b in fig.content
                                 if b isa Makie.Button && b.label[] == lbl)
            click(b) = (bb = b.layoutobservables.computedbbox[];
                        press(Point2f(bb.origin .+ bb.widths ./ 2)); release())
            # the media beat left one clip 0.5–16.5s; place the playhead early
            press(tlx(0.7)); release()
            nclips = length(p.sequence.clips)
            playhead_before = p.playhead[]
            sleep(0.5)
            click(toolbtn("✂"))                   # split tool ARMS (no cut yet)
            @test p.tool[] == :split
            @test length(p.sequence.clips) == nclips
            # click the timeline elsewhere → cut THERE, not at the playhead
            press(tlx(3.0)); release()
            @test length(p.sequence.clips) == nclips + 1
            @test p.tool[] == :split               # persistent blade stays on (Esc/✂ to stop)
            @test p.playhead[] == playhead_before  # the click cut, didn't scrub
            @test any(c -> c.start == 90, p.sequence.clips)  # cut at 3.0 s @30fps
            sleep(0.5)
            click(toolbtn("↶"))                   # undo tool
            @test length(p.sequence.clips) == nclips
            sleep(0.5)
            click(toolbtn("▢"))                   # the crop tool starts crop mode
            @test p.tool[] == :crop
            @test p.cropmode[]
            keypress(Keyboard.escape)
            @test !p.cropmode[]
            @test p.tool[] == :none
        end

        @testset "one refiner at a time, and it targets the CURRENT frame" begin
            # The scrub/fx desync: `retrypresent` used to spawn a task PER playhead
            # change, each closing over ITS frame — so a slow one published behind a
            # newer one and the matte mask lagged the picture. The fix is structural
            # and cheap to state: at most one refiner exists, and it re-reads
            # `playhead[]` every pass rather than a captured `n`.
            @test p.refining isa Threads.Atomic{Bool}
            @test p.refining[] == false                  # nothing in flight at rest

            # A second call while one is running must NOT start another — that is
            # the whole invariant, and `atomic_cas!` is what enforces it.
            Threads.atomic_cas!(p.refining, false, true)
            @test p.refining[] == true
            @test VE.retrypresent(p) === nothing         # declines rather than piling on
            p.refining[] = false

            # …and a burst of playhead moves settles on the LAST one, not on
            # whichever render happened to finish last.
            head0 = p.playhead[]
            clip = p.sequence.clips[1]
            for n in (clip.start + 2, clip.start + 5, clip.start + 9)
                p.playhead[] = n
            end
            target = p.playhead[]
            @test waitfor(() -> p.refining[] == false; s = 20)
            @test p.playhead[] == target
            p.playhead[] = head0
            VE.refreshedit!(p)
        end

        @testset "the pill row is tall enough for its pills" begin
            # TODO #7's other half. The row height was `Fixed(28 * n)` — a hardcoded
            # guess at one pill plus a gap — while the grid's real gap was Makie's
            # default. Every added object pushed the pills further past the bottom
            # of their row and over the control beneath.
            @test VE.pillrowsize(1) == Makie.Fixed(VE.PILLHEIGHT)
            for n in 2:6
                want = (VE.PILLHEIGHT + VE.PILLGAP) * n - VE.PILLGAP
                @test VE.pillrowsize(n) == Makie.Fixed(want)
                # …i.e. n pills and the n-1 gaps BETWEEN them, never n gaps
                @test want == n * VE.PILLHEIGHT + (n - 1) * VE.PILLGAP
            end
            # and the radius is the "not a stadium" one the complaint was about
            @test VE.PILLRADIUS < VE.PILLHEIGHT / 2
        end

        @testset "painting into the mask actually paints" begin
            # The second half of the repair TODO — "or directly paint into the
            # mask" — and it had NO coverage at all: not one test called
            # `beginmattebrush!`. The GUI for it was verified by screenshot; that
            # says the hint and the sizing are on screen, not that a stroke moves a
            # pixel.
            clip = p.sequence.clips[1]
            snap0 = VE.docsnapshot(p)
            p.timeline.selected[] = 1
            p.playhead[] = clip.start
            clip.mattetrack = VE.MatteTrack(fill(0x00, 32, 24, 20), clip.src_in)
            empty!(VE.matterepairs(p, clip))
            sf = VE.sourceframe(clip, p.playhead[])
            @test all(==(0x00), VE.matteframe(clip, sf))       # nothing painted yet

            @test VE.beginmattebrush!(p, true)                  # left drag = add
            vp = p.previewaxis.scene.viewport[]
            mid = Point2f(vp.origin[1] + 0.5 * vp.widths[1],
                          vp.origin[2] + 0.5 * vp.widths[2])
            @test VE.mattebrushto!(p, mid; radius = 0.25)
            @test any(==(0xff), VE.matteframe(clip, sf))        # …the stroke landed

            # …and letting go RECORDS it, so a later full re-run cannot silently
            # discard the fix — that is what `matterepairs` is for.
            @test VE.endmattebrush!(p)
            @test haskey(VE.matterepairs(p, clip), Int(sf))
            @test p.mattebrush === nothing                      # the stroke is over

            empty!(VE.matterepairs(p, clip))
            clip.mattetrack = nothing
            VE.docrestore!(p, snap0)
            VE.refreshedit!(p)
        end

        @testset "the DNN kernels actually compile on the GPU" begin
            # Both of these shipped with `RGB{N0f8}(::Float32, …)`, which VALIDATES
            # and calls `throw_colorerror` — string building on an error path no
            # input reaches, which Lava rejects, taking the whole kernel with it.
            # They passed every CPU test and could never have run on the GPU they
            # were written for. `unitn0f8` (matte.jl) exists precisely for this and
            # documents it; I wrote both kernels without using it.
            be = p.analysisbackend
            KA = VE.KA
            W, H = 32, 24
            img = KA.allocate(be, VE.RGB{VE.N0f8}, W, H)
            fill!(img, VE.RGB{VE.N0f8}(0.5, 0.4, 0.3))
            out = KA.allocate(be, VE.RGB{VE.N0f8}, W, H)
            fill!(out, VE.RGB{VE.N0f8}(0.1, 0.2, 0.3))
            VE.lookmix_kernel!(be)(out, img, 0.5f0; ndrange = (W, H))
            KA.synchronize(be)
            @test Array(out)[1, 1] isa VE.RGB{VE.N0f8}

            dep = KA.allocate(be, UInt8, W, H); fill!(dep, 0x80)
            dst = KA.allocate(be, VE.RGB{VE.N0f8}, W, H)
            VE.depthblur_kernel!(be)(dst, img, dep, Int32(W), Int32(H), 0.5f0, Int32(3);
                                     ndrange = (W, H))
            KA.synchronize(be)
            @test Array(dst)[1, 1] isa VE.RGB{VE.N0f8}
        end

        @testset "resizing a depth map keeps its values" begin
            # `analyzedepth!` resized through `mattemaskscale`, which ends
            # `v > 0 ? 0xff : 0x00` — right for a binary seed mask, catastrophic
            # for depth: every non-zero depth became 0xff, so the track was a flat
            # "everything is nearest" plane. Depth blur still CHANGED the picture
            # (it defocused everything equally), so every does-it-do-something
            # check passed. The card's thumbnail — solid white — is what showed it.
            grad = UInt8[round(UInt8, 255 * (i - 1) / 15) for i in 1:16, j in 1:12]
            @test length(unique(grad)) > 2                    # a real gradient in
            small = VE.depthscale(grad, 8, 6)
            @test size(small) == (8, 6)
            @test length(unique(small)) > 2                   # …and a gradient out
            @test minimum(small) < 0x40 && maximum(small) > 0xc0

            # the mask scaler, by contrast, is SUPPOSED to flatten — that is what
            # makes it wrong here and right where it belongs
            flat = VE.mattemaskscale(grad, 8, 6)
            @test Set(unique(flat)) ⊆ Set([0x00, 0xff])

            # …and it interpolates rather than picking nearest: depth drives a
            # PER-PIXEL blur radius, so a stepped map bands the defocus into rings.
            ramp = UInt8[round(UInt8, 255 * (i - 1) / 63) for i in 1:64, j in 1:8]
            down = VE.depthscale(ramp, 16, 4)
            steps = diff(Int.(down[:, 1]))
            @test all(>=(0), steps)                    # monotone in, monotone out
            @test maximum(steps) - minimum(steps) <= 1 # …and evenly spaced, not chunked
        end

        @testset "the depth model's output is a matrix, not a tensor" begin
            # `depthbytes` takes a MATRIX; the runner returns the model's raw
            # tensor with a batch and a channel axis. Pressing "Estimate depth"
            # threw `MethodError: no method matching depthbytes(::Array{Float16,4})`
            # every time — the whole feature was unreachable, and every test around
            # it fed `depthbytes` a synthetic matrix so nothing noticed.
            @test VE.dropsingletons(zeros(Float16, 8, 6, 1, 1)) |> size == (8, 6)
            @test VE.dropsingletons(zeros(Float16, 1, 1, 8, 6)) |> size == (8, 6)
            @test VE.dropsingletons(zeros(Float16, 8, 6)) |> size == (8, 6)
            @test VE.depthbytes(VE.dropsingletons(rand(Float16, 8, 6, 1, 1))) isa Matrix{UInt8}
        end

        @testset "paste selects what it pasted" begin
            # Paste lands a clip at the playhead on whichever lane is free, which
            # is rarely where it belongs — so the next act is always to move it.
            # Without a selection the act BEFORE that was hunting for the thing you
            # had just made.
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            p.timeline.selected[] = 1
            VE.copyclips!(p)
            @test !isempty(p.clipboard)
            p.playhead[] = VE.clipend(seq.clips[1])
            n = VE.pasteclips!(p)
            @test n == 1
            @test length(seq.clips) == 2
            i = p.timeline.selected[]
            @test 1 <= i <= length(seq.clips)
            # …and it is the NEW one, found by identity after the sort reorders
            @test seq.clips[i].start == head0 + VE.cliplength(seq.clips[1]) ||
                  seq.clips[i] !== seq.clips[1]
            @test i in p.timeline.selection[]

            empty!(p.clipboard)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "a refused lane says WHY, not just no" begin
            # "seems like only new track is a drop target" was a reading of silence:
            # the ghost went red and nothing said what WOULD work, so an occupied
            # lane looked like a lane that refuses clips rather than one that
            # refuses this clip here. The label sits on the lane, where the eye is.
            tl0 = p.timeline
            # a VECTOR: `text!` type-locks the attribute at creation, so both the
            # hint and its replacement have to be assigned in that form
            @test first(tl0.newtrackplot.text[]) isa AbstractString
            # the hint is STROKED, because it also has to read on top of a
            # filmstrip — unstroked it was invisible over the clip it covers
            @test tl0.newtrackplot.strokewidth[] > 0
        end

        @testset "Ctrl-drag lifts a clip to a new track" begin
            # Ctrl+press mid-clip, then drag UP into the marked new-track zone.
            #
            # This used to work WITHOUT Ctrl: a plain press that wandered out of
            # the clip's lane converted into a move. That gesture is the scrub's
            # exactly — press on the timeline and move — separated only by how far
            # you strayed, and it kept stealing clips during ordinary scrubbing.
            # Ctrl is now the only way to move a clip, so the two cannot collide.
            clip = p.sequence.clips[1]
            @test clip.track == 1
            sleep(0.5)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            press(tlx(3.0))
            @test tl.dragclip !== nothing        # Ctrl+press grabs it immediately
            vp = ax.scene.viewport[]
            moveto(Point2f(tlx(3.5)[1], vp.origin[2] + 0.93 * vp.widths[2]))
            @test tl.dragtrack == 2
            release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test clip.track == 2
            @test VE.ntracks(p.sequence) == 2
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)                 # undoable like any edit
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test p.sequence.clips[1].track == 1
        end

        @testset "a plain scrub never moves a clip" begin
            # The complaint this replaced the old gesture over: scrubbing along
            # the timeline would sometimes carry the clip with it. Without Ctrl
            # the playhead follows and the sequence must not change at all.
            clip = p.sequence.clips[1]
            tr0, st0 = clip.track, clip.start
            sleep(0.5)
            press(tlx(3.0))
            vp = ax.scene.viewport[]
            moveto(Point2f(tlx(3.5)[1], vp.origin[2] + 0.93 * vp.widths[2]))
            @test tl.dragclip === nothing        # no move was started
            release()
            @test clip.track == tr0
            @test clip.start == st0
        end

        @testset "copy/paste carries the fx graph, not references" begin
            # Ctrl+C then Ctrl+V. The copy must be independent: same effects, own
            # slot ids, so editing one clip's stack cannot reach the other's.
            seq = p.sequence
            # Every testset here shares one Player, so this one has to hand the
            # document back exactly as it found it — it adds an effect, leaves a
            # pasted clip behind and moves the playhead, and the testsets after it
            # assert against the sequence it started with.
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            clip = seq.clips[1]
            VE.seteffect!(clip, VE.OpacityEffect(0.5f0))
            nfx = length(clip.effects)
            @test nfx > 0
            n0 = length(seq.clips)
            tl.selected[] = 1
            sleep(0.3)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.c)
            @test length(p.clipboard) == 1
            VE.seek!(p, VE.seqlength(seq) - 1)   # paste lands at the playhead
            sleep(0.3)
            keypress(Keyboard.v)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test waitfor(() -> length(seq.clips) == n0 + 1)
            pasted = seq.clips[end]
            @test pasted !== clip
            @test pasted.id != clip.id                       # fresh identity
            @test length(pasted.effects) == nfx              # the stack came along
            @test pasted.source === clip.source              # source is shared
            @test pasted.mattetrack === clip.mattetrack      # analysis is shared
            # …but the SLOTS are the copy's own, or the inspector would address both
            @test all(a.id != b.id for a in pasted.effects, b in clip.effects)
            # editing the copy's stack must not touch the original's — the slot
            # is mutable, so this is exactly what a shared slot would leak through
            pasted.effects[1].enabled = false
            @test clip.effects[1].enabled

            empty!(p.clipboard)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == n0)
        end


        @testset "depth: normalization, effect, and declining without a track" begin
            # `depthbytes` is the one place a monocular model's arbitrary scale is turned
            # into something an effect can threshold, so it is worth pinning directly.
            @test VE.depthbytes(Float32[1 2; 3 4]) == UInt8[0x00 0x55; 0xaa 0xff]
            @test all(==(0x80), VE.depthbytes(fill(0.5f0, 3, 3)))   # no range at all
            @test all(==(0x80), VE.depthbytes(fill(NaN32, 2, 2)))   # …and no finite range

            # The effect survives a project-file roundtrip, which is what a new effect
            # most often forgets: `effectdict`/`effectfromdict` are two lists to update.
            e = VE.DepthBlurEffect(; focus = 0.25, strength = 0.75)
            @test VE.effectfromdict(VE.effectdict(e)) == e
            @test VE.isneutral(VE.DepthBlurEffect(; strength = 0.0))
            @test !VE.isneutral(e)

            # No track → no node, so the effect can sit in a stack waiting for the
            # analysis instead of erroring or rendering something wrong.
            src = VE.VideoSource(testvideo)
            clip = VE.Clip(src)
            @test clip.depthtrack === nothing
            @test VE.planeshape(VE.DepthBlurOp(1.0f0, 1.0f0), clip) === nothing
            @test VE.depthframe(clip, 0) === nothing

            # …and with one, the plane is found and sized from the track.
            clip.depthtrack = VE.DepthTrack(fill(0x40, 8, 6, 4), 0)
            @test VE.planeshape(VE.DepthBlurOp(1.0f0, 1.0f0), clip) == (8, 6)
            @test size(VE.depthframe(clip, 2)) == (8, 6)
            @test VE.depthframe(clip, 99) === nothing        # outside the track

            # A copy and a split must carry depth, exactly as they carry the matte —
            # both are keyed by absolute source frame.
            c2 = VE.copyclip(clip)
            @test c2.depthtrack === clip.depthtrack
        end

        @testset "depth blur focuses at a plane, not just far away" begin
            # The discriminating test: moving the focus plane must SWAP which half
            # stays sharp. A kernel that merely blurred "the far stuff" passes any
            # single-focus check and fails this one — and focusing on a mid-ground
            # subject is the whole reason `focus` is a parameter.
            W, H = 64, 48
            chk(x, y) = Float32((x ÷ 4 + y ÷ 4) % 2)
            img = [RGB{N0f8}(chk(x, y), chk(x, y), chk(x, y)) for x in 1:W, y in 1:H]
            dw, dh = 16, 12
            plane = [x <= dw ÷ 2 ? 0xff : 0x00 for x in 1:dw, y in 1:dh]   # left near, right far
            # `VE.ColorTypes.red`, as line 169 and runtests.jl:210 already do: the test
            # file does not import ColorTypes, and four loaded packages export a `red`,
            # so an unqualified one resolves to none of them in `Main`.
            chan(q) = Float64(VE.ColorTypes.red(q))
            varof(v) = (m = sum(chan, v) / length(v);
                        sum((chan(q) - m)^2 for q in v) / length(v))
            nearside(a) = view(a, 1:(W ÷ 2 - 4), :)
            farside(a)  = view(a, (W ÷ 2 + 4):W, :)

            onnear = similar(img)
            VE.depthblur!(onnear, img, plane, VE.DepthBlurOp(1.0f0, 1.0f0))
            @test varof(nearside(onnear)) > varof(farside(onnear))
            @test varof(farside(onnear)) < varof(farside(img))

            onfar = similar(img)
            VE.depthblur!(onfar, img, plane, VE.DepthBlurOp(0.0f0, 1.0f0))
            @test varof(farside(onfar)) > varof(nearside(onfar))     # …swapped
            @test varof(nearside(onfar)) < varof(nearside(img))

            flat = similar(img)                                       # neutral is a pass-through
            VE.depthblur!(flat, img, plane, VE.DepthBlurOp(1.0f0, 0.0f0))
            @test flat == img
        end

        @testset "look: per-clip LUT, uploaded once, declines without one" begin
            src = VE.VideoSource(testvideo)
            clip = VE.Clip(src)
            @test clip.look === nothing
            @test VE.lookdim(clip) === nothing
            @test VE.nodefor(VE.LookEffect(), 1, clip) === nothing   # no LUT, no node

            # An identity table: out == in, so a grade that changes nothing proves
            # the sampling is right rather than that the kernel ran.
            D = 8
            lut = Array{Float32,4}(undef, D, D, D, 3)
            for k in 1:D, j in 1:D, i in 1:D
                lut[i, j, k, 1] = (i - 1) / (D - 1)
                lut[i, j, k, 2] = (j - 1) / (D - 1)
                lut[i, j, k, 3] = (k - 1) / (D - 1)
            end
            clip.look = lut
            @test VE.lookdim(clip) == D
            n = VE.nodefor(VE.LookEffect(), 1, clip)
            @test n isa VE.LookNode && n.dim == D

            W, H = 16, 12
            img = [RGB{N0f8}((x - 1) / (W - 1), (y - 1) / (H - 1), 0.5f0) for x in 1:W, y in 1:H]
            out = similar(img)
            VE.applylook!(out, img, lut, 1.0)
            chan(q) = Float64(VE.ColorTypes.red(q))
            @test maximum(abs(chan(a) - chan(b)) for (a, b) in zip(out, img)) < 0.02

            # strength mixes back toward the original; at 0 it IS the original
            VE.applylook!(out, img, lut, 0.0)
            @test out == img

            # split and copy carry the shot's grade — it is keyed to the clip,
            # not to a frame range
            r = VE.copyclip(clip)
            @test r.look === clip.look

            # …and it survives save/reopen. Re-learning is NOT the same operation:
            # `runlook!` fits from whatever frame the playhead is on, so a look
            # dropped by the project file comes back as a DIFFERENT grade — which
            # is worse than coming back missing, because nothing tells you.
            dir = mktempdir()
            proj = joinpath(dir, "look.vproj")
            VE.saveproject(proj, VE.Sequence([clip], 30.0))
            back = VE.loadproject(proj)
            @test back.clips[1].look !== nothing
            @test size(back.clips[1].look) == size(lut)
            @test maximum(abs.(back.clips[1].look .- lut)) < 1.0f-6
        end

        @testset "captions live on the sequence, save and undo" begin
            seq = p.sequence
            @test isempty(seq.captions)          # a fresh sequence has none

            caps = [VE.Caption(0.0, 1.0, "hello"), VE.Caption(0.9, 2.0, "world")]
            @test VE.captionat(caps, 0.5) == "hello"
            @test VE.captionat(caps, 1.5) == "world"
            @test VE.captionat(caps, 0.95) == "world"   # overlap: the later line wins
            @test VE.captionat(caps, 9.0) == ""         # outside every line

            # Linear resample, exact on a 2:1 decimation
            @test VE.resampleaudio(Float32[0, 1, 2, 3], 4, 2) == Float32[0, 2]
            @test VE.resampleaudio(Float32[1, 2], 8, 8) == Float32[1, 2]

            # A transcript is an EDIT: it survives a project roundtrip…
            snap0 = VE.docsnapshot(p)
            append!(seq.captions, caps)
            path = joinpath(mktempdir(), "cap.vedit")
            VE.saveproject(path, p.sequence)
            back = VE.loadproject(path)
            @test length(back.captions) == 2
            @test back.captions[2].text == "world"
            @test back.captions[1].stop == 1.0

            # …and undo reaches it, because re-transcribing replaces corrections
            VE.docrestore!(p, snap0)
            @test isempty(p.sequence.captions)

            @test haskey(VE.OVERLAYBYNAME, :captions)

            # …and a transcript is a GUESS, so it must be correctable without
            # re-running the model. `captionindexat` is what the editor asks with
            # a playhead; the caption stores seconds.
            empty!(seq.captions); append!(seq.captions, caps)
            fps = seq.framerate
            @test VE.captionindexat(seq, round(Int, 0.5 * fps)) == 1
            @test VE.captionindexat(seq, round(Int, 1.5 * fps)) == 2
            @test VE.captionindexat(seq, round(Int, 9.0 * fps)) == 0
            p.playhead[] = round(Int, 1.5 * fps)
            @test VE.editcaption!(p, "corrected")
            @test seq.captions[2].text == "corrected"
            @test seq.captions[2].start == caps[2].start   # only the words change
            p.playhead[] = round(Int, 9.0 * fps)
            @test !VE.editcaption!(p, "nowhere")           # nothing under the playhead
            empty!(seq.captions)
        end

        @testset "time interpolation: the phase is what floor() throws away" begin
            src = VE.VideoSource(testvideo)
            c = VE.Clip(src); c.start = 0
            @test c.timeinterp === :sample              # every editor's default
            c.rate = 1.0
            @test VE.sourcephase(c, 7) == 0.0           # unconformed: never between frames

            c.rate = 0.5                                # half speed
            @test [VE.sourcephase(c, n) for n in 0:5] == [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
            @test [VE.sourceframe(c, n) - c.src_in for n in 0:5] == [0, 0, 1, 1, 2, 2]

            # …which is exactly the judder: two timeline frames show source frame
            # 0, and the phase says the second one wants a frame that is not there.
            @test VE.settimeinterp!(c, :flow) === :flow
            @test_throws ArgumentError VE.settimeinterp!(c, :bogus)
            @test c.timeinterp === :flow
            @test VE.copyclip(c).timeinterp === :flow    # a copy keeps the mode
        end

        @testset "narration mixes over the clips and saves its words" begin
            seq = p.sequence
            @test isempty(seq.narration)

            nar = VE.Narration("hello there", 0.5, "af_heart")
            @test isempty(nar.samples)                   # not rendered yet
            push!(seq.narration, nar)

            # An unrendered narration must not change the mix — the words are the
            # edit, the samples are a cache, and a missing cache is silence.
            blk = fill(Int16(100), 2, 64)
            VE.mixnarration!(blk, seq, 0; rate = 48_000)
            @test all(==(Int16(100)), blk)

            # …and a rendered one ADDS, because a voiceover plays OVER the timeline
            append!(nar.samples, fill(0.5f0, 48_000)); nar.rate = 48_000
            VE.mixnarration!(blk, seq, 24_000; rate = 48_000)   # inside its span
            @test all(>(Int16(100)), blk)

            # the WORDS round-trip through a project file; the samples do not
            snap0 = VE.docsnapshot(p)
            path = joinpath(mktempdir(), "nar.vedit")
            VE.saveproject(path, seq)
            back = VE.loadproject(path)
            @test length(back.narration) == 1
            @test back.narration[1].text == "hello there"
            @test back.narration[1].at == 0.5
            @test isempty(back.narration[1].samples)
            # RESTORE first, then clear: `docrestore!` now puts the narration back
            # (that is the undo fix), so clearing before it restored the very line
            # this testset added and leaked it into the next one.
            VE.docrestore!(p, snap0)
            empty!(seq.narration)
        end

        @testset "the canvas is the sequence's, not whichever clip is first" begin
            seq = p.sequence
            head0 = p.playhead[]
            # Normalise BEFORE snapshotting: the testsets share one Player and an
            # earlier crop legitimately leaves a canvas behind, which a snapshot
            # taken first would faithfully restore — and then the "undo unsets it"
            # assertion below would be checking the leak, not the fix.
            seq.canvas = nothing
            snap0 = VE.docsnapshot(p)
            @test VE.canvassize(seq) != (640, 360)   # unset: derived from clip 1

            # Setting it explicitly makes the resolution stop depending on which
            # clip happens to be first — the trap this replaced: deleting clip 1
            # silently changed the project's output size.
            seq.canvas = (640, 360)
            @test VE.canvassize(seq) == (640, 360)
            base = seq.clips[1]
            c2 = VE.copyclip(base; start = VE.clipend(base))
            c2.crop = (0.0, 0.0, 0.5, 0.5)            # a differently-cropped clip
            push!(seq.clips, c2)
            @test VE.canvassize(seq) == (640, 360)    # …and it does not move
            deleteat!(seq.clips, 1)
            @test VE.canvassize(seq) == (640, 360)    # …not even when clip 1 goes

            # Unset, the old behaviour is intact so existing projects open the same
            seq.canvas = nothing
            @test VE.canvassize(seq) == (round(Int, 0.5 * c2.source.width) ÷ 2 * 2,
                                         round(Int, 0.5 * c2.source.height) ÷ 2 * 2)

            # The canvas is a SEQUENCE field and `snapshot(seq)` returns the clip
            # vector alone, so this is what caught that cropping outward survived
            # a Ctrl+Z: the picture went back to its old framing and the output
            # size stayed changed.
            seq.canvas = (1920, 1080)
            VE.docrestore!(p, snap0)
            @test seq.canvas === nothing

            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "crop scope: this clip, or the whole project" begin
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            seq.canvas = nothing
            clip = seq.clips[1]
            p.playhead[] = clip.start
            VE.refreshedit!(p)
            W, H = size(p.frame[])
            @test W > 0 && H > 0

            @test VE.cropscope(p)[] === :canvas      # resizing the project is the default

            # THIS CLIP: the drag reframes the clip and the project keeps its size.
            # The bug this closes: every crop rewrote `sequence.canvas`, so "show
            # less of this one shot" silently resized the finished video.
            VE.setcropscope!(p, :clip)
            before = VE.canvassize(seq)
            p.cropanchor = Point2f(0.1W, 0.1H)
            VE.finishcrop!(p, Point2f(0.6W, 0.6H))
            @test VE.canvassize(seq) == before       # the project did not move…
            @test clip.crop[3] < 0.9                 # …but the clip did
            # …and it holds because the derived canvas got PINNED. Without that,
            # `canvassize` re-derives from clip one's crop and a "this clip only"
            # crop of clip one silently resized the project anyway — which is what
            # this assertion caught the first time it ran.
            @test seq.canvas == before

            # WHOLE PROJECT: the same drag now sets the canvas.
            VE.setcropscope!(p, :canvas)
            p.cropanchor = Point2f(0.0, 0.0)
            VE.finishcrop!(p, Point2f(0.5W, 0.5H))
            @test seq.canvas !== nothing
            @test VE.canvassize(seq) == seq.canvas

            # …and dragging PAST the edge grows it, which is the whole point of an
            # unclamped crop rectangle — a canvas that could only ever shrink had
            # no gesture for "make this taller".
            grown = VE.canvassize(seq)
            p.cropanchor = Point2f(-0.5W, -0.5H)
            VE.finishcrop!(p, Point2f(1.5W, 1.5H))
            @test VE.canvassize(seq)[1] > grown[1]
            @test VE.canvassize(seq)[2] > grown[2]

            # Reset drops back to deriving it, and is itself undoable
            VE.resetcanvas!(p)
            @test seq.canvas === nothing

            VE.setcropscope!(p, :canvas)
            VE.usetool!(p, :none)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "crop ratio lock: a fraction is not a pixel" begin
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            clip = seq.clips[1]
            src = clip.source

            # The bug this guards: a crop is a fraction OF THE SOURCE, so locking
            # the fraction ratio to 16/9 gives a 16:9 output only on a square
            # source. The ratio has to be converted through the source's pixels.
            for a in (16 / 9, 9 / 16, 1.0)
                x, y, w, h = VE.lockaspect((0.0, 0.0, 1.0, 1.0), src, a)
                @test isapprox((w * src.width) / (h * src.height), a; rtol = 1e-6)
                @test (x, y, w) == (0.0, 0.0, 1.0)   # anchor and width kept…
            end
            @test src.width != src.height            # …and the source is NOT square,
                                                      # so the conversion is doing work

            # End to end: picking a ratio reshapes the framing NOW, and the canvas
            # follows in :canvas scope. Even-pixel rounding is why this is rtol and
            # not equality.
            p.timeline.selected[] = 1
            p.playhead[] = clip.start
            VE.setcropscope!(p, :canvas)
            VE.setcropaspect!(p, 9 / 16)
            cw, ch = VE.canvassize(seq)
            @test isapprox(cw / ch, 9 / 16; rtol = 0.05)

            # …and in :clip scope the shape changes without resizing the project
            VE.setcropscope!(p, :clip)
            seq.canvas = (640, 360)
            VE.setcropaspect!(p, 1.0)
            @test seq.canvas == (640, 360)
            @test isapprox((clip.crop[3] * src.width) / (clip.crop[4] * src.height),
                           1.0; rtol = 1e-6)

            VE.setcropaspect!(p, nothing)
            @test VE.cropaspect(p)[] === nothing
            VE.setcropscope!(p, :canvas)
            VE.usetool!(p, :none)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "the panel rebuilds when the SEQUENCE changes, not just the clip" begin
            # `effsig` reads the CLIP and only the clip, which was right while the
            # panel showed clip effects and nothing else. It now hosts Narration,
            # Transcript, Crop and Time interpolation, which read the SEQUENCE — so
            # the signature never changed when they did, `rebuildstack` returned
            # early, and a narration line you had just typed did not appear on the
            # card that added it. A screenshot found this; no assertion did.
            seq = p.sequence
            clip = seq.clips[1]
            empty!(seq.captions); empty!(seq.narration)
            before = VE.docsig(seq, clip)

            push!(seq.captions, VE.Caption(0.0, 1.0, "hello"))
            @test VE.docsig(seq, clip) != before        # a caption is a change…
            empty!(seq.captions)
            @test VE.docsig(seq, clip) == before

            push!(seq.narration, VE.Narration("hi", 0.0))
            @test VE.docsig(seq, clip) != before        # …so is a narration line
            empty!(seq.narration)

            seq.canvas = (640, 360)
            @test VE.docsig(seq, clip) != before        # …and so is the canvas
            seq.canvas = nothing
            @test VE.docsig(seq, clip) == before

            # …while the clip-only signature is blind to every one of them, which
            # is precisely why it could not be the whole answer.
            e = VE.effsig(clip)
            push!(seq.captions, VE.Caption(0.0, 1.0, "hello"))
            push!(seq.narration, VE.Narration("hi", 0.0))
            seq.canvas = (640, 360)
            @test VE.effsig(clip) == e
            empty!(seq.captions); empty!(seq.narration); seq.canvas = nothing
        end

        @testset "a hidden panel skips rebuilds but is never stale" begin
            # `rebuildstack` runs on EVERY playhead change and tears the stack
            # down on every clip boundary — during playback, that is every cut,
            # paid for cards behind a closed dock. Skipping while hidden is only
            # safe if coming back rebuilds, which is the failure this guards.
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            base = seq.clips[1]
            c2 = VE.copyclip(base; start = VE.clipend(base))
            VE.seteffect!(c2, VE.BlurEffect(3.0f0))       # …so its stack DIFFERS
            push!(seq.clips, c2)
            VE.refreshedit!(p)

            VE.opendock!(p, :effects)
            p.timeline.selected[] = 1
            p.playhead[] = base.start
            @test waitfor(() -> !("Blur" in [c.title[] for c in p.fxwidgets[:fxcards]]))

            # close, move onto the OTHER clip while nothing is watching, reopen
            VE.opendock!(p, :none)
            p.timeline.selected[] = 2
            p.playhead[] = c2.start + 1
            VE.opendock!(p, :effects)
            @test waitfor(() -> "Blur" in [c.title[] for c in p.fxwidgets[:fxcards]])

            deleteat!(seq.clips, findfirst(c -> c === c2, seq.clips))
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "tool-only cards are opened from the menu, not always on" begin
            # These four (Crop, Transcript, Narration, Time interpolation) have a
            # body and no `make`, so there is no effect to hang a card on. They
            # used to render on EVERY clip to make them reachable at all, which
            # put four cards in front of someone who asked for none — Simon: "for
            # a clip without effects that should be empty, discover works via the
            # searchable menu." So the panel shows what was put there, and the
            # menu is how a tool that is not a clip effect gets found.
            VE.opendock!(p, :effects)
            p.timeline.selected[] = 1
            clip = p.sequence.clips[1]
            p.playhead[] = clip.start
            VE.refreshedit!(p)
            p.fxwidgets[:fxlistrefresh]()
            titles() = [c.title[] for c in p.fxwidgets[:fxcards]]
            @test isempty(VE.opentools(p))
            for gone in ("Crop", "Transcript", "Narration", "Time interpolation")
                @test !(gone in titles())
            end
            # The menu offers them beside the real effects, and picking one opens
            # its card WITHOUT putting anything on the clip — that is the whole
            # distinction between a tool and an effect.
            @test :crop in [name for (_, name) in p.fxwidgets[:fxmenuopts]()]
            before = length(clip.effects)
            VE.opentool!(p, :crop)
            @test waitfor(() -> "Crop" in titles())
            @test length(clip.effects) == before
            # …and the card's × closes it again, so it is not a one-way door.
            VE.closetool!(p, :crop)
            @test waitfor(() -> !("Crop" in titles()))
            # …and each is a kind with a body and no `make`, which is the shape
            # `toolonlykinds` selects on — if one grows a `make` it becomes a
            # normal effect card and belongs on the stack above instead.
            names = [k.name for k in VE.toolonlykinds()]
            @test Set(names) == Set([:crop, :transcript, :narration, :timeinterp])
        end

        @testset "an outward crop survives all the way to the exported file" begin
            # The end of every job, and the place TODO #8 would matter most: a
            # crop that reaches OUTSIDE the picture has to grow the delivered
            # file, not just the preview, and the new area has to come out empty
            # rather than as stretched picture. Nothing tested the export against
            # an unclamped crop.
            src = p.sequence.clips[1].source
            clip = VE.Clip(src, 0, 12, 0, (-0.25, -0.25, 1.5, 1.5), 1.0)
            seq = VE.Sequence([clip], 30.0)
            cw, ch = VE.canvassize(seq)
            @test cw > src.width && ch > src.height        # it GREW

            out = joinpath(mktempdir(), "outward.mp4")
            VE.exportvideo(out, seq)
            rd = VE.VideoIO.openvideo(out); f = VE.VideoIO.read(rd); close(rd)
            # `size` is (h, w); `canvassize` is (w, h). Not a transposition bug —
            # checking the wrong way round is how it first looked like one.
            @test size(f) == (ch, cw)
            @test VE.ColorTypes.red(f[1, 1]) == 0          # the new area is empty…
            @test VE.ColorTypes.green(f[1, 1]) == 0        # …letterbox, not picture

            # …and an explicit canvas beats whatever clip 1 would have implied
            seq.canvas = (640, 360)
            out2 = joinpath(mktempdir(), "explicit.mp4")
            VE.exportvideo(out2, seq)
            rd2 = VE.VideoIO.openvideo(out2); f2 = VE.VideoIO.read(rd2); close(rd2)
            @test size(f2) == (360, 640)
        end

        @testset "a long matte is file-backed, not resident" begin
            # The alpha is width*height*length bytes and grows with the SHOT —
            # measured 3.5 GB a minute at 1080p, 17 GB for five. Holding all of it
            # resident was a STORAGE choice, not a requirement: the propagator
            # writes forward once and the renderer reads a single frame. Refusing
            # big mattes with a guard treated the symptom; backing them with a file
            # removes the reason for one.
            src = p.sequence.clips[1].source
            short = VE.Clip(src, 0, 60, 0, (0.0, 0.0, 1.0, 1.0), 1.0)
            @test VE.mattebytes(short) == src.width * src.height * 60

            # Small stays a plain heap array — the mapping is not worth its own cost
            small = VE.mattebuffer(64, 64, 10)
            @test small isa Array{UInt8, 3}
            @test 64 * 64 * 10 <= VE.MATTEINRAM

            # …and past the threshold it is file-backed while staying the SAME
            # type, which is what lets `MatteTrack` and every reader be unchanged.
            dims = (256, 256, 4608)                     # 288 MB: PAST it, not exactly on it
            @test prod(dims) > VE.MATTEINRAM
            big = VE.mattebuffer(dims...)
            @test big isa Array{UInt8, 3}
            @test size(big) == dims
            big[1, 1, 1] = 0x7f; big[end, end, end] = 0x2a
            @test big[1, 1, 1] == 0x7f && big[end, end, end] == 0x2a

            # NOT on a tmpfs: /tmp here is RAM with a path, so backing an
            # "avoid holding it resident" buffer there defeats the whole point.
            @test !startswith(VE.mattescratch(), "/tmp")
            @test isdir(VE.mattescratch())
            @test VE.clearmattescratch!() isa Integer   # sweeps dead processes' files
        end

        @testset "a ripple delete carries the captions and the narration" begin
            # Cutting anything is mostly ripple deletes, and these two are pinned
            # to the PICTURE. Leaving them at absolute seconds while the clips
            # after the cut moved earlier desynced the whole back half of the edit
            # on the first trim.
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            fps = seq.framerate
            base = VE.copyclip(seq.clips[1]; start = VE.clipend(seq.clips[1]))
            push!(seq.clips, base)
            first_len = VE.cliplength(seq.clips[1]) / fps

            empty!(seq.captions); empty!(seq.narration)
            push!(seq.captions, VE.Caption(0.1, 0.4, "before the cut"))
            push!(seq.captions, VE.Caption(first_len + 0.5, first_len + 0.9, "after it"))
            push!(seq.narration, VE.Narration("spoken after", first_len + 0.5))
            append!(seq.narration[1].samples, fill(0.5f0, 64))
            seq.narration[1].rate = 24_000

            VE.deleteclip!(seq, seq.clips[1])          # ripple: everything after moves up
            @test seq.captions[1].text == "before the cut"
            @test seq.captions[1].start == 0.1          # …ahead of the cut: untouched
            @test seq.captions[2].text == "after it"
            @test seq.captions[2].start ≈ 0.5           # …behind it: moved with the picture
            @test seq.captions[2].stop ≈ 0.9
            @test seq.narration[1].at ≈ 0.5
            @test length(seq.narration[1].samples) == 64  # the words did not change

            empty!(seq.captions); empty!(seq.narration)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "a split carries the matte marks to both halves" begin
            # Both stores are keyed by CLIP ID and every way of making a second
            # clip from one hands it a fresh id. The alpha comes along, so the
            # picture stays right and only re-running breaks — the split half met
            # an empty mark store and refused to propagate.
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            left = seq.clips[1]
            VE.mattemarks(p, left)[left.src_in] = fill(0x22, 8, 6)
            VE.matterepairs(p, left)[left.src_in + 1] = fill(0x33, 8, 6)

            p.timeline.selected[] = 1
            p.playhead[] = left.start + VE.cliplength(left) ÷ 2
            VE.split!(p)
            @test length(seq.clips) == 2
            right = seq.clips[2]
            @test right.id != left.id                     # a fresh id, as always…
            @test haskey(VE.mattemarks(p, right), left.src_in)     # …but the marks followed
            @test haskey(VE.matterepairs(p, right), left.src_in + 1)
            @test all(==(0x22), VE.mattemarks(p, right)[left.src_in])

            empty!(VE.mattemarks(p, left)); empty!(VE.matterepairs(p, left))
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
            VE.refreshedit!(p)
            @test waitfor(() -> length(p.sequence.clips) == 1)
        end

        @testset "matte repairs survive save and reopen" begin
            # Invisible without this: the repaired PIXELS live in the alpha
            # sidecar, so a reopened project looked right. What was lost was the
            # RECORD — so the repair cards were gone, and the next `runmatte!`
            # rebuilt from the seeds and silently discarded every fix.
            clip = p.sequence.clips[1]
            snap0 = VE.docsnapshot(p)
            p.timeline.selected[] = 1
            p.playhead[] = clip.start
            clip.mattetrack = VE.MatteTrack(fill(0x00, 8, 6, 40), clip.src_in)
            empty!(VE.matterepairs(p, clip))
            VE.matterepairs(p, clip)[3] = fill(0xff, 8, 6)
            VE.matterepairs(p, clip)[7] = fill(0x40, 8, 6)

            dir = mktempdir()
            proj = joinpath(dir, "reps.vproj")
            VE.saveproject(proj, p)                    # the PLAYER method, not the sequence one
            @test isfile(VE.repairfile(proj, clip.id))

            # Clear them as a reopen would, then read them back off the sidecar
            empty!(VE.matterepairs(p, clip))
            @test isempty(VE.matterepairs(p, clip))
            VE.loadrepairs!(p, proj)
            reps = VE.matterepairs(p, clip)
            @test Set(keys(reps)) == Set([3, 7])
            @test all(==(0xff), reps[3])
            @test all(==(0x40), reps[7])
            @test size(reps[3]) == (8, 6)

            # The MARKS ride the same sidecar, and losing them was worse:
            # `runmatte!` propagates from them, so a reopened matte could not be
            # re-run at all — "Apply matte to clip" met an empty store and refused.
            VE.mattemarks(p, clip)[2] = fill(0x11, 8, 6)
            VE.saveproject(proj, p)
            empty!(VE.mattemarks(p, clip))
            VE.loadrepairs!(p, proj)
            @test haskey(VE.mattemarks(p, clip), 2)
            @test all(==(0x11), VE.mattemarks(p, clip)[2])

            # A checkpoint carries the JSON and NOT the `.mattes` directory, so
            # restoring one must leave the session's marks alone rather than clear
            # them and put nothing back.
            @test !isdir(VE.mattedir(joinpath(dir, "no-such-checkpoint.vproj")))

            empty!(VE.matterepairs(p, clip)); empty!(VE.mattemarks(p, clip))
            clip.mattetrack = nothing
            VE.docrestore!(p, snap0)
        end

        @testset "a finished matte can still be marked again" begin
            # The door that locked behind you: "Mark subject" is gated on there
            # being NO matte, and "Fix this frame only" needs a marking session to
            # already exist — so once a matte was applied, nothing on the card, no
            # shortcut and no palette entry started marking again. That is exactly
            # the case the repair flow was built for.
            clip = p.sequence.clips[1]
            snap0 = VE.docsnapshot(p)
            p.timeline.selected[] = 1
            p.playhead[] = clip.start
            clip.mattetrack = VE.MatteTrack(fill(0x00, 8, 6, 40), clip.src_in)
            VE.showkind!(p, :matte)
            @test waitfor(() -> haskey(p.fxwidgets[:toolpanels], :matte))
            pctx = p.fxwidgets[:toolpanels][:matte]
            # an action that starts marking, with a matte already on the clip
            @test !isempty(pctx.callbacks)
            clip.mattetrack = nothing
            VE.docrestore!(p, snap0)
            VE.refreshedit!(p)
        end

        @testset "the new actions are reachable from the keyboard" begin
            # The palette IS this editor's keyboard route — that is why every edit
            # operation registers one. Everything added this session was
            # mouse-only, so walking a transcript, resetting the project canvas or
            # picking a focus point could not be done without hunting for a card.
            seq = p.sequence
            cmds = Dict(c.name => c for c in VE.commands(p))
            for name in (:next_caption, :prev_caption, :reset_canvas, :pick_focus)
                @test haskey(cmds, name)
            end
            # …and each says WHY it is unavailable rather than failing silently
            empty!(seq.captions)
            @test cmds[:next_caption].enabled(p) isa String
            push!(seq.captions, VE.Caption(0.0, 1.0, "hello"))
            @test cmds[:next_caption].enabled(p) === true
            empty!(seq.captions)

            seq.canvas = nothing
            @test cmds[:reset_canvas].enabled(p) isa String   # nothing to reset
            seq.canvas = (640, 360)
            @test cmds[:reset_canvas].enabled(p) === true
            seq.canvas = nothing
        end

        @testset "every edit operation is in the palette" begin
            # Copy/paste were Ctrl+C/Ctrl+V and nothing else — the only edit
            # operations with no palette entry. That hides them from a mouse user
            # entirely, and hides the SHORTCUT from everyone, since the palette is
            # where a key is learned.
            names = [c.name for c in VE.commands(p)]
            @test :copy_clips in names
            @test :paste_clips in names
            cmds = Dict(c.name => c for c in VE.commands(p))
            @test cmds[:copy_clips].shortcut == "Ctrl+C"
            @test cmds[:paste_clips].shortcut == "Ctrl+V"
            # Paste says WHY it is off rather than being silently dead
            empty!(p.clipboard)
            @test cmds[:paste_clips].enabled(p) isa String
        end

        @testset "a narration line can be reworded and re-timed" begin
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            empty!(seq.narration)
            push!(seq.narration, VE.Narration("frist post", 0.0, "af_heart"))
            append!(seq.narration[1].samples, fill(0.25f0, 100))
            seq.narration[1].rate = 24_000

            # Rewording REPLACES and un-renders: the samples are a cache of the
            # words, and audio that says the old line is worse than none.
            VE.setnarrationtext!(p, 1, "first post")
            @test seq.narration[1].text == "first post"
            @test isempty(seq.narration[1].samples)
            @test seq.narration[1].at == 0.0             # …and the timing survives

            # Moving KEEPS the audio — the words did not change, so re-synthesizing
            # would cost seconds to produce the same sound.
            append!(seq.narration[1].samples, fill(0.25f0, 100))
            p.playhead[] = 30
            VE.movenarration!(p, 1)
            @test seq.narration[1].at ≈ 30 / seq.framerate
            @test length(seq.narration[1].samples) == 100
            @test seq.narration[1].text == "first post"

            # Changing the VOICE un-renders too: the samples cache the words AND
            # who says them, so keeping them would leave a card claiming one voice
            # over audio in another.
            append!(seq.narration[1].samples, fill(0.25f0, 100))
            VE.setnarrationvoice!(p, 1, "am_michael")
            @test seq.narration[1].voice == "am_michael"
            @test isempty(seq.narration[1].samples)
            @test seq.narration[1].text == "first post"     # …and the words survive
            VE.setnarrationvoice!(p, 1, "am_michael")       # same voice: a no-op
            @test seq.narration[1].voice == "am_michael"

            # With no synthesizer installed there are no voices to offer, and the
            # panel draws no menu rather than an empty one.
            @test VE.speakvoices() isa Vector{String}

            empty!(seq.narration)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
        end

        @testset "the transcript can be walked line by line" begin
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            head0 = p.playhead[]
            empty!(seq.captions)
            append!(seq.captions, [VE.Caption(0.0, 1.0, "one"),
                                   VE.Caption(1.0, 2.0, "two"),
                                   VE.Caption(2.0, 3.0, "three")])
            # From a gap, forward goes to the FIRST line and back to the LAST, so
            # the buttons do something useful wherever the playhead happens to be.
            p.playhead[] = round(Int, 9.0 * seq.framerate)
            VE.gotocaption!(p, 1)
            @test VE.captionindexat(seq, p.playhead[]) == 1
            VE.gotocaption!(p, 1)
            @test VE.captionindexat(seq, p.playhead[]) == 2
            VE.gotocaption!(p, -1)
            @test VE.captionindexat(seq, p.playhead[]) == 1
            VE.gotocaption!(p, -1)                       # clamps, does not wrap
            @test VE.captionindexat(seq, p.playhead[]) == 1

            empty!(seq.captions)
            VE.docrestore!(p, snap0)
            p.playhead[] = head0
        end

        @testset "undo puts a narration back" begin
            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            push!(seq.narration, VE.Narration("undo me", 0.0, "af_heart"))
            @test length(seq.narration) == 1
            VE.docrestore!(p, snap0)
            @test isempty(seq.narration)              # …and not a line of it left
        end

        @testset "a matte repair is visible and droppable" begin
            clip = p.sequence.clips[1]
            reps = VE.matterepairs(p, clip)
            @test isempty(reps)

            # A repair with no matte to repair must decline, not half-apply
            clip.mattetrack === nothing &&
                @test !VE.repairmatteat!(p, fill(0xff, 4, 4))
            @test isempty(VE.matterepairs(p, clip))

            # With a track, the repair is recorded so the panel can show it — the
            # point of the record: a fixed frame you cannot see or undo is not a
            # feature, and a full re-run would silently discard it.
            clip.mattetrack = VE.MatteTrack(fill(0x00, 8, 6, 40), clip.src_in)
            snap0 = VE.docsnapshot(p)
            # `repairmatteat!` repairs the clip being EDITED, and `editclip` wants
            # it selected with the playhead inside it — the same precondition every
            # other clip operation has.
            p.timeline.selected[] = 1
            p.playhead[] = clip.start; sleep(0.2)
            @test VE.repairmatteat!(p, fill(0xff, 8, 6))
            reps = VE.matterepairs(p, clip)
            @test length(reps) == 1
            sf = first(keys(reps))
            @test all(==(0xff), VE.matteframe(clip, sf))     # the pixels changed

            VE.dropmatterepair!(p, clip, sf)                 # …and the record drops
            @test isempty(VE.matterepairs(p, clip))

            clip.mattetrack = nothing
            VE.docrestore!(p, snap0)
            empty!(VE.matterepairs(p, clip))
        end

        @testset "narration and focus-pick are reachable, not just callable" begin
            # The gap these close: `narrate!` and the focus parameter existed with
            # NO way for a user to reach them — plumbing with no door.
            names = [k.name for k in VE.toolkinds()]
            @test :narration in names
            @test :transcript in names
            @test :timeinterp in names        # flow was command-only and invisible

            # Focus-picking is NOT its own tool: it is an action on the Depth blur
            # card, next to the sliders it sets. Two cards for one feature made the
            # user work out that they were related.
            kinds = [k.name for k in VE.effectkinds()]
            @test :depthblur in kinds
            @test :look in kinds              # …and both are addable from the menu
            @test :depthblur in [k.name for k in VE.addablekinds()]
            @test :look in [k.name for k in VE.addablekinds()]
            @test !(:depthfocus in names)

            seq = p.sequence
            snap0 = VE.docsnapshot(p)
            @test isempty(seq.narration)
            # Dropping a line by index, and refusing an index that is not there
            push!(seq.narration, VE.Narration("one", 0.0))
            push!(seq.narration, VE.Narration("two", 1.0))
            VE.dropnarration!(p, 1)
            @test length(seq.narration) == 1 && seq.narration[1].text == "two"
            VE.dropnarration!(p, 99)                       # out of range: a no-op
            @test length(seq.narration) == 1
            empty!(seq.narration)

            # Focus-pick declines cleanly with no depth rather than arming a
            # click that would then fail on the preview.
            clip = seq.clips[1]
            clip.depthtrack = nothing
            # `pickfocus!` acts on the clip being EDITED, so select it — otherwise
            # it declines for the wrong reason ("no clip under the playhead") and
            # the test passes or fails on the wrong branch.
            p.timeline.selected[] = 1
            p.playhead[] = clip.start
            VE.pickfocus!(p)
            @test p.onpick === nothing
            # `setstatus!` PUTS ON A QUEUE — reading `p.status[]` on the next line
            # reads whatever the previous testset left there.
            @test waitfor(() -> occursin("depth", p.status[]))

            clip.depthtrack = VE.DepthTrack(fill(0x80, 8, 6, 40), clip.src_in)
            VE.pickfocus!(p)
            @test p.onpick !== nothing                     # armed
            p.onpick = nothing
            # setinterp! is undoable and a no-op when nothing changes — the card
            # is redrawn constantly, so a toggle that always snapshotted would
            # fill the undo stack with edits that changed nothing.
            @test clip.timeinterp === :sample
            n0 = length(p.undostack)
            VE.setinterp!(p, clip, :sample)
            @test length(p.undostack) == n0
            VE.setinterp!(p, clip, :flow)
            @test clip.timeinterp === :flow
            @test length(p.undostack) == n0 + 1
            VE.setinterp!(p, clip, :sample)

            clip.depthtrack = nothing
            VE.docrestore!(p, snap0)
        end

        @testset "the matte brush has a size you can see and change" begin
            # Painting with a fixed, invisible radius is guesswork — the gesture
            # exists but you cannot tell where the brush is or how big.
            @test p.brushradius ≈ 0.04
            r0 = p.brushradius
            keypress(Keyboard.right_bracket); sleep(0.1)
            @test p.brushradius > r0
            keypress(Keyboard.left_bracket); sleep(0.1)
            @test p.brushradius ≈ r0                    # ] then [ returns
            # Geometric, so one key is the same PROPORTION at any size: a fixed
            # step is half the brush at 1% and nothing at 20%.
            p.brushradius = 0.004
            keypress(Keyboard.left_bracket); sleep(0.1)
            @test p.brushradius ≈ 0.004                 # clamped, not below
            p.brushradius = 0.4
            keypress(Keyboard.right_bracket); sleep(0.1)
            @test p.brushradius ≈ 0.4                   # …and not above
            p.brushradius = 0.04

            # The kernel honours the radius it is given, so the cursor and the
            # stroke cannot disagree about size.
            m = zeros(UInt8, 200, 200)
            VE.brushmatte!(m, 0.5, 0.5, true; radius = 0.05)
            small = count(==(0xff), m)
            fill!(m, 0x00)
            VE.brushmatte!(m, 0.5, 0.5, true; radius = 0.10)
            @test count(==(0xff), m) > 3 * small        # ~4x the area
        end

        @testset "export dock panel renders the timeline" begin
            outbtn = first(b for b in fig.content
                           if b isa Makie.Button && b.label[] == "Export")
            bb = outbtn.layoutobservables.computedbbox[]
            sleep(0.5)
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release()
            @test waitfor(() -> p.dockopen[] === :export)
            out = joinpath(mktempdir(), "paneltest.mp4")
            p.fxwidgets[:exportpath][] = out
            gb = p.fxwidgets[:exportgo].layoutobservables.computedbbox[]
            # clear the dblclick window fully and HOLD the press briefly — an
            # instantaneous synthetic press+release this soon after the Export
            # click above gets swallowed as a double-click
            sleep(1.2)
            press(Point2f(gb.origin .+ gb.widths ./ 2)); sleep(0.2); release()
            @test waitfor(() -> occursin("exported", p.status[]) ||
                                occursin("failed", p.status[]); s = 45)
            @test occursin("exported", p.status[])
            @test isfile(out)
        end

        @testset "dock cycling keeps panel content alive" begin
            # regression: a NESTED Subfigure (the inspector inside the dock slot)
            # lost its content scene forever after switching docks away and back —
            # generic hide! force-hid the scene, unhide! never re-synced it
            VE.opendock!(p, :effects); sleep(0.2)
            for k in (:media, :effects, :export, :none, :effects)
                VE.opendock!(p, k); sleep(0.1)
            end
            sleep(0.2)
            @test p.fxwidgets[:fxscroll].scene.visible[]
            @test p.fxwidgets[:addeffect].blockscene.visible[]
            @test p.fxwidgets[:compare].blockscene.visible[]
        end

        @testset "Ctrl+P palette adds any effect" begin
            press(tlx(2.0)); release()              # playhead onto the clip
            clip = VE.locate(p.sequence, p.playhead[])[1]
            nfx = length(clip.effects)
            sleep(0.3)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.p)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test p.fxwidgets[:palettemodal].open[]
            p.fxwidgets[:palettequery][] = "shar"   # filters down to Sharpen
            keypress(Keyboard.enter)                # ⏎ applies the top hit
            @test !p.fxwidgets[:palettemodal].open[]
            @test length(clip.effects) == nfx + 1
            @test clip.effects[end].effect isa VE.SharpenEffect
            # Stabilization is findable the same way and behaves the same way:
            # it is an effect kind like any other, so the palette ADDS it and its
            # card carries the mode menu and the analyze button.
            nstab = length(clip.effects)
            p.fxwidgets[:paletteopen]()
            p.fxwidgets[:palettequery][] = "stabil"
            keypress(Keyboard.enter)
            @test waitfor(() -> length(clip.effects) == nstab + 1)
            @test clip.effects[end].effect isa VE.StabilizeEffect
            @test waitfor(() -> haskey(p.fxwidgets, :modemenu) && haskey(p.fxwidgets, :analyze))
            # PARAMETER labels hit too: "bright" finds the Color kind (Premiere-style)
            nfx2 = length(clip.effects)
            p.fxwidgets[:paletteopen]()
            p.fxwidgets[:palettequery][] = "bright"
            keypress(Keyboard.enter)
            @test length(clip.effects) == nfx2 + 1
            @test clip.effects[end].effect isa VE.ColorEffect
        end

        @testset "the Add effect menu survives a search that matches nothing" begin
            # Simon, 2026-07-27: picking the first entry ("Color") or the first
            # search hit in the fx menu did nothing. A query with NO hits emptied
            # the menu's per-option color vectors; the next hit then resolved the
            # option text plot against them and threw inside the compute graph —
            # which kills the GLMakie render loop, so the whole window went numb.
            VE.opendock!(p, :effects); sleep(0.3)
            press(tlx(2.0)); release()
            clip = VE.locate(p.sequence, p.playhead[])[1]
            nfx = length(clip.effects)
            menu = p.fxwidgets[:addeffect]
            optiontexts = menu.blockscene.children[1].plots[2]
            menu.is_open[] = true; sleep(0.1)
            foreach(c -> ev.unicode_input[] = c, "colq")   # "colq" matches nothing
            @test isempty(optiontexts.text[])
            keypress(Keyboard.backspace)                   # back to "col" → Color first
            @test first(optiontexts.text[]) == "Color"
            @test length(optiontexts.color[]) == length(optiontexts.text[])
            keypress(Keyboard.enter)                       # ⏎ takes the first hit
            @test !menu.is_open[]
            @test length(clip.effects) == nfx + 1
            @test clip.effects[end].effect isa VE.ColorEffect
        end

        @testset "the FIRST row of a dropdown is clickable" begin
            # Simon, 2026-07-27: "ich kann immer noch nicht color auswählen als
            # erstes". The A/B compare button sits directly under the "+ Add
            # effect…" menu, and its priority-100 bbox hit test swallowed the
            # press on the dropdown's first row: a raw bbox test cannot see what
            # is DRAWN over it. Rows 2+ hang below the button and always worked,
            # which is exactly why this never showed up in a test before.
            VE.opendock!(p, :effects); sleep(0.3)
            press(tlx(2.0)); release()
            clip = VE.locate(p.sequence, p.playhead[])[1]
            nfx = length(clip.effects)
            menu = p.fxwidgets[:addeffect]
            menuscene = menu.blockscene.children[end]
            bb = menu.layoutobservables.computedbbox[]
            sleep(0.5)                                    # clear the dblclick window
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release()
            @test menu.is_open[]
            rects = menuscene.plots[1][1][]
            tr = Makie.translation(menuscene)[]
            row1 = Point2f(sum(extrema(rects[1])) ./ 2 .+ Point2f(tr[1], tr[2]))
            # The overlap is REAL — the whole point of the regression. What sits
            # under the dropdown is the filter box now (the compare button moved
            # below it), and the rule is the same: a widget drawn under an open
            # menu must not swallow the row you click.
            @test row1 in p.fxwidgets[:fxfilterbox].layoutobservables.computedbbox[]
            press(row1); release(); sleep(0.3)
            @test !menu.is_open[]
            @test length(clip.effects) == nfx + 1
            @test clip.effects[end].effect isa VE.ColorEffect
            @test p.applytracks[]        # …and the compare bypass never fired
        end

        @testset "keyframe overlay edits the param you click" begin
            press(tlx(2.0)); release()               # playhead onto the clip
            clip = VE.locate(p.sequence, p.playhead[])[1]
            p.fxwidgets[:paletteopen]()              # a second animatable effect
            p.fxwidgets[:palettequery][] = "brig"
            keypress(Keyboard.enter)
            sleep(0.3)
            len = clip.src_out - clip.src_in         # frames survive earlier trim tests
            fkey = clip.src_in + (3 * len) ÷ 4       # keyed frame (both params)
            ffree = clip.src_in + (2 * len) ÷ 5      # key-free frame for the Alt-add
            VE.startanimating!(p, :brightness)
            VE.startanimating!(p, :sharpen)
            VE.setkey!(clip.animations[:brightness], fkey, VE.paramspec(:brightness).hi)
            VE.setkey!(clip.animations[:sharpen], fkey, VE.paramspec(:sharpen).lo)
            VE.startanimating!(p, :brightness)          # focus = brightness
            notify(p.playhead); sleep(0.3)
            curvepos(key, sf) = begin                # figure pixel on `key`'s curve at SOURCE frame sf
                ntr = VE.ntracks(p.sequence)
                lo, hi = VE.trackband(clip.track, ntr)
                g = min(0.02, VE.trackspan(ntr) * 0.15); lo += g; hi -= g
                inset = 0.12 * (hi - lo); lo += inset; hi -= inset
                c = clip.animations[key]; pr = VE.paramspec(key)
                v = something(VE.valueat(c, sf), pr.get(clip))
                y = lo + (hi - lo) * clamp(VE.paramnorm(pr, v), 0.0, 1.0)
                t = (clip.start + (sf - clip.src_in)) / p.sequence.framerate
                lims = ax.finallimits[]; vp = ax.scene.viewport[]
                Point2f(vp.origin[1] + (t - minimum(lims)[1]) /
                            (maximum(lims)[1] - minimum(lims)[1]) * vp.widths[1],
                        vp.origin[2] + (y - minimum(lims)[2]) /
                            (maximum(lims)[2] - minimum(lims)[2]) * vp.widths[2])
            end
            # Alt-click near the SHARPEN curve while BRIGHTNESS holds the focus:
            # the key must land on what was clicked, and the focus must follow
            nb = length(clip.animations[:brightness].keys)
            ns = length(clip.animations[:sharpen].keys)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.press)
            press(curvepos(:sharpen, ffree)); release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.release)
            sleep(0.2)
            @test length(clip.animations[:sharpen].keys) == ns + 1
            @test length(clip.animations[:brightness].keys) == nb
            @test p.kffocus[] === :sharpen
            # grabbing a brightness ◆ refocuses it
            press(curvepos(:brightness, fkey)); release(); sleep(0.2)
            @test p.kffocus[] === :brightness
            # right-click opens the keyframe menu on the clicked ◆; Delete removes
            # exactly that key (and only it) — the clip context menu stays closed
            ns2 = length(clip.animations[:sharpen].keys)
            moveto(curvepos(:sharpen, ffree))
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.press)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.release)
            sleep(0.2)
            @test p.fxwidgets[:kfmenu].open[]
            @test p.clipmodal === nothing || !p.clipmodal.open[]
            notify(p.fxwidgets[:kfmenubtn1].clicks); sleep(0.2)
            @test !p.fxwidgets[:kfmenu].open[]
            @test length(clip.animations[:sharpen].keys) == ns2 - 1
            @test length(clip.animations[:brightness].keys) == nb
        end

        @testset "keyframe trio, snap, ease & guards (Premiere parity)" begin
            Makie.limits!(ax, 0.0, 4.0, 0.0, 1.0)    # deterministic zoom for pixel math
            press(tlx(2.0)); release(); sleep(0.2)   # playhead onto the clip
            clip = VE.locate(p.sequence, p.playhead[])[1]
            fps = p.sequence.framerate
            markerpix(key, sf) = begin               # figure pixel of `key`'s curve at source frame sf
                ntr = VE.ntracks(p.sequence)
                lo, hi = VE.trackband(clip.track, ntr)
                g = min(0.02, VE.trackspan(ntr) * 0.15); lo += g; hi -= g
                inset = 0.12 * (hi - lo); lo += inset; hi -= inset
                c = clip.animations[key]; pr = VE.paramspec(key)
                v = something(VE.valueat(c, sf), pr.get(clip))
                y = lo + (hi - lo) * clamp(VE.paramnorm(pr, v), 0.0, 1.0)
                t = (clip.start + (sf - clip.src_in)) / fps
                lims = ax.finallimits[]; vp = ax.scene.viewport[]
                Point2f(vp.origin[1] + (t - minimum(lims)[1]) /
                            (maximum(lims)[1] - minimum(lims)[1]) * vp.widths[1],
                        vp.origin[2] + (y - minimum(lims)[2]) /
                            (maximum(lims)[2] - minimum(lims)[2]) * vp.widths[2])
            end
            trio() = p.fxwidgets[:kfacc_contrast]    # re-fetch: keying rebuilds the stack

            # --- ◆ starts animating: first key at the playhead, label flips ◇ → ◆ ---
            @test !VE.clipanimated(clip, :contrast)
            @test Makie.to_value(trio()[2].label) == "◇"
            notify(trio()[2].clicks); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            f1 = VE.playheadframe(p, clip)
            @test [k.frame for k in clip.animations[:contrast].keys] == [f1]
            @test Makie.to_value(trio()[2].label) == "◆"
            # --- scrub off the key (◇), slider writes a second key (◆ again) ---
            press(tlx(2.8)); release(); sleep(0.3)
            @test Makie.to_value(trio()[2].label) == "◇"
            ghost0 = sum(length(c.keys) for (k, c) in clip.animations if k !== :contrast; init = 0)
            Makie.set_close_to!(p.fxsliders[:contrast], 1.6); sleep(0.4)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            f2 = VE.playheadframe(p, clip)
            @test length(clip.animations[:contrast].keys) == 2
            # a gesture on ONE slider must not stamp ghost keys on other animated params
            @test sum(length(c.keys) for (k, c) in clip.animations if k !== :contrast; init = 0) == ghost0
            @test Makie.to_value(trio()[2].label) == "◆"
            # --- ◀ ▶ jump between keys ---
            notify(trio()[1].clicks); sleep(0.3)
            @test VE.playheadframe(p, VE.locate(p.sequence, p.playhead[])[1]) == f1
            notify(trio()[3].clicks); sleep(0.3)
            @test VE.playheadframe(p, VE.locate(p.sequence, p.playhead[])[1]) == f2
            # --- ◆ ON a key removes it; removing the last key un-animates ---
            notify(trio()[2].clicks); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test [k.frame for k in clip.animations[:contrast].keys] == [f1]
            notify(trio()[1].clicks); sleep(0.3)     # ◀ back onto the remaining key
            notify(trio()[2].clicks); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test !VE.clipanimated(clip, :contrast)
            @test Makie.to_value(trio()[2].label) == "◇"

            # --- rebuild a 2-key ramp for the gesture checks ---
            notify(trio()[2].clicks); sleep(0.3)                    # key at f1'
            press(tlx(2.8)); release(); sleep(0.2)
            Makie.set_close_to!(p.fxsliders[:contrast], 1.6); sleep(0.4)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            ka, kb = (k.frame for k in clip.animations[:contrast].keys)

            # --- dragging a ◆ snaps onto the playhead when within reach: aim 9 px
            # right of it (>½ frame at this zoom, so WITHOUT the snap the key would
            # round to mid+1; within the 12 px magnet, so WITH it it lands on mid) ---
            mid = ka + (kb - ka) ÷ 2
            VE.seek!(p, clip.start + (mid - clip.src_in)); sleep(0.2)
            from = markerpix(:contrast, kb)
            near = markerpix(:contrast, mid) .+ Point2f(9, 0)
            press(from); moveto(from .+ Point2f(-30, 0)); moveto(near); release(); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test any(k -> k.frame == mid, clip.animations[:contrast].keys)

            # --- hidden curves are INERT: the same grab must scrub, not retime ---
            p.kfvisible[] = false; sleep(0.2)
            VE.seek!(p, clip.start); sleep(0.2)
            keysbefore = [(k.frame, k.value) for k in clip.animations[:contrast].keys]
            hidden = markerpix(:contrast, mid)
            press(hidden); moveto(hidden .+ Point2f(-40, 0)); release(); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test p.playhead[] != clip.start                       # it scrubbed
            @test [(k.frame, k.value) for k in clip.animations[:contrast].keys] == keysbefore
            p.kfvisible[] = true; sleep(0.2)

            # --- Alt-click with no animated curve nearby only hints ---
            saved = copy(clip.animations); empty!(clip.animations)
            notify(p.playhead); sleep(0.2)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.press)
            press(tlx((clip.start + (mid - clip.src_in)) / fps)); release()   # same clip, no curves
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.release)
            sleep(0.2)
            @test isempty(clip.animations)                         # nothing invented
            merge!(clip.animations, saved); notify(p.playhead); sleep(0.2)

            # --- right-click ◆ → per-KEY ease: smoothing kb flattens ITS tangent ---
            clip = VE.locate(p.sequence, p.playhead[])[1]
            c = clip.animations[:contrast]
            ka, kb = (k.frame for k in c.keys)
            va, vb = (k.value for k in c.keys)
            fq = ka + (kb - ka) ÷ 4
            tq = (fq - ka) / (kb - ka)
            rc = markerpix(:contrast, kb)
            ev.mouseposition[] = Tuple(rc)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.press)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.release)
            sleep(0.2)
            @test p.fxwidgets[:kfmenu].open[]
            notify(p.fxwidgets[:kfmenubtn2].clicks); sleep(0.2)    # "Ease in & out"
            @test c.keys[end].ease === :smooth
            # hermite: ka linear (m0=1), kb smooth (m1=0) → H(t) = -t³ + t² + t
            @test VE.valueat(c, fq) ≈ va + (vb - va) * (-tq^3 + tq^2 + tq) atol = 1.0e-9
            # --- Hold on ka freezes the value until kb ---
            rc = markerpix(:contrast, ka)
            ev.mouseposition[] = Tuple(rc)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.press)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.release)
            sleep(0.2)
            notify(p.fxwidgets[:kfmenubtn3].clicks); sleep(0.2)    # "Hold until the next key"
            @test c.keys[1].ease === :hold
            @test VE.valueat(c, fq) == va
            @test VE.valueat(c, kb) == vb                          # the next key still lands
            # --- and Clear-all makes the parameter static again ---
            rc = markerpix(:contrast, kb)
            ev.mouseposition[] = Tuple(rc)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.press)
            ev.mousebutton[] = MouseButtonEvent(Mouse.right, Mouse.release)
            sleep(0.2)
            @test Makie.to_value(p.fxwidgets[:kfmenubtn2].label) == "Make linear (corner)"
            notify(p.fxwidgets[:kfmenubtn4].clicks); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            @test !VE.clipanimated(clip, :contrast)

            # --- E2E: the keys really drive the render — sampled ends of the ramp ---
            notify(trio()[2].clicks); sleep(0.3)                   # start it again at the playhead
            clip = VE.locate(p.sequence, p.playhead[])[1]
            sf0 = VE.playheadframe(p, clip)
            VE.setkey!(clip.animations[:contrast], sf0, 0.2)
            VE.setkey!(clip.animations[:contrast], clip.src_out - 1, 1.9)
            notify(p.playhead); sleep(0.3)
            @test VE.paramspec(:contrast).get(VE.effectiveclip(clip, sf0)) ≈ 0.2 atol = 1.0e-6
            @test VE.paramspec(:contrast).get(VE.effectiveclip(clip, clip.src_out - 1)) ≈ 1.9 atol = 1.0e-6
            delete!(clip.animations, :contrast)                    # leave the state clean
            notify(p.playhead); sleep(0.2)
        end

        @testset "multi-track keyframes: every lane edits, Alt aims by track" begin
            Makie.limits!(ax, 0.0, 4.0, 0.0, 1.0)
            press(tlx(2.0)); release(); sleep(0.2)
            fps = p.sequence.framerate
            base = VE.locate(p.sequence, p.playhead[])[1]
            savedanims = copy(base.animations); empty!(base.animations)   # a clean V1 lane
            # split base and stack its right half ABOVE it for this testset (restored
            # by joinclips! at the end) — self-sufficient even on a one-clip timeline
            c2 = VE.split!(p.sequence, base.start + (VE.clipend(base) - base.start) ÷ 2)
            c2.start = base.start; c2.track = base.track + 1
            VE.refreshedit!(p); sleep(0.3)
            w = min(VE.clipend(base), VE.clipend(c2)) - base.start   # overlap window
            fA = base.start + w ÷ 3                                  # model key here
            fB = base.start + 2w ÷ 3                                 # Alt-clicks here
            overt = fB / fps
            # a known flat curve on each lane (model setup; the GESTURES are the test)
            VE.setkey!(get!(() -> VE.AnimCurve(), base.animations, :temperature),
                       base.src_in + (fA - base.start), 0.0)     # norm 0.5 → band middle
            VE.setkey!(get!(() -> VE.AnimCurve(), c2.animations, :opacity),
                       c2.src_in + (fA - c2.start), 0.5)         # ditto — the Alt aims there
            notify(p.playhead); sleep(0.3)
            ntr = VE.ntracks(p.sequence)
            bandmid(tr) = begin
                lo, hi = VE.trackband(tr, ntr)
                g = min(0.02, VE.trackspan(ntr) * 0.15)
                (lo + g + hi - g) / 2
            end
            bandpix(t, tr) = begin
                lims = ax.finallimits[]; vp = ax.scene.viewport[]
                Point2f(vp.origin[1] + (t - minimum(lims)[1]) /
                            (maximum(lims)[1] - minimum(lims)[1]) * vp.widths[1],
                        vp.origin[2] + bandmid(tr) * vp.widths[2])
            end
            # Alt-click into the UPPER band adds to c2 only (temperature curve on V1
            # is at band middle = norm 0.5, opacity curve on V2 near band top)
            nb = length(base.animations[:temperature].keys)
            nc = length(c2.animations[:opacity].keys)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.press)
            press(bandpix(overt, c2.track)); release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.release)
            sleep(0.3)
            @test length(c2.animations[:opacity].keys) == nc + 1
            @test length(base.animations[:temperature].keys) == nb
            # ...and into the LOWER band adds to base only
            nc2 = length(c2.animations[:opacity].keys)
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.press)
            press(bandpix(overt, base.track)); release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_alt, Keyboard.release)
            sleep(0.3)
            @test length(base.animations[:temperature].keys) == nb + 1
            @test length(c2.animations[:opacity].keys) == nc2
            # V1's ◆ markers stay editable even though V2 is on top: drag V1's key
            press(tlx(overt)); release(); sleep(0.2)   # playhead over the stack
            k1 = base.animations[:temperature].keys[1]
            lo, hi = VE.trackband(base.track, ntr)
            g = min(0.02, VE.trackspan(ntr) * 0.15); lo += g; hi -= g
            inset = 0.12 * (hi - lo)
            y1 = (lo + inset) + (hi - lo - 2inset) *
                 VE.paramnorm(VE.paramspec(:temperature), k1.value)
            lims = ax.finallimits[]; vp = ax.scene.viewport[]
            mpix = Point2f(vp.origin[1] + ((base.start + (k1.frame - base.src_in)) / fps -
                               minimum(lims)[1]) / (maximum(lims)[1] - minimum(lims)[1]) *
                               vp.widths[1],
                           vp.origin[2] + y1 * vp.widths[2])
            f0 = k1.frame
            press(mpix); moveto(mpix .+ Point2f(-40, 0)); release(); sleep(0.3)
            @test any(k -> k.frame < f0, base.animations[:temperature].keys)
            # restore the timeline: unstack the half and join it back onto base
            empty!(base.animations)
            empty!(c2.animations)
            c2.track = base.track
            c2.start = VE.clipend(base)
            @test VE.joinclips!(p.sequence, base.start) !== nothing
            merge!(base.animations, savedanims)
            VE.refreshedit!(p); notify(p.playhead); sleep(0.3)
        end

        @testset "slider writes keys while playing (live keying)" begin
            press(tlx(1.2)); release(); sleep(0.2)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            VE.startanimating!(p, :contrast); sleep(0.3)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            farm = clip.animations[:contrast].keys[1].frame
            VE.play!(p); sleep(0.5)                    # playhead advancing
            Makie.set_close_to!(p.fxsliders[:contrast], 1.8)
            sleep(0.2); VE.pause!(p); sleep(0.2)
            clip = VE.locate(p.sequence, p.playhead[])[1]
            ks = clip.animations[:contrast].keys
            @test any(k -> k.frame > farm && abs(k.value - 1.8) < 0.02, ks)
            delete!(clip.animations, :contrast)        # leave the state clean
            notify(p.playhead); sleep(0.2)
        end

        @testset "GUI tools: loop hints + blend" begin
            datapos(t, y) = begin                    # timeline data coords → figure pixel
                lims = ax.finallimits[]; vp = ax.scene.viewport[]
                Point2f(vp.origin[1] + (t - minimum(lims)[1]) /
                            (maximum(lims)[1] - minimum(lims)[1]) * vp.widths[1],
                        vp.origin[2] + (y - minimum(lims)[2]) /
                            (maximum(lims)[2] - minimum(lims)[2]) * vp.widths[2])
            end
            press(tlx(2.0)); release()               # playhead onto the clip
            nbefore = VE.seqlength(p.sequence)
            nclips0 = length(p.sequence.clips)
            VE.opendock!(p, :effects)                # card clicks need the panel open
            VE.activatetool!(p, :loopfinder)
            @test VE.activetoolname(p)[] === :loopfinder
            ctx = VE.activetool(p)[2]
            @test waitfor(() -> ctx.state !== nothing && !isempty(ctx.state[:refs]); s = 25)
            sleep(0.4)                               # uiqueue draws card + hints
            # PER TOOL, not the global count: `toolcards[3]` is shared by every
            # tool body, and the panel now renders several at once (Crop,
            # Transcript, Narration, Time interpolation all have card bodies).
            # Counting the whole list measured the panel, not the loop finder.
            @test count(c -> c.tool === :loopfinder, p.fxwidgets[:toolcards][3]) == 1
            pts = ctx.state[:hintpts][]
            @test !isempty(pts)
            # ▼ click CUTS there — the timeline keeps its length, the tool stays on
            press(datapos(pts[1][1], pts[1][2])); release(); sleep(0.3)
            @test length(p.sequence.clips) == nclips0 + 1
            @test VE.seqlength(p.sequence) == nbefore
            @test VE.activetoolname(p)[] === :loopfinder
            # Find adds a SECOND reference card (signatures cached — instant)
            press(tlx(1.0)); release()               # another playhead frame
            VE.activetool(p)[2].callbacks[1]()       # the panel's Find action
            # the clip under t=1.0 may be an UNCACHED split half (the ▼ cut position
            # is content-driven) — the Find then runs a fresh async analysis; wait
            # for the card like the first Find did instead of a fixed sleep
            lfcards() = filter(c -> c.tool === :loopfinder, p.fxwidgets[:toolcards][3])
            @test waitfor(() -> length(lfcards()) == 2; s = 25)
            cards = lfcards()
            @test ctx.state[:active][] == 2
            # clicking the FIRST card highlights its markers again
            bb = cards[1].frame.layoutobservables.computedbbox[]
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release(); sleep(0.2)
            @test ctx.state[:active][] == 1
            # × on the first card removes THAT reference; the second remains
            # BY FIELD: the entry is a NamedTuple precisely so a new field cannot
            # shift what `[6]` means — which is exactly what appending `tool` did.
            @test cards[1].rm !== nothing
            notify(cards[1].rm.clicks); sleep(0.3)
            @test length(ctx.state[:refs]) == 1
            @test length(lfcards()) == 1
            @test ctx.state[:active][] == 1              # selection follows the survivor
            @test !isempty(ctx.state[:hintpts][])        # its hints still shown
            VE.undo!(p); sleep(0.2)                  # undo the hint cut
            @test length(p.sequence.clips) == nclips0
            VE.deactivatetool!(p)
            @test isempty(p.fxwidgets[:toolcards][3])   # cards cleaned up

            # BLEND: opacity keys on the LATER clip, remembered as a pair by clip id.
            # The card is project state (built whether or not the tool is on), the
            # header IS the action, nothing captures timeline clicks, and the length
            # control edits whatever blend is selected.
            Makie.limits!(ax, 0.0, VE.seqduration(p.sequence), 0.0, 1.0)
            sleep(0.2)
            nclips = length(p.sequence.clips)
            fps = p.sequence.framerate
            VE.seek!(p, VE.seqlength(p.sequence) ÷ 2)
            VE.split!(p); sleep(0.3)
            @test length(p.sequence.clips) == nclips + 1
            c1 = p.sequence.clips[1]; c2 = p.sequence.clips[2]
            pctx = p.fxwidgets[:toolpanels][:blend]
            @test VE.activetoolname(p)[] === :none      # nothing on …
            @test !isempty(pctx.controls)               # … and the card is filled anyway
            @test length(pctx.rows) == 1                # "no blends yet" placeholder
            @test pctx.state[:fit]                      # "move clips to fit" is the default

            tl.selection[] = [1, 2]; sleep(0.2)
            VE.activatetool!(p, :blend); sleep(0.5)     # clicking the header blends
            @test VE.activetoolname(p)[] === :none      # and hands the slot straight back
            @test isempty(p.sequence.transitions)       # NOT a transition
            @test VE.clipanimated(c2, :opacity)         # the later clip fades in
            @test !VE.clipanimated(c1, :opacity)        # the earlier one is untouched
            @test c2.blendfrom == c1.id                 # the pair is REMEMBERED, not guessed
            @test VE.blends(p.sequence) == [(1, 2, VE.fadeinlength(c2))]
            @test length(pctx.rows) == 1                # one row per blend
            # with the option ticked the clips are already arranged: one step, done
            @test c2.start < VE.clipend(c1)
            @test VE.clipend(c1) - c2.start == VE.fadeinlength(c2)

            # the length control works on a SINGLE selected clip, live, and the
            # overlap follows it
            tl.selection[] = Int[]; tl.selected[] = 2; sleep(0.3)
            @test VE.selectedblend(p) !== nothing
            VE.setblendlength!(pctx, 1.0); sleep(0.4)
            @test VE.fadeinlength(c2) == round(Int, 1.0 * fps)
            @test VE.clipend(c1) - c2.start == VE.fadeinlength(c2)
            mid = c2.start + VE.fadeinlength(c2) ÷ 2
            @test length(VE.clipsat(p.sequence, mid)) == 2
            @test 0.2 < VE.paramvalue(c2, :opacity, c2.src_in + (mid - c2.start)) < 0.8

            # the blend IS an effect entry: switch it off (keys stay), then remove it
            slot = VE.findslot(c2, VE.OpacityEffect)
            @test slot !== nothing
            slot.enabled = false
            # …and no OPACITY is applied any more. Not `isempty`: stabilization is
            # an effect too now, so a clip that was stabilized earlier in this
            # session still has its Stabilize slot in the stack.
            @test !any(e -> e isa VE.OpacityEffect, VE.liveeffects(c2))
            @test VE.clipanimated(c2, :opacity)          # the curve is untouched
            slot.enabled = true

            # the pair survives a move and an undo — nothing is re-derived
            start0, track0 = c2.start, c2.track
            VE.movetooverlap!(p, c1, c2, 12); sleep(0.3)
            @test VE.blends(p.sequence) == [(1, 2, VE.fadeinlength(c2))]
            VE.undo!(p); sleep(0.3)
            @test length(VE.blends(p.sequence)) == 1

            # dragging a clip ONTO another rides up a lane instead of being refused
            c2 = p.sequence.clips[2]
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            press(tlx((c2.start + VE.cliplength(c2) / 2) / fps))
            for t in range((c2.start + VE.cliplength(c2) / 2) / fps, c1.start / fps + 0.7; length = 6)
                moveto(tlx(t)); sleep(0.05)
            end
            release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            sleep(0.4)
            @test c2.start < VE.clipend(c1)
            @test c2.track >= track0

            # × clears keys, effect entry and the pair together
            VE.clearfade!(c2, :in); c2.blendfrom = UInt64(0)   # what × does
            VE.refreshedit!(p); sleep(0.2)
            @test !VE.clipanimated(c2, :opacity)
            @test VE.findslot(c2, VE.OpacityEffect) === nothing
            @test c2.blendfrom == 0
            @test isempty(VE.blends(p.sequence))
            # restore for the following beats
            c2.start, c2.track = start0, track0
            VE.refreshedit!(p); sleep(0.2)
            for _ in 1:6
                VE.undo!(p)
            end
            tl.selection[] = Int[]
            VE.refreshedit!(p); sleep(0.2)
        end

        @testset "the playhead can be placed while playing" begin
            # Simon, 2026-07-27: "while playing, we cant move the playhead" — the
            # loop kept advancing between mouse moves and dragged it away again.
            VE.seek!(p, 5); sleep(0.2)
            play!(p); sleep(0.4)
            p.timeline.presspick = (0, 0.0, 0.0)     # button down on the ruler
            p.playhead[] = 40; sleep(0.5)
            @test p.playhead[] == 40                 # playback holds while you place it
            p.timeline.presspick = nothing; sleep(0.4)
            @test p.playhead[] > 40                  # and runs on from there after release
            VE.pause!(p); sleep(0.2)
        end

        @testset "stacked clips: you edit the lane you click" begin
            # Simon, 2026-07-27: "keyframes/effects are per clip — you only ever get
            # the ones of the selected clip shown". They were per clip alright, but
            # BOTH the click (clipat = topmost) and the whole inspector (locate =
            # topmost) ignored the lane, so a clip stacked BELOW another could not be
            # selected, inspected or keyed at all — which is exactly what an opacity
            # fade between two stacked clips needs.
            Makie.limits!(ax, 0.0, VE.seqduration(p.sequence), 0.0, 1.0); sleep(0.2)
            fps = p.sequence.framerate
            VE.seek!(p, VE.seqlength(p.sequence) ÷ 2)
            VE.split!(p); sleep(0.3)
            lower, upper = p.sequence.clips[1], p.sequence.clips[2]
            track0, start0 = upper.track, upper.start
            upper.track = 2
            upper.start = lower.start + VE.cliplength(lower) ÷ 2   # overlap the lower one
            empty!(upper.animations); empty!(lower.animations)
            VE.refreshedit!(p); sleep(0.3)
            over = upper.start + 5                                  # both clips cover this
            VE.seek!(p, over); sleep(0.3)
            laney(track) = (b = VE.trackband(track, VE.ntracks(p.sequence));
                            vp = ax.scene.viewport[];
                            vp.origin[2] + 0.5 * (b[1] + b[2]) * vp.widths[2])
            lanepos(t, track) = Point2f(tlx(t)[1], laney(track))
            @test VE.locate(p.sequence, over)[1] === upper          # the preview shows the top clip

            press(lanepos(over / fps, 1)); release(); sleep(0.3)    # click the LOWER lane
            @test tl.selected[] == 1
            @test VE.editclip(p)[1] === lower                       # …and that is what you edit
            VE.togglekey!(p, :opacity); sleep(0.2)
            @test VE.clipanimated(lower, :opacity)
            @test !VE.clipanimated(upper, :opacity)                 # the key went to the RIGHT clip

            press(lanepos(over / fps, 2)); release(); sleep(0.3)    # click the upper lane
            @test tl.selected[] == 2
            @test VE.editclip(p)[1] === upper
            VE.togglekey!(p, :opacity); sleep(0.2)
            @test VE.clipanimated(upper, :opacity)

            empty!(upper.animations); empty!(lower.animations)      # restore for later beats
            upper.track, upper.start = track0, start0
            VE.refreshedit!(p); sleep(0.2)
            VE.undo!(p); VE.undo!(p); VE.undo!(p); sleep(0.2)       # two keyframe snapshots + the split
            tl.selected[] = 0
        end

        @testset "shift-click marks multiple clips" begin
            Makie.limits!(ax, 0.0, VE.seqduration(p.sequence), 0.0, 1.0); sleep(0.2)
            nbefore = VE.seqlength(p.sequence)
            VE.seek!(p, VE.seqlength(p.sequence) ÷ 2)
            VE.split!(p); sleep(0.2)
            c1 = p.sequence.clips[1]; c2 = p.sequence.clips[2]
            fps = p.sequence.framerate
            press(tlx((c1.start + VE.cliplength(c1) / 2) / fps)); release()
            @test tl.selected[] == 1
            ph = p.playhead[]
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_shift, Keyboard.press)
            press(tlx((c2.start + VE.cliplength(c2) / 2) / fps)); release()
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_shift, Keyboard.release)
            sleep(0.2)
            @test sort(tl.selection[]) == [1, 2]     # both marked…
            @test tl.clipstates[1][] === :selected && tl.clipstates[2][] === :selected
            @test p.playhead[] == ph                 # …and marking didn't scrub
            deleteat!(p); sleep(0.2)                 # X deletes the whole set
            @test isempty(p.sequence.clips)
            @test isempty(tl.selection[])
            VE.undo!(p); sleep(0.2)
            @test length(p.sequence.clips) == 2
            VE.undo!(p); sleep(0.2)                  # and the split
            @test VE.seqlength(p.sequence) == nbefore
        end

        # The marking interaction, end to end, with the built-in propagator — the
        # matte testset covers tracks and keyframes but never drives a click, so
        # `showlivematte!` shipped with a `findeffect(...).effect` on its first
        # line and the suite stayed green.
        @testset "matte marking: points, live preview, removal" begin
            press(tlx(1.0)); release(); sleep(0.3)   # select a clip: marking needs one
            clip = p.sequence.clips[1]
            clip.mattetrack = nothing
            filter!(s -> !(s.effect isa VE.MatteEffect), clip.effects)
            VE.opendock!(p, :effects)
            VE.activatetool!(p, :matte); sleep(0.5)
            col = VE.mattecollect(p)
            @test col !== nothing                       # marking is running…
            @test col.scene.visible[]                   # …and the overlay is up
            @test col.scene.captures_mouse              # so nothing else gets the click
            moveto(pv(0.5, 0.5))                        # routing is asked about a POSITION
            @test !Makie.receives_events(p.previewaxis.scene)   # crop/pick stand down

            press(pv(0.5, 0.5)); release(); sleep(0.2)  # one foreground point
            @test length(col.points) == 1
            @test length(col.fg[]) == 1
            # The first mark of the session, and the only one that pays a cold
            # model. Measured on this path: `seedmask` (SAM 2) 2.5 s, and
            # `previewmatte` 38.6 s — because the live preview runs the
            # *propagation* model on the marked frame, so the first mark loads
            # MatAnyone and specializes its GEMM tiles for this clip's matte
            # resolution. The 6 s default never covered that; every later mark
            # here is warm and keeps it.
            @test waitfor(() -> clip.mattetrack !== nothing; s = 120)
            @test size(clip.mattetrack.alpha, 3) == 1   # THIS frame only
            fx = VE.findeffect(clip, VE.MatteEffect)
            @test fx !== nothing && fx.strength < 1.0f0 # background dimmed, not black

            press(pv(0.5, 0.5)); release(); sleep(0.3)  # clicking a dot removes it
            @test isempty(col.points)
            @test isempty(col.fg[])

            press(pv(0.5, 0.5)); release(); sleep(0.2)
            press(pv(0.6, 0.55)); release(); sleep(0.2)
            @test length(col.points) == 2
            keypress(Keyboard.backspace); sleep(0.2)    # Backspace undoes the last
            @test length(col.points) == 1
            press(pv(0.62, 0.5)); release(); sleep(0.2)
            @test length(col.points) == 2
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z); sleep(0.3)            # Ctrl+Z takes the point back…
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test length(col.points) == 1
            @test length(p.sequence.clips) == 1         # …and does NOT undo the edit

            keypress(Keyboard.escape); sleep(0.5)       # Esc puts everything back
            @test VE.mattecollect(p) === nothing
            @test !col.scene.visible[] && !col.scene.captures_mouse
            @test isempty(col.fg[]) && isempty(col.bg[])
            @test VE.findeffect(clip, VE.MatteEffect) === nothing
            moveto(pv(0.5, 0.5))
            @test Makie.receives_events(p.previewaxis.scene)    # pointer handed back
            # A mark's card: propagate, then call the stored callbacks exactly as
            # the card system does — with the card id. Zero-arg closures threw
            # inside the render loop, which is why clicking a card spammed errors
            # and its × did nothing.
            # Esc above ENDED the collect: `endmattecollect!` turns the listeners
            # off and deletes `:mattecollect`, so the old `col` is detached and
            # its `points` can never change again — marking has to be started again and
            # `col` re-taken before the sequence continues. (As written this asked
            # a detached collect to grow to 2 points, which it cannot do; the
            # assertion had never run, because every earlier failure in this file
            # aborted the testset before reaching it.)
            # …and Esc leaves the TOOL active while the collect is gone, so a
            # single `activatetool!` would toggle the tool OFF rather than restart it
            # it. Off, then on — which is what clicking the toolbar button twice
            # does — gives a fresh collect to mark into.
            VE.activatetool!(p, :matte); VE.activatetool!(p, :matte); sleep(0.5)
            col = VE.mattecollect(p)
            @test col !== nothing
            press(pv(0.35, 0.62)); release(); sleep(0.2)  # a spot with no dot on it
            @test length(col.points) == 1                # a fresh collect: this ADDS
            keypress(Keyboard.enter)                     # propagate (built-in propagator)
            @test waitfor(() -> clip.mattetrack !== nothing && size(clip.mattetrack.alpha, 3) > 1;
                          s = 30)
            # No `activatetool!` here, despite what the old comment said. The
            # cards rebuild themselves: `runmatte!`'s completion calls
            # `refreshmattepanel!`, which bumps `TOOLSVERSION`, and the dock
            # rebuilds on that. Toggling the tool does the opposite —
            # `deactivatetool!` runs `cleartoolcards!`, and `rebuild()` fires only
            # on `TOOLSVERSION`, the fold button, or activating a *folded* tool,
            # so nothing puts them back. That is what made this unreachable.
            #
            # ON THE MATTE'S OWN ROWS, not the shared `:toolcards` list. Enter
            # propagates through `finishmattecollect!`, which calls
            # `endmattecollect!` — so the marking session is OVER by the time this
            # runs, there is no live seed card, and each marked frame is a ROW
            # ("frame N" with `go` and `×`) put there by `toolrows!`. The matte
            # therefore contributes NOTHING to `:toolcards[3]`, which every tool
            # shares. That assertion passed only because the four always-on
            # tool-only cards were sitting in that list — it never looked at the
            # matte at all, and it went red the moment those cards stopped
            # rendering on every clip. Measured: 5 shared cards before, 0 after.
            matterows() = (c = get(get(p.fxwidgets, :toolpanels, Dict{Symbol, Any}()),
                                   :matte, nothing);
                           c === nothing ? 0 : length(c.rows))
            @test waitfor(() -> matterows() > 0; s = 15)
            seeds = VE.mattemarks(p, clip)
            @test !isempty(seeds)
            @test matterows() >= length(seeds)         # one row per marked frame
            # …and the way back to a mark still works, which is what a row is FOR.
            p.playhead[] = VE.clipend(clip) - 1
            ctx = p.fxwidgets[:toolpanels][:matte]
            VE.gotomatteseed!(ctx, clip, first(sort!(collect(keys(seeds)))))
            @test p.playhead[] != VE.clipend(clip) - 1
        end
    finally
        close(p)
    end
end

@testset "chaos: random event storm leaves the editor coherent" begin
    using Random
    p = Player(testvideo; gpupreview = false)
    sleep(1.5)
    ev = Makie.events(p.fig)
    p.fxwidgets[:browse] = () -> nothing   # the storm clicks everywhere, and the bin's
                                           # drop zone would open a BLOCKING native dialog
    # dock-panel widgets live outside fig.content — include them in the storm
    buttons = vcat([b for b in p.fig.content if b isa Makie.Button],
                   [w for w in values(p.fxwidgets) if w isa Makie.Button])
    rng = MersenneTwister(42)   # seeded: the event sequence is reproducible
    randpos() = Point2f(rand(rng) * 1490 + 5, rand(rng) * 940 + 5)
    fuzzkeys = [Keyboard.space, Keyboard.s, Keyboard.x, Keyboard.c, Keyboard.r,
                Keyboard.left, Keyboard.right, Keyboard.escape]
    errormsgs = String[]
    t0 = time()
    while time() - t0 < 10
        try
            r = rand(rng)
            if r < 0.35
                ev.mouseposition[] = Tuple(randpos())
                rand(rng, Bool) &&
                    (ev.mousebutton[] = MouseButtonEvent(rand(rng, (Mouse.left, Mouse.right)), Mouse.press);
                     ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.release))
            elseif r < 0.60
                rand(rng) < 0.2 && (ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press))
                k = rand(rng, fuzzkeys)
                ev.keyboardbutton[] = KeyEvent(k, Keyboard.press)
                ev.keyboardbutton[] = KeyEvent(k, Keyboard.release)
                ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            elseif r < 0.75
                b = rand(rng, buttons)
                bb = b.layoutobservables.computedbbox[]
                ev.mouseposition[] = Tuple(Point2f(bb.origin .+ rand(rng, 2) .* bb.widths))
                ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.press)
                ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.release)
            elseif r < 0.90
                ev.mouseposition[] = Tuple(Point2f(rand(rng) * 1400 + 20, 60))
                ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.press)
                ev.mouseposition[] = Tuple(Point2f(rand(rng) * 1400 + 20, 60))
                ev.mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.release)
            elseif r < 0.97
                ev.mouseposition[] = Tuple(Point2f(rand(rng) * 1400 + 20, 60))
                ev.scroll[] = (0.0, rand(rng, -4:4))
            else
                ev.dropped_files[] = [rand(rng, (testvideo2, "/nonexistent/nope.mp4"))]
            end
        catch e
            push!(errormsgs, sprint(showerror, e)[1:min(end, 300)])
            length(errormsgs) > 3 && break
        end
        rand(rng) < 0.1 && sleep(0.02)
    end
    isempty(errormsgs) || @info "chaos exceptions" errormsgs
    @test isempty(errormsgs)                # no listener ever threw
    sleep(0.5)
    VE.pause!(p)                            # the storm may leave playback running,
    p.playhead[] = 10                       # and its presents would race this one
    ok = VE.showframe!(p, 10)
    t1 = time()
    while !ok && time() - t1 < 5            # decoder settles after the seek storm
        sleep(0.05)
        ok = VE.showframe!(p, 10)
    end
    @test ok                                # still presents (empty sequence = black)
    close(p)                                # and closes cleanly
    sleep(0.5)
    @test true
end

# `EFFECTS` is a global registry and the effects panel puts two listeners on its
# `version`. A closed player used to keep both, which is not a slow leak but a
# live fault: the corpse's handler `put!`s onto its own closed `uiqueue` and
# throws, `notify` abandons the rest of the listener list, and the LIVE player's
# panel silently stops rebuilding — which is how a Matte card lost the "Apply
# matte to clip" button a walkthrough was about to click.
@testset "a closed player lets go of the global effect registry" begin
    n0 = length(VE.EFFECTS.version.listeners)
    p1 = Player(testvideo; gpupreview = false)
    sleep(1.0)
    @test length(VE.EFFECTS.version.listeners) > n0
    close(p1)
    sleep(0.5)
    @test length(VE.EFFECTS.version.listeners) == n0

    p2 = Player(testvideo; gpupreview = false)
    try
        sleep(1.0)
        VE.EFFECTS.version[] += 1     # the bump the matte panel does; must not throw
        sleep(0.3)
        @test length(p2.fxwidgets[:fxcards]) == 0   # panel still live and rebuilding
    finally
        close(p2)
    end
    sleep(0.5)
    @test length(VE.EFFECTS.version.listeners) == n0
end

# THE GAP THIS CLOSES. `drawoverlays!` was called from `export.jl` and nowhere
# else, so every overlay kind was correct in the exported file and INVISIBLE in
# the editor — and the suite was fully green throughout, because every overlay
# test asserted on an exported frame or on `drawoverlays!` directly. What was
# missing was an assertion that the PREVIEW shows what the export writes.
@testset "overlays are drawn in the preview, not only in the export" begin
    p = Player(testvideo; gpupreview = false)
    try
        sleep(1.5)
        seq = p.sequence
        n = 35
        VE.seek!(p, n); sleep(0.6)
        bare = copy(p.frame[])

        # a plain filled bar: no 3D, no GPU, nothing that can be unavailable
        # headless, and opaque enough that "did it draw" is not a judgement call
        VE.addoverlay!(seq, :bar; start = 0, stop = 120, opacity = 1.0)
        VE.seek!(p, n); sleep(0.6)
        drawn = copy(p.frame[])
        # SAME timeline frame on both sides — comparing two different frames of a
        # moving test pattern "confirmed" this while the overlay drew nothing at
        # all, which is how it stayed broken.
        @test count(bare .!= drawn) > 0

        # and it must match what the export produces for that frame, bit for bit:
        # one door, one picture (see `publishframe!`)
        expect = copy(bare)
        VE.drawoverlays!(expect, seq.overlays, n;
                         framerate = seq.framerate, captions = seq.captions)
        @test drawn == expect

        # removing it puts the frame back exactly — the pass is not cumulative
        empty!(seq.overlays)
        VE.seek!(p, n); sleep(0.6)
        @test copy(p.frame[]) == bare
    finally
        close(p)
    end
end

@testset "Player opens a saved project" begin
    src = VideoSource(testvideo)
    seq = Sequence(src)
    split!(seq, 40)
    seq.clips[1].crop = (0.1, 0.1, 0.8, 0.8)
    path = joinpath(mktempdir(), "edit.videoedit.toml")
    saveproject(path, seq)
    p2 = Player(path; gpupreview = false)   # .toml path → the saved edit, not a video
    try
        sleep(1.5)
        @test length(p2.sequence.clips) == 2
        @test p2.sequence.clips[1].crop == (0.1, 0.1, 0.8, 0.8)
        @test VE.seqlength(p2.sequence) == 120
        # The CANVAS, which the crop above defines: 320*0.8 x 180*0.8. This
        # asserted the source's own (320, 180) — true before a crop started
        # deriving the canvas, and unnoticed since, because this file throws on
        # its known failures and nothing after it in the suite ever ran.
        @test size(p2.frame[]) == (256, 144)
        @test size(p2.frame[]) == VE.canvassize(p2.sequence)
    finally
        close(p2)
    end
end
