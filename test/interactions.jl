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
        stabopen() = (p.fxwidgets[:stabopen](); sleep(0.3))
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
            @test p.cropmode[]                 # persistent crop tool stays armed (re-drag to refine)
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

        @testset "hold-to-compare" begin
            VE.opendock!(p, :effects); sleep(0.3)
            comparebtn = p.fxwidgets[:compare]   # inspector: it bypasses the WHOLE stack
            center = comparebtn.layoutobservables.computedbbox[]
            pos = Point2f(center.origin .+ center.widths ./ 2)
            @test p.applytracks[]
            press(pos)
            @test !p.applytracks[]
            release()
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

        @testset "drop a second source" begin
            # state: two clips (0.7s split of a 45-frame clip), 45 frames total
            nclips = length(p.sequence.clips)
            ev.dropped_files[] = [testvideo2]      # what GLFW delivers on file drop
            @test length(p.sequence.clips) == nclips + 1
            @test length(tl.clipplots) == nclips + 1
            added = p.sequence.clips[end]
            @test added.source.width == 480
            @test added.start == 45                # appended at the sequence end
            @test maximum(ax.finallimits[])[1] >= VE.seqduration(p.sequence) - 1e-6

            # scrubbing onto the new clip switches the preview buffers
            press(tlx((45 + 15) / 30)); release()
            sleep(0.6)
            @test size(p.frame[]) == (480, 270)

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

            stabbtn = p.fxwidgets[:analyze]
            stabcenter() = (sc = stabbtn.layoutobservables.computedbbox[]; Point2f(sc.origin .+ sc.widths ./ 2))
            press(stabcenter()); release()       # arms the pick AND closes the modal
            @test p.onpick !== nothing

            keypress(Keyboard.escape)            # Esc cancels the pick
            @test p.onpick === nothing

            # re-arm (reopen the modal, click Stabilize again) then pick on the preview
            # (pause first: a second click at the same spot within the dblclick window
            # would register as dblclick, not click)
            sleep(0.5)
            stabopen()
            press(stabcenter()); release()
            @test p.onpick !== nothing
            clip = VE.locate(p.sequence, p.playhead[])[1]
            clip.motiontrack = nothing
            press(pv(0.5, 0.5)); release()       # modal closed by the arm → click reaches the preview
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
            # the analysis rebuilt the Stabilization section — re-fetch the live menu
            menu = p.fxwidgets[:modemenu]
            press(mpos(0.5)); release()
            press(mpos(-0.5)); release()
            @test menu.selection[] == :similarity
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
            # goes) — pressing Remove on a clean clip says so instead of doing damage
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release()
            @test clip.motiontrack === nothing
            @test waitfor(() -> occursin("no stabilization", p.status[]))
            stabclose()
        end

        @testset "proxy swap keeps the preview consistent" begin
            press(tlx(0.3)); release()               # onto clip 1
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

        @testset "toolbar tools: arm, click-to-act, cursor" begin
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
            @test p.tool[] == :split               # persistent blade stays armed (Esc/✂ to stop)
            @test p.playhead[] == playhead_before  # the click cut, didn't scrub
            @test any(c -> c.start == 90, p.sequence.clips)  # cut at 3.0 s @30fps
            sleep(0.5)
            click(toolbtn("↶"))                   # undo tool
            @test length(p.sequence.clips) == nclips
            sleep(0.5)
            click(toolbtn("▢"))                   # crop tool arms crop mode
            @test p.tool[] == :crop
            @test p.cropmode[]
            keypress(Keyboard.escape)
            @test !p.cropmode[]
            @test p.tool[] == :none
        end

        @testset "plain-drag lifts a clip to a new track" begin
            # press mid-clip (starts as a scrub), then drag UP into the marked
            # new-track zone — the gesture converts into a clip move, no Ctrl
            clip = p.sequence.clips[1]
            @test clip.track == 1
            sleep(0.5)
            press(tlx(3.0))
            @test tl.presspick !== nothing       # button down on the ruler (not a drag yet)
            vp = ax.scene.viewport[]
            moveto(Point2f(tlx(3.5)[1], vp.origin[2] + 0.93 * vp.widths[2]))
            @test !tl.scrubbing[]
            @test tl.dragclip !== nothing        # converted to a move
            @test tl.dragtrack == 2
            release()
            @test clip.track == 2
            @test VE.ntracks(p.sequence) == 2
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.press)
            keypress(Keyboard.z)                 # undoable like any edit
            ev.keyboardbutton[] = KeyEvent(Keyboard.left_control, Keyboard.release)
            @test p.sequence.clips[1].track == 1
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
            for k in (:tools, :media, :effects, :export, :none, :effects)
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
            # Stabilization is findable the same way, but it is a TOOL now: the entry
            # takes you to the Tools dock instead of stacking a section (Simon,
            # 2026-07-27 — "stabilization should move to tools")
            p.fxwidgets[:paletteopen]()
            p.fxwidgets[:palettequery][] = "stab"
            keypress(Keyboard.enter)
            @test waitfor(() -> p.dockopen[] === :tools)
            @test haskey(p.fxwidgets, :modemenu) && haskey(p.fxwidgets, :analyze)
            VE.opendock!(p, :effects); sleep(0.3)
            # PARAMETER labels hit too: "bright" finds the Color kind (Premiere-style)
            nfx2 = length(clip.effects)
            p.fxwidgets[:paletteopen]()
            p.fxwidgets[:palettequery][] = "bright"
            keypress(Keyboard.enter)
            @test length(clip.effects) == nfx2 + 1
            @test clip.effects[end].effect isa VE.ColorEffect
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
            VE.armkeyframe!(p, :brightness)
            VE.armkeyframe!(p, :sharpen)
            VE.setkey!(clip.animations[:brightness], fkey, VE.paramspec(:brightness).hi)
            VE.setkey!(clip.animations[:sharpen], fkey, VE.paramspec(:sharpen).lo)
            VE.armkeyframe!(p, :brightness)          # focus = brightness
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
            trio() = p.fxwidgets[:kfacc_contrast]    # re-fetch: arming rebuilds the stack

            # --- ◆ arms: first key at the playhead, label flips ◇ → ◆ ---
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
            notify(trio()[2].clicks); sleep(0.3)                   # re-arm at the playhead
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
            VE.armkeyframe!(p, :contrast); sleep(0.3)
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
            VE.opendock!(p, :tools)                  # card clicks need the dock open
            VE.activatetool!(p, :loopfinder)
            @test VE.activetoolname(p)[] === :loopfinder
            ctx = VE.activetool(p)[2]
            @test waitfor(() -> ctx.state !== nothing && !isempty(ctx.state[:refs]); s = 25)
            sleep(0.4)                               # uiqueue draws card + hints
            @test length(p.fxwidgets[:toolcards][3]) == 1   # reference card added
            pts = ctx.state[:hintpts][]
            @test !isempty(pts)
            # ▼ click CUTS there — the timeline keeps its length, the tool stays armed
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
            @test waitfor(() -> length(p.fxwidgets[:toolcards][3]) == 2; s = 25)
            cards = p.fxwidgets[:toolcards][3]
            @test ctx.state[:active][] == 2
            # clicking the FIRST card highlights its markers again
            bb = cards[1][2].layoutobservables.computedbbox[]
            press(Point2f(bb.origin .+ bb.widths ./ 2)); release(); sleep(0.2)
            @test ctx.state[:active][] == 1
            # × on the first card removes THAT reference; the second remains
            @test cards[1][6] !== nothing
            notify(cards[1][6].clicks); sleep(0.3)
            @test length(ctx.state[:refs]) == 1
            @test length(p.fxwidgets[:toolcards][3]) == 1
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
            @test isempty(collect(VE.liveeffects(c2)))
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
    finally
        close(p)
    end
end

@testset "chaos: random event storm leaves the editor coherent" begin
    using Random
    p = Player(testvideo; gpupreview = false)
    sleep(1.5)
    ev = Makie.events(p.fig)
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
        @test size(p2.frame[]) == (320, 180)   # presents from the project's source
    finally
        close(p2)
    end
end
