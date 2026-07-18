# Records a walkthrough of an "LLM" driving the editor THROUGH THE MCP TOOL
# LAYER to turn a handheld bird clip into a seamless loop:
#   get_state → trim to a stable window → analyze_motion (camera lock) →
#   analyze_color → find_loop (content frame-matching) → trim to the loop →
#   play → export a looping GIF.
# Every edit goes through VideoEditor.calltool — the exact code path an MCP
# client hits — while the editor window is captured to a video.

ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
ENV["XAUTHORITY"] = get(ENV, "XAUTHORITY", "/run/user/1000/xauth_hqQZRv")
ENV["XDG_RUNTIME_DIR"] = get(ENV, "XDG_RUNTIME_DIR", "/run/user/1000")

using VideoEditor, GLMakie, Lava
import VideoEditor as VE
import VideoEditor.JSON as JSON
import VideoEditor.Makie as MK
using VideoEditor.Makie: VideoStream, recordframe!, save

GLMakie.activate!(; visible = false)

const BIRD = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
const OUT_MP4 = joinpath(@__DIR__, "..", "..", "..", "media", "loop_demo.mp4")
const OUT_GIF = joinpath(tempdir(), "bird_loop.gif")

player = Player(BIRD; analysisbackend = LavaBackend())
srv = VE.mcpserve!(player; port = 8901)
sleep(2.0)

# call a tool the way an MCP client would; parse its JSON payload
function mcp(name; kw...)
    r = VE.calltool(srv, name, Dict(string(k) => v for (k, v) in kw))
    txt = try r["content"][1]["text"] catch; "" end
    return try JSON.parse(txt) catch; txt end
end
caption(s) = (player.status[] = s)

# warm the GPU kernels off-camera (first Lava analysis compiles ~40s)
let warm = VE.Clip(player.sequence.clips[1].source, 720, 770, 0, (0.0, 0.0, 1.0, 1.0)),
    done = Channel{Bool}(1)
    VE.rungpu(() -> (VE.analyzemotion!(warm; backend = player.analysisbackend); put!(done, true)), player)
    take!(done)
end
@info "GPU warm — recording"

io = VideoStream(player.fig; framerate = 30, px_per_unit = 1)
hold(sec) = for _ in 1:round(Int, sec * 30); sleep(1 / 30); recordframe!(io); end
# capture while waiting for `cond`, compressed (~1 frame / 0.3s wall)
function waitcap(cond; maxsec = 20)
    t0 = time(); i = 0
    while time() - t0 < maxsec && !cond()
        sleep(0.1); i += 1
        i % 3 == 0 && recordframe!(io)
    end
    hold(0.5)
end
hasmotion() = player.sequence.clips[1].motiontrack !== nothing
hascolor() = player.sequence.clips[1].motiontrack !== nothing &&
             any(c -> c.colortrack !== nothing, player.sequence.clips)

# [1] the raw clip — a shaky 39s handheld bird video
caption("MCP: get_state — a 39s handheld clip of a bird at a nest box")
mcp("get_state"); mcp("seek"; time = 14.0); hold(2.0)

# [2] trim to a stable 8s window (skip the shaky zoom-in intro), src 720–1200
caption("MCP: trim to a stable 8-second window")
mcp("split_at"; time = 12.0); hold(0.8)
mcp("split_at"; time = 20.0); hold(0.8)
mcp("delete_clip_at"; time = 6.0); hold(0.8)     # drop the intro [0,12]
mcp("delete_clip_at"; time = 15.0); hold(0.8)    # drop the tail  → keep [0,8]
mcp("seek"; time = 2.0); hold(1.0)

# [3] camera-lock stabilization (GPU) — the nest box locks in place
caption("MCP: analyze_motion (camera lock) — locking the camera like a tripod")
mcp("analyze_motion"; time = 2.0, mode = "similarity")
waitcap(hasmotion; maxsec = 20)
mcp("seek"; time = 2.0); hold(2.0)               # refresh → stabilized + auto-cropped

# [4] color/exposure stabilization — kill the flicker
caption("MCP: analyze_color — removing exposure flicker")
mcp("analyze_color"; time = 2.0)
waitcap(hascolor; maxsec = 12)
mcp("seek"; time = 2.0); hold(1.0)

# [5] find the seamless loop point (content frame-matching)
caption("MCP: find_loop — matching frames so the bird is ~back where it started")
hold(0.8)
loop = mcp("find_loop"; time = 2.0, min_seconds = 1.5, max_seconds = 5.0)
t0 = Float64(loop["start_time"]); t1 = Float64(loop["end_time"])
caption("found a $(loop["loop_seconds"])s loop — trimming to it")
hold(1.2)

# [6] trim the timeline down to just the loop
if t1 < VE.seqduration(player.sequence) - 0.05
    mcp("split_at"; time = t1); hold(0.7)
    mcp("delete_clip_at"; time = t1 + 0.3); hold(0.7)   # drop everything after
end
if t0 > 0.01
    mcp("split_at"; time = t0); hold(0.7)
    mcp("delete_clip_at"; time = t0 / 2); hold(0.7)     # drop everything before
end
mcp("seek"; time = 0.0); hold(1.0)

# [7] play the loop a few times
caption("the finished loop — bird returns to the same spot, no reversing")
mcp("play"); hold(6.0); mcp("pause")

# [8] export as a forever-looping GIF
caption("MCP: export as a looping GIF")
hold(0.6)
res = mcp("export"; path = OUT_GIF, format = "gif", fps = 20, loop = 0)
caption("exported $(basename(OUT_GIF)) — a seamless, stabilized bird loop")
mcp("seek"; time = 0.0); mcp("play"); hold(4.0); mcp("pause")

save(OUT_MP4, io)
@info "saved walkthrough" OUT_MP4 gif = OUT_GIF exported = res
VE.stop!(srv)
close(player)
