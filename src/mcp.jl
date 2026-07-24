"""
    mcpserve!(player; port=8765) -> MCPServer

Expose the live editor as an MCP server (streamable HTTP transport), so an
AI agent can inspect and edit the open project. Attach from Claude Code:

    claude mcp add --transport http videoedit http://localhost:8765

Tools: get_state, get_frame (rendered preview incl. stabilization/effects/
crop), seek, play, pause, split_at, delete_clip_at, move_clip, set_crop,
set_color, set_blur, set_sharpen, analyze_color, analyze_motion (both
async — poll get_state), export, save_project.

All editor mutations are funneled through a command channel executed on
the task that created the server (call `mcpserve!` from the main thread),
so HTTP handler threads never touch Makie observables directly.
"""
mutable struct MCPServer
    const player::Player
    const commands::Channel{Function}
    server::Any
    executor::Task

    function MCPServer(player::Player)
        return new(player, Channel{Function}(32), nothing)
    end
end

function mcpserve!(player::Player; port::Integer = 8765)
    srv = MCPServer(player)
    srv.executor = @async for f in srv.commands
        f()
    end
    srv.server = HTTP.serve!("127.0.0.1", port) do req
        handlemcp(srv, req)
    end
    @info "MCP server listening — attach with: claude mcp add --transport http videoedit http://localhost:$port"
    return srv
end

function stop!(srv::MCPServer)
    close(srv.commands)
    srv.server === nothing || close(srv.server)
    return nothing
end

"Run `f()` on the editor task and return its result to the calling (HTTP) task."
function runoneditor(f::Function, srv::MCPServer)
    result = Channel{Any}(1)
    put!(srv.commands, () -> put!(result, try
        f()
    catch e
        Dict("isError" => true, "message" => sprint(showerror, e))
    end))
    return take!(result)
end

# ------------------------------------------------------------------ protocol

function handlemcp(srv::MCPServer, req::HTTP.Request)
    req.method == "POST" || return HTTP.Response(405)
    msg = JSON.parse(String(req.body))
    method = get(msg, "method", "")
    id = get(msg, "id", nothing)
    id === nothing && return HTTP.Response(202)  # notification
    result = if method == "initialize"
        Dict("protocolVersion" => get(get(msg, "params", Dict()), "protocolVersion", "2025-03-26"),
             "capabilities" => Dict("tools" => Dict()),
             "serverInfo" => Dict("name" => "VideoEditor", "version" => "0.1.0"))
    elseif method == "tools/list"
        Dict("tools" => tooldefinitions())
    elseif method == "tools/call"
        calltool(srv, msg["params"]["name"], get(msg["params"], "arguments", Dict()))
    else
        return jsonrpc(id; error = Dict("code" => -32601, "message" => "unknown method $method"))
    end
    return jsonrpc(id; result)
end

function jsonrpc(id; result = nothing, error = nothing)
    body = Dict{String, Any}("jsonrpc" => "2.0", "id" => id)
    error === nothing ? (body["result"] = result) : (body["error"] = error)
    return HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(body))
end

textcontent(x) = Dict("content" => [Dict("type" => "text", "text" => x isa String ? x : JSON.json(x))])

tool(name, description, props = Dict(), required = String[]) =
    Dict("name" => name, "description" => description,
         "inputSchema" => Dict("type" => "object", "properties" => props, "required" => required))

num(description) = Dict("type" => "number", "description" => description)
str(description) = Dict("type" => "string", "description" => description)

function tooldefinitions()
    t = num("timeline time in seconds")
    tools = Any[
        tool("get_state", "Project state: clips (source ranges, timeline placement, crop, effects, stabilization flags), playhead, duration."),
        tool("get_frame", "Rendered preview PNG at a timeline time — includes stabilization, color correction, effects and crop (what export will produce).",
             Dict("time" => t, "width" => Dict("type" => "integer", "description" => "preview width px (default 480)")), ["time"]),
        tool("seek", "Move the playhead.", Dict("time" => t), ["time"]),
        tool("play", "Start playback."),
        tool("pause", "Pause playback."),
        tool("split_at", "Split the clip containing the given time.", Dict("time" => t), ["time"]),
        tool("delete_clip_at", "Ripple-delete the clip containing the given time.", Dict("time" => t), ["time"]),
        tool("move_clip", "Move clip (1-based index) to start at a new time. Fails on overlap.",
             Dict("clip" => Dict("type" => "integer"), "start_time" => t), ["clip", "start_time"]),
        tool("set_crop", "Set a clip's crop as a normalized rect (x, y from top-left, w, h in 0..1). Full frame = 0,0,1,1.",
             Dict("time" => t, "x" => num("left"), "y" => num("top"), "w" => num("width"), "h" => num("height")),
             ["time", "x", "y", "w", "h"]),
        tool("set_color", "Set color correction of the clip at time. Neutral: brightness 0, contrast 1, saturation 1, temperature 0.",
             Dict("time" => t, "brightness" => num("-1..1"), "contrast" => num("0.5..2"),
                  "saturation" => num("0..2"), "temperature" => num("-1..1 (positive=warmer)")), ["time"]),
        tool("set_blur", "Set gaussian blur sigma (0 = off) of the clip at time.",
             Dict("time" => t, "sigma" => num("0..8")), ["time", "sigma"]),
        tool("set_sharpen", "Set unsharp-mask amount (0 = off) of the clip at time.",
             Dict("time" => t, "amount" => num("0..2")), ["time", "amount"]),
        tool("analyze_color", "Start color/exposure flicker stabilization analysis for the clip at time (async — poll get_state for has_colortrack).",
             Dict("time" => t), ["time"]),
        tool("analyze_motion", "Start camera shake stabilization analysis for the clip at time (async, ~0.4x realtime — poll get_state for has_motiontrack).",
             Dict("time" => t, "mode" => str("similarity (default, camera lock — like a tripod) | tripod (affine, legacy) | perspective (also fix keystoning) | smooth (keep camera moves)")), ["time"]),
        tool("analyze_object", "Object lock: track the subject at the given position (normalized 0..1, from the top-left, in the clip's first frame) and pin it in place (async — poll get_state for has_motiontrack).",
             Dict("time" => t, "x" => num("0..1"), "y" => num("0..1")), ["time", "x", "y"]),
        tool("find_loop", "Find the best seamless-loop cut inside the clip at `time`: returns start_time/end_time (seconds) of the segment whose stabilized, brightness-normalized content matches most closely (e.g. a bird back in ~the same pose), so it loops with the least seam. Analyze motion first for a clean match. Trim to [start_time, end_time] then export as a looping gif.",
             Dict("time" => t, "min_seconds" => num("shortest loop, default 1.5"), "max_seconds" => num("longest loop, default 6")), ["time"]),
        tool("export", "Render the sequence to a file. `format` mp4 (default) or gif (animated, looping). Blocks until done.",
             Dict("path" => str("output path"), "format" => str("mp4 (default) or gif"),
                  "crf" => Dict("type" => "integer", "description" => "mp4 quality, default 20"),
                  "fps" => Dict("type" => "integer", "description" => "gif frame rate, default 15"),
                  "loop" => Dict("type" => "integer", "description" => "gif looping: 0 = forever (default), -1 = play once")), ["path"]),
        tool("save_project", "Save edit metadata as TOML.", Dict("path" => str("output path")), ["path"]),
        tool("add_source", "Append a video file as a new clip at the end of the timeline (framerate must match the sequence).",
             Dict("path" => str("video file path")), ["path"]),
    ]
    # registered effect plugins → one tool each (reflects new plugins on every list)
    for p in PLUGINS
        props = Dict{String, Any}("time" => t)
        for pr in p.params
            props[String(pr.name)] = num("$(pr.label) ($(pr.min)..$(pr.max), default $(pr.default))")
        end
        push!(tools, tool("effect_$(p.name)", "Apply the '$(p.label)' effect to the clip at `time`.", props, ["time"]))
    end
    push!(tools, tool("define_effect",
        "Author and register a NEW effect plugin at runtime, then use it via effect_<name> " *
        "(it also appears in the editor's Add-effect menu and is keyframable). `code` is a Julia " *
        "snippet that calls `registerplugin!(name::Symbol, label, params::Vector{FxParam}, kind)`, " *
        "where `kind(p) -> Pointwise((c,uv) -> Vec3f)` or `Stencil(radius) do sample,r,uv ... end`; " *
        "`c` is the pixel (Vec3f, 0..1), `uv` its coordinate (Vec2f, 0..1²), `sample(di,dj)` a neighbor. " *
        "Example: `registerplugin!(:invert, \"Invert\", FxParam[], p -> Pointwise((c,uv)->Vec3f(1,1,1)-c))`. " *
        "After defining, re-list tools to see effect_<name>.",
        Dict("code" => str("Julia snippet calling registerplugin!")), ["code"]))
    return tools
end

# ---------------------------------------------------------------- tool calls

function calltool(srv::MCPServer, name::String, args)
    player = srv.player
    seq = player.sequence
    attime(t) = clamp(round(Int, Float64(t) * seq.framerate), 0, max(seqlength(seq) - 1, 0))
    clipattime(t) = begin
        loc = locate(seq, attime(t))
        loc === nothing ? nothing : loc[1]
    end

    if name == "get_frame"
        img = renderpreview(player, Float64(args["time"]), Int(get(args, "width", 480)))
        io = IOBuffer()
        PNGFiles.save(io, PermutedDimsArray(img, (2, 1)))
        return Dict("content" => [Dict("type" => "image", "data" => Base64.base64encode(take!(io)),
                                       "mimeType" => "image/png")])
    elseif name == "export"
        # runs on the HTTP task: only reads edit metadata, and must not
        # starve the editor's executor (and with it the render loop) for its
        # multi-second duration. Stays on the CPU backend — GPU dispatches
        # would have to come from the pinned worker, and this call must
        # return the path synchronously.
        fmt = lowercase(String(get(args, "format", "mp4")))
        path = if fmt == "gif"
            exportgif(String(args["path"]), seq; fps = Int(get(args, "fps", 15)),
                      loop = Int(get(args, "loop", 0)))
        else
            crf = Int(get(args, "crf", 20))
            exportvideo(String(args["path"]), seq; encoder_options = (crf = crf, preset = "medium"))
        end
        return textcontent(Dict("exported" => path))
    elseif name == "find_loop"
        # decode-heavy read of the source; safe off the editor task
        clip = clipattime(args["time"])
        clip === nothing && return textcontent("no clip at that time")
        a, b, score = runanalysissync(player) do   # GPU decode runs on the worker
            findloop(clip; minseconds = Float64(get(args, "min_seconds", 1.5)),
                     maxseconds = Float64(get(args, "max_seconds", 6.0)),
                     backend = player.analysisbackend)
        end
        fps = seq.framerate
        # clip-relative source offsets → timeline seconds
        t0 = (clip.start + a) / fps
        t1 = (clip.start + b) / fps
        return textcontent(Dict("start_time" => round(t0, digits = 3),
                                "end_time" => round(t1, digits = 3),
                                "loop_seconds" => round((b - a) / fps, digits = 2),
                                "seam_score" => round(score, digits = 4)))
    elseif name == "define_effect"
        # author + register a new plugin (mutates the global registry, not the player)
        before = Set(p.name for p in PLUGINS)
        definepluginfromcode!(String(args["code"]))
        added = [String(p.name) for p in PLUGINS if !(p.name in before)]
        return textcontent(Dict("registered" => added,
            "call_with" => ["effect_$n" for n in added],
            "note" => "re-list tools (tools/list) to see the new effect_<name> tool"))
    end

    result = runoneditor(srv) do
        if name == "get_state"
            statedict(player)
        elseif name == "seek"
            player.playhead[] = attime(args["time"])
            "playhead at frame $(player.playhead[])"
        elseif name == "play"
            play!(player); "playing"
        elseif name == "pause"
            pause!(player); "paused"
        elseif name == "split_at"
            snapshot!(player)
            split!(seq, attime(args["time"]))
            refreshedit!(player)
            statedict(player)
        elseif name == "delete_clip_at"
            snapshot!(player)
            deleteclip!(seq, attime(args["time"]))
            player.playhead[] = clamp(player.playhead[], 0, max(seqlength(seq) - 1, 0))
            refreshedit!(player)
            statedict(player)
        elseif name == "move_clip"
            snapshot!(player)
            clip = seq.clips[Int(args["clip"])]
            ok = moveclip!(seq, clip, round(Int, Float64(args["start_time"]) * seq.framerate))
            refreshedit!(player)
            ok ? statedict(player) : "move rejected: would overlap another clip"
        elseif name == "set_crop"
            snapshot!(player)
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            clip.crop = (Float64(args["x"]), Float64(args["y"]), Float64(args["w"]), Float64(args["h"]))
            refreshedit!(player)
            "crop set"
        elseif name == "set_color"
            snapshot!(player)
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            seteffect!(clip, ColorEffect(brightness = get(args, "brightness", 0), contrast = get(args, "contrast", 1),
                                         saturation = get(args, "saturation", 1), temperature = get(args, "temperature", 0)))
            syncsliders!(player, clip)
            refreshedit!(player)
            "color set"
        elseif name == "set_blur"
            snapshot!(player)
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            seteffect!(clip, BlurEffect(Float32(args["sigma"])))
            syncsliders!(player, clip)
            refreshedit!(player)
            "blur set"
        elseif name == "set_sharpen"
            snapshot!(player)
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            seteffect!(clip, SharpenEffect(1.0f0, Float32(args["amount"])))
            syncsliders!(player, clip)
            refreshedit!(player)
            "sharpen set"
        elseif name == "analyze_color"
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            runanalysis(() -> analyzecolor!(clip; backend = player.analysisbackend), player)
            "color analysis started — poll get_state"
        elseif name == "analyze_motion"
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            mode = Symbol(get(args, "mode", "similarity"))
            mode in (:similarity, :tripod, :perspective, :smooth) || return "unknown mode $mode"
            oldcrop = clip.crop
            job = () -> try
                analyzemotion!(clip; mode, backend = player.analysisbackend)
                # hide the warp's replicate borders, like the GUI's Stabilize
                W, H = clip.source.width, clip.source.height
                newcrop = cropintersect(oldcrop, bordercrop(clip.motiontrack, W, H))
                put!(player.uiqueue, () -> (clip.crop = newcrop; refreshedit!(player)))
            catch e
                @error "motion analysis failed" exception = (e, catch_backtrace())
            end
            runanalysis(job, player)
            "motion analysis ($mode) started — poll get_state"
        elseif name == "analyze_object"
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            point = (Float64(args["x"]) * clip.source.width,
                     Float64(args["y"]) * clip.source.height)
            job = () -> analyzeobject!(clip, point; backend = player.analysisbackend)
            runanalysis(job, player)
            "object lock analysis started — poll get_state"
        elseif name == "save_project"
            isempty(seq.clips) && return "nothing to save — the timeline is empty"
            saveproject(String(args["path"]), seq)
        elseif name == "add_source"
            clip = addsource!(player, String(args["path"]))  # snapshots itself
            statedict(player)
        elseif startswith(name, "effect_") && haskey(PLUGINBYNAME, Symbol(name[8:end]))
            clip = clipattime(args["time"])
            clip === nothing && return "no clip at that time"
            p = PLUGINBYNAME[Symbol(name[8:end])]
            snapshot!(player)
            kw = (; (pr.name => Float64(get(args, String(pr.name), pr.default)) for pr in p.params)...)
            seteffect!(clip, plugineffect(p.name; kw...))
            syncsliders!(player, clip)
            refreshedit!(player)
            "applied $(p.label)"
        else
            Dict("isError" => true, "message" => "unknown tool $name")
        end
    end
    return textcontent(result)
end

function statedict(player::Player)
    seq = player.sequence
    fps = seq.framerate
    return Dict(
        "duration_seconds" => seqduration(seq),
        "framerate" => fps,
        "playhead_seconds" => player.playhead[] / fps,
        "playing" => player.playing[],
        "clips" => [Dict(
            "index" => i,
            "timeline_start" => clip.start / fps,
            "timeline_end" => clipend(clip) / fps,
            "source" => clip.source.path,
            "source_in" => clip.src_in / clip.source.framerate,
            "crop" => collect(clip.crop),
            "effects" => [effectdict(e) for e in clip.effects],
            "has_colortrack" => clip.colortrack !== nothing,
            "has_motiontrack" => clip.motiontrack !== nothing,
        ) for (i, clip) in enumerate(seq.clips)],
    )
end

"Render a WYSIWYG preview frame (stabilization + effects + crop) at time `t`."
function renderpreview(player::Player, t::Float64, width::Int)
    seq = player.sequence
    isempty(seq.clips) && return zeros(RGB{N0f8}, clamp(width, 64, 1920),
                                       round(Int, 9 / 16 * clamp(width, 64, 1920)))
    n = clamp(round(Int, t * seq.framerate), 0, max(seqlength(seq) - 1, 0))
    loc = locate(seq, n)
    canvas = canvassize(seq)
    pw = clamp(width, 64, 1920)
    ph = max(round(Int, canvas[2] / canvas[1] * pw), 16)
    preview = RGBFrame(undef, pw, ph)
    loc === nothing && return fill!(preview, RGB{N0f8}(0, 0, 0))
    clip, srcframe = loc

    sp = pool(player, clip.source)
    scratch = RGBFrame(undef, sp.source.width, sp.source.height)
    settarget!(sp.worker, srcframe)
    deadline = time() + 3.0
    while !fetchframe!(scratch, sp.ring, srcframe)
        sleep(0.01)
        time() > deadline && error("frame $srcframe not decodable within 3s")
    end
    # the same graph as preview/export; a private engine — this runs on the MCP
    # task, the player's cpuengine pool belongs to the render thread
    ec = effectiveclip(clip, srcframe)
    render(FxEngine(KA.CPU()), scratch, ec, Int(srcframe)) do out
        warp!(preview, out, ec.crop)
        KA.synchronize(KA.get_backend(preview))
    end
    return preview
end
