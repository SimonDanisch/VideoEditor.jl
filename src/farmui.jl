"""
    registerfarm!(name, label; directory, connect, inputs=[], root=nothing) -> Command

Add a render farm to the editor's command palette and Export panel.
`connect(job)` prepares independent renderer sessions and returns `FarmWorker`s.
The connection owns machine discovery, file synchronization, and GPU selection;
VideoEditor owns the saved job, timeline rendering, resume, progress, and encode.
`inputs` can be a file list or a function of the saved project path; files under
`root` keep their relative paths in the job (see [`renderjob`](@ref)).
"""
function registerfarm!(name::Symbol, label::AbstractString;
                       directory::AbstractString, connect, inputs = String[], root = nothing)
    return registercommand!(Command(Symbol("farm_", name), "Render on $label";
        category = :renderfarm, keywords = ["export", "render", "farm", "gpu"],
        enabled = p -> isempty(p.sequence.clips) ? "the timeline is empty" :
                       isnan(p.jobprogress[]) ? true : "an export is already running",
        run = player -> begin
            widgets = player.fxwidgets
            format = something(widgets[:exportformat].selection[], ".mp4")
            if format == ".gif"
                setstatus!(player, "select MP4, MKV or MOV for a farm export")
                return nothing
            end
            output = abspath(splitext(widgets[:exportpath][])[1] * format)
            options = (codec_name = something(widgets[:exportcodec].selection[], "libx264"),
                       crf = round(Int, widgets[:exportcrf].value[]),
                       preset = something(widgets[:exportpreset].selection[], "medium"),
                       audio = widgets[:exportaudio].checked[])
            root = abspath(directory)
            mkpath(root)
            # Snapshot on the UI thread before the background job starts.
            # Edits made afterwards belong to the next job.
            dir = mktempdir(root; prefix = "render-", cleanup = false)
            project = joinpath(dir, "edit.videoedit")
            saveproject(project, player.sequence)
            extra = inputs isa Function ? inputs(project) : inputs
            job = renderjob(project, dir; inputs = extra, root = something(root, dir))
            player.jobprogress[] = 0.0
            setstatus!(player, "connecting to $label…")
            @async try
                workers = connect(job)
                status = renderfarm!(job, workers; progress = (done, total) -> begin
                    player.jobprogress[] = done / total
                    done % 30 == 0 && setstatus!(player, "rendering on $label: $done / $total")
                end)
                if status["paused"]
                    setstatus!(player, "farm render paused: $(job.directory)")
                else
                    setstatus!(player, "encoding farm render…")
                    encodefarm!(job.directory, output; options...)
                    setstatus!(player, "exported $output")
                end
            catch e
                setstatus!(player, "farm render stopped: $(sprint(showerror, e))")
                @error "farm render stopped; completed frames are retained" job = dir exception = (e, catch_backtrace())
            finally
                player.jobprogress[] = NaN
            end
            return job
        end))
end

function choosefarm!(player::Player)
    farms = filter(c -> c.category === :renderfarm, COMMANDS)
    if isempty(farms)
        setstatus!(player, "no render farm connected")
        return nothing
    end
    # Farms are ordinary commands, so the same action is also available to MCP
    # and Ctrl+P. The export panel has no second dispatch or profile registry.
    length(farms) == 1 && return runcommand!(player, only(farms).name)
    opencommandpalette!(player; query = "render on")
    return nothing
end

"""Where a coordinator's farm config lives: the machines it renders on (see `farmmachines`)."""
farmconfig() = joinpath(homedir(), ".videoeditor", "farm.toml")

"""
    registerlanfarm!(; config = farmconfig(), directory = ~/.videoeditor/farm-jobs, inputs = []) -> Command

"Render on LAN farm": every free GPU slot of the farm daemons `config` lists
(see `farmmachines`), read when a render starts; `inputs` and `root` as for
[`registerfarm!`](@ref). A `Player` registers it by itself when the config
exists and nothing has registered it before.
"""
function registerlanfarm!(; config::AbstractString = farmconfig(),
                          directory::AbstractString = joinpath(homedir(), ".videoeditor", "farm-jobs"),
                          inputs = String[], root = nothing)
    return registerfarm!(:lan, "LAN farm"; directory, inputs, root,
                         connect = job -> farmworkers(job, farmmachines(config)))
end
