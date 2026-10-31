"""
    bundlefarm(job, directory; root)

Copy the job and its declared animation inputs for another machine. `root` is
the animation project's root; inputs keep relative paths so `include` and asset
loading work unchanged. Files outside that root get individual mappings.

Julia packages and compiled caches stay in the worker's ordinary Pkg environment.
Prepare that runtime once, sync this bundle, then call `openfarmbundle(directory)`.
Existing bundles are immutable; use a new directory for a new edit.
"""
function bundlefarm(job::RenderJob, directory::AbstractString; root::AbstractString)
    directory = abspath(directory)
    ispath(directory) && error("bundle already exists: $directory")
    root = abspath(root)
    # Build beside the destination, publishing only a complete bundle.
    mkpath(dirname(directory))
    mktempdir(dirname(directory)) do temp
        relocations = Dict(root => "data")
        for (file, checksum) in job.inputs
            filehash(file) == checksum || error("input changed before bundling: $file")
            mappings = Dict(source => joinpath(temp, relative) for (source, relative) in relocations)
            target = farmpath(file, mappings)
            if target == file
                relative = joinpath("inputs", checksum, basename(file))
                relocations[file] = relative
                target = joinpath(temp, relative)
            end
            mkpath(dirname(target))
            cp(file, target; force = true)
        end
        destjob = joinpath(temp, "job")
        mkpath(destjob)
        for file in ("job.json", "project.videoedit", "project.videoedit.mattes", "project.videoedit.bakes")
            source = joinpath(job.directory, file)
            ispath(source) && cp(source, joinpath(destjob, file))
        end
        farmwritejson(joinpath(temp, "bundle.json"), Dict("relocations" => relocations,
                      "identity" => job.identity))
        mv(temp, directory)
    end
    return (directory = directory, job_directory = joinpath(directory, "job"))
end

"Open a synced job bundle, verifying declared inputs and rebasing their paths."
function openfarmbundle(directory::AbstractString; kw...)
    d = JSON.parsefile(joinpath(directory, "bundle.json"))
    mappings = Dict(source => joinpath(directory, relative) for (source, relative) in d["relocations"])
    job = RenderJob(joinpath(directory, "job"))
    job.identity == d["identity"] || error("bundle has a different render job")
    return farmrenderer(job; pathmap = mappings, kw...)
end
