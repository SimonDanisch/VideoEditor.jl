# Which code a farm job runs: an environment, written out as a Project.toml that
# pins every package the job needs to exactly what this machine runs. A worker
# instantiates it, so its frames come from the same code by construction, not by
# comparing hashes after the fact.

"""
    farmenvironment(roots; project = Base.active_project(), portable = true) -> Dict

The Project.toml (as a dictionary) of an environment running what the active
one runs for `roots` (package names): every package they need, by `[deps]`
and weak dependencies as the active manifest resolved them, each pinned.

  - A package developed from a git checkout is a `[sources]` entry with its
    remote's `url`, the checked-out commit as `rev`, and its `subdir` in a
    monorepo. The checkout must be committed and the commit pushed: a worker
    can only fetch what a remote has. Every package that is not is listed in
    one error.
  - A package added from a repository is pinned to its commit the same way;
    one that follows a branch is refused (the branch moves).
  - A registered package gets an exact `[compat]` version, and so does Julia.
  - The environment's package preferences come along, as `[preferences]`.

With `portable = false`, developed packages are `path` sources instead: an
environment for farm daemons on this machine only (its second GPU), which can
render what is not committed yet.
"""
function farmenvironment(roots; project::AbstractString = Base.active_project(), portable::Bool = true)
    manifestpath = Base.project_file_manifest_path(project)
    manifestpath === nothing && error("the environment $project has no manifest: instantiate it first")
    entries = Dict(name => only(es) for (name, es) in TOML.parsefile(manifestpath)["deps"])
    for root in roots
        haskey(entries, root) || error("$root is not in the environment $project")
    end
    deps, sources, compat = Dict{String, Any}(), Dict{String, Any}(), Dict{String, Any}()
    problems = String[]
    for name in sort!(collect(dependencyclosure(entries, roots)))
        entry = entries[name]
        isstdlib(entry) && continue
        deps[name] = entry["uuid"]
        if haskey(entry, "path")
            path = abspath(dirname(manifestpath), entry["path"])
            sources[name] = portable ? pinnedsource!(problems, name, path) : Dict{String, Any}("path" => path)
        elseif haskey(entry, "repo-url")
            rev = get(entry, "repo-rev", "")
            occursin(r"^[0-9a-f]{40}$", rev) ||
                push!(problems, "$name follows \"$rev\" of $(entry["repo-url"]): add it at a commit, or develop it")
            sources[name] = Dict{String, Any}("url" => entry["repo-url"], "rev" => rev)
        else
            # a build number (a JLL's `+1`) has no place in compat; the version is exact
            compat[name] = "=" * first(split(entry["version"], '+'))
        end
    end
    isempty(problems) || error("the farm can only render code a worker can fetch:\n  " * join(problems, "\n  "))
    compat["julia"] = "=$(VERSION.major).$(VERSION.minor).$(VERSION.patch)"
    env = Dict{String, Any}("deps" => deps, "sources" => sources, "compat" => compat)
    # preferences change what a package does: they travel with its code
    preferences = environmentpreferences(project)
    filter!(p -> haskey(deps, first(p)), preferences)
    isempty(preferences) || (env["preferences"] = preferences)
    return env
end

"""The package preferences an environment sets: its Project.toml's `[preferences]` and its `LocalPreferences.toml`."""
function environmentpreferences(project::AbstractString)
    preferences = Dict{String, Any}(get(TOML.parsefile(project), "preferences", Dict{String, Any}()))
    for name in ("LocalPreferences.toml", "JuliaLocalPreferences.toml")
        file = joinpath(dirname(project), name)
        isfile(file) && mergewith!(merge, preferences, TOML.parsefile(file))
    end
    return preferences
end

"""The packages `roots` need in a manifest's `entries`: themselves, their dependencies and weak dependencies."""
function dependencyclosure(entries, roots)
    seen = Set{String}()
    todo = collect(String, roots)
    while !isempty(todo)
        name = pop!(todo)
        name in seen && continue
        push!(seen, name)
        entry = entries[name]
        for key in ("deps", "weakdeps")
            listed = get(entry, key, String[])
            for dep in (listed isa AbstractDict ? keys(listed) : listed)
                haskey(entries, dep) && push!(todo, dep)
            end
        end
    end
    return seen
end

"""A standard library: a manifest entry with no tree hash, path or repository of its own."""
isstdlib(entry) = !any(k -> haskey(entry, k), ("git-tree-sha1", "path", "repo-url"))

"""
The `[sources]` entry of the package checked out at `path`: the remote that has
its commit, the commit, and its folder in the repository. What keeps it from
being fetched exactly goes into `problems`.
"""
function pinnedsource!(problems, name, path)
    top = git(path, "rev-parse", "--show-toplevel")
    if top === nothing
        push!(problems, "$name at $path is not a git checkout")
        return Dict{String, Any}()
    end
    changes = git(path, "status", "--porcelain", "--", ".")
    isempty(changes) || push!(problems, "$name has uncommitted changes in $path")
    head = git(path, "rev-parse", "HEAD")
    # the remote branches holding the commit, `origin/main` or `fork/feature`
    holders = [strip(l) for l in split(something(git(path, "branch", "-r", "--contains", "HEAD"), ""), '\n')
               if !isempty(strip(l)) && !occursin("->", l)]
    if isempty(holders)
        push!(problems, "$name: commit $(head[1:min(end, 10)]) in $path is on no remote branch: push it")
        return Dict{String, Any}()
    end
    remote = first(split(first(holders), '/'))
    source = Dict{String, Any}("url" => git(path, "remote", "get-url", remote), "rev" => head)
    subdir = relpath(path, top)
    subdir == "." || (source["subdir"] = replace(subdir, '\\' => '/'))
    return source
end

"""Git's answer in `dir`, or `nothing` where git fails (not a checkout)."""
function git(dir, args...)
    out = IOBuffer()
    ok = success(pipeline(Cmd(`git -C $dir $args`); stdout = out, stderr = devnull))
    return ok ? strip(String(take!(out))) : nothing
end

"""
    farmpackages(seq) -> Vector{String}

The packages a farm worker loads to render `seq`: VideoEditor, the backends its
scenes are finished with, and the packages their recipes come from.
"""
farmpackages(seq::Sequence) = unique(["VideoEditor"; farmbackends(seq); recipepackages(seq)])

"""The packages `seq`'s scene recipes come from (see `packagescene`)."""
recipepackages(seq::Sequence) =
    unique([String(c.source.build["package"]) for c in seq.clips
            if c.source isa SceneSource && c.source.build isa AbstractDict && haskey(c.source.build, "package")])

"""The backends `seq`'s scenes are finished with: what a worker loads and registers."""
farmbackends(seq::Sequence) =
    unique([String(s.bakewith === :auto ? s.backend : s.bakewith)
            for s in unique(c.source for c in seq.clips if c.source isa SceneSource)])
