"Validate a job against the current application root without activating or installing another environment."
function checkfarmroot(project::AbstractString)
    wanted=TOML.parsefile(project)
    names=String.(collect(keys(wanted["deps"])))
    # A job of path sources (`portable = false`) pins developed packages by their code.
    pinned=get(wanted,"content",Dict{String,Any}())
    actual=farmenvironment(names; portable=isempty(pinned))
    problems=String[]
    for name in names
        get(actual["deps"],name,nothing)==wanted["deps"][name] || push!(problems,"$name: UUID differs")
        if haskey(pinned,name)
            # a path is each machine's own; the code is what has to agree
            get(get(actual,"content",Dict()),name,nothing)==pinned[name] ||
                push!(problems,"$name: code differs (synchronize the root project)")
        else
            isequal(get(get(wanted,"sources",Dict()),name,nothing),get(get(actual,"sources",Dict()),name,nothing)) ||
                push!(problems,"$name: sources differs (synchronize the root project)")
        end
        isequal(get(get(wanted,"compat",Dict()),name,nothing),get(get(actual,"compat",Dict()),name,nothing)) ||
            push!(problems,"$name: compat differs (synchronize the root project)")
        isequal(get(get(actual,"preferences",Dict()),name,nothing),
                get(get(wanted,"preferences",Dict()),name,nothing)) || push!(problems,"$name: preferences differ")
    end
    get(actual["compat"],"julia",nothing)==get(wanted["compat"],"julia",nothing) || push!(problems,"Julia version differs")
    isempty(problems) || error("Farm job does not match the application root:\n"*join(problems,"\n"))
    dirname(Base.active_project())
end
