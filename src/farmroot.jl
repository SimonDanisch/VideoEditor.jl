"Validate a job against the current application root without activating or installing another environment."
function checkfarmroot(project::AbstractString)
    wanted=TOML.parsefile(project)
    names=String.(collect(keys(wanted["deps"])))
    # A job of path sources (`portable = false`) pins developed packages by their code.
    portable=!any(s -> haskey(s,"path"), values(get(wanted,"sources",Dict())))
    actual=farmenvironment(names; portable)
    problems=String[]
    for name in names
        get(actual["deps"],name,nothing)==wanted["deps"][name] || push!(problems,"$name: UUID differs")
        for section in ("sources","compat")
            expected=get(get(wanted,section,Dict()),name,nothing)
            found=get(get(actual,section,Dict()),name,nothing)
            samesource(expected,found) || push!(problems,"$name: $section differs (synchronize the root project)")
        end
        isequal(get(get(actual,"preferences",Dict()),name,nothing),
                get(get(wanted,"preferences",Dict()),name,nothing)) || push!(problems,"$name: preferences differ")
    end
    get(actual["compat"],"julia",nothing)==get(wanted["compat"],"julia",nothing) || push!(problems,"Julia version differs")
    isempty(problems) || error("Farm job does not match the application root:\n"*join(problems,"\n"))
    dirname(Base.active_project())
end

"The same code: a developed package by its content (its path is each machine's own), anything else exactly."
samesource(expected::AbstractDict, found::AbstractDict) =
    haskey(expected,"content") ? get(found,"content",nothing)==expected["content"] : isequal(expected,found)
samesource(expected, found) = isequal(expected, found)
