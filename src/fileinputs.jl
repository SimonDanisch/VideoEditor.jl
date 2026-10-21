"""
    FileSeriesRef(paths, times; package="", loader=:load, options=Dict())

An immutable file-backed time sequence. Its `:sample` input takes this reference
and a numeric time parameter. Only the two bracketing files are loaded. `package`
and `loader` optionally name an installed package's public file loader; empty
`package` uses FileIO. The project saves addresses/options, never volume arrays.
"""
struct FileSeriesRef <: InputRef
    paths::Vector{String}
    times::Vector{Float64}
    package::String
    loader::Symbol
    options::Dict{String,Any}
    function FileSeriesRef(paths,times;package="",loader=:load,options=Dict())
        length(paths)==length(times)>=2 || throw(ArgumentError("file series needs at least two matching paths/times"))
        ts=Float64.(times)
        all(isfinite,ts) && all(>(0),diff(ts)) || throw(ArgumentError("file series times must increase strictly"))
        new(abspath.(String.(paths)),ts,String(package),Symbol(loader),Dict{String,Any}(options))
    end
end

"A lazy blend descriptor. Consumers interpolate on device, not into a host-sized copy."
struct BlendedInput{A,B}
    a::A
    b::B
    weight::Float32
end

mutable struct FileSeries
    ref::FileSeriesRef
    loaded::Dict{Int,Tuple{Tuple{Float64,Int},Any}}
    loader::Any
end
FileSeries(ref::FileSeriesRef)=FileSeries(ref,Dict{Int,Tuple{Tuple{Float64,Int},Any}}(),nothing)

function sampleinput(series::FileSeries,time::Real)
    isfinite(time) || throw(ArgumentError("non-finite file-series time"))
    ref=series.ref
    i=clamp(searchsortedlast(ref.times,time),1,length(ref.times)-1)
    weight=Float32(clamp((time-ref.times[i])/(ref.times[i+1]-ref.times[i]),0,1))
    for k in collect(keys(series.loaded))
        k in (i,i+1) || delete!(series.loaded,k)
    end
    if series.loader===nothing
        mod=isempty(ref.package) ? Makie.FileIO : Base.require(Main,Symbol(ref.package))
        series.loader=getproperty(mod,ref.loader)
    end
    function getframe(k)
        path=ref.paths[k]
        isfile(path) || error("Missing file-series input: $path")
        stamp=(mtime(path),filesize(path))
        cached=get(series.loaded,k,nothing)
        if cached===nothing || cached[1]!=stamp
            value=Base.invokelatest(series.loader,path; (Symbol(k)=>v for (k,v) in ref.options)...)
            series.loaded[k]=(stamp,value)
        end
        series.loaded[k][2]
    end
    BlendedInput(getframe(i),getframe(i+1),weight)
end

inputarity(::Val{:sample})=2
inputvalue(::Val{:sample},values)=sampleinput(values[1],values[2])

"Explicit file dependencies, shared by cache freshness and farm packaging."
inputfiles(::InputRef)=String[]
inputfiles(ref::FileRef)=[ref.path]
inputfiles(ref::RecordedRef)=[ref.path]
inputfiles(ref::FileSeriesRef)=ref.paths
inputpackages(::InputRef)=String[]
inputpackages(ref::FileSeriesRef)=isempty(ref.package) ? String[] : [ref.package]
