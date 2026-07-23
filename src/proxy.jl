"Scratch cache directory for `space` (\"proxies\" or \"pcm\") — always keyed
by VideoEditor's UUID, wherever the caller lives (Scratch's macro keys by
the calling module otherwise)."
cachedir(space::String) = Scratch.@get_scratch!(space)

"""
    proxypath(source, height) -> String

Location of `source`'s preview proxy in the package scratch space, keyed by
the original's identity (absolute path + size + mtime) and the proxy
height — editing or replacing the original invalidates its proxy.
"""
function proxypath(source::VideoSource, height::Integer)
    st = stat(source.path)
    key = string(hash((abspath(source.path), st.size, st.mtime, Int(height))), base = 16)
    return joinpath(cachedir("proxies"), key * ".mp4")
end

"""
    prunecache!(; maxbytes = 20 * 2^30) -> bytes_removed

Keep the regenerable scratch caches (preview proxies + audio PCM) under
`maxbytes` by deleting least-recently-modified files first. Runs once per
`Player` construction; entries for edited or moved sources go stale (the
cache key includes mtime) and would otherwise accumulate forever.
"""
function prunecache!(; maxbytes::Integer = 20 * 2^30)
    files = Tuple{Float64, Int, String}[]  # (mtime, size, path)
    for space in ("proxies", "pcm", "mezzanine")
        dir = cachedir(space)
        for name in readdir(dir; join = true)
            endswith(name, ".part") && (rm(name; force = true); continue)  # aborted job
            push!(files, (mtime(name), filesize(name), name))
        end
    end
    total = sum(f -> f[2], files; init = 0)
    removed = 0
    for (_, size, path) in sort(files)          # oldest first
        total - removed <= maxbytes && break
        rm(path; force = true)
        removed += size
    end
    return removed
end

"""
Whether `source` deserves a preview proxy: more pixels than `maxpixels`
(above full HD by default) OR a heavyweight codec — intra-frame formats
like DNxHR/ProRes carry hundreds of KB per frame, which strains decode
bandwidth and blows up the preview ring's RAM even at modest resolutions.
"""
needsproxy(source::VideoSource; maxpixels::Integer = 2_100_000,
           maxbytesperframe::Integer = 250_000) =
    source.width * source.height > maxpixels ||
    filesize(source.path) ÷ max(source.nframes, 1) > maxbytesperframe

"Location of `source`'s editing mezzanine (same identity key as proxies)."
function mezzaninepath(source::VideoSource)
    st = stat(source.path)
    key = string(hash((abspath(source.path), st.size, st.mtime, :mezzanine)), base = 16)
    return joinpath(cachedir("mezzanine"), key * ".mp4")
end

"Run an ffmpeg encode with `-progress pipe:1`, reporting frames to `progress`."
function runencode(cmd::Cmd, nframes::Integer, progress)
    proc = open(pipeline(ignorestatus(cmd); stderr = devnull))
    for line in eachline(proc)
        progress === nothing && continue
        startswith(line, "frame=") || continue
        f = tryparse(Int, strip(line[7:end]))
        f === nothing || progress(f, nframes)
    end
    wait(proc)
    return success(proc)
end

"""
    generatemezzanine(source; progress=nothing) -> path

Transcode `source` ONCE into the editing mezzanine — the codec/GOP structure
the Vulkan pipeline streams best: ≤4K → H.264 High (refs 2, no B-frames),
larger → HEVC Main (NVDEC's H.264 engine tops out at 4096×4096); both with
~1 s closed GOPs for snappy GOP seeks and a small VRAM ring. Decodes AND
encodes in hardware when the system ffmpeg has CUDA/NVENC, falling back to
the bundled software encoders. Frame-count validated; cached like proxies.
Only the DECODE side uses it — export still reads the original.
"""
function generatemezzanine(source::VideoSource; progress = nothing)
    path = mezzaninepath(source)
    if !isfile(path)
        g = max(round(Int, source.framerate), 1)
        hevc = source.width > 4096 || source.height > 4096
        tmp = path * ".part.mp4"
        tail = `-g $g -pix_fmt yuv420p -frames:v $(source.nframes) -an
                -progress pipe:1 -nostats $tmp`
        sys = cudaffmpeg()
        enc = hevc ? `-c:v hevc_nvenc -preset p4 -cq 23 -weighted_pred 0` :
                     `-c:v h264_nvenc -preset p4 -cq 21 -refs 2 -weighted_pred 0`
        done = sys !== nothing &&
               runencode(`$sys -y -v error -hwaccel cuda -i $(source.path) -map 0:v:0
                          $enc -bf 0 -forced-idr 1 $tail`, source.nframes, progress)
        if !done
            enc = hevc ? `-c:v libx265 -preset fast -crf 22 -x265-params
                          keyint=$g:min-keyint=$g:bframes=0:ref=2:no-open-gop=1:weightp=0` :
                         `-c:v libx264 -preset veryfast -crf 20 -refs 2 -bf 0`
            runencode(`$(FFMPEG_jll.ffmpeg()) -y -v error -i $(source.path) -map 0:v:0
                       $enc $tail`, source.nframes, progress) ||
                error("mezzanine transcode failed for $(source.path)")
        end
        mv(tmp, path; force = true)
    end
    mz = VideoSource(path)
    mz.nframes >= source.nframes - 2 ||   # a short mezzanine breaks tail frames
        error("mezzanine has $(mz.nframes) frames, original $(source.nframes)")
    return path
end

"""
System ffmpeg with CUDA hardware decode, or `nothing`. Software HEVC decode of
an 8K source runs ~10 fps — NVDEC does it at hardware speed, turning proxy
generation from many minutes into roughly realtime.
"""
function cudaffmpeg()
    sys = Sys.which("ffmpeg")
    sys === nothing && return nothing
    try
        occursin("cuda", read(`$sys -hide_banner -hwaccels`, String)) ? sys : nothing
    catch
        nothing
    end
end

"""
    generateproxy(source; height=720) -> VideoSource

Encode a lightweight preview proxy of `source` (h264 `veryfast`, dense
keyframes for snappy scrub seeks, no audio) unless a cached one already
exists, and return it as a `VideoSource`. Frame-count exact, so proxy and
original share frame indices — only the resolution differs. Decodes through
the system ffmpeg's CUDA path when available ([`cudaffmpeg`](@ref)).
"""
function generateproxy(source::VideoSource; height::Integer = 720)
    path = proxypath(source, height)
    if !isfile(path)
        h = min(2 * (Int(height) ÷ 2), 2 * (source.height ÷ 2))
        tmp = path * ".part.mp4"
        args = `-y -i $(source.path)
                -vf scale=-2:$h -c:v libx264 -preset veryfast -crf 23
                -g 30 -pix_fmt yuv420p -frames:v $(source.nframes) -an $tmp`
        sys = cudaffmpeg()   # decode on the GPU when the system ffmpeg can
        encoded = sys !== nothing &&
                  success(pipeline(`$sys -hwaccel cuda $args`, stdout = devnull, stderr = devnull))
        encoded ||
            run(pipeline(`$(FFMPEG_jll.ffmpeg()) $args`, stdout = devnull, stderr = devnull))
        mv(tmp, path; force = true)
    end
    proxy = VideoSource(path)
    # a truncated proxy would make tail frames undecodable — refuse it
    proxy.nframes >= source.nframes - 2 ||
        error("proxy has $(proxy.nframes) frames, original $(source.nframes)")
    return proxy
end
