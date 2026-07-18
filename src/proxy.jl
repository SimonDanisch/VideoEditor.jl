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
    for space in ("proxies", "pcm")
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

"""
    generateproxy(source; height=720) -> VideoSource

Encode a lightweight preview proxy of `source` (h264 `veryfast`, dense
keyframes for snappy scrub seeks, no audio) unless a cached one already
exists, and return it as a `VideoSource`. Frame-count exact, so proxy and
original share frame indices — only the resolution differs.
"""
function generateproxy(source::VideoSource; height::Integer = 720)
    path = proxypath(source, height)
    if !isfile(path)
        h = min(2 * (Int(height) ÷ 2), 2 * (source.height ÷ 2))
        tmp = path * ".part.mp4"
        run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -i $(source.path)
                      -vf scale=-2:$h -c:v libx264 -preset veryfast -crf 23
                      -g 30 -pix_fmt yuv420p -frames:v $(source.nframes) -an $tmp`,
                     stdout = devnull, stderr = devnull))
        mv(tmp, path; force = true)
    end
    proxy = VideoSource(path)
    # a truncated proxy would make tail frames undecodable — refuse it
    proxy.nframes >= source.nframes - 2 ||
        error("proxy has $(proxy.nframes) frames, original $(source.nframes)")
    return proxy
end
