"""
    analyzecolor!(clip; cutoff=0.5, progress=nothing) -> ColorTrack

Color/exposure stabilization: measure per-channel mean/std for every source
frame of the clip, low-pass the trajectories (`cutoff` Hz — slow intentional
changes survive, flicker doesn't), and store per-frame gain/offset
corrections that pull each frame onto the smoothed trajectory.

The track is keyed by absolute source frame, so it stays valid across
later splits/moves of the clip. Applied automatically (before user
effects) in preview and export. Saved with the project.
"""
function analyzecolor!(clip::Clip; cutoff::Real = 0.5, progress = nothing)
    n = cliplength(clip)
    n >= 24 || return nothing  # too short to separate flicker from content
    means = Matrix{Float32}(undef, n, 3)
    stds = Matrix{Float32}(undef, n, 3)
    sr = SequentialReader(clip.source)
    frame = RGBFrame(undef, clip.source.width, clip.source.height)
    try
        for i in 1:n
            readframe!(frame, sr, clip.src_in + i - 1)
            μ, σ = channelstats(frame)
            means[i, :] .= Tuple(μ)
            stds[i, :] .= Tuple(σ)
            progress === nothing || i % 60 == 0 && progress(i, n)
        end
    finally
        close(sr)
    end

    fps = clip.source.framerate
    normalized = clamp(2 * cutoff / fps, 1.0e-3, 0.95)  # cutoff relative to Nyquist
    lowpass = DSP.digitalfilter(DSP.Lowpass(normalized), DSP.Butterworth(2))
    target_μ = similar(means)
    target_σ = similar(stds)
    for ch in 1:3
        target_μ[:, ch] = DSP.filtfilt(lowpass, Float64.(means[:, ch]))
        target_σ[:, ch] = DSP.filtfilt(lowpass, Float64.(stds[:, ch]))
    end

    gains = Vector{Vec3f}(undef, n)
    offsets = Vector{Vec3f}(undef, n)
    for i in 1:n
        g = Vec3f(ntuple(ch -> clamp(target_σ[i, ch] / max(stds[i, ch], 1.0f-4), 0.5f0, 2.0f0), 3))
        o = Vec3f(ntuple(ch -> target_μ[i, ch] - means[i, ch] * g[ch], 3))
        gains[i] = g
        offsets[i] = o
    end
    clip.colortrack = ColorTrack(gains, offsets, clip.src_in)
    progress === nothing || progress(n, n)
    return clip.colortrack
end

"Apply the clip's color stabilization for `srcframe`, if analyzed."
function applycolortrack!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer)
    track = clip.colortrack
    track === nothing && return buf
    i = srcframe - track.src_in + 1
    1 <= i <= length(track.gains) || return buf
    channellinear!(buf, track.gains[i], track.offsets[i])
    KA.synchronize(KA.get_backend(buf))
    return buf
end
