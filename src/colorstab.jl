"""
    analyzecolor!(clip; cutoff=0.5, backend=KA.CPU(), progress=nothing) -> ColorTrack

Color/exposure stabilization: measure per-channel mean/std for every source
frame of the clip, low-pass the trajectories (`cutoff` Hz — slow intentional
changes survive, flicker doesn't), and store per-frame gain/offset
corrections that pull each frame onto the smoothed trajectory.

The track is keyed by absolute source frame, so it stays valid across
later splits/moves of the clip. Applied automatically (before user
effects) in preview and export. Saved with the project.
"""
function analyzecolor!(clip::Clip; cutoff::Real = 0.5, backend = KA.CPU(), progress = nothing)
    n = srclength(clip)
    n >= 24 || return nothing  # too short to separate flicker from content
    decodable(clip.source) || return nothing   # a rendered clip has no flicker
    means = Matrix{Float32}(undef, n, 3)
    stds = Matrix{Float32}(undef, n, 3)
    dec = graysource(backend, clip.source)   # GPU decode on a GPU backend
    try
        for i in 1:n
            # stats reduce ON the decoder's backend — with a stream nothing but
            # the six floats ever crosses the bus
            μ, σ = channelstats(rgbframe!(dec, clip.src_in + i - 1))
            means[i, :] .= Tuple(μ)
            stds[i, :] .= Tuple(σ)
            progress === nothing || i % 60 == 0 && progress(i, n)
        end
    finally
        close(dec)
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
    # what the fix actually buys, in the same units the eye notices: mean
    # frame-to-frame change of overall brightness, before vs after the correction
    # (the corrected means are the smoothed targets by construction)
    jump(m) = n < 2 ? 0.0 : mean(abs.(diff(vec(sum(m, dims = 2)))))
    was, becomes = jump(means), jump(target_μ)
    reduction = was > 0 ? clamp(1 - becomes / was, 0.0, 1.0) : 0.0
    setcolortrack!(clip, ColorTrack(gains, offsets, clip.src_in, 1.0f0, reduction))
    progress === nothing || progress(n, n)
    return clip.colortrack
end

"""
    colortrackgainoffset(clip, srcframe, strength) -> (gain, offset)

The colour track's per-channel gain and offset at this frame, or `(1, 0)` when
there is nothing to apply.

Its own function for the reason [`motionwarp`](@ref) is: the render path needs
the numbers as parameters without running the kernel, and the pass gates on
whether they are neutral. `strength` scales both toward neutral, so a keyframed
strength of zero IS `(1, 0)` and the gate closes.
"""
function colortrackgainoffset(clip::Clip, srcframe::Integer, strength::Real)
    track = clip.colortrack
    track === nothing && return (Vec3f(1), Vec3f(0))
    i = srcframe - track.src_in + 1
    1 <= i <= length(track.gains) || return (Vec3f(1), Vec3f(0))
    s = clamp(Float32(strength), 0.0f0, 1.0f0)
    s <= 0.0f0 && return (Vec3f(1), Vec3f(0))
    return (Vec3f(1.0f0 .+ s .* (track.gains[i] .- 1.0f0)), Vec3f(s .* track.offsets[i]))
end

"""
Apply the clip's colour stabilization for `srcframe`, if analyzed. `strength`
scales the correction toward identity; it comes from the `FlickerEffect` in the
clip's stack, which is what the user tunes, and falls back to the track's own
value for a track loaded from a project written before the effect existed.
"""
function applycolortrack!(buf::AnyRGBFrame, clip::Clip, srcframe::Integer;
                          strength::Real = clip.colortrack === nothing ? 1.0f0 :
                                           clip.colortrack.strength)
    track = clip.colortrack
    track === nothing && return buf
    i = srcframe - track.src_in + 1
    1 <= i <= length(track.gains) || return buf
    g, o = colortrackgainoffset(clip, srcframe, strength)
    (g, o) === (Vec3f(1), Vec3f(0)) && return buf
    channellinear!(buf, g, o)
    return buf
end
