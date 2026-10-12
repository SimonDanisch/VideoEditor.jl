"Upgrade a frame's primary-ray guides without resampling its HDR radiance."
function refreshguides!(clip::Clip, sf::Integer)
    original = cachedsourceframe(clip, sf)
    original === nothing && error("no HDR source master at frame $sf")
    src = clip.source
    src isa SceneSource || error("guide refresh requires a scene source")
    mode = src.mode;src.mode = :bake
    try
        name = renderwith(src);backend = getbackend(name)
        isdefined(backend, :refreshguidebuffers!) || error("backend cannot refresh guides")
        fresh = onthread(renderthread(backend)) do
            lock(SCENELOCK) do
                opts = renderopts(src);opts[:rasterize] = false;opts[:denoise] = false
                src.live = livescene!(src.live, src.root, (src.width, src.height), name, backend; opts)
                isdefined(@__MODULE__, :updatesceneprogram!) &&
                    updatesceneprogram!(src.live.target, src, clip, sf, src.framerate; reset = true)
                applysceneparams!(src, clip, sf)
                Base.invokelatest(backend.refreshguidebuffers!, src.live.screen)
            end
        end
        invalidateclip!(clip; frames = sf:sf, source = false)
        cache = sourcecache(clip)
        dirtyframes!(cache, clip.id, sf:sf)
        upgraded = merge(
            original, (
                albedo = fresh.albedo, normal = fresh.normal, depth = fresh.depth,
                metadata = merge(original.metadata, fresh.metadata),
            )
        )
        cachewrite!(cache, clip.id, sf, upgraded)
        return upgraded
    finally
        src.mode = mode
    end
end
