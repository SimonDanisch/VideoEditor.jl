using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE

bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
# bird = "/home/simon/Downloads/First 8K Video From Space~Orig.mkv"
# No `analysisbackend = LavaBackend()`: Lava's context belongs to the thread that
# touches it FIRST, so building it here makes MAIN the owner and every later
# analysis on the pinned worker asserts "BatchQueue is single-writer".
# `autodetectgpu!` establishes worker ownership in the right order.
player = Player(bird; gpupreview=true);

begin
    using VideoEditor, Lava, RayMakie, GLMakie
    import VideoEditor as VE
    GLMakie.activate!()
    # Loading RayMakie is not enough: a scene names its renderer in text, and a
    # name resolves only once the module is registered.
    VE.usebackend!(RayMakie)
    # No `coverage` method needed: a raytracer knows which rays hit nothing, and
    # RayMakie's frame carries that in its alpha (a transparent scene background
    # makes misses read as uncovered), which is the plane convention already.
    project = "/sim/Programmieren/VideoEdit/media/lego.videoedit"
    player = Player(project)
end
isfile("/sim/Programmieren/VideoEdit/media/lego.videoedit")
