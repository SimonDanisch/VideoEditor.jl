using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE

bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
# bird = "/home/simon/Downloads/First 8K Video From Space~Orig.mkv"
# No `analysisbackend = LavaBackend()`: Lava's context belongs to the thread that
# touches it FIRST, so building it here makes MAIN the owner and every later
# analysis on the pinned worker asserts "BatchQueue is single-writer".
# `autodetectgpu!` establishes worker ownership in the right order.
player = Player(bird; gpupreview=true);
