using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE

bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
# bird = "/home/simon/Downloads/First 8K Video From Space~Orig.mkv"
player = Player(bird; analysisbackend=LavaBackend(), gpupreview=true);
