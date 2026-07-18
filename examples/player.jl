using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE

bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
player = Player(bird; analysisbackend=LavaBackend())
