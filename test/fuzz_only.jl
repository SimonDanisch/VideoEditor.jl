# Drive `fuzz.jl` on its own.
#
# `runtests.jl` includes it at line 996, AFTER `interactions.jl` — which throws on
# its known failures and so aborts the file before the include is reached. That is
# why the fuzzer rotted unnoticed; until the suite's include order is settled, this
# is how it gets run.
using VideoEditor, Test, GLMakie
import VideoEditor as VE
import FFMPEG_jll

testvideo = joinpath(mktempdir(), "test.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 4 -c:v libx264 -g 30 -pix_fmt yuv420p $testvideo`,
             stdout = devnull, stderr = devnull))
testvideo15 = joinpath(mktempdir(), "test15.mp4")
run(pipeline(`$(FFMPEG_jll.ffmpeg()) -y -f lavfi -i testsrc2=size=180x320:rate=15 -t 4 -c:v libx264 -g 15 -pix_fmt yuv420p $testvideo15`,
             stdout = devnull, stderr = devnull))

include("fuzz.jl")
