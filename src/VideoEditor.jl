module VideoEditor

using ColorTypes
using FixedPointNumbers
using GeometryBasics
using GLMakie
using GPUFiltering
using Lava
using Makie
using Observables
using Printf
using TOML
using VideoIO
import Base64
import DSP
import FFMPEG_jll
import HTTP
import JSON
import KernelAbstractions as KA
import Mmap
import PNGFiles
import Scratch
import Statistics: median, mean

const RGBFrame = Matrix{RGB{N0f8}}
# Track/effect appliers accept this so the same code runs on CPU frames and
# GPU-resident LavaArrays (GPUFiltering kernels are backend-generic).
const AnyRGBFrame = AbstractMatrix{RGB{N0f8}}

include("source.jl")
include("proxy.jl")
include("buffer.jl")
include("decoder.jl")
include("keyframes.jl")
include("clips.jl")
include("effects.jl")
include("project.jl")
include("thumbnails.jl")
include("clipview.jl")
include("timeline.jl")
include("player.jl")
include("glbridge.jl")
include("audiopreview.jl")
include("audio.jl")
include("colorstab.jl")
include("motionstab.jl")
include("bundlestab.jl")
include("gpudecode.jl")
include("gpustream.jl")
include("gpugraph.jl")
include("export.jl")   # decodeinto! dispatches on GpuVideoStream — needs the type
include("campath.jl")  # GrayReader wraps export's SequentialReader; grayinto! needs GpuVideoStream
include("plugins.jl")
include("tools.jl")    # GUI tools: timeline-overlay hints + premade operations
include("mcp.jl")

export VideoSource, Player, Clip, Sequence
export play!, pause!, step!
export split!, deleteclip!, moveclip!, addsource!, saveproject, loadproject
export ColorEffect, BlurEffect, SharpenEffect, seteffect!
export registerplugin!, FxParam, FxPlugin, Pointwise, Stencil
export registertool!, EditorTool, ToolContext, toolplot!, ontool!, tooltime, toolband
export activatetool!, deactivatetool!
export exportvideo, exportgif, analyzecolor!, analyzemotion!, analyzeobject!, findloop
export loopsignatures, similarframes
export generateproxy, startproxy!
export mcpserve!

end
