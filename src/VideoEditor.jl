module VideoEditor

using ColorTypes
using FixedPointNumbers
using GeometryBasics
using GLMakie
using GPUFiltering
using Lava
using Makie
using Observables
using SAM2Runner
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
include("overlays.jl")   # Overlay: the Sequence carries a list of them
include("clips.jl")
include("effects.jl")
include("project.jl")
include("thumbnails.jl")
include("clipview.jl")
include("timeline.jl")
include("gpudecode.jl")
include("gpustream.jl")
include("gpugraph.jl")   # FxEngine: the Player struct carries the CPU-tier engine
include("player.jl")
include("glbridge.jl")
include("audiopreview.jl")
include("audio.jl")
include("matte.jl")
include("restore.jl")
include("colorstab.jl")
include("motionstab.jl")
include("bundlestab.jl")
include("export.jl")   # sourceinto! dispatches on GpuVideoStream — needs the type
include("campath.jl")  # GrayReader wraps export's SequentialReader; grayinto! needs GpuVideoStream
include("plugins.jl")
include("tools.jl")    # GUI tools: timeline-overlay hints + premade operations
include("agentview.jl")  # what an AGENT sees: contact sheets, zoom, change search
include("mcp.jl")
include("precompile.jl")  # LAST: the workload runs the render + matte paths, so
                          # everything it calls has to be defined already

export VideoSource, Player, Clip, Sequence
export play!, pause!, step!
export split!, deleteclip!, moveclip!, addsource!, saveproject, loadproject
export ColorEffect, BlurEffect, SharpenEffect, MatteEffect, seteffect!
export MatteTrack, analyzematte!, applymatte!, registermatte!, seedmask
export sam2seed, sam2ready, defaultsegmenter
export RestoreEffect, registerrestore!, restorewindow!, applyrestore!
export registerplugin!, FxParam, FxPlugin, Pointwise, Stencil
export Overlay, registeroverlay!, addoverlay!, removeoverlay!, setoverlaykey!
export registertool!, EditorTool, ToolContext, toolplot!, ontool!, tooltime, toolband
export activatetool!, deactivatetool!
export exportvideo, exportgif, analyzecolor!, analyzemotion!, analyzeobject!, findloop
export loopsignatures, similarframes
export contactsheet, filmstrip, framegrab, regiongrab, findchange, viewsummary
export generateproxy, startproxy!
export mcpserve!

end
