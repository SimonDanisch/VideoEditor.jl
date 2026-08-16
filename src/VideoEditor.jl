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
import DepthAnythingRunner
using Printf
using TOML
using JSON
using Dates
using VideoIO
import Artifacts
import Base64
import DSP
import FFMPEG_jll
import HTTP
import JSON
import KernelAbstractions as KA
import Mantle
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
include("registry.jl")  # ONE registry: built-ins, plugins and tools are all EffectKinds
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
include("depth.jl")     # DepthBlurOp lives in gpugraph.jl; its hooks are here
include("colorstab.jl")
include("motionstab.jl")
include("bundlestab.jl")
include("export.jl")   # sourceinto! dispatches on GpuVideoStream — needs the type
include("campath.jl")  # GrayReader wraps export's SequentialReader; grayinto! needs GpuVideoStream
include("tools.jl")    # GUI tools: timeline-overlay hints + premade operations
include("commands.jl") # everything the editor can do, as data
include("links.jl")    # effects that reference other effects (a blend is two halves)
include("fxpanel.jl")  # THE effects panel: one card list for the selected clip
include("palette.jl")  # Ctrl+P over the commands
include("transform.jl")  # the preview gizmo for TransformEffect
include("agentview.jl")  # what an AGENT sees: contact sheets, zoom, change search
include("mcp.jl")
include("precompile.jl")  # LAST: the workload runs the render + matte paths, so
                          # everything it calls has to be defined already

export VideoSource, Player, Clip, Sequence
export play!, pause!, step!
export split!, deleteclip!, moveclip!, copyclip, copyclips!, pasteclips!,
       addsource!, saveproject, loadproject
export ColorEffect, BlurEffect, SharpenEffect, MatteEffect, seteffect!
export MatteTrack, analyzematte!, applymatte!, registermatte!, seedmask,
       repairframe!, repairmatteat!, matterepairs
export sam2seed, sam2ready, defaultsegmenter
export RestoreEffect, registerrestore!, restorewindow!, applyrestore!
export DepthBlurEffect, DepthTrack, registerdepth!, installdepth!, analyzedepth!, depthframe
export registerplugin!, registereffect!, EffectKind, FxParam, Pointwise, Stencil
export Overlay, registeroverlay!, addoverlay!, removeoverlay!, setoverlaykey!
export registertool!, ToolContext, toolplot!, ontool!, tooltime, toolband
export activatetool!, deactivatetool!
export repairmattecollect!, brushmatte!, matteframe
export beginmattebrush!, mattebrushto!, endmattebrush!
export exportvideo, exportgif, analyzecolor!, analyzemotion!, analyzeobject!, findloop
export loopsignatures, similarframes
export contactsheet, filmstrip, framegrab, regiongrab, findchange, viewsummary
export generateproxy, startproxy!
export mcpserve!
export Command, registercommand!, runcommand!, commands
export FxLink, linkeffects!, resolvelink, prunelinks!

end
