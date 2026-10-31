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
import Mmap
import DepthAnythingRunner
import NeuralLUTRunner
import WhisperRunner
import RIFERunner
import KokoroRunner
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
import MsgPack
import Scratch
import Statistics: median, mean

const RGBFrame = Matrix{RGB{N0f8}}

"""
What a plane in the render graph is made of: RGBA, PREMULTIPLIED.

Alpha is COVERAGE — how much of this pixel is there — and it is what a matte
writes when it removes background, what a crop leaves behind, and what a rendered
3D scene has outside its subject. Before it, "removed" was painted black, which is
right over nothing and wrong over a track: the black is opaque, so a matte over a
background plate showed a silhouette, and the compositor had to be handed a second
image saying which black was which.

Premultiplied because every op that MIXES pixels — a blur, the placement warp, a
cross-fade — is a weighted sum, and a weighted sum of premultiplied colours is
correct while one of straight colours drags the colour of absent pixels into
present ones. That is the halo along a matte edge, and it is a property of the
FORMAT, so no op has to know about it (see `GPUFiltering.AnyRGB`).

The CANVAS is not this. It is `RGB{N0f8}`: nothing shows through the finished
picture, and it is what goes to the screen and to the encoder.
"""
const PlanePixel = RGBA{N0f8}

# Track/effect appliers accept this so the same code runs on CPU frames and
# GPU-resident LavaArrays (GPUFiltering kernels are backend-generic), and on a
# host RGB frame as well as a graph plane.
const AnyRGBFrame = AbstractMatrix{<:Union{RGB{N0f8}, RGBA{N0f8}}}

include("source.jl")
include("proxy.jl")
include("buffer.jl")
include("decoder.jl")
include("keyframes.jl")
include("clips.jl")      # Effect, Clip, Sequence — in that order, because each
                         # of them is a field of the next
include("pack.jl")   # how our own types go into a project file
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
include("look.jl")      # …and LookNode likewise
include("captions.jl")  # needs overlays.jl AND audiopreview.jl, both above
include("flow.jl")      # SmoothSourceNode is in gpugraph.jl; the model is here
include("narration.jl") # after audio.jl/audiopreview.jl: mixes into both paths
include("colorstab.jl")
include("motionstab.jl")
include("bundlestab.jl")
include("export.jl")   # sourceinto! dispatches on GpuVideoStream — needs the type
include("campath.jl")  # GrayReader wraps export's SequentialReader; grayinto! needs GpuVideoStream
include("tools.jl")    # GUI tools: timeline-overlay hints + premade operations
include("scenespec.jl") # a Makie scene AS DATA: plots, backend, theme, paths
include("scenerender.jl") # …and rendering it, with the backend as a parameter
include("scenesource.jl") # a scene AS A CLIP: the source pass, and making one
include("overlays.jl")   # the stock non-footage clips, each as a SceneSpec preset
include("bake.jl")       # pre-rendering a clip's chain to disk
include("commands.jl") # everything the editor can do, as data
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
export LookEffect, registerlook!, installlook!, analyzelook!, lookdim
export Caption, transcribe!, registertranscribe!, installtranscribe!, captionat
export captionindexat, editcaption!
export registerinterpolate!, installinterpolate!, settimeinterp!, smoothslowmo!
export addnarration!, dropnarration!, rendernarration!, pickfocus!, setinterp!
export Narration, narrate!, registerspeak!, installspeak!
export registerplugin!, registereffect!, EffectKind, FxParam, Pointwise, Stencil
export SceneSource, SceneSpec, sceneclip, textscene, barscene, timecodescene, curvescene
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
export ParamInput, ParamRef, FileRef, ClipRef, bindinput!, unbindinput!, bindinputs!

end
