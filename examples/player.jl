using VideoEditor, GLMakie, Makie, Lava
import VideoEditor as VE

bird = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
# bird = "/home/simon/Downloads/First 8K Video From Space~Orig.mkv"
player = Player(bird; analysisbackend=LavaBackend(), gpupreview=true);


```

julia> Error in callback:
Failed to resolve fast_string_boundingboxes:
[ComputeEdge] fast_string_boundingboxes = (::MapFunctionWrapper(#register_fast_string_boundingboxes!##0))((text_blocks, raw_glyph_boundingboxes, marker_offset, text_rotation, linesegments, linewidths, lineindices, ), changed, cached)
  @ unknown method location
[ComputeEdge] glyphcollections, glyphindices, font_per_char, glyph_origins, glyph_extents, text_blocks, text_color, text_rotation, text_scales, text_strokewidth, text_strokecolor, linesegments, linewidths, linecolors, lineindices = #compute_glyph_collections!##0((input_text, fontsize, selected_font, align, rotation, justification, lineheight, word_wrap_width, offset, fonts, computed_color, strokecolor, strokewidth, ), changed, cached)
  @ /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:349
  with edge inputs:
    input_text = ["Stabilization"]
    fontsize = 14.0f0
    selected_font = FTFont (family = TeX Gyre Heros Makie, style = Regular)
    align = (:left, :center)
    rotation = 1.0 + 0.0im + 0.0jm + 0.0km
    justification = Makie.Automatic()
    lineheight = 1.0
    word_wrap_width = -1
    offset = Float32[0.0, 0.0, 0.0]
    fonts = Attributes()
    computed_color = ColorTypes.RGBA{Float32}[]
    strokecolor = RGBA{Float32}(0.0, 0.0, 0.0, 0.0)
    strokewidth = 0.0f0
Triggered by update of:
  alpha, strokecolor, position, colormap, colorscale, text, fonts, highclip, word_wrap_width, justification, font, align, nan_color, strokewidth, rotation, arg1, lineheight, colorrange, offset, color, fontsize or lowclip
Due to ERROR: BoundsError: attempt to access 0-element Vector{ColorTypes.RGBA{Float32}} at index [0]
Stacktrace:
  [1] throw_boundserror(A::Vector{ColorTypes.RGBA{Float32}}, I::Tuple{Int64})
    @ Base ./essentials.jl:15
  [2] getindex
    @ ./essentials.jl:919 [inlined]
  [3] per_glyph_block(data::Vector{ColorTypes.RGBA{Float32}}, block_idx::Int64, N_blocks::Int64, block::UnitRange{Int64})
    @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:192
  [4] convert_text_string!(outputs::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}}, input_text::String, i::Int64, N::Int64, fontsize::Float32, font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32)
    @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:215
  [5] (::Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1")(::@NamedTuple{input_text::Vector{String}, fontsize::Float32, selected_font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, computed_color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32}, changed::@NamedTuple{input_text::Bool, fontsize::Bool, selected_font::Bool, align::Bool, rotation::Bool, justification::Bool, lineheight::Bool, word_wrap_width::Bool, offset::Bool, fonts::Bool, computed_color::Bool, strokecolor::Bool, strokewidth::Bool}, cached::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}})
    @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:370
  [6] locked_resolve!(edge::ComputePipeline.TypedEdge{@NamedTuple{input_text::Base.RefValue{Any}, fontsize::Base.RefValue{Float32}, selected_font::Base.RefValue{FreeTypeAbstraction.FTFont}, align::Base.RefValue{Any}, rotation::Base.RefValue{Quaternion{Float64}}, justification::Base.RefValue{Makie.Automatic}, lineheight::Base.RefValue{Float64}, word_wrap_width::Base.RefValue{Int64}, offset::Base.RefValue{Vec{3, Float32}}, fonts::Base.RefValue{Attributes}, computed_color::Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, strokecolor::Base.RefValue{ColorTypes.RGBA{Float32}}, strokewidth::Base.RefValue{Float32}}, Tuple{Base.RefValue{Vector{Makie.GlyphCollection}}, Base.RefValue{Vector{UInt64}}, Base.RefValue{Vector{FreeTypeAbstraction.FTFont}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Makie.GlyphExtent}}, Base.RefValue{Vector{UnitRange{Int64}}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Quaternionf}}, Base.RefValue{Vector{Vec{2, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Pair{Int64, Int64}}}}, Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1"})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:935
  [7] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:980
  [8] locked_resolve!(computed::ComputePipeline.Computed)
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:969
  [9] foreach
    @ ./abstractarray.jl:3188 [inlined]
 [10] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:976
 [11] macro expansion
    @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:1004 [inlined]
 [12] macro expansion
    @ ./lock.jl:376 [inlined]
 [13] resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:1003
 [14] resolve!(computed::ComputePipeline.Computed)
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:994
 [15] getindex
    @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:861 [inlined]
 [16] (::ComputePipeline.var"#ComputeGraph##0#ComputeGraph##1"{ComputePipeline.ComputeGraph})(changeset::Set{Symbol})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:408
 [17] notify
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:206 [inlined]
 [18] foreach(f::typeof(notify), itr::Vector{Observable})
    @ Base ./abstractarray.jl:3188
 [19] update_observables!
    @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:550 [inlined]
 [20] update_observables!
    @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:548 [inlined]
 [21] setproperty!(attr::ComputePipeline.ComputeGraph, key::Symbol, value::Vector{String})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:601
 [22] (::ComputePipeline.var"#add_input!##2#add_input!##3"{ComputePipeline.ComputeGraph, Symbol})(new_val::Vector{String})
    @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:1267
 [23] notify
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:206 [inlined]
 [24] setindex!(observable::Observable, val::Any)
    @ Observables ~/.julia/packages/Observables/YdEbO/src/Observables.jl:123
 [25] (::Observables.MapCallback)(value::Any)
    @ Observables ~/.julia/packages/Observables/YdEbO/src/Observables.jl:436
 [26] notify
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:206 [inlined]
 [27] setindex!(observable::Observable, val::Any)
    @ Observables ~/.julia/packages/Observables/YdEbO/src/Observables.jl:123
 [28] (::Observables.MapCallback)(value::Any)
    @ Observables ~/.julia/packages/Observables/YdEbO/src/Observables.jl:436
 [29] notify
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:206 [inlined]
 [30] setindex!
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:123 [inlined]
 [31] (::Makie.var"#2523#2524"{Menu, Observable{Vector{Int64}}, Observable{String}, Scene})(ev::Makie.KeyEvent)
    @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/makielayout/blocks/menu.jl:154
 [32] notify
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:206 [inlined]
 [33] setindex!
    @ ~/.julia/packages/Observables/YdEbO/src/Observables.jl:123 [inlined]
 [34] (::GLMakie.var"#keyoardbuttons#keyoardbuttons##0"{Observable{Makie.KeyEvent}})(window::GLFW.Window, button::GLFW.Key, scancode::Int32, action::GLFW.Action, mods::Int32)
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/events.jl:118
 [35] _KeyCallbackWrapper(window::GLFW.Window, key::GLFW.Key, scancode::Int32, action::GLFW.Action, mods::Int32)
    @ GLFW ~/.julia/packages/GLFW/wA4ue/src/callback.jl:46
 [36] macro expansion
    @ ~/.julia/packages/GLFW/wA4ue/src/GLFW.jl:34 [inlined]
 [37] PollEvents
    @ ~/.julia/packages/GLFW/wA4ue/src/glfw3.jl:773 [inlined]
 [38] pollevents(screen::GLMakie.Screen{GLFW.Window}, frame_state::Makie.TickState)
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:547
 [39] (::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}})()
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1100
 [40] with_context(f::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}}, context::GLFW.Window)
    @ GLMakie.GLAbstraction /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/GLAbstraction/GLAbstraction.jl:59
 [41] on_demand_renderloop(screen::GLMakie.Screen{GLFW.Window})
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1099
 [42] renderloop(screen::GLMakie.Screen{GLFW.Window})
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1127
 [43] (::GLMakie.var"#start_renderloop!##0#start_renderloop!##1"{GLMakie.Screen{GLFW.Window}})()
    @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:952
┌ Error: Failed to update renderobject - skipping update
│   exception =
│    Failed to resolve gl_renderobject:
│    [ComputeEdge] gl_renderobject = RenderObjectUpdater((per_char_positions_transformed_f32c, text_color, text_strokecolor, text_rotation, marker_offset, quad_offset, sdf_uv, quad_scale, lowclip_color, highclip_color, nan_color, strokewidth, glowcolor, glowwidth, model_f32c, transform_marker, gl_indices, gl_len, f32c_scale, uniform_clip_planes, uniform_num_clip_planes, depth_shift, visible, fxaa, resolution, projection, projectionview, view, upvector, eyeposition, view_direction, preprojection, ), changed, cached)
│      @ /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/plot-primitives.jl:87
│    [ComputeEdge] per_char_positions_transformed_f32c = (::MapFunctionWrapper(#register_text_computations!##12))((text_blocks, positions_transformed_f32c, ), changed, cached)
│      @ unknown method location
│    [ComputeEdge] glyphcollections, glyphindices, font_per_char, glyph_origins, glyph_extents, text_blocks, text_color, text_rotation, text_scales, text_strokewidth, text_strokecolor, linesegments, linewidths, linecolors, lineindices = #compute_glyph_collections!##0((input_text, fontsize, selected_font, align, rotation, justification, lineheight, word_wrap_width, offset, fonts, computed_color, strokecolor, strokewidth, ), changed, cached)
│      @ /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:349
│      with edge inputs:
│        input_text = ["Stabilization"]
│        fontsize = 14.0f0
│        selected_font = FTFont (family = TeX Gyre Heros Makie, style = Regular)
│        align = (:left, :center)
│        rotation = 1.0 + 0.0im + 0.0jm + 0.0km
│        justification = Makie.Automatic()
│        lineheight = 1.0
│        word_wrap_width = -1
│        offset = Float32[0.0, 0.0, 0.0]
│        fonts = Attributes()
│        computed_color = ColorTypes.RGBA{Float32}[]
│        strokecolor = RGBA{Float32}(0.0, 0.0, 0.0, 0.0)
│        strokewidth = 0.0f0
│    Triggered by update of:
│      alpha, strokecolor, position, colormap, colorscale, text, fonts, highclip, word_wrap_width, justification, font, align, nan_color, strokewidth, rotation, arg1, lineheight, colorrange, offset, color, fontsize or lowclip
│    Due to ERROR: BoundsError: attempt to access 0-element Vector{ColorTypes.RGBA{Float32}} at index [0]
│    Stacktrace:
│      [1] throw_boundserror(A::Vector{ColorTypes.RGBA{Float32}}, I::Tuple{Int64})
│        @ Base ./essentials.jl:15
│      [2] getindex
│        @ ./essentials.jl:919 [inlined]
│      [3] per_glyph_block(data::Vector{ColorTypes.RGBA{Float32}}, block_idx::Int64, N_blocks::Int64, block::UnitRange{Int64})
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:192
│      [4] convert_text_string!(outputs::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}}, input_text::String, i::Int64, N::Int64, fontsize::Float32, font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32)
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:215
│      [5] (::Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1")(::@NamedTuple{input_text::Vector{String}, fontsize::Float32, selected_font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, computed_color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32}, changed::@NamedTuple{input_text::Bool, fontsize::Bool, selected_font::Bool, align::Bool, rotation::Bool, justification::Bool, lineheight::Bool, word_wrap_width::Bool, offset::Bool, fonts::Bool, computed_color::Bool, strokecolor::Bool, strokewidth::Bool}, cached::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}})
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:370
│      [6] locked_resolve!(edge::ComputePipeline.TypedEdge{@NamedTuple{input_text::Base.RefValue{Any}, fontsize::Base.RefValue{Float32}, selected_font::Base.RefValue{FreeTypeAbstraction.FTFont}, align::Base.RefValue{Any}, rotation::Base.RefValue{Quaternion{Float64}}, justification::Base.RefValue{Makie.Automatic}, lineheight::Base.RefValue{Float64}, word_wrap_width::Base.RefValue{Int64}, offset::Base.RefValue{Vec{3, Float32}}, fonts::Base.RefValue{Attributes}, computed_color::Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, strokecolor::Base.RefValue{ColorTypes.RGBA{Float32}}, strokewidth::Base.RefValue{Float32}}, Tuple{Base.RefValue{Vector{Makie.GlyphCollection}}, Base.RefValue{Vector{UInt64}}, Base.RefValue{Vector{FreeTypeAbstraction.FTFont}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Makie.GlyphExtent}}, Base.RefValue{Vector{UnitRange{Int64}}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Quaternionf}}, Base.RefValue{Vector{Vec{2, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Pair{Int64, Int64}}}}, Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1"})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:935
│      [7] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:980
│      [8] locked_resolve!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:969
│      [9] foreach
│        @ ./abstractarray.jl:3188 [inlined]
│     [10] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:976
│     [11] locked_resolve!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:969
│     [12] foreach
│        @ ./abstractarray.jl:3188 [inlined]
│     [13] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:976
│     [14] macro expansion
│        @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:1004 [inlined]
│     [15] macro expansion
│        @ ./lock.jl:376 [inlined]
│     [16] resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:1003
│     [17] resolve!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:994
│     [18] getindex(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:861
│     [19] (::GLMakie.var"#poll_updates##2#poll_updates##3"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1079
│     [20] with_context(f::GLMakie.var"#poll_updates##2#poll_updates##3"{GLMakie.Screen{GLFW.Window}}, context::GLFW.Window)
│        @ GLMakie.GLAbstraction /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/GLAbstraction/GLAbstraction.jl:59
│     [21] (::GLMakie.var"#poll_updates##0#poll_updates##1"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1070
│     [22] poll_updates
│        @ /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1069 [inlined]
│     [23] (::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1101
│     [24] with_context(f::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}}, context::GLFW.Window)
│        @ GLMakie.GLAbstraction /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/GLAbstraction/GLAbstraction.jl:59
│     [25] on_demand_renderloop(screen::GLMakie.Screen{GLFW.Window})
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1099
│     [26] renderloop(screen::GLMakie.Screen{GLFW.Window})
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1127
│     [27] (::GLMakie.var"#start_renderloop!##0#start_renderloop!##1"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:952
└ @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1081
┌ Warning: error in renderloop
│   exception =
│    Failed to resolve gl_renderobject:
│    [ComputeEdge] gl_renderobject = RenderObjectUpdater((per_char_positions_transformed_f32c, text_color, text_strokecolor, text_rotation, marker_offset, quad_offset, sdf_uv, quad_scale, lowclip_color, highclip_color, nan_color, strokewidth, glowcolor, glowwidth, model_f32c, transform_marker, gl_indices, gl_len, f32c_scale, uniform_clip_planes, uniform_num_clip_planes, depth_shift, visible, fxaa, resolution, projection, projectionview, view, upvector, eyeposition, view_direction, preprojection, ), changed, cached)
│      @ /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/plot-primitives.jl:87
│    [ComputeEdge] per_char_positions_transformed_f32c = (::MapFunctionWrapper(#register_text_computations!##12))((text_blocks, positions_transformed_f32c, ), changed, cached)
│      @ unknown method location
│    [ComputeEdge] glyphcollections, glyphindices, font_per_char, glyph_origins, glyph_extents, text_blocks, text_color, text_rotation, text_scales, text_strokewidth, text_strokecolor, linesegments, linewidths, linecolors, lineindices = #compute_glyph_collections!##0((input_text, fontsize, selected_font, align, rotation, justification, lineheight, word_wrap_width, offset, fonts, computed_color, strokecolor, strokewidth, ), changed, cached)
│      @ /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:349
│      with edge inputs:
│        input_text = ["Stabilization"]
│        fontsize = 14.0f0
│        selected_font = FTFont (family = TeX Gyre Heros Makie, style = Regular)
│        align = (:left, :center)
│        rotation = 1.0 + 0.0im + 0.0jm + 0.0km
│        justification = Makie.Automatic()
│        lineheight = 1.0
│        word_wrap_width = -1
│        offset = Float32[0.0, 0.0, 0.0]
│        fonts = Attributes()
│        computed_color = ColorTypes.RGBA{Float32}[]
│        strokecolor = RGBA{Float32}(0.0, 0.0, 0.0, 0.0)
│        strokewidth = 0.0f0
│    Triggered by update of:
│      alpha, strokecolor, position, colormap, colorscale, text, fonts, highclip, word_wrap_width, justification, font, align, nan_color, strokewidth, rotation, arg1, lineheight, colorrange, offset, color, fontsize or lowclip
│    Due to ERROR: BoundsError: attempt to access 0-element Vector{ColorTypes.RGBA{Float32}} at index [0]
│    Stacktrace:
│      [1] throw_boundserror(A::Vector{ColorTypes.RGBA{Float32}}, I::Tuple{Int64})
│        @ Base ./essentials.jl:15
│      [2] getindex
│        @ ./essentials.jl:919 [inlined]
│      [3] per_glyph_block(data::Vector{ColorTypes.RGBA{Float32}}, block_idx::Int64, N_blocks::Int64, block::UnitRange{Int64})
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:192
│      [4] convert_text_string!(outputs::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}}, input_text::String, i::Int64, N::Int64, fontsize::Float32, font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32)
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:215
│      [5] (::Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1")(::@NamedTuple{input_text::Vector{String}, fontsize::Float32, selected_font::FreeTypeAbstraction.FTFont, align::Tuple{Symbol, Symbol}, rotation::Quaternion{Float64}, justification::Makie.Automatic, lineheight::Float64, word_wrap_width::Int64, offset::Vec{3, Float32}, fonts::Attributes, computed_color::Vector{ColorTypes.RGBA{Float32}}, strokecolor::ColorTypes.RGBA{Float32}, strokewidth::Float32}, changed::@NamedTuple{input_text::Bool, fontsize::Bool, selected_font::Bool, align::Bool, rotation::Bool, justification::Bool, lineheight::Bool, word_wrap_width::Bool, offset::Bool, fonts::Bool, computed_color::Bool, strokecolor::Bool, strokewidth::Bool}, cached::@NamedTuple{glyphcollections::Vector{Makie.GlyphCollection}, glyphindices::Vector{UInt64}, font_per_char::Vector{FreeTypeAbstraction.FTFont}, glyph_origins::Vector{Point{3, Float32}}, glyph_extents::Vector{Makie.GlyphExtent}, text_blocks::Vector{UnitRange{Int64}}, text_color::Vector{ColorTypes.RGBA{Float32}}, text_rotation::Vector{Quaternionf}, text_scales::Vector{Vec{2, Float32}}, text_strokewidth::Vector{Float32}, text_strokecolor::Vector{ColorTypes.RGBA{Float32}}, linesegments::Vector{Point{3, Float32}}, linewidths::Vector{Float32}, linecolors::Vector{ColorTypes.RGBA{Float32}}, lineindices::Vector{Pair{Int64, Int64}}})
│        @ Makie /sim/Programmieren/VideoEdit/dev/Makie/Makie/src/basic_recipes/text.jl:370
│      [6] locked_resolve!(edge::ComputePipeline.TypedEdge{@NamedTuple{input_text::Base.RefValue{Any}, fontsize::Base.RefValue{Float32}, selected_font::Base.RefValue{FreeTypeAbstraction.FTFont}, align::Base.RefValue{Any}, rotation::Base.RefValue{Quaternion{Float64}}, justification::Base.RefValue{Makie.Automatic}, lineheight::Base.RefValue{Float64}, word_wrap_width::Base.RefValue{Int64}, offset::Base.RefValue{Vec{3, Float32}}, fonts::Base.RefValue{Attributes}, computed_color::Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, strokecolor::Base.RefValue{ColorTypes.RGBA{Float32}}, strokewidth::Base.RefValue{Float32}}, Tuple{Base.RefValue{Vector{Makie.GlyphCollection}}, Base.RefValue{Vector{UInt64}}, Base.RefValue{Vector{FreeTypeAbstraction.FTFont}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Makie.GlyphExtent}}, Base.RefValue{Vector{UnitRange{Int64}}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Quaternionf}}, Base.RefValue{Vector{Vec{2, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Point{3, Float32}}}, Base.RefValue{Vector{Float32}}, Base.RefValue{Vector{ColorTypes.RGBA{Float32}}}, Base.RefValue{Vector{Pair{Int64, Int64}}}}, Makie.var"#compute_glyph_collections!##0#compute_glyph_collections!##1"})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:935
│      [7] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:980
│      [8] locked_resolve!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:969
│      [9] foreach
│        @ ./abstractarray.jl:3188 [inlined]
│     [10] locked_resolve!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:976
│     [11] locked_resolve!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:969
│     [12] foreach
│        @ ./abstractarray.jl:3188 [inlined]
│     [13] macro expansion
│        @ /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:482 [inlined]
│     [14] macro expansion
│        @ ./lock.jl:376 [inlined]
│     [15] mark_resolved!(edge::ComputePipeline.ComputeEdge{ComputePipeline.ComputeGraph})
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:481
│     [16] mark_resolved!(computed::ComputePipeline.Computed)
│        @ ComputePipeline /sim/Programmieren/VideoEdit/dev/Makie/ComputePipeline/src/ComputePipeline.jl:472
│     [17] (::GLMakie.var"#poll_updates##2#poll_updates##3"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1085
│     [18] with_context(f::GLMakie.var"#poll_updates##2#poll_updates##3"{GLMakie.Screen{GLFW.Window}}, context::GLFW.Window)
│        @ GLMakie.GLAbstraction /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/GLAbstraction/GLAbstraction.jl:59
│     [19] (::GLMakie.var"#poll_updates##0#poll_updates##1"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1070
│     [20] poll_updates
│        @ /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1069 [inlined]
│     [21] (::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1101
│     [22] with_context(f::GLMakie.var"#on_demand_renderloop##0#on_demand_renderloop##1"{GLMakie.Screen{GLFW.Window}, Base.RefValue{Makie.TickState}}, context::GLFW.Window)
│        @ GLMakie.GLAbstraction /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/GLAbstraction/GLAbstraction.jl:59
│     [23] on_demand_renderloop(screen::GLMakie.Screen{GLFW.Window})
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1099
│     [24] renderloop(screen::GLMakie.Screen{GLFW.Window})
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1127
│     [25] (::GLMakie.var"#start_renderloop!##0#start_renderloop!##1"{GLMakie.Screen{GLFW.Window}})()
│        @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:952
└ @ GLMakie /sim/Programmieren/VideoEdit/dev/Makie/GLMakie/src/screen.jl:1136

```
