function build_shader() {
	$1 $2
	tmp="$2.tmp"
	mv "$2" "$tmp"
	spirv-opt -O "$tmp" -o "$2"
	rm "$tmp"
}

vulkan_version="vulkan100"

# primitives
build_shader "glslangValidator --client $vulkan_version prim.frag -o " "prim.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TEXTURED prim.frag -o " "prim_textured.frag.spv" &

build_shader "glslangValidator --client $vulkan_version prim.vert -o " "prim.vert.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TEXTURED prim.vert -o " "prim_textured.vert.spv" &

# text
build_shader "glslangValidator --client $vulkan_version text.frag -o " "text.frag.spv" &
build_shader "glslangValidator --client $vulkan_version text.vert -o " "text.vert.spv" &

# quad container
build_shader "glslangValidator --client $vulkan_version primex.frag -o " "primex.frag.spv" &
build_shader "glslangValidator --client $vulkan_version primex.vert -o " "primex.vert.spv" &

build_shader "glslangValidator --client $vulkan_version primex.frag -o " "primex_rotationless.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_ROTATIONLESS primex.vert -o " "primex_rotationless.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TEXTURED primex.frag -o " "primex_tex.frag.spv" &
build_shader "glslangValidator --client $vulkan_version primex.vert -o " "primex_tex.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TEXTURED primex.frag -o " "primex_tex_rotationless.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_ROTATIONLESS primex.vert -o " "primex_tex_rotationless.vert.spv" &

build_shader "glslangValidator --client $vulkan_version spritemulti.frag -o " "spritemulti.frag.spv" &
build_shader "glslangValidator --client $vulkan_version spritemulti.vert -o " "spritemulti.vert.spv" &

# tile layer
build_shader "glslangValidator --client $vulkan_version tile.frag -o " "tile.frag.spv" &
build_shader "glslangValidator --client $vulkan_version tile.vert -o " "tile.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TILE_TEXTURED tile.frag -o " "tile_textured.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TILE_TEXTURED tile.vert -o " "tile_textured.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER tile.frag -o " "tile_border.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER tile.vert -o " "tile_border.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER -DTW_TILE_TEXTURED tile.frag -o " "tile_border_textured.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER -DTW_TILE_TEXTURED tile.vert -o " "tile_border_textured.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER_LINE tile.frag -o " "tile_border_line.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER_LINE tile.vert -o " "tile_border_line.vert.spv" &

build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER_LINE -DTW_TILE_TEXTURED tile.frag -o " "tile_border_line_textured.frag.spv" &
build_shader "glslangValidator --client $vulkan_version -DTW_TILE_BORDER_LINE -DTW_TILE_TEXTURED tile.vert -o " "tile_border_line_textured.vert.spv" &

wait

echo "Done compiling shaders"
