#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec4 inVertex;
layout (location = 1) in vec4 inColor;
#ifdef TW_QUAD_TEXTURED
layout (location = 2) in vec2 inVertexTexCoord;
#endif

layout(push_constant) uniform SPosBO {
	layout(offset = 0) uniform mat4x2 gPos;
	layout(offset = 32) uniform int gQuadOffset;
} gPosBO;

layout (std140, set = 1, binding = 1) uniform SOffBO {
	uniform vec2 gOffsets[TW_MAX_QUADS];
} gOffBO;

layout (std140, set = 2, binding = 2) uniform SRotBO {
	uniform float gRotations[TW_MAX_QUADS];
} gRotBO;

noperspective out vec4 QuadColor;
flat out int QuadIndex;
#ifdef TW_QUAD_TEXTURED
noperspective out vec2 TexCoord;
#endif

void main()
{
	vec2 FinalPos = vec2(inVertex.xy);
	
	int TmpQuadIndex = int(gl_VertexIndex / 4) - gPosBO.gQuadOffset;

	if(gRotBO.gRotations[TmpQuadIndex] != 0.0)
	{
		float X = FinalPos.x - inVertex.z;
		float Y = FinalPos.y - inVertex.w;
		
		FinalPos.x = X * cos(gRotBO.gRotations[TmpQuadIndex]) - Y * sin(gRotBO.gRotations[TmpQuadIndex]) + inVertex.z;
		FinalPos.y = X * sin(gRotBO.gRotations[TmpQuadIndex]) + Y * cos(gRotBO.gRotations[TmpQuadIndex]) + inVertex.w;
	}
	
	FinalPos.x = FinalPos.x / 1024.0 + gOffBO.gOffsets[TmpQuadIndex].x;
	FinalPos.y = FinalPos.y / 1024.0 + gOffBO.gOffsets[TmpQuadIndex].y;

	gl_Position = vec4(gPosBO.gPos * vec4(FinalPos, 0.0, 1.0), 0.0, 1.0);
	QuadColor = inColor;
	QuadIndex = TmpQuadIndex;
#ifdef TW_QUAD_TEXTURED
	TexCoord = inVertexTexCoord;
#endif
}
