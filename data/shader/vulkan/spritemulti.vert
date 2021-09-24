#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 inVertex;
layout (location = 1) in vec2 inVertexTexCoord;
layout (location = 2) in vec4 inVertexColor;

layout(push_constant) uniform SPosBO {
	layout(offset = 0) uniform mat4x2 gPos;
	layout(offset = 32) uniform vec2 gCenter;
} gPosBO;

layout (std140, set = 1, binding = 1) uniform SRSPBO {
	vec4 gRSP[512];
} gRSPBO;

layout (location = 0) noperspective out vec2 texCoord;
layout (location = 1) noperspective out vec4 vertColor;

void main()
{
	vec2 FinalPos = vec2(inVertex.xy);
	if(gRSPBO.gRSP[gl_InstanceIndex].w != 0.0)
	{
		float X = FinalPos.x - gPosBO.gCenter.x;
		float Y = FinalPos.y - gPosBO.gCenter.y;
		
		FinalPos.x = X * cos(gRSPBO.gRSP[gl_InstanceIndex].w) - Y * sin(gRSPBO.gRSP[gl_InstanceIndex].w) + gPosBO.gCenter.x;
		FinalPos.y = X * sin(gRSPBO.gRSP[gl_InstanceIndex].w) + Y * cos(gRSPBO.gRSP[gl_InstanceIndex].w) + gPosBO.gCenter.y;
	}
	
	FinalPos.x *= gRSPBO.gRSP[gl_InstanceIndex].z;
	FinalPos.y *= gRSPBO.gRSP[gl_InstanceIndex].z;
		
	FinalPos.x += gRSPBO.gRSP[gl_InstanceIndex].x;
	FinalPos.y += gRSPBO.gRSP[gl_InstanceIndex].y;

	gl_Position = vec4(gPosBO.gPos * vec4(FinalPos, 0.0, 1.0), 0.0, 1.0);
	texCoord = inVertexTexCoord;
	vertColor = inVertexColor;
}
