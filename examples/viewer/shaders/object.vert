#version 450

#include "common.glsl"

// Scale is kept OUT of the matrix so the normal transform stays correct and
// cheap. For a diagonal scale the inverse transpose is just the reciprocal, so
// dividing the normal by the scale is exact - baking scale into the matrix and
// using mat3(model) would shade every stretched box wrong, and the boat's hull
// is stretched 3:1.
layout(push_constant) uniform Push {
    mat4 model;
    vec4 scale;
    vec4 color;
    vec4 surf;    // see object.frag; unused here, but the block must match
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;

void main()
{
    vec4 wp = pc.model * vec4(inPos * pc.scale.xyz, 1.0);
    vWorld  = wp.xyz;
    vNormal = normalize(mat3(pc.model) * (inNormal / pc.scale.xyz));
    gl_Position = g.viewProj * wp;
}
