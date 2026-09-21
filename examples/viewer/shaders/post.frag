#version 450

#include "tonemap.glsl"

// The scene's linear-HDR render target. Sampled, not read as an input
// attachment, because later phases of this pass will want to read neighbouring
// texels (bloom) and at reduced rates, which an input attachment cannot do.
layout(set = 0, binding = 0) uniform sampler2D uScene;

layout(push_constant) uniform Push {
    // x = exposure, yzw reserved for the bloom parameters this pass will grow.
    vec4 params;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 hdr = texture(uScene, vUV).rgb;
    outColor = vec4(tonemap(hdr, pc.params.x), 1.0);
}
