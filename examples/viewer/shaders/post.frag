#version 450

#include "tonemap.glsl"

// The scene's linear-HDR render target. Sampled, not read as an input
// attachment, because later phases of this pass will want to read neighbouring
// texels (bloom) and at reduced rates, which an input attachment cannot do.
layout(set = 0, binding = 0) uniform sampler2D uScene;
// Level 0 of the bloom chain, which by this point holds every level summed.
layout(set = 0, binding = 1) uniform sampler2D uBloom;

layout(push_constant) uniform Push {
    // x = exposure, y = tone curve (0 Reinhard, 1 ACES), z = bloom intensity,
    // w reserved.
    vec4 params;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 hdr = texture(uScene, vUV).rgb;

    // Guarded rather than multiplied by zero. With bloom off the chain has not
    // run and that texture holds whatever was in memory - and mix() at a
    // weight of zero still evaluates x*(1-a) + y*a, so a single Inf or NaN
    // texel would propagate straight through a weight that should have made
    // it irrelevant. Uniform control flow, so it costs nothing.
    if (pc.params.z <= 0.0) {
        outColor = vec4(tonemap(hdr, pc.params.x, pc.params.y), 1.0);
        return;
    }
    vec3 bloom = texture(uBloom, vUV).rgb;

    // LERP toward the bloom, not add it.
    //
    // Adding is the common choice and it is an energy leak: every bright pixel
    // gets its own light back plus a blurred copy of itself, so the image gets
    // brighter overall and the tone curve has to absorb it. Mixing conserves
    // what is there - light is REDISTRIBUTED from where it was into the
    // neighbourhood around it, which is what a lens actually does to it.
    //
    // It also fails gracefully: at intensity 1 the image becomes the fully
    // blurred version rather than an ever-brighter white, so no value of the
    // knob produces the blown-out look an additive chain reaches easily.
    vec3 c = mix(hdr, bloom, clamp(pc.params.z, 0.0, 1.0));

    outColor = vec4(tonemap(c, pc.params.x, pc.params.y), 1.0);
}
