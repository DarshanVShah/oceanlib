#version 450

// One step back up the bloom chain: read the smaller level below, add it into
// this one.
//
// The addition is done by the blend state (ONE, ONE), not in the shader, so
// this writes only its own contribution. That is what makes the chain
// progressive: every level ends up holding its own gathered light plus
// everything coarser, and level 0 therefore carries the whole multi-scale glow
// by the time the composite reads it. One texture read per level, no
// accumulation buffer, no ping-pong.
//
// A 3x3 tent rather than another wide gather. Going up, the source is already
// smooth - it has been filtered on the way down - so the only job here is to
// interpolate it without reintroducing the box-shaped blockiness a straight
// bilinear upsample leaves behind. The tent is the cheapest kernel that does
// not, and stacking it once per level is what turns five discrete blurs into
// something that reads as one continuous falloff.

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(push_constant) uniform Push {
    // xy = 1 / source resolution, z unused, w = filter radius in source texels
    vec4 params;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    // The radius is in SOURCE texels and deliberately larger than one. At
    // exactly one texel the tent degenerates to the bilinear filter it exists
    // to improve on; widening it is what spreads each level's light far enough
    // to overlap the next, so the sum across levels has no visible rings.
    vec2 r = pc.params.xy * pc.params.w;

    vec3 sum = texture(uSource, vUV + vec2(-1.0,  1.0) * r).rgb * 1.0;
    sum     += texture(uSource, vUV + vec2( 0.0,  1.0) * r).rgb * 2.0;
    sum     += texture(uSource, vUV + vec2( 1.0,  1.0) * r).rgb * 1.0;

    sum     += texture(uSource, vUV + vec2(-1.0,  0.0) * r).rgb * 2.0;
    sum     += texture(uSource, vUV                       ).rgb * 4.0;
    sum     += texture(uSource, vUV + vec2( 1.0,  0.0) * r).rgb * 2.0;

    sum     += texture(uSource, vUV + vec2(-1.0, -1.0) * r).rgb * 1.0;
    sum     += texture(uSource, vUV + vec2( 0.0, -1.0) * r).rgb * 2.0;
    sum     += texture(uSource, vUV + vec2( 1.0, -1.0) * r).rgb * 1.0;

    outColor = vec4(sum * (1.0 / 16.0), 1.0);
}
