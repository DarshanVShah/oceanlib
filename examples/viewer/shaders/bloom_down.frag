#version 450

// One step down the bloom chain: read the level above, write this one at half
// its resolution.
//
// The filter is the 13-tap from Jimenez's Call of Duty presentation, not a
// plain bilinear halving. A bilinear downsample of an HDR image aliases
// horribly, and it aliases in the worst possible way for bloom: what it
// aliases ARE the bright pixels, so the glow flickers as the camera moves,
// which is exactly when anyone is looking at it. The 13 taps form a wider,
// smoother kernel whose support overlaps between neighbouring output texels,
// so no bright texel can fall between the cracks the way it can with four.

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(push_constant) uniform Push {
    // xy = 1 / source resolution, z = 1 on the first step only, w = threshold
    vec4 params;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

// Karis average: weight each group by 1/(1+luma) before summing, so a single
// blazing texel cannot dominate its neighbourhood.
//
// This is the fireflies fix, and it is only applied on the FIRST step. Sun
// glitter on water is the pathological case for bloom: isolated pixels
// thousands of times brighter than the sea around them, appearing and
// vanishing between frames as the wave slope crosses the mirror angle. Averaged
// naively they become a sparkling mess of popping halos. Weighted this way, a
// lone spike contributes roughly its share of a bright neighbourhood instead of
// all of it. Applied on later steps it would just flatten the glow, since by
// then the spikes are already averaged away.
float karis_weight(vec3 c)
{
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma);
}

void main()
{
    vec2 t = pc.params.xy;

    // The 13 taps: an inner 2x2 group at half-texel offsets, plus four corner
    // groups and a centre, arranged so the weights sum to 1.
    vec3 a = texture(uSource, vUV + vec2(-2.0, -2.0) * t).rgb;
    vec3 b = texture(uSource, vUV + vec2( 0.0, -2.0) * t).rgb;
    vec3 c = texture(uSource, vUV + vec2( 2.0, -2.0) * t).rgb;
    vec3 d = texture(uSource, vUV + vec2(-2.0,  0.0) * t).rgb;
    vec3 e = texture(uSource, vUV                     ).rgb;
    vec3 f = texture(uSource, vUV + vec2( 2.0,  0.0) * t).rgb;
    vec3 g = texture(uSource, vUV + vec2(-2.0,  2.0) * t).rgb;
    vec3 h = texture(uSource, vUV + vec2( 0.0,  2.0) * t).rgb;
    vec3 i = texture(uSource, vUV + vec2( 2.0,  2.0) * t).rgb;

    vec3 j = texture(uSource, vUV + vec2(-1.0, -1.0) * t).rgb;
    vec3 k = texture(uSource, vUV + vec2( 1.0, -1.0) * t).rgb;
    vec3 l = texture(uSource, vUV + vec2(-1.0,  1.0) * t).rgb;
    vec3 m = texture(uSource, vUV + vec2( 1.0,  1.0) * t).rgb;

    vec3 result;
    if (pc.params.z > 0.5) {
        // Five overlapping 2x2 groups, each averaged, each Karis-weighted.
        vec3 g0 = (j + k + l + m) * 0.25;
        vec3 g1 = (a + b + d + e) * 0.25;
        vec3 g2 = (b + c + e + f) * 0.25;
        vec3 g3 = (d + e + g + h) * 0.25;
        vec3 g4 = (e + f + h + i) * 0.25;

        float w0 = karis_weight(g0) * 0.5;
        float w1 = karis_weight(g1) * 0.125;
        float w2 = karis_weight(g2) * 0.125;
        float w3 = karis_weight(g3) * 0.125;
        float w4 = karis_weight(g4) * 0.125;

        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) /
                 max(w0 + w1 + w2 + w3 + w4, 1e-5);

        // No hard brightness threshold anywhere in this chain.
        //
        // The usual "only bloom pixels above 1.0" cutoff is a hangover from
        // LDR rendering, and on an HDR target it does visible harm: a wave
        // crest drifting across the cutoff pops its glow on and off, and the
        // sea is full of crests hovering right at it. Letting everything bloom
        // a little and relying on the intensity in the composite keeps it
        // continuous, which matters far more on moving water than a crisp
        // separation between "bright" and "not" ever would.
    } else {
        result = e * 0.125;
        result += (a + c + g + i) * 0.03125;
        result += (b + d + f + h) * 0.0625;
        result += (j + k + l + m) * 0.125;
    }

    outColor = vec4(max(result, vec3(0.0)), 1.0);
}
