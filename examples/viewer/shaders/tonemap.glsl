// Tone mapping, split out of common.glsl so the post pass can use it without
// also declaring the Globals uniform block - the post pipeline binds only the
// HDR image it is reading, and a block it never writes to would still have to
// be bound for the pipeline layout to be valid.
//
// Nothing in the scene shaders calls this any more. They write linear radiance
// into an fp16 target and the post pass maps it once, at the end, where a
// tone curve belongs: applying it per material meant every shader had to agree
// about exposure, and it made bloom impossible, because bloom has to gather
// light that a tone curve has not yet compressed.

#ifndef OCEAN_TONEMAP_GLSL
#define OCEAN_TONEMAP_GLSL

vec3 tonemap(vec3 linear, float exposure)
{
    vec3 c = linear * exposure;

    // Reinhard on LUMINANCE, not per channel.
    //
    // Per-channel Reinhard compresses a bright channel harder than a dim one,
    // so it desaturates everything bright toward grey. That is exactly what
    // made the Preetham sky look washed out: at the zenith the model produced
    // linear (0.139, 0.236, 0.471) - blue is 3.4x red, a proper sky - and
    // per-channel mapping delivered (104, 126, 158), nearly neutral. The
    // chromaticity the whole model exists to compute was being thrown away in
    // the last line of the shader.
    //
    // Compressing luminance and carrying the chroma through unchanged
    // preserves hue and saturation exactly.
    float L  = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float Lt = L / (1.0 + L);
    vec3 mapped = c * (Lt / max(L, 1e-5));

    // Except that genuinely intense sources DO read as white - a photograph of
    // the sun is a white disc, not a saturated orange one. So fade back toward
    // the per-channel result as luminance climbs, which restores that
    // behaviour for the sun and its glitter without touching the sky.
    vec3 perChannel = c / (c + vec3(1.0));
    mapped = mix(mapped, perChannel, clamp(L * 0.12, 0.0, 1.0));

    return pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
}

#endif
