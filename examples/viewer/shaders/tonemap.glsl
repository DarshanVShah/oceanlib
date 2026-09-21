// Tone mapping, split out of common.glsl so the post pass can use it without
// also declaring the Globals uniform block - the post pipeline binds only the
// HDR image it is reading, and a block it never writes to would still have to
// be bound for the pipeline layout to be valid.
//
// Nothing in the scene shaders calls this any more. They write linear radiance
// into an fp16 target and the post pass maps it once, at the end, where a tone
// curve belongs: applying it per material meant every shader had to agree
// about exposure, and it made bloom impossible, because bloom has to gather
// light that a tone curve has not yet compressed.

#ifndef OCEAN_TONEMAP_GLSL
#define OCEAN_TONEMAP_GLSL

// --- Reinhard on luminance -------------------------------------------------
//
// The original curve, kept because it is the A/B control for the ACES fit
// below and because the reasoning it was built on is still worth having in
// front of anyone changing it.
vec3 tonemap_reinhard(vec3 linear, float exposure)
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

// --- ACES ------------------------------------------------------------------
//
// Stephen Hill's fit to the ACES RRT+ODT, which is the curve most film and
// game pipelines have converged on. It replaces the hand-built Reinhard above
// as the default, and it is worth being precise about what it actually buys,
// because "filmic" on its own is not a reason.
//
// The Reinhard curve had to be argued into behaving at both ends. Its
// luminance-only form kept the sky's chromaticity but made the sun a
// saturated disc, so a second term faded toward per-channel compression as
// luminance climbed - two curves and a blend, tuned by eye, to get
// desaturation that only happens where it should.
//
// ACES gets that for free, because the matrices bracketing the curve do the
// work. Input is rotated into a space where the tone curve is applied, then
// back, and that rotation is what makes bright saturated colour drift toward
// white on the way up instead of clipping to a channel edge. A sunset over
// water is the case that shows it: Reinhard clips the sun's reflection to
// flat orange where the red channel saturates first, and ACES rolls the whole
// highlight to white the way an exposure actually does.
//
// The cost is a slight overall desaturation and a lifted toe, which is exactly
// the "film look" complaint levelled at it. Keep --tonemap reinhard to see it.
const mat3 kAcesInput = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777);

const mat3 kAcesOutput = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602);

vec3 aces_rrt_odt(vec3 v)
{
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 tonemap_aces(vec3 linear, float exposure)
{
    vec3 c = linear * exposure;
    c = kAcesInput * c;
    c = aces_rrt_odt(c);
    c = kAcesOutput * c;
    // The swapchain is UNORM, not _SRGB, so the transfer function is ours to
    // apply - see the format choice in vk_context.cpp. 2.2 rather than the
    // piecewise sRGB curve: they differ only in the bottom few codes, and
    // nothing in this scene lives down there.
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}

// `mode`: 0 = Reinhard on luminance, anything else = ACES.
vec3 tonemap(vec3 linear, float exposure, float mode)
{
    return (mode < 0.5) ? tonemap_reinhard(linear, exposure)
                        : tonemap_aces(linear, exposure);
}

#endif
