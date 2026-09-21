// Shared by every stage. glslc resolves quoted includes relative to the
// including file.
//
// The sky model lives here rather than being duplicated because the water
// reflects it: if the sky pass and the reflection disagreed, the horizon would
// visibly seam where the ocean meets the sky.

#ifndef OCEAN_COMMON_GLSL
#define OCEAN_COMMON_GLSL

// The viewer demonstrates exactly 3 cascades: a far scale for big swells, a
// mid scale, and a near scale for fine ripples (see ADR-020). The library's
// own CascadeStack is not limited to 3 - this is a viewer-only simplification
// so the descriptor layout and shader loops can be fixed-size rather than
// driven by a runtime count, which keeps the demo simple without limiting
// what the library itself can do.
const int kCascadeCount = 3;

layout(set = 0, binding = 0) uniform Globals {
    mat4 viewProj;
    mat4 invViewProj;
    vec4 camPos;       // xyz
    vec4 sunDir;       // xyz, normalised, pointing TOWARD the sun
    vec4 cascadePatch; // x,y,z = patch_length (m) of cascades 0,1,2; w = tiles
                       // per side of the outer mesh, sized to cascade 0 (the
                       // largest/farthest scale)
    vec4 params;       // x = time, y = mesh resolution, z = sky
                       // turbidity, w = sky luminance scale
    vec4 shading;      // x foam strength, y exposure, z fog density, w unused
    vec4 cascadeTexel; // xyz = world size of one texel in cascades 0,1,2;
                       // w   = world units per pixel per metre of distance
    vec4 interaction;  // xy = world position of the field's low corner,
                       // z  = field extent in metres,
                       // w  = 1 to show the interaction field in isolation
    vec4 water;        // x = camera depth below the surface, metres, positive
                       // when submerged; yzw unused
    vec4 slopeVar;     // xyz = mean-square slope of cascades 0,1,2, both axes
                       // summed - a property of the SPECTRUM, so it is
                       // measured once per sea state rather than per frame;
                       // w   = the BRDF's base roughness, as GGX alpha
} g;

// How much of a cascade survives at this distance, in [0,1].
//
// A cascade whose texels project to less than a pixel cannot be resolved: what
// reaches the screen is not detail but ALIASING, and it is the worst kind -
// it shimmers as the camera moves, because which sub-texel each pixel lands on
// changes every frame. The finest cascade is the worst offender, since it has
// the smallest texels and is still being sampled out to the horizon.
//
// So each cascade is faded out once its texel drops below roughly a pixel.
// This is mip-mapping's argument applied at the level of a whole frequency
// band: the band is removed rather than filtered, because the cascades ARE a
// frequency decomposition and dropping the top band is exactly what a low-pass
// filter would do.
//
// The far cascade is unaffected in any normal view - its texels are metres
// across - so the horizon keeps its large-scale shape and only loses the
// detail that could not have been drawn correctly anyway.
float detail_fade(float dist, float texelWorld)
{
    // Texels per pixel at this distance. Above 1 the band is resolvable.
    float perPixel = texelWorld / max(dist * g.cascadeTexel.w, 1e-6);
    return smoothstep(0.9, 2.6, perPixel);
}

// The local interaction field (ADR-021): eta, dEta/dx, dEta/dz, dEta/dt.
//
// Sampled with CLAMP_TO_BORDER and a transparent-black border, which is the
// EXACT OPPOSITE of the cascade textures above. Those are genuinely periodic,
// so REPEAT is correct for them. This one is not periodic at all - a REPEAT
// sampler here would tile the ripples from one splash across the entire ocean.
layout(set = 0, binding = 7) uniform sampler2D uInteraction;

vec4 interaction_sample(vec2 worldXZ)
{
    vec2 uv = (worldXZ - g.interaction.xy) / g.interaction.z;
    // Explicit reject as well as the border colour: belt and braces, and it
    // makes the intent legible rather than depending on sampler state set up
    // several hundred lines away in C++.
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0);
    // textureLod, not texture: this helper is called from the vertex shader
    // too, where there are no derivatives and an implicit LOD is undefined.
    return textureLod(uInteraction, uv, 0.0);
}

const float kPi = 3.14159265359;

// Preetham analytic sky.
//
// Replaces a two-colour gradient with a real scattering model, and it improves
// the WATER more than any change to the water would: most of what you see
// looking at an ocean is reflected sky, so the reflection is only ever as good
// as the thing being reflected. A flat gradient gives flat water.
//
// Preetham, Shirley & Smits 1999, "A Practical Analytic Model for Daylight".
// The model fits Perez's five-parameter sky distribution to output from a full
// spectral atmospheric simulation, parameterised by a single TURBIDITY number -
// roughly the ratio of scattering by aerosols to scattering by molecules. 2 is
// a very clear arctic sky, 3 clear, 6 hazy, 10+ murky.
//
// Chosen over Hosek-Wilkie (which is more accurate, especially near sunset)
// because Hosek-Wilkie needs a large table of fitted coefficients that would
// have to be baked into the binary, and ADR-001's zero-dependency rule makes
// a compact closed form worth more here than the last few per cent of fidelity.
//
// Everything is evaluated per pixel. It is a few dozen ALU ops against a
// cubemap fetch, which on any GPU that can run this demo is not the bottleneck,
// and it buys a sky that responds continuously to sun angle and turbidity
// rather than being baked at one time of day.

// Perez sky distribution. `cosTheta` is the cosine of the angle from the
// zenith, `gamma` the angle between the view direction and the sun.
float perez(float cosTheta, float gamma, float A, float B, float C, float D,
            float E)
{
    // cosTheta is clamped away from zero rather than allowed to reach it: the
    // exp(B/cosTheta) term diverges at the horizon, and the model is not
    // defined below it anyway.
    float ct = max(cosTheta, 0.01);
    float cg = cos(gamma);
    return (1.0 + A * exp(B / ct)) * (1.0 + C * exp(D * gamma) + E * cg * cg);
}

vec3 xyY_to_linear_rgb(float x, float y, float Y)
{
    // xyY -> XYZ. y is a denominator, so guard it.
    float yy = max(y, 1e-4);
    vec3 XYZ = vec3(x * Y / yy, Y, (1.0 - x - y) * Y / yy);

    // XYZ -> linear sRGB (Rec. 709 primaries, D65).
    return vec3(
        dot(XYZ, vec3( 3.2406, -1.5372, -0.4986)),
        dot(XYZ, vec3(-0.9689,  1.8758,  0.0415)),
        dot(XYZ, vec3( 0.0557, -0.2040,  1.0570)));
}

// `dir` and `sunDir` must be normalised. `turbidity` in roughly [1.7, 10].
vec3 sky_color_turbid(vec3 dir, vec3 sunDir, float turbidity)
{
    float T = clamp(turbidity, 1.7, 10.0);

    // Angles. thetaS is the sun's zenith angle; a sun below the horizon is
    // clamped to just above it so the model stays defined during a sunset
    // rather than producing negative luminance.
    float cosTheta = dir.y;
    float sunUp    = clamp(sunDir.y, 0.02, 1.0);
    float thetaS   = acos(sunUp);
    float gamma    = acos(clamp(dot(dir, sunDir), -1.0, 1.0));

    // Perez coefficients, linear in turbidity (Preetham table 1).
    float AY =  0.1787 * T - 1.4630;
    float BY = -0.3554 * T + 0.4275;
    float CY = -0.0227 * T + 5.3251;
    float DY =  0.1206 * T - 2.5771;
    float EY = -0.0670 * T + 0.3703;

    float Ax = -0.0193 * T - 0.2592;
    float Bx = -0.0665 * T + 0.0008;
    float Cx = -0.0004 * T + 0.2125;
    float Dx = -0.0641 * T - 0.8989;
    float Ex = -0.0033 * T + 0.0452;

    float Ay = -0.0167 * T - 0.2608;
    float By = -0.0950 * T + 0.0092;
    float Cy = -0.0079 * T + 0.2102;
    float Dy = -0.0441 * T - 1.6537;
    float Ey = -0.0109 * T + 0.0529;

    // Zenith absolute luminance, in kcd/m^2.
    float chi = (4.0 / 9.0 - T / 120.0) * (kPi - 2.0 * thetaS);
    float Yz  = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;

    // Zenith chromaticity: a cubic in the solar zenith angle, quadratic in
    // turbidity.
    float t2 = thetaS * thetaS;
    float t3 = t2 * thetaS;
    float T2 = T * T;

    float xz =
        ( 0.00166 * t3 - 0.00375 * t2 + 0.00209 * thetaS)             * T2 +
        (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * thetaS + 0.00394)   * T  +
        ( 0.11693 * t3 - 0.21196 * t2 + 0.06052 * thetaS + 0.25886);

    float yz =
        ( 0.00275 * t3 - 0.00610 * t2 + 0.00317 * thetaS)             * T2 +
        (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * thetaS + 0.00516)   * T  +
        ( 0.15346 * t3 - 0.26756 * t2 + 0.06670 * thetaS + 0.26688);

    // The model gives RATIOS against the zenith, so every quantity is the
    // zenith value scaled by perez(view) / perez(zenith).
    float denomY = perez(1.0, thetaS, AY, BY, CY, DY, EY);
    float denomx = perez(1.0, thetaS, Ax, Bx, Cx, Dx, Ex);
    float denomy = perez(1.0, thetaS, Ay, By, Cy, Dy, Ey);

    float Y = Yz * perez(cosTheta, gamma, AY, BY, CY, DY, EY) / max(denomY, 1e-4);
    float x = xz * perez(cosTheta, gamma, Ax, Bx, Cx, Dx, Ex) / max(denomx, 1e-4);
    float y = yz * perez(cosTheta, gamma, Ay, By, Cy, Dy, Ey) / max(denomy, 1e-4);

    // kcd/m^2 into a range the tonemapper expects.
    //
    // This constant matters more than it looks. Preetham returns absolute
    // luminance, and feeding it in too hot drives every channel into the
    // Reinhard curve's saturation region, where they converge on each other
    // and the sky turns white - taking the chromaticity, which is the whole
    // point of the model, with it. The first value did exactly that: a sunset
    // rendered as grey-blue because the warm hue was being clipped away rather
    // than because the model had not produced one.
    vec3 col = xyY_to_linear_rgb(x, y, Y * g.params.w);

    // Preetham can dip slightly negative at extreme angles where the fit runs
    // out; clamping is honest about the model's domain rather than letting a
    // negative channel propagate into the water's reflection.
    return max(col, vec3(0.0));
}

vec3 sky_color(vec3 dir, vec3 sunDir)
{
    // Below the horizon the model is undefined. Mirroring rather than clamping
    // keeps the gradient continuous across y = 0, which matters because the
    // water reflects rays that legitimately point downward off a steep wave
    // face.
    vec3 d = vec3(dir.x, abs(dir.y), dir.z);
    vec3 col = sky_color_turbid(d, sunDir, g.params.z);

    // The sun itself. Preetham describes the sky's scattered light, not the
    // solar disc, so the disc and its bloom are added separately.
    float cosA = max(dot(dir, sunDir), 0.0);
    float disc  = pow(cosA, 5000.0) * 120.0;  // the disc, for sun glitter
    float bloom = pow(cosA, 60.0) * 0.20;     // aureole around it

    // The disc reddens as it sets, for the same reason the sky does: a low sun
    // is seen through far more atmosphere. Preetham models the SKY's scattered
    // light, not the direct beam, so this has to be applied here.
    float lowSun = 1.0 - clamp(sunDir.y * 3.0, 0.0, 1.0);
    vec3 sunTint = mix(vec3(1.0, 0.96, 0.90), vec3(1.0, 0.52, 0.22), lowSun);
    col += sunTint * (disc + bloom);

    return col;
}

// Reinhard tone map followed by gamma. The scene is rendered in linear light
// with an unbounded sun, so something has to bring it back into range; without
// a tone map the specular highlight clips to a flat white disc.
// ---------------------------------------------------------------------------
// Underwater
// ---------------------------------------------------------------------------

// Beer-Lambert extinction of clear sea water, per metre, per channel.
//
// Red is absorbed roughly ten times faster than blue. This one fact is most of
// why underwater footage looks the way it does: everything goes blue-green
// with distance, and a red object is grey by a few metres down. Faking it with
// a blue fog colour gets the hue but not the behaviour - real absorption
// changes the RATIO between channels with distance, so contrast collapses
// toward monochrome rather than toward a tint.
const vec3 kExtinction = vec3(0.42, 0.075, 0.035);

vec3 absorb(vec3 color, float pathMetres)
{
    return color * exp(-kExtinction * max(pathMetres, 0.0));
}

// Water refractive index, and the cosine of the critical angle measured from
// vertical: asin(1/1.333) = 48.6 degrees, so cos = 0.6614.
const float kWaterIor  = 1.333;
const float kCosCrit   = 0.6614;

// What a submerged viewer sees looking along `dir`.
//
// SNELL'S WINDOW is the striking part, and it is real rather than stylised:
// from below, the entire 180-degree sky is refracted into a cone of about 97
// degrees directly overhead. Look up inside that cone and you see the world
// above, squashed toward the edge; look outside it and the surface is a
// perfect mirror by total internal reflection, showing the water below.
vec3 underwater_background(vec3 dir, vec3 sunDir)
{
    vec3 deepColor = vec3(0.018, 0.075, 0.105);

    if (dir.y > kCosCrit) {
        // Inside the window. Refract back out to air: Snell gives
        // sin(air) = n * sin(water), and the angle is measured from vertical.
        float sinW = sqrt(max(0.0, 1.0 - dir.y * dir.y));
        float sinA = clamp(kWaterIor * sinW, 0.0, 1.0);
        float cosA = sqrt(max(0.0, 1.0 - sinA * sinA));
        vec3  horiz = vec3(dir.x, 0.0, dir.z);
        float hlen = length(horiz);
        horiz = (hlen > 1e-5) ? horiz / hlen : vec3(1.0, 0.0, 0.0);
        return sky_color(horiz * sinA + vec3(0.0, cosA, 0.0), sunDir);
    }

    // Outside the window: total internal reflection, so the surface mirrors
    // the water. Darkening downward is the light falling off with depth.
    float down = clamp(-dir.y * 0.5 + 0.5, 0.0, 1.0);
    return deepColor * mix(0.35, 1.0, down);
}

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
