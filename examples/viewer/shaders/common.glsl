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
    vec4 params;       // x = time, y = mesh resolution, z,w unused
    vec4 shading;      // x foam strength, y exposure, z fog density, w unused
} g;

const float kPi = 3.14159265359;

// Analytic sky. Cheap, and good enough that the reflection sells the water.
vec3 sky_color(vec3 dir, vec3 sunDir)
{
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    vec3 horizon = vec3(0.62, 0.72, 0.84);
    vec3 zenith  = vec3(0.14, 0.32, 0.68);
    vec3 col = mix(horizon, zenith, pow(up, 0.55));

    // Sun: a tight disc for the specular highlight plus a broad glow that
    // gives the haze around it.
    float cosA = max(dot(dir, sunDir), 0.0);
    col += vec3(1.0, 0.92, 0.74) * (pow(cosA, 900.0) * 90.0 + pow(cosA, 12.0) * 0.30);

    // Thicken the haze near the horizon so the ocean can fade into it and the
    // edge of the tiled patch never reads as a hard boundary.
    col = mix(col, vec3(0.80, 0.84, 0.88), pow(1.0 - abs(dir.y), 8.0) * 0.75);

    return col;
}

// Reinhard tone map followed by gamma. The scene is rendered in linear light
// with an unbounded sun, so something has to bring it back into range; without
// a tone map the specular highlight clips to a flat white disc.
vec3 tonemap(vec3 linear, float exposure)
{
    vec3 c = linear * exposure;
    c = c / (c + vec3(1.0));
    return pow(c, vec3(1.0 / 2.2));
}

#endif
