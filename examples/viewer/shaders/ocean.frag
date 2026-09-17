#version 450

#include "common.glsl"

layout(set = 0, binding = 2) uniform sampler2D uNormal;

layout(location = 0) in  vec3  vWorld;
layout(location = 1) in  vec2  vUV;
layout(location = 2) in  float vFoam;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 L = normalize(g.sunDir.xyz);
    vec3 V = normalize(g.camPos.xyz - vWorld);

    // The normal comes straight from the library. It is the exact normal of
    // the DISPLACED surface - derived from the same displacement gradients the
    // foam term uses - not a heightfield approximation, so the shading agrees
    // with the geometry even where the chop has dragged vertices sideways.
    vec3 N = normalize(texture(uNormal, vUV).xyz);

    // Backfacing normals occur where the surface has folded (jacobian < 0).
    // Flipping rather than discarding keeps breaking crests lit instead of
    // punching black holes in them.
    if (dot(N, V) < 0.0) N = reflect(N, V);

    // --- reflection ------------------------------------------------------
    vec3 R = reflect(-V, N);
    R.y = abs(R.y);  // never sample the sky from below the horizon
    vec3 reflection = sky_color(R, L);

    // --- Fresnel ---------------------------------------------------------
    // Schlick, with F0 = 0.02 for an air/water interface (IOR 1.33). This one
    // constant is most of why water looks like water: nearly transparent
    // looking straight down, a mirror at grazing angles.
    float cosTheta = max(dot(N, V), 0.0);
    float F = 0.02 + 0.98 * pow(1.0 - cosTheta, 5.0);

    // --- refraction / subsurface ------------------------------------------
    // Cheap stand-in for subsurface scattering: wave crests glow when you are
    // looking through them toward the sun. Keyed on height above the mean
    // surface and on how much the view direction opposes the light.
    float lift    = clamp(vWorld.y * 0.45 + 0.25, 0.0, 1.0);
    float through = pow(clamp(dot(-V, L) * 0.5 + 0.5, 0.0, 1.0), 3.0);

    vec3 deep    = vec3(0.004, 0.036, 0.072);
    vec3 scatter = vec3(0.045, 0.30, 0.26);
    vec3 refraction = deep + scatter * lift * through * 1.6;

    // --- sun specular (GGX) ----------------------------------------------
    vec3  H     = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float rough = 0.055;
    float a     = rough * rough;
    float denom = NdotH * NdotH * (a * a - 1.0) + 1.0;
    float D     = (a * a) / (kPi * denom * denom);
    vec3  spec  = vec3(1.0, 0.95, 0.85) * D * 0.35 * max(dot(N, L), 0.0);

    vec3 color = mix(refraction, reflection, F) + spec;

    // --- foam -------------------------------------------------------------
    // The library's foam term is the Jacobian of the horizontal displacement:
    // it is the actual measure of the surface folding onto itself, which is
    // physically where whitecaps form - not a height threshold.
    float foam = clamp(vFoam * g.shading.x, 0.0, 1.0);
    vec3  foamColor = vec3(0.86, 0.91, 0.95) * (0.45 + 0.55 * max(dot(N, L), 0.0));
    color = mix(color, foamColor, foam);

    // --- atmospheric fade -------------------------------------------------
    // Fades the ocean into the sky at distance, which both looks right and
    // hides the outer edge of the tiled patch.
    float dist = length(g.camPos.xyz - vWorld);
    float fog  = 1.0 - exp(-dist * g.shading.z);
    color = mix(color, sky_color(normalize(vWorld - g.camPos.xyz), L), fog);

    outColor = vec4(tonemap(color, g.shading.y), 1.0);
}
