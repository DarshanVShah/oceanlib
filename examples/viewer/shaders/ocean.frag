#version 450

#include "common.glsl"

// Displacement textures, for their ALPHA channel only: that is where foam
// lives, and it has to be read per pixel rather than interpolated from the
// vertex shader.
layout(set = 0, binding = 1) uniform sampler2D uDisplacement0;
layout(set = 0, binding = 3) uniform sampler2D uDisplacement1;
layout(set = 0, binding = 5) uniform sampler2D uDisplacement2;

layout(set = 0, binding = 2) uniform sampler2D uNormal0;
layout(set = 0, binding = 4) uniform sampler2D uNormal1;
layout(set = 0, binding = 6) uniform sampler2D uNormal2;

layout(location = 0) in  vec3  vWorld;
layout(location = 1) in  vec2  vUV0;
layout(location = 2) in  vec2  vUV1;
layout(location = 3) in  vec2  vUV2;
layout(location = 4) in  float vFoam;
layout(location = 5) in  vec4  vInteraction;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 L = normalize(g.sunDir.xyz);
    vec3 V = normalize(g.camPos.xyz - vWorld);

    // Sum-then-renormalise: the same technique CascadeStack::sample_at uses
    // on the CPU side (ADR-020), applied here per-pixel instead of per-query.
    // This is the standard way multi-scale detail normals are combined in
    // real-time rendering - not exact (an exact answer needs the raw slope
    // gradients from every cascade recombined before deriving one normal,
    // which the public API does not expose - see ADR-005), but it is the
    // well-understood, visually correct approximation, and using the SAME
    // approximation here as the CPU query keeps rendering and physics
    // consistent with each other rather than just each being locally
    // plausible.
    vec3 n0 = texture(uNormal0, vUV0).xyz;
    vec3 n1 = texture(uNormal1, vUV1).xyz;
    vec3 n2 = texture(uNormal2, vUV2).xyz;
    vec3 N = normalize(n0 + n1 + n2);

    // Add the interaction field's slope, so the ripples CATCH THE LIGHT rather
    // than merely displacing the mesh.
    //
    // Composed through SLOPES, not by summing unit normals. The cascade sum
    // above has to blend normals because each cascade only exposes an already
    // normalised one (ADR-020); here we have a genuine height field, and
    // heights adding in world space means slopes add in world space - an
    // identity, not an approximation. This is the same arithmetic
    // WaterSurface::sample_at performs on the CPU, so shading and physics
    // agree rather than each being separately plausible.
    // Sampled HERE, per pixel, not taken from the interpolated vertex value.
    //
    // This matters more than it looks. The mesh is about 3 m per quad (256
    // quads across the far cascade's 800 m patch), while the interaction field
    // carries ripples of 0.5 to 4 m. Interpolating a per-vertex sample across
    // a 3 m quad aliases those ripples away completely - the field is there,
    // the texture is bound, and the water still looks flat. A per-pixel fetch
    // reads the field at its own resolution, which is what makes the ripples
    // catch the light rather than merely move the mesh.
    vec4 inter = interaction_sample(vWorld.xz);

    if (g.interaction.w > 0.5) {
        // Isolation mode: the surface IS the interaction field, so its normal
        // is the plain height-field normal.
        N = normalize(vec3(-inter.y, 1.0, -inter.z));
    } else if (N.y > 1e-6) {
        vec2 slope = vec2(-N.x / N.y + inter.y,
                          -N.z / N.y + inter.z);
        N = normalize(vec3(-slope.x, 1.0, -slope.y));
    }

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

    // Tint the subsurface term by the light actually falling on the water.
    //
    // Without this the water below the Fresnel crossover keeps a fixed
    // turquoise no matter what the sky is doing, so a sunset gives a warm sky
    // over cold cyan water and the image does not cohere. The scattered light
    // leaving the water is light that entered it, so it has to carry the
    // sky's colour. Sampled straight up, which is where most of the downwelling
    // light comes from, and normalised against its own luminance so this
    // changes HUE without also changing how bright the water is.
    vec3 ambient = sky_color_turbid(vec3(0.0, 1.0, 0.0), L, g.params.z);
    ambient /= max(dot(ambient, vec3(0.2126, 0.7152, 0.0722)), 1e-3);

    vec3 refraction = (deep + scatter * lift * through * 1.6) * ambient;

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
    //
    // Sampled PER PIXEL, not taken from the interpolated vFoam the vertex
    // shader computes. The mesh is about 3 m per quad while foam structure is
    // metres or less across, so interpolating a per-vertex sample smears sharp
    // whitewater into broad pale washes - the water goes milky instead of
    // getting foam on it. Exactly the same mistake, and the same fix, as the
    // interaction field's normals above.
    //
    // The screen blend 1 - product(1 - f_i) is the same one
    // CascadeStack::sample_at uses on the CPU, so a physics query and a pixel
    // agree about how much foam is at a point.
    float f0 = texture(uDisplacement0, vUV0).w;
    float f1 = texture(uDisplacement1, vUV1).w;
    float f2 = texture(uDisplacement2, vUV2).w;
    float coverage = 1.0 - (1.0 - f0) * (1.0 - f1) * (1.0 - f2);

    // A gentle contrast curve. Advected foam has a long thin tail as it decays,
    // and mapping that tail linearly to opacity turns half the ocean hazy;
    // pushing the low end down keeps the faint remnants as texture and lets
    // fresh whitewater read as actually white.
    float foam = clamp(coverage * g.shading.x, 0.0, 1.0);
    foam = foam * foam * (3.0 - 2.0 * foam);

    // Foam is a diffuse, near-Lambertian scatterer - it has no specular
    // highlight of its own, which is why it reads as matte against the water
    // it sits on.
    vec3 foamColor = vec3(0.94, 0.96, 0.98) * (0.55 + 0.45 * max(dot(N, L), 0.0));
    color = mix(color, foamColor, foam);

    // --- atmospheric fade -------------------------------------------------
    // Fades the ocean into the sky at distance, which both looks right and
    // hides the outer edge of the tiled patch.
    float dist = length(g.camPos.xyz - vWorld);
    float fog  = 1.0 - exp(-dist * g.shading.z);
    color = mix(color, sky_color(normalize(vWorld - g.camPos.xyz), L), fog);

    outColor = vec4(tonemap(color, g.shading.y), 1.0);
}
