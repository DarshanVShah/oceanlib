#version 450

#include "common.glsl"

// Must match viewer::PropInstance field for field, up to `mesh` - the viewer
// pushes the struct's own bytes (ocean_view.cpp's kPropPushSize is literally
// offsetof(PropInstance, mesh)).
layout(push_constant) uniform Push {
    mat4 model;
    vec4 scale;
    vec4 color;   // rgb albedo, a = translucency (light passed through from behind)
    vec4 surf;    // x = water height at the model origin, yz = dEta/dx, dEta/dz,
                  // w = wet-band softness in metres; <= 0 disables the band
} pc;

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 L = normalize(g.sunDir.xyz);
    vec3 V = normalize(g.camPos.xyz - vWorld);

    // Props are drawn with culling off, so both sides of every surface reach
    // here and the back ones arrive with their normal pointing away. Flip on
    // gl_FrontFacing, which is the winding the rasteriser actually saw.
    //
    // Testing dot(N, V) < 0 instead - which is what this used to do - is not
    // the same thing and is wrong on curved geometry: an INTERPOLATED normal
    // on a smooth hull tips past 90 degrees from the eye well before the true
    // silhouette, so genuine front faces near the edge got flipped and shaded
    // inside-out. That was the dark rim around the hull.
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;

    float ndl = max(dot(N, L), 0.0);

    // Thin fabric passes light. Without this a sail is a flat silhouette the
    // instant the sun goes behind it, which is the one lighting condition a
    // sail is most recognisable in. Lambertian on the REVERSED normal is the
    // cheapest honest stand-in for transmission through a thin sheet: no
    // scattering depth, no thickness, just light arriving from the far side.
    float trans = max(dot(-N, L), 0.0) * pc.color.a;

    // Ambient straight from the sky model, biased upward. Cheap image-based
    // lighting, and it is what keeps a prop sitting in the SAME light as the
    // water - a boat lit by a hardcoded constant reads as pasted on, most
    // obviously at sunset when the water has gone warm and the hull has not.
    vec3 ambient = sky_color_turbid(normalize(N + vec3(0.0, 0.9, 0.0)), L,
                                    g.params.z) * 1.1;

    // How wet this fragment is, from the local water plane. The plane is
    // fitted from the same four probes that float the hull, so it tilts with
    // the wave the boat is on rather than cutting it dead level - on a 9 m
    // hull in a real swell that difference is a good fraction of the
    // freeboard. Softness <= 0 means "not a floating thing", so no band.
    //
    // The band runs UPWARD from the waterline, not down. Everything below the
    // line is hidden behind the ocean surface itself, so wetting only what is
    // submerged changes nothing anyone can see; what is visible is the strip
    // just ABOVE the line that the sea keeps washing over and that has not
    // dried yet. That strip is the cue that reads as "floating in" rather than
    // "resting on", and it is the one a hard geometric intersection lacks.
    float wet = 0.0;
    if (pc.surf.w > 0.0) {
        float waterY = pc.surf.x + pc.surf.y * (vWorld.x - pc.model[3].x)
                                 + pc.surf.z * (vWorld.z - pc.model[3].z);
        wet = 1.0 - smoothstep(0.0, pc.surf.w, vWorld.y - waterY);
    }

    // Soaked wood is darker and glossier than dry wood - the water fills the
    // surface roughness, so less light scatters back out and more reflects.
    vec3  albedo = pc.color.rgb * mix(1.0, 0.40, wet);
    float gloss  = mix(40.0, 180.0, wet);
    float spec   = mix(0.10, 0.45, wet);

    vec3 col = albedo * (vec3(1.0, 0.95, 0.86) * (ndl + trans) + ambient);

    vec3 H = normalize(L + V);
    col += vec3(1.0) * pow(max(dot(N, H), 0.0), gloss) * spec * ndl;

    float dist = length(g.camPos.xyz - vWorld);
    if (g.water.x > 0.0) {
        col = absorb(col, dist);
    } else {
        float fog = 1.0 - exp(-dist * g.shading.z);
        col = mix(col, sky_color(normalize(vWorld - g.camPos.xyz), L), fog);
    }
    outColor = vec4(tonemap(col, g.shading.y), 1.0);
}
