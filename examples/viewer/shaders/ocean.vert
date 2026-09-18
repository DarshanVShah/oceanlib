#version 450

#include "common.glsl"

// A flat unit grid, tiled to cover cascade 0's (the far/largest scale's) own
// patch. The mesh itself carries no per-cascade information - all three
// scales are sampled at the SAME world (x,z), each through its own patch's
// wrapped UV, exactly the "simple additive sum, no distance weighting"
// decision in ADR-020: cascades are superposed frequency bands of the same
// real surface, so summing samples is the mathematically correct thing to
// do, not a heuristic blend.
layout(location = 0) in vec2 inGrid;  // [0,1] x [0,1]

layout(set = 0, binding = 1) uniform sampler2D uDisplacement0;
layout(set = 0, binding = 3) uniform sampler2D uDisplacement1;
layout(set = 0, binding = 5) uniform sampler2D uDisplacement2;

layout(location = 0) out vec3  vWorld;
layout(location = 1) out vec2  vUV0;
layout(location = 2) out vec2  vUV1;
layout(location = 3) out vec2  vUV2;
layout(location = 4) out float vFoam;
layout(location = 5) out vec4  vInteraction;

void main()
{
    // Note: `patch` is a reserved word in GLSL (tessellation), hence patchLen.
    float patchLen0 = g.cascadePatch.x;
    float patchLen1 = g.cascadePatch.y;
    float patchLen2 = g.cascadePatch.z;
    int   tiles     = int(g.cascadePatch.w);

    // The outer mesh tiles cascade 0's patch, centred on the origin - the
    // same instancing scheme a single-cascade ocean already used, just fixed
    // to the largest scale so the visible area is set by the far cascade.
    int ti = gl_InstanceIndex;
    int tx = (ti % tiles) - (tiles / 2);
    int tz = (ti / tiles) - (tiles / 2);

    vec2 base = (inGrid + vec2(float(tx), float(tz))) * patchLen0;

    // Each cascade's own UV is world position divided by ITS OWN patch length.
    // A REPEAT sampler makes this correct rather than approximate for cascade
    // 1 and 2 even when |base| exceeds their (smaller) patch many times over:
    // the GPU's wrap hardware performs the exact modulo, so uv = 37.4 reads
    // back the exact same texel as uv = 0.4 would, because that cascade's
    // field really is periodic with that period.
    vec2 uv0 = base / patchLen0;
    vec2 uv1 = base / patchLen1;
    vec2 uv2 = base / patchLen2;

    // textureLod, not texture: a vertex shader has no derivatives, so an
    // implicit-LOD sample would be undefined here.
    vec4 d0 = textureLod(uDisplacement0, uv0, 0.0);
    vec4 d1 = textureLod(uDisplacement1, uv1, 0.0);
    vec4 d2 = textureLod(uDisplacement2, uv2, 0.0);

    // Height and horizontal offset: exact linear superposition (ADR-020) -
    // each cascade's displacement is a real field over world position, so
    // summing them is not an approximation.
    vec3 dSum = d0.xyz + d1.xyz + d2.xyz;

    // The interaction field is a pure HEIGHT field - no horizontal
    // displacement - so it is sampled at the world position the vertex
    // actually lands at, after the cascades have moved it sideways. Sampling
    // at the undisplaced position would smear the ripples wherever choppiness
    // is doing anything.
    vec2 worldXZ = vec2(base.x + dSum.x, base.y + dSum.z);
    vec4 inter = interaction_sample(worldXZ);
    vInteraction = inter;

    // Isolation mode: show ONLY the interaction field, on flat water, so the
    // ripples can be read without the swell moving underneath them.
    if (g.interaction.w > 0.5) {
        vWorld = vec3(base.x, inter.x, base.y);
        vUV0 = uv0; vUV1 = uv1; vUV2 = uv2;
        vFoam = 0.0;
        gl_Position = g.viewProj * vec4(vWorld, 1.0);
        return;
    }

    vWorld = vec3(base.x + dSum.x, dSum.y + inter.x, base.y + dSum.z);
    vUV0   = uv0;
    vUV1   = uv1;
    vUV2   = uv2;

    // Foam: same "screen"/OR blend as CascadeStack::sample_at on the CPU side
    // (1 - product(1-foam_i)), kept identical so the GPU-rendered foam and a
    // physics query at the same point agree in character, not just height.
    vFoam = 1.0 - (1.0 - d0.w) * (1.0 - d1.w) * (1.0 - d2.w);

    gl_Position = g.viewProj * vec4(vWorld, 1.0);
}
