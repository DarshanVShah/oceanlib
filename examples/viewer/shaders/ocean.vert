#version 450

#include "common.glsl"

// One ring of a geometry clipmap (ADR-025). The mesh carries no per-cascade
// information - all three scales are sampled at the SAME world (x,z), each
// through its own patch's wrapped UV, exactly the "simple additive sum, no
// distance weighting" decision in ADR-020: cascades are superposed frequency
// bands of the same real surface, so summing samples is the mathematically
// correct thing to do, not a heuristic blend.
//
// LOD and cascade scale are separate axes and must stay that way. A cascade is
// a band of WAVELENGTHS; a ring is a band of DISTANCES. The old mesh conflated
// them by tiling cascade 0's patch, which is why it drew 6.42M triangles to
// cover a view whose far half was sub-pixel.
//
// Local coordinates are integers in [-kRingHalf, kRingHalf] on each axis, and
// kRingHalf must match clipmap.hpp exactly - the one invariant here kept by
// hand, because GLSL cannot include a C++ header.
const int kRingHalf = 32;

layout(push_constant) uniform RingPush {
    vec2  offset;      // this ring's snapped world-space centre
    float cellSize;    // world metres per local unit at this ring's level
    float morphStart;  // fraction of kRingHalf where the geomorph begins
} ring;

layout(location = 0) in vec2 inLocal;  // this ring's own local (x, z)

layout(set = 0, binding = 1) uniform sampler2D uDisplacement0;
layout(set = 0, binding = 3) uniform sampler2D uDisplacement1;
layout(set = 0, binding = 5) uniform sampler2D uDisplacement2;

layout(location = 0) out vec3  vWorld;
layout(location = 1) out vec2  vUV0;
layout(location = 2) out vec2  vUV1;
layout(location = 3) out vec2  vUV2;
layout(location = 4) out float vFoam;
layout(location = 5) out vec4  vInteraction;
layout(location = 6) out vec3  vFade;

void main()
{
    // Note: `patch` is a reserved word in GLSL (tessellation), hence patchLen.
    float patchLen0 = g.cascadePatch.x;
    float patchLen1 = g.cascadePatch.y;
    float patchLen2 = g.cascadePatch.z;

    // --- geomorphing (ADR-025) ---------------------------------------------
    //
    // Near this ring's OUTER boundary it hands off to the next COARSER ring -
    // ring i's footprint edge is exactly ring (i+1)'s hole boundary by
    // construction - so blend this ring's sample position toward the position
    // its coarser sibling would use there, across the outer (1 - morphStart)
    // fraction of the ring's own half-extent.
    //
    // Chebyshev distance, not Euclidean, because a ring is a square. Using
    // length() would make the morph band a circle inscribed in a square
    // boundary, so the corners would reach the edge un-morphed and pop.
    float ringDist = max(abs(inLocal.x), abs(inLocal.y)) / float(kRingHalf);
    float blend    = smoothstep(ring.morphStart, 1.0, ringDist);

    // Where the coarser ring would sample: its cells are twice as wide, so its
    // nearest grid line is this local position rounded to the nearest even
    // integer. At blend = 1 the odd vertices land exactly on their even
    // neighbours, the triangles between them go degenerate, and the boundary
    // becomes the same straight segment the coarser ring draws - which is what
    // removes the T-junction crack rather than hiding it behind a skirt.
    vec2 coarseLocal = round(inLocal * 0.5) * 2.0;
    vec2 morphed     = mix(inLocal, coarseLocal, blend);

    vec2 base = ring.offset + morphed * ring.cellSize;

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

    // Distance-based detail fade. Weighted BEFORE the sum, because the
    // cascades are a frequency decomposition and fading one out is dropping
    // its band - which is what a low-pass filter would do to geometry too
    // small to draw.
    //
    // Distance is measured to the undisplaced position: using the displaced
    // one would make the weight depend on the displacement it is about to
    // scale, which is circular and pops as waves move.
    float dist = length(g.camPos.xyz - vec3(base.x, 0.0, base.y));
    vFade = vec3(detail_fade(dist, g.cascadeTexel.x),
                 detail_fade(dist, g.cascadeTexel.y),
                 detail_fade(dist, g.cascadeTexel.z));

    // Height and horizontal offset: exact linear superposition (ADR-020) -
    // each cascade's displacement is a real field over world position, so
    // summing them is not an approximation.
    vec3 dSum = d0.xyz * vFade.x + d1.xyz * vFade.y + d2.xyz * vFade.z;

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
        vFade = vec3(1.0);
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
    vFoam = 1.0 - (1.0 - d0.w * vFade.x) * (1.0 - d1.w * vFade.y) *
                  (1.0 - d2.w * vFade.z);

    gl_Position = g.viewProj * vec4(vWorld, 1.0);
}
