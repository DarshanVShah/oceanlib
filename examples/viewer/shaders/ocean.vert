#version 450

#include "common.glsl"

// A clipmap ring's local grid (ADR-021): integers in [-kRingHalf, kRingHalf]
// along each axis (kRingHalf must match clipmap.hpp's kRingHalf exactly -
// this is the one place that invariant has to be kept by hand, since GLSL
// cannot #include a C++ header). World position is local * cellSize + offset,
// both pushed per ring draw - the direct generalisation of the single-tile
// mesh's old "unit grid * patchLen0 + tile offset".
const int kRingHalf = 32;

layout(push_constant) uniform RingPush {
    vec2  offset;      // this ring's world-space centre
    float cellSize;    // world metres per local unit at this ring's level
    float morphStart;  // fraction of kRingHalf where the geomorph blend begins
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
layout(location = 5) out float vFade1;
layout(location = 6) out float vFade2;

void main()
{
    float patchLen0 = g.cascadePatch.x;
    float patchLen1 = g.cascadePatch.y;
    float patchLen2 = g.cascadePatch.z;
    float cascadeN  = g.cascadePatch.w;

    // --- geomorphing (ADR-021) ---------------------------------------------
    //
    // Near this ring's OUTER boundary it hands off to the next COARSER ring
    // (ring i's own footprint edge is exactly ring (i+1)'s hole boundary by
    // construction), so blend this ring's sample position toward the
    // position its coarser sibling would use there, over the outer
    // (1 - morphStart) fraction of the ring's own half-extent. Chebyshev
    // distance, not Euclidean, because the ring is a square, not a circle.
    float dist  = max(abs(inLocal.x), abs(inLocal.y)) / float(kRingHalf);
    float blend = smoothstep(ring.morphStart, 1.0, dist);
    // "What the coarser ring would sample here": its cells are twice as
    // wide, so its nearest grid line is the local position rounded to the
    // nearest even integer.
    vec2 coarseLocal = round(inLocal * 0.5) * 2.0;
    vec2 morphed = mix(inLocal, coarseLocal, blend);

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

    // --- fine-cascade distance fade -----------------------------------------
    //
    // Far enough out, the local mesh resolution is already several times
    // coarser than a cascade's own texel size, so that cascade's high-
    // frequency content is being undersampled by the MESH regardless of
    // what the texture holds - the triangle spans many texels, so the fine
    // ripples it carries are already averaged away geometrically. Sampling
    // and summing it at that point buys nothing but bandwidth and a risk of
    // shimmer from an undersampled signal, so fade it to zero smoothly.
    // Cascade 0 (the big swell) never fades - it is always the coarsest
    // scale present, so it is never the one being oversampled by the mesh.
    //
    // Driven by world-space distance from the camera, NOT by ring.cellSize:
    // cellSize is a per-ring CONSTANT that doubles discretely at every ring
    // boundary, so fading on it would make a cascade's contribution jump
    // between two different values exactly at that boundary - a real height
    // discontinuity between the last vertex of one ring and the first of the
    // next, independent of how well the mesh itself is stitched. Distance is
    // continuous across the whole clipmap, including at ring seams, so this
    // has no jump to begin with.
    float distFromCam = length(base - g.camPos.xz);
    float texel1 = patchLen1 / cascadeN;
    float texel2 = patchLen2 / cascadeN;
    vFade1 = 1.0 - smoothstep(texel1 * 40.0, texel1 * 160.0, distFromCam);
    vFade2 = 1.0 - smoothstep(texel2 * 40.0, texel2 * 160.0, distFromCam);

    // textureLod, not texture: a vertex shader has no derivatives, so an
    // implicit-LOD sample would be undefined here.
    vec4 d0 = textureLod(uDisplacement0, uv0, 0.0);
    vec4 d1 = textureLod(uDisplacement1, uv1, 0.0) * vFade1;
    vec4 d2 = textureLod(uDisplacement2, uv2, 0.0) * vFade2;

    // Height and horizontal offset: exact linear superposition (ADR-020) -
    // each cascade's displacement is a real field over world position, so
    // summing them is not an approximation. The fade above scales an entire
    // cascade's contribution uniformly, so this sum is still an exact
    // superposition of whichever cascades are actually contributing.
    vec3 dSum = d0.xyz + d1.xyz + d2.xyz;

    vWorld = vec3(base.x + dSum.x, dSum.y, base.y + dSum.z);
    vUV0   = uv0;
    vUV1   = uv1;
    vUV2   = uv2;

    // Foam: same "screen"/OR blend as CascadeStack::sample_at on the CPU side
    // (1 - product(1-foam_i)), kept identical so the GPU-rendered foam and a
    // physics query at the same point agree in character, not just height.
    // A faded-out cascade's foam term is scaled by the same vFadeN, so it
    // drops out of the product exactly as its displacement already did.
    vFoam = 1.0 - (1.0 - d0.w) * (1.0 - d1.w) * (1.0 - d2.w);

    gl_Position = g.viewProj * vec4(vWorld, 1.0);
}
