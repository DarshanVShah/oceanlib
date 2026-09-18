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

    // --- fine-cascade subpixel fade -----------------------------------------
    //
    // "Subpixel" means exactly that: a cascade's texel projects to less than
    // one screen pixel. Past that point its high-frequency content cannot be
    // resolved regardless of what the texture holds, so sampling and summing
    // it buys nothing but bandwidth and a risk of shimmer from an
    // undersampled signal - fade it to zero smoothly instead. Cascade 0 (the
    // big swell) never fades - its texels are the largest of the three, so
    // it is never the one going subpixel first.
    //
    // A first version of this drove the fade from ring.cellSize (mesh
    // resolution) and, separately, from an arbitrary multiple of texel size -
    // both were wrong: the first made a cascade's contribution jump at every
    // ring boundary (fixed by switching to distance, see the ADR), and the
    // second was never checked against a real image and turned out to fade
    // the near cascade out by 15 m from the camera, visibly flattening the
    // water (see BENCHMARKS.md / ADR-021's account of the regression). This
    // version instead computes the actual world-space footprint of one
    // screen pixel at this vertex's distance - the standard perspective
    // relationship, footprint = 2 * distance * tan(fovY/2) / viewportHeight -
    // and compares each cascade's OWN texel size against it directly. That
    // is what "subpixel" literally means, so there is no arbitrary constant
    // left to get wrong: get the projection math right and the threshold
    // falls out of it.
    float distFromCam   = length(base - g.camPos.xz);
    float pixelFootprint = 2.0 * distFromCam * tan(g.params.y * 0.5) / g.params.z;
    float texel1 = patchLen1 / cascadeN;
    float texel2 = patchLen2 / cascadeN;
    // Fade over a 4x range of the texel-to-pixel ratio, centred where a
    // texel and a pixel are the same size (ratio 1): comfortably resolved
    // (ratio >= 2, a texel spans at least 2 pixels) down to clearly aliasing
    // (ratio <= 0.5, at least 2 texels landing in one pixel).
    vFade1 = smoothstep(0.5, 2.0, texel1 / pixelFootprint);
    vFade2 = smoothstep(0.5, 2.0, texel2 / pixelFootprint);

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
