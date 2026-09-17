#version 450

#include "common.glsl"

// A flat unit grid. The displacement comes entirely from the texture, which is
// the point: the library's output buffers were laid out to BE an RGBA32F
// texture, so integration is two uploads and a sampler.
layout(location = 0) in vec2 inGrid;  // [0,1] x [0,1]

layout(set = 0, binding = 1) uniform sampler2D uDisplacement;

layout(location = 0) out vec3  vWorld;
layout(location = 1) out vec2  vUV;
layout(location = 2) out float vFoam;

void main()
{
    // Note: `patch` is a reserved word in GLSL (tessellation), hence patchLen.
    float patchLen = g.params.x;
    int   tiles    = int(g.params.y);

    // Each instance is one tile of the periodic patch, laid out in a square
    // centred on the origin. The FFT surface is exactly periodic, so tiles
    // join seamlessly with no blending, overlap or edge fixup.
    int ti = gl_InstanceIndex;
    int tx = (ti % tiles) - (tiles / 2);
    int tz = (ti / tiles) - (tiles / 2);

    vec2 base = (inGrid + vec2(float(tx), float(tz))) * patchLen;

    // textureLod, not texture: a vertex shader has no derivatives, so an
    // implicit-LOD sample would be undefined here.
    vec4 d = textureLod(uDisplacement, inGrid, 0.0);

    // d.xz is the horizontal chop, d.y the height, d.w the foam coverage.
    // Exactly the layout documented in ocean.h - no unpacking, no swizzle
    // table, no per-engine convention to remember.
    vWorld = vec3(base.x + d.x, d.y, base.y + d.z);
    vUV    = inGrid;
    vFoam  = d.w;

    gl_Position = g.viewProj * vec4(vWorld, 1.0);
}
