#version 450

#include "common.glsl"

layout(location = 0) in  vec2 vNdc;
layout(location = 0) out vec4 outColor;

void main()
{
    // Unproject two points down the same pixel ray and subtract. Doing it this
    // way means the shader never has to know how the projection matrix was
    // built - near/far planes, reversed depth, aspect - it just works.
    vec4 nearP = g.invViewProj * vec4(vNdc, 0.0, 1.0);
    vec4 farP  = g.invViewProj * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(farP.xyz / farP.w - nearP.xyz / nearP.w);

    vec3 L = normalize(g.sunDir.xyz);

    // Submerged: the background is no longer sky at all. Everything the viewer
    // sees past the water surface has travelled through water to reach them,
    // so it is absorbed as well as refracted.
    vec3 col;
    if (g.water.x > 0.0) {
        col = underwater_background(dir, L);
        // Looking down, the path is effectively unbounded; looking up it is
        // roughly the depth divided by how steeply you are looking. This is a
        // slab approximation, not a volume integral, which is the right level
        // of effort for a background with no geometry in it.
        float path = (dir.y > 0.01) ? (g.water.x / dir.y) : 60.0;
        col = absorb(col, min(path, 60.0));
    } else {
        col = sky_color(dir, L);
    }
    outColor = vec4(tonemap(col, g.shading.y), 1.0);
}
