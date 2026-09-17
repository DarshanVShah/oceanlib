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

    vec3 col = sky_color(dir, normalize(g.sunDir.xyz));
    outColor = vec4(tonemap(col, g.shading.y), 1.0);
}
