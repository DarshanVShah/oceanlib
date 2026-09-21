#version 450

// A fullscreen triangle, not a quad.
//
// Three vertices covering the screen beat two triangles covering it exactly:
// there is no shared edge, so no pixels along a diagonal seam get shaded twice
// by neighbouring quads, and the rasteriser walks one primitive instead of
// two. The vertices sit outside the viewport and are clipped, which costs
// nothing.
//
// No vertex buffer: positions come from gl_VertexIndex, the same trick sky.vert
// already uses.

layout(location = 0) out vec2 vUV;

void main()
{
    //  index 0 -> (-1,-1)   index 1 -> (3,-1)   index 2 -> (-1,3)
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vUV = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
