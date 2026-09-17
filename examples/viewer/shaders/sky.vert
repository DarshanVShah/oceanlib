#version 450

// Fullscreen triangle with no vertex buffer: three vertices generated from
// gl_VertexIndex. A triangle rather than a quad because the quad's diagonal
// would make the GPU shade the seam twice.
layout(location = 0) out vec2 vNdc;

void main()
{
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vNdc = p * 2.0 - 1.0;
    // z = 1 puts it on the far plane, so the ocean always wins the depth test.
    gl_Position = vec4(vNdc, 1.0, 1.0);
}
