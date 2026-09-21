// Solid props drawn on top of the ocean: a boat that floats, and rocks that
// are thrown into the water.
//
// The rock is a faceted sphere generated procedurally. The boat's hull, mast,
// rigging and sail are a real model (assets/gislinge_viking_boat.glb), baked
// at build time by tools/gltf_bake.cpp into the plain vertex/index arrays
// `boat_mesh.h` declares - so the demo still ships no asset file to find at
// launch and still launches from any working directory (ADR-015's reasoning
// for embedding the shaders, applied to a real mesh instead of a procedural
// one). The box primitive remains for anything that does not need a sculpted
// shape.
//
// The point of the props is not the rendering. It is that a boat floating
// correctly is the most direct demonstration of promise #2 there is: the hull
// sits on the water the renderer actually drew, because both come from the same
// query. A boat that visibly hovers or sinks into a crest is what a naive
// height lookup gets you.
#pragma once

#include "boat_mesh.h"
#include "vk_math.hpp"

#include <cmath>
#include <cstdint>
#include <iterator>
#include <vector>

namespace viewer {

struct PropVertex {
    float px, py, pz;
    float nx, ny, nz;
};

enum class PropMesh : std::uint32_t {
    Box      = 0,   // generic box, unused by the boat but kept for reuse
    Rock     = 1,   // a thrown rock
    BoatHull = 2,   // hull, mast, rigging, rudder - everything but the sail
    BoatSail = 3,
    Count
};

// One thing to draw. Scale is kept OUT of the matrix so the shader can build a
// correct normal transform: for a diagonal scale the inverse transpose is just
// the reciprocal, which is far cheaper than passing a second matrix and far
// less error-prone than forgetting to and shading a stretched box wrong.
//
// Everything up to (not including) `mesh` is pushed to the shader verbatim as
// one push-constant block - see ocean_view.cpp's kPropPushSize, which is
// literally offsetof(PropInstance, mesh). Adding a field here therefore adds
// it to the shader's `Push` too, and the two must stay in the same order.
struct PropInstance {
    vkm::Mat4 model{};        // rotation and translation only
    float     scale[4]{1, 1, 1, 0};

    // rgb = albedo. a = TRANSLUCENCY: how much light passes through the
    // surface from behind, for thin fabric. 0 for anything solid.
    float     color[4]{1, 1, 1, 0};

    // The local water plane, in the prop's own frame, for the wet band below
    // the waterline:
    //   x    = water height at the model origin (world Y)
    //   y, z = dEta/dx, dEta/dz, so the band tilts with the wave rather than
    //          cutting the hull dead level while it rides a slope
    //   w    = softness of the band edge in metres; <= 0 disables it entirely,
    //          which is what anything not floating (a rock in flight) wants
    float     surf[4]{0, 0, 0, 0};

    PropMesh  mesh = PropMesh::Box;
};

// A CPU mesh plus where it lives in the shared index buffer.
struct PropMeshRange {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t  vertex_base = 0;
};

// --- generation ------------------------------------------------------------

// Unit box spanning [-1, 1] on every axis, with FLAT normals - 24 vertices
// rather than 8, because a shared corner cannot carry three different face
// normals and smoothing them would round off the very edges that make a box
// read as a box.
inline void append_box(std::vector<PropVertex>& vtx,
                       std::vector<std::uint32_t>& idx, PropMeshRange& range)
{
    range.first_index = static_cast<std::uint32_t>(idx.size());
    range.vertex_base = static_cast<std::int32_t>(vtx.size());

    const float n[6][3] = {{ 1, 0, 0}, {-1, 0, 0}, {0,  1, 0},
                           {0, -1, 0}, { 0, 0, 1}, {0,  0,-1}};
    // Two in-plane axes per face, chosen so the winding comes out
    // counter-clockwise when seen from outside.
    const float u[6][3] = {{0, 0,-1}, {0, 0, 1}, {1, 0, 0},
                           {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}};
    const float v[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, 1},
                           {0, 0,-1}, {0, 1, 0}, {0, 1, 0}};

    for (int f = 0; f < 6; ++f) {
        const std::uint32_t base = static_cast<std::uint32_t>(vtx.size()) -
                                   static_cast<std::uint32_t>(range.vertex_base);
        for (int c = 0; c < 4; ++c) {
            const float su = (c == 0 || c == 3) ? -1.0f : 1.0f;
            const float sv = (c < 2) ? -1.0f : 1.0f;
            PropVertex p{};
            p.px = n[f][0] + u[f][0] * su + v[f][0] * sv;
            p.py = n[f][1] + u[f][1] * su + v[f][1] * sv;
            p.pz = n[f][2] + u[f][2] * su + v[f][2] * sv;
            p.nx = n[f][0]; p.ny = n[f][1]; p.nz = n[f][2];
            vtx.push_back(p);
        }
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
    range.index_count = static_cast<std::uint32_t>(idx.size()) - range.first_index;
}

// A rock: an icosahedron with each vertex pushed in or out a little, and FLAT
// normals so it reads as chipped stone rather than a ball. Twenty faces is
// enough - at the size a thrown rock appears on screen, more would be invisible.
//
// The perturbation is a deterministic hash of the vertex index, not a random
// number, so every rock in every run is the same rock. That keeps --screenshot
// reproducible, which is the same reason the scripted splash exists.
inline void append_rock(std::vector<PropVertex>& vtx,
                        std::vector<std::uint32_t>& idx, PropMeshRange& range)
{
    range.first_index = static_cast<std::uint32_t>(idx.size());
    range.vertex_base = static_cast<std::int32_t>(vtx.size());

    const float t = 1.6180339887f;
    float base[12][3] = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}};
    const int faces[20][3] = {
        {0,11,5}, {0,5,1},  {0,1,7},   {0,7,10},  {0,10,11},
        {1,5,9},  {5,11,4}, {11,10,2}, {10,7,6},  {7,1,8},
        {3,9,4},  {3,4,2},  {3,2,6},   {3,6,8},   {3,8,9},
        {4,9,5},  {2,4,11}, {6,2,10},  {8,6,7},   {9,8,1}};

    for (int i = 0; i < 12; ++i) {
        const float len = std::sqrt(base[i][0] * base[i][0] +
                                    base[i][1] * base[i][1] +
                                    base[i][2] * base[i][2]);
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u;
        h ^= h >> 15;
        const float jitter = 0.80f + 0.32f * (static_cast<float>(h & 0xFFFFu) / 65535.0f);
        for (int c = 0; c < 3; ++c) base[i][c] = base[i][c] / len * jitter;
    }

    for (const auto& f : faces) {
        const float* a = base[f[0]];
        const float* b = base[f[1]];
        const float* c = base[f[2]];
        const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float nx = e1[1] * e2[2] - e1[2] * e2[1];
        float ny = e1[2] * e2[0] - e1[0] * e2[2];
        float nz = e1[0] * e2[1] - e1[1] * e2[0];
        const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nl > 0.0f) { nx /= nl; ny /= nl; nz /= nl; }

        const std::uint32_t vbase =
            static_cast<std::uint32_t>(vtx.size()) -
            static_cast<std::uint32_t>(range.vertex_base);
        for (const float* p : {a, b, c}) {
            vtx.push_back({p[0], p[1], p[2], nx, ny, nz});
        }
        idx.push_back(vbase + 0); idx.push_back(vbase + 1); idx.push_back(vbase + 2);
    }
    range.index_count = static_cast<std::uint32_t>(idx.size()) - range.first_index;
}

// The baked boat meshes: a straight copy of boat_mesh.h's flat arrays into
// the shared vertex/index buffer, exactly like append_box/append_rock build
// their procedural geometry - the only difference is where the numbers came
// from. Both are already centred and scaled to metres by gltf_bake.
inline void append_boat_hull(std::vector<PropVertex>& vtx,
                             std::vector<std::uint32_t>& idx, PropMeshRange& range)
{
    range.first_index = static_cast<std::uint32_t>(idx.size());
    range.vertex_base = static_cast<std::int32_t>(vtx.size());

    const std::size_t vertex_count = std::size(boat_mesh::kHullVerts) / 6;
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const float* v = &boat_mesh::kHullVerts[i * 6];
        vtx.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
    }
    for (std::uint32_t i : boat_mesh::kHullIndices) idx.push_back(i);
    range.index_count = static_cast<std::uint32_t>(idx.size()) - range.first_index;
}

inline void append_boat_sail(std::vector<PropVertex>& vtx,
                             std::vector<std::uint32_t>& idx, PropMeshRange& range)
{
    range.first_index = static_cast<std::uint32_t>(idx.size());
    range.vertex_base = static_cast<std::int32_t>(vtx.size());

    const std::size_t vertex_count = std::size(boat_mesh::kSailVerts) / 6;
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const float* v = &boat_mesh::kSailVerts[i * 6];
        vtx.push_back({v[0], v[1], v[2], v[3], v[4], v[5]});
    }
    for (std::uint32_t i : boat_mesh::kSailIndices) idx.push_back(i);
    range.index_count = static_cast<std::uint32_t>(idx.size()) - range.first_index;
}

}  // namespace viewer
