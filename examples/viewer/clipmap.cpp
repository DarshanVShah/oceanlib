#include "clipmap.hpp"

#include <cmath>

namespace viewer {
namespace {

constexpr std::int32_t kHole = kRingHalf / 2;  // half-width of the centred hole, in local units
constexpr std::int32_t kVertsPerSide = kRingSegments + 1;

std::uint32_t vertex_index(std::int32_t local_i, std::int32_t local_z)
{
    return static_cast<std::uint32_t>((local_z + kRingHalf) * kVertsPerSide +
                                      (local_i + kRingHalf));
}

}  // namespace

std::vector<RingPlacement> RingLayout::place(float camera_x, float camera_z) const
{
    std::vector<RingPlacement> out;
    out.reserve(ring_count);
    for (std::uint32_t r = 0; r < ring_count; ++r) {
        const float cs = cell_size(r);
        RingPlacement p;
        p.cell_size = cs;
        // Snap to the nearest multiple of this ring's OWN cell size - the
        // finest grid available at this level. Rounding to nearest rather
        // than flooring keeps the ring centred under the camera rather than
        // trailing it by up to one cell.
        p.world_x = std::round(camera_x / cs) * cs;
        p.world_z = std::round(camera_z / cs) * cs;
        out.push_back(p);
    }
    return out;
}

ClipmapMesh ClipmapMesh::build()
{
    ClipmapMesh mesh;

    mesh.vertices.reserve(static_cast<std::size_t>(kVertsPerSide) * kVertsPerSide * 2);
    for (std::int32_t z = -kRingHalf; z <= kRingHalf; ++z) {
        for (std::int32_t x = -kRingHalf; x <= kRingHalf; ++x) {
            mesh.vertices.push_back(static_cast<float>(x));
            mesh.vertices.push_back(static_cast<float>(z));
        }
    }

    auto push_quad = [](std::vector<std::uint32_t>& idx, std::int32_t x, std::int32_t z) {
        // Counter-clockwise viewed from +Y, matching the existing tile mesh's
        // winding (ocean_view.cpp's create_mesh) and the pipeline's front face.
        const std::uint32_t i00 = vertex_index(x, z);
        const std::uint32_t i10 = vertex_index(x + 1, z);
        const std::uint32_t i01 = vertex_index(x, z + 1);
        const std::uint32_t i11 = vertex_index(x + 1, z + 1);
        idx.push_back(i00); idx.push_back(i01); idx.push_back(i10);
        idx.push_back(i10); idx.push_back(i01); idx.push_back(i11);
    };

    mesh.solid_indices.reserve(static_cast<std::size_t>(kRingSegments) * kRingSegments * 6);
    for (std::int32_t z = -kRingHalf; z < kRingHalf; ++z) {
        for (std::int32_t x = -kRingHalf; x < kRingHalf; ++x) {
            push_quad(mesh.solid_indices, x, z);
        }
    }

    mesh.annulus_indices.reserve(mesh.solid_indices.size());  // upper bound
    for (std::int32_t z = -kRingHalf; z < kRingHalf; ++z) {
        for (std::int32_t x = -kRingHalf; x < kRingHalf; ++x) {
            // Cell (x, z) occupies local [x, x+1) x [z, z+1); it is inside the
            // centred hole - and therefore omitted, since the next finer ring
            // covers this area at its own resolution - iff its whole extent
            // falls within [-kHole, kHole).
            const bool in_hole = x >= -kHole && x < kHole && z >= -kHole && z < kHole;
            if (in_hole) continue;
            push_quad(mesh.annulus_indices, x, z);
        }
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// StitchBand
// ---------------------------------------------------------------------------

namespace {

// One side of the hole boundary. `along` is the axis the side runs along
// (0 = x varies, boundary at fixed z; 1 = z varies, boundary at fixed x);
// `fixed_sign` is which of the two parallel sides (-1 or +1).
struct Side {
    int along;
    int fixed_sign;
};
constexpr Side kSides[4] = {
    {0, -1},  // z = -kHole, x varies:  "south" edge
    {0, +1},  // z = +kHole, x varies:  "north" edge
    {1, -1},  // x = -kHole, z varies:  "west" edge
    {1, +1},  // x = +kHole, z varies:  "east" edge
};

// Coarse vertices per side (outer ring's own hole-boundary resolution) and
// fine vertices per side (inner ring's own boundary resolution, exactly 2x -
// see ADR-021: ring (i-1)'s cell size is half of ring i's).
constexpr std::int32_t kCoarseSegs = 2 * kHole;         // = kRingHalf
constexpr std::int32_t kCoarseVerts = kCoarseSegs + 1;
constexpr std::int32_t kFineSegs = 2 * kCoarseSegs;      // = 2 * kRingHalf
constexpr std::int32_t kFineVerts = kFineSegs + 1;

constexpr std::size_t kVertsPerSideTotal =
    static_cast<std::size_t>(kCoarseVerts) + static_cast<std::size_t>(kFineVerts);
constexpr std::size_t kTrianglesPerSide = static_cast<std::size_t>(kCoarseSegs) * 3;

}  // namespace

StitchBand StitchBand::build()
{
    StitchBand band;
    band.vertices.resize(4 * kVertsPerSideTotal * 2, 0.0f);  // 2 floats/vertex
    band.indices.reserve(4 * kTrianglesPerSide * 3);

    // Topology only, fixed for the life of the program: for side `s`, the
    // coarse vertex block starts at vertex index (s * kVertsPerSideTotal),
    // the fine block immediately after it. Connectivity is the standard
    // N-to-2N stitch: each coarse segment [c_k, c_{k+1}] meets 2 fine
    // segments [f_{2k}, f_{2k+1}, f_{2k+2}], closed with 3 triangles sharing
    // the fine midpoint - this stays correct regardless of the ACTUAL vertex
    // positions StitchBand::update() writes each frame, since positions and
    // topology are independent here.
    for (std::uint32_t s = 0; s < 4; ++s) {
        const std::uint32_t coarse0 = s * static_cast<std::uint32_t>(kVertsPerSideTotal);
        const std::uint32_t fine0   = coarse0 + static_cast<std::uint32_t>(kCoarseVerts);
        for (std::int32_t k = 0; k < kCoarseSegs; ++k) {
            const std::uint32_t c0 = coarse0 + static_cast<std::uint32_t>(k);
            const std::uint32_t c1 = c0 + 1;
            const std::uint32_t f0 = fine0 + static_cast<std::uint32_t>(2 * k);
            const std::uint32_t f1 = f0 + 1;
            const std::uint32_t f2 = f0 + 2;
            // Wound so the stitch faces the same way (CCW from +Y) as the
            // rest of the mesh; which triple is CCW depends on which side of
            // the hole this is, since opposite sides face opposite ways.
            const bool flip = kSides[s].fixed_sign < 0;
            auto tri = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                if (flip) { band.indices.push_back(a); band.indices.push_back(c); band.indices.push_back(b); }
                else      { band.indices.push_back(a); band.indices.push_back(b); band.indices.push_back(c); }
            };
            if (kSides[s].along == 1) {
                // "west"/"east" sides: swap the winding sense relative to the
                // x-varying sides, since z now plays the role x played there.
                tri(c0, f1, f0);
                tri(c0, c1, f1);
                tri(c1, f2, f1);
            } else {
                tri(c0, f0, f1);
                tri(c0, f1, c1);
                tri(c1, f1, f2);
            }
        }
    }
    return band;
}

void StitchBand::update(const RingPlacement& outer, const RingPlacement& inner)
{
    // Every position is computed in world space, then converted into ring
    // `outer`'s own local frame (world - outer.world) / outer.cell_size, the
    // same convention the shared ClipmapMesh vertex buffer uses - so the
    // vertex shader applies one identical local*cell_size+offset transform
    // to this geometry as to the rest of ring `outer`'s draw.
    auto write = [&](std::size_t v, float world_x, float world_z) {
        vertices[2 * v + 0] = (world_x - outer.world_x) / outer.cell_size;
        vertices[2 * v + 1] = (world_z - outer.world_z) / outer.cell_size;
    };

    // vertices already sized by build(); rewrite in place, no reallocation.
    for (std::uint32_t s = 0; s < 4; ++s) {
        const Side side = kSides[s];
        const std::size_t coarse0 = s * kVertsPerSideTotal;
        const std::size_t fine0   = coarse0 + static_cast<std::size_t>(kCoarseVerts);

        for (std::int32_t k = 0; k < kCoarseVerts; ++k) {
            const std::int32_t along_local = -kHole + k;  // -kHole..+kHole
            float world_x, world_z;
            if (side.along == 0) {
                world_x = outer.world_x + static_cast<float>(along_local) * outer.cell_size;
                world_z = outer.world_z + static_cast<float>(side.fixed_sign * kHole) * outer.cell_size;
            } else {
                world_x = outer.world_x + static_cast<float>(side.fixed_sign * kHole) * outer.cell_size;
                world_z = outer.world_z + static_cast<float>(along_local) * outer.cell_size;
            }
            write(coarse0 + static_cast<std::size_t>(k), world_x, world_z);
        }

        for (std::int32_t k = 0; k < kFineVerts; ++k) {
            const std::int32_t along_local = -kRingHalf + k;  // -kRingHalf..+kRingHalf
            float world_x, world_z;
            if (side.along == 0) {
                world_x = inner.world_x + static_cast<float>(along_local) * inner.cell_size;
                world_z = inner.world_z + static_cast<float>(side.fixed_sign * kRingHalf) * inner.cell_size;
            } else {
                world_x = inner.world_x + static_cast<float>(side.fixed_sign * kRingHalf) * inner.cell_size;
                world_z = inner.world_z + static_cast<float>(along_local) * inner.cell_size;
            }
            write(fine0 + static_cast<std::size_t>(k), world_x, world_z);
        }
    }
}

}  // namespace viewer
