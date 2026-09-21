// Geometry clipmap mesh generation for the viewer's LOD (ADR-025).
//
// Pure host-side geometry: no Vulkan types here, so the layout and index math
// can be read (and eyeballed for correctness) without any graphics API noise,
// the same separation vk_math.hpp already keeps for linear algebra.
//
// A ring's local coordinates are integers in [-half, half] (half = segments/2)
// along each axis, with the origin at the ring's own centre. World position is
// local * cell_size + world_offset, computed in the vertex shader exactly like
// today's single-tile mesh multiplies its unit grid by patch_length - a ring
// is that same idea, just with its own independent scale and offset per level.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace viewer {

// Quads per side of a ring's full (uncut) square. Even, so the centred hole
// cut from every non-innermost ring is an exact segments/2 square. 64 keeps
// any one ring's own triangle count modest (64*64*2 = 8192 for the solid
// centre, 3/4 of that for an annulus) while the ring count carries the LOD
// range - see RingLayout below.
inline constexpr std::int32_t kRingSegments = 64;
inline constexpr std::int32_t kRingHalf     = kRingSegments / 2;

// Compile-time cap on ring count, so RingLayout::place can fill a fixed
// array rather than allocate one every frame (this is viewer/demo code, not
// the core library, but there is no reason to be sloppier about it here).
inline constexpr std::uint32_t kMaxRings = 8;

// One ring's placement for the current frame: independent of every other
// ring, following ADR-025's "finest possible camera-following precision at
// every scale" reasoning.
struct RingPlacement {
    float cell_size = 1.0f;  // world metres per local unit at this level
    float world_x   = 0.0f;  // snapped centre, world space
    float world_z   = 0.0f;
};

// Ring 0 is the innermost, finest, solid (no hole) ring; ring i > 0 is an
// annulus with a centred hole of segments/2 sized for ring (i-1) to nest in
// exactly, twice ring (i-1)'s cell size and footprint, standard clipmap
// doubling.
struct RingLayout {
    std::uint32_t ring_count    = 6;
    float         base_cell_size = 1.0f;  // ring 0's world metres per local unit

    [[nodiscard]] float cell_size(std::uint32_t ring) const
    {
        return base_cell_size * static_cast<float>(1u << ring);
    }

    // World footprint (full square side, before any hole is cut) of a ring.
    [[nodiscard]] float footprint(std::uint32_t ring) const
    {
        return cell_size(ring) * static_cast<float>(kRingSegments);
    }

    // Every ring's own centre, independently snapped to a multiple of ITS
    // OWN cell size - the finest snap available at that scale (ADR-025).
    // Because each ring snaps independently, ring i's centre and ring
    // (i-1)'s centre generally differ by a small offset (up to half of ring
    // i's own cell) - that gap is what StitchBand closes.
    //
    // Fills `out[0..ring_count)` in place; `out` must have at least
    // ring_count entries (callers size it kMaxRings, like the descriptor/
    // texture arrays already do for kMaxCascades) so this never allocates.
    void place(float camera_x, float camera_z,
              std::array<RingPlacement, kMaxRings>& out) const;
};

// Host-side vertex/index buffers for one shared clipmap mesh, reused by every
// ring at draw time via a different (cell_size, world_offset) pushed per
// draw - see RingPlacement and OceanView::record.
struct ClipmapMesh {
    // Local-space positions (2 floats per vertex: local x, local z), shared
    // by both the solid grid and every annulus.
    std::vector<float> vertices;

    // segments x segments grid, no hole: ring 0 only.
    std::vector<std::uint32_t> solid_indices;

    // Same grid with the centred (segments/2)^2 hole removed, boundary
    // always at the fixed, symmetric local +-(segments/4): every ring > 0.
    std::vector<std::uint32_t> annulus_indices;

    static ClipmapMesh build();
};

// The boundary stitch: closes the gap between ring `i`'s fixed hole edge and
// ring `i-1`'s ACTUAL outer edge, which can be offset from ring i's hole by
// up to half of ring i's own cell (see ADR-025) because the two rings snap
// to the camera independently. That offset is a continuous, small value, not
// one of a handful of discrete cases, so rather than precomputing multiple
// trim variants this is regenerated - positions only, the topology never
// changes - into a small preallocated buffer every frame. It is a few
// hundred vertices around one ring's hole perimeter, not a per-frame
// allocation: the buffer is sized once at init for the worst case and only
// its contents are rewritten, exactly like the existing per-frame texture
// upload already does for the cascade buffers.
//
// Positions are in ring i's own local space (the same convention as
// ClipmapMesh), i.e. already divided by ring i's cell_size, so the vertex
// shader applies the identical local * cell_size + world_offset transform
// to this geometry as to the rest of ring i's mesh.
struct StitchBand {
    std::vector<float>         vertices;  // local x, local z pairs
    std::vector<std::uint32_t> indices;   // fixed topology, built once

    // Regenerates `vertices` in place (no reallocation once capacity is
    // reserved) for the boundary between `outer` (ring i, provides the
    // fixed hole edge) and `inner` (ring i-1, provides the actual offset
    // edge this stitches to).
    void update(const RingPlacement& outer, const RingPlacement& inner);

    static StitchBand build();
};

}  // namespace viewer
