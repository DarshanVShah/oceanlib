// oceanlib - public API.
//
// Multiple overlapping ocean patches at different scales, summed into one
// composite sea state. A single Ocean patch must compromise between capturing
// large swells (needs a big patch_length) and fine ripples (needs many texels
// per metre, i.e. either a huge N or a small patch) - cascades resolve that by
// running several small, cheap patches side by side, each covering its own
// wavelength band, and adding their contributions together.
//
// This header is entirely additive: ocean::Ocean is completely unchanged, and
// every existing single-patch integration is unaffected by this addition.
//
// C++ only, deliberately, for v1.0: there is no ocean_cascade_* surface in
// ocean.h yet, so bindings from other languages cannot reach this class.
// A C API for CascadeStack is planned for a v1.1 release; see CHANGELOG.md.
#pragma once

#include "ocean/ocean.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>

namespace ocean {

// One scale in a cascade stack. Exactly an OceanDesc - a cascade level is
// simply an ocean patch, configured the same way a standalone one would be.
// The alias exists purely for readability at cascade call sites.
using CascadeLevel = OceanDesc;

// Sums several independently-scaled Ocean patches into one composite surface.
//
// IMPORTANT: this does NOT produce one combined CPU buffer. Each level keeps
// its own resolution and patch_length (that is the entire point - a "far"
// cascade might be 800 m across at 128x128, a "near" one 50 m across at
// 256x256), so their texel grids do not share a common indexing and cannot be
// flattened into a single array. A renderer samples each level's own
// displacement/normal textures at world position mapped into THAT level's own
// UV space, and sums the samples - exactly as buffers(i) below is meant to be
// used. See ARCHITECTURE.md ADR-020 for the full reasoning, including exactly
// what is exact (height, horizontal offset - true linear superposition of the
// same real surface) and what is a stated approximation (normal and foam
// blending, and the per-cascade-independent choppy inversion).
class CascadeStack {
public:
    // At least one level is required. Each level is validated exactly as a
    // standalone Ocean would be (throws std::invalid_argument on a bad
    // descriptor); levels already constructed before a later one throws are
    // cleaned up automatically.
    explicit CascadeStack(std::span<const CascadeLevel> levels);
    CascadeStack(std::initializer_list<CascadeLevel> levels);
    ~CascadeStack();

    CascadeStack(CascadeStack&&) noexcept;
    CascadeStack& operator=(CascadeStack&&) noexcept;
    CascadeStack(const CascadeStack&)            = delete;
    CascadeStack& operator=(const CascadeStack&) = delete;

    // Advances every level to the same absolute time. Performs no heap
    // allocation, exactly like Ocean::update() - it is a fixed-size loop over
    // already-allocated Ocean instances.
    void update(double time) noexcept;

    [[nodiscard]] std::size_t level_count() const noexcept;

    // Level i's own buffers, unmodified - see the class comment for how a
    // renderer is meant to combine them.
    [[nodiscard]] Buffers buffers(std::size_t level) const noexcept;
    [[nodiscard]] const OceanDesc& desc(std::size_t level) const noexcept;

    // Combined physics query, summed across every level at the same world
    // position. Height and horizontal offset are exact (see ADR-020); normal
    // and foam are blended from each level's own already-normalised output.
    [[nodiscard]] float height_at(float world_x, float world_z) const noexcept;
    [[nodiscard]] Surface sample_at(float world_x, float world_z) const noexcept;

    [[nodiscard]] double time() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ocean
