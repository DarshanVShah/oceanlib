#include "ocean/cascade.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ocean {

struct CascadeStack::Impl {
    std::vector<Ocean> levels;
    double             time = 0.0;

    explicit Impl(std::span<const CascadeLevel> descs)
    {
        // Each Ocean(desc) validates and allocates exactly as a standalone
        // patch would. If level k throws, std::vector unwinds and destroys
        // levels [0, k) that already constructed successfully - no manual
        // cleanup needed here.
        levels.reserve(descs.size());
        for (const CascadeLevel& d : descs) {
            levels.emplace_back(d);
        }
    }
};

CascadeStack::CascadeStack(std::span<const CascadeLevel> levels)
{
    if (levels.empty()) {
        throw std::invalid_argument("CascadeStack: at least one level is required");
    }
    impl_ = std::make_unique<Impl>(levels);
}

CascadeStack::CascadeStack(std::initializer_list<CascadeLevel> levels)
    : CascadeStack(std::span<const CascadeLevel>(levels.begin(), levels.size()))
{}

CascadeStack::~CascadeStack()                                     = default;
CascadeStack::CascadeStack(CascadeStack&&) noexcept                = default;
CascadeStack& CascadeStack::operator=(CascadeStack&&) noexcept     = default;

void CascadeStack::update(double time) noexcept
{
    // Sequential across levels for now: each level already parallelises
    // internally (its own thread pool or the shared host scheduler), and this
    // loop itself performs no allocation, preserving the same guarantee
    // Ocean::update() makes. Interleaving levels' internal stages (e.g.
    // running level 0's FFT concurrently with level 1's spectrum evolution)
    // is a plausible future optimisation, but only worth the complexity once
    // measured to matter - see ADR-020.
    for (Ocean& o : impl_->levels) {
        o.update(time);
    }
    impl_->time = time;
}

std::size_t CascadeStack::level_count() const noexcept
{
    return impl_->levels.size();
}

Buffers CascadeStack::buffers(std::size_t level) const noexcept
{
    return impl_->levels[level].buffers();
}

const OceanDesc& CascadeStack::desc(std::size_t level) const noexcept
{
    return impl_->levels[level].desc();
}
const Ocean& CascadeStack::level(std::size_t level) const noexcept
{
    return impl_->levels[level];
}


float CascadeStack::height_at(float world_x, float world_z) const noexcept
{
    return sample_at(world_x, world_z).height;
}

Surface CascadeStack::sample_at(float world_x, float world_z) const noexcept
{
    // Height and horizontal offset: TRUE linear superposition. Each level's
    // height/offset field is an independent real-valued function of world
    // position (its own periodic tiling), and summing several real fields at
    // the same point is exact - not an approximation - for as long as each
    // level solves ITS OWN fixed-point inversion correctly in isolation. That
    // "in isolation" is the one place approximation actually enters: level i
    // inverts against only its own chop, not the combined chop of every
    // level, so the (u,v) it samples is the exact answer to "where does level
    // i's own displacement alone place a point here", not to the fully joint
    // multi-level inverse problem. See ADR-020 for the size of that residual.
    float height = 0.0f, offset_x = 0.0f, offset_z = 0.0f;

    // Normal and foam are combined from each level's own ALREADY-NORMALISED
    // output, not re-derived from summed raw slope/Jacobian gradients (the
    // public Ocean API does not expose those, by design - see ADR-005). This
    // is the same sum-then-renormalise technique used for detail normal map
    // blending in real-time rendering generally: not exact, but the standard,
    // well-understood approximation for combining independent surface detail
    // at different scales.
    float normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;

    // "Screen"/OR blend for foam: each level independently reports a folding
    // coverage in [0,1], and 1 - product(1 - foam_i) combines them the way
    // independent coverage layers combine in compositing - if any level's
    // scale is breaking here, the point should read as foamy, and multiple
    // simultaneous fold events compound toward but never exceed full coverage.
    float foam_transparency = 1.0f;

    // Orbital velocity sums EXACTLY, for the same reason height does: each
    // level's velocity field is an independent real vector field over world
    // position, and the composite flow is the superposition of the individual
    // wave trains. No renormalising, no blending - unlike the normal below.
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;

    for (const Ocean& o : impl_->levels) {
        const Surface s = o.sample_at(world_x, world_z);
        height   += s.height;
        offset_x += s.offset_x;
        offset_z += s.offset_z;
        normal_x += s.normal_x;
        normal_y += s.normal_y;
        normal_z += s.normal_z;
        vx += s.velocity_x;
        vy += s.velocity_y;
        vz += s.velocity_z;
        foam_transparency *= (1.0f - s.foam);
    }

    Surface out;
    out.height   = height;
    out.offset_x = offset_x;
    out.offset_z = offset_z;
    out.foam     = 1.0f - foam_transparency;
    out.velocity_x = vx;
    out.velocity_y = vy;
    out.velocity_z = vz;

    const float len2 = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z;
    if (len2 > 0.0f) {
        const float inv = 1.0f / std::sqrt(len2);
        out.normal_x = normal_x * inv;
        out.normal_y = normal_y * inv;
        out.normal_z = normal_z * inv;
    } else {
        out.normal_x = 0.0f;
        out.normal_y = 1.0f;
        out.normal_z = 0.0f;
    }
    return out;
}

double CascadeStack::time() const noexcept { return impl_->time; }

}  // namespace ocean
