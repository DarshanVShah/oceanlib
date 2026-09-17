#include "ocean/ocean.hpp"

#include "core/aligned.hpp"

#include <stdexcept>
#include <string>

namespace ocean {
namespace {

constexpr bool is_power_of_two(std::uint32_t v) noexcept
{
    return v != 0 && (v & (v - 1)) == 0;
}

void validate(const OceanDesc& d)
{
    // Radix-2 FFT, so N must be a power of two. The lower bound keeps the
    // butterfly loops from having degenerate stages; the upper bound is a
    // sanity guard (1024^2 complex fields already cost tens of MB).
    if (!is_power_of_two(d.size) || d.size < 16 || d.size > 4096) {
        throw std::invalid_argument(
            "ocean: size must be a power of two in [16, 4096], got " +
            std::to_string(d.size));
    }
    if (!(d.patch_length > 0.0f)) {
        throw std::invalid_argument("ocean: patch_length must be positive");
    }
    if (!(d.spectrum.gravity > 0.0f)) {
        throw std::invalid_argument("ocean: gravity must be positive");
    }
    if (!(d.spectrum.fetch > 0.0f)) {
        throw std::invalid_argument("ocean: fetch must be positive");
    }
    if (!(d.spectrum.wind_speed >= 0.0f)) {
        throw std::invalid_argument("ocean: wind_speed must be non-negative");
    }
    if (!(d.spectrum.peak_enhancement >= 1.0f)) {
        // gamma < 1 would *suppress* the spectral peak, which JONSWAP does not
        // model; gamma == 1 is the Pierson-Moskowitz degenerate case.
        throw std::invalid_argument(
            "ocean: peak_enhancement must be >= 1 (1 = Pierson-Moskowitz)");
    }
    if (!(d.choppiness >= 0.0f)) {
        throw std::invalid_argument("ocean: choppiness must be non-negative");
    }
}

}  // namespace

// ---------------------------------------------------------------------------

struct Ocean::Impl {
    OceanDesc   desc;
    double      time  = 0.0;
    std::size_t cells = 0;  // N*N

    // Public outputs, GPU-texture-shaped: 4 floats per cell.
    detail::AlignedBuffer<float> displacement;
    detail::AlignedBuffer<float> normal;

    explicit Impl(const OceanDesc& d)
        : desc(d),
          cells(static_cast<std::size_t>(d.size) * d.size),
          displacement(cells * 4),
          normal(cells * 4)
    {
        // Until the spectrum and FFT land, the buffers describe a perfectly
        // flat ocean at y = 0: zero displacement, zero foam, +Y normals, and a
        // Jacobian of 1 (no folding). That is a real, self-consistent surface,
        // not a placeholder - every query below agrees with it.
        for (std::size_t i = 0; i < cells; ++i) {
            normal[4 * i + 1] = 1.0f;  // ny
            normal[4 * i + 3] = 1.0f;  // jacobian
        }
    }
};

Ocean::Ocean(const OceanDesc& desc)
{
    validate(desc);
    impl_ = std::make_unique<Impl>(desc);
}

Ocean::~Ocean() = default;
Ocean::Ocean(Ocean&&) noexcept = default;
Ocean& Ocean::operator=(Ocean&&) noexcept = default;

void Ocean::update(double time)
{
    // Absolute, not incremental: the surface is a pure function of
    // (seed, desc, time). Seeking or replaying reproduces it exactly.
    impl_->time = time;
}

Buffers Ocean::buffers() const noexcept
{
    Buffers b;
    b.displacement = impl_->displacement.data();
    b.normal       = impl_->normal.data();
    b.size         = impl_->desc.size;
    b.patch_length = impl_->desc.patch_length;
    return b;
}

float Ocean::height_at(float world_x, float world_z) const noexcept
{
    return sample_at(world_x, world_z).height;
}

Surface Ocean::sample_at(float, float) const noexcept
{
    // Flat ocean for now; the defaults in Surface already describe it.
    return Surface{};
}

const OceanDesc& Ocean::desc() const noexcept { return impl_->desc; }
double           Ocean::time() const noexcept { return impl_->time; }

}  // namespace ocean
