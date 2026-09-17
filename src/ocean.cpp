#include "ocean/ocean.hpp"

#include "core/aligned.hpp"
#include "core/evolve.hpp"
#include "core/fft.hpp"
#include "core/spectrum.hpp"

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
    std::size_t cells = 0;

    // Everything below is allocated exactly once, here in the constructor.
    detail::SpectrumTables spectrum;
    detail::FftPlan        plan;
    detail::FieldSet       fields;
    detail::AlignedBuffer<float> fft_scratch;

    // Public outputs, GPU-texture-shaped: 4 floats per cell.
    detail::AlignedBuffer<float> displacement;
    detail::AlignedBuffer<float> normal;

    explicit Impl(const OceanDesc& d)
        : desc(d),
          cells(static_cast<std::size_t>(d.size) * d.size),
          plan(d.size),
          fft_scratch(plan.scratch_floats()),
          displacement(cells * 4),
          normal(cells * 4)
    {
        detail::build_spectrum(desc, spectrum);
        fields.allocate(d.size);
    }

    void run(double t) noexcept
    {
        const std::uint32_t n = desc.size;

        // 1. Advance every wavenumber to absolute time t and pack the eight
        //    field spectra into four complex fields.
        detail::evolve_rows(spectrum, t, fields, 0, n);

        // 2. Four inverse 2D transforms. No 1/N^2 normalisation: Tessendorf's
        //    surface is literally the unnormalised inverse sum
        //    h(x) = sum_k h(k) e^{i k.x}, and the spectrum amplitudes already
        //    carry the correct physical scale.
        for (int f = 0; f < 4; ++f) {
            plan.transform_2d(fields.re(f), fields.im(f), fft_scratch.data(),
                              detail::FftSign::Inverse);
        }

        // 3. Unpack, build the displaced-surface normal, the Jacobian and foam,
        //    and write the interleaved output in one sweep.
        detail::finalize_rows(fields, desc.choppiness, desc.foam_threshold,
                              displacement.data(), normal.data(), 0, n);

        time = t;
    }
};

Ocean::Ocean(const OceanDesc& desc)
{
    validate(desc);
    impl_ = std::make_unique<Impl>(desc);
    // Leave the ocean in a valid, fully-evaluated state at t = 0 so that
    // buffers() and the query API are meaningful before the first update().
    impl_->run(0.0);
}

Ocean::~Ocean() = default;
Ocean::Ocean(Ocean&&) noexcept = default;
Ocean& Ocean::operator=(Ocean&&) noexcept = default;

void Ocean::update(double time)
{
    // Absolute, not incremental: the surface is a pure function of
    // (seed, desc, time). Seeking or replaying reproduces it exactly.
    impl_->run(time);
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
    // Implemented in the next step (fixed-point inversion of the choppy
    // displacement). Returns flat-water defaults until then.
    return Surface{};
}

const OceanDesc& Ocean::desc() const noexcept { return impl_->desc; }
double           Ocean::time() const noexcept { return impl_->time; }

}  // namespace ocean
