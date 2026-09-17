#include "ocean/ocean.hpp"

#include "core/aligned.hpp"
#include "core/evolve.hpp"
#include "core/fft.hpp"
#include "core/spectrum.hpp"

#include <cmath>
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

namespace {

// Bilinear tap into a periodic N x N grid of 4-component texels.
//
// Precomputed once per sample point because every query reads several
// components (dx, dz, height, normal, foam) at the same location, and the
// index arithmetic and wrapping would otherwise be repeated for each.
struct Tap {
    std::size_t i00 = 0, i10 = 0, i01 = 0, i11 = 0;
    float w00 = 1.0f, w10 = 0.0f, w01 = 0.0f, w11 = 0.0f;
};

std::uint32_t wrap_index(int i, int n) noexcept
{
    i %= n;
    if (i < 0) i += n;
    return static_cast<std::uint32_t>(i);
}

// `u`, `v` are positions in the undisplaced parameter plane, in metres.
Tap make_tap(float u, float v, std::uint32_t n, float inv_cell) noexcept
{
    const float gu = u * inv_cell;
    const float gv = v * inv_cell;

    const float fu = std::floor(gu);
    const float fv = std::floor(gv);

    const int ni = static_cast<int>(n);
    const std::uint32_t x0 = wrap_index(static_cast<int>(fu), ni);
    const std::uint32_t z0 = wrap_index(static_cast<int>(fv), ni);
    const std::uint32_t x1 = (x0 + 1u) % n;
    const std::uint32_t z1 = (z0 + 1u) % n;

    // Wrapping rather than clamping is correct, not a convenience: the FFT
    // surface is exactly periodic with period `patch_length`, so the tile
    // genuinely tiles, and a query far outside the patch is well defined.
    const float tu = gu - fu;
    const float tv = gv - fv;

    Tap t;
    t.i00 = (static_cast<std::size_t>(z0) * n + x0) * 4;
    t.i10 = (static_cast<std::size_t>(z0) * n + x1) * 4;
    t.i01 = (static_cast<std::size_t>(z1) * n + x0) * 4;
    t.i11 = (static_cast<std::size_t>(z1) * n + x1) * 4;
    t.w00 = (1.0f - tu) * (1.0f - tv);
    t.w10 = tu * (1.0f - tv);
    t.w01 = (1.0f - tu) * tv;
    t.w11 = tu * tv;
    return t;
}

float fetch(const float* buf, const Tap& t, int c) noexcept
{
    return buf[t.i00 + c] * t.w00 + buf[t.i10 + c] * t.w10 +
           buf[t.i01 + c] * t.w01 + buf[t.i11 + c] * t.w11;
}

// How many fixed-point steps to take when inverting the displacement.
//
// The iteration converges linearly with rate |lambda * grad D|, which is below
// 1 exactly where the surface has not folded. Four steps drive the residual to
// well under a millimetre for ordinary choppiness, and the error is bounded by
// rate^4 - see the convergence test, which measures it directly.
constexpr int kInversionSteps = 4;

}  // namespace

float Ocean::height_at(float world_x, float world_z) const noexcept
{
    return sample_at(world_x, world_z).height;
}

Surface Ocean::sample_at(float world_x, float world_z) const noexcept
{
    const Impl& m = *impl_;
    const std::uint32_t n = m.desc.size;
    const float inv_cell = static_cast<float>(n) / m.desc.patch_length;
    const float* disp = m.displacement.data();

    // Invert the horizontal displacement map.
    //
    // The renderer draws the cell at parameter position (u,v) at world
    // position (u + lambda*Dx(u,v), v + lambda*Dz(u,v)). So the vertex that
    // LANDS at (world_x, world_z) did not start there, and a direct lookup
    // would read the wrong cell - the error growing exactly where choppiness
    // matters most, at the crests.
    //
    // Solve  u + lambda*Dx(u,v) = world_x,  v + lambda*Dz(u,v) = world_z
    // by fixed-point iteration, starting from the undisplaced guess:
    //
    //     u <- world_x - lambda*Dx(u,v)
    //     v <- world_z - lambda*Dz(u,v)
    //
    // This is a contraction while |lambda * grad D| < 1, which is precisely
    // the condition that the Jacobian stays positive - i.e. that the surface
    // has not folded over itself. So the iteration converges wherever the
    // surface is single-valued, and degrades exactly where it genuinely is
    // not: inside a breaking wave there really are several surface points
    // above one (x,z), and no solver can pick one for us.
    //
    // Newton would converge quadratically here, and we even have the Jacobian
    // matrix on hand in the normal buffer. It is not worth it: each Newton
    // step needs three extra bilinear fetches for the gradient terms, so two
    // Newton steps cost about as much as five fixed-point steps, and four
    // fixed-point steps are already past the accuracy the bilinear
    // interpolation itself can deliver.
    float u = world_x;
    float v = world_z;
    for (int it = 0; it < kInversionSteps; ++it) {
        const Tap t = make_tap(u, v, n, inv_cell);
        // Channels 0 and 2 already hold lambda*Dx and lambda*Dz.
        u = world_x - fetch(disp, t, 0);
        v = world_z - fetch(disp, t, 2);
    }

    const Tap t = make_tap(u, v, n, inv_cell);
    const float* nrm = m.normal.data();

    Surface s;
    s.offset_x = fetch(disp, t, 0);
    s.height   = fetch(disp, t, 1);
    s.offset_z = fetch(disp, t, 2);
    s.foam     = fetch(disp, t, 3);

    // Interpolating unit vectors does not preserve length, so renormalise.
    float nx = fetch(nrm, t, 0);
    float ny = fetch(nrm, t, 1);
    float nz = fetch(nrm, t, 2);
    const float len2 = nx * nx + ny * ny + nz * nz;
    if (len2 > 0.0f) {
        const float inv = 1.0f / std::sqrt(len2);
        nx *= inv; ny *= inv; nz *= inv;
    } else {
        nx = 0.0f; ny = 1.0f; nz = 0.0f;
    }
    s.normal_x = nx;
    s.normal_y = ny;
    s.normal_z = nz;
    return s;
}

const OceanDesc& Ocean::desc() const noexcept { return impl_->desc; }
double           Ocean::time() const noexcept { return impl_->time; }

}  // namespace ocean
