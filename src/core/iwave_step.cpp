#include "core/iwave_step.hpp"

#include "core/cpu_features.hpp"

namespace ocean::detail {

void iwave_step_rows_scalar(const IWaveGrid& g, std::uint32_t row_begin,
                            std::uint32_t row_end) noexcept
{
    const int          p      = static_cast<int>(g.p);
    const int          w      = 2 * p + 1;
    const std::size_t  stride = g.stride;
    const std::uint32_t n     = g.n;

    for (std::uint32_t z = row_begin; z < row_end; ++z) {
        const std::size_t row = static_cast<std::size_t>(z) * stride;
        const std::size_t orow = static_cast<std::size_t>(z) * n;

        for (std::uint32_t x = 0; x < n; ++x) {
            const float* centre = g.cur + row + x;

            // The convolution, accumulated in a FIXED order: taps sweep j
            // outer, i inner, and the accumulator is a single float.
            //
            // The order is part of the contract, not an implementation detail.
            // The vector kernels process eight different OUTPUT cells at once,
            // one per lane, each walking this same tap sequence - so every lane
            // performs exactly these operations in exactly this order and the
            // vector path is bit-identical to this one by construction, rather
            // than by luck. Accumulating i-outer, or pairing symmetric taps to
            // save multiplies, would each give a different (equally valid, but
            // different) rounding and break that.
            float acc = 0.0f;
            const float* tap = g.taps;
            for (int j = -p; j <= p; ++j) {
                const float* srow = centre + static_cast<std::ptrdiff_t>(j) *
                                                 static_cast<std::ptrdiff_t>(stride) - p;
                for (int i = 0; i < w; ++i) {
                    acc += tap[i] * srow[i];
                }
                tap += w;
            }

            const float hc = *centre;
            const float hp = g.old[row + x];
            const float v =
                g.c1[orow + x] * (2.0f * hc - g.c2[orow + x] * hp - g.gdt2 * acc);

            // Flush negligible values to zero. See kIWaveFlush - this is a
            // performance fix with a correctness constraint, not a tolerance.
            g.old[row + x] = (v > -kIWaveFlush && v < kIWaveFlush) ? 0.0f : v;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

namespace {

using StepKernel = void (*)(const IWaveGrid&, std::uint32_t,
                            std::uint32_t) noexcept;

StepKernel select_step_kernel() noexcept
{
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    switch (detect_simd_level()) {
        case SimdLevel::Avx2: return &iwave_step_rows_avx2;
        case SimdLevel::Sse2: return &iwave_step_rows_sse2;
        default: break;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (detect_simd_level() == SimdLevel::Neon) return &iwave_step_rows_neon;
#endif
    return &iwave_step_rows_scalar;
}

}  // namespace

void iwave_step_rows(const IWaveGrid& g, std::uint32_t row_begin,
                     std::uint32_t row_end) noexcept
{
    // Resolved on first call, like the FFT and evolve kernels. Every
    // implementation is required to be bit-identical to the scalar one, so
    // this changes speed and nothing else.
    static const StepKernel kernel = select_step_kernel();
    kernel(g, row_begin, row_end);
}

// ---------------------------------------------------------------------------

void iwave_finalize_rows(const IWaveGrid& g, float inv_two_cell, float inv_dt,
                         float* out, std::uint32_t row_begin,
                         std::uint32_t row_end) noexcept
{
    const std::size_t   stride = g.stride;
    const std::uint32_t n      = g.n;

    for (std::uint32_t z = row_begin; z < row_end; ++z) {
        const std::size_t row = static_cast<std::size_t>(z) * stride;
        float* dst = out + static_cast<std::size_t>(z) * n * 4;

        for (std::uint32_t x = 0; x < n; ++x) {
            const float* c = g.cur + row + x;

            // Central differences. The halo means the edge cells need no
            // special case - they simply see the zero water outside, which is
            // the truth here rather than a convenient fiction.
            const float gx = (c[1] - c[-1]) * inv_two_cell;
            const float gz = (c[stride] - c[-static_cast<std::ptrdiff_t>(stride)]) *
                             inv_two_cell;

            // dEta/dt comes free from the two stored time levels - the same
            // two the leapfrog already needs.
            const float vy = (*c - g.old[row + x]) * inv_dt;

            dst[4 * x + 0] = *c;
            dst[4 * x + 1] = gx;
            dst[4 * x + 2] = gz;
            dst[4 * x + 3] = vy;
        }
    }
}

}  // namespace ocean::detail
