#include "core/fft_kernel.hpp"

namespace ocean::detail {

void stage_scalar(const float* xr, const float* xi, float* yr, float* yi,
                  const float* twr, const float* twi, std::uint32_t len,
                  std::uint32_t s, std::uint32_t tw_step) noexcept
{
    const std::uint32_t m = len >> 1;
    for (std::uint32_t p = 0; p < m; ++p) {
        const float wr = twr[static_cast<std::size_t>(p) * tw_step];
        const float wi = twi[static_cast<std::size_t>(p) * tw_step];

        const std::uint32_t src0 = s * p;
        const std::uint32_t src1 = s * (p + m);
        const std::uint32_t dst0 = s * (2u * p);
        const std::uint32_t dst1 = s * (2u * p + 1u);

        for (std::uint32_t q = 0; q < s; ++q) {
            const float ar = xr[src0 + q];
            const float ai = xi[src0 + q];
            const float br = xr[src1 + q];
            const float bi = xi[src1 + q];

            // Sum leg carries no twiddle; difference leg is rotated by w.
            yr[dst0 + q] = ar + br;
            yi[dst0 + q] = ai + bi;

            const float dr = ar - br;
            const float di = ai - bi;
            yr[dst1 + q] = dr * wr - di * wi;
            yi[dst1 + q] = dr * wi + di * wr;
        }
    }
}

StageKernel select_stage_kernel(SimdLevel level) noexcept
{
    switch (level) {
#if defined(OCEAN_HAS_X86_KERNELS)
        case SimdLevel::Avx2: return &stage_avx2;
        case SimdLevel::Sse2: return &stage_sse2;
#endif
#if defined(OCEAN_HAS_NEON_KERNEL)
        case SimdLevel::Neon: return &stage_neon;
#endif
        default: break;
    }
    return &stage_scalar;
}

}  // namespace ocean::detail
