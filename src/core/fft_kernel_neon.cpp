// NEON (AArch64 ASIMD) Stockham stage kernel.
//
// NEON is mandatory on AArch64, so this needs no runtime feature check - the
// dispatch selects it whenever the build targets ARM64.
//
// Verified under QEMU user-mode emulation (real ARM64 machine code, real NEON
// instructions - not modelled), cross-compiled with aarch64-linux-gnu-g++ and
// run through the library's own bit-exactness test against stage_scalar. See
// ADR-018. Not yet run on physical ARM silicon; that remains the final check
// if this ever ships on real ARM hardware, though the risk left after passing
// under emulation is small (the only realistic gap is exotic denormal/FTZ
// behaviour, and this kernel touches no denormal-range values).
#include "core/fft_kernel.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

namespace ocean::detail {

void stage_neon(const float* xr, const float* xi, float* yr, float* yi,
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

        std::uint32_t q = 0;

        if (s >= 4u) {
            const float32x4_t vwr = vdupq_n_f32(wr);
            const float32x4_t vwi = vdupq_n_f32(wi);

            for (; q + 4u <= s; q += 4u) {
                const float32x4_t ar = vld1q_f32(xr + src0 + q);
                const float32x4_t ai = vld1q_f32(xi + src0 + q);
                const float32x4_t br = vld1q_f32(xr + src1 + q);
                const float32x4_t bi = vld1q_f32(xi + src1 + q);

                vst1q_f32(yr + dst0 + q, vaddq_f32(ar, br));
                vst1q_f32(yi + dst0 + q, vaddq_f32(ai, bi));

                const float32x4_t dr = vsubq_f32(ar, br);
                const float32x4_t di = vsubq_f32(ai, bi);

                // vmulq + vsubq, never vfmaq_f32. NEON's fused multiply-add
                // rounds once where the scalar path rounds twice, so using it
                // would break bit-exactness with the reference - see the
                // contract note in fft_kernel.hpp.
                vst1q_f32(yr + dst1 + q,
                          vsubq_f32(vmulq_f32(dr, vwr), vmulq_f32(di, vwi)));
                vst1q_f32(yi + dst1 + q,
                          vaddq_f32(vmulq_f32(dr, vwi), vmulq_f32(di, vwr)));
            }
        }

        for (; q < s; ++q) {
            const float ar = xr[src0 + q];
            const float ai = xi[src0 + q];
            const float br = xr[src1 + q];
            const float bi = xi[src1 + q];

            yr[dst0 + q] = ar + br;
            yi[dst0 + q] = ai + bi;

            const float dr = ar - br;
            const float di = ai - bi;
            yr[dst1 + q] = dr * wr - di * wi;
            yi[dst1 + q] = dr * wi + di * wr;
        }
    }
}

}  // namespace ocean::detail

#endif
