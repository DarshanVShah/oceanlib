// NEON iWave convolution kernel.
//
// Same structure and the same bit-exactness contract as the SSE2 and AVX2
// kernels: one output cell per lane, every lane walking the scalar tap
// sequence in the scalar order. vmlaq_f32 is deliberately NOT used - on
// AArch64 it contracts to a fused multiply-add, which rounds once instead of
// twice and would make the NEON path MORE accurate than scalar. More accurate
// is still different, and different breaks determinism across machines.
#include "core/iwave_step.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

namespace ocean::detail {

void iwave_step_rows_neon(const IWaveGrid& g, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept
{
    const int           p      = static_cast<int>(g.p);
    const int           w      = 2 * p + 1;
    const std::size_t   stride = g.stride;
    const std::uint32_t n      = g.n;

    const float32x4_t two  = vdupq_n_f32(2.0f);
    const float32x4_t gdt2 = vdupq_n_f32(g.gdt2);

    for (std::uint32_t z = row_begin; z < row_end; ++z) {
        const std::size_t row  = static_cast<std::size_t>(z) * stride;
        const std::size_t orow = static_cast<std::size_t>(z) * n;

        for (std::uint32_t x = 0; x < n; x += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            const float* tap = g.taps;
            for (int j = -p; j <= p; ++j) {
                const float* srow = g.cur + row + x +
                                    static_cast<std::ptrdiff_t>(j) *
                                        static_cast<std::ptrdiff_t>(stride) - p;
                for (int i = 0; i < w; ++i) {
                    // Separate multiply and add, never vmlaq_f32.
                    acc = vaddq_f32(acc, vmulq_f32(vdupq_n_f32(tap[i]),
                                                   vld1q_f32(srow + i)));
                }
                tap += w;
            }
            const float32x4_t hc  = vld1q_f32(g.cur + row + x);
            const float32x4_t hp  = vld1q_f32(g.old + row + x);
            const float32x4_t c1v = vld1q_f32(g.c1 + orow + x);
            const float32x4_t c2v = vld1q_f32(g.c2 + orow + x);

            const float32x4_t r = vmulq_f32(
                c1v, vsubq_f32(vsubq_f32(vmulq_f32(two, hc),
                                         vmulq_f32(c2v, hp)),
                               vmulq_f32(gdt2, acc)));
            vst1q_f32(g.old + row + x, r);
        }
    }
}

}  // namespace ocean::detail

#endif
