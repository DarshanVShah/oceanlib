// SSE2 iWave convolution kernel.
//
// Bit-exactness with iwave_step_rows_scalar is a hard requirement (ADR-013):
// if the vector path produced even slightly different numbers, the ripples a
// player saw would depend on which CPU rendered them.
//
// That requirement is satisfied STRUCTURALLY here, not by careful matching.
// Each lane computes a DIFFERENT output cell, and every lane walks exactly the
// tap sequence the scalar loop walks, in the same order, into its own
// accumulator. So each lane performs precisely the scalar operations. No FMA
// anywhere - fmadd would round once instead of twice and be more accurate than
// scalar, which is exactly why it is banned.
#include "core/iwave_step.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)

#include <emmintrin.h>

namespace ocean::detail {

void iwave_step_rows_sse2(const IWaveGrid& g, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept
{
    const int           p      = static_cast<int>(g.p);
    const int           w      = 2 * p + 1;
    const std::size_t   stride = g.stride;
    const std::uint32_t n      = g.n;

    const __m128 two  = _mm_set1_ps(2.0f);
    const __m128 gdt2 = _mm_set1_ps(g.gdt2);

    // N is a power of two of at least 32, so it is always a multiple of 4 and
    // there is never a tail to special-case.
    for (std::uint32_t z = row_begin; z < row_end; ++z) {
        const std::size_t row  = static_cast<std::size_t>(z) * stride;
        const std::size_t orow = static_cast<std::size_t>(z) * n;

        for (std::uint32_t x = 0; x < n; x += 4) {
            __m128 acc = _mm_setzero_ps();
            const float* tap = g.taps;
            for (int j = -p; j <= p; ++j) {
                const float* srow = g.cur + row + x +
                                    static_cast<std::ptrdiff_t>(j) *
                                        static_cast<std::ptrdiff_t>(stride) - p;
                for (int i = 0; i < w; ++i) {
                    acc = _mm_add_ps(
                        acc, _mm_mul_ps(_mm_set1_ps(tap[i]),
                                        _mm_loadu_ps(srow + i)));
                }
                tap += w;
            }
            const __m128 hc  = _mm_loadu_ps(g.cur + row + x);
            const __m128 hp  = _mm_loadu_ps(g.old + row + x);
            const __m128 c1v = _mm_loadu_ps(g.c1 + orow + x);
            const __m128 c2v = _mm_loadu_ps(g.c2 + orow + x);

            // (2*hc - c2*hp) - gdt2*acc, then scaled - the same association as
            // the scalar expression, which parses left to right.
            const __m128 r = _mm_mul_ps(
                c1v, _mm_sub_ps(_mm_sub_ps(_mm_mul_ps(two, hc),
                                           _mm_mul_ps(c2v, hp)),
                                _mm_mul_ps(gdt2, acc)));
            // Flush negligible values to zero, branchlessly. Same threshold and
            // same result as the scalar path - see kIWaveFlush.
            const __m128 mag = _mm_andnot_ps(_mm_set1_ps(-0.0f), r);
            const __m128 keep = _mm_cmpge_ps(mag, _mm_set1_ps(kIWaveFlush));
            _mm_storeu_ps(g.old + row + x, _mm_and_ps(r, keep));
        }
    }
}

}  // namespace ocean::detail

#endif
