// SSE2 Stockham stage kernel.
//
// SSE2 is part of the x86-64 base architecture, so this TU needs no special
// compile flags and no runtime guard beyond "are we on x86". It exists so the
// library still gets 4-wide vectors on machines too old for AVX2 - promise #1
// is "runs on anything", and "anything" includes hardware from before 2013.
#include "core/fft_kernel.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)

#include <emmintrin.h>

namespace ocean::detail {

void stage_sse2(const float* xr, const float* xi, float* yr, float* yi,
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
            const __m128 vwr = _mm_set1_ps(wr);
            const __m128 vwi = _mm_set1_ps(wi);

            for (; q + 4u <= s; q += 4u) {
                const __m128 ar = _mm_loadu_ps(xr + src0 + q);
                const __m128 ai = _mm_loadu_ps(xi + src0 + q);
                const __m128 br = _mm_loadu_ps(xr + src1 + q);
                const __m128 bi = _mm_loadu_ps(xi + src1 + q);

                _mm_storeu_ps(yr + dst0 + q, _mm_add_ps(ar, br));
                _mm_storeu_ps(yi + dst0 + q, _mm_add_ps(ai, bi));

                const __m128 dr = _mm_sub_ps(ar, br);
                const __m128 di = _mm_sub_ps(ai, bi);

                // Same operation order as the scalar path, four lanes at a
                // time. SSE2 has no FMA, so bit-exactness here is automatic
                // rather than something we have to defend against.
                _mm_storeu_ps(yr + dst1 + q,
                              _mm_sub_ps(_mm_mul_ps(dr, vwr), _mm_mul_ps(di, vwi)));
                _mm_storeu_ps(yi + dst1 + q,
                              _mm_add_ps(_mm_mul_ps(dr, vwi), _mm_mul_ps(di, vwr)));
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
