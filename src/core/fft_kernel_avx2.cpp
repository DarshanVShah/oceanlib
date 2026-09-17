// AVX2 Stockham stage kernel.
//
// This translation unit is compiled with /arch:AVX2 (MSVC) or -mavx2
// (GCC/Clang), and is reached only through the runtime dispatch in
// cpu_features.cpp. Nothing else in the library may call into it directly,
// because a binary built this way still has to boot on a CPU without AVX2.
#include "core/fft_kernel.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)

#include <immintrin.h>

namespace ocean::detail {

void stage_avx2(const float* xr, const float* xi, float* yr, float* yi,
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

        // The twiddle is constant across the whole q loop - it depends only on
        // p - so it is broadcast once per butterfly row rather than reloaded.
        // This is exactly why split re[]/im[] storage was chosen (ADR-008):
        // every lane wants the same component, so there is nothing to shuffle.
        if (s >= 8u) {
            const __m256 vwr = _mm256_set1_ps(wr);
            const __m256 vwi = _mm256_set1_ps(wi);

            for (; q + 8u <= s; q += 8u) {
                const __m256 ar = _mm256_loadu_ps(xr + src0 + q);
                const __m256 ai = _mm256_loadu_ps(xi + src0 + q);
                const __m256 br = _mm256_loadu_ps(xr + src1 + q);
                const __m256 bi = _mm256_loadu_ps(xi + src1 + q);

                _mm256_storeu_ps(yr + dst0 + q, _mm256_add_ps(ar, br));
                _mm256_storeu_ps(yi + dst0 + q, _mm256_add_ps(ai, bi));

                const __m256 dr = _mm256_sub_ps(ar, br);
                const __m256 di = _mm256_sub_ps(ai, bi);

                // Deliberately separate mul and sub/add rather than _mm256_fmsub_ps
                // / _mm256_fmadd_ps. An FMA rounds once instead of twice, so it
                // is *more* accurate - and that is exactly the problem: it would
                // no longer match the scalar reference bit for bit, which would
                // make the ocean depend on which CPU rendered it. See the
                // contract note in fft_kernel.hpp.
                _mm256_storeu_ps(yr + dst1 + q,
                                 _mm256_sub_ps(_mm256_mul_ps(dr, vwr),
                                               _mm256_mul_ps(di, vwi)));
                _mm256_storeu_ps(yi + dst1 + q,
                                 _mm256_add_ps(_mm256_mul_ps(dr, vwi),
                                               _mm256_mul_ps(di, vwr)));
            }
        }

        // Tail, and the whole loop for the first three stages where s < 8.
        // Those stages are where the remaining headroom is; see ADR-013.
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
