// The vector kernels behind half.hpp, in their own translation unit so that
// AVX2 and F16C can be enabled for them and only them. See the header.
#include "half.hpp"

#if defined(VIEWER_HALF_X86)

#include <immintrin.h>

namespace viewer {

#if defined(__GNUC__) || defined(__clang__)
#define VIEWER_HALF_TARGET __attribute__((target("avx2,f16c")))
#else
#define VIEWER_HALF_TARGET
#endif

VIEWER_HALF_TARGET
void pack_half_f16c(std::uint16_t* dst, const float* src,
                    std::size_t count) noexcept
{
    std::size_t i = 0;
    // Four vectors per iteration: 32 floats in, one full 64-byte line out. The
    // destination is write-combined, where a partially filled line is flushed
    // as its own bus transaction, so filling whole lines is worth more here
    // than the unrolling is.
    for (; i + 32 <= count; i += 32) {
        const __m128i h0 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i +  0), 0);
        const __m128i h1 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i +  8), 0);
        const __m128i h2 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i + 16), 0);
        const __m128i h3 = _mm256_cvtps_ph(_mm256_loadu_ps(src + i + 24), 0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_set_m128i(h1, h0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 16),
                            _mm256_set_m128i(h3, h2));
    }
    for (; i + 8 <= count; i += 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), 0));
    }
    for (; i < count; ++i) dst[i] = float_to_half(src[i]);
}

VIEWER_HALF_TARGET
void pack_rgba16_f16c(std::uint16_t* dst, const float* src, const float* alpha,
                      std::size_t cells) noexcept
{
    if (alpha == nullptr) {
        pack_half_f16c(dst, src, cells * 4);
        return;
    }

    // Eight texels per iteration: four source vectors, one vector of eight
    // alpha values. Each source vector covers two texels, so it needs two of
    // those alphas, at lanes 3 and 7 - which is exactly one cross-lane
    // permute, and then a blend that touches nothing else.
    const __m256i lane0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 1);
    const __m256i lane1 = _mm256_setr_epi32(0, 0, 0, 2, 0, 0, 0, 3);
    const __m256i lane2 = _mm256_setr_epi32(0, 0, 0, 4, 0, 0, 0, 5);
    const __m256i lane3 = _mm256_setr_epi32(0, 0, 0, 6, 0, 0, 0, 7);

    std::size_t c = 0;
    for (; c + 8 <= cells; c += 8) {
        const float* s  = src + 4 * c;
        const __m256 av = _mm256_loadu_ps(alpha + c);

        const __m256 v0 = _mm256_blend_ps(_mm256_loadu_ps(s +  0),
                              _mm256_permutevar8x32_ps(av, lane0), 0x88);
        const __m256 v1 = _mm256_blend_ps(_mm256_loadu_ps(s +  8),
                              _mm256_permutevar8x32_ps(av, lane1), 0x88);
        const __m256 v2 = _mm256_blend_ps(_mm256_loadu_ps(s + 16),
                              _mm256_permutevar8x32_ps(av, lane2), 0x88);
        const __m256 v3 = _mm256_blend_ps(_mm256_loadu_ps(s + 24),
                              _mm256_permutevar8x32_ps(av, lane3), 0x88);

        std::uint16_t* d = dst + 4 * c;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d),
                            _mm256_set_m128i(_mm256_cvtps_ph(v1, 0),
                                             _mm256_cvtps_ph(v0, 0)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 16),
                            _mm256_set_m128i(_mm256_cvtps_ph(v3, 0),
                                             _mm256_cvtps_ph(v2, 0)));
    }
    pack_rgba16_scalar(dst + 4 * c, src + 4 * c, alpha + c, cells - c);
}

}  // namespace viewer

#endif  // VIEWER_HALF_X86
