// AVX2 spectrum-evolution kernel.
//
// Compiled with /arch:AVX2 (or -mavx2) and reached only through the runtime
// dispatch in evolve.cpp, exactly like the AVX2 FFT kernel.
//
// Bit-exactness with evolve_rows_scalar is a hard requirement, for the same
// reason it is in the FFT: if the vector path produced even slightly different
// numbers, the ocean would depend on which CPU rendered it. Every operation
// below is the same operation, in the same order, as the scalar code - just
// eight lanes wide. No FMA anywhere.
#include "core/evolve.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)

#include "core/trig.hpp"

#include <immintrin.h>

namespace ocean::detail {
namespace {

// Eight phases at once, folded into [0, 2*pi).
//
// The fold has to happen in double - omega*t reaches thousands of radians -
// so the eight floats are widened to two 4-wide double vectors, folded, and
// narrowed back. That widening is why this is not simply eight times faster
// than scalar, but it is still far cheaper than eight libm calls.
inline __m256 reduced_phase8(__m256 omega, double time_over_two_pi) noexcept
{
    const __m256d scale = _mm256_set1_pd(time_over_two_pi);
    const __m256d two_pi = _mm256_set1_pd(kTwoPiD);

    __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(omega));
    __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(omega, 1));

    lo = _mm256_mul_pd(lo, scale);
    hi = _mm256_mul_pd(hi, scale);

    // d - floor(d), matching reduced_phase exactly.
    lo = _mm256_sub_pd(lo, _mm256_floor_pd(lo));
    hi = _mm256_sub_pd(hi, _mm256_floor_pd(hi));

    lo = _mm256_mul_pd(lo, two_pi);
    hi = _mm256_mul_pd(hi, two_pi);

    return _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
}

// Vector form of sincos_f32. The scalar version was deliberately written with
// selects rather than a switch so this could mirror it operation for
// operation.
inline void sincos8(__m256 x, __m256& out_sin, __m256& out_cos) noexcept
{
    const __m256 qf = _mm256_floor_ps(
        _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(kTwoOverPi)),
                      _mm256_set1_ps(0.5f)));
    // qf is already integral, so truncation and rounding agree; this matches
    // the scalar static_cast<int32_t>.
    const __m256i q = _mm256_cvttps_epi32(qf);

    // Cody-Waite reduction in two steps.
    __m256 y = _mm256_sub_ps(x, _mm256_mul_ps(qf, _mm256_set1_ps(kPiOver2A)));
    y = _mm256_sub_ps(y, _mm256_mul_ps(qf, _mm256_set1_ps(kPiOver2B)));

    const __m256 z = _mm256_mul_ps(y, y);

    // sin(y) = y + ((s0*z + s1)*z + s2)*z*y
    __m256 p = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(kSin0), z),
                             _mm256_set1_ps(kSin1));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(kSin2));
    const __m256 sin_y =
        _mm256_add_ps(y, _mm256_mul_ps(_mm256_mul_ps(p, z), y));

    // cos(y) = (1 - 0.5*z) + ((c0*z + c1)*z + c2)*z*z
    __m256 r = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(kCos0), z),
                             _mm256_set1_ps(kCos1));
    r = _mm256_add_ps(_mm256_mul_ps(r, z), _mm256_set1_ps(kCos2));
    const __m256 cos_y = _mm256_add_ps(
        _mm256_sub_ps(_mm256_set1_ps(1.0f),
                      _mm256_mul_ps(_mm256_set1_ps(0.5f), z)),
        _mm256_mul_ps(_mm256_mul_ps(r, z), z));

    const __m256i one  = _mm256_set1_epi32(1);
    const __m256i two  = _mm256_set1_epi32(2);
    const __m256i quad = _mm256_and_si256(q, _mm256_set1_epi32(3));

    // swap when quadrant is odd
    const __m256 swap = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(quad, one), one));
    const __m256 base_s = _mm256_blendv_ps(sin_y, cos_y, swap);
    const __m256 base_c = _mm256_blendv_ps(cos_y, sin_y, swap);

    // sin negative in quadrants 2,3; cos negative in 1,2
    const __m256 neg_s = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(quad, two), two));
    const __m256 neg_c = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(_mm256_add_epi32(quad, one), two), two));

    // XOR with the sign bit is bit-identical to unary minus for every finite
    // value, including negative zero.
    const __m256 signbit =
        _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));
    out_sin = _mm256_xor_ps(base_s, _mm256_and_ps(neg_s, signbit));
    out_cos = _mm256_xor_ps(base_c, _mm256_and_ps(neg_c, signbit));
}

}  // namespace

void evolve_rows_avx2(const SpectrumTables& tables, double time,
                      FieldSet& fields, std::uint32_t row_begin,
                      std::uint32_t row_end) noexcept
{
    const std::uint32_t n = tables.n;
    const double time_over_two_pi = time * kInvTwoPiD;

    float* p0_re = fields.re(0); float* p0_im = fields.im(0);
    float* p1_re = fields.re(1); float* p1_im = fields.im(1);
    float* p2_re = fields.re(2); float* p2_im = fields.im(2);
    float* p3_re = fields.re(3); float* p3_im = fields.im(3);

    // N is a power of two of at least 16, so rows divide evenly into vectors
    // of 8 and there is never a scalar tail to handle.
    for (std::uint32_t y = row_begin; y < row_end; ++y) {
        const std::size_t row = static_cast<std::size_t>(y) * n;

        for (std::uint32_t x = 0; x < n; x += 8) {
            const std::size_t i = row + x;

            const __m256 omega = _mm256_loadu_ps(tables.omega.data() + i);
            __m256 s, c;
            sincos8(reduced_phase8(omega, time_over_two_pi), s, c);

            const __m256 a_re = _mm256_loadu_ps(tables.h0_re.data() + i);
            const __m256 a_im = _mm256_loadu_ps(tables.h0_im.data() + i);
            const __m256 b_re = _mm256_loadu_ps(tables.h0c_re.data() + i);
            const __m256 b_im = _mm256_loadu_ps(tables.h0c_im.data() + i);

            // h(k,t) = h0 e^{iwt} + conj(h0(-k)) e^{-iwt}
            const __m256 fwd_re = _mm256_sub_ps(_mm256_mul_ps(a_re, c),
                                                _mm256_mul_ps(a_im, s));
            const __m256 fwd_im = _mm256_add_ps(_mm256_mul_ps(a_re, s),
                                                _mm256_mul_ps(a_im, c));
            const __m256 bwd_re = _mm256_add_ps(_mm256_mul_ps(b_re, c),
                                                _mm256_mul_ps(b_im, s));
            const __m256 bwd_im = _mm256_sub_ps(_mm256_mul_ps(b_im, c),
                                                _mm256_mul_ps(b_re, s));

            const __m256 hr = _mm256_add_ps(fwd_re, bwd_re);
            const __m256 hi = _mm256_add_ps(fwd_im, bwd_im);

            const __m256 kx = _mm256_loadu_ps(tables.kx.data() + i);
            const __m256 kz = _mm256_loadu_ps(tables.kz.data() + i);
            const __m256 ki = _mm256_loadu_ps(tables.k_inv.data() + i);

            const __m256 sx = _mm256_mul_ps(kx, ki);
            const __m256 sz = _mm256_mul_ps(kz, ki);

            // Negation is done by flipping the sign bit, NOT by subtracting
            // from zero. The scalar code writes `-sx * hr`, which is
            // (-sx) * hr, and IEEE gives that -0.0 when sx is +0.0 and hr is
            // positive. `0.0 - (sx*hr)` would instead give +0.0, because
            // x - x is +0 in round-to-nearest. sx IS exactly zero at the DC
            // bin, so this is a difference that actually occurs - and it would
            // show up as a one-bit mismatch in the bit-exactness test.
            const __m256 negate =
                _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));

            // The eight spectra, as in the scalar path.
            const __m256 h_re  = hr;
            const __m256 h_im  = hi;
            const __m256 dx_re = _mm256_mul_ps(sx, hi);
            const __m256 dx_im = _mm256_xor_ps(_mm256_mul_ps(sx, hr), negate);
            const __m256 dz_re = _mm256_mul_ps(sz, hi);
            const __m256 dz_im = _mm256_xor_ps(_mm256_mul_ps(sz, hr), negate);
            const __m256 hx_re = _mm256_xor_ps(_mm256_mul_ps(kx, hi), negate);
            const __m256 hx_im = _mm256_mul_ps(kx, hr);
            const __m256 hz_re = _mm256_xor_ps(_mm256_mul_ps(kz, hi), negate);
            const __m256 hz_im = _mm256_mul_ps(kz, hr);

            const __m256 jxx = _mm256_mul_ps(kx, sx);
            const __m256 jzz = _mm256_mul_ps(kz, sz);
            const __m256 jxz = _mm256_mul_ps(kx, sz);

            const __m256 jxx_re = _mm256_mul_ps(jxx, hr);
            const __m256 jxx_im = _mm256_mul_ps(jxx, hi);
            const __m256 jzz_re = _mm256_mul_ps(jzz, hr);
            const __m256 jzz_im = _mm256_mul_ps(jzz, hi);
            const __m256 jxz_re = _mm256_mul_ps(jxz, hr);
            const __m256 jxz_im = _mm256_mul_ps(jxz, hi);

            // Pack A + i*B:  re = A_re - B_im,  im = A_im + B_re.
            _mm256_storeu_ps(p0_re + i, _mm256_sub_ps(h_re, dx_im));
            _mm256_storeu_ps(p0_im + i, _mm256_add_ps(h_im, dx_re));
            _mm256_storeu_ps(p1_re + i, _mm256_sub_ps(dz_re, hx_im));
            _mm256_storeu_ps(p1_im + i, _mm256_add_ps(dz_im, hx_re));
            _mm256_storeu_ps(p2_re + i, _mm256_sub_ps(hz_re, jxx_im));
            _mm256_storeu_ps(p2_im + i, _mm256_add_ps(hz_im, jxx_re));
            _mm256_storeu_ps(p3_re + i, _mm256_sub_ps(jzz_re, jxz_im));
            _mm256_storeu_ps(p3_im + i, _mm256_add_ps(jzz_im, jxz_re));
        }
    }
}

}  // namespace ocean::detail

#endif
