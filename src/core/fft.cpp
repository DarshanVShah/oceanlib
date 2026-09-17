#include "core/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocean::detail {

FftPlan::FftPlan(std::uint32_t n) : n_(n)
{
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("FftPlan: size must be a power of two");
    }
    while ((1u << log2n_) < n) ++log2n_;

    const std::size_t half = std::max<std::size_t>(n / 2, 1);
    tw_re_     = AlignedBuffer<float>(half);
    tw_im_fwd_ = AlignedBuffer<float>(half);
    tw_im_inv_ = AlignedBuffer<float>(half);

    constexpr double kTwoPi = 6.283185307179586476925286766559;
    for (std::size_t j = 0; j < n / 2; ++j) {
        // Computed in double and narrowed on store. The table is built once,
        // so the extra precision is free, and it keeps the twiddles correctly
        // rounded rather than accumulating error from a recurrence.
        const double angle = kTwoPi * static_cast<double>(j) / static_cast<double>(n);
        tw_re_[j]     = static_cast<float>(std::cos(angle));
        tw_im_fwd_[j] = static_cast<float>(-std::sin(angle));
        tw_im_inv_[j] = static_cast<float>(std::sin(angle));
    }
}

void FftPlan::transform(float* re, float* im, std::ptrdiff_t stride,
                        float* scratch, FftSign sign) const noexcept
{
    const std::uint32_t n = n_;

    float* a_re = scratch;
    float* a_im = scratch + n;
    float* b_re = scratch + 2u * static_cast<std::size_t>(n);
    float* b_im = scratch + 3u * static_cast<std::size_t>(n);

    for (std::uint32_t j = 0; j < n; ++j) {
        a_re[j] = re[static_cast<std::ptrdiff_t>(j) * stride];
        a_im[j] = im[static_cast<std::ptrdiff_t>(j) * stride];
    }

    const float* twr = tw_re_.data();
    const float* twi = (sign == FftSign::Forward) ? tw_im_fwd_.data()
                                                  : tw_im_inv_.data();

    float* xr = a_re;
    float* xi = a_im;
    float* yr = b_re;
    float* yi = b_im;

    // Stockham auto-sort. `len` is the length of the sub-transform being
    // combined and `s` the distance between its elements; len*s == n at every
    // stage. Reading from x and writing to y (then swapping) is what removes
    // the bit-reversal pass: the permutation is folded into the write indices
    // 2p and 2p+1, and both the read and the write walk memory sequentially in
    // q - which is the property that lets a SIMD version load whole vectors.
    for (std::uint32_t len = n, s = 1; len > 1; len >>= 1, s <<= 1) {
        const std::uint32_t m = len >> 1;
        for (std::uint32_t p = 0; p < m; ++p) {
            const float wr = twr[static_cast<std::size_t>(p) * s];
            const float wi = twi[static_cast<std::size_t>(p) * s];

            const std::uint32_t src0 = s * p;
            const std::uint32_t src1 = s * (p + m);
            const std::uint32_t dst0 = s * (2u * p);
            const std::uint32_t dst1 = s * (2u * p + 1u);

            for (std::uint32_t q = 0; q < s; ++q) {
                const float ar = xr[src0 + q];
                const float ai = xi[src0 + q];
                const float br = xr[src1 + q];
                const float bi = xi[src1 + q];

                // Sum leg needs no twiddle; difference leg is rotated by w.
                yr[dst0 + q] = ar + br;
                yi[dst0 + q] = ai + bi;

                const float dr = ar - br;
                const float di = ai - bi;
                yr[dst1 + q] = dr * wr - di * wi;
                yi[dst1 + q] = dr * wi + di * wr;
            }
        }
        std::swap(xr, yr);
        std::swap(xi, yi);
    }

    // After log2(n) swaps the result sits in x, whichever physical buffer that
    // now is. The scatter below absorbs the parity, so no copy-back is needed.
    for (std::uint32_t j = 0; j < n; ++j) {
        re[static_cast<std::ptrdiff_t>(j) * stride] = xr[j];
        im[static_cast<std::ptrdiff_t>(j) * stride] = xi[j];
    }
}

void FftPlan::transform_row_range(float* re, float* im, std::uint32_t begin,
                                  std::uint32_t end, float* scratch,
                                  FftSign sign) const noexcept
{
    const std::size_t n = n_;
    for (std::uint32_t r = begin; r < end; ++r) {
        const std::size_t off = static_cast<std::size_t>(r) * n;
        transform(re + off, im + off, 1, scratch, sign);
    }
}

void FftPlan::transform_col_range(float* re, float* im, std::uint32_t begin,
                                  std::uint32_t end, float* scratch,
                                  FftSign sign) const noexcept
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(n_);
    for (std::uint32_t c = begin; c < end; ++c) {
        transform(re + c, im + c, n, scratch, sign);
    }
}

void FftPlan::transform_2d(float* re, float* im, float* scratch,
                           FftSign sign) const noexcept
{
    // A 2D DFT is separable: exp(-2pi i (kx*x + ky*y)/N) factorises into
    // exp(-2pi i kx*x/N) * exp(-2pi i ky*y/N). So N row transforms followed by
    // N column transforms give the full 2D result, at O(N^2 log N) instead of
    // the O(N^4) of the direct double sum.
    transform_row_range(re, im, 0, n_, scratch, sign);
    transform_col_range(re, im, 0, n_, scratch, sign);
}

}  // namespace ocean::detail
