#include "core/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocean::detail {

FftPlan::FftPlan(std::uint32_t n) : FftPlan(n, detect_simd_level()) {}

FftPlan::FftPlan(std::uint32_t n, SimdLevel level)
    : n_(n), level_(level), kernel_(select_stage_kernel(level))
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
    // q - which is the property that lets the SIMD kernels load whole vectors.
    //
    // The per-stage work lives behind `kernel_`, chosen once when the plan was
    // built. Every kernel is required to be bit-identical to the scalar one,
    // so this dispatch changes speed and nothing else.
    for (std::uint32_t len = n, s = 1; len > 1; len >>= 1, s <<= 1) {
        kernel_(xr, xi, yr, yi, twr, twi, len, s, s);
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

void FftPlan::transform_batch(float* re, float* im, std::ptrdiff_t stride,
                              std::uint32_t batch, float* scratch,
                              FftSign sign) const noexcept
{
    const std::uint32_t n = n_;
    const std::size_t   lanes = static_cast<std::size_t>(batch);
    const std::size_t   span  = static_cast<std::size_t>(n) * lanes;

    float* a_re = scratch;
    float* a_im = scratch + span;
    float* b_re = scratch + 2u * span;
    float* b_im = scratch + 3u * span;

    // Gather: element j of every sequence in the batch, side by side. Each
    // inner run is contiguous in memory, so a whole cache line gets used
    // instead of one float out of sixteen.
    for (std::uint32_t j = 0; j < n; ++j) {
        const float* src_re = re + static_cast<std::ptrdiff_t>(j) * stride;
        const float* src_im = im + static_cast<std::ptrdiff_t>(j) * stride;
        float* dst_re = a_re + static_cast<std::size_t>(j) * lanes;
        float* dst_im = a_im + static_cast<std::size_t>(j) * lanes;
        for (std::uint32_t c = 0; c < batch; ++c) {
            dst_re[c] = src_re[c];
            dst_im[c] = src_im[c];
        }
    }

    const float* twr = tw_re_.data();
    const float* twi = (sign == FftSign::Forward) ? tw_im_fwd_.data()
                                                  : tw_im_inv_.data();

    float* xr = a_re;
    float* xi = a_im;
    float* yr = b_re;
    float* yi = b_im;

    // Identical Stockham schedule, with every address scaled by `batch`. The
    // twiddle step stays unscaled because a twiddle depends only on the
    // position within a sequence, not on which lane of the batch it is.
    for (std::uint32_t len = n, s = 1; len > 1; len >>= 1, s <<= 1) {
        kernel_(xr, xi, yr, yi, twr, twi, len, s * batch, s);
        std::swap(xr, yr);
        std::swap(xi, yi);
    }

    for (std::uint32_t j = 0; j < n; ++j) {
        float* dst_re = re + static_cast<std::ptrdiff_t>(j) * stride;
        float* dst_im = im + static_cast<std::ptrdiff_t>(j) * stride;
        const float* src_re = xr + static_cast<std::size_t>(j) * lanes;
        const float* src_im = xi + static_cast<std::size_t>(j) * lanes;
        for (std::uint32_t c = 0; c < batch; ++c) {
            dst_re[c] = src_re[c];
            dst_im[c] = src_im[c];
        }
    }
}

void FftPlan::transform_col_range(float* re, float* im, std::uint32_t begin,
                                  std::uint32_t end, float* scratch,
                                  FftSign sign) const noexcept
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(n_);

    std::uint32_t c = begin;
    for (; c + kColumnBatch <= end; c += kColumnBatch) {
        transform_batch(re + c, im + c, n, kColumnBatch, scratch, sign);
    }
    // Remainder, when the caller's range is not a multiple of the batch.
    // Handled by the same batched routine with a smaller batch rather than by
    // a second code path, so there is only one thing to get right.
    if (c < end) {
        transform_batch(re + c, im + c, n, end - c, scratch, sign);
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
