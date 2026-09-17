// oceanlib - internal header. Not part of the public API.
//
// In-tree complex FFT. Split (structure-of-arrays) storage, Stockham
// auto-sort radix-2.
#pragma once

#include "core/aligned.hpp"

#include <cstddef>
#include <cstdint>

namespace ocean::detail {

// Sign of the exponent in the transform kernel.
//
//   Forward: X[k] = sum_j x[j] * exp(-2*pi*i * j*k/N)
//   Inverse: x[j] = sum_k X[k] * exp(+2*pi*i * j*k/N)
//
// NEITHER direction applies a 1/N factor, so forward(inverse(x)) == N*x.
// That is deliberate: Tessendorf writes the surface as
//     h(x,t) = sum_k h~(k,t) * exp(i k.x)
// which is exactly the unnormalised inverse transform. Folding a 1/N in here
// would mean multiplying it back out in the spectrum, for no benefit.
enum class FftSign { Forward, Inverse };

// A precomputed transform of one fixed power-of-two size.
//
// "Plan" in the FFTW sense: the twiddle factors depend only on N, so they are
// computed once at init and shared by every transform. This is also what keeps
// the per-frame path allocation-free.
class FftPlan {
public:
    FftPlan() = default;

    // n must be a power of two, >= 1.
    explicit FftPlan(std::uint32_t n);

    [[nodiscard]] std::uint32_t size() const noexcept { return n_; }
    [[nodiscard]] std::uint32_t log2_size() const noexcept { return log2n_; }

    // Scratch floats required by ONE caller. Concurrent callers each need
    // their own block.
    [[nodiscard]] std::size_t scratch_floats() const noexcept
    {
        return 4u * static_cast<std::size_t>(n_);
    }

    // 1D transform in place: element j lives at re[j*stride], im[j*stride].
    //
    // The strided access is gathered into contiguous scratch first and
    // scattered back at the end, so the strided walk happens exactly twice
    // rather than once per stage. For the column pass of a 2D transform that
    // turns log2(N) strided sweeps into 2.
    void transform(float* re, float* im, std::ptrdiff_t stride, float* scratch,
                   FftSign sign) const noexcept;

    // Separable 2D transform of a row-major n x n field: rows, then columns.
    void transform_2d(float* re, float* im, float* scratch,
                      FftSign sign) const noexcept;

    // Half-transforms over an index range, so a scheduler can split the work.
    // Rows and columns are each fully independent, which is what makes the
    // parallel-for decomposition trivial - and, because each 1D transform is
    // self-contained, the result cannot depend on the order they ran in.
    void transform_row_range(float* re, float* im, std::uint32_t begin,
                             std::uint32_t end, float* scratch,
                             FftSign sign) const noexcept;
    void transform_col_range(float* re, float* im, std::uint32_t begin,
                             std::uint32_t end, float* scratch,
                             FftSign sign) const noexcept;

private:
    std::uint32_t n_     = 0;
    std::uint32_t log2n_ = 0;

    // w_j = exp(-2*pi*i * j/N) for j in [0, N/2). A single table serves every
    // stage: at the stage with sub-transform length `len` and stride `s`
    // (len*s == N always), the twiddle for butterfly p is exp(-2*pi*i*p/len),
    // which is exactly table entry p*s. The inverse table is the conjugate,
    // stored separately so the inner loop never has to negate.
    AlignedBuffer<float> tw_re_;
    AlignedBuffer<float> tw_im_fwd_;
    AlignedBuffer<float> tw_im_inv_;
};

}  // namespace ocean::detail
