// oceanlib - internal header. Not part of the public API.
//
// In-tree complex FFT. Split (structure-of-arrays) storage, Stockham
// auto-sort radix-2.
#pragma once

#include "core/aligned.hpp"
#include "core/cpu_features.hpp"
#include "core/fft_kernel.hpp"

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
// How many adjacent columns the column pass transforms at once.
//
// The column pass walks memory with stride N. Gathering one column at a time
// touches a separate cache line per element and cannot be vectorised at all;
// gathering EIGHT adjacent columns reads 8 contiguous floats per element
// instead, so each cache line fetched is actually used.
//
// It also removes the other weakness of the single-sequence form. Batching
// multiplies the memory stride by 8, so `s` is at least 8 from the very first
// stage - the stages where s < vector width, which previously fell back to
// scalar, now vectorise like all the others.
//
// 8 suits both AVX2 (8 floats) and SSE2 (two 4-float vectors per element).
inline constexpr std::uint32_t kColumnBatch = 8;

class FftPlan {
public:
    FftPlan() = default;

    // n must be a power of two, >= 1. The SIMD level defaults to whatever the
    // running CPU supports; passing it explicitly lets tests compare every
    // kernel in the build against the scalar reference, rather than only the
    // one this particular machine happens to select.
    explicit FftPlan(std::uint32_t n);
    FftPlan(std::uint32_t n, SimdLevel level);

    [[nodiscard]] SimdLevel simd_level() const noexcept { return level_; }

    [[nodiscard]] std::uint32_t size() const noexcept { return n_; }
    [[nodiscard]] std::uint32_t log2_size() const noexcept { return log2n_; }

    // Scratch floats required by ONE caller. Concurrent callers each need
    // their own block.
    [[nodiscard]] std::size_t scratch_floats() const noexcept
    {
        // Sized for the widest user, the batched column pass: two ping-pong
        // buffers x two components x n elements x kColumnBatch lanes.
        return 4u * static_cast<std::size_t>(n_) * kColumnBatch;
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
    // Transforms `batch` interleaved sequences at once: element j of sequence
    // c lives at re[j*stride + c], for c in [0, batch). Used by the column
    // pass, where `stride` is n and the batch is a run of adjacent columns.
    void transform_batch(float* re, float* im, std::ptrdiff_t stride,
                         std::uint32_t batch, float* scratch,
                         FftSign sign) const noexcept;

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

    // Chosen once at plan construction, not per call: the dispatch is a cold
    // decision and must never sit inside the butterfly loop.
    SimdLevel   level_  = SimdLevel::Scalar;
    StageKernel kernel_ = nullptr;

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
