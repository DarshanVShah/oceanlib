// oceanlib - internal header. Not part of the public API.
//
// The per-frame path: evolve the spectrum to time t, and turn the transformed
// fields into the output buffers. Everything here is allocation-free and
// splittable by row range so a scheduler can drive it.
#pragma once

#include "core/aligned.hpp"
#include "core/spectrum.hpp"

#include <cstdint>

namespace ocean::detail {

// The four complex fields that one frame needs.
//
// Each holds two real output fields packed as real + i*imaginary, exploiting
// the fact that every one of our eight spectra is Hermitian and therefore
// transforms to a purely real field:
//
//   field 0:  h      + i * Dx          (height, x displacement)
//   field 1:  Dz     + i * dh/dx       (z displacement, x slope)
//   field 2:  dh/dz  + i * dDx/dx      (z slope, x-x displacement gradient)
//   field 3:  dDz/dz + i * dDx/dz      (remaining displacement gradients)
//
// Eight fields, four transforms.
//
// Note there is no fourth displacement gradient: dDz/dx equals dDx/dz, because
// the horizontal displacement is the gradient of a potential, so its Jacobian
// matrix is symmetric. That symmetry saves a whole extra field for free.
struct FieldSet {
    std::uint32_t n     = 0;
    std::size_t   cells = 0;

    // One allocation for all eight planes, so the sub-arrays are contiguous
    // and the whole working set can be prefetched as a unit. Each plane stays
    // 64-byte aligned because cells is a multiple of 16 for every legal N.
    AlignedBuffer<float> storage;

    // Number of COMPLEX fields: 4 normally, 6 when orbital velocity is on.
    int complex_fields = 4;

    void allocate(std::uint32_t size, bool with_velocity = false);

    [[nodiscard]] float* re(int f) noexcept
    {
        return storage.data() + static_cast<std::size_t>(2 * f) * cells;
    }
    [[nodiscard]] float* im(int f) noexcept
    {
        return storage.data() + static_cast<std::size_t>(2 * f + 1) * cells;
    }
    [[nodiscard]] const float* re(int f) const noexcept
    {
        return storage.data() + static_cast<std::size_t>(2 * f) * cells;
    }
    [[nodiscard]] const float* im(int f) const noexcept
    {
        return storage.data() + static_cast<std::size_t>(2 * f + 1) * cells;
    }
};

// Advance the spectrum to absolute time `time` and write the four packed
// complex fields, for rows [row_begin, row_end).
//
// Pure function of (tables, time, row): rows may be processed in any order on
// any thread with bit-identical results.
void evolve_rows(const SpectrumTables& tables, double time, FieldSet& fields,
                 std::uint32_t row_begin, std::uint32_t row_end) noexcept;

// The individual kernels, exposed so tests can compare them directly rather
// than only through whichever one this machine happens to select.
void evolve_rows_scalar(const SpectrumTables& tables, double time,
                        FieldSet& fields, std::uint32_t row_begin,
                        std::uint32_t row_end) noexcept;

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
void evolve_rows_avx2(const SpectrumTables& tables, double time,
                      FieldSet& fields, std::uint32_t row_begin,
                      std::uint32_t row_end) noexcept;
#endif

// Write the two extra velocity spectra into fields 4 and 5, for rows
// [row_begin, row_end).
//
// A SECOND PASS over the same rows, rather than an extension of evolve_rows.
// It costs a second argument reduction and sincos pair per cell, which is not
// free - but it leaves the existing AVX2 evolve kernel completely untouched,
// so enabling velocity cannot perturb a single bit of the surface an existing
// integration already renders. Fusing the two and writing the AVX2 velocity
// kernel is the obvious optimisation, and it is worth doing only once
// profiling shows a real scene is limited by it (the standing practice in
// ADR-020).
//
// Scalar only for now, so bit-exactness across kernels is trivially satisfied
// while the vector version does not exist.
void evolve_velocity_rows(const SpectrumTables& tables, double time,
                          FieldSet& fields, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept;

// Unpack the transformed velocity fields into the public interleaved buffer.
void finalize_velocity_rows(const FieldSet& fields, float* velocity,
                            std::uint32_t row_begin,
                            std::uint32_t row_end) noexcept;

// Turn the four transformed fields into the public interleaved buffers, for
// rows [row_begin, row_end). Computes displacement, the exact normal of the
// displaced surface, the Jacobian and the foam term.
void finalize_rows(const FieldSet& fields, float choppiness,
                   float foam_threshold, float* displacement, float* normal,
                   std::uint32_t row_begin, std::uint32_t row_end) noexcept;

}  // namespace ocean::detail
