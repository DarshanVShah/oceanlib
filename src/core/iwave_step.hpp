// oceanlib - internal header. Not part of the public API.
//
// The iWave per-substep hot loops. Everything here is allocation-free and
// splittable by row range so a scheduler can drive it, exactly like the FFT
// pipeline's stages.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ocean::detail {

// Values smaller than this are flushed to exactly zero at the end of each
// substep.
//
// MEASURED, NOT PRECAUTIONARY. The absorbing layer multiplies its cells by
// about 0.8 every substep, so after a few hundred steps a ring of cells is
// sitting in the denormal range - and denormal arithmetic on x86 traps to
// microcode, costing on the order of 100x per operation. It showed up as the
// interaction field at 256^2 measuring FOUR TIMES slower per cell than at
// 512^2, which is impossible for a loop whose work is exactly proportional to
// cell count. The difference was how many substeps each size had run: 512^2
// had not yet decayed into the denormal range, and 128^2 had already passed
// through it to exact zero.
//
// The alternative fix is setting FTZ/DAZ in MXCSR, and it is rejected
// deliberately: that is a process-wide CPU mode, it would change results in
// the host's own code, and it is an x86 register with no portable equivalent -
// so the scalar, SSE2, AVX2 and NEON paths would stop agreeing bit for bit and
// ADR-013's contract would quietly break. An explicit compare-and-zero is
// portable, deterministic, and identical on every path.
//
// 1e-30 m is thirty orders of magnitude below a millimetre of wave height, and
// eight orders above the float denormal threshold of 1.18e-38, so it discards
// nothing that could ever be seen and never leaves a denormal behind.
inline constexpr float kIWaveFlush = 1.0e-30f;

// One substep's worth of state, laid out for the convolution.
//
// The grid is stored with a P-cell halo on every side, so the stencil never
// needs a bounds test in the inner loop: reading (x+i, z+j) for |i|,|j| <= P is
// always in bounds. The halo holds zeros (open water outside the field), or,
// with an obstruction mask, whatever the boundary condition put there.
//
// `cur` and `old` point at interior cell (0,0), not at the allocation base.
struct IWaveGrid {
    std::uint32_t n      = 0;   // interior resolution
    std::uint32_t p      = 0;   // kernel radius
    std::size_t   stride = 0;   // floats between vertically adjacent cells

    float*       cur = nullptr;  // h^n
    float*       old = nullptr;  // h^{n-1}, overwritten with h^{n+1}
    const float* taps = nullptr; // (2P+1)^2 stencil, row-major over (j,i)

    // Per-cell damping coefficients, N*N, row-major. Precomputed because the
    // absorbing layer makes damping spatially varying, and 1/(1+alpha*dt) is a
    // divide we are not going to do per cell per substep.
    //
    //   c1 = 1 / (1 + alpha*dt)
    //   c2 = (1 - alpha*dt)
    const float* c1 = nullptr;
    const float* c2 = nullptr;

    float gdt2 = 0.0f;   // g * dt^2
};

// Advance rows [row_begin, row_end) by one substep, writing h^{n+1} into
// `old`. The caller swaps `cur` and `old` afterwards.
//
// The update is the centred discretisation of
//
//     d^2h/dt^2 + 2*alpha*dh/dt = -g * G{h}
//
// namely
//
//     h^{n+1} = [ 2h^n - (1-alpha*dt) h^{n-1} - g dt^2 G{h^n} ] / (1+alpha*dt)
//
// Second order in time, and the centred damping term is unconditionally
// dissipative for alpha >= 0 - it can never add energy, whatever the timestep.
//
// Pure function of (`cur`, `old`, coefficients, row): rows may be processed in
// any order on any thread with bit-identical results, because nothing written
// here is read here. That is the same property the FFT stages rely on, and it
// is what makes the threaded and reverse-order scheduler hooks safe.
void iwave_step_rows(const IWaveGrid& g, std::uint32_t row_begin,
                     std::uint32_t row_end) noexcept;

// The scalar reference. Every other kernel must match it to the last bit.
void iwave_step_rows_scalar(const IWaveGrid& g, std::uint32_t row_begin,
                            std::uint32_t row_end) noexcept;

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
void iwave_step_rows_avx2(const IWaveGrid& g, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept;
void iwave_step_rows_sse2(const IWaveGrid& g, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void iwave_step_rows_neon(const IWaveGrid& g, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept;
#endif

// Pack rows [row_begin, row_end) of the public RGBA-shaped output:
// (eta, dEta/dx, dEta/dz, dEta/dt).
//
// Gradients are central differences over the halo, so edge cells need no
// special case. inv_dt is 1/fixed_dt.
void iwave_finalize_rows(const IWaveGrid& g, float inv_two_cell, float inv_dt,
                         float* out, std::uint32_t row_begin,
                         std::uint32_t row_end) noexcept;

}  // namespace ocean::detail
