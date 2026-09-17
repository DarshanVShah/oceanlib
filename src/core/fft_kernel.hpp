// oceanlib - internal header. Not part of the public API.
//
// The one hot loop of the FFT, in scalar and vector forms.
#pragma once

#include "core/cpu_features.hpp"

#include <cstddef>
#include <cstdint>

namespace ocean::detail {

// Runs a single Stockham auto-sort stage:
//
//   m = len/2
//   for p in [0, m):
//       w = (twr[p*tw_step], twi[p*tw_step])
//       for q in [0, s):
//           a = x[s*p + q],  b = x[s*(p+m) + q]
//           y[s*2p     + q] = a + b
//           y[s*(2p+1) + q] = (a - b) * w
//
// Reads from x, writes to y; the caller swaps them between stages.
//
// `s` is the MEMORY stride and `tw_step` the LOGICAL one. For a single
// sequence they are equal. They diverge when several sequences are transformed
// side by side: batching B interleaved sequences multiplies every address by B
// while the twiddles, which depend only on the position within a sequence,
// stay put. Splitting the two is what lets one kernel serve both cases - and,
// because s is then always at least B, it is also what lets the batched form
// vectorise from the very first stage instead of falling back to scalar for
// s < vector width.
//
// BIT-EXACTNESS IS PART OF THE CONTRACT. Every implementation must produce
// results identical to `stage_scalar` down to the last bit, not merely close.
// That means:
//   - no FMA (a fused multiply-add keeps more intermediate precision than the
//     separate multiply and subtract the scalar path performs),
//   - no reassociation,
//   - the same operations in the same order, just several lanes at a time.
//
// The payoff is that the chosen SIMD level is NOT part of the determinism
// contract: a player on a 2006 SSE2 machine and one on a modern AVX2 machine
// see byte-identical oceans. Without this, promise #3 would quietly depend on
// which CPU the binary happened to run on. A test enforces it with memcmp.
using StageKernel = void (*)(const float* xr, const float* xi, float* yr,
                             float* yi, const float* twr, const float* twi,
                             std::uint32_t len, std::uint32_t s,
                             std::uint32_t tw_step) noexcept;

void stage_scalar(const float* xr, const float* xi, float* yr, float* yi,
                  const float* twr, const float* twi, std::uint32_t len,
                  std::uint32_t s, std::uint32_t tw_step) noexcept;

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define OCEAN_HAS_X86_KERNELS 1
void stage_sse2(const float* xr, const float* xi, float* yr, float* yi,
                const float* twr, const float* twi, std::uint32_t len,
                std::uint32_t s, std::uint32_t tw_step) noexcept;
void stage_avx2(const float* xr, const float* xi, float* yr, float* yi,
                const float* twr, const float* twi, std::uint32_t len,
                std::uint32_t s, std::uint32_t tw_step) noexcept;
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define OCEAN_HAS_NEON_KERNEL 1
void stage_neon(const float* xr, const float* xi, float* yr, float* yi,
                const float* twr, const float* twi, std::uint32_t len,
                std::uint32_t s, std::uint32_t tw_step) noexcept;
#endif

// The kernel for a given level, falling back to scalar when a build does not
// contain that architecture's kernels.
StageKernel select_stage_kernel(SimdLevel level) noexcept;

}  // namespace ocean::detail
