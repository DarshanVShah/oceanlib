// oceanlib - internal header. Not part of the public API.
//
// Runtime CPU feature detection, so one binary runs everywhere: the library
// ships with SSE2, AVX2 and NEON kernels and picks at startup. Promise #1 is
// "runs on anything", which means we cannot require the user to build separate
// binaries per instruction set.
#pragma once

namespace ocean::detail {

enum class SimdLevel {
    Scalar,  // portable fallback; also the correctness reference
    Sse2,    // x86-64 baseline, always available there
    Avx2,    // 8-wide float
    Neon,    // AArch64 baseline, always available there
};

// Detected once, cached. Safe to call repeatedly.
SimdLevel detect_simd_level() noexcept;

const char* simd_level_name(SimdLevel level) noexcept;

// Forces a level for testing, so the test suite can compare every kernel the
// build contains against the scalar reference rather than only the one this
// particular machine happens to select. Returns the level actually installed,
// which may be lower than requested if the CPU does not support it.
SimdLevel force_simd_level(SimdLevel level) noexcept;

// Highest level this CPU supports, ignoring any override.
SimdLevel max_simd_level() noexcept;

}  // namespace ocean::detail
