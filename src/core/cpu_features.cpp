#include "core/cpu_features.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define OCEAN_X86 1
#include <immintrin.h>
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#define OCEAN_X86 1
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define OCEAN_ARM64 1
#endif

namespace ocean::detail {
namespace {

#if defined(OCEAN_X86)

void cpuid_count(int leaf, int subleaf, int out[4]) noexcept
{
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#else
    unsigned int a, b, c, d;
    __cpuid_count(static_cast<unsigned int>(leaf),
                  static_cast<unsigned int>(subleaf), a, b, c, d);
    out[0] = static_cast<int>(a);
    out[1] = static_cast<int>(b);
    out[2] = static_cast<int>(c);
    out[3] = static_cast<int>(d);
#endif
}

unsigned long long read_xcr0() noexcept
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

bool cpu_has_avx2() noexcept
{
    int info[4];
    cpuid_count(0, 0, info);
    if (info[0] < 7) return false;

    cpuid_count(1, 0, info);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx     = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    // The OSXSAVE/XGETBV check is not optional paranoia. A CPU can report AVX2
    // support while the operating system does not save and restore the upper
    // halves of the YMM registers on a context switch. Using them then silently
    // corrupts state across a preemption - a bug that appears as rare,
    // irreproducible garbage. Bits 1 and 2 of XCR0 are the XMM and YMM state.
    const unsigned long long xcr0 = read_xcr0();
    if ((xcr0 & 0x6ULL) != 0x6ULL) return false;

    cpuid_count(7, 0, info);
    return (info[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
}

#endif  // OCEAN_X86

SimdLevel compute_max_level() noexcept
{
#if defined(OCEAN_X86)
    if (cpu_has_avx2()) return SimdLevel::Avx2;
    // Every x86-64 CPU has SSE2; it is part of the base architecture, so no
    // detection is needed and no 32-bit-only fallback is required here.
#if defined(_M_X64) || defined(__x86_64__)
    return SimdLevel::Sse2;
#else
    return SimdLevel::Scalar;
#endif
#elif defined(OCEAN_ARM64)
    // NEON (ASIMD) is mandatory on AArch64, so it needs no runtime check.
    return SimdLevel::Neon;
#else
    return SimdLevel::Scalar;
#endif
}

SimdLevel& active_level() noexcept
{
    static SimdLevel level = compute_max_level();
    return level;
}

}  // namespace

SimdLevel max_simd_level() noexcept
{
    static const SimdLevel cached = compute_max_level();
    return cached;
}

SimdLevel detect_simd_level() noexcept { return active_level(); }

SimdLevel force_simd_level(SimdLevel level) noexcept
{
    const SimdLevel cap = max_simd_level();

    // Never install a level the CPU cannot execute: the caller is asking for a
    // test configuration, not permission to fault.
    auto rank = [](SimdLevel l) {
        switch (l) {
            case SimdLevel::Scalar: return 0;
            case SimdLevel::Sse2:   return 1;
            case SimdLevel::Neon:   return 1;
            case SimdLevel::Avx2:   return 2;
        }
        return 0;
    };
    if (rank(level) > rank(cap)) level = cap;

    // An x86 machine cannot run the NEON kernel and vice versa.
    if (level == SimdLevel::Neon && cap != SimdLevel::Neon) level = SimdLevel::Scalar;
    if (level == SimdLevel::Sse2 && cap == SimdLevel::Neon) level = SimdLevel::Scalar;

    active_level() = level;
    return level;
}

const char* simd_level_name(SimdLevel level) noexcept
{
    switch (level) {
        case SimdLevel::Scalar: return "scalar";
        case SimdLevel::Sse2:   return "SSE2";
        case SimdLevel::Avx2:   return "AVX2";
        case SimdLevel::Neon:   return "NEON";
    }
    return "unknown";
}

}  // namespace ocean::detail
