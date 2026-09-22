// Viewer-only: IEEE binary32 -> binary16 conversion for the texture upload.
//
// WHY THIS EXISTS. Measured per pass, handing the library's output to the GPU
// was 53% of the frame once the clipmap cut the geometry (ADR-025). It is not
// GPU work: 7 MiB of staging data per frame in 0.575 ms is 12.8 GB/s, which is
// a PCIe 4.0 x8 link running at its width, not a GPU doing anything. The only
// lever on a bandwidth wall is to send fewer bytes, and RGBA16F halves them
// exactly. See ARCHITECTURE.md ADR-026.
//
// WHY THE ALPHA CHANNEL IS FUSED IN HERE rather than patched afterwards. The
// staging buffer is HOST_VISIBLE|HOST_COHERENT, which on a discrete GPU is
// write-combined: writes are gathered in a handful of line-sized buffers and
// flushed whole. A second pass that revisits one component per texel writes
// 16 bytes of every 64-byte line, long after the first pass flushed it, so
// each line is sent twice and the second time mostly empty. Measured, that
// second pass cost MORE than converting the entire frame. Fusing it into the
// first pass is the whole fix, and it applies just as much to the fp32 path.
//
// NOTE ON DETERMINISM. The library bans FMA and ships a bit-exact scalar/
// SSE2/AVX2/NEON FFT because its OUTPUT is a promise. Nothing here is: this
// converts an already-computed frame on its way to a texture, in the viewer,
// and the two paths below agree bit for bit anyway: F16C's round-to-nearest-
// even is the rounding the scalar path implements. tools/half_check.cpp proves
// that exhaustively over all 2^32 float bit patterns rather than by assertion.
// It stays out of tests/, which is about the library and stays that way.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define VIEWER_HALF_X86 1
// CPUID and XGETBV only. The vector kernels live in half.cpp, which is the
// only translation unit compiled with a wide -march, precisely so that this
// detection can run on a CPU that has none of it.
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace viewer {

// Round-to-nearest-even, with subnormals and infinities handled. Values past
// half's range saturate to infinity rather than wrapping; nothing in a
// displacement, normal or interaction field comes near 65504, but silently
// producing a NaN there would be the kind of failure that only shows up in
// somebody else's scene.
inline std::uint16_t float_to_half(float value) noexcept
{
    std::uint32_t x;
    std::memcpy(&x, &value, sizeof x);

    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t e32  = (x >> 23) & 0xFFu;
    const std::uint32_t m32  = x & 0x007FFFFFu;

    if (e32 == 0xFFu) {  // Inf, or a NaN that must stay a NaN
        return static_cast<std::uint16_t>(sign | 0x7C00u | (m32 ? 0x0200u : 0u));
    }

    const std::int32_t e = static_cast<std::int32_t>(e32) - 127 + 15;

    if (e >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);

    if (e <= 0) {  // subnormal in half, or under even that
        if (e < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m    = m32 | 0x00800000u;  // restore the implicit 1
        const int           s    = 14 - e;             // 14..24
        std::uint32_t       h    = m >> s;
        const std::uint32_t rem  = m & ((1u << s) - 1u);
        const std::uint32_t tie  = 1u << (s - 1);
        if (rem > tie || (rem == tie && (h & 1u))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }

    // Normal. A mantissa that rounds up carries into the exponent field on its
    // own, which is why the exponent is packed before rounding rather than
    // after: 0x3FF + 1 becomes the next exponent with a zero mantissa, and an
    // exponent of 30 rounding up becomes infinity, both correctly.
    std::uint32_t       h   = (static_cast<std::uint32_t>(e) << 10) | (m32 >> 13);
    const std::uint32_t rem = m32 & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}

inline void pack_half_scalar(std::uint16_t* dst, const float* src,
                             std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = float_to_half(src[i]);
}

// One RGBA texel per `cells`: RGB from `src` (4 floats per texel, its own
// fourth component discarded), A from `alpha` (1 float per texel). `alpha` may
// be null, in which case src's own fourth component is kept.
//
// Sequential in the destination by construction, which is the entire point -
// see the note on write-combined memory above.
inline void pack_rgba16_scalar(std::uint16_t* dst, const float* src,
                               const float* alpha, std::size_t cells) noexcept
{
    if (alpha == nullptr) {
        pack_half_scalar(dst, src, cells * 4);
        return;
    }
    for (std::size_t c = 0; c < cells; ++c) {
        dst[4 * c + 0] = float_to_half(src[4 * c + 0]);
        dst[4 * c + 1] = float_to_half(src[4 * c + 1]);
        dst[4 * c + 2] = float_to_half(src[4 * c + 2]);
        dst[4 * c + 3] = float_to_half(alpha[c]);
    }
}

// The same, staying in fp32. No conversion to do, but the fusion still matters
// for exactly the same reason.
inline void pack_rgba32(float* dst, const float* src, const float* alpha,
                        std::size_t cells) noexcept
{
    if (alpha == nullptr) {
        std::memcpy(dst, src, cells * 4 * sizeof(float));
        return;
    }
    for (std::size_t c = 0; c < cells; ++c) {
        dst[4 * c + 0] = src[4 * c + 0];
        dst[4 * c + 1] = src[4 * c + 1];
        dst[4 * c + 2] = src[4 * c + 2];
        dst[4 * c + 3] = alpha[c];
    }
}

#if defined(VIEWER_HALF_X86)

// Both live in half.cpp, the one translation unit built with AVX2 and F16C
// enabled. Confining the flags there is what keeps the rest of the viewer
// runnable on a CPU that has neither: the detection below is plain CPUID in
// this header, compiled for the baseline, and nothing calls into half.cpp
// until it has answered yes.
void pack_half_f16c(std::uint16_t* dst, const float* src,
                    std::size_t count) noexcept;
void pack_rgba16_f16c(std::uint16_t* dst, const float* src, const float* alpha,
                      std::size_t cells) noexcept;

namespace detail {

inline void cpuid_count(std::uint32_t leaf, std::uint32_t sub,
                        std::uint32_t r[4]) noexcept
{
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<std::uint32_t>(regs[i]);
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

inline bool detect_avx2_f16c() noexcept
{
    std::uint32_t r[4] = {0, 0, 0, 0};
    cpuid_count(0, 0, r);
    if (r[0] < 7) return false;  // no leaf 7, so no AVX2 to ask about

    cpuid_count(1, 0, r);
    const std::uint32_t ecx = r[2];
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx     = (ecx & (1u << 28)) != 0;
    const bool f16c    = (ecx & (1u << 29)) != 0;
    if (!osxsave || !avx || !f16c) return false;

    // The instruction existing is not enough: the OS has to have enabled the
    // wide register state, or YMM is not preserved across a context switch and
    // using it corrupts other threads. Checking CPUID alone is the classic way
    // to get this wrong.
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    std::uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 =
        (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
    if ((xcr0 & 0x6ull) != 0x6ull) return false;  // XMM and YMM state saved

    cpuid_count(7, 0, r);
    return (r[1] & (1u << 5)) != 0;  // AVX2
}

}  // namespace detail

inline bool have_fast_half() noexcept
{
    static const bool yes = detail::detect_avx2_f16c();
    return yes;
}

#endif  // VIEWER_HALF_X86

// The two entry points. Dispatch is resolved once, and the branch is per call
// - seven per frame - not per texel.
inline void pack_half(std::uint16_t* dst, const float* src,
                      std::size_t count) noexcept
{
#if defined(VIEWER_HALF_X86)
    if (have_fast_half()) { pack_half_f16c(dst, src, count); return; }
#endif
    pack_half_scalar(dst, src, count);
}

inline void pack_rgba16(std::uint16_t* dst, const float* src,
                        const float* alpha, std::size_t cells) noexcept
{
#if defined(VIEWER_HALF_X86)
    if (have_fast_half()) { pack_rgba16_f16c(dst, src, alpha, cells); return; }
#endif
    pack_rgba16_scalar(dst, src, alpha, cells);
}

}  // namespace viewer
