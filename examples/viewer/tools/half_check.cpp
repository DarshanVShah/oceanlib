// Exhaustive check of viewer/half.hpp, over all 2^32 float bit patterns.
//
// Three claims to settle, none of them by assertion:
//
//   1. The scalar and vector paths produce IDENTICAL bits. The dispatch in
//      pack_half() is a performance choice, and a performance choice that
//      changes results is a correctness bug wearing a disguise.
//   2. Both match the hardware's own rounding for every finite input. F16C is
//      the reference here: it IS the IEEE 754 binary16 conversion.
//   3. The fused RGB+alpha kernel routes the right value to the right lane.
//      It does a cross-lane permute per vector, which is the one thing in here
//      that can be wrong in a way that still produces plausible output - a
//      foam value landing in the wrong texel is a subtle smear, not a crash.
//
// NaN payloads are compared only for NaN-ness, which is all either path
// promises and all a texture upload can use.
//
// Build: cmake --build . --target half_check.  Run: about ten seconds.
#include "../half.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::size_t kTexels = 2048;              // per block
constexpr std::size_t kFloats = kTexels * 4;

float         g_src[kFloats];
float         g_alpha[kTexels];
std::uint16_t g_a[kFloats];
std::uint16_t g_b[kFloats];

bool same(std::uint16_t x, std::uint16_t y)
{
    if (x == y) return true;
    const bool x_nan = (x & 0x7C00u) == 0x7C00u && (x & 0x03FFu);
    const bool y_nan = (y & 0x7C00u) == 0x7C00u && (y & 0x03FFu);
    return x_nan && y_nan;
}

float from_bits(std::uint32_t bits)
{
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

}  // namespace

int main()
{
#if !defined(VIEWER_HALF_X86)
    std::printf("not an x86 build: no vector path to compare against\n");
    return 0;
#else
    if (!viewer::have_fast_half()) {
        std::printf("no AVX2/F16C on this CPU: nothing to compare against\n");
        return 0;
    }

    // --- 1 and 2: the plain conversion, every bit pattern ------------------
    std::uint64_t bad = 0;
    for (std::uint64_t base = 0; base < (1ull << 32); base += kFloats) {
        for (std::size_t i = 0; i < kFloats; ++i)
            g_src[i] = from_bits(static_cast<std::uint32_t>(base + i));

        viewer::pack_half_scalar(g_a, g_src, kFloats);
        viewer::pack_half_f16c(g_b, g_src, kFloats);

        for (std::size_t i = 0; i < kFloats; ++i) {
            if (same(g_a[i], g_b[i])) continue;
            if (bad == 0) {
                std::printf("  pack_half differs at 0x%08X: scalar %04X, "
                            "f16c %04X\n",
                            static_cast<std::uint32_t>(base + i), g_a[i], g_b[i]);
            }
            ++bad;
        }
    }
    std::printf("pack_half:   4294967296 patterns, %llu mismatches\n",
                static_cast<unsigned long long>(bad));

    // --- 3: the fused kernel, same sweep, with alpha drawn from a different
    // part of the space so a lane mix-up cannot go unnoticed by coincidence.
    std::uint64_t bad_rgba = 0;
    for (std::uint64_t base = 0; base < (1ull << 32); base += kFloats) {
        for (std::size_t i = 0; i < kFloats; ++i)
            g_src[i] = from_bits(static_cast<std::uint32_t>(base + i));
        for (std::size_t c = 0; c < kTexels; ++c) {
            // A large odd stride: unrelated to the source values, and to the
            // block size, so alpha and RGB never fall into step.
            g_alpha[c] = from_bits(static_cast<std::uint32_t>(
                base + c * 2654435761ull + 0x5F3759DFull));
        }

        viewer::pack_rgba16_scalar(g_a, g_src, g_alpha, kTexels);
        viewer::pack_rgba16_f16c(g_b, g_src, g_alpha, kTexels);

        for (std::size_t i = 0; i < kFloats; ++i) {
            if (same(g_a[i], g_b[i])) continue;
            if (bad_rgba == 0) {
                std::printf("  pack_rgba16 differs at texel %zu lane %zu: "
                            "scalar %04X, f16c %04X\n",
                            i / 4, i % 4, g_a[i], g_b[i]);
            }
            ++bad_rgba;
        }
    }
    std::printf("pack_rgba16: 4294967296 patterns, %llu mismatches\n",
                static_cast<unsigned long long>(bad_rgba));

    return (bad || bad_rgba) ? 1 : 0;
#endif
}
