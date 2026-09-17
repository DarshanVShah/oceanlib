// oceanlib - internal header. Not part of the public API.
//
// Deterministic pseudo-random numbers.
//
// We deliberately do NOT use <random>'s distributions. The C++ standard fixes
// the *engine* algorithms (std::mt19937 is bit-exact everywhere) but says
// nothing about how a *distribution* turns engine output into numbers, so
// std::normal_distribution produces different values on libstdc++, libc++ and
// MSVC. Our whole "same seed => same ocean" promise would die at the first
// platform boundary. So both the engine and the normal distribution live here.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ocean::detail {

// PCG32 (Melissa O'Neill, 2014): a 64-bit LCG whose weak high bits are fixed up
// by an output permutation - xorshift then a *data-dependent* rotate. The LCG
// alone fails statistical tests badly (low bits are nearly periodic); the
// permutation costs ~3 instructions and buys us BigCrush-quality output from
// 8 bytes of state. Compared to the alternatives:
//   - std::mt19937: 2.5 KB of state, cache-hostile, and we would still have to
//     write our own normal distribution.
//   - xorshift128+: comparable speed/quality, but PCG has cheap, exact seeking
//     and a clean multi-stream design (see `stream` below).
struct Pcg32 {
    std::uint64_t state = 0;
    std::uint64_t inc   = 1;  // must always be odd
};

// LCG multiplier from Knuth / PCG reference implementation.
inline constexpr std::uint64_t kPcgMult = 6364136223846793005ULL;

inline std::uint32_t next_u32(Pcg32& rng) noexcept
{
    const std::uint64_t old = rng.state;
    rng.state = old * kPcgMult + rng.inc;

    // XSH-RR output function: xorshift the entropy from the high bits down,
    // then rotate right by an amount taken from the top 5 bits. The rotate is
    // what defeats the lattice structure a plain LCG leaves behind.
    const std::uint32_t xorshifted =
        static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
    const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

// `seed` picks the position in the sequence, `stream` picks *which* sequence.
// Every odd increment gives a distinct period-2^64 stream, so independent
// parts of the simulation can each own a stream and stay reproducible no
// matter what order they run in - which matters once generation is threaded.
inline Pcg32 seed_pcg32(std::uint64_t seed, std::uint64_t stream = 0) noexcept
{
    Pcg32 rng;
    rng.state = 0u;
    rng.inc   = (stream << 1u) | 1u;
    next_u32(rng);
    rng.state += seed;
    next_u32(rng);
    return rng;
}

// Uniform in the OPEN interval (0,1). Mapping u -> (u + 0.5) * 2^-32 keeps the
// result strictly inside the interval, so the log() below can never see 0.
// Resolution is 2^-32; the smallest value is ~1.2e-10, whose Box-Muller radius
// is ~6.8 sigma - far past any amplitude that matters for an ocean surface.
inline double next_open01(Pcg32& rng) noexcept
{
    return (static_cast<double>(next_u32(rng)) + 0.5) * (1.0 / 4294967296.0);
}

// SplitMix64 finaliser (Steele et al. 2014). Used to derive independent
// per-row seeds from one user seed.
//
// We deliberately do NOT use PCG's `stream` parameter for that. Distinct
// streams are guaranteed to be *different* sequences, but not to be
// statistically independent - nearby increments are known to produce
// correlated output. Running a strong mixer over the seed instead scatters
// adjacent row indices to unrelated points in state space, which is what we
// actually need: visible structure correlated between grid rows would show up
// as banding in the wave field.
inline std::uint64_t mix64(std::uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ULL;  // golden-ratio odd constant
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// An independent generator for row `row` of the grid, derived from `seed`.
// Because the generator is a pure function of (seed, row), rows can be built
// in any order, on any thread, with bit-identical results.
inline Pcg32 row_rng(std::uint64_t seed, std::uint32_t row) noexcept
{
    return seed_pcg32(mix64(seed ^ (0x9E3779B97F4A7C15ULL * (row + 1u))), 0);
}

struct GaussianPair {
    float a;
    float b;
};

// Box-Muller transform: two independent uniforms on (0,1) map to two
// independent standard normals via the polar identity
//     r = sqrt(-2 ln u1),  theta = 2*pi*u2,  (r cos t, r sin t).
//
// Why the basic form and not Marsaglia polar or Ziggurat: both of those
// *reject* samples, so the number of engine draws consumed per output varies.
// Box-Muller consumes exactly two draws per two outputs, always. That makes the
// stream position a pure function of the sample index, so we can generate any
// tile of the spectrum from any thread, in any order, and get identical bits.
// It costs a log, a sqrt and a sincos, but this runs only at init, never
// per frame, so the cost is irrelevant. We compute in double and narrow on
// store: float Box-Muller loses precision in the tail where -log(u) is large.
GaussianPair next_gaussian_pair(Pcg32& rng) noexcept;

// Bulk fill of `count` standard normals (count may be odd).
void fill_gaussians(Pcg32& rng, float* dst, std::size_t count) noexcept;

}  // namespace ocean::detail
