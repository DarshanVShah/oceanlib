#include <doctest/doctest.h>

#include "core/rng.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace ocean::detail;

TEST_CASE("pcg32 reproduces the reference implementation's known answers")
{
    // Known-answer test. These are the first outputs of the canonical
    // pcg32 demo for seed=42, stream=54, re-derived independently from the
    // algorithm definition. If a refactor ever perturbs the engine, this
    // fires before any downstream test has a chance to hide it.
    const std::uint32_t expected[] = {
        0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u,
        0xbfa4784bu, 0xcbed606eu, 0xbfc6a3adu, 0x812fff6du,
    };

    Pcg32 rng = seed_pcg32(42, 54);
    for (std::uint32_t want : expected) {
        CHECK(next_u32(rng) == want);
    }
}

TEST_CASE("streams with the same seed are independent sequences")
{
    Pcg32 a = seed_pcg32(42, 54);
    Pcg32 b = seed_pcg32(42, 55);

    int matches = 0;
    for (int i = 0; i < 64; ++i) {
        if (next_u32(a) == next_u32(b)) ++matches;
    }
    // Two independent 32-bit streams agreeing even once in 64 draws is a
    // ~1.5e-8 event; any real overlap would show up as many matches.
    CHECK(matches == 0);
}

TEST_CASE("next_open01 stays strictly inside (0,1)")
{
    Pcg32 rng = seed_pcg32(7, 1);
    double lo = 1.0, hi = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const double u = next_open01(rng);
        CHECK(u > 0.0);
        CHECK(u < 1.0);
        lo = std::fmin(lo, u);
        hi = std::fmax(hi, u);
    }
    // Sanity: we really are covering the interval, not sitting in one corner.
    CHECK(lo < 0.001);
    CHECK(hi > 0.999);
}

TEST_CASE("gaussian pairs have the right first four moments")
{
    Pcg32 rng = seed_pcg32(12345, 0);
    const std::size_t n = 1u << 20;
    std::vector<float> v(n);
    fill_gaussians(rng, v.data(), n);

    double m1 = 0.0;
    for (float x : v) m1 += x;
    m1 /= static_cast<double>(n);

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (float x : v) {
        const double d = static_cast<double>(x) - m1;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);

    const double sd       = std::sqrt(m2);
    const double skew     = m3 / (sd * sd * sd);
    const double kurtosis = m4 / (m2 * m2);

    // Standard errors at n = 2^20 are ~1e-3 (mean/skew) and ~2.4e-3 (var);
    // these bounds are several sigma wide but would still catch a wrong
    // constant, a missing sqrt, or a swapped sin/cos.
    CHECK(std::abs(m1) < 0.01);
    CHECK(m2 == doctest::Approx(1.0).epsilon(0.01));
    CHECK(std::abs(skew) < 0.02);
    CHECK(kurtosis == doctest::Approx(3.0).epsilon(0.02));  // normal kurtosis
}

TEST_CASE("gaussian tails reach several sigma")
{
    // Guards against a transform that silently truncates the distribution
    // (e.g. a uniform that can return values clamped away from 0).
    Pcg32 rng = seed_pcg32(999, 3);
    const std::size_t n = 1u << 20;
    std::vector<float> v(n);
    fill_gaussians(rng, v.data(), n);

    float max_abs = 0.0f;
    std::size_t beyond3 = 0;
    for (float x : v) {
        max_abs = std::fmax(max_abs, std::fabs(x));
        if (std::fabs(x) > 3.0f) ++beyond3;
    }
    CHECK(max_abs > 4.0f);
    // P(|x| > 3) = 0.0027, so expect ~2800 of 2^20.
    CHECK(beyond3 > 2000);
    CHECK(beyond3 < 4000);
}

TEST_CASE("identical seeds produce bit-identical buffers")
{
    const std::size_t n = 4096;
    std::vector<float> a(n), b(n);

    Pcg32 ra = seed_pcg32(0xDEADBEEF, 2);
    Pcg32 rb = seed_pcg32(0xDEADBEEF, 2);
    fill_gaussians(ra, a.data(), n);
    fill_gaussians(rb, b.data(), n);

    // Bit-exact, not approximate: this is the determinism promise.
    CHECK(std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0);
}

TEST_CASE("bulk fill agrees with element-wise generation")
{
    const std::size_t n = 101;  // odd, to exercise the tail path
    std::vector<float> bulk(n), one_by_one(n);

    Pcg32 ra = seed_pcg32(5, 9);
    fill_gaussians(ra, bulk.data(), n);

    Pcg32 rb = seed_pcg32(5, 9);
    for (std::size_t i = 0; i + 1 < n; i += 2) {
        const GaussianPair g = next_gaussian_pair(rb);
        one_by_one[i]     = g.a;
        one_by_one[i + 1] = g.b;
    }
    one_by_one[n - 1] = next_gaussian_pair(rb).a;

    CHECK(std::memcmp(bulk.data(), one_by_one.data(), n * sizeof(float)) == 0);
}
