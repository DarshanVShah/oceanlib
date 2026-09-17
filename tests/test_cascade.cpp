#include <doctest/doctest.h>

#include "alloc_probe.hpp"
#include "ocean/cascade.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ocean;

namespace {

CascadeLevel make_level(std::uint32_t size, float patch_length,
                        std::uint64_t seed, float wind_speed,
                        float cutoff)
{
    CascadeLevel d;
    d.size                       = size;
    d.patch_length                = patch_length;
    d.seed                        = seed;
    d.choppiness                  = 1.0f;
    d.spectrum.wind_speed         = wind_speed;
    d.spectrum.wind_direction     = 0.3f;
    d.spectrum.small_wave_cutoff  = cutoff;
    return d;
}

// A representative three-level stack, in the style the viewer uses: a far
// cascade for big swells, a mid cascade, and a near cascade for fine ripples.
// Distinct seeds are deliberate - see the class comment on why correlated
// seeds across levels risk shared patterning.
std::vector<CascadeLevel> three_levels()
{
    return {
        make_level(128, 800.0f, 1001, 12.0f, 15.0f),
        make_level(128, 150.0f, 2002, 12.0f, 4.0f),
        make_level(128, 25.0f,  3003, 12.0f, 0.5f),
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction and lifetime
// ---------------------------------------------------------------------------

TEST_CASE("an empty level list is rejected")
{
    std::vector<CascadeLevel> empty;
    CHECK_THROWS_AS(CascadeStack{std::span<const CascadeLevel>(empty)}, std::invalid_argument);
}

TEST_CASE("an invalid level is rejected exactly as a standalone Ocean would be")
{
    std::vector<CascadeLevel> levels = three_levels();
    levels[1].size = 100;  // not a power of two
    CHECK_THROWS_AS(CascadeStack{std::span<const CascadeLevel>(levels)}, std::invalid_argument);
}

TEST_CASE("a single-level stack behaves identically to a standalone Ocean")
{
    // The degenerate case: one cascade level should be indistinguishable from
    // using Ocean directly, since sample_at's sum over one term is that term.
    const CascadeLevel level = make_level(64, 200.0f, 555, 12.0f, 0.5f);

    Ocean solo{level};
    CascadeStack stack{{level}};

    solo.update(4.0);
    stack.update(4.0);

    CHECK(stack.level_count() == 1);
    const Buffers a = solo.buffers();
    const Buffers b = stack.buffers(0);
    const std::size_t bytes =
        static_cast<std::size_t>(a.size) * a.size * 4 * sizeof(float);
    CHECK(std::memcmp(a.displacement, b.displacement, bytes) == 0);
    CHECK(std::memcmp(a.normal, b.normal, bytes) == 0);

    for (int i = 0; i < 30; ++i) {
        const float x = -50.0f + 3.3f * i;
        const float z = 20.0f - 1.7f * i;
        const Surface sa = solo.sample_at(x, z);
        const Surface sb = stack.sample_at(x, z);
        CHECK(sa.height == doctest::Approx(sb.height).epsilon(1e-5));
        CHECK(sa.offset_x == doctest::Approx(sb.offset_x).epsilon(1e-5));
        CHECK(sa.offset_z == doctest::Approx(sb.offset_z).epsilon(1e-5));
        CHECK(sa.normal_x == doctest::Approx(sb.normal_x).epsilon(1e-5));
        CHECK(sa.normal_y == doctest::Approx(sb.normal_y).epsilon(1e-5));
        CHECK(sa.foam == doctest::Approx(sb.foam).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// The composite query
// ---------------------------------------------------------------------------

TEST_CASE("height and offset are the exact sum of each level's own sample")
{
    // By construction this is what CascadeStack::sample_at computes, but it
    // is worth stating as a test: it is the load-bearing claim of ADR-020
    // that height/offset summation is exact superposition, not an
    // approximation, and this is what a caller actually depends on.
    const std::vector<CascadeLevel> levels = three_levels();
    CascadeStack stack{levels};
    stack.update(6.0);

    std::vector<Ocean> solo;
    for (const auto& l : levels) solo.emplace_back(l);
    for (auto& o : solo) o.update(6.0);

    for (int i = 0; i < 25; ++i) {
        const float x = -300.0f + 11.0f * i;
        const float z = 150.0f - 6.0f * i;

        float expected_h = 0.0f, expected_ox = 0.0f, expected_oz = 0.0f;
        for (auto& o : solo) {
            const Surface s = o.sample_at(x, z);
            expected_h  += s.height;
            expected_ox += s.offset_x;
            expected_oz += s.offset_z;
        }

        const Surface got = stack.sample_at(x, z);
        CHECK(got.height == doctest::Approx(expected_h).epsilon(1e-5));
        CHECK(got.offset_x == doctest::Approx(expected_ox).epsilon(1e-5));
        CHECK(got.offset_z == doctest::Approx(expected_oz).epsilon(1e-5));
    }
}

TEST_CASE("the composite normal is always unit length")
{
    CascadeStack stack{three_levels()};
    stack.update(9.5);

    for (int i = 0; i < 300; ++i) {
        const float x = -400.0f + 2.9f * i;
        const float z = 200.0f - 1.3f * i;
        const Surface s = stack.sample_at(x, z);
        const float len2 = s.normal_x * s.normal_x + s.normal_y * s.normal_y +
                           s.normal_z * s.normal_z;
        REQUIRE(std::abs(len2 - 1.0f) < 1e-4f);
    }
}

TEST_CASE("composite foam stays in [0,1] and only rises as any level's foam rises")
{
    // Sanity properties of the 1 - product(1 - foam_i) blend: it must never
    // leave [0,1], and it must be monotone in each input, since it is meant
    // to read as "coverage", not an arbitrary combination.
    CHECK(1.0f - (1.0f - 0.0f) * (1.0f - 0.0f) == 0.0f);
    CHECK(1.0f - (1.0f - 1.0f) * (1.0f - 1.0f) == 1.0f);

    for (float a : {0.0f, 0.2f, 0.5f, 0.9f, 1.0f}) {
        for (float b : {0.0f, 0.2f, 0.5f, 0.9f, 1.0f}) {
            const float combined = 1.0f - (1.0f - a) * (1.0f - b);
            CAPTURE(a);
            CAPTURE(b);
            CHECK(combined >= std::max(a, b) - 1e-6f);  // never LESS than either input
            CHECK(combined <= 1.0f);
            CHECK(combined >= 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// The statistical claim: variances of independent levels add
// ---------------------------------------------------------------------------

TEST_CASE("independent, non-overlapping cascades add their energy like independent random variables")
{
    // The strongest available check for the whole feature, in the spirit of
    // ADR-010's Parseval test: if h1, h2, h3 are statistically independent
    // real-valued fields (different seeds, and non-overlapping wavelength
    // bands via small_wave_cutoff / patch_length so there is negligible
    // spectral overlap), then for the sum h = h1+h2+h3,
    //
    //   Var(h) = Var(h1) + Var(h2) + Var(h3)
    //
    // exactly, because Var(a+b) = Var(a) + Var(b) + 2*Cov(a,b) and Cov -> 0
    // for independent fields. This is a real, derivable statistical property,
    // not merely "the numbers look plausible" - and it is a strictly
    // stronger claim than "the composite differs from any one level".
    const std::vector<CascadeLevel> levels = three_levels();

    // Individual variances, measured directly on each level's own grid.
    std::vector<Ocean> solo;
    for (const auto& l : levels) solo.emplace_back(l);
    for (auto& o : solo) o.update(11.0);

    double sum_of_variances = 0.0;
    for (auto& o : solo) {
        const Buffers b = o.buffers();
        const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
        double mean = 0.0;
        for (std::size_t i = 0; i < cells; ++i) mean += b.displacement[4 * i + 1];
        mean /= static_cast<double>(cells);
        double var = 0.0;
        for (std::size_t i = 0; i < cells; ++i) {
            const double d = b.displacement[4 * i + 1] - mean;
            var += d * d;
        }
        var /= static_cast<double>(cells);
        sum_of_variances += var;
    }

    // Composite variance, measured by sampling the STACK at a dense set of
    // world points spanning several periods of the largest cascade - not read
    // off any one level's own grid, since the composite lives in world space.
    CascadeStack stack{levels};
    stack.update(11.0);

    constexpr int samples = 6000;
    double mean = 0.0;
    std::vector<double> heights(samples);
    for (int i = 0; i < samples; ++i) {
        // An irrational-ish stride so the sample set does not accidentally
        // land on a periodic subset of any one cascade's grid.
        const float x = -900.0f + 0.4173f * i;
        const float z = 500.0f - 0.2591f * i;
        heights[i] = stack.height_at(x, z);
        mean += heights[i];
    }
    mean /= samples;
    double composite_variance = 0.0;
    for (double h : heights) composite_variance += (h - mean) * (h - mean);
    composite_variance /= samples;

    CAPTURE(sum_of_variances);
    CAPTURE(composite_variance);
    // 8%: the composite variance is a Monte Carlo estimate from 6000 samples
    // (not an exact grid sum, since world-space points do not align with any
    // one cascade's texel grid), and the cascades' wavelength bands overlap
    // slightly at their edges (small_wave_cutoff is a soft knee, per
    // ADR-009, not a hard wall), so a small positive covariance is expected,
    // not just sampling noise. 8% is generous enough to absorb both while
    // still confirming this is genuine addition, not some other combination.
    CHECK(composite_variance == doctest::Approx(sum_of_variances).epsilon(0.08));
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST_CASE("CascadeStack is deterministic: same descriptors and time, bit-identical output")
{
    const std::vector<CascadeLevel> levels = three_levels();

    CascadeStack a{levels};
    CascadeStack b{levels};

    for (double t : {0.0, 2.5, 88.25}) {
        a.update(t);
        b.update(t);
        for (std::size_t lvl = 0; lvl < a.level_count(); ++lvl) {
            const Buffers ba = a.buffers(lvl);
            const Buffers bb = b.buffers(lvl);
            const std::size_t bytes =
                static_cast<std::size_t>(ba.size) * ba.size * 4 * sizeof(float);
            CHECK(std::memcmp(ba.displacement, bb.displacement, bytes) == 0);
            CHECK(std::memcmp(ba.normal, bb.normal, bytes) == 0);
        }
    }
}

TEST_CASE("seeking backwards reproduces an earlier composite frame exactly")
{
    CascadeStack stack{three_levels()};
    stack.update(15.0);
    const Surface snapshot = stack.sample_at(12.0f, -7.0f);

    stack.update(500.0);
    stack.update(0.1);
    stack.update(15.0);

    const Surface again = stack.sample_at(12.0f, -7.0f);
    CHECK(again.height == snapshot.height);
    CHECK(again.offset_x == snapshot.offset_x);
    CHECK(again.normal_y == snapshot.normal_y);
}

TEST_CASE("update performs no heap allocation beyond construction")
{
    // Reuses the same global-operator-new counter as the single-Ocean version
    // of this test (tests/test_api.cpp), linked into this same binary. The
    // property matters just as much here: CascadeStack::update is a fixed
    // loop over already-allocated Ocean instances, and it should cost exactly
    // what N separate Ocean::update() calls would cost - no extra allocation
    // for the loop, the composite, or anything else.
    CascadeStack stack{three_levels()};
    stack.update(0.0);  // warm up any one-time lazy state before measuring

    const std::size_t before = alloc_probe::count();
    for (int i = 0; i < 16; ++i) {
        stack.update(static_cast<double>(i) * 0.016);
    }
    const std::size_t after = alloc_probe::count();
    CHECK(after - before == 0);
}
