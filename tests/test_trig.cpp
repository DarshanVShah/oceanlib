#include <doctest/doctest.h>

#include "core/trig.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace ocean::detail;

namespace {
constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;
}  // namespace

TEST_CASE("sincos matches libm across the reduced range")
{
    // The reference runs in double so that what we measure is our polynomial's
    // error, not a contest between two float approximations.
    //
    // Range is [-2pi, 2pi] because that is exactly what evolve_rows produces:
    // std::fmod of a positive modulus keeps the sign of the dividend, so the
    // phase arrives in (-2pi, 2pi).
    double worst_sin = 0.0, worst_cos = 0.0;
    float worst_at = 0.0f;

    constexpr int samples = 2000001;
    for (int i = 0; i < samples; ++i) {
        const double t = -kTwoPi + 2.0 * kTwoPi * i / (samples - 1);
        const float x = static_cast<float>(t);

        float s = 0.0f, c = 0.0f;
        sincos_f32(x, s, c);

        const double es = std::abs(static_cast<double>(s) - std::sin(static_cast<double>(x)));
        const double ec = std::abs(static_cast<double>(c) - std::cos(static_cast<double>(x)));
        if (es > worst_sin) { worst_sin = es; worst_at = x; }
        worst_cos = std::max(worst_cos, ec);
    }

    CAPTURE(worst_sin);
    CAPTURE(worst_cos);
    CAPTURE(worst_at);
    MESSAGE("max |sin error| = " << worst_sin << ", max |cos error| = " << worst_cos);

    // Float epsilon is 1.19e-7. Anything at or below a couple of ULP is as
    // good as float can represent, and the wave amplitudes this multiplies are
    // metres, so 1e-7 of relative error is nanometres of surface.
    CHECK(worst_sin < 2.0e-7);
    CHECK(worst_cos < 2.0e-7);
}

TEST_CASE("sincos is exact at the landmark angles")
{
    struct Case { double angle; double sin_v; double cos_v; };
    const Case cases[] = {
        {0.0, 0.0, 1.0},
        {kPi / 2.0, 1.0, 0.0},
        {kPi, 0.0, -1.0},
        {-kPi / 2.0, -1.0, 0.0},
        {-kPi, 0.0, -1.0},
        {3.0 * kPi / 2.0, -1.0, 0.0},
        {kTwoPi, 0.0, 1.0},
    };

    for (const Case& c : cases) {
        CAPTURE(c.angle);
        float s = 0.0f, co = 0.0f;
        sincos_f32(static_cast<float>(c.angle), s, co);
        // The tolerance covers the error in narrowing the angle to float, not
        // just the polynomial: float(pi) differs from pi by ~8.7e-8.
        CHECK(std::abs(s - c.sin_v) < 2.0e-7);
        CHECK(std::abs(co - c.cos_v) < 2.0e-7);
    }
}

TEST_CASE("every quadrant gets the right signs")
{
    // A quadrant-fold bug is the classic failure in this algorithm, and it
    // produces a result that is still smooth and still bounded by 1 - so it
    // looks plausible and only shows up as an ocean travelling the wrong way.
    struct Q { double angle; int sin_sign; int cos_sign; };
    const Q quadrants[] = {
        {0.4,             +1, +1},  // Q1
        {kPi / 2 + 0.4,   +1, -1},  // Q2
        {kPi + 0.4,       -1, -1},  // Q3
        {3 * kPi / 2 + 0.4, -1, +1},  // Q4
        {-0.4,            -1, +1},
        {-kPi / 2 - 0.4,  -1, -1},
        {-kPi - 0.4,      +1, -1},
    };

    for (const Q& q : quadrants) {
        CAPTURE(q.angle);
        float s = 0.0f, c = 0.0f;
        sincos_f32(static_cast<float>(q.angle), s, c);
        CHECK((s > 0.0f ? 1 : -1) == q.sin_sign);
        CHECK((c > 0.0f ? 1 : -1) == q.cos_sign);
    }
}

TEST_CASE("the Pythagorean identity holds everywhere")
{
    // Independent of libm entirely: sin^2 + cos^2 must be 1 whatever the
    // reference says. This catches a sin/cos swap in one quadrant, which an
    // error-versus-libm test would also catch but which this states directly.
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const float x = static_cast<float>(-kTwoPi + 2.0 * kTwoPi * i / 199999.0);
        float s = 0.0f, c = 0.0f;
        sincos_f32(x, s, c);
        worst = std::max(worst, std::abs(static_cast<double>(s) * s +
                                         static_cast<double>(c) * c - 1.0));
    }
    CAPTURE(worst);
    CHECK(worst < 5.0e-7);
}

TEST_CASE("sincos is a pure function")
{
    // No cached state, no rounding-mode dependence, no lazy init: the same
    // input must give bit-identical output every time it is called.
    for (int i = 0; i < 1000; ++i) {
        const float x = -6.0f + 0.012f * i;
        float s1 = 0.0f, c1 = 0.0f, s2 = 0.0f, c2 = 0.0f;
        sincos_f32(x, s1, c1);
        sincos_f32(x, s2, c2);
        REQUIRE(s1 == s2);
        REQUIRE(c1 == c2);
    }
}
