// Orbital velocity of the FFT surface.
//
// The derivation solves the dynamic boundary condition per branch and claims
// three spectra. Every claim below is checked against something independent of
// that derivation: a finite difference of the height field, a textbook
// identity about deep-water orbits, or a bit-for-bit comparison with the
// velocity feature switched off.
#include "ocean/cascade.hpp"
#include "ocean/ocean.hpp"

#include "alloc_probe.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ocean;

namespace {

OceanDesc velocity_desc(std::uint32_t n = 128)
{
    OceanDesc d;
    d.size                = n;
    d.patch_length        = 200.0f;
    d.spectrum.wind_speed = 12.0f;
    d.choppiness          = 1.0f;
    d.seed                = 4242;
    d.compute_velocity    = true;
    return d;
}

}  // namespace

TEST_CASE("velocity is absent unless asked for, and costs nothing when absent")
{
    OceanDesc d = velocity_desc();
    d.compute_velocity = false;
    Ocean sim{d};
    sim.update(3.0);

    CHECK(sim.buffers().velocity == nullptr);

    const Surface s = sim.sample_at(12.0f, -7.0f);
    CHECK(s.velocity_x == 0.0f);
    CHECK(s.velocity_y == 0.0f);
    CHECK(s.velocity_z == 0.0f);
}

TEST_CASE("enabling velocity does not change the surface by a single bit")
{
    // The load-bearing compatibility test. The velocity spectra are computed in
    // a SECOND pass precisely so that the existing evolve kernel - including
    // its AVX2 path - is untouched. If that separation were imperfect, every
    // existing integration would silently render a different ocean the moment
    // someone switched velocity on.
    OceanDesc off = velocity_desc();
    off.compute_velocity = false;
    OceanDesc on = velocity_desc();

    Ocean a{off};
    Ocean b{on};
    for (double t : {0.0, 1.5, 40.0, 1000.0}) {
        a.update(t);
        b.update(t);
        const Buffers ba = a.buffers(), bb = b.buffers();
        const std::size_t bytes =
            static_cast<std::size_t>(ba.size) * ba.size * 4 * sizeof(float);
        INFO("t = " << t);
        REQUIRE(std::memcmp(ba.displacement, bb.displacement, bytes) == 0);
        REQUIRE(std::memcmp(ba.normal, bb.normal, bytes) == 0);
    }
}

TEST_CASE("vertical velocity equals dh/dt, the kinematic boundary condition")
{
    // THE test for the derivation. u_y is claimed to be i*omega*(A-B); the
    // kinematic boundary condition says it must also be dh/dt. Those are
    // derived from different equations, so agreeing is a real check rather
    // than a tautology - and it is exactly what a sign error on one branch
    // would break.
    //
    // Checked against a CENTRAL DIFFERENCE of the rendered height field, which
    // never touches the velocity code path at all.
    Ocean sim{velocity_desc(128)};
    const double t = 21.0, dt = 0.004;

    std::vector<float> h_plus, h_minus;
    sim.update(t + dt);
    {
        const Buffers b = sim.buffers();
        h_plus.assign(b.displacement, b.displacement + static_cast<std::size_t>(b.size) * b.size * 4);
    }
    sim.update(t - dt);
    {
        const Buffers b = sim.buffers();
        h_minus.assign(b.displacement, b.displacement + static_cast<std::size_t>(b.size) * b.size * 4);
    }
    sim.update(t);

    const Buffers b = sim.buffers();
    REQUIRE(b.velocity != nullptr);

    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(b.size) * b.size; ++i) {
        const double fd = (static_cast<double>(h_plus[4 * i + 1]) -
                           static_cast<double>(h_minus[4 * i + 1])) / (2.0 * dt);
        const double uy = b.velocity[4 * i + 1];
        num += (fd - uy) * (fd - uy);
        den += fd * fd;
        worst = std::max(worst, std::fabs(fd - uy));
    }
    const double rel = std::sqrt(num / den);
    std::printf("\n  u_y vs central difference of h: RMS relative error %.3e, "
                "worst absolute %.3e m/s\n", rel, worst);

    // The residual is the O(dt^2) truncation of the finite difference, not an
    // error in u_y.
    CHECK(rel < 1e-3);
}

TEST_CASE("horizontal velocity matches the spatial gradient relationship")
{
    // u_x is claimed to be -omega*(kx/|k|)*D while u_y is i*omega*D, so
    // u_x = i*(kx/|k|)*u_y. In real space that is the same operator that turns
    // height into Tessendorf's horizontal displacement - which means the
    // horizontal velocity field must be, to within a constant, the same shape
    // as the displacement field is to the height field.
    //
    // Rather than rebuild that operator, this checks the consequence that is
    // easiest to state and hardest to fake: deep-water orbits are CIRCULAR, so
    // the horizontal and vertical orbital speeds have equal variance.
    Ocean sim{velocity_desc(256)};
    sim.update(17.0);
    const Buffers b = sim.buffers();
    REQUIRE(b.velocity != nullptr);

    double hx = 0.0, hy = 0.0, hz = 0.0;
    const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
    for (std::size_t i = 0; i < cells; ++i) {
        hx += static_cast<double>(b.velocity[4 * i + 0]) * b.velocity[4 * i + 0];
        hy += static_cast<double>(b.velocity[4 * i + 1]) * b.velocity[4 * i + 1];
        hz += static_cast<double>(b.velocity[4 * i + 2]) * b.velocity[4 * i + 2];
    }
    hx /= cells; hy /= cells; hz /= cells;

    const double horiz = hx + hz;
    std::printf("  orbital velocity variance: horizontal %.5f, vertical %.5f, "
                "ratio %.4f (circular orbits => 1)\n", horiz, hy, horiz / hy);

    // For deep water the orbit is a circle, so <u_x^2 + u_z^2> = <u_y^2>
    // exactly in the continuum. This is a textbook identity, and it holds here
    // for the same reason Parseval does in ADR-010: it is an exact statement
    // about the spectrum, not a statistical one.
    CHECK(horiz / hy == doctest::Approx(1.0).epsilon(0.02));

    // And the flow must actually be going somewhere.
    CHECK(hy > 1e-4);
}

TEST_CASE("queries agree with the velocity buffer")
{
    OceanDesc d = velocity_desc(128);
    d.choppiness = 0.0f;   // no inversion, so a query maps straight to a texel
    Ocean sim{d};
    sim.update(9.0);

    const Buffers b = sim.buffers();
    const float cell = b.patch_length / static_cast<float>(b.size);

    double worst = 0.0;
    for (std::uint32_t z = 10; z < 110; z += 7) {
        for (std::uint32_t x = 10; x < 110; x += 5) {
            const Surface s = sim.sample_at(x * cell, z * cell);
            const std::size_t i = static_cast<std::size_t>(z) * b.size + x;
            worst = std::max(worst,
                             std::fabs(static_cast<double>(s.velocity_y) -
                                       b.velocity[4 * i + 1]));
        }
    }
    std::printf("  velocity query vs buffer: worst %.3e m/s\n", worst);
    CHECK(worst < 1e-5);
}

TEST_CASE("velocity is deterministic and allocation-free")
{
    Ocean a{velocity_desc(128)};
    Ocean b{velocity_desc(128)};
    a.update(55.0);
    b.update(55.0);
    const std::size_t bytes =
        static_cast<std::size_t>(a.buffers().size) * a.buffers().size * 4 * sizeof(float);
    CHECK(std::memcmp(a.buffers().velocity, b.buffers().velocity, bytes) == 0);

    // Seeking backwards reproduces exactly, like the rest of the surface.
    a.update(12.0);
    a.update(55.0);
    CHECK(std::memcmp(a.buffers().velocity, b.buffers().velocity, bytes) == 0);

    const std::size_t before = alloc_probe::count();
    for (int i = 0; i < 20; ++i) a.update(60.0 + i * 0.1);
    CHECK(alloc_probe::count() == before);
}

TEST_CASE("cascade velocity sums exactly across levels")
{
    // Unlike normals and foam, velocity is an exact linear superposition: each
    // level's velocity field is an independent real vector field over world
    // position, so the composite flow is their sum with nothing to renormalise.
    std::vector<OceanDesc> levels;
    for (int i = 0; i < 3; ++i) {
        OceanDesc d = velocity_desc(128);
        d.patch_length = 400.0f / (1.0f + static_cast<float>(i) * 2.0f);
        d.seed         = 900 + static_cast<std::uint64_t>(i);
        levels.push_back(d);
    }
    CascadeStack stack{std::span<const OceanDesc>(levels)};
    stack.update(6.0);

    std::vector<Ocean> singles;
    singles.reserve(3);
    for (const auto& d : levels) singles.emplace_back(d);
    for (auto& o : singles) o.update(6.0);

    double worst = 0.0;
    for (float x = -80.0f; x < 80.0f; x += 13.7f) {
        for (float z = -80.0f; z < 80.0f; z += 11.3f) {
            const Surface c = stack.sample_at(x, z);
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            for (auto& o : singles) {
                const Surface s = o.sample_at(x, z);
                sx += s.velocity_x; sy += s.velocity_y; sz += s.velocity_z;
            }
            worst = std::max({worst,
                              std::fabs(static_cast<double>(c.velocity_x - sx)),
                              std::fabs(static_cast<double>(c.velocity_y - sy)),
                              std::fabs(static_cast<double>(c.velocity_z - sz))});
        }
    }
    std::printf("  cascade velocity vs sum of levels: worst %.3e m/s\n", worst);
    CHECK(worst < 1e-5);
}
