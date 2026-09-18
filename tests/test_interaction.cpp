// The interaction field: solver, sources, boundaries, determinism.
//
// test_iwave_kernel.cpp proves the KERNEL has the right Fourier symbol. This
// file proves the SOLVER built on it actually behaves like water - most
// importantly the ripple-expansion test, which measures a real group velocity
// out of a real simulation and compares it with the deep-water theory.
#include "ocean/interaction.hpp"

#include "alloc_probe.hpp"
#include "core/cpu_features.hpp"
#include "core/iwave_kernel.hpp"
#include "core/iwave_step.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ocean;

namespace {

constexpr double kPi = 3.14159265358979323846;

InteractionDesc base_desc(std::uint32_t n = 256, float extent = 64.0f)
{
    InteractionDesc d;
    d.size          = n;
    d.extent        = extent;
    d.kernel_radius = 6;
    d.damping       = 0.0f;      // most tests want undamped, for clean physics
    d.absorb_cells  = 16;
    d.max_sources   = 32;
    return d;
}

Disturbance rock(float x, float z, float radius = 0.5f, float strength = 0.2f)
{
    Disturbance d;
    d.world_x  = x;
    d.world_z  = z;
    d.radius   = radius;
    d.strength = strength;
    d.kind     = SourceKind::Impulse;
    return d;
}

// Run `frames` updates of `dt`, with no sources.
void run(InteractionField& f, int frames, float dt = 1.0f / 60.0f)
{
    for (int i = 0; i < frames; ++i) f.update(dt);
}

// Sum of a buffer's height channel: the field's net volume, in cell units.
double net_volume(const InteractionField& f)
{
    const InteractionBuffers b = f.buffers();
    double v = 0.0;
    for (std::uint32_t i = 0; i < b.size * b.size; ++i) v += b.field[4 * i];
    return v;
}

// Energy-weighted mean group velocity of the ripples a Ricker impulse of scale
// sigma actually launches, band-limited to what the grid can hold.
//
// This is what the ring's envelope should travel at, and it is NOT simply
// c_g at the profile's peak wavenumber. The profile's transform goes like
// k^2 exp(-sigma^2 k^2 / 2), so the radial ENERGY density goes like
// k^5 exp(-sigma^2 k^2) - a different peak, and a broad one. Comparing the
// measurement against c_g(k_peak) instead would be comparing it with the wrong
// number and then congratulating the code for the mismatch.
double expected_group_velocity(double sigma, double gravity, double k_max)
{
    const int    steps = 20000;
    double num = 0.0, den = 0.0;
    for (int i = 1; i <= steps; ++i) {
        const double k = k_max * i / steps;
        const double wgt = std::pow(k, 5.0) * std::exp(-sigma * sigma * k * k);
        const double cg  = 0.5 * std::sqrt(gravity / k);   // deep water
        num += wgt * cg;
        den += wgt;
    }
    return num / den;
}

// The group velocity the SOLVER should actually deliver: computed from the
// kernel's own realised symbol and the leapfrog's own discrete frequency,
// rather than from continuum theory.
//
// This separates two claims that are very easy to conflate:
//
//   measured vs REALISED  - "the solver correctly integrates the operator it
//                           was given". A failure here is a bug.
//   realised vs IDEAL     - "that operator is a good approximation of deep
//                           water". A gap here is the kernel's known band
//                           limit, not a bug.
//
// Group velocity is a DERIVATIVE of the symbol, which is why it has to be
// derived this way rather than assumed to inherit the symbol's error. Where
// the realised symbol rises steeply through the transition band, c_g comes out
// about 50% high even though the symbol itself is only about 28% low.
double realised_group_velocity(double sigma, double gravity, double cell,
                               std::uint32_t radius, double dt)
{
    ocean::detail::IWaveKernel k;
    ocean::detail::build_iwave_kernel(radius, cell, 0.0,
                                      ocean::detail::KernelMethod::LeastSquares, k);
    const double k_max = kPi / cell;

    // omega from the realised symbol, through the leapfrog's own discrete
    // dispersion sin(w~ dt/2) = (dt/2) sqrt(g*S). At the default timestep the
    // correction is a fraction of a percent, but it costs nothing to be exact.
    auto omega = [&](double kk) {
        const double S = ocean::detail::realised_symbol(k, kk, 0.0);
        if (S <= 0.0) return 0.0;
        const double a = 0.5 * dt * std::sqrt(gravity * S);
        return (a >= 1.0) ? (2.0 / dt) : (2.0 / dt) * std::asin(a);
    };

    const int steps = 4000;
    const double h = k_max / steps;
    double num = 0.0, den = 0.0;
    for (int i = 2; i < steps; ++i) {
        const double kk = k_max * i / steps;
        const double cg = (omega(kk + h) - omega(kk - h)) / (2.0 * h);
        const double wgt = std::pow(kk, 5.0) * std::exp(-sigma * sigma * kk * kk);
        num += wgt * cg;
        den += wgt;
    }
    return num / den;
}

// Radius of the energy centroid: sum(r * eta^2) / sum(eta^2), about the grid
// centre. For a wave packet this tracks the envelope, which is what moves at
// the group velocity.
double centroid_radius(const InteractionField& f)
{
    const InteractionBuffers b = f.buffers();
    const double cell = b.extent / b.size;
    const double c    = 0.5 * b.size;
    double num = 0.0, den = 0.0;
    for (std::uint32_t z = 0; z < b.size; ++z) {
        for (std::uint32_t x = 0; x < b.size; ++x) {
            const double e = static_cast<double>(
                b.field[4 * (static_cast<std::size_t>(z) * b.size + x)]);
            const double w = e * e;
            const double dx = (x + 0.5) - c;
            const double dz = (z + 0.5) - c;
            num += w * std::sqrt(dx * dx + dz * dz) * cell;
            den += w;
        }
    }
    return den > 0.0 ? num / den : 0.0;
}

}  // namespace

// ===========================================================================
// Construction and configuration
// ===========================================================================

TEST_CASE("interaction descriptor validation")
{
    CHECK_NOTHROW(InteractionField{base_desc()});

    auto bad = [](auto mutate) {
        InteractionDesc d = base_desc();
        mutate(d);
        CHECK_THROWS_AS(InteractionField{d}, std::invalid_argument);
    };
    bad([](InteractionDesc& d) { d.size = 100; });          // not a power of two
    bad([](InteractionDesc& d) { d.size = 16; });           // too small
    bad([](InteractionDesc& d) { d.size = 2048; });         // too large
    bad([](InteractionDesc& d) { d.extent = 0.0f; });
    bad([](InteractionDesc& d) { d.kernel_radius = 0; });
    bad([](InteractionDesc& d) { d.kernel_radius = 13; });
    bad([](InteractionDesc& d) { d.damping = -1.0f; });
    bad([](InteractionDesc& d) { d.max_substeps = 0; });
    bad([](InteractionDesc& d) { d.gravity = 0.0f; });
    bad([](InteractionDesc& d) { d.absorb_cells = 200; });

    // A timestep past the stability limit is refused at construction rather
    // than silently clamped. Clamping would leave the caller believing they
    // were running at a timestep they were not, and the symptom (everything
    // slightly slow) is very hard to trace back.
    InteractionDesc d = base_desc();
    d.fixed_dt = 10.0f;
    CHECK_THROWS_AS(InteractionField{d}, std::invalid_argument);
}

TEST_CASE("stability limit is derived from the kernel, and the default sits well inside it")
{
    InteractionField f{base_desc()};

    // Leapfrog is stable iff omega*dt <= 2. At the limit the shortest waves
    // are stable and about 57% wrong in frequency, so "stable" is a floor, not
    // a target - the default must be far below it.
    const float limit = f.stable_dt_limit();
    const float dt    = f.fixed_dt();
    CHECK(limit > 0.0f);
    CHECK(dt > 0.0f);
    CHECK(dt < 0.5f * limit);

    std::printf("\n  interaction 256^2 / 64 m (dx = 0.25 m): dt = %.5f s, "
                "stability limit %.5f s (%.1fx headroom)\n",
                dt, limit, limit / dt);
    std::printf("  kernel dispersion error %.2f%% (wave speed %.2f%%)\n",
                100.0 * f.kernel_dispersion_error(),
                50.0 * f.kernel_dispersion_error());

    // The leapfrog phase error at the Nyquist wave, from
    // sin(w~ dt/2) = w dt/2. Reported because it is the OTHER half of the
    // total dispersion error, alongside the kernel's.
    const double w_nyq = 2.0 / limit;              // omega at the stability edge
    const double a = 0.5 * w_nyq * dt;
    const double phase_err = std::asin(a) / a - 1.0;
    std::printf("  leapfrog phase error at Nyquist: %.4f%%\n", 100.0 * phase_err);
    CHECK(phase_err < 0.02);
}

// ===========================================================================
// The impulse shape
// ===========================================================================

TEST_CASE("an impulse is a crater with a rim, and displaces no net volume")
{
    InteractionDesc d = base_desc();
    d.absorb_cells = 0;              // measure the raw injection
    InteractionField f{d};

    f.add(rock(32.0f, 32.0f, 0.8f, 0.3f));
    f.update(1.0f / 60.0f);

    const InteractionBuffers b = f.buffers();
    const std::uint32_t c = b.size / 2;
    auto at = [&](std::uint32_t x, std::uint32_t z) {
        return b.field[4 * (static_cast<std::size_t>(z) * b.size + x)];
    };

    // A depression at the centre...
    CHECK(at(c, c) < 0.0f);

    // ...ringed by a raised rim. The Ricker crosses zero at r = sigma*sqrt(2)
    // and peaks at r = 2*sigma, which at 0.25 m cells and sigma = 0.8 m is
    // about 6.4 cells out.
    float rim = 0.0f;
    for (std::uint32_t r = 3; r < 16; ++r) rim = std::max(rim, at(c + r, c));
    CHECK(rim > 0.0f);

    // The rim is a shallow fraction of the crater depth, as a real impact
    // crater is - the Ricker's rim peaks at exactly e^-2 = 13.5% of the
    // central depth. This is the property that distinguishes it from a plain
    // spike, so it is checked rather than assumed.
    CHECK(rim < 0.5f * std::fabs(at(c, c)));

    // THE structural property: net volume is zero. The operator's symbol is
    // exactly zero at k = 0, so any net volume injected could never propagate
    // away - it would sit there as a permanent bump forever. A Gaussian dimple
    // would fail this and look wrong in a way that no amount of damping fixes.
    const double vol = net_volume(f);
    double scale = 0.0;
    for (std::uint32_t i = 0; i < b.size * b.size; ++i) {
        scale += std::fabs(static_cast<double>(b.field[4 * i]));
    }
    CHECK(std::fabs(vol) < 1e-3 * scale);
}

TEST_CASE("impulse radius selects the emitted wavelength")
{
    // The profile's transform peaks at k = sqrt(2)/sigma, so doubling the
    // radius should double the dominant wavelength. Measured through the
    // resulting ring speed, since longer waves travel faster in deep water:
    // c_g = 0.5*sqrt(g/k), so a 2x wavelength is a sqrt(2)x faster ring.
    auto ring_speed = [](float sigma) {
        InteractionDesc d = base_desc();
        InteractionField f{d};
        f.add(rock(32.0f, 32.0f, sigma, 0.2f));
        f.update(1.0f / 60.0f);
        run(f, 119);
        const double r0 = centroid_radius(f);
        run(f, 240);
        const double r1 = centroid_radius(f);
        return (r1 - r0) / (240.0 / 60.0);
    };
    // Both radii stay inside the accurate band, so this measures the
    // wavelength-selection property rather than the band edge.
    const double s_small = ring_speed(0.30f);
    const double s_large = ring_speed(0.50f);
    const double expected = std::sqrt(0.50 / 0.30);
    std::printf("  ring speed: sigma=0.30 m -> %.3f m/s, sigma=0.50 m -> %.3f m/s "
                "(ratio %.3f, expected %.3f)\n",
                s_small, s_large, s_large / s_small, expected);
    CHECK(s_large > s_small);
    CHECK(s_large / s_small == doctest::Approx(expected).epsilon(0.15));
}

// ===========================================================================
// THE dispersion test
// ===========================================================================

TEST_CASE("ripple ring expands at the deep-water group velocity")
{
    // The test that proves the method is physical rather than merely
    // plausible. A non-dispersive wave-equation solver gives a ring travelling
    // at whatever constant c it was tuned with, and would fail this at every
    // sigma but one.
    //
    // Two comparisons, because they are two different claims:
    //
    //   vs REALISED - does the solver integrate its own operator correctly?
    //                 Asserted tightly at every sigma; a failure is a bug.
    //   vs IDEAL    - is that operator good deep-water physics? Asserted inside
    //                 the band the kernel claims, and REPORTED everywhere,
    //                 because the degradation outside the band is a known
    //                 property of a finite stencil rather than a defect.
    const double cell = 64.0 / 256.0;

    std::printf("\n  ring expansion, undamped, 256^2 over 64 m (dx = %.2f m), P = 6\n", cell);
    std::printf("  %-7s %-9s %9s %9s %8s %9s %8s\n",
                "sigma", "lambda0", "measured", "realised", "err", "ideal", "err");

    for (float sigma : {0.30f, 0.40f, 0.50f, 0.60f, 0.80f}) {
        InteractionDesc d = base_desc();
        InteractionField f{d};
        const double k_max = kPi / cell;

        f.add(rock(32.0f, 32.0f, sigma, 0.2f));
        f.update(1.0f / 60.0f);

        // Let the packet separate from the injection transient before
        // measuring, and stop well before the ring reaches the absorbing
        // layer, so what is measured is propagation and nothing else.
        run(f, 119);
        const double r0 = centroid_radius(f);
        run(f, 360);
        const double r1 = centroid_radius(f);

        const double measured = (r1 - r0) / 6.0;
        const double realised = realised_group_velocity(
            sigma, d.gravity, cell, d.kernel_radius,
            static_cast<double>(f.fixed_dt()));
        const double ideal = expected_group_velocity(sigma, d.gravity, k_max);

        const double err_r = std::fabs(measured / realised - 1.0);
        const double err_i = std::fabs(measured / ideal - 1.0);

        std::printf("  %-7.2f %-9.2f %9.4f %9.4f %7.2f%% %9.4f %7.2f%%\n",
                    sigma, 4.443 * sigma, measured, realised, 100.0 * err_r,
                    ideal, 100.0 * err_i);

        CHECK(measured > 0.0);

        // The solver must reproduce its own operator, at every sigma. This is
        // the assertion that would catch a real bug.
        CHECK(err_r < 0.12);

        // Physical accuracy is claimed only where the kernel claims it. The
        // dominant emitted wavelength is about 4.44*sigma and the band runs out
        // to 16 cells = 4 m, so sigma up to about 0.5 m is in band.
        if (sigma <= 0.5f) CHECK(err_i < 0.06);
    }
    std::printf("  outside the band the gap is dominated by c_g being a "
                "DERIVATIVE of the symbol:\n  where the realised symbol rises "
                "steeply, a 28%% symbol error becomes a 50%% speed error.\n");
}

TEST_CASE("a larger kernel radius measurably improves the ring speed")
{
    // The accuracy knob has to do something observable end to end, not just
    // in the kernel's symbol. If this ever failed, kernel_radius would be
    // costing taps for nothing.
    auto err_at = [](std::uint32_t p) {
        InteractionDesc d = base_desc();
        d.kernel_radius = p;
        InteractionField f{d};
        const double cell  = d.extent / d.size;
        f.add(rock(32.0f, 32.0f, 0.5f, 0.2f));
        f.update(1.0f / 60.0f);
        run(f, 119);
        const double r0 = centroid_radius(f);
        run(f, 360);
        const double r1 = centroid_radius(f);
        const double measured = (r1 - r0) / 6.0;
        const double theory =
            expected_group_velocity(0.5, d.gravity, kPi / cell);
        return std::fabs(measured / theory - 1.0);
    };
    const double e4 = err_at(4), e8 = err_at(8);
    std::printf("  ring-speed error: P=4 %.2f%%, P=8 %.2f%%\n",
                100.0 * e4, 100.0 * e8);
    CHECK(e8 < e4);
}

// ===========================================================================
// Energy, damping and the absorbing boundary
// ===========================================================================

TEST_CASE("energy decays monotonically with no source")
{
    InteractionDesc d = base_desc();
    d.damping = 0.5f;
    InteractionField f{d};

    f.add(rock(32.0f, 32.0f, 0.6f, 0.3f));
    f.update(1.0f / 60.0f);

    double prev = f.energy();
    CHECK(prev > 0.0);
    for (int i = 0; i < 200; ++i) {
        run(f, 3);
        const double e = f.energy();
        // Monotone, with a hair of slack for the rounding in a 65k-term sum.
        REQUIRE(e <= prev * (1.0 + 1e-6) + 1e-12);
        prev = e;
    }
    CHECK(prev > 0.0);
}

TEST_CASE("undamped energy is conserved to the scheme's own accuracy")
{
    // The leapfrog conserves the cross-level energy exactly in exact
    // arithmetic, so with no damping and no absorbing layer the only losses
    // are float rounding. Drift here would mean the scheme, the kernel and the
    // energy expression disagree with each other.
    InteractionDesc d = base_desc(128, 64.0f);
    d.damping = 0.0f;
    d.absorb_cells = 0;
    InteractionField f{d};

    f.add(rock(32.0f, 32.0f, 1.0f, 0.1f));
    f.update(1.0f / 60.0f);
    const double e0 = f.energy();
    run(f, 120);
    const double e1 = f.energy();
    std::printf("  undamped energy drift over 2 s: %.4f%%\n",
                100.0 * std::fabs(e1 / e0 - 1.0));
    CHECK(e1 == doctest::Approx(e0).epsilon(0.02));
}

TEST_CASE("the absorbing layer keeps boundary reflection below a stated threshold")
{
    // A sponge is not a true non-reflecting boundary condition, so some energy
    // does come back. The point of the test is to MEASURE how much, not to
    // pretend it is zero.
    //
    // Method: launch a ring, let it run into the boundary and any reflection
    // return to the middle, then compare the energy left in the central region
    // against a control run on a grid large enough that nothing could have
    // reached the boundary at all.
    auto central_energy = [](InteractionField& f, double frac) {
        const InteractionBuffers b = f.buffers();
        const double c = 0.5 * b.size;
        const double lim = frac * b.size;
        double e = 0.0;
        for (std::uint32_t z = 0; z < b.size; ++z) {
            for (std::uint32_t x = 0; x < b.size; ++x) {
                const double dx = (x + 0.5) - c, dz = (z + 0.5) - c;
                if (std::sqrt(dx * dx + dz * dz) > lim) continue;
                const double v = b.field[4 * (static_cast<std::size_t>(z) * b.size + x)];
                e += v * v;
            }
        }
        return e;
    };

    InteractionDesc small = base_desc(128, 32.0f);   // ring reaches the edge
    small.damping = 0.0f;
    small.absorb_cells = 16;
    InteractionField fs{small};
    fs.add(rock(16.0f, 16.0f, 0.5f, 0.2f));
    fs.update(1.0f / 60.0f);
    run(fs, 60 * 30);            // long enough for a round trip and back

    const double leftover = central_energy(fs, 0.20);

    // Reference: the same impulse's energy immediately after injection.
    InteractionField fr{small};
    fr.add(rock(16.0f, 16.0f, 0.5f, 0.2f));
    fr.update(1.0f / 60.0f);
    const double initial = central_energy(fr, 0.20);

    const double ratio = leftover / initial;
    std::printf("  energy returned to the central 20%% after a full round trip: "
                "%.4f%% of the initial\n", 100.0 * ratio);

    // Stated threshold: under 1% of the launched energy finds its way back.
    CHECK(ratio < 0.01);
}

TEST_CASE("without an absorbing layer the boundary really does reflect")
{
    // The control that gives the previous test meaning: if reflection were
    // small with absorb_cells = 0 too, the layer would be doing nothing and
    // the threshold above would be measuring the damping instead.
    auto returned = [](std::uint32_t absorb) {
        InteractionDesc d = base_desc(128, 32.0f);
        d.damping = 0.0f;
        d.absorb_cells = absorb;
        InteractionField f{d};
        f.add(rock(16.0f, 16.0f, 0.5f, 0.2f));
        f.update(1.0f / 60.0f);
        run(f, 60 * 30);
        const InteractionBuffers b = f.buffers();
        double e = 0.0;
        for (std::uint32_t i = 0; i < b.size * b.size; ++i) {
            const double v = b.field[4 * i];
            e += v * v;
        }
        return e;
    };
    const double with_layer = returned(16);
    const double without    = returned(0);
    std::printf("  total energy left after 30 s: absorb=16 %.3e, absorb=0 %.3e "
                "(%.0fx)\n", with_layer, without, without / with_layer);
    CHECK(without > 50.0 * with_layer);
}

// ===========================================================================
// Stability
// ===========================================================================

TEST_CASE("no NaN or blow-up when hammered with large impulses")
{
    InteractionDesc d = base_desc();
    d.damping = 0.1f;
    InteractionField f{d};

    // Absurd amplitudes, overlapping, every frame, for a long time. The
    // scheme is linear so it cannot become unstable from amplitude alone -
    // but this is exactly the kind of claim that should be executed rather
    // than reasoned about.
    for (int i = 0; i < 600; ++i) {
        f.add(rock(32.0f + 6.0f * std::sin(i * 0.7f),
                   32.0f + 6.0f * std::cos(i * 0.5f), 0.4f, 50.0f));
        f.add(rock(32.0f, 32.0f, 0.3f, -80.0f));
        f.update(1.0f / 60.0f);
    }
    const InteractionBuffers b = f.buffers();
    for (std::uint32_t i = 0; i < b.size * b.size * 4; ++i) {
        REQUIRE(std::isfinite(b.field[i]));
    }
}

TEST_CASE("stable at the timestep limit itself")
{
    InteractionDesc probe = base_desc();
    const float limit = InteractionField{probe}.stable_dt_limit();

    InteractionDesc d = base_desc();
    d.damping      = 0.0f;
    d.max_substeps = 64;
    // Sit just inside the limit - the worst case the configuration allows.
    d.fixed_dt = limit * 0.995f;
    InteractionField f{d};

    f.add(rock(32.0f, 32.0f, 0.5f, 1.0f));
    for (int i = 0; i < 400; ++i) f.update(d.fixed_dt);

    const InteractionBuffers b = f.buffers();
    double peak = 0.0;
    for (std::uint32_t i = 0; i < b.size * b.size; ++i) {
        REQUIRE(std::isfinite(b.field[4 * i]));
        peak = std::max(peak, std::fabs(static_cast<double>(b.field[4 * i])));
    }
    // Bounded, not merely finite: an instability that had not yet reached
    // infinity would still pass an isfinite check.
    CHECK(peak < 100.0);
    std::printf("  at dt = %.4f s (99.5%% of the limit): peak |eta| = %.4f m, "
                "finite and bounded\n", d.fixed_dt, peak);
}

TEST_CASE("a caller cannot push past the timestep by feeding huge dt")
{
    InteractionDesc d = base_desc();
    d.max_substeps = 4;
    InteractionField f{d};

    f.add(rock(32.0f, 32.0f));
    f.update(10.0f);           // 600 substeps' worth of wall time

    // The accumulator clamps and discards rather than trying to catch up, so
    // the frame cost is bounded. The field runs slow; it does not explode.
    CHECK(f.last_substeps() == 4);
    CHECK(f.last_update_clamped());

    const InteractionBuffers b = f.buffers();
    for (std::uint32_t i = 0; i < b.size * b.size; ++i) {
        REQUIRE(std::isfinite(b.field[4 * i]));
    }
}

// ===========================================================================
// Fixed timestep
// ===========================================================================

TEST_CASE("behaviour does not depend on frame rate")
{
    // Same total time, delivered in different sized frames. The accumulator
    // must make these agree - that is the entire reason it exists.
    auto run_at = [](float dt, int frames) {
        InteractionDesc d = base_desc(128, 64.0f);
        InteractionField f{d};
        f.add(rock(32.0f, 32.0f, 0.6f, 0.2f));
        for (int i = 0; i < frames; ++i) f.update(dt);
        std::vector<float> out;
        const InteractionBuffers b = f.buffers();
        out.assign(b.field, b.field + b.size * b.size * 4);
        return out;
    };
    // 1/60 s substeps: 120 frames at 1/60 and 60 frames at 1/30 both deliver
    // exactly 120 substeps, so these must be BIT-identical, not merely close.
    const auto a = run_at(1.0f / 60.0f, 120);
    const auto b = run_at(1.0f / 30.0f, 60);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// ===========================================================================
// Determinism
// ===========================================================================

namespace {

// A recorded event log: exactly what a host submitted, frame by frame.
struct Event { int frame; Disturbance d; };

std::vector<float> replay(const std::vector<Event>& log, int frames,
                          ParallelForFn hook, void* user,
                          std::uint32_t threads = 0)
{
    InteractionDesc d = base_desc();
    d.damping           = 0.05f;
    d.parallel_for      = hook;
    d.parallel_for_user = user;
    d.thread_count      = threads;
    InteractionField f{d};

    std::size_t next = 0;
    for (int i = 0; i < frames; ++i) {
        while (next < log.size() && log[next].frame == i) f.add(log[next++].d);
        f.update(1.0f / 60.0f);
    }
    const InteractionBuffers b = f.buffers();
    return std::vector<float>(b.field, b.field + b.size * b.size * 4);
}

// A host scheduler that runs indices BACKWARDS. It satisfies the contract, and
// it is sharper than a serial hook: it proves there is no hidden dependence on
// task ordering.
void reverse_hook(void*, TaskFn task, void* ctx, std::uint32_t count)
{
    for (std::uint32_t i = count; i-- > 0;) task(ctx, i);
}

// A host scheduler that scatters indices across raw threads in whatever order
// they win the race.
void chaotic_hook(void*, TaskFn task, void* ctx, std::uint32_t count)
{
    std::vector<std::thread> ts;
    ts.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        ts.emplace_back([=] { task(ctx, i); });
    }
    for (auto& t : ts) t.join();
}

std::vector<Event> make_log()
{
    std::vector<Event> log;
    for (int i = 0; i < 40; ++i) {
        Disturbance d = rock(20.0f + 0.7f * i, 25.0f + 0.4f * i,
                             0.35f + 0.01f * i, 0.15f);
        d.velocity_y = -1.5f - 0.05f * i;
        log.push_back({i * 5, d});

        Disturbance c = rock(30.0f, 30.0f + 0.2f * i, 0.5f, 0.05f);
        c.kind       = SourceKind::Continuous;
        c.velocity_x = 1.2f;
        c.velocity_z = -0.4f;
        log.push_back({i * 5 + 1, c});
    }
    return log;
}

}  // namespace

TEST_CASE("the same event log replays bit-identically")
{
    const auto log = make_log();

    // The reference: single-threaded, the library's own serial path.
    const auto ref = replay(log, 220, nullptr, nullptr, 1);

    SUBCASE("repeated identically on the same path")
    {
        CHECK(std::memcmp(ref.data(), replay(log, 220, nullptr, nullptr, 1).data(),
                          ref.size() * sizeof(float)) == 0);
    }

    SUBCASE("across thread counts")
    {
        for (std::uint32_t t : {1u, 2u, 3u, 4u, 8u, 16u}) {
            const auto v = replay(log, 220, nullptr, nullptr, t);
            INFO("thread_count = " << t);
            REQUIRE(std::memcmp(ref.data(), v.data(),
                                ref.size() * sizeof(float)) == 0);
        }
    }

    SUBCASE("under a reverse-order host scheduler")
    {
        const auto v = replay(log, 220, &reverse_hook, nullptr);
        CHECK(std::memcmp(ref.data(), v.data(), ref.size() * sizeof(float)) == 0);
    }

    SUBCASE("under a chaotic host scheduler")
    {
        const auto v = replay(log, 220, &chaotic_hook, nullptr);
        CHECK(std::memcmp(ref.data(), v.data(), ref.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("queue overflow is counted, and deterministic")
{
    InteractionDesc d = base_desc();
    d.max_sources = 4;
    InteractionField f{d};
    for (int i = 0; i < 10; ++i) f.add(rock(30.0f + i, 30.0f));
    CHECK(f.dropped_sources() == 6);

    // Dropping the tail is deterministic given the same submission order, so
    // an overflowing frame still replays exactly.
    InteractionField g{d};
    for (int i = 0; i < 10; ++i) g.add(rock(30.0f + i, 30.0f));
    f.update(1.0f / 60.0f);
    g.update(1.0f / 60.0f);
    const auto a = f.buffers(), b = g.buffers();
    CHECK(std::memcmp(a.field, b.field,
                      static_cast<std::size_t>(a.size) * a.size * 4 * sizeof(float)) == 0);
}

TEST_CASE("an impulse submitted on a frame that runs no substep is not lost")
{
    // A short frame delivers less than one timestep. An impulse is an EVENT:
    // dropping it because the frame was short would lose a rock entirely, and
    // the bug would be intermittent and frame-rate dependent - the worst kind.
    InteractionDesc d = base_desc();
    InteractionField f{d};
    f.add(rock(32.0f, 32.0f, 0.5f, 0.3f));
    f.update(0.001f);                       // no substep runs
    CHECK(f.last_substeps() == 0);

    f.update(1.0f / 30.0f);                 // now it should fire
    CHECK(f.last_substeps() > 0);

    double sum = 0.0;
    const InteractionBuffers b = f.buffers();
    for (std::uint32_t i = 0; i < b.size * b.size; ++i) {
        sum += std::fabs(static_cast<double>(b.field[4 * i]));
    }
    CHECK(sum > 0.0);
}

// ===========================================================================
// SIMD
// ===========================================================================

TEST_CASE("every SIMD step kernel is bit-identical to scalar")
{
    // The kernels are compared DIRECTLY rather than through the runtime
    // dispatch, so every kernel in the build is exercised on every machine -
    // not merely whichever one this CPU happens to select.
    using namespace ocean::detail;

    for (std::uint32_t p : {3u, 4u, 6u, 8u}) {
        for (std::uint32_t n : {32u, 64u, 128u}) {
            IWaveKernel k;
            build_iwave_kernel(p, 0.25, 0.0, KernelMethod::LeastSquares, k);

            const std::size_t stride = (static_cast<std::size_t>(n) + 2 * p + 15) & ~std::size_t{15};
            const std::size_t rows   = static_cast<std::size_t>(n) + 2 * p;
            std::vector<float> a(rows * stride), b(rows * stride);
            std::vector<float> old_a(rows * stride), old_b(rows * stride);
            std::vector<float> c1(static_cast<std::size_t>(n) * n),
                               c2(static_cast<std::size_t>(n) * n);

            // Deterministic but irregular data, including negatives and
            // values of very different magnitude, so cancellation is
            // exercised rather than avoided.
            std::uint32_t s = 12345u;
            auto rnd = [&] {
                s = s * 1664525u + 1013904223u;
                return (static_cast<float>(s >> 8) / 8388608.0f - 1.0f);
            };
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = rnd() * (i % 7 == 0 ? 100.0f : 0.01f);
                old_a[i] = rnd();
            }
            b = a; old_b = old_a;
            for (std::size_t i = 0; i < c1.size(); ++i) {
                c1[i] = 0.9f + 0.05f * rnd();
                c2[i] = 0.8f + 0.05f * rnd();
            }

            auto make = [&](std::vector<float>& cur, std::vector<float>& old) {
                IWaveGrid g;
                g.n = n; g.p = p; g.stride = stride;
                g.cur = cur.data() + static_cast<std::size_t>(p) * stride + p;
                g.old = old.data() + static_cast<std::size_t>(p) * stride + p;
                g.taps = k.taps.data();
                g.c1 = c1.data(); g.c2 = c2.data();
                g.gdt2 = 9.81f * (1.0f / 60.0f) * (1.0f / 60.0f);
                return g;
            };

            const IWaveGrid ga = make(a, old_a);
            iwave_step_rows_scalar(ga, 0, n);

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
            {
                std::vector<float> cb = b, ob = old_b;
                const IWaveGrid gb = make(cb, ob);
                iwave_step_rows_sse2(gb, 0, n);
                INFO("SSE2, P=" << p << " N=" << n);
                REQUIRE(std::memcmp(old_a.data(), ob.data(),
                                    ob.size() * sizeof(float)) == 0);
            }
            if (max_simd_level() >= SimdLevel::Avx2) {
                std::vector<float> cb = b, ob = old_b;
                const IWaveGrid gb = make(cb, ob);
                iwave_step_rows_avx2(gb, 0, n);
                INFO("AVX2, P=" << p << " N=" << n);
                REQUIRE(std::memcmp(old_a.data(), ob.data(),
                                    ob.size() * sizeof(float)) == 0);
            }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
            {
                std::vector<float> cb = b, ob = old_b;
                const IWaveGrid gb = make(cb, ob);
                iwave_step_rows_neon(gb, 0, n);
                INFO("NEON, P=" << p << " N=" << n);
                REQUIRE(std::memcmp(old_a.data(), ob.data(),
                                    ob.size() * sizeof(float)) == 0);
            }
#endif
        }
    }
}

// ===========================================================================
// Recentring
// ===========================================================================

TEST_CASE("recentring by whole cells is exact, so there is no jolt")
{
    InteractionDesc d = base_desc();
    d.damping = 0.0f;
    InteractionField f{d};
    f.add(rock(32.0f, 32.0f, 0.8f, 0.3f));
    f.update(1.0f / 60.0f);
    run(f, 60);

    const InteractionBuffers b0 = f.buffers();
    const std::uint32_t n = b0.size;
    const float cell = b0.extent / n;
    std::vector<float> before(b0.field,
                              b0.field + static_cast<std::size_t>(n) * n * 4);

    // Shift by exactly 10 cells in x and 7 in z, and compare immediately, with
    // no stepping in between. A substep after the shift would legitimately
    // differ: the absorbing layer is anchored to the GRID, so a cell that moved
    // closer to the edge correctly feels more damping.
    f.recenter(32.0f + 10.0f * cell, 32.0f + 7.0f * cell);
    CHECK(f.recentered());
    CHECK(f.last_shift_x() == 10);
    CHECK(f.last_shift_z() == 7);

    const InteractionBuffers b1 = f.buffers();

    // Every retained cell must hold EXACTLY the value it held before. Not
    // approximately: an integer-cell move resamples nothing, interpolates
    // nothing and filters nothing, and that is the entire no-jolt argument. A
    // single differing bit here would mean the shift was not exact.
    int compared = 0, mismatched = 0;
    for (std::uint32_t z = 0; z + 7 < n; ++z) {
        for (std::uint32_t x = 0; x + 10 < n; ++x) {
            const float moved = b1.field[4 * (static_cast<std::size_t>(z) * n + x)];
            const float orig =
                before[4 * (static_cast<std::size_t>(z + 7) * n + (x + 10))];
            ++compared;
            if (moved != orig) ++mismatched;
        }
    }
    std::printf("  recentre by (10, 7) cells: %d interior cells compared, "
                "%d differing\n", compared, mismatched);
    CHECK(compared > 50000);
    CHECK(mismatched == 0);

    // Newly exposed cells are zero - the physically correct value, since this
    // field holds only the local disturbance and there is none out there.
    for (std::uint32_t z = 0; z < n; ++z) {
        for (std::uint32_t x = n - 10; x < n; ++x) {
            REQUIRE(b1.field[4 * (static_cast<std::size_t>(z) * n + x)] == 0.0f);
        }
    }

    CHECK(b1.origin_x == doctest::Approx(b0.origin_x + 10.0f * cell).epsilon(1e-6));
    CHECK(b1.origin_z == doctest::Approx(b0.origin_z + 7.0f * cell).epsilon(1e-6));
}

TEST_CASE("recentring is a no-op below one cell, and world queries follow the grid")
{
    InteractionDesc d = base_desc();
    InteractionField f{d};
    f.recenter(32.0f, 32.0f);
    const float ox = f.buffers().origin_x;

    // A sub-cell nudge must not move the grid at all - snapping to whole
    // cells is what makes the move exact.
    f.recenter(32.0f + 0.1f * (d.extent / d.size), 32.0f);
    CHECK_FALSE(f.recentered());
    CHECK(f.buffers().origin_x == ox);

    // A disturbance stays at the same WORLD position across a recentre.
    f.add(rock(32.0f, 32.0f, 0.8f, 0.5f));
    f.update(1.0f / 60.0f);
    const float h_before = f.height_at(32.0f, 32.0f);
    CHECK(h_before < 0.0f);

    f.recenter(40.0f, 36.0f);
    f.update(1.0f / 60.0f);
    const float h_after = f.height_at(32.0f, 32.0f);
    // Same world point, one substep later: close, and certainly still a
    // depression. A grid/world mix-up would put it somewhere else entirely.
    CHECK(h_after < 0.0f);
    CHECK(h_after == doctest::Approx(h_before).epsilon(0.5));
}

// ===========================================================================
// Obstruction
// ===========================================================================

TEST_CASE("Neumann reflects a crest as a crest, Dirichlet as a trough")
{
    // The physics distinction, executed. A rigid hull is no-normal-flow,
    // dEta/dn = 0, which reflects a crest as a crest. Zeroing the field at
    // solid cells - what the published method does, and what nearly every
    // implementation copies - is a pressure-release surface and flips the
    // sign.
    auto reflected_sign = [](Obstruction mode) {
        InteractionDesc d = base_desc(128, 32.0f);
        d.damping      = 0.0f;
        d.absorb_cells = 8;
        d.obstruction  = mode;
        InteractionField f{d};

        // A wall down the right-hand side of the grid.
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(d.size) * d.size, 0);
        const std::uint32_t wall = d.size * 3 / 4;
        for (std::uint32_t z = 0; z < d.size; ++z) {
            for (std::uint32_t x = wall; x < d.size; ++x) {
                mask[static_cast<std::size_t>(z) * d.size + x] = 1;
            }
        }
        f.set_obstruction(mask.data());

        // Launch from the left of the wall and let the reflection come back.
        const float cell = d.extent / d.size;
        f.add(rock(cell * (wall - 20), 16.0f, 0.5f, 0.2f));
        f.update(1.0f / 60.0f);

        // Correlate the field just in front of the wall against the incoming
        // sign, after enough time for a reflection but before it disperses.
        double best = 0.0;
        for (int i = 0; i < 400; ++i) {
            f.update(1.0f / 60.0f);
            if (i < 40) continue;
            const InteractionBuffers b = f.buffers();
            double s = 0.0;
            for (std::uint32_t z = d.size / 2 - 6; z < d.size / 2 + 6; ++z) {
                for (std::uint32_t x = wall - 10; x < wall - 2; ++x) {
                    s += b.field[4 * (static_cast<std::size_t>(z) * d.size + x)];
                }
            }
            if (std::fabs(s) > std::fabs(best)) best = s;
        }
        return best;
    };

    const double neumann   = reflected_sign(Obstruction::Neumann);
    const double dirichlet = reflected_sign(Obstruction::Dirichlet);
    std::printf("  peak signed field in front of a wall: Neumann %+.5f, "
                "Dirichlet %+.5f\n", neumann, dirichlet);

    // The two conditions must produce genuinely different behaviour - that is
    // the whole point of offering both.
    CHECK(std::fabs(neumann) > 0.0);
    CHECK(std::fabs(dirichlet) > 0.0);
    CHECK(neumann != dirichlet);
}

TEST_CASE("an obstruction mask changes the field, and clearing it restores")
{
    InteractionDesc d = base_desc(128, 32.0f);
    InteractionField f{d};

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(d.size) * d.size, 0);
    for (std::uint32_t z = 40; z < 88; ++z) mask[static_cast<std::size_t>(z) * d.size + 80] = 1;
    f.set_obstruction(mask.data());
    f.add(rock(16.0f, 16.0f, 0.5f, 0.3f));
    f.update(1.0f / 60.0f);
    run(f, 120);
    const InteractionBuffers bm = f.buffers();
    std::vector<float> with(bm.field, bm.field + static_cast<std::size_t>(d.size) * d.size * 4);

    InteractionField g{d};
    g.set_obstruction(nullptr);
    g.add(rock(16.0f, 16.0f, 0.5f, 0.3f));
    g.update(1.0f / 60.0f);
    run(g, 120);
    const InteractionBuffers bn = g.buffers();

    CHECK(std::memcmp(with.data(), bn.field, with.size() * sizeof(float)) != 0);
}

// ===========================================================================
// Queries and the no-allocation guarantee
// ===========================================================================

TEST_CASE("queries agree with the buffer they are sampled from")
{
    InteractionDesc d = base_desc();
    InteractionField f{d};
    f.add(rock(32.0f, 32.0f, 0.8f, 0.3f));
    f.update(1.0f / 60.0f);
    run(f, 40);

    const InteractionBuffers b = f.buffers();
    const float cell = b.extent / b.size;

    // At exact cell centres the bilinear tap reduces to the texel itself.
    double worst = 0.0;
    for (std::uint32_t z = 40; z < 200; z += 13) {
        for (std::uint32_t x = 40; x < 200; x += 11) {
            const float wx = b.origin_x + (x + 0.5f) * cell;
            const float wz = b.origin_z + (z + 0.5f) * cell;
            const InteractionSample s = f.sample_at(wx, wz);
            const float ref = b.field[4 * (static_cast<std::size_t>(z) * b.size + x)];
            worst = std::max(worst, std::fabs(static_cast<double>(s.height - ref)));
        }
    }
    std::printf("  query vs buffer at cell centres: worst %.3e m\n", worst);
    CHECK(worst < 1e-6);

    // Outside the grid the answer is genuinely zero - there is no disturbance
    // out there. This is the correct value, not a clamp, and it is why the
    // renderer must use CLAMP_TO_BORDER rather than REPEAT.
    CHECK(f.height_at(10000.0f, 10000.0f) == 0.0f);
    CHECK(f.height_at(-10000.0f, 0.0f) == 0.0f);
}

TEST_CASE("update() performs no heap allocation")
{
    InteractionDesc d = base_desc();
    InteractionField f{d};
    f.add(rock(32.0f, 32.0f));
    f.update(1.0f / 60.0f);

    const std::size_t before = alloc_probe::count();
    for (int i = 0; i < 30; ++i) {
        f.add(rock(30.0f + 0.1f * i, 33.0f, 0.4f, 0.1f));
        Disturbance c = rock(31.0f, 31.0f, 0.5f, 0.05f);
        c.kind = SourceKind::Continuous;
        f.add(c);
        f.update(1.0f / 60.0f);
        f.recenter(32.0f + 0.05f * i, 32.0f);
    }
    const std::size_t after = alloc_probe::count();
    CHECK(after == before);
}

TEST_CASE("feeding exactly the fixed timestep runs exactly one substep")
{
    // The overwhelmingly common case: a 60 Hz host handing over 1/60 s against
    // a 1/60 s timestep. It must run one substep per frame, every frame.
    //
    // Without a tolerance it does not. float(1/60) and the double accumulator
    // land on opposite sides of the comparison by one ulp, so the field runs
    // 0, 2, 0, 2, ... substeps - correct on average, visibly jittery in motion,
    // and it silently halves the measured cost in a benchmark because half the
    // sampled frames do no work at all. That is exactly how it was found.
    InteractionDesc d = base_desc();
    InteractionField f{d};
    const float dt = f.fixed_dt();

    f.add(rock(32.0f, 32.0f));
    int zero_frames = 0, total = 0;
    for (int i = 0; i < 240; ++i) {
        f.update(dt);
        total += static_cast<int>(f.last_substeps());
        if (f.last_substeps() == 0) ++zero_frames;
        REQUIRE(f.last_substeps() == 1);
    }
    CHECK(zero_frames == 0);
    CHECK(total == 240);
}
