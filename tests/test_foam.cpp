// Persistent, advected foam.
//
// The two behaviours this class exists to add - drift and persistence - are
// tested in ISOLATION from each other, by switching the source and the
// advection off in turn. Testing them together would only show that something
// changed, which is the weakest possible claim.
#include "ocean/foam.hpp"
#include "ocean/ocean.hpp"

#include "alloc_probe.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ocean;

namespace {

OceanDesc foam_ocean(std::uint32_t n = 128)
{
    OceanDesc d;
    d.size                = n;
    d.patch_length        = 200.0f;
    d.spectrum.wind_speed = 16.0f;   // windy, so there is real foam to work with
    d.choppiness          = 1.4f;
    d.foam_threshold      = 0.9f;
    d.seed                = 31337;
    d.compute_velocity    = true;    // required by FoamField
    return d;
}

double mean_of(const float* f, std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += f[i];
    return s / static_cast<double>(n);
}

}  // namespace

TEST_CASE("FoamField refuses an ocean with no velocity field")
{
    // Refused rather than silently degraded: without velocity there is nothing
    // to advect with, and quietly producing non-advected foam would leave a
    // caller believing they had the feature they asked for.
    OceanDesc d = foam_ocean();
    d.compute_velocity = false;
    Ocean sim{d};
    CHECK_THROWS_AS(FoamField(sim, FoamDesc{}), std::invalid_argument);

    Ocean ok{foam_ocean()};
    CHECK_NOTHROW(FoamField(ok, FoamDesc{}));

    auto bad = [&](auto mutate) {
        FoamDesc fd;
        mutate(fd);
        CHECK_THROWS_AS(FoamField(ok, fd), std::invalid_argument);
    };
    bad([](FoamDesc& f) { f.decay = -1.0f; });
    bad([](FoamDesc& f) { f.source_gain = -1.0f; });
    bad([](FoamDesc& f) { f.fixed_dt = 0.0f; });
    bad([](FoamDesc& f) { f.max_substeps = 0; });
}

TEST_CASE("foam persists after the source stops, decaying at the stated rate")
{
    // Decay in isolation: build foam up, then switch the source off with
    // set_desc and watch it fall. The claim is exp(-decay * t), so this checks
    // the actual exponential rather than merely that it goes down.
    Ocean sim{foam_ocean()};
    sim.update(20.0);

    FoamDesc fd;
    fd.decay        = 0.5f;
    fd.source_gain  = 6.0f;
    fd.advect_scale = 0.0f;   // isolate decay from drift
    FoamField foam{sim, fd};

    for (int i = 0; i < 120; ++i) foam.update(1.0f / 60.0f);
    const double built = foam.coverage();
    CHECK(built > 1e-4);

    // Source off. Nothing else changes, and the ocean is deliberately NOT
    // advanced, so the only thing acting on the field is decay.
    fd.source_gain = 0.0f;
    foam.set_desc(fd);

    const double t = 2.0;
    for (int i = 0; i < 120; ++i) foam.update(1.0f / 60.0f);
    const double after = foam.coverage();

    const double expected = built * std::exp(-fd.decay * t);
    std::printf("\n  foam decay over %.1f s at %.2f /s: %.6f -> %.6f "
                "(expected %.6f, error %.3f%%)\n",
                t, fd.decay, built, after, expected,
                100.0 * std::fabs(after / expected - 1.0));

    // The residual is the difference between the discrete per-step factor
    // exp(-decay*dt) applied 120 times and the continuous exp(-decay*t) - which
    // is exactly zero in real arithmetic, so this is float rounding only.
    CHECK(after == doctest::Approx(expected).epsilon(1e-3));

    // And crucially: it is still non-zero. The instantaneous Jacobian foam
    // would have been gone the instant the crest unfolded; this is the whole
    // point of the class.
    CHECK(after > 0.0);
}

TEST_CASE("foam is advected, exactly, by the flow it is given")
{
    // Advection in isolation, and made EXACT by choosing a drift that moves
    // the field a whole number of cells per step. At a whole-cell shift the
    // bilinear weights are exactly 1 and 0, so semi-Lagrangian advection is a
    // pure relabelling and the result must be bit-identical to the same field
    // shifted by hand. Any smearing, any off-by-one, any wrap error shows up
    // as a hard failure rather than as a tolerance being widened.
    Ocean sim{foam_ocean()};
    sim.update(20.0);

    const float cell = 200.0f / 128.0f;
    const float dt   = 1.0f / 60.0f;

    FoamDesc fd;
    fd.decay        = 0.0f;
    fd.source_gain  = 5.0f;
    fd.advect_scale = 0.0f;
    fd.fixed_dt     = dt;
    FoamField foam{sim, fd};

    for (int i = 0; i < 60; ++i) foam.update(dt);

    const std::uint32_t n = foam.size();
    std::vector<float> before(foam.data(), foam.data() + static_cast<std::size_t>(n) * n);
    CHECK(mean_of(before.data(), before.size()) > 1e-4);

    // Source off, decay off, pure uniform drift of exactly one cell per step.
    fd.source_gain  = 0.0f;
    fd.wind_drift_x = cell / dt;
    foam.set_desc(fd);

    const int steps = 5;
    for (int i = 0; i < steps; ++i) foam.update(dt);

    const float* after = foam.data();
    int mismatched = 0;
    for (std::uint32_t z = 0; z < n; ++z) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::uint32_t sx = (x + n - static_cast<std::uint32_t>(steps) % n) % n;
            const float expect = before[static_cast<std::size_t>(z) * n + sx];
            if (after[static_cast<std::size_t>(z) * n + x] != expect) ++mismatched;
        }
    }
    std::printf("  advected exactly %d cells in +x: %u cells compared, "
                "%d differing\n", steps, n * n, mismatched);
    CHECK(mismatched == 0);
}

TEST_CASE("foam advection wraps, because the surface it rides on does")
{
    // A drift large enough to carry the field right around the patch must
    // return it to where it started. Clamping instead of wrapping would smear
    // the edge into a streak - and a REPEAT sampler would then tile that
    // streak across the entire ocean.
    Ocean sim{foam_ocean()};
    sim.update(12.0);

    const float cell = 200.0f / 128.0f;
    const float dt   = 1.0f / 60.0f;

    FoamDesc fd;
    fd.decay        = 0.0f;
    fd.source_gain  = 5.0f;
    fd.advect_scale = 0.0f;
    fd.fixed_dt     = dt;
    FoamField foam{sim, fd};
    for (int i = 0; i < 60; ++i) foam.update(dt);

    const std::uint32_t n = foam.size();
    std::vector<float> before(foam.data(), foam.data() + static_cast<std::size_t>(n) * n);

    fd.source_gain  = 0.0f;
    fd.wind_drift_x = cell / dt;     // one cell per step
    foam.set_desc(fd);
    for (std::uint32_t i = 0; i < n; ++i) foam.update(dt);   // exactly one lap

    CHECK(std::memcmp(before.data(), foam.data(),
                      before.size() * sizeof(float)) == 0);
}

TEST_CASE("persistent foam outlives the instantaneous Jacobian foam")
{
    // The behavioural claim, stated as a comparison against what the library
    // did before: on a real moving ocean, the persistent field must hold more
    // foam than the instantaneous threshold does, because it remembers.
    Ocean sim{foam_ocean(256)};

    FoamDesc fd;
    fd.decay       = 0.3f;
    fd.source_gain = 4.0f;
    FoamField foam{sim, fd};

    double worst_ratio = 1e9;
    for (int i = 0; i < 400; ++i) {
        sim.update(30.0 + i / 60.0);
        foam.update(1.0f / 60.0f);
    }
    for (int i = 0; i < 200; ++i) {
        sim.update(36.666 + i / 60.0);
        foam.update(1.0f / 60.0f);

        const Buffers b = sim.buffers();
        double instant = 0.0;
        const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
        for (std::size_t c = 0; c < cells; ++c) instant += b.displacement[4 * c + 3];
        instant /= static_cast<double>(cells);

        if (instant > 1e-6) {
            worst_ratio = std::min(worst_ratio, foam.coverage() / instant);
        }
    }
    std::printf("  persistent/instantaneous coverage, worst over 200 frames: "
                "%.2fx\n", worst_ratio);

    // Never less foam than the instantaneous threshold, at any frame. If the
    // ratio ever dropped below 1 the field would be forgetting faster than the
    // surface generates, which is the failure this class exists to prevent.
    CHECK(worst_ratio > 1.0);
}

TEST_CASE("foam stays within [0,1] under a saturating source")
{
    Ocean sim{foam_ocean()};
    FoamDesc fd;
    fd.decay       = 0.01f;
    fd.source_gain = 500.0f;    // absurd, to force the clamp
    FoamField foam{sim, fd};

    for (int i = 0; i < 300; ++i) {
        sim.update(5.0 + i / 60.0);
        foam.update(1.0f / 60.0f);
    }
    const float* f = foam.data();
    const std::size_t cells =
        static_cast<std::size_t>(foam.size()) * foam.size();
    for (std::size_t i = 0; i < cells; ++i) {
        REQUIRE(std::isfinite(f[i]));
        REQUIRE(f[i] >= 0.0f);
        REQUIRE(f[i] <= 1.0f);
    }
    CHECK(foam.coverage() > 0.5);   // it really did saturate
}

TEST_CASE("foam is deterministic, including under threading")
{
    auto run = [](ParallelForFn hook, std::uint32_t threads) {
        Ocean sim{foam_ocean(128)};
        FoamDesc fd;
        fd.parallel_for = hook;
        fd.thread_count = threads;
        FoamField foam{sim, fd};
        for (int i = 0; i < 90; ++i) {
            sim.update(10.0 + i / 60.0);
            foam.update(1.0f / 60.0f);
        }
        return std::vector<float>(
            foam.data(),
            foam.data() + static_cast<std::size_t>(foam.size()) * foam.size());
    };

    const auto ref = run(nullptr, 1);

    for (std::uint32_t t : {1u, 2u, 4u, 8u, 16u}) {
        const auto v = run(nullptr, t);
        INFO("thread_count = " << t);
        REQUIRE(std::memcmp(ref.data(), v.data(), ref.size() * sizeof(float)) == 0);
    }

    // A reverse-order host scheduler, which is sharper than a serial one: it
    // proves there is no hidden dependence on the order rows are processed in.
    // The ping-pong buffers are what make this hold - advecting in place would
    // let a row read a value another row had already overwritten.
    struct Hooks {
        static void reverse(void*, TaskFn task, void* ctx, std::uint32_t count)
        {
            for (std::uint32_t i = count; i-- > 0;) task(ctx, i);
        }
        static void chaotic(void*, TaskFn task, void* ctx, std::uint32_t count)
        {
            std::vector<std::thread> ts;
            ts.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                ts.emplace_back([=] { task(ctx, i); });
            }
            for (auto& t : ts) t.join();
        }
    };
    CHECK(std::memcmp(ref.data(), run(&Hooks::reverse, 0).data(),
                      ref.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(ref.data(), run(&Hooks::chaotic, 0).data(),
                      ref.size() * sizeof(float)) == 0);
}

TEST_CASE("foam does not depend on frame rate")
{
    auto run_at = [](float dt, int frames) {
        Ocean sim{foam_ocean(128)};
        FoamDesc fd;
        FoamField foam{sim, fd};
        for (int i = 0; i < frames; ++i) {
            // Same simulated time delivered either way.
            sim.update(10.0 + static_cast<double>(i + 1) * dt);
            foam.update(dt);
        }
        return foam.coverage();
    };
    // 120 frames at 1/60 and 60 at 1/30 both deliver 120 substeps of 1/60.
    const double a = run_at(1.0f / 60.0f, 120);
    const double b = run_at(1.0f / 30.0f, 60);
    std::printf("  coverage at 60 fps %.6f, at 30 fps %.6f (%.2f%% apart)\n",
                a, b, 100.0 * std::fabs(a / b - 1.0));

    // Not bit-identical, because the OCEAN is sampled at different instants in
    // between - the source term genuinely differs. The accumulator's job is
    // that the foam dynamics run at the same rate, and a few per cent is the
    // difference in what was sampled, not in how it was integrated.
    CHECK(a == doctest::Approx(b).epsilon(0.05));
}

TEST_CASE("foam queries and buffer agree, and clear() empties the field")
{
    // Choppiness stays HIGH here on purpose. Foam only exists where the
    // surface folds, so a zero-choppiness ocean has a Jacobian of exactly 1
    // everywhere and generates no foam at all - which makes it useless for
    // testing a foam query, and is what a first attempt at this test got
    // wrong. It also means the query has to invert the displacement for real,
    // which is the part worth testing.
    OceanDesc od = foam_ocean(128);
    Ocean sim{od};
    sim.update(14.0);

    FoamDesc fd;
    fd.source_gain = 6.0f;
    FoamField foam{sim, fd};
    for (int i = 0; i < 60; ++i) foam.update(1.0f / 60.0f);
    REQUIRE(foam.coverage() > 0.0);

    // Each texel's WORLD position is its parameter position plus that texel's
    // own horizontal displacement. Querying there must recover that texel's
    // foam - which only works if foam_at inverts the displacement the same way
    // Ocean::sample_at does.
    const Buffers b = sim.buffers();
    const float cell = b.patch_length / static_cast<float>(b.size);

    double worst = 0.0, worst_naive = 0.0;
    for (std::uint32_t z = 5; z < 120; z += 9) {
        for (std::uint32_t x = 5; x < 120; x += 7) {
            const std::size_t i = static_cast<std::size_t>(z) * b.size + x;
            const float wx = x * cell + b.displacement[4 * i + 0];
            const float wz = z * cell + b.displacement[4 * i + 2];

            const float ref = foam.data()[i];
            worst = std::max(worst,
                             std::fabs(static_cast<double>(foam.foam_at(wx, wz) - ref)));

            // What a naive lookup - treating world position as if it were
            // parameter position - would have given, for contrast.
            const int nx = static_cast<int>(wx / cell + 0.5f) & 127;
            const int nz = static_cast<int>(wz / cell + 0.5f) & 127;
            worst_naive = std::max(
                worst_naive,
                std::fabs(static_cast<double>(
                    foam.data()[static_cast<std::size_t>(nz) * 128 + nx] - ref)));
        }
    }
    std::printf("  foam query at displaced world positions: worst %.4f "
                "(naive lookup would be %.4f)\n", worst, worst_naive);

    // The residual is the fixed-point inversion's own error plus bilinear
    // interpolation between texels, not a mismatch in what is being asked.
    CHECK(worst < 0.15);
    // And the inversion is doing real work - a naive lookup is far worse.
    CHECK(worst_naive > worst);

    CHECK(foam.coverage() > 0.0);
    foam.clear();
    CHECK(foam.coverage() == 0.0);
}

TEST_CASE("foam update performs no heap allocation")
{
    Ocean sim{foam_ocean(128)};
    FoamField foam{sim, FoamDesc{}};
    sim.update(1.0);
    foam.update(1.0f / 60.0f);

    const std::size_t before = alloc_probe::count();
    for (int i = 0; i < 60; ++i) {
        sim.update(2.0 + i / 60.0);
        foam.update(1.0f / 60.0f);
    }
    CHECK(alloc_probe::count() == before);
}
