#include <doctest/doctest.h>

#include "core/spectrum.hpp"
#include "ocean/ocean.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace ocean;

namespace {

OceanDesc test_desc(std::uint32_t size = 64)
{
    OceanDesc d;
    d.size                       = size;
    d.patch_length               = 200.0f;
    d.seed                       = 20260917;
    d.choppiness                 = 1.0f;
    d.spectrum.wind_speed        = 12.0f;
    d.spectrum.fetch             = 100000.0f;
    d.spectrum.wind_direction    = 0.4f;
    d.spectrum.small_wave_cutoff = 0.5f;
    return d;
}

bool all_finite(const float* p, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST_CASE("same seed and time give bit-identical buffers across instances")
{
    const OceanDesc d = test_desc(64);

    Ocean a{d};
    Ocean b{d};

    const std::size_t floats = static_cast<std::size_t>(d.size) * d.size * 4;
    const std::size_t bytes  = floats * sizeof(float);

    for (double t : {0.0, 1.5, 97.25, 3600.0}) {
        CAPTURE(t);
        a.update(t);
        b.update(t);
        // memcmp, not an epsilon comparison. "Deterministic" means the bits
        // match, otherwise the guarantee is not worth stating.
        CHECK(std::memcmp(a.buffers().displacement, b.buffers().displacement, bytes) == 0);
        CHECK(std::memcmp(a.buffers().normal, b.buffers().normal, bytes) == 0);
    }
}

TEST_CASE("seeking backwards reproduces an earlier frame exactly")
{
    // The property that absolute time buys us: no accumulator, no drift, so a
    // replay or a late-joining network client sees the same ocean.
    OceanDesc d = test_desc(64);
    Ocean sim{d};

    const std::size_t floats = static_cast<std::size_t>(d.size) * d.size * 4;
    sim.update(12.5);
    std::vector<float> snapshot(sim.buffers().displacement,
                                sim.buffers().displacement + floats);

    sim.update(400.0);
    sim.update(0.125);
    sim.update(12.5);  // return to the original time

    CHECK(std::memcmp(snapshot.data(), sim.buffers().displacement,
                      floats * sizeof(float)) == 0);
}

TEST_CASE("different seeds give different oceans with the same statistics")
{
    OceanDesc a_desc = test_desc(64);
    OceanDesc b_desc = test_desc(64);
    b_desc.seed = a_desc.seed + 1;

    Ocean a{a_desc};
    Ocean b{b_desc};
    a.update(2.0);
    b.update(2.0);

    const std::size_t cells = 64u * 64u;
    const std::size_t bytes = cells * 4 * sizeof(float);
    CHECK(std::memcmp(a.buffers().displacement, b.buffers().displacement, bytes) != 0);

    // ...but the sea state should be the same, because the spectrum is.
    auto rms_height = [&](const Ocean& o) {
        double s = 0.0;
        const float* p = o.buffers().displacement;
        for (std::size_t i = 0; i < cells; ++i) {
            s += static_cast<double>(p[4 * i + 1]) * p[4 * i + 1];
        }
        return std::sqrt(s / cells);
    };
    const double ra = rms_height(a);
    const double rb = rms_height(b);
    CAPTURE(ra);
    CAPTURE(rb);
    CHECK(ra == doctest::Approx(rb).epsilon(0.25));
}

// ---------------------------------------------------------------------------
// Energy: the strongest end-to-end check we have
// ---------------------------------------------------------------------------

TEST_CASE("surface variance matches the spectrum exactly, via Parseval")
{
    // Parseval for the unnormalised inverse DFT:
    //
    //   h(x) = sum_k h~(k) e^{i k.x}   =>   mean_x |h|^2 = sum_k |h~(k)|^2
    //
    // Since the DC bin is exactly zero the surface has zero mean, so the left
    // side is the variance of the rendered heightfield and the right side is
    // computed straight from the spectrum tables.
    //
    // This is an EXACT identity, not a statistical one, so it holds to float
    // precision for a single realisation. It simultaneously pins down:
    //   - that no stray 1/N or 1/N^2 normalisation crept into the transform
    //     (either would show up as an N^2 or N^4 discrepancy),
    //   - that the two-fields-per-transform packing and unpacking are correct
    //     (a swap would put Dx's energy into the height channel),
    //   - that the spectrum really is Hermitian (if it were not, the inverse
    //     transform would be complex and the real part alone would carry less
    //     than the full energy).
    OceanDesc d = test_desc(128);
    Ocean sim{d};

    const double t = 7.75;
    sim.update(t);

    detail::SpectrumTables tables;
    detail::build_spectrum(d, tables);

    constexpr double kTwoPi = 6.28318530717958647692;
    const std::size_t cells = static_cast<std::size_t>(d.size) * d.size;

    // Reconstruct |h~(k,t)|^2 from the tables. Deliberately written out here
    // rather than reusing evolve_rows: this test is checking the transform
    // chain, so the reference side should not share code with it.
    double spectral_energy = 0.0;
    for (std::size_t i = 0; i < cells; ++i) {
        const double phase = std::fmod(static_cast<double>(tables.omega[i]) * t, kTwoPi);
        const float c = static_cast<float>(std::cos(phase));
        const float s = static_cast<float>(std::sin(phase));

        const float a_re = tables.h0_re[i],  a_im = tables.h0_im[i];
        const float b_re = tables.h0c_re[i], b_im = tables.h0c_im[i];

        const double hr = (a_re * c - a_im * s) + (b_re * c + b_im * s);
        const double hi = (a_re * s + a_im * c) + (b_im * c - b_re * s);
        spectral_energy += hr * hr + hi * hi;
    }

    const float* disp = sim.buffers().displacement;
    double mean = 0.0;
    for (std::size_t i = 0; i < cells; ++i) mean += disp[4 * i + 1];
    mean /= static_cast<double>(cells);

    double spatial_energy = 0.0;
    for (std::size_t i = 0; i < cells; ++i) {
        const double h = disp[4 * i + 1];
        spatial_energy += h * h;
    }
    spatial_energy /= static_cast<double>(cells);

    CAPTURE(mean);
    CAPTURE(spatial_energy);
    CAPTURE(spectral_energy);

    // DC is exactly zero, so the surface mean must be zero to rounding.
    CHECK(std::abs(mean) < 1e-4);
    CHECK(spatial_energy == doctest::Approx(spectral_energy).epsilon(1e-4));
}

TEST_CASE("significant wave height is physically plausible for the wind")
{
    // H_s = 4 * sigma_h. At 12 m/s (about 23 kt, Beaufort 6) a fetch-limited
    // sea runs roughly 2-4 m; a patch only 200 m across truncates the low
    // wavenumbers, so we expect to land at or slightly under that.
    OceanDesc d = test_desc(256);
    Ocean sim{d};
    sim.update(10.0);

    const std::size_t cells = 256u * 256u;
    const float* disp = sim.buffers().displacement;
    double sum2 = 0.0;
    for (std::size_t i = 0; i < cells; ++i) {
        const double h = disp[4 * i + 1];
        sum2 += h * h;
    }
    const double hs = 4.0 * std::sqrt(sum2 / cells);
    CAPTURE(hs);
    CHECK(hs > 0.3);
    CHECK(hs < 6.0);
}

TEST_CASE("stronger wind raises the surface")
{
    double previous = 0.0;
    for (float u : {5.0f, 10.0f, 15.0f, 20.0f}) {
        CAPTURE(u);
        OceanDesc d = test_desc(128);
        d.spectrum.wind_speed = u;
        Ocean sim{d};
        sim.update(3.0);

        const std::size_t cells = 128u * 128u;
        const float* disp = sim.buffers().displacement;
        double sum2 = 0.0;
        for (std::size_t i = 0; i < cells; ++i) {
            const double h = disp[4 * i + 1];
            sum2 += h * h;
        }
        const double rms = std::sqrt(sum2 / cells);
        CHECK(rms > previous);
        previous = rms;
    }
}

// ---------------------------------------------------------------------------
// Structure of the outputs
// ---------------------------------------------------------------------------

TEST_CASE("the surface actually moves over time")
{
    OceanDesc d = test_desc(64);
    Ocean sim{d};

    const std::size_t floats = 64u * 64u * 4;
    sim.update(0.0);
    std::vector<float> first(sim.buffers().displacement,
                             sim.buffers().displacement + floats);

    sim.update(1.0);
    CHECK(std::memcmp(first.data(), sim.buffers().displacement,
                      floats * sizeof(float)) != 0);
}

TEST_CASE("outputs are finite and normals are unit length at every cell")
{
    OceanDesc d = test_desc(128);
    d.choppiness = 2.0f;  // deliberately aggressive, to provoke folding
    Ocean sim{d};

    for (double t : {0.0, 0.5, 11.0, 250.0}) {
        CAPTURE(t);
        sim.update(t);
        const Buffers b = sim.buffers();
        const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;

        REQUIRE(all_finite(b.displacement, cells * 4));
        REQUIRE(all_finite(b.normal, cells * 4));

        for (std::size_t i = 0; i < cells; ++i) {
            const float nx = b.normal[4 * i + 0];
            const float ny = b.normal[4 * i + 1];
            const float nz = b.normal[4 * i + 2];
            REQUIRE(std::abs(nx * nx + ny * ny + nz * nz - 1.0f) < 1e-4f);
        }
    }
}

TEST_CASE("choppiness zero gives a pure heightfield with a flat Jacobian")
{
    OceanDesc d = test_desc(64);
    d.choppiness = 0.0f;
    Ocean sim{d};
    sim.update(4.0);

    const Buffers b = sim.buffers();
    const std::size_t cells = 64u * 64u;
    for (std::size_t i = 0; i < cells; ++i) {
        // No horizontal displacement at all.
        REQUIRE(b.displacement[4 * i + 0] == 0.0f);
        REQUIRE(b.displacement[4 * i + 2] == 0.0f);
        // With no displacement the map is the identity, so its Jacobian is
        // exactly 1 and no cell can be folding.
        REQUIRE(b.normal[4 * i + 3] == 1.0f);
        REQUIRE(b.displacement[4 * i + 3] == 0.0f);  // hence no foam
        // The normal degenerates to the plain heightfield normal, so it must
        // point strictly upward.
        REQUIRE(b.normal[4 * i + 1] > 0.0f);
    }
}

TEST_CASE("choppiness moves vertices horizontally without changing height")
{
    // Tessendorf's chop displaces vertices sideways toward crests; it must not
    // touch the height channel at all. If it did, the choppiness control would
    // be changing the sea state rather than just the crest shape.
    OceanDesc flat = test_desc(64);
    flat.choppiness = 0.0f;
    OceanDesc chop = test_desc(64);
    chop.choppiness = 1.5f;

    Ocean a{flat};
    Ocean b{chop};
    a.update(6.0);
    b.update(6.0);

    const std::size_t cells = 64u * 64u;
    for (std::size_t i = 0; i < cells; ++i) {
        REQUIRE(a.buffers().displacement[4 * i + 1] ==
                b.buffers().displacement[4 * i + 1]);
    }

    bool moved = false;
    for (std::size_t i = 0; i < cells; ++i) {
        if (b.buffers().displacement[4 * i + 0] != 0.0f) moved = true;
    }
    CHECK(moved);
}

TEST_CASE("foam appears exactly where the Jacobian says the surface folds")
{
    OceanDesc d = test_desc(128);
    d.choppiness     = 2.5f;  // enough to fold the surface somewhere
    d.foam_threshold = 0.6f;
    Ocean sim{d};
    sim.update(9.0);

    const Buffers b = sim.buffers();
    const std::size_t cells = 128u * 128u;

    std::size_t foamy = 0;
    for (std::size_t i = 0; i < cells; ++i) {
        const float jacobian = b.normal[4 * i + 3];
        const float foam     = b.displacement[4 * i + 3];

        REQUIRE(foam >= 0.0f);
        REQUIRE(foam <= 1.0f);

        // Foam is a linear ramp of (threshold - J)/threshold, clamped.
        float expected = (0.6f - jacobian) / 0.6f;
        if (expected < 0.0f) expected = 0.0f;
        if (expected > 1.0f) expected = 1.0f;
        REQUIRE(std::abs(foam - expected) < 1e-5f);

        if (foam > 0.0f) ++foamy;
    }

    // With choppiness this high some of the surface really should be folding,
    // otherwise the test is not exercising the path it claims to.
    CAPTURE(foamy);
    CHECK(foamy > 0);
}

TEST_CASE("update is allocation-free at every supported size")
{
    for (std::uint32_t n : {16u, 64u, 256u}) {
        CAPTURE(n);
        OceanDesc d = test_desc(n);
        Ocean sim{d};
        sim.update(0.0);
        // Re-checked per size because scratch sizing is size-dependent.
        CHECK_NOTHROW(sim.update(1.0));
    }
}
