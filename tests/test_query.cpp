#include <doctest/doctest.h>

#include "ocean/ocean.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ocean;

namespace {

OceanDesc query_desc(std::uint32_t size = 128, float choppiness = 1.0f)
{
    OceanDesc d;
    d.size                    = size;
    d.patch_length            = 200.0f;
    d.seed                    = 4242;
    d.choppiness              = choppiness;
    d.spectrum.wind_speed     = 12.0f;
    d.spectrum.wind_direction = 0.3f;
    return d;
}

// Worst and mean error between the height the query returns and the height the
// renderer actually draws, measured over every vertex of the displaced grid.
struct ErrorStats {
    double worst = 0.0;
    double mean  = 0.0;
    double rms_height = 0.0;
};

ErrorStats measure_agreement(const Ocean& sim)
{
    const Buffers b = sim.buffers();
    const std::uint32_t n = b.size;
    const float cell = b.patch_length / static_cast<float>(n);

    ErrorStats st;
    double total = 0.0, h2 = 0.0;
    const std::size_t cells = static_cast<std::size_t>(n) * n;

    for (std::uint32_t z = 0; z < n; ++z) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(z) * n + x;

            // Where this vertex is actually drawn in the world.
            const float world_x = x * cell + b.displacement[4 * i + 0];
            const float world_z = z * cell + b.displacement[4 * i + 2];
            const float drawn_h = b.displacement[4 * i + 1];

            const float queried = sim.height_at(world_x, world_z);
            const double err = std::abs(static_cast<double>(queried) - drawn_h);

            st.worst = std::max(st.worst, err);
            total += err;
            h2 += static_cast<double>(drawn_h) * drawn_h;
        }
    }
    st.mean = total / static_cast<double>(cells);
    st.rms_height = std::sqrt(h2 / static_cast<double>(cells));
    return st;
}

}  // namespace

// ---------------------------------------------------------------------------
// The promise: what you query is what you see
// ---------------------------------------------------------------------------

TEST_CASE("the height query agrees with the displaced grid the renderer draws")
{
    // Promise #2, stated as a test. For every vertex, take the world position
    // the renderer will actually place it at, ask the query API for the height
    // there, and require the answer to be the height of that vertex.
    //
    // A naive implementation that looked up cell (x,z) directly would fail
    // this badly, and would fail it worst at the crests - exactly where a boat
    // or a character notices.
    for (float chop : {0.0f, 0.5f, 1.0f, 1.5f}) {
        CAPTURE(chop);
        OceanDesc d = query_desc(128, chop);
        Ocean sim{d};
        sim.update(5.0);

        const ErrorStats st = measure_agreement(sim);
        CAPTURE(st.rms_height);
        CAPTURE(st.mean);
        CAPTURE(st.worst);

        // Tolerance is relative to the sea state: an error of 1 cm means
        // something different on a 10 cm chop than on a 3 m swell.
        CHECK(st.mean < 0.02 * st.rms_height);
        CHECK(st.worst < 0.25 * st.rms_height);
    }
}

TEST_CASE("with no choppiness the query is an exact grid lookup")
{
    // With lambda = 0 the displacement map is the identity, so the fixed-point
    // solve converges on the first step and the query reduces to a bilinear
    // fetch. At exact grid positions that fetch must reproduce the stored
    // value to the last bit of the interpolation arithmetic.
    OceanDesc d = query_desc(64, 0.0f);
    Ocean sim{d};
    sim.update(2.0);

    const Buffers b = sim.buffers();
    const float cell = b.patch_length / static_cast<float>(b.size);

    for (std::uint32_t z = 0; z < b.size; z += 7) {
        for (std::uint32_t x = 0; x < b.size; x += 5) {
            const std::size_t i = static_cast<std::size_t>(z) * b.size + x;
            const float queried = sim.height_at(x * cell, z * cell);
            REQUIRE(std::abs(queried - b.displacement[4 * i + 1]) < 1e-5f);
        }
    }
}

TEST_CASE("sample_at returns normal and foam consistent with the buffers")
{
    OceanDesc d = query_desc(64, 0.0f);
    Ocean sim{d};
    sim.update(3.0);

    const Buffers b = sim.buffers();
    const float cell = b.patch_length / static_cast<float>(b.size);

    for (std::uint32_t z = 0; z < b.size; z += 9) {
        for (std::uint32_t x = 0; x < b.size; x += 11) {
            const std::size_t i = static_cast<std::size_t>(z) * b.size + x;
            const Surface s = sim.sample_at(x * cell, z * cell);

            REQUIRE(std::abs(s.normal_x - b.normal[4 * i + 0]) < 1e-4f);
            REQUIRE(std::abs(s.normal_y - b.normal[4 * i + 1]) < 1e-4f);
            REQUIRE(std::abs(s.normal_z - b.normal[4 * i + 2]) < 1e-4f);
            REQUIRE(std::abs(s.foam - b.displacement[4 * i + 3]) < 1e-4f);
        }
    }
}

TEST_CASE("returned normals are unit length everywhere")
{
    // Bilinear interpolation of unit vectors does not preserve length, so the
    // query has to renormalise. A renderer that skipped this would get
    // darkened bands between texels.
    OceanDesc d = query_desc(64, 1.2f);
    Ocean sim{d};
    sim.update(8.0);

    for (int i = 0; i < 500; ++i) {
        const float x = -450.0f + 1.7f * i;
        const float z =  380.0f - 2.3f * i;
        const Surface s = sim.sample_at(x, z);
        const float len2 = s.normal_x * s.normal_x + s.normal_y * s.normal_y +
                           s.normal_z * s.normal_z;
        REQUIRE(std::abs(len2 - 1.0f) < 1e-4f);
    }
}

// ---------------------------------------------------------------------------
// Periodicity and robustness
// ---------------------------------------------------------------------------

TEST_CASE("queries are periodic with the patch length")
{
    // The FFT surface is exactly periodic, so the tile genuinely tiles and a
    // query anywhere in the world is well defined. This is why the sampler
    // wraps rather than clamps.
    OceanDesc d = query_desc(64, 1.0f);
    Ocean sim{d};
    sim.update(4.0);

    const float L = d.patch_length;
    for (int i = 0; i < 40; ++i) {
        const float x = -137.0f + 9.3f * i;
        const float z =   64.0f - 5.1f * i;

        const float base = sim.height_at(x, z);
        REQUIRE(std::abs(sim.height_at(x + L, z) - base) < 1e-4f);
        REQUIRE(std::abs(sim.height_at(x, z + L) - base) < 1e-4f);
        REQUIRE(std::abs(sim.height_at(x - 3.0f * L, z + 2.0f * L) - base) < 1e-4f);
    }
}

TEST_CASE("queries far outside the patch stay finite and bounded")
{
    OceanDesc d = query_desc(64, 1.0f);
    Ocean sim{d};
    sim.update(6.0);

    for (float m : {1.0f, 1e3f, 1e5f, -1e5f}) {
        CAPTURE(m);
        const Surface s = sim.sample_at(m * 3.1f, m * -7.9f);
        REQUIRE(std::isfinite(s.height));
        REQUIRE(std::isfinite(s.normal_y));
        REQUIRE(std::isfinite(s.foam));
        REQUIRE(std::abs(s.height) < 100.0f);
    }
}

TEST_CASE("extreme choppiness degrades gracefully rather than exploding")
{
    // Past the folding threshold the surface is genuinely multi-valued - at a
    // breaking wave there really are several surface points above one (x,z) -
    // so the fixed-point iteration stops being a contraction. It must still
    // return something finite and plausible rather than diverging or NaN-ing.
    OceanDesc d = query_desc(64, 4.0f);
    Ocean sim{d};
    sim.update(7.0);

    for (int i = 0; i < 2000; ++i) {
        const float x = -100.0f + 0.13f * i;
        const float z =   77.0f - 0.09f * i;
        const Surface s = sim.sample_at(x, z);
        REQUIRE(std::isfinite(s.height));
        REQUIRE(std::isfinite(s.offset_x));
        REQUIRE(std::abs(s.height) < 100.0f);
    }
}

TEST_CASE("the query follows the surface as it moves")
{
    OceanDesc d = query_desc(64, 1.0f);
    Ocean sim{d};

    sim.update(0.0);
    const float h0 = sim.height_at(37.0f, 91.0f);
    sim.update(2.5);
    const float h1 = sim.height_at(37.0f, 91.0f);
    CHECK(h0 != h1);

    // ...and going back reproduces it exactly.
    sim.update(0.0);
    CHECK(sim.height_at(37.0f, 91.0f) == h0);
}

TEST_CASE("queries on a flat ocean return exactly flat water")
{
    OceanDesc d = query_desc(32, 1.0f);
    d.spectrum.wind_speed = 0.0f;
    Ocean sim{d};
    sim.update(12.0);

    for (int i = 0; i < 50; ++i) {
        const Surface s = sim.sample_at(3.7f * i, -8.1f * i);
        REQUIRE(s.height == 0.0f);
        REQUIRE(s.offset_x == 0.0f);
        REQUIRE(s.offset_z == 0.0f);
        REQUIRE(s.normal_y == 1.0f);
        REQUIRE(s.foam == 0.0f);
    }
}
