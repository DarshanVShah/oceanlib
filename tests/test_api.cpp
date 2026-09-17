#include <doctest/doctest.h>

#include "alloc_probe.hpp"
#include "ocean/ocean.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

using namespace ocean;

TEST_CASE("descriptor validation rejects malformed configurations")
{
    OceanDesc d;

    SUBCASE("non-power-of-two size") {
        d.size = 100;
        CHECK_THROWS_AS(Ocean{d}, std::invalid_argument);
    }
    SUBCASE("size below the supported range") {
        d.size = 8;
        CHECK_THROWS_AS(Ocean{d}, std::invalid_argument);
    }
    SUBCASE("zero patch length") {
        d.patch_length = 0.0f;
        CHECK_THROWS_AS(Ocean{d}, std::invalid_argument);
    }
    SUBCASE("negative choppiness") {
        d.choppiness = -1.0f;
        CHECK_THROWS_AS(Ocean{d}, std::invalid_argument);
    }
    SUBCASE("peak enhancement below Pierson-Moskowitz") {
        d.spectrum.peak_enhancement = 0.5f;
        CHECK_THROWS_AS(Ocean{d}, std::invalid_argument);
    }
    SUBCASE("valid descriptor is accepted") {
        d.size = 64;
        CHECK_NOTHROW(Ocean{d});
    }
}

TEST_CASE("buffers are correctly shaped and cache-line aligned")
{
    OceanDesc d;
    d.size         = 64;
    d.patch_length = 137.0f;
    Ocean sim{d};

    const Buffers b = sim.buffers();
    REQUIRE(b.displacement != nullptr);
    REQUIRE(b.normal != nullptr);
    CHECK(b.size == 64);
    CHECK(b.patch_length == doctest::Approx(137.0f));

    // Alignment is what lets threaded tiles avoid false sharing.
    CHECK(reinterpret_cast<std::uintptr_t>(b.displacement) % kBufferAlignment == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(b.normal) % kBufferAlignment == 0);
}

TEST_CASE("a freshly constructed ocean is already evaluated at t = 0")
{
    // The constructor runs a full update, so buffers() and the query API are
    // meaningful before the caller has called update() even once. An engine
    // that builds the ocean during level load and uploads immediately should
    // not get a frame of flat water.
    OceanDesc d;
    d.size = 32;
    Ocean sim{d};

    CHECK(sim.time() == 0.0);

    const Buffers b = sim.buffers();
    const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;

    bool any_height = false;
    for (std::size_t i = 0; i < cells; ++i) {
        if (b.displacement[4 * i + 1] != 0.0f) any_height = true;
        // Normals are unit length everywhere, always.
        const float nx = b.normal[4 * i + 0];
        const float ny = b.normal[4 * i + 1];
        const float nz = b.normal[4 * i + 2];
        REQUIRE(std::abs(nx * nx + ny * ny + nz * nz - 1.0f) < 1e-4f);
    }
    CHECK(any_height);
}

TEST_CASE("zero wind speed produces a genuinely flat ocean")
{
    OceanDesc d;
    d.size = 32;
    d.spectrum.wind_speed = 0.0f;
    Ocean sim{d};
    sim.update(3.0);

    const Buffers b = sim.buffers();
    const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
    for (std::size_t i = 0; i < cells; ++i) {
        REQUIRE(b.displacement[4 * i + 0] == 0.0f);
        REQUIRE(b.displacement[4 * i + 1] == 0.0f);
        REQUIRE(b.displacement[4 * i + 2] == 0.0f);
        REQUIRE(b.displacement[4 * i + 3] == 0.0f);  // no foam on flat water
        REQUIRE(b.normal[4 * i + 1] == 1.0f);        // +Y up
        REQUIRE(b.normal[4 * i + 3] == 1.0f);        // unfolded Jacobian
    }
}

TEST_CASE("the allocation probe is actually installed")
{
    // Guards the test below from passing vacuously. If the global operator new
    // replacement silently failed to link, the counter would never move and
    // "update allocates nothing" would be meaningless.
    const std::size_t before = alloc_probe::count();
    OceanDesc d;
    d.size = 64;
    Ocean sim{d};
    CHECK(alloc_probe::count() > before);
}

TEST_CASE("update performs no heap allocation")
{
    OceanDesc d;
    d.size = 128;
    Ocean sim{d};

    sim.update(0.0);  // warm any one-time lazy state before measuring

    const std::size_t before = alloc_probe::count();
    for (int i = 0; i < 16; ++i) {
        sim.update(static_cast<double>(i) * 0.016);
    }
    const std::size_t after = alloc_probe::count();

    // Zero, not "few": a per-frame allocation is a frame-time spike waiting to
    // happen, and on a console it is a fragmentation bug.
    CHECK(after - before == 0);
}

TEST_CASE("update is absolute, not incremental")
{
    OceanDesc d;
    d.size = 32;
    Ocean sim{d};

    sim.update(5.0);
    CHECK(sim.time() == doctest::Approx(5.0));
    sim.update(1.0);  // seeking backwards is legal
    CHECK(sim.time() == doctest::Approx(1.0));
}

TEST_CASE("Ocean is movable and the descriptor round-trips")
{
    OceanDesc d;
    d.size                   = 64;
    d.seed                   = 99;
    d.spectrum.wind_speed    = 14.5f;
    d.spectrum.wind_direction = 1.25f;

    Ocean a{d};
    const float* original = a.buffers().displacement;

    Ocean b{std::move(a)};
    CHECK(b.buffers().displacement == original);  // moved, not copied
    CHECK(b.desc().seed == 99);
    CHECK(b.desc().spectrum.wind_speed == doctest::Approx(14.5f));
    CHECK(b.desc().spectrum.wind_direction == doctest::Approx(1.25f));
}
