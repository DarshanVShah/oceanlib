#include <doctest/doctest.h>

#include "ocean/ocean.h"
#include "ocean/ocean.hpp"

#include <cstring>
#include <vector>

extern "C" int ocean_c_header_smoke_test(void);

namespace {

ocean_desc make_c_desc(std::uint32_t n)
{
    ocean_desc d;
    ocean_desc_init(&d);
    d.size                    = n;
    d.patch_length            = 200.0f;
    d.seed                    = 555777;
    d.choppiness              = 1.0f;
    d.spectrum.wind_speed     = 11.0f;
    d.spectrum.wind_direction = 0.25f;
    return d;
}

ocean::OceanDesc make_cpp_desc(std::uint32_t n)
{
    ocean::OceanDesc d;
    d.size                    = n;
    d.patch_length            = 200.0f;
    d.seed                    = 555777;
    d.choppiness              = 1.0f;
    d.spectrum.wind_speed     = 11.0f;
    d.spectrum.wind_direction = 0.25f;
    return d;
}

}  // namespace

TEST_CASE("the header compiles and links as real C")
{
    // The value of this test is in the build, not the assertion: the
    // translation unit behind it is compiled by the C compiler. If the header
    // acquired a C++-only construct, or if an extern "C" declaration drifted,
    // this would fail to compile or fail to link rather than failing here.
    const int result = ocean_c_header_smoke_test();
    CAPTURE(result);
    CHECK(result == 0);
}

TEST_CASE("the C API reproduces the C++ API bit for bit")
{
    // The C wrapper must be a pure forwarding layer. Anything that changed the
    // numbers - a float narrowed somewhere, a defaulted field diverging -
    // would mean bindings quietly simulate a different ocean from the C++ API,
    // which is the kind of difference nobody finds for months.
    constexpr std::uint32_t n = 64;
    const std::size_t floats = static_cast<std::size_t>(n) * n * 4;
    const std::size_t bytes  = floats * sizeof(float);

    ocean::Ocean cpp_sim{make_cpp_desc(n)};

    ocean_desc cd = make_c_desc(n);
    ocean_status status = OCEAN_ERROR_UNKNOWN;
    ocean_sim* c_sim = ocean_create(&cd, &status);
    REQUIRE(c_sim != nullptr);
    REQUIRE(status == OCEAN_OK);

    for (double t : {0.0, 1.25, 60.5}) {
        CAPTURE(t);
        cpp_sim.update(t);
        ocean_update(c_sim, t);

        const ocean::Buffers cpp_b = cpp_sim.buffers();
        const ocean_buffers  c_b   = ocean_get_buffers(c_sim);

        REQUIRE(c_b.size == cpp_b.size);
        REQUIRE(c_b.patch_length == cpp_b.patch_length);
        CHECK(std::memcmp(cpp_b.displacement, c_b.displacement, bytes) == 0);
        CHECK(std::memcmp(cpp_b.normal, c_b.normal, bytes) == 0);

        // Queries too, not just the buffers.
        for (int i = 0; i < 20; ++i) {
            const float x = -90.0f + 7.3f * i;
            const float z = 45.0f - 3.1f * i;
            const ocean::Surface a = cpp_sim.sample_at(x, z);
            const ocean_surface  b = ocean_sample_at(c_sim, x, z);
            REQUIRE(a.height == b.height);
            REQUIRE(a.normal_y == b.normal_y);
            REQUIRE(a.foam == b.foam);
        }
    }

    ocean_destroy(c_sim);
}

TEST_CASE("invalid descriptors return null and a reason, never an exception")
{
    // A C caller cannot catch std::invalid_argument, and letting one unwind
    // through a C frame is undefined behaviour. Every failure must arrive as a
    // null handle plus a status.
    SUBCASE("non-power-of-two size") {
        ocean_desc d = make_c_desc(100);
        ocean_status s = OCEAN_OK;
        CHECK(ocean_create(&d, &s) == nullptr);
        CHECK(s == OCEAN_ERROR_INVALID_ARG);
    }
    SUBCASE("zero patch length") {
        ocean_desc d = make_c_desc(64);
        d.patch_length = 0.0f;
        ocean_status s = OCEAN_OK;
        CHECK(ocean_create(&d, &s) == nullptr);
        CHECK(s == OCEAN_ERROR_INVALID_ARG);
    }
    SUBCASE("null descriptor") {
        ocean_status s = OCEAN_OK;
        CHECK(ocean_create(nullptr, &s) == nullptr);
        CHECK(s == OCEAN_ERROR_INVALID_ARG);
    }
    SUBCASE("zero struct_size") {
        ocean_desc d = make_c_desc(64);
        d.struct_size = 0;
        ocean_status s = OCEAN_OK;
        CHECK(ocean_create(&d, &s) == nullptr);
        CHECK(s == OCEAN_ERROR_INVALID_ARG);
    }
    SUBCASE("a caller from the future") {
        // struct_size larger than we know about means the caller was built
        // against a NEWER header. We cannot guess what the extra fields mean,
        // and silently ignoring them is worse than refusing.
        ocean_desc d = make_c_desc(64);
        d.struct_size = sizeof(ocean_desc) + 64;
        ocean_status s = OCEAN_OK;
        CHECK(ocean_create(&d, &s) == nullptr);
        CHECK(s == OCEAN_ERROR_INVALID_ARG);
    }
    SUBCASE("out_status may be null") {
        ocean_desc d = make_c_desc(100);
        CHECK(ocean_create(&d, nullptr) == nullptr);
    }
}

namespace {

// Exactly the prefix of ocean_desc as it stood before `parallel_for`,
// `parallel_for_user` and `thread_count` were added: a caller compiled against
// an older header.
struct LegacyDesc {
    std::size_t         struct_size;
    std::uint32_t       size;
    float               patch_length;
    ocean_spectrum_desc spectrum;
    float               choppiness;
    float               foam_threshold;
    std::uint64_t       seed;
};

}  // namespace

TEST_CASE("struct_size lets an older caller keep working")
{
    // This is what struct_size is for. Without it, adding a field to
    // ocean_desc would make the library read past the end of an old caller's
    // smaller allocation - garbage at best, a page fault at worst. With it,
    // the library copies only the bytes the caller actually provided and
    // leaves its own defaults in the rest.
    static_assert(sizeof(LegacyDesc) < sizeof(ocean_desc),
                  "LegacyDesc must model a strictly older, smaller struct");

    LegacyDesc legacy{};
    legacy.struct_size = sizeof(LegacyDesc);
    legacy.size        = 64;
    legacy.patch_length = 200.0f;
    legacy.spectrum.wind_speed        = 11.0f;
    legacy.spectrum.fetch             = 100000.0f;
    legacy.spectrum.wind_direction    = 0.25f;
    legacy.spectrum.peak_enhancement  = 3.3f;
    legacy.spectrum.swell             = 0.0f;
    legacy.spectrum.small_wave_cutoff = 0.5f;
    legacy.spectrum.gravity           = 9.81f;
    legacy.choppiness     = 1.0f;
    legacy.foam_threshold = 0.5f;
    legacy.seed           = 555777;

    ocean_status s = OCEAN_ERROR_UNKNOWN;
    ocean_sim* sim =
        ocean_create(reinterpret_cast<const ocean_desc*>(&legacy), &s);
    REQUIRE(sim != nullptr);
    REQUIRE(s == OCEAN_OK);

    ocean_update(sim, 1.25);

    // And it must produce exactly the same ocean as a current caller that set
    // the new fields to their defaults.
    ocean::Ocean reference{make_cpp_desc(64)};
    reference.update(1.25);

    const std::size_t bytes = 64u * 64u * 4 * sizeof(float);
    CHECK(std::memcmp(reference.buffers().displacement,
                      ocean_get_buffers(sim).displacement, bytes) == 0);

    ocean_destroy(sim);
}

TEST_CASE("every entry point tolerates a null handle")
{
    // Bindings written in other languages routinely call into a handle that
    // failed to construct. Crashing inside the library makes that a support
    // ticket instead of a caller-side bug report.
    CHECK_NOTHROW(ocean_destroy(nullptr));
    CHECK_NOTHROW(ocean_update(nullptr, 1.0));
    CHECK(ocean_get_time(nullptr) == 0.0);
    CHECK(ocean_height_at(nullptr, 1.0f, 2.0f) == 0.0f);

    const ocean_buffers b = ocean_get_buffers(nullptr);
    CHECK(b.displacement == nullptr);
    CHECK(b.size == 0);

    const ocean_surface s = ocean_sample_at(nullptr, 0.0f, 0.0f);
    CHECK(s.height == 0.0f);
    CHECK(s.normal_y == 1.0f);  // still a usable normal, not a zero vector

    CHECK_NOTHROW(ocean_desc_init(nullptr));
    CHECK_NOTHROW(ocean_version(nullptr, nullptr, nullptr));
}

TEST_CASE("C defaults match C++ defaults exactly")
{
    // Duplicated defaults are a classic source of divergence between an API
    // and its wrapper. ocean_desc_init reads them from the C++ descriptor, and
    // this test holds that arrangement in place.
    ocean_desc c;
    ocean_desc_init(&c);
    const ocean::OceanDesc cpp{};

    CHECK(c.size == cpp.size);
    CHECK(c.patch_length == cpp.patch_length);
    CHECK(c.choppiness == cpp.choppiness);
    CHECK(c.foam_threshold == cpp.foam_threshold);
    CHECK(c.seed == cpp.seed);
    CHECK(c.thread_count == cpp.thread_count);
    CHECK(c.spectrum.wind_speed == cpp.spectrum.wind_speed);
    CHECK(c.spectrum.fetch == cpp.spectrum.fetch);
    CHECK(c.spectrum.wind_direction == cpp.spectrum.wind_direction);
    CHECK(c.spectrum.peak_enhancement == cpp.spectrum.peak_enhancement);
    CHECK(c.spectrum.swell == cpp.spectrum.swell);
    CHECK(c.spectrum.small_wave_cutoff == cpp.spectrum.small_wave_cutoff);
    CHECK(c.spectrum.gravity == cpp.spectrum.gravity);
}

namespace {
int g_hook_calls = 0;
void counting_hook(void* user, ocean_task_fn task, void* ctx, std::uint32_t count)
{
    (void)user;
    ++g_hook_calls;
    for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
}
}  // namespace

TEST_CASE("a scheduler hook supplied through the C API is used")
{
    g_hook_calls = 0;
    ocean_desc d = make_c_desc(64);
    d.parallel_for = &counting_hook;

    ocean_status s = OCEAN_ERROR_UNKNOWN;
    ocean_sim* sim = ocean_create(&d, &s);
    REQUIRE(sim != nullptr);

    CHECK(g_hook_calls == 4);  // the constructor evaluates one frame
    ocean_update(sim, 1.0);
    CHECK(g_hook_calls == 8);

    // The hook types must be layout-identical across the boundary, so a host
    // scheduler pays no trampoline: results must match the built-in path.
    ocean::Ocean reference{make_cpp_desc(64)};
    reference.update(1.0);
    const std::size_t bytes = 64u * 64u * 4 * sizeof(float);
    CHECK(std::memcmp(reference.buffers().displacement,
                      ocean_get_buffers(sim).displacement, bytes) == 0);

    ocean_destroy(sim);
}

TEST_CASE("status strings and version are reported")
{
    CHECK(std::strcmp(ocean_status_string(OCEAN_OK), "ok") == 0);
    CHECK(std::strlen(ocean_status_string(OCEAN_ERROR_INVALID_ARG)) > 0);
    CHECK(std::strlen(ocean_simd_level()) > 0);

    std::uint32_t major = 99, minor = 99, patch = 99;
    ocean_version(&major, &minor, &patch);
    CHECK(major == 0);
    CHECK(minor == 1);
}

TEST_CASE("forcing the SIMD level is reflected by ocean_simd_level")
{
    // Every build can run scalar, regardless of what the host CPU supports -
    // this is the QA hook's most basic promise: "does this repro on scalar?"
    // must always be answerable.
    const char* got = ocean_force_simd_level("scalar");
    REQUIRE(got != nullptr);
    CHECK(std::strcmp(got, "scalar") == 0);
    CHECK(std::strcmp(ocean_simd_level(), "scalar") == 0);

    // Case-insensitive, as documented.
    got = ocean_force_simd_level("SCALAR");
    CHECK(std::strcmp(got, "scalar") == 0);

    // An unrecognised name is a no-op: the level stays whatever it was, and
    // the call still returns a valid, non-null string rather than crashing or
    // silently corrupting the active level.
    got = ocean_force_simd_level("not-a-real-level");
    CHECK(std::strcmp(got, "scalar") == 0);  // unchanged from the force above

    // Restore whatever this machine actually supports, so later tests in this
    // binary are not left running in forced scalar mode.
    ocean_force_simd_level(nullptr);
}

TEST_CASE("a simulation built while a SIMD level is forced still matches scalar bit for bit")
{
    // The point of forcing a level is testing on hardware you do not have.
    // That is worthless unless the forced kernel actually agrees with every
    // other kernel - so this checks the ACTUAL simulation output, not just
    // that the reported name changed.
    ocean_desc d = make_c_desc(32);
    d.seed = 555;

    ocean_force_simd_level("scalar");
    ocean_sim* scalar_sim = ocean_create(&d, nullptr);
    REQUIRE(scalar_sim != nullptr);
    ocean_update(scalar_sim, 4.0);
    const ocean_buffers scalar_b = ocean_get_buffers(scalar_sim);

    ocean_force_simd_level(nullptr);  // back to native max for this CPU
    ocean_sim* native_sim = ocean_create(&d, nullptr);
    REQUIRE(native_sim != nullptr);
    ocean_update(native_sim, 4.0);
    const ocean_buffers native_b = ocean_get_buffers(native_sim);

    const std::size_t bytes = 32u * 32u * 4 * sizeof(float);
    CHECK(std::memcmp(scalar_b.displacement, native_b.displacement, bytes) == 0);
    CHECK(std::memcmp(scalar_b.normal, native_b.normal, bytes) == 0);

    ocean_destroy(scalar_sim);
    ocean_destroy(native_sim);
}
