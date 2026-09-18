// The C API for orbital velocity and the interaction field.
//
// The load-bearing checks are the ones that would let bindings quietly
// simulate a DIFFERENT ocean from the C++ API: defaults drifting apart, a
// legacy struct being misread, and the two paths disagreeing numerically.
#include "ocean/interaction.hpp"
#include "ocean/ocean.h"
#include "ocean/ocean.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <vector>

TEST_CASE("interaction desc defaults match the C++ defaults exactly")
{
    // Read from the C++ defaults rather than restated, so they cannot drift -
    // but that is the implementation's claim, and this is the test that holds
    // it to it. Duplicated defaults are a classic source of silent divergence
    // between an API and its wrapper.
    ocean_interaction_desc c;
    ocean_interaction_desc_init(&c);
    const ocean::InteractionDesc d{};

    CHECK(c.struct_size   == sizeof(ocean_interaction_desc));
    CHECK(c.size          == d.size);
    CHECK(c.extent        == d.extent);
    CHECK(c.kernel_radius == d.kernel_radius);
    CHECK(c.damping       == d.damping);
    CHECK(c.absorb_cells  == d.absorb_cells);
    CHECK(c.fixed_dt      == d.fixed_dt);
    CHECK(c.max_substeps  == d.max_substeps);
    CHECK(c.max_sources   == d.max_sources);
    CHECK(c.gravity       == d.gravity);
    CHECK(c.water_depth   == d.depth);
    CHECK(c.thread_count  == d.thread_count);
    CHECK(c.obstruction   == OCEAN_OBSTRUCTION_NEUMANN);
    CHECK(d.obstruction   == ocean::Obstruction::Neumann);
}

TEST_CASE("the C and C++ interaction paths produce identical fields")
{
    // If these ever diverged, every non-C++ binding would be simulating a
    // different ocean - the kind of discrepancy nobody finds for months.
    ocean_interaction_desc cd;
    ocean_interaction_desc_init(&cd);
    cd.size         = 128;
    cd.extent       = 32.0f;
    cd.damping      = 0.1f;
    cd.thread_count = 1;

    ocean_status st = OCEAN_ERROR_UNKNOWN;
    ocean_interaction* cf = ocean_interaction_create(&cd, &st);
    REQUIRE(cf != nullptr);
    CHECK(st == OCEAN_OK);

    ocean::InteractionDesc pd;
    pd.size         = 128;
    pd.extent       = 32.0f;
    pd.damping      = 0.1f;
    pd.thread_count = 1;
    ocean::InteractionField pf{pd};

    for (int i = 0; i < 40; ++i) {
        if (i % 7 == 0) {
            ocean_disturbance d{};
            d.world_x    = 12.0f + 0.3f * i;
            d.world_z    = 14.0f;
            d.radius     = 0.5f;
            d.strength   = 0.2f;
            d.velocity_y = -1.0f;
            d.kind       = OCEAN_SOURCE_IMPULSE;
            ocean_interaction_add(cf, &d);

            ocean::Disturbance p;
            p.world_x    = d.world_x;
            p.world_z    = d.world_z;
            p.radius     = d.radius;
            p.strength   = d.strength;
            p.velocity_y = d.velocity_y;
            p.kind       = ocean::SourceKind::Impulse;
            pf.add(p);
        }
        ocean_interaction_update(cf, 1.0f / 60.0f);
        pf.update(1.0f / 60.0f);
    }

    const ocean_interaction_buffers cb = ocean_interaction_get_buffers(cf);
    const ocean::InteractionBuffers pb = pf.buffers();
    REQUIRE(cb.size == pb.size);
    const std::size_t bytes =
        static_cast<std::size_t>(cb.size) * cb.size * 4 * sizeof(float);
    CHECK(std::memcmp(cb.field, pb.field, bytes) == 0);

    CHECK(ocean_interaction_fixed_dt(cf) == pf.fixed_dt());
    CHECK(ocean_interaction_dt_limit(cf) == pf.stable_dt_limit());

    ocean_interaction_sample sc{};
    ocean_interaction_sample_at(cf, 13.0f, 14.5f, &sc);
    const ocean::InteractionSample sp = pf.sample_at(13.0f, 14.5f);
    CHECK(sc.height     == sp.height);
    CHECK(sc.slope_x    == sp.slope_x);
    CHECK(sc.velocity_y == sp.velocity_y);

    ocean_interaction_destroy(cf);
}

TEST_CASE("a legacy interaction descriptor still works")
{
    // Models a caller compiled against a header that stopped before
    // `obstruction` and the threading fields existed. The library must copy
    // exactly the bytes they provided and default the rest.
    //
    // This is the mechanism ADR-019 documents, exercised for the new struct.
    struct LegacyDesc {
        size_t   struct_size;
        uint32_t size;
        float    extent;
        uint32_t kernel_radius;
        float    damping;
        uint32_t absorb_cells;
        float    fixed_dt;
        uint32_t max_substeps;
        uint32_t max_sources;
        float    gravity;
        float    water_depth;
    };
    static_assert(offsetof(LegacyDesc, water_depth) ==
                      offsetof(ocean_interaction_desc, water_depth),
                  "legacy struct must be a byte-for-byte prefix of the current one");

    LegacyDesc legacy;
    std::memset(&legacy, 0, sizeof(legacy));   // zero first, as the convention requires
    legacy.struct_size   = sizeof(LegacyDesc);
    legacy.size          = 128;
    legacy.extent        = 32.0f;
    legacy.kernel_radius = 6;
    legacy.damping       = 0.1f;
    legacy.absorb_cells  = 16;
    legacy.fixed_dt      = 0.0f;
    legacy.max_substeps  = 8;
    legacy.max_sources   = 64;
    legacy.gravity       = 9.81f;
    legacy.water_depth   = 0.0f;

    ocean_status st = OCEAN_ERROR_UNKNOWN;
    ocean_interaction* old_caller = ocean_interaction_create(
        reinterpret_cast<const ocean_interaction_desc*>(&legacy), &st);
    REQUIRE(old_caller != nullptr);
    CHECK(st == OCEAN_OK);

    ocean_interaction_desc current;
    ocean_interaction_desc_init(&current);
    current.size   = 128;
    current.extent = 32.0f;
    current.damping = 0.1f;
    ocean_interaction* new_caller = ocean_interaction_create(&current, &st);
    REQUIRE(new_caller != nullptr);

    for (int i = 0; i < 20; ++i) {
        if (i == 0) {
            ocean_disturbance d{};
            d.world_x = 16.0f; d.world_z = 16.0f;
            d.radius = 0.5f;   d.strength = 0.3f;
            d.kind = OCEAN_SOURCE_IMPULSE;
            ocean_interaction_add(old_caller, &d);
            ocean_interaction_add(new_caller, &d);
        }
        ocean_interaction_update(old_caller, 1.0f / 60.0f);
        ocean_interaction_update(new_caller, 1.0f / 60.0f);
    }

    const ocean_interaction_buffers a = ocean_interaction_get_buffers(old_caller);
    const ocean_interaction_buffers b = ocean_interaction_get_buffers(new_caller);
    const std::size_t bytes =
        static_cast<std::size_t>(a.size) * a.size * 4 * sizeof(float);
    CHECK(std::memcmp(a.field, b.field, bytes) == 0);

    ocean_interaction_destroy(old_caller);
    ocean_interaction_destroy(new_caller);
}

TEST_CASE("a struct_size from the future is refused, not truncated")
{
    ocean_interaction_desc d;
    ocean_interaction_desc_init(&d);
    d.struct_size = sizeof(ocean_interaction_desc) + 16;

    ocean_status st = OCEAN_OK;
    CHECK(ocean_interaction_create(&d, &st) == nullptr);
    CHECK(st == OCEAN_ERROR_INVALID_ARG);

    // Zero is meaningless too.
    d.struct_size = 0;
    CHECK(ocean_interaction_create(&d, &st) == nullptr);
    CHECK(st == OCEAN_ERROR_INVALID_ARG);
}

TEST_CASE("ocean_surface is frozen, and velocity comes through the ex path")
{
    // ocean_surface is returned BY VALUE, so appending to it would change the
    // ABI of ocean_sample_at and corrupt the return slot of every binary
    // already compiled against it. This pins the size so nobody grows it by
    // accident - struct_size cannot save this case, because the caller never
    // passes the struct in.
    CHECK(sizeof(ocean_surface) == 7 * sizeof(float));

    ocean_desc d;
    ocean_desc_init(&d);
    d.size             = 64;
    d.seed             = 99;
    d.compute_velocity = 1;
    CHECK(d.compute_velocity == 1);

    ocean_status st = OCEAN_ERROR_UNKNOWN;
    ocean_sim* sim = ocean_create(&d, &st);
    REQUIRE(sim != nullptr);
    ocean_update(sim, 5.0);

    const ocean_surface base = ocean_sample_at(sim, 3.0f, 4.0f);
    ocean_surface_ex ex{};
    ocean_sample_ex(sim, 3.0f, 4.0f, &ex);

    // The shared fields must agree exactly - the two entry points are the same
    // query.
    CHECK(ex.height   == base.height);
    CHECK(ex.offset_x == base.offset_x);
    CHECK(ex.normal_y == base.normal_y);
    CHECK(ex.foam     == base.foam);

    CHECK(ocean_get_velocity(sim) != nullptr);

    // And it matches the C++ API.
    ocean::OceanDesc pd;
    pd.size             = 64;
    pd.seed             = 99;
    pd.compute_velocity = true;
    ocean::Ocean psim{pd};
    psim.update(5.0);
    const ocean::Surface ps = psim.sample_at(3.0f, 4.0f);
    CHECK(ex.velocity_x == ps.velocity_x);
    CHECK(ex.velocity_y == ps.velocity_y);
    CHECK(ex.velocity_z == ps.velocity_z);

    ocean_destroy(sim);
}

TEST_CASE("velocity is absent through C unless requested")
{
    ocean_desc d;
    ocean_desc_init(&d);
    d.size = 64;
    CHECK(d.compute_velocity == 0);   // the default a zeroed legacy struct gets

    ocean_sim* sim = ocean_create(&d, nullptr);
    REQUIRE(sim != nullptr);
    ocean_update(sim, 1.0);
    CHECK(ocean_get_velocity(sim) == nullptr);

    ocean_surface_ex ex{};
    ocean_sample_ex(sim, 1.0f, 1.0f, &ex);
    CHECK(ex.velocity_x == 0.0f);
    CHECK(ex.velocity_y == 0.0f);
    CHECK(ex.velocity_z == 0.0f);
    ocean_destroy(sim);
}

TEST_CASE("combined query through C matches WaterSurface")
{
    ocean_desc od;
    ocean_desc_init(&od);
    od.size             = 64;
    od.seed             = 7;
    od.compute_velocity = 1;
    ocean_sim* sim = ocean_create(&od, nullptr);
    REQUIRE(sim != nullptr);
    ocean_update(sim, 3.0);

    ocean_interaction_desc id;
    ocean_interaction_desc_init(&id);
    id.size         = 128;
    id.extent       = 32.0f;
    id.thread_count = 1;
    ocean_interaction* field = ocean_interaction_create(&id, nullptr);
    REQUIRE(field != nullptr);

    ocean_disturbance dd{};
    dd.world_x = 16.0f; dd.world_z = 16.0f;
    dd.radius = 0.6f;   dd.strength = 0.4f;
    dd.kind = OCEAN_SOURCE_IMPULSE;
    ocean_interaction_add(field, &dd);
    for (int i = 0; i < 25; ++i) ocean_interaction_update(field, 1.0f / 60.0f);

    ocean_surface_ex c{};
    ocean_sample_combined(sim, field, 16.5f, 16.2f, &c);

    // Same thing through C++.
    ocean::OceanDesc pod;
    pod.size             = 64;
    pod.seed             = 7;
    pod.compute_velocity = true;
    ocean::Ocean psim{pod};
    psim.update(3.0);

    ocean::InteractionDesc pid;
    pid.size         = 128;
    pid.extent       = 32.0f;
    pid.thread_count = 1;
    ocean::InteractionField pfield{pid};
    ocean::Disturbance pd2;
    pd2.world_x = 16.0f; pd2.world_z = 16.0f;
    pd2.radius = 0.6f;   pd2.strength = 0.4f;
    pfield.add(pd2);
    for (int i = 0; i < 25; ++i) pfield.update(1.0f / 60.0f);

    const ocean::WaterSurface water{psim, pfield};
    const ocean::Surface ps = water.sample_at(16.5f, 16.2f);

    CHECK(c.height     == ps.height);
    CHECK(c.normal_x   == ps.normal_x);
    CHECK(c.normal_y   == ps.normal_y);
    CHECK(c.velocity_y == ps.velocity_y);

    // The interaction field really did contribute.
    CHECK(std::fabs(c.height - psim.sample_at(16.5f, 16.2f).height) > 1e-6f);

    ocean_interaction_destroy(field);
    ocean_destroy(sim);
}

TEST_CASE("null handles are tolerated everywhere")
{
    // Bindings routinely call into a handle that failed to construct; crashing
    // inside the library turns a caller-side bug into a support ticket.
    ocean_interaction_destroy(nullptr);
    ocean_interaction_update(nullptr, 0.016f);
    ocean_interaction_add(nullptr, nullptr);
    ocean_interaction_set_obstruction(nullptr, nullptr);
    CHECK(ocean_interaction_dropped(nullptr) == 0u);
    CHECK(ocean_interaction_recenter(nullptr, 0.0f, 0.0f) == 0);
    CHECK(ocean_interaction_height_at(nullptr, 0.0f, 0.0f) == 0.0f);
    CHECK(ocean_interaction_fixed_dt(nullptr) == 0.0f);
    CHECK(ocean_interaction_dt_limit(nullptr) == 0.0f);
    CHECK(ocean_interaction_dispersion_error(nullptr) == 0.0f);
    CHECK(ocean_interaction_get_buffers(nullptr).field == nullptr);
    CHECK(ocean_get_velocity(nullptr) == nullptr);

    ocean_interaction_sample s{};
    ocean_interaction_sample_at(nullptr, 0.0f, 0.0f, &s);
    CHECK(s.height == 0.0f);

    ocean_surface_ex ex{};
    ocean_sample_ex(nullptr, 0.0f, 0.0f, &ex);
    CHECK(ex.normal_y == 1.0f);
    ocean_sample_combined(nullptr, nullptr, 0.0f, 0.0f, &ex);
    CHECK(ex.normal_y == 1.0f);
}

TEST_CASE("queue overflow and recentre report through C")
{
    ocean_interaction_desc d;
    ocean_interaction_desc_init(&d);
    d.size        = 64;
    d.extent      = 16.0f;
    d.max_sources = 2;
    ocean_interaction* f = ocean_interaction_create(&d, nullptr);
    REQUIRE(f != nullptr);

    ocean_disturbance dd{};
    dd.radius = 0.4f; dd.strength = 0.1f;
    for (int i = 0; i < 5; ++i) ocean_interaction_add(f, &dd);
    CHECK(ocean_interaction_dropped(f) == 3u);

    // The origin is the grid's LOW CORNER, not its centre, so centring on
    // world (0,0) legitimately moves it by half the extent on the first call.
    CHECK(ocean_interaction_recenter(f, 0.0f, 0.0f) == 1);
    // Idempotent: asking for the same centre again must not move anything.
    CHECK(ocean_interaction_recenter(f, 0.0f, 0.0f) == 0);
    // A sub-cell nudge must not move the grid - snapping to whole cells is
    // what makes the move exact.
    const float cell = 16.0f / 64.0f;
    CHECK(ocean_interaction_recenter(f, 0.1f * cell, 0.0f) == 0);
    // A whole-cell step must.
    CHECK(ocean_interaction_recenter(f, 100.0f, 100.0f) == 1);

    ocean_interaction_destroy(f);
}
