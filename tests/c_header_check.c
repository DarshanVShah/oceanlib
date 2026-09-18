/* Compiled as C, not C++.
 *
 * The point of this file is the compiler invocation, not the code: a "C API"
 * header that only ever gets included from C++ drifts into using C++-only
 * constructs within a release or two - default arguments, `bool` without
 * stdbool, `//` comments in a C89 target, a stray `class`. Building one real
 * C translation unit against it on every CI run is the only way to know the
 * header is what it claims to be.
 *
 * It also exercises the calling convention end to end: this object file is
 * compiled by the C compiler and linked against C++ objects, so a mismatch in
 * `extern "C"` linkage shows up as a link error here rather than in a user's
 * project.
 */

#include "ocean/ocean.h"

/* A scheduler hook written in C, matching the published signature exactly. */
static void c_serial_for(void* user, ocean_task_fn task, void* ctx,
                         uint32_t count)
{
    uint32_t i;
    (void)user;
    for (i = 0; i < count; ++i) {
        task(ctx, i);
    }
}

/* Returns 0 on success, or a small non-zero code identifying what failed. The
 * C++ test suite calls this and reports the result. */
int ocean_c_header_smoke_test(void)
{
    ocean_desc    desc;
    ocean_status  status = OCEAN_ERROR_UNKNOWN;
    ocean_sim*    sim;
    ocean_buffers buffers;
    ocean_surface surface;
    float         height;
    uint32_t      major = 0, minor = 0, patch = 0;

    ocean_desc_init(&desc);
    if (desc.struct_size != sizeof(ocean_desc)) return 1;
    if (desc.spectrum.gravity <= 0.0f) return 2;

    desc.size         = 32;
    desc.seed         = 12345u;
    desc.parallel_for = &c_serial_for;

    sim = ocean_create(&desc, &status);
    if (sim == NULL) return 3;
    if (status != OCEAN_OK) return 4;

    ocean_update(sim, 2.5);
    if (ocean_get_time(sim) != 2.5) return 5;

    buffers = ocean_get_buffers(sim);
    if (buffers.displacement == NULL) return 6;
    if (buffers.normal == NULL) return 7;
    if (buffers.size != 32u) return 8;

    height  = ocean_height_at(sim, 3.0f, 4.0f);
    surface = ocean_sample_at(sim, 3.0f, 4.0f);
    if (surface.height != height) return 9;

    ocean_version(&major, &minor, &patch);
    if (ocean_simd_level() == NULL) return 10;
    if (ocean_status_string(OCEAN_OK) == NULL) return 11;

    /* The v3 surface, exercised from real C. The value here is in the BUILD,
     * not the assertions: a "C API" header that is only ever included from C++
     * drifts into C++-only constructs within a release or two, and a missing
     * extern "C" shows up here as a link error rather than in a user's
     * project. */
    {
        ocean_surface_ex ex;
        ocean_interaction_desc idesc;
        ocean_interaction* field;
        ocean_interaction_buffers ib;
        ocean_interaction_sample isample;
        ocean_disturbance d;
        unsigned char mask[32 * 32];
        int i;

        ocean_sample_ex(sim, 3.0f, 4.0f, &ex);
        if (ex.height != surface.height) return 12;
        /* compute_velocity was not requested, so this must be absent. */
        if (ocean_get_velocity(sim) != NULL) return 13;

        ocean_interaction_desc_init(&idesc);
        if (idesc.struct_size != sizeof(ocean_interaction_desc)) return 14;
        idesc.size         = 64;
        idesc.extent       = 16.0f;
        idesc.thread_count = 1;
        idesc.obstruction  = OCEAN_OBSTRUCTION_NEUMANN;

        field = ocean_interaction_create(&idesc, &status);
        if (field == NULL) return 15;
        if (status != OCEAN_OK) return 16;

        for (i = 0; i < 32 * 32; ++i) mask[i] = 0;
        ocean_interaction_set_obstruction(field, mask);
        ocean_interaction_set_obstruction(field, NULL);

        d.world_x    = 8.0f;
        d.world_z    = 8.0f;
        d.radius     = 0.5f;
        d.strength   = 0.25f;
        d.velocity_y = -1.0f;
        d.velocity_x = 0.0f;
        d.velocity_z = 0.0f;
        d.kind       = OCEAN_SOURCE_IMPULSE;
        ocean_interaction_add(field, &d);

        for (i = 0; i < 10; ++i) ocean_interaction_update(field, 1.0f / 60.0f);

        ib = ocean_interaction_get_buffers(field);
        if (ib.field == NULL) return 17;
        if (ib.size != 64u) return 18;

        ocean_interaction_sample_at(field, 8.0f, 8.0f, &isample);
        /* The impulse makes a crater, so the centre must be below zero. */
        if (!(isample.height < 0.0f)) return 19;

        if (!(ocean_interaction_fixed_dt(field) > 0.0f)) return 20;
        if (!(ocean_interaction_dt_limit(field) >
              ocean_interaction_fixed_dt(field))) return 21;
        if (ocean_interaction_dropped(field) != 0u) return 22;

        ocean_sample_combined(sim, field, 8.0f, 8.0f, &ex);

        ocean_interaction_destroy(field);
        ocean_interaction_destroy(NULL); /* no-op, like free() */
    }

    ocean_destroy(sim);
    ocean_destroy(NULL); /* must be a no-op, like free() */

    return 0;
}
