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

    ocean_destroy(sim);
    ocean_destroy(NULL); /* must be a no-op, like free() */

    return 0;
}
