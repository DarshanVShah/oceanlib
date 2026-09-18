/* oceanlib - C API.
 *
 * A thin wrapper over the C++ core, so the library can be bound from C, Rust,
 * C#, Python, Swift, or any engine that speaks the C ABI.
 *
 * Design rules this header follows:
 *
 *   - Opaque handle. `ocean_sim` is never defined, so callers cannot depend on
 *     the layout of anything inside, and we stay free to change it.
 *   - No exceptions cross the boundary. Every entry point is `noexcept` on the
 *     C++ side; constructor failures come back as a null handle plus an error
 *     code, because a C caller has no way to catch a std::invalid_argument and
 *     letting one unwind through a C frame is undefined behaviour.
 *   - Explicit struct versioning. `struct_size` lets a newer library detect an
 *     older caller's smaller struct and fill the difference with defaults,
 *     which is what keeps a shared library upgradeable without recompiling
 *     every binding.
 *   - Plain data only. No callbacks that allocate, no ownership transfer of
 *     anything but the handle itself.
 */
#ifndef OCEANLIB_OCEAN_H
#define OCEANLIB_OCEAN_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Handle and status
 * ------------------------------------------------------------------------- */

typedef struct ocean_sim ocean_sim;

typedef enum ocean_status {
    OCEAN_OK                  = 0,
    OCEAN_ERROR_INVALID_ARG   = 1, /* a descriptor field is out of range      */
    OCEAN_ERROR_OUT_OF_MEMORY = 2,
    OCEAN_ERROR_UNKNOWN       = 3
} ocean_status;

/* Human-readable form of a status code. Points to static storage; never null,
 * never needs freeing. */
const char* ocean_status_string(ocean_status status);

/* -------------------------------------------------------------------------
 * Threading hook
 * ------------------------------------------------------------------------- */

/* One unit of work; `index` runs over [0, count). */
typedef void (*ocean_task_fn)(void* ctx, uint32_t index);

/* Must invoke task(ctx, i) for every i in [0, count) and MUST NOT return until
 * all of them have completed. Passing NULL selects the built-in thread pool.
 *
 * The blocking contract is what keeps this ABI small: the library never holds
 * a job handle, so there is no handle lifetime or ownership question to answer
 * across the boundary. */
typedef void (*ocean_parallel_for_fn)(void* user, ocean_task_fn task, void* ctx,
                                      uint32_t count);

/* -------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */

/* DO NOT add fields to this struct. It is embedded BY VALUE inside
 * ocean_desc, ahead of choppiness, foam_threshold, seed and the threading
 * fields - so any field appended here would shift the byte offset of every
 * field that follows it in ocean_desc, silently breaking the struct_size
 * prefix-compatibility guarantee documented below for existing callers.
 * A new spectrum option belongs at the true end of ocean_desc instead,
 * exactly like water_depth - see the comment on that field. */
typedef struct ocean_spectrum_desc {
    float wind_speed;        /* U10, m/s                                     */
    float fetch;             /* metres of open water the wind has crossed    */
    float wind_direction;    /* radians; 0 is +X                             */
    float peak_enhancement;  /* JONSWAP gamma; 1.0 = Pierson-Moskowitz       */
    float swell;             /* 0 = wind sea, 1 = narrow swell               */
    float small_wave_cutoff; /* metres; shorter waves are rolled off         */
    float gravity;           /* m/s^2                                        */
} ocean_spectrum_desc;

typedef struct ocean_desc {
    /* MUST be set to sizeof(ocean_desc) by the caller. This is the version
     * tag: it lets a newer library recognise a struct compiled against an
     * older header and default the fields the caller never knew about. Without
     * it, adding a field to this struct would silently read garbage from
     * beyond the end of an old caller's allocation. */
    size_t struct_size;

    uint32_t size;         /* grid resolution N; power of two in [16, 4096]  */
    float    patch_length; /* world-space tile edge, metres                  */

    ocean_spectrum_desc spectrum;

    float    choppiness;     /* Tessendorf's lambda                          */
    float    foam_threshold; /* Jacobian below which foam appears            */
    uint64_t seed;

    ocean_parallel_for_fn parallel_for;      /* NULL = built-in thread pool  */
    void*                 parallel_for_user;
    uint32_t              thread_count;      /* 0 = one per hardware thread  */

    /* Water depth in metres, for the finite-depth dispersion relation
     * omega^2 = g*k*tanh(k*water_depth). <= 0 (the default) means deep water.
     * See ocean::SpectrumDesc::depth for the full explanation.
     *
     * This lives HERE, at the very end of ocean_desc, rather than inside
     * ocean_spectrum_desc where it would sit more naturally - because
     * ocean_spectrum_desc is embedded by value ahead of choppiness,
     * foam_threshold, seed and the threading fields, inserting a field into
     * it would shift every field after it to a new byte offset, and the
     * struct_size versioning above only works because an old caller's struct
     * is a byte-for-byte PREFIX of the current one. Only ever append new
     * fields here, at the true end - never inside a nested struct that is not
     * itself the last member. */
    float water_depth;
} ocean_desc;

/* Fills `desc` with the same defaults the C++ API uses. Callers should always
 * start here rather than zeroing the struct: a zeroed ocean_desc has
 * gravity = 0 and would be rejected. Sets struct_size for you. */
void ocean_desc_init(ocean_desc* desc);

/* -------------------------------------------------------------------------
 * Outputs
 * ------------------------------------------------------------------------- */

/* Pointers into storage owned by the simulation. Valid until ocean_destroy;
 * rewritten in place by every ocean_update. Do not free them.
 *
 * Both buffers are N*N*4 floats laid out exactly as an RGBA32F texture:
 *   displacement[4*i + 0..2] = dx, dy, dz   (dy is wave height, metres)
 *   displacement[4*i + 3]    = foam         (0..1)
 *   normal      [4*i + 0..2] = nx, ny, nz   (unit length, +Y up)
 *   normal      [4*i + 3]    = jacobian     (< 0 means the surface folded)
 * Row-major, i = z * size + x. */
typedef struct ocean_buffers {
    const float* displacement;
    const float* normal;
    uint32_t     size;
    float        patch_length;
} ocean_buffers;

typedef struct ocean_surface {
    float height;
    float offset_x;
    float offset_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float foam;
} ocean_surface;

/* -------------------------------------------------------------------------
 * Lifetime
 * ------------------------------------------------------------------------- */

/* Allocates everything up front. Returns NULL on failure and, when
 * `out_status` is non-NULL, writes the reason there.
 *
 * Returning the handle rather than a status keeps the common path terse while
 * still letting a caller that cares distinguish a bad descriptor from an
 * allocation failure. */
ocean_sim* ocean_create(const ocean_desc* desc, ocean_status* out_status);

/* Safe to call with NULL. */
void ocean_destroy(ocean_sim* sim);

/* -------------------------------------------------------------------------
 * Simulation
 * ------------------------------------------------------------------------- */

/* Advances to ABSOLUTE simulation time, in seconds - not a delta. The surface
 * is a pure function of (seed, desc, time), so seeking, pausing and replaying
 * all reproduce bit-identically. Performs no heap allocation. */
void ocean_update(ocean_sim* sim, double time);

ocean_buffers ocean_get_buffers(const ocean_sim* sim);

double ocean_get_time(const ocean_sim* sim);

/* -------------------------------------------------------------------------
 * Queries
 * ------------------------------------------------------------------------- */

/* Surface height at a world position. Because choppy displacement moves the
 * surface sideways, this is a small fixed-point solve rather than a lookup. */
float ocean_height_at(const ocean_sim* sim, float world_x, float world_z);

/* Same solve, also returning normal and foam. Cheaper than calling
 * ocean_height_at and then sampling separately. */
ocean_surface ocean_sample_at(const ocean_sim* sim, float world_x, float world_z);

/* -------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */

/* Name of the SIMD kernel set selected at runtime ("scalar", "SSE2", "AVX2",
 * "NEON"). Static storage. Useful in bug reports - though note that all
 * kernels are bit-identical, so this never changes results. */
const char* ocean_simd_level(void);

/* Forces a specific SIMD kernel level, clamped to what this CPU can actually
 * execute (asking for AVX2 on a machine without it falls back rather than
 * faulting). Returns the level actually installed, as ocean_simd_level()
 * would report it afterwards.
 *
 * This is a QA/support hook, not something a normal integration needs: it
 * lets you ask "does this repro on scalar?" without a different machine, and
 * it is how this library's own benchmarks simulate weaker hardware. It
 * affects every ocean_sim created AFTER the call, not ones already running.
 * name is one of "scalar", "sse2", "avx2", "neon" (case-insensitive); any
 * other value is a no-op that returns the current level unchanged. */
const char* ocean_force_simd_level(const char* name);

/* Library version, for a host that loads the shared object dynamically. */
void ocean_version(uint32_t* major, uint32_t* minor, uint32_t* patch);

#if defined(__cplusplus)
}  /* extern "C" */
#endif

#endif /* OCEANLIB_OCEAN_H */
