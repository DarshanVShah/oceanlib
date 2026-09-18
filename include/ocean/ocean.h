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

    /* Compute the 3-D orbital velocity field as well. Non-zero to enable.
     *
     * Appended at the TRUE END of the struct, which is the only placement that
     * keeps struct_size meaningful - see the comment on water_depth, and
     * ADR-019 on why a field can hide in tail padding without changing
     * sizeof(). An old caller who used ocean_desc_init() has a zero here,
     * which is the correct default.
     *
     * The velocity itself is read through ocean_sample_ex() and
     * ocean_buffers_ex(); ocean_surface and ocean_sample_at() are deliberately
     * frozen, because ocean_surface is returned BY VALUE and growing it would
     * break every binary already compiled against it. */
    int32_t compute_velocity;
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
 * ocean_height_at and then sampling separately.
 *
 * FROZEN. ocean_surface is returned by value, so appending a field to it would
 * change the ABI of this function and corrupt the return slot of every binary
 * already compiled against the old size. New outputs go in ocean_surface_ex
 * below instead. This is the one case where struct_size cannot help: the
 * caller does not pass this struct in, so there is nothing to version. */
ocean_surface ocean_sample_at(const ocean_sim* sim, float world_x, float world_z);

/* Everything ocean_surface carries, plus orbital velocity.
 *
 * A separate struct rather than an extension, for the reason above. Velocity
 * is zero unless ocean_desc::compute_velocity was set. */
typedef struct ocean_surface_ex {
    float height;
    float offset_x;
    float offset_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float foam;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} ocean_surface_ex;

void ocean_sample_ex(const ocean_sim* sim, float world_x, float world_z,
                     ocean_surface_ex* out);

/* The orbital velocity buffer, or NULL when compute_velocity was not set.
 * N*N*4 floats: vx, vy, vz, 0 (reserved). Same lifetime rules as
 * ocean_get_buffers. */
const float* ocean_get_velocity(const ocean_sim* sim);

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

/* -------------------------------------------------------------------------
 * Local interaction (ADR-021)
 *
 * A local height field, evolved by the iWave convolution and added on top of
 * the FFT surface: objects disturb the water, the disturbance propagates and
 * reflects, and queries see it.
 *
 * Deliberately a SEPARATE handle with a delta-time update, mirroring the C++
 * API. ocean_update() takes ABSOLUTE time and is a pure function of
 * (seed, desc, time); an interaction field is a time-stepped ODE and cannot
 * be. Keeping them apart means that purity is not quietly demoted for callers
 * who never asked for interaction.
 * ------------------------------------------------------------------------- */

typedef struct ocean_interaction ocean_interaction;

/* How solid cells reflect. A rigid hull is no-normal-flow (Neumann) and
 * reflects a crest as a CREST; zeroing the field (Dirichlet) is physically a
 * pressure-release surface and flips the sign. */
typedef enum ocean_obstruction {
    OCEAN_OBSTRUCTION_NEUMANN   = 0,  /* dEta/dn = 0; correct for hulls   */
    OCEAN_OBSTRUCTION_DIRICHLET = 1   /* eta = 0; the published iWave     */
} ocean_obstruction;

typedef enum ocean_source_kind {
    OCEAN_SOURCE_IMPULSE    = 0,  /* one-off; strength in metres          */
    OCEAN_SOURCE_CONTINUOUS = 1   /* ongoing; strength in metres/second   */
} ocean_source_kind;

typedef struct ocean_interaction_desc {
    /* MUST be set to sizeof(ocean_interaction_desc). Same version tag, same
     * rules, as ocean_desc: append only, at the true end. */
    size_t struct_size;

    uint32_t size;          /* grid resolution N, power of two in [32, 1024] */
    float    extent;        /* world edge length, metres                     */
    uint32_t kernel_radius; /* P; stencil is (2P+1)^2 taps                   */
    float    damping;       /* alpha, 1/s                                    */
    uint32_t absorb_cells;  /* width of the absorbing layer, in cells        */
    float    fixed_dt;      /* 0 = derive a safe default                     */
    uint32_t max_substeps;  /* spiral-of-death clamp                         */
    uint32_t max_sources;   /* queue capacity; submitting never allocates    */
    float    gravity;
    float    water_depth;   /* <= 0 = deep water                             */
    int32_t  obstruction;   /* ocean_obstruction                             */

    ocean_parallel_for_fn parallel_for;      /* NULL = built-in thread pool  */
    void*                 parallel_for_user;
    uint32_t              thread_count;
} ocean_interaction_desc;

/* Fills with the same defaults the C++ API uses, and sets struct_size.
 * Callers should always start here. */
void ocean_interaction_desc_init(ocean_interaction_desc* desc);

ocean_interaction* ocean_interaction_create(const ocean_interaction_desc* desc,
                                            ocean_status* out_status);
void ocean_interaction_destroy(ocean_interaction* field);

/* One disturbance, in WORLD space.
 *
 * `radius` is a wavelength selector, not a vague size: the profile's transform
 * peaks at sqrt(2)/radius, so the dominant emitted wavelength is about
 * 4.44*radius. `strength` is the depth of the central depression (metres for
 * an impulse, metres/second for a continuous source). `velocity_y` is the
 * vertical speed imparted to the water, negative for a falling rock -
 * genuinely different from strength, which says the water HAS been pushed
 * down rather than that it is STILL being pushed. */
typedef struct ocean_disturbance {
    float world_x;
    float world_z;
    float radius;
    float strength;
    float velocity_y;
    float velocity_x;   /* horizontal motion of the source itself, m/s */
    float velocity_z;
    int32_t kind;       /* ocean_source_kind */
} ocean_disturbance;

/* Queue a disturbance. Never allocates; dropped and counted if the queue is
 * full. Immediate-mode by design - no handles, so there is no lifetime or
 * ownership question to answer across this boundary. */
void     ocean_interaction_add(ocean_interaction* field,
                               const ocean_disturbance* d);
uint64_t ocean_interaction_dropped(const ocean_interaction* field);

/* Advance by dt seconds of wall time. Runs a whole number of fixed internal
 * timesteps from an accumulator, so behaviour does not change with frame rate.
 * Performs no heap allocation. */
void ocean_interaction_update(ocean_interaction* field, float dt);

/* Move the field so it is centred as close to (x, z) as a whole number of
 * cells allows. The shift is exact - nothing is resampled - so there is no
 * jolt. Returns non-zero if the grid actually moved, which is when a host
 * owning a world-space obstruction mask must refresh it. */
int32_t ocean_interaction_recenter(ocean_interaction* field, float world_x,
                                   float world_z);

/* Mark cells solid. `mask` is size*size bytes, row-major, non-zero = solid.
 * Copied. NULL clears. Not a per-frame call. */
void ocean_interaction_set_obstruction(ocean_interaction* field,
                                       const uint8_t* mask);

typedef struct ocean_interaction_buffers {
    /* N*N*4 floats: eta, dEta/dx, dEta/dz, dEta/dt.
     *
     * NOT periodic. Sample with CLAMP_TO_BORDER and a zero border, never
     * REPEAT - the exact opposite of the FFT buffers, and a REPEAT sampler
     * here tiles one splash across the whole ocean. */
    const float* field;
    uint32_t     size;
    float        extent;
    float        origin_x;   /* world position of cell (0,0); changes on a  */
    float        origin_z;   /* recentre, so read it every frame            */
} ocean_interaction_buffers;

ocean_interaction_buffers ocean_interaction_get_buffers(
    const ocean_interaction* field);

typedef struct ocean_interaction_sample {
    float height;
    float slope_x;
    float slope_z;
    float velocity_y;
} ocean_interaction_sample;

void  ocean_interaction_sample_at(const ocean_interaction* field, float world_x,
                                  float world_z,
                                  ocean_interaction_sample* out);
float ocean_interaction_height_at(const ocean_interaction* field, float world_x,
                                  float world_z);

/* The timestep actually in use, and the largest one this configuration is
 * stable at (derived from the kernel's realised symbol, not from theory).
 * Note that stable is not accurate: at the limit the shortest waves are
 * stable and about 57% wrong in frequency. */
float ocean_interaction_fixed_dt(const ocean_interaction* field);
float ocean_interaction_dt_limit(const ocean_interaction* field);

/* Peak relative error of the kernel's dispersion symbol over the band it
 * claims. The wave-SPEED error is half this. */
float ocean_interaction_dispersion_error(const ocean_interaction* field);

/* Combined query: the FFT surface plus the interaction field, with normals
 * composed through slopes (exact - heights adding in world space means
 * world-space slopes add). */
void ocean_sample_combined(const ocean_sim* sim, const ocean_interaction* field,
                           float world_x, float world_z,
                           ocean_surface_ex* out);

#if defined(__cplusplus)
}  /* extern "C" */
#endif

#endif /* OCEANLIB_OCEAN_H */
