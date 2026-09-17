// oceanlib - public API.
//
// Anything declared in include/ocean/ is a compatibility commitment. Internal
// headers live under src/ so that users cannot accidentally depend on them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ocean {

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

// One unit of work. `index` runs over [0, count).
using TaskFn = void (*)(void* ctx, std::uint32_t index);

// A blocking parallel-for. The implementation must invoke
// `task(ctx, i)` for every i in [0, count) and MUST NOT return until all of
// them have completed.
//
// The blocking contract is the whole design. It means the library never holds
// a job handle, never has to define handle lifetime across the C boundary, and
// keeps full control of its own frame ordering - so the result cannot depend
// on how the host scheduler happened to interleave things. It maps onto every
// engine scheduler unchanged: Unreal's ParallelFor, Unity's IJobParallelFor
// followed by Complete(), tbb::parallel_for, #pragma omp parallel for.
using ParallelForFn = void (*)(void* user, TaskFn task, void* ctx,
                               std::uint32_t count);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Directional wave spectrum parameters (JONSWAP + directional spreading,
// following Horvath 2015 on top of Tessendorf 2001).
struct SpectrumDesc {
    // Wind speed at the standard 10 m reference height, in m/s. This is the
    // single most important knob: JONSWAP energy scales steeply with it.
    float wind_speed = 10.0f;

    // Fetch: the distance over open water the wind has blown, in metres.
    // Together with wind_speed it fixes the peak frequency - short fetch gives
    // young, steep, high-frequency seas; long fetch gives long swell.
    float fetch = 100000.0f;

    // Wind heading in radians; 0 points along +X.
    float wind_direction = 0.0f;

    // JONSWAP peak enhancement factor (gamma). 1.0 degenerates to
    // Pierson-Moskowitz (fully developed sea); 3.3 is the North Sea mean.
    float peak_enhancement = 3.3f;

    // 0 = wind sea (broad directional spread), 1 = swell (narrow, nearly
    // unidirectional). Blends the directional spreading function.
    float swell = 0.0f;

    // Waves shorter than this (metres) are suppressed, via Tessendorf's
    // exp(-k^2 * l^2) factor. Kills sub-texel ripples that would only alias.
    float small_wave_cutoff = 0.5f;

    float gravity = 9.81f;

    // Water depth in metres, for the finite-depth dispersion relation
    // omega^2 = g*k*tanh(k*depth). <= 0 (the default) means deep water,
    // omega = sqrt(g*k) - the classic open-ocean assumption, exact whenever
    // depth exceeds roughly half the longest wavelength represented. Set a
    // finite positive value for a coastal or shallow scene: waves slow down
    // and shorten as the bottom shoals, which this changes correctly, but the
    // JONSWAP spectral *shape* itself is not re-derived for shallow water
    // (that would be the TMA spectrum, a further refinement this does not
    // attempt) - only the dispersion relation used to place that shape in
    // wavenumber space.
    float depth = 0.0f;
};

struct OceanDesc {
    // Grid resolution N. Must be a power of two (the FFT is radix-2).
    std::uint32_t size = 256;

    // World-space edge length of one tile, in metres. size/patch_length sets
    // the spatial sampling rate, and hence the shortest wave we can represent.
    float patch_length = 200.0f;

    SpectrumDesc spectrum{};

    // Tessendorf's lambda: how far vertices are pulled horizontally toward
    // wave crests. 0 gives a pure heightfield; ~1 gives sharp, realistic
    // crests; too high makes the surface self-intersect.
    float choppiness = 1.0f;

    // Foam appears where the Jacobian of the horizontal displacement drops
    // below this value, i.e. where the surface is folding onto itself.
    float foam_threshold = 0.5f;

    // Same seed + same time => identical waves, always.
    std::uint64_t seed = 1337;

    // Host scheduler hook. If null, the library uses its own thread pool.
    ParallelForFn parallel_for = nullptr;
    void*         parallel_for_user = nullptr;

    // Only consulted when parallel_for is null. 0 means "one worker per
    // hardware thread".
    std::uint32_t thread_count = 0;
};

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

// All output buffers are aligned to this many bytes: enough for AVX-512 loads
// and, more usefully, exactly one cache line, so a tile of work never shares a
// line with its neighbour and threads cannot false-share.
inline constexpr std::size_t kBufferAlignment = 64;

// Per-frame results, laid out to match GPU RGBA32F textures exactly, so
// uploading the whole ocean is two memcpy-shaped calls with no repacking.
//
//   displacement[4*i + 0..2] = dx, dy, dz   (metres, dy is wave height)
//   displacement[4*i + 3]    = foam         (0..1 coverage)
//   normal      [4*i + 0..2] = nx, ny, nz   (unit length, +Y up)
//   normal      [4*i + 3]    = jacobian     (folding measure; < 0 => inverted)
//
// Row-major, i = z * size + x. Pointers stay valid for the lifetime of the
// Ocean and are rewritten in place by every update().
struct Buffers {
    const float*  displacement = nullptr;
    const float*  normal       = nullptr;
    std::uint32_t size         = 0;     // N; each buffer holds N*N*4 floats
    float         patch_length = 0.0f;  // metres per tile edge
};

// A full surface query at one world position.
struct Surface {
    float height     = 0.0f;  // world Y of the displaced surface
    float offset_x   = 0.0f;  // horizontal displacement applied at that point
    float offset_z   = 0.0f;
    float normal_x   = 0.0f;
    float normal_y   = 1.0f;
    float normal_z   = 0.0f;
    float foam       = 0.0f;
};

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

class Ocean {
public:
    // Allocates everything up front. Throws std::invalid_argument if the
    // descriptor is malformed (non-power-of-two size, non-positive lengths).
    explicit Ocean(const OceanDesc& desc);
    ~Ocean();

    Ocean(Ocean&&) noexcept;
    Ocean& operator=(Ocean&&) noexcept;
    Ocean(const Ocean&)            = delete;
    Ocean& operator=(const Ocean&) = delete;

    // Advances the surface to absolute simulation time `time` (seconds).
    // This is a pure function of (seed, desc, time) - not of previous calls -
    // which is what makes promise #3 hold under seeking, pausing and replay.
    // Performs no heap allocation.
    void update(double time);

    // Valid until the Ocean is destroyed; contents change on every update().
    [[nodiscard]] Buffers buffers() const noexcept;

    // Surface height in world space. Because choppy displacement moves the
    // surface horizontally, the point that *lands* at (x, z) did not start
    // there, so this is a small fixed-point solve, not a table lookup.
    [[nodiscard]] float height_at(float world_x, float world_z) const noexcept;

    // Same solve, but also returns normal and foam - cheaper than calling
    // height_at and then sampling separately.
    [[nodiscard]] Surface sample_at(float world_x, float world_z) const noexcept;

    [[nodiscard]] const OceanDesc& desc() const noexcept;
    [[nodiscard]] double           time() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ocean
