// oceanlib - public API.
//
// Local interaction: objects disturb the water, the disturbance propagates and
// reflects, and gameplay queries see it. Added on top of the FFT surface,
// never replacing it.
//
// METHOD. Tessendorf's iWave (Game Programming Gems 4, 2004): a local height
// field evolved by a convolution whose Fourier symbol is the linearised
// deep-water surface operator. Both this field and the FFT ocean solve the
// SAME linearised equations, which is what makes adding them legitimate rather
// than a blend - the same argument ADR-020 already makes for cascades, applied
// across a different axis. See ARCHITECTURE.md ADR-021.
//
// WHY THIS IS A SEPARATE OBJECT, and not a flag on Ocean. Ocean::update() takes
// ABSOLUTE time and is a pure function of (seed, desc, time); that purity is
// core promise #3 and it is what makes seeking, pausing and replay work. An
// interaction field is a time-stepped ODE - it is inherently incremental and
// cannot be made a pure function of absolute time. Folding it into Ocean would
// quietly demote that promise for everyone. So the two stay separate, and the
// difference is visible at every call site:
//
//     ocean.update(absolute_time);   // double, absolute
//     field.update(frame_dt);        // float, a delta
//
// WaterSurface below composes them for queries.
#pragma once

#include "ocean/ocean.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ocean {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// How solid cells reflect waves.
//
// This is a real physics distinction, not a style option. For a rigid hull the
// boundary condition on the velocity potential is no normal flow, dphi/dn = 0,
// which for surface elevation is dEta/dn = 0 - a Neumann condition. A crest
// arriving at a wall reflects as a CREST, which is what you see off a pier.
//
// Zeroing the field at solid cells, which is what the published iWave does and
// what almost every implementation copies, is a Dirichlet condition eta = 0.
// That is physically a pressure-release surface, not a wall, and it flips the
// sign of the reflection: a crest comes back as a trough.
enum class Obstruction {
    // dEta/dn = 0. Solid cells are filled from their nearest fluid neighbour
    // before each convolution, so the field has zero gradient into the solid.
    // Correct for hulls, piers and terrain. The default.
    Neumann,

    // eta = 0 at solid cells. Cheaper (one multiply), matches the published
    // method, and available for anyone who wants that look or is comparing
    // against a reference implementation.
    Dirichlet,
};

struct InteractionDesc {
    // Grid resolution N. Power of two in [32, 1024]. Not an FFT requirement -
    // there is no transform here - but it keeps the index arithmetic cheap and
    // matches the rest of the library.
    std::uint32_t size = 256;

    // World-space edge length of the field, in metres. Mirrors
    // OceanDesc::patch_length; cell size is extent / size.
    //
    // Cell size is what actually matters physically: it sets the Nyquist
    // wavelength, and the kernel is only accurate over a band of roughly 2 to
    // 16 cells (see kernel_dispersion_error() below). At the defaults, 64 m
    // over 256 cells = 0.25 m cells, the field carries ripples of about 0.5 m
    // to 4 m - which is exactly the range a rock splash produces.
    float extent = 64.0f;

    // Convolution kernel radius P; the stencil is (2P+1)^2 taps, so cost grows
    // as P^2. Larger P widens the band over which dispersion is accurate.
    //
    // Measured at 0.25 m cells, peak wave-speed error over wavelengths of 2 to
    // 16 cells: P=4 13.3%, P=6 6.6%, P=8 3.2%, P=10 1.5%. P=6 is the knee and
    // the default. Construction throws if the resulting kernel would be
    // unstable.
    std::uint32_t kernel_radius = 6;

    // Linear damping alpha, in 1/s. Ripples decay as exp(-alpha t).
    float damping = 0.20f;

    // Width, in cells, of the absorbing layer at the grid edge.
    //
    // Without it the boundary reflects and the scene fills with waves that
    // bounced off nothing. The layer ramps damping up smoothly (quadratically)
    // from the inner edge rather than switching it on: a step change in
    // damping is an impedance discontinuity and reflects almost as badly as
    // the hard boundary it replaces. It is a sponge, not a true non-reflecting
    // boundary condition, so some energy does come back - measured rather than
    // assumed, see the reflection test.
    std::uint32_t absorb_cells = 16;

    // Fixed simulation timestep, in seconds. 0 selects a safe default derived
    // from the kernel's own realised symbol at construction.
    //
    // Fixed, with an accumulator, so behaviour does not change with frame rate.
    // A value above the stability limit is rejected at construction rather
    // than silently clamped - see stable_dt_limit().
    float fixed_dt = 0.0f;

    // Most substeps one update() will run before giving up and discarding the
    // backlog. This is the guard against the spiral of death: after a long
    // hitch, the field runs slow rather than the frame exploding.
    //
    // Discarding is deterministic - the clamp is a function of the accumulator,
    // which is a function of the dt sequence - so replay is unaffected.
    std::uint32_t max_substeps = 8;

    // Capacity of the per-frame disturbance queue. Fixed at construction so
    // that submitting sources never allocates. Submissions past it are dropped
    // and counted, never silently lost - see dropped_sources().
    std::uint32_t max_sources = 64;

    float gravity = 9.81f;

    // Water depth in metres for the kernel's dispersion relation,
    // omega^2 = g k tanh(k*depth). <= 0 (the default) is deep water, taken as
    // a literal special case exactly as SpectrumDesc::depth is. Set it to the
    // same value as the ocean's if the ocean is shallow, or the two will
    // disagree about how fast waves travel.
    float depth = 0.0f;

    Obstruction obstruction = Obstruction::Neumann;

    // Host scheduler hook, same blocking contract as OceanDesc::parallel_for.
    ParallelForFn parallel_for      = nullptr;
    void*         parallel_for_user = nullptr;
    std::uint32_t thread_count      = 0;
};

// ---------------------------------------------------------------------------
// Disturbances
// ---------------------------------------------------------------------------

enum class SourceKind {
    // A one-off event: a rock landing, an explosion. Applied exactly once, at
    // the first substep that runs after it is submitted. `strength` is in
    // metres.
    Impulse,

    // An ongoing condition: a hull pushing water, a paddle. Applied at every
    // substep, scaled by the timestep, so the result does not depend on frame
    // rate. `strength` is in metres per second. Re-submit it every frame for
    // as long as it exists.
    Continuous,
};

// One disturbance, in WORLD space. The library maps it to grid cells.
struct Disturbance {
    float world_x = 0.0f;
    float world_z = 0.0f;

    // Shape radius sigma, in metres. This is a wavelength selector, not a
    // vague size knob: the impulse profile's transform peaks at k = sqrt(2)/r,
    // so the dominant emitted wavelength is about 4.44 * radius. Choose it so
    // that wavelength lands inside the kernel's accurate band.
    float radius = 0.35f;

    // Depth of the central depression (Impulse, metres) or the rate at which
    // it is driven (Continuous, metres/second). Positive makes a crater.
    float strength = 0.25f;

    // Vertical velocity imparted to the water, m/s, positive up. A falling
    // rock passes a negative value. This is a genuinely different thing from
    // `strength`: strength says the water HAS been pushed down, velocity says
    // it is STILL being pushed down, and a real impact is both.
    float velocity_y = 0.0f;

    // Horizontal velocity of the source itself, m/s. For a Continuous source
    // the position is advanced by this across the substeps of one update, so a
    // moving hull draws a continuous wake instead of a row of separate stamps.
    float velocity_x = 0.0f;
    float velocity_z = 0.0f;

    SourceKind kind = SourceKind::Impulse;
};

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

// The interaction field's contribution at one world position. It is a pure
// height field - no horizontal displacement - so it composes with the FFT
// surface by adding heights and adding SLOPES. See WaterSurface.
struct InteractionSample {
    float height     = 0.0f;  // metres, added to the FFT surface height
    float slope_x    = 0.0f;  // dEta/dx
    float slope_z    = 0.0f;  // dEta/dz
    float velocity_y = 0.0f;  // dEta/dt, m/s
};

// GPU-texture-shaped output, exactly as ADR-004 established for the FFT
// buffers: one RGBA32F image, uploaded with a single memcpy.
//
//   field[4*i + 0] = eta        (metres)
//   field[4*i + 1] = dEta/dx
//   field[4*i + 2] = dEta/dz
//   field[4*i + 3] = dEta/dt    (m/s)
//
// Row-major, i = z * size + x.
//
// SAMPLING NOTE, and it is the exact opposite of the FFT textures: this field
// is NOT periodic. It must be sampled with CLAMP_TO_BORDER and a zero border,
// never REPEAT - outside the grid there is genuinely no disturbance, and a
// REPEAT sampler would tile the ripples across the whole ocean.
struct InteractionBuffers {
    const float*  field  = nullptr;
    std::uint32_t size   = 0;
    float         extent = 0.0f;

    // World position of the low corner of cell (0,0). Changes when the field
    // is recentred, so a renderer must read it every frame rather than caching
    // it once.
    float origin_x = 0.0f;
    float origin_z = 0.0f;
};

// ---------------------------------------------------------------------------
// The field
// ---------------------------------------------------------------------------

class InteractionField {
public:
    // Allocates everything up front. Throws std::invalid_argument on a
    // malformed descriptor, including the case where the requested kernel
    // radius would produce an unstable operator or fixed_dt exceeds the
    // stability limit.
    explicit InteractionField(const InteractionDesc& desc);
    ~InteractionField();

    InteractionField(InteractionField&&) noexcept;
    InteractionField& operator=(InteractionField&&) noexcept;
    InteractionField(const InteractionField&)            = delete;
    InteractionField& operator=(const InteractionField&) = delete;

    // --- sources -----------------------------------------------------------

    // Queue a disturbance. Never allocates; if the queue is full the
    // submission is dropped and counted (see dropped_sources()).
    //
    // Immediate-mode by design: no handles, so there is no lifetime or
    // ownership question to answer across the C boundary - the same reasoning
    // ADR-006 gives for the blocking parallel-for. It also means the sequence
    // of submissions IS the event log, which is what the replay determinism
    // test replays.
    void add(const Disturbance& d) noexcept;

    // Number of submissions dropped for lack of queue capacity since
    // construction. Non-zero means max_sources is too small.
    [[nodiscard]] std::uint64_t dropped_sources() const noexcept;

    // --- stepping ----------------------------------------------------------

    // Advance by `dt` seconds of wall time. Internally this runs a whole
    // number of fixed timesteps from an accumulator, so behaviour does not
    // change with frame rate. Performs no heap allocation.
    void update(float dt) noexcept;

    // Substeps actually run by the last update(), and whether the backlog was
    // discarded because max_substeps was hit.
    [[nodiscard]] std::uint32_t last_substeps() const noexcept;
    [[nodiscard]] bool          last_update_clamped() const noexcept;

    // --- recentring --------------------------------------------------------

    // Move the field so it is centred as close to (world_x, world_z) as a
    // whole number of cells allows.
    //
    // The shift is always an integer number of cells, so it is an exact move:
    // every retained cell keeps its bit-exact value, nothing is resampled or
    // filtered, and there is therefore NO jolt by construction - not "a small
    // jolt we smooth over". Newly exposed cells are set to zero, which is the
    // physically correct value (this field holds only the local disturbance),
    // and they arrive at the grid edge where the absorbing layer has already
    // attenuated everything to near nothing.
    //
    // Both stored time levels shift together. Shifting one and not the other
    // would misalign them and turn the second time derivative into noise.
    void recenter(float world_x, float world_z) noexcept;

    // Whether the last recenter() actually moved the grid, and by how many
    // cells. A host that owns an obstruction mask in world space must refresh
    // it when this is true - newly exposed cells default to fluid.
    [[nodiscard]] bool recentered() const noexcept;
    [[nodiscard]] std::int32_t last_shift_x() const noexcept;
    [[nodiscard]] std::int32_t last_shift_z() const noexcept;

    // --- obstruction -------------------------------------------------------

    // Mark cells solid. `mask` is size*size bytes, row-major, non-zero meaning
    // solid. Copied, so the caller may free it. Not a per-frame call: it
    // rebuilds the nearest-fluid table the Neumann condition needs.
    //
    // Passing nullptr clears the mask.
    void set_obstruction(const std::uint8_t* mask) noexcept;

    // --- queries -----------------------------------------------------------

    // Bilinear sample of this field alone. Returns zeros outside the grid -
    // correct, not a clamp: there really is no disturbance out there.
    [[nodiscard]] InteractionSample sample_at(float world_x,
                                              float world_z) const noexcept;
    [[nodiscard]] float height_at(float world_x, float world_z) const noexcept;

    // --- introspection -----------------------------------------------------

    [[nodiscard]] InteractionBuffers buffers() const noexcept;
    [[nodiscard]] const InteractionDesc& desc() const noexcept;

    // The timestep actually in use, after defaulting and validation.
    [[nodiscard]] float fixed_dt() const noexcept;

    // Largest timestep this configuration is stable at, derived from the
    // kernel's REALISED symbol rather than from theory: truncation ringing can
    // push the realised symbol above the ideal |k|, and the leapfrog bound
    // depends on the largest frequency the state can actually contain.
    //
    // Note that stable is not the same as accurate. At the limit the shortest
    // waves are stable and about 57% wrong in frequency, so the default sits
    // far below it.
    [[nodiscard]] float stable_dt_limit() const noexcept;

    // Peak relative error of the kernel's realised dispersion symbol over the
    // band it claims. The wave-SPEED error is half this, because
    // omega = sqrt(g*S).
    [[nodiscard]] float kernel_dispersion_error() const noexcept;

    // Total energy in the field, in the norm the scheme conserves up to
    // damping. Used by the tests to show decay is monotone and that boundary
    // reflection stays under a stated threshold.
    [[nodiscard]] double energy() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ocean
