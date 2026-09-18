// oceanlib - public API.
//
// Persistent, drifting foam.
//
// The FFT surface already reports foam, but only as an INSTANTANEOUS Jacobian
// threshold: it appears the moment a crest folds and vanishes the moment it
// unfolds, pinned to the wave rather than to the water. Real foam is a passive
// tracer - it is created by breaking, then it rides the surface current and
// dissolves over seconds. The difference is the single cheapest thing that
// makes an ocean read as alive rather than as a moving surface.
//
// This field adds the two missing behaviours:
//
//   ADVECTION - foam is carried by the horizontal orbital velocity, so it
//               drifts off the crest that made it and streaks along the flow.
//   PERSISTENCE - foam decays exponentially instead of switching off, so a
//               patch of whitewater outlives the wave that produced it.
//
// WHY THIS IS A SEPARATE OBJECT, for exactly the reason ADR-021 gives for the
// interaction field: Ocean::update() takes ABSOLUTE time and is a pure
// function of (seed, desc, time), which is core promise #3 and what makes
// seeking and replay work. Advected foam is a time-stepped state that depends
// on its own history and cannot be. Folding it into Ocean would demote that
// promise for everyone, including integrations that never asked for foam.
//
//     ocean.update(absolute_time);   // double, absolute, pure
//     foam.update(frame_dt);         // float, a delta, stateful
#pragma once

#include "ocean/ocean.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ocean {

struct FoamDesc {
    // Exponential decay rate, 1/s. Foam falls as exp(-decay * t), so the
    // half-life is ln(2)/decay - about 2 s at the default.
    float decay = 0.35f;

    // How fast the instantaneous Jacobian foam injects into the persistent
    // field, per second. Steady-state coverage under a sustained source is
    // source_gain/decay, so anything above `decay` saturates a genuinely
    // breaking crest to full white quickly while leaving a briefly-folding one
    // faint.
    float source_gain = 4.0f;

    // Multiplier on the advecting velocity. 1 is the physical orbital
    // velocity; raising it exaggerates the streaking, which reads well on a
    // stylised ocean and is wrong on a realistic one.
    float advect_scale = 1.0f;

    // Constant drift added to the orbital velocity, m/s, for wind pushing
    // surface foam downwind. Purely cosmetic: the spectrum's own wind does not
    // produce a mean surface current in linear theory.
    float wind_drift_x = 0.0f;
    float wind_drift_z = 0.0f;

    // Fixed simulation timestep, seconds, with an accumulator - so the look
    // does not change with frame rate. Semi-Lagrangian advection is
    // unconditionally stable, so this is an accuracy choice and not a
    // stability one.
    float fixed_dt = 1.0f / 60.0f;

    // Spiral-of-death guard, as InteractionField has. The backlog is
    // discarded rather than carried.
    std::uint32_t max_substeps = 4;

    ParallelForFn parallel_for      = nullptr;
    void*         parallel_for_user = nullptr;
    std::uint32_t thread_count      = 0;
};

// Persistent foam over one Ocean patch.
//
// Holds a NON-OWNING reference to the ocean, which must outlive it, and reads
// whatever state that ocean is in when update() is called - so the host must
// advance the ocean first:
//
//     ocean.update(t);
//     foam.update(dt);
//
// The ocean must have been created with OceanDesc::compute_velocity set: there
// is nothing to advect foam WITH otherwise, and silently falling back to
// non-advected foam would defeat the entire point of this class.
class FoamField {
public:
    // Throws std::invalid_argument if the ocean has no velocity field or the
    // descriptor is malformed. Allocates everything up front.
    FoamField(const Ocean& ocean, const FoamDesc& desc);
    ~FoamField();

    FoamField(FoamField&&) noexcept;
    FoamField& operator=(FoamField&&) noexcept;
    FoamField(const FoamField&)            = delete;
    FoamField& operator=(const FoamField&) = delete;

    // Advance by `dt` seconds of wall time. Performs no heap allocation.
    void update(float dt) noexcept;

    // Reset to no foam anywhere. Useful after a seek, since this field - unlike
    // the ocean itself - cannot reconstruct a past state from the clock.
    void clear() noexcept;

    // N*N single-channel coverage in [0,1], row-major, i = z * size + x.
    //
    // One channel rather than an RGBA texel because there is nothing else to
    // put beside it; a renderer uploads this as R32_SFLOAT. Indexed by the
    // ocean's UNDISPLACED parameter position, exactly like the displacement
    // and normal buffers, and periodic with the same period - so it tiles
    // with them and a REPEAT sampler is correct.
    [[nodiscard]] const float*  data() const noexcept;
    [[nodiscard]] std::uint32_t size() const noexcept;

    // World-space query, matching Ocean::sample_at's inversion so that the
    // foam reported at a point belongs to the same piece of water as the
    // height reported there.
    [[nodiscard]] float foam_at(float world_x, float world_z) const noexcept;

    [[nodiscard]] const FoamDesc& desc() const noexcept;

    // Retune between updates. Never reallocates and never touches the stored
    // foam, so a weather transition can ramp wind drift and decay smoothly
    // without a discontinuity in what is already on the water.
    //
    // The threading and timestep fields are IGNORED: changing them would mean
    // rebuilding the pool or resetting the accumulator, which is a
    // construction-time decision here for exactly the reason OceanDesc makes
    // choppiness one (ADR-015's note on the viewer).
    void set_desc(const FoamDesc& desc);
    [[nodiscard]] std::uint32_t   last_substeps() const noexcept;

    // Mean coverage over the patch, in [0,1]. Cheap enough to call per frame
    // and the natural thing for a test to assert decay against.
    [[nodiscard]] double coverage() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ocean
