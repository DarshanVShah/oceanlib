// Things that float, and things that are thrown at the water.
//
// No Vulkan in this file. It is the part of the demo that USES the library the
// way a game would: querying the surface and integrating rigid bodies against
// it, with no access to anything the public API does not expose.
#pragma once

#include "props.hpp"
#include "vk_math.hpp"

#include "ocean/interaction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace viewer {

// ---------------------------------------------------------------------------
// Boat
// ---------------------------------------------------------------------------

// A hull floated by sampling the water under four points and integrating the
// buoyancy they produce.
//
// This is the most direct demonstration of promise #2 in the whole project. The
// boat sits on the water THE RENDERER ACTUALLY DREW, because the hull's
// buoyancy and the vertex shader's displacement come from the same field. With
// a naive height lookup the hull would ride visibly wrong at the crests -
// ADR-011 measures that error at 725 mm against 535 mm RMS waves, which on a
// 6 m boat is the difference between floating and hovering.
//
// Three degrees of freedom: heave, pitch and roll. Surge, sway and yaw are
// left out deliberately - they need a hull hydrodynamics model rather than a
// surface query, and inventing one would say nothing about the library.
class Boat {
public:
    // Hull half-extents in metres: length (x), draft (y), beam (z). Taken
    // from the baked model itself (gltf_bake's hull-shell bounding box -
    // keel/planking/frames, not the mast or sail), so these track whatever
    // boat model is actually in assets/ rather than being guessed by hand.
    vkm::Vec3 half{boat_mesh::kHalfExtent[0], boat_mesh::kHalfExtent[1],
                  boat_mesh::kHalfExtent[2]};

    vkm::Vec3 position{0.0f, 0.0f, 0.0f};
    float     pitch = 0.0f;     // about Z, bow up positive
    float     roll  = 0.0f;     // about X
    float     yaw   = 0.0f;

    float heave_vel = 0.0f;
    float pitch_vel = 0.0f;
    float roll_vel  = 0.0f;

    // Tuning. Mass is implicit: what matters is the ratio of buoyant
    // stiffness to inertia, so these are expressed directly as accelerations.
    //
    // `buoyancy` comes from gltf_bake too: it is g/draft, where the draft is
    // the one that puts 55% of the hull's AMIDSHIPS depth under water at rest.
    // Amidships because a double-ender's gunwale sweeps up at both ends, so
    // the bounding box measures the hull where it is deepest and leaves the
    // middle - the part that decides how wet the boat gets - riding far lower
    // than intended. A hand-picked constant here previously left the hull
    // about 87% submerged, visibly swamped, because it had no relationship to
    // the hull's actual shape at all.
    float buoyancy   = boat_mesh::kBuoyancy;   // vertical accel per metre of submersion

    // `drag` is heave's damping in the same spring-damper: for a spring of
    // stiffness `buoyancy`, critical damping is 2*sqrt(buoyancy). This sits
    // ABOVE that (critical ratio ~1.2), deliberately overdamped.
    //
    // The clamp in update() below - depth can't go negative - makes this a
    // one-sided spring: it only pushes up while submerged, so a hull that
    // pops above the local surface free-falls until it lands again rather
    // than being pulled back down. At the ~0.75 damping this used to carry,
    // that combination could still build up amplitude over a few wave
    // cycles into a real bounce-and-splashdown cycle, which is what read as
    // "still bouncing" on an otherwise-calm sea, not just as normal wave
    // following. Tied to `buoyancy` rather than a bare constant so the ratio
    // survives whatever half.y the next boat model bakes to.
    float drag       = 2.4f * std::sqrt(boat_mesh::kBuoyancy);
    float ang_stiff  = 9.0f;    // how hard the hull aligns to the surface
    float ang_damp   = 3.6f;

    // Speed at which a hull section re-entering the water counts as a SLAM
    // rather than a settle, and the metres of depression each m/s of entry
    // speed is worth, capped.
    //
    // The threshold is the honest part: below it the hull is following the
    // water and there is nothing impulsive happening, so emitting anything
    // would be inventing a splash. The gain is a stated fudge - real slamming
    // pressure depends on the deadrise angle of the section that hits, which
    // a four-probe model does not have - and the cap is what keeps a violent
    // entry from injecting a ripple taller than the chop it is riding in.
    static constexpr float kSlamSpeed = 0.60f;   // m/s of entry
    static constexpr float kSlamGain  = 0.10f;   // metres of crater per m/s
    static constexpr float kSlamMax   = 0.30f;   // metres, hard ceiling

    // The water plane under the hull as of the last update(): height at
    // (position.x, position.z), and the two slopes. Output only - the physics
    // uses the probe samples directly, and this is what the renderer needs to
    // draw a waterline on the hull that tilts with the wave.
    float water_y    = 0.0f;
    float water_dhdx = 0.0f;
    float water_dhdz = 0.0f;

    void reset(float x, float z)
    {
        position = {x, 0.0f, z};
        pitch = roll = 0.0f;
        heave_vel = pitch_vel = roll_vel = 0.0f;
    }

    // `water` must already be up to date for this frame.
    void update(const ocean::WaterSurface& water, float dt)
    {
        wake_count_ = 0;
        if (!(dt > 0.0f)) return;
        dt = std::min(dt, 1.0f / 20.0f);   // a long hitch must not launch the boat

        // Four probes at the hull corners. Four is the minimum that can
        // distinguish pitch from roll, and on a hull this size the surface is
        // close to planar across it, so more would buy nothing.
        const float px[4] = { half.x, -half.x,  half.x, -half.x};
        const float pz[4] = { half.z,  half.z, -half.z, -half.z};

        float force = 0.0f;         // net vertical acceleration
        float torque_pitch = 0.0f;  // about Z, from the fore/aft imbalance
        float torque_roll  = 0.0f;  // about X, from the port/starboard one
        float water_vy_sum = 0.0f;

        // Kept per probe so the wake and the waterline plane below can reuse
        // the samples the buoyancy has already paid for, rather than querying
        // the surface a second time for each.
        float probe_x[4], probe_z[4], probe_h[4], probe_depth[4], probe_vy[4];

        for (int i = 0; i < 4; ++i) {
            // Probe position in world space, with the hull's current attitude
            // applied. Small-angle terms are kept because pitch and roll stay
            // well under 20 degrees on any sea this model is valid for.
            const float wx = position.x + px[i] - pz[i] * yaw;
            const float wz = position.z + pz[i] + px[i] * yaw;
            const float wy = position.y - px[i] * pitch + pz[i] * roll;

            const ocean::Surface s = water.sample_at(wx, wz);

            // Submersion measured from the KEEL, capped at the hull's full
            // depth: past that the hull is entirely under and pushing it
            // deeper adds no buoyancy. Without the cap a probe that briefly
            // goes metres under a crest produces an impulse that throws the
            // boat into the air.
            //
            // From the keel rather than from the hull centre, which is what
            // this used to do. The centre is not a waterline, so the spring it
            // produced had a stiffness unrelated to the hull's actual draft -
            // 2.2x too stiff here, giving a 9 m boat a 0.99 s heave period
            // against 1.47 s now. From the keel the stiffness IS g/draft,
            // which is the right answer for a wall-sided hull, and gltf_bake
            // picks the buoyancy constant to match.
            const float depth = std::clamp(s.height - (wy - half.y),
                                           0.0f, 2.0f * half.y);

            const float lift = buoyancy * depth * 0.25f;   // four probes share it
            force        += lift;
            torque_pitch += lift * (-px[i]);
            torque_roll  += lift * ( pz[i]);
            water_vy_sum += s.velocity_y;

            probe_x[i]     = wx;
            probe_z[i]     = wz;
            probe_h[i]     = s.height;
            probe_depth[i] = depth;
            probe_vy[i]    = s.velocity_y;
        }

        // Gravity, and drag measured RELATIVE TO THE WATER rather than to the
        // world. That distinction is what makes the boat ride a wave instead of
        // being dragged through it, and it is only possible because the library
        // reports orbital velocity.
        const float water_vy = water_vy_sum * 0.25f;
        force -= 9.81f;
        force -= drag * (heave_vel - water_vy);

        heave_vel += force * dt;
        position.y += heave_vel * dt;

        // Attitude: a spring toward level relative to the buoyancy imbalance,
        // damped. Normalising by the lever arm keeps the response independent
        // of hull size.
        pitch_vel += (torque_pitch * ang_stiff / half.x - ang_damp * pitch_vel) * dt;
        roll_vel  += (torque_roll  * ang_stiff / half.z - ang_damp * roll_vel)  * dt;
        pitch += pitch_vel * dt;
        roll  += roll_vel  * dt;

        pitch = std::clamp(pitch, -0.7f, 0.7f);
        roll  = std::clamp(roll,  -0.7f, 0.7f);

        // A hull that has somehow left the water entirely is snapped back
        // rather than allowed to sail off; this only fires if the simulation is
        // driven far outside what it models.
        const float surf = water.height_at(position.x, position.z);
        if (position.y > surf + 12.0f || position.y < surf - 12.0f) {
            position.y = surf;
            heave_vel  = 0.0f;
        }

        // --- the water plane under the hull, for rendering -----------------
        //
        // Fitted from the same four samples the buoyancy used. The probes sit
        // symmetrically about the hull centre, so their mean IS the height at
        // that centre for a linear fit, and the two differences are the
        // slopes. (The yaw shear above skews the layout very slightly; at the
        // yaw angles this model allows, that is far below anything a shading
        // band could show.) The renderer wets the hull below this plane; the
        // physics never reads it, but it comes free with the probes.
        water_y    = 0.25f * (probe_h[0] + probe_h[1] + probe_h[2] + probe_h[3]);
        water_dhdx = ((probe_h[0] + probe_h[2]) - (probe_h[1] + probe_h[3])) /
                     (4.0f * half.x);
        water_dhdz = ((probe_h[0] + probe_h[1]) - (probe_h[2] + probe_h[3])) /
                     (4.0f * half.z);

        // --- the hull's own disturbance ------------------------------------
        //
        // A floating hull is not a passive reader of the water: it pushes back,
        // and the library has both of the source kinds that takes. Which one
        // applies is decided by measurement, not by taste.
        //
        // The shared quantity is CLOSING SPEED: how fast a part of the hull is
        // moving down relative to the water under it. That is the same fact
        // that lets drag be measured against the water rather than the world,
        // and both velocities are already in hand, so it costs no extra
        // sampling. A hull merely riding a swell moves WITH the water and its
        // closing speed is near zero; a hull falling off a crest into the back
        // of the next wave has a large one.
        //
        // ONE source, at the hull centre - not one per probe. This is a stated
        // approximation and it is load-bearing, so here is the measurement
        // behind it. Four corner sources put the emitted ripple (about 2.6 m,
        // see the radius below) at almost exactly the hull's 2.15 m beam, so
        // the ripple the boat had just made arrived at its two beam probes
        // with OPPOSITE sign. That is a roll torque the boat applies to
        // itself, and it is regenerative: over four minutes it drove roll into
        // its +-40 degree clamp in half of the 30 s windows measured, and
        // widened the hull's travel from 28..108% submerged to -5..136%. Four
        // point probes alias their own wake.
        //
        // A single centred source cannot do that. Its ripple is radially
        // symmetric about the hull centre, so it reaches all four probes
        // alike: it contributes a small symmetric heave - which is a real
        // added-mass-like effect, and self-limiting - and essentially no pitch
        // or roll torque. The boat still disturbs the water, and still reads
        // everything ELSE in the field, so a rock dropped alongside rocks it
        // exactly as before.
        //
        // What is given up is telling a bow slam from a stern one, which at
        // this field's 0.5-4 m accurate band against a 9 m hull it could not
        // represent faithfully anyway.
        float closing_sum = 0.0f, closing_max = 0.0f, depth_sum = 0.0f;
        int   submerged_count = 0;
        bool  slam = false;

        for (int i = 0; i < 4; ++i) {
            const bool  submerged = probe_depth[i] > 0.0f;
            // The probe's own vertical velocity, hull-rigid: heave, plus what
            // pitch and roll contribute at that lever arm.
            const float hull_vy   = heave_vel - px[i] * pitch_vel + pz[i] * roll_vel;
            const float closing   = probe_vy[i] - hull_vy;   // > 0 = driving down

            // A slam is a TRANSITION - a section that was clear of the water
            // re-entering it fast - so it cannot be read off one frame's
            // state. This is the only history the hull keeps.
            if (submerged && !probe_was_wet_[i] && closing > kSlamSpeed) slam = true;
            probe_was_wet_[i] = submerged;

            if (!submerged || closing <= 0.0f) continue;
            closing_sum += closing;
            closing_max  = std::max(closing_max, closing);
            depth_sum   += probe_depth[i];
            ++submerged_count;
        }

        if (submerged_count > 0) {
            const float n       = static_cast<float>(submerged_count);
            const float closing = closing_sum / n;

            ocean::Disturbance& d = wake_[wake_count_++];
            d.world_x = position.x;
            d.world_z = position.z;
            // The emitted wavelength is about 4.44 * radius, so half the beam
            // puts it near 2.6 m - inside the 0.5-4 m band the viewer's 0.25 m
            // field resolves accurately. A hull-sized radius would overshoot.
            d.radius     = 0.55f * half.z;
            d.velocity_x = 0.0f;
            d.velocity_z = 0.0f;

            if (slam) {
                // Impulse, because a hull section re-entering the water IS an
                // impulsive event - the same kind of event as the thrown rock,
                // and given the same treatment: a crater sized by the entry
                // speed, plus the velocity still driving it down.
                d.strength = std::min(kSlamGain * closing_max, kSlamMax);
                // Clamped to the range the rock's impact uses, for the same
                // reason: a real velocity, but bounded against the field's
                // stability limits.
                d.velocity_y = std::clamp(-closing_max, -8.0f, -0.5f);
                d.kind       = ocean::SourceKind::Impulse;
            } else {
                // Continuous, which is what that kind documents itself for -
                // "an ongoing condition: a hull pushing water". Submitted at
                // the closing speed itself, with no gain, because the source
                // takes metres per second and that is what this is.
                //
                // Be clear about what it produces: almost nothing, and
                // correctly so. Measured against this viewer's own field, a
                // 0.05 m/s drive settles at a peak of 0.24 mm - a slow push
                // does not pile water up, it just makes it flow, so the wave
                // equation radiates the drive away about as fast as it
                // arrives. That is why a moored boat bobbing on a swell leaves
                // the water around it visibly flat. The term is kept because
                // it is the physically right one and it costs one queue push;
                // it is NOT scaled up to make it show, because the ~60x gain
                // that would take is not a hull pushing water any more.
                //
                // Faded in with submersion, so a hull skimming the surface
                // does not switch its source on and off between frames.
                const float wet = std::min(depth_sum / (n * 0.35f * half.y), 1.0f);
                d.strength   = closing * wet;
                d.velocity_y = 0.0f;
                d.kind       = ocean::SourceKind::Continuous;
            }
        }
    }

    // This frame's hull disturbances, ready to hand to an InteractionField.
    // Empty until update() has run, and refilled by every call to it.
    [[nodiscard]] std::span<const ocean::Disturbance> wake() const
    {
        return {wake_.data(), wake_count_};
    }

    // The baked Viking boat model: hull/mast/rigging as one mesh, the sail as
    // another so it can carry its own (paler, fabric-like) colour without a
    // texture. Both share the same body transform, and both were already
    // centred and scaled to metres by gltf_bake, so no per-instance scale or
    // offset is needed - unlike the old three-box hull, which needed a
    // separate offset per part because it had no single sculpted shape to
    // draw instead.
    void append(std::vector<PropInstance>& out) const
    {
        const vkm::Mat4 body =
            vkm::translate(position) * vkm::rotate_y(yaw) *
            vkm::rotate_z(pitch) * vkm::rotate_x(roll);

        auto add = [&](PropMesh mesh, float r, float g, float b, float translucency,
                       float band) {
            PropInstance p;
            p.model    = body;
            p.scale[0] = p.scale[1] = p.scale[2] = 1.0f;
            p.color[0] = r; p.color[1] = g; p.color[2] = b;
            p.color[3] = translucency;
            p.surf[0]  = water_y;
            p.surf[1]  = water_dhdx;
            p.surf[2]  = water_dhdz;
            p.surf[3]  = band;
            p.mesh     = mesh;
            out.push_back(p);
        };

        // Oak is opaque and gets a wet band: planking below the waterline is
        // soaked and dark, and that hard line is most of what tells the eye the
        // hull is IN the water rather than resting on a picture of it.
        //
        // Canvas is the other way round. It is thin enough to pass light, so
        // it gets translucency and no band - a sail that goes flat grey the
        // moment the sun is behind it reads as a painted board, and a sail lit
        // through from behind is the single most recognisable thing about one.
        // The albedo is lower than it looks: 0.42 reads as pale sand once it is
        // lit at full sun and sRGB-encoded, not as oak. Weathered oak sits
        // nearer 0.2, and at that value the planking keeps its colour in
        // sunlight instead of washing out to cream.
        //
        // The band is 0.30 m, about a fifth of the hull's depth - roughly how
        // far up the planking this boat's own chop actually washes.
        add(PropMesh::BoatHull, 0.26f, 0.17f, 0.10f, 0.00f, 0.30f);  // weathered oak
        add(PropMesh::BoatSail, 0.86f, 0.80f, 0.68f, 0.55f, 0.00f);  // canvas
    }

private:
    // At most one source per frame, and storage for it that never allocates -
    // the same reasoning InteractionField::add() gives for its own queue. It
    // stays a span rather than an optional so the call site is a plain range,
    // matching how the rock splashes are submitted next to it.
    std::array<ocean::Disturbance, 1> wake_{};
    std::size_t                       wake_count_ = 0;

    // Whether each probe was under water last frame. A slam is a TRANSITION,
    // so it cannot be read off a single frame's state; this is the one bit of
    // history the hull needs. Starting true means a boat spawned already
    // floating does not slam on its first frame.
    bool probe_was_wet_[4] = {true, true, true, true};
};

// ---------------------------------------------------------------------------
// Thrown rock
// ---------------------------------------------------------------------------

// A rock in flight. It is a real ballistic projectile, not an effect: it leaves
// the camera, arcs under gravity, and the splash happens WHEN AND WHERE IT
// LANDS - tested against the displaced surface each step, so it lands on a
// crest sooner than it would on a trough.
struct Rock {
    vkm::Vec3 pos{};
    vkm::Vec3 vel{};
    vkm::Vec3 spin{};
    float     angle = 0.0f;
    float     radius = 0.22f;
    bool      alive = true;
};

class RockThrower {
public:
    // Launch a rock from `from` so that it lands on `target`.
    //
    // Flight time is chosen from the distance rather than the speed, which is
    // what makes a near throw look like a lob and a far one look like a hurl.
    // Given a time, the launch velocity is exact: v = (d - 0.5*g*T^2) / T.
    void throw_at(vkm::Vec3 from, vkm::Vec3 target, float radius)
    {
        const vkm::Vec3 d = target - from;
        const float dist = std::sqrt(d.x * d.x + d.z * d.z);
        const float T = std::clamp(0.35f + dist * 0.035f, 0.45f, 2.6f);

        Rock r;
        r.pos    = from;
        r.radius = radius;
        r.vel    = {d.x / T, d.y / T + 0.5f * 9.81f * T, d.z / T};
        // A deterministic tumble, so a screenshot of a rock in flight is
        // reproducible like everything else in this viewer.
        const float h = static_cast<float>(rocks_.size() * 2654435761u % 1000u) / 1000.0f;
        r.spin = {6.0f + 4.0f * h, 3.0f - 5.0f * h, 7.5f * h};
        rocks_.push_back(r);
    }

    // Advances every rock. Returns splashes to hand to the interaction field:
    // position, and the speed the rock was travelling when it hit.
    struct Splash { float x, z, impact_speed; };

    std::vector<Splash> update(const ocean::WaterSurface& water, float dt)
    {
        std::vector<Splash> splashes;
        if (!(dt > 0.0f)) return splashes;
        dt = std::min(dt, 1.0f / 20.0f);

        for (Rock& r : rocks_) {
            if (!r.alive) continue;
            r.vel.y -= 9.81f * dt;
            r.pos   += r.vel * dt;
            r.angle += dt;

            // The surface it is falling toward is the DISPLACED one, including
            // any wake already there - so a rock thrown into ripples from an
            // earlier rock lands on those ripples.
            const float surf = water.height_at(r.pos.x, r.pos.z);
            if (r.pos.y - r.radius <= surf) {
                splashes.push_back({r.pos.x, r.pos.z, -r.vel.y});
                r.alive = false;
            } else if (r.pos.y < surf - 200.0f) {
                r.alive = false;   // thrown at the sky and never came back
            }
        }
        rocks_.erase(std::remove_if(rocks_.begin(), rocks_.end(),
                                    [](const Rock& r) { return !r.alive; }),
                     rocks_.end());
        return splashes;
    }

    void append(std::vector<PropInstance>& out) const
    {
        for (const Rock& r : rocks_) {
            PropInstance p;
            p.model = vkm::translate(r.pos) *
                      vkm::rotate_y(r.angle * r.spin.y) *
                      vkm::rotate_x(r.angle * r.spin.x) *
                      vkm::rotate_z(r.angle * r.spin.z);
            p.scale[0] = p.scale[1] = p.scale[2] = r.radius;
            p.color[0] = 0.30f; p.color[1] = 0.29f; p.color[2] = 0.27f;
            p.color[3] = 0.0f;   // stone passes no light
            // surf stays zeroed, and a band softness of 0 disables the wet
            // band outright: a rock in flight has no waterline, and one that
            // has hit the water is already gone.
            p.mesh = PropMesh::Rock;
            out.push_back(p);
        }
    }

    [[nodiscard]] std::size_t in_flight() const { return rocks_.size(); }

private:
    std::vector<Rock> rocks_;
};

}  // namespace viewer
