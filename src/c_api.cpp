// oceanlib - C API implementation.
//
// Every entry point here is a firewall: no exception may cross it, because
// unwinding through a C frame is undefined behaviour and a Rust or C# caller
// has no way to catch anything anyway.

#include "ocean/ocean.h"

#include "ocean/interaction.hpp"

#include "core/cpu_features.hpp"
#include "ocean/ocean.hpp"

#include <cctype>
#include <cstring>
#include <new>
#include <string>
#include <stdexcept>

namespace {

// The handle is just the C++ object. No wrapper struct, no extra indirection:
// `ocean_sim` is declared but never defined in the header, so the cast is the
// entire binding. This is the payoff of ADR-005's pimpl - the C++ class is
// already one pointer, so the C API costs nothing on top of it.
ocean::Ocean* to_cpp(ocean_sim* h) noexcept
{
    return reinterpret_cast<ocean::Ocean*>(h);
}
const ocean::Ocean* to_cpp(const ocean_sim* h) noexcept
{
    return reinterpret_cast<const ocean::Ocean*>(h);
}
ocean_sim* to_c(ocean::Ocean* p) noexcept
{
    return reinterpret_cast<ocean_sim*>(p);
}

// Copies a caller's descriptor, honouring struct_size.
//
// A caller compiled against an older header passes a smaller struct. Reading
// past `struct_size` would read memory it never allocated, so we copy only
// what it actually provided and leave our defaults in place for the rest.
// This is what makes the shared library upgradeable without recompiling every
// binding against the new header.
bool copy_desc(const ocean_desc* src, ocean_desc& dst) noexcept
{
    if (src == nullptr) return false;
    if (src->struct_size == 0 || src->struct_size > sizeof(ocean_desc)) {
        // Larger than we know about means the caller was built against a
        // NEWER header than this library. We cannot guess what the extra
        // fields mean, and silently ignoring them would be worse than saying
        // no, so this is an error rather than a truncation.
        return false;
    }
    ocean_desc_init(&dst);
    std::memcpy(&dst, src, src->struct_size);
    dst.struct_size = sizeof(ocean_desc);
    return true;
}

ocean::OceanDesc translate(const ocean_desc& c)
{
    ocean::OceanDesc d;
    d.size         = c.size;
    d.patch_length = c.patch_length;

    d.spectrum.wind_speed        = c.spectrum.wind_speed;
    d.spectrum.fetch             = c.spectrum.fetch;
    d.spectrum.wind_direction    = c.spectrum.wind_direction;
    d.spectrum.peak_enhancement  = c.spectrum.peak_enhancement;
    d.spectrum.swell             = c.spectrum.swell;
    d.spectrum.small_wave_cutoff = c.spectrum.small_wave_cutoff;
    d.spectrum.gravity           = c.spectrum.gravity;

    d.choppiness     = c.choppiness;
    d.foam_threshold = c.foam_threshold;
    d.seed           = c.seed;

    // water_depth lives at the very end of the flat C struct (see ocean.h for
    // why); it maps onto the nested C++ field here. This translate layer is
    // exactly what lets the two layouts differ.
    d.spectrum.depth = c.water_depth;

    // The callback types are layout-identical by construction: both are plain
    // function pointers taking (void*, uint32_t) and (void*, fn, void*,
    // uint32_t). Keeping them identical rather than wrapping means a host
    // scheduler costs no trampoline on the hot path.
    d.parallel_for =
        reinterpret_cast<ocean::ParallelForFn>(c.parallel_for);
    d.parallel_for_user = c.parallel_for_user;
    d.thread_count      = c.thread_count;
    d.compute_velocity  = (c.compute_velocity != 0);
    return d;
}

}  // namespace

extern "C" {

const char* ocean_status_string(ocean_status status)
{
    switch (status) {
        case OCEAN_OK:                  return "ok";
        case OCEAN_ERROR_INVALID_ARG:   return "invalid argument";
        case OCEAN_ERROR_OUT_OF_MEMORY: return "out of memory";
        case OCEAN_ERROR_UNKNOWN:       return "unknown error";
    }
    return "unrecognised status";
}

void ocean_desc_init(ocean_desc* desc)
{
    if (desc == nullptr) return;

    // Defaults are taken from the C++ descriptor rather than duplicated, so
    // the two APIs cannot drift apart.
    const ocean::OceanDesc d{};

    std::memset(desc, 0, sizeof(*desc));
    desc->struct_size  = sizeof(ocean_desc);
    desc->size         = d.size;
    desc->patch_length = d.patch_length;

    desc->spectrum.wind_speed        = d.spectrum.wind_speed;
    desc->spectrum.fetch             = d.spectrum.fetch;
    desc->spectrum.wind_direction    = d.spectrum.wind_direction;
    desc->spectrum.peak_enhancement  = d.spectrum.peak_enhancement;
    desc->spectrum.swell             = d.spectrum.swell;
    desc->spectrum.small_wave_cutoff = d.spectrum.small_wave_cutoff;
    desc->spectrum.gravity           = d.spectrum.gravity;

    desc->choppiness     = d.choppiness;
    desc->foam_threshold = d.foam_threshold;
    desc->seed           = d.seed;
    desc->thread_count   = d.thread_count;
    desc->water_depth    = d.spectrum.depth;
    desc->compute_velocity = d.compute_velocity ? 1 : 0;
}

ocean_sim* ocean_create(const ocean_desc* desc, ocean_status* out_status)
{
    auto fail = [&](ocean_status s) -> ocean_sim* {
        if (out_status != nullptr) *out_status = s;
        return nullptr;
    };

    ocean_desc resolved;
    if (!copy_desc(desc, resolved)) return fail(OCEAN_ERROR_INVALID_ARG);

    try {
        auto* sim = new ocean::Ocean(translate(resolved));
        if (out_status != nullptr) *out_status = OCEAN_OK;
        return to_c(sim);
    } catch (const std::invalid_argument&) {
        return fail(OCEAN_ERROR_INVALID_ARG);
    } catch (const std::bad_alloc&) {
        return fail(OCEAN_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        // A catch-all is mandatory, not defensive programming: letting any
        // other exception escape would unwind through the caller's C frame,
        // which is undefined behaviour.
        return fail(OCEAN_ERROR_UNKNOWN);
    }
}

void ocean_destroy(ocean_sim* sim)
{
    // Deliberately tolerant of NULL, matching free() - it removes a null check
    // from every caller's cleanup path.
    delete to_cpp(sim);
}

void ocean_update(ocean_sim* sim, double time)
{
    if (sim == nullptr) return;
    to_cpp(sim)->update(time);
}

ocean_buffers ocean_get_buffers(const ocean_sim* sim)
{
    ocean_buffers out;
    std::memset(&out, 0, sizeof(out));
    if (sim == nullptr) return out;

    const ocean::Buffers b = to_cpp(sim)->buffers();
    out.displacement = b.displacement;
    out.normal       = b.normal;
    out.size         = b.size;
    out.patch_length = b.patch_length;
    return out;
}

double ocean_get_time(const ocean_sim* sim)
{
    return (sim == nullptr) ? 0.0 : to_cpp(sim)->time();
}

float ocean_height_at(const ocean_sim* sim, float world_x, float world_z)
{
    return (sim == nullptr) ? 0.0f : to_cpp(sim)->height_at(world_x, world_z);
}

ocean_surface ocean_sample_at(const ocean_sim* sim, float world_x, float world_z)
{
    ocean_surface out;
    std::memset(&out, 0, sizeof(out));
    out.normal_y = 1.0f;  // flat water, so a null handle still yields a usable normal
    if (sim == nullptr) return out;

    const ocean::Surface s = to_cpp(sim)->sample_at(world_x, world_z);
    out.height   = s.height;
    out.offset_x = s.offset_x;
    out.offset_z = s.offset_z;
    out.normal_x = s.normal_x;
    out.normal_y = s.normal_y;
    out.normal_z = s.normal_z;
    out.foam     = s.foam;
    return out;
}

const char* ocean_simd_level(void)
{
    return ocean::detail::simd_level_name(ocean::detail::detect_simd_level());
}

const char* ocean_force_simd_level(const char* name)
{
    ocean::detail::SimdLevel level;
    if (name == nullptr) {
        level = ocean::detail::max_simd_level();
    } else {
        std::string s(name);
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (s == "scalar")      level = ocean::detail::SimdLevel::Scalar;
        else if (s == "sse2")   level = ocean::detail::SimdLevel::Sse2;
        else if (s == "avx2")   level = ocean::detail::SimdLevel::Avx2;
        else if (s == "neon")   level = ocean::detail::SimdLevel::Neon;
        else {
            // Unrecognised name: leave the current level untouched rather than
            // guessing, and report what is actually active.
            return ocean_simd_level();
        }
    }
    return ocean::detail::simd_level_name(ocean::detail::force_simd_level(level));
}

void ocean_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    if (major != nullptr) *major = 1;
    if (minor != nullptr) *minor = 0;
    if (patch != nullptr) *patch = 0;
}
void ocean_sample_ex(const ocean_sim* sim, float world_x, float world_z,
                     ocean_surface_ex* out)
{
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));
    out->normal_y = 1.0f;
    if (sim == nullptr) return;

    const ocean::Surface s = to_cpp(sim)->sample_at(world_x, world_z);
    out->height     = s.height;
    out->offset_x   = s.offset_x;
    out->offset_z   = s.offset_z;
    out->normal_x   = s.normal_x;
    out->normal_y   = s.normal_y;
    out->normal_z   = s.normal_z;
    out->foam       = s.foam;
    out->velocity_x = s.velocity_x;
    out->velocity_y = s.velocity_y;
    out->velocity_z = s.velocity_z;
}

const float* ocean_get_velocity(const ocean_sim* sim)
{
    if (sim == nullptr) return nullptr;
    return to_cpp(sim)->buffers().velocity;
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

namespace {

// Named distinctly rather than overloading to_cpp: these live inside the
// extern "C" block, where overloading is not allowed.
ocean::InteractionField* to_field(ocean_interaction* f) noexcept
{
    return reinterpret_cast<ocean::InteractionField*>(f);
}
const ocean::InteractionField* to_cfield(const ocean_interaction* f) noexcept
{
    return reinterpret_cast<const ocean::InteractionField*>(f);
}
ocean_interaction* field_to_c(ocean::InteractionField* f) noexcept
{
    return reinterpret_cast<ocean_interaction*>(f);
}

// Same struct_size contract as copy_desc(): a caller's smaller struct is a
// byte-for-byte PREFIX of ours, so copy exactly what they provided and leave
// our defaults in the rest. A LARGER struct_size is an error rather than a
// truncation - that caller was built against a newer header, and silently
// ignoring a field they believe they set is worse than refusing.
bool copy_interaction_desc(const ocean_interaction_desc* src,
                           ocean_interaction_desc& dst) noexcept
{
    if (src == nullptr) return false;
    if (src->struct_size == 0 ||
        src->struct_size > sizeof(ocean_interaction_desc)) {
        return false;
    }
    ocean_interaction_desc_init(&dst);
    std::memcpy(&dst, src, src->struct_size);
    dst.struct_size = sizeof(ocean_interaction_desc);
    return true;
}

ocean::InteractionDesc translate(const ocean_interaction_desc& c) noexcept
{
    ocean::InteractionDesc d;
    d.size          = c.size;
    d.extent        = c.extent;
    d.kernel_radius = c.kernel_radius;
    d.damping       = c.damping;
    d.absorb_cells  = c.absorb_cells;
    d.fixed_dt      = c.fixed_dt;
    d.max_substeps  = c.max_substeps;
    d.max_sources   = c.max_sources;
    d.gravity       = c.gravity;
    d.depth         = c.water_depth;
    d.obstruction   = (c.obstruction == OCEAN_OBSTRUCTION_DIRICHLET)
                          ? ocean::Obstruction::Dirichlet
                          : ocean::Obstruction::Neumann;
    // Layout-identical by construction, so this is a cast and not a
    // trampoline on the hot path - exactly as for OceanDesc.
    d.parallel_for      = reinterpret_cast<ocean::ParallelForFn>(c.parallel_for);
    d.parallel_for_user = c.parallel_for_user;
    d.thread_count      = c.thread_count;
    return d;
}

}  // namespace

void ocean_interaction_desc_init(ocean_interaction_desc* desc)
{
    if (desc == nullptr) return;
    // Read from the C++ defaults rather than restating them, so the two APIs
    // cannot drift.
    const ocean::InteractionDesc d{};
    std::memset(desc, 0, sizeof(*desc));
    desc->struct_size   = sizeof(ocean_interaction_desc);
    desc->size          = d.size;
    desc->extent        = d.extent;
    desc->kernel_radius = d.kernel_radius;
    desc->damping       = d.damping;
    desc->absorb_cells  = d.absorb_cells;
    desc->fixed_dt      = d.fixed_dt;
    desc->max_substeps  = d.max_substeps;
    desc->max_sources   = d.max_sources;
    desc->gravity       = d.gravity;
    desc->water_depth   = d.depth;
    desc->obstruction   = (d.obstruction == ocean::Obstruction::Dirichlet)
                              ? OCEAN_OBSTRUCTION_DIRICHLET
                              : OCEAN_OBSTRUCTION_NEUMANN;
    desc->thread_count  = d.thread_count;
}

ocean_interaction* ocean_interaction_create(const ocean_interaction_desc* desc,
                                            ocean_status* out_status)
{
    auto fail = [&](ocean_status st) -> ocean_interaction* {
        if (out_status != nullptr) *out_status = st;
        return nullptr;
    };

    ocean_interaction_desc resolved;
    if (!copy_interaction_desc(desc, resolved)) {
        return fail(OCEAN_ERROR_INVALID_ARG);
    }
    try {
        auto* f = new ocean::InteractionField(translate(resolved));
        if (out_status != nullptr) *out_status = OCEAN_OK;
        return field_to_c(f);
    } catch (const std::invalid_argument&) {
        return fail(OCEAN_ERROR_INVALID_ARG);
    } catch (const std::bad_alloc&) {
        return fail(OCEAN_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        return fail(OCEAN_ERROR_UNKNOWN);
    }
}

void ocean_interaction_destroy(ocean_interaction* field)
{
    delete to_field(field);
}

void ocean_interaction_add(ocean_interaction* field, const ocean_disturbance* d)
{
    if (field == nullptr || d == nullptr) return;
    ocean::Disturbance s;
    s.world_x    = d->world_x;
    s.world_z    = d->world_z;
    s.radius     = d->radius;
    s.strength   = d->strength;
    s.velocity_y = d->velocity_y;
    s.velocity_x = d->velocity_x;
    s.velocity_z = d->velocity_z;
    s.kind = (d->kind == OCEAN_SOURCE_CONTINUOUS) ? ocean::SourceKind::Continuous
                                                  : ocean::SourceKind::Impulse;
    to_field(field)->add(s);
}

uint64_t ocean_interaction_dropped(const ocean_interaction* field)
{
    return field == nullptr ? 0u : to_cfield(field)->dropped_sources();
}

void ocean_interaction_update(ocean_interaction* field, float dt)
{
    if (field == nullptr) return;
    to_field(field)->update(dt);
}

int32_t ocean_interaction_recenter(ocean_interaction* field, float world_x,
                                   float world_z)
{
    if (field == nullptr) return 0;
    to_field(field)->recenter(world_x, world_z);
    return to_field(field)->recentered() ? 1 : 0;
}

void ocean_interaction_set_obstruction(ocean_interaction* field,
                                       const uint8_t* mask)
{
    if (field == nullptr) return;
    to_field(field)->set_obstruction(mask);
}

ocean_interaction_buffers ocean_interaction_get_buffers(
    const ocean_interaction* field)
{
    ocean_interaction_buffers b{};
    if (field == nullptr) return b;
    const ocean::InteractionBuffers ib = to_cfield(field)->buffers();
    b.field    = ib.field;
    b.size     = ib.size;
    b.extent   = ib.extent;
    b.origin_x = ib.origin_x;
    b.origin_z = ib.origin_z;
    return b;
}

void ocean_interaction_sample_at(const ocean_interaction* field, float world_x,
                                 float world_z, ocean_interaction_sample* out)
{
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));
    if (field == nullptr) return;
    const ocean::InteractionSample s = to_cfield(field)->sample_at(world_x, world_z);
    out->height     = s.height;
    out->slope_x    = s.slope_x;
    out->slope_z    = s.slope_z;
    out->velocity_y = s.velocity_y;
}

float ocean_interaction_height_at(const ocean_interaction* field, float world_x,
                                  float world_z)
{
    return field == nullptr ? 0.0f : to_cfield(field)->height_at(world_x, world_z);
}

float ocean_interaction_fixed_dt(const ocean_interaction* field)
{
    return field == nullptr ? 0.0f : to_cfield(field)->fixed_dt();
}

float ocean_interaction_dt_limit(const ocean_interaction* field)
{
    return field == nullptr ? 0.0f : to_cfield(field)->stable_dt_limit();
}

float ocean_interaction_dispersion_error(const ocean_interaction* field)
{
    return field == nullptr ? 0.0f
                            : to_cfield(field)->kernel_dispersion_error();
}

void ocean_sample_combined(const ocean_sim* sim, const ocean_interaction* field,
                           float world_x, float world_z, ocean_surface_ex* out)
{
    if (out == nullptr) return;
    if (field == nullptr) { ocean_sample_ex(sim, world_x, world_z, out); return; }
    if (sim == nullptr) {
        std::memset(out, 0, sizeof(*out));
        out->normal_y = 1.0f;
        const ocean::InteractionSample e =
            to_cfield(field)->sample_at(world_x, world_z);
        out->height     = e.height;
        out->velocity_y = e.velocity_y;
        return;
    }
    const ocean::WaterSurface water{*to_cpp(sim), *to_cfield(field)};
    const ocean::Surface s = water.sample_at(world_x, world_z);
    out->height     = s.height;
    out->offset_x   = s.offset_x;
    out->offset_z   = s.offset_z;
    out->normal_x   = s.normal_x;
    out->normal_y   = s.normal_y;
    out->normal_z   = s.normal_z;
    out->foam       = s.foam;
    out->velocity_x = s.velocity_x;
    out->velocity_y = s.velocity_y;
    out->velocity_z = s.velocity_z;
}


}  // extern "C"
