#include "ocean/foam.hpp"

#include "core/aligned.hpp"
#include "core/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ocean {
namespace {

constexpr std::size_t kMinCellsForThreading = 32768;

// Bilinear fetch from a periodic N x N single-channel grid, in cell units.
//
// Wrapping rather than clamping is correct, not a convenience: the FFT surface
// is exactly periodic with period patch_length, so the foam riding on it is
// too. Clamping would smear the edge rows into a streak that a REPEAT sampler
// would then tile across the whole ocean.
inline float sample_wrap(const float* f, float u, float v, int n) noexcept
{
    const float fu = std::floor(u);
    const float fv = std::floor(v);
    const float tu = u - fu;
    const float tv = v - fv;

    int x0 = static_cast<int>(fu) % n; if (x0 < 0) x0 += n;
    int z0 = static_cast<int>(fv) % n; if (z0 < 0) z0 += n;
    const int x1 = (x0 + 1) % n;
    const int z1 = (z0 + 1) % n;

    const float a = f[static_cast<std::size_t>(z0) * n + x0];
    const float b = f[static_cast<std::size_t>(z0) * n + x1];
    const float c = f[static_cast<std::size_t>(z1) * n + x0];
    const float d = f[static_cast<std::size_t>(z1) * n + x1];

    return a * (1.0f - tu) * (1.0f - tv) + b * tu * (1.0f - tv) +
           c * (1.0f - tu) * tv + d * tu * tv;
}

}  // namespace

// ---------------------------------------------------------------------------

struct FoamField::Impl {
    const Ocean* ocean = nullptr;
    FoamDesc     desc;

    std::uint32_t n     = 0;
    std::size_t   cells = 0;
    float         cell  = 0.0f;   // metres per cell

    // Ping-pong, because semi-Lagrangian advection reads a scattered
    // neighbourhood of the previous state while writing this one. Advecting in
    // place would let a cell read a value that had already been overwritten -
    // and, worse, WHICH values had been overwritten would depend on how the
    // scheduler split the rows, so the result would stop being deterministic
    // under threading.
    detail::AlignedBuffer<float> buf_a, buf_b;
    float* cur  = nullptr;
    float* next = nullptr;

    double        accum         = 0.0;
    std::uint32_t substeps_last = 0;

    std::unique_ptr<detail::ThreadPool> pool;
    ParallelForFn dispatch      = nullptr;
    void*         dispatch_user = nullptr;
    std::uint32_t task_count    = 1;

    struct StepCtx {
        Impl*        self;
        const float* disp;
        const float* vel;
        float        dt;
    };

    static void task_step(void* ctx, std::uint32_t i) noexcept
    {
        StepCtx& sc = *static_cast<StepCtx*>(ctx);
        Impl& m = *sc.self;
        std::uint32_t begin, end;
        detail::chunk_range(i, m.task_count, m.n, begin, end);
        if (begin < end) m.step_rows(sc, begin, end);
    }

    void step_rows(const StepCtx& sc, std::uint32_t row_begin,
                   std::uint32_t row_end) noexcept
    {
        const int   ni       = static_cast<int>(n);
        const float dt       = sc.dt;
        const float inv_cell = 1.0f / cell;
        const float decay    = std::exp(-desc.decay * dt);
        const float gain     = desc.source_gain * dt;
        const float scale    = desc.advect_scale;
        const float drift_x  = desc.wind_drift_x;
        const float drift_z  = desc.wind_drift_z;

        for (std::uint32_t z = row_begin; z < row_end; ++z) {
            for (std::uint32_t x = 0; x < n; ++x) {
                const std::size_t i = static_cast<std::size_t>(z) * n + x;

                // Semi-Lagrangian: trace BACKWARDS from where this cell is to
                // where the water now in it came from, and take the foam that
                // was there. Unconditionally stable for any timestep, which is
                // why no CFL limit appears anywhere in this class - the price
                // is numerical diffusion, which for foam reads as the patch
                // softening at its edges and is, if anything, welcome.
                const float vx = sc.vel[4 * i + 0] * scale + drift_x;
                const float vz = sc.vel[4 * i + 2] * scale + drift_z;

                const float su = static_cast<float>(x) - vx * dt * inv_cell;
                const float sv = static_cast<float>(z) - vz * dt * inv_cell;

                // Advection happens in the ocean's PARAMETER space, not in
                // world space. The two differ by the choppy displacement, so
                // this is an approximation - stated rather than hidden. It is
                // the right one to make: parameter space is where the grid is
                // periodic, so foam wraps exactly as the surface does, and at
                // ordinary choppiness the displacement gradient is a few per
                // cent of a cell per step.
                float f = sample_wrap(cur, su, sv, ni);

                // Source: the instantaneous Jacobian foam the FFT already
                // computes, in displacement.w. Reusing it rather than
                // rederiving from the Jacobian keeps this field and the
                // surface agreeing about WHERE foam is born, so only the
                // persistence and drift are new behaviour.
                const float src = sc.disp[4 * i + 3];

                f = f * decay + src * gain;
                if (f > 1.0f) f = 1.0f;
                if (f < 0.0f) f = 0.0f;
                next[i] = f;
            }
        }
    }
};

// ---------------------------------------------------------------------------

namespace {

void validate(const FoamDesc& d)
{
    if (!(d.decay >= 0.0f)) {
        throw std::invalid_argument("ocean: foam decay must be non-negative");
    }
    if (!(d.source_gain >= 0.0f)) {
        throw std::invalid_argument("ocean: foam source_gain must be non-negative");
    }
    if (!(d.fixed_dt > 0.0f)) {
        throw std::invalid_argument("ocean: foam fixed_dt must be positive");
    }
    if (d.max_substeps < 1) {
        throw std::invalid_argument("ocean: foam max_substeps must be >= 1");
    }
    if (!std::isfinite(d.advect_scale) || !std::isfinite(d.wind_drift_x) ||
        !std::isfinite(d.wind_drift_z)) {
        throw std::invalid_argument("ocean: foam drift parameters must be finite");
    }
}

}  // namespace

FoamField::FoamField(const Ocean& ocean, const FoamDesc& desc)
{
    validate(desc);

    const Buffers b = ocean.buffers();
    if (b.velocity == nullptr) {
        // Refused rather than silently degraded. Without a velocity field
        // there is nothing to advect foam with, and quietly producing
        // non-advected foam would leave a caller believing they had the
        // feature they asked for.
        throw std::invalid_argument(
            "ocean: FoamField requires OceanDesc::compute_velocity to be set");
    }

    impl_ = std::make_unique<Impl>();
    Impl& m = *impl_;
    m.ocean = &ocean;
    m.desc  = desc;
    m.n     = b.size;
    m.cells = static_cast<std::size_t>(b.size) * b.size;
    m.cell  = b.patch_length / static_cast<float>(b.size);

    m.buf_a = detail::AlignedBuffer<float>(m.cells);
    m.buf_b = detail::AlignedBuffer<float>(m.cells);
    m.cur   = m.buf_a.data();
    m.next  = m.buf_b.data();

    unsigned workers = desc.thread_count;
    if (workers == 0) workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 1;
    m.task_count = std::min(m.n, std::max(1u, workers * 4u));
    if (m.cells < kMinCellsForThreading) m.task_count = 1;

    if (desc.parallel_for != nullptr) {
        m.dispatch      = desc.parallel_for;
        m.dispatch_user = desc.parallel_for_user;
    } else if (m.task_count > 1) {
        m.pool = std::make_unique<detail::ThreadPool>(desc.thread_count);
        m.dispatch      = &detail::pool_dispatch;
        m.dispatch_user = m.pool.get();
    } else {
        m.dispatch      = &detail::serial_dispatch;
        m.dispatch_user = nullptr;
    }
}

FoamField::~FoamField() = default;
FoamField::FoamField(FoamField&&) noexcept = default;
FoamField& FoamField::operator=(FoamField&&) noexcept = default;

void FoamField::update(float dt) noexcept
{
    Impl& m = *impl_;
    if (!(dt > 0.0f)) dt = 0.0f;
    m.accum += static_cast<double>(dt);

    const double step = static_cast<double>(m.desc.fixed_dt);
    // A relative epsilon, so a 60 Hz host feeding exactly 1/60 s against a
    // 1/60 s step runs one substep every frame instead of alternating 0 and 2
    // as float rounding pushes the comparison either side of the boundary.
    const double eps = step * 1e-6;

    std::uint32_t steps = 0;
    if (m.accum + eps >= step) {
        steps = static_cast<std::uint32_t>((m.accum + eps) / step);
    }
    if (steps > m.desc.max_substeps) {
        steps   = m.desc.max_substeps;
        m.accum = 0.0;           // discard the backlog, never carry it
    } else {
        m.accum -= static_cast<double>(steps) * step;
        if (m.accum < 0.0) m.accum = 0.0;
    }
    m.substeps_last = steps;
    if (steps == 0) return;

    const Buffers b = m.ocean->buffers();
    if (b.velocity == nullptr) return;   // ocean was moved from; nothing to do

    Impl::StepCtx ctx{&m, b.displacement, b.velocity, m.desc.fixed_dt};
    for (std::uint32_t s = 0; s < steps; ++s) {
        m.dispatch(m.dispatch_user, &Impl::task_step, &ctx, m.task_count);
        std::swap(m.cur, m.next);
    }
}

void FoamField::clear() noexcept
{
    Impl& m = *impl_;
    std::memset(m.buf_a.data(), 0, m.buf_a.size() * sizeof(float));
    std::memset(m.buf_b.data(), 0, m.buf_b.size() * sizeof(float));
    m.accum = 0.0;
}

const float*  FoamField::data() const noexcept { return impl_->cur; }
std::uint32_t FoamField::size() const noexcept { return impl_->n; }
const FoamDesc& FoamField::desc() const noexcept { return impl_->desc; }

void FoamField::set_desc(const FoamDesc& d)
{
    validate(d);
    Impl& m = *impl_;
    // Preserve the fields that are construction-time decisions, so a caller
    // who passes a default-constructed FoamDesc cannot silently change the
    // timestep or orphan the thread pool.
    const float         keep_dt    = m.desc.fixed_dt;
    const std::uint32_t keep_steps = m.desc.max_substeps;
    ParallelForFn       keep_pf    = m.desc.parallel_for;
    void*               keep_user  = m.desc.parallel_for_user;
    const std::uint32_t keep_tc    = m.desc.thread_count;

    m.desc = d;
    m.desc.fixed_dt          = keep_dt;
    m.desc.max_substeps      = keep_steps;
    m.desc.parallel_for      = keep_pf;
    m.desc.parallel_for_user = keep_user;
    m.desc.thread_count      = keep_tc;
}
std::uint32_t FoamField::last_substeps() const noexcept
{
    return impl_->substeps_last;
}

float FoamField::foam_at(float world_x, float world_z) const noexcept
{
    const Impl& m = *impl_;

    // Ask the ocean where this world position came from in parameter space -
    // the same fixed-point inversion Ocean::sample_at runs (ADR-011) - so the
    // foam reported here belongs to the same piece of water as the height
    // reported there. Doing a naive lookup instead would put the foam on the
    // wrong side of a crest wherever choppiness is doing anything, which is
    // exactly where foam lives.
    const Surface s = m.ocean->sample_at(world_x, world_z);
    const float u = (world_x - s.offset_x) / m.cell;
    const float v = (world_z - s.offset_z) / m.cell;
    return sample_wrap(m.cur, u, v, static_cast<int>(m.n));
}

double FoamField::coverage() const noexcept
{
    const Impl& m = *impl_;
    double sum = 0.0;
    for (std::size_t i = 0; i < m.cells; ++i) sum += m.cur[i];
    return sum / static_cast<double>(m.cells);
}

}  // namespace ocean
