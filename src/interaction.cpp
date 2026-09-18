#include "ocean/interaction.hpp"

#include "core/aligned.hpp"
#include "core/iwave_kernel.hpp"
#include "core/iwave_step.hpp"
#include "core/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ocean {
namespace {

// Work below this many cells runs serially.
//
// ADR-012 measured 8192 for the FFT pipeline; this workload is different and
// so is its crossover. Measured here: 128^2 (16384 cells) is SLOWER threaded -
// 0.137 ms against 0.112 ms serial, 0.82x - while 256^2 wins by 2.35x. One
// dispatch per substep is cheaper than the FFT's four barriers, but there is
// also far less work to spread at small sizes.
//
// The crossover is therefore somewhere in (16384, 65536]; only powers of two
// are legal sizes, so those are the only two points that exist to measure and
// the threshold is placed between them. The principle is the defensible part,
// and it is ADR-012's: threading must never make things worse.
constexpr std::size_t kMinCellsForThreading = 32768;

constexpr bool is_power_of_two(std::uint32_t v) noexcept
{
    return v != 0 && (v & (v - 1)) == 0;
}

// Round a row length up so each row starts on a 64-byte boundary. Cache-line
// alignment here is about false sharing between row-range tasks, exactly as in
// ADR-004 - the stencil's taps are at arbitrary offsets, so the loads
// themselves are unaligned no matter what we do.
constexpr std::size_t align_stride(std::size_t v) noexcept
{
    return (v + 15u) & ~static_cast<std::size_t>(15u);
}

// The impulse profile: the 2-D Ricker wavelet, (1 - u) e^{-u} with
// u = r^2 / 2*sigma^2.
//
// Three reasons, in increasing order of how much they matter.
//
// 1. It looks like an impact crater: a central depression ringed by a raised
//    rim of 13.5% the depth at r = 2*sigma, which is the shallow broad rim real
//    craters have. A single Gaussian spike looks like nothing in nature.
//
// 2. ITS INTEGRAL OVER THE PLANE IS EXACTLY ZERO. Under the radial
//    substitution the integral is proportional to integral (1-u) e^-u du = 0.
//    This is not aesthetic. The operator's symbol is exactly zero at k = 0, so
//    any net volume injected can NEVER propagate away - it would sit there as a
//    permanent bump forever. A Gaussian dimple, whose integral is not zero,
//    does exactly that. Zero net volume is the physically forced choice, and it
//    is the same statement as "the rock displaces water, and the rim IS that
//    displaced water".
//
// 3. Its transform is proportional to k^2 e^{-sigma^2 k^2 / 2}, which peaks at
//    k = sqrt(2)/sigma. So `radius` is a WAVELENGTH SELECTOR: the dominant
//    emitted wavelength is 2*pi*sigma/sqrt(2) ~= 4.44*sigma. That is what makes
//    the dispersion test clean - inject at a known sigma, predict the group
//    velocity, measure the ring.
//
// Returns +1 at the centre; the caller applies the sign so that a positive
// `strength` makes a crater and a negative `velocity_y` pushes water down.
inline float ricker(float r2, float inv_two_sigma2) noexcept
{
    const float u = r2 * inv_two_sigma2;
    return (1.0f - u) * std::exp(-u);
}

}  // namespace

// ---------------------------------------------------------------------------

struct InteractionField::Impl {
    InteractionDesc desc;
    detail::IWaveKernel kernel;

    std::uint32_t n      = 0;
    std::size_t   stride = 0;
    std::size_t   cells  = 0;
    float         cell   = 0.0f;     // metres per cell
    float         dt     = 0.0f;
    float         dt_limit = 0.0f;

    // Two time levels, each with a P-cell halo so the stencil never needs a
    // bounds test. `cur_` and `old_` point at interior cell (0,0).
    detail::AlignedBuffer<float> buf_a, buf_b;
    float* cur = nullptr;
    float* old = nullptr;

    // Per-cell damping coefficients; the absorbing layer makes damping
    // spatially varying.
    detail::AlignedBuffer<float> c1, c2;

    // Public RGBA-shaped output.
    detail::AlignedBuffer<float> out;

    // Obstruction.
    detail::AlignedBuffer<std::uint8_t>  mask;
    detail::AlignedBuffer<std::uint32_t> nearest_fluid;  // for Neumann
    detail::AlignedBuffer<std::uint32_t> solid_list;
    detail::AlignedBuffer<std::uint32_t> bfs_queue;
    std::uint32_t solid_count = 0;
    bool          has_mask    = false;

    // Source queue. Fixed capacity, so add() never allocates.
    detail::AlignedBuffer<Disturbance> queue;
    std::uint32_t queue_count    = 0;
    std::uint64_t dropped        = 0;

    // Accumulator state.
    double        accum          = 0.0;
    std::uint32_t substeps_last  = 0;
    bool          clamped_last   = false;

    // Grid origin, in cells, relative to world origin. Integer by
    // construction - that is the whole no-jolt argument.
    std::int32_t origin_cell_x = 0;
    std::int32_t origin_cell_z = 0;
    std::int32_t shift_x       = 0;
    std::int32_t shift_z       = 0;
    bool         moved         = false;

    // Scheduling, mirroring Ocean::Impl exactly - one dispatch path, no branch
    // for "our threads" versus "theirs", so the host hook is exercised by every
    // test run.
    std::unique_ptr<detail::ThreadPool> pool;
    ParallelForFn dispatch      = nullptr;
    void*         dispatch_user = nullptr;
    std::uint32_t task_count    = 1;

    [[nodiscard]] float* base(detail::AlignedBuffer<float>& b) noexcept
    {
        return b.data() + static_cast<std::size_t>(desc.kernel_radius) * stride +
               desc.kernel_radius;
    }

    explicit Impl(const InteractionDesc& d) : desc(d)
    {
        n     = d.size;
        cell  = d.extent / static_cast<float>(d.size);
        cells = static_cast<std::size_t>(n) * n;

        const std::size_t p = d.kernel_radius;
        stride = align_stride(static_cast<std::size_t>(n) + 2 * p);
        const std::size_t rows  = static_cast<std::size_t>(n) + 2 * p;
        const std::size_t total = rows * stride;

        detail::build_iwave_kernel(d.kernel_radius, cell, d.depth,
                                   detail::KernelMethod::LeastSquares, kernel);
        if (!detail::kernel_is_stable(kernel)) {
            throw std::invalid_argument(
                "ocean: interaction kernel_radius " +
                std::to_string(d.kernel_radius) +
                " yields a negative Fourier symbol (unconditionally unstable)");
        }

        // Stability limit from the REALISED symbol, not from theory. Leapfrog
        // on h'' = -omega^2 h is stable iff omega*dt <= 2, and the largest
        // omega present is sqrt(g * max_k S_realised(k)) - which truncation
        // ringing can push above the ideal Nyquist value. Deriving the limit
        // from the kernel we actually built is the difference between a bound
        // that holds and one that merely looks right.
        dt_limit = static_cast<float>(
            2.0 / std::sqrt(static_cast<double>(d.gravity) * kernel.max_symbol));

        // Default timestep from ACCURACY, not stability. At the stability
        // limit the shortest waves are stable and 57% wrong in frequency
        // (sin(w~ dt/2) = w dt/2 gives w~ dt = pi against a true 2), so the
        // default sits an order of magnitude below it. 1/60 s is also exactly
        // one substep per frame at 60 Hz, which keeps the common case cheap.
        dt = (d.fixed_dt > 0.0f) ? d.fixed_dt
                                 : std::min(1.0f / 60.0f, 0.35f * dt_limit);
        if (d.fixed_dt > 0.0f && d.fixed_dt > dt_limit) {
            throw std::invalid_argument(
                "ocean: interaction fixed_dt exceeds the stability limit of " +
                std::to_string(dt_limit) + " s for this configuration");
        }

        buf_a = detail::AlignedBuffer<float>(total);
        buf_b = detail::AlignedBuffer<float>(total);
        cur   = base(buf_a);
        old   = base(buf_b);

        c1  = detail::AlignedBuffer<float>(cells);
        c2  = detail::AlignedBuffer<float>(cells);
        out = detail::AlignedBuffer<float>(cells * 4);

        mask          = detail::AlignedBuffer<std::uint8_t>(cells);
        nearest_fluid = detail::AlignedBuffer<std::uint32_t>(cells);
        solid_list    = detail::AlignedBuffer<std::uint32_t>(cells);
        bfs_queue     = detail::AlignedBuffer<std::uint32_t>(cells);
        queue         = detail::AlignedBuffer<Disturbance>(d.max_sources);

        build_damping();
    }

    // The absorbing layer.
    //
    // Damping ramps up QUADRATICALLY from the inner edge of the layer rather
    // than switching on at it. A step change in damping is an impedance
    // discontinuity and reflects almost as badly as the hard boundary it is
    // supposed to replace; a smooth ramp lets the wave enter the layer without
    // seeing an interface, and then attenuates it before it reaches the edge.
    void build_damping() noexcept
    {
        const float a0 = desc.damping;
        const int   w  = static_cast<int>(desc.absorb_cells);

        // Strength at the very edge, chosen so a wave crossing the layer is
        // attenuated to a few thousandths. Expressed relative to the timestep
        // so the layer behaves the same at any dt.
        const float a_edge = 4.0f / std::max(dt * static_cast<float>(std::max(w, 1)), 1e-6f);

        for (std::uint32_t z = 0; z < n; ++z) {
            for (std::uint32_t x = 0; x < n; ++x) {
                const int dx = std::min<int>(static_cast<int>(x),
                                             static_cast<int>(n - 1 - x));
                const int dz = std::min<int>(static_cast<int>(z),
                                             static_cast<int>(n - 1 - z));
                const int d  = std::min(dx, dz);

                float alpha = a0;
                if (w > 0 && d < w) {
                    const float t = static_cast<float>(w - d) / static_cast<float>(w);
                    alpha += a_edge * t * t;
                }
                const float ad = alpha * dt;
                const std::size_t i = static_cast<std::size_t>(z) * n + x;
                c1[i] = 1.0f / (1.0f + ad);
                c2[i] = 1.0f - ad;
            }
        }
    }

    [[nodiscard]] detail::IWaveGrid grid() noexcept
    {
        detail::IWaveGrid g;
        g.n = n; g.p = desc.kernel_radius; g.stride = stride;
        g.cur = cur; g.old = old; g.taps = kernel.taps.data();
        g.c1 = c1.data(); g.c2 = c2.data();
        g.gdt2 = desc.gravity * dt * dt;
        return g;
    }

    // --- obstruction -------------------------------------------------------

    void rebuild_mask_tables() noexcept
    {
        solid_count = 0;
        if (!has_mask) return;

        // Multi-source BFS outward from every fluid cell, so each solid cell
        // learns the index of its nearest fluid cell. Done once per
        // set_obstruction, never per frame.
        std::uint32_t* q = bfs_queue.data();
        std::uint32_t head = 0, tail = 0;
        for (std::size_t i = 0; i < cells; ++i) {
            if (mask[i] == 0) {
                nearest_fluid[i] = static_cast<std::uint32_t>(i);
                q[tail++] = static_cast<std::uint32_t>(i);
            } else {
                nearest_fluid[i] = 0xFFFFFFFFu;
                solid_list[solid_count++] = static_cast<std::uint32_t>(i);
            }
        }
        while (head < tail) {
            const std::uint32_t i = q[head++];
            const std::uint32_t x = i % n, z = i / n;
            const int dx[4] = {1, -1, 0, 0};
            const int dz[4] = {0, 0, 1, -1};
            for (int e = 0; e < 4; ++e) {
                const int nx = static_cast<int>(x) + dx[e];
                const int nz = static_cast<int>(z) + dz[e];
                if (nx < 0 || nz < 0 || nx >= static_cast<int>(n) ||
                    nz >= static_cast<int>(n)) continue;
                const std::uint32_t j =
                    static_cast<std::uint32_t>(nz) * n + static_cast<std::uint32_t>(nx);
                if (nearest_fluid[j] == 0xFFFFFFFFu) {
                    nearest_fluid[j] = nearest_fluid[i];
                    q[tail++] = j;
                }
            }
        }
    }

    // Apply the boundary condition to h^n before convolving.
    void apply_obstruction() noexcept
    {
        if (!has_mask || solid_count == 0) return;

        if (desc.obstruction == Obstruction::Dirichlet) {
            // eta = 0: a pressure-release surface. Reflects a crest as a
            // trough. Kept because it is what the published method does.
            for (std::uint32_t s = 0; s < solid_count; ++s) {
                const std::uint32_t i = solid_list[s];
                cur[(i / n) * stride + (i % n)] = 0.0f;
            }
        } else {
            // dEta/dn = 0: fill each solid cell from its nearest fluid cell, so
            // the field has zero gradient into the solid. Reflects a crest as
            // a crest, which is what a rigid hull actually does.
            for (std::uint32_t s = 0; s < solid_count; ++s) {
                const std::uint32_t i = solid_list[s];
                const std::uint32_t f = nearest_fluid[i];
                if (f == 0xFFFFFFFFu) continue;    // no fluid anywhere
                cur[(i / n) * stride + (i % n)] =
                    cur[(f / n) * stride + (f % n)];
            }
        }
    }

    // --- sources -----------------------------------------------------------

    void inject(const Disturbance& d, float sub_time, float scale) noexcept
    {
        const float sigma = std::max(d.radius, 0.25f * cell);
        const float inv2s2 = 1.0f / (2.0f * sigma * sigma);

        // Continuous sources move across the substeps of one update, so a
        // moving hull draws one wake instead of a row of separate stamps.
        const float wx = d.world_x + d.velocity_x * sub_time;
        const float wz = d.world_z + d.velocity_z * sub_time;

        // Grid position of the source, in cell units.
        const float gx = (wx - static_cast<float>(origin_cell_x) * cell) / cell;
        const float gz = (wz - static_cast<float>(origin_cell_z) * cell) / cell;

        // Clip at 4 sigma, then force the DISCRETE sum to zero.
        //
        // The continuous Ricker integrates to exactly zero, but a clipped,
        // cell-sampled copy of it does not: the tail beyond radius R carries
        // u*exp(-u) with u = R^2/2*sigma^2, which is 5% of the peak at 3 sigma.
        // That residual is net volume, and net volume is the one thing this
        // operator can never propagate away - it would sit there as a permanent
        // bump forever, because the symbol is exactly zero at k = 0.
        //
        // So the mean over the stamped region is subtracted, exactly as the
        // kernel's own DC term is. 4 sigma rather than 3 makes the residual
        // 0.27% instead of 5%, so the constant being subtracted is small enough
        // that the step it leaves at the clip edge is invisible.
        const int rad = static_cast<int>(std::ceil(4.0f * sigma / cell));
        const int x0 = std::max(0, static_cast<int>(std::floor(gx)) - rad);
        const int x1 = std::min(static_cast<int>(n) - 1,
                                static_cast<int>(std::floor(gx)) + rad);
        const int z0 = std::max(0, static_cast<int>(std::floor(gz)) - rad);
        const int z1 = std::min(static_cast<int>(n) - 1,
                                static_cast<int>(std::floor(gz)) + rad);
        if (x1 < x0 || z1 < z0) return;

        double shape_sum = 0.0;
        for (int z = z0; z <= z1; ++z) {
            for (int x = x0; x <= x1; ++x) {
                const float px = (static_cast<float>(x) + 0.5f) - gx;
                const float pz = (static_cast<float>(z) + 0.5f) - gz;
                shape_sum += ricker((px * px + pz * pz) * cell * cell, inv2s2);
            }
        }
        const float shape_mean = static_cast<float>(
            shape_sum / (static_cast<double>(x1 - x0 + 1) *
                         static_cast<double>(z1 - z0 + 1)));

        const float disp = -d.strength * scale;
        const float vel  =  d.velocity_y * scale;

        for (int z = z0; z <= z1; ++z) {
            for (int x = x0; x <= x1; ++x) {
                const float px = (static_cast<float>(x) + 0.5f) - gx;
                const float pz = (static_cast<float>(z) + 0.5f) - gz;
                const float r2 = (px * px + pz * pz) * cell * cell;
                const float s  = ricker(r2, inv2s2) - shape_mean;

                // Displacement must be added to BOTH time levels, or it would
                // also inject a velocity of disp/dt - an enormous one. Velocity
                // is injected by perturbing only the older level, since
                // dEta/dt = (h^n - h^{n-1}) / dt.
                const float dd = disp * s;
                const float dv = vel * s;
                const std::size_t ci = static_cast<std::size_t>(z) * stride + x;
                cur[ci] += dd;
                old[ci] += dd - dv * dt;
            }
        }
    }

    // --- stepping ----------------------------------------------------------

    struct StepCtx { Impl* self; };

    static void task_step(void* ctx, std::uint32_t i) noexcept
    {
        Impl& m = *static_cast<StepCtx*>(ctx)->self;
        std::uint32_t b, e;
        detail::chunk_range(i, m.task_count, m.n, b, e);
        if (b < e) {
            const detail::IWaveGrid g = m.grid();
            detail::iwave_step_rows(g, b, e);
        }
    }

    static void task_finalize(void* ctx, std::uint32_t i) noexcept
    {
        Impl& m = *static_cast<StepCtx*>(ctx)->self;
        std::uint32_t b, e;
        detail::chunk_range(i, m.task_count, m.n, b, e);
        if (b < e) {
            const detail::IWaveGrid g = m.grid();
            detail::iwave_finalize_rows(g, 0.5f / m.cell, 1.0f / m.dt,
                                        m.out.data(), b, e);
        }
    }

    void substep() noexcept
    {
        apply_obstruction();
        StepCtx ctx{this};
        dispatch(dispatch_user, &task_step, &ctx, task_count);
        std::swap(cur, old);
    }

    void finalize() noexcept
    {
        StepCtx ctx{this};
        dispatch(dispatch_user, &task_finalize, &ctx, task_count);
    }
};

// ---------------------------------------------------------------------------

namespace {

void validate(const InteractionDesc& d)
{
    if (!is_power_of_two(d.size) || d.size < 32 || d.size > 1024) {
        throw std::invalid_argument(
            "ocean: interaction size must be a power of two in [32, 1024], got " +
            std::to_string(d.size));
    }
    if (!(d.extent > 0.0f)) {
        throw std::invalid_argument("ocean: interaction extent must be positive");
    }
    if (d.kernel_radius < 1 || d.kernel_radius > 12) {
        throw std::invalid_argument(
            "ocean: interaction kernel_radius must be in [1, 12]");
    }
    if (2 * d.kernel_radius >= d.size) {
        throw std::invalid_argument(
            "ocean: interaction kernel_radius is too large for this size");
    }
    if (!(d.damping >= 0.0f)) {
        throw std::invalid_argument("ocean: interaction damping must be non-negative");
    }
    if (d.absorb_cells * 2 >= d.size) {
        throw std::invalid_argument(
            "ocean: interaction absorb_cells must be less than half the size");
    }
    if (d.max_substeps < 1) {
        throw std::invalid_argument("ocean: interaction max_substeps must be >= 1");
    }
    if (d.max_sources < 1) {
        throw std::invalid_argument("ocean: interaction max_sources must be >= 1");
    }
    if (!(d.gravity > 0.0f)) {
        throw std::invalid_argument("ocean: gravity must be positive");
    }
    if (d.fixed_dt < 0.0f) {
        throw std::invalid_argument("ocean: interaction fixed_dt must be non-negative");
    }
}

}  // namespace

InteractionField::InteractionField(const InteractionDesc& desc)
{
    validate(desc);
    impl_ = std::make_unique<Impl>(desc);

    Impl& m = *impl_;
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

    m.finalize();   // a valid, all-zero output before the first update()
}

InteractionField::~InteractionField() = default;
InteractionField::InteractionField(InteractionField&&) noexcept = default;
InteractionField& InteractionField::operator=(InteractionField&&) noexcept = default;

// ---------------------------------------------------------------------------

void InteractionField::add(const Disturbance& d) noexcept
{
    Impl& m = *impl_;
    if (m.queue_count >= m.desc.max_sources) {
        // Dropped and counted, never silently lost. Dropping the tail is
        // deterministic given the same submission order, so replay is
        // unaffected even when the queue overflows.
        ++m.dropped;
        return;
    }
    m.queue[m.queue_count++] = d;
}

std::uint64_t InteractionField::dropped_sources() const noexcept
{
    return impl_->dropped;
}

void InteractionField::update(float dt) noexcept
{
    Impl& m = *impl_;
    m.moved = false;

    if (!(dt > 0.0f)) dt = 0.0f;
    m.accum += static_cast<double>(dt);

    std::uint32_t steps = 0;
    if (m.accum >= m.dt) {
        steps = static_cast<std::uint32_t>(m.accum / static_cast<double>(m.dt));
    }
    m.clamped_last = steps > m.desc.max_substeps;
    if (m.clamped_last) {
        // The spiral-of-death guard: run what we allow and DISCARD the
        // backlog rather than carrying it, or a single long hitch would make
        // every following frame late too. The field runs slow for a moment;
        // the frame does not explode. Deterministic, because the clamp is a
        // function of the accumulator alone.
        steps   = m.desc.max_substeps;
        m.accum = 0.0;
    } else {
        m.accum -= static_cast<double>(steps) * static_cast<double>(m.dt);
    }
    m.substeps_last = steps;

    if (steps == 0) {
        // No time passed, so continuous sources contribute nothing. Impulses
        // are KEPT: an impulse is an event, and dropping it because the frame
        // was short would lose a rock entirely.
        std::uint32_t keep = 0;
        for (std::uint32_t i = 0; i < m.queue_count; ++i) {
            if (m.queue[i].kind == SourceKind::Impulse) m.queue[keep++] = m.queue[i];
        }
        m.queue_count = keep;
        return;
    }

    for (std::uint32_t s = 0; s < steps; ++s) {
        const float sub_time = static_cast<float>(s) * m.dt;
        for (std::uint32_t i = 0; i < m.queue_count; ++i) {
            const Disturbance& d = m.queue[i];
            if (d.kind == SourceKind::Impulse) {
                // Exactly once, at the first substep. `strength` is in metres.
                if (s == 0) m.inject(d, 0.0f, 1.0f);
            } else {
                // Every substep, scaled by dt. `strength` is metres/second, so
                // the result does not depend on how many substeps ran.
                m.inject(d, sub_time, m.dt);
            }
        }
        m.substep();
    }
    m.queue_count = 0;

    m.finalize();
}

std::uint32_t InteractionField::last_substeps() const noexcept
{
    return impl_->substeps_last;
}

bool InteractionField::last_update_clamped() const noexcept
{
    return impl_->clamped_last;
}

// ---------------------------------------------------------------------------

void InteractionField::recenter(float world_x, float world_z) noexcept
{
    Impl& m = *impl_;
    const float half = 0.5f * m.desc.extent;

    // Snap to whole cells. This is the entire no-jolt argument: an integer
    // shift is an exact move, so every retained cell keeps its bit-exact value
    // and nothing is resampled, interpolated or filtered.
    const std::int32_t tx = static_cast<std::int32_t>(
        std::lround((world_x - half) / m.cell));
    const std::int32_t tz = static_cast<std::int32_t>(
        std::lround((world_z - half) / m.cell));

    const std::int32_t dx = tx - m.origin_cell_x;
    const std::int32_t dz = tz - m.origin_cell_z;
    m.shift_x = dx;
    m.shift_z = dz;
    if (dx == 0 && dz == 0) { m.moved = false; return; }
    m.moved = true;

    const std::int32_t n = static_cast<std::int32_t>(m.n);
    m.origin_cell_x = tx;
    m.origin_cell_z = tz;

    if (std::abs(dx) >= n || std::abs(dz) >= n) {
        // Moved further than the whole grid: nothing is retained.
        std::memset(m.buf_a.data(), 0, m.buf_a.size() * sizeof(float));
        std::memset(m.buf_b.data(), 0, m.buf_b.size() * sizeof(float));
        std::memset(m.mask.data(), 0, m.mask.size());
        m.rebuild_mask_tables();
        m.finalize();
        return;
    }

    // Both time levels shift together. Shifting one and not the other would
    // misalign them and turn the second time derivative into noise - the
    // subtlest way to get this wrong.
    auto shift_plane = [&](float* p) {
        const std::int32_t zlo = (dz > 0) ? 0 : n - 1;
        const std::int32_t zhi = (dz > 0) ? n : -1;
        const std::int32_t zst = (dz > 0) ? 1 : -1;
        for (std::int32_t z = zlo; z != zhi; z += zst) {
            const std::int32_t sz = z + dz;
            float* dst = p + static_cast<std::ptrdiff_t>(z) * m.stride;
            if (sz < 0 || sz >= n) {
                std::memset(dst, 0, static_cast<std::size_t>(n) * sizeof(float));
                continue;
            }
            const float* src = p + static_cast<std::ptrdiff_t>(sz) * m.stride;
            // Row-direction order follows the shift so overlapping ranges
            // cannot corrupt; memmove handles the overlap within a row.
            const std::int32_t copy_lo = std::max<std::int32_t>(0, -dx);
            const std::int32_t copy_hi = std::min<std::int32_t>(n, n - dx);
            if (copy_hi > copy_lo) {
                std::memmove(dst + copy_lo, src + copy_lo + dx,
                             static_cast<std::size_t>(copy_hi - copy_lo) * sizeof(float));
            }
            if (copy_lo > 0) {
                std::memset(dst, 0, static_cast<std::size_t>(copy_lo) * sizeof(float));
            }
            if (copy_hi < n) {
                std::memset(dst + copy_hi, 0,
                            static_cast<std::size_t>(n - copy_hi) * sizeof(float));
            }
        }
    };
    shift_plane(m.cur);
    shift_plane(m.old);

    if (m.has_mask) {
        // The mask travels with the field. Newly exposed cells default to
        // fluid; a host that owns a world-space mask should refresh it when
        // recentered() reports true.
        auto* mp = m.mask.data();
        const std::int32_t zlo = (dz > 0) ? 0 : n - 1;
        const std::int32_t zhi = (dz > 0) ? n : -1;
        const std::int32_t zst = (dz > 0) ? 1 : -1;
        for (std::int32_t z = zlo; z != zhi; z += zst) {
            const std::int32_t sz = z + dz;
            std::uint8_t* dst = mp + static_cast<std::ptrdiff_t>(z) * n;
            if (sz < 0 || sz >= n) {
                std::memset(dst, 0, static_cast<std::size_t>(n));
                continue;
            }
            const std::uint8_t* src = mp + static_cast<std::ptrdiff_t>(sz) * n;
            const std::int32_t copy_lo = std::max<std::int32_t>(0, -dx);
            const std::int32_t copy_hi = std::min<std::int32_t>(n, n - dx);
            if (copy_hi > copy_lo) {
                std::memmove(dst + copy_lo, src + copy_lo + dx,
                             static_cast<std::size_t>(copy_hi - copy_lo));
            }
            if (copy_lo > 0) std::memset(dst, 0, static_cast<std::size_t>(copy_lo));
            if (copy_hi < n) {
                std::memset(dst + copy_hi, 0, static_cast<std::size_t>(n - copy_hi));
            }
        }
        m.rebuild_mask_tables();
    }

    // Refresh the public buffer so buffers() and sample_at() always describe
    // the CURRENT grid. Without this a caller that recentred and then queried
    // before the next update() would read the old field through the new
    // origin - every value off by the shift, and silently so.
    m.finalize();
}

bool InteractionField::recentered() const noexcept { return impl_->moved; }
std::int32_t InteractionField::last_shift_x() const noexcept { return impl_->shift_x; }
std::int32_t InteractionField::last_shift_z() const noexcept { return impl_->shift_z; }

// ---------------------------------------------------------------------------

void InteractionField::set_obstruction(const std::uint8_t* m8) noexcept
{
    Impl& m = *impl_;
    if (m8 == nullptr) {
        std::memset(m.mask.data(), 0, m.mask.size());
        m.has_mask = false;
        m.solid_count = 0;
        return;
    }
    std::memcpy(m.mask.data(), m8, m.mask.size());
    m.has_mask = true;
    m.rebuild_mask_tables();
}

// ---------------------------------------------------------------------------

InteractionSample InteractionField::sample_at(float world_x,
                                              float world_z) const noexcept
{
    const Impl& m = *impl_;
    InteractionSample s;

    const float gx = (world_x - static_cast<float>(m.origin_cell_x) * m.cell) /
                     m.cell - 0.5f;
    const float gz = (world_z - static_cast<float>(m.origin_cell_z) * m.cell) /
                     m.cell - 0.5f;

    // Outside the grid the answer is genuinely zero - no disturbance out
    // there - so this is not a clamp, it is the correct value. The 1-cell
    // inset keeps the bilinear tap in bounds.
    if (gx < 0.0f || gz < 0.0f ||
        gx >= static_cast<float>(m.n - 1) || gz >= static_cast<float>(m.n - 1)) {
        return s;
    }

    const int   x0 = static_cast<int>(gx);
    const int   z0 = static_cast<int>(gz);
    const float tx = gx - static_cast<float>(x0);
    const float tz = gz - static_cast<float>(z0);

    const float* f = m.out.data();
    const std::size_t i00 = (static_cast<std::size_t>(z0) * m.n + x0) * 4;
    const std::size_t i10 = i00 + 4;
    const std::size_t i01 = i00 + static_cast<std::size_t>(m.n) * 4;
    const std::size_t i11 = i01 + 4;

    const float w00 = (1.0f - tx) * (1.0f - tz);
    const float w10 = tx * (1.0f - tz);
    const float w01 = (1.0f - tx) * tz;
    const float w11 = tx * tz;

    auto tap = [&](int c) {
        return f[i00 + c] * w00 + f[i10 + c] * w10 +
               f[i01 + c] * w01 + f[i11 + c] * w11;
    };
    s.height     = tap(0);
    s.slope_x    = tap(1);
    s.slope_z    = tap(2);
    s.velocity_y = tap(3);
    return s;
}

float InteractionField::height_at(float world_x, float world_z) const noexcept
{
    return sample_at(world_x, world_z).height;
}

// ---------------------------------------------------------------------------

InteractionBuffers InteractionField::buffers() const noexcept
{
    const Impl& m = *impl_;
    InteractionBuffers b;
    b.field    = m.out.data();
    b.size     = m.n;
    b.extent   = m.desc.extent;
    b.origin_x = static_cast<float>(m.origin_cell_x) * m.cell;
    b.origin_z = static_cast<float>(m.origin_cell_z) * m.cell;
    return b;
}

const InteractionDesc& InteractionField::desc() const noexcept { return impl_->desc; }
float InteractionField::fixed_dt() const noexcept { return impl_->dt; }
float InteractionField::stable_dt_limit() const noexcept { return impl_->dt_limit; }

float InteractionField::kernel_dispersion_error() const noexcept
{
    return static_cast<float>(impl_->kernel.peak_relative_error);
}

double InteractionField::energy() const noexcept
{
    const Impl& m = *impl_;
    const int p = static_cast<int>(m.desc.kernel_radius);
    const int w = 2 * p + 1;

    // The quantity leapfrog conserves exactly in the undamped case:
    //
    //     E = 1/2 |v|^2 + 1/2 g * h^n . G{h^{n-1}}
    //
    // The cross-level product in the potential term is what makes it exactly
    // conserved rather than merely conserved on average - the naive
    // same-level form oscillates by O(dt^2) every step, which would drown the
    // decay this is used to measure.
    //
    // The potential term is non-negative precisely because the realised symbol
    // is non-negative (it equals sum_k S(k)|h(k)|^2 in Fourier space), which is
    // the same property that makes the scheme stable at all.
    const double inv_dt = 1.0 / static_cast<double>(m.dt);
    double kinetic = 0.0, potential = 0.0;

    for (std::uint32_t z = 0; z < m.n; ++z) {
        const std::size_t row = static_cast<std::size_t>(z) * m.stride;
        for (std::uint32_t x = 0; x < m.n; ++x) {
            const double hc = m.cur[row + x];
            const double hp = m.old[row + x];
            const double v  = (hc - hp) * inv_dt;
            kinetic += v * v;

            const float* centre = m.old + row + x;
            const float* tap = m.kernel.taps.data();
            double acc = 0.0;
            for (int j = -p; j <= p; ++j) {
                const float* srow =
                    centre + static_cast<std::ptrdiff_t>(j) *
                                 static_cast<std::ptrdiff_t>(m.stride) - p;
                for (int i = 0; i < w; ++i) acc += tap[i] * srow[i];
                tap += w;
            }
            potential += hc * acc;
        }
    }
    return 0.5 * (kinetic + static_cast<double>(m.desc.gravity) * potential);
}

// ---------------------------------------------------------------------------
// WaterSurface
// ---------------------------------------------------------------------------

WaterSurface::WaterSurface(const Ocean& o, const InteractionField& f) noexcept
    : ocean_(&o), field_(&f) {}

WaterSurface::WaterSurface(const CascadeStack& s, const InteractionField& f) noexcept
    : stack_(&s), field_(&f) {}

InteractionSample WaterSurface::interaction_at(float x, float z) const noexcept
{
    return field_->sample_at(x, z);
}

Surface WaterSurface::sample_at(float world_x, float world_z) const noexcept
{
    Surface s = stack_ ? stack_->sample_at(world_x, world_z)
                       : ocean_->sample_at(world_x, world_z);
    const InteractionSample e = field_->sample_at(world_x, world_z);

    s.height += e.height;

    // Compose through SLOPES, not by summing unit normals.
    //
    // Heights add in world space, so world-space slopes add too - that is an
    // identity. Summing the two unit normals and renormalising, which is the
    // usual reflex and what ADR-020 had to do for cascades, would be an
    // approximation here for no reason: we have a genuine height field on one
    // side, not a second already-normalised normal.
    if (s.normal_y > 1.0e-6f) {
        const float inv = 1.0f / s.normal_y;
        const float sx = -s.normal_x * inv + e.slope_x;
        const float sz = -s.normal_z * inv + e.slope_z;
        const float len = std::sqrt(sx * sx + sz * sz + 1.0f);
        const float il = 1.0f / len;
        s.normal_x = -sx * il;
        s.normal_y = il;
        s.normal_z = -sz * il;
    }
    // normal_y <= 0 means the FFT surface has folded over itself (a breaking
    // crest). There is no single well-defined slope there to add to, so the
    // FFT normal is left as it stands rather than invented - the same honesty
    // ADR-011 applies to the query inside a breaker.

    return s;
}

float WaterSurface::height_at(float world_x, float world_z) const noexcept
{
    const float base = stack_ ? stack_->height_at(world_x, world_z)
                              : ocean_->height_at(world_x, world_z);
    return base + field_->height_at(world_x, world_z);
}

}  // namespace ocean
