// oceanlib - update-loop benchmark.
//
// Prints per-frame update cost at several grid sizes, plus a breakdown of the
// three pipeline stages so we can see what actually dominates before trying to
// optimise anything.

#include "core/aligned.hpp"
#include "core/evolve.hpp"
#include "core/fft.hpp"
#include "core/spectrum.hpp"
#include "ocean/ocean.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Timing {
    double median_ms = 0.0;
    double min_ms    = 0.0;
};

// Median rather than mean: on a laptop, a scheduler hiccup or a thermal blip
// produces occasional huge outliers that drag a mean around and tell us
// nothing about the code. The minimum is reported alongside as the best case
// the machine is capable of.
Timing summarise(std::vector<double>& samples)
{
    std::sort(samples.begin(), samples.end());
    Timing t;
    t.median_ms = samples[samples.size() / 2];
    t.min_ms    = samples.front();
    return t;
}

template <typename Fn>
Timing measure(Fn&& fn, int warmup, int iterations)
{
    for (int i = 0; i < warmup; ++i) fn(static_cast<double>(i) * 0.013);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        // Advance simulation time every iteration so nothing degenerates into
        // a cached special case (e.g. t = 0 making every phase identical).
        const double t = 100.0 + static_cast<double>(i) * 0.0137;
        const auto start = Clock::now();
        fn(t);
        const auto end = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    return summarise(samples);
}

int iterations_for(std::uint32_t n)
{
    // Keep total wall time per size roughly constant.
    if (n <= 64) return 2000;
    if (n <= 128) return 800;
    if (n <= 256) return 300;
    return 100;
}

ocean::OceanDesc make_desc(std::uint32_t n)
{
    ocean::OceanDesc d;
    d.size                    = n;
    d.patch_length            = 200.0f;
    d.seed                    = 1337;
    d.choppiness              = 1.0f;
    d.spectrum.wind_speed     = 12.0f;
    d.spectrum.wind_direction = 0.4f;
    return d;
}

}  // namespace

int main()
{
    std::printf("oceanlib update benchmark\n");
    std::printf("-------------------------\n\n");
    std::printf("%6s %11s %11s %11s %11s %11s %11s\n", "N", "total(ms)",
                "best(ms)", "evolve(ms)", "fft(ms)", "final(ms)", "Mcell/s");

    for (std::uint32_t n : {64u, 128u, 256u, 512u}) {
        const ocean::OceanDesc desc = make_desc(n);
        const int iters  = iterations_for(n);
        const int warmup = std::max(10, iters / 10);

        // ---- whole update(), through the public API --------------------
        ocean::Ocean sim{desc};
        const Timing total = measure([&](double t) { sim.update(t); }, warmup, iters);

        // ---- stage breakdown, using the internal pieces directly -------
        using namespace ocean::detail;
        SpectrumTables tables;
        build_spectrum(desc, tables);
        FftPlan plan{n};
        FieldSet fields;
        fields.allocate(n);
        AlignedBuffer<float> scratch(plan.scratch_floats());
        const std::size_t cells = static_cast<std::size_t>(n) * n;
        AlignedBuffer<float> displacement(cells * 4);
        AlignedBuffer<float> normal(cells * 4);

        const Timing evolve = measure(
            [&](double t) { evolve_rows(tables, t, fields, 0, n); }, warmup, iters);

        const Timing fft = measure(
            [&](double) {
                for (int f = 0; f < 4; ++f) {
                    plan.transform_2d(fields.re(f), fields.im(f), scratch.data(),
                                      FftSign::Inverse);
                }
            },
            warmup, iters);

        const Timing fin = measure(
            [&](double) {
                finalize_rows(fields, desc.choppiness, desc.foam_threshold,
                              displacement.data(), normal.data(), 0, n);
            },
            warmup, iters);

        const double mcells_per_s =
            static_cast<double>(cells) / (total.median_ms * 1e-3) / 1e6;

        std::printf("%6u %11.3f %11.3f %11.3f %11.3f %11.3f %11.1f\n", n,
                    total.median_ms, total.min_ms, evolve.median_ms,
                    fft.median_ms, fin.median_ms, mcells_per_s);
    }

    std::printf("\nStage columns are measured separately and include their own\n");
    std::printf("warmup, so they will not sum exactly to the total.\n");
    return 0;
}
