// oceanlib - interaction field benchmark.
//
// The convolution is the hot loop of the whole feature: (2P+1)^2 taps per cell
// per substep. This measures it against grid size and kernel radius, next to
// the FFT ocean's own cost, so the two can be compared in the terms an
// integrator actually budgets in - milliseconds per frame.
//
// Flags mirror bench_update:
//   --simd scalar|sse2|avx2|neon|max
//   --threads N
//   --sizes 128,256,512
//   --radii 4,6,8

#include "core/cpu_features.hpp"
#include "ocean/interaction.hpp"
#include "ocean/ocean.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ocean::detail::SimdLevel;

struct Timing {
    double median_ms = 0.0;
    double min_ms    = 0.0;
};

Timing summarise(std::vector<double>& s)
{
    std::sort(s.begin(), s.end());
    Timing t;
    t.median_ms = s[s.size() / 2];
    t.min_ms    = s.front();
    return t;
}

template <typename Fn>
Timing measure(Fn&& fn, int warmup, int iterations)
{
    for (int i = 0; i < warmup; ++i) fn(i);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        fn(i);
        const auto end = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    return summarise(samples);
}

int iterations_for(std::uint32_t n)
{
    if (n <= 128) return 1200;
    if (n <= 256) return 400;
    return 120;
}

std::vector<std::uint32_t> parse_list(const char* s, std::vector<std::uint32_t> fallback)
{
    if (s == nullptr || *s == '\0') return fallback;
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(static_cast<std::uint32_t>(std::strtoul(tok.c_str(), nullptr, 10)));
    }
    return out.empty() ? fallback : out;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string simd;
    std::uint32_t threads = 0;
    std::vector<std::uint32_t> sizes{128, 256, 512};
    std::vector<std::uint32_t> radii{3, 4, 6, 8, 10};

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--simd")         simd = next();
        else if (a == "--threads") threads = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--sizes")   sizes = parse_list(next(), sizes);
        else if (a == "--radii")   radii = parse_list(next(), radii);
    }

    if (!simd.empty() && simd != "max") {
        SimdLevel want = SimdLevel::Scalar;
        if (simd == "sse2") want = SimdLevel::Sse2;
        else if (simd == "avx2") want = SimdLevel::Avx2;
        else if (simd == "neon") want = SimdLevel::Neon;
        ocean::detail::force_simd_level(want);
    }
    std::printf("SIMD: %s, threads: %s\n",
                ocean::detail::simd_level_name(ocean::detail::detect_simd_level()),
                threads == 0 ? "hardware_concurrency" : std::to_string(threads).c_str());

    // --- cost against grid size, at the default radius ---------------------
    std::printf("\nInteraction update, P = 6 (169 taps), one substep per update\n");
    std::printf("%6s %12s %12s %12s %10s %12s\n", "N", "serial ms",
                "threaded ms", "speedup", "Mcell/s", "dt limit s");

    for (std::uint32_t n : sizes) {
        ocean::InteractionDesc d;
        d.size          = n;
        d.extent        = 0.25f * static_cast<float>(n);   // hold dx at 0.25 m
        d.kernel_radius = 6;
        d.max_sources   = 8;

        d.thread_count = 1;
        ocean::InteractionDesc ds = d;
        ds.size = n;
        // A single-task dispatch, to isolate the kernel from the scheduler.
        ds.parallel_for = [](void*, ocean::TaskFn task, void* ctx, std::uint32_t count) {
            for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
        };
        ocean::InteractionField serial{ds};

        ocean::InteractionDesc dt_ = d;
        dt_.thread_count = threads;
        ocean::InteractionField threaded{dt_};

        // Keep a live ripple in the field so the convolution is operating on
        // real data rather than a field of exact zeros, which could in
        // principle be optimised differently by the hardware.
        auto seed = [](ocean::InteractionField& f) {
            ocean::Disturbance s;
            s.world_x = s.world_z = 0.0f;
            s.radius = 0.5f; s.strength = 0.3f;
            f.add(s);
            f.update(1.0f / 60.0f);
            for (int i = 0; i < 20; ++i) f.update(1.0f / 60.0f);
        };
        seed(serial);
        seed(threaded);

        const int iters = iterations_for(n);
        const Timing st = measure([&](int) { serial.update(1.0f / 60.0f); }, 20, iters);
        const Timing tt = measure([&](int) { threaded.update(1.0f / 60.0f); }, 20, iters);

        const double cells = static_cast<double>(n) * n;
        std::printf("%6u %12.3f %12.3f %11.2fx %10.1f %12.4f\n", n,
                    st.median_ms, tt.median_ms, st.median_ms / tt.median_ms,
                    cells / (tt.median_ms * 1e-3) / 1e6,
                    static_cast<double>(threaded.stable_dt_limit()));
    }

    // --- cost against kernel radius ----------------------------------------
    std::printf("\nInteraction update at 256^2, against kernel radius\n");
    std::printf("%4s %7s %13s %13s %12s %14s\n", "P", "taps", "serial ms",
                "threaded ms", "ms/Mtap", "dispersion err");

    for (std::uint32_t p : radii) {
        ocean::InteractionDesc d;
        d.size          = 256;
        d.extent        = 64.0f;
        d.kernel_radius = p;
        d.max_sources   = 8;

        ocean::InteractionDesc ds = d;
        ds.parallel_for = [](void*, ocean::TaskFn task, void* ctx, std::uint32_t count) {
            for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
        };
        ocean::InteractionField serial{ds};

        ocean::InteractionDesc dtd = d;
        dtd.thread_count = threads;
        ocean::InteractionField threaded{dtd};

        ocean::Disturbance s;
        s.radius = 0.5f; s.strength = 0.3f;
        serial.add(s); threaded.add(s);
        for (int i = 0; i < 21; ++i) { serial.update(1.0f/60.0f); threaded.update(1.0f/60.0f); }

        const Timing st = measure([&](int) { serial.update(1.0f / 60.0f); }, 20, 400);
        const Timing tt = measure([&](int) { threaded.update(1.0f / 60.0f); }, 20, 400);

        const double taps = static_cast<double>(2 * p + 1) * (2 * p + 1);
        const double mtaps = taps * 256.0 * 256.0 / 1e6;
        std::printf("%4u %7.0f %13.3f %13.3f %12.4f %13.2f%%\n", p, taps,
                    st.median_ms, tt.median_ms, st.median_ms / mtaps,
                    100.0 * threaded.kernel_dispersion_error());
    }

    // --- next to the FFT ocean ---------------------------------------------
    std::printf("\nFor scale, the FFT ocean at the same resolutions\n");
    std::printf("%6s %14s %16s\n", "N", "ocean ms", "interaction ms");
    for (std::uint32_t n : sizes) {
        ocean::OceanDesc od;
        od.size         = n;
        od.patch_length = 200.0f;
        od.thread_count = threads;
        ocean::Ocean sim{od};

        ocean::InteractionDesc id;
        id.size         = n;
        id.extent       = 0.25f * static_cast<float>(n);
        id.thread_count = threads;
        id.max_sources  = 8;
        ocean::InteractionField field{id};
        ocean::Disturbance s;
        s.radius = 0.5f; s.strength = 0.3f;
        field.add(s);
        for (int i = 0; i < 21; ++i) field.update(1.0f / 60.0f);

        const int iters = iterations_for(n);
        const Timing ot = measure([&](int i) { sim.update(100.0 + i * 0.0137); }, 20, iters);
        const Timing it = measure([&](int) { field.update(1.0f / 60.0f); }, 20, iters);
        std::printf("%6u %14.3f %16.3f\n", n, ot.median_ms, it.median_ms);
    }

    std::printf("\nOne update() at 60 fps runs exactly one substep, so the\n");
    std::printf("interaction column is also the per-frame cost at 60 fps.\n");
    return 0;
}
