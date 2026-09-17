#include <doctest/doctest.h>

#include "core/thread_pool.hpp"
#include "ocean/ocean.hpp"

#include <atomic>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

using namespace ocean;

namespace {

OceanDesc threaded_desc(std::uint32_t size, std::uint32_t threads)
{
    OceanDesc d;
    d.size                    = size;
    d.patch_length            = 200.0f;
    d.seed                    = 90210;
    d.choppiness              = 1.0f;
    d.spectrum.wind_speed     = 13.0f;
    d.spectrum.wind_direction = 0.7f;
    d.thread_count            = threads;
    return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// The property that matters: threading must not change a single bit
// ---------------------------------------------------------------------------

TEST_CASE("results are bit-identical regardless of thread count")
{
    // This is the whole reason the pipeline is structured as four barriered
    // stages of pure per-row work. If any stage read data another stage was
    // still writing, this test would fail intermittently - which is exactly
    // the kind of bug that ships.
    const std::uint32_t n = 128;
    const std::size_t bytes = static_cast<std::size_t>(n) * n * 4 * sizeof(float);

    Ocean reference{threaded_desc(n, 1)};
    reference.update(6.25);

    std::vector<float> ref_disp(
        reference.buffers().displacement,
        reference.buffers().displacement + static_cast<std::size_t>(n) * n * 4);
    std::vector<float> ref_nrm(
        reference.buffers().normal,
        reference.buffers().normal + static_cast<std::size_t>(n) * n * 4);

    for (std::uint32_t threads : {1u, 2u, 3u, 4u, 8u, 16u, 32u}) {
        CAPTURE(threads);
        Ocean sim{threaded_desc(n, threads)};
        sim.update(6.25);
        CHECK(std::memcmp(ref_disp.data(), sim.buffers().displacement, bytes) == 0);
        CHECK(std::memcmp(ref_nrm.data(), sim.buffers().normal, bytes) == 0);
    }
}

TEST_CASE("repeated threaded updates are stable, not just first-run correct")
{
    // A race would not necessarily show up on the first frame. Running the
    // same time repeatedly and requiring identical output every time gives the
    // scheduler many chances to interleave differently.
    const std::uint32_t n = 128;
    const std::size_t floats = static_cast<std::size_t>(n) * n * 4;

    Ocean sim{threaded_desc(n, 16)};
    sim.update(3.5);
    std::vector<float> first(sim.buffers().displacement,
                             sim.buffers().displacement + floats);

    for (int i = 0; i < 30; ++i) {
        sim.update(100.0 + i);   // wander away
        sim.update(3.5);         // and come back
        REQUIRE(std::memcmp(first.data(), sim.buffers().displacement,
                            floats * sizeof(float)) == 0);
    }
}

// ---------------------------------------------------------------------------
// The host scheduler hook
// ---------------------------------------------------------------------------

namespace {

struct HookState {
    std::atomic<int>           dispatches{0};
    std::atomic<std::uint32_t> items{0};
};

// The simplest legal implementation of the contract: run everything inline on
// the calling thread. An engine's ParallelFor is a drop-in replacement.
void serial_hook(void* user, TaskFn task, void* ctx, std::uint32_t count)
{
    HookState* st = static_cast<HookState*>(user);
    st->dispatches.fetch_add(1, std::memory_order_relaxed);
    st->items.fetch_add(count, std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
}

// A hook that runs indices in REVERSE order. Still satisfies the contract -
// it runs every index and blocks until all are done - and must therefore give
// identical results. This is a much sharper test than a serial hook: it proves
// we have no hidden dependency on task ordering.
void reverse_hook(void* user, TaskFn task, void* ctx, std::uint32_t count)
{
    static_cast<HookState*>(user)->dispatches.fetch_add(1, std::memory_order_relaxed);
    for (std::uint32_t i = count; i-- > 0;) task(ctx, i);
}

// A hook that scatters work across freshly spawned std::threads in an
// unpredictable order, to model a host job system we know nothing about.
void chaotic_hook(void* user, TaskFn task, void* ctx, std::uint32_t count)
{
    static_cast<HookState*>(user)->dispatches.fetch_add(1, std::memory_order_relaxed);

    std::atomic<std::uint32_t> next{0};
    const unsigned n_threads = 4;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned w = 0; w < n_threads; ++w) {
        workers.emplace_back([&] {
            for (;;) {
                const std::uint32_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) return;
                task(ctx, i);
            }
        });
    }
    for (std::thread& t : workers) t.join();  // the blocking half of the contract
}

}  // namespace

TEST_CASE("a host scheduler hook is used instead of the built-in pool")
{
    HookState st;
    OceanDesc d = threaded_desc(64, 0);
    d.parallel_for      = &serial_hook;
    d.parallel_for_user = &st;

    Ocean sim{d};
    // The constructor already runs one full frame.
    CHECK(st.dispatches.load() == 4);  // evolve, fft rows, fft cols, finalise

    sim.update(1.0);
    CHECK(st.dispatches.load() == 8);
    CHECK(st.items.load() > 0);
}

TEST_CASE("hooks that reorder or scatter work give identical results")
{
    const std::uint32_t n = 64;
    const std::size_t bytes = static_cast<std::size_t>(n) * n * 4 * sizeof(float);
    const double t = 4.75;

    Ocean builtin{threaded_desc(n, 4)};
    builtin.update(t);
    std::vector<float> expected(
        builtin.buffers().displacement,
        builtin.buffers().displacement + static_cast<std::size_t>(n) * n * 4);

    HookState st;

    SUBCASE("serial hook") {
        OceanDesc d = threaded_desc(n, 0);
        d.parallel_for = &serial_hook;
        d.parallel_for_user = &st;
        Ocean sim{d};
        sim.update(t);
        CHECK(std::memcmp(expected.data(), sim.buffers().displacement, bytes) == 0);
    }
    SUBCASE("reverse-order hook") {
        OceanDesc d = threaded_desc(n, 0);
        d.parallel_for = &reverse_hook;
        d.parallel_for_user = &st;
        Ocean sim{d};
        sim.update(t);
        CHECK(std::memcmp(expected.data(), sim.buffers().displacement, bytes) == 0);
    }
    SUBCASE("chaotic multi-threaded hook") {
        OceanDesc d = threaded_desc(n, 0);
        d.parallel_for = &chaotic_hook;
        d.parallel_for_user = &st;
        Ocean sim{d};
        for (int i = 0; i < 5; ++i) {
            sim.update(t);
            REQUIRE(std::memcmp(expected.data(), sim.buffers().displacement, bytes) == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// The pool itself
// ---------------------------------------------------------------------------

TEST_CASE("the thread pool runs every index exactly once")
{
    for (unsigned threads : {1u, 2u, 4u, 8u}) {
        CAPTURE(threads);
        detail::ThreadPool pool{threads};

        for (std::uint32_t count : {0u, 1u, 7u, 64u, 1000u}) {
            CAPTURE(count);
            std::vector<std::atomic<int>> visits(count == 0 ? 1 : count);
            for (auto& v : visits) v.store(0);

            struct Ctx { std::vector<std::atomic<int>>* v; };
            Ctx ctx{&visits};

            pool.parallel_for(
                [](void* c, std::uint32_t i) {
                    (*static_cast<Ctx*>(c)->v)[i].fetch_add(
                        1, std::memory_order_relaxed);
                },
                &ctx, count);

            for (std::uint32_t i = 0; i < count; ++i) {
                REQUIRE(visits[i].load() == 1);
            }
        }
    }
}

TEST_CASE("parallel_for does not return until all work has finished")
{
    // The blocking contract in ADR-006. If it returned early, the FFT column
    // pass would start reading rows the row pass had not written yet.
    detail::ThreadPool pool{8};
    constexpr std::uint32_t count = 512;

    std::vector<int> results(count, 0);
    struct Ctx { int* out; };
    Ctx ctx{results.data()};

    for (int repeat = 0; repeat < 20; ++repeat) {
        std::fill(results.begin(), results.end(), 0);
        pool.parallel_for(
            [](void* c, std::uint32_t i) {
                // A little work, so a non-blocking implementation would be
                // caught rather than racing to finish in time anyway.
                volatile double acc = 0.0;
                for (int k = 0; k < 200; ++k) acc += k * 0.5;
                static_cast<Ctx*>(c)->out[i] = static_cast<int>(i) + 1;
            },
            &ctx, count);

        // Immediately after the call returns, every slot must be written.
        for (std::uint32_t i = 0; i < count; ++i) {
            REQUIRE(results[i] == static_cast<int>(i) + 1);
        }
    }
}

TEST_CASE("the pool can be created and destroyed repeatedly without leaking threads")
{
    for (int i = 0; i < 20; ++i) {
        detail::ThreadPool pool{4};
        int value = 0;
        pool.parallel_for([](void* c, std::uint32_t) { *static_cast<int*>(c) += 1; },
                          &value, 1);
        REQUIRE(value == 1);
    }
}

TEST_CASE("chunk_range covers the whole span exactly once")
{
    for (std::uint32_t total : {0u, 1u, 5u, 64u, 127u, 1000u}) {
        for (std::uint32_t tasks : {1u, 2u, 3u, 7u, 64u, 200u}) {
            CAPTURE(total);
            CAPTURE(tasks);
            std::uint32_t covered = 0;
            std::uint32_t previous_end = 0;
            std::uint32_t largest = 0, smallest = 0xFFFFFFFFu;

            for (std::uint32_t i = 0; i < tasks; ++i) {
                std::uint32_t b, e;
                detail::chunk_range(i, tasks, total, b, e);
                REQUIRE(b == previous_end);  // contiguous, no gaps or overlap
                REQUIRE(e >= b);
                previous_end = e;
                covered += e - b;
                largest = std::max(largest, e - b);
                smallest = std::min(smallest, e - b);
            }
            REQUIRE(previous_end == total);
            REQUIRE(covered == total);
            // Remainder is spread, so chunks differ by at most one element.
            REQUIRE(largest - smallest <= 1);
        }
    }
}
