// oceanlib - internal header. Not part of the public API.
//
// The built-in thread pool, used only when the host supplies no scheduler of
// its own via OceanDesc::parallel_for.
#pragma once

#include "ocean/ocean.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace ocean::detail {

// A minimal blocking parallel-for pool.
//
// Deliberately not a general task system: it implements exactly the contract
// in ADR-006 and nothing more. There is no queue of pending jobs, no futures,
// no task graph - one job is in flight at a time and the caller blocks until
// it is done. That is all the library asks for, and it means the pool has no
// lifetime or ownership questions to answer.
class ThreadPool {
public:
    // `count` == 0 means one worker per hardware thread.
    explicit ThreadPool(unsigned count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Runs task(ctx, i) for every i in [0, count) and returns only when all of
    // them have finished.
    void parallel_for(TaskFn task, void* ctx, std::uint32_t count);

    // Number of worker threads, not counting the calling thread (which also
    // participates in every job).
    [[nodiscard]] unsigned worker_count() const noexcept
    {
        return static_cast<unsigned>(threads_.size());
    }

private:
    struct Job {
        TaskFn        fn    = nullptr;
        void*         ctx   = nullptr;
        std::uint32_t count = 0;
    };

    void worker_loop();
    void drain() noexcept;

    std::mutex              mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;

    Job                        job_{};
    std::atomic<std::uint32_t> next_{0};
    unsigned                   active_     = 0;
    std::uint64_t              generation_ = 0;
    bool                       stop_       = false;

    std::vector<std::thread> threads_;
};

// Adapter matching the public ParallelForFn signature, so the built-in pool is
// plugged in through exactly the same hook a host scheduler would use. No
// separate code path exists for "our threads" versus "their threads", which
// means the hook is exercised by every single test run.
void pool_dispatch(void* user, TaskFn task, void* ctx,
                   std::uint32_t count);

// Runs every index inline on the calling thread. Used when the work is too
// small to be worth distributing, so that the "serial" case still goes through
// the same dispatch pointer as everything else and needs no branch in run().
void serial_dispatch(void* user, TaskFn task, void* ctx, std::uint32_t count);

// Splits [0, total) into `task_count` contiguous ranges and returns the one
// belonging to `index`. Remainder is spread over the first few tasks rather
// than dumped on the last, so chunk sizes differ by at most one.
inline void chunk_range(std::uint32_t index, std::uint32_t task_count,
                        std::uint32_t total, std::uint32_t& begin,
                        std::uint32_t& end) noexcept
{
    const std::uint32_t per = total / task_count;
    const std::uint32_t rem = total % task_count;
    const std::uint32_t lead = (index < rem) ? index : rem;
    begin = index * per + lead;
    end   = begin + per + ((index < rem) ? 1u : 0u);
}

}  // namespace ocean::detail
