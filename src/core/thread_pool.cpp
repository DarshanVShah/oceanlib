#include "core/thread_pool.hpp"

namespace ocean::detail {

ThreadPool::ThreadPool(unsigned count)
{
    if (count == 0) {
        count = std::thread::hardware_concurrency();
        if (count == 0) count = 1;
    }
    // The calling thread participates in every job, so it accounts for one
    // worker. Spawning `count` threads on top of it would oversubscribe by one
    // and leave the pool fighting itself for the last core.
    const unsigned spawn = (count > 1) ? (count - 1) : 0;

    threads_.reserve(spawn);
    for (unsigned i = 0; i < spawn; ++i) {
        threads_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        ++generation_;
    }
    work_cv_.notify_all();
    for (std::thread& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::drain() noexcept
{
    // Every worker, plus the calling thread, races for indices from a single
    // atomic counter. This is work-stealing in its simplest possible form, and
    // it is what keeps the pool balanced when tasks take unequal time - which
    // they do here, because the FFT column pass touches memory very differently
    // from the row pass.
    for (;;) {
        const std::uint32_t i = next_.fetch_add(1, std::memory_order_relaxed);
        if (i >= job_.count) return;
        job_.fn(job_.ctx, i);
    }
}

void ThreadPool::worker_loop()
{
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            // job_ was written before generation_ advanced, both under this
            // mutex, so acquiring it here gives us the happens-before edge that
            // makes the lock-free read inside drain() safe.
        }

        drain();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--active_ == 0) done_cv_.notify_one();
        }
    }
}

void ThreadPool::parallel_for(TaskFn task, void* ctx, std::uint32_t count)
{
    if (count == 0) return;

    // Running inline for trivial jobs avoids paying two condition-variable
    // round trips to do less work than the synchronisation costs.
    if (threads_.empty() || count == 1) {
        for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = Job{task, ctx, count};
        next_.store(0, std::memory_order_relaxed);
        active_ = static_cast<unsigned>(threads_.size());
        ++generation_;
    }
    work_cv_.notify_all();

    // The calling thread is a worker too. Besides using the core it is sitting
    // on anyway, this means a pool of size 1 degenerates to a plain inline loop
    // with no threads at all.
    drain();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return active_ == 0; });
}

void pool_dispatch(void* user, TaskFn task, void* ctx, std::uint32_t count)
{
    static_cast<ThreadPool*>(user)->parallel_for(task, ctx, count);
}

void serial_dispatch(void*, TaskFn task, void* ctx, std::uint32_t count)
{
    for (std::uint32_t i = 0; i < count; ++i) task(ctx, i);
}

}  // namespace ocean::detail
