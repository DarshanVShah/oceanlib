#pragma once
#include <atomic>
#include <cstddef>

// Counts every trip through global operator new in the test binary.
//
// The library promises zero heap allocation inside update(). The only way to
// *prove* that - rather than assert it by reading the code - is to replace the
// global allocator and watch the counter across the call. This catches the
// accidental std::vector temporary or std::function copy that code review
// misses.
namespace alloc_probe {
extern std::atomic<std::size_t> allocation_count;
inline std::size_t count() noexcept
{
    return allocation_count.load(std::memory_order_relaxed);
}
}  // namespace alloc_probe
