#include "alloc_probe.hpp"

#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace alloc_probe {
std::atomic<std::size_t> allocation_count{0};
}

namespace {

void bump() noexcept
{
    alloc_probe::allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void* raw_alloc(std::size_t bytes) noexcept
{
    return std::malloc(bytes == 0 ? 1 : bytes);
}

void* raw_alloc_aligned(std::size_t bytes, std::size_t alignment) noexcept
{
    if (bytes == 0) bytes = 1;
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, alignment);
#else
    // C11 aligned_alloc requires size to be a multiple of alignment.
    const std::size_t rounded = (bytes + alignment - 1) / alignment * alignment;
    return std::aligned_alloc(alignment, rounded);
#endif
}

void raw_free(void* p) noexcept { std::free(p); }

void raw_free_aligned(void* p) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace

// --- plain ------------------------------------------------------------------
void* operator new(std::size_t n)
{
    void* p = raw_alloc(n);
    if (!p) throw std::bad_alloc();
    bump();
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }

void* operator new(std::size_t n, const std::nothrow_t&) noexcept
{
    void* p = raw_alloc(n);
    if (p) bump();
    return p;
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept
{
    return ::operator new(n, t);
}

void operator delete(void* p) noexcept { raw_free(p); }
void operator delete[](void* p) noexcept { raw_free(p); }
void operator delete(void* p, std::size_t) noexcept { raw_free(p); }
void operator delete[](void* p, std::size_t) noexcept { raw_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { raw_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { raw_free(p); }

// --- over-aligned (C++17) ---------------------------------------------------
void* operator new(std::size_t n, std::align_val_t a)
{
    void* p = raw_alloc_aligned(n, static_cast<std::size_t>(a));
    if (!p) throw std::bad_alloc();
    bump();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a)
{
    return ::operator new(n, a);
}
void* operator new(std::size_t n, std::align_val_t a,
                   const std::nothrow_t&) noexcept
{
    void* p = raw_alloc_aligned(n, static_cast<std::size_t>(a));
    if (p) bump();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t& t) noexcept
{
    return ::operator new(n, a, t);
}

void operator delete(void* p, std::align_val_t) noexcept { raw_free_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { raw_free_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept
{
    raw_free_aligned(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept
{
    raw_free_aligned(p);
}
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept
{
    raw_free_aligned(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept
{
    raw_free_aligned(p);
}
