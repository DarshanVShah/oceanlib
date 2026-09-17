// oceanlib - internal header. Not part of the public API.
//
// Cache-line-aligned, heap-allocated POD storage.
#pragma once

#include "ocean/ocean.hpp"

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace ocean::detail {

// Owns one aligned block of trivially-copyable elements.
//
// We use C++17's aligned `operator new` rather than posix_memalign or
// _aligned_malloc so there is exactly one portable code path; MSVC's CRT has
// no C11 aligned_alloc, so that route would need an #ifdef anyway.
//
// Alignment defaults to kBufferAlignment (64 B = one cache line). Cache-line
// alignment is not about SIMD load instructions - AVX2 loads work unaligned -
// it is about false sharing. When threads write neighbouring tiles of the same
// buffer, two threads touching one line would ping-pong that line between
// cores. Aligning the base and sizing tiles in whole lines removes that.
template <typename T>
class AlignedBuffer {
    static_assert(std::is_trivially_destructible_v<T>,
                  "AlignedBuffer does not run destructors");
    static_assert(std::is_trivially_copyable_v<T>,
                  "AlignedBuffer zero-initialises via memset");

public:
    AlignedBuffer() = default;

    explicit AlignedBuffer(std::size_t count,
                           std::size_t alignment = kBufferAlignment)
        : count_(count), alignment_(alignment)
    {
        if (count_ == 0) return;
        data_ = static_cast<T*>(
            ::operator new(count_ * sizeof(T), std::align_val_t{alignment_}));
        // Zero rather than leave indeterminate: an un-updated buffer must read
        // as a well-defined flat ocean, not as whatever was in that page.
        std::memset(static_cast<void*>(data_), 0, count_ * sizeof(T));
    }

    ~AlignedBuffer() { release(); }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)),
          alignment_(other.alignment_)
    {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            data_      = std::exchange(other.data_, nullptr);
            count_     = std::exchange(other.count_, 0);
            alignment_ = other.alignment_;
        }
        return *this;
    }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    [[nodiscard]] T*       data() noexcept       { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    T&       operator[](std::size_t i) noexcept       { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    void release() noexcept
    {
        if (data_) {
            ::operator delete(static_cast<void*>(data_),
                              std::align_val_t{alignment_});
            data_ = nullptr;
        }
        count_ = 0;
    }

    T*          data_      = nullptr;
    std::size_t count_     = 0;
    std::size_t alignment_ = kBufferAlignment;
};

}  // namespace ocean::detail
