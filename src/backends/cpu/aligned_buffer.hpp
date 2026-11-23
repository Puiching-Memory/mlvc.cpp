#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <utility>

namespace mlvc::cpu_backend {

inline constexpr std::size_t align_up_64(std::size_t v) noexcept
{
    return (v + 63U) & ~std::size_t{63U};
}

// 64-byte aligned contiguous byte buffer for SIMD-friendly tensor arenas and I/O.
class AlignedByteBuffer final {
public:
    static constexpr std::size_t kAlignment = 64U;

    AlignedByteBuffer() = default;
    ~AlignedByteBuffer() { reset(); }

    AlignedByteBuffer(const AlignedByteBuffer&) = delete;
    AlignedByteBuffer& operator=(const AlignedByteBuffer&) = delete;

    AlignedByteBuffer(AlignedByteBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0U))
    {
    }

    AlignedByteBuffer& operator=(AlignedByteBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0U);
        }
        return *this;
    }

    void assign_zero(std::size_t bytes)
    {
        reset();
        if (bytes == 0U)
            return;
        data_ = static_cast<std::byte*>(
            ::operator new(bytes, std::align_val_t{kAlignment}));
        size_ = bytes;
        std::memset(data_, 0, bytes);
    }

    void zero() noexcept
    {
        if (data_ && size_ > 0U)
            std::memset(data_, 0, size_);
    }

    void swap(AlignedByteBuffer& other) noexcept
    {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    std::byte* data() noexcept { return data_; }
    const std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    void reset() noexcept
    {
        if (data_) {
            ::operator delete(data_, std::align_val_t{kAlignment});
            data_ = nullptr;
            size_ = 0U;
        }
    }

    std::byte* data_ = nullptr;
    std::size_t size_ = 0U;
};

}  // namespace mlvc::cpu_backend
