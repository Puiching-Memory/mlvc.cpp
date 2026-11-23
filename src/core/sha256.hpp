#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace mlvc {

using Sha256Digest = std::array<std::byte, 32>;

class Sha256 final {
public:
    Sha256() noexcept;

    void update(std::span<const std::byte> data) noexcept;
    void update(const void* data, std::size_t size) noexcept;
    Sha256Digest finalize() noexcept;

private:
    void transform(const std::uint8_t block[64]) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_len_ = 0;
    std::uint64_t total_bytes_ = 0;
};

Sha256Digest sha256_digest(std::span<const std::byte> data) noexcept;
bool sha256_equals(const Sha256Digest& digest, std::string_view expected_raw) noexcept;

}  // namespace mlvc
