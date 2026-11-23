#include "core/sha256.hpp"

#include <cstring>

namespace mlvc {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept
{
    return (x >> n) | (x << (32U - n));
}

std::uint32_t load_be32(const std::uint8_t* p) noexcept
{
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

void store_be32(std::byte* p, std::uint32_t v) noexcept
{
    p[0] = static_cast<std::byte>((v >> 24) & 0xffU);
    p[1] = static_cast<std::byte>((v >> 16) & 0xffU);
    p[2] = static_cast<std::byte>((v >> 8) & 0xffU);
    p[3] = static_cast<std::byte>(v & 0xffU);
}

}  // namespace

Sha256::Sha256() noexcept
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}
{
}

void Sha256::transform(const std::uint8_t block[64]) noexcept
{
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = load_be32(block + i * 4);
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (int i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[static_cast<std::size_t>(i)] + w[i];
        const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const std::byte> data) noexcept
{
    update(data.data(), data.size());
}

void Sha256::update(const void* data, std::size_t size) noexcept
{
    if (!data || size == 0)
        return;
    const auto* ptr = static_cast<const std::uint8_t*>(data);
    total_bytes_ += static_cast<std::uint64_t>(size);

    if (buffer_len_ > 0) {
        const std::size_t need = 64U - buffer_len_;
        const std::size_t take = size < need ? size : need;
        std::memcpy(buffer_.data() + buffer_len_, ptr, take);
        buffer_len_ += take;
        ptr += take;
        size -= take;
        if (buffer_len_ == 64U) {
            transform(buffer_.data());
            buffer_len_ = 0;
        }
    }

    while (size >= 64U) {
        transform(ptr);
        ptr += 64U;
        size -= 64U;
    }

    if (size > 0) {
        std::memcpy(buffer_.data(), ptr, size);
        buffer_len_ = size;
    }
}

Sha256Digest Sha256::finalize() noexcept
{
    const std::uint64_t total_bits = total_bytes_ * 8ULL;
    buffer_[buffer_len_++] = 0x80U;
    if (buffer_len_ > 56U) {
        std::memset(buffer_.data() + buffer_len_, 0, 64U - buffer_len_);
        transform(buffer_.data());
        buffer_len_ = 0;
    }
    std::memset(buffer_.data() + buffer_len_, 0, 56U - buffer_len_);
    for (int i = 0; i < 8; ++i) {
        buffer_[56U + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((total_bits >> (56 - i * 8)) & 0xffULL);
    }
    transform(buffer_.data());

    Sha256Digest digest{};
    for (std::size_t i = 0; i < 8U; ++i)
        store_be32(digest.data() + i * 4U, state_[i]);
    return digest;
}

Sha256Digest sha256_digest(std::span<const std::byte> data) noexcept
{
    Sha256 hasher;
    hasher.update(data);
    return hasher.finalize();
}

bool sha256_equals(const Sha256Digest& digest, std::string_view expected_raw) noexcept
{
    if (expected_raw.size() != digest.size())
        return false;
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < digest.size(); ++i) {
        diff |= static_cast<std::uint8_t>(digest[i]) ^
                static_cast<std::uint8_t>(static_cast<unsigned char>(expected_raw[i]));
    }
    return diff == 0;
}

}  // namespace mlvc
