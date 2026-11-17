// Derived from microsoft/libmlvc (https://github.com/microsoft/libmlvc)
// Copyright (c) 2026 Microsoft Corporation
// Licensed under the MIT License. See licenses/libmlvc/LICENSE.

#include "bit_io.hpp"

#include "mlvc/core/bitstream.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlvc::detail {

void add_emulation_prevention_bytes(std::span<const std::byte> input,
                                    std::vector<std::byte>& output)
{
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (i >= 2 && output.size() >= 2 &&
            output[output.size() - 2] == std::byte{0x00} &&
            output[output.size() - 1] == std::byte{0x00} &&
            input[i] <= std::byte{0x03}) {
            output.push_back(std::byte{0x03});
        }
        output.push_back(input[i]);
    }
}

void remove_emulation_prevention_bytes(std::span<const std::byte> input,
                                       std::vector<std::byte>& output)
{
    output.clear();
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (i >= 2 && input[i - 2] == std::byte{0x00} &&
            input[i - 1] == std::byte{0x00} && input[i] == std::byte{0x03}) {
            continue;
        }
        output.push_back(input[i]);
    }
}

BitWriter::BitWriter(std::size_t reserved_bytes)
{
    buffer_.reserve(reserved_bytes);
}

void BitWriter::reset()
{
    buffer_.clear();
    bit_pos_ = 0;
}

void BitWriter::write_bit(bool value)
{
    const std::size_t byte_pos = bit_pos_ / CHAR_BIT;
    while (byte_pos >= buffer_.size())
        buffer_.push_back(std::byte{0});
    const std::byte mask = std::byte{0x01} << (CHAR_BIT - 1 - (bit_pos_ % CHAR_BIT));
    if (value)
        buffer_[byte_pos] |= mask;
    else
        buffer_[byte_pos] &= ~mask;
    ++bit_pos_;
}

void BitWriter::write_uint(int value, int num_bits)
{
    if (num_bits < 0 || num_bits > static_cast<int>(sizeof(int) * CHAR_BIT) - 1)
        throw std::runtime_error("BitWriter::write_uint: bit width out of range");
    if (value < 0)
        throw std::runtime_error("BitWriter::write_uint: value must be non-negative");
    if (num_bits < 31 && (static_cast<std::uint32_t>(value) >> num_bits) != 0U)
        throw std::runtime_error("BitWriter::write_uint: value exceeds field bit width");
    for (int i = num_bits - 1; i >= 0; --i)
        write_bit(((value >> i) & 0x01) != 0);
}

void BitWriter::write_ue(int value)
{
    if (value < 0 || value >= std::numeric_limits<int>::max())
        throw std::runtime_error("BitWriter::write_ue: value out of range");

    int num_bits = 0;
    int tmp = value + 1;
    while (tmp > 0) {
        tmp >>= 1;
        ++num_bits;
    }
    write_uint(0, num_bits - 1);
    write_uint(value + 1, num_bits);
}

void BitWriter::write_se(int value)
{
    if (value <= 0) {
        if (value <= std::numeric_limits<int>::min() / 2)
            throw std::runtime_error("BitWriter::write_se: negative value out of range");
        write_ue(-2 * value);
    } else {
        if (value > std::numeric_limits<int>::max() / 2)
            throw std::runtime_error("BitWriter::write_se: positive value out of range");
        write_ue(2 * value - 1);
    }
}

void BitWriter::write_byte_padding()
{
    const int padding = static_cast<int>(CHAR_BIT - (bit_pos_ % CHAR_BIT));
    if (padding != CHAR_BIT)
        write_uint(0, padding);
}

void BitWriter::write_stop_bit()
{
    write_bit(true);
    write_byte_padding();
}

void BitWriter::write_bytes(std::span<const std::byte> bytes)
{
    write_byte_padding();
    const std::size_t byte_pos = bit_pos_ / CHAR_BIT;
    buffer_.resize(byte_pos + bytes.size());
    std::copy(bytes.begin(), bytes.end(),
              buffer_.begin() + static_cast<std::ptrdiff_t>(byte_pos));
    bit_pos_ += bytes.size() * CHAR_BIT;
}

bool BitReader::read_bit()
{
    if (bit_pos_ >= buffer_.size() * CHAR_BIT) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "bitstream read out of bounds");
    }
    const std::size_t byte_pos = bit_pos_ / CHAR_BIT;
    const std::byte mask = std::byte{0x01} << (CHAR_BIT - 1 - (bit_pos_ % CHAR_BIT));
    const bool value = (buffer_[byte_pos] & mask) != std::byte{0};
    ++bit_pos_;
    return value;
}

int BitReader::read_uint(int num_bits)
{
    if (num_bits < 0 || num_bits > static_cast<int>(sizeof(int) * CHAR_BIT) - 1) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "bitstream integer bit width out of range");
    }
    int value = 0;
    for (int i = 0; i < num_bits; ++i)
        value = (value << 1) | (read_bit() ? 1 : 0);
    return value;
}

int BitReader::read_ue()
{
    int num_bits = 1;
    while (true) {
        if (read_bit()) {
            --bit_pos_;
            break;
        }
        ++num_bits;
        if (num_bits >= static_cast<int>(sizeof(int) * CHAR_BIT)) {
            throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                 "Exp-Golomb code exceeds 31-bit integer range");
        }
    }
    return read_uint(num_bits) - 1;
}

int BitReader::read_se()
{
    const int ue = read_ue();
    return (ue % 2 == 0) ? -(ue / 2) : ((ue + 1) / 2);
}

void BitReader::read_byte_padding()
{
    const std::size_t padding = CHAR_BIT - (bit_pos_ % CHAR_BIT);
    if (padding != CHAR_BIT) {
        if (read_uint(static_cast<int>(padding)) != 0) {
            throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                 "non-zero RBSP byte alignment padding");
        }
    }
}

void BitReader::read_stop_bit()
{
    if (!read_bit()) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "missing RBSP stop bit");
    }
    read_byte_padding();
}

std::span<const std::byte> BitReader::read_bytes()
{
    read_byte_padding();
    if (bit_pos_ >= buffer_.size() * CHAR_BIT) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "bitstream payload read out of bounds");
    }
    // Subtract 1 bit for the trailing RBSP stop bit.
    const std::size_t num_bytes = (buffer_.size() * CHAR_BIT - bit_pos_ - 1) / CHAR_BIT;
    const auto result = buffer_.subspan(bit_pos_ / CHAR_BIT, num_bytes);
    bit_pos_ += num_bytes * CHAR_BIT;
    return result;
}

}  // namespace mlvc::detail
