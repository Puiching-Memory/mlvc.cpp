// Derived from microsoft/libmlvc (https://github.com/microsoft/libmlvc)
// Copyright (c) 2026 Microsoft Corporation
// Licensed under the MIT License. See third_party/libmlvc/LICENSE.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mlvc::detail {

void add_emulation_prevention_bytes(std::span<const std::byte> input,
                                    std::vector<std::byte>& output);
void remove_emulation_prevention_bytes(std::span<const std::byte> input,
                                       std::vector<std::byte>& output);

class BitWriter {
public:
    explicit BitWriter(std::size_t reserved_bytes = 64);

    void reset();
    bool empty() const noexcept { return bit_pos_ == 0; }

    void write_bit(bool value);
    void write_uint(int value, int num_bits);
    void write_ue(int value);
    void write_se(int value);
    void write_byte_padding();
    void write_stop_bit();
    void write_bytes(std::span<const std::byte> bytes);

    std::span<const std::byte> bytes() const noexcept { return buffer_; }

private:
    std::vector<std::byte> buffer_;
    std::size_t bit_pos_ = 0;
};

class BitReader {
public:
    explicit BitReader(std::span<const std::byte> buffer) noexcept
        : buffer_(buffer)
    {
    }

    bool read_bit();
    int read_uint(int num_bits);
    int read_ue();
    int read_se();
    void read_byte_padding();
    void read_stop_bit();
    std::span<const std::byte> read_bytes();

private:
    std::span<const std::byte> buffer_;
    std::size_t bit_pos_ = 0;
};

}  // namespace mlvc::detail
