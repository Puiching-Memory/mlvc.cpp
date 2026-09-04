#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace mlvc {

inline constexpr std::size_t kMaxEncodedFramePayloadBytes =
    std::size_t{256} << 20;

struct EncodedFrame {
    std::int32_t q_index = 0;
    std::vector<std::byte> payload;
};

bool read_encoded_frame(
    std::istream& input, EncodedFrame& frame,
    std::size_t max_payload_bytes = kMaxEncodedFramePayloadBytes);
void write_encoded_frame(std::ostream& output, const EncodedFrame& frame);

}  // namespace mlvc
