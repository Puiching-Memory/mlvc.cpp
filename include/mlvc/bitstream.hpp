#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace mlvc {

struct EncodedFrame {
    std::int32_t q_index = 0;
    std::vector<std::byte> payload;
};

bool read_encoded_frame(std::istream& input, EncodedFrame& frame);
void write_encoded_frame(std::ostream& output, const EncodedFrame& frame);

}  // namespace mlvc
