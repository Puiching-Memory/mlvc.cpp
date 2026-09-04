#include "mlvc/core/bitstream.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace mlvc {
namespace {

std::uint32_t load_u32_le(const std::byte* data)
{
    return std::to_integer<std::uint32_t>(data[0]) |
        (std::to_integer<std::uint32_t>(data[1]) << 8) |
        (std::to_integer<std::uint32_t>(data[2]) << 16) |
        (std::to_integer<std::uint32_t>(data[3]) << 24);
}

void store_u32_le(std::byte* data, std::uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i)
        data[i] = static_cast<std::byte>((value >> (i * 8)) & 0xffU);
}

}  // namespace

bool read_encoded_frame(std::istream& input, EncodedFrame& frame)
{
    std::array<std::byte, 8> header{};
    input.read(reinterpret_cast<char*>(header.data()), header.size());
    if (input.gcount() == 0)
        return false;
    if (input.gcount() != static_cast<std::streamsize>(header.size()))
        throw std::runtime_error("truncated MLVC frame header");

    frame.q_index = static_cast<std::int32_t>(load_u32_le(header.data()));
    const std::uint32_t payload_size = load_u32_le(header.data() + 4);
    frame.payload.resize(payload_size);
    input.read(reinterpret_cast<char*>(frame.payload.data()), payload_size);
    if (input.gcount() != static_cast<std::streamsize>(payload_size))
        throw std::runtime_error("truncated MLVC frame payload");
    return true;
}

void write_encoded_frame(std::ostream& output, const EncodedFrame& frame)
{
    if (frame.payload.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("MLVC frame payload is too large");
    std::array<std::byte, 8> header{};
    store_u32_le(header.data(), static_cast<std::uint32_t>(frame.q_index));
    store_u32_le(header.data() + 4,
                 static_cast<std::uint32_t>(frame.payload.size()));
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    output.write(reinterpret_cast<const char*>(frame.payload.data()),
                 static_cast<std::streamsize>(frame.payload.size()));
    if (!output)
        throw std::runtime_error("failed to write MLVC frame");
}

}  // namespace mlvc
