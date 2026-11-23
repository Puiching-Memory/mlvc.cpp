#include "core/bitstream.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <sstream>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    if (!data || size == 0U || size > (1U << 20))
        return 0;

    const std::span<const std::byte> bytes(
        reinterpret_cast<const std::byte*>(data), size);

    try {
        mlvc::BitstreamDecoder decoder;
        static_cast<void>(decoder.decode(bytes));
    } catch (const std::exception&) {
    }

    try {
        std::string raw(reinterpret_cast<const char*>(data), size);
        std::istringstream stream(raw, std::ios::binary);
        mlvc::AccessUnitReader reader(stream, 1U << 20);
        mlvc::BitstreamDecoder decoder;
        std::vector<std::byte> au;
        for (int i = 0; i < 8 && reader.read(au); ++i)
            static_cast<void>(decoder.decode(au));
    } catch (const std::exception&) {
    }
    return 0;
}
