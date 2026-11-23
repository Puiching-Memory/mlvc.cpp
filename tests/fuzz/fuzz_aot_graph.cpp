#include "runtime/aot_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    if (!data || size == 0U || size > (1U << 20))
        return 0;

    try {
        const std::span<const std::byte> bytes(
            reinterpret_cast<const std::byte*>(data), size);
        static_cast<void>(mlvc::aot::decode_graph_binary(bytes));
    } catch (const std::exception&) {
    }
    return 0;
}
