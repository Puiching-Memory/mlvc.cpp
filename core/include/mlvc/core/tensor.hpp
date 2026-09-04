#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace mlvc {

enum class TensorDataType { kFloat16, kInt32 };

// IEEE-754 binary16 is stored as raw 16-bit data at every backend boundary.
using Float16Storage = std::uint16_t;
using TensorStorage = std::variant<
    std::vector<Float16Storage>, std::vector<std::int32_t>>;

struct Tensor {
    std::string name;
    std::vector<std::int64_t> shape;
    TensorStorage data;

    TensorDataType data_type() const noexcept
    {
        return std::holds_alternative<std::vector<Float16Storage>>(data)
            ? TensorDataType::kFloat16 : TensorDataType::kInt32;
    }

    std::size_t element_count() const noexcept
    {
        return std::visit([](const auto& values) { return values.size(); }, data);
    }

    std::size_t byte_size() const noexcept
    {
        return std::visit([](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            return values.size() * sizeof(Value);
        }, data);
    }

    const void* raw_data() const noexcept
    {
        return std::visit([](const auto& values) -> const void* {
            return values.data();
        }, data);
    }
};

}  // namespace mlvc
