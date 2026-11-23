#include "aot_graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mlvc::driver_cubin_backend {

std::size_t element_count(const std::vector<int64_t>& shape)
{
    return aot::checked_element_count(shape);
}

std::size_t dtype_bytes(DataType dtype)
{
    return aot::dtype_byte_size(dtype);
}

TensorDataType public_dtype(DataType dtype)
{
    return aot::to_public_dtype(dtype);
}

std::array<int, 4> dims4(const std::vector<int64_t>& shape)
{
    return aot::pad_dims_to_4d(shape);
}

unsigned int divide_up(std::size_t value, unsigned int divisor)
{
    return aot::divide_up(value, divisor);
}

const Value& AotGraph::value(TensorId name) const
{
    if (name == 0 || name >= values_.size() || values_[name].address == 0)
        throw std::runtime_error("driver-cubin: unresolved tensor ID " + std::to_string(name));
    return values_[name];
}

TensorId AotGraph::tensor_id(std::string_view name) const
{
    for (std::size_t i = 1; i < values_.size(); ++i)
        if (values_[i].name == name)
            return static_cast<TensorId>(i);
    throw std::runtime_error("driver-cubin: named codec tensor not found: " + std::string(name));
}

float AotGraph::scalar_fp16(TensorId name) const
{
    const auto values = initializer_values<Float16Storage>(name);
    if (values.size() != 1)
        throw std::runtime_error("driver-cubin: Clip bound must be scalar");
    return half_to_float(values[0]);
}

}  // namespace mlvc::driver_cubin_backend
