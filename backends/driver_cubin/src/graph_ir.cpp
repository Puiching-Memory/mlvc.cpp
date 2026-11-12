#include "aot_graph.hpp"

#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlvc::driver_cubin_backend {

std::size_t element_count(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           [](std::size_t value, int64_t dimension) {
                               if (dimension <= 0 ||
                                   value > std::numeric_limits<std::size_t>::max() /
                                               static_cast<std::size_t>(dimension))
                                   throw std::runtime_error("driver-cubin: invalid tensor shape");
                               return value * static_cast<std::size_t>(dimension);
                           });
}

std::size_t dtype_bytes(DataType dtype)
{
    if (dtype == DataType::FP16) return 2;
    if (dtype == DataType::INT32) return 4;
    if (dtype == DataType::INT64) return 8;
    throw std::runtime_error("driver-cubin: unsupported AOT dtype " + std::to_string(static_cast<int>(dtype)));
}

TensorDataType public_dtype(DataType dtype)
{
    if (dtype == DataType::FP16) return TensorDataType::kFloat16;
    if (dtype == DataType::INT32) return TensorDataType::kInt32;
    throw std::runtime_error("driver-cubin: unsupported graph I/O dtype " + std::to_string(static_cast<int>(dtype)));
}

std::array<int, 4> dims4(const std::vector<int64_t>& shape)
{
    if (shape.size() > 4)
        throw std::runtime_error("driver-cubin: tensors with rank > 4 are unsupported");
    std::array<int, 4> result{1, 1, 1, 1};
    const std::size_t offset = 4 - shape.size();
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0 || shape[i] > std::numeric_limits<int>::max())
            throw std::runtime_error("driver-cubin: invalid tensor dimension");
        result[offset + i] = static_cast<int>(shape[i]);
    }
    return result;
}

unsigned int divide_up(std::size_t value, unsigned int divisor)
{
    return static_cast<unsigned int>((value + divisor - 1) / divisor);
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
