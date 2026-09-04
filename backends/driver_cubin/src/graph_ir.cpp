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

std::size_t dtype_bytes(const std::string& dtype)
{
    if (dtype == "fp16") return 2;
    if (dtype == "int32") return 4;
    if (dtype == "int64") return 8;
    throw std::runtime_error("driver-cubin: unsupported AOT dtype " + dtype);
}

TensorDataType public_dtype(const std::string& dtype)
{
    if (dtype == "fp16") return TensorDataType::kFloat16;
    if (dtype == "int32") return TensorDataType::kInt32;
    throw std::runtime_error("driver-cubin: unsupported graph I/O dtype " + dtype);
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

const Value& AotGraph::value(const std::string& name) const
{
    const auto found = values_.find(name);
    if (found == values_.end() || found->second.address == 0)
        throw std::runtime_error("driver-cubin: unresolved tensor " + name);
    return found->second;
}

float AotGraph::scalar_fp16(const std::string& name) const
{
    const auto values = initializer_values<Float16Storage>(name);
    if (values.size() != 1)
        throw std::runtime_error("driver-cubin: Clip bound must be scalar");
    return half_to_float(values[0]);
}

}  // namespace mlvc::driver_cubin_backend
