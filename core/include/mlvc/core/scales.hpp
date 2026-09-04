#pragma once

#include "mlvc/core/model.hpp"
#include "mlvc/core/tensor.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace mlvc {

using ScalePair = std::pair<std::vector<std::int32_t>,
                            std::vector<std::int32_t>>;

ScalePair extract_scales(const Tensor& z_raw, const ModelConfig& config,
                         int scale_max_index);
void extract_scales(TensorView z_raw, const ModelConfig& config,
                    int scale_max_index, std::vector<std::int32_t>& scales_0,
                    std::vector<std::int32_t>& scales_1);

}  // namespace mlvc
