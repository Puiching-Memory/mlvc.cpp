#pragma once

#include "mlvc/backend.hpp"
#include "mlvc/model.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace mlvc {

using ScalePair = std::pair<std::vector<std::int32_t>,
                            std::vector<std::int32_t>>;

ScalePair extract_scales(const Tensor& z_raw, const ModelConfig& config,
                         int scale_max_index);

}  // namespace mlvc
