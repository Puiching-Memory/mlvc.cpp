#pragma once

#include "model_assets.hpp"
#include "mlvc_model.pb.h"

#include <limits>
#include <stdexcept>

namespace mlvc::detail {

template <typename Message>
Message read_model_proto(const std::filesystem::path& directory,
                         const std::filesystem::path& filename)
{
    const auto bytes = read_model_binary(directory, filename);
    constexpr std::size_t limit = 64U * 1024U * 1024U;
    if (bytes.empty() || bytes.size() > limit)
        throw std::runtime_error("invalid Protobuf model message size: " + filename.string());
    Message result;
    if (!result.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
        throw std::runtime_error("invalid Protobuf model message: " + filename.string());
    return result;
}

}  // namespace mlvc::detail
