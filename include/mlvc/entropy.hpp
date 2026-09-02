#pragma once

#include "mlvc/backend.hpp"
#include "mlvc/model.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace mlvc {

class EntropyCodec final {
public:
    EntropyCodec(const std::filesystem::path& model_dir,
                 const ModelConfig& config);
    ~EntropyCodec();

    EntropyCodec(const EntropyCodec&) = delete;
    EntropyCodec& operator=(const EntropyCodec&) = delete;
    EntropyCodec(EntropyCodec&&) noexcept;
    EntropyCodec& operator=(EntropyCodec&&) noexcept;

    std::vector<std::byte> encode(const Tensor& y_raw_1,
                                  const Tensor& y_raw_0,
                                  const Tensor& z_raw, int q_index) const;

    struct DecodedLatents {
        Tensor z_raw;
        Tensor y_raw_0;
        Tensor y_raw_1;
    };
    DecodedLatents decode(const std::vector<std::byte>& payload,
                          int q_index) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mlvc
