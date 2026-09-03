#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace mlvc::detail {

bool model_asset_exists(const std::filesystem::path& model_dir,
                        const std::filesystem::path& relative_path);
std::string read_model_text(const std::filesystem::path& model_dir,
                            const std::filesystem::path& relative_path);
std::vector<std::byte> read_model_binary(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& relative_path);
std::vector<std::string> list_embedded_model_profiles();
std::filesystem::path default_model_location();

}  // namespace mlvc::detail
