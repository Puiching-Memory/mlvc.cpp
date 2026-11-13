#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mlvc {

// Abstract read-only source of canonical MLVC bundle files (metadata.pb,
// model_bundle.pb, PMF tables, and AOT graph/weights artifacts). Enables
// directory bundles, ELF-embedded images, or platform-specific readers such as
// Android AAssetManager/in-memory buffers without duplicating model parsers.
class ModelAssetReader {
public:
    virtual ~ModelAssetReader() = default;

    virtual std::string location_label() const = 0;
    virtual bool exists(const std::filesystem::path& relative_path) const = 0;
    virtual std::vector<std::byte> read_binary(
        const std::filesystem::path& relative_path) const = 0;
};

std::shared_ptr<const ModelAssetReader> make_directory_asset_reader(
    std::filesystem::path directory);

std::shared_ptr<const ModelAssetReader> make_embedded_asset_reader(
    std::string_view profile);

std::shared_ptr<const ModelAssetReader> make_model_asset_reader(
    const std::filesystem::path& location);

std::shared_ptr<const ModelAssetReader> resolve_model_asset_reader(
    const std::filesystem::path& model_dir,
    std::shared_ptr<const ModelAssetReader> override_reader = nullptr);

std::vector<std::string> list_embedded_model_profiles();
std::filesystem::path default_model_location();

}  // namespace mlvc
