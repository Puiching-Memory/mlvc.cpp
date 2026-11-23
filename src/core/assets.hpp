#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlvc {

// Read-only view of a model asset's bytes. May borrow static memory (e.g.
// embedded .rodata), hold a read-only memory mapping, or own a heap buffer.
class AssetView final {
public:
    AssetView() = default;

    static AssetView borrowed(std::span<const std::byte> data) noexcept
    {
        return AssetView(data, nullptr);
    }

    static AssetView owned(std::vector<std::byte> bytes)
    {
        auto holder = std::make_shared<std::vector<std::byte>>(std::move(bytes));
        const std::span<const std::byte> view(holder->data(), holder->size());
        return AssetView(view, std::move(holder));
    }

    static AssetView mapped(const std::byte* data, std::size_t size,
                            std::shared_ptr<const void> owner) noexcept
    {
        return AssetView(std::span<const std::byte>(data, size), std::move(owner));
    }

    std::span<const std::byte> span() const noexcept { return data_; }
    const std::byte* data() const noexcept { return data_.data(); }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    operator std::span<const std::byte>() const noexcept { return data_; }

private:
    AssetView(std::span<const std::byte> data, std::shared_ptr<const void> owner) noexcept
        : data_(data), owner_(std::move(owner))
    {
    }

    std::span<const std::byte> data_{};
    std::shared_ptr<const void> owner_{};
};

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

    // Optional zero-copy read path. Defaults to wrapping read_binary() so
    // custom subclasses do not have to override it.
    virtual AssetView read_view(const std::filesystem::path& relative_path) const
    {
        return AssetView::owned(read_binary(relative_path));
    }
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
