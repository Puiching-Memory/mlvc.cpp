#include "model_assets.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

#ifdef MLVC_HAS_EMBEDDED_MODELS
extern "C" const unsigned char mlvc_embedded_models_start[];
extern "C" const unsigned char mlvc_embedded_models_end[];
#endif

namespace mlvc::detail {
namespace {

constexpr std::string_view kEmbeddedPrefix = "embedded:";

#ifdef MLVC_HAS_EMBEDDED_MODELS
struct EmbeddedIndexEntry {
    std::string_view profile;
    std::string_view path;
    std::size_t offset;
    std::size_t size;
};

constexpr EmbeddedIndexEntry kEmbeddedIndex[] = {
#include "mlvc_embedded_models_index.inc"
};
#endif

std::optional<std::string> embedded_profile(
    const std::filesystem::path& model_dir)
{
    const std::string location = model_dir.generic_string();
    if (!location.starts_with(kEmbeddedPrefix))
        return std::nullopt;
    std::string profile = location.substr(kEmbeddedPrefix.size());
    if (profile.empty() || profile.find('/') != std::string::npos ||
        profile.find('\\') != std::string::npos) {
        throw std::runtime_error("invalid embedded model profile: " + profile);
    }
    return profile;
}

#ifdef MLVC_HAS_EMBEDDED_MODELS
std::span<const std::byte> embedded_data()
{
    const auto begin_address = reinterpret_cast<std::uintptr_t>(
        mlvc_embedded_models_start);
    const auto end_address = reinterpret_cast<std::uintptr_t>(
        mlvc_embedded_models_end);
    if (end_address < begin_address)
        throw std::runtime_error("embedded model section is corrupt");
    return {
        reinterpret_cast<const std::byte*>(begin_address),
        static_cast<std::size_t>(end_address - begin_address)};
}

std::optional<std::span<const std::byte>> find_embedded_asset(
    std::string_view profile, std::string_view path)
{
    const auto data = embedded_data();
    for (const auto& entry : kEmbeddedIndex) {
        if (entry.profile != profile || entry.path != path)
            continue;
        if (entry.offset > data.size() || entry.size > data.size() - entry.offset)
            throw std::runtime_error("embedded model index is corrupt");
        return data.subspan(entry.offset, entry.size);
    }
    return std::nullopt;
}
#endif

std::vector<std::byte> read_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open model asset: " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("cannot size model asset: " + path.string());
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    if (!input)
        throw std::runtime_error("cannot read model asset: " + path.string());
    return result;
}

}  // namespace

bool model_asset_exists(const std::filesystem::path& model_dir,
                        const std::filesystem::path& relative_path)
{
    const auto profile = embedded_profile(model_dir);
    if (!profile)
        return std::filesystem::is_regular_file(model_dir / relative_path);
#ifdef MLVC_HAS_EMBEDDED_MODELS
    return find_embedded_asset(*profile, relative_path.generic_string()).has_value();
#else
    return false;
#endif
}

std::vector<std::byte> read_model_binary(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& relative_path)
{
    const auto profile = embedded_profile(model_dir);
    if (!profile)
        return read_file(model_dir / relative_path);
#ifdef MLVC_HAS_EMBEDDED_MODELS
    const auto asset = find_embedded_asset(
        *profile, relative_path.generic_string());
    if (asset)
        return {asset->begin(), asset->end()};
#endif
    throw std::runtime_error(
        "embedded model asset not found: " + *profile + "/" +
        relative_path.generic_string());
}

std::string read_model_text(const std::filesystem::path& model_dir,
                            const std::filesystem::path& relative_path)
{
    const std::vector<std::byte> data = read_model_binary(model_dir, relative_path);
    return {reinterpret_cast<const char*>(data.data()), data.size()};
}

std::vector<std::string> list_embedded_model_profiles()
{
    std::vector<std::string> result;
#ifdef MLVC_HAS_EMBEDDED_MODELS
    for (const auto& entry : kEmbeddedIndex) {
        if (std::find(result.begin(), result.end(), entry.profile) == result.end())
            result.emplace_back(entry.profile);
    }
#endif
    return result;
}

std::filesystem::path default_model_location()
{
    const std::vector<std::string> profiles = list_embedded_model_profiles();
    if (profiles.empty())
        return {};
    constexpr std::string_view preferred = "mlvc-psnr-v1";
    const auto found = std::find(profiles.begin(), profiles.end(), preferred);
    const std::string& profile = found == profiles.end() ? profiles.front() : *found;
    return std::string(kEmbeddedPrefix) + profile;
}

}  // namespace mlvc::detail
