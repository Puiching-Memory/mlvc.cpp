// FP16-only backend microbenchmark and output parity checker.

#include "mlvc/backend.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

struct CliOptions {
    mlvc::BackendOptions backend;
    std::filesystem::path case_path;
    std::filesystem::path result_path;
    std::filesystem::path save_outputs_dir;
    std::size_t warmup_iterations = 10;
    std::size_t iterations = 100;
    std::optional<double> max_abs_error;
    std::optional<double> max_rmse;
    bool print_backend_name = false;
};

struct BenchmarkCase {
    std::string case_id;
    std::string model_name;
    std::vector<mlvc::Tensor> inputs;
    std::vector<mlvc::Tensor> expected_outputs;
};

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s --model-dir DIR --case FILE [options]\n"
        "options:\n"
        "  --device-id N             CUDA device index (default: 0)\n"
        "  --warmup N                warm-up iterations (default: 10)\n"
        "  --iterations N            measured iterations (default: 100)\n"
        "  --engine-cache-dir DIR    TensorRT engine cache directory\n"
        "  --workspace-mib N         TensorRT workspace MiB (default: 4096)\n"
        "  --result FILE             write result JSON instead of stdout only\n"
        "  --save-outputs DIR        save final output tensors as raw files\n"
        "  --max-abs-error VALUE     fail if an FP16 output exceeds this error\n"
        "  --max-rmse VALUE          fail if an FP16 output exceeds this RMSE\n"
        "  --backend-name            print the compiled backend and exit\n",
        argv0);
}

template <typename T>
T parse_unsigned(const char* text, std::string_view option, bool allow_zero)
{
    try {
        std::size_t parsed = 0;
        const unsigned long long value = std::stoull(text, &parsed);
        if (text[parsed] == '\0' && (allow_zero || value > 0) &&
            value <= static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
            return static_cast<T>(value);
        }
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(option) +
                             (allow_zero ? " must be a non-negative integer"
                                         : " must be a positive integer"));
}

double parse_tolerance(const char* text, std::string_view option)
{
    try {
        std::size_t parsed = 0;
        const double value = std::stod(text, &parsed);
        if (text[parsed] == '\0' && std::isfinite(value) && value >= 0.0)
            return value;
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(option) +
                             " must be a finite non-negative number");
}

CliOptions parse_cli(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        auto value = [&](std::string_view option) -> const char* {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(option) + " requires a value");
            return argv[++i];
        };

        if (argument == "--model-dir") {
            options.backend.model_dir = value(argument);
        } else if (argument == "--case") {
            options.case_path = value(argument);
        } else if (argument == "--device-id") {
            options.backend.device_id = parse_unsigned<int>(value(argument), argument, true);
        } else if (argument == "--warmup") {
            options.warmup_iterations =
                parse_unsigned<std::size_t>(value(argument), argument, true);
        } else if (argument == "--iterations") {
            options.iterations = parse_unsigned<std::size_t>(value(argument), argument, false);
        } else if (argument == "--engine-cache-dir") {
            options.backend.engine_cache_dir = value(argument);
        } else if (argument == "--workspace-mib") {
            const std::size_t mib =
                parse_unsigned<std::size_t>(value(argument), argument, false);
            if (mib > (std::numeric_limits<std::size_t>::max() >> 20))
                throw std::runtime_error("--workspace-mib is too large");
            options.backend.workspace_size = mib << 20;
        } else if (argument == "--result") {
            options.result_path = value(argument);
        } else if (argument == "--save-outputs") {
            options.save_outputs_dir = value(argument);
        } else if (argument == "--max-abs-error") {
            options.max_abs_error = parse_tolerance(value(argument), argument);
        } else if (argument == "--max-rmse") {
            options.max_rmse = parse_tolerance(value(argument), argument);
        } else if (argument == "--backend-name") {
            options.print_backend_name = true;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }

    if (!options.print_backend_name) {
        if (options.backend.model_dir.empty())
            throw std::runtime_error("--model-dir is required");
        if (options.case_path.empty())
            throw std::runtime_error("--case is required");
    }
    return options;
}

std::size_t checked_element_count(const std::vector<std::int64_t>& shape)
{
    std::size_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0)
            throw std::runtime_error("tensor dimensions must be positive");
        const auto value = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / value)
            throw std::runtime_error("tensor element count overflows size_t");
        count *= value;
    }
    return count;
}

std::vector<std::byte> read_binary(const std::filesystem::path& path,
                                   std::size_t expected_bytes)
{
    std::error_code error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
    if (error)
        throw std::runtime_error("cannot stat tensor file: " + path.string());
    if (file_bytes != expected_bytes) {
        throw std::runtime_error("tensor file size mismatch for " + path.string() +
                                 ": expected " + std::to_string(expected_bytes) +
                                 ", got " + std::to_string(file_bytes));
    }

    std::vector<std::byte> bytes(expected_bytes);
    std::ifstream input(path, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(bytes.data()),
                              static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read tensor file: " + path.string());
    }
    return bytes;
}

mlvc::Tensor load_tensor(const Json& spec, const std::filesystem::path& base_dir)
{
    mlvc::Tensor tensor;
    tensor.name = spec.at("name").get<std::string>();
    tensor.shape = spec.at("shape").get<std::vector<std::int64_t>>();
    const std::size_t count = checked_element_count(tensor.shape);
    const std::string dtype = spec.at("dtype").get<std::string>();
    const std::filesystem::path file = base_dir / spec.at("file").get<std::string>();

    if (dtype == "fp16") {
        const auto bytes = read_binary(file, count * sizeof(mlvc::Float16Storage));
        std::vector<mlvc::Float16Storage> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        tensor.data = std::move(values);
    } else if (dtype == "int32") {
        const auto bytes = read_binary(file, count * sizeof(std::int32_t));
        std::vector<std::int32_t> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        tensor.data = std::move(values);
    } else {
        throw std::runtime_error("unsupported tensor dtype '" + dtype +
                                 "'; only fp16 and int32 are accepted");
    }
    return tensor;
}

BenchmarkCase load_case(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open benchmark case: " + path.string());

    Json document;
    input >> document;
    if (document.value("schema_version", 0) != 1)
        throw std::runtime_error("benchmark case schema_version must be 1");
    if (document.value("precision", std::string{}) != "fp16")
        throw std::runtime_error("benchmark case precision must be fp16");

    BenchmarkCase result;
    result.case_id = document.value("case_id", std::string{});
    result.model_name = document.at("model").get<std::string>();
    if (result.model_name.empty())
        throw std::runtime_error("benchmark case model must not be empty");

    const std::filesystem::path base_dir = path.parent_path();
    for (const Json& spec : document.at("inputs"))
        result.inputs.push_back(load_tensor(spec, base_dir));
    if (result.inputs.empty())
        throw std::runtime_error("benchmark case must contain at least one input");

    if (document.contains("expected_outputs")) {
        for (const Json& spec : document.at("expected_outputs"))
            result.expected_outputs.push_back(load_tensor(spec, base_dir));
    }
    return result;
}

float half_to_float(mlvc::Float16Storage half) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fU;
    std::uint32_t mantissa = half & 0x03ffU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign |
                (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

std::string dtype_name(mlvc::TensorDataType type)
{
    return type == mlvc::TensorDataType::kFloat16 ? "fp16" : "int32";
}

std::string utc_timestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    if (gmtime_r(&now, &utc) == nullptr)
        throw std::runtime_error("failed to obtain UTC timestamp");
    char buffer[sizeof("2026-09-02T00:00:00Z")];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0)
        throw std::runtime_error("failed to format UTC timestamp");
    return buffer;
}

Json compare_fp16(const std::vector<mlvc::Float16Storage>& actual,
                  const std::vector<mlvc::Float16Storage>& expected,
                  const CliOptions& options, bool& pass)
{
    constexpr double relative_error_epsilon = 1e-6;
    std::size_t bit_mismatches = 0;
    std::size_t numerical_mismatches = 0;
    std::size_t signed_zero_mismatches = 0;
    std::size_t non_finite_values = 0;
    std::size_t non_finite_mismatches = 0;
    std::size_t finite_values = 0;
    double max_abs_error = 0.0;
    double max_relative_error = 0.0;
    double absolute_error = 0.0;
    double squared_error = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i])
            ++bit_mismatches;
        const double a = half_to_float(actual[i]);
        const double e = half_to_float(expected[i]);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++non_finite_values;
            if (actual[i] != expected[i]) {
                ++non_finite_mismatches;
                ++numerical_mismatches;
            }
            continue;
        }

        if (a != e) {
            ++numerical_mismatches;
        } else if (actual[i] != expected[i] && a == 0.0) {
            ++signed_zero_mismatches;
        }

        const double difference = std::abs(a - e);
        max_abs_error = std::max(max_abs_error, difference);
        max_relative_error = std::max(
            max_relative_error,
            difference / std::max(std::abs(e), relative_error_epsilon));
        absolute_error += difference;
        squared_error += difference * difference;
        dot += a * e;
        actual_norm += a * a;
        expected_norm += e * e;
        ++finite_values;
    }

    const double rmse = finite_values == 0
        ? 0.0
        : std::sqrt(squared_error / static_cast<double>(finite_values));
    const double mean_abs_error = finite_values == 0
        ? 0.0
        : absolute_error / static_cast<double>(finite_values);
    double cosine_similarity = 0.0;
    if (actual_norm == 0.0 && expected_norm == 0.0) {
        cosine_similarity = 1.0;
    } else if (actual_norm > 0.0 && expected_norm > 0.0) {
        cosine_similarity = dot / std::sqrt(actual_norm * expected_norm);
    }

    const bool thresholds_configured = options.max_abs_error || options.max_rmse;
    bool within_tolerance = non_finite_mismatches == 0;
    if (options.max_abs_error)
        within_tolerance = within_tolerance && max_abs_error <= *options.max_abs_error;
    if (options.max_rmse)
        within_tolerance = within_tolerance && rmse <= *options.max_rmse;
    if (thresholds_configured)
        pass = pass && within_tolerance;

    return Json{
        {"bit_exact", bit_mismatches == 0},
        {"numerically_exact", numerical_mismatches == 0},
        {"bit_mismatched_elements", bit_mismatches},
        {"numerically_mismatched_elements", numerical_mismatches},
        {"signed_zero_mismatches", signed_zero_mismatches},
        {"max_abs_error", max_abs_error},
        {"mean_abs_error", mean_abs_error},
        {"max_relative_error", max_relative_error},
        {"relative_error_epsilon", relative_error_epsilon},
        {"rmse", rmse},
        {"cosine_similarity", cosine_similarity},
        {"non_finite_values", non_finite_values},
        {"non_finite_mismatches", non_finite_mismatches},
        {"within_tolerance", thresholds_configured ? Json(within_tolerance) : Json(nullptr)},
    };
}

Json compare_outputs(const std::vector<mlvc::Tensor>& actual,
                     const std::vector<mlvc::Tensor>& expected,
                     const CliOptions& options, bool& pass)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error("output count mismatch: expected " +
                                 std::to_string(expected.size()) + ", got " +
                                 std::to_string(actual.size()));
    }

    Json comparisons = Json::array();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const mlvc::Tensor& a = actual[i];
        const mlvc::Tensor& e = expected[i];
        if (a.shape != e.shape)
            throw std::runtime_error("output shape mismatch at index " + std::to_string(i));
        if (a.data_type() != e.data_type())
            throw std::runtime_error("output dtype mismatch at index " + std::to_string(i));
        if (!a.name.empty() && !e.name.empty() && a.name != e.name) {
            throw std::runtime_error("output name mismatch at index " + std::to_string(i) +
                                     ": expected " + e.name + ", got " + a.name);
        }

        Json item{
            {"index", i},
            {"name", e.name.empty() ? a.name : e.name},
            {"dtype", dtype_name(a.data_type())},
            {"shape", a.shape},
            {"elements", a.element_count()},
        };
        if (a.data_type() == mlvc::TensorDataType::kFloat16) {
            item["comparison"] = compare_fp16(
                std::get<std::vector<mlvc::Float16Storage>>(a.data),
                std::get<std::vector<mlvc::Float16Storage>>(e.data), options, pass);
        } else {
            const bool exact = std::get<std::vector<std::int32_t>>(a.data) ==
                               std::get<std::vector<std::int32_t>>(e.data);
            pass = pass && exact;
            item["comparison"] = Json{
                {"bit_exact", exact},
                {"numerically_exact", exact},
                {"within_tolerance", exact},
            };
        }
        comparisons.push_back(std::move(item));
    }
    return comparisons;
}

double percentile(const std::vector<double>& sorted_values, double quantile)
{
    const double position = quantile * static_cast<double>(sorted_values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper)
        return sorted_values[lower];
    const double weight = position - static_cast<double>(lower);
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

Json latency_statistics(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());
    return Json{
        {"min", values.front()},
        {"mean", mean},
        {"p50", percentile(values, 0.50)},
        {"p95", percentile(values, 0.95)},
        {"p99", percentile(values, 0.99)},
        {"max", values.back()},
    };
}

std::string safe_file_name(std::string name)
{
    for (char& character : name) {
        const bool safe = (character >= 'a' && character <= 'z') ||
                          (character >= 'A' && character <= 'Z') ||
                          (character >= '0' && character <= '9') ||
                          character == '-' || character == '_';
        if (!safe)
            character = '_';
    }
    return name.empty() ? "unnamed" : name;
}

void save_outputs(const std::filesystem::path& directory,
                  const std::string& model_name,
                  const std::vector<mlvc::Tensor>& outputs)
{
    std::filesystem::create_directories(directory);
    Json manifest{
        {"schema_version", 1},
        {"precision", "fp16"},
        {"model", model_name},
        {"outputs", Json::array()},
    };

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const mlvc::Tensor& output = outputs[i];
        const std::string file_name = std::to_string(i) + "-" +
            safe_file_name(output.name) + "." + dtype_name(output.data_type()) + ".raw";
        const std::filesystem::path path = directory / file_name;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
            throw std::runtime_error("cannot write output tensor: " + path.string());
        file.write(static_cast<const char*>(output.raw_data()),
                   static_cast<std::streamsize>(output.byte_size()));
        if (!file)
            throw std::runtime_error("cannot write output tensor: " + path.string());
        manifest["outputs"].push_back(Json{
            {"index", i},
            {"name", output.name},
            {"dtype", dtype_name(output.data_type())},
            {"shape", output.shape},
            {"file", file_name},
        });
    }

    std::ofstream manifest_file(directory / "outputs.json");
    if (!manifest_file)
        throw std::runtime_error("cannot write output manifest");
    manifest_file << manifest.dump(2) << '\n';
}

void write_result(const Json& result, const std::filesystem::path& path)
{
    if (path.empty())
        return;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot write result JSON: " + path.string());
    output << result.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.print_backend_name) {
            std::printf("%.*s\n", static_cast<int>(mlvc::compiled_backend_name().size()),
                        mlvc::compiled_backend_name().data());
            return 0;
        }

        const BenchmarkCase benchmark_case = load_case(options.case_path);

        const auto create_start = Clock::now();
        std::unique_ptr<mlvc::InferenceBackend> backend =
            mlvc::create_backend(options.backend);
        const auto create_end = Clock::now();

        const auto load_start = Clock::now();
        backend->load(benchmark_case.model_name, {});
        const auto load_end = Clock::now();

        std::vector<mlvc::Tensor> outputs;
        for (std::size_t i = 0; i < options.warmup_iterations; ++i)
            outputs = backend->run(benchmark_case.inputs);

        std::vector<double> latency_ms;
        latency_ms.reserve(options.iterations);
        for (std::size_t i = 0; i < options.iterations; ++i) {
            const auto start = Clock::now();
            outputs = backend->run(benchmark_case.inputs);
            const auto end = Clock::now();
            latency_ms.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
        }

        Json latency = latency_statistics(latency_ms);
        Json result{
            {"schema_version", 1},
            {"timestamp_utc", utc_timestamp()},
            {"case_id", benchmark_case.case_id.empty()
                ? Json(nullptr) : Json(benchmark_case.case_id)},
            {"backend", std::string(backend->name())},
            {"precision", "fp16"},
            {"model", benchmark_case.model_name},
            {"device_id", options.backend.device_id},
            {"warmup_iterations", options.warmup_iterations},
            {"iterations", options.iterations},
            {"backend_create_ms", std::chrono::duration<double, std::milli>(
                                      create_end - create_start).count()},
            {"model_load_ms", std::chrono::duration<double, std::milli>(
                                  load_end - load_start).count()},
            {"latency_ms", latency},
            {"throughput_inferences_per_second", 1000.0 / latency.at("mean").get<double>()},
            {"input_bytes_per_inference", std::accumulate(
                benchmark_case.inputs.begin(), benchmark_case.inputs.end(), std::size_t{0},
                [](std::size_t total, const mlvc::Tensor& tensor) {
                    return total + tensor.byte_size();
                })},
            {"output_bytes_per_inference", std::accumulate(
                outputs.begin(), outputs.end(), std::size_t{0},
                [](std::size_t total, const mlvc::Tensor& tensor) {
                    return total + tensor.byte_size();
                })},
            {"timing_scope", "host_input_to_host_output"},
            {"implementation", "host-roundtrip-v1"},
        };

        bool comparison_pass = true;
        const bool thresholds_configured = options.max_abs_error || options.max_rmse;
        if (!benchmark_case.expected_outputs.empty()) {
            result["reference"] = Json{
                {"thresholds_configured", thresholds_configured},
                {"max_abs_error", options.max_abs_error
                    ? Json(*options.max_abs_error) : Json(nullptr)},
                {"max_rmse", options.max_rmse ? Json(*options.max_rmse) : Json(nullptr)},
                {"outputs", compare_outputs(outputs, benchmark_case.expected_outputs,
                                             options, comparison_pass)},
                {"pass", thresholds_configured ? Json(comparison_pass) : Json(nullptr)},
            };
        } else {
            result["reference"] = nullptr;
        }

        if (!options.save_outputs_dir.empty())
            save_outputs(options.save_outputs_dir, benchmark_case.model_name, outputs);
        write_result(result, options.result_path);
        std::printf("%s\n", result.dump(2).c_str());
        return !thresholds_configured || comparison_pass ? 0 : 3;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
