// Fixed-shape ONNX graph interpreter backed by embedded CUDA Driver kernels.

#include "mlvc/backend.hpp"
#include "mlvc/driver/cutlass.hpp"
#include "mlvc/driver/driver.hpp"
#include "mlvc/half.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" const unsigned char mlvc_driver_kernels_fatbin[];
extern "C" const std::size_t mlvc_driver_kernels_fatbin_size;

namespace mlvc {
namespace {

using json = nlohmann::json;
using DeviceAddress = driver::abi::DeviceAddress;

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

std::vector<std::byte> read_binary(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("driver-cubin: cannot open " + path.string());
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("driver-cubin: cannot size " + path.string());
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size()));
    if (!input)
        throw std::runtime_error("driver-cubin: cannot read " + path.string());
    return result;
}

json read_json(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("driver-cubin: cannot open " + path.string());
    json result;
    input >> result;
    return result;
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

struct Value {
    std::string dtype;
    std::vector<int64_t> shape;
    DeviceAddress address = 0;
    std::size_t host_offset = 0;
    bool initializer = false;
};

class AotGraph final {
public:
    AotGraph(const std::filesystem::path& model_dir, const std::string& model_name,
             driver::Driver& driver, const driver::Module& module)
        : driver_(driver), module_(module)
    {
        const std::filesystem::path graph_dir = model_dir / "aot" / model_name;
        manifest_ = read_json(graph_dir / "graph.json");
        if (manifest_.at("schema_version").get<int>() != 1)
            throw std::runtime_error("driver-cubin: unsupported AOT graph schema");
        weights_host_ = read_binary(graph_dir / "weights.bin");
        if (weights_host_.size() != manifest_.at("weights_bytes").get<std::size_t>())
            throw std::runtime_error("driver-cubin: weights size does not match graph");

        weights_device_ = driver_.allocate(std::max<std::size_t>(weights_host_.size(), 1));
        if (!weights_host_.empty())
            driver_.upload_async(weights_device_, weights_host_.data(), weights_host_.size());
        arena_device_ = driver_.allocate(manifest_.at("arena_bytes").get<std::size_t>());

        for (const auto& item : manifest_.at("weights")) {
            Value value;
            value.dtype = item.at("dtype").get<std::string>();
            value.shape = item.at("shape").get<std::vector<int64_t>>();
            value.host_offset = item.at("offset").get<std::size_t>();
            value.address = weights_device_.address() + value.host_offset;
            value.initializer = true;
            values_.emplace(item.at("name").get<std::string>(), std::move(value));
        }

        for (auto it = manifest_.at("tensors").begin();
             it != manifest_.at("tensors").end(); ++it) {
            Value value;
            value.dtype = it.value().at("dtype").get<std::string>();
            value.shape = it.value().at("shape").get<std::vector<int64_t>>();
            if (manifest_.at("arena").contains(it.key())) {
                value.address = arena_device_.address() +
                    manifest_.at("arena").at(it.key()).at("offset").get<std::size_t>();
            }
            values_.emplace(it.key(), std::move(value));
        }

        for (const auto& input : manifest_.at("inputs")) {
            const std::string name = input.at("name").get<std::string>();
            Value& value = values_.at(name);
            input_names_.push_back(name);
            input_buffers_.push_back(driver_.allocate(
                element_count(value.shape) * dtype_bytes(value.dtype)));
            value.address = input_buffers_.back().address();
        }

        for (const auto& node : manifest_.at("nodes")) {
            if (node.at("op") != "Slice")
                continue;
            const auto inputs =
                node.at("inputs").get<std::vector<std::string>>();
            const auto outputs =
                node.at("outputs").get<std::vector<std::string>>();
            if (inputs.empty() || outputs.size() != 1 ||
                std::find(input_names_.begin(), input_names_.end(), inputs[0]) ==
                    input_names_.end()) {
                continue;
            }
            const Value& input = value(inputs[0]);
            Value& output = values_.at(outputs[0]);
            if (input.dtype != output.dtype || input.shape.size() != 4 ||
                output.shape.size() != 4 || input.shape[0] != 1 ||
                output.shape[0] != 1 || input.shape[2] != output.shape[2] ||
                input.shape[3] != output.shape[3] ||
                output.shape[1] > input.shape[1]) {
                continue;
            }
            const int output_channels = static_cast<int>(output.shape[1]);
            for (int start = 0;
                 start + output_channels <= static_cast<int>(input.shape[1]);
                 ++start) {
                if (!is_channel_slice(node, start, start + output_channels))
                    continue;
                const std::size_t channel_bytes =
                    static_cast<std::size_t>(input.shape[2] * input.shape[3]) *
                    dtype_bytes(input.dtype);
                output.address = input.address +
                    static_cast<std::size_t>(start) * channel_bytes;
                aliased_input_slices_.push_back(
                    node.at("index").get<std::size_t>());
                break;
            }
        }

        if (driver_.device_info().compute_major >= 8) {
            struct EpilogueInterval {
                std::string name;
                std::size_t birth;
                std::size_t death;
                std::size_t bytes;
                std::size_t slot;
            };
            std::vector<EpilogueInterval> intervals;
            const auto& nodes = manifest_.at("nodes");
            for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
                const auto& convolution = nodes.at(index);
                const auto& epilogue = nodes.at(index + 1);
                const std::string epilogue_op =
                    epilogue.at("op").get<std::string>();
                if (convolution.at("op") != "Conv" ||
                    (epilogue_op != "Add" && epilogue_op != "LeakyRelu")) {
                    continue;
                }
                const auto convolution_outputs = convolution.at("outputs")
                    .get<std::vector<std::string>>();
                const auto epilogue_inputs = epilogue.at("inputs")
                    .get<std::vector<std::string>>();
                const auto epilogue_outputs = epilogue.at("outputs")
                    .get<std::vector<std::string>>();
                if (convolution_outputs.size() != 1 ||
                    epilogue_outputs.size() != 1 ||
                    std::find(epilogue_inputs.begin(), epilogue_inputs.end(),
                              convolution_outputs[0]) == epilogue_inputs.end() ||
                    !manifest_.at("arena").contains(epilogue_outputs[0])) {
                    continue;
                }
                const auto& allocation =
                    manifest_.at("arena").at(epilogue_outputs[0]);
                intervals.push_back({
                    epilogue_outputs[0],
                    allocation.at("birth").get<std::size_t>(),
                    allocation.at("death").get<std::size_t>(),
                    allocation.at("bytes").get<std::size_t>(), 0});
            }

            std::vector<std::size_t> slot_deaths;
            std::vector<std::size_t> slot_bytes;
            for (EpilogueInterval& interval : intervals) {
                std::size_t slot = 0;
                while (slot < slot_deaths.size() &&
                       slot_deaths[slot] >= interval.birth) {
                    ++slot;
                }
                if (slot == slot_deaths.size()) {
                    slot_deaths.push_back(interval.death);
                    slot_bytes.push_back(interval.bytes);
                } else {
                    slot_deaths[slot] = interval.death;
                    slot_bytes[slot] = std::max(slot_bytes[slot], interval.bytes);
                }
                interval.slot = slot;
            }
            epilogue_buffers_.reserve(slot_bytes.size());
            for (std::size_t bytes : slot_bytes)
                epilogue_buffers_.push_back(driver_.allocate(bytes));
            for (const EpilogueInterval& interval : intervals)
                values_.at(interval.name).address =
                    epilogue_buffers_.at(interval.slot).address();
        }

        std::size_t reglu_bytes = 0;
        const auto& nodes = manifest_.at("nodes");
        for (std::size_t index = 0; index + 3 < nodes.size(); ++index) {
            if (nodes.at(index).at("op") == "Slice" &&
                nodes.at(index + 1).at("op") == "Slice" &&
                nodes.at(index + 2).at("op") == "Clip" &&
                nodes.at(index + 3).at("op") == "Mul") {
                const auto outputs = nodes.at(index + 3).at("outputs")
                    .get<std::vector<std::string>>();
                if (outputs.size() == 1) {
                    const Value& output = value(outputs[0]);
                    reglu_bytes = std::max(
                        reglu_bytes,
                        element_count(output.shape) * dtype_bytes(output.dtype));
                }
            }
        }
        if (reglu_bytes != 0)
            reglu_buffer_ = driver_.allocate(reglu_bytes);

        if (driver_.device_info().compute_major >= 8) {
            std::size_t spatial_input_bytes = 0;
            std::size_t spatial_output_bytes = 0;
            for (const auto& node : nodes) {
                if (node.at("op") != "Conv")
                    continue;
                const auto inputs =
                    node.at("inputs").get<std::vector<std::string>>();
                const auto outputs =
                    node.at("outputs").get<std::vector<std::string>>();
                const auto& attributes = node.at("attributes");
                if (inputs.size() < 3 || inputs[2].empty() ||
                    outputs.size() != 1 || attributes.value("group", 1) != 1) {
                    continue;
                }
                const Value& input = value(inputs[0]);
                const Value& weight = value(inputs[1]);
                const Value& output = value(outputs[0]);
                if (input.dtype != "fp16" || weight.dtype != "fp16" ||
                    output.dtype != "fp16" || input.shape.size() != 4 ||
                    weight.shape.size() != 4 || output.shape.size() != 4 ||
                    input.shape[0] != 1 || weight.shape[2] * weight.shape[3] == 1 ||
                    input.shape[1] % 8 != 0 || output.shape[1] % 8 != 0) {
                    continue;
                }
                spatial_input_bytes = std::max(
                    spatial_input_bytes,
                    element_count(input.shape) * dtype_bytes(input.dtype));
                spatial_output_bytes = std::max(
                    spatial_output_bytes,
                    element_count(output.shape) * dtype_bytes(output.dtype));
            }
            if (spatial_input_bytes != 0 && spatial_output_bytes != 0) {
                spatial_input_buffer_ = driver_.allocate(spatial_input_bytes);
                spatial_output_buffer_ = driver_.allocate(spatial_output_bytes);
            }
        }

        for (const auto& output : manifest_.at("outputs"))
            output_names_.push_back(output.at("name").get<std::string>());

        concat_buffers_.reserve(nodes.size());
        for (auto iterator = nodes.rbegin(); iterator != nodes.rend(); ++iterator) {
            const json& node = *iterator;
            if (node.at("op") != "Concat" ||
                node.at("attributes").at("axis").get<int>() != 1) {
                continue;
            }
            const auto inputs =
                node.at("inputs").get<std::vector<std::string>>();
            const auto outputs =
                node.at("outputs").get<std::vector<std::string>>();
            if (outputs.size() != 1)
                continue;
            Value& output = values_.at(outputs[0]);
            if (output.dtype != "fp16" || output.shape.size() != 4 ||
                output.shape[0] != 1) {
                continue;
            }

            std::vector<bool> redirect(inputs.size(), false);
            bool has_redirect = false;
            for (std::size_t input_index = 0; input_index < inputs.size();
                 ++input_index) {
                const std::string& input_name = inputs[input_index];
                const Value& input = value(input_name);
                if (input.initializer || input.dtype != output.dtype ||
                    input.shape.size() != 4 || input.shape[0] != 1 ||
                    input.shape[2] != output.shape[2] ||
                    input.shape[3] != output.shape[3] ||
                    std::find(input_names_.begin(), input_names_.end(),
                              input_name) != input_names_.end() ||
                    std::find(output_names_.begin(), output_names_.end(),
                              input_name) != output_names_.end()) {
                    continue;
                }
                std::size_t consumers = 0;
                bool produced = false;
                for (const auto& candidate : nodes) {
                    const auto candidate_inputs = candidate.at("inputs")
                        .get<std::vector<std::string>>();
                    consumers += static_cast<std::size_t>(std::count(
                        candidate_inputs.begin(), candidate_inputs.end(),
                        input_name));
                    const auto candidate_outputs = candidate.at("outputs")
                        .get<std::vector<std::string>>();
                    produced = produced || std::find(
                        candidate_outputs.begin(), candidate_outputs.end(),
                        input_name) != candidate_outputs.end();
                }
                redirect[input_index] = produced && consumers == 1;
                has_redirect = has_redirect || redirect[input_index];
            }
            if (!has_redirect)
                continue;

            concat_buffers_.push_back(driver_.allocate(
                element_count(output.shape) * dtype_bytes(output.dtype)));
            output.address = concat_buffers_.back().address();
            std::size_t channel_offset = 0;
            const std::size_t channel_bytes =
                static_cast<std::size_t>(output.shape[2] * output.shape[3]) *
                dtype_bytes(output.dtype);
            for (std::size_t input_index = 0; input_index < inputs.size();
                 ++input_index) {
                Value& input = values_.at(inputs[input_index]);
                if (redirect[input_index]) {
                    input.address = output.address + channel_offset * channel_bytes;
                    direct_concat_values_.push_back(inputs[input_index]);
                }
                channel_offset += static_cast<std::size_t>(input.shape[1]);
            }
        }

        binary_ = module_.function("mlvc_binary_fp16");
        binary_contiguous_ = module_.function("mlvc_binary_contiguous_fp16");
        unary_ = module_.function("mlvc_unary_fp16");
        reglu_ = module_.function("mlvc_reglu_fp16");
        reglu_vec8_ = module_.function("mlvc_reglu_vec8_fp16");
        convolution_ = module_.function("mlvc_conv_fp16");
        pointwise_convolution_ = module_.function("mlvc_pointwise_conv_fp16");
        pointwise_convolution_wide_ =
            module_.function("mlvc_pointwise_conv_wide_fp16");
        pointwise_convolution_balanced_ =
            module_.function("mlvc_pointwise_conv_balanced_fp16");
        pointwise_convolution_mma_ =
            module_.function("mlvc_pointwise_conv_mma_fp16");
        pointwise_convolution_mma_small_ =
            module_.function("mlvc_pointwise_conv_mma_small_fp16");
        pointwise_reglu_mma_ = module_.function("mlvc_pointwise_reglu_mma_fp16");
        pointwise_reglu_mma_small_ =
            module_.function("mlvc_pointwise_reglu_mma_small_fp16");
        if (driver_.device_info().compute_major >= 8) {
            cutlass_pointwise_init_ =
                module_.function("mlvc_cutlass_pointwise_init_fp16");
            cutlass_pointwise_ =
                module_.function("mlvc_cutlass_pointwise_fp16");
            cutlass_pointwise_leaky_relu_init_ = module_.function(
                "mlvc_cutlass_pointwise_leaky_relu_init_fp16");
            cutlass_pointwise_leaky_relu_ = module_.function(
                "mlvc_cutlass_pointwise_leaky_relu_fp16");
            cutlass_pointwise_residual_init_ = module_.function(
                "mlvc_cutlass_pointwise_residual_init_fp16");
            cutlass_pointwise_residual_ = module_.function(
                "mlvc_cutlass_pointwise_residual_fp16");
            cutlass_pointwise_medium_init_ = module_.function(
                "mlvc_cutlass_pointwise_medium_init_fp16");
            cutlass_pointwise_medium_ = module_.function(
                "mlvc_cutlass_pointwise_medium_fp16");
            cutlass_pointwise_medium_stage4_init_ = module_.function(
                "mlvc_cutlass_pointwise_medium_stage4_init_fp16");
            cutlass_pointwise_medium_stage4_ = module_.function(
                "mlvc_cutlass_pointwise_medium_stage4_fp16");
            driver_.set_max_dynamic_shared_memory(
                cutlass_pointwise_medium_stage4_, 65536U);
            cutlass_pointwise_medium_leaky_relu_init_ = module_.function(
                "mlvc_cutlass_pointwise_medium_leaky_relu_init_fp16");
            cutlass_pointwise_medium_leaky_relu_ = module_.function(
                "mlvc_cutlass_pointwise_medium_leaky_relu_fp16");
            cutlass_pointwise_medium_residual_init_ = module_.function(
                "mlvc_cutlass_pointwise_medium_residual_init_fp16");
            cutlass_pointwise_medium_residual_ = module_.function(
                "mlvc_cutlass_pointwise_medium_residual_fp16");
            cutlass_pointwise_spatial_wide_residual_init_ = module_.function(
                "mlvc_cutlass_pointwise_spatial_wide_residual_init_fp16");
            cutlass_pointwise_spatial_wide_residual_ = module_.function(
                "mlvc_cutlass_pointwise_spatial_wide_residual_fp16");
            driver_.set_max_dynamic_shared_memory(
                cutlass_pointwise_spatial_wide_residual_, 61440U);
            cutlass_spatial_convolution_init_ =
                module_.function("mlvc_cutlass_spatial_conv_init_fp16");
            cutlass_spatial_convolution_ =
                module_.function("mlvc_cutlass_spatial_conv_fp16");
            driver_.set_max_dynamic_shared_memory(
                cutlass_spatial_convolution_, 73728U);
            transpose_ = module_.function("mlvc_transpose_fp16");
            oihw_to_krsc_ = module_.function("mlvc_oihw_to_krsc_fp16");
        }
        spatial_convolution_mma_ =
            module_.function("mlvc_spatial_conv_mma_fp16");
        spatial_convolution_mma_wide_ =
            module_.function("mlvc_spatial_conv_mma_wide_fp16");
        spatial_convolution_ = module_.function("mlvc_spatial_conv_fp16");
        depthwise_convolution_ = module_.function("mlvc_depthwise_conv_fp16");
        depthwise_convolution_pair_ =
            module_.function("mlvc_depthwise_conv_pair_fp16");
        depthwise_convolution_quad_ =
            module_.function("mlvc_depthwise_conv_quad_fp16");
        gather_ = module_.function("mlvc_gather_axis0_fp16");
        slice_ = module_.function("mlvc_slice_fp16");
        concat_ = module_.function("mlvc_concat_copy_fp16");
        depth_to_space_ = module_.function("mlvc_depth_to_space_fp16");
        space_to_depth_ = module_.function("mlvc_space_to_depth_fp16");
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs)
    {
        if (inputs.size() != input_names_.size())
            throw std::runtime_error("driver-cubin: graph input count mismatch");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Value& expected = value(input_names_[i]);
            if (inputs[i].shape != expected.shape ||
                inputs[i].data_type() != public_dtype(expected.dtype))
                throw std::runtime_error("driver-cubin: graph input shape or dtype mismatch");
            driver_.upload_async(input_buffers_[i], inputs[i].raw_data(),
                                 inputs[i].byte_size());
        }

        if (!cutlass_parameters_ready_ &&
            driver_.device_info().compute_major >= 8) {
            execute_schedule();
            cutlass_parameters_ready_ = true;
        } else if (!executable_graph_) {
            driver_.begin_capture();
            execute_schedule();
            executable_graph_ = driver_.end_capture();
            driver_.launch_graph(executable_graph_);
        } else {
            driver_.launch_graph(executable_graph_);
        }

        std::vector<Tensor> outputs;
        outputs.reserve(output_names_.size());
        for (const std::string& name : output_names_) {
            const Value& result = value(name);
            Tensor tensor;
            tensor.name = name;
            tensor.shape = result.shape;
            if (result.dtype == "fp16") {
                tensor.data = std::vector<Float16Storage>(element_count(result.shape));
            } else if (result.dtype == "int32") {
                tensor.data = std::vector<std::int32_t>(element_count(result.shape));
            } else {
                throw std::runtime_error("driver-cubin: unsupported graph output dtype");
            }
            std::visit([&](auto& storage) {
                driver_.download_async(storage.data(), result.address,
                                       storage.size() * sizeof(typename std::decay_t<decltype(storage)>::value_type));
            }, tensor.data);
            outputs.push_back(std::move(tensor));
        }
        driver_.synchronize();
        return outputs;
    }

private:
    const Value& value(const std::string& name) const
    {
        const auto found = values_.find(name);
        if (found == values_.end() || found->second.address == 0)
            throw std::runtime_error("driver-cubin: unresolved tensor " + name);
        return found->second;
    }

    template <typename T>
    std::vector<T> initializer_values(const std::string& name) const
    {
        const Value& item = value(name);
        if (!item.initializer || item.dtype != (sizeof(T) == 8 ? "int64" : "fp16"))
            throw std::runtime_error("driver-cubin: unexpected initializer type " + name);
        const std::size_t count = element_count(item.shape);
        std::vector<T> result(count);
        std::memcpy(result.data(), weights_host_.data() + item.host_offset,
                    result.size() * sizeof(T));
        return result;
    }

    float scalar_fp16(const std::string& name) const
    {
        const auto values = initializer_values<Float16Storage>(name);
        if (values.size() != 1)
            throw std::runtime_error("driver-cubin: Clip bound must be scalar");
        return half_to_float(values[0]);
    }

    bool is_channel_slice(const json& node, int start, int end) const
    {
        const auto inputs = node.at("inputs").get<std::vector<std::string>>();
        if (inputs.size() < 4 || inputs[1].empty() || inputs[2].empty() ||
            inputs[3].empty()) {
            return false;
        }
        const auto starts = initializer_values<std::int64_t>(inputs[1]);
        const auto ends = initializer_values<std::int64_t>(inputs[2]);
        const auto axes = initializer_values<std::int64_t>(inputs[3]);
        const auto steps = inputs.size() > 4 && !inputs[4].empty()
            ? initializer_values<std::int64_t>(inputs[4])
            : std::vector<std::int64_t>{1};
        return starts.size() == 1 && ends.size() == 1 && axes.size() == 1 &&
               steps.size() == 1 && starts[0] == start && ends[0] == end &&
               (axes[0] == 1 || axes[0] == -3) && steps[0] == 1;
    }

    static bool ranges_overlap(const Value& lhs, const Value& rhs)
    {
        const std::size_t lhs_bytes =
            element_count(lhs.shape) * dtype_bytes(lhs.dtype);
        const std::size_t rhs_bytes =
            element_count(rhs.shape) * dtype_bytes(rhs.dtype);
        return lhs.address < rhs.address + rhs_bytes &&
               rhs.address < lhs.address + lhs_bytes;
    }

    void execute_schedule()
    {
        const auto& nodes = manifest_.at("nodes");
        for (std::size_t index = 0; index < nodes.size();) {
            if (try_execute_pointwise_reglu(nodes, index))
                index += 5;
            else if (try_execute_pointwise_epilogue(nodes, index))
                index += 2;
            else if (try_execute_reglu(nodes, index))
                index += 4;
            else
                execute(nodes.at(index++));
        }
    }

    bool launch_cutlass_spatial(
        const json& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress output, int batch_count,
        int in_channels, int input_height, int input_width, int out_channels,
        int output_height, int output_width, int kernel_height,
        int kernel_width, int stride_height, int stride_width,
        int pad_height, int pad_width)
    {
        if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
            bias == 0 || !spatial_input_buffer_ || !spatial_output_buffer_ ||
            in_channels % 8 != 0 || out_channels % 8 != 0 ||
            kernel_height <= 1 || kernel_width <= 1) {
            return false;
        }

        const std::size_t node_index = node.at("index").get<std::size_t>();
        auto [params, params_inserted] =
            cutlass_spatial_parameter_buffers_.try_emplace(node_index);
        auto [transformed_weight, weight_inserted] =
            cutlass_spatial_weight_buffers_.try_emplace(node_index);
        if (params_inserted != weight_inserted)
            throw std::runtime_error("driver-cubin: inconsistent spatial CUTLASS state");

        DeviceAddress input_nhwc = spatial_input_buffer_.address();
        DeviceAddress output_nhwc = spatial_output_buffer_.address();
        if (params_inserted) {
            const std::size_t weight_elements =
                static_cast<std::size_t>(out_channels) * in_channels *
                kernel_height * kernel_width;
            params->second = driver_.allocate(
                driver::kCutlassPointwiseParamsStorageBytes);
            transformed_weight->second =
                driver_.allocate(weight_elements * sizeof(Float16Storage));
            DeviceAddress weight_krsc = transformed_weight->second.address();
            void* transform_parameters[] = {
                &weight, &weight_krsc, &out_channels, &in_channels,
                &kernel_height, &kernel_width};
            driver_.launch(
                oihw_to_krsc_, {divide_up(weight_elements, 256), 1, 1},
                {256, 1, 1}, 0, transform_parameters);

            DeviceAddress params_storage = params->second.address();
            void* init_parameters[] = {
                &params_storage, &input_nhwc, &weight_krsc, &bias, &output_nhwc,
                &batch_count, &in_channels, &input_height, &input_width,
                &out_channels, &output_height, &output_width, &kernel_height,
                &kernel_width, &stride_height, &stride_width, &pad_height,
                &pad_width};
            driver_.launch(
                cutlass_spatial_convolution_init_, {1, 1, 1}, {1, 1, 1}, 0,
                init_parameters);
            auto [host_params, host_params_inserted] =
                cutlass_spatial_host_parameters_.try_emplace(node_index);
            if (!host_params_inserted)
                throw std::runtime_error(
                    "driver-cubin: duplicate spatial CUTLASS host state");
            driver_.download_async(
                host_params->second.data(), params->second,
                host_params->second.size());
            driver_.synchronize();
        }

        int input_rows = in_channels;
        int input_columns = input_height * input_width;
        void* input_transform_parameters[] = {
            &input, &input_nhwc, &input_rows, &input_columns};
        driver_.launch(
            transpose_, {divide_up(input_columns, 32), divide_up(input_rows, 32), 1},
            {32, 8, 1}, 0, input_transform_parameters);

        auto host_params = cutlass_spatial_host_parameters_.find(node_index);
        if (host_params == cutlass_spatial_host_parameters_.end())
            throw std::runtime_error(
                "driver-cubin: missing spatial CUTLASS host parameters");
        void* convolution_parameters[] = {host_params->second.data()};
        const int output_spatial = output_height * output_width;
        driver_.launch(
            cutlass_spatial_convolution_,
            {divide_up(output_spatial, 128), divide_up(out_channels, 256), 1},
            {256, 1, 1}, 73728U, convolution_parameters);

        int output_rows = output_spatial;
        int output_columns = out_channels;
        void* output_transform_parameters[] = {
            &output_nhwc, &output, &output_rows, &output_columns};
        driver_.launch(
            transpose_,
            {divide_up(output_columns, 32), divide_up(output_rows, 32), 1},
            {32, 8, 1}, 0, output_transform_parameters);
        return true;
    }

    bool launch_cutlass_pointwise(
        const json& node, DeviceAddress input, DeviceAddress weight,
        DeviceAddress bias, DeviceAddress residual, DeviceAddress output,
        int batch_count, int in_channels, int spatial_count, int out_channels,
        int epilogue, float epilogue_alpha)
    {
        if (driver_.device_info().compute_major < 8 || batch_count != 1 ||
            bias == 0 ||
            spatial_count < 920 || (spatial_count % 8) != 0 ||
            (in_channels % 8) != 0 || (out_channels % 8) != 0 ||
            epilogue < 0 || epilogue > 2 ||
            (epilogue == 1 &&
             std::abs(epilogue_alpha - 0.01F) > 1.0e-7F) ||
            (epilogue == 2 && residual == 0)) {
            return false;
        }

        const bool use_medium_tile = in_channels <= 384 &&
            ((spatial_count >= 3000 && out_channels >= 256) ||
             (spatial_count < 3000 && out_channels >= 384 &&
              in_channels >= 256));
        const bool use_spatial_wide_tile =
            epilogue == 2 && spatial_count >= 3000 && in_channels >= 512;
        const bool use_medium_stage4_tile =
            epilogue == 0 && use_medium_tile && spatial_count >= 3000;
        driver::abi::Function init = use_medium_stage4_tile
            ? cutlass_pointwise_medium_stage4_init_
            : use_medium_tile ? cutlass_pointwise_medium_init_
                              : cutlass_pointwise_init_;
        driver::abi::Function function = use_medium_stage4_tile
            ? cutlass_pointwise_medium_stage4_
            : use_medium_tile ? cutlass_pointwise_medium_
                              : cutlass_pointwise_;
        if (epilogue == 1) {
            init = use_medium_tile
                ? cutlass_pointwise_medium_leaky_relu_init_
                : cutlass_pointwise_leaky_relu_init_;
            function = use_medium_tile
                ? cutlass_pointwise_medium_leaky_relu_
                : cutlass_pointwise_leaky_relu_;
        } else if (epilogue == 2) {
            if (use_spatial_wide_tile) {
                init = cutlass_pointwise_spatial_wide_residual_init_;
                function = cutlass_pointwise_spatial_wide_residual_;
            } else {
                init = use_medium_tile
                    ? cutlass_pointwise_medium_residual_init_
                    : cutlass_pointwise_residual_init_;
                function = use_medium_tile
                    ? cutlass_pointwise_medium_residual_
                    : cutlass_pointwise_residual_;
            }
        }

        const std::size_t node_index = node.at("index").get<std::size_t>();
        auto [found, inserted] = cutlass_parameter_buffers_.try_emplace(node_index);
        if (inserted) {
            found->second = driver_.allocate(
                driver::kCutlassPointwiseParamsStorageBytes);
            DeviceAddress params_storage = found->second.address();
            void* init_parameters[] = {
                &params_storage, &input, &weight, &bias, &residual, &output,
                &out_channels, &spatial_count, &in_channels};
            driver_.launch(init, {1, 1, 1}, {1, 1, 1}, 0, init_parameters);
            auto [host_params, host_params_inserted] =
                cutlass_host_parameters_.try_emplace(node_index);
            if (!host_params_inserted)
                throw std::runtime_error(
                    "driver-cubin: duplicate CUTLASS host state");
            driver_.download_async(
                host_params->second.data(), found->second,
                host_params->second.size());
            driver_.synchronize();
        }

        auto host_params = cutlass_host_parameters_.find(node_index);
        if (host_params == cutlass_host_parameters_.end())
            throw std::runtime_error(
                "driver-cubin: missing CUTLASS host parameters");
        void* parameters[] = {host_params->second.data()};
        driver_.launch(
            function,
            {divide_up(out_channels, use_spatial_wide_tile ? 64 : 128),
             divide_up(spatial_count,
                       use_spatial_wide_tile ? 256 : use_medium_tile ? 128 : 64),
             1},
            {128, 1, 1},
            use_spatial_wide_tile ? 61440U
                : use_medium_stage4_tile ? 65536U
                : use_medium_tile ? 49152U : 36864U,
            parameters);
        return true;
    }

    bool try_execute_pointwise_epilogue(const json& nodes, std::size_t index)
    {
        if (driver_.device_info().compute_major < 8 || index + 1 >= nodes.size())
            return false;
        const json& convolution = nodes.at(index);
        const json& epilogue_node = nodes.at(index + 1);
        const std::string epilogue_op =
            epilogue_node.at("op").get<std::string>();
        if (convolution.at("op") != "Conv" ||
            (epilogue_op != "LeakyRelu" && epilogue_op != "Add")) {
            return false;
        }

        const auto convolution_inputs =
            convolution.at("inputs").get<std::vector<std::string>>();
        const auto convolution_outputs =
            convolution.at("outputs").get<std::vector<std::string>>();
        const auto epilogue_inputs =
            epilogue_node.at("inputs").get<std::vector<std::string>>();
        const auto epilogue_outputs =
            epilogue_node.at("outputs").get<std::vector<std::string>>();
        if (convolution_inputs.size() < 2 || convolution_outputs.size() != 1 ||
            epilogue_outputs.size() != 1 ||
            std::find(output_names_.begin(), output_names_.end(),
                      convolution_outputs[0]) != output_names_.end()) {
            return false;
        }

        const std::string& convolution_output_name = convolution_outputs[0];
        std::size_t consumer_count = 0;
        for (const auto& candidate : nodes) {
            for (const std::string& input :
                 candidate.at("inputs").get<std::vector<std::string>>()) {
                consumer_count += input == convolution_output_name ? 1 : 0;
            }
        }
        if (consumer_count != 1)
            return false;

        DeviceAddress epilogue_input_address = 0;
        int epilogue = 1;
        float alpha = epilogue_node.at("attributes").value("alpha", 0.01F);
        const Value* residual = nullptr;
        if (epilogue_op == "LeakyRelu") {
            if (epilogue_inputs != std::vector<std::string>{convolution_output_name})
                return false;
        } else {
            if (epilogue_inputs.size() != 2)
                return false;
            const auto found = std::find(epilogue_inputs.begin(),
                                         epilogue_inputs.end(),
                                         convolution_output_name);
            if (found == epilogue_inputs.end())
                return false;
            const std::string& residual_name =
                epilogue_inputs[found == epilogue_inputs.begin() ? 1 : 0];
            residual = &value(residual_name);
            epilogue_input_address = residual->address;
            epilogue = 2;
        }

        const Value& input = value(convolution_inputs[0]);
        const Value& weight = value(convolution_inputs[1]);
        const Value& convolution_output = value(convolution_output_name);
        const Value& output = value(epilogue_outputs[0]);
        const auto& attributes = convolution.at("attributes");
        const auto strides = attributes.at("strides").get<std::vector<int>>();
        const auto pads = attributes.at("pads").get<std::vector<int>>();
        if (input.dtype != "fp16" || weight.dtype != "fp16" ||
            output.dtype != "fp16" || input.shape.size() != 4 ||
            weight.shape.size() != 4 || output.shape.size() != 4 ||
            attributes.value("group", 1) != 1 || weight.shape[2] != 1 ||
            weight.shape[3] != 1 || strides != std::vector<int>{1, 1} ||
            pads != std::vector<int>{0, 0, 0, 0} ||
            input.shape[0] != output.shape[0] ||
            input.shape[1] != weight.shape[1] ||
            input.shape[2] != output.shape[2] ||
            input.shape[3] != output.shape[3] ||
            output.shape[1] != weight.shape[0] ||
            convolution_output.shape != output.shape ||
            input.shape[1] % 16 != 0 || output.shape[1] % 128 != 0 ||
            (residual && residual->shape != output.shape) ||
            ranges_overlap(output, input) ||
            (residual && ranges_overlap(output, *residual))) {
            return false;
        }

        execute(convolution, &output, epilogue_input_address, epilogue, alpha);
        return true;
    }

    bool try_execute_reglu(const json& nodes, std::size_t index)
    {
        if (index + 3 >= nodes.size())
            return false;
        const json& first_slice = nodes.at(index);
        const json& second_slice = nodes.at(index + 1);
        const json& clip = nodes.at(index + 2);
        const json& multiply = nodes.at(index + 3);
        if (first_slice.at("op") != "Slice" || second_slice.at("op") != "Slice" ||
            clip.at("op") != "Clip" || multiply.at("op") != "Mul") {
            return false;
        }

        const auto first_inputs =
            first_slice.at("inputs").get<std::vector<std::string>>();
        const auto second_inputs =
            second_slice.at("inputs").get<std::vector<std::string>>();
        const auto first_outputs =
            first_slice.at("outputs").get<std::vector<std::string>>();
        const auto second_outputs =
            second_slice.at("outputs").get<std::vector<std::string>>();
        const auto clip_inputs = clip.at("inputs").get<std::vector<std::string>>();
        const auto clip_outputs = clip.at("outputs").get<std::vector<std::string>>();
        const auto multiply_inputs =
            multiply.at("inputs").get<std::vector<std::string>>();
        const auto multiply_outputs =
            multiply.at("outputs").get<std::vector<std::string>>();
        if (first_inputs.empty() || second_inputs.empty() ||
            first_inputs[0] != second_inputs[0] || first_outputs.size() != 1 ||
            second_outputs.size() != 1 || clip_inputs.empty() ||
            clip_outputs.size() != 1 || multiply_inputs.size() != 2 ||
            multiply_outputs.size() != 1 || clip_inputs[0] != first_outputs[0] ||
            multiply_inputs[0] != clip_outputs[0] ||
            multiply_inputs[1] != second_outputs[0]) {
            return false;
        }

        const Value& input = value(first_inputs[0]);
        Value& output = values_.at(multiply_outputs[0]);
        if (input.dtype != "fp16" || output.dtype != "fp16" ||
            input.shape.size() != 4 || output.shape.size() != 4 ||
            input.shape[0] != output.shape[0] ||
            input.shape[1] != output.shape[1] * 2 ||
            input.shape[2] != output.shape[2] ||
            input.shape[3] != output.shape[3] ||
            value(first_outputs[0]).shape != output.shape ||
            value(second_outputs[0]).shape != output.shape ||
            value(clip_outputs[0]).shape != output.shape) {
            return false;
        }

        int channels = static_cast<int>(output.shape[1]);
        if (!is_channel_slice(first_slice, 0, channels) ||
            !is_channel_slice(second_slice, channels, channels * 2)) {
            return false;
        }

        float minimum = -std::numeric_limits<float>::infinity();
        float maximum = std::numeric_limits<float>::infinity();
        if (clip_inputs.size() > 1 && !clip_inputs[1].empty())
            minimum = scalar_fp16(clip_inputs[1]);
        if (clip_inputs.size() > 2 && !clip_inputs[2].empty())
            maximum = scalar_fp16(clip_inputs[2]);
        DeviceAddress input_address = input.address;
        if (std::find(direct_concat_values_.begin(),
                      direct_concat_values_.end(), multiply_outputs[0]) ==
            direct_concat_values_.end()) {
            output.address = reglu_buffer_.address();
        }
        DeviceAddress output_address = output.address;
        int batch_count = static_cast<int>(output.shape[0]);
        int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
        int work_count = spatial_count % 2 == 0 ? spatial_count / 2 : spatial_count;
        void* parameters[] = {
            &input_address, &output_address, &batch_count, &channels,
            &spatial_count, &minimum, &maximum};
        if (spatial_count % 8 == 0) {
            launch_linear(
                reglu_vec8_,
                static_cast<std::size_t>(batch_count) * channels *
                    (spatial_count / 8),
                parameters);
        } else {
            driver_.launch(reglu_, {divide_up(work_count, 256),
                                    static_cast<unsigned int>(channels),
                                    static_cast<unsigned int>(batch_count)},
                           {256, 1, 1}, 0, parameters);
        }
        return true;
    }

    bool try_execute_pointwise_reglu(const json& nodes, std::size_t index)
    {
        if (driver_.device_info().compute_major < 8 || index + 4 >= nodes.size())
            return false;
        const json& convolution = nodes.at(index);
        const json& first_slice = nodes.at(index + 1);
        const json& second_slice = nodes.at(index + 2);
        const json& clip = nodes.at(index + 3);
        const json& multiply = nodes.at(index + 4);
        if (convolution.at("op") != "Conv" || first_slice.at("op") != "Slice" ||
            second_slice.at("op") != "Slice" || clip.at("op") != "Clip" ||
            multiply.at("op") != "Mul") {
            return false;
        }

        const auto convolution_inputs =
            convolution.at("inputs").get<std::vector<std::string>>();
        const auto convolution_outputs =
            convolution.at("outputs").get<std::vector<std::string>>();
        const auto first_inputs =
            first_slice.at("inputs").get<std::vector<std::string>>();
        const auto second_inputs =
            second_slice.at("inputs").get<std::vector<std::string>>();
        const auto first_outputs =
            first_slice.at("outputs").get<std::vector<std::string>>();
        const auto second_outputs =
            second_slice.at("outputs").get<std::vector<std::string>>();
        const auto clip_inputs = clip.at("inputs").get<std::vector<std::string>>();
        const auto clip_outputs = clip.at("outputs").get<std::vector<std::string>>();
        const auto multiply_inputs =
            multiply.at("inputs").get<std::vector<std::string>>();
        const auto multiply_outputs =
            multiply.at("outputs").get<std::vector<std::string>>();
        if (convolution_inputs.size() < 2 || convolution_outputs.size() != 1 ||
            first_inputs.empty() || second_inputs.empty() ||
            first_inputs[0] != convolution_outputs[0] ||
            second_inputs[0] != convolution_outputs[0] ||
            first_outputs.size() != 1 || second_outputs.size() != 1 ||
            clip_inputs.empty() || clip_outputs.size() != 1 ||
            multiply_inputs.size() != 2 || multiply_outputs.size() != 1 ||
            clip_inputs[0] != first_outputs[0] ||
            multiply_inputs[0] != clip_outputs[0] ||
            multiply_inputs[1] != second_outputs[0]) {
            return false;
        }

        const Value& input = value(convolution_inputs[0]);
        const Value& weight = value(convolution_inputs[1]);
        const Value& convolution_output = value(convolution_outputs[0]);
        Value& output = values_.at(multiply_outputs[0]);
        const auto& attributes = convolution.at("attributes");
        const auto strides = attributes.at("strides").get<std::vector<int>>();
        const auto pads = attributes.at("pads").get<std::vector<int>>();
        const int groups = attributes.value("group", 1);
        if (input.dtype != "fp16" || weight.dtype != "fp16" ||
            output.dtype != "fp16" || input.shape.size() != 4 ||
            weight.shape.size() != 4 || output.shape.size() != 4 ||
            convolution_output.shape.size() != 4 || groups != 1 ||
            weight.shape[2] != 1 || weight.shape[3] != 1 ||
            strides != std::vector<int>{1, 1} ||
            pads != std::vector<int>{0, 0, 0, 0} ||
            input.shape[0] != output.shape[0] ||
            input.shape[2] != output.shape[2] ||
            input.shape[3] != output.shape[3] ||
            convolution_output.shape[1] != output.shape[1] * 2 ||
            convolution_output.shape[0] != output.shape[0] ||
            convolution_output.shape[2] != output.shape[2] ||
            convolution_output.shape[3] != output.shape[3] ||
            value(first_outputs[0]).shape != output.shape ||
            value(second_outputs[0]).shape != output.shape ||
            value(clip_outputs[0]).shape != output.shape) {
            return false;
        }

        int gate_channels = static_cast<int>(output.shape[1]);
        int in_channels = static_cast<int>(input.shape[1]);
        if (gate_channels % 64 != 0 || in_channels % 16 != 0 ||
            !is_channel_slice(first_slice, 0, gate_channels) ||
            !is_channel_slice(second_slice, gate_channels, gate_channels * 2)) {
            return false;
        }

        float minimum = -std::numeric_limits<float>::infinity();
        float maximum = std::numeric_limits<float>::infinity();
        if (clip_inputs.size() > 1 && !clip_inputs[1].empty())
            minimum = scalar_fp16(clip_inputs[1]);
        if (clip_inputs.size() > 2 && !clip_inputs[2].empty())
            maximum = scalar_fp16(clip_inputs[2]);
        DeviceAddress input_address = input.address;
        DeviceAddress weight_address = weight.address;
        DeviceAddress bias_address = convolution_inputs.size() > 2 &&
            !convolution_inputs[2].empty()
            ? value(convolution_inputs[2]).address : 0;
        if (std::find(direct_concat_values_.begin(),
                      direct_concat_values_.end(), multiply_outputs[0]) ==
            direct_concat_values_.end()) {
            output.address = reglu_buffer_.address();
        }
        DeviceAddress output_address = output.address;
        int batch_count = static_cast<int>(output.shape[0]);
        int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
        DeviceAddress convolution_output_address = convolution_output.address;
        if (launch_cutlass_pointwise(
                convolution, input_address, weight_address, bias_address, 0,
                convolution_output_address, batch_count, in_channels,
                spatial_count, gate_channels * 2, 0, 0.0F)) {
            const int work_count = (spatial_count & 1) == 0
                ? spatial_count / 2 : spatial_count;
            void* reglu_parameters[] = {
                &convolution_output_address, &output_address, &batch_count,
                &gate_channels, &spatial_count, &minimum, &maximum};
            if (spatial_count % 8 == 0) {
                launch_linear(
                    reglu_vec8_,
                    static_cast<std::size_t>(batch_count) * gate_channels *
                        (spatial_count / 8),
                    reglu_parameters);
            } else {
                driver_.launch(
                    reglu_,
                    {divide_up(work_count, 256),
                     static_cast<unsigned int>(gate_channels),
                     static_cast<unsigned int>(batch_count)},
                    {256, 1, 1}, 0, reglu_parameters);
            }
            return true;
        }

        void* parameters[] = {
            &input_address, &weight_address, &bias_address, &output_address,
            &batch_count, &in_channels, &spatial_count, &gate_channels,
            &minimum, &maximum};
        const unsigned int large_tile_blocks =
            divide_up(spatial_count, 64) * divide_up(gate_channels, 64) *
            batch_count;
        const bool use_small = large_tile_blocks <
            static_cast<unsigned int>(
                driver_.device_info().multiprocessor_count * 2);
        driver_.launch(
            use_small ? pointwise_reglu_mma_small_ : pointwise_reglu_mma_,
            {divide_up(spatial_count, 64),
             divide_up(gate_channels, use_small ? 32 : 64),
             static_cast<unsigned int>(batch_count)},
            {32, 8, 1}, 0, parameters);
        return true;
    }

    void launch_linear(driver::abi::Function function, std::size_t count,
                       std::span<void*> parameters)
    {
        constexpr unsigned int block = 256;
        driver_.launch(function, {divide_up(count, block), 1, 1}, {block, 1, 1},
                       0, parameters);
    }

    void execute(const json& node, const Value* output_override = nullptr,
                 DeviceAddress epilogue_input = 0, int epilogue = 0,
                 float epilogue_alpha = 0.0F)
    {
        const std::string op = node.at("op").get<std::string>();
        const auto inputs = node.at("inputs").get<std::vector<std::string>>();
        const auto outputs = node.at("outputs").get<std::vector<std::string>>();
        if (outputs.size() != 1)
            throw std::runtime_error("driver-cubin: only single-output nodes are supported");
        const Value& output = output_override ? *output_override : value(outputs[0]);
        const std::size_t count_size = element_count(output.shape);
        if (count_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("driver-cubin: tensor is too large");
        int count = static_cast<int>(count_size);

        if (op == "Add" || op == "Mul" || op == "Sub") {
            DeviceAddress lhs = value(inputs.at(0)).address;
            DeviceAddress rhs = value(inputs.at(1)).address;
            DeviceAddress result = output.address;
            int operation = op == "Add" ? 0 : (op == "Mul" ? 1 : 2);
            const auto od = dims4(output.shape);
            const auto ld = dims4(value(inputs[0]).shape);
            const auto rd = dims4(value(inputs[1]).shape);
            void* parameters[] = {
                &lhs, &rhs, &result, &count, &operation,
                const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                const_cast<int*>(&ld[0]), const_cast<int*>(&ld[1]),
                const_cast<int*>(&ld[2]), const_cast<int*>(&ld[3]),
                const_cast<int*>(&rd[0]), const_cast<int*>(&rd[1]),
                const_cast<int*>(&rd[2]), const_cast<int*>(&rd[3]),
            };
            if (value(inputs[0]).shape == output.shape &&
                value(inputs[1]).shape == output.shape) {
                void* contiguous_parameters[] = {
                    &lhs, &rhs, &result, &count, &operation};
                launch_linear(binary_contiguous_, (count_size + 1) / 2,
                              contiguous_parameters);
            } else {
                launch_linear(binary_, count_size, parameters);
            }
            return;
        }

        if (op == "LeakyRelu" || op == "Clip" || op == "Sigmoid" ||
            op == "Reciprocal" || op == "Round") {
            DeviceAddress input = value(inputs.at(0)).address;
            DeviceAddress result = output.address;
            int operation = op == "LeakyRelu" ? 0 : op == "Clip" ? 1 :
                            op == "Sigmoid" ? 2 : op == "Reciprocal" ? 3 : 4;
            float alpha = node.at("attributes").value("alpha", 0.01F);
            float minimum = -std::numeric_limits<float>::infinity();
            float maximum = std::numeric_limits<float>::infinity();
            if (op == "Clip") {
                if (inputs.size() > 1 && !inputs[1].empty()) minimum = scalar_fp16(inputs[1]);
                if (inputs.size() > 2 && !inputs[2].empty()) maximum = scalar_fp16(inputs[2]);
            }
            void* parameters[] = {
                &input, &result, &count, &operation, &alpha, &minimum, &maximum};
            launch_linear(unary_, count_size, parameters);
            return;
        }

        if (op == "Conv") {
            const Value& input_value = value(inputs.at(0));
            const Value& weight_value = value(inputs.at(1));
            const auto& attributes = node.at("attributes");
            const auto strides = attributes.at("strides").get<std::vector<int>>();
            const auto pads = attributes.at("pads").get<std::vector<int>>();
            int groups = attributes.value("group", 1);
            const auto in = dims4(input_value.shape);
            const auto out = dims4(output.shape);
            const auto kernel = weight_value.shape;
            DeviceAddress input = input_value.address;
            DeviceAddress weight = weight_value.address;
            DeviceAddress bias = inputs.size() > 2 && !inputs[2].empty()
                ? value(inputs[2]).address : 0;
            DeviceAddress result = output.address;
            int n = in[0], ic = in[1], ih = in[2], iw = in[3];
            int oc = out[1], oh = out[2], ow = out[3];
            int kh = static_cast<int>(kernel.at(2));
            int kw = static_cast<int>(kernel.at(3));
            int sh = strides.at(0), sw = strides.at(1);
            int ph = pads.at(0), pw = pads.at(1);
            int spatial = oh * ow;
            const bool use_pointwise =
                groups == 1 && kh == 1 && kw == 1 &&
                sh == 1 && sw == 1 && ph == 0 && pw == 0 &&
                ih == oh && iw == ow && ic % 16 == 0 && oc % 64 == 0;
            if (use_pointwise) {
                if (launch_cutlass_pointwise(
                        node, input, weight, bias, epilogue_input, result,
                        n, ic, spatial, oc, epilogue, epilogue_alpha)) {
                    return;
                }
                if (oc % 128 == 0) {
                    const bool use_mma = driver_.device_info().compute_major >= 8;
                    if (use_mma) {
                        void* mma_parameters[] = {
                            &input, &weight, &bias, &result, &n, &ic, &spatial,
                            &oc, &epilogue_input, &epilogue, &epilogue_alpha};
                        const unsigned int large_tile_blocks =
                            divide_up(spatial, 64) * divide_up(oc, 128) * n;
                        const bool use_small_tile = large_tile_blocks <
                            static_cast<unsigned int>(
                                driver_.device_info().multiprocessor_count * 2);
                        driver_.launch(
                            use_small_tile ? pointwise_convolution_mma_small_
                                           : pointwise_convolution_mma_,
                            {divide_up(spatial, 64),
                             divide_up(oc, use_small_tile ? 64 : 128),
                             static_cast<unsigned int>(n)},
                            {32, 8, 1}, 0, mma_parameters);
                    } else {
                        void* pointwise_parameters[] = {
                            &input, &weight, &bias, &result, &n, &ic, &spatial,
                            &oc};
                        driver_.launch(pointwise_convolution_balanced_,
                            {divide_up(spatial, 64), divide_up(oc, 128),
                             static_cast<unsigned int>(n)},
                            {32, 8, 1}, 0, pointwise_parameters);
                    }
                } else {
                    void* pointwise_parameters[] = {
                        &input, &weight, &bias, &result, &n, &ic, &spatial, &oc};
                    driver_.launch(pointwise_convolution_,
                        {divide_up(spatial, 16), divide_up(oc, 64),
                         static_cast<unsigned int>(n)},
                        {32, 4, 1}, 0, pointwise_parameters);
                }
                return;
            }
            const int reduction = ic * kh * kw;
            const bool use_spatial = groups == 1 && reduction % 16 == 0 &&
                                     oc % 64 == 0;
            if (use_spatial) {
                if (launch_cutlass_spatial(
                        node, input, weight, bias, result, n, ic, ih, iw, oc,
                        oh, ow, kh, kw, sh, sw, ph, pw)) {
                    return;
                }
                void* spatial_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                    &oc, &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
                const unsigned int mma_blocks =
                    divide_up(spatial, 16) * divide_up(oc, 128) * n;
                const bool use_mma =
                    driver_.device_info().compute_major >= 8 && mma_blocks >= 1;
                const bool use_wide_mma = use_mma && oc >= 512;
                driver_.launch(
                    use_wide_mma ? spatial_convolution_mma_wide_
                                 : use_mma ? spatial_convolution_mma_
                                           : spatial_convolution_,
                    {divide_up(spatial, use_wide_mma ? 32 : 16),
                     divide_up(oc, use_mma ? 128 : 64),
                     static_cast<unsigned int>(n)},
                    {32, use_mma ? 8U : 4U, 1}, 0, spatial_parameters);
                return;
            }
            const bool use_depthwise = groups == ic && oc == ic &&
                                       weight_value.shape.at(1) == 1;
            if (use_depthwise) {
                void* depthwise_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                    &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
                const bool use_quads = ow % 4 == 0 && kh == 3 && kw == 3 &&
                    sh == 1 && sw == 1 && ph == 1 && pw == 1;
                const bool use_pairs = !use_quads && ow % 2 == 0 && sw == 1;
                const int outputs_per_thread = use_quads ? 4 : use_pairs ? 2 : 1;
                const std::size_t depthwise_count =
                    static_cast<std::size_t>(n) * oc * oh *
                    (ow / outputs_per_thread);
                driver_.launch(use_quads ? depthwise_convolution_quad_
                                         : use_pairs ? depthwise_convolution_pair_
                                                     : depthwise_convolution_,
                    {divide_up(depthwise_count, 256), 1, 1}, {256, 1, 1},
                    0, depthwise_parameters);
                return;
            }
            void* parameters[] = {
                &input, &weight, &bias, &result, &n, &ic, &ih, &iw, &oc, &oh, &ow,
                &kh, &kw, &sh, &sw, &ph, &pw, &groups};
            driver_.launch(convolution_,
                {divide_up(static_cast<std::size_t>(n) * oh * ow, 32),
                 divide_up(oc, 4), 1},
                {32, 4, 1}, 0, parameters);
            return;
        }

        if (op == "Gather") {
            const Value& data = value(inputs.at(0));
            if (node.at("attributes").value("axis", 0) != 0)
                throw std::runtime_error("driver-cubin: only Gather axis 0 is supported");
            DeviceAddress input = data.address;
            DeviceAddress index = value(inputs.at(1)).address;
            DeviceAddress result = output.address;
            int rows = static_cast<int>(data.shape.at(0));
            int row_elements = static_cast<int>(element_count(data.shape) / rows);
            void* parameters[] = {&input, &index, &result, &row_elements, &rows};
            launch_linear(gather_, row_elements, parameters);
            return;
        }

        if (op == "Slice") {
            const std::size_t node_index = node.at("index").get<std::size_t>();
            if (std::find(aliased_input_slices_.begin(),
                          aliased_input_slices_.end(), node_index) !=
                aliased_input_slices_.end()) {
                return;
            }
            const Value& input_value = value(inputs.at(0));
            const auto starts_values = initializer_values<std::int64_t>(inputs.at(1));
            const auto axes_values = inputs.size() > 3 && !inputs[3].empty()
                ? initializer_values<std::int64_t>(inputs[3])
                : std::vector<std::int64_t>{};
            const auto steps_values = inputs.size() > 4 && !inputs[4].empty()
                ? initializer_values<std::int64_t>(inputs[4])
                : std::vector<std::int64_t>{};
            auto id = dims4(input_value.shape);
            const auto od = dims4(output.shape);
            std::array<int, 4> starts{0, 0, 0, 0};
            std::array<int, 4> steps{1, 1, 1, 1};
            for (std::size_t i = 0; i < starts_values.size(); ++i) {
                int axis = axes_values.empty() ? static_cast<int>(i) :
                           static_cast<int>(axes_values.at(i));
                if (axis < 0) axis += static_cast<int>(input_value.shape.size());
                axis += 4 - static_cast<int>(input_value.shape.size());
                int start = static_cast<int>(starts_values[i]);
                if (start < 0) start += id.at(static_cast<std::size_t>(axis));
                starts.at(static_cast<std::size_t>(axis)) = start;
                if (!steps_values.empty())
                    steps.at(static_cast<std::size_t>(axis)) =
                        static_cast<int>(steps_values.at(i));
            }
            DeviceAddress input = input_value.address;
            DeviceAddress result = output.address;
            void* parameters[] = {
                &input, &result, &count,
                const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                &id[0], &id[1], &id[2], &id[3],
                &starts[0], &starts[1], &starts[2], &starts[3],
                &steps[0], &steps[1], &steps[2], &steps[3],
            };
            launch_linear(slice_, count_size, parameters);
            return;
        }

        if (op == "Concat") {
            int axis = node.at("attributes").at("axis").get<int>();
            if (axis < 0) axis += static_cast<int>(output.shape.size());
            axis += 4 - static_cast<int>(output.shape.size());
            const auto od = dims4(output.shape);
            int axis_offset = 0;
            for (const std::string& input_name : inputs) {
                const Value& input_value = value(input_name);
                const auto id = dims4(input_value.shape);
                const std::size_t input_count = element_count(input_value.shape);
                int input_count_parameter = static_cast<int>(input_count);
                DeviceAddress input = input_value.address;
                DeviceAddress result = output.address;
                const std::size_t channel_bytes =
                    static_cast<std::size_t>(od[2]) * od[3] *
                    dtype_bytes(output.dtype);
                const DeviceAddress direct_address = result +
                    static_cast<std::size_t>(axis_offset) * channel_bytes;
                if (axis != 1 || od[0] != 1 || input != direct_address) {
                    void* parameters[] = {
                        &input, &result, &input_count_parameter,
                        const_cast<int*>(&id[0]), const_cast<int*>(&id[1]),
                        const_cast<int*>(&id[2]), const_cast<int*>(&id[3]),
                        const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                        const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                        &axis, &axis_offset,
                    };
                    launch_linear(concat_, input_count, parameters);
                }
                axis_offset += id.at(static_cast<std::size_t>(axis));
            }
            return;
        }

        if (op == "DepthToSpace") {
            const Value& input_value = value(inputs.at(0));
            const auto in = dims4(input_value.shape);
            DeviceAddress input = input_value.address;
            DeviceAddress result = output.address;
            int channels = in[1], height = in[2], width = in[3];
            int block = node.at("attributes").at("blocksize").get<int>();
            int mode = node.at("attributes").value("mode", std::string("DCR")) == "DCR"
                ? 0 : 1;
            void* parameters[] = {
                &input, &result, &count, &channels, &height, &width, &block, &mode};
            launch_linear(depth_to_space_, count_size, parameters);
            return;
        }

        if (op == "SpaceToDepth") {
            const Value& input_value = value(inputs.at(0));
            const auto in = dims4(input_value.shape);
            DeviceAddress input = input_value.address;
            DeviceAddress result = output.address;
            int channels = in[1], height = in[2], width = in[3];
            int block = node.at("attributes").at("blocksize").get<int>();
            void* parameters[] = {
                &input, &result, &count, &channels, &height, &width, &block};
            launch_linear(space_to_depth_, count_size, parameters);
            return;
        }

        throw std::runtime_error("driver-cubin: unsupported operator " + op);
    }

    driver::Driver& driver_;
    const driver::Module& module_;
    json manifest_;
    std::vector<std::byte> weights_host_;
    driver::DeviceBuffer weights_device_;
    driver::DeviceBuffer arena_device_;
    driver::DeviceBuffer reglu_buffer_;
    driver::DeviceBuffer spatial_input_buffer_;
    driver::DeviceBuffer spatial_output_buffer_;
    std::vector<driver::DeviceBuffer> epilogue_buffers_;
    std::vector<driver::DeviceBuffer> concat_buffers_;
    std::unordered_map<std::size_t, driver::DeviceBuffer>
        cutlass_parameter_buffers_;
    std::unordered_map<std::size_t,
                       std::array<std::byte,
                                  driver::kCutlassPointwiseParamsStorageBytes>>
        cutlass_host_parameters_;
    std::unordered_map<std::size_t, driver::DeviceBuffer>
        cutlass_spatial_parameter_buffers_;
    std::unordered_map<std::size_t, driver::DeviceBuffer>
        cutlass_spatial_weight_buffers_;
    std::unordered_map<std::size_t,
                       std::array<std::byte,
                                  driver::kCutlassPointwiseParamsStorageBytes>>
        cutlass_spatial_host_parameters_;
    std::vector<driver::DeviceBuffer> input_buffers_;
    std::vector<std::string> input_names_;
    std::vector<std::size_t> aliased_input_slices_;
    std::vector<std::string> direct_concat_values_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, Value> values_;
    driver::abi::Function binary_ = nullptr;
    driver::abi::Function binary_contiguous_ = nullptr;
    driver::abi::Function unary_ = nullptr;
    driver::abi::Function reglu_ = nullptr;
    driver::abi::Function reglu_vec8_ = nullptr;
    driver::abi::Function convolution_ = nullptr;
    driver::abi::Function pointwise_convolution_ = nullptr;
    driver::abi::Function pointwise_convolution_wide_ = nullptr;
    driver::abi::Function pointwise_convolution_balanced_ = nullptr;
    driver::abi::Function pointwise_convolution_mma_ = nullptr;
    driver::abi::Function pointwise_convolution_mma_small_ = nullptr;
    driver::abi::Function pointwise_reglu_mma_ = nullptr;
    driver::abi::Function pointwise_reglu_mma_small_ = nullptr;
    driver::abi::Function cutlass_pointwise_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_ = nullptr;
    driver::abi::Function cutlass_pointwise_leaky_relu_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_leaky_relu_ = nullptr;
    driver::abi::Function cutlass_pointwise_residual_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_residual_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_stage4_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_stage4_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_leaky_relu_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_leaky_relu_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_residual_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_medium_residual_ = nullptr;
    driver::abi::Function cutlass_pointwise_spatial_wide_residual_init_ = nullptr;
    driver::abi::Function cutlass_pointwise_spatial_wide_residual_ = nullptr;
    driver::abi::Function cutlass_spatial_convolution_init_ = nullptr;
    driver::abi::Function cutlass_spatial_convolution_ = nullptr;
    driver::abi::Function transpose_ = nullptr;
    driver::abi::Function oihw_to_krsc_ = nullptr;
    driver::abi::Function spatial_convolution_mma_ = nullptr;
    driver::abi::Function spatial_convolution_mma_wide_ = nullptr;
    driver::abi::Function spatial_convolution_ = nullptr;
    driver::abi::Function depthwise_convolution_ = nullptr;
    driver::abi::Function depthwise_convolution_pair_ = nullptr;
    driver::abi::Function depthwise_convolution_quad_ = nullptr;
    driver::abi::Function gather_ = nullptr;
    driver::abi::Function slice_ = nullptr;
    driver::abi::Function concat_ = nullptr;
    driver::abi::Function depth_to_space_ = nullptr;
    driver::abi::Function space_to_depth_ = nullptr;
    bool cutlass_parameters_ready_ = false;
    driver::ExecutableGraph executable_graph_;
};

class DriverCubinBackend final : public InferenceBackend {
public:
    explicit DriverCubinBackend(BackendOptions options)
        : options_(std::move(options)), driver_(options_.device_id),
          module_(driver_.load_module(std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(mlvc_driver_kernels_fatbin),
              mlvc_driver_kernels_fatbin_size)))
    {
    }

    std::string_view name() const noexcept override { return "driver-cubin"; }

    void load(const std::string& model_name) override
    {
        graph_ = std::make_unique<AotGraph>(options_.model_dir, model_name,
                                            driver_, module_);
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!graph_)
            throw std::runtime_error("driver-cubin: load() must be called first");
        return graph_->run(inputs);
    }

private:
    BackendOptions options_;
    driver::Driver driver_;
    driver::Module module_;
    std::unique_ptr<AotGraph> graph_;
};

}  // namespace

std::string_view compiled_backend_name() noexcept
{
    return "driver-cubin";
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    return std::make_unique<DriverCubinBackend>(options);
}

}  // namespace mlvc
