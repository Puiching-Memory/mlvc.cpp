// Fixed-shape ONNX graph interpreter backed by embedded CUDA Driver kernels.

#include "mlvc/backend.hpp"
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
        for (const auto& output : manifest_.at("outputs"))
            output_names_.push_back(output.at("name").get<std::string>());

        binary_ = module_.function("mlvc_binary_fp16");
        unary_ = module_.function("mlvc_unary_fp16");
        convolution_ = module_.function("mlvc_conv_fp16");
        pointwise_convolution_ = module_.function("mlvc_pointwise_conv_fp16");
        spatial_convolution_ = module_.function("mlvc_spatial_conv_fp16");
        depthwise_convolution_ = module_.function("mlvc_depthwise_conv_fp16");
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

        for (const auto& node : manifest_.at("nodes"))
            execute(node);

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

    void launch_linear(driver::abi::Function function, std::size_t count,
                       std::span<void*> parameters)
    {
        constexpr unsigned int block = 256;
        driver_.launch(function, {divide_up(count, block), 1, 1}, {block, 1, 1},
                       0, parameters);
    }

    void execute(const json& node)
    {
        const std::string op = node.at("op").get<std::string>();
        const auto inputs = node.at("inputs").get<std::vector<std::string>>();
        const auto outputs = node.at("outputs").get<std::vector<std::string>>();
        if (outputs.size() != 1)
            throw std::runtime_error("driver-cubin: only single-output nodes are supported");
        const Value& output = value(outputs[0]);
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
            launch_linear(binary_, count_size, parameters);
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
                void* pointwise_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &spatial, &oc};
                driver_.launch(pointwise_convolution_,
                    {divide_up(spatial, 16), divide_up(oc, 64),
                     static_cast<unsigned int>(n)},
                    {32, 4, 1}, 0, pointwise_parameters);
                return;
            }
            const int reduction = ic * kh * kw;
            const bool use_spatial = groups == 1 && reduction % 16 == 0 &&
                                     oc % 64 == 0;
            if (use_spatial) {
                void* spatial_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                    &oc, &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
                driver_.launch(spatial_convolution_,
                    {divide_up(spatial, 16), divide_up(oc, 64),
                     static_cast<unsigned int>(n)},
                    {32, 4, 1}, 0, spatial_parameters);
                return;
            }
            const bool use_depthwise = groups == ic && oc == ic &&
                                       weight_value.shape.at(1) == 1;
            if (use_depthwise) {
                void* depthwise_parameters[] = {
                    &input, &weight, &bias, &result, &n, &ic, &ih, &iw,
                    &oh, &ow, &kh, &kw, &sh, &sw, &ph, &pw};
                const std::size_t depthwise_count =
                    static_cast<std::size_t>(n) * oc * spatial;
                driver_.launch(depthwise_convolution_,
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
                void* parameters[] = {
                    &input, &result, &input_count_parameter,
                    const_cast<int*>(&id[0]), const_cast<int*>(&id[1]),
                    const_cast<int*>(&id[2]), const_cast<int*>(&id[3]),
                    const_cast<int*>(&od[0]), const_cast<int*>(&od[1]),
                    const_cast<int*>(&od[2]), const_cast<int*>(&od[3]),
                    &axis, &axis_offset,
                };
                launch_linear(concat_, input_count, parameters);
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
    std::vector<driver::DeviceBuffer> input_buffers_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, Value> values_;
    driver::abi::Function binary_ = nullptr;
    driver::abi::Function unary_ = nullptr;
    driver::abi::Function convolution_ = nullptr;
    driver::abi::Function pointwise_convolution_ = nullptr;
    driver::abi::Function spatial_convolution_ = nullptr;
    driver::abi::Function depthwise_convolution_ = nullptr;
    driver::abi::Function gather_ = nullptr;
    driver::abi::Function slice_ = nullptr;
    driver::abi::Function concat_ = nullptr;
    driver::abi::Function depth_to_space_ = nullptr;
    driver::abi::Function space_to_depth_ = nullptr;
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
