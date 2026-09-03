// TensorRT 11.2 backend: builds or loads hardware-specific CUDA engines.

#include "mlvc/backend.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlvc {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[tensorrt] %s\n", msg);
    }
};

template <typename T>
struct TrtDeleter {
    void operator()(T* p) const
    {
        if (p)
            delete p;
    }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("tensorrt: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}

std::size_t element_count(const nvinfer1::Dims& dims)
{
    std::size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0)
            throw std::runtime_error("tensorrt: dynamic dims not supported yet");
        count *= static_cast<std::size_t>(dims.d[i]);
    }
    return count;
}

std::size_t data_type_size(nvinfer1::DataType type)
{
    switch (type) {
    case nvinfer1::DataType::kHALF:  return sizeof(Float16Storage);
    case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
    default: throw std::runtime_error("tensorrt: unsupported tensor dtype");
    }
}

TensorDataType tensor_data_type(nvinfer1::DataType type)
{
    switch (type) {
    case nvinfer1::DataType::kHALF:  return TensorDataType::kFloat16;
    case nvinfer1::DataType::kINT32: return TensorDataType::kInt32;
    default: throw std::runtime_error("tensorrt: unsupported tensor dtype");
    }
}

TensorStorage allocate_storage(nvinfer1::DataType type, std::size_t count)
{
    switch (type) {
    case nvinfer1::DataType::kHALF:  return std::vector<Float16Storage>(count);
    case nvinfer1::DataType::kINT32: return std::vector<std::int32_t>(count);
    default: throw std::runtime_error("tensorrt: unsupported output dtype");
    }
}

class Buffer {
public:
    Buffer(std::string tensor_name, nvinfer1::DataType tensor_type,
           nvinfer1::Dims tensor_dims, bool input)
        : name(std::move(tensor_name)), type(tensor_type), dims(tensor_dims),
          bytes(element_count(dims) * data_type_size(type)), is_input(input)
    {
        check_cuda(cudaMalloc(&device_ptr, bytes), "cudaMalloc");
        try {
            check_cuda(cudaMallocHost(&host_ptr, bytes), "cudaMallocHost");
        } catch (...) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
            throw;
        }
    }

    ~Buffer()
    {
        if (host_ptr)
            cudaFreeHost(host_ptr);
        if (device_ptr)
            cudaFree(device_ptr);
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept
        : name(std::move(other.name)), type(other.type), dims(other.dims),
          device_ptr(std::exchange(other.device_ptr, nullptr)),
          host_ptr(std::exchange(other.host_ptr, nullptr)), bytes(other.bytes),
          is_input(other.is_input)
    {
    }
    Buffer& operator=(Buffer&&) = delete;

    std::string name;
    nvinfer1::DataType type;
    nvinfer1::Dims dims;
    void* device_ptr{};
    void* host_ptr{};
    std::size_t bytes{};
    bool is_input{};
};

class TensorRtBackend final : public InferenceBackend {
public:
    explicit TensorRtBackend(BackendOptions options) : options_(std::move(options))
    {
        check_cuda(cudaSetDevice(options_.device_id), "cudaSetDevice");
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }

    ~TensorRtBackend() override
    {
        buffers_.clear();
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    std::string_view name() const noexcept override { return "tensorrt"; }

    void load(const std::string& model_name,
              const ModelExecutionConfig& config) override
    {
        const std::filesystem::path onnx_path =
            std::filesystem::path(options_.model_dir) / (model_name + ".onnx");

        buffers_.clear();
        state_bindings_.clear();
        state_initialized_ = false;
        context_.reset();
        engine_.reset();
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
            throw std::runtime_error("tensorrt: failed to create runtime");

        const std::filesystem::path engine_path = cache_path(model_name);
        if (cache_is_current(engine_path, onnx_path))
            engine_ = deserialize_engine(engine_path);

        if (!engine_) {
            TrtPtr<nvinfer1::IHostMemory> plan = build_engine(onnx_path);
            engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
            if (!engine_)
                throw std::runtime_error("tensorrt: engine deserialization failed");
            save_engine(engine_path, *plan);
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_)
            throw std::runtime_error("tensorrt: failed to create execution context");

        buffers_.reserve(engine_->getNbIOTensors());
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto type = engine_->getTensorDataType(name);
            if (type != nvinfer1::DataType::kHALF &&
                type != nvinfer1::DataType::kINT32) {
                throw std::runtime_error(
                    std::string("tensorrt: tensor ") + name +
                    " is not fp16/int32; mlvc.cpp accepts FP16 models only");
            }
            buffers_.emplace_back(
                name, type, engine_->getTensorShape(name),
                engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        }
        configure_state(config);
    }

    void reset_state() override
    {
        if (!context_)
            throw std::runtime_error("tensorrt: load() must be called first");
        for (const StateBinding& binding : state_bindings_) {
            Buffer& input = buffers_.at(binding.input_buffer_index);
            check_cuda(cudaMemsetAsync(input.device_ptr, 0, input.bytes, stream_),
                       "cudaMemsetAsync state");
        }
        state_initialized_ = true;
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!context_)
            throw std::runtime_error("tensorrt: load() must be called first");
        if (!state_bindings_.empty() && !state_initialized_)
            throw std::runtime_error(
                "tensorrt: reset_state() must be called before run()");

        const std::size_t graph_input_count = static_cast<std::size_t>(
            std::count_if(buffers_.begin(), buffers_.end(),
                          [](const Buffer& buffer) { return buffer.is_input; }));
        if (inputs.size() + state_bindings_.size() != graph_input_count)
            throw std::runtime_error("tensorrt: graph input count mismatch");
        std::size_t external_index = 0;
        for (std::size_t buffer_index = 0; buffer_index < buffers_.size();
             ++buffer_index) {
            Buffer& buffer = buffers_[buffer_index];
            if (!buffer.is_input)
                continue;
            if (is_state_input(buffer_index))
                continue;
            const Tensor& input = inputs.at(external_index++);
            if (input.data_type() != tensor_data_type(buffer.type))
                throw std::runtime_error("tensorrt: input dtype mismatch for " + buffer.name);
            if (input.byte_size() != buffer.bytes)
                throw std::runtime_error("tensorrt: input size mismatch for " + buffer.name);

            std::memcpy(buffer.host_ptr, input.raw_data(), buffer.bytes);
            check_cuda(cudaMemcpyAsync(buffer.device_ptr, buffer.host_ptr, buffer.bytes,
                                       cudaMemcpyHostToDevice, stream_),
                       "cudaMemcpyAsync host-to-device");
        }

        for (Buffer& buffer : buffers_) {
            if (!context_->setTensorAddress(buffer.name.c_str(), buffer.device_ptr))
                throw std::runtime_error("tensorrt: setTensorAddress failed for " +
                                         buffer.name);
        }
        if (!context_->enqueueV3(stream_))
            throw std::runtime_error("tensorrt: enqueueV3 failed");

        for (std::size_t index = 0; index < buffers_.size(); ++index) {
            Buffer& buffer = buffers_[index];
            if (!buffer.is_input && !is_state_output(index))
                check_cuda(cudaMemcpyAsync(buffer.host_ptr, buffer.device_ptr, buffer.bytes,
                                           cudaMemcpyDeviceToHost, stream_),
                           "cudaMemcpyAsync device-to-host");
        }
        // The just-produced output becomes the next invocation's input. The
        // old input allocation becomes the next output target, avoiding both
        // host traffic and a device-to-device copy.
        for (const StateBinding& binding : state_bindings_) {
            std::swap(buffers_[binding.input_buffer_index].device_ptr,
                      buffers_[binding.output_buffer_index].device_ptr);
        }
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

        std::vector<Tensor> result;
        for (std::size_t index = 0; index < buffers_.size(); ++index) {
            Buffer& buffer = buffers_[index];
            if (buffer.is_input || is_state_output(index))
                continue;
            Tensor tensor;
            tensor.name = buffer.name;
            for (int i = 0; i < buffer.dims.nbDims; ++i)
                tensor.shape.push_back(buffer.dims.d[i]);
            tensor.data = allocate_storage(buffer.type, element_count(buffer.dims));
            std::visit([&](auto& values) {
                std::memcpy(values.data(), buffer.host_ptr, buffer.bytes);
            }, tensor.data);
            result.push_back(std::move(tensor));
        }
        return result;
    }

private:
    struct StateBinding {
        std::size_t input_buffer_index;
        std::size_t output_buffer_index;
    };

    void configure_state(const ModelExecutionConfig& config)
    {
        std::vector<std::size_t> input_indices;
        std::vector<std::size_t> output_indices;
        for (std::size_t index = 0; index < buffers_.size(); ++index) {
            (buffers_[index].is_input ? input_indices : output_indices)
                .push_back(index);
        }
        for (const StateTensorBinding& requested : config.state_bindings) {
            if (requested.input_index >= input_indices.size() ||
                requested.output_index >= output_indices.size()) {
                throw std::runtime_error(
                    "tensorrt: state tensor index is out of range");
            }
            const std::size_t input_index = input_indices[requested.input_index];
            const std::size_t output_index = output_indices[requested.output_index];
            if (is_state_input(input_index) || is_state_output(output_index))
                throw std::runtime_error("tensorrt: duplicate state tensor binding");

            const Buffer& input = buffers_[input_index];
            const Buffer& output = buffers_[output_index];
            std::vector<int64_t> shape;
            shape.reserve(static_cast<std::size_t>(input.dims.nbDims));
            for (int i = 0; i < input.dims.nbDims; ++i)
                shape.push_back(input.dims.d[i]);
            if (input.type != output.type || input.bytes != output.bytes ||
                input.dims.nbDims != output.dims.nbDims ||
                !std::equal(input.dims.d, input.dims.d + input.dims.nbDims,
                            output.dims.d) ||
                requested.data_type != tensor_data_type(input.type) ||
                requested.shape != shape) {
                throw std::runtime_error(
                    "tensorrt: state tensor shape or dtype mismatch");
            }
            state_bindings_.push_back({input_index, output_index});
        }
    }

    bool is_state_input(std::size_t index) const
    {
        return std::any_of(state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.input_buffer_index == index;
            });
    }

    bool is_state_output(std::size_t index) const
    {
        return std::any_of(state_bindings_.begin(), state_bindings_.end(),
            [&](const StateBinding& binding) {
                return binding.output_buffer_index == index;
            });
    }

    std::filesystem::path cache_path(const std::string& model_name) const
    {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, options_.device_id),
                   "cudaGetDeviceProperties");
        std::filesystem::path directory = options_.engine_cache_dir.empty()
            ? std::filesystem::path(options_.model_dir) / ".mlvc-cache"
            : std::filesystem::path(options_.engine_cache_dir);
        // A shared cache root is commonly used for both MLVC profiles.  Keep
        // their engines in separate namespaces because TensorRT engine names
        // alone do not identify the ONNX graph or tensor shapes.
        std::string profile = "model";
        try {
            std::ifstream metadata(
                std::filesystem::path(options_.model_dir) / "metadata.json");
            nlohmann::json root;
            if (metadata)
                metadata >> root;
            if (root.contains("name") && root.at("name").is_string())
                profile = root.at("name").get<std::string>();
        } catch (const std::exception&) {
            // Model validation reports the authoritative metadata error.  A
            // fallback namespace keeps backend benchmark diagnostics usable
            // for an ONNX-only directory without a metadata file.
        }
        for (char& value : profile) {
            if (!(std::isalnum(static_cast<unsigned char>(value)) ||
                  value == '-' || value == '_' || value == '.'))
                value = '_';
        }
        directory /= profile;
        return directory /
            (model_name + ".trt" + std::to_string(NV_TENSORRT_MAJOR) + "." +
             std::to_string(NV_TENSORRT_MINOR) + ".sm" +
             std::to_string(properties.major) + std::to_string(properties.minor) +
             ".ws" + std::to_string(options_.workspace_size >> 20) +
             ".fp16" +
             ".engine");
    }

    static bool cache_is_current(const std::filesystem::path& engine_path,
                                 const std::filesystem::path& onnx_path)
    {
        std::error_code error;
        if (!std::filesystem::is_regular_file(engine_path, error))
            return false;
        const auto engine_time = std::filesystem::last_write_time(engine_path, error);
        if (error)
            return false;
        const auto onnx_time = std::filesystem::last_write_time(onnx_path, error);
        return !error && engine_time >= onnx_time;
    }

    TrtPtr<nvinfer1::ICudaEngine> deserialize_engine(
        const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            return {};
        const auto length = input.tellg();
        if (length <= 0)
            return {};
        std::vector<char> plan(static_cast<std::size_t>(length));
        input.seekg(0);
        if (!input.read(plan.data(), static_cast<std::streamsize>(length)))
            return {};
        TrtPtr<nvinfer1::ICudaEngine> engine(
            runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (engine)
            std::fprintf(stderr, "[tensorrt] loaded engine cache: %s\n",
                         path.string().c_str());
        return engine;
    }

    TrtPtr<nvinfer1::IHostMemory> build_engine(
        const std::filesystem::path& onnx_path)
    {
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
        if (!builder)
            throw std::runtime_error("tensorrt: failed to create builder");
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
        if (!network)
            throw std::runtime_error("tensorrt: failed to create network");
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger_));
        if (!parser ||
            !parser->parseFromFile(onnx_path.string().c_str(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
            throw std::runtime_error("tensorrt: failed to parse " + onnx_path.string());

        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config)
            throw std::runtime_error("tensorrt: failed to create builder config");
        config->setBuilderOptimizationLevel(5);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   options_.workspace_size);
        // TensorRT 11 uses the ONNX tensor types directly; the exported graphs
        // are FP16 and the runtime contract rejects implicit FP32 tensors.
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);

        TrtPtr<nvinfer1::IHostMemory> plan(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
            throw std::runtime_error("tensorrt: engine build failed for " +
                                     onnx_path.string());
        return plan;
    }

    static void save_engine(const std::filesystem::path& path,
                            const nvinfer1::IHostMemory& plan)
    {
        try {
            std::filesystem::create_directories(path.parent_path());
            std::filesystem::path temporary = path;
            temporary += ".tmp";
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                output.write(static_cast<const char*>(plan.data()),
                             static_cast<std::streamsize>(plan.size()));
                if (!output)
                    throw std::runtime_error("failed to write engine cache");
            }
            std::error_code error;
            std::filesystem::remove(path, error);
            std::filesystem::rename(temporary, path);
            std::fprintf(stderr, "[tensorrt] saved engine cache: %s\n",
                         path.string().c_str());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[tensorrt] warning: %s\n", error.what());
        }
    }

    BackendOptions options_;
    TrtLogger logger_;
    cudaStream_t stream_{};
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    std::vector<Buffer> buffers_;
    std::vector<StateBinding> state_bindings_;
    bool state_initialized_ = false;
};

}  // namespace

std::string_view compiled_backend_name() noexcept
{
    return "tensorrt";
}

std::unique_ptr<InferenceBackend> create_backend(const BackendOptions& options)
{
    return std::make_unique<TensorRtBackend>(options);
}

}  // namespace mlvc
