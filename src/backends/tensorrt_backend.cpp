// TensorRT backend: parses the exported MLVC ONNX graphs and builds
// TensorRT engines (TensorRT 10 API). Requires CUDA.

#include "mlvc/backend.hpp"

#ifdef MLVC_WITH_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
            delete p;  // TRT 10: objects implement virtual destructors
    }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

size_t element_count(const nvinfer1::Dims& dims)
{
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0)
            throw std::runtime_error("tensorrt: dynamic dims not supported yet");
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

class TensorRtBackend final : public InferenceBackend {
public:
    explicit TensorRtBackend(BackendOptions options) : options_(std::move(options))
    {
        if (options_.device != "cuda")
            throw std::runtime_error("tensorrt backend requires --device cuda");
        if (cudaStreamCreate(&stream_) != cudaSuccess)
            throw std::runtime_error("tensorrt: cudaStreamCreate failed");
    }

    ~TensorRtBackend() override { cudaStreamDestroy(stream_); }

    BackendKind kind() const noexcept override { return BackendKind::kTensorRt; }

    void load(const std::string& model_name) override
    {
        const std::string path = options_.model_dir + "/" + model_name + ".onnx";

        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
        TrtPtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(0));  // explicit batch
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger_));
        if (!parser->parseFromFile(path.c_str(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
            throw std::runtime_error("tensorrt: failed to parse " + path);

        // MLVC exports carry fp16 weights; allow fp16 math when supported.
        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (builder->platformHasFastFp16())
            config->setFlag(nvinfer1::BuilderFlag::kFP16);

        TrtPtr<nvinfer1::IHostMemory> plan(
            builder->buildSerializedNetwork(*network, *config));
        if (!plan)
            throw std::runtime_error("tensorrt: engine build failed for " + path);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
        context_.reset(engine_->createExecutionContext());
        if (!context_)
            throw std::runtime_error("tensorrt: failed to create execution context");

        buffers_.clear();
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            const size_t bytes =
                element_count(engine_->getTensorShape(name)) * sizeof(float);
            void* device_ptr = nullptr;
            if (cudaMalloc(&device_ptr, bytes) != cudaSuccess)
                throw std::runtime_error("tensorrt: cudaMalloc failed");
            buffers_.push_back({name, device_ptr, bytes,
                                engine_->getTensorIOMode(name) ==
                                    nvinfer1::TensorIOMode::kINPUT});
        }
    }

    std::vector<Tensor> run(const std::vector<Tensor>& inputs) override
    {
        if (!context_)
            throw std::runtime_error("tensorrt: load() must be called first");

        for (Buffer& b : buffers_) {
            if (!b.is_input)
                continue;
            auto it = std::find_if(inputs.begin(), inputs.end(),
                                   [&](const Tensor& t) { return t.name == b.name; });
            if (it == inputs.end())
                throw std::runtime_error("tensorrt: missing input " + b.name);
            if (it->data.size() * sizeof(float) != b.bytes)
                throw std::runtime_error("tensorrt: input size mismatch for " + b.name);
            cudaMemcpyAsync(b.device_ptr, it->data.data(), b.bytes,
                            cudaMemcpyHostToDevice, stream_);
            context_->setTensorAddress(b.name.c_str(), b.device_ptr);
        }
        for (Buffer& b : buffers_)
            if (!b.is_input)
                context_->setTensorAddress(b.name.c_str(), b.device_ptr);

        if (!context_->enqueueV3(stream_))
            throw std::runtime_error("tensorrt: enqueueV3 failed");

        std::vector<Tensor> result;
        for (Buffer& b : buffers_) {
            if (b.is_input)
                continue;
            Tensor t;
            t.name = b.name;
            const nvinfer1::Dims dims = engine_->getTensorShape(b.name.c_str());
            for (int i = 0; i < dims.nbDims; ++i)
                t.shape.push_back(dims.d[i]);
            t.data.resize(b.bytes / sizeof(float));
            cudaMemcpyAsync(t.data.data(), b.device_ptr, b.bytes,
                            cudaMemcpyDeviceToHost, stream_);
            result.push_back(std::move(t));
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess)
            throw std::runtime_error("tensorrt: inference stream sync failed");
        return result;
    }

private:
    struct Buffer {
        std::string name;
        void* device_ptr;
        size_t bytes;
        bool is_input;
    };

    BackendOptions options_;
    TrtLogger logger_;
    cudaStream_t stream_{};
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    std::vector<Buffer> buffers_;
};

}  // namespace

std::unique_ptr<InferenceBackend> create_tensorrt_backend(const BackendOptions& options)
{
    return std::make_unique<TensorRtBackend>(options);
}

}  // namespace mlvc

#else  // MLVC_WITH_TENSORRT

#include <stdexcept>

namespace mlvc {

std::unique_ptr<InferenceBackend> create_tensorrt_backend(const BackendOptions&)
{
    throw std::runtime_error("tensorrt backend not compiled in (MLVC_WITH_TENSORRT=OFF)");
}

}  // namespace mlvc

#endif  // MLVC_WITH_TENSORRT
