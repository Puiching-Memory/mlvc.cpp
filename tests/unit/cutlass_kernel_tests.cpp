#include "mlvc/core/half.hpp"
#include "mlvc/driver_cubin/cutlass.hpp"
#include "mlvc/driver_cubin/driver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" const unsigned char mlvc_driver_kernels_fatbin[];
extern "C" const std::size_t mlvc_driver_kernels_fatbin_size;

namespace {
using Half = mlvc::Float16Storage;
using mlvc::half_to_float;
using mlvc::float_to_half;
using namespace mlvc::driver_cubin;

std::vector<Half> values(std::size_t count, std::uint32_t seed)
{
    std::vector<Half> result(count);
    for (auto& value : result) {
        seed = seed * 1664525U + 1013904223U;
        value = float_to_half(float(int(seed >> 24) - 128) / 256.0F);
    }
    return result;
}

DeviceBuffer upload(Driver& driver, const std::vector<Half>& values)
{
    auto buffer = driver.allocate(values.size() * sizeof(Half));
    driver.upload(buffer, values.data(), buffer.size());
    return buffer;
}

struct Variant {
    const char* name;
    int m, n, stages, epilogue;
    int log_tile = 0;
};

std::string pointwise_init_name(const Variant& v)
{
    std::string name(v.name);
    if (v.log_tile > 0)
        name += "_swizzle";
    return name + "_init_fp16";
}

std::string pointwise_main_name(const Variant& v)
{
    return std::string(v.name) + "_fp16";
}

Dim3 pointwise_grid(const Variant& v, int m, int n)
{
    const unsigned tiles_m = unsigned((m + v.m - 1) / v.m);
    const unsigned tiles_n = unsigned((n + v.n - 1) / v.n);
    const unsigned group = 1U << v.log_tile;
    return {tiles_m * group, (tiles_n + group - 1) / group, 1};
}

float check(const std::vector<Half>& actual, const std::vector<Half>& expected,
            const std::string& name, float tolerance)
{
    float max_error = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = half_to_float(actual[i]);
        const float b = half_to_float(expected[i]);
        const float error = std::abs(a - b);
        max_error = std::max(max_error, error);
        if (!std::isfinite(a) || error > tolerance)
            throw std::runtime_error(name + " mismatch at " + std::to_string(i) +
                                     ": " + std::to_string(a) + " vs " +
                                     std::to_string(b));
    }
    return max_error;
}

struct PointwiseCase {
    std::vector<Half> input;
    std::vector<Half> weight;
    std::vector<Half> bias;
    std::vector<Half> residual;
    DeviceBuffer device_input;
    DeviceBuffer device_weight;
    DeviceBuffer device_bias;
    DeviceBuffer device_residual;
    DeviceBuffer output;
    DeviceBuffer storage;
    std::array<std::byte, kCutlassPointwiseParamsStorageBytes> params{};
    abi::Function function = nullptr;
    Dim3 grid{};
    unsigned shared = 0;
};

PointwiseCase prepare_pointwise(Driver& driver, Module& module,
                                const Variant& v, int m, int n, int k)
{
    PointwiseCase state;
    state.input = values(std::size_t(k) * n, 1);
    state.weight = values(std::size_t(m) * k, 2);
    state.bias = values(std::size_t(m), 3);
    state.residual = values(std::size_t(m) * n, 4);
    state.device_input = upload(driver, state.input);
    state.device_weight = upload(driver, state.weight);
    state.device_bias = upload(driver, state.bias);
    state.device_residual = upload(driver, state.residual);
    state.output = driver.allocate(state.residual.size() * sizeof(Half));
    state.storage = driver.allocate(kCutlassPointwiseParamsStorageBytes);
    auto pi = state.device_input.address();
    auto pw = state.device_weight.address();
    auto pb = state.device_bias.address();
    auto pr = state.device_residual.address();
    auto po = state.output.address();
    auto ps = state.storage.address();
    int log_tile = v.log_tile;
    void* init[] = {&ps, &pi, &pw, &pb, &pr, &po, &m, &n, &k, &log_tile};
    const std::size_t init_argc = v.log_tile > 0 ? 10 : 9;
    driver.launch(module.function(pointwise_init_name(v)),
                  {1, 1, 1}, {1, 1, 1}, 0,
                  std::span<void*>(init, init_argc));
    driver.download(state.params.data(), ps, state.params.size());
    state.function = module.function(pointwise_main_name(v));
    state.shared = unsigned((v.m + v.n) * 32 * v.stages * 2);
    if (state.shared > 49152)
        driver.set_max_dynamic_shared_memory(state.function, state.shared);
    state.grid = pointwise_grid(v, m, n);
    return state;
}

float pointwise(Driver& driver, Module& module, const Variant& v,
                int m, int n, int k)
{
    auto state = prepare_pointwise(driver, module, v, m, n, k);
    void* args[] = {state.params.data()};
    driver.launch(state.function, state.grid, {128, 1, 1}, state.shared, args);
    std::vector<Half> actual(state.residual.size());
    std::vector<Half> expected(state.residual.size());
    driver.download(actual.data(), state.output.address(),
                    state.output.size());
    for (int row = 0; row < m; ++row) {
        for (int column = 0; column < n; ++column) {
            float sum = 0;
            for (int c = 0; c < k; ++c)
                sum += half_to_float(state.weight[std::size_t(row) * k + c]) *
                       half_to_float(state.input[std::size_t(c) * n + column]);
            sum += half_to_float(state.bias[row]);
            if (v.epilogue == 1)
                sum = sum >= 0 ? sum : sum * 0.01F;
            if (v.epilogue == 2)
                sum += half_to_float(
                    state.residual[std::size_t(row) * n + column]);
            expected[std::size_t(row) * n + column] = float_to_half(sum);
        }
    }
    return check(actual, expected, pointwise_main_name(v), 0.002F);
}

void spatial(Driver& driver, Module& module, int channels, int outputs,
             int stride, int kernel, int padding)
{
    int batch = 1, height = 13, width = 17;
    int oh = (height + 2 * padding - kernel) / stride + 1;
    int ow = (width + 2 * padding - kernel) / stride + 1;
    auto input = values(std::size_t(height) * width * channels, 5);
    auto weight = values(std::size_t(outputs) * kernel * kernel * channels, 6);
    auto bias = values(outputs, 7);
    auto di = upload(driver, input), dw = upload(driver, weight), db = upload(driver, bias);
    auto output = driver.allocate(std::size_t(oh) * ow * outputs * sizeof(Half));
    auto storage = driver.allocate(kCutlassPointwiseParamsStorageBytes);
    auto pi = di.address(), pw = dw.address(), pb = db.address();
    auto po = output.address(), ps = storage.address();
    void* init[] = {&ps, &pi, &pw, &pb, &po, &batch, &channels, &height, &width,
                   &outputs, &oh, &ow, &kernel, &kernel, &stride, &stride, &padding, &padding};
    driver.launch(module.function("mlvc_cutlass_spatial_conv_init_fp16"),
                  {1, 1, 1}, {1, 1, 1}, 0, init);
    alignas(16) std::array<std::byte, kCutlassPointwiseParamsStorageBytes> params{};
    driver.download(params.data(), ps, params.size());
    void* args[] = {params.data()};
    auto function = module.function("mlvc_cutlass_spatial_conv_fp16");
    driver.set_max_dynamic_shared_memory(function, 73728);
    driver.launch(function, {unsigned((oh * ow + 127) / 128),
                            unsigned((outputs + 255) / 256), 1},
                  {256, 1, 1}, 73728, args);
    std::vector<Half> actual(std::size_t(oh) * ow * outputs), expected(actual.size());
    driver.download(actual.data(), po, output.size());
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            for (int o = 0; o < outputs; ++o) {
                float sum = 0;
                for (int cb = 0; cb < channels; cb += 32) {
                    for (int r = 0; r < kernel; ++r) {
                        for (int s = 0; s < kernel; ++s) {
                            int iy = y * stride - padding + r, ix = x * stride - padding + s;
                            if (iy < 0 || iy >= height || ix < 0 || ix >= width)
                                continue;
                            for (int c = cb; c < std::min(cb + 32, channels); ++c)
                                sum += half_to_float(input[(iy * width + ix) * channels + c]) *
                                    half_to_float(weight[((o * kernel + r) * kernel + s) * channels + c]);
                        }
                    }
                }
                expected[(y * ow + x) * outputs + o] = float_to_half(sum + half_to_float(bias[o]));
            }
        }
    }
    check(actual, expected, "spatial", 0.002F);
}

void bench_pointwise(Driver& driver, Module& module, const Variant& v,
                     int m, int n, int k, int warmup, int iterations)
{
    auto state = prepare_pointwise(driver, module, v, m, n, k);
    void* args[] = {state.params.data()};
    for (int i = 0; i < warmup; ++i)
        driver.launch(state.function, state.grid, {128, 1, 1}, state.shared,
                      args);
    driver.synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        driver.launch(state.function, state.grid, {128, 1, 1}, state.shared,
                      args);
    driver.synchronize();
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count() /
                      iterations;
    const double tflops = 2.0 * m * n * k / (us * 1.0e6);
    std::printf("%s_l%d,%d,%d,%d,%.2f,%.2f\n", pointwise_main_name(v).c_str(),
                v.log_tile, m, n, k, us, tflops);
}
} // namespace

int main(int argc, char** argv)
{
    try {
        Driver driver;
        if (driver.device_info().compute_major < 8)
            return 77;
        auto module = driver.load_module({reinterpret_cast<const std::byte*>(mlvc_driver_kernels_fatbin),
                                         mlvc_driver_kernels_fatbin_size});
        const Variant variants[] = {
            {"mlvc_cutlass_pointwise", 128, 64, 3, 0},
            {"mlvc_cutlass_pointwise_leaky_relu", 128, 64, 3, 1},
            {"mlvc_cutlass_pointwise_residual", 128, 64, 3, 2},
            {"mlvc_cutlass_pointwise_medium", 128, 128, 3, 0},
            {"mlvc_cutlass_pointwise_medium_stage4", 128, 128, 4, 0},
            {"mlvc_cutlass_pointwise_medium_leaky_relu", 128, 128, 3, 1},
            {"mlvc_cutlass_pointwise_medium_residual", 128, 128, 3, 2},
            {"mlvc_cutlass_pointwise_spatial_wide_residual", 64, 256, 3, 2},
            {"mlvc_cutlass_pointwise", 128, 64, 3, 0, 3},
            {"mlvc_cutlass_pointwise_leaky_relu", 128, 64, 3, 1, 3},
            {"mlvc_cutlass_pointwise_residual", 128, 64, 3, 2, 3},
            {"mlvc_cutlass_pointwise_medium", 128, 128, 3, 0, 3},
            {"mlvc_cutlass_pointwise_medium_leaky_relu", 128, 128, 3, 1,
             3},
            {"mlvc_cutlass_pointwise_medium_residual", 128, 128, 3, 2,
             3},
            {"mlvc_cutlass_pointwise_spatial_wide_residual", 64, 256, 3, 2,
             3},
        };
        if (argc > 1 && std::string(argv[1]) == "--bench") {
            const std::array<int, 3> bench_shapes[] = {
                {136, 920, 24},  {136, 920, 72},   {136, 920, 128},
                {256, 960, 512}, {256, 3680, 64},  {384, 920, 256},
                {512, 3680, 512}, {128, 14720, 64},
            };
            std::puts("variant,m,n,k,us,tflops");
            for (const auto& v : variants) {
                for (const auto& shape : bench_shapes)
                    bench_pointwise(driver, module, v, shape[0], shape[1],
                                    shape[2], 10, 100);
                if (v.log_tile == 0 &&
                    v.name != std::string("mlvc_cutlass_pointwise_medium_stage4")) {
                    for (int log_tile = 1; log_tile <= 2; ++log_tile) {
                        Variant extra = v;
                        extra.log_tile = log_tile;
                        for (const auto& shape : bench_shapes)
                            bench_pointwise(driver, module, extra, shape[0],
                                            shape[1], shape[2], 10, 100);
                    }
                }
            }
            return 0;
        }
        // Exercise partial M/N/K tiles, exact tiles, a pipeline shorter than
        // its prefetch depth, and the larger reductions used by the models.
        const std::array<int, 3> shapes[] = {
            {136, 920, 24}, {136, 920, 72}, {136, 920, 128},
            {128, 256, 32}, {8, 8, 8}, {256, 960, 512},
        };
        float max_error = 0.0F;
        for (const auto& v : variants)
            for (const auto& shape : shapes)
                max_error = std::max(
                    max_error,
                    pointwise(driver, module, v, shape[0], shape[1], shape[2]));
        spatial(driver, module, 24, 24, 1, 3, 1);
        spatial(driver, module, 40, 264, 2, 3, 1);
        spatial(driver, module, 8, 8, 2, 2, 0);
        spatial(driver, module, 128, 256, 2, 2, 0);
        std::printf("CUTLASS kernels: %d GEMM and 4 convolution cases passed"
                    " (max abs error %.5f)\n",
                    int(sizeof(variants) / sizeof(variants[0]) * 6),
                    max_error);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
