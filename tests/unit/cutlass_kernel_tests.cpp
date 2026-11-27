#include "core/half.hpp"
#include "backends/driver_cubin/cutlass.hpp"
#include "backends/driver_cubin/driver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
using DeviceAddress = mlvc::driver_cubin::abi::DeviceAddress;

std::vector<Half> values(std::size_t count, std::uint32_t seed)
{
    std::vector<Half> result(count);
    for (auto& value : result) {
        seed = seed * 1664525U + 1013904223U;
        value = float_to_half(float(int(seed >> 24) - 128) / 256.0F);
    }
    return result;
}

std::vector<Half> make_values(std::size_t count, std::uint32_t seed)
{
    return values(count, seed);
}

DeviceBuffer upload(Driver& driver, const std::vector<Half>& values)
{
    auto buffer = driver.allocate(values.size() * sizeof(Half));
    driver.upload(buffer, values.data(), buffer.size());
    return buffer;
}

void check_native_pixel_layout(Driver& driver, Module& module)
{
    // Three channels distinguish native channel-first layout from block-first
    // layout. Exercise both factors used by the checkpoint model family.
    for (int factor : {2, 8}) {
        int channels = 3, height = factor * 2, width = factor * 3;
        int count = 2 * channels * height * width;
        std::vector<Half> input(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
            input[static_cast<std::size_t>(i)] = float_to_half(static_cast<float>(i));
        auto device_input = upload(driver, input);
        auto shuffled = driver.allocate(input.size() * sizeof(Half));
        auto restored = driver.allocate(input.size() * sizeof(Half));
        auto pi = device_input.address();
        auto ps = shuffled.address();
        auto pr = restored.address();
        void* unshuffle[] = {&pi, &ps, &count, &channels, &height, &width, &factor};
        driver.launch(module.function("mlvc_space_to_depth_fp16"),
                      {static_cast<unsigned>((count + 255) / 256), 1, 1},
                      {256, 1, 1}, 0, unshuffle);
        std::vector<Half> actual(input.size());
        driver.download(actual.data(), ps, actual.size() * sizeof(Half));
        std::size_t output = 0;
        for (int n = 0; n < 2; ++n)
            for (int c = 0; c < channels; ++c)
                for (int by = 0; by < factor; ++by)
                    for (int bx = 0; bx < factor; ++bx)
                        for (int y = 0; y < height / factor; ++y)
                            for (int x = 0; x < width / factor; ++x) {
                                const int index = ((n * channels + c) * height +
                                    y * factor + by) * width + x * factor + bx;
                                if (actual[output++] != input[static_cast<std::size_t>(index)])
                                    throw std::runtime_error("native pixel_unshuffle channel order mismatch");
                            }
        channels *= factor * factor;
        height /= factor;
        width /= factor;
        int mode = 1; // Native CRD pixel_shuffle.
        void* shuffle[] = {&ps, &pr, &count, &channels, &height, &width, &factor, &mode};
        driver.launch(module.function("mlvc_depth_to_space_fp16"),
                      {static_cast<unsigned>((count + 255) / 256), 1, 1},
                      {256, 1, 1}, 0, shuffle);
        driver.download(actual.data(), pr, actual.size() * sizeof(Half));
        if (actual != input)
            throw std::runtime_error("native pixel shuffle round trip mismatch");
    }
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
             int stride, int kernel, int padding, bool small_tile = false)
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
    driver.launch(module.function(small_tile ? "mlvc_cutlass_spatial_conv_small_init_fp16"
                                             : "mlvc_cutlass_spatial_conv_init_fp16"),
                  {1, 1, 1}, {1, 1, 1}, 0, init);
    alignas(16) std::array<std::byte, kCutlassPointwiseParamsStorageBytes> params{};
    driver.download(params.data(), ps, params.size());
    void* args[] = {params.data()};
    auto function = module.function(small_tile ? "mlvc_cutlass_spatial_conv_small_fp16"
                                               : "mlvc_cutlass_spatial_conv_fp16");
    if (small_tile) {
        driver.launch(function, {unsigned((oh * ow + 63) / 64),
                                 unsigned((outputs + 63) / 64), 1},
                      {128, 1, 1}, 32768, args);
    } else {
        driver.set_max_dynamic_shared_memory(function, 73728);
        driver.launch(function, {unsigned((oh * ow + 127) / 128),
                                 unsigned((outputs + 255) / 256), 1},
                      {256, 1, 1}, 73728, args);
    }
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

void check_tiny_pointwise(Driver& driver, Module& module, int m, int n, int k,
                          int epilogue)
{
    auto input = values(std::size_t(k) * n, 11);
    auto weight = values(std::size_t(m) * k, 12);
    auto bias = values(std::size_t(m), 13);
    auto residual = values(std::size_t(m) * n, 14);
    auto di = upload(driver, input);
    auto dw = upload(driver, weight);
    auto db = upload(driver, bias);
    auto dr = upload(driver, residual);
    auto output = driver.allocate(std::size_t(m) * n * sizeof(Half));
    auto pi = di.address(), pw = dw.address(), pb = db.address(),
         pr = dr.address(), po = output.address();
    int batch = 1;
    float alpha = 0.01F;
    void* args[] = {&pi, &pw, &pb, &po, &batch, &k, &n, &m, &pr, &epilogue,
                    &alpha};
    driver.launch(module.function("mlvc_pointwise_conv_mma_tiny_fp16"),
                  {unsigned((n + 15) / 16), unsigned((m + 15) / 16), 1},
                  {32, 4, 1}, 0, args);
    std::vector<Half> actual(std::size_t(m) * n), expected(actual.size());
    driver.download(actual.data(), po, output.size());
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            float sum = half_to_float(bias[r]);
            for (int kk = 0; kk < k; ++kk)
                sum += half_to_float(weight[std::size_t(r) * k + kk]) *
                       half_to_float(input[std::size_t(kk) * n + c]);
            Half h = float_to_half(sum);
            if (epilogue == 1) {
                float v = half_to_float(h);
                h = float_to_half(v >= 0.0F ? v : v * alpha);
            } else if (epilogue == 2) {
                h = float_to_half(
                    half_to_float(h) +
                    half_to_float(residual[std::size_t(r) * n + c]));
            }
            expected[std::size_t(r) * n + c] = h;
        }
    }
    check(actual, expected, "pointwise_tiny", 0.002F);
}

void check_tiny_reglu(Driver& driver, Module& module, int gate, int n, int k)
{
    auto input = values(std::size_t(k) * n, 21);
    auto weight = values(std::size_t(gate * 2) * k, 22);
    auto bias = values(std::size_t(gate * 2), 23);
    auto di = upload(driver, input);
    auto dw = upload(driver, weight);
    auto db = upload(driver, bias);
    auto output = driver.allocate(std::size_t(gate) * n * sizeof(Half));
    auto pi = di.address(), pw = dw.address(), pb = db.address(),
         po = output.address();
    int batch = 1;
    float minimum = 0.0F, maximum = 65504.0F;
    void* args[] = {&pi, &pw, &pb, &po, &batch, &k, &n, &gate, &minimum,
                    &maximum};
    driver.launch(module.function("mlvc_pointwise_reglu_mma_tiny_fp16"),
                  {unsigned((n + 15) / 16), unsigned((gate + 15) / 16), 1},
                  {32, 4, 1}, 0, args);
    std::vector<Half> actual(std::size_t(gate) * n), expected(actual.size());
    driver.download(actual.data(), po, output.size());
    for (int r = 0; r < gate; ++r) {
        for (int c = 0; c < n; ++c) {
            float g = half_to_float(bias[r]);
            float l = half_to_float(bias[gate + r]);
            for (int kk = 0; kk < k; ++kk) {
                const float x = half_to_float(input[std::size_t(kk) * n + c]);
                g += half_to_float(weight[std::size_t(r) * k + kk]) * x;
                l += half_to_float(weight[std::size_t(gate + r) * k + kk]) * x;
            }
            const float gh = half_to_float(float_to_half(g));
            const float lh = half_to_float(float_to_half(l));
            const float clamped = std::min(std::max(gh, minimum), maximum);
            expected[std::size_t(r) * n + c] = float_to_half(clamped * lh);
        }
    }
    check(actual, expected, "reglu_tiny", 0.002F);
}

void check_dwconv_3x3(Driver& driver, Module& module, int c, int h, int w)
{
    auto input = values(std::size_t(c) * h * w, 31);
    auto weight = values(std::size_t(c) * 9, 32);
    auto bias = values(std::size_t(c), 33);
    auto di = upload(driver, input);
    auto dw = upload(driver, weight);
    auto db = upload(driver, bias);
    auto output = driver.allocate(input.size() * sizeof(Half));
    auto pi = di.address(), pw = dw.address(), pb = db.address(),
         po = output.address();
    int batch = 1;
    if (w % 8 == 0) {
        void* args[] = {&pi, &pw, &pb, &po, &batch, &c, &h, &w, &h, &w};
        const unsigned total_pairs = unsigned(((h + 1) / 2) * (w / 8));
        const unsigned block = total_pairs <= 64 ? 64U : 128U;
        driver.launch(module.function("mlvc_depthwise_conv_oct2_fp16"),
                      {(total_pairs + block - 1) / block, unsigned(c), 1},
                      {block, 1, 1}, 0, args);
    } else {
        void* args[] = {&pi, &pw, &pb, &po, &batch, &c, &h, &w};
        driver.launch(module.function("mlvc_depthwise_conv_3x3_small_fp16"),
                      {unsigned((c + 1) / 2), 1, 1},
                      {32, 2, 1}, 0, args);
    }
    std::vector<Half> actual(input.size()), expected(input.size());
    driver.download(actual.data(), po, output.size());
    for (int ch = 0; ch < c; ++ch) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float acc = half_to_float(bias[ch]);
                for (int ky = 0; ky < 3; ++ky) {
                    const int iy = y + ky - 1;
                    if (iy < 0 || iy >= h) continue;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int ix = x + kx - 1;
                        if (ix < 0 || ix >= w) continue;
                        acc += half_to_float(input[(ch * h + iy) * w + ix]) *
                               half_to_float(weight[ch * 9 + ky * 3 + kx]);
                    }
                }
                expected[(ch * h + y) * w + x] = float_to_half(acc);
            }
        }
    }
    check(actual, expected, "dwconv_3x3", 0.002F);
}

void check_cutlass_reglu(Driver& driver, Module& module, const char* base_name,
                         int tile_m, int tile_n, int stages, int gate, int n,
                         int k, int log_tile = 0)
{
    int out_channels = gate * 2;
    auto input = values(std::size_t(k) * n, 41);
    auto weight = values(std::size_t(out_channels) * k, 42);
    auto bias = values(std::size_t(out_channels), 43);
    auto di = upload(driver, input);
    auto dw = upload(driver, weight);
    auto db = upload(driver, bias);
    auto dw_inter = driver.allocate(weight.size() * sizeof(Half));
    auto db_inter = driver.allocate(bias.size() * sizeof(Half));
    auto output = driver.allocate(std::size_t(gate) * n * sizeof(Half));
    auto storage = driver.allocate(kCutlassPointwiseParamsStorageBytes);

    auto pw = dw.address(), pwi = dw_inter.address();
    void* wi_args[] = {&pw, &pwi, &gate, &k};
    driver.launch(module.function("mlvc_interleave_reglu_rows_fp16"),
                  {unsigned((k + 127) / 128), unsigned(out_channels), 1},
                  {128, 1, 1}, 0, wi_args);
    auto pb = db.address(), pbi = db_inter.address();
    int bw = 1;
    void* bi_args[] = {&pb, &pbi, &gate, &bw};
    driver.launch(module.function("mlvc_interleave_reglu_rows_fp16"),
                  {1, unsigned(out_channels), 1}, {32, 1, 1}, 0, bi_args);

    auto pi = di.address(), po = output.address(), ps = storage.address();
    float minimum = 0.0F, maximum = 6.0F;
    void* init_args[] = {&ps, &pi, &pwi, &pbi, &po, &out_channels, &n, &k,
                         &minimum, &maximum, &log_tile};
    driver.launch(module.function(std::string(base_name) + "_init_fp16"),
                  {1, 1, 1}, {1, 1, 1}, 0, init_args);
    alignas(16) std::array<std::byte, kCutlassPointwiseParamsStorageBytes> params{};
    driver.download(params.data(), ps, params.size());

    auto fn = module.function(std::string(base_name) + "_fp16");
    unsigned shared = unsigned((tile_m + tile_n) * 32 * stages * 2);
    if (shared > 49152)
        driver.set_max_dynamic_shared_memory(fn, shared);
    unsigned gm = unsigned((out_channels + tile_m - 1) / tile_m);
    unsigned gn = unsigned((n + tile_n - 1) / tile_n);
    if (log_tile > 0) {
        unsigned grp = 1U << log_tile;
        gm *= grp;
        gn = (gn + grp - 1) / grp;
    }
    void* args[] = {params.data()};
    driver.launch(fn, {gm, gn, 1}, {128, 1, 1}, shared, args);

    std::vector<Half> actual(std::size_t(gate) * n), expected(actual.size());
    driver.download(actual.data(), po, output.size());
    for (int r = 0; r < gate; ++r) {
        for (int c = 0; c < n; ++c) {
            float g = half_to_float(bias[r]);
            float l = half_to_float(bias[gate + r]);
            for (int kk = 0; kk < k; ++kk) {
                const float x = half_to_float(input[std::size_t(kk) * n + c]);
                g += half_to_float(weight[std::size_t(r) * k + kk]) * x;
                l += half_to_float(weight[std::size_t(gate + r) * k + kk]) * x;
            }
            const float gh = half_to_float(float_to_half(g));
            const float lh = half_to_float(float_to_half(l));
            const float clamped = std::min(std::max(gh, minimum), maximum);
            expected[std::size_t(r) * n + c] = float_to_half(clamped * lh);
        }
    }
    check(actual, expected, base_name, 0.002F);
}

void bench_hot_tiles(Driver& driver, Module& module)
{
    std::puts("shape,tile,log_tile,min_us,p10_us,med_us,blocks");
    const std::array<int, 4> shapes[] = {
        {256, 3680, 256, 2}, {256, 3680, 256, 1},
        {256, 3680, 512, 2}, {192, 3680, 192, 2},
        {192, 3680, 192, 1}, {192, 3680, 384, 2},
        {144, 3680, 288, 2}, {256, 920, 512, 2},
        {384, 920, 256, 0},
    };
    const Variant tiles[] = {
        {"mlvc_cutlass_pointwise_medium", 128, 128, 3, 0, 0},
        {"mlvc_cutlass_pointwise_medium_stage4", 128, 128, 4, 0, 0},
        {"mlvc_cutlass_pointwise_m64n128", 64, 128, 3, 0, 0},
        {"mlvc_cutlass_pointwise", 128, 64, 3, 0, 0},
        {"mlvc_cutlass_pointwise_spatial_wide", 64, 256, 3, 0, 0},
        {"mlvc_cutlass_pointwise_m64n64", 64, 64, 3, 0, 0},
        {"mlvc_cutlass_pointwise_medium_residual", 128, 128, 3, 2, 0},
        {"mlvc_cutlass_pointwise_m64n128_residual", 64, 128, 3, 2, 0},
        {"mlvc_cutlass_pointwise_residual", 128, 64, 3, 2, 0},
        {"mlvc_cutlass_pointwise_spatial_wide_residual", 64, 256, 3, 2, 0},
        {"mlvc_cutlass_pointwise_m64n64_residual", 64, 64, 3, 2, 0},
        {"mlvc_cutlass_pointwise_medium_leaky_relu", 128, 128, 3, 1, 0},
        {"mlvc_cutlass_pointwise_m64n128_leaky_relu", 64, 128, 3, 1, 0},
        {"mlvc_cutlass_pointwise_leaky_relu", 128, 64, 3, 1, 0},
        {"mlvc_cutlass_pointwise_spatial_wide_leaky_relu", 64, 256, 3, 1, 0},
        {"mlvc_cutlass_pointwise_m64n64_leaky_relu", 64, 64, 3, 1, 0},
    };
    constexpr int kRounds = 60;
    constexpr int kBatch = 5;
    for (const auto& s : shapes) {
        struct Cand {
            const char* name;
            int log_tile;
            PointwiseCase st;
            std::vector<double> samples;
        };
        std::vector<Cand> cands;
        for (const auto& base : tiles) {
            if (base.epilogue != s[3]) continue;
            const bool is_s4 = std::string(base.name) ==
                "mlvc_cutlass_pointwise_medium_stage4";
            for (int lt : {0, 1}) {
                if (is_s4 && lt > 0) continue;
                Variant v = base;
                v.log_tile = lt;
                cands.push_back({base.name, lt,
                                 prepare_pointwise(driver, module, v, s[0], s[1], s[2]),
                                 std::vector<double>(kRounds)});
            }
        }
        // Warmup all candidates
        for (auto& c : cands) {
            void* args[] = {c.st.params.data()};
            for (int i = 0; i < 5; ++i)
                driver.launch(c.st.function, c.st.grid, {128, 1, 1}, c.st.shared, args);
        }
        driver.synchronize();
        // Round-robin interleaved rounds so all candidates see identical GPU windows
        for (int r = 0; r < kRounds; ++r) {
            for (auto& c : cands) {
                void* args[] = {c.st.params.data()};
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(c.st.function, c.st.grid, {128, 1, 1}, c.st.shared, args);
                driver.synchronize();
                c.samples[r] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count() / kBatch;
            }
        }
        for (auto& c : cands) {
            std::sort(c.samples.begin(), c.samples.end());
            std::printf("%dx%dx%d_ep%d,%s,l%d,%.2f,%.2f,%.2f,%u\n",
                        s[0], s[1], s[2], s[3], c.name, c.log_tile,
                        c.samples[0], c.samples[kRounds / 10], c.samples[kRounds / 2],
                        c.st.grid.x * c.st.grid.y);
        }
    }
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

struct NhwcVariant {
    const char* name;
    int tm, tn, stages, epilogue; // epilogue: 0=bias, 1=lrelu, 2=res, 3=reglu
    unsigned int threads = 128;
    int log_tile = 0;
};

unsigned int nhwc_shared_bytes(const NhwcVariant& v)
{
    if (v.tm == 128 && v.tn == 256) return 73728U;
    if (v.tm == 128 && v.tn == 128) return v.stages == 4 ? 65536U : 49152U;
    if (v.tm == 64 && v.tn == 64) return 24576U;
    return 36864U;
}

Dim3 nhwc_grid(const NhwcVariant& v, int sp, int oc)
{
    const unsigned gm = unsigned((sp + v.tm - 1) / v.tm);
    const unsigned gn = unsigned((oc + v.tn - 1) / v.tn);
    const unsigned group = 1U << v.log_tile;
    return {gm * group, (gn + group - 1) / group, 1};
}

struct NhwcPrepared {
    DeviceBuffer d_in, d_w, d_b, d_res, d_out;
    std::array<std::byte, kCutlassPointwiseParamsStorageBytes> params{};
    abi::Function fn = nullptr;
    Dim3 grid{};
    Dim3 block{128, 1, 1};
    unsigned int shared = 0;
};

NhwcPrepared prepare_nhwc_pointwise(
    Driver& driver, Module& module, const NhwcVariant& v,
    int sp, int oc, int ic,
    int in_stride = 0, int res_stride = 0, int out_stride = 0,
    const std::vector<Half>* h_in_ptr = nullptr,
    const std::vector<Half>* h_w_ptr = nullptr,
    const std::vector<Half>* h_b_ptr = nullptr,
    const std::vector<Half>* h_res_ptr = nullptr,
    float reglu_min = 0.0F, float reglu_max = 6.0F)
{
    if (in_stride == 0) in_stride = ic;
    const int eff_out_c = (v.epilogue == 3) ? (oc / 2) : oc;
    if (res_stride == 0) res_stride = eff_out_c;
    if (out_stride == 0) out_stride = eff_out_c;

    std::vector<Half> h_in = h_in_ptr ? *h_in_ptr
                                      : make_values(std::size_t(sp) * in_stride, 7);
    std::vector<Half> h_w = h_w_ptr ? *h_w_ptr
                                    : make_values(std::size_t(oc) * ic, 19);
    std::vector<Half> h_b = h_b_ptr ? *h_b_ptr
                                    : make_values(std::size_t(oc), 31);
    std::vector<Half> h_res = h_res_ptr ? *h_res_ptr
                                        : make_values(std::size_t(sp) * res_stride, 43);

    NhwcPrepared p;
    p.d_in = driver.allocate(h_in.size() * sizeof(Half));
    p.d_w = driver.allocate(h_w.size() * sizeof(Half));
    p.d_b = driver.allocate(h_b.size() * sizeof(Half));
    p.d_res = driver.allocate(h_res.size() * sizeof(Half));
    p.d_out = driver.allocate(std::size_t(sp) * out_stride * sizeof(Half));
    driver.upload(p.d_in, h_in.data(), h_in.size() * sizeof(Half));
    driver.upload(p.d_w, h_w.data(), h_w.size() * sizeof(Half));
    driver.upload(p.d_b, h_b.data(), h_b.size() * sizeof(Half));
    driver.upload(p.d_res, h_res.data(), h_res.size() * sizeof(Half));

    DeviceAddress w_addr = p.d_w.address();
    DeviceAddress b_addr = p.d_b.address();
    DeviceBuffer w_inter, b_inter;
    if (v.epilogue == 3) {
        w_inter = driver.allocate(h_w.size() * sizeof(Half));
        b_inter = driver.allocate(h_b.size() * sizeof(Half));
        auto inter_fn = module.function("mlvc_interleave_reglu_pairs_fp16");
        int g = oc / 2;
        DeviceAddress wi_a = w_inter.address();
        void* wp[] = {&w_addr, &wi_a, &g, &ic};
        driver.launch(inter_fn, {unsigned((ic + 127) / 128), unsigned(oc), 1},
                      {128, 1, 1}, 0, wp);
        int one = 1;
        DeviceAddress bi_a = b_inter.address();
        void* bp[] = {&b_addr, &bi_a, &g, &one};
        driver.launch(inter_fn, {1, unsigned(oc), 1}, {32, 1, 1}, 0, bp);
        p.d_w = std::move(w_inter);
        p.d_b = std::move(b_inter);
        w_addr = p.d_w.address();
        b_addr = p.d_b.address();
    }

    auto init_fn = module.function(std::string(v.name) + "_init_fp16");
    p.fn = module.function(std::string(v.name) + "_fp16");
    p.shared = nhwc_shared_bytes(v);
    if (p.shared > 49152U)
        driver.set_max_dynamic_shared_memory(p.fn, p.shared);
    p.grid = nhwc_grid(v, sp, oc);
    p.block = {v.threads, 1, 1};

    DeviceBuffer d_params = driver.allocate(p.params.size());
    DeviceAddress pa = d_params.address();
    DeviceAddress ia = p.d_in.address();
    DeviceAddress ra = p.d_res.address();
    DeviceAddress oa = p.d_out.address();
    int lt = v.log_tile;
    if (v.epilogue == 3) {
        void* args[] = {&pa, &ia, &w_addr, &b_addr, &oa, &sp, &oc, &ic,
                        &in_stride, &out_stride, &reglu_min, &reglu_max, &lt};
        driver.launch(init_fn, {1, 1, 1}, {1, 1, 1}, 0, args);
    } else {
        void* args[] = {&pa, &ia, &w_addr, &b_addr, &ra, &oa, &sp, &oc, &ic,
                        &in_stride, &res_stride, &out_stride, &lt};
        driver.launch(init_fn, {1, 1, 1}, {1, 1, 1}, 0, args);
    }
    driver.download(p.params.data(), d_params.address(), p.params.size());
    return p;
}

void check_nhwc_cutlass(Driver& driver, Module& module, const NhwcVariant& v,
                        int sp, int oc, int ic, int pad_in = 0, int pad_out = 0)
{
    const int in_stride = ic + pad_in;
    const int eff_out_c = (v.epilogue == 3) ? (oc / 2) : oc;
    const int res_stride = eff_out_c + pad_out;
    const int out_stride = eff_out_c + pad_out;
    const auto h_in = make_values(std::size_t(sp) * in_stride, 7);
    const auto h_w = make_values(std::size_t(oc) * ic, 19);
    const auto h_b = make_values(std::size_t(oc), 31);
    const auto h_res = make_values(std::size_t(sp) * res_stride, 43);
    const float lo = -1.0F, hi = 2.0F;

    auto p = prepare_nhwc_pointwise(driver, module, v, sp, oc, ic,
                                    in_stride, res_stride, out_stride,
                                    &h_in, &h_w, &h_b, &h_res, lo, hi);
    void* args[] = {p.params.data()};
    driver.launch(p.fn, p.grid, p.block, p.shared, args);
    std::vector<Half> actual_full(std::size_t(sp) * out_stride);
    driver.download(actual_full.data(), p.d_out.address(),
                    actual_full.size() * sizeof(Half));

    std::vector<Half> actual(std::size_t(sp) * eff_out_c);
    std::vector<Half> expected(std::size_t(sp) * eff_out_c);
    for (int m = 0; m < sp; ++m) {
        if (v.epilogue == 3) {
            const int g = oc / 2;
            for (int c = 0; c < g; ++c) {
                float ag = half_to_float(h_b[c]);
                float al = half_to_float(h_b[g + c]);
                for (int k = 0; k < ic; ++k) {
                    const float x = half_to_float(h_in[std::size_t(m) * in_stride + k]);
                    ag += x * half_to_float(h_w[std::size_t(c) * ic + k]);
                    al += x * half_to_float(h_w[std::size_t(g + c) * ic + k]);
                }
                const float gh = half_to_float(float_to_half(ag));
                const float lh = half_to_float(float_to_half(al));
                const float cl = std::min(std::max(gh, lo), hi);
                expected[std::size_t(m) * g + c] = float_to_half(cl * lh);
                actual[std::size_t(m) * g + c] =
                    actual_full[std::size_t(m) * out_stride + c];
            }
        } else {
            for (int c = 0; c < oc; ++c) {
                float acc = half_to_float(h_b[c]);
                for (int k = 0; k < ic; ++k) {
                    acc += half_to_float(h_in[std::size_t(m) * in_stride + k]) *
                           half_to_float(h_w[std::size_t(c) * ic + k]);
                }
                if (v.epilogue == 1)
                    acc = acc >= 0.0F ? acc : 0.01F * acc;
                else if (v.epilogue == 2)
                    acc += half_to_float(h_res[std::size_t(m) * res_stride + c]);
                expected[std::size_t(m) * oc + c] = float_to_half(acc);
                actual[std::size_t(m) * oc + c] =
                    actual_full[std::size_t(m) * out_stride + c];
            }
        }
    }
    check(actual, expected, std::string(v.name), 0.002F);
}

void check_nhwc_kernels(Driver& driver, Module& module)
{
    auto up = [&](const std::vector<Half>& v) {
        auto b = driver.allocate(v.size() * sizeof(Half));
        driver.upload(b, v.data(), v.size() * sizeof(Half));
        return b;
    };
    auto cpu_dw = [](const std::vector<Half>& in, const std::vector<Half>& w,
                     const std::vector<Half>& b, int h, int ww, int c,
                     int in_stride) {
        std::vector<Half> o(std::size_t(h) * ww * c);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < ww; ++x)
                for (int ch = 0; ch < c; ++ch) {
                    float acc = half_to_float(b[ch]);
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= h) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= ww) continue;
                            acc += half_to_float(
                                       in[(std::size_t(iy) * ww + ix) * in_stride + ch]) *
                                   half_to_float(w[std::size_t(ch) * 9 + ky * 3 + kx]);
                        }
                    }
                    o[(std::size_t(y) * ww + x) * c + ch] = float_to_half(acc);
                }
        return o;
    };

    // 1) Test NHWC DWConv sliding kernels (h4_u2, h2_u2) with channel chunking
    //    and strided input/output across 128, 144, 192, 256, 384 channels.
    auto pack_fn = module.function("mlvc_dwconv_weight_oihw_to_k9c_fp16");
    auto h4_fn = module.function("mlvc_depthwise_conv_3x3_nhwc_h4_u2_fp16");
    auto h2_fn = module.function("mlvc_depthwise_conv_3x3_nhwc_h2_u2_fp16");
    const std::array<std::array<int, 4>, 6> dw_cases{{
        {256, 46, 80, 0}, {192, 46, 80, 64}, {384, 23, 40, 0},
        {144, 23, 40, 16}, {128, 12, 20, 0}, {128, 3, 5, 128}
    }};
    for (const auto& dc : dw_cases) {
        int c = dc[0], h = dc[1], w = dc[2], pad = dc[3];
        int in_s = c + pad, out_s = c + pad, n = 1;
        const auto h_in = make_values(std::size_t(h) * w * in_s, 11 + c);
        const auto h_w = make_values(std::size_t(c) * 9, 23 + c);
        const auto h_b = make_values(std::size_t(c), 37 + c);
        auto d_in = up(h_in), d_w = up(h_w), d_b = up(h_b);
        auto d_wk = driver.allocate(h_w.size() * sizeof(Half));
        auto d_out = driver.allocate(std::size_t(h) * w * out_s * sizeof(Half));
        DeviceAddress wa = d_w.address(), wka = d_wk.address();
        void* pp[] = {&wa, &wka, &c};
        driver.launch(pack_fn, {unsigned((c + 127) / 128), 9, 1}, {128, 1, 1}, 0, pp);

        const auto ref = cpu_dw(h_in, h_w, h_b, h, w, c, in_s);
        const unsigned cg = unsigned(c / 4);
        const unsigned chunk_thr = (cg % 32U == 0U) ? 32U
                                 : (cg % 16U == 0U) ? 16U
                                 : std::min(cg, 32U);
        const unsigned tpb = (chunk_thr == 16U) ? 2U : 1U;
        int c_chunks = int((cg + chunk_thr - 1) / chunk_thr);
        DeviceAddress ia = d_in.address(), ba = d_b.address(), oa = d_out.address();
        void* dp[] = {&ia, &wka, &ba, &oa, &n, &c, &h, &w, &in_s, &out_s, &c_chunks};

        auto verify_out = [&](const char* label) {
            std::vector<Half> full(std::size_t(h) * w * out_s);
            driver.download(full.data(), oa, full.size() * sizeof(Half));
            std::vector<Half> packed(std::size_t(h) * w * c);
            for (int p = 0; p < h * w; ++p)
                for (int ch = 0; ch < c; ++ch)
                    packed[std::size_t(p) * c + ch] = full[std::size_t(p) * out_s + ch];
            check(packed, ref, label, 0.005F);
        };

        driver.launch(h4_fn,
                      {unsigned(((w + 3) / 4 + tpb - 1) / tpb),
                       unsigned((h + 1) / 2), unsigned(c_chunks)},
                      {chunk_thr, tpb, 1}, 0, dp);
        verify_out("mlvc_depthwise_conv_3x3_nhwc_h4_u2_fp16");

        driver.launch(h2_fn,
                      {unsigned(((w + 1) / 2 + tpb - 1) / tpb),
                       unsigned((h + 1) / 2), unsigned(c_chunks)},
                      {chunk_thr, tpb, 1}, 0, dp);
        verify_out("mlvc_depthwise_conv_3x3_nhwc_h2_u2_fp16");
    }

    // 2) Test tiny NHWC MMA pointwise conv (with residual epilogue)
    {
        auto tiny_fn = module.function("mlvc_pointwise_conv_mma_tiny_nhwc_fp16");
        int sp = 15, ic = 128, oc = 128, n = 1, ep = 2;
        float al = 0.01F;
        const auto x = make_values(std::size_t(sp) * ic, 71);
        const auto w1 = make_values(std::size_t(oc) * ic, 3);
        const auto b1 = make_values(std::size_t(oc), 5);
        const auto res = make_values(std::size_t(sp) * oc, 73);
        auto dx = up(x), dw1 = up(w1), db1 = up(b1), dres = up(res);
        auto dout = driver.allocate(std::size_t(sp) * oc * sizeof(Half));
        DeviceAddress xa = dx.address(), wa = dw1.address(), ba = db1.address(),
                      oa = dout.address(), ra = dres.address();
        void* tp[] = {&xa, &wa, &ba, &oa, &n, &ic, &sp, &oc, &ic, &oc, &oc, &ra, &ep, &al};
        driver.launch(tiny_fn, {unsigned((sp + 15) / 16), unsigned((oc + 15) / 16), 1},
                      {32, 4, 1}, 0, tp);
        std::vector<Half> actual(std::size_t(sp) * oc), ref(std::size_t(sp) * oc);
        driver.download(actual.data(), oa, actual.size() * sizeof(Half));
        for (int m = 0; m < sp; ++m)
            for (int c = 0; c < oc; ++c) {
                float acc = half_to_float(b1[c]);
                for (int k = 0; k < ic; ++k)
                    acc += half_to_float(x[std::size_t(m) * ic + k]) *
                           half_to_float(w1[std::size_t(c) * ic + k]);
                ref[std::size_t(m) * oc + c] = float_to_half(
                    half_to_float(float_to_half(acc)) +
                    half_to_float(res[std::size_t(m) * oc + c]));
            }
        check(actual, ref, "mlvc_pointwise_conv_mma_tiny_nhwc_fp16", 0.01F);
    }

    // 3) Test vectorized 128-bit NHWC channel-broadcast + strided binary kernels
    {
        auto bcast_fn = module.function("mlvc_binary_channel_broadcast_nhwc_fp16");
        for (int c : {128, 192, 132}) {
            int pixels = 257, in_s = c + (c % 8 == 0 ? 64 : 0), out_s = c;
            int op = 1, swapped = 0;
            const auto t = make_values(std::size_t(pixels) * in_s, 81 + c);
            const auto cv = make_values(std::size_t(c), 91 + c);
            auto dt = up(t), dcv = up(cv);
            auto dout = driver.allocate(std::size_t(pixels) * out_s * sizeof(Half));
            DeviceAddress ta = dt.address(), cva = dcv.address(), oa = dout.address();
            const bool vec8 = ((c & 7) == 0) && ((in_s & 7) == 0) &&
                ((out_s & 7) == 0) && ((ta & 15U) == 0) &&
                ((cva & 15U) == 0) && ((oa & 15U) == 0);
            const unsigned work = unsigned(pixels * (vec8 ? (c / 8) : c));
            void* bp[] = {&ta, &cva, &oa, &pixels, &c, &in_s, &out_s, &op, &swapped};
            driver.launch(bcast_fn, {(work + 255U) / 256U, 1, 1}, {256, 1, 1}, 0, bp);
            std::vector<Half> actual(std::size_t(pixels) * c), ref(std::size_t(pixels) * c);
            driver.download(actual.data(), oa, actual.size() * sizeof(Half));
            for (int p = 0; p < pixels; ++p)
                for (int ch = 0; ch < c; ++ch)
                    ref[std::size_t(p) * c + ch] = float_to_half(
                        half_to_float(t[std::size_t(p) * in_s + ch]) *
                        half_to_float(cv[ch]));
            check(actual, ref, "mlvc_binary_channel_broadcast_nhwc_fp16", 0.0F);
        }
    }
}

void bench_nhwc_vs_nchw(Driver& driver, Module& module)
{
    constexpr int kRounds = 50;
    constexpr int kBatch = 5;

    std::puts("=== 1. NCHW vs NHWC CUTLASS Pointwise (min/p10 us) ===");
    struct ShapeCase { int oc, sp, ic, ep; };
    const ShapeCase pw_shapes[] = {
        {256, 3680, 256, 2}, {256, 3680, 256, 1}, {256, 3680, 512, 2},
        {384, 3680, 256, 0}, {192, 3680, 192, 2}, {192, 3680, 384, 2},
        {256, 920, 512, 2},  {384, 920, 256, 0},
    };
    for (const auto& s : pw_shapes) {
        const char* nchw_name = s.sp >= 3000
            ? (s.ep == 0 && s.oc >= 384 ? "mlvc_cutlass_pointwise_medium_stage4"
               : s.ep == 2 ? "mlvc_cutlass_pointwise_m64n128_residual"
               : s.ep == 1 ? "mlvc_cutlass_pointwise_m64n128_leaky_relu"
                           : "mlvc_cutlass_pointwise_m64n128")
            : (s.oc == 384 ? "mlvc_cutlass_pointwise"
                           : "mlvc_cutlass_pointwise_m64n64_residual");
        Variant nv{nchw_name,
                   (s.sp >= 3000 && s.ep == 0 && s.oc >= 384) ? 128
                       : (s.sp >= 3000 ? 64 : (s.oc == 384 ? 128 : 64)),
                   (s.sp >= 3000 ? 128 : 64),
                   (s.sp >= 3000 && s.ep == 0 && s.oc >= 384) ? 4 : 3, s.ep, 0};
        auto nchw_st = prepare_pointwise(driver, module, nv, s.oc, s.sp, s.ic);

        std::vector<NhwcVariant> nhwc_vars;
        auto add_v = [&](const char* base, int tm, int tn, int stg) {
            nhwc_vars.push_back({base, tm, tn, stg, s.ep, 128, 0});
        };
        if (s.ep == 2) {
            add_v("mlvc_cutlass_nhwc_m128n64_residual", 128, 64, 3);
            add_v("mlvc_cutlass_nhwc_m128n128_residual", 128, 128, 3);
            add_v("mlvc_cutlass_nhwc_m64n128_residual", 64, 128, 3);
            add_v("mlvc_cutlass_nhwc_m64n64_residual", 64, 64, 3);
        } else if (s.ep == 1) {
            add_v("mlvc_cutlass_nhwc_m128n64_leaky_relu", 128, 64, 3);
            add_v("mlvc_cutlass_nhwc_m128n128_leaky_relu", 128, 128, 3);
            add_v("mlvc_cutlass_nhwc_m64n128_leaky_relu", 64, 128, 3);
            add_v("mlvc_cutlass_nhwc_m64n64_leaky_relu", 64, 64, 3);
        } else {
            add_v("mlvc_cutlass_nhwc_m128n64", 128, 64, 3);
            add_v("mlvc_cutlass_nhwc_m128n128", 128, 128, 3);
            add_v("mlvc_cutlass_nhwc_m128n128_stage4", 128, 128, 4);
            add_v("mlvc_cutlass_nhwc_m64n128", 64, 128, 3);
            add_v("mlvc_cutlass_nhwc_m64n64", 64, 64, 3);
        }
        std::vector<NhwcPrepared> nhwc_sts;
        for (const auto& hv : nhwc_vars)
            nhwc_sts.push_back(prepare_nhwc_pointwise(driver, module, hv, s.sp, s.oc, s.ic));

        std::vector<double> nchw_s(kRounds);
        std::vector<std::vector<double>> nhwc_s(nhwc_sts.size(), std::vector<double>(kRounds));
        for (int r = 0; r < kRounds; ++r) {
            void* a0[] = {nchw_st.params.data()};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kBatch; ++i)
                driver.launch(nchw_st.function, nchw_st.grid, {128, 1, 1}, nchw_st.shared, a0);
            driver.synchronize();
            nchw_s[r] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count() / kBatch;
            for (std::size_t j = 0; j < nhwc_sts.size(); ++j) {
                void* aj[] = {nhwc_sts[j].params.data()};
                auto tj = std::chrono::steady_clock::now();
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(nhwc_sts[j].fn, nhwc_sts[j].grid, nhwc_sts[j].block,
                                  nhwc_sts[j].shared, aj);
                driver.synchronize();
                nhwc_s[j][r] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - tj).count() / kBatch;
            }
        }
        std::sort(nchw_s.begin(), nchw_s.end());
        std::printf("\n[%dx%dx%d ep%d]\n", s.oc, s.sp, s.ic, s.ep);
        std::printf("  NCHW best  min=%6.2f p10=%6.2f  %s\n",
                    nchw_s[0], nchw_s[kRounds / 10], nchw_name);
        for (std::size_t j = 0; j < nhwc_sts.size(); ++j) {
            std::sort(nhwc_s[j].begin(), nhwc_s[j].end());
            std::printf("  NHWC       min=%6.2f p10=%6.2f  blks=%3u  %s\n",
                        nhwc_s[j][0], nhwc_s[j][kRounds / 10],
                        nhwc_sts[j].grid.x * nhwc_sts[j].grid.y, nhwc_vars[j].name);
        }
    }

    std::puts("\n=== 2. NCHW vs NHWC CUTLASS ReGLU (min/p10 us) ===");
    const std::array<int, 3> reglu_shapes[] = {
        {1024, 3680, 256}, {768, 3680, 192}, {1536, 920, 384},
        {1024, 920, 256},  {576, 920, 144}
    };
    for (const auto& s : reglu_shapes) {
        int oc = s[0], sp = s[1], ic = s[2], g = oc / 2;
        const bool s4 = sp >= 3000;
        const char* nchw_rname = s4 ? "mlvc_cutlass_pointwise_medium_stage4_reglu"
                                    : (oc >= 1024 ? "mlvc_cutlass_pointwise_medium_reglu"
                                                  : "mlvc_cutlass_pointwise_m64n64_reglu");
        int ntm = (oc >= 1024 || s4) ? 128 : 64;
        int ntn = (oc >= 1024 || s4) ? 128 : 64;
        unsigned int nsh = s4 ? 65536U : (ntm == 128 ? 49152U : 24576U);
        auto di = driver.allocate(std::size_t(ic) * sp * 2);
        auto dw = driver.allocate(std::size_t(oc) * ic * 2);
        auto db = driver.allocate(std::size_t(oc) * 2);
        auto do_nchw = driver.allocate(std::size_t(g) * sp * 2);
        auto dp = driver.allocate(kCutlassPointwiseParamsStorageBytes);
        std::array<std::byte, kCutlassPointwiseParamsStorageBytes> nparams{};
        auto rinit = module.function(std::string(nchw_rname) + "_init_fp16");
        auto rfn = module.function(std::string(nchw_rname) + "_fp16");
        if (nsh > 49152U) driver.set_max_dynamic_shared_memory(rfn, nsh);
        DeviceAddress pa = dp.address(), ia = di.address(), wa = dw.address(),
                      ba = db.address(), oa = do_nchw.address();
        float lo = 0.0F, hi = 6.0F; int lt = 0;
        void* ria[] = {&pa, &ia, &wa, &ba, &oa, &oc, &sp, &ic, &lo, &hi, &lt};
        driver.launch(rinit, {1, 1, 1}, {1, 1, 1}, 0, ria);
        driver.download(nparams.data(), pa, nparams.size());
        Dim3 ngrid{unsigned((oc + ntm - 1) / ntm), unsigned((sp + ntn - 1) / ntn), 1};

        const NhwcVariant hvs[] = {
            {"mlvc_cutlass_nhwc_m128n128_stage4_reglu", 128, 128, 4, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_reglu", 128, 128, 3, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m64n128_reglu", 64, 128, 3, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m64n64_reglu", 64, 64, 3, 3, 128, 0},
        };
        std::vector<NhwcPrepared> hsts;
        for (const auto& hv : hvs)
            hsts.push_back(prepare_nhwc_pointwise(driver, module, hv, sp, oc, ic));

        std::vector<double> ns(kRounds);
        std::vector<std::vector<double>> hs(hsts.size(), std::vector<double>(kRounds));
        for (int r = 0; r < kRounds; ++r) {
            void* a0[] = {nparams.data()};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < kBatch; ++i)
                driver.launch(rfn, ngrid, {128, 1, 1}, nsh, a0);
            driver.synchronize();
            ns[r] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count() / kBatch;
            for (std::size_t j = 0; j < hsts.size(); ++j) {
                void* aj[] = {hsts[j].params.data()};
                auto tj = std::chrono::steady_clock::now();
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(hsts[j].fn, hsts[j].grid, hsts[j].block,
                                  hsts[j].shared, aj);
                driver.synchronize();
                hs[j][r] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - tj).count() / kBatch;
            }
        }
        std::sort(ns.begin(), ns.end());
        std::printf("\n[ReGLU %dx%dx%d]\n", oc, sp, ic);
        std::printf("  NCHW best  min=%6.2f p10=%6.2f  %s\n", ns[0], ns[kRounds / 10], nchw_rname);
        for (std::size_t j = 0; j < hsts.size(); ++j) {
            std::sort(hs[j].begin(), hs[j].end());
            std::printf("  NHWC       min=%6.2f p10=%6.2f  blks=%3u  %s\n",
                        hs[j][0], hs[j][kRounds / 10],
                        hsts[j].grid.x * hsts[j].grid.y, hvs[j].name);
        }
    }

    std::puts("\n=== 3. NCHW vs NHWC DWConv 3x3 half2-Sliding (h4_u2 / h2_u2) (min us) ===");
    auto oct2_fn = module.function("mlvc_depthwise_conv_oct2_fp16");
    auto nchw_sm_fn = module.function("mlvc_depthwise_conv_3x3_small_fp16");
    auto h4_fn = module.function("mlvc_depthwise_conv_3x3_nhwc_h4_u2_fp16");
    auto h2_fn = module.function("mlvc_depthwise_conv_3x3_nhwc_h2_u2_fp16");
    for (auto d : std::array<std::array<int, 3>, 8>{{
             {256, 46, 80}, {192, 46, 80}, {256, 23, 40}, {384, 23, 40},
             {144, 23, 40}, {128, 12, 20}, {128, 6, 10},  {128, 3, 5}}}) {
        int c = d[0], h = d[1], w = d[2], n = 1;
        auto di = driver.allocate(std::size_t(c) * h * w * 2);
        auto dw = driver.allocate(std::size_t(c) * 9 * 2);
        auto db = driver.allocate(std::size_t(c) * 2);
        auto do1 = driver.allocate(std::size_t(c) * h * w * 2);
        DeviceAddress ia = di.address(), wa = dw.address(), ba = db.address(), oa = do1.address();
        void* oct_p[] = {&ia, &wa, &ba, &oa, &n, &c, &h, &w, &h, &w};
        void* nsm_p[] = {&ia, &wa, &ba, &oa, &n, &c, &h, &w};
        const unsigned cg = unsigned(c / 4);
        const unsigned chunk_thr = (cg % 32U == 0U) ? 32U
                                 : (cg % 16U == 0U) ? 16U
                                 : std::min(cg, 32U);
        const unsigned tpb = (chunk_thr == 16U) ? 2U : (w >= 80 ? 2U : 1U);
        int c_chunks = int((cg + chunk_thr - 1) / chunk_thr);
        void* nh_p[] = {&ia, &wa, &ba, &oa, &n, &c, &h, &w, &c, &c, &c_chunks};

        std::size_t pairs = std::size_t((h + 1) / 2) * (w / 8);
        unsigned ob = pairs <= 64 ? 64U : 128U;
        Dim3 ogrid{unsigned((pairs + ob - 1) / ob), unsigned(c), 1};
        Dim3 g4{unsigned(((w + 3) / 4 + tpb - 1) / tpb), unsigned((h + 1) / 2), unsigned(c_chunks)};
        Dim3 g2{unsigned(((w + 1) / 2 + tpb - 1) / tpb), unsigned((h + 1) / 2), unsigned(c_chunks)};

        std::vector<double> s_nc(kRounds), s_h4(kRounds), s_h2(kRounds);
        for (int r = 0; r < kRounds; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            if (w % 8 == 0) {
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(oct2_fn, ogrid, {ob, 1, 1}, 0, oct_p);
            } else {
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(nchw_sm_fn, {unsigned((c + 1) / 2), 1, 1}, {32, 2, 1}, 0, nsm_p);
            }
            driver.synchronize();
            s_nc[r] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count() / kBatch;

            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < kBatch; ++i)
                driver.launch(h4_fn, g4, {chunk_thr, tpb, 1}, 0, nh_p);
            driver.synchronize();
            s_h4[r] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t1).count() / kBatch;

            auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < kBatch; ++i)
                driver.launch(h2_fn, g2, {chunk_thr, tpb, 1}, 0, nh_p);
            driver.synchronize();
            s_h2[r] = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t2).count() / kBatch;
        }
        std::sort(s_nc.begin(), s_nc.end());
        std::sort(s_h4.begin(), s_h4.end());
        std::sort(s_h2.begin(), s_h2.end());
        std::printf("  [DW %3dx%2dx%2d] nchw=%5.2f | h4=%5.2f (%3u) | h2=%5.2f (%3u)\n",
                    c, h, w, s_nc[0],
                    s_h4[0], g4.x * g4.y * g4.z,
                    s_h2[0], g2.x * g2.y * g2.z);
    }

    {
        std::puts("\n=== 4. Compact sp<920 NHWC: tiny_nhwc vs CUTLASS m64n64 (min/p10 us) ===");
        auto tiny_fn = module.function("mlvc_pointwise_conv_mma_tiny_nhwc_fp16");
        float al = 0.01F;
        for (auto s : std::array<std::array<int, 3>, 5>{{
                 {128, 15, 128}, {768, 60, 48}, {128, 240, 128},
                 {256, 240, 128}, {512, 240, 128}}}) {
            int oc = s[0], sp = s[1], ic = s[2], n = 1, ep = 0;
            auto dt_in = driver.allocate(std::size_t(sp) * ic * 2);
            auto dt_w = driver.allocate(std::size_t(oc) * ic * 2);
            auto dt_b = driver.allocate(std::size_t(oc) * 2);
            auto dt_o = driver.allocate(std::size_t(sp) * oc * 2);
            DeviceAddress tia = dt_in.address(), twa = dt_w.address(),
                          tba = dt_b.address(), toa = dt_o.address(), tra = 0;
            void* tp[] = {&tia, &twa, &tba, &toa, &n, &ic, &sp, &oc, &ic, &oc, &oc, &tra, &ep, &al};
            Dim3 tgrid{unsigned((sp + 15) / 16), unsigned((oc + 15) / 16), 1};
            NhwcVariant cv{"mlvc_cutlass_nhwc_m64n64", 64, 64, 3, 0, 128, 0};
            auto cprep = prepare_nhwc_pointwise(driver, module, cv, sp, oc, ic);
            std::vector<double> st(kRounds), sc(kRounds);
            for (int r = 0; r < kRounds; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(tiny_fn, tgrid, {32, 4, 1}, 0, tp);
                driver.synchronize();
                st[r] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count() / kBatch;
                void* ca[] = {cprep.params.data()};
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < kBatch; ++i)
                    driver.launch(cprep.fn, cprep.grid, cprep.block, cprep.shared, ca);
                driver.synchronize();
                sc[r] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t1).count() / kBatch;
            }
            std::sort(st.begin(), st.end());
            std::sort(sc.begin(), sc.end());
            std::printf("  [PW %3dx%3dx%3d]  tiny_nhwc min=%5.2f p10=%5.2f | m64n64 min=%5.2f p10=%5.2f\n",
                        oc, sp, ic, st[0], st[kRounds / 10], sc[0], sc[kRounds / 10]);
        }
    }
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
        check_native_pixel_layout(driver, module);
        const Variant variants[] = {
            {"mlvc_cutlass_pointwise", 128, 64, 3, 0},
            {"mlvc_cutlass_pointwise_leaky_relu", 128, 64, 3, 1},
            {"mlvc_cutlass_pointwise_residual", 128, 64, 3, 2},
            {"mlvc_cutlass_pointwise_medium", 128, 128, 3, 0},
            {"mlvc_cutlass_pointwise_medium_stage4", 128, 128, 4, 0},
            {"mlvc_cutlass_pointwise_medium_leaky_relu", 128, 128, 3, 1},
            {"mlvc_cutlass_pointwise_medium_residual", 128, 128, 3, 2},
            {"mlvc_cutlass_pointwise_spatial_wide", 64, 256, 3, 0},
            {"mlvc_cutlass_pointwise_spatial_wide_leaky_relu", 64, 256, 3, 1},
            {"mlvc_cutlass_pointwise_spatial_wide_residual", 64, 256, 3, 2},
            {"mlvc_cutlass_pointwise_m64n128", 64, 128, 3, 0},
            {"mlvc_cutlass_pointwise_m64n128_leaky_relu", 64, 128, 3, 1},
            {"mlvc_cutlass_pointwise_m64n128_residual", 64, 128, 3, 2},
            {"mlvc_cutlass_pointwise_m64n64", 64, 64, 3, 0},
            {"mlvc_cutlass_pointwise_m64n64_leaky_relu", 64, 64, 3, 1},
            {"mlvc_cutlass_pointwise_m64n64_residual", 64, 64, 3, 2},
            {"mlvc_cutlass_pointwise", 128, 64, 3, 0, 3},
            {"mlvc_cutlass_pointwise_leaky_relu", 128, 64, 3, 1, 3},
            {"mlvc_cutlass_pointwise_residual", 128, 64, 3, 2, 3},
            {"mlvc_cutlass_pointwise_medium", 128, 128, 3, 0, 3},
            {"mlvc_cutlass_pointwise_medium_leaky_relu", 128, 128, 3, 1,
             3},
            {"mlvc_cutlass_pointwise_medium_residual", 128, 128, 3, 2,
             3},
            {"mlvc_cutlass_pointwise_spatial_wide", 64, 256, 3, 0, 3},
            {"mlvc_cutlass_pointwise_spatial_wide_leaky_relu", 64, 256, 3, 1,
             3},
            {"mlvc_cutlass_pointwise_spatial_wide_residual", 64, 256, 3, 2,
             3},
            {"mlvc_cutlass_pointwise_m64n128", 64, 128, 3, 0, 3},
            {"mlvc_cutlass_pointwise_m64n128_leaky_relu", 64, 128, 3, 1, 3},
            {"mlvc_cutlass_pointwise_m64n128_residual", 64, 128, 3, 2, 3},
            {"mlvc_cutlass_pointwise_m64n64", 64, 64, 3, 0, 3},
            {"mlvc_cutlass_pointwise_m64n64_leaky_relu", 64, 64, 3, 1, 3},
            {"mlvc_cutlass_pointwise_m64n64_residual", 64, 64, 3, 2, 3},
        };
        if (argc > 1 && std::string(argv[1]) == "--bench-nhwc") {
            bench_nhwc_vs_nchw(driver, module);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--bench-tiles") {
            bench_hot_tiles(driver, module);
            return 0;
        }
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
        for (bool small_tile : {false, true}) {
            spatial(driver, module, 24, 24, 1, 3, 1, small_tile);
            spatial(driver, module, 40, 264, 2, 3, 1, small_tile);
            spatial(driver, module, 8, 8, 2, 2, 0, small_tile);
            spatial(driver, module, 128, 256, 2, 2, 0, small_tile);
        }
        for (int epilogue = 0; epilogue <= 2; ++epilogue) {
            check_tiny_pointwise(driver, module, 128, 15, 128, epilogue);
            check_tiny_pointwise(driver, module, 256, 60, 256, epilogue);
            check_tiny_pointwise(driver, module, 768, 60, 48, epilogue);
            check_tiny_pointwise(driver, module, 128, 240, 256, epilogue);
            check_tiny_pointwise(driver, module, 16, 15, 32, epilogue);
        }
        check_tiny_reglu(driver, module, 256, 15, 128);
        check_tiny_reglu(driver, module, 256, 60, 256);
        check_tiny_reglu(driver, module, 192, 60, 48);
        check_tiny_reglu(driver, module, 256, 240, 128);
        check_dwconv_3x3(driver, module, 32, 46, 80);
        check_dwconv_3x3(driver, module, 32, 23, 40);
        check_dwconv_3x3(driver, module, 32, 24, 40);
        check_dwconv_3x3(driver, module, 32, 12, 20);
        check_dwconv_3x3(driver, module, 32, 6, 10);
        check_dwconv_3x3(driver, module, 32, 3, 5);
        for (const auto& rs : std::array<std::array<int, 3>, 4>{
                 {{72, 920, 48}, {288, 920, 144}, {256, 960, 128}, {384, 256, 192}}}) {
            check_cutlass_reglu(driver, module, "mlvc_cutlass_pointwise_reglu",
                                128, 64, 3, rs[0], rs[1], rs[2]);
            check_cutlass_reglu(driver, module, "mlvc_cutlass_pointwise_medium_reglu",
                                128, 128, 3, rs[0], rs[1], rs[2]);
            check_cutlass_reglu(driver, module, "mlvc_cutlass_pointwise_medium_stage4_reglu",
                                128, 128, 4, rs[0], rs[1], rs[2]);
            check_cutlass_reglu(driver, module, "mlvc_cutlass_pointwise_m64n64_reglu",
                                64, 64, 3, rs[0], rs[1], rs[2]);
            check_cutlass_reglu(driver, module, "mlvc_cutlass_pointwise_medium_reglu",
                                128, 128, 3, rs[0], rs[1], rs[2], 2);
        }
        // Verify NHWC CUTLASS pointwise + ReGLU (including strided lda/ldd),
        // 3-tier NHWC DWConv 3x3, tiny NHWC MMA, and vectorized NHWC elementwise.
        const NhwcVariant nhwc_checks[] = {
            {"mlvc_cutlass_nhwc_m128n64", 128, 64, 3, 0, 128, 0},
            {"mlvc_cutlass_nhwc_m128n64_leaky_relu", 128, 64, 3, 1, 128, 0},
            {"mlvc_cutlass_nhwc_m128n64_residual", 128, 64, 3, 2, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128", 128, 128, 3, 0, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_stage4", 128, 128, 4, 0, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_leaky_relu", 128, 128, 3, 1, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_residual", 128, 128, 3, 2, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_reglu", 128, 128, 3, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m128n128_stage4_reglu", 128, 128, 4, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m64n128", 64, 128, 3, 0, 128, 0},
            {"mlvc_cutlass_nhwc_m64n128_leaky_relu", 64, 128, 3, 1, 128, 0},
            {"mlvc_cutlass_nhwc_m64n128_residual", 64, 128, 3, 2, 128, 0},
            {"mlvc_cutlass_nhwc_m64n128_reglu", 64, 128, 3, 3, 128, 0},
            {"mlvc_cutlass_nhwc_m64n64", 64, 64, 3, 0, 128, 0},
            {"mlvc_cutlass_nhwc_m64n64_leaky_relu", 64, 64, 3, 1, 128, 0},
            {"mlvc_cutlass_nhwc_m64n64_residual", 64, 64, 3, 2, 128, 0},
            {"mlvc_cutlass_nhwc_m64n64_reglu", 64, 64, 3, 3, 128, 0},
        };
        for (const auto& hv : nhwc_checks) {
            check_nhwc_cutlass(driver, module, hv, 135, 144, 48, 0, 0);
            check_nhwc_cutlass(driver, module, hv, 256, 256, 128, 128, 64);
        }
        check_nhwc_kernels(driver, module);
        std::printf("CUTLASS kernels: %d GEMM and 8 convolution cases passed"
                    " (max abs error %.5f)\n",
                    int(sizeof(variants) / sizeof(variants[0]) * 6),
                    max_error);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
