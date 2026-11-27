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
        std::printf("CUTLASS kernels: %d GEMM and 8 convolution cases passed"
                    " (max abs error %.5f)\n",
                    int(sizeof(variants) / sizeof(variants[0]) * 6),
                    max_error);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
