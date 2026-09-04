#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace {

__device__ __forceinline__ int offset4(int linear, int d1, int d2, int d3,
                                       int s0, int s1, int s2, int s3)
{
    const int c3 = linear % d3;
    linear /= d3;
    const int c2 = linear % d2;
    linear /= d2;
    const int c1 = linear % d1;
    const int c0 = linear / d1;
    return c0 * s0 + c1 * s1 + c2 * s2 + c3 * s3;
}

}  // namespace

extern "C" __global__ void mlvc_binary_fp16(
    const __half* lhs, const __half* rhs, __half* output, int count, int op,
    int od0, int od1, int od2, int od3,
    int ld0, int ld1, int ld2, int ld3,
    int rd0, int rd1, int rd2, int rd3)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int lhs_index = offset4(
        index, od1, od2, od3,
        ld0 == 1 ? 0 : ld1 * ld2 * ld3,
        ld1 == 1 ? 0 : ld2 * ld3,
        ld2 == 1 ? 0 : ld3,
        ld3 == 1 ? 0 : 1);
    const int rhs_index = offset4(
        index, od1, od2, od3,
        rd0 == 1 ? 0 : rd1 * rd2 * rd3,
        rd1 == 1 ? 0 : rd2 * rd3,
        rd2 == 1 ? 0 : rd3,
        rd3 == 1 ? 0 : 1);
    const float a = __half2float(lhs[lhs_index]);
    const float b = __half2float(rhs[rhs_index]);
    float value = 0.0F;
    if (op == 0)
        value = a + b;
    else if (op == 1)
        value = a * b;
    else
        value = a - b;
    output[index] = __float2half_rn(value);
}

__device__ __forceinline__ __half binary_value(__half lhs, __half rhs, int op)
{
    const float a = __half2float(lhs);
    const float b = __half2float(rhs);
    return __float2half_rn(op == 0 ? a + b : (op == 1 ? a * b : a - b));
}

extern "C" __global__ void mlvc_binary_contiguous_fp16(
    const __half* lhs, const __half* rhs, __half* output, int count, int op)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int index = pair * 2;
    if (index >= count)
        return;
    if (index + 1 < count) {
        const __half2 a = reinterpret_cast<const __half2*>(lhs)[pair];
        const __half2 b = reinterpret_cast<const __half2*>(rhs)[pair];
        reinterpret_cast<__half2*>(output)[pair] = __halves2half2(
            binary_value(__low2half(a), __low2half(b), op),
            binary_value(__high2half(a), __high2half(b), op));
    } else {
        output[index] = binary_value(lhs[index], rhs[index], op);
    }
}

extern "C" __global__ void mlvc_unary_fp16(
    const __half* input, __half* output, int count, int op,
    float alpha, float minimum, float maximum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const float x = __half2float(input[index]);
    float value = x;
    if (op == 0)
        value = x >= 0.0F ? x : alpha * x;
    else if (op == 1)
        value = fminf(fmaxf(x, minimum), maximum);
    else if (op == 2)
        value = 1.0F / (1.0F + expf(-x));
    else if (op == 3)
        value = 1.0F / x;
    else if (op == 4)
        value = nearbyintf(x);
    output[index] = __float2half_rn(value);
}

// Fused recurrent-feature update used by the encoder tail.  The exported graph
// expresses this as three slices, two sigmoids, six binary operations, and a
// concat.  Keeping the fp16 rounding points here preserves those operator
// boundaries while avoiding eleven intermediate tensor round trips.
extern "C" __global__ void mlvc_feature_update_fp16(
    const __half* input, const __half* history, const __half* scale,
    __half* output,
    int batch_count, int channels, int spatial_count)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pairs_per_plane = (spatial_count + 1) / 2;
    const int plane = pair / pairs_per_plane;
    const int spatial = (pair - plane * pairs_per_plane) * 2;
    if (plane >= batch_count * channels || spatial >= spatial_count)
        return;

    const int batch = plane / channels;
    const int channel = plane - batch * channels;
    const std::size_t input_base =
        (static_cast<std::size_t>(batch) * channels * 3 + channel) *
        spatial_count + spatial;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) *
        spatial_count + spatial;
    const std::size_t channel_stride =
        static_cast<std::size_t>(channels) * spatial_count;
    const __half one = __float2half_rn(1.0F);
    const __half channel_scale = scale[channel];

#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
        if (spatial + offset >= spatial_count)
            break;
        const __half a = input[input_base + offset];
        const __half b = input[input_base + channel_stride + offset];
        const __half c = input[input_base + channel_stride * 2 + offset];
        const __half previous = history[
            static_cast<std::size_t>(plane) * spatial_count + spatial + offset];
        const __half sigmoid_b = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(b))));
        const __half sigmoid_c = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(c))));
        const __half weighted_b = binary_value(sigmoid_b, previous, 1);
        const __half inverse_gate = binary_value(one, sigmoid_b, 2);
        const __half weighted_a = binary_value(inverse_gate, a, 1);
        const __half blended = binary_value(weighted_b, weighted_a, 0);
        const __half gated = binary_value(sigmoid_c, blended, 1);
        output[output_base + offset] =
            binary_value(gated, channel_scale, 1);
        output[output_base + channel_stride + offset] = blended;
    }
}

extern "C" __global__ void mlvc_transpose_fp16(
    const __half* input, __half* output, int rows, int columns)
{
    __shared__ __half tile[32][33];
    int x = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
    int y = static_cast<int>(blockIdx.y) * 32 + threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        if (x < columns && y + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(y + offset) * columns + x];
        }
    }
    __syncthreads();

    x = static_cast<int>(blockIdx.y) * 32 + threadIdx.x;
    y = static_cast<int>(blockIdx.x) * 32 + threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        if (x < rows && y + offset < columns) {
            output[(y + offset) * rows + x] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

extern "C" __global__ void mlvc_oihw_to_krsc_fp16(
    const __half* input, __half* output, int out_channels, int in_channels,
    int kernel_height, int kernel_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count =
        out_channels * in_channels * kernel_height * kernel_width;
    if (index >= count)
        return;

    int linear = index;
    const int input_channel = linear % in_channels;
    linear /= in_channels;
    const int kernel_x = linear % kernel_width;
    linear /= kernel_width;
    const int kernel_y = linear % kernel_height;
    const int output_channel = linear / kernel_height;
    output[index] = input[
        ((output_channel * in_channels + input_channel) * kernel_height +
         kernel_y) * kernel_width + kernel_x];
}

__device__ __forceinline__ __half reglu_value(
    __half gate, __half linear, float minimum, float maximum)
{
    const float clipped = fminf(fmaxf(__half2float(gate), minimum), maximum);
    const __half rounded_clip = __float2half_rn(clipped);
    return __float2half_rn(__half2float(rounded_clip) * __half2float(linear));
}

extern "C" __global__ void mlvc_reglu_fp16(
    const __half* input, __half* output, int batch_count, int channels,
    int spatial_count, float minimum, float maximum)
{
    const bool vectorized = (spatial_count & 1) == 0;
    const int work_count = vectorized ? spatial_count / 2 : spatial_count;
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= work_count || channel >= channels || batch >= batch_count)
        return;

    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;

    if (vectorized) {
        const __half2 gates = reinterpret_cast<const __half2*>(
            input + gate_base)[spatial];
        const __half2 linears = reinterpret_cast<const __half2*>(
            input + linear_base)[spatial];
        reinterpret_cast<__half2*>(output + output_base)[spatial] =
            __halves2half2(
                reglu_value(__low2half(gates), __low2half(linears),
                            minimum, maximum),
                reglu_value(__high2half(gates), __high2half(linears),
                            minimum, maximum));
    } else {
        output[output_base + spatial] =
            reglu_value(input[gate_base + spatial],
                        input[linear_base + spatial], minimum, maximum);
    }
}

struct __align__(16) RegLuHalf8 {
    __half2 values[4];
};

extern "C" __global__ void mlvc_reglu_vec8_fp16(
    const __half* input, __half* output, int batch_count, int channels,
    int spatial_count, float minimum, float maximum)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int chunks_per_channel = spatial_count / 8;
    const int count = batch_count * channels * chunks_per_channel;
    if (index >= count)
        return;

    int linear = index;
    const int chunk = linear % chunks_per_channel;
    linear /= chunks_per_channel;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;
    const RegLuHalf8 gates = *reinterpret_cast<const RegLuHalf8*>(
        input + gate_base + chunk * 8);
    const RegLuHalf8 linears = *reinterpret_cast<const RegLuHalf8*>(
        input + linear_base + chunk * 8);
    RegLuHalf8 result;

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        result.values[item] = __halves2half2(
            reglu_value(__low2half(gates.values[item]),
                        __low2half(linears.values[item]), minimum, maximum),
            reglu_value(__high2half(gates.values[item]),
                        __high2half(linears.values[item]), minimum, maximum));
    }
    *reinterpret_cast<RegLuHalf8*>(output + output_base + chunk * 8) = result;
}

extern "C" __global__ void mlvc_reglu_bias_fp16(
    const __half* input, const __half* bias, __half* output,
    int batch_count, int channels, int spatial_count,
    float minimum, float maximum)
{
    const bool vectorized = (spatial_count & 1) == 0;
    const int work_count = vectorized ? spatial_count / 2 : spatial_count;
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= work_count || channel >= channels || batch >= batch_count)
        return;

    const std::size_t plane = static_cast<std::size_t>(spatial_count);
    const std::size_t gate_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) * plane;
    const std::size_t linear_base = gate_base + channels * plane;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels + channel) * plane;
    const float gate_bias = bias ? __half2float(bias[channel]) : 0.0F;
    const float linear_bias =
        bias ? __half2float(bias[channel + channels]) : 0.0F;

    auto apply = [&](const __half gate, const __half linear) {
        const __half biased_gate =
            __float2half_rn(__half2float(gate) + gate_bias);
        const __half biased_linear =
            __float2half_rn(__half2float(linear) + linear_bias);
        return reglu_value(biased_gate, biased_linear, minimum, maximum);
    };

    if (vectorized) {
        const __half2 gates = reinterpret_cast<const __half2*>(
            input + gate_base)[spatial];
        const __half2 linears = reinterpret_cast<const __half2*>(
            input + linear_base)[spatial];
        reinterpret_cast<__half2*>(output + output_base)[spatial] =
            __halves2half2(
                apply(__low2half(gates), __low2half(linears)),
                apply(__high2half(gates), __high2half(linears)));
    } else {
        output[output_base + spatial] =
            apply(input[gate_base + spatial], input[linear_base + spatial]);
    }
}

extern "C" __global__ void mlvc_conv_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int n_count, int in_channels, int input_height,
    int input_width, int out_channels, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width, int groups)
{
    constexpr int kSpatialTile = 32;
    constexpr int kChannelTile = 4;
    constexpr int kReductionTile = 8;
    __shared__ __half weight_tile[kChannelTile][kReductionTile];

    const int spatial = static_cast<int>(blockIdx.x) * kSpatialTile +
                        static_cast<int>(threadIdx.x);
    const int output_channel = static_cast<int>(blockIdx.y) * kChannelTile +
                               static_cast<int>(threadIdx.y);
    const int output_plane = output_height * output_width;
    const int spatial_count = n_count * output_plane;
    const bool valid = spatial < spatial_count && output_channel < out_channels;
    const int channels_per_group = in_channels / groups;
    const int outputs_per_group = out_channels / groups;
    const int reduction = channels_per_group * kernel_height * kernel_width;
    float accumulator = valid && bias ? __half2float(bias[output_channel]) : 0.0F;

    int n = 0;
    int output_y = 0;
    int output_x = 0;
    int input_channel_base = 0;
    if (valid) {
        n = spatial / output_plane;
        const int plane_index = spatial - n * output_plane;
        output_y = plane_index / output_width;
        output_x = plane_index - output_y * output_width;
        input_channel_base = (output_channel / outputs_per_group) * channels_per_group;
    }

    for (int base = 0; base < reduction; base += kReductionTile) {
        if (threadIdx.x < kReductionTile) {
            const int k = base + static_cast<int>(threadIdx.x);
            weight_tile[threadIdx.y][threadIdx.x] =
                output_channel < out_channels && k < reduction
                    ? weight[output_channel * reduction + k]
                    : __float2half(0.0F);
        }
        __syncthreads();
        if (valid) {
#pragma unroll
            for (int inner = 0; inner < kReductionTile; ++inner) {
                const int k = base + inner;
                if (k >= reduction)
                    break;
                const int kernel_index = k % (kernel_height * kernel_width);
                const int input_channel = input_channel_base +
                    k / (kernel_height * kernel_width);
                const int kernel_y = kernel_index / kernel_width;
                const int kernel_x = kernel_index - kernel_y * kernel_width;
                const int input_y = output_y * stride_height + kernel_y - pad_height;
                const int input_x = output_x * stride_width + kernel_x - pad_width;
                if (input_y >= 0 && input_y < input_height &&
                    input_x >= 0 && input_x < input_width) {
                    const int input_index =
                        ((n * in_channels + input_channel) * input_height + input_y) *
                            input_width + input_x;
                    accumulator = fmaf(__half2float(input[input_index]),
                                       __half2float(weight_tile[threadIdx.y][inner]),
                                       accumulator);
                }
            }
        }
        __syncthreads();
    }
    if (valid) {
        const int plane_index = spatial - n * output_plane;
        const int output_index = (n * out_channels + output_channel) * output_plane +
                                 plane_index;
        output[output_index] = __float2half_rn(accumulator);
    }
}

// Treat a 1x1 convolution as W[OC, IC] * X[IC, N*H*W]. Four warps share
// each input tile while producing 64 output channels with FP32 accumulation.
extern "C" __global__ void mlvc_pointwise_conv_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels)
{
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kWarps = 4;
    constexpr int kOutputChannelsPerBlock = kTile * kWarps;

    __shared__ __align__(32) __half input_tile[kTile * kTile];
    __shared__ __align__(32) float accumulator_tiles[kWarps][kTile * kTile];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial_base = static_cast<int>(blockIdx.x) * kTile;
    const int channel_base =
        static_cast<int>(blockIdx.y) * kOutputChannelsPerBlock + warp * kTile;

    if (batch >= batch_count || channel_base >= out_channels)
        return;

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                   __half, wmma::row_major> weight_fragment;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                   __half, wmma::row_major> input_fragment;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        accumulator_fragment;

    for (int index = lane; index < kTile * kTile; index += 32) {
        const int row = index / kTile;
        accumulator_tiles[warp][index] = bias
            ? __half2float(bias[channel_base + row]) : 0.0F;
    }
    __syncwarp();
    wmma::load_matrix_sync(accumulator_fragment, accumulator_tiles[warp],
                           kTile, wmma::mem_row_major);

    for (int channel = 0; channel < in_channels; channel += kTile) {
        for (int index = linear_thread; index < kTile * kTile;
             index += 32 * kWarps) {
            const int row = index / kTile;
            const int column = index - row * kTile;
            const int spatial = spatial_base + column;
            input_tile[index] = spatial < spatial_count
                ? input[(batch * in_channels + channel + row) * spatial_count +
                        spatial]
                : __float2half(0.0F);
        }
        __syncthreads();

        wmma::load_matrix_sync(
            weight_fragment,
            weight + channel_base * in_channels + channel,
            in_channels);
        wmma::load_matrix_sync(input_fragment, input_tile, kTile);
        wmma::mma_sync(accumulator_fragment, weight_fragment, input_fragment,
                       accumulator_fragment);
        __syncthreads();
    }

    wmma::store_matrix_sync(accumulator_tiles[warp], accumulator_fragment,
                            kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTile * kTile; index += 32) {
        const int row = index / kTile;
        const int column = index - row * kTile;
        const int spatial = spatial_base + column;
        if (spatial < spatial_count) {
            output[(batch * out_channels + channel_base + row) * spatial_count +
                   spatial] = __float2half_rn(accumulator_tiles[warp][index]);
        }
    }
}

struct __align__(32) PointwiseOperandTile {
    __half weight[128 * 16];
    __half input[16 * 128];
};

struct __align__(32) PointwiseOperandTile64 {
    __half weight[128 * 16];
    __half input[16 * 72];
};

struct __align__(32) PointwiseOperandSwizzled {
    __half weight[128 * 16];
    __half input[16 * 64];
};

struct __align__(32) PointwiseOperandSwizzled64 {
    __half weight[64 * 16];
    __half input[16 * 64];
};

struct __align__(32) SpatialOperandK32 {
    PointwiseOperandSwizzled halves[2];
};

__device__ __forceinline__ void copy_async_16(
    __half* destination, const __half* source, int valid_halves)
{
#if __CUDA_ARCH__ >= 800
    const auto source_address = reinterpret_cast<unsigned long long>(source);
    if (valid_halves == 8 && (source_address & 15ULL) == 0) {
        const unsigned int shared_address =
            static_cast<unsigned int>(__cvta_generic_to_shared(destination));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     : : "r"(shared_address), "l"(source) : "memory");
    } else {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
            destination[index] = index < valid_halves
                ? source[index] : __float2half(0.0F);
        }
    }
#else
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        destination[index] = index < valid_halves
            ? source[index] : __float2half(0.0F);
    }
#endif
}

__device__ __forceinline__ void commit_async_copies()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : : "memory");
#endif
}

__device__ __forceinline__ void wait_for_async_copies()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" : : : "memory");
#endif
}

__device__ __forceinline__ void load_pointwise_tile(
    PointwiseOperandTile& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + weight_row * 16 + weight_column,
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    const int input_row = linear_thread / 16;
    const int input_column = (linear_thread % 16) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves = remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + input_row * 128 + input_column,
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

__device__ __forceinline__ void load_pointwise_tile64(
    PointwiseOperandTile64& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + weight_row * 16 + weight_column,
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + input_row * 72 + input_column,
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ int swizzle_input_offset(int row, int column)
{
    const int chunk = column / 8;
    return row * 64 + ((chunk ^ (row & 7)) * 8) + (column & 7);
}

__device__ __forceinline__ int swizzle_weight_offset(int row, int column)
{
    return row * 16 + (column ^ ((row & 4) * 2));
}

__device__ __forceinline__ void load_pointwise_tile_swizzled(
    PointwiseOperandSwizzled& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int channel, int in_channels,
    int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + swizzle_input_offset(input_row, input_column),
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ void load_pointwise_tile_swizzled64(
    PointwiseOperandSwizzled64& tile, const __half* input,
    const __half* weight, int batch, int block_channel_base, int channel,
    int in_channels, int spatial_base, int spatial_count, int linear_thread)
{
    if (linear_thread >= 128)
        return;

    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + (block_channel_base + weight_row) * in_channels + channel +
            weight_column,
        8);

    const int input_row = linear_thread / 8;
    const int input_column = (linear_thread % 8) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves =
        remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + swizzle_input_offset(input_row, input_column),
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

__device__ __forceinline__ void load_pointwise_reglu_tile_swizzled(
    PointwiseOperandSwizzled& tile, const __half* input, const __half* weight,
    int batch, int block_channel_base, int gate_channels, int channel,
    int in_channels, int spatial_base, int spatial_count, int linear_thread)
{
    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    const int output_channel = weight_row < 64
        ? block_channel_base + weight_row
        : gate_channels + block_channel_base + weight_row - 64;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + output_channel * in_channels + channel + weight_column, 8);

    if (linear_thread < 128) {
        const int input_row = linear_thread / 8;
        const int input_column = (linear_thread % 8) * 8;
        const int remaining = spatial_count - spatial_base - input_column;
        const int valid_halves =
            remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
        copy_async_16(
            tile.input + swizzle_input_offset(input_row, input_column),
            input + (batch * in_channels + channel + input_row) * spatial_count +
                spatial_base + input_column,
            valid_halves);
    }
}

__device__ __forceinline__ void load_pointwise_reglu_tile_swizzled_small(
    PointwiseOperandSwizzled64& tile, const __half* input,
    const __half* weight, int batch, int block_channel_base, int gate_channels,
    int channel, int in_channels, int spatial_base, int spatial_count,
    int linear_thread)
{
    if (linear_thread >= 128)
        return;

    const int weight_row = linear_thread / 2;
    const int weight_column = (linear_thread % 2) * 8;
    const int output_channel = weight_row < 32
        ? block_channel_base + weight_row
        : gate_channels + block_channel_base + weight_row - 32;
    copy_async_16(
        tile.weight + swizzle_weight_offset(weight_row, weight_column),
        weight + output_channel * in_channels + channel + weight_column, 8);

    const int input_row = linear_thread / 8;
    const int input_column = (linear_thread % 8) * 8;
    const int remaining = spatial_count - spatial_base - input_column;
    const int valid_halves =
        remaining <= 0 ? 0 : (remaining < 8 ? remaining : 8);
    copy_async_16(
        tile.input + swizzle_input_offset(input_row, input_column),
        input + (batch * in_channels + channel + input_row) * spatial_count +
            spatial_base + input_column,
        valid_halves);
}

struct MmaAccumulator {
    float values[4];
};

__device__ __forceinline__ void load_matrix_x4(
    const __half* source, unsigned int (&values)[4])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void load_matrix_x2_transpose(
    const __half* source, unsigned int (&values)[2])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(values[0]), "=r"(values[1])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void load_matrix_x4_transpose(
    const __half* source, unsigned int (&values)[4])
{
#if __CUDA_ARCH__ >= 800
    const unsigned int address =
        static_cast<unsigned int>(__cvta_generic_to_shared(source));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
        : "r"(address) : "memory");
#endif
}

__device__ __forceinline__ void mma_m16n8k16(
    MmaAccumulator& accumulator, const unsigned int (&a)[4],
    const unsigned int* b)
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(accumulator.values[0]), "+f"(accumulator.values[1]),
          "+f"(accumulator.values[2]), "+f"(accumulator.values[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ __half pointwise_epilogue_value(
    float accumulator, const __half* residual, int epilogue, float alpha)
{
    const __half rounded = __float2half_rn(accumulator);
    if (epilogue == 1) {
        const float value = __half2float(rounded);
        return __float2half_rn(value >= 0.0F ? value : alpha * value);
    }
    if (epilogue == 2)
        return binary_value(rounded, *residual, 0);
    return rounded;
}

extern "C" __global__ void mlvc_pointwise_postprocess_fp16(
    const __half* input, const __half* bias, const __half* residual,
    __half* output, int batch_count, int channels, int spatial_count,
    int epilogue, float alpha)
{
    const bool vectorized = (spatial_count & 1) == 0;
    const int work_count = vectorized ? spatial_count / 2 : spatial_count;
    const int spatial = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int channel = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    if (spatial >= work_count || channel >= channels || batch >= batch_count)
        return;

    const std::size_t base =
        (static_cast<std::size_t>(batch) * channels + channel) * spatial_count;
    const float channel_bias = bias ? __half2float(bias[channel]) : 0.0F;
    const int element = vectorized ? spatial * 2 : spatial;
    const __half* source = input + base + element;
    __half* destination = output + base + element;
    const __half* residual_source =
        epilogue == 2 ? residual + base + element : nullptr;

    const __half first = pointwise_epilogue_value(
        __half2float(source[0]) + channel_bias,
        residual_source, epilogue, alpha);
    if (vectorized) {
        const __half second = pointwise_epilogue_value(
            __half2float(source[1]) + channel_bias,
            residual_source ? residual_source + 1 : nullptr, epilogue, alpha);
        *reinterpret_cast<__half2*>(destination) =
            __halves2half2(first, second);
    } else {
        destination[0] = first;
    }
}

template <int kBlockRows, typename OperandTile>
__device__ __forceinline__ void pointwise_conv_mma_body(
    OperandTile (&operand_tiles)[2],
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels, const __half* epilogue_input, int epilogue, float alpha)
{
    constexpr int kBlockColumns = 64;
    constexpr int kWarpRows = kBlockRows / 32;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial_base = static_cast<int>(blockIdx.x) * kBlockColumns;
    const int block_channel_base = static_cast<int>(blockIdx.y) * kBlockRows;
    const int channel_base = block_channel_base + warp_row * (kBlockRows / 2);
    const int column_base = warp_column * 16;

    if (batch >= batch_count || block_channel_base >= out_channels)
        return;

    const int lane_group = lane / 4;
    MmaAccumulator accumulators[kWarpRows][2];
#pragma unroll
    for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
        const float low_bias = bias
            ? __half2float(bias[channel_base + row_tile * 16 + lane_group])
            : 0.0F;
        const float high_bias = bias
            ? __half2float(
                  bias[channel_base + row_tile * 16 + lane_group + 8])
            : 0.0F;
#pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            accumulators[row_tile][column_tile].values[0] = low_bias;
            accumulators[row_tile][column_tile].values[1] = low_bias;
            accumulators[row_tile][column_tile].values[2] = high_bias;
            accumulators[row_tile][column_tile].values[3] = high_bias;
        }
    }

    int stage = 0;
    if constexpr (kBlockRows == 128) {
        load_pointwise_tile_swizzled(
            operand_tiles[stage], input, weight, batch, block_channel_base, 0,
            in_channels, spatial_base, spatial_count, linear_thread);
    } else {
        load_pointwise_tile_swizzled64(
            operand_tiles[stage], input, weight, batch, block_channel_base, 0,
            in_channels, spatial_base, spatial_count, linear_thread);
    }
    commit_async_copies();

    for (int channel = 0; channel < in_channels; channel += 16) {
        wait_for_async_copies();
        __syncthreads();
        const int next_channel = channel + 16;
        if (next_channel < in_channels) {
            if constexpr (kBlockRows == 128) {
                load_pointwise_tile_swizzled(
                    operand_tiles[stage ^ 1], input, weight, batch,
                    block_channel_base, next_channel, in_channels, spatial_base,
                    spatial_count, linear_thread);
            } else {
                load_pointwise_tile_swizzled64(
                    operand_tiles[stage ^ 1], input, weight, batch,
                    block_channel_base, next_channel, in_channels, spatial_base,
                    spatial_count, linear_thread);
            }
            commit_async_copies();
        }

        const int matrix_row = lane & 15;
        const int matrix_column = (lane / 16) * 8;
        unsigned int a[kWarpRows][4];
#pragma unroll
        for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
            load_matrix_x4(
                operand_tiles[stage].weight + swizzle_weight_offset(
                    warp_row * (kBlockRows / 2) + row_tile * 16 + matrix_row,
                    matrix_column),
                a[row_tile]);
        }

        unsigned int b[4];
        const int matrix_input_row = lane & 15;
        const int matrix_input_column = (lane / 16) * 8;
        load_matrix_x4_transpose(
            operand_tiles[stage].input + swizzle_input_offset(
                matrix_input_row, column_base + matrix_input_column),
            b);
#pragma unroll
        for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
            mma_m16n8k16(accumulators[row_tile][0], a[row_tile], b);
            mma_m16n8k16(accumulators[row_tile][1], a[row_tile], b + 2);
        }
        __syncthreads();
        stage ^= 1;
    }

    const int thread_column = (lane & 3) * 2;
#pragma unroll
    for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            const int spatial = spatial_base + column_base + column_tile * 8 +
                                thread_column;
            const int low_channel = channel_base + row_tile * 16 + lane_group;
            const int high_channel = low_channel + 8;
            if (spatial < spatial_count) {
                __half* low = output +
                    (batch * out_channels + low_channel) * spatial_count + spatial;
                __half* high = output +
                    (batch * out_channels + high_channel) * spatial_count + spatial;
                if (epilogue == 0 && (spatial_count & 1) == 0) {
                    *reinterpret_cast<__half2*>(low) = __floats2half2_rn(
                        accumulators[row_tile][column_tile].values[0],
                        accumulators[row_tile][column_tile].values[1]);
                    *reinterpret_cast<__half2*>(high) = __floats2half2_rn(
                        accumulators[row_tile][column_tile].values[2],
                        accumulators[row_tile][column_tile].values[3]);
                } else {
                    const __half* low_residual = epilogue == 2
                        ? epilogue_input +
                            (batch * out_channels + low_channel) * spatial_count +
                            spatial
                        : nullptr;
                    const __half* high_residual = epilogue == 2
                        ? epilogue_input +
                            (batch * out_channels + high_channel) * spatial_count +
                            spatial
                        : nullptr;
                    const __half low_first = pointwise_epilogue_value(
                        accumulators[row_tile][column_tile].values[0],
                        low_residual, epilogue, alpha);
                    const __half high_first = pointwise_epilogue_value(
                        accumulators[row_tile][column_tile].values[2],
                        high_residual, epilogue, alpha);
                    if (spatial + 1 < spatial_count) {
                        const __half low_second = pointwise_epilogue_value(
                            accumulators[row_tile][column_tile].values[1],
                            low_residual ? low_residual + 1 : nullptr,
                            epilogue, alpha);
                        const __half high_second = pointwise_epilogue_value(
                            accumulators[row_tile][column_tile].values[3],
                            high_residual ? high_residual + 1 : nullptr,
                            epilogue, alpha);
                        if ((spatial_count & 1) == 0) {
                            *reinterpret_cast<__half2*>(low) =
                                __halves2half2(low_first, low_second);
                            *reinterpret_cast<__half2*>(high) =
                                __halves2half2(high_first, high_second);
                        } else {
                            low[0] = low_first;
                            low[1] = low_second;
                            high[0] = high_first;
                            high[1] = high_second;
                        }
                    } else {
                        low[0] = low_first;
                        high[0] = high_first;
                    }
                }
            }
        }
    }
}

// Ampere+ pointwise kernels with explicit fragment registers. The large tile
// favors arithmetic reuse; the small tile supplies more blocks for compact
// feature maps.
extern "C" __global__ void mlvc_pointwise_conv_mma_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels, const __half* epilogue_input, int epilogue, float alpha)
{
#if __CUDA_ARCH__ >= 800
    __shared__ PointwiseOperandSwizzled operand_tiles[2];
    pointwise_conv_mma_body<128>(
        operand_tiles, input, weight, bias, output, batch_count, in_channels,
        spatial_count, out_channels, epilogue_input, epilogue, alpha);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)spatial_count;
    (void)out_channels;
    (void)epilogue_input;
    (void)epilogue;
    (void)alpha;
#endif
}

extern "C" __global__ void mlvc_pointwise_conv_mma_small_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels, const __half* epilogue_input, int epilogue, float alpha)
{
#if __CUDA_ARCH__ >= 800
    __shared__ PointwiseOperandSwizzled64 operand_tiles[2];
    pointwise_conv_mma_body<64>(
        operand_tiles, input, weight, bias, output, batch_count, in_channels,
        spatial_count, out_channels, epilogue_input, epilogue, alpha);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)spatial_count;
    (void)out_channels;
    (void)epilogue_input;
    (void)epilogue;
    (void)alpha;
#endif
}

template <int kBlockRows, typename OperandTile>
__device__ __forceinline__ void pointwise_reglu_mma_body(
    OperandTile (&operand_tiles)[2],
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int gate_channels, float minimum, float maximum)
{
    constexpr int kBlockColumns = 64;
    constexpr int kWarpRows = kBlockRows / 32;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int warp_row = warp / 4;
    const int warp_column = warp % 4;
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial_base = static_cast<int>(blockIdx.x) * kBlockColumns;
    const int block_channel_base = static_cast<int>(blockIdx.y) * kBlockRows;
    const int channel_base = block_channel_base + warp_row * (kBlockRows / 2);
    const int column_base = warp_column * 16;

    if (batch >= batch_count || block_channel_base >= gate_channels)
        return;

    const int lane_group = lane / 4;
    MmaAccumulator accumulators[2][kWarpRows][2];
#pragma unroll
    for (int side = 0; side < 2; ++side) {
#pragma unroll
        for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
            const int output_channel = side * gate_channels + channel_base +
                                       row_tile * 16 + lane_group;
            const float low_bias = bias
                ? __half2float(bias[output_channel]) : 0.0F;
            const float high_bias = bias
                ? __half2float(bias[output_channel + 8]) : 0.0F;
#pragma unroll
            for (int column_tile = 0; column_tile < 2; ++column_tile) {
                accumulators[side][row_tile][column_tile].values[0] = low_bias;
                accumulators[side][row_tile][column_tile].values[1] = low_bias;
                accumulators[side][row_tile][column_tile].values[2] = high_bias;
                accumulators[side][row_tile][column_tile].values[3] = high_bias;
            }
        }
    }

    int stage = 0;
    if constexpr (kBlockRows == 64) {
        load_pointwise_reglu_tile_swizzled(
            operand_tiles[stage], input, weight, batch, block_channel_base,
            gate_channels, 0, in_channels, spatial_base, spatial_count,
            linear_thread);
    } else {
        load_pointwise_reglu_tile_swizzled_small(
            operand_tiles[stage], input, weight, batch, block_channel_base,
            gate_channels, 0, in_channels, spatial_base, spatial_count,
            linear_thread);
    }
    commit_async_copies();

    for (int channel = 0; channel < in_channels; channel += 16) {
        wait_for_async_copies();
        __syncthreads();
        const int next_channel = channel + 16;
        if (next_channel < in_channels) {
            if constexpr (kBlockRows == 64) {
                load_pointwise_reglu_tile_swizzled(
                    operand_tiles[stage ^ 1], input, weight, batch,
                    block_channel_base, gate_channels, next_channel, in_channels,
                    spatial_base, spatial_count, linear_thread);
            } else {
                load_pointwise_reglu_tile_swizzled_small(
                    operand_tiles[stage ^ 1], input, weight, batch,
                    block_channel_base, gate_channels, next_channel, in_channels,
                    spatial_base, spatial_count, linear_thread);
            }
            commit_async_copies();
        }

        const int matrix_row = lane & 15;
        const int matrix_column = (lane / 16) * 8;
        const int matrix_input_row = lane & 15;
        const int matrix_input_column = (lane / 16) * 8;
        unsigned int b[4];
        load_matrix_x4_transpose(
            operand_tiles[stage].input + swizzle_input_offset(
                matrix_input_row, column_base + matrix_input_column),
            b);
#pragma unroll
        for (int side = 0; side < 2; ++side) {
#pragma unroll
            for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
                const int shared_row = side * kBlockRows +
                    warp_row * (kBlockRows / 2) + row_tile * 16 + matrix_row;
                unsigned int a[4];
                load_matrix_x4(
                    operand_tiles[stage].weight +
                        swizzle_weight_offset(shared_row, matrix_column),
                    a);
                mma_m16n8k16(accumulators[side][row_tile][0],
                             a, b);
                mma_m16n8k16(accumulators[side][row_tile][1],
                             a, b + 2);
            }
        }
        __syncthreads();
        stage ^= 1;
    }

    const int thread_column = (lane & 3) * 2;
#pragma unroll
    for (int row_tile = 0; row_tile < kWarpRows; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            const int spatial = spatial_base + column_base + column_tile * 8 +
                                thread_column;
            if (spatial >= spatial_count)
                continue;
            const int low_channel = channel_base + row_tile * 16 + lane_group;
            const int high_channel = low_channel + 8;
            __half low_values[2];
            __half high_values[2];
#pragma unroll
            for (int item = 0; item < 2; ++item) {
                const __half gate = __float2half_rn(
                    accumulators[0][row_tile][column_tile].values[item]);
                const __half linear = __float2half_rn(
                    accumulators[1][row_tile][column_tile].values[item]);
                low_values[item] = reglu_value(
                    gate, linear, minimum, maximum);
                const __half high_gate = __float2half_rn(
                    accumulators[0][row_tile][column_tile].values[item + 2]);
                const __half high_linear = __float2half_rn(
                    accumulators[1][row_tile][column_tile].values[item + 2]);
                high_values[item] = reglu_value(
                    high_gate, high_linear, minimum, maximum);
            }
            __half* low = output +
                (batch * gate_channels + low_channel) * spatial_count + spatial;
            __half* high = output +
                (batch * gate_channels + high_channel) * spatial_count + spatial;
            if ((spatial_count & 1) == 0) {
                *reinterpret_cast<__half2*>(low) =
                    __halves2half2(low_values[0], low_values[1]);
                *reinterpret_cast<__half2*>(high) =
                    __halves2half2(high_values[0], high_values[1]);
            } else {
                low[0] = low_values[0];
                high[0] = high_values[0];
                if (spatial + 1 < spatial_count) {
                    low[1] = low_values[1];
                    high[1] = high_values[1];
                }
            }
        }
    }
}

// Fuses the common FFN pointwise Conv(2C), channel split, Clip, and Mul.
extern "C" __global__ void mlvc_pointwise_reglu_mma_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int gate_channels, float minimum, float maximum)
{
#if __CUDA_ARCH__ >= 800
    __shared__ PointwiseOperandSwizzled operand_tiles[2];
    pointwise_reglu_mma_body<64>(
        operand_tiles, input, weight, bias, output, batch_count, in_channels,
        spatial_count, gate_channels, minimum, maximum);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)spatial_count;
    (void)gate_channels;
    (void)minimum;
    (void)maximum;
#endif
}

extern "C" __global__ void mlvc_pointwise_reglu_mma_small_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int gate_channels, float minimum, float maximum)
{
#if __CUDA_ARCH__ >= 800
    __shared__ PointwiseOperandSwizzled64 operand_tiles[2];
    pointwise_reglu_mma_body<32>(
        operand_tiles, input, weight, bias, output, batch_count, in_channels,
        spatial_count, gate_channels, minimum, maximum);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)spatial_count;
    (void)gate_channels;
    (void)minimum;
    (void)maximum;
#endif
}

// Balanced tile for pointwise layers whose 128-column grid is too small to
// occupy the device. Each warp keeps four accumulator fragments live.
extern "C" __global__ void mlvc_pointwise_conv_balanced_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels)
{
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kWarps = 8;
    constexpr int kBlockRows = 128;
    constexpr int kBlockColumns = 64;

    __shared__ PointwiseOperandTile64 operand_tiles[2];
    __shared__ __align__(32) float accumulator_tile[kWarps][kTile * kTile];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int warp_row = warp / 2;
    const int warp_column = warp % 2;
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial_base = static_cast<int>(blockIdx.x) * kBlockColumns;
    const int block_channel_base = static_cast<int>(blockIdx.y) * kBlockRows;
    const int channel_base = block_channel_base + warp_row * 32;
    const int column_base = warp_column * 32;

    if (batch >= batch_count || block_channel_base >= out_channels)
        return;

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                   __half, wmma::row_major> weight_fragments[2];
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                   __half, wmma::row_major> input_fragments[2];
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        accumulators[2][2];

    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            for (int index = lane; index < kTile * kTile; index += 32) {
                const int row = index / kTile;
                accumulator_tile[warp][index] = bias
                    ? __half2float(bias[channel_base + row_tile * kTile + row])
                    : 0.0F;
            }
            __syncwarp();
            wmma::load_matrix_sync(accumulators[row_tile][column_tile],
                                   accumulator_tile[warp], kTile,
                                   wmma::mem_row_major);
        }
    }

    int stage = 0;
    load_pointwise_tile64(operand_tiles[stage], input, weight, batch,
                          block_channel_base, 0, in_channels, spatial_base,
                          spatial_count, linear_thread);
    commit_async_copies();

    for (int channel = 0; channel < in_channels; channel += kTile) {
        wait_for_async_copies();
        __syncthreads();
        const int next_channel = channel + kTile;
        if (next_channel < in_channels) {
            load_pointwise_tile64(
                operand_tiles[stage ^ 1], input, weight, batch,
                block_channel_base, next_channel, in_channels, spatial_base,
                spatial_count, linear_thread);
            commit_async_copies();
        }

        wmma::load_matrix_sync(
            weight_fragments[0],
            operand_tiles[stage].weight + warp_row * 32 * kTile, kTile);
        wmma::load_matrix_sync(
            weight_fragments[1],
            operand_tiles[stage].weight +
                (warp_row * 32 + kTile) * kTile,
            kTile);
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            wmma::load_matrix_sync(
                input_fragments[column_tile],
                operand_tiles[stage].input + column_base + column_tile * kTile,
                72);
        }
        for (int row_tile = 0; row_tile < 2; ++row_tile) {
            for (int column_tile = 0; column_tile < 2; ++column_tile) {
                wmma::mma_sync(accumulators[row_tile][column_tile],
                               weight_fragments[row_tile],
                               input_fragments[column_tile],
                               accumulators[row_tile][column_tile]);
            }
        }
        __syncthreads();
        stage ^= 1;
    }

    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            wmma::store_matrix_sync(accumulator_tile[warp],
                                    accumulators[row_tile][column_tile],
                                    kTile, wmma::mem_row_major);
            __syncwarp();
            for (int index = lane; index < kTile * kTile; index += 32) {
                const int row = index / kTile;
                const int column = index - row * kTile;
                const int spatial = spatial_base + column_base +
                                    column_tile * kTile + column;
                if (spatial < spatial_count) {
                    const int output_channel =
                        channel_base + row_tile * kTile + row;
                    output[(batch * out_channels + output_channel) *
                               spatial_count + spatial] =
                        __float2half_rn(accumulator_tile[warp][index]);
                }
            }
            __syncwarp();
        }
    }
}

// Wide pointwise tile for aligned MLVC layers. Eight warps compute a 128x128
// output tile while double-buffered operand loads overlap tensor-core work.
extern "C" __global__ void mlvc_pointwise_conv_wide_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int spatial_count,
    int out_channels)
{
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kWarps = 8;
    constexpr int kBlockRows = 128;
    constexpr int kBlockColumns = 128;

    __shared__ PointwiseOperandTile operand_tiles[2];
    __shared__ __align__(32) float accumulator_tile[kWarps][kTile * kTile];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int warp_row = warp / 2;
    const int warp_column = warp % 2;
    const int batch = static_cast<int>(blockIdx.z);
    const int spatial_base = static_cast<int>(blockIdx.x) * kBlockColumns;
    const int block_channel_base = static_cast<int>(blockIdx.y) * kBlockRows;
    const int channel_base = block_channel_base + warp_row * 32;
    const int column_base = warp_column * 64;

    if (batch >= batch_count || block_channel_base >= out_channels)
        return;

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                   __half, wmma::row_major> weight_fragments[2];
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                   __half, wmma::row_major> input_fragments[4];
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        accumulators[2][4];

    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        for (int column_tile = 0; column_tile < 4; ++column_tile) {
            for (int index = lane; index < kTile * kTile; index += 32) {
                const int row = index / kTile;
                accumulator_tile[warp][index] = bias
                    ? __half2float(bias[channel_base + row_tile * kTile + row])
                    : 0.0F;
            }
            __syncwarp();
            wmma::load_matrix_sync(accumulators[row_tile][column_tile],
                                   accumulator_tile[warp], kTile,
                                   wmma::mem_row_major);
        }
    }

    int stage = 0;
    load_pointwise_tile(operand_tiles[stage], input, weight, batch,
                        block_channel_base, 0, in_channels, spatial_base,
                        spatial_count, linear_thread);
    commit_async_copies();

    for (int channel = 0; channel < in_channels; channel += kTile) {
        wait_for_async_copies();
        __syncthreads();
        const int next_channel = channel + kTile;
        if (next_channel < in_channels) {
            load_pointwise_tile(operand_tiles[stage ^ 1], input, weight, batch,
                                block_channel_base, next_channel, in_channels,
                                spatial_base, spatial_count, linear_thread);
            commit_async_copies();
        }

        wmma::load_matrix_sync(
            weight_fragments[0],
            operand_tiles[stage].weight + warp_row * 32 * kTile, kTile);
        wmma::load_matrix_sync(
            weight_fragments[1],
            operand_tiles[stage].weight +
                (warp_row * 32 + kTile) * kTile,
            kTile);
        for (int column_tile = 0; column_tile < 4; ++column_tile) {
            wmma::load_matrix_sync(
                input_fragments[column_tile],
                operand_tiles[stage].input + column_base + column_tile * kTile,
                kBlockColumns);
        }
        for (int row_tile = 0; row_tile < 2; ++row_tile) {
            for (int column_tile = 0; column_tile < 4; ++column_tile) {
                wmma::mma_sync(accumulators[row_tile][column_tile],
                               weight_fragments[row_tile],
                               input_fragments[column_tile],
                               accumulators[row_tile][column_tile]);
            }
        }
        __syncthreads();
        stage ^= 1;
    }

    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        for (int column_tile = 0; column_tile < 4; ++column_tile) {
            wmma::store_matrix_sync(accumulator_tile[warp],
                                    accumulators[row_tile][column_tile],
                                    kTile, wmma::mem_row_major);
            __syncwarp();
            for (int index = lane; index < kTile * kTile; index += 32) {
                const int row = index / kTile;
                const int column = index - row * kTile;
                const int spatial = spatial_base + column_base +
                                    column_tile * kTile + column;
                if (spatial < spatial_count) {
                    const int output_channel =
                        channel_base + row_tile * kTile + row;
                    output[(batch * out_channels + output_channel) *
                               spatial_count + spatial] =
                        __float2half_rn(accumulator_tile[warp][index]);
                }
            }
            __syncwarp();
        }
    }
}

template <int kBlockColumns>
__device__ __forceinline__ void spatial_conv_mma_body(
    SpatialOperandK32& operand,
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int input_height,
    int input_width, int out_channels, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
    constexpr int kBlockChannels = 128;
    constexpr int kColumnTiles = kBlockColumns / 8;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int batch = static_cast<int>(blockIdx.z);
    const int block_channel_base =
        static_cast<int>(blockIdx.y) * kBlockChannels;
    const int spatial_base = static_cast<int>(blockIdx.x) * kBlockColumns;
    const int channel_base = block_channel_base + warp * 16;
    const int output_spatial_count = output_height * output_width;
    const int kernel_plane = kernel_height * kernel_width;
    const int reduction_count = in_channels * kernel_plane;

    if (batch >= batch_count || block_channel_base >= out_channels)
        return;

    const int local_spatial = linear_thread % kBlockColumns;
    const int spatial = spatial_base + local_spatial;
    const bool valid_spatial = spatial < output_spatial_count;
    const int output_y = valid_spatial ? spatial / output_width : 0;
    const int output_x = valid_spatial
        ? spatial - output_y * output_width : 0;

    const int lane_group = lane / 4;
    MmaAccumulator accumulators[kColumnTiles];
#pragma unroll
    for (int column_tile = 0; column_tile < kColumnTiles; ++column_tile) {
        const float low_bias = bias
            ? __half2float(bias[channel_base + lane_group]) : 0.0F;
        const float high_bias = bias
            ? __half2float(bias[channel_base + lane_group + 8]) : 0.0F;
        accumulators[column_tile].values[0] = low_bias;
        accumulators[column_tile].values[1] = low_bias;
        accumulators[column_tile].values[2] = high_bias;
        accumulators[column_tile].values[3] = high_bias;
    }

    for (int reduction_base = 0; reduction_base < reduction_count;
         reduction_base += 32) {
#pragma unroll
        for (int reduction_half = 0; reduction_half < 2; ++reduction_half) {
            PointwiseOperandSwizzled& tile = operand.halves[reduction_half];
            const int weight_row = linear_thread / 2;
            const int weight_column = (linear_thread % 2) * 8;
            copy_async_16(
                tile.weight + swizzle_weight_offset(
                    weight_row, weight_column),
                weight + (block_channel_base + weight_row) * reduction_count +
                    reduction_base + reduction_half * 16 + weight_column,
                8);
        }
        commit_async_copies();

#pragma unroll
        for (int reduction_half = 0; reduction_half < 2; ++reduction_half) {
            PointwiseOperandSwizzled& tile = operand.halves[reduction_half];
            if constexpr (kBlockColumns == 16) {
                const int row = warp * 2 + lane / 16;
                const int leader = lane < 16 ? 0 : 16;
                const int reduction =
                    reduction_base + reduction_half * 16 + row;
                int input_channel = 0;
                int kernel_y = 0;
                int kernel_x = 0;
                if (lane == leader) {
                    if (kernel_plane == 9 && kernel_width == 3) {
                        input_channel = reduction / 9;
                        const int kernel_index = reduction - input_channel * 9;
                        kernel_y = kernel_index / 3;
                        kernel_x = kernel_index - kernel_y * 3;
                    } else if (kernel_plane == 4 && kernel_width == 2) {
                        input_channel = reduction / 4;
                        const int kernel_index = reduction - input_channel * 4;
                        kernel_y = kernel_index / 2;
                        kernel_x = kernel_index - kernel_y * 2;
                    } else {
                        input_channel = reduction / kernel_plane;
                        const int kernel_index =
                            reduction - input_channel * kernel_plane;
                        kernel_y = kernel_index / kernel_width;
                        kernel_x = kernel_index - kernel_y * kernel_width;
                    }
                }
                input_channel =
                    __shfl_sync(0xffffffffU, input_channel, leader);
                kernel_y = __shfl_sync(0xffffffffU, kernel_y, leader);
                kernel_x = __shfl_sync(0xffffffffU, kernel_x, leader);

                __half input_value = __float2half(0.0F);
                if (valid_spatial) {
                    const int input_y = output_y * stride_height + kernel_y -
                        pad_height;
                    const int input_x = output_x * stride_width + kernel_x -
                        pad_width;
                    if (input_y >= 0 && input_y < input_height &&
                        input_x >= 0 && input_x < input_width) {
                        input_value = input[
                            ((batch * in_channels + input_channel) *
                                 input_height + input_y) *
                                input_width + input_x];
                    }
                }
                tile.input[swizzle_input_offset(row, local_spatial)] =
                    input_value;
            } else {
                for (int index = linear_thread; index < 16 * kBlockColumns;
                     index += 256) {
                    const int reduction = reduction_base +
                        reduction_half * 16 + index / kBlockColumns;
                    int input_channel = 0;
                    int kernel_y = 0;
                    int kernel_x = 0;
                    if (kernel_plane == 9 && kernel_width == 3) {
                        input_channel = reduction / 9;
                        const int kernel_index = reduction - input_channel * 9;
                        kernel_y = kernel_index / 3;
                        kernel_x = kernel_index - kernel_y * 3;
                    } else if (kernel_plane == 4 && kernel_width == 2) {
                        input_channel = reduction / 4;
                        const int kernel_index = reduction - input_channel * 4;
                        kernel_y = kernel_index / 2;
                        kernel_x = kernel_index - kernel_y * 2;
                    } else {
                        input_channel = reduction / kernel_plane;
                        const int kernel_index =
                            reduction - input_channel * kernel_plane;
                        kernel_y = kernel_index / kernel_width;
                        kernel_x = kernel_index - kernel_y * kernel_width;
                    }
                    __half input_value = __float2half(0.0F);
                    if (valid_spatial) {
                        const int input_y = output_y * stride_height +
                            kernel_y - pad_height;
                        const int input_x = output_x * stride_width +
                            kernel_x - pad_width;
                        if (input_y >= 0 && input_y < input_height &&
                            input_x >= 0 && input_x < input_width) {
                            input_value = input[
                                ((batch * in_channels + input_channel) *
                                     input_height + input_y) *
                                    input_width + input_x];
                        }
                    }
                    tile.input[swizzle_input_offset(
                        index / kBlockColumns, local_spatial)] = input_value;
                }
            }
        }
        wait_for_async_copies();
        __syncthreads();

#pragma unroll
        for (int reduction_half = 0; reduction_half < 2; ++reduction_half) {
            const PointwiseOperandSwizzled& tile =
                operand.halves[reduction_half];
            unsigned int a[4];
            const int matrix_row = lane & 15;
            const int matrix_column = (lane / 16) * 8;
            load_matrix_x4(
                tile.weight + swizzle_weight_offset(
                    warp * 16 + matrix_row, matrix_column),
                a);

            unsigned int b[kBlockColumns / 16][4];
#pragma unroll
            for (int column_group = 0; column_group < kBlockColumns / 16;
                 ++column_group) {
                load_matrix_x4_transpose(
                    tile.input + swizzle_input_offset(
                        matrix_row, column_group * 16 + matrix_column),
                    b[column_group]);
                mma_m16n8k16(
                    accumulators[column_group * 2], a, b[column_group]);
                mma_m16n8k16(
                    accumulators[column_group * 2 + 1], a,
                    b[column_group] + 2);
            }
        }
        __syncthreads();
    }

    const int thread_column = (lane & 3) * 2;
#pragma unroll
    for (int column_tile = 0; column_tile < kColumnTiles; ++column_tile) {
        const int spatial = spatial_base + column_tile * 8 + thread_column;
        if (spatial >= output_spatial_count)
            continue;
        const int low_channel = channel_base + lane_group;
        const int high_channel = low_channel + 8;
        __half* low = output +
            (batch * out_channels + low_channel) * output_spatial_count + spatial;
        __half* high = output +
            (batch * out_channels + high_channel) * output_spatial_count + spatial;
        if ((output_spatial_count & 1) == 0) {
            *reinterpret_cast<__half2*>(low) = __floats2half2_rn(
                accumulators[column_tile].values[0],
                accumulators[column_tile].values[1]);
            *reinterpret_cast<__half2*>(high) = __floats2half2_rn(
                accumulators[column_tile].values[2],
                accumulators[column_tile].values[3]);
        } else {
            low[0] = __float2half_rn(accumulators[column_tile].values[0]);
            high[0] = __float2half_rn(accumulators[column_tile].values[2]);
            if (spatial + 1 < output_spatial_count) {
                low[1] = __float2half_rn(accumulators[column_tile].values[1]);
                high[1] = __float2half_rn(accumulators[column_tile].values[3]);
            }
        }
    }
}

// Ampere+ implicit-GEMM spatial convolutions. The wide variant amortizes each
// 128x16 weight tile over twice as many output positions.
extern "C" __global__ void mlvc_spatial_conv_mma_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int input_height,
    int input_width, int out_channels, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
#if __CUDA_ARCH__ >= 800
    __shared__ SpatialOperandK32 operand;
    spatial_conv_mma_body<16>(
        operand, input, weight, bias, output, batch_count, in_channels,
        input_height, input_width, out_channels, output_height, output_width,
        kernel_height, kernel_width, stride_height, stride_width, pad_height,
        pad_width);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)input_height;
    (void)input_width;
    (void)out_channels;
    (void)output_height;
    (void)output_width;
    (void)kernel_height;
    (void)kernel_width;
    (void)stride_height;
    (void)stride_width;
    (void)pad_height;
    (void)pad_width;
#endif
}

extern "C" __global__ void mlvc_spatial_conv_mma_wide_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int input_height,
    int input_width, int out_channels, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
#if __CUDA_ARCH__ >= 800
    __shared__ SpatialOperandK32 operand;
    spatial_conv_mma_body<32>(
        operand, input, weight, bias, output, batch_count, in_channels,
        input_height, input_width, out_channels, output_height, output_width,
        kernel_height, kernel_width, stride_height, stride_width, pad_height,
        pad_width);
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)batch_count;
    (void)in_channels;
    (void)input_height;
    (void)input_width;
    (void)out_channels;
    (void)output_height;
    (void)output_width;
    (void)kernel_height;
    (void)kernel_width;
    (void)stride_height;
    (void)stride_width;
    (void)pad_height;
    (void)pad_width;
#endif
}

// Implicit-GEMM fallback for dense spatial kernels. The input matrix tile is
// gathered from NCHW storage, avoiding a materialized im2col buffer.
extern "C" __global__ void mlvc_spatial_conv_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int in_channels, int input_height,
    int input_width, int out_channels, int output_height, int output_width,
    int kernel_height, int kernel_width, int stride_height, int stride_width,
    int pad_height, int pad_width)
{
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kWarps = 4;
    constexpr int kOutputChannelsPerBlock = kTile * kWarps;

    __shared__ __align__(32) __half input_tile[kTile * kTile];
    __shared__ __align__(32) float accumulator_tiles[kWarps][kTile * kTile];

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int linear_thread = warp * 32 + lane;
    const int batch = static_cast<int>(blockIdx.z);
    const int output_spatial_count = output_height * output_width;
    const int spatial_base = static_cast<int>(blockIdx.x) * kTile;
    const int channel_base =
        static_cast<int>(blockIdx.y) * kOutputChannelsPerBlock + warp * kTile;
    const int kernel_plane = kernel_height * kernel_width;
    const int reduction_count = in_channels * kernel_plane;

    if (batch >= batch_count || channel_base >= out_channels)
        return;

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                   __half, wmma::row_major> weight_fragment;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                   __half, wmma::row_major> input_fragment;
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        accumulator_fragment;

    for (int index = lane; index < kTile * kTile; index += 32) {
        const int row = index / kTile;
        accumulator_tiles[warp][index] = bias
            ? __half2float(bias[channel_base + row]) : 0.0F;
    }
    __syncwarp();
    wmma::load_matrix_sync(accumulator_fragment, accumulator_tiles[warp],
                           kTile, wmma::mem_row_major);

    for (int reduction_base = 0; reduction_base < reduction_count;
         reduction_base += kTile) {
        for (int index = linear_thread; index < kTile * kTile;
             index += 32 * kWarps) {
            const int reduction = reduction_base + index / kTile;
            const int spatial = spatial_base + index % kTile;
            __half value = __float2half(0.0F);
            if (spatial < output_spatial_count) {
                const int output_y = spatial / output_width;
                const int output_x = spatial - output_y * output_width;
                const int input_channel = reduction / kernel_plane;
                const int kernel_index = reduction - input_channel * kernel_plane;
                const int kernel_y = kernel_index / kernel_width;
                const int kernel_x = kernel_index - kernel_y * kernel_width;
                const int input_y =
                    output_y * stride_height + kernel_y - pad_height;
                const int input_x =
                    output_x * stride_width + kernel_x - pad_width;
                if (input_y >= 0 && input_y < input_height &&
                    input_x >= 0 && input_x < input_width) {
                    value = input[
                        ((batch * in_channels + input_channel) * input_height +
                         input_y) * input_width + input_x];
                }
            }
            input_tile[index] = value;
        }
        __syncthreads();

        wmma::load_matrix_sync(
            weight_fragment,
            weight + channel_base * reduction_count + reduction_base,
            reduction_count);
        wmma::load_matrix_sync(input_fragment, input_tile, kTile);
        wmma::mma_sync(accumulator_fragment, weight_fragment, input_fragment,
                       accumulator_fragment);
        __syncthreads();
    }

    wmma::store_matrix_sync(accumulator_tiles[warp], accumulator_fragment,
                            kTile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTile * kTile; index += 32) {
        const int row = index / kTile;
        const int column = index - row * kTile;
        const int spatial = spatial_base + column;
        if (spatial < output_spatial_count) {
            output[(batch * out_channels + channel_base + row) *
                       output_spatial_count + spatial] =
                __float2half_rn(accumulator_tiles[warp][index]);
        }
    }
}

extern "C" __global__ void mlvc_depthwise_conv_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_plane = output_height * output_width;
    const int count = batch_count * channels * output_plane;
    if (index >= count)
        return;

    int linear = index;
    const int output_x = linear % output_width;
    linear /= output_width;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int kernel_plane = kernel_height * kernel_width;
    float accumulator = bias ? __half2float(bias[channel]) : 0.0F;

    for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
            const int input_x =
                output_x * stride_width + kernel_x - pad_width;
            if (input_x < 0 || input_x >= input_width)
                continue;
            const int input_index =
                ((batch * channels + channel) * input_height + input_y) *
                    input_width + input_x;
            const int weight_index =
                channel * kernel_plane + kernel_y * kernel_width + kernel_x;
            accumulator = fmaf(__half2float(input[input_index]),
                               __half2float(weight[weight_index]), accumulator);
        }
    }
    output[index] = __float2half_rn(accumulator);
}

extern "C" __global__ void mlvc_depthwise_conv_pair_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_pairs_per_row = output_width / 2;
    const int output_pairs = output_height * output_pairs_per_row;
    const int count = batch_count * channels * output_pairs;
    if (index >= count)
        return;

    int linear = index;
    const int output_pair_x = linear % output_pairs_per_row;
    linear /= output_pairs_per_row;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int output_x = output_pair_x * 2;
    const int kernel_plane = kernel_height * kernel_width;
    const float initial = bias ? __half2float(bias[channel]) : 0.0F;
    float low_accumulator = initial;
    float high_accumulator = initial;

    for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
            const int input_x =
                output_x * stride_width + kernel_x - pad_width;
            const float weight_value = __half2float(
                weight[channel * kernel_plane +
                       kernel_y * kernel_width + kernel_x]);
            const int input_row =
                ((batch * channels + channel) * input_height + input_y) *
                input_width;
            if (input_x >= 0 && input_x < input_width) {
                low_accumulator = fmaf(
                    __half2float(input[input_row + input_x]), weight_value,
                    low_accumulator);
            }
            const int high_input_x = input_x + stride_width;
            if (high_input_x >= 0 && high_input_x < input_width) {
                high_accumulator = fmaf(
                    __half2float(input[input_row + high_input_x]), weight_value,
                    high_accumulator);
            }
        }
    }

    const int output_index =
        ((batch * channels + channel) * output_height + output_y) *
        output_width + output_x;
    reinterpret_cast<__half2*>(output + output_index)[0] =
        __floats2half2_rn(low_accumulator, high_accumulator);
}

extern "C" __global__ void mlvc_depthwise_conv_quad_fp16(
    const __half* input, const __half* weight, const __half* bias,
    __half* output, int batch_count, int channels, int input_height,
    int input_width, int output_height, int output_width, int kernel_height,
    int kernel_width, int stride_height, int stride_width, int pad_height,
    int pad_width)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int output_quads_per_row = output_width / 4;
    const int output_quads = output_height * output_quads_per_row;
    const int count = batch_count * channels * output_quads;
    if (index >= count)
        return;

    int linear = index;
    const int output_quad_x = linear % output_quads_per_row;
    linear /= output_quads_per_row;
    const int output_y = linear % output_height;
    linear /= output_height;
    const int channel = linear % channels;
    const int batch = linear / channels;
    const int output_x = output_quad_x * 4;
    const float initial = bias ? __half2float(bias[channel]) : 0.0F;
    float accumulators[4] = {initial, initial, initial, initial};

#pragma unroll
    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
        const int input_y =
            output_y * stride_height + kernel_y - pad_height;
        if (input_y < 0 || input_y >= input_height)
            continue;
        const int input_x_base = output_x * stride_width - pad_width;
        const int input_row =
            ((batch * channels + channel) * input_height + input_y) *
            input_width;
        float input_values[6];
#pragma unroll
        for (int item = 0; item < 6; ++item) {
            const int input_x = input_x_base + item;
            input_values[item] = input_x >= 0 && input_x < input_width
                ? __half2float(input[input_row + input_x]) : 0.0F;
        }
#pragma unroll
        for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
            const float weight_value = __half2float(
                weight[channel * 9 + kernel_y * 3 + kernel_x]);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                accumulators[item] = fmaf(
                    input_values[item + kernel_x], weight_value,
                    accumulators[item]);
            }
        }
    }

    const int output_index =
        ((batch * channels + channel) * output_height + output_y) *
        output_width + output_x;
    reinterpret_cast<__half2*>(output + output_index)[0] =
        __floats2half2_rn(accumulators[0], accumulators[1]);
    reinterpret_cast<__half2*>(output + output_index)[1] =
        __floats2half2_rn(accumulators[2], accumulators[3]);

    (void)kernel_height;
    (void)kernel_width;
}

extern "C" __global__ void mlvc_gather_axis0_fp16(
    const __half* input, const std::int32_t* index, __half* output,
    int row_elements, int rows)
{
    const int element = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (element >= row_elements)
        return;
    int row = index[0];
    if (row < 0)
        row += rows;
    if (row >= 0 && row < rows)
        output[element] = input[row * row_elements + element];
}

extern "C" __global__ void mlvc_slice_fp16(
    const __half* input, __half* output, int count,
    int od0, int od1, int od2, int od3,
    int id0, int id1, int id2, int id3,
    int start0, int start1, int start2, int start3,
    int step0, int step1, int step2, int step3)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    int linear = index;
    const int c3 = linear % od3;
    linear /= od3;
    const int c2 = linear % od2;
    linear /= od2;
    const int c1 = linear % od1;
    const int c0 = linear / od1;
    const int input_index =
        (((start0 + c0 * step0) * id1 + (start1 + c1 * step1)) * id2 +
          (start2 + c2 * step2)) * id3 + (start3 + c3 * step3);
    output[index] = input[input_index];
}

extern "C" __global__ void mlvc_concat_copy_fp16(
    const __half* input, __half* output, int count,
    int id0, int id1, int id2, int id3,
    int od0, int od1, int od2, int od3,
    int axis, int axis_offset)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    int linear = index;
    int coordinates[4];
    coordinates[3] = linear % id3;
    linear /= id3;
    coordinates[2] = linear % id2;
    linear /= id2;
    coordinates[1] = linear % id1;
    coordinates[0] = linear / id1;
    coordinates[axis] += axis_offset;
    const int output_index =
        ((coordinates[0] * od1 + coordinates[1]) * od2 + coordinates[2]) * od3 +
        coordinates[3];
    output[output_index] = input[index];
}

extern "C" __global__ void mlvc_depth_to_space_fp16(
    const __half* input, __half* output, int count,
    int in_channels, int input_height, int input_width, int block_size, int mode)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int output_width = input_width * block_size;
    const int output_height = input_height * block_size;
    const int output_channels = in_channels / (block_size * block_size);
    int linear = index;
    const int x = linear % output_width;
    linear /= output_width;
    const int y = linear % output_height;
    linear /= output_height;
    const int channel = linear % output_channels;
    const int n = linear / output_channels;
    const int offset_y = y % block_size;
    const int offset_x = x % block_size;
    const int input_channel = mode == 0
        ? (offset_y * block_size + offset_x) * output_channels + channel
        : channel * block_size * block_size + offset_y * block_size + offset_x;
    const int input_index =
        ((n * in_channels + input_channel) * input_height + y / block_size) *
            input_width + x / block_size;
    output[index] = input[input_index];
}

extern "C" __global__ void mlvc_space_to_depth_fp16(
    const __half* input, __half* output, int count,
    int in_channels, int input_height, int input_width, int block_size)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    const int output_width = input_width / block_size;
    const int output_height = input_height / block_size;
    const int output_channels = in_channels * block_size * block_size;
    int linear = index;
    const int x = linear % output_width;
    linear /= output_width;
    const int y = linear % output_height;
    linear /= output_height;
    const int output_channel = linear % output_channels;
    const int n = linear / output_channels;
    // ONNX SpaceToDepth is the inverse of DCR DepthToSpace: block offset first.
    const int input_channel = output_channel % in_channels;
    const int offset = output_channel / in_channels;
    const int input_y = y * block_size + offset / block_size;
    const int input_x = x * block_size + offset % block_size;
    const int input_index =
        ((n * in_channels + input_channel) * input_height + input_y) * input_width +
        input_x;
    output[index] = input[input_index];
}
