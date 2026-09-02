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

// Implicit-GEMM convolution for dense spatial kernels. The input matrix tile
// is gathered from NCHW storage, avoiding a materialized im2col buffer.
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
    const int input_channel = output_channel / (block_size * block_size);
    const int offset = output_channel % (block_size * block_size);
    const int input_y = y * block_size + offset / block_size;
    const int input_x = x * block_size + offset % block_size;
    const int input_index =
        ((n * in_channels + input_channel) * input_height + input_y) * input_width +
        input_x;
    output[index] = input[input_index];
}
