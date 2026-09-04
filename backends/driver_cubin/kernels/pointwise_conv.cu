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

#include "mma_common.cuh"

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

