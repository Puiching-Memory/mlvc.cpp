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

