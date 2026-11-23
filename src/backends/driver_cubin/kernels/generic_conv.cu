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

