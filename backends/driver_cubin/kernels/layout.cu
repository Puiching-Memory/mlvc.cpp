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

