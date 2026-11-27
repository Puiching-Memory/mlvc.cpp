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

// Interleaves [gate(0..G-1), linear(0..G-1)] rows into 16-row MMA blocks
// [gate(8b..8b+7), linear(8b..8b+7)] so CUTLASS epilogue steps (8 rows/step)
// visit each gate block immediately before its matching linear block.
extern "C" __global__ void mlvc_interleave_reglu_rows_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int gate_channels, int row_width)
{
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int out_row = static_cast<int>(blockIdx.y);
    if (col >= row_width || out_row >= gate_channels * 2)
        return;
    const int block = out_row >> 4;
    const int in_block = out_row & 15;
    const int src_row = in_block < 8
        ? (block << 3) + in_block
        : gate_channels + (block << 3) + (in_block - 8);
    output[static_cast<std::size_t>(out_row) * row_width + col] =
        input[static_cast<std::size_t>(src_row) * row_width + col];
}

// Interleaves [gate(0..G-1), linear(0..G-1)] rows into adjacent pairs
// [gate(0), linear(0), gate(1), linear(1), ...] for NHWC CUTLASS ReGLU where
// each 8-element fragment holds 4 (gate, linear) pairs in one thread.
extern "C" __global__ void mlvc_interleave_reglu_pairs_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int gate_channels, int row_width)
{
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int out_row = static_cast<int>(blockIdx.y);
    if (col >= row_width || out_row >= gate_channels * 2)
        return;
    const int pair = out_row >> 1;
    const int src_row = (out_row & 1) == 0 ? pair : (gate_channels + pair);
    output[static_cast<std::size_t>(out_row) * row_width + col] =
        input[static_cast<std::size_t>(src_row) * row_width + col];
}

// Repacks 3x3 depthwise weights from [C, 9] (OIHW) to [9, C] so NHWC depthwise
// convolution loads all 9 taps with 128-bit coalesced channel vector loads.
extern "C" __global__ void mlvc_dwconv_weight_oihw_to_k9c_fp16(
    const __half* __restrict__ input, __half* __restrict__ output, int channels)
{
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int k = static_cast<int>(blockIdx.y);
    if (c >= channels || k >= 9)
        return;
    output[static_cast<std::size_t>(k) * channels + c] =
        input[static_cast<std::size_t>(c) * 9 + k];
}

