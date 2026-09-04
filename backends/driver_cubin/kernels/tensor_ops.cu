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

