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
    // Native pixel_unshuffle layout: channel first, followed by block offsets.
    const int input_channel = output_channel / (block_size * block_size);
    const int offset = output_channel % (block_size * block_size);
    const int input_y = y * block_size + offset / block_size;
    const int input_x = x * block_size + offset % block_size;
    const int input_index =
        ((n * in_channels + input_channel) * input_height + input_y) * input_width +
        input_x;
    output[index] = input[input_index];
}

extern "C" __global__ void mlvc_copy_fp16(
    const __half* __restrict__ input, __half* __restrict__ output, int count)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool aligned16 =
        ((reinterpret_cast<std::uintptr_t>(input) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output) & 15U) == 0);
    if (aligned16) {
        const int vec_count = count >> 3;
        if (idx < vec_count) {
            reinterpret_cast<uint4*>(output)[idx] =
                __ldg(reinterpret_cast<const uint4*>(input) + idx);
        }
        const int tail_start = vec_count << 3;
        if (idx < count - tail_start)
            output[tail_start + idx] = input[tail_start + idx];
        return;
    }
    const int base = idx << 3;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (base + i < count)
            output[base + i] = input[base + i];
    }
}

extern "C" __global__ void mlvc_depth_to_space_3d_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int batch_count, int in_channels, int input_height, int input_width,
    int block_size, int mode)
{
    const int output_width = input_width * block_size;
    const int output_height = input_height * block_size;
    const int block_sq = block_size * block_size;
    const int output_channels = in_channels / block_sq;
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int plane = static_cast<int>(blockIdx.z);
    if (x >= output_width || y >= output_height ||
        plane >= batch_count * output_channels) {
        return;
    }

    const int n = batch_count == 1 ? 0 : (plane / output_channels);
    const int channel = batch_count == 1 ? plane : (plane - n * output_channels);
    int iy, ix, offset_y, offset_x;
    if ((block_size & (block_size - 1)) == 0) {
        const int shift = __ffs(block_size) - 1;
        const int mask = block_size - 1;
        iy = y >> shift;
        ix = x >> shift;
        offset_y = y & mask;
        offset_x = x & mask;
    } else {
        iy = y / block_size;
        ix = x / block_size;
        offset_y = y - iy * block_size;
        offset_x = x - ix * block_size;
    }
    const int input_channel = mode == 0
        ? (offset_y * block_size + offset_x) * output_channels + channel
        : channel * block_sq + offset_y * block_size + offset_x;
    const std::size_t input_index =
        ((static_cast<std::size_t>(n) * in_channels + input_channel) *
             input_height + iy) * input_width + ix;
    const std::size_t output_index =
        (static_cast<std::size_t>(plane) * output_height + y) * output_width + x;
    output[output_index] = __ldg(input + input_index);
}

extern "C" __global__ void mlvc_space_to_depth_3d_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int batch_count, int in_channels, int input_height, int input_width,
    int block_size)
{
    const int block_sq = block_size * block_size;
    const int output_width = input_width / block_size;
    const int output_height = input_height / block_size;
    const int output_channels = in_channels * block_sq;
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int plane = static_cast<int>(blockIdx.z);
    if (x >= output_width || y >= output_height ||
        plane >= batch_count * output_channels) {
        return;
    }

    const int n = batch_count == 1 ? 0 : (plane / output_channels);
    const int output_channel =
        batch_count == 1 ? plane : (plane - n * output_channels);
    int input_channel, input_y, input_x;
    if ((block_size & (block_size - 1)) == 0) {
        const int shift = __ffs(block_size) - 1;
        const int mask = block_size - 1;
        const int sq_shift = shift * 2;
        const int sq_mask = block_sq - 1;
        input_channel = output_channel >> sq_shift;
        const int offset = output_channel & sq_mask;
        input_y = (y << shift) + (offset >> shift);
        input_x = (x << shift) + (offset & mask);
    } else {
        input_channel = output_channel / block_sq;
        const int offset = output_channel - input_channel * block_sq;
        input_y = y * block_size + offset / block_size;
        input_x = x * block_size + offset % block_size;
    }
    const std::size_t input_index =
        ((static_cast<std::size_t>(n) * in_channels + input_channel) *
             input_height + input_y) * input_width + input_x;
    const std::size_t output_index =
        (static_cast<std::size_t>(plane) * output_height + y) * output_width + x;
    output[output_index] = __ldg(input + input_index);
}

