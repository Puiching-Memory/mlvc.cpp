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

// Fast 2D row copy for NHWC channel Slice and channel Concat (128-bit coalesced
// when channels, strides, and base pointers are 8-half aligned).
// Launch: block = (32, 8, 1),
//         grid  = (ceil(channels / 32) [or ceil((channels/8)/32)],
//                  ceil(pixels / 8), 1)
// Note: pass use_vec8 = 1 when (channels % 8 == 0 && strides % 8 == 0 && 16B aligned).
extern "C" __global__ void mlvc_copy_2d_nhwc_fp16(
    const __half* __restrict__ src, __half* __restrict__ dst,
    int pixels, int channels, int src_stride, int dst_stride)
{
    const int p = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (p >= pixels)
        return;
    const bool vec8 =
        ((channels & 7) == 0) && ((src_stride & 7) == 0) &&
        ((dst_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(src) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(dst) & 15U) == 0);
    if (vec8) {
        const int cg = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int vec_c = channels >> 3;
        if (cg < vec_c) {
            reinterpret_cast<uint4*>(
                dst + static_cast<std::size_t>(p) * dst_stride)[cg] =
                __ldg(reinterpret_cast<const uint4*>(
                          src + static_cast<std::size_t>(p) * src_stride) +
                      cg);
        }
        return;
    }
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c < channels) {
        dst[static_cast<std::size_t>(p) * dst_stride + c] =
            __ldg(src + static_cast<std::size_t>(p) * src_stride + c);
    }
}

extern "C" __global__ void mlvc_slice_nhwc_fp16(
    const __half* __restrict__ input, __half* __restrict__ output, int count,
    int od0, int od1, int od2, int od3,
    int id0, int id1, int id2, int id3,
    int start0, int start1, int start2, int start3,
    int step0, int step1, int step2, int step3,
    int in_stride, int out_stride)
{
    (void)od0; (void)id0; (void)id1;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    int linear = index;
    const int c = linear % od1;
    linear /= od1;
    const int w = linear % od3;
    linear /= od3;
    const int h = linear % od2;
    const int n = linear / od2;

    const int in_n = start0 + n * step0;
    const int in_c = start1 + c * step1;
    const int in_h = start2 + h * step2;
    const int in_w = start3 + w * step3;
    const std::size_t in_idx =
        (static_cast<std::size_t>(in_n * id2 + in_h) * id3 + in_w) * in_stride +
        in_c;
    const std::size_t out_idx =
        (static_cast<std::size_t>(n * od2 + h) * od3 + w) * out_stride + c;
    output[out_idx] = __ldg(input + in_idx);
}

extern "C" __global__ void mlvc_concat_copy_nhwc_fp16(
    const __half* __restrict__ input, __half* __restrict__ output, int count,
    int id0, int id1, int id2, int id3,
    int od0, int od1, int od2, int od3,
    int axis, int axis_offset, int in_stride, int out_stride)
{
    (void)id0; (void)od0; (void)od1;
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count)
        return;
    int linear = index;
    int coords[4];
    coords[1] = linear % id1;
    linear /= id1;
    coords[3] = linear % id3;
    linear /= id3;
    coords[2] = linear % id2;
    coords[0] = linear / id2;

    const std::size_t in_idx =
        (static_cast<std::size_t>(coords[0] * id2 + coords[2]) * id3 +
         coords[3]) * in_stride + coords[1];
    coords[axis] += axis_offset;
    const std::size_t out_idx =
        (static_cast<std::size_t>(coords[0] * od2 + coords[2]) * od3 +
         coords[3]) * out_stride + coords[1];
    output[out_idx] = __ldg(input + in_idx);
}

// NHWC DepthToSpace: stages one input pixel [in_channels] into SMEM with
// 128-bit coalesced reads, then scatters to [blk*blk] output pixels with
// coalesced channel writes.
// Launch: grid = (input_width, input_height, batch_count), block = (128, 1, 1).
extern "C" __global__ void mlvc_depth_to_space_nhwc_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int batch_count, int in_channels, int input_height, int input_width,
    int block_size, int mode, int in_stride, int out_stride)
{
    const int ix = static_cast<int>(blockIdx.x);
    const int iy = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    if (ix >= input_width || iy >= input_height || b >= batch_count)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int blk_sq = block_size * block_size;
    const int out_channels = in_channels / blk_sq;
    const int out_width = input_width * block_size;
    const int out_height = input_height * block_size;
    const __half* in_pix = input +
        (static_cast<std::size_t>(b * input_height + iy) * input_width + ix) *
            in_stride;

    if (in_channels <= 2048) {
        __shared__ __align__(16) __half smem[2048];
        const bool vec8 = ((in_channels & 7) == 0) &&
            ((reinterpret_cast<std::uintptr_t>(in_pix) & 15U) == 0);
        if (vec8) {
            const int vec_n = in_channels >> 3;
            for (int v = tid; v < vec_n; v += static_cast<int>(blockDim.x)) {
                reinterpret_cast<uint4*>(smem)[v] =
                    __ldg(reinterpret_cast<const uint4*>(in_pix) + v);
            }
        } else {
            for (int c = tid; c < in_channels; c += static_cast<int>(blockDim.x))
                smem[c] = __ldg(in_pix + c);
        }
        __syncthreads();

        if (out_channels >= 32) {
            for (int sub = 0; sub < blk_sq; ++sub) {
                const int oy = sub / block_size;
                const int ox = sub - oy * block_size;
                __half* out_pix = output +
                    (static_cast<std::size_t>(
                         b * out_height + iy * block_size + oy) * out_width +
                     ix * block_size + ox) * out_stride;
                for (int oc = tid; oc < out_channels;
                     oc += static_cast<int>(blockDim.x)) {
                    const int ic = mode == 0 ? (sub * out_channels + oc)
                                             : (oc * blk_sq + sub);
                    out_pix[oc] = smem[ic];
                }
            }
        } else {
            for (int idx = tid; idx < in_channels;
                 idx += static_cast<int>(blockDim.x)) {
                const int sub = idx / out_channels;
                const int oc = idx - sub * out_channels;
                const int oy = sub / block_size;
                const int ox = sub - oy * block_size;
                const int ic = mode == 0 ? idx : (oc * blk_sq + sub);
                __half* out_pix = output +
                    (static_cast<std::size_t>(
                         b * out_height + iy * block_size + oy) * out_width +
                     ix * block_size + ox) * out_stride;
                out_pix[oc] = smem[ic];
            }
        }
        return;
    }

    // Fallback for in_channels > 2048.
    for (int sub = 0; sub < blk_sq; ++sub) {
        const int oy = sub / block_size;
        const int ox = sub - oy * block_size;
        __half* out_pix = output +
            (static_cast<std::size_t>(b * out_height + iy * block_size + oy) *
                 out_width + ix * block_size + ox) * out_stride;
        for (int oc = tid; oc < out_channels;
             oc += static_cast<int>(blockDim.x)) {
            const int ic = mode == 0 ? (sub * out_channels + oc)
                                     : (oc * blk_sq + sub);
            out_pix[oc] = __ldg(in_pix + ic);
        }
    }
}

// NHWC SpaceToDepth (CRD pixel_unshuffle).
// Launch: grid = (out_width, out_height, batch_count), block = (128, 1, 1).
extern "C" __global__ void mlvc_space_to_depth_nhwc_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int batch_count, int in_channels, int input_height, int input_width,
    int block_size, int in_stride, int out_stride)
{
    const int ox = static_cast<int>(blockIdx.x);
    const int oy = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    const int out_width = input_width / block_size;
    const int out_height = input_height / block_size;
    if (ox >= out_width || oy >= out_height || b >= batch_count)
        return;

    const int blk_sq = block_size * block_size;
    const int out_channels = in_channels * blk_sq;
    __half* out_pix = output +
        (static_cast<std::size_t>(b * out_height + oy) * out_width + ox) *
            out_stride;
    for (int oc = static_cast<int>(threadIdx.x); oc < out_channels;
         oc += static_cast<int>(blockDim.x)) {
        const int ic = oc / blk_sq;
        const int rem = oc - ic * blk_sq;
        const int dy = rem / block_size;
        const int dx = rem - dy * block_size;
        const int iy = oy * block_size + dy;
        const int ix = ox * block_size + dx;
        out_pix[oc] = __ldg(
            input + (static_cast<std::size_t>(b * input_height + iy) *
                         input_width + ix) * in_stride + ic);
    }
}

// Fused Round + NHWC -> NCHW for external z_raw latent output.
extern "C" __global__ void mlvc_round_nhwc_to_nchw_fp16(
    const __half* __restrict__ input, __half* __restrict__ output,
    int batch_count, int channels, int spatial_count, int in_stride)
{
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int s = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    if (c >= channels || s >= spatial_count || b >= batch_count)
        return;
    const std::size_t in_idx =
        (static_cast<std::size_t>(b) * spatial_count + s) * in_stride + c;
    const std::size_t out_idx =
        (static_cast<std::size_t>(b) * channels + c) * spatial_count + s;
    output[out_idx] = __float2half_rn(
        nearbyintf(__half2float(__ldg(input + in_idx))));
}

