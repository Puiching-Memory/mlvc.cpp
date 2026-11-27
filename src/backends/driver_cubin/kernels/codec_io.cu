namespace {

__device__ __forceinline__ std::uint8_t quantize_yuv(float value)
{
    value = fminf(fmaxf(value, 0.0F), 1.0F);
    return static_cast<std::uint8_t>(__float2int_rn(value * 255.0F));
}

__device__ __forceinline__ float model_sample(
    const __half* input, int channel, int y, int x, int width, int height,
    int model_width, int model_height, int pad_left, int pad_top, int rotated,
    float inverse_pixel_range)
{
    int model_y = rotated ? x : y;
    int model_x = rotated ? y : x;
    model_y += pad_top;
    model_x += pad_left;
    if (model_y < 0 || model_y >= model_height ||
        model_x < 0 || model_x >= model_width ||
        y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
    }
    const std::size_t index =
        (static_cast<std::size_t>(channel) * model_height + model_y) *
            model_width + model_x;
    return __half2float(input[index]) * inverse_pixel_range;
}

}  // namespace

extern "C" __global__ void mlvc_yuv420_to_nchw_fp16(
    const std::uint8_t* __restrict__ input, const __half* __restrict__ byte_lut,
    __half* __restrict__ output, int width, int height, int model_width,
    int model_height, int rotated)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int channel = static_cast<int>(blockIdx.z);
    if (x >= model_width || y >= model_height || channel >= 3)
        return;

    const int active_width = rotated ? height : width;
    const int active_height = rotated ? width : height;
    const int edge_y = y < active_height ? y : active_height - 1;
    const int edge_x = x < active_width ? x : active_width - 1;
    const int source_y = rotated ? edge_x : edge_y;
    const int source_x = rotated ? edge_y : edge_x;
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    const std::size_t uv_size = y_size >> 2;

    std::uint8_t sample = 0;
    if (channel == 0) {
        sample = __ldg(input + static_cast<std::size_t>(source_y) * width + source_x);
    } else {
        const std::size_t plane_offset =
            y_size + static_cast<std::size_t>(channel - 1) * uv_size;
        sample = __ldg(
            input + plane_offset +
            static_cast<std::size_t>(source_y >> 1) * (width >> 1) +
            (source_x >> 1));
    }
    const std::size_t out_index =
        (static_cast<std::size_t>(channel) * model_height + y) * model_width + x;
    output[out_index] = __ldg(byte_lut + sample);
}

extern "C" __global__ void mlvc_nchw_to_yuv420(
    const __half* __restrict__ input, std::uint8_t* __restrict__ output,
    int width, int height, int model_width, int model_height, int pad_left,
    int pad_top, int rotated, float inverse_pixel_range)
{
    const int uv_width = width >> 1;
    const int uv_height = height >> 1;
    const int uv_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (uv_x >= uv_width || uv_y >= uv_height)
        return;

    const int y0 = uv_y << 1;
    const int x0 = uv_x << 1;
    const std::size_t y_count = static_cast<std::size_t>(width) * height;
    const std::size_t uv_count = y_count >> 2;
    const std::size_t uv_index = static_cast<std::size_t>(uv_y) * uv_width + uv_x;

    float u = 0.0F;
    float v = 0.0F;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int py = y0 + dy;
        std::uint8_t* y_row = output + static_cast<std::size_t>(py) * width + x0;
        const std::uint8_t y_l = quantize_yuv(model_sample(
            input, 0, py, x0, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range));
        const std::uint8_t y_r = quantize_yuv(model_sample(
            input, 0, py, x0 + 1, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range));
        *reinterpret_cast<uchar2*>(y_row) = make_uchar2(y_l, y_r);
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            u = __fadd_rn(u, model_sample(
                input, 1, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
            v = __fadd_rn(v, model_sample(
                input, 2, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
        }
    }
    output[y_count + uv_index] = quantize_yuv(__fmul_rn(u, 0.25F));
    output[y_count + uv_count + uv_index] =
        quantize_yuv(__fmul_rn(v, 0.25F));
}

namespace {

__device__ __forceinline__ std::uint8_t sample_yuv420_plane(
    const std::uint8_t* __restrict__ input, int plane, int py, int px,
    int width, int height, int rotated)
{
    const int active_width = rotated ? height : width;
    const int active_height = rotated ? width : height;
    const int edge_y = py < active_height ? py : active_height - 1;
    const int edge_x = px < active_width ? px : active_width - 1;
    const int source_y = rotated ? edge_x : edge_y;
    const int source_x = rotated ? edge_y : edge_x;
    const std::size_t y_size = static_cast<std::size_t>(width) * height;
    if (plane == 0) {
        return __ldg(
            input + static_cast<std::size_t>(source_y) * width + source_x);
    }
    const std::size_t uv_size = y_size >> 2;
    const std::size_t plane_offset =
        y_size + static_cast<std::size_t>(plane - 1) * uv_size;
    return __ldg(
        input + plane_offset +
        static_cast<std::size_t>(source_y >> 1) * (width >> 1) +
        (source_x >> 1));
}

__device__ __forceinline__ float model_sample_nhwc(
    const __half* input, int channel, int y, int x, int width, int height,
    int model_width, int model_height, int pad_left, int pad_top, int rotated,
    float inverse_pixel_range)
{
    int model_y = rotated ? x : y;
    int model_x = rotated ? y : x;
    model_y += pad_top;
    model_x += pad_left;
    if (model_y < 0 || model_y >= model_height ||
        model_x < 0 || model_x >= model_width ||
        y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
    }
    const std::size_t index =
        (static_cast<std::size_t>(model_y) * model_width + model_x) * 3 +
        channel;
    return __half2float(__ldg(input + index)) * inverse_pixel_range;
}

__device__ __forceinline__ float model_sample_d2s8_nhwc(
    const __half* input, int channel, int y, int x, int width, int height,
    int model_width, int model_height, int pad_left, int pad_top, int rotated,
    float inverse_pixel_range, int has_clip, float clip_min, float clip_max)
{
    int model_y = rotated ? x : y;
    int model_x = rotated ? y : x;
    model_y += pad_top;
    model_x += pad_left;
    if (model_y < 0 || model_y >= model_height ||
        model_x < 0 || model_x >= model_width ||
        y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0F;
    }
    const int bw = model_width >> 3;
    const int by = model_y >> 3;
    const int bx = model_x >> 3;
    const int oy = model_y & 7;
    const int ox = model_x & 7;
    const int in_c = (channel << 6) | (oy << 3) | ox;
    const std::size_t index =
        (static_cast<std::size_t>(by) * bw + bx) * 192 + in_c;
    float v = __half2float(__ldg(input + index));
    if (has_clip)
        v = __half2float(__float2half_rn(fminf(fmaxf(v, clip_min), clip_max)));
    return v * inverse_pixel_range;
}

}  // namespace

extern "C" __global__ void mlvc_yuv420_to_nhwc_fp16(
    const std::uint8_t* __restrict__ input, const __half* __restrict__ byte_lut,
    __half* __restrict__ output, int width, int height, int model_width,
    int model_height, int rotated)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= model_width || y >= model_height)
        return;

    const std::size_t base =
        (static_cast<std::size_t>(y) * model_width + x) * 3;
    output[base + 0] = __ldg(
        byte_lut + sample_yuv420_plane(input, 0, y, x, width, height, rotated));
    output[base + 1] = __ldg(
        byte_lut + sample_yuv420_plane(input, 1, y, x, width, height, rotated));
    output[base + 2] = __ldg(
        byte_lut + sample_yuv420_plane(input, 2, y, x, width, height, rotated));
}

// Fuses YUV420 -> FP16 + SpaceToDepth(block=8, CRD) directly into NHWC
// [1, model_height/8, model_width/8, 192] using 128-bit coalesced stores.
// Launch: block = (24, 8, 1), grid = (ceil((model_width/8)/8), model_height/8, 1).
extern "C" __global__ void mlvc_yuv420_to_s2d8_nhwc_fp16(
    const std::uint8_t* __restrict__ input, const __half* __restrict__ byte_lut,
    __half* __restrict__ output, int width, int height, int model_width,
    int model_height, int rotated)
{
    const int cg = static_cast<int>(threadIdx.x);  // 0..23 -> 8 channels each
    const int bw = model_width >> 3;
    const int bh = model_height >> 3;
    const int bx = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    const int by = static_cast<int>(blockIdx.y);
    if (cg >= 24 || bx >= bw || by >= bh)
        return;

    const int plane = cg >= 16 ? 2 : (cg >= 8 ? 1 : 0);
    const int dy = cg - (plane << 3);
    const int py = (by << 3) + dy;
    const int px0 = bx << 3;

    __half vals[8];
#pragma unroll
    for (int dx = 0; dx < 8; ++dx) {
        const std::uint8_t sample = sample_yuv420_plane(
            input, plane, py, px0 + dx, width, height, rotated);
        vals[dx] = __ldg(byte_lut + sample);
    }

    const __half2 h0 = __halves2half2(vals[0], vals[1]);
    const __half2 h1 = __halves2half2(vals[2], vals[3]);
    const __half2 h2 = __halves2half2(vals[4], vals[5]);
    const __half2 h3 = __halves2half2(vals[6], vals[7]);
    uint4 packed;
    packed.x = *reinterpret_cast<const unsigned int*>(&h0);
    packed.y = *reinterpret_cast<const unsigned int*>(&h1);
    packed.z = *reinterpret_cast<const unsigned int*>(&h2);
    packed.w = *reinterpret_cast<const unsigned int*>(&h3);
    const std::size_t out_offset =
        (static_cast<std::size_t>(by) * bw + bx) * 192 + (cg << 3);
    *reinterpret_cast<uint4*>(output + out_offset) = packed;
}

extern "C" __global__ void mlvc_nhwc_to_yuv420(
    const __half* __restrict__ input, std::uint8_t* __restrict__ output,
    int width, int height, int model_width, int model_height, int pad_left,
    int pad_top, int rotated, float inverse_pixel_range)
{
    const int uv_width = width >> 1;
    const int uv_height = height >> 1;
    const int uv_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (uv_x >= uv_width || uv_y >= uv_height)
        return;

    const int y0 = uv_y << 1;
    const int x0 = uv_x << 1;
    const std::size_t y_count = static_cast<std::size_t>(width) * height;
    const std::size_t uv_count = y_count >> 2;
    const std::size_t uv_index = static_cast<std::size_t>(uv_y) * uv_width + uv_x;

    float u = 0.0F;
    float v = 0.0F;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int py = y0 + dy;
        std::uint8_t* y_row = output + static_cast<std::size_t>(py) * width + x0;
        const std::uint8_t y_l = quantize_yuv(model_sample_nhwc(
            input, 0, py, x0, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range));
        const std::uint8_t y_r = quantize_yuv(model_sample_nhwc(
            input, 0, py, x0 + 1, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range));
        *reinterpret_cast<uchar2*>(y_row) = make_uchar2(y_l, y_r);
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            u = __fadd_rn(u, model_sample_nhwc(
                input, 1, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
            v = __fadd_rn(v, model_sample_nhwc(
                input, 2, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range));
        }
    }
    output[y_count + uv_index] = quantize_yuv(__fmul_rn(u, 0.25F));
    output[y_count + uv_count + uv_index] =
        quantize_yuv(__fmul_rn(v, 0.25F));
}

// Fuses DepthToSpace(block=8, CRD) + optional Clip + NHWC -> YUV420 directly
// from [1, model_height/8, model_width/8, 192] NHWC.
extern "C" __global__ void mlvc_d2s8_nhwc_to_yuv420(
    const __half* __restrict__ input, std::uint8_t* __restrict__ output,
    int width, int height, int model_width, int model_height, int pad_left,
    int pad_top, int rotated, float inverse_pixel_range, int has_clip,
    float clip_min, float clip_max)
{
    const int uv_width = width >> 1;
    const int uv_height = height >> 1;
    const int uv_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (uv_x >= uv_width || uv_y >= uv_height)
        return;

    const int y0 = uv_y << 1;
    const int x0 = uv_x << 1;
    const std::size_t y_count = static_cast<std::size_t>(width) * height;
    const std::size_t uv_count = y_count >> 2;
    const std::size_t uv_index = static_cast<std::size_t>(uv_y) * uv_width + uv_x;

    float u = 0.0F;
    float v = 0.0F;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int py = y0 + dy;
        std::uint8_t* y_row = output + static_cast<std::size_t>(py) * width + x0;
        const std::uint8_t y_l = quantize_yuv(model_sample_d2s8_nhwc(
            input, 0, py, x0, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range,
            has_clip, clip_min, clip_max));
        const std::uint8_t y_r = quantize_yuv(model_sample_d2s8_nhwc(
            input, 0, py, x0 + 1, width, height, model_width, model_height,
            pad_left, pad_top, rotated, inverse_pixel_range,
            has_clip, clip_min, clip_max));
        *reinterpret_cast<uchar2*>(y_row) = make_uchar2(y_l, y_r);
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            u = __fadd_rn(u, model_sample_d2s8_nhwc(
                input, 1, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range,
                has_clip, clip_min, clip_max));
            v = __fadd_rn(v, model_sample_d2s8_nhwc(
                input, 2, py, x0 + dx, width, height, model_width, model_height,
                pad_left, pad_top, rotated, inverse_pixel_range,
                has_clip, clip_min, clip_max));
        }
    }
    output[y_count + uv_index] = quantize_yuv(__fmul_rn(u, 0.25F));
    output[y_count + uv_count + uv_index] =
        quantize_yuv(__fmul_rn(v, 0.25F));
}
