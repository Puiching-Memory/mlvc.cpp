// Fused recurrent-feature update used by the encoder tail.  The exported graph
// expresses this as three slices, two sigmoids, six binary operations, and a
// concat.  Keeping the fp16 rounding points here preserves those operator
// boundaries while avoiding eleven intermediate tensor round trips.
extern "C" __global__ void mlvc_feature_update_fp16(
    const __half* input, const __half* history, const __half* scale,
    __half* output,
    int batch_count, int channels, int spatial_count)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pairs_per_plane = (spatial_count + 1) / 2;
    const int plane = pair / pairs_per_plane;
    const int spatial = (pair - plane * pairs_per_plane) * 2;
    if (plane >= batch_count * channels || spatial >= spatial_count)
        return;

    const int batch = plane / channels;
    const int channel = plane - batch * channels;
    const std::size_t input_base =
        (static_cast<std::size_t>(batch) * channels * 3 + channel) *
        spatial_count + spatial;
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) *
        spatial_count + spatial;
    const std::size_t channel_stride =
        static_cast<std::size_t>(channels) * spatial_count;
    const __half one = __float2half_rn(1.0F);
    const __half channel_scale = scale[channel];

#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
        if (spatial + offset >= spatial_count)
            break;
        const __half a = input[input_base + offset];
        const __half b = input[input_base + channel_stride + offset];
        const __half c = input[input_base + channel_stride * 2 + offset];
        const __half previous = history[
            static_cast<std::size_t>(plane) * spatial_count + spatial + offset];
        const __half sigmoid_b = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(b))));
        const __half sigmoid_c = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(c))));
        const __half weighted_b = binary_value(sigmoid_b, previous, 1);
        const __half inverse_gate = binary_value(one, sigmoid_b, 2);
        const __half weighted_a = binary_value(inverse_gate, a, 1);
        const __half blended = binary_value(weighted_b, weighted_a, 0);
        const __half gated = binary_value(sigmoid_c, blended, 1);
        output[output_base + offset] =
            binary_value(gated, channel_scale, 1);
        output[output_base + channel_stride + offset] = blended;
    }
}

// Decoder variant of the recurrent-feature update.  The scaled and blended
// halves remain live as inputs to the reconstruction network, while the final
// concatenated feature is also materialized here to remove the deferred copy.
extern "C" __global__ void mlvc_feature_update_outputs_fp16(
    const __half* input, const __half* history, const __half* scale,
    __half* scaled_output, __half* blended_output, __half* feature_output,
    int batch_count, int channels, int spatial_count)
{
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pairs_per_plane = (spatial_count + 1) / 2;
    const int plane = pair / pairs_per_plane;
    const int spatial = (pair - plane * pairs_per_plane) * 2;
    if (plane >= batch_count * channels || spatial >= spatial_count)
        return;

    const int batch = plane / channels;
    const int channel = plane - batch * channels;
    const std::size_t input_base =
        (static_cast<std::size_t>(batch) * channels * 3 + channel) *
        spatial_count + spatial;
    const std::size_t side_output_base =
        static_cast<std::size_t>(plane) * spatial_count + spatial;
    const std::size_t feature_base =
        (static_cast<std::size_t>(batch) * channels * 2 + channel) *
        spatial_count + spatial;
    const std::size_t channel_stride =
        static_cast<std::size_t>(channels) * spatial_count;
    const __half one = __float2half_rn(1.0F);
    const __half channel_scale = scale[channel];

#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
        if (spatial + offset >= spatial_count)
            break;
        const __half a = input[input_base + offset];
        const __half b = input[input_base + channel_stride + offset];
        const __half c = input[input_base + channel_stride * 2 + offset];
        const __half previous = history[side_output_base + offset];
        const __half sigmoid_b = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(b))));
        const __half sigmoid_c = __float2half_rn(
            1.0F / (1.0F + expf(-__half2float(c))));
        const __half weighted_b = binary_value(sigmoid_b, previous, 1);
        const __half inverse_gate = binary_value(one, sigmoid_b, 2);
        const __half weighted_a = binary_value(inverse_gate, a, 1);
        const __half blended = binary_value(weighted_b, weighted_a, 0);
        const __half gated = binary_value(sigmoid_c, blended, 1);
        const __half scaled = binary_value(gated, channel_scale, 1);
        scaled_output[side_output_base + offset] = scaled;
        blended_output[side_output_base + offset] = blended;
        feature_output[feature_base + offset] = scaled;
        feature_output[feature_base + channel_stride + offset] = blended;
    }
}

__device__ __forceinline__ void feature_update_scalar(
    __half a, __half b, __half c_val, __half previous, __half channel_scale,
    __half& scaled_out, __half& blended_out)
{
    const __half one = __float2half_rn(1.0F);
    const __half sigmoid_b = __float2half_rn(
        1.0F / (1.0F + expf(-__half2float(b))));
    const __half sigmoid_c = __float2half_rn(
        1.0F / (1.0F + expf(-__half2float(c_val))));
    const __half weighted_b = binary_value(sigmoid_b, previous, 1);
    const __half inverse_gate = binary_value(one, sigmoid_b, 2);
    const __half weighted_a = binary_value(inverse_gate, a, 1);
    const __half blended = binary_value(weighted_b, weighted_a, 0);
    const __half gated = binary_value(sigmoid_c, blended, 1);
    scaled_out = binary_value(gated, channel_scale, 1);
    blended_out = blended;
}

__device__ __forceinline__ void feature_update_half2_bits(
    unsigned int a_bits, unsigned int b_bits, unsigned int c_bits,
    unsigned int p_bits, unsigned int s_bits,
    unsigned int& scaled_bits, unsigned int& blended_bits)
{
    const __half2 a = *reinterpret_cast<const __half2*>(&a_bits);
    const __half2 b = *reinterpret_cast<const __half2*>(&b_bits);
    const __half2 c = *reinterpret_cast<const __half2*>(&c_bits);
    const __half2 p = *reinterpret_cast<const __half2*>(&p_bits);
    const __half2 s = *reinterpret_cast<const __half2*>(&s_bits);
    __half sc0, bl0, sc1, bl1;
    feature_update_scalar(__low2half(a), __low2half(b), __low2half(c),
                          __low2half(p), __low2half(s), sc0, bl0);
    feature_update_scalar(__high2half(a), __high2half(b), __high2half(c),
                          __high2half(p), __high2half(s), sc1, bl1);
    const __half2 sc2 = __halves2half2(sc0, sc1);
    const __half2 bl2 = __halves2half2(bl0, bl1);
    scaled_bits = *reinterpret_cast<const unsigned int*>(&sc2);
    blended_bits = *reinterpret_cast<const unsigned int*>(&bl2);
}

// NHWC fused recurrent-feature update (128-bit uint4 when 16B aligned).
// Launch: 1D linear over batch_count * spatial_count * (vec8 ? channels / 8 : channels).
extern "C" __global__ void mlvc_feature_update_nhwc_fp16(
    const __half* __restrict__ input, const __half* __restrict__ history,
    const __half* __restrict__ scale, __half* __restrict__ output,
    int batch_count, int channels, int spatial_count,
    int input_stride, int history_stride, int output_stride)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixels =
        static_cast<std::size_t>(batch_count) * spatial_count;
    const bool vec8 =
        ((channels & 7) == 0) && ((input_stride & 7) == 0) &&
        ((history_stride & 7) == 0) && ((output_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(input) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(history) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(scale) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(output) & 15U) == 0);
    if (vec8) {
        const std::size_t vec_cols = static_cast<std::size_t>(channels >> 3);
        const std::size_t total = pixels * vec_cols;
        if (idx >= total)
            return;
        const std::size_t pix = idx / vec_cols;
        const std::size_t c = (idx - pix * vec_cols) << 3;
        const std::size_t in_base = pix * static_cast<std::size_t>(input_stride) + c;
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(input + in_base));
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(input + in_base + channels));
        const uint4 cv = __ldg(reinterpret_cast<const uint4*>(input + in_base + channels * 2));
        const uint4 prev = __ldg(reinterpret_cast<const uint4*>(
            history + pix * static_cast<std::size_t>(history_stride) + c));
        const uint4 sc = __ldg(reinterpret_cast<const uint4*>(scale + c));
        uint4 out_sc, out_bl;
        feature_update_half2_bits(a.x, b.x, cv.x, prev.x, sc.x, out_sc.x, out_bl.x);
        feature_update_half2_bits(a.y, b.y, cv.y, prev.y, sc.y, out_sc.y, out_bl.y);
        feature_update_half2_bits(a.z, b.z, cv.z, prev.z, sc.z, out_sc.z, out_bl.z);
        feature_update_half2_bits(a.w, b.w, cv.w, prev.w, sc.w, out_sc.w, out_bl.w);
        const std::size_t out_base = pix * static_cast<std::size_t>(output_stride) + c;
        *reinterpret_cast<uint4*>(output + out_base) = out_sc;
        *reinterpret_cast<uint4*>(output + out_base + channels) = out_bl;
        return;
    }
    const std::size_t total = pixels * static_cast<std::size_t>(channels);
    if (idx >= total)
        return;
    const std::size_t pix = idx / static_cast<std::size_t>(channels);
    const std::size_t c = idx - pix * static_cast<std::size_t>(channels);
    const std::size_t in_base = pix * static_cast<std::size_t>(input_stride) + c;
    __half scaled, blended;
    feature_update_scalar(
        __ldg(input + in_base), __ldg(input + in_base + channels),
        __ldg(input + in_base + channels * 2),
        __ldg(history + pix * static_cast<std::size_t>(history_stride) + c),
        __ldg(scale + c), scaled, blended);
    const std::size_t out_base = pix * static_cast<std::size_t>(output_stride) + c;
    output[out_base] = scaled;
    output[out_base + channels] = blended;
}

extern "C" __global__ void mlvc_feature_update_outputs_nhwc_fp16(
    const __half* __restrict__ input, const __half* __restrict__ history,
    const __half* __restrict__ scale, __half* __restrict__ scaled_output,
    __half* __restrict__ blended_output, __half* __restrict__ feature_output,
    int batch_count, int channels, int spatial_count,
    int input_stride, int history_stride, int scaled_stride,
    int blended_stride, int feature_stride)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixels =
        static_cast<std::size_t>(batch_count) * spatial_count;
    const bool vec8 =
        ((channels & 7) == 0) && ((input_stride & 7) == 0) &&
        ((history_stride & 7) == 0) && ((scaled_stride & 7) == 0) &&
        ((blended_stride & 7) == 0) && ((feature_stride & 7) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(input) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(history) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(scale) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(scaled_output) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(blended_output) & 15U) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(feature_output) & 15U) == 0);
    if (vec8) {
        const std::size_t vec_cols = static_cast<std::size_t>(channels >> 3);
        const std::size_t total = pixels * vec_cols;
        if (idx >= total)
            return;
        const std::size_t pix = idx / vec_cols;
        const std::size_t c = (idx - pix * vec_cols) << 3;
        const std::size_t in_base = pix * static_cast<std::size_t>(input_stride) + c;
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(input + in_base));
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(input + in_base + channels));
        const uint4 cv = __ldg(reinterpret_cast<const uint4*>(input + in_base + channels * 2));
        const uint4 prev = __ldg(reinterpret_cast<const uint4*>(
            history + pix * static_cast<std::size_t>(history_stride) + c));
        const uint4 sc = __ldg(reinterpret_cast<const uint4*>(scale + c));
        uint4 out_sc, out_bl;
        feature_update_half2_bits(a.x, b.x, cv.x, prev.x, sc.x, out_sc.x, out_bl.x);
        feature_update_half2_bits(a.y, b.y, cv.y, prev.y, sc.y, out_sc.y, out_bl.y);
        feature_update_half2_bits(a.z, b.z, cv.z, prev.z, sc.z, out_sc.z, out_bl.z);
        feature_update_half2_bits(a.w, b.w, cv.w, prev.w, sc.w, out_sc.w, out_bl.w);
        *reinterpret_cast<uint4*>(
            scaled_output + pix * static_cast<std::size_t>(scaled_stride) + c) = out_sc;
        *reinterpret_cast<uint4*>(
            blended_output + pix * static_cast<std::size_t>(blended_stride) + c) = out_bl;
        const std::size_t feat_base =
            pix * static_cast<std::size_t>(feature_stride) + c;
        *reinterpret_cast<uint4*>(feature_output + feat_base) = out_sc;
        *reinterpret_cast<uint4*>(feature_output + feat_base + channels) = out_bl;
        return;
    }
    const std::size_t total = pixels * static_cast<std::size_t>(channels);
    if (idx >= total)
        return;
    const std::size_t pix = idx / static_cast<std::size_t>(channels);
    const std::size_t c = idx - pix * static_cast<std::size_t>(channels);
    const std::size_t in_base = pix * static_cast<std::size_t>(input_stride) + c;
    __half scaled, blended;
    feature_update_scalar(
        __ldg(input + in_base), __ldg(input + in_base + channels),
        __ldg(input + in_base + channels * 2),
        __ldg(history + pix * static_cast<std::size_t>(history_stride) + c),
        __ldg(scale + c), scaled, blended);
    scaled_output[pix * static_cast<std::size_t>(scaled_stride) + c] = scaled;
    blended_output[pix * static_cast<std::size_t>(blended_stride) + c] = blended;
    const std::size_t feat_base =
        pix * static_cast<std::size_t>(feature_stride) + c;
    feature_output[feat_base] = scaled;
    feature_output[feat_base + channels] = blended;
}
