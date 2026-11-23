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
