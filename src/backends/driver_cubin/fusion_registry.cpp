#include "aot_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mlvc::driver_cubin_backend {

std::size_t AotGraph::try_execute_y0_tail(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 18 >= nodes.size())
        return 0;

    auto inputs = [&](std::size_t node_idx) {
        return nodes.at(node_idx).inputs;
    };
    auto output = [&](std::size_t node_idx) -> TensorId {
        const auto names = nodes.at(node_idx).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    if (nodes.at(index).op != Op::SLICE ||
        nodes.at(index + 1).op != Op::SLICE ||
        nodes.at(index + 2).op != Op::CLIP ||
        nodes.at(index + 3).op != Op::RECIPROCAL) {
        return 0;
    }

    const auto slice0_inputs = inputs(index);
    const auto slice1_inputs = inputs(index + 1);
    if (slice0_inputs.empty() || slice1_inputs.empty() ||
        slice0_inputs[0] == 0 || slice0_inputs[0] != slice1_inputs[0]) {
        return 0;
    }
    const TensorId prior_name = slice0_inputs[0];
    const TensorId slice0 = output(index);
    const TensorId slice1 = output(index + 1);
    const TensorId clipped = output(index + 2);
    const TensorId reciprocal_raw = output(index + 3);
    if (slice0 == 0 || slice1 == 0 || clipped == 0 || reciprocal_raw == 0)
        return 0;

    const auto clip_inputs = inputs(index + 2);
    if ((clip_inputs.size() != 2 && clip_inputs.size() != 3) ||
        clip_inputs[0] != slice0 || clip_inputs[1] == 0 ||
        inputs(index + 3) != std::vector<TensorId>{clipped}) {
        return 0;
    }
    if (value(clip_inputs[1]).dtype != DataType::FP16 ||
        value(clip_inputs[1]).shape != std::vector<int64_t>{1}) {
        return 0;
    }
    float clip_max = 65504.0F;
    if (clip_inputs.size() == 3 && clip_inputs[2] != 0) {
        if (value(clip_inputs[2]).dtype != DataType::FP16 ||
            value(clip_inputs[2]).shape != std::vector<int64_t>{1}) {
            return 0;
        }
        clip_max = scalar_fp16(clip_inputs[2]);
        if (clip_max < 65504.0F)
            return 0;
    }

    // PyTorch AOT lowering emits an optional `* 1.0` Mul right after Reciprocal.
    std::size_t cursor = index + 4;
    TensorId reciprocal = reciprocal_raw;
    if (cursor + 15 <= nodes.size() && nodes.at(cursor).op == Op::MUL &&
        inputs(cursor).size() == 2 && output(cursor) != 0) {
        TensorId other = 0;
        if (inputs(cursor)[0] == reciprocal_raw)
            other = inputs(cursor)[1];
        else if (inputs(cursor)[1] == reciprocal_raw)
            other = inputs(cursor)[0];
        if (other != 0 && value(other).initializer &&
            value(other).dtype == DataType::FP16 &&
            value(other).shape == std::vector<int64_t>{1} &&
            std::abs(scalar_fp16(other) - 1.0F) <= 1.0e-4F) {
            if (tensor_consumer_count(reciprocal_raw) != 1)
                return 0;
            reciprocal = output(cursor);
            ++cursor;
        }
    }
    if (cursor + 15 > nodes.size())
        return 0;

    static constexpr std::array<Op, 15> tail_ops{
        Op::MUL, Op::SUB, Op::MUL, Op::SLICE, Op::SLICE, Op::ADD, Op::ROUND,
        Op::SLICE, Op::SLICE, Op::ADD, Op::MUL, Op::ADD, Op::MUL,
        Op::CONCAT, Op::CONCAT};
    for (std::size_t offset = 0; offset < tail_ops.size(); ++offset) {
        if (nodes.at(cursor + offset).op != tail_ops[offset])
            return 0;
    }

    const TensorId normalized = output(cursor + 0);
    const TensorId difference = output(cursor + 1);
    const TensorId scaled = output(cursor + 2);
    const TensorId scaled0 = output(cursor + 3);
    const TensorId scaled1 = output(cursor + 4);
    const TensorId rounded_input = output(cursor + 5);
    const TensorId y_raw = output(cursor + 6);
    const TensorId prior0 = output(cursor + 7);
    const TensorId prior1 = output(cursor + 8);
    const TensorId updated_input0 = output(cursor + 9);
    const TensorId updated0 = output(cursor + 10);
    const TensorId updated_input1 = output(cursor + 11);
    const TensorId updated1 = output(cursor + 12);
    const TensorId concat2 = output(cursor + 13);
    const TensorId concat3 = output(cursor + 14);
    if (normalized == 0 || difference == 0 || scaled == 0 || scaled0 == 0 ||
        scaled1 == 0 || rounded_input == 0 || y_raw == 0 || prior0 == 0 ||
        prior1 == 0 || updated_input0 == 0 || updated0 == 0 ||
        updated_input1 == 0 || updated1 == 0 || concat2 == 0 || concat3 == 0) {
        return 0;
    }

    TensorId latent_name = 0;
    if (inputs(cursor + 0).size() != 2)
        return 0;
    if (inputs(cursor + 0)[1] == reciprocal)
        latent_name = inputs(cursor + 0)[0];
    else if (inputs(cursor + 0)[0] == reciprocal)
        latent_name = inputs(cursor + 0)[1];
    else
        return 0;

    if (latent_name == 0 ||
        inputs(cursor + 1) != std::vector<TensorId>{normalized, slice1} ||
        inputs(cursor + 2).size() != 2 || inputs(cursor + 2)[0] != difference ||
        inputs(cursor + 2)[1] == 0 || inputs(cursor + 3).empty() ||
        inputs(cursor + 3)[0] != scaled || inputs(cursor + 4).empty() ||
        inputs(cursor + 4)[0] != scaled ||
        inputs(cursor + 5) != std::vector<TensorId>{scaled0, scaled1} ||
        inputs(cursor + 6) != std::vector<TensorId>{rounded_input} ||
        inputs(cursor + 7).empty() || inputs(cursor + 7)[0] != slice1 ||
        inputs(cursor + 8).empty() || inputs(cursor + 8)[0] != slice1 ||
        inputs(cursor + 9) != std::vector<TensorId>{y_raw, prior0} ||
        inputs(cursor + 10).size() != 2 ||
        inputs(cursor + 10)[0] != updated_input0 ||
        inputs(cursor + 10)[1] == 0 ||
        inputs(cursor + 11) != std::vector<TensorId>{y_raw, prior1} ||
        inputs(cursor + 12).size() != 2 ||
        inputs(cursor + 12)[0] != updated_input1 ||
        inputs(cursor + 12)[1] == 0 ||
        inputs(cursor + 13) != std::vector<TensorId>{updated0, updated1} ||
        inputs(cursor + 14) != std::vector<TensorId>{concat2, prior_name} ||
        nodes.at(cursor + 13).axis_or_zero() != 1 ||
        nodes.at(cursor + 14).axis_or_zero() != 1) {
        return 0;
    }

    const std::array<std::pair<const TensorId*, std::size_t>, 19>
        expected_consumers{{
            {&prior_name, 3}, {&slice0, 1}, {&slice1, 3},
            {&clipped, 2}, {&reciprocal, 1}, {&normalized, 2},
            {&difference, 1}, {&scaled, 2}, {&scaled0, 1},
            {&scaled1, 1}, {&rounded_input, 1}, {&y_raw, 2},
            {&prior0, 1}, {&prior1, 1}, {&updated_input0, 1},
            {&updated0, 1}, {&updated_input1, 1}, {&updated1, 1},
            {&concat2, 2}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (tensor_consumer_count(*name) != expected)
            return 0;
    }

    const Value& prior_value = value(prior_name);
    const Value& latent_value = value(latent_name);
    const Value& quant_scale = value(inputs(cursor + 2)[1]);
    const Value& update_scale0 = value(inputs(cursor + 10)[1]);
    const Value& update_scale1 = value(inputs(cursor + 12)[1]);
    const Value& normalized_value = value(normalized);
    const Value& clipped_value = value(clipped);
    const Value& raw_value = value(y_raw);
    const Value& concat2_value = value(concat2);
    const Value& concat3_value = value(concat3);
    if (prior_value.dtype != DataType::FP16 || latent_value.dtype != DataType::FP16 ||
        quant_scale.dtype != DataType::FP16 || update_scale0.dtype != DataType::FP16 ||
        update_scale1.dtype != DataType::FP16 || normalized_value.dtype != DataType::FP16 ||
        clipped_value.dtype != DataType::FP16 || raw_value.dtype != DataType::FP16 ||
        concat2_value.dtype != DataType::FP16 || concat3_value.dtype != DataType::FP16 ||
        prior_value.shape.size() != 4 || latent_value.shape.size() != 4 ||
        raw_value.shape.size() != 4 || concat3_value.shape.size() != 4) {
        return 0;
    }

    int quant_channels = static_cast<int>(raw_value.shape[1]);
    if (quant_channels < 16)
        return 0;
    const std::vector<int64_t> raw_shape{
        raw_value.shape[0], raw_value.shape[1], raw_value.shape[2],
        raw_value.shape[3]};
    const std::vector<int64_t> normalized_expected{
        raw_shape[0], raw_shape[1] * 2, raw_shape[2], raw_shape[3]};
    const std::vector<int64_t> prior_expected{
        raw_shape[0], raw_shape[1] * 4, raw_shape[2], raw_shape[3]};
    const std::vector<int64_t> concat3_expected{
        raw_shape[0], raw_shape[1] * 6, raw_shape[2], raw_shape[3]};
    if (quant_channels <= 0 || latent_value.shape != normalized_expected ||
        normalized_value.shape != normalized_expected ||
        clipped_value.shape != normalized_expected ||
        quant_scale.shape != normalized_expected ||
        update_scale0.shape != raw_shape || update_scale1.shape != raw_shape ||
        prior_value.shape != prior_expected || value(slice0).shape !=
            std::vector<int64_t>{raw_shape[0], raw_shape[1] * 2,
                                 raw_shape[2], raw_shape[3]} ||
        value(slice1).shape != value(slice0).shape ||
        value(scaled).shape != normalized_expected ||
        value(scaled0).shape != raw_shape || value(scaled1).shape != raw_shape ||
        value(rounded_input).shape != raw_shape ||
        value(prior0).shape != raw_shape || value(prior1).shape != raw_shape ||
        value(updated_input0).shape != raw_shape ||
        value(updated_input1).shape != raw_shape ||
        value(updated0).shape != raw_shape || value(updated1).shape != raw_shape ||
        concat2_value.shape != normalized_expected ||
        concat3_value.shape != concat3_expected) {
        return 0;
    }
    if (!is_channel_slice(nodes.at(index), 0, quant_channels * 2) ||
        !is_channel_slice(nodes.at(index + 1), quant_channels * 2,
                          quant_channels * 4) ||
        !is_channel_slice(nodes.at(cursor + 3), 0, quant_channels) ||
        !is_channel_slice(nodes.at(cursor + 4), quant_channels,
                          quant_channels * 2) ||
        !is_channel_slice(nodes.at(cursor + 7), 0, quant_channels) ||
        !is_channel_slice(nodes.at(cursor + 8), quant_channels,
                          quant_channels * 2)) {
        return 0;
    }

    const int batch_count = static_cast<int>(raw_shape[0]);
    const bool raw_in_place_latent =
        !use_nhwc_ && batch_count == 1 &&
        raw_value.address == latent_value.address;
    if (use_nhwc_ &&
        (!is_packed_nhwc(normalized_value) || !is_packed_nhwc(clipped_value) ||
         !is_packed_nhwc(concat2_value))) {
        return 0;
    }
    if (ranges_overlap(concat3_value, prior_value) ||
        ranges_overlap(concat3_value, latent_value) ||
        ranges_overlap(concat3_value, normalized_value) ||
        ranges_overlap(concat3_value, clipped_value) ||
        ranges_overlap(concat3_value, raw_value) ||
        ranges_overlap(concat3_value, concat2_value) ||
        ranges_overlap(concat2_value, prior_value) ||
        ranges_overlap(concat2_value, normalized_value) ||
        ranges_overlap(concat2_value, clipped_value) ||
        (ranges_overlap(concat2_value, latent_value) &&
         (concat2_value.address != latent_value.address ||
          (use_nhwc_ && value_stride(latent_value) != quant_channels * 2))) ||
        ranges_overlap(raw_value, prior_value) ||
        ranges_overlap(raw_value, normalized_value) ||
        ranges_overlap(raw_value, clipped_value) ||
        ranges_overlap(raw_value, concat2_value) ||
        (ranges_overlap(raw_value, latent_value) && !raw_in_place_latent) ||
        (ranges_overlap(normalized_value, latent_value) &&
         (normalized_value.address != latent_value.address ||
          (use_nhwc_ && value_stride(latent_value) != quant_channels * 2))) ||
        ranges_overlap(normalized_value, prior_value) ||
        ranges_overlap(normalized_value, clipped_value) ||
        ranges_overlap(clipped_value, prior_value) ||
        ranges_overlap(clipped_value, latent_value)) {
        return 0;
    }

    if (batch_initializing_cutlass_)
        return (cursor + 15) - index;
    DeviceAddress latent = latent_value.address;
    DeviceAddress prior = prior_value.address;
    DeviceAddress quant = quant_scale.address;
    DeviceAddress update0 = update_scale0.address;
    DeviceAddress update1 = update_scale1.address;
    DeviceAddress normalized_address = normalized_value.address;
    DeviceAddress clipped_address = clipped_value.address;
    DeviceAddress raw = raw_value.address;
    DeviceAddress concat2_address = concat2_value.address;
    DeviceAddress concat3_address = concat3_value.address;
    float clip_min = scalar_fp16(clip_inputs[1]);
    int batch_param = batch_count;
    int spatial_count = static_cast<int>(raw_shape[2] * raw_shape[3]);
    if (use_nhwc_) {
        int latent_stride = value_stride(latent_value);
        int prior_stride = value_stride(prior_value);
        int concat3_stride = value_stride(concat3_value);
        void* parameters[] = {
            &latent, &prior, &quant, &update0, &update1, &normalized_address,
            &clipped_address, &raw, &concat2_address, &concat3_address,
            &clip_min, &clip_max, &batch_param, &quant_channels, &spatial_count,
            &latent_stride, &prior_stride, &concat3_stride};
        driver_.launch(
            y0_tail_nhwc_,
            {divide_up(static_cast<std::size_t>(quant_channels), 16),
             divide_up(static_cast<std::size_t>(spatial_count), 16),
             static_cast<unsigned int>(batch_count)},
            {16, 16, 1}, 0, parameters);
        return (cursor + 15) - index;
    }
    void* parameters[] = {
        &latent, &prior, &quant, &update0, &update1, &normalized_address,
        &clipped_address, &raw, &concat2_address, &concat3_address,
        &clip_min, &clip_max, &batch_param, &quant_channels, &spatial_count};
    driver_.launch(
        y0_tail_,
        {divide_up(static_cast<std::size_t>(spatial_count), 128),
         static_cast<unsigned int>(quant_channels),
         static_cast<unsigned int>(batch_count)},
        {128, 1, 1}, 0, parameters);
    return (cursor + 15) - index;
}

bool AotGraph::try_execute_y1_tail(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 14 >= nodes.size())
        return false;
    static constexpr std::array<Op, 15> operations{
        Op::SUB, Op::MUL, Op::SLICE, Op::SLICE, Op::ADD, Op::ROUND, Op::SLICE,
        Op::SLICE, Op::ADD, Op::MUL, Op::ADD, Op::MUL, Op::CONCAT, Op::ADD, Op::MUL};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).op != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).inputs;
    };
    auto output = [&](std::size_t offset) -> TensorId {
        const auto names = nodes.at(index + offset).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    const auto first_inputs = inputs(0);
    if (first_inputs.size() != 2 || (first_inputs[0] == 0) ||
        (first_inputs[1] == 0)) {
        return false;
    }
    const TensorId normalized_name = first_inputs[0];
    const TensorId prior_name = first_inputs[1];
    const TensorId difference = output(0);
    const TensorId scaled = output(1);
    const TensorId scaled0 = output(2);
    const TensorId scaled1 = output(3);
    const TensorId rounded_input = output(4);
    const TensorId y_raw = output(5);
    const TensorId prior0 = output(6);
    const TensorId prior1 = output(7);
    const TensorId updated_input0 = output(8);
    const TensorId updated0 = output(9);
    const TensorId updated_input1 = output(10);
    const TensorId updated1 = output(11);
    const TensorId concat4 = output(12);
    const TensorId added = output(13);
    const TensorId result_name = output(14);
    if ((difference == 0) || (scaled == 0) || (scaled0 == 0) ||
        (scaled1 == 0) || (rounded_input == 0) || (y_raw == 0) ||
        (prior0 == 0) || (prior1 == 0) || (updated_input0 == 0) ||
        (updated0 == 0) || (updated_input1 == 0) || (updated1 == 0) ||
        (concat4 == 0) || (added == 0) || (result_name == 0) ||
        inputs(1).size() != 2 || inputs(1)[0] != difference ||
        (inputs(1)[1] == 0) || inputs(2).size() < 1 ||
        inputs(2)[0] != scaled || inputs(3).size() < 1 ||
        inputs(3)[0] != scaled || inputs(4) !=
            std::vector<TensorId>{scaled0, scaled1} ||
        inputs(5) != std::vector<TensorId>{rounded_input} ||
        inputs(6).size() < 1 || inputs(6)[0] != prior_name ||
        inputs(7).size() < 1 || inputs(7)[0] != prior_name ||
        inputs(8) != std::vector<TensorId>{y_raw, prior0} ||
        inputs(9).size() != 2 || inputs(9)[0] != updated_input0 ||
        (inputs(9)[1] == 0) || inputs(10) !=
            std::vector<TensorId>{y_raw, prior1} ||
        inputs(11).size() != 2 || inputs(11)[0] != updated_input1 ||
        (inputs(11)[1] == 0) || inputs(12) !=
            std::vector<TensorId>{updated0, updated1} ||
        inputs(13).size() != 2 || (inputs(13)[0] == 0) ||
        inputs(13)[1] != concat4 ||
        inputs(14).size() != 2 || inputs(14)[0] != added ||
        (inputs(14)[1] == 0) ||
        nodes.at(index + 12).axis_or_zero() != 1) {
        return false;
    }

    const TensorId concat2_name = inputs(13)[0];
    const TensorId clipped_name = inputs(14)[1];
    if (tensor_consumer_count(difference) != 1 || tensor_consumer_count(scaled) != 2 ||
        tensor_consumer_count(scaled0) != 1 || tensor_consumer_count(scaled1) != 1 ||
        tensor_consumer_count(rounded_input) != 1 || tensor_consumer_count(y_raw) != 2 ||
        tensor_consumer_count(prior_name) != 3 || tensor_consumer_count(prior0) != 1 ||
        tensor_consumer_count(prior1) != 1 || tensor_consumer_count(updated_input0) != 1 ||
        tensor_consumer_count(updated0) != 1 || tensor_consumer_count(updated_input1) != 1 ||
        tensor_consumer_count(updated1) != 1 || tensor_consumer_count(concat4) != 1 ||
        tensor_consumer_count(added) != 1 || tensor_consumer_count(result_name) != 1 ||
        tensor_consumer_count(concat2_name) != 2 || tensor_consumer_count(clipped_name) != 2) {
        return false;
    }

    const Value& normalized_value = value(normalized_name);
    const Value& prior_value = value(prior_name);
    const Value& quant_scale = value(inputs(1)[1]);
    const Value& update_scale0 = value(inputs(9)[1]);
    const Value& update_scale1 = value(inputs(11)[1]);
    const Value& concat2_value = value(concat2_name);
    const Value& clipped_value = value(clipped_name);
    const Value& raw_value = value(y_raw);
    const Value& result = value(result_name);
    if (normalized_value.dtype != DataType::FP16 || prior_value.dtype != DataType::FP16 ||
        quant_scale.dtype != DataType::FP16 || update_scale0.dtype != DataType::FP16 ||
        update_scale1.dtype != DataType::FP16 || concat2_value.dtype != DataType::FP16 ||
        clipped_value.dtype != DataType::FP16 || raw_value.dtype != DataType::FP16 ||
        result.dtype != DataType::FP16 || normalized_value.shape.size() != 4 ||
        prior_value.shape.size() != 4 || raw_value.shape.size() != 4 ||
        result.shape.size() != 4) {
        return false;
    }
    int quant_channels = static_cast<int>(raw_value.shape[1]);
    if (quant_channels < 16)
        return false;
    const std::vector<int64_t> raw_shape{
        raw_value.shape[0], raw_value.shape[1], raw_value.shape[2],
        raw_value.shape[3]};
    const std::vector<int64_t> normalized_shape{
        raw_shape[0], raw_shape[1] * 2, raw_shape[2], raw_shape[3]};
    if (quant_channels <= 0 || normalized_value.shape != normalized_shape ||
        prior_value.shape != normalized_shape ||
        quant_scale.shape != normalized_shape ||
        update_scale0.shape != raw_shape || update_scale1.shape != raw_shape ||
        concat2_value.shape != normalized_shape ||
        clipped_value.shape != normalized_shape || result.shape != normalized_shape ||
        value(difference).shape != normalized_shape ||
        value(scaled).shape != normalized_shape ||
        value(scaled0).shape != raw_shape || value(scaled1).shape != raw_shape ||
        value(rounded_input).shape != raw_shape || value(prior0).shape != raw_shape ||
        value(prior1).shape != raw_shape || value(updated_input0).shape != raw_shape ||
        value(updated_input1).shape != raw_shape || value(updated0).shape != raw_shape ||
        value(updated1).shape != raw_shape || value(concat4).shape != normalized_shape ||
        value(added).shape != normalized_shape ||
        !is_channel_slice(nodes.at(index + 2), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 3), quant_channels,
                          quant_channels * 2) ||
        !is_channel_slice(nodes.at(index + 6), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 7), quant_channels,
                          quant_channels * 2) ||
        nodes.at(index + 13).axis_or_zero() != 0) {
        return false;
    }

    if (use_nhwc_ &&
        (!is_packed_nhwc(normalized_value) || !is_packed_nhwc(concat2_value) ||
         !is_packed_nhwc(clipped_value))) {
        return false;
    }
    if (ranges_overlap(raw_value, normalized_value) ||
        ranges_overlap(raw_value, prior_value) ||
        ranges_overlap(raw_value, concat2_value) ||
        ranges_overlap(raw_value, clipped_value) ||
        (ranges_overlap(result, concat2_value) &&
         (result.address != concat2_value.address ||
          (use_nhwc_ && value_stride(result) != quant_channels * 2))) ||
        (ranges_overlap(result, normalized_value) &&
         (result.address != normalized_value.address ||
          (use_nhwc_ && value_stride(result) != quant_channels * 2))) ||
        (ranges_overlap(result, prior_value) &&
         (result.address != prior_value.address ||
          (use_nhwc_ && value_stride(result) != value_stride(prior_value)))) ||
        (ranges_overlap(result, clipped_value) &&
         (result.address != clipped_value.address ||
          (use_nhwc_ && value_stride(result) != quant_channels * 2))) ||
        ranges_overlap(result, raw_value)) {
        return false;
    }

    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress normalized = normalized_value.address;
    DeviceAddress prior = prior_value.address;
    DeviceAddress quant = quant_scale.address;
    DeviceAddress update0 = update_scale0.address;
    DeviceAddress update1 = update_scale1.address;
    DeviceAddress concat2 = concat2_value.address;
    DeviceAddress clipped = clipped_value.address;
    DeviceAddress raw = raw_value.address;
    DeviceAddress result_address = result.address;
    int batch_count = static_cast<int>(raw_shape[0]);
    int spatial_count = static_cast<int>(raw_shape[2] * raw_shape[3]);
    if (use_nhwc_) {
        int prior_stride = value_stride(prior_value);
        int output_stride = value_stride(result);
        void* parameters[] = {
            &normalized, &prior, &quant, &update0, &update1, &concat2,
            &clipped, &raw, &result_address, &batch_count, &quant_channels,
            &spatial_count, &prior_stride, &output_stride};
        driver_.launch(
            y1_tail_nhwc_,
            {divide_up(static_cast<std::size_t>(quant_channels), 16),
             divide_up(static_cast<std::size_t>(spatial_count), 16),
             static_cast<unsigned int>(batch_count)},
            {16, 16, 1}, 0, parameters);
        return true;
    }
    void* parameters[] = {
        &normalized, &prior, &quant, &update0, &update1, &concat2,
        &clipped, &raw, &result_address, &batch_count, &quant_channels,
        &spatial_count};
    driver_.launch(
        y1_tail_,
        {divide_up(static_cast<std::size_t>(spatial_count), 128),
         static_cast<unsigned int>(quant_channels),
         static_cast<unsigned int>(batch_count)},
        {128, 1, 1}, 0, parameters);
    return true;
}

bool AotGraph::try_execute_dec_y0_tail(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 10 >= nodes.size())
        return false;
    static constexpr std::array<Op, 11> operations{
        Op::SLICE, Op::SLICE, Op::CLIP, Op::SLICE, Op::SLICE,
        Op::ADD, Op::MUL, Op::ADD, Op::MUL, Op::CONCAT, Op::CONCAT};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).op != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).inputs;
    };
    auto output = [&](std::size_t offset) -> TensorId {
        const auto names = nodes.at(index + offset).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    const auto slice0_inputs = inputs(0);
    const auto slice1_inputs = inputs(1);
    if (slice0_inputs.empty() || slice1_inputs.empty() ||
        slice0_inputs[0] == 0 || slice0_inputs[0] != slice1_inputs[0]) {
        return false;
    }
    const TensorId prior_name = slice0_inputs[0];
    const TensorId slice0 = output(0);
    const TensorId slice1 = output(1);
    const TensorId clipped = output(2);
    const TensorId prior0 = output(3);
    const TensorId prior1 = output(4);
    const TensorId updated_input0 = output(5);
    const TensorId updated0 = output(6);
    const TensorId updated_input1 = output(7);
    const TensorId updated1 = output(8);
    const TensorId concat2 = output(9);
    const TensorId concat3 = output(10);
    if (slice0 == 0 || slice1 == 0 || clipped == 0 || prior0 == 0 ||
        prior1 == 0 || updated_input0 == 0 || updated0 == 0 ||
        updated_input1 == 0 || updated1 == 0 || concat2 == 0 || concat3 == 0) {
        return false;
    }

    const auto clip_inputs = inputs(2);
    if ((clip_inputs.size() != 2 && clip_inputs.size() != 3) ||
        clip_inputs[0] != slice0 || clip_inputs[1] == 0) {
        return false;
    }
    if (value(clip_inputs[1]).dtype != DataType::FP16 ||
        value(clip_inputs[1]).shape != std::vector<int64_t>{1}) {
        return false;
    }
    float clip_max = 65504.0F;
    if (clip_inputs.size() == 3 && clip_inputs[2] != 0) {
        if (value(clip_inputs[2]).dtype != DataType::FP16 ||
            value(clip_inputs[2]).shape != std::vector<int64_t>{1}) {
            return false;
        }
        clip_max = scalar_fp16(clip_inputs[2]);
    }

    if (inputs(3).empty() || inputs(3)[0] != slice1 ||
        inputs(4).empty() || inputs(4)[0] != slice1 ||
        inputs(5).size() != 2 || inputs(7).size() != 2) {
        return false;
    }
    TensorId y_raw_name = 0;
    if (inputs(5)[1] == prior0)
        y_raw_name = inputs(5)[0];
    else if (inputs(5)[0] == prior0)
        y_raw_name = inputs(5)[1];
    else
        return false;
    if (y_raw_name == 0 ||
        !((inputs(7)[0] == y_raw_name && inputs(7)[1] == prior1) ||
          (inputs(7)[1] == y_raw_name && inputs(7)[0] == prior1))) {
        return false;
    }
    if (inputs(6).size() != 2 || inputs(6)[0] != updated_input0 ||
        inputs(6)[1] == 0 || inputs(8).size() != 2 ||
        inputs(8)[0] != updated_input1 || inputs(8)[1] == 0 ||
        inputs(9) != std::vector<TensorId>{updated0, updated1} ||
        inputs(10) != std::vector<TensorId>{concat2, prior_name} ||
        nodes.at(index + 9).axis_or_zero() != 1 ||
        nodes.at(index + 10).axis_or_zero() != 1) {
        return false;
    }

    const std::array<std::pair<const TensorId*, std::size_t>, 10>
        expected_consumers{{
            {&prior_name, 3}, {&slice0, 1}, {&slice1, 2},
            {&clipped, 1}, {&prior0, 1}, {&prior1, 1},
            {&updated_input0, 1}, {&updated0, 1},
            {&updated_input1, 1}, {&updated1, 1}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (tensor_consumer_count(*name) != expected)
            return false;
    }
    if (tensor_consumer_count(concat2) != 2)
        return false;

    const Value& prior_value = value(prior_name);
    const Value& raw_value = value(y_raw_name);
    const Value& update_scale0 = value(inputs(6)[1]);
    const Value& update_scale1 = value(inputs(8)[1]);
    const Value& clipped_value = value(clipped);
    const Value& concat2_value = value(concat2);
    const Value& concat3_value = value(concat3);
    if (prior_value.dtype != DataType::FP16 || raw_value.dtype != DataType::FP16 ||
        update_scale0.dtype != DataType::FP16 || update_scale1.dtype != DataType::FP16 ||
        clipped_value.dtype != DataType::FP16 || concat2_value.dtype != DataType::FP16 ||
        concat3_value.dtype != DataType::FP16 || prior_value.shape.size() != 4 ||
        raw_value.shape.size() != 4 || concat3_value.shape.size() != 4) {
        return false;
    }

    int quant_channels = static_cast<int>(raw_value.shape[1]);
    if (quant_channels < 16)
        return false;
    const std::vector<int64_t> raw_shape{
        raw_value.shape[0], raw_value.shape[1], raw_value.shape[2],
        raw_value.shape[3]};
    const std::vector<int64_t> norm_shape{
        raw_shape[0], raw_shape[1] * 2, raw_shape[2], raw_shape[3]};
    const std::vector<int64_t> prior_shape{
        raw_shape[0], raw_shape[1] * 4, raw_shape[2], raw_shape[3]};
    const std::vector<int64_t> concat3_shape{
        raw_shape[0], raw_shape[1] * 6, raw_shape[2], raw_shape[3]};
    if (prior_value.shape != prior_shape || update_scale0.shape != raw_shape ||
        update_scale1.shape != raw_shape || clipped_value.shape != norm_shape ||
        concat2_value.shape != norm_shape || concat3_value.shape != concat3_shape ||
        !is_channel_slice(nodes.at(index), 0, quant_channels * 2) ||
        !is_channel_slice(nodes.at(index + 1), quant_channels * 2,
                          quant_channels * 4) ||
        !is_channel_slice(nodes.at(index + 3), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 4), quant_channels,
                          quant_channels * 2)) {
        return false;
    }

    if (use_nhwc_ &&
        (!is_packed_nhwc(clipped_value) || !is_packed_nhwc(concat2_value))) {
        return false;
    }
    if (ranges_overlap(concat3_value, prior_value) ||
        ranges_overlap(concat3_value, raw_value) ||
        ranges_overlap(concat3_value, clipped_value) ||
        ranges_overlap(concat3_value, concat2_value) ||
        ranges_overlap(concat2_value, prior_value) ||
        ranges_overlap(concat2_value, raw_value) ||
        ranges_overlap(concat2_value, clipped_value) ||
        ranges_overlap(clipped_value, prior_value) ||
        ranges_overlap(clipped_value, raw_value)) {
        return false;
    }

    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress prior = prior_value.address;
    DeviceAddress raw = raw_value.address;
    DeviceAddress update0 = update_scale0.address;
    DeviceAddress update1 = update_scale1.address;
    DeviceAddress clipped_address = clipped_value.address;
    DeviceAddress concat2_address = concat2_value.address;
    DeviceAddress concat3_address = concat3_value.address;
    float clip_min = scalar_fp16(clip_inputs[1]);
    int batch_count = static_cast<int>(raw_shape[0]);
    int spatial_count = static_cast<int>(raw_shape[2] * raw_shape[3]);
    if (use_nhwc_) {
        int prior_stride = value_stride(prior_value);
        int concat3_stride = value_stride(concat3_value);
        void* parameters[] = {
            &prior, &raw, &update0, &update1, &clipped_address,
            &concat2_address, &concat3_address, &clip_min, &clip_max,
            &batch_count, &quant_channels, &spatial_count,
            &prior_stride, &concat3_stride};
        driver_.launch(
            dec_y0_tail_nhwc_,
            {divide_up(static_cast<std::size_t>(quant_channels), 16),
             divide_up(static_cast<std::size_t>(spatial_count), 16),
             static_cast<unsigned int>(batch_count)},
            {16, 16, 1}, 0, parameters);
        return true;
    }
    void* parameters[] = {
        &prior, &raw, &update0, &update1, &clipped_address,
        &concat2_address, &concat3_address, &clip_min, &clip_max,
        &batch_count, &quant_channels, &spatial_count};
    driver_.launch(
        dec_y0_tail_,
        {divide_up(static_cast<std::size_t>(spatial_count), 128),
         static_cast<unsigned int>(quant_channels),
         static_cast<unsigned int>(batch_count)},
        {128, 1, 1}, 0, parameters);
    return true;
}

bool AotGraph::try_execute_dec_y1_tail(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 8 >= nodes.size())
        return false;
    static constexpr std::array<Op, 9> operations{
        Op::SLICE, Op::SLICE, Op::ADD, Op::MUL, Op::ADD, Op::MUL,
        Op::CONCAT, Op::ADD, Op::MUL};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).op != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).inputs;
    };
    auto output = [&](std::size_t offset) -> TensorId {
        const auto names = nodes.at(index + offset).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    if (inputs(0).empty() || inputs(1).empty() ||
        inputs(0)[0] == 0 || inputs(0)[0] != inputs(1)[0]) {
        return false;
    }
    const TensorId prior_name = inputs(0)[0];
    const TensorId prior0 = output(0);
    const TensorId prior1 = output(1);
    const TensorId updated_input0 = output(2);
    const TensorId updated0 = output(3);
    const TensorId updated_input1 = output(4);
    const TensorId updated1 = output(5);
    const TensorId concat4 = output(6);
    const TensorId added = output(7);
    const TensorId result_name = output(8);
    if (prior0 == 0 || prior1 == 0 || updated_input0 == 0 || updated0 == 0 ||
        updated_input1 == 0 || updated1 == 0 || concat4 == 0 || added == 0 ||
        result_name == 0 || inputs(2).size() != 2 || inputs(4).size() != 2) {
        return false;
    }

    TensorId y_raw_name = 0;
    if (inputs(2)[1] == prior0)
        y_raw_name = inputs(2)[0];
    else if (inputs(2)[0] == prior0)
        y_raw_name = inputs(2)[1];
    else
        return false;
    if (y_raw_name == 0 ||
        !((inputs(4)[0] == y_raw_name && inputs(4)[1] == prior1) ||
          (inputs(4)[1] == y_raw_name && inputs(4)[0] == prior1))) {
        return false;
    }
    if (inputs(3).size() != 2 || inputs(3)[0] != updated_input0 ||
        inputs(3)[1] == 0 || inputs(5).size() != 2 ||
        inputs(5)[0] != updated_input1 || inputs(5)[1] == 0 ||
        inputs(6) != std::vector<TensorId>{updated0, updated1} ||
        nodes.at(index + 6).axis_or_zero() != 1 ||
        inputs(7).size() != 2 || inputs(8).size() != 2) {
        return false;
    }

    TensorId concat2_name = 0;
    if (inputs(7)[1] == concat4)
        concat2_name = inputs(7)[0];
    else if (inputs(7)[0] == concat4)
        concat2_name = inputs(7)[1];
    else
        return false;

    TensorId clipped_name = 0;
    if (inputs(8)[0] == added)
        clipped_name = inputs(8)[1];
    else if (inputs(8)[1] == added)
        clipped_name = inputs(8)[0];
    else
        return false;
    if (concat2_name == 0 || clipped_name == 0)
        return false;

    if (tensor_consumer_count(prior_name) != 2 ||
        tensor_consumer_count(prior0) != 1 ||
        tensor_consumer_count(prior1) != 1 ||
        tensor_consumer_count(updated_input0) != 1 ||
        tensor_consumer_count(updated0) != 1 ||
        tensor_consumer_count(updated_input1) != 1 ||
        tensor_consumer_count(updated1) != 1 ||
        tensor_consumer_count(concat4) != 1 ||
        tensor_consumer_count(added) != 1) {
        return false;
    }

    const Value& prior_value = value(prior_name);
    const Value& raw_value = value(y_raw_name);
    const Value& update_scale0 = value(inputs(3)[1]);
    const Value& update_scale1 = value(inputs(5)[1]);
    const Value& concat2_value = value(concat2_name);
    const Value& clipped_value = value(clipped_name);
    const Value& result = value(result_name);
    if (prior_value.dtype != DataType::FP16 || raw_value.dtype != DataType::FP16 ||
        update_scale0.dtype != DataType::FP16 || update_scale1.dtype != DataType::FP16 ||
        concat2_value.dtype != DataType::FP16 || clipped_value.dtype != DataType::FP16 ||
        result.dtype != DataType::FP16 || prior_value.shape.size() != 4 ||
        raw_value.shape.size() != 4 || result.shape.size() != 4) {
        return false;
    }

    int quant_channels = static_cast<int>(raw_value.shape[1]);
    if (quant_channels < 16)
        return false;
    const std::vector<int64_t> raw_shape{
        raw_value.shape[0], raw_value.shape[1], raw_value.shape[2],
        raw_value.shape[3]};
    const std::vector<int64_t> norm_shape{
        raw_shape[0], raw_shape[1] * 2, raw_shape[2], raw_shape[3]};
    if (prior_value.shape != norm_shape || update_scale0.shape != raw_shape ||
        update_scale1.shape != raw_shape || concat2_value.shape != norm_shape ||
        clipped_value.shape != norm_shape || result.shape != norm_shape ||
        !is_channel_slice(nodes.at(index), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 1), quant_channels,
                          quant_channels * 2)) {
        return false;
    }

    if (use_nhwc_ &&
        (!is_packed_nhwc(concat2_value) || !is_packed_nhwc(clipped_value))) {
        return false;
    }
    if (ranges_overlap(result, raw_value) ||
        (ranges_overlap(result, prior_value) &&
         (result.address != prior_value.address ||
          (use_nhwc_ && value_stride(result) != value_stride(prior_value)))) ||
        (ranges_overlap(result, concat2_value) &&
         (result.address != concat2_value.address ||
          (use_nhwc_ && value_stride(result) != quant_channels * 2))) ||
        (ranges_overlap(result, clipped_value) &&
         (result.address != clipped_value.address ||
          (use_nhwc_ && value_stride(result) != quant_channels * 2)))) {
        return false;
    }

    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress prior = prior_value.address;
    DeviceAddress raw = raw_value.address;
    DeviceAddress update0 = update_scale0.address;
    DeviceAddress update1 = update_scale1.address;
    DeviceAddress concat2 = concat2_value.address;
    DeviceAddress clipped = clipped_value.address;
    DeviceAddress result_address = result.address;
    int batch_count = static_cast<int>(raw_shape[0]);
    int spatial_count = static_cast<int>(raw_shape[2] * raw_shape[3]);
    if (use_nhwc_) {
        int prior_stride = value_stride(prior_value);
        int output_stride = value_stride(result);
        void* parameters[] = {
            &prior, &raw, &update0, &update1, &concat2, &clipped,
            &result_address, &batch_count, &quant_channels, &spatial_count,
            &prior_stride, &output_stride};
        driver_.launch(
            dec_y1_tail_nhwc_,
            {divide_up(static_cast<std::size_t>(quant_channels), 16),
             divide_up(static_cast<std::size_t>(spatial_count), 16),
             static_cast<unsigned int>(batch_count)},
            {16, 16, 1}, 0, parameters);
        return true;
    }
    void* parameters[] = {
        &prior, &raw, &update0, &update1, &concat2, &clipped,
        &result_address, &batch_count, &quant_channels, &spatial_count};
    driver_.launch(
        dec_y1_tail_,
        {divide_up(static_cast<std::size_t>(spatial_count), 128),
         static_cast<unsigned int>(quant_channels),
         static_cast<unsigned int>(batch_count)},
        {128, 1, 1}, 0, parameters);
    return true;
}

bool AotGraph::try_execute_feature_update(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 11 >= nodes.size())
        return false;
    static constexpr std::array<Op, 12> operations{
        Op::SLICE, Op::SLICE, Op::SLICE, Op::SIGMOID, Op::SIGMOID, Op::MUL,
        Op::SUB, Op::MUL, Op::ADD, Op::MUL, Op::MUL, Op::CONCAT};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).op != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).inputs;
    };
    auto output = [&](std::size_t offset) -> TensorId {
        const auto names = nodes.at(index + offset).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    const auto slice0_inputs = inputs(0);
    const auto slice1_inputs = inputs(1);
    const auto slice2_inputs = inputs(2);
    if (slice0_inputs.empty() || slice1_inputs.empty() ||
        slice2_inputs.empty() || slice0_inputs[0] != slice1_inputs[0] ||
        slice0_inputs[0] != slice2_inputs[0]) {
        return false;
    }
    const TensorId slice0 = output(0);
    const TensorId slice1 = output(1);
    const TensorId slice2 = output(2);
    const TensorId sigmoid0 = output(3);
    const TensorId sigmoid1 = output(4);
    const TensorId weighted1 = output(5);
    const TensorId inverse_gate = output(6);
    const TensorId weighted0 = output(7);
    const TensorId blended = output(8);
    const TensorId gated = output(9);
    const TensorId scaled = output(10);
    const TensorId result_name = output(11);
    if ((slice0 == 0) || (slice1 == 0) || (slice2 == 0) ||
        (sigmoid0 == 0) || (sigmoid1 == 0) || (weighted1 == 0) ||
        (inverse_gate == 0) || (weighted0 == 0) || (blended == 0) ||
        (gated == 0) || (scaled == 0) || (result_name == 0) ||
        inputs(3) != std::vector<TensorId>{slice1} ||
        inputs(4) != std::vector<TensorId>{slice2} ||
        inputs(5).size() != 2 || inputs(5)[0] != sigmoid0 ||
        inputs(6).size() != 2 || inputs(6)[1] != sigmoid0 ||
        inputs(7) != std::vector<TensorId>{inverse_gate, slice0} ||
        inputs(8) != std::vector<TensorId>{weighted1, weighted0} ||
        inputs(9) != std::vector<TensorId>{sigmoid1, blended} ||
        inputs(10).size() != 2 || inputs(10)[0] != gated ||
        inputs(11) != std::vector<TensorId>{scaled, blended} ||
        nodes.at(index + 11).axis() != 1) {
        return false;
    }

    const std::array<std::pair<const TensorId*, std::size_t>, 11>
        expected_consumers{{
            {&slice0, 1}, {&slice1, 1}, {&slice2, 1}, {&sigmoid0, 2},
            {&sigmoid1, 1}, {&weighted1, 1}, {&inverse_gate, 1},
            {&weighted0, 1}, {&blended, 2}, {&gated, 1}, {&scaled, 1}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (tensor_consumer_count(*name) != expected)
            return false;
    }

    const Value& source = value(slice0_inputs[0]);
    const Value& first = value(slice0);
    const Value& history = value(inputs(5)[1]);
    const Value& scale = value(inputs(10)[1]);
    const Value& result = value(result_name);
    if (source.dtype != DataType::FP16 || first.dtype != DataType::FP16 ||
        history.dtype != DataType::FP16 || scale.dtype != DataType::FP16 ||
        result.dtype != DataType::FP16 ||
        source.shape.size() != 4 || first.shape.size() != 4 ||
        result.shape.size() != 4 || first.shape[0] != source.shape[0] ||
        source.shape[1] != first.shape[1] * 3 ||
        result.shape[0] != first.shape[0] ||
        result.shape[1] != first.shape[1] * 2 ||
        result.shape[2] != first.shape[2] ||
        result.shape[3] != first.shape[3] ||
        history.shape != first.shape ||
        scale.shape != std::vector<int64_t>{1, first.shape[1], 1, 1} ||
        value(slice1).shape != first.shape ||
        value(slice2).shape != first.shape ||
        scalar_fp16(inputs(6)[0]) != 1.0F) {
        return false;
    }
    for (std::size_t offset = 3; offset <= 10; ++offset) {
        if (value(output(offset)).dtype != DataType::FP16 ||
            value(output(offset)).shape != first.shape) {
            return false;
        }
    }

    int channels = static_cast<int>(first.shape[1]);
    if (!is_channel_slice(nodes.at(index), 0, channels) ||
        !is_channel_slice(nodes.at(index + 1), channels, channels * 2) ||
        !is_channel_slice(nodes.at(index + 2), channels * 2, channels * 3)) {
        return false;
    }

    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress source_address = source.address;
    DeviceAddress history_address = history.address;
    DeviceAddress scale_address = scale.address;
    DeviceAddress result_address = result.address;
    int batch_count = static_cast<int>(first.shape[0]);
    int spatial_count = static_cast<int>(first.shape[2] * first.shape[3]);
    if (use_nhwc_) {
        int input_stride = value_stride(source);
        int history_stride = value_stride(history);
        int output_stride = value_stride(result);
        const bool vec8 = ((channels & 7) == 0) && ((input_stride & 7) == 0) &&
            ((history_stride & 7) == 0) && ((output_stride & 7) == 0) &&
            ((source_address & 15U) == 0) && ((history_address & 15U) == 0) &&
            ((scale_address & 15U) == 0) && ((result_address & 15U) == 0);
        const std::size_t work = static_cast<std::size_t>(batch_count) *
            static_cast<std::size_t>(spatial_count) *
            static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
        void* parameters[] = {
            &source_address, &history_address, &scale_address, &result_address,
            &batch_count, &channels, &spatial_count,
            &input_stride, &history_stride, &output_stride};
        launch_linear(feature_update_nhwc_, work, parameters);
        return true;
    }
    void* parameters[] = {
        &source_address, &history_address, &scale_address, &result_address,
        &batch_count, &channels, &spatial_count};
    const std::size_t pairs = static_cast<std::size_t>(batch_count) *
        channels * ((spatial_count + 1) / 2);
    launch_linear(feature_update_, pairs, parameters);
    return true;
}

bool AotGraph::try_execute_feature_update_outputs(
    const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 10 >= nodes.size())
        return false;
    static constexpr std::array<Op, 11> operations{
        Op::SLICE, Op::SLICE, Op::SLICE, Op::SIGMOID, Op::SIGMOID, Op::MUL,
        Op::SUB, Op::MUL, Op::ADD, Op::MUL, Op::MUL};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).op != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).inputs;
    };
    auto output = [&](std::size_t offset) -> TensorId {
        const auto names = nodes.at(index + offset).outputs;
        return names.size() == 1 ? names[0] : TensorId{};
    };

    const auto slice0_inputs = inputs(0);
    const auto slice1_inputs = inputs(1);
    const auto slice2_inputs = inputs(2);
    if (slice0_inputs.empty() || slice1_inputs.empty() ||
        slice2_inputs.empty() || slice0_inputs[0] != slice1_inputs[0] ||
        slice0_inputs[0] != slice2_inputs[0]) {
        return false;
    }
    const TensorId slice0 = output(0);
    const TensorId slice1 = output(1);
    const TensorId slice2 = output(2);
    const TensorId sigmoid0 = output(3);
    const TensorId sigmoid1 = output(4);
    const TensorId weighted1 = output(5);
    const TensorId inverse_gate = output(6);
    const TensorId weighted0 = output(7);
    const TensorId blended = output(8);
    const TensorId gated = output(9);
    const TensorId scaled = output(10);
    if ((slice0 == 0) || (slice1 == 0) || (slice2 == 0) ||
        (sigmoid0 == 0) || (sigmoid1 == 0) || (weighted1 == 0) ||
        (inverse_gate == 0) || (weighted0 == 0) || (blended == 0) ||
        (gated == 0) || (scaled == 0) ||
        inputs(3) != std::vector<TensorId>{slice1} ||
        inputs(4) != std::vector<TensorId>{slice2} ||
        inputs(5).size() != 2 || inputs(5)[0] != sigmoid0 ||
        inputs(6).size() != 2 || inputs(6)[1] != sigmoid0 ||
        inputs(7) != std::vector<TensorId>{inverse_gate, slice0} ||
        inputs(8) != std::vector<TensorId>{weighted1, weighted0} ||
        inputs(9) != std::vector<TensorId>{sigmoid1, blended} ||
        inputs(10).size() != 2 || inputs(10)[0] != gated) {
        return false;
    }

    std::size_t concat_index = nodes.size();
    TensorId result_name = 0;
    for (std::size_t candidate_index = index + operations.size();
         candidate_index < nodes.size(); ++candidate_index) {
        const Node& candidate = nodes.at(candidate_index);
        if (candidate.op != Op::CONCAT)
            continue;
        const auto candidate_inputs = candidate.inputs;
        const auto candidate_outputs = candidate.outputs;
        if (candidate_inputs != std::vector<TensorId>{scaled, blended} ||
            candidate_outputs.size() != 1 ||
            candidate.axis_or_zero() != 1) {
            continue;
        }
        if (concat_index != nodes.size())
            return false;
        concat_index = candidate_index;
        result_name = candidate_outputs[0];
    }
    if (concat_index == nodes.size() || (result_name == 0))
        return false;

    const std::array<std::pair<const TensorId*, std::size_t>, 11>
        expected_consumers{{
            {&slice0, 1}, {&slice1, 1}, {&slice2, 1}, {&sigmoid0, 2},
            {&sigmoid1, 1}, {&weighted1, 1}, {&inverse_gate, 1},
            {&weighted0, 1}, {&blended, 2}, {&gated, 1}, {&scaled, 2}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (tensor_consumer_count(*name) != expected)
            return false;
    }

    const Value& source = value(slice0_inputs[0]);
    const Value& first = value(slice0);
    const Value& history = value(inputs(5)[1]);
    const Value& scale = value(inputs(10)[1]);
    const Value& scaled_value = value(scaled);
    const Value& blended_value = value(blended);
    Value& result = values_.at(result_name);
    if (source.dtype != DataType::FP16 || first.dtype != DataType::FP16 ||
        history.dtype != DataType::FP16 || scale.dtype != DataType::FP16 ||
        scaled_value.dtype != DataType::FP16 || blended_value.dtype != DataType::FP16 ||
        result.dtype != DataType::FP16 || source.shape.size() != 4 ||
        first.shape.size() != 4 || result.shape.size() != 4 ||
        first.shape[0] != source.shape[0] ||
        source.shape[1] != first.shape[1] * 3 ||
        result.shape[0] != first.shape[0] ||
        result.shape[1] != first.shape[1] * 2 ||
        result.shape[2] != first.shape[2] ||
        result.shape[3] != first.shape[3] || history.shape != first.shape ||
        scale.shape != std::vector<int64_t>{1, first.shape[1], 1, 1} ||
        scaled_value.shape != first.shape || blended_value.shape != first.shape ||
        value(slice1).shape != first.shape || value(slice2).shape != first.shape ||
        scalar_fp16(inputs(6)[0]) != 1.0F) {
        return false;
    }
    for (std::size_t offset = 3; offset <= 10; ++offset) {
        if (value(output(offset)).dtype != DataType::FP16 ||
            value(output(offset)).shape != first.shape) {
            return false;
        }
    }

    int channels = static_cast<int>(first.shape[1]);
    if (!is_channel_slice(nodes.at(index), 0, channels) ||
        !is_channel_slice(nodes.at(index + 1), channels, channels * 2) ||
        !is_channel_slice(nodes.at(index + 2), channels * 2, channels * 3) ||
        ranges_overlap(scaled_value, blended_value) ||
        ranges_overlap(blended_value, source) ||
        (ranges_overlap(scaled_value, source) &&
         scaled_value.address != source.address) ||
        ranges_overlap(scaled_value, history) ||
        ranges_overlap(blended_value, history)) {
        return false;
    }

    const auto output_iterator = std::find(
        output_ids_.begin(), output_ids_.end(), result_name);
    const bool bound_state_output =
        output_iterator != output_ids_.end() &&
        is_state_output(static_cast<std::size_t>(
            std::distance(output_ids_.begin(), output_iterator)));
    if (!bound_state_output) {
        auto [buffer, inserted] =
            feature_update_output_buffers_.try_emplace(concat_index);
        if (inserted) {
            buffer->second = driver_.allocate(
                element_count(result.shape) * dtype_bytes(result.dtype));
        }
        result.address = buffer->second.address();
    }
    if (std::find(elided_schedule_nodes_.begin(),
                  elided_schedule_nodes_.end(), concat_index) ==
        elided_schedule_nodes_.end()) {
        elided_schedule_nodes_.push_back(concat_index);
    }

    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress source_address = source.address;
    DeviceAddress history_address = history.address;
    DeviceAddress scale_address = scale.address;
    DeviceAddress scaled_address = scaled_value.address;
    DeviceAddress blended_address = blended_value.address;
    DeviceAddress result_address = result.address;
    int batch_count = static_cast<int>(first.shape[0]);
    int spatial_count = static_cast<int>(first.shape[2] * first.shape[3]);
    if (use_nhwc_) {
        int input_stride = value_stride(source);
        int history_stride = value_stride(history);
        int scaled_stride = value_stride(scaled_value);
        int blended_stride = value_stride(blended_value);
        int feature_stride = value_stride(result);
        const bool vec8 = ((channels & 7) == 0) && ((input_stride & 7) == 0) &&
            ((history_stride & 7) == 0) && ((scaled_stride & 7) == 0) &&
            ((blended_stride & 7) == 0) && ((feature_stride & 7) == 0) &&
            ((source_address & 15U) == 0) && ((history_address & 15U) == 0) &&
            ((scale_address & 15U) == 0) && ((scaled_address & 15U) == 0) &&
            ((blended_address & 15U) == 0) && ((result_address & 15U) == 0);
        const std::size_t work = static_cast<std::size_t>(batch_count) *
            static_cast<std::size_t>(spatial_count) *
            static_cast<std::size_t>(vec8 ? (channels / 8) : channels);
        void* parameters[] = {
            &source_address, &history_address, &scale_address, &scaled_address,
            &blended_address, &result_address, &batch_count, &channels,
            &spatial_count, &input_stride, &history_stride, &scaled_stride,
            &blended_stride, &feature_stride};
        launch_linear(feature_update_outputs_nhwc_, work, parameters);
        return true;
    }
    void* parameters[] = {
        &source_address, &history_address, &scale_address, &scaled_address,
        &blended_address, &result_address, &batch_count, &channels,
        &spatial_count};
    const std::size_t pairs = static_cast<std::size_t>(batch_count) *
        channels * ((spatial_count + 1) / 2);
    launch_linear(feature_update_outputs_, pairs, parameters);
    return true;
}

bool AotGraph::try_execute_pointwise_epilogue(const std::vector<Node>& nodes, std::size_t index)
{
    if (driver_.device_info().compute_major < 8 || index + 1 >= nodes.size())
        return false;
    const Node& convolution = nodes.at(index);
    const Node& epilogue_node = nodes.at(index + 1);
    const Op epilogue_op =
        epilogue_node.op;
    if (convolution.op != Op::CONV ||
        (epilogue_op != Op::LEAKY_RELU && epilogue_op != Op::ADD)) {
        return false;
    }

    const auto convolution_inputs =
        convolution.inputs;
    const auto convolution_outputs =
        convolution.outputs;
    const auto epilogue_inputs =
        epilogue_node.inputs;
    const auto epilogue_outputs =
        epilogue_node.outputs;
    if (convolution_inputs.size() < 2 || convolution_outputs.size() != 1 ||
        epilogue_outputs.size() != 1 ||
        std::find(output_ids_.begin(), output_ids_.end(),
                  convolution_outputs[0]) != output_ids_.end()) {
        return false;
    }

    const TensorId convolution_output_name = convolution_outputs[0];
    if (tensor_consumer_count(convolution_output_name) != 1)
        return false;

    DeviceAddress epilogue_input_address = 0;
    int epilogue = 1;
    float alpha = epilogue_node.alpha_or_default();
    const Value* residual = nullptr;
    if (epilogue_op == Op::LEAKY_RELU) {
        if (epilogue_inputs != std::vector<TensorId>{convolution_output_name})
            return false;
    } else {
        if (epilogue_inputs.size() != 2)
            return false;
        const auto found = std::find(epilogue_inputs.begin(),
                                     epilogue_inputs.end(),
                                     convolution_output_name);
        if (found == epilogue_inputs.end())
            return false;
        TensorId residual_name =
            epilogue_inputs[found == epilogue_inputs.begin() ? 1 : 0];
        residual = &value(residual_name);
        epilogue_input_address = residual->address;
        epilogue = 2;
    }

    const Value& input = value(convolution_inputs[0]);
    const Value& weight = value(convolution_inputs[1]);
    const Value& convolution_output = value(convolution_output_name);
    const Value& output = value(epilogue_outputs[0]);
    const auto& attributes = convolution.conv();
    const auto strides = attributes.strides;
    const auto pads = attributes.pads;
    if (input.dtype != DataType::FP16 || weight.dtype != DataType::FP16 ||
        output.dtype != DataType::FP16 || input.shape.size() != 4 ||
        weight.shape.size() != 4 || output.shape.size() != 4 ||
        attributes.groups != 1 || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || strides != std::array<int, 2>{1, 1} ||
        pads != std::array<int, 4>{0, 0, 0, 0} ||
        input.shape[0] != output.shape[0] ||
        input.shape[1] != weight.shape[1] ||
        input.shape[2] != output.shape[2] ||
        input.shape[3] != output.shape[3] ||
        output.shape[1] != weight.shape[0] ||
        convolution_output.shape != output.shape ||
        (residual && residual->shape != output.shape) ||
        ranges_overlap(output, input) ||
        (residual && ranges_overlap(output, *residual))) {
        return false;
    }

    const bool has_bias = convolution_inputs.size() > 2 &&
                          !(convolution_inputs[2] == 0);
    const int batch_count = static_cast<int>(input.shape[0]);
    const int in_channels = static_cast<int>(input.shape[1]);
    const int out_channels = static_cast<int>(output.shape[1]);
    const int spatial_count =
        static_cast<int>(output.shape[2] * output.shape[3]);
    const int in_stride = use_nhwc_ ? value_stride(input) : 0;
    const int res_stride = (use_nhwc_ && residual) ? value_stride(*residual) : 0;
    const int out_stride = use_nhwc_ ? value_stride(output) : 0;
    const bool cutlass_supported = can_use_cutlass_pointwise(
        batch_count, has_bias, in_channels, spatial_count, out_channels,
        epilogue, alpha, epilogue_input_address != 0,
        in_stride, res_stride, out_stride);
    const bool mma_supported = use_nhwc_
        ? ((in_channels % 16 == 0) && (out_channels % 16 == 0) &&
           (in_stride % 8 == 0))
        : ((in_channels % 16 == 0) &&
           ((out_channels % 128 == 0) ||
            (spatial_count < 920 && (out_channels % 16 == 0))));
    if (!cutlass_supported && !mma_supported)
        return false;

    execute(convolution, &output, epilogue_input_address, epilogue, alpha,
            res_stride);
    return true;
}

bool AotGraph::try_execute_reglu(const std::vector<Node>& nodes, std::size_t index)
{
    if (index + 3 >= nodes.size())
        return false;
    const Node& first_slice = nodes.at(index);
    const Node& second_slice = nodes.at(index + 1);
    const Node& clip = nodes.at(index + 2);
    const Node& multiply = nodes.at(index + 3);
    if (first_slice.op != Op::SLICE || second_slice.op != Op::SLICE ||
        clip.op != Op::CLIP || multiply.op != Op::MUL) {
        return false;
    }

    const auto first_inputs =
        first_slice.inputs;
    const auto second_inputs =
        second_slice.inputs;
    const auto first_outputs =
        first_slice.outputs;
    const auto second_outputs =
        second_slice.outputs;
    const auto clip_inputs = clip.inputs;
    const auto clip_outputs = clip.outputs;
    const auto multiply_inputs =
        multiply.inputs;
    const auto multiply_outputs =
        multiply.outputs;
    if (first_inputs.empty() || second_inputs.empty() ||
        first_inputs[0] != second_inputs[0] || first_outputs.size() != 1 ||
        second_outputs.size() != 1 || clip_inputs.empty() ||
        clip_outputs.size() != 1 || multiply_inputs.size() != 2 ||
        multiply_outputs.size() != 1 || clip_inputs[0] != first_outputs[0] ||
        multiply_inputs[0] != clip_outputs[0] ||
        multiply_inputs[1] != second_outputs[0]) {
        return false;
    }

    const Value& input = value(first_inputs[0]);
    Value& output = values_.at(multiply_outputs[0]);
    if (input.dtype != DataType::FP16 || output.dtype != DataType::FP16 ||
        input.shape.size() != 4 || output.shape.size() != 4 ||
        input.shape[0] != output.shape[0] ||
        input.shape[1] != output.shape[1] * 2 ||
        input.shape[2] != output.shape[2] ||
        input.shape[3] != output.shape[3] ||
        value(first_outputs[0]).shape != output.shape ||
        value(second_outputs[0]).shape != output.shape ||
        value(clip_outputs[0]).shape != output.shape) {
        return false;
    }

    int channels = static_cast<int>(output.shape[1]);
    if (!is_channel_slice(first_slice, 0, channels) ||
        !is_channel_slice(second_slice, channels, channels * 2)) {
        return false;
    }

    float minimum = -std::numeric_limits<float>::infinity();
    float maximum = std::numeric_limits<float>::infinity();
    if (clip_inputs.size() > 1 && !(clip_inputs[1] == 0))
        minimum = scalar_fp16(clip_inputs[1]);
    if (clip_inputs.size() > 2 && !(clip_inputs[2] == 0))
        maximum = scalar_fp16(clip_inputs[2]);
    DeviceAddress input_address = input.address;
    if (std::find(direct_concat_values_.begin(),
                  direct_concat_values_.end(), multiply_outputs[0]) ==
            direct_concat_values_.end() &&
        std::find(aliased_spatial_values_.begin(),
                  aliased_spatial_values_.end(), multiply_outputs[0]) ==
            aliased_spatial_values_.end()) {
        output.address = reglu_buffer_.address();
        output.nhwc_stride = 0;
    }
    if (batch_initializing_cutlass_)
        return true;
    DeviceAddress output_address = output.address;
    int batch_count = static_cast<int>(output.shape[0]);
    int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
    if (use_nhwc_) {
        int pixels = batch_count * spatial_count;
        int in_stride = value_stride(input);
        int out_stride = value_stride(output);
        void* parameters[] = {
            &input_address, &output_address, &pixels, &channels,
            &in_stride, &out_stride, &minimum, &maximum};
        driver_.launch(
            reglu_nhwc_,
            {divide_up(static_cast<std::size_t>(channels), 32),
             divide_up(static_cast<std::size_t>(pixels), 8), 1},
            {32, 8, 1}, 0, parameters);
        return true;
    }
    int work_count = spatial_count % 2 == 0 ? spatial_count / 2 : spatial_count;
    void* parameters[] = {
        &input_address, &output_address, &batch_count, &channels,
        &spatial_count, &minimum, &maximum};
    if (spatial_count % 8 == 0) {
        driver_.launch(
            reglu_vec8_,
            {divide_up(static_cast<std::size_t>(spatial_count / 8), 128),
             static_cast<unsigned int>(channels),
             static_cast<unsigned int>(batch_count)},
            {128, 1, 1}, 0, parameters);
    } else {
        driver_.launch(reglu_, {divide_up(work_count, 256),
                                static_cast<unsigned int>(channels),
                                static_cast<unsigned int>(batch_count)},
                       {256, 1, 1}, 0, parameters);
    }
    return true;
}

bool AotGraph::try_execute_pointwise_reglu(const std::vector<Node>& nodes, std::size_t index)
{
    if (driver_.device_info().compute_major < 8 || index + 4 >= nodes.size())
        return false;
    const Node& convolution = nodes.at(index);
    const Node& first_slice = nodes.at(index + 1);
    const Node& second_slice = nodes.at(index + 2);
    const Node& clip = nodes.at(index + 3);
    const Node& multiply = nodes.at(index + 4);
    if (convolution.op != Op::CONV || first_slice.op != Op::SLICE ||
        second_slice.op != Op::SLICE || clip.op != Op::CLIP ||
        multiply.op != Op::MUL) {
        return false;
    }

    const auto convolution_inputs =
        convolution.inputs;
    const auto convolution_outputs =
        convolution.outputs;
    const auto first_inputs =
        first_slice.inputs;
    const auto second_inputs =
        second_slice.inputs;
    const auto first_outputs =
        first_slice.outputs;
    const auto second_outputs =
        second_slice.outputs;
    const auto clip_inputs = clip.inputs;
    const auto clip_outputs = clip.outputs;
    const auto multiply_inputs = multiply.inputs;
    const auto multiply_outputs = multiply.outputs;
    if (convolution_inputs.size() < 2 || convolution_outputs.size() != 1 ||
        first_inputs.empty() || second_inputs.empty() ||
        first_inputs[0] != convolution_outputs[0] ||
        second_inputs[0] != convolution_outputs[0] ||
        first_outputs.size() != 1 || second_outputs.size() != 1 ||
        clip_inputs.empty() || clip_outputs.size() != 1 ||
        multiply_inputs.size() != 2 || multiply_outputs.size() != 1 ||
        clip_inputs[0] != first_outputs[0] ||
        multiply_inputs[0] != clip_outputs[0] ||
        multiply_inputs[1] != second_outputs[0]) {
        return false;
    }

    const Value& input = value(convolution_inputs[0]);
    const Value& weight = value(convolution_inputs[1]);
    const Value& convolution_output = value(convolution_outputs[0]);
    Value& output = values_.at(multiply_outputs[0]);
    const auto& attributes = convolution.conv();
    const auto strides = attributes.strides;
    const auto pads = attributes.pads;
    const int groups = attributes.groups;
    if (input.dtype != DataType::FP16 || weight.dtype != DataType::FP16 ||
        output.dtype != DataType::FP16 || input.shape.size() != 4 ||
        weight.shape.size() != 4 || output.shape.size() != 4 ||
        convolution_output.shape.size() != 4 || groups != 1 ||
        weight.shape[2] != 1 || weight.shape[3] != 1 ||
        strides != std::array<int, 2>{1, 1} ||
        pads != std::array<int, 4>{0, 0, 0, 0} ||
        input.shape[0] != output.shape[0] ||
        input.shape[2] != output.shape[2] ||
        input.shape[3] != output.shape[3] ||
        convolution_output.shape[1] != output.shape[1] * 2 ||
        convolution_output.shape[0] != output.shape[0] ||
        convolution_output.shape[2] != output.shape[2] ||
        convolution_output.shape[3] != output.shape[3] ||
        value(first_outputs[0]).shape != output.shape ||
        value(second_outputs[0]).shape != output.shape ||
        value(clip_outputs[0]).shape != output.shape) {
        return false;
    }

    int gate_channels = static_cast<int>(output.shape[1]);
    int in_channels = static_cast<int>(input.shape[1]);
    int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
    int batch_count = static_cast<int>(output.shape[0]);
    DeviceAddress bias_address = convolution_inputs.size() > 2 &&
        !(convolution_inputs[2] == 0)
        ? value(convolution_inputs[2]).address : 0;
    const int in_stride = use_nhwc_ ? value_stride(input) : 0;
    const bool redirect_reglu_out =
        std::find(direct_concat_values_.begin(),
                  direct_concat_values_.end(), multiply_outputs[0]) ==
            direct_concat_values_.end() &&
        std::find(aliased_spatial_values_.begin(),
                  aliased_spatial_values_.end(), multiply_outputs[0]) ==
            aliased_spatial_values_.end();
    const int out_stride = use_nhwc_
        ? (redirect_reglu_out ? gate_channels : value_stride(output)) : 0;
    const bool cutlass_reglu_eligible = use_nhwc_
        ? (batch_count == 1 && bias_address != 0 && (in_channels % 8) == 0 &&
           (gate_channels % 8) == 0 && (in_stride % 8) == 0 &&
           (out_stride % 8) == 0 &&
           tensor_consumer_count(convolution_outputs[0]) == 2 &&
           tensor_consumer_count(first_outputs[0]) == 1 &&
           tensor_consumer_count(second_outputs[0]) == 1 &&
           tensor_consumer_count(clip_outputs[0]) == 1)
        : (batch_count == 1 && bias_address != 0 && spatial_count >= 920 &&
           (spatial_count % 8) == 0 && (in_channels % 8) == 0 &&
           (gate_channels % 8) == 0 &&
           tensor_consumer_count(convolution_outputs[0]) == 2 &&
           tensor_consumer_count(first_outputs[0]) == 1 &&
           tensor_consumer_count(second_outputs[0]) == 1 &&
           tensor_consumer_count(clip_outputs[0]) == 1);
    const bool mma_reglu_eligible = !use_nhwc_ &&
        (in_channels % 16 == 0) &&
        ((spatial_count < 920 && (gate_channels % 16 == 0)) ||
         (gate_channels % 64 == 0));
    if ((!cutlass_reglu_eligible && !mma_reglu_eligible) ||
        !is_channel_slice(first_slice, 0, gate_channels) ||
        !is_channel_slice(second_slice, gate_channels, gate_channels * 2)) {
        return false;
    }

    float minimum = -std::numeric_limits<float>::infinity();
    float maximum = std::numeric_limits<float>::infinity();
    if (clip_inputs.size() > 1 && !(clip_inputs[1] == 0))
        minimum = scalar_fp16(clip_inputs[1]);
    if (clip_inputs.size() > 2 && !(clip_inputs[2] == 0))
        maximum = scalar_fp16(clip_inputs[2]);
    DeviceAddress input_address = input.address;
    DeviceAddress weight_address = weight.address;
    if (redirect_reglu_out) {
        output.address = reglu_buffer_.address();
        output.nhwc_stride = 0;
    }
    DeviceAddress output_address = output.address;
    if (cutlass_reglu_eligible &&
        launch_cutlass_pointwise_reglu(
            convolution, input_address, weight_address, bias_address,
            output_address, batch_count, in_channels, spatial_count,
            gate_channels, minimum, maximum, in_stride, out_stride)) {
        return true;
    }
    DeviceAddress convolution_output_address = convolution_output.address;
    const int conv_out_stride =
        use_nhwc_ ? value_stride(convolution_output) : 0;
    if (launch_cutlass_pointwise(
            convolution, input_address, weight_address, bias_address, 0,
            convolution_output_address, batch_count, in_channels,
            spatial_count, gate_channels * 2, 0, 0.0F,
            in_stride, 0, conv_out_stride)) {
        if (batch_initializing_cutlass_)
            return true;
        if (use_nhwc_) {
            int pixels = batch_count * spatial_count;
            int r_in_stride = conv_out_stride;
            int r_out_stride = value_stride(output);
            void* reglu_parameters[] = {
                &convolution_output_address, &output_address, &pixels,
                &gate_channels, &r_in_stride, &r_out_stride,
                &minimum, &maximum};
            driver_.launch(
                reglu_nhwc_,
                {divide_up(static_cast<std::size_t>(gate_channels), 32),
                 divide_up(static_cast<std::size_t>(pixels), 8), 1},
                {32, 8, 1}, 0, reglu_parameters);
            return true;
        }
        const int work_count = (spatial_count & 1) == 0
            ? spatial_count / 2 : spatial_count;
        void* reglu_parameters[] = {
            &convolution_output_address, &output_address, &batch_count,
            &gate_channels, &spatial_count, &minimum, &maximum};
        if (spatial_count % 8 == 0) {
            driver_.launch(
                reglu_vec8_,
                {divide_up(static_cast<std::size_t>(spatial_count / 8), 128),
                 static_cast<unsigned int>(gate_channels),
                 static_cast<unsigned int>(batch_count)},
                {128, 1, 1}, 0, reglu_parameters);
        } else {
            driver_.launch(
                reglu_,
                {divide_up(work_count, 256),
                 static_cast<unsigned int>(gate_channels),
                 static_cast<unsigned int>(batch_count)},
                {256, 1, 1}, 0, reglu_parameters);
        }
        return true;
    }

    if (batch_initializing_cutlass_)
        return true;
    void* parameters[] = {
        &input_address, &weight_address, &bias_address, &output_address,
        &batch_count, &in_channels, &spatial_count, &gate_channels,
        &minimum, &maximum};
    if (spatial_count < 920) {
        driver_.launch(
            pointwise_reglu_mma_tiny_,
            {divide_up(spatial_count, 16), divide_up(gate_channels, 16),
             static_cast<unsigned int>(batch_count)},
            {32, 4, 1}, 0, parameters);
        return true;
    }
    const unsigned int large_tile_blocks =
        divide_up(spatial_count, 64) * divide_up(gate_channels, 64) *
        batch_count;
    const bool use_small = large_tile_blocks <
        static_cast<unsigned int>(
            driver_.device_info().multiprocessor_count * 2);
    driver_.launch(
        use_small ? pointwise_reglu_mma_small_ : pointwise_reglu_mma_,
        {divide_up(spatial_count, 64),
         divide_up(gate_channels, use_small ? 32 : 64),
         static_cast<unsigned int>(batch_count)},
        {32, 8, 1}, 0, parameters);
    return true;
}

}  // namespace mlvc::driver_cubin_backend
