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

bool AotGraph::try_execute_y0_tail(const json& nodes, std::size_t index)
{
    if (index + 18 >= nodes.size())
        return false;
    static constexpr std::array<const char*, 19> operations{
        "Slice", "Slice", "Clip", "Reciprocal", "Mul", "Sub", "Mul",
        "Slice", "Slice", "Add", "Round", "Slice", "Slice", "Add",
        "Mul", "Add", "Mul", "Concat", "Concat"};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).at("op") != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).at("inputs")
            .get<std::vector<std::string>>();
    };
    auto output = [&](std::size_t offset) -> std::string {
        const auto names = nodes.at(index + offset).at("outputs")
            .get<std::vector<std::string>>();
        return names.size() == 1 ? names[0] : std::string{};
    };
    auto consumer_count = [&](const std::string& name) {
        std::size_t count = 0;
        for (const auto& candidate : nodes) {
            const auto candidate_inputs = candidate.at("inputs")
                .get<std::vector<std::string>>();
            count += static_cast<std::size_t>(std::count(
                candidate_inputs.begin(), candidate_inputs.end(), name));
        }
        return count;
    };

    const auto slice0_inputs = inputs(0);
    const auto slice1_inputs = inputs(1);
    if (slice0_inputs.size() < 1 || slice1_inputs.size() < 1 ||
        slice0_inputs[0].empty() || slice0_inputs[0] != slice1_inputs[0]) {
        return false;
    }
    const std::string prior_name = slice0_inputs[0];
    const std::string slice0 = output(0);
    const std::string slice1 = output(1);
    const std::string clipped = output(2);
    const std::string reciprocal = output(3);
    const std::string normalized = output(4);
    const std::string difference = output(5);
    const std::string scaled = output(6);
    const std::string scaled0 = output(7);
    const std::string scaled1 = output(8);
    const std::string rounded_input = output(9);
    const std::string y_raw = output(10);
    const std::string prior0 = output(11);
    const std::string prior1 = output(12);
    const std::string updated_input0 = output(13);
    const std::string updated0 = output(14);
    const std::string updated_input1 = output(15);
    const std::string updated1 = output(16);
    const std::string concat2 = output(17);
    const std::string concat3 = output(18);
    if (slice0.empty() || slice1.empty() || clipped.empty() ||
        reciprocal.empty() || normalized.empty() || difference.empty() ||
        scaled.empty() || scaled0.empty() || scaled1.empty() ||
        rounded_input.empty() || y_raw.empty() || prior0.empty() ||
        prior1.empty() || updated_input0.empty() || updated0.empty() ||
        updated_input1.empty() || updated1.empty() || concat2.empty() ||
        concat3.empty() ||
        inputs(2).size() != 2 || inputs(2)[0] != slice0 ||
        inputs(2)[1].empty() || inputs(3) != std::vector<std::string>{clipped} ||
        inputs(4).size() != 2 || inputs(4)[0].empty() ||
        inputs(4)[1] != reciprocal ||
        inputs(5) != std::vector<std::string>{normalized, slice1} ||
        inputs(6).size() != 2 || inputs(6)[0] != difference ||
        inputs(6)[1].empty() || inputs(7).size() < 1 ||
        inputs(7)[0] != scaled || inputs(8).size() < 1 ||
        inputs(8)[0] != scaled || inputs(9) !=
            std::vector<std::string>{scaled0, scaled1} ||
        inputs(10) != std::vector<std::string>{rounded_input} ||
        inputs(11).size() < 1 || inputs(11)[0] != slice1 ||
        inputs(12).size() < 1 || inputs(12)[0] != slice1 ||
        inputs(13) != std::vector<std::string>{y_raw, prior0} ||
        inputs(14).size() != 2 || inputs(14)[0] != updated_input0 ||
        inputs(14)[1].empty() || inputs(15) !=
            std::vector<std::string>{y_raw, prior1} ||
        inputs(16).size() != 2 || inputs(16)[0] != updated_input1 ||
        inputs(16)[1].empty() || inputs(17) !=
            std::vector<std::string>{updated0, updated1} ||
        inputs(18) != std::vector<std::string>{concat2, prior_name} ||
        nodes.at(index + 17).at("attributes").value("axis", 0) != 1 ||
        nodes.at(index + 18).at("attributes").value("axis", 0) != 1) {
        return false;
    }

    const std::array<std::pair<const std::string*, std::size_t>, 19>
        expected_consumers{{
            {&prior_name, 3}, {&slice0, 1}, {&slice1, 3},
            {&clipped, 2}, {&reciprocal, 1}, {&normalized, 2},
            {&difference, 1}, {&scaled, 2}, {&scaled0, 1},
            {&scaled1, 1}, {&rounded_input, 1}, {&y_raw, 2},
            {&prior0, 1}, {&prior1, 1}, {&updated_input0, 1},
            {&updated0, 1}, {&updated_input1, 1}, {&updated1, 1},
            {&concat2, 2}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (consumer_count(*name) != expected)
            return false;
    }

    const Value& prior_value = value(prior_name);
    const Value& latent_value = value(inputs(4)[0]);
    const Value& quant_scale = value(inputs(6)[1]);
    const Value& update_scale0 = value(inputs(14)[1]);
    const Value& update_scale1 = value(inputs(16)[1]);
    const Value& normalized_value = value(normalized);
    const Value& clipped_value = value(clipped);
    const Value& raw_value = value(y_raw);
    const Value& concat2_value = value(concat2);
    const Value& concat3_value = value(concat3);
    if (prior_value.dtype != "fp16" || latent_value.dtype != "fp16" ||
        quant_scale.dtype != "fp16" || update_scale0.dtype != "fp16" ||
        update_scale1.dtype != "fp16" || normalized_value.dtype != "fp16" ||
        clipped_value.dtype != "fp16" || raw_value.dtype != "fp16" ||
        concat2_value.dtype != "fp16" || concat3_value.dtype != "fp16" ||
        prior_value.shape.size() != 4 || latent_value.shape.size() != 4 ||
        raw_value.shape.size() != 4 || concat3_value.shape.size() != 4) {
        return false;
    }

    int quant_channels = static_cast<int>(raw_value.shape[1]);
    // The small profile has too little parallel work to amortize this
    // fused kernel's register and launch footprint on the A30.
    if (quant_channels < 32)
        return false;
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
        concat3_value.shape != concat3_expected ||
        value(inputs(2)[1]).dtype != "fp16" ||
        value(inputs(2)[1]).shape != std::vector<int64_t>{1}) {
        return false;
    }
    if (!is_channel_slice(nodes.at(index), 0, quant_channels * 2) ||
        !is_channel_slice(nodes.at(index + 1), quant_channels * 2,
                          quant_channels * 4) ||
        !is_channel_slice(nodes.at(index + 7), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 8), quant_channels,
                          quant_channels * 2) ||
        !is_channel_slice(nodes.at(index + 11), 0, quant_channels) ||
        !is_channel_slice(nodes.at(index + 12), quant_channels,
                          quant_channels * 2)) {
        return false;
    }

    if (ranges_overlap(concat3_value, prior_value) ||
        ranges_overlap(concat3_value, latent_value) ||
        ranges_overlap(concat3_value, normalized_value) ||
        ranges_overlap(concat3_value, clipped_value) ||
        ranges_overlap(concat2_value, prior_value) ||
        ranges_overlap(concat2_value, normalized_value) ||
        ranges_overlap(concat2_value, clipped_value) ||
        (ranges_overlap(concat2_value, latent_value) &&
         concat2_value.address != latent_value.address) ||
        ranges_overlap(raw_value, prior_value) ||
        ranges_overlap(raw_value, normalized_value) ||
        ranges_overlap(raw_value, clipped_value) ||
        ranges_overlap(raw_value, concat2_value) ||
        (ranges_overlap(raw_value, latent_value) &&
         raw_value.address != latent_value.address)) {
        return false;
    }

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
    float clip_min = scalar_fp16(inputs(2)[1]);
    int batch_count = static_cast<int>(raw_shape[0]);
    int spatial_count = static_cast<int>(raw_shape[2] * raw_shape[3]);
    void* parameters[] = {
        &latent, &prior, &quant, &update0, &update1, &normalized_address,
        &clipped_address, &raw, &concat2_address, &concat3_address,
        &clip_min, &batch_count, &quant_channels, &spatial_count};
    const std::size_t work = static_cast<std::size_t>(batch_count) *
        quant_channels * 4 * spatial_count;
    launch_linear(y0_tail_, work, parameters);
    return true;
}

bool AotGraph::try_execute_y1_tail(const json& nodes, std::size_t index)
{
    if (index + 14 >= nodes.size())
        return false;
    static constexpr std::array<const char*, 15> operations{
        "Sub", "Mul", "Slice", "Slice", "Add", "Round", "Slice",
        "Slice", "Add", "Mul", "Add", "Mul", "Concat", "Add", "Mul"};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).at("op") != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).at("inputs")
            .get<std::vector<std::string>>();
    };
    auto output = [&](std::size_t offset) -> std::string {
        const auto names = nodes.at(index + offset).at("outputs")
            .get<std::vector<std::string>>();
        return names.size() == 1 ? names[0] : std::string{};
    };
    auto consumer_count = [&](const std::string& name) {
        std::size_t count = 0;
        for (const auto& candidate : nodes) {
            const auto candidate_inputs = candidate.at("inputs")
                .get<std::vector<std::string>>();
            count += static_cast<std::size_t>(std::count(
                candidate_inputs.begin(), candidate_inputs.end(), name));
        }
        return count;
    };

    const auto first_inputs = inputs(0);
    if (first_inputs.size() != 2 || first_inputs[0].empty() ||
        first_inputs[1].empty()) {
        return false;
    }
    const std::string normalized_name = first_inputs[0];
    const std::string prior_name = first_inputs[1];
    const std::string difference = output(0);
    const std::string scaled = output(1);
    const std::string scaled0 = output(2);
    const std::string scaled1 = output(3);
    const std::string rounded_input = output(4);
    const std::string y_raw = output(5);
    const std::string prior0 = output(6);
    const std::string prior1 = output(7);
    const std::string updated_input0 = output(8);
    const std::string updated0 = output(9);
    const std::string updated_input1 = output(10);
    const std::string updated1 = output(11);
    const std::string concat4 = output(12);
    const std::string added = output(13);
    const std::string result_name = output(14);
    if (difference.empty() || scaled.empty() || scaled0.empty() ||
        scaled1.empty() || rounded_input.empty() || y_raw.empty() ||
        prior0.empty() || prior1.empty() || updated_input0.empty() ||
        updated0.empty() || updated_input1.empty() || updated1.empty() ||
        concat4.empty() || added.empty() || result_name.empty() ||
        inputs(1).size() != 2 || inputs(1)[0] != difference ||
        inputs(1)[1].empty() || inputs(2).size() < 1 ||
        inputs(2)[0] != scaled || inputs(3).size() < 1 ||
        inputs(3)[0] != scaled || inputs(4) !=
            std::vector<std::string>{scaled0, scaled1} ||
        inputs(5) != std::vector<std::string>{rounded_input} ||
        inputs(6).size() < 1 || inputs(6)[0] != prior_name ||
        inputs(7).size() < 1 || inputs(7)[0] != prior_name ||
        inputs(8) != std::vector<std::string>{y_raw, prior0} ||
        inputs(9).size() != 2 || inputs(9)[0] != updated_input0 ||
        inputs(9)[1].empty() || inputs(10) !=
            std::vector<std::string>{y_raw, prior1} ||
        inputs(11).size() != 2 || inputs(11)[0] != updated_input1 ||
        inputs(11)[1].empty() || inputs(12) !=
            std::vector<std::string>{updated0, updated1} ||
        inputs(13).size() != 2 || inputs(13)[0].empty() ||
        inputs(13)[1] != concat4 ||
        inputs(14).size() != 2 || inputs(14)[0] != added ||
        inputs(14)[1].empty() ||
        nodes.at(index + 12).at("attributes").value("axis", 0) != 1) {
        return false;
    }

    const std::string concat2_name = inputs(13)[0];
    const std::string clipped_name = inputs(14)[1];
    if (consumer_count(difference) != 1 || consumer_count(scaled) != 2 ||
        consumer_count(scaled0) != 1 || consumer_count(scaled1) != 1 ||
        consumer_count(rounded_input) != 1 || consumer_count(y_raw) != 2 ||
        consumer_count(prior_name) != 3 || consumer_count(prior0) != 1 ||
        consumer_count(prior1) != 1 || consumer_count(updated_input0) != 1 ||
        consumer_count(updated0) != 1 || consumer_count(updated_input1) != 1 ||
        consumer_count(updated1) != 1 || consumer_count(concat4) != 1 ||
        consumer_count(added) != 1 || consumer_count(result_name) != 1 ||
        consumer_count(concat2_name) != 2 || consumer_count(clipped_name) != 2) {
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
    if (normalized_value.dtype != "fp16" || prior_value.dtype != "fp16" ||
        quant_scale.dtype != "fp16" || update_scale0.dtype != "fp16" ||
        update_scale1.dtype != "fp16" || concat2_value.dtype != "fp16" ||
        clipped_value.dtype != "fp16" || raw_value.dtype != "fp16" ||
        result.dtype != "fp16" || normalized_value.shape.size() != 4 ||
        prior_value.shape.size() != 4 || raw_value.shape.size() != 4 ||
        result.shape.size() != 4) {
        return false;
    }
    int quant_channels = static_cast<int>(raw_value.shape[1]);
    // Keep the generic schedule for small fixed-shape profiles; their
    // simpler kernels are faster than this wider fused tail.
    if (quant_channels < 32)
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
        nodes.at(index + 13).at("attributes").value("axis", 0) != 0) {
        return false;
    }

    if (ranges_overlap(raw_value, normalized_value) ||
        ranges_overlap(raw_value, prior_value) ||
        ranges_overlap(raw_value, concat2_value) ||
        ranges_overlap(raw_value, clipped_value) ||
        (ranges_overlap(result, concat2_value) &&
         result.address != concat2_value.address) ||
        ranges_overlap(result, normalized_value) ||
        ranges_overlap(result, prior_value) ||
        ranges_overlap(result, clipped_value) ||
        ranges_overlap(result, raw_value)) {
        return false;
    }

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
    void* parameters[] = {
        &normalized, &prior, &quant, &update0, &update1, &concat2,
        &clipped, &raw, &result_address, &batch_count, &quant_channels,
        &spatial_count};
    const std::size_t work = static_cast<std::size_t>(batch_count) *
        quant_channels * spatial_count;
    launch_linear(y1_tail_, work, parameters);
    return true;
}

bool AotGraph::try_execute_feature_update(const json& nodes, std::size_t index)
{
    if (index + 11 >= nodes.size())
        return false;
    static constexpr std::array<const char*, 12> operations{
        "Slice", "Slice", "Slice", "Sigmoid", "Sigmoid", "Mul",
        "Sub", "Mul", "Add", "Mul", "Mul", "Concat"};
    for (std::size_t offset = 0; offset < operations.size(); ++offset) {
        if (nodes.at(index + offset).at("op") != operations[offset])
            return false;
    }

    auto inputs = [&](std::size_t offset) {
        return nodes.at(index + offset).at("inputs")
            .get<std::vector<std::string>>();
    };
    auto output = [&](std::size_t offset) -> std::string {
        const auto names = nodes.at(index + offset).at("outputs")
            .get<std::vector<std::string>>();
        return names.size() == 1 ? names[0] : std::string{};
    };

    const auto slice0_inputs = inputs(0);
    const auto slice1_inputs = inputs(1);
    const auto slice2_inputs = inputs(2);
    if (slice0_inputs.empty() || slice1_inputs.empty() ||
        slice2_inputs.empty() || slice0_inputs[0] != slice1_inputs[0] ||
        slice0_inputs[0] != slice2_inputs[0]) {
        return false;
    }
    const std::string slice0 = output(0);
    const std::string slice1 = output(1);
    const std::string slice2 = output(2);
    const std::string sigmoid0 = output(3);
    const std::string sigmoid1 = output(4);
    const std::string weighted1 = output(5);
    const std::string inverse_gate = output(6);
    const std::string weighted0 = output(7);
    const std::string blended = output(8);
    const std::string gated = output(9);
    const std::string scaled = output(10);
    const std::string result_name = output(11);
    if (slice0.empty() || slice1.empty() || slice2.empty() ||
        sigmoid0.empty() || sigmoid1.empty() || weighted1.empty() ||
        inverse_gate.empty() || weighted0.empty() || blended.empty() ||
        gated.empty() || scaled.empty() || result_name.empty() ||
        inputs(3) != std::vector<std::string>{slice1} ||
        inputs(4) != std::vector<std::string>{slice2} ||
        inputs(5).size() != 2 || inputs(5)[0] != sigmoid0 ||
        inputs(6).size() != 2 || inputs(6)[1] != sigmoid0 ||
        inputs(7) != std::vector<std::string>{inverse_gate, slice0} ||
        inputs(8) != std::vector<std::string>{weighted1, weighted0} ||
        inputs(9) != std::vector<std::string>{sigmoid1, blended} ||
        inputs(10).size() != 2 || inputs(10)[0] != gated ||
        inputs(11) != std::vector<std::string>{scaled, blended} ||
        nodes.at(index + 11).at("attributes").at("axis").get<int>() != 1) {
        return false;
    }

    auto consumer_count = [&](const std::string& name) {
        std::size_t count = 0;
        for (const auto& candidate : nodes) {
            const auto candidate_inputs = candidate.at("inputs")
                .get<std::vector<std::string>>();
            count += static_cast<std::size_t>(std::count(
                candidate_inputs.begin(), candidate_inputs.end(), name));
        }
        return count;
    };
    const std::array<std::pair<const std::string*, std::size_t>, 11>
        expected_consumers{{
            {&slice0, 1}, {&slice1, 1}, {&slice2, 1}, {&sigmoid0, 2},
            {&sigmoid1, 1}, {&weighted1, 1}, {&inverse_gate, 1},
            {&weighted0, 1}, {&blended, 2}, {&gated, 1}, {&scaled, 1}}};
    for (const auto& [name, expected] : expected_consumers) {
        if (consumer_count(*name) != expected)
            return false;
    }

    const Value& source = value(slice0_inputs[0]);
    const Value& first = value(slice0);
    const Value& history = value(inputs(5)[1]);
    const Value& scale = value(inputs(10)[1]);
    const Value& result = value(result_name);
    if (source.dtype != "fp16" || first.dtype != "fp16" ||
        history.dtype != "fp16" || scale.dtype != "fp16" ||
        result.dtype != "fp16" ||
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
        if (value(output(offset)).dtype != "fp16" ||
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

    DeviceAddress source_address = source.address;
    DeviceAddress history_address = history.address;
    DeviceAddress scale_address = scale.address;
    DeviceAddress result_address = result.address;
    int batch_count = static_cast<int>(first.shape[0]);
    int spatial_count = static_cast<int>(first.shape[2] * first.shape[3]);
    void* parameters[] = {
        &source_address, &history_address, &scale_address, &result_address,
        &batch_count, &channels, &spatial_count};
    const std::size_t pairs = static_cast<std::size_t>(batch_count) *
        channels * ((spatial_count + 1) / 2);
    launch_linear(feature_update_, pairs, parameters);
    return true;
}

bool AotGraph::try_execute_pointwise_epilogue(const json& nodes, std::size_t index)
{
    if (driver_.device_info().compute_major < 8 || index + 1 >= nodes.size())
        return false;
    const json& convolution = nodes.at(index);
    const json& epilogue_node = nodes.at(index + 1);
    const std::string epilogue_op =
        epilogue_node.at("op").get<std::string>();
    if (convolution.at("op") != "Conv" ||
        (epilogue_op != "LeakyRelu" && epilogue_op != "Add")) {
        return false;
    }

    const auto convolution_inputs =
        convolution.at("inputs").get<std::vector<std::string>>();
    const auto convolution_outputs =
        convolution.at("outputs").get<std::vector<std::string>>();
    const auto epilogue_inputs =
        epilogue_node.at("inputs").get<std::vector<std::string>>();
    const auto epilogue_outputs =
        epilogue_node.at("outputs").get<std::vector<std::string>>();
    if (convolution_inputs.size() < 2 || convolution_outputs.size() != 1 ||
        epilogue_outputs.size() != 1 ||
        std::find(output_names_.begin(), output_names_.end(),
                  convolution_outputs[0]) != output_names_.end()) {
        return false;
    }

    const std::string& convolution_output_name = convolution_outputs[0];
    std::size_t consumer_count = 0;
    for (const auto& candidate : nodes) {
        for (const std::string& input :
             candidate.at("inputs").get<std::vector<std::string>>()) {
            consumer_count += input == convolution_output_name ? 1 : 0;
        }
    }
    if (consumer_count != 1)
        return false;

    DeviceAddress epilogue_input_address = 0;
    int epilogue = 1;
    float alpha = epilogue_node.at("attributes").value("alpha", 0.01F);
    const Value* residual = nullptr;
    if (epilogue_op == "LeakyRelu") {
        if (epilogue_inputs != std::vector<std::string>{convolution_output_name})
            return false;
    } else {
        if (epilogue_inputs.size() != 2)
            return false;
        const auto found = std::find(epilogue_inputs.begin(),
                                     epilogue_inputs.end(),
                                     convolution_output_name);
        if (found == epilogue_inputs.end())
            return false;
        const std::string& residual_name =
            epilogue_inputs[found == epilogue_inputs.begin() ? 1 : 0];
        residual = &value(residual_name);
        epilogue_input_address = residual->address;
        epilogue = 2;
    }

    const Value& input = value(convolution_inputs[0]);
    const Value& weight = value(convolution_inputs[1]);
    const Value& convolution_output = value(convolution_output_name);
    const Value& output = value(epilogue_outputs[0]);
    const auto& attributes = convolution.at("attributes");
    const auto strides = attributes.at("strides").get<std::vector<int>>();
    const auto pads = attributes.at("pads").get<std::vector<int>>();
    if (input.dtype != "fp16" || weight.dtype != "fp16" ||
        output.dtype != "fp16" || input.shape.size() != 4 ||
        weight.shape.size() != 4 || output.shape.size() != 4 ||
        attributes.value("group", 1) != 1 || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || strides != std::vector<int>{1, 1} ||
        pads != std::vector<int>{0, 0, 0, 0} ||
        input.shape[0] != output.shape[0] ||
        input.shape[1] != weight.shape[1] ||
        input.shape[2] != output.shape[2] ||
        input.shape[3] != output.shape[3] ||
        output.shape[1] != weight.shape[0] ||
        convolution_output.shape != output.shape ||
        input.shape[1] % 16 != 0 || output.shape[1] % 128 != 0 ||
        (residual && residual->shape != output.shape) ||
        ranges_overlap(output, input) ||
        (residual && ranges_overlap(output, *residual))) {
        return false;
    }

    execute(convolution, &output, epilogue_input_address, epilogue, alpha);
    return true;
}

bool AotGraph::try_execute_reglu(const json& nodes, std::size_t index)
{
    if (index + 3 >= nodes.size())
        return false;
    const json& first_slice = nodes.at(index);
    const json& second_slice = nodes.at(index + 1);
    const json& clip = nodes.at(index + 2);
    const json& multiply = nodes.at(index + 3);
    if (first_slice.at("op") != "Slice" || second_slice.at("op") != "Slice" ||
        clip.at("op") != "Clip" || multiply.at("op") != "Mul") {
        return false;
    }

    const auto first_inputs =
        first_slice.at("inputs").get<std::vector<std::string>>();
    const auto second_inputs =
        second_slice.at("inputs").get<std::vector<std::string>>();
    const auto first_outputs =
        first_slice.at("outputs").get<std::vector<std::string>>();
    const auto second_outputs =
        second_slice.at("outputs").get<std::vector<std::string>>();
    const auto clip_inputs = clip.at("inputs").get<std::vector<std::string>>();
    const auto clip_outputs = clip.at("outputs").get<std::vector<std::string>>();
    const auto multiply_inputs =
        multiply.at("inputs").get<std::vector<std::string>>();
    const auto multiply_outputs =
        multiply.at("outputs").get<std::vector<std::string>>();
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
    if (input.dtype != "fp16" || output.dtype != "fp16" ||
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
    if (clip_inputs.size() > 1 && !clip_inputs[1].empty())
        minimum = scalar_fp16(clip_inputs[1]);
    if (clip_inputs.size() > 2 && !clip_inputs[2].empty())
        maximum = scalar_fp16(clip_inputs[2]);
    DeviceAddress input_address = input.address;
    if (std::find(direct_concat_values_.begin(),
                  direct_concat_values_.end(), multiply_outputs[0]) ==
        direct_concat_values_.end()) {
        output.address = reglu_buffer_.address();
    }
    DeviceAddress output_address = output.address;
    int batch_count = static_cast<int>(output.shape[0]);
    int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
    int work_count = spatial_count % 2 == 0 ? spatial_count / 2 : spatial_count;
    void* parameters[] = {
        &input_address, &output_address, &batch_count, &channels,
        &spatial_count, &minimum, &maximum};
    if (spatial_count % 8 == 0) {
        launch_linear(
            reglu_vec8_,
            static_cast<std::size_t>(batch_count) * channels *
                (spatial_count / 8),
            parameters);
    } else {
        driver_.launch(reglu_, {divide_up(work_count, 256),
                                static_cast<unsigned int>(channels),
                                static_cast<unsigned int>(batch_count)},
                       {256, 1, 1}, 0, parameters);
    }
    return true;
}

bool AotGraph::try_execute_pointwise_reglu(const json& nodes, std::size_t index)
{
    if (driver_.device_info().compute_major < 8 || index + 4 >= nodes.size())
        return false;
    const json& convolution = nodes.at(index);
    const json& first_slice = nodes.at(index + 1);
    const json& second_slice = nodes.at(index + 2);
    const json& clip = nodes.at(index + 3);
    const json& multiply = nodes.at(index + 4);
    if (convolution.at("op") != "Conv" || first_slice.at("op") != "Slice" ||
        second_slice.at("op") != "Slice" || clip.at("op") != "Clip" ||
        multiply.at("op") != "Mul") {
        return false;
    }

    const auto convolution_inputs =
        convolution.at("inputs").get<std::vector<std::string>>();
    const auto convolution_outputs =
        convolution.at("outputs").get<std::vector<std::string>>();
    const auto first_inputs =
        first_slice.at("inputs").get<std::vector<std::string>>();
    const auto second_inputs =
        second_slice.at("inputs").get<std::vector<std::string>>();
    const auto first_outputs =
        first_slice.at("outputs").get<std::vector<std::string>>();
    const auto second_outputs =
        second_slice.at("outputs").get<std::vector<std::string>>();
    const auto clip_inputs = clip.at("inputs").get<std::vector<std::string>>();
    const auto clip_outputs = clip.at("outputs").get<std::vector<std::string>>();
    const auto multiply_inputs = multiply.at("inputs").get<std::vector<std::string>>();
    const auto multiply_outputs = multiply.at("outputs").get<std::vector<std::string>>();
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
    const auto& attributes = convolution.at("attributes");
    const auto strides = attributes.at("strides").get<std::vector<int>>();
    const auto pads = attributes.at("pads").get<std::vector<int>>();
    const int groups = attributes.value("group", 1);
    if (input.dtype != "fp16" || weight.dtype != "fp16" ||
        output.dtype != "fp16" || input.shape.size() != 4 ||
        weight.shape.size() != 4 || output.shape.size() != 4 ||
        convolution_output.shape.size() != 4 || groups != 1 ||
        weight.shape[2] != 1 || weight.shape[3] != 1 ||
        strides != std::vector<int>{1, 1} ||
        pads != std::vector<int>{0, 0, 0, 0} ||
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
    if (gate_channels % 64 != 0 || in_channels % 16 != 0 ||
        !is_channel_slice(first_slice, 0, gate_channels) ||
        !is_channel_slice(second_slice, gate_channels, gate_channels * 2)) {
        return false;
    }

    float minimum = -std::numeric_limits<float>::infinity();
    float maximum = std::numeric_limits<float>::infinity();
    if (clip_inputs.size() > 1 && !clip_inputs[1].empty())
        minimum = scalar_fp16(clip_inputs[1]);
    if (clip_inputs.size() > 2 && !clip_inputs[2].empty())
        maximum = scalar_fp16(clip_inputs[2]);
    DeviceAddress input_address = input.address;
    DeviceAddress weight_address = weight.address;
    DeviceAddress bias_address = convolution_inputs.size() > 2 &&
        !convolution_inputs[2].empty()
        ? value(convolution_inputs[2]).address : 0;
    if (std::find(direct_concat_values_.begin(),
                  direct_concat_values_.end(), multiply_outputs[0]) ==
        direct_concat_values_.end()) {
        output.address = reglu_buffer_.address();
    }
    DeviceAddress output_address = output.address;
    int batch_count = static_cast<int>(output.shape[0]);
    int spatial_count = static_cast<int>(output.shape[2] * output.shape[3]);
    DeviceAddress convolution_output_address = convolution_output.address;
    if (launch_cutlass_pointwise(
            convolution, input_address, weight_address, bias_address, 0,
            convolution_output_address, batch_count, in_channels,
            spatial_count, gate_channels * 2, 0, 0.0F)) {
        const int work_count = (spatial_count & 1) == 0
            ? spatial_count / 2 : spatial_count;
        void* reglu_parameters[] = {
            &convolution_output_address, &output_address, &batch_count,
            &gate_channels, &spatial_count, &minimum, &maximum};
        if (spatial_count % 8 == 0) {
            launch_linear(
                reglu_vec8_,
                static_cast<std::size_t>(batch_count) * gate_channels *
                    (spatial_count / 8),
                reglu_parameters);
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

    void* parameters[] = {
        &input_address, &weight_address, &bias_address, &output_address,
        &batch_count, &in_channels, &spatial_count, &gate_channels,
        &minimum, &maximum};
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
