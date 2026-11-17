// Derived from microsoft/libmlvc (https://github.com/microsoft/libmlvc)
// Copyright (c) 2026 Microsoft Corporation
// Licensed under the MIT License. See licenses/libmlvc/LICENSE.

#include "mlvc/core/bitstream.hpp"

#include "bit_io.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <streambuf>
#include <string>
#include <utility>

namespace mlvc {
namespace {

constexpr int kNaluTypeBits = 6;
constexpr int kLayerIdBits = 6;
constexpr int kTemporalIdPlus1Bits = 3;
constexpr int kMlvcVersionBits = 8;
constexpr int kModelSizeBits = 12;
constexpr int kMaxTemporalLayersMinus1Bits = 3;
constexpr int kLtrNumSlotsBits = 4;

bool is_start_code(std::span<const std::byte> data) noexcept
{
    return data.size() >= kNaluStartCode.size() &&
        std::equal(kNaluStartCode.begin(), kNaluStartCode.end(), data.begin());
}

void write_nalu_header(detail::BitWriter& writer, const NaluHeader& header)
{
    if (header.forbidden_zero != 0)
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "NALU forbidden_zero bit must be 0");
    if (header.layer_id < 0 || header.layer_id >= (1 << kLayerIdBits))
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "NALU layer_id is out of range");
    if (header.temporal_id_plus1 <= 0 ||
        header.temporal_id_plus1 >= (1 << kTemporalIdPlus1Bits)) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "NALU temporal_id_plus1 is out of range");
    }
    writer.write_bit(false);
    writer.write_uint(static_cast<int>(header.nalu_type), kNaluTypeBits);
    writer.write_uint(header.layer_id, kLayerIdBits);
    writer.write_uint(header.temporal_id_plus1, kTemporalIdPlus1Bits);
}

NaluHeader parse_nalu_header(detail::BitReader& reader)
{
    if (reader.read_bit()) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "forbidden_zero bit is set in NALU header");
    }
    NaluHeader header;
    header.forbidden_zero = 0;
    header.nalu_type = static_cast<NaluType>(reader.read_uint(kNaluTypeBits));
    header.layer_id = reader.read_uint(kLayerIdBits);
    header.temporal_id_plus1 = reader.read_uint(kTemporalIdPlus1Bits);
    return header;
}

FrameHeader parse_frame_header(detail::BitReader& reader, const SpsNalu& sps,
                               const NaluHeader& nalu_header)
{
    FrameHeader frame_header;
    frame_header.pps_id = reader.read_ue();
    frame_header.qp_delta = reader.read_se();

    const int frame_idx_bits = sps.frame_idx_bits_minus8 + kMinFrameIdxBits;
    std::fill(frame_header.ltr_slots.begin(), frame_header.ltr_slots.end(),
              LtrSlotInfo{});
    const int num_ltr_slots = reader.read_uint(kLtrNumSlotsBits);
    if (num_ltr_slots > kMaxLtrSlots) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "num_ltr_slots exceeds maximum");
    }
    for (int i = 0; i < num_ltr_slots; ++i) {
        const bool is_empty = reader.read_bit();
        if (!is_empty) {
            frame_header.ltr_slots[static_cast<std::size_t>(i)] =
                LtrSlotInfo{reader.read_uint(frame_idx_bits)};
        }
    }

    if (nalu_header.nalu_type != NaluType::kIdrNLp) {
        frame_header.frame_idx = reader.read_uint(frame_idx_bits);
        frame_header.ref_frame_idx = reader.read_uint(frame_idx_bits);
        frame_header.feature_reset_flag = reader.read_bit();
    } else {
        frame_header.frame_idx = 0;
        frame_header.ref_frame_idx = 0;
        frame_header.feature_reset_flag = false;
    }
    return frame_header;
}

void validate_frame_index_value(int value, int frame_idx_bits, const char* name)
{
    if (value < 0 ||
        static_cast<std::uint64_t>(value) >= (std::uint64_t{1} << frame_idx_bits)) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             std::string(name) + " is outside frame_idx_bits range");
    }
}

void validate_frame_data(const FrameData& data)
{
    if (data.mlvc_version.major != kSupportedMlvcMajorVersion ||
        data.mlvc_version.minor < 0 || data.mlvc_version.minor > 255) {
        throw BitstreamError(BitstreamErrorCode::kUnsupportedVersion,
                             "unsupported MLVC bitstream model version");
    }
    if (data.model_width <= 0 || data.model_height <= 0 ||
        data.model_width % 2 != 0 || data.model_height % 2 != 0 ||
        data.model_width / 2 >= (1 << kModelSizeBits) ||
        data.model_height / 2 >= (1 << kModelSizeBits)) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "model dimensions are outside SPS NALU range");
    }
    if (data.crop_offsets.left < 0 || data.crop_offsets.right < 0 ||
        data.crop_offsets.top < 0 || data.crop_offsets.bottom < 0 ||
        data.crop_offsets.left % 2 != 0 || data.crop_offsets.right % 2 != 0 ||
        data.crop_offsets.top % 2 != 0 || data.crop_offsets.bottom % 2 != 0 ||
        static_cast<std::int64_t>(data.crop_offsets.left) + data.crop_offsets.right >=
            data.model_width ||
        static_cast<std::int64_t>(data.crop_offsets.top) + data.crop_offsets.bottom >=
            data.model_height) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "crop offsets are invalid for model dimensions");
    }
    if (data.max_temporal_layers < 1 || data.max_temporal_layers > kMaxTemporalLayers) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "max_temporal_layers is out of range");
    }
    if (data.temporal_id < 0 || data.temporal_id >= data.max_temporal_layers) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "temporal_id is out of range");
    }
    if (data.frame_idx_bits < kMinFrameIdxBits ||
        data.frame_idx_bits > kMaxFrameIdxBits) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "frame_idx_bits is outside [8, 30]");
    }
    if (data.qp < kMinBitstreamQp || data.qp > kMaxBitstreamQp) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "QP must be in [0, 51] for MLVC NALU bitstream");
    }
    if (data.frame_type != FrameType::kIFrame) {
        validate_frame_index_value(data.cur_frame_idx, data.frame_idx_bits, "cur_frame_idx");
        validate_frame_index_value(data.ref_frame_idx, data.frame_idx_bits, "ref_frame_idx");
    }
    for (const LtrSlotInfo& slot : data.ltr_slots) {
        if (slot.has_value())
            validate_frame_index_value(slot.frame_idx, data.frame_idx_bits, "ltr_frame_idx");
    }
    if (data.payload.empty()) {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "frame entropy payload must not be empty");
    }
}

}  // namespace

struct NaluBuilder::Impl {
    detail::BitWriter writer{64U * 1024U};
    std::vector<std::byte> output_buffer;

    Impl() { output_buffer.reserve(64U * 1024U); }

    void append_start_code()
    {
        output_buffer.insert(output_buffer.end(),
                             kNaluStartCode.begin(), kNaluStartCode.end());
    }

    void flush_bit_writer()
    {
        writer.write_stop_bit();
        detail::add_emulation_prevention_bytes(writer.bytes(), output_buffer);
        writer.reset();
    }
};

NaluBuilder::NaluBuilder() : impl_(std::make_unique<Impl>()) {}
NaluBuilder::~NaluBuilder() = default;
NaluBuilder::NaluBuilder(NaluBuilder&&) noexcept = default;
NaluBuilder& NaluBuilder::operator=(NaluBuilder&&) noexcept = default;

void NaluBuilder::reset()
{
    impl_->output_buffer.clear();
    impl_->writer.reset();
}

void NaluBuilder::append_sps(const SpsNalu& sps)
{
    impl_->append_start_code();
    impl_->writer.reset();
    write_nalu_header(impl_->writer, sps.nalu_header);
    impl_->writer.write_uint(sps.mlvc_version_major, kMlvcVersionBits);
    impl_->writer.write_uint(sps.mlvc_version_minor, kMlvcVersionBits);
    impl_->writer.write_ue(sps.sps_id);
    impl_->writer.write_uint(sps.model_width_div2, kModelSizeBits);
    impl_->writer.write_uint(sps.model_height_div2, kModelSizeBits);
    impl_->writer.write_bit(sps.transpose_flag);
    impl_->writer.write_bit(sps.crop_flag);
    if (sps.crop_flag) {
        impl_->writer.write_ue(sps.crop_left_div2);
        impl_->writer.write_ue(sps.crop_right_div2);
        impl_->writer.write_ue(sps.crop_top_div2);
        impl_->writer.write_ue(sps.crop_bottom_div2);
    }
    impl_->writer.write_uint(sps.max_temporal_layers_minus1,
                             kMaxTemporalLayersMinus1Bits);
    impl_->writer.write_ue(sps.frame_idx_bits_minus8);
    impl_->flush_bit_writer();
}

void NaluBuilder::append_pps(const PpsNalu& pps)
{
    impl_->append_start_code();
    impl_->writer.reset();
    write_nalu_header(impl_->writer, pps.nalu_header);
    impl_->writer.write_ue(pps.pps_id);
    impl_->writer.write_ue(pps.sps_id);
    impl_->writer.write_se(pps.init_qp_minus26);
    impl_->flush_bit_writer();
}

void NaluBuilder::append_frame(const SpsNalu& sps, const FrameNalu& frame)
{
    impl_->append_start_code();
    impl_->writer.reset();
    write_nalu_header(impl_->writer, frame.nalu_header);

    impl_->writer.write_ue(frame.frame_header.pps_id);
    impl_->writer.write_se(frame.frame_header.qp_delta);

    const int frame_idx_bits = sps.frame_idx_bits_minus8 + kMinFrameIdxBits;
    int num_ltr_slots = 0;
    for (int i = 0; i < static_cast<int>(frame.frame_header.ltr_slots.size()); ++i) {
        if (frame.frame_header.ltr_slots[static_cast<std::size_t>(i)].has_value())
            num_ltr_slots = i + 1;
    }
    impl_->writer.write_uint(num_ltr_slots, kLtrNumSlotsBits);
    for (int i = 0; i < num_ltr_slots; ++i) {
        const LtrSlotInfo& slot =
            frame.frame_header.ltr_slots[static_cast<std::size_t>(i)];
        impl_->writer.write_bit(slot.empty);
        if (slot.has_value())
            impl_->writer.write_uint(slot.frame_idx, frame_idx_bits);
    }

    if (frame.nalu_header.nalu_type != NaluType::kIdrNLp) {
        impl_->writer.write_uint(frame.frame_header.frame_idx, frame_idx_bits);
        impl_->writer.write_uint(frame.frame_header.ref_frame_idx, frame_idx_bits);
        impl_->writer.write_bit(frame.frame_header.feature_reset_flag);
    }

    impl_->writer.write_bytes(frame.payload);
    impl_->flush_bit_writer();
}

std::span<const std::byte> NaluBuilder::output() const noexcept
{
    return impl_->output_buffer;
}

struct NaluParser::Impl {
    std::span<const std::byte> buffer;
    std::vector<std::byte> rbsp_buffer;

    std::span<const std::byte> read_next_nalu()
    {
        if (buffer.empty()) {
            throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                 "unexpected end of NALU stream");
        }
        std::size_t next_nalu_start = 0;
        while (next_nalu_start < buffer.size() &&
               !is_start_code(buffer.subspan(next_nalu_start))) {
            ++next_nalu_start;
        }
        const auto nalu_bytes = buffer.subspan(0, next_nalu_start);
        if (next_nalu_start + kNaluStartCode.size() <= buffer.size())
            buffer = buffer.subspan(next_nalu_start + kNaluStartCode.size());
        else
            buffer = {};
        return nalu_bytes;
    }

    detail::BitReader prepare_reader(std::span<const std::byte> ebsp)
    {
        detail::remove_emulation_prevention_bytes(ebsp, rbsp_buffer);
        return detail::BitReader{rbsp_buffer};
    }
};

NaluParser::NaluParser() : impl_(std::make_unique<Impl>()) {}
NaluParser::~NaluParser() = default;
NaluParser::NaluParser(NaluParser&&) noexcept = default;
NaluParser& NaluParser::operator=(NaluParser&&) noexcept = default;

void NaluParser::open(std::span<const std::byte> buffer)
{
    impl_->buffer = buffer;
    if (is_start_code(impl_->buffer))
        impl_->buffer = impl_->buffer.subspan(kNaluStartCode.size());
}

bool NaluParser::has_next() const noexcept
{
    return !impl_->buffer.empty();
}

NaluHeader NaluParser::peek_nalu_header() const
{
    if (impl_->buffer.empty()) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "cannot peek NALU header from empty buffer");
    }
    detail::BitReader reader{impl_->buffer};
    return parse_nalu_header(reader);
}

SpsNalu NaluParser::read_sps()
{
    const auto ebsp = impl_->read_next_nalu();
    detail::BitReader reader = impl_->prepare_reader(ebsp);

    SpsNalu sps{};
    sps.nalu_header = parse_nalu_header(reader);
    sps.mlvc_version_major = reader.read_uint(kMlvcVersionBits);
    sps.mlvc_version_minor = reader.read_uint(kMlvcVersionBits);
    sps.sps_id = reader.read_ue();
    sps.model_width_div2 = reader.read_uint(kModelSizeBits);
    sps.model_height_div2 = reader.read_uint(kModelSizeBits);
    sps.transpose_flag = reader.read_bit();
    sps.crop_flag = reader.read_bit();
    if (sps.crop_flag) {
        sps.crop_left_div2 = reader.read_ue();
        sps.crop_right_div2 = reader.read_ue();
        sps.crop_top_div2 = reader.read_ue();
        sps.crop_bottom_div2 = reader.read_ue();
    } else {
        sps.crop_left_div2 = 0;
        sps.crop_right_div2 = 0;
        sps.crop_top_div2 = 0;
        sps.crop_bottom_div2 = 0;
    }

    const std::int64_t crop_width_div2 =
        static_cast<std::int64_t>(sps.crop_left_div2) + sps.crop_right_div2;
    const std::int64_t crop_height_div2 =
        static_cast<std::int64_t>(sps.crop_top_div2) + sps.crop_bottom_div2;
    if (crop_width_div2 >= sps.model_width_div2 ||
        crop_height_div2 >= sps.model_height_div2) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "invalid SPS geometry or crop offsets");
    }

    sps.max_temporal_layers_minus1 = reader.read_uint(kMaxTemporalLayersMinus1Bits);
    if (sps.max_temporal_layers_minus1 >= kMaxTemporalLayers) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "SPS max_temporal_layers exceeds supported limit");
    }

    sps.frame_idx_bits_minus8 = reader.read_ue();
    if (sps.frame_idx_bits_minus8 > kMaxFrameIdxBits - kMinFrameIdxBits) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "SPS frame_idx_bits exceeds supported limit");
    }

    reader.read_stop_bit();
    return sps;
}

PpsNalu NaluParser::read_pps()
{
    const auto ebsp = impl_->read_next_nalu();
    detail::BitReader reader = impl_->prepare_reader(ebsp);

    PpsNalu pps{};
    pps.nalu_header = parse_nalu_header(reader);
    pps.pps_id = reader.read_ue();
    pps.sps_id = reader.read_ue();
    pps.init_qp_minus26 = reader.read_se();
    if (pps.init_qp_minus26 < kMinBitstreamQp - kDefaultInitQp ||
        pps.init_qp_minus26 > kMaxBitstreamQp - kDefaultInitQp) {
        throw BitstreamError(BitstreamErrorCode::kUnexpected,
                             "PPS init_qp_minus26 is outside valid QP [0, 51] range");
    }

    reader.read_stop_bit();
    return pps;
}

FrameNalu NaluParser::read_frame(const SpsNalu& sps)
{
    const auto ebsp = impl_->read_next_nalu();
    detail::BitReader reader = impl_->prepare_reader(ebsp);

    FrameNalu frame{};
    frame.nalu_header = parse_nalu_header(reader);
    frame.frame_header = parse_frame_header(reader, sps, frame.nalu_header);
    frame.payload = reader.read_bytes();
    reader.read_stop_bit();
    return frame;
}

CustomNalu NaluParser::read_custom_nalu()
{
    const auto ebsp = impl_->read_next_nalu();
    detail::BitReader reader = impl_->prepare_reader(ebsp);

    CustomNalu custom{};
    custom.nalu_header = parse_nalu_header(reader);
    custom.payload = reader.read_bytes();
    return custom;
}

struct BitstreamEncoder::Impl {
    SpsNalu sps{};
    PpsNalu pps{};
    NaluBuilder nalu_builder;

    void reset()
    {
        sps = SpsNalu{};
        sps.sps_id = -1;
        pps = PpsNalu{};
        pps.pps_id = -1;
        pps.sps_id = -2;
        nalu_builder.reset();
    }

    Impl() { reset(); }
};

BitstreamEncoder::BitstreamEncoder() : impl_(std::make_unique<Impl>()) {}
BitstreamEncoder::~BitstreamEncoder() = default;
BitstreamEncoder::BitstreamEncoder(BitstreamEncoder&&) noexcept = default;
BitstreamEncoder& BitstreamEncoder::operator=(BitstreamEncoder&&) noexcept = default;

void BitstreamEncoder::reset()
{
    impl_->reset();
}

std::span<const std::byte> BitstreamEncoder::encode(const FrameData& data)
{
    validate_frame_data(data);
    impl_->nalu_builder.reset();

    if (data.frame_type == FrameType::kIFrame) {
        SpsNalu new_sps{};
        new_sps.nalu_header = NaluHeader{NaluType::kSps};
        new_sps.mlvc_version_major = data.mlvc_version.major;
        new_sps.mlvc_version_minor = data.mlvc_version.minor;
        new_sps.sps_id = impl_->sps.sps_id;
        new_sps.model_width_div2 = data.model_width / 2;
        new_sps.model_height_div2 = data.model_height / 2;
        new_sps.transpose_flag = data.transpose_flag;
        new_sps.crop_flag = !data.crop_offsets.is_empty();
        new_sps.crop_left_div2 = data.crop_offsets.left / 2;
        new_sps.crop_right_div2 = data.crop_offsets.right / 2;
        new_sps.crop_top_div2 = data.crop_offsets.top / 2;
        new_sps.crop_bottom_div2 = data.crop_offsets.bottom / 2;
        new_sps.max_temporal_layers_minus1 = data.max_temporal_layers - 1;
        new_sps.frame_idx_bits_minus8 = data.frame_idx_bits - kMinFrameIdxBits;

        const bool update_sps =
            impl_->sps.sps_id < 0 || impl_->pps.pps_id < 0 || new_sps != impl_->sps;
        if (update_sps) {
            const int new_sps_id = (impl_->sps.sps_id + 1) % kMaxSpsId;
            impl_->sps = new_sps;
            impl_->sps.sps_id = new_sps_id;

            const int new_pps_id = (impl_->pps.pps_id + 1) % kMaxPpsId;
            impl_->pps.pps_id = new_pps_id;
            impl_->pps.sps_id = impl_->sps.sps_id;
            impl_->pps.init_qp_minus26 = data.qp - kDefaultInitQp;
        }

        impl_->nalu_builder.append_sps(impl_->sps);
        impl_->nalu_builder.append_pps(impl_->pps);
    } else if (impl_->sps.sps_id < 0 || impl_->pps.pps_id < 0) {
        throw BitstreamError(BitstreamErrorCode::kMissingSps,
                             "cannot encode non-IDR frame before an IDR frame");
    }

    FrameNalu frame_nalu{};
    frame_nalu.nalu_header.temporal_id_plus1 = data.temporal_id + 1;
    if (data.frame_type == FrameType::kIFrame) {
        frame_nalu.nalu_header.nalu_type = NaluType::kIdrNLp;
    } else if (data.frame_type == FrameType::kPFrame ||
               data.frame_type == FrameType::kLtrRecovery) {
        frame_nalu.nalu_header.nalu_type =
            (data.temporal_id == 0) ? NaluType::kTrailR : NaluType::kTsaR;
    } else {
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument,
                             "unsupported frame type");
    }

    frame_nalu.frame_header.pps_id = impl_->pps.pps_id;
    frame_nalu.frame_header.qp_delta =
        data.qp - (impl_->pps.init_qp_minus26 + kDefaultInitQp);
    frame_nalu.frame_header.ltr_slots = data.ltr_slots;
    frame_nalu.frame_header.frame_idx = data.cur_frame_idx;
    frame_nalu.frame_header.ref_frame_idx = data.ref_frame_idx;
    frame_nalu.frame_header.feature_reset_flag = data.feature_reset_flag;
    frame_nalu.payload = data.payload;

    impl_->nalu_builder.append_frame(impl_->sps, frame_nalu);
    return impl_->nalu_builder.output();
}

struct BitstreamDecoder::Impl {
    SpsNalu sps{};
    PpsNalu pps{};
    NaluParser parser;

    void reset()
    {
        sps = SpsNalu{};
        sps.sps_id = -1;
        pps = PpsNalu{};
        pps.pps_id = -1;
        pps.sps_id = -2;
    }

    Impl() { reset(); }

    std::optional<FrameData> decode_next_nalu()
    {
        const NaluHeader header = parser.peek_nalu_header();
        const NaluType type = header.nalu_type;

        if (type == NaluType::kSps) {
            const SpsNalu parsed_sps = parser.read_sps();
            if (parsed_sps.mlvc_version_major != kSupportedMlvcMajorVersion) {
                throw BitstreamError(BitstreamErrorCode::kUnsupportedVersion,
                                     "unsupported MLVC bitstream major version");
            }
            sps = parsed_sps;
            return std::nullopt;
        }
        if (type == NaluType::kPps) {
            const PpsNalu parsed_pps = parser.read_pps();
            pps = parsed_pps;
            if (pps.sps_id != sps.sps_id) {
                throw BitstreamError(BitstreamErrorCode::kMissingSps,
                                     "PPS references unknown or inactive SPS");
            }
            return std::nullopt;
        }
        if (type == NaluType::kIdrNLp || type == NaluType::kTrailR ||
            type == NaluType::kTsaR || type == NaluType::kTrailN ||
            type == NaluType::kTsaN) {
            const FrameNalu frame = parser.read_frame(sps);
            if (frame.frame_header.pps_id != pps.pps_id) {
                throw BitstreamError(BitstreamErrorCode::kMissingPps,
                                     "frame header references unknown or inactive PPS");
            }
            if (pps.sps_id != sps.sps_id || sps.sps_id < 0) {
                throw BitstreamError(BitstreamErrorCode::kMissingSps,
                                     "frame header has no active matching SPS");
            }
            if (header.temporal_id_plus1 == 0) {
                throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                     "temporal_id_plus1 must be positive");
            }
            const int temporal_id = header.temporal_id_plus1 - 1;
            const int max_temporal_layers = sps.max_temporal_layers_minus1 + 1;
            if (temporal_id >= max_temporal_layers) {
                throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                     "frame temporal_id exceeds SPS max_temporal_layers");
            }

            FrameType frame_type = FrameType::kPFrame;
            if (type == NaluType::kIdrNLp) {
                frame_type = FrameType::kIFrame;
            } else if (temporal_id == 0 &&
                       (frame.frame_header.frame_idx - frame.frame_header.ref_frame_idx) > 2) {
                frame_type = FrameType::kLtrRecovery;
            }

            const std::int64_t qp =
                static_cast<std::int64_t>(frame.frame_header.qp_delta) +
                pps.init_qp_minus26 + kDefaultInitQp;
            if (qp < kMinBitstreamQp || qp > kMaxBitstreamQp) {
                throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                     "decoded frame QP is outside [0, 51]");
            }

            FrameData result{};
            result.mlvc_version = {sps.mlvc_version_major, sps.mlvc_version_minor};
            result.model_width = 2 * sps.model_width_div2;
            result.model_height = 2 * sps.model_height_div2;
            result.crop_offsets = {
                2 * sps.crop_left_div2,
                2 * sps.crop_right_div2,
                2 * sps.crop_top_div2,
                2 * sps.crop_bottom_div2,
            };
            result.transpose_flag = sps.transpose_flag;
            result.frame_idx_bits = sps.frame_idx_bits_minus8 + kMinFrameIdxBits;
            result.temporal_id = temporal_id;
            result.max_temporal_layers = max_temporal_layers;
            result.frame_type = frame_type;
            result.qp = static_cast<int>(qp);
            result.feature_reset_flag = frame.frame_header.feature_reset_flag;
            result.cur_frame_idx = frame.frame_header.frame_idx;
            result.ref_frame_idx = frame.frame_header.ref_frame_idx;
            result.ltr_slots = frame.frame_header.ltr_slots;
            result.payload = frame.payload;
            return result;
        }

        static_cast<void>(parser.read_custom_nalu());
        return std::nullopt;
    }
};

BitstreamDecoder::BitstreamDecoder() : impl_(std::make_unique<Impl>()) {}
BitstreamDecoder::~BitstreamDecoder() = default;
BitstreamDecoder::BitstreamDecoder(BitstreamDecoder&&) noexcept = default;
BitstreamDecoder& BitstreamDecoder::operator=(BitstreamDecoder&&) noexcept = default;

void BitstreamDecoder::reset()
{
    impl_->reset();
}

FrameData BitstreamDecoder::decode(std::span<const std::byte> access_unit)
{
    impl_->parser.open(access_unit);
    while (impl_->parser.has_next()) {
        std::optional<FrameData> frame = impl_->decode_next_nalu();
        if (frame.has_value() && impl_->parser.has_next()) {
            throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                 "extra NALUs after frame NALU are not supported");
        }
        if (frame.has_value())
            return *frame;
    }
    throw BitstreamError(BitstreamErrorCode::kPartialAccessUnit,
                         "incomplete MLVC access unit without frame NALU");
}

const SpsNalu& BitstreamDecoder::sps() const noexcept
{
    return impl_->sps;
}

const PpsNalu& BitstreamDecoder::pps() const noexcept
{
    return impl_->pps;
}

AccessUnitReader::AccessUnitReader(std::istream& input,
                                   std::size_t max_access_unit_bytes)
    : input_(&input), max_access_unit_bytes_(max_access_unit_bytes)
{
}

bool AccessUnitReader::fill_more()
{
    if (!input_ || !input_->rdbuf())
        throw BitstreamError(BitstreamErrorCode::kInvalidArgument, "invalid MLVC input stream");
    if (input_->bad() || (input_->fail() && !input_->eof()))
        throw BitstreamError(BitstreamErrorCode::kUnexpected, "I/O error reading MLVC bitstream");

    if (offset_ > 0) {
        buffer_.erase(buffer_.begin(),
                      buffer_.begin() + static_cast<std::ptrdiff_t>(offset_));
        offset_ = 0;
    }

    constexpr std::size_t kMaxReadChunk = 64U * 1024U;
    std::streambuf* stream_buf = input_->rdbuf();
    std::streamsize available = stream_buf->in_avail();
    if (available > 0) {
        const std::size_t to_read =
            std::min<std::size_t>(static_cast<std::size_t>(available), kMaxReadChunk);
        const std::size_t old_size = buffer_.size();
        buffer_.resize(old_size + to_read);
        const std::streamsize got = stream_buf->sgetn(
            reinterpret_cast<char*>(buffer_.data() + old_size),
            static_cast<std::streamsize>(to_read));
        buffer_.resize(old_size + (got > 0 ? static_cast<std::size_t>(got) : 0U));
        return got > 0;
    }

    const int first = stream_buf->sbumpc();
    if (first == std::char_traits<char>::eof()) {
        input_->setstate(std::ios::eofbit);
        return false;
    }
    buffer_.push_back(static_cast<std::byte>(static_cast<unsigned char>(first)));

    available = stream_buf->in_avail();
    if (available > 0) {
        const std::size_t to_read =
            std::min<std::size_t>(static_cast<std::size_t>(available), kMaxReadChunk - 1U);
        const std::size_t old_size = buffer_.size();
        buffer_.resize(old_size + to_read);
        const std::streamsize got = stream_buf->sgetn(
            reinterpret_cast<char*>(buffer_.data() + old_size),
            static_cast<std::streamsize>(to_read));
        buffer_.resize(old_size + (got > 0 ? static_cast<std::size_t>(got) : 0U));
    }
    return true;
}

bool AccessUnitReader::ensure_available(std::size_t count)
{
    while (buffer_.size() - offset_ < count) {
        if (!fill_more())
            return false;
    }
    return true;
}

std::size_t AccessUnitReader::find_nalu_end()
{
    std::size_t pos = offset_ + kNaluStartCode.size();
    while (true) {
        for (; pos + kNaluStartCode.size() <= buffer_.size(); ++pos) {
            if (is_start_code(std::span<const std::byte>(buffer_).subspan(pos)))
                return pos;
        }
        if (pos - offset_ > max_access_unit_bytes_) {
            throw BitstreamError(BitstreamErrorCode::kSizeLimitExceeded,
                                 "MLVC NALU exceeds configured size limit");
        }
        pos -= offset_;
        if (!fill_more())
            return buffer_.size();
        pos += offset_;
    }
}

bool AccessUnitReader::read(std::vector<std::byte>& access_unit)
{
    std::vector<std::byte> pending;
    while (true) {
        const bool have_header = ensure_available(kNaluStartCode.size() + 1U);
        if (offset_ == buffer_.size()) {
            if (!pending.empty()) {
                throw BitstreamError(BitstreamErrorCode::kPartialAccessUnit,
                                     "truncated MLVC access unit before VCL frame NALU");
            }
            return false;
        }
        if (!have_header ||
            !is_start_code(std::span<const std::byte>(buffer_).subspan(offset_))) {
            throw BitstreamError(BitstreamErrorCode::kUnexpected,
                                 "missing NALU start code in MLVC bitstream");
        }

        const std::size_t nal_end = find_nalu_end();
        const std::size_t nalu_size = nal_end - offset_;
        if (nalu_size > max_access_unit_bytes_ ||
            pending.size() > max_access_unit_bytes_ - nalu_size) {
            throw BitstreamError(BitstreamErrorCode::kSizeLimitExceeded,
                                 "MLVC access unit exceeds configured size limit");
        }

        const int nalu_type =
            (static_cast<int>(buffer_[offset_ + kNaluStartCode.size()]) >> 1) & 0x3F;
        pending.insert(pending.end(),
                       buffer_.begin() + static_cast<std::ptrdiff_t>(offset_),
                       buffer_.begin() + static_cast<std::ptrdiff_t>(nal_end));
        offset_ = nal_end;

        if (nalu_type < 32) {
            access_unit = std::move(pending);
            return true;
        }
    }
}

void write_access_unit(std::ostream& output, std::span<const std::byte> access_unit)
{
    if (!access_unit.empty()) {
        output.write(reinterpret_cast<const char*>(access_unit.data()),
                     static_cast<std::streamsize>(access_unit.size()));
    }
    if (!output)
        throw std::runtime_error("failed to write MLVC access unit");
}

}  // namespace mlvc
