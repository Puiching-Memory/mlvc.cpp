// Derived from microsoft/libmlvc (https://github.com/microsoft/libmlvc)
// Copyright (c) 2026 Microsoft Corporation
// Licensed under the MIT License. See licenses/libmlvc/LICENSE.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlvc {

inline constexpr std::size_t kMaxEncodedFramePayloadBytes =
    std::size_t{256} << 20;

inline constexpr int kSupportedMlvcMajorVersion = 0;
inline constexpr int kDefaultMlvcMinorVersion = 1;
inline constexpr int kMaxTemporalLayers = 2;
inline constexpr int kMaxLtrSlots = 8;
inline constexpr int kMinBitstreamQp = 0;
inline constexpr int kMaxBitstreamQp = 51;
inline constexpr int kDefaultInitQp = 26;
inline constexpr int kMinFrameIdxBits = 8;
inline constexpr int kMaxFrameIdxBits = 30;
inline constexpr int kDefaultFrameIdxBits = 10;
inline constexpr int kMaxSpsId = 3;
inline constexpr int kMaxPpsId = 3;

inline constexpr std::array<std::byte, 4> kNaluStartCode = {
    std::byte{0x00},
    std::byte{0x00},
    std::byte{0x00},
    std::byte{0x01},
};

enum class BitstreamErrorCode {
    kUnexpected,
    kPartialAccessUnit,
    kMissingSps,
    kMissingPps,
    kUnsupportedVersion,
    kInvalidArgument,
    kSizeLimitExceeded,
};

class BitstreamError final : public std::runtime_error {
public:
    BitstreamError(BitstreamErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code)
    {
    }

    BitstreamErrorCode code() const noexcept { return code_; }

private:
    BitstreamErrorCode code_;
};

enum class NaluType : std::uint8_t {
    kTrailN = 0,
    kTrailR = 1,
    kTsaN = 2,
    kTsaR = 3,
    kIdrNLp = 20,
    kSps = 33,
    kPps = 34,
};

enum class FrameType {
    kIFrame,
    kPFrame,
    kLtrRecovery,
};

struct MlvcVersion {
    int major = kSupportedMlvcMajorVersion;
    int minor = kDefaultMlvcMinorVersion;

    bool operator==(const MlvcVersion&) const = default;
};

struct CropOffsets {
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;

    bool is_empty() const noexcept
    {
        return left == 0 && right == 0 && top == 0 && bottom == 0;
    }

    bool operator==(const CropOffsets&) const = default;
};

struct NaluHeader {
    int forbidden_zero = 0;
    NaluType nalu_type = NaluType::kIdrNLp;
    int layer_id = 0;
    int temporal_id_plus1 = 1;

    NaluHeader() = default;
    explicit NaluHeader(NaluType type) : nalu_type(type) {}

    bool operator==(const NaluHeader&) const = default;
};

struct SpsNalu {
    NaluHeader nalu_header{NaluType::kSps};
    int mlvc_version_major = kSupportedMlvcMajorVersion;
    int mlvc_version_minor = kDefaultMlvcMinorVersion;
    int sps_id = -1;
    int model_width_div2 = 0;
    int model_height_div2 = 0;
    bool transpose_flag = false;
    bool crop_flag = false;
    int crop_left_div2 = 0;
    int crop_right_div2 = 0;
    int crop_top_div2 = 0;
    int crop_bottom_div2 = 0;
    int max_temporal_layers_minus1 = 0;
    int frame_idx_bits_minus8 = 0;

    bool operator==(const SpsNalu&) const = default;
};

struct PpsNalu {
    NaluHeader nalu_header{NaluType::kPps};
    int pps_id = -1;
    int sps_id = -2;
    int init_qp_minus26 = 0;

    bool operator==(const PpsNalu&) const = default;
};

struct LtrSlotInfo {
    bool empty = true;
    int frame_idx = 0;

    LtrSlotInfo() = default;
    explicit LtrSlotInfo(int idx) : empty(false), frame_idx(idx) {}

    bool has_value() const noexcept { return !empty; }
    bool operator==(const LtrSlotInfo&) const = default;
};

struct FrameHeader {
    int pps_id = 0;
    int qp_delta = 0;
    std::array<LtrSlotInfo, kMaxLtrSlots> ltr_slots{};
    int frame_idx = 0;
    int ref_frame_idx = 0;
    bool feature_reset_flag = false;
};

struct FrameNalu {
    NaluHeader nalu_header{NaluType::kIdrNLp};
    FrameHeader frame_header{};
    std::span<const std::byte> payload{};
};

struct CustomNalu {
    NaluHeader nalu_header{};
    std::span<const std::byte> payload{};
};

struct FrameData {
    MlvcVersion mlvc_version{};
    int model_width = 0;
    int model_height = 0;
    CropOffsets crop_offsets{};
    bool transpose_flag = false;
    int frame_idx_bits = kDefaultFrameIdxBits;
    int temporal_id = 0;
    int max_temporal_layers = 1;
    FrameType frame_type = FrameType::kIFrame;
    int qp = 0;
    bool feature_reset_flag = false;
    int cur_frame_idx = 0;
    int ref_frame_idx = 0;
    std::array<LtrSlotInfo, kMaxLtrSlots> ltr_slots{};
    // Non-owning span: when produced by BitstreamDecoder::decode(), valid until
    // the next decode() call or decoder destruction.
    std::span<const std::byte> payload{};

    int display_width() const noexcept
    {
        if (!transpose_flag)
            return model_width - crop_offsets.left - crop_offsets.right;
        return model_height - crop_offsets.top - crop_offsets.bottom;
    }

    int display_height() const noexcept
    {
        if (!transpose_flag)
            return model_height - crop_offsets.top - crop_offsets.bottom;
        return model_width - crop_offsets.left - crop_offsets.right;
    }
};

struct EncodedFrame {
    std::int32_t q_index = 0;
    FrameType frame_type = FrameType::kIFrame;
    std::vector<std::byte> bitstream;
};

class NaluBuilder final {
public:
    NaluBuilder();
    ~NaluBuilder();

    NaluBuilder(const NaluBuilder&) = delete;
    NaluBuilder& operator=(const NaluBuilder&) = delete;
    NaluBuilder(NaluBuilder&&) noexcept;
    NaluBuilder& operator=(NaluBuilder&&) noexcept;

    void reset();
    void append_sps(const SpsNalu& sps);
    void append_pps(const PpsNalu& pps);
    void append_frame(const SpsNalu& sps, const FrameNalu& frame);
    std::span<const std::byte> output() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NaluParser final {
public:
    NaluParser();
    ~NaluParser();

    NaluParser(const NaluParser&) = delete;
    NaluParser& operator=(const NaluParser&) = delete;
    NaluParser(NaluParser&&) noexcept;
    NaluParser& operator=(NaluParser&&) noexcept;

    void open(std::span<const std::byte> buffer);
    bool has_next() const noexcept;
    NaluHeader peek_nalu_header() const;
    SpsNalu read_sps();
    PpsNalu read_pps();
    FrameNalu read_frame(const SpsNalu& sps);
    CustomNalu read_custom_nalu();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class BitstreamEncoder final {
public:
    BitstreamEncoder();
    ~BitstreamEncoder();

    BitstreamEncoder(const BitstreamEncoder&) = delete;
    BitstreamEncoder& operator=(const BitstreamEncoder&) = delete;
    BitstreamEncoder(BitstreamEncoder&&) noexcept;
    BitstreamEncoder& operator=(BitstreamEncoder&&) noexcept;

    void reset();
    std::span<const std::byte> encode(const FrameData& data);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class BitstreamDecoder final {
public:
    BitstreamDecoder();
    ~BitstreamDecoder();

    BitstreamDecoder(const BitstreamDecoder&) = delete;
    BitstreamDecoder& operator=(const BitstreamDecoder&) = delete;
    BitstreamDecoder(BitstreamDecoder&&) noexcept;
    BitstreamDecoder& operator=(BitstreamDecoder&&) noexcept;

    void reset();
    FrameData decode(std::span<const std::byte> access_unit);
    const SpsNalu& sps() const noexcept;
    const PpsNalu& pps() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class AccessUnitReader final {
public:
    explicit AccessUnitReader(
        std::istream& input,
        std::size_t max_access_unit_bytes = kMaxEncodedFramePayloadBytes);

    bool read(std::vector<std::byte>& access_unit);

private:
    std::istream* input_ = nullptr;
    std::size_t max_access_unit_bytes_ = kMaxEncodedFramePayloadBytes;
    std::vector<std::byte> buffer_;
    std::size_t offset_ = 0;

    bool fill_more();
    bool ensure_available(std::size_t count);
    std::size_t find_nalu_end();
};

void write_access_unit(std::ostream& output,
                       std::span<const std::byte> access_unit);

}  // namespace mlvc
