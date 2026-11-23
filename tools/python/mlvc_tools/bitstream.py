# Derived from microsoft/libmlvc (https://github.com/microsoft/libmlvc)
# Copyright (c) 2026 Microsoft Corporation
# Licensed under the MIT License. See third_party/libmlvc/LICENSE.

"""MLVC v0.x NALU Annex-B bitstream builder and parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable


START_CODE = b"\x00\x00\x00\x01"
SUPPORTED_MAJOR_VERSION = 0
DEFAULT_MINOR_VERSION = 1
MAX_TEMPORAL_LAYERS = 2
MAX_LTR_SLOTS = 8
MIN_BITSTREAM_QP = 0
MAX_BITSTREAM_QP = 51
DEFAULT_INIT_QP = 26
MIN_FRAME_IDX_BITS = 8
MAX_FRAME_IDX_BITS = 30
DEFAULT_FRAME_IDX_BITS = 10
MAX_SPS_ID = 3
MAX_PPS_ID = 3

_NALU_TYPE_BITS = 6
_LAYER_ID_BITS = 6
_TEMPORAL_ID_PLUS1_BITS = 3
_VERSION_BITS = 8
_MODEL_SIZE_BITS = 12
_MAX_TEMPORAL_LAYERS_MINUS1_BITS = 3
_LTR_NUM_SLOTS_BITS = 4


class BitstreamError(ValueError):
    """Raised when an MLVC NALU stream violates the v0.x specification."""


class NaluType(IntEnum):
    TRAIL_N = 0
    TRAIL_R = 1
    TSA_N = 2
    TSA_R = 3
    IDR_N_LP = 20
    SPS = 33
    PPS = 34


class FrameType(str, Enum):
    I_FRAME = "I"
    P_FRAME = "P"
    LTR_RECOVERY = "LTR_RECOVERY"


@dataclass(frozen=True)
class CropOffsets:
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            self.left == 0
            and self.right == 0
            and self.top == 0
            and self.bottom == 0
        )


@dataclass(frozen=True)
class NaluHeader:
    nalu_type: int
    layer_id: int = 0
    temporal_id_plus1: int = 1
    forbidden_zero: int = 0


@dataclass(frozen=True)
class SpsNalu:
    nalu_header: NaluHeader = field(default_factory=lambda: NaluHeader(NaluType.SPS))
    mlvc_version_major: int = 0
    mlvc_version_minor: int = 0
    sps_id: int = -1
    model_width_div2: int = 0
    model_height_div2: int = 0
    transpose_flag: bool = False
    crop_flag: bool = False
    crop_left_div2: int = 0
    crop_right_div2: int = 0
    crop_top_div2: int = 0
    crop_bottom_div2: int = 0
    max_temporal_layers_minus1: int = 0
    frame_idx_bits_minus8: int = 0


@dataclass(frozen=True)
class PpsNalu:
    nalu_header: NaluHeader = field(default_factory=lambda: NaluHeader(NaluType.PPS))
    pps_id: int = -1
    sps_id: int = -2
    init_qp_minus26: int = 0


@dataclass(frozen=True)
class FrameHeader:
    pps_id: int = 0
    qp_delta: int = 0
    ltr_slots: tuple[int | None, ...] = (None,) * MAX_LTR_SLOTS
    frame_idx: int = 0
    ref_frame_idx: int = 0
    feature_reset_flag: bool = False


@dataclass(frozen=True)
class FrameNalu:
    nalu_header: NaluHeader
    frame_header: FrameHeader
    payload: bytes


@dataclass(frozen=True)
class FrameData:
    mlvc_version: tuple[int, int] = (
        SUPPORTED_MAJOR_VERSION,
        DEFAULT_MINOR_VERSION,
    )
    model_width: int = 0
    model_height: int = 0
    crop_offsets: CropOffsets = field(default_factory=CropOffsets)
    transpose_flag: bool = False
    frame_idx_bits: int = DEFAULT_FRAME_IDX_BITS
    temporal_id: int = 0
    max_temporal_layers: int = 1
    frame_type: FrameType = FrameType.I_FRAME
    qp: int = 0
    feature_reset_flag: bool = False
    cur_frame_idx: int = 0
    ref_frame_idx: int = 0
    ltr_slots: tuple[int | None, ...] = (None,) * MAX_LTR_SLOTS
    payload: bytes = b""

    @property
    def display_width(self) -> int:
        cropped_width = self.model_width - self.crop_offsets.left - self.crop_offsets.right
        cropped_height = self.model_height - self.crop_offsets.top - self.crop_offsets.bottom
        return cropped_height if self.transpose_flag else cropped_width

    @property
    def display_height(self) -> int:
        cropped_width = self.model_width - self.crop_offsets.left - self.crop_offsets.right
        cropped_height = self.model_height - self.crop_offsets.top - self.crop_offsets.bottom
        return cropped_width if self.transpose_flag else cropped_height


def frame_padding(
    width: int, height: int, model_width: int, model_height: int
) -> tuple[CropOffsets, bool]:
    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0:
        raise ValueError("input width and height must be positive even integers")
    if (
        model_width <= 0
        or model_height <= 0
        or model_width % 2 != 0
        or model_height % 2 != 0
    ):
        raise ValueError("model width and height must be positive even integers")
    native = width <= model_width and height <= model_height
    rotated = height <= model_width and width <= model_height
    if not native and not rotated:
        raise ValueError("input frame is larger than the model dimensions")
    active_width = width if native else height
    active_height = height if native else width
    return (
        CropOffsets(
            left=0,
            right=model_width - active_width,
            top=0,
            bottom=model_height - active_height,
        ),
        not native,
    )


def add_emulation_prevention_bytes(rbsp: bytes) -> bytes:
    out = bytearray()
    for index, byte in enumerate(rbsp):
        if index >= 2 and len(out) >= 2 and out[-2] == 0 and out[-1] == 0 and byte <= 3:
            out.append(3)
        out.append(byte)
    return bytes(out)


def remove_emulation_prevention_bytes(ebsp: bytes) -> bytes:
    out = bytearray()
    for index, byte in enumerate(ebsp):
        if index >= 2 and ebsp[index - 2] == 0 and ebsp[index - 1] == 0 and byte == 3:
            continue
        out.append(byte)
    return bytes(out)


class BitWriter:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._bit_pos = 0

    def reset(self) -> None:
        self._buf.clear()
        self._bit_pos = 0

    def write_bit(self, value: bool) -> None:
        byte_pos = self._bit_pos >> 3
        while byte_pos >= len(self._buf):
            self._buf.append(0)
        mask = 1 << (7 - (self._bit_pos & 7))
        if value:
            self._buf[byte_pos] |= mask
        else:
            self._buf[byte_pos] &= ~mask & 0xFF
        self._bit_pos += 1

    def write_uint(self, value: int, num_bits: int) -> None:
        if num_bits < 0 or num_bits > 30:
            raise BitstreamError("bit width out of range")
        if value < 0 or (value >> num_bits) != 0:
            raise BitstreamError(f"value {value} does not fit in {num_bits} bits")
        for shift in range(num_bits - 1, -1, -1):
            self.write_bit(((value >> shift) & 1) != 0)

    def write_ue(self, value: int) -> None:
        if value < 0 or value >= 0x7FFFFFFF:
            raise BitstreamError(f"ue(v) value out of range: {value}")
        code_num = value + 1
        num_bits = code_num.bit_length()
        self.write_uint(0, num_bits - 1)
        self.write_uint(code_num, num_bits)

    def write_se(self, value: int) -> None:
        if value <= 0:
            self.write_ue(-2 * value)
        else:
            self.write_ue(2 * value - 1)

    def write_byte_padding(self) -> None:
        padding = (8 - (self._bit_pos & 7)) & 7
        if padding:
            self.write_uint(0, padding)

    def write_stop_bit(self) -> None:
        self.write_bit(True)
        self.write_byte_padding()

    def write_bytes(self, data: bytes) -> None:
        self.write_byte_padding()
        self._buf.extend(data)
        self._bit_pos += len(data) * 8

    def to_bytes(self) -> bytes:
        return bytes(self._buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bit_pos = 0

    def read_bit(self) -> bool:
        if self._bit_pos >= len(self._data) * 8:
            raise BitstreamError("bitstream read out of bounds")
        byte = self._data[self._bit_pos >> 3]
        bit = (byte >> (7 - (self._bit_pos & 7))) & 1
        self._bit_pos += 1
        return bit != 0

    def read_uint(self, num_bits: int) -> int:
        if num_bits < 0 or num_bits > 30:
            raise BitstreamError("bitstream integer bit width out of range")
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | (1 if self.read_bit() else 0)
        return value

    def read_ue(self) -> int:
        num_bits = 1
        while not self.read_bit():
            num_bits += 1
            if num_bits >= 32:
                raise BitstreamError("Exp-Golomb code exceeds 31-bit integer range")
        self._bit_pos -= 1
        return self.read_uint(num_bits) - 1

    def read_se(self) -> int:
        ue = self.read_ue()
        return -(ue // 2) if (ue % 2 == 0) else ((ue + 1) // 2)

    def read_byte_padding(self) -> None:
        padding = (8 - (self._bit_pos & 7)) & 7
        if padding and self.read_uint(padding) != 0:
            raise BitstreamError("non-zero RBSP byte alignment padding")

    def read_stop_bit(self) -> None:
        if not self.read_bit():
            raise BitstreamError("missing RBSP stop bit")
        self.read_byte_padding()

    def read_bytes(self) -> bytes:
        self.read_byte_padding()
        total_bits = len(self._data) * 8
        if self._bit_pos >= total_bits:
            raise BitstreamError("bitstream payload read out of bounds")
        num_bytes = (total_bits - self._bit_pos - 1) // 8
        start = self._bit_pos >> 3
        self._bit_pos += num_bytes * 8
        return self._data[start : start + num_bytes]


def _write_nalu_header(writer: BitWriter, header: NaluHeader) -> None:
    if header.forbidden_zero != 0:
        raise BitstreamError("NALU forbidden_zero bit must be 0")
    writer.write_bit(False)
    writer.write_uint(int(header.nalu_type), _NALU_TYPE_BITS)
    writer.write_uint(header.layer_id, _LAYER_ID_BITS)
    writer.write_uint(header.temporal_id_plus1, _TEMPORAL_ID_PLUS1_BITS)


def _parse_nalu_header(reader: BitReader) -> NaluHeader:
    if reader.read_bit():
        raise BitstreamError("forbidden_zero bit is set in NALU header")
    return NaluHeader(
        nalu_type=reader.read_uint(_NALU_TYPE_BITS),
        layer_id=reader.read_uint(_LAYER_ID_BITS),
        temporal_id_plus1=reader.read_uint(_TEMPORAL_ID_PLUS1_BITS),
    )


def _validate_frame_data(data: FrameData) -> None:
    if data.mlvc_version[0] != SUPPORTED_MAJOR_VERSION:
        raise BitstreamError("unsupported MLVC bitstream major version")
    if data.mlvc_version[1] < 0 or data.mlvc_version[1] >= 256:
        raise BitstreamError("MLVC bitstream minor version out of range")
    if (
        data.model_width <= 0
        or data.model_height <= 0
        or data.model_width % 2 != 0
        or data.model_height % 2 != 0
        or data.model_width >= 8192
        or data.model_height >= 8192
    ):
        raise BitstreamError("invalid MLVC model dimensions for bitstream")
    for offset in (
        data.crop_offsets.left,
        data.crop_offsets.right,
        data.crop_offsets.top,
        data.crop_offsets.bottom,
    ):
        if offset < 0 or offset % 2 != 0:
            raise BitstreamError("crop offsets must be non-negative even integers")
    if (
        data.crop_offsets.left + data.crop_offsets.right >= data.model_width
        or data.crop_offsets.top + data.crop_offsets.bottom >= data.model_height
    ):
        raise BitstreamError("crop offsets exceed model dimensions")
    if not (1 <= data.max_temporal_layers <= MAX_TEMPORAL_LAYERS):
        raise BitstreamError("max_temporal_layers out of range")
    if data.temporal_id < 0 or data.temporal_id >= data.max_temporal_layers:
        raise BitstreamError("temporal_id out of range")
    if not (MIN_FRAME_IDX_BITS <= data.frame_idx_bits <= MAX_FRAME_IDX_BITS):
        raise BitstreamError("frame_idx_bits out of range")
    if not (MIN_BITSTREAM_QP <= data.qp <= MAX_BITSTREAM_QP):
        raise BitstreamError("frame QP must be in [0, 51]")
    if len(data.ltr_slots) != MAX_LTR_SLOTS:
        raise BitstreamError("expected 8 LTR slots")
    max_idx = 1 << data.frame_idx_bits
    if not (0 <= data.cur_frame_idx < max_idx) or not (0 <= data.ref_frame_idx < max_idx):
        raise BitstreamError("frame index exceeds frame_idx_bits")
    for slot in data.ltr_slots:
        if slot is not None and not (0 <= slot < max_idx):
            raise BitstreamError("LTR slot frame index exceeds frame_idx_bits")
    if not data.payload:
        raise BitstreamError("frame payload must not be empty")


class NaluBuilder:
    def __init__(self) -> None:
        self._output = bytearray()
        self._writer = BitWriter()

    def reset(self) -> None:
        self._output.clear()
        self._writer.reset()

    def _flush_nalu(self) -> None:
        self._writer.write_stop_bit()
        self._output.extend(START_CODE)
        self._output.extend(add_emulation_prevention_bytes(self._writer.to_bytes()))
        self._writer.reset()

    def append_sps(self, sps: SpsNalu) -> None:
        self._writer.reset()
        _write_nalu_header(self._writer, sps.nalu_header)
        self._writer.write_uint(sps.mlvc_version_major, _VERSION_BITS)
        self._writer.write_uint(sps.mlvc_version_minor, _VERSION_BITS)
        self._writer.write_ue(sps.sps_id)
        self._writer.write_uint(sps.model_width_div2, _MODEL_SIZE_BITS)
        self._writer.write_uint(sps.model_height_div2, _MODEL_SIZE_BITS)
        self._writer.write_bit(sps.transpose_flag)
        self._writer.write_bit(sps.crop_flag)
        if sps.crop_flag:
            self._writer.write_ue(sps.crop_left_div2)
            self._writer.write_ue(sps.crop_right_div2)
            self._writer.write_ue(sps.crop_top_div2)
            self._writer.write_ue(sps.crop_bottom_div2)
        self._writer.write_uint(
            sps.max_temporal_layers_minus1, _MAX_TEMPORAL_LAYERS_MINUS1_BITS
        )
        self._writer.write_ue(sps.frame_idx_bits_minus8)
        self._flush_nalu()

    def append_pps(self, pps: PpsNalu) -> None:
        self._writer.reset()
        _write_nalu_header(self._writer, pps.nalu_header)
        self._writer.write_ue(pps.pps_id)
        self._writer.write_ue(pps.sps_id)
        self._writer.write_se(pps.init_qp_minus26)
        self._flush_nalu()

    def append_frame(self, sps: SpsNalu, frame: FrameNalu) -> None:
        self._writer.reset()
        _write_nalu_header(self._writer, frame.nalu_header)
        self._writer.write_ue(frame.frame_header.pps_id)
        self._writer.write_se(frame.frame_header.qp_delta)

        frame_idx_bits = sps.frame_idx_bits_minus8 + MIN_FRAME_IDX_BITS
        num_ltr_slots = 0
        for idx, slot in enumerate(frame.frame_header.ltr_slots):
            if slot is not None:
                num_ltr_slots = idx + 1
        self._writer.write_uint(num_ltr_slots, _LTR_NUM_SLOTS_BITS)
        for idx in range(num_ltr_slots):
            slot = frame.frame_header.ltr_slots[idx]
            self._writer.write_bit(slot is None)
            if slot is not None:
                self._writer.write_uint(slot, frame_idx_bits)

        if frame.nalu_header.nalu_type != NaluType.IDR_N_LP:
            self._writer.write_uint(frame.frame_header.frame_idx, frame_idx_bits)
            self._writer.write_uint(frame.frame_header.ref_frame_idx, frame_idx_bits)
            self._writer.write_bit(frame.frame_header.feature_reset_flag)

        self._writer.write_bytes(frame.payload)
        self._flush_nalu()

    def output(self) -> bytes:
        return bytes(self._output)


class BitstreamEncoder:
    def __init__(self) -> None:
        self._builder = NaluBuilder()
        self.reset()

    def reset(self) -> None:
        self._sps: SpsNalu | None = None
        self._pps: PpsNalu | None = None
        self._builder.reset()

    def encode(self, data: FrameData) -> bytes:
        _validate_frame_data(data)
        self._builder.reset()

        if data.frame_type == FrameType.I_FRAME:
            prev_sps_id = self._sps.sps_id if self._sps is not None else -1
            new_sps = SpsNalu(
                nalu_header=NaluHeader(NaluType.SPS),
                mlvc_version_major=data.mlvc_version[0],
                mlvc_version_minor=data.mlvc_version[1],
                sps_id=prev_sps_id,
                model_width_div2=data.model_width // 2,
                model_height_div2=data.model_height // 2,
                transpose_flag=data.transpose_flag,
                crop_flag=not data.crop_offsets.is_empty,
                crop_left_div2=data.crop_offsets.left // 2,
                crop_right_div2=data.crop_offsets.right // 2,
                crop_top_div2=data.crop_offsets.top // 2,
                crop_bottom_div2=data.crop_offsets.bottom // 2,
                max_temporal_layers_minus1=data.max_temporal_layers - 1,
                frame_idx_bits_minus8=data.frame_idx_bits - MIN_FRAME_IDX_BITS,
            )
            if self._sps is None or self._pps is None or new_sps != self._sps:
                new_sps_id = (prev_sps_id + 1) % MAX_SPS_ID
                new_pps_id = (
                    (self._pps.pps_id if self._pps is not None else -1) + 1
                ) % MAX_PPS_ID
                self._sps = SpsNalu(
                    nalu_header=new_sps.nalu_header,
                    mlvc_version_major=new_sps.mlvc_version_major,
                    mlvc_version_minor=new_sps.mlvc_version_minor,
                    sps_id=new_sps_id,
                    model_width_div2=new_sps.model_width_div2,
                    model_height_div2=new_sps.model_height_div2,
                    transpose_flag=new_sps.transpose_flag,
                    crop_flag=new_sps.crop_flag,
                    crop_left_div2=new_sps.crop_left_div2,
                    crop_right_div2=new_sps.crop_right_div2,
                    crop_top_div2=new_sps.crop_top_div2,
                    crop_bottom_div2=new_sps.crop_bottom_div2,
                    max_temporal_layers_minus1=new_sps.max_temporal_layers_minus1,
                    frame_idx_bits_minus8=new_sps.frame_idx_bits_minus8,
                )
                self._pps = PpsNalu(
                    nalu_header=NaluHeader(NaluType.PPS),
                    pps_id=new_pps_id,
                    sps_id=new_sps_id,
                    init_qp_minus26=data.qp - DEFAULT_INIT_QP,
                )
            assert self._sps is not None and self._pps is not None
            self._builder.append_sps(self._sps)
            self._builder.append_pps(self._pps)
        elif self._sps is None or self._pps is None:
            raise BitstreamError("cannot encode non-IDR frame before an IDR frame")

        if data.frame_type == FrameType.I_FRAME:
            nalu_type = NaluType.IDR_N_LP
        elif data.frame_type in (FrameType.P_FRAME, FrameType.LTR_RECOVERY):
            nalu_type = NaluType.TRAIL_R if data.temporal_id == 0 else NaluType.TSA_R
        else:
            raise BitstreamError("unsupported frame type")

        assert self._sps is not None and self._pps is not None
        frame_nalu = FrameNalu(
            nalu_header=NaluHeader(
                nalu_type=nalu_type,
                layer_id=0,
                temporal_id_plus1=data.temporal_id + 1,
            ),
            frame_header=FrameHeader(
                pps_id=self._pps.pps_id,
                qp_delta=data.qp - (self._pps.init_qp_minus26 + DEFAULT_INIT_QP),
                ltr_slots=tuple(data.ltr_slots),
                frame_idx=data.cur_frame_idx,
                ref_frame_idx=data.ref_frame_idx,
                feature_reset_flag=data.feature_reset_flag,
            ),
            payload=data.payload,
        )
        self._builder.append_frame(self._sps, frame_nalu)
        return self._builder.output()


def _split_ebsp_nalus(access_unit: bytes) -> list[bytes]:
    if not access_unit:
        return []
    if not access_unit.startswith(START_CODE):
        raise BitstreamError("missing NALU start code in MLVC access unit")
    nalus: list[bytes] = []
    pos = len(START_CODE)
    while pos <= len(access_unit):
        next_pos = access_unit.find(START_CODE, pos)
        if next_pos == -1:
            nalus.append(access_unit[pos:])
            break
        nalus.append(access_unit[pos:next_pos])
        pos = next_pos + len(START_CODE)
    return nalus


class BitstreamDecoder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sps: SpsNalu | None = None
        self._pps: PpsNalu | None = None

    @property
    def sps(self) -> SpsNalu | None:
        return self._sps

    @property
    def pps(self) -> PpsNalu | None:
        return self._pps

    def decode(self, access_unit: bytes) -> FrameData:
        ebsp_nalus = _split_ebsp_nalus(access_unit)
        frame_data: FrameData | None = None
        for index, ebsp in enumerate(ebsp_nalus):
            reader = BitReader(remove_emulation_prevention_bytes(ebsp))
            header = _parse_nalu_header(reader)
            if header.nalu_type == NaluType.SPS:
                major = reader.read_uint(_VERSION_BITS)
                if major != SUPPORTED_MAJOR_VERSION:
                    raise BitstreamError("unsupported MLVC bitstream major version")
                minor = reader.read_uint(_VERSION_BITS)
                sps_id = reader.read_ue()
                w2 = reader.read_uint(_MODEL_SIZE_BITS)
                h2 = reader.read_uint(_MODEL_SIZE_BITS)
                transpose = reader.read_bit()
                crop_flag = reader.read_bit()
                if crop_flag:
                    cl2 = reader.read_ue()
                    cr2 = reader.read_ue()
                    ct2 = reader.read_ue()
                    cb2 = reader.read_ue()
                else:
                    cl2 = cr2 = ct2 = cb2 = 0
                if cl2 + cr2 >= w2 or ct2 + cb2 >= h2:
                    raise BitstreamError("invalid SPS geometry or crop offsets")
                max_tl_m1 = reader.read_uint(_MAX_TEMPORAL_LAYERS_MINUS1_BITS)
                if max_tl_m1 >= MAX_TEMPORAL_LAYERS:
                    raise BitstreamError("SPS max_temporal_layers exceeds supported limit")
                fidx_m8 = reader.read_ue()
                if fidx_m8 > MAX_FRAME_IDX_BITS - MIN_FRAME_IDX_BITS:
                    raise BitstreamError("SPS frame_idx_bits exceeds supported limit")
                reader.read_stop_bit()
                self._sps = SpsNalu(
                    nalu_header=header,
                    mlvc_version_major=major,
                    mlvc_version_minor=minor,
                    sps_id=sps_id,
                    model_width_div2=w2,
                    model_height_div2=h2,
                    transpose_flag=transpose,
                    crop_flag=crop_flag,
                    crop_left_div2=cl2,
                    crop_right_div2=cr2,
                    crop_top_div2=ct2,
                    crop_bottom_div2=cb2,
                    max_temporal_layers_minus1=max_tl_m1,
                    frame_idx_bits_minus8=fidx_m8,
                )
            elif header.nalu_type == NaluType.PPS:
                pps_id = reader.read_ue()
                sps_id = reader.read_ue()
                init_qp_minus26 = reader.read_se()
                if not (
                    MIN_BITSTREAM_QP - DEFAULT_INIT_QP
                    <= init_qp_minus26
                    <= MAX_BITSTREAM_QP - DEFAULT_INIT_QP
                ):
                    raise BitstreamError("PPS init_qp_minus26 outside [0, 51]")
                reader.read_stop_bit()
                if self._sps is None or sps_id != self._sps.sps_id:
                    raise BitstreamError("PPS references unknown or inactive SPS")
                self._pps = PpsNalu(
                    nalu_header=header,
                    pps_id=pps_id,
                    sps_id=sps_id,
                    init_qp_minus26=init_qp_minus26,
                )
            elif header.nalu_type in (
                NaluType.IDR_N_LP,
                NaluType.TRAIL_R,
                NaluType.TSA_R,
                NaluType.TRAIL_N,
                NaluType.TSA_N,
            ):
                if self._sps is None or self._pps is None:
                    raise BitstreamError("frame header has no active SPS/PPS")
                if index != len(ebsp_nalus) - 1:
                    raise BitstreamError("extra NALUs after frame NALU are not supported")
                pps_id = reader.read_ue()
                if pps_id != self._pps.pps_id or self._pps.sps_id != self._sps.sps_id:
                    raise BitstreamError("frame header references unknown or inactive PPS")
                qp_delta = reader.read_se()
                frame_idx_bits = self._sps.frame_idx_bits_minus8 + MIN_FRAME_IDX_BITS
                num_ltr_slots = reader.read_uint(_LTR_NUM_SLOTS_BITS)
                if num_ltr_slots > MAX_LTR_SLOTS:
                    raise BitstreamError("num_ltr_slots exceeds maximum")
                ltr_slots: list[int | None] = [None] * MAX_LTR_SLOTS
                for slot_idx in range(num_ltr_slots):
                    if not reader.read_bit():
                        ltr_slots[slot_idx] = reader.read_uint(frame_idx_bits)
                if header.nalu_type != NaluType.IDR_N_LP:
                    cur_frame_idx = reader.read_uint(frame_idx_bits)
                    ref_frame_idx = reader.read_uint(frame_idx_bits)
                    feature_reset_flag = reader.read_bit()
                else:
                    cur_frame_idx = 0
                    ref_frame_idx = 0
                    feature_reset_flag = False
                payload = reader.read_bytes()
                reader.read_stop_bit()

                if header.temporal_id_plus1 == 0:
                    raise BitstreamError("temporal_id_plus1 must be positive")
                temporal_id = header.temporal_id_plus1 - 1
                max_temporal_layers = self._sps.max_temporal_layers_minus1 + 1
                if temporal_id >= max_temporal_layers:
                    raise BitstreamError("frame temporal_id exceeds SPS limit")
                qp = qp_delta + self._pps.init_qp_minus26 + DEFAULT_INIT_QP
                if not (MIN_BITSTREAM_QP <= qp <= MAX_BITSTREAM_QP):
                    raise BitstreamError("decoded frame QP is outside [0, 51]")

                if header.nalu_type == NaluType.IDR_N_LP:
                    frame_type = FrameType.I_FRAME
                elif temporal_id == 0 and (cur_frame_idx - ref_frame_idx) > 2:
                    frame_type = FrameType.LTR_RECOVERY
                else:
                    frame_type = FrameType.P_FRAME

                frame_data = FrameData(
                    mlvc_version=(
                        self._sps.mlvc_version_major,
                        self._sps.mlvc_version_minor,
                    ),
                    model_width=2 * self._sps.model_width_div2,
                    model_height=2 * self._sps.model_height_div2,
                    crop_offsets=CropOffsets(
                        left=2 * self._sps.crop_left_div2,
                        right=2 * self._sps.crop_right_div2,
                        top=2 * self._sps.crop_top_div2,
                        bottom=2 * self._sps.crop_bottom_div2,
                    ),
                    transpose_flag=self._sps.transpose_flag,
                    frame_idx_bits=frame_idx_bits,
                    temporal_id=temporal_id,
                    max_temporal_layers=max_temporal_layers,
                    frame_type=frame_type,
                    qp=qp,
                    feature_reset_flag=feature_reset_flag,
                    cur_frame_idx=cur_frame_idx,
                    ref_frame_idx=ref_frame_idx,
                    ltr_slots=tuple(ltr_slots),
                    payload=payload,
                )
            else:
                _ = reader.read_bytes()

        if frame_data is None:
            raise BitstreamError("incomplete MLVC access unit without frame NALU")
        return frame_data


def split_access_units(data: bytes) -> list[bytes]:
    if not data:
        return []
    if not data.startswith(START_CODE):
        raise BitstreamError("missing NALU start code in MLVC bitstream")
    access_units: list[bytes] = []
    au_start = 0
    pos = len(START_CODE)
    while pos <= len(data):
        if pos == len(data):
            raise BitstreamError("truncated MLVC NALU header at end of stream")
        nalu_type = (data[pos] >> 1) & 0x3F
        next_pos = data.find(START_CODE, pos)
        nalu_end = len(data) if next_pos == -1 else next_pos
        if nalu_type < 32:
            access_units.append(data[au_start:nalu_end])
            au_start = nalu_end
        elif next_pos == -1:
            raise BitstreamError("truncated MLVC access unit before VCL frame NALU")
        if next_pos == -1:
            break
        pos = next_pos + len(START_CODE)
    return access_units


def decode_bitstream_frames(data: bytes) -> Iterable[tuple[bytes, FrameData]]:
    decoder = BitstreamDecoder()
    for access_unit in split_access_units(data):
        yield access_unit, decoder.decode(access_unit)
