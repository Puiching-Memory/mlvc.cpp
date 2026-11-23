# Codec Pipeline & Bitstream Specification

`mlvc.cpp` re-implements the `microsoft/mlvc` `video/conversion` split-model deployment pipeline (`dmc61sbr_e1d1`) for both `mlvc-psnr-v1` (`dmc61sbr_reglu`) and `mlvc-s-psnr-v1` (`dmc61sbr_mini_reglu`). The inference backend runs fixed-shape FP16 neural graphs while `src/core/` and `src/codec/` own frame preprocessing, DPB/GOP scheduling, scale extraction, rANS entropy coding, and the `mlvc-nalu-v0` bitstream container.

---

## 1. Split-Model Graph Contract

All model dimensions come from `metadata.pb` inside the loaded canonical bundle — nothing hard-codes channel counts or canvas sizes.

### Fixed Graph I/O (640×368 Example for `mlvc-psnr-v1`)

**Encoder (`MLVCEncoder`)**:

```text
Inputs:
  x                [1, 3, 368, 640]   fp16   YUV444 input pixels in [0, 1]
  ref_feature      [1, 256, 46, 80]   fp16   previous encoder DPB feature (bound state)
  q_index_shifted  [1]                int32  q_index + qp_shift[fa_idx]

Outputs:
  feature          [1, 256, 46, 80]   fp16   next encoder DPB feature (bound state)
  z_raw            [1, 128, 3, 5]     fp16   integer-valued quantized hyper-latent
  y_raw_0          [1, 64, 23, 40]    fp16   integer-valued quantized latent half 0
  y_raw_1          [1, 64, 23, 40]    fp16   integer-valued quantized latent half 1
```

**Decoder (`MLVCDecoder`)**:

```text
Inputs:
  z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted

Outputs:
  x_hat            [1, 3, 368, 640]   fp16   reconstructed YUV444 pixels in [0, 1]
  feature          [1, 256, 46, 80]   fp16   next decoder DPB feature (bound state)
```

### Registered Profile Dimensions (`640×368` Canvas)

| Field                                                      | `mlvc-psnr-v1` (`dmc61sbr_reglu`) | `mlvc-s-psnr-v1` (`dmc61sbr_mini_reglu`) |
| ---------------------------------------------------------- | --------------------------------- | ---------------------------------------- |
| `feature_channels`                                         | `256`                             | `96`                                     |
| `latent_channels` (`y`)                                    | `128` (`64` per half)             | `48` (`24` per half)                     |
| `hyperprior_channels` (`z`)                                | `128`                             | `48`                                     |
| `downsample_feature` / `latent` / `hyperprior`             | `8` / `16` / `8`                  | `8` / `16` / `4`                         |
| `y_scale_repeat`                                           | `2`                               | `4`                                      |
| Feature plane (`H × W`)                                    | `46 × 80`                         | `46 × 80`                                |
| Latent `y` plane (`ceil(H/16) × ceil(W/16)`)               | `23 × 40`                         | `23 × 40`                                |
| Hyper-latent `z` plane                                     | `3 × 5`                           | `6 × 10`                                 |
| `pixel_range`                                              | `1.0`                             | `1.0`                                    |
| `qp_num` / `total_qp_num`                                  | `64` / `72`                       | `64` / `72`                              |
| `frame_index_map`                                          | `[0, 1, 0, 2, 0, 2, 0, 2]`        | `[0, 1, 0, 2, 0, 2, 0, 2]`               |
| `qp_shift`                                                 | `[0, 8, 4]`                       | `[0, 8, 4]`                              |
| `iframe_period` / `reset_period` / `disable_feature_reset` | `64` / `null` / `true`            | `64` / `null` / `true`                   |

> **Note**: `mlvc-s-psnr-v1` uses a mini-hyperprior architecture (`y_scale_repeat=4`, `6 × 10` `z` plane) rather than a simple channel-scaled copy of `mlvc-psnr-v1`.

---

## 2. Pre- and Post-Processing (`src/core/yuv.cpp`)

Input and output frames are 8-bit planar YUV420 / I420 (full `Y` plane followed by quarter-resolution `U` and `V` planes; `width` and `height` must be positive even integers; frame byte size = `width * height * 3 / 2`).

- **Encode preprocessing**:
  1. Normalize 8-bit samples by `pixel_range / 255.0` (`pixel_range = 1.0`).
  2. Upsample `U`/`V` to YUV444 via 2× nearest-neighbor replication.
  3. Compute `FramePadding` (`left`, `right`, `top`, `bottom`, `rotated`) to fit the fixed model canvas (for example, `640×360` on a `640×368` canvas adds 8 bottom edge-replicated rows). Portrait inputs may set `rotated = true` (`transpose_flag`) to match a landscape canvas.
  4. Apply edge-replicated padding to `[1, 3, model_height, model_width]` FP16.
- **Decode postprocessing**:
  1. Crop (and un-transpose if `rotated` is set) `x_hat` back to `width × height`.
  2. Downsample `U`/`V` from YUV444 to YUV420 by 2×2 box averaging.
  3. Scale by `255.0 / pixel_range`, round, and clamp to `[0, 255]`.

On both `driver-cubin` and `cpu`, the `BufferedCodecBackend` fast path performs YUV conversion directly inside the backend's persistent frame slots (on GPU kernels for `driver-cubin`, vectorized SIMD for `cpu`).

---

## 3. GOP, DPB & QP Scheduling (`src/codec/pipeline.cpp`)

Matches upstream `FrameLoop` (`video/conversion/_frame_loop.py`) under the registered profile defaults:

- **GOP boundaries**: With `iframe_period = 64`, frame `0`, `64`, `128`, … is an IDR I-frame (`FrameType::kIFrame`); all intervening frames are P-frames (`FrameType::kPFrame`). `reset_period` is unset (`disable_feature_reset = true`).
- **State binding & lockstep DPBs**:
  - Encoder and decoder maintain independent recurrent features (`ref_feature` → `feature`) kept in lockstep across frames.
  - On every I-frame, `reset_state()` zeroes `ref_feature`.
  - In normal execution (`debug_dir` empty), `ModelExecutionConfig::state_bindings` binds output `feature` directly to the next frame's `ref_feature` inside the backend, eliminating host↔device feature transfers.
  - Enabling `--debug-dir` disables state binding so both `ref_feature` and `feature` can be dumped to disk for diagnosis.
- **Shifted QP vs. coded QP**:
  - Neural graph input:
    ```text
    fa_idx          = frame_index_map[(gop_idx + 1) % len(frame_index_map)]
    q_index_shifted = q_index + qp_shift[fa_idx]
    ```
  - Entropy coding and NALU headers always use the **unshifted** `q_index` (validated in `[0, 51]` and `< config.qp_num`).

---

## 4. Dual-Prior Scale Extraction (`src/core/scales.cpp`)

`scales_0` and `scales_1` (Gaussian PMF indices for coding `y_raw_0` / `y_raw_1`) are derived deterministically from `z_raw` symmetrically on both encoder and decoder — they are never emitted by the neural graph.

Reference: `UpsampleScaleDecoder.extract_scales` (`video/conversion/_scale_decoder.py`):

```text
base_channels = latent_channels / y_scale_repeat
base = z_raw[:, :base_channels, :, :]
s = |base|
s = repeat(channels: y_scale_repeat)
    × repeat(height: spatial_repeat)
    × repeat(width:  spatial_repeat)
s = s[:, :, :y_height, :y_width]             # crop to exact y plane
s = clip(s, 0, scale_levels - 1)             # [0, 127]

# Dual-prior checkerboard split:
#   m0 = [[1, 0], [0, 1]],  m1 = [[0, 1], [1, 0]]
#   mask_0 = concat(first_half * m0, second_half * m1)
#   mask_1 = concat(first_half * m1, second_half * m0)
scales_0 = sum(split(mask_0 * s, 2 halves))  # [1, latent_channels/2, y_h, y_w]
scales_1 = sum(split(mask_1 * s, 2 halves))
```

All repeat factors are computed from `metadata.pb`, handling both `mlvc-psnr-v1` (channel repeat `2`, spatial repeat `8`) and `mlvc-s-psnr-v1` (channel repeat `4`, spatial repeat `4`) uniformly.

---

## 5. Entropy Coding (`src/core/entropy.cpp`)

- **Identifier**: `canonical-pmf-v1`
- **Coder**: `msrtc_rans` (`RansByte`, `symbolBits = 16`, `bypassBits = 2`)
- **Gaussian PMF (`gaussian_pmf.pb`)**: `index_space = true` — clipped scale values directly index the quantized Gaussian PMF table (`indices = int32(scales)`, `symbols = int32(y_raw)`).
- **Bit Estimator PMF (`bit_estimator_pmf.pb`)**: One PMF table per `(q_index, z_channel)` pair: `indices = arange(z_channels) + q_index * z_channels`.
- **LIFO stream order** (each frame is an independent rANS stream; encode pushes in reverse of decode):

```text
Encode (push): y_raw_1 (scales_1) -> y_raw_0 (scales_0) -> z_raw (q_index)
Decode (pop):  z_raw   (q_index)  -> y_raw_0 (scales_0) -> y_raw_1 (scales_1) -> EOF
```

---

## 6. Bitstream Container Format (`mlvc-nalu-v0`)

Implemented in [`src/core/bitstream.hpp`](../src/core/bitstream.hpp), [`src/core/bitstream.cpp`](../src/core/bitstream.cpp), and [`src/core/bit_io.cpp`](../src/core/bit_io.cpp), derived from [`microsoft/libmlvc`](https://github.com/microsoft/libmlvc) (MIT License; see [`third_party/libmlvc/LICENSE`](../third_party/libmlvc/LICENSE)).

### Framing & EBSP

```text
access_unit := [SPS_NALU] [PPS_NALU] VCL_FRAME_NALU
nalu        := 0x00 0x00 0x00 0x01 | ebsp
```

- Start codes are 4-byte Annex-B markers (`0x00 0x00 0x00 0x01`).
- Inside EBSP payloads, byte sequences `0x00 0x00 0x00`..`0x03` are escaped with emulation-prevention byte `0x03` (`0x00 0x00 0x03 0x0x`), stripped transparently on decode.
- Every RBSP terminates with a `1` stop bit followed by zero padding to a byte boundary.
- Maximum Access Unit size is bounded by `kMaxEncodedFramePayloadBytes` (256 MiB). Legacy `mlvc-frame-le-v1` containers are rejected.

### NALU Types (`NaluType`)

| Value | Name       | Role                                                                                 |
| ----- | ---------- | ------------------------------------------------------------------------------------ |
| `0`   | `TRAIL_N`  | Non-reference trailing inter frame (accepted by decoder)                             |
| `1`   | `TRAIL_R`  | Reference trailing inter P-frame                                                     |
| `2`   | `TSA_N`    | Non-reference temporal sub-layer access frame (accepted by decoder)                  |
| `3`   | `TSA_R`    | Reference temporal sub-layer access frame                                            |
| `20`  | `IDR_N_LP` | Instantaneous Decoder Refresh keyframe (I-frame); always preceded by `SPS` and `PPS` |
| `33`  | `SPS`      | Sequence Parameter Set                                                               |
| `34`  | `PPS`      | Picture Parameter Set                                                                |

### Parameter Sets & Frame Syntax

- **16-bit NALU Header**: `forbidden_zero (1b)` | `nalu_type (6b)` | `layer_id (6b)` | `temporal_id_plus1 (3b)`.
- **SPS (`33`)**: `mlvc_version` (`major=0`, `minor=1`), `sps_id` (`0..3`), `model_width_div2`, `model_height_div2`, `transpose_flag`, `crop_flag` + `crop_{left,right,top,bottom}_div2`, `max_temporal_layers_minus1` (up to `kMaxTemporalLayers = 2`), `frame_idx_bits_minus8` (`frame_idx_bits` in `[8, 30]`, default `10`).
- **PPS (`34`)**: `pps_id` (`0..3`), `sps_id`, `init_qp_minus26` (default `init_qp = 26`).
- **VCL Frame NALU (`IDR_N_LP` / `TRAIL_*` / `TSA_*`)**:
  ```text
  nalu_header(16b) | pps_id(ue) | qp_delta(se) | ltr_slots(up to 8)
    | [frame_idx, ref_frame_idx, feature_reset_flag] | byte_align
    | rans_entropy_payload | rbsp_stop_bit(1)
  ```
- **QP Validation**: Effective frame QP (`init_qp_minus26 + 26 + qp_delta`) is strictly validated in `[0, 51]` (`kMinBitstreamQp`..`kMaxBitstreamQp`).
- **Auto-detection on decode**: `SPS` carries full model dimensions, crop offsets, and `transpose_flag`. Calling `decode` with `width=0, height=0` (or omitting `--width`/`--height` on the CLI) automatically derives display dimensions from `SPS`; explicit dimensions are checked against `SPS` and fail fast on mismatch.
