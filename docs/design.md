# MLVC C++ runtime pipeline — design notes

Reference: microsoft/mlvc `video/conversion` split-model deployment path
(`dmc61sbr_e1d1`, model `dmc61sbr_reglu`, weights `mlvc-psnr-v1`).

This repository re-implements the same pipeline in portable C++: the neural
parts run as exported ONNX graphs, quantization/entropy coding run with
`msrtc_rans`.

---

## 1. Model artifacts

Produced by the official converter:

```
python video/convert.py export \
  --model-version dmc61sbr_reglu \
  --model-type onnx --target-device generic \
  --split-type dmc61sbr_e1d1 \
  --model-width 640 --model-height 368 \
  --output-path <dir>
```

Outputs (per input size, here `640x368`):

| File | Role |
|---|---|
| `MLVCEncoder.onnx` | Encoder graph |
| `MLVCDecoder.onnx` | Decoder graph |
| `gaussian_pmf.json` | Quantized Gaussian PMF tables (y coding) |
| `bit_estimator_pmf.json` | Quantized bit-estimator PMF tables (z coding) |
| `metadata.json` | Model params (dims, GOP, scales, ...) |

### ONNX IO (verified, `opset 18`, FP16 weights)

Encoder:

```
x             [1,3,368,640]  fp16   input pixels, range [0,1]
ref_feature   [1,256,46,80] fp16   reference feature from DPB
q_index_shifted [1]         int32  q_index + qp_shift[fa_idx]
----------------------------------------------
feature       [1,256,46,80] fp16   new DPB feature
z_raw         [1,128,3,5]   fp16   quantized latent z (integer-valued)
y_raw_0       [1,64,23,40]  fp16   quantized residual part 0
y_raw_1       [1,64,23,40]  fp16   quantized residual part 1
```

Decoder (same latent shapes as encoder outputs):

```
z_raw, y_raw_0, y_raw_1, ref_feature, q_index_shifted
  -> x_hat [1,3,368,640] (reconstruction, [0,1])
  -> feature [1,256,46,80]
```

Key dimensions for `640x368` (from `metadata.json`):

- `feature_channels=256`, `latent_channels=128`, `hyperprior_channels=128`
- `downsample_feature=8`, `downsample_latent=16`, `downsample_hyperprior=8`
- `y_scale_repeat=2`, `pixel_range=1.0`, `frame_index_map=[0,1,0,2,0,2,0,2]`,
  `qp_shift=[0,8,4]`, `iframe_period=64`, `reset_period=null`
- Latent (y) plane: `ceil(640/16) x ceil(368/16)` = `40 x 23`
- Hyper latent (z) plane: `ceil(640/128) x ceil(368/128)` = `5 x 3`

## 2. Scale extraction (`extract_scales`)

`scales_0/scales_1` (PMF indices for y coding) are re-derived from `z_raw`
*symmetrically on encode and decode* — they are NOT produced by the encoder
graph. Python reference: `UpsampleScaleDecoder.extract_scales`
(`video/conversion/_scale_decoder.py`).

```
base = z_raw[:, :64, :, :]             # first latent_channels/y_scale_repeat chans
s = |base|
s = repeat(channels: 2) x repeat(spatial: 8) x repeat(spatial: 8)  -> [1,128,24,40]
s = s[:, :, :23, :40]                  # crop to y plane
s = clip(s, 0, 127)                    # scale_levels-1
# mask-based split (dual prior):
#   mask_0 = concat(m*m0, m*m1), m0 = [[1,0],[0,1]] checkerboard,
#   mask_1 = concat(m*m1, m*m0), m1 = [[0,1],[1,0]]
scales_0 = sum(split(mask_0 * s, chans))   # [1,64,23,40]
scales_1 = sum(split(mask_1 * s, chans))
```

## 3. Entropy coding (`msrtc_rans`)

- Variant: `RansByte`, `symbolBits=16`, `bypassBits=2`
  (matches `_coder.py`: `EntropyEncoder(..., symbolBits=16, bypassBits=2)`)
- Gaussian PMF: `index_space=True` → scale values *are* PMF indices.
  `indices = int32(scales)`, `symbols = int32(y_raw)`.
- Bit estimator (z): `indices = arange(128) + q_index * 128` (one table per q
  index, `qp_num=72`, `channels=128`).
- Message order (**encode** pushes, **decode** pops in reverse):

```
encode: push(y_raw_1, scales_1) -> push(y_raw_0, scales_0) -> push(z_raw, q_index)
decode: pop(z_raw, q_index) -> pop(y_raw_0, scales_0) -> pop(y_raw_1, scales_1) [EOF]
```

Each frame is an independent rANS stream.

## 4. Bitstream container format

Compatible with the official helpers (`save_mlvc_bitstreams` /
`read_mlvc_bitstreams` in `video/conversion/utils.py`):

```
frame  := header (8 bytes LE) + payload
header := int32 q_index | uint32 payload_size
```

## 5. Frame loop / DPB (GOP)

Reference: `FrameLoop` (`video/conversion/_frame_loop.py`) with model defaults:

- `iframe_period=64`: frame 0 (and every 64th) is an I-frame; clears both DPBs.
- `reset_period=null` + `disable_feature_reset=true`: no feature resets.
- `ltr_period=null`: no LTR frames. All other frames are P-frames using the
  previous frame's feature as reference.
- I-frame ref: zero `ref_feature [1,256,46,80]`; P-frame ref: previous feature.
- Encoder DPB stores the *encoder* feature; decoder DPB stores the *decoder*
  feature (kept in lockstep).
- `q_index_shifted = q_index + qp_shift[frame_index_map[(gop_idx+1) % 8]]`
- Entropy coding always uses the raw `q_index` (not shifted).

## 6. Pre/post-processing

- Input: YUV420 8-bit → `/255.0` → YUV444 (UV nearest-neighbor x2) →
  pad to `640x368` (EDGE padding, BOTTOM-RIGHT; `640x360` gets 8 rows of
  edge-replicated samples at the bottom).
- Output: unpad (crop), `/1.0` (already in [0,1]), YUV444 → YUV420
  (2x2 average downsample) → x255 → bytes.
- `pixel_range=1.0`, so the graph works directly in [0,1].

## 7. Inference backends

Three interchangeable backends implement `mlvc::InferenceBackend`
(`include/mlvc/backend.hpp`), selected at runtime via `--backend` and compiled
in via `MLVC_WITH_*` CMake options:

| Backend | Model artifacts | Execution | Notes |
|---|---|---|---|
| `onnxruntime` | `MLVC{Encoder,Decoder}.onnx` | CPU / CUDA EP | Reference path, mirrors the official deployment |
| `libtorch` | `MLVC{Encoder,Decoder}.ts` | CPU / CUDA | Needs TorchScript exports (converter emits ONNX only); libtorch >= 2.13 requires C++20 |
| `tensorrt` | `MLVC{Encoder,Decoder}.onnx` | CUDA | Engines built at load time via nvonnxparser (TensorRT 10 API, fp16 enabled) |

Host-side tensors crossing the backend boundary are fp32; the MLVC graphs are
fp16, so each backend adds dtype conversion when the pipeline lands.

## 8. Implementation plan

- [ ] `src/pmf.{h,cpp}` — PMF JSON loading + rANS encoder/decoder wrappers
- [ ] `src/scales.{h,cpp}` — dual-prior scale extraction + masks
- [ ] `src/yuv.{h,cpp}` — YUV420 IO, padding, color conversion
- [ ] `src/pipeline.{h,cpp}` — backend sessions, DPB, encode/decode loop
- [ ] `src/main.cpp` — CLI (`encode` / `decode` / `transcode` / `bench`)
- [ ] Backend parity test: same inputs → bit-exact latents across backends
- [ ] Cross-validation against python reference (`validate_conversion` outputs)
