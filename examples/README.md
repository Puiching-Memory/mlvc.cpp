# Examples

Sample inputs for the demo (git-ignored, generate locally):

```bash
# Generate a 60-frame 640x360 YUV420 test sequence
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=30 -pix_fmt yuv420p -frames:v 60 test60.yuv
```

MLVC model artifacts: use the official converter (see docs/design.md §1) and
point `--model-dir` at the exported `640x368` directory containing
`MLVCEncoder.onnx`, `MLVCDecoder.onnx`, `gaussian_pmf.json`,
`bit_estimator_pmf.json` and `metadata.json`.
