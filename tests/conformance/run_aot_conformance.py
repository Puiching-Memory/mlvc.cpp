#!/usr/bin/env python3
"""Check serialized checkpoint AOT graphs against official eager CPU execution.

This validates capture/lowering separately from CUDA arithmetic. GPU codec
acceptance still requires run_codec_conformance.py.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/unit"))
from model_wire import read_metadata
from model_source import example_inputs, load_model, load_profiles
from aot_tests import execute_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models/canonical")
    parser.add_argument("--q-indices", type=int, nargs="+", default=[0, 21, 63])
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.manual_seed(123)
    results = []
    for profile in load_profiles():
        bundles = list((args.model_root / profile).glob("*/model_bundle.pb"))
        if len(bundles) != 1:
            raise ValueError(f"expected one canonical bundle for {profile}")
        bundle = bundles[0].parent
        metadata = read_metadata(bundle)
        shape = metadata["params"]["split_model_params"]
        width, height = shape["model_width"], shape["model_height"]
        full, split, _, _ = load_model(profile, args.checkpoint_root / f"{profile}.ckpt", width, height)
        for part, wrapper in split.model_parts.items():
            inputs = list(example_inputs(part.value, full.model_params, width, height))
            for q in args.q_indices:
                if q < 0 or q >= full.model_params.total_qp_num:
                    raise ValueError(f"q-index outside model range: {q}")
                inputs[-1] = torch.tensor([q], dtype=torch.int32)
                for index, tensor in enumerate(inputs[:-1]):
                    if part.value == "MLVCDecoder" and index < 3:
                        inputs[index] = torch.randint(-2, 3, tensor.shape).half()
                    else:
                        inputs[index] = torch.rand_like(tensor) * (1.0 if index == 0 else 0.1)
                with torch.inference_mode():
                    expected = wrapper.torch_model(*inputs)
                    actual = execute_graph(bundle / "aot" / part.value, inputs)
                for a, e in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, e, rtol=0, atol=0)
                result = dict(profile=profile, part=part.value, q_index=q, numerically_exact=True)
                results.append(result)
                print(json.dumps(result), flush=True)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(dict(cases=results, passed=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
