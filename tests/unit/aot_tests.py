"""Host checks for checkpoint capture, AOT lowering, and buffer lifetimes."""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from model_aot import compile_module


@torch.inference_mode()
def execute_graph(directory, inputs):
    """Independent host interpretation of the serialized runtime contract."""
    graph = json.loads((directory / "graph.json").read_text())
    weights = (directory / "weights.bin").read_bytes()
    values = dict(zip((v["name"] for v in graph["inputs"]), inputs, strict=True))
    for weight in graph["weights"]:
        dtype = {"fp16": "<f2", "int64": "<i8"}[weight["dtype"]]
        array = np.frombuffer(weights, dtype=dtype, count=weight["bytes"] // np.dtype(dtype).itemsize,
                              offset=weight["offset"]).copy().reshape(weight["shape"])
        values[weight["name"]] = torch.from_numpy(array)
    for node in graph["nodes"]:
        op, attrs = node["op"], node["attributes"]
        args = [values[n] for n in node["inputs"]]
        if op == "Conv":
            result = F.conv2d(*args[:2], bias=args[2] if len(args) > 2 else None,
                stride=attrs["strides"], padding=attrs["pads"][:2],
                dilation=attrs["dilations"], groups=attrs["group"])
        elif op in {"Add", "Sub", "Mul"}:
            result = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul}[op](*args)
        elif op == "Clip":
            result = args[0].clamp(float(args[1].item()), float(args[2].item()))
        elif op == "LeakyRelu":
            result = F.leaky_relu(args[0], attrs["alpha"])
        elif op in {"Round", "Reciprocal", "Sigmoid"}:
            result = getattr(torch, op.lower())(args[0])
        elif op == "Gather":
            result = args[0][args[1].long()]
        elif op == "Slice":
            slices = [slice(None)] * args[0].ndim
            for start, end, axis, step in zip(*(t.tolist() for t in args[1:]), strict=True):
                slices[axis] = slice(start, end, step)
            result = args[0][tuple(slices)]
        elif op == "Concat":
            result = torch.cat(args, attrs["axis"])
        elif op in {"SpaceToDepth", "DepthToSpace"}:
            assert attrs["mode"] == "CRD"
            result = (F.pixel_unshuffle if op == "SpaceToDepth" else F.pixel_shuffle)(args[0], attrs["blocksize"])
        else:
            raise ValueError(op)
        values[node["outputs"][0]] = result
    return tuple(values[v["name"]] for v in graph["outputs"])


class Example(torch.nn.Module):
    InputType = namedtuple("Inputs", "x ref q")
    OutputType = namedtuple("Outputs", "reconstruction feature")

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 12, 1).half()
        self.scales = torch.nn.Parameter(torch.arange(1, 5, dtype=torch.float16).reshape(4, 1, 1, 1))

    def forward(self, x, ref, q):
        shuffled = F.pixel_unshuffle(x, 2)
        y = self.conv(shuffled) * self.scales[q]
        a, b = y.chunk(2, 1)
        # Unsupported dead computation must not enter the deployment graph.
        _ = torch.sin(y).flatten()
        reconstruction = F.pixel_shuffle(y, 2).clamp_(0, 1)
        feature = torch.cat((a + ref[:, :6], 1.0 - b), 1)
        return self.OutputType(reconstruction, feature)


class AotTests(unittest.TestCase):
    def test_capture_replays_new_inputs_and_qp(self):
        torch.manual_seed(7)
        model = Example().eval()
        inputs = (torch.rand(1, 3, 8, 8).half(), torch.zeros(1, 12, 4, 4).half(), torch.tensor([0], dtype=torch.int32))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            graph, _ = compile_module(model, inputs, directory, "test")
            for q in (0, 1, 3):
                sample = (torch.rand_like(inputs[0]), torch.rand_like(inputs[1]), torch.tensor([q], dtype=torch.int32))
                for actual, expected in zip(execute_graph(directory, sample), model(*sample), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertNotIn("Sin", graph["operator_counts"])
            self.assertEqual(graph["nodes"][-1]["outputs"], ["feature"])
            self.assertEqual(hashlib.sha256((directory / "weights.bin").read_bytes()).hexdigest(), graph["weights_sha256"])
            # Intervals that overlap in time must not overlap in memory.
            entries = list(graph["arena"].values())
            for i, a in enumerate(entries):
                for b in entries[i + 1:]:
                    if max(a["birth"], b["birth"]) <= min(a["death"], b["death"]):
                        self.assertTrue(a["offset"] + a["bytes"] <= b["offset"] or
                                        b["offset"] + b["bytes"] <= a["offset"])

    def test_rejects_unsupported_live_operator(self):
        class Unsupported(torch.nn.Module):
            InputType = namedtuple("Inputs", "x")
            OutputType = namedtuple("Outputs", "y")

            def forward(self, x):
                return self.OutputType(torch.sin(x))

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsupported live.*sin"):
                compile_module(Unsupported(), (torch.ones(1).half(),), Path(directory), "test")


if __name__ == "__main__":
    unittest.main()
