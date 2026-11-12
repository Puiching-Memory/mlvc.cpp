#!/usr/bin/env python3
"""Compile official checkpoint + model code directly into fixed-shape AOT data.

PyTorch is an offline dependency. The deployed codec loads only AOT graph data,
weights, and CUDA kernels; no framework model serialization is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import weakref
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

ALIGNMENT = 256
DTYPES = {torch.float16: "fp16", torch.int32: "int32", torch.int64: "int64"}
DTYPE_BYTES = {"fp16": 2, "int32": 4, "int64": 8}


def align_up(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def plan_arena(nodes, tensors, inputs, outputs):
    consumers, producer = defaultdict(list), {}
    for index, node in enumerate(nodes):
        for name in node["inputs"]:
            consumers[name].append(index)
        for name in node["outputs"]:
            producer[name] = index
    intervals = []
    for name, birth in producer.items():
        if name in inputs:
            continue
        tensor = tensors[name]
        size = align_up(math.prod(tensor["shape"]) * DTYPE_BYTES[tensor["dtype"]])
        death = len(nodes) if name in outputs else max(consumers.get(name, [birth]))
        intervals.append((birth, death, name, size))
    active, free, result, arena_end = [], [], {}, 0
    for birth, death, name, size in sorted(intervals):
        retained = []
        for old_death, offset, old_size in active:
            if old_death < birth:
                free.append((offset, old_size))
            else:
                retained.append((old_death, offset, old_size))
        active = retained
        merged = []
        for offset, block_size in sorted(free):
            if merged and sum(merged[-1]) == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + block_size)
            else:
                merged.append((offset, block_size))
        free = merged
        candidates = [(s, i) for i, (_, s) in enumerate(free) if s >= size]
        if candidates:
            _, index = min(candidates)
            offset, block_size = free.pop(index)
            if block_size > size:
                free.append((offset + size, block_size - size))
        else:
            offset = arena_end
            arena_end += size
        active.append((death, offset, size))
        result[name] = dict(offset=offset, bytes=size, birth=birth, death=death)
    return result, arena_end


def reachable_nodes(nodes, outputs):
    needed, retained = set(outputs), []
    for node in reversed(nodes):
        if needed.intersection(node["outputs"]):
            retained.append(node)
            needed.update(v for v in tree_flatten(node["inputs"])[0] if isinstance(v, str))
    return list(reversed(retained))


class Capture(TorchDispatchMode):
    """Record fixed-shape eager dependencies; fold constant-only computation.

    Weak references prevent recycled tensor IDs from creating false graph edges.
    Unsupported live operators are rejected after dead branches are pruned.
    """

    def __init__(self, inputs, names):
        super().__init__()
        self.names, self.constants, self.tensors, self.nodes = {}, {}, {}, []
        self.live = set()
        for tensor, name in zip(inputs, names, strict=True):
            self.bind(tensor, name)
            self.live.add(name)

    def bind(self, tensor, name):
        self.names[id(tensor)] = (weakref.ref(tensor), name)
        self.tensors[name] = dict(dtype=DTYPES.get(tensor.dtype, str(tensor.dtype)), shape=list(tensor.shape))

    def name(self, tensor):
        entry = self.names.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        name = f"constant_{len(self.constants):05d}"
        self.constants[name] = tensor.detach().cpu().numpy().copy()
        self.bind(tensor, name)
        return name

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        mapped = tree_map(lambda x: self.name(x) if isinstance(x, torch.Tensor) else x, args)
        mapped_kwargs = tree_map(lambda x: self.name(x) if isinstance(x, torch.Tensor) else x, kwargs)
        live = any(isinstance(x, str) and x in self.live for x in tree_flatten((mapped, mapped_kwargs))[0])
        result = func(*args, **kwargs)
        if live:
            outputs = []
            for tensor in tree_flatten(result)[0]:
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"data-dependent non-tensor output: {func}")
                name = f"value_{len(self.nodes):05d}_{len(outputs)}"
                self.bind(tensor, name)
                self.live.add(name)
                outputs.append(name)
            self.nodes.append(dict(op=str(func), inputs=mapped, kwargs=mapped_kwargs, outputs=outputs))
        return result


class Lowering:
    def __init__(self, capture):
        self.nodes, self.aliases = [], {}
        self.tensors, self.constants = dict(capture.tensors), dict(capture.constants)

    def resolve(self, name):
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def constant(self, values, dtype=np.int64):
        name = f"literal_{len(self.constants):05d}"
        self.constants[name] = np.asarray(values, dtype=dtype).reshape(-1)
        return name

    def emit(self, op, inputs, output, attributes=None):
        self.nodes.append(dict(op=op, inputs=inputs, outputs=[output], attributes=attributes or {}))

    def slice(self, source, output, axis, start, end, step=1):
        self.emit("Slice", [source, *(self.constant([v]) for v in (start, end, axis, step))], output)

    def lower(self, raw):
        args = tree_map(lambda x: self.resolve(x) if isinstance(x, str) else x, raw["inputs"])
        kwargs, op, out = raw["kwargs"], raw["op"], raw["outputs"][0]
        get = lambda i, default: args[i] if len(args) > i else default
        if op in {"aten.detach.default", "aten.clone.default", "aten.alias.default", "aten.contiguous.default"}:
            self.aliases[out] = args[0]
        elif op in {"aten.convolution.default", "aten._convolution.default", "aten.conv2d.default"}:
            if op == "aten.conv2d.default":
                x, weight, bias = args[:3]
                stride, pad, dilation, groups = get(3, [1, 1]), get(4, [0, 0]), get(5, [1, 1]), get(6, 1)
                transposed, output_pad = False, [0, 0]
            else:
                x, weight, bias, stride, pad, dilation, transposed, output_pad, groups = args[:9]
            if transposed or any(output_pad):
                raise ValueError("transposed convolution is unsupported")
            self.emit("Conv", [x, weight] + ([bias] if bias is not None else []), out,
                      dict(strides=list(stride), pads=list(pad) * 2, dilations=list(dilation), group=groups))
        elif op in {f"aten.{n}.{k}" for n in ("add", "sub", "mul") for k in ("Tensor", "Scalar")}:
            if kwargs.get("alpha", 1) != 1:
                raise ValueError("scaled add/sub is unsupported")
            inputs = [a if isinstance(a, str) else self.constant([a], np.float16) for a in args[:2]]
            self.emit({"add": "Add", "sub": "Sub", "mul": "Mul"}[op.split('.')[1]], inputs, out)
        elif op == "aten.rsub.Scalar":
            if get(2, 1) != 1:
                raise ValueError("scaled rsub is unsupported")
            self.emit("Sub", [self.constant([args[1]], np.float16), args[0]], out)
        elif op in {"aten.reciprocal.default", "aten.round.default", "aten.sigmoid.default"}:
            self.emit(op.split('.')[1].capitalize(), [args[0]], out)
        elif op in {"aten.leaky_relu.default", "aten.leaky_relu_.default"}:
            self.emit("LeakyRelu", [args[0]], out, {"alpha": get(1, 0.01)})
        elif op in {"aten.clamp.default", "aten.clamp_.default", "aten.clamp_min.default", "aten.clamp_max.default"}:
            lower, upper = get(1, kwargs.get("min")), get(2, kwargs.get("max"))
            if op == "aten.clamp_min.default":
                upper = None
            elif op == "aten.clamp_max.default":
                lower, upper = None, get(1, None)
            bounds = [lower if lower is not None else -65504.0, upper if upper is not None else 65504.0]
            self.emit("Clip", [args[0], *(self.constant([v], np.float16) for v in bounds)], out)
        elif op in {"aten.relu.default", "aten.relu_.default"}:
            self.emit("Clip", [args[0], self.constant([0], np.float16), self.constant([65504], np.float16)], out)
        elif op == "aten.index.Tensor":
            if len(args[1]) != 1 or args[1][0] is None:
                raise ValueError("only axis-0 tensor indexing is supported")
            index = args[1][0]
            if self.tensors[index]["dtype"] != "int32" or self.tensors[index]["shape"] != [1]:
                raise ValueError("Gather requires one int32 index")
            self.emit("Gather", [args[0], index], out, {"axis": 0})
        elif op == "aten.slice.Tensor":
            axis, start, end, step = get(1, 0), get(2, 0), get(3, 2**63-1), get(4, 1)
            start, end, step = slice(start, end, step).indices(self.tensors[args[0]]["shape"][axis])
            self.slice(args[0], out, axis, start, end, step)
        elif op in {"aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default"}:
            axis, offset = get(2, 0), 0
            for output in raw["outputs"]:
                size = self.tensors[output]["shape"][axis]
                self.slice(args[0], output, axis, offset, offset + size)
                offset += size
        elif op == "aten.cat.default":
            self.emit("Concat", list(args[0]), out, {"axis": get(1, 0)})
        elif op in {"aten.pixel_shuffle.default", "aten.pixel_unshuffle.default"}:
            self.emit("DepthToSpace" if op == "aten.pixel_shuffle.default" else "SpaceToDepth",
                      [args[0]], out, {"blocksize": args[1], "mode": "CRD"})
        else:
            raise ValueError(f"unsupported live checkpoint operator: {op}")


@torch.inference_mode()
def compile_module(model, inputs, destination, source_hash):
    capture = Capture(inputs, model.InputType._fields)
    with capture:
        result = model(*inputs)
    raw_outputs = [capture.name(t) for t in result]
    lower = Lowering(capture)
    for node in reachable_nodes(capture.nodes, raw_outputs):
        lower.lower(node)
    raw_outputs = [lower.resolve(n) for n in raw_outputs]
    output_names = list(model.OutputType._fields)
    renames = dict(zip(raw_outputs, output_names, strict=True))
    nodes = reachable_nodes(lower.nodes, raw_outputs)
    nodes = tree_map(lambda x: renames.get(x, x) if isinstance(x, str) else x, nodes)
    input_names = list(model.InputType._fields)
    used = set(input_names + output_names)
    for index, node in enumerate(nodes):
        node.update(index=index, name=f"{node['op']}_{index}")
        used.update(node["inputs"] + node["outputs"])
    tensors = {renames.get(n, n): v for n, v in lower.tensors.items()
               if renames.get(n, n) in used and n not in lower.constants}
    for name, info in tensors.items():
        if info["dtype"] not in {"fp16", "int32"} or len(info["shape"]) > 4 or any(d <= 0 for d in info["shape"]):
            raise ValueError(f"unsupported runtime tensor {name}: {info}")
    weights, content = [], bytearray()
    for name, array in sorted(lower.constants.items()):
        if name not in used:
            continue
        array = np.ascontiguousarray(array)
        dtype = {np.dtype("float16"): "fp16", np.dtype("int64"): "int64"}.get(array.dtype)
        if dtype is None:
            raise ValueError(f"unsupported weight dtype: {name} {array.dtype}")
        content.extend(b"\0" * (align_up(len(content)) - len(content)))
        weights.append(dict(name=name, dtype=dtype, shape=list(array.shape), offset=len(content), bytes=array.nbytes))
        content.extend(array.tobytes())
    arena, arena_bytes = plan_arena(nodes, tensors, input_names, output_names)
    manifest = dict(schema_version=2, source="checkpoint", source_sha256=source_hash,
        inputs=[dict(name=n, **tensors[n]) for n in input_names],
        outputs=[dict(name=n, **tensors[n]) for n in output_names], nodes=nodes, tensors=tensors, weights=weights,
        operator_counts=dict(sorted(Counter(n["op"] for n in nodes).items())),
        weights_sha256=hashlib.sha256(content).hexdigest(), weights_bytes=len(content),
        arena_alignment=ALIGNMENT, arena_bytes=arena_bytes, arena=arena)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "weights.bin").write_bytes(content)
    (destination / "graph.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, result


def main():
    from model_source import example_inputs, load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-width", type=int, default=640)
    parser.add_argument("--model-height", type=int, default=368)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    full, split, gaussian, estimator = load_model(args.profile, args.checkpoint, args.model_width, args.model_height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(name=args.profile, params=dict(full_model_params=asdict(full.model_params),
                                                  split_model_params=asdict(split.split_model_params)))
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, table in (("gaussian_pmf", gaussian), ("bit_estimator_pmf", estimator)):
        (args.output_dir / f"{name}.json").write_text(json.dumps(asdict(table)) + "\n")
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    summaries = []
    for part, wrapper in split.model_parts.items():
        inputs = example_inputs(part.value, full.model_params, args.model_width, args.model_height)
        graph, result = compile_module(wrapper.torch_model, inputs, args.output_dir / "aot" / part.value, source_hash)
        snapshot = {f"{part.value}_input_{n}": t.numpy() for n, t in zip(wrapper.metadata.input_fields, inputs, strict=True)}
        snapshot.update({f"{part.value}_output_{n}": t.numpy() for n, t in zip(wrapper.metadata.output_fields, result, strict=True)})
        np.savez(args.output_dir / f"benchmark-{part.value}.npz", **snapshot)
        summaries.append(dict(part=part.value, nodes=len(graph["nodes"]), weights_bytes=graph["weights_bytes"], arena_bytes=graph["arena_bytes"]))
        print(summaries[-1])
    (args.output_dir / "aot/manifest.json").write_text(json.dumps(
        dict(schema_version=1, source_sha256=source_hash, graphs=summaries), indent=2) + "\n")


if __name__ == "__main__":
    main()
