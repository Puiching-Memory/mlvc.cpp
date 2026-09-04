#!/usr/bin/env python3
"""Compile fixed-shape MLVC ONNX graphs into deterministic AOT model data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


ALIGNMENT = 256
SUPPORTED_OPERATORS = {
    "Add",
    "Clip",
    "Concat",
    "Conv",
    "DepthToSpace",
    "Gather",
    "LeakyRelu",
    "Mul",
    "Reciprocal",
    "Round",
    "Sigmoid",
    "Slice",
    "SpaceToDepth",
    "Sub",
}
DTYPE_NAMES = {
    TensorProto.FLOAT16: "fp16",
    TensorProto.FLOAT: "fp32",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
}
DTYPE_BYTES = {
    TensorProto.FLOAT16: 2,
    TensorProto.FLOAT: 4,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
}


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parts", nargs="+", default=["MLVCEncoder", "MLVCDecoder"]
    )
    return parser.parse_args()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    result = []
    for dim in tensor.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        result.append(dim.dim_value)
    return result


def json_attribute(attribute: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "dtype": DTYPE_NAMES.get(value.data_type, str(value.data_type)),
            "shape": list(array.shape),
            "values": array.reshape(-1).tolist(),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def merge_free_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not blocks:
        return []
    merged = []
    for offset, size in sorted(blocks):
        if merged and merged[-1][0] + merged[-1][1] == offset:
            previous_offset, previous_size = merged[-1]
            merged[-1] = previous_offset, previous_size + size
        else:
            merged.append((offset, size))
    return merged


def plan_arena(
    graph: onnx.GraphProto,
    shapes: dict[str, list[int]],
    dtypes: dict[str, int],
    initializer_names: set[str],
) -> tuple[dict[str, dict[str, int]], int]:
    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}
    consumers: dict[str, list[int]] = defaultdict(list)
    producer: dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            if name:
                consumers[name].append(index)
        for name in node.output:
            if name:
                producer[name] = index

    intervals = []
    for name, birth in producer.items():
        if name in initializer_names or name in graph_inputs:
            continue
        shape = shapes.get(name)
        dtype = dtypes.get(name)
        if shape is None or dtype not in DTYPE_BYTES:
            raise ValueError(f"missing static type/shape for runtime tensor {name!r}")
        count = int(np.prod(shape, dtype=np.int64))
        size = align_up(count * DTYPE_BYTES[dtype])
        death = max(consumers.get(name, [birth]))
        if name in graph_outputs:
            death = len(graph.node)
        intervals.append((birth, death, name, size))

    active: list[tuple[int, int, int, str]] = []
    free: list[tuple[int, int]] = []
    result: dict[str, dict[str, int]] = {}
    arena_end = 0
    peak_live = 0
    for birth, death, name, size in sorted(intervals):
        retained = []
        for active_death, offset, active_size, active_name in active:
            if active_death < birth:
                free.append((offset, active_size))
            else:
                retained.append((active_death, offset, active_size, active_name))
        active = retained
        free = merge_free_blocks(free)

        candidate = None
        for index, (_, block_size) in enumerate(free):
            if block_size >= size and (
                candidate is None or block_size < free[candidate][1]
            ):
                candidate = index
        if candidate is None:
            offset = arena_end
            arena_end += size
        else:
            offset, block_size = free.pop(candidate)
            if block_size > size:
                free.append((offset + size, block_size - size))
        active.append((death, offset, size, name))
        result[name] = {
            "offset": offset,
            "bytes": size,
            "birth": birth,
            "death": death,
        }
        peak_live = max(peak_live, sum(item[2] for item in active))
    return result, max(arena_end, peak_live)


def write_weights(
    initializers: list[onnx.TensorProto], destination: Path
) -> tuple[list[dict[str, Any]], str]:
    records = []
    content = bytearray()
    for tensor in sorted(initializers, key=lambda item: item.name):
        array = np.ascontiguousarray(numpy_helper.to_array(tensor))
        if tensor.data_type not in DTYPE_NAMES:
            raise ValueError(f"unsupported initializer dtype {tensor.data_type}: {tensor.name}")
        aligned = align_up(len(content))
        content.extend(b"\0" * (aligned - len(content)))
        raw = array.tobytes(order="C")
        records.append(
            {
                "name": tensor.name,
                "dtype": DTYPE_NAMES[tensor.data_type],
                "shape": list(array.shape),
                "offset": aligned,
                "bytes": len(raw),
            }
        )
        content.extend(raw)
    destination.write_bytes(content)
    return records, hashlib.sha256(content).hexdigest()


def compile_graph(source: Path, destination: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    model = onnx.load_model_from_string(source_bytes)
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    graph = model.graph
    unsupported_operators = sorted(
        {node.op_type for node in graph.node} - SUPPORTED_OPERATORS
    )
    if unsupported_operators:
        raise ValueError(
            "driver-cubin has no kernel for ONNX operators: "
            + ", ".join(unsupported_operators)
        )

    values = list(graph.input) + list(graph.output) + list(graph.value_info)
    shapes = {value.name: tensor_shape(value) for value in values}
    dtypes = {value.name: value.type.tensor_type.elem_type for value in values}
    for initializer in graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
        dtypes[initializer.name] = initializer.data_type
    if any(shape is None for shape in shapes.values()):
        missing = sorted(name for name, shape in shapes.items() if shape is None)
        raise ValueError(f"dynamic tensors remain after shape inference: {missing}")
    concrete_shapes = {name: shape for name, shape in shapes.items() if shape is not None}

    initializer_names = {tensor.name for tensor in graph.initializer}
    unsupported_runtime_dtypes = sorted(
        {
            DTYPE_NAMES.get(dtype, str(dtype))
            for name, dtype in dtypes.items()
            if name not in initializer_names
            and dtype not in {TensorProto.FLOAT16, TensorProto.INT32}
        }
    )
    if unsupported_runtime_dtypes:
        raise ValueError(
            "driver-cubin has no runtime storage for dtypes: "
            + ", ".join(unsupported_runtime_dtypes)
        )
    unsupported_initializer_dtypes = sorted(
        {
            DTYPE_NAMES.get(tensor.data_type, str(tensor.data_type))
            for tensor in graph.initializer
            if tensor.data_type not in {TensorProto.FLOAT16, TensorProto.INT64}
        }
    )
    if unsupported_initializer_dtypes:
        raise ValueError(
            "driver-cubin has no initializer support for dtypes: "
            + ", ".join(unsupported_initializer_dtypes)
        )
    arena, arena_bytes = plan_arena(
        graph, concrete_shapes, dtypes, initializer_names
    )
    destination.mkdir(parents=True, exist_ok=True)
    weight_records, weights_sha256 = write_weights(
        list(graph.initializer), destination / "weights.bin"
    )

    nodes = []
    for index, node in enumerate(graph.node):
        nodes.append(
            {
                "index": index,
                "name": node.name or f"{node.op_type}_{index}",
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {
                    attribute.name: json_attribute(attribute)
                    for attribute in node.attribute
                },
            }
        )

    def value_record(value: onnx.ValueInfoProto) -> dict[str, Any]:
        return {
            "name": value.name,
            "dtype": DTYPE_NAMES.get(dtypes[value.name], str(dtypes[value.name])),
            "shape": concrete_shapes[value.name],
        }

    manifest = {
        "schema_version": 1,
        "source": source.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "opset": max(item.version for item in model.opset_import),
        "inputs": [value_record(value) for value in graph.input],
        "outputs": [value_record(value) for value in graph.output],
        "operator_counts": dict(sorted(Counter(node.op_type for node in graph.node).items())),
        "nodes": nodes,
        "weights": weight_records,
        "weights_sha256": weights_sha256,
        "weights_bytes": (destination / "weights.bin").stat().st_size,
        "arena_alignment": ALIGNMENT,
        "arena_bytes": arena_bytes,
        "arena": arena,
        "tensors": {
            name: {
                "dtype": DTYPE_NAMES.get(dtypes[name], str(dtypes[name])),
                "shape": shape,
            }
            for name, shape in sorted(concrete_shapes.items())
            if name not in initializer_names
        },
    }
    manifest_path = destination / "graph.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "source": source.as_posix(),
        "manifest": manifest_path.as_posix(),
        "nodes": len(nodes),
        "weights_bytes": manifest["weights_bytes"],
        "arena_bytes": arena_bytes,
        "operator_counts": manifest["operator_counts"],
    }


def main() -> None:
    args = parse_args()
    summary = {"schema_version": 1, "graphs": []}
    for part in args.parts:
        result = compile_graph(
            args.model_dir / f"{part}.onnx", args.output_dir / part
        )
        summary["graphs"].append(result)
        print(
            f"{part}: nodes={result['nodes']} weights={result['weights_bytes']} "
            f"arena={result['arena_bytes']}"
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
