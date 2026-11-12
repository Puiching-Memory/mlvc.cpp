"""Typed Protobuf model serialization shared by offline tools.

The dictionary adapters describe the compiler's in-memory IR and diagnostic
reports. Model assets on disk are exclusively Protobuf and aligned weight blobs.
"""

from pathlib import Path

from google.protobuf.message import DecodeError
from mlvc_proto import mlvc_model_pb2 as pb

GRAPH_SCHEMA = 3
BUNDLE_SCHEMA = 2
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
DTYPES = {"fp16": pb.FP16, "int32": pb.INT32, "int64": pb.INT64}
DTYPE_NAMES = {value: name for name, value in DTYPES.items()}
OPS = dict(Add=pb.ADD, Sub=pb.SUB, Mul=pb.MUL, Clip=pb.CLIP, Concat=pb.CONCAT,
    Conv=pb.CONV, DepthToSpace=pb.DEPTH_TO_SPACE, Gather=pb.GATHER,
    LeakyRelu=pb.LEAKY_RELU, Reciprocal=pb.RECIPROCAL, Round=pb.ROUND,
    Sigmoid=pb.SIGMOID, Slice=pb.SLICE, SpaceToDepth=pb.SPACE_TO_DEPTH)
OP_NAMES = {value: name for name, value in OPS.items()}


def read_message(path: Path, message_type):
    if path.stat().st_size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Protobuf model message is too large: {path}")
    message = message_type()
    try:
        message.ParseFromString(path.read_bytes())
    except DecodeError as error:
        raise ValueError(f"invalid Protobuf model message: {path}") from error
    return message


def write_message(path: Path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(message.SerializeToString(deterministic=True))


def metadata_from_dict(source, checkpoint_hash):
    full = source["params"]["full_model_params"]
    split = source["params"]["split_model_params"]
    result = pb.ModelMetadata(schema_version=1, profile=source["name"],
        model_version=full["model_version"], checkpoint_sha256=bytes.fromhex(checkpoint_hash),
        width=split["model_width"], height=split["model_height"])
    fields = ("pixel_range", "qp_num", "total_qp_num", "frame_index_map", "qp_shift",
        "feature_channels", "latent_channels", "hyperprior_channels", "downsample_feature",
        "downsample_latent", "downsample_hyperprior", "y_scale_repeat", "disable_feature_reset")
    for name in fields:
        if name in ("frame_index_map", "qp_shift"):
            getattr(result, name).extend(full[name])
        else:
            setattr(result, name, full[name])
    for name in ("iframe_period", "reset_period"):
        if full.get(name) is not None:
            setattr(result, name, full[name])
    return result


def metadata_to_dict(message):
    if message.schema_version != 1:
        raise ValueError("unsupported model metadata schema")
    fields = ("model_version", "pixel_range", "qp_num", "total_qp_num",
        "feature_channels", "latent_channels", "hyperprior_channels", "downsample_feature",
        "downsample_latent", "downsample_hyperprior", "y_scale_repeat", "disable_feature_reset")
    full = {name: getattr(message, name) for name in fields}
    full.update(frame_index_map=list(message.frame_index_map), qp_shift=list(message.qp_shift))
    for name in ("iframe_period", "reset_period"):
        full[name] = getattr(message, name) if message.HasField(name) else None
    return dict(name=message.profile, params=dict(full_model_params=full,
        split_model_params=dict(model_width=message.width, model_height=message.height)))


def read_metadata(directory):
    return metadata_to_dict(read_message(directory / "metadata.pb", pb.ModelMetadata))


def pmf_from_dict(source, gaussian):
    result = pb.GaussianPmf() if gaussian else pb.BitEstimatorPmf()
    result.schema_version = 1
    result.table.lengths.extend(source["pmf_lengths"])
    result.table.offsets.extend(source["pmf_offsets"])
    result.table.values.extend(source["pmf_table"])
    for name in (("scale_min", "scale_max", "scale_levels", "index_space") if gaussian
                 else ("qp_num", "channels")):
        setattr(result, name, source[name])
    return result


def pmf_to_dict(message):
    if message.schema_version != 1:
        raise ValueError("unsupported PMF schema")
    result = dict(pmf_lengths=list(message.table.lengths), pmf_offsets=list(message.table.offsets),
                  pmf_table=list(message.table.values))
    fields = ("scale_min", "scale_max", "scale_levels", "index_space") if isinstance(message, pb.GaussianPmf) else ("qp_num", "channels")
    result.update({name: getattr(message, name) for name in fields})
    return result


def graph_from_dict(source):
    result = pb.Graph(schema_version=GRAPH_SCHEMA,
        checkpoint_sha256=bytes.fromhex(source["source_sha256"]),
        arena_bytes=source["arena_bytes"], arena_alignment=source["arena_alignment"],
        weights_bytes=source["weights_bytes"], weights_sha256=bytes.fromhex(source["weights_sha256"]))
    weights = {item["name"]: item for item in source["weights"]}
    all_tensors = dict(source["tensors"])
    all_tensors.update(weights)
    ids = {name: i for i, name in enumerate(sorted(all_tensors), 1)}
    for name, identifier in ids.items():
        data = all_tensors[name]
        tensor = result.tensors.add(id=identifier, name=name, dtype=DTYPES[data["dtype"]], shape=data["shape"])
        if name in weights:
            tensor.weight.offset = data["offset"]
            tensor.weight.size_bytes = data["bytes"]
        elif name in source["arena"]:
            allocation = source["arena"][name]
            tensor.arena.CopyFrom(pb.ArenaAllocation(offset=allocation["offset"],
                size_bytes=allocation["bytes"], birth=allocation["birth"], death=allocation["death"]))
    result.inputs.extend(ids[item["name"]] for item in source["inputs"])
    result.outputs.extend(ids[item["name"]] for item in source["outputs"])
    for data in source["nodes"]:
        node = result.nodes.add(op=OPS[data["op"]],
            inputs=[ids[name] if name else 0 for name in data["inputs"]],
            outputs=[ids[name] for name in data["outputs"]])
        attrs = data["attributes"]
        if node.op == pb.CONV:
            node.conv.CopyFrom(pb.ConvAttributes(strides=attrs["strides"], pads=attrs["pads"],
                dilations=attrs["dilations"], groups=attrs.get("group", 1)))
        elif node.op in (pb.CONCAT, pb.GATHER):
            node.axis.axis = attrs.get("axis", 0)
        elif node.op == pb.LEAKY_RELU:
            node.activation.alpha = attrs.get("alpha", 0.01)
        elif node.op in (pb.DEPTH_TO_SPACE, pb.SPACE_TO_DEPTH):
            if attrs.get("mode", "CRD") != "CRD":
                raise ValueError("only native channel-first pixel layout is supported")
            node.pixel.block_size = attrs["blocksize"]
    return result


def graph_to_dict(message):
    if message.schema_version != GRAPH_SCHEMA:
        raise ValueError("unsupported AOT Protobuf graph schema")
    names = {item.id: item.name for item in message.tensors}
    names[0] = ""
    if len(names) != len(message.tensors) + 1:
        raise ValueError("duplicate/zero tensor ID")
    tensors, weights, arena = {}, [], {}
    for tensor in message.tensors:
        data = dict(dtype=DTYPE_NAMES[tensor.dtype], shape=list(tensor.shape))
        storage = tensor.WhichOneof("storage")
        if storage == "weight":
            weights.append(dict(name=tensor.name, **data, offset=tensor.weight.offset,
                                bytes=tensor.weight.size_bytes))
        else:
            tensors[tensor.name] = data
            if storage == "arena":
                arena[tensor.name] = dict(offset=tensor.arena.offset, bytes=tensor.arena.size_bytes,
                    birth=tensor.arena.birth, death=tensor.arena.death)
    nodes = []
    for index, node in enumerate(message.nodes):
        op, attrs = OP_NAMES[node.op], {}
        if node.op == pb.CONV:
            attrs = dict(strides=list(node.conv.strides), pads=list(node.conv.pads),
                         dilations=list(node.conv.dilations), group=node.conv.groups)
        elif node.op in (pb.CONCAT, pb.GATHER):
            attrs = dict(axis=node.axis.axis)
        elif node.op == pb.LEAKY_RELU:
            attrs = dict(alpha=node.activation.alpha)
        elif node.op in (pb.DEPTH_TO_SPACE, pb.SPACE_TO_DEPTH):
            attrs = dict(blocksize=node.pixel.block_size, mode="CRD")
        nodes.append(dict(index=index, name=f"{op}_{index}", op=op,
            inputs=[names[n] for n in node.inputs], outputs=[names[n] for n in node.outputs], attributes=attrs))
    return dict(schema_version=message.schema_version, source="checkpoint", source_sha256=message.checkpoint_sha256.hex(),
        inputs=[dict(name=names[n], **tensors[names[n]]) for n in message.inputs],
        outputs=[dict(name=names[n], **tensors[names[n]]) for n in message.outputs],
        nodes=nodes, tensors=tensors, weights=weights, arena=arena,
        arena_bytes=message.arena_bytes, arena_alignment=message.arena_alignment,
        weights_bytes=message.weights_bytes, weights_sha256=message.weights_sha256.hex())


def bundle_to_dict(message):
    if message.schema_version != BUNDLE_SCHEMA:
        raise ValueError("unsupported Protobuf bundle schema")
    return dict(schema_version=message.schema_version, profile=message.profile,
        model_version=message.model_version, model_size=dict(width=message.width, height=message.height),
        entropy_model_sha256=message.entropy_model_sha256.hex(),
        artifacts={item.path: dict(path=item.path, bytes=item.size_bytes, sha256=item.sha256.hex())
                   for item in message.artifacts},
        compatibility=dict(container=message.contract.container, entropy=message.contract.entropy,
            decoder_max_yuv_sample_error=message.contract.decoder_max_yuv_sample_error,
            decoder_min_psnr_db=message.contract.decoder_min_psnr_db))


def read_bundle(directory):
    return bundle_to_dict(read_message(directory / "model_bundle.pb", pb.ModelBundle))
