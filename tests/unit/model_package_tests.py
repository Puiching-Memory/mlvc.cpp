"""Protobuf-only model assembly, integrity checks, and deterministic embedding."""

import argparse
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import model_package as package
from model_wire import pb, write_message, read_message, GRAPH_SCHEMA


class PackageTests(unittest.TestCase):
    def make_bundle(self, root, profile):
        registration = package.load_json(package.PROFILE_PATH)["profiles"][profile]
        checkpoint = bytes.fromhex(registration["weights_sha256"])
        source = root / "source" / profile
        source.mkdir(parents=True)
        write_message(source / "metadata.pb", pb.ModelMetadata(schema_version=1,
            profile=profile, model_version=registration["model_version"],
            checkpoint_sha256=checkpoint, width=64, height=64))
        table = pb.PmfTable(lengths=[1], offsets=[0], values=[1])
        write_message(source / "gaussian_pmf.pb", pb.GaussianPmf(schema_version=1,
            table=table, scale_levels=1, index_space=True))
        write_message(source / "bit_estimator_pmf.pb", pb.BitEstimatorPmf(schema_version=1,
            table=table, qp_num=1, channels=1))
        for part in ("MLVCEncoder", "MLVCDecoder"):
            directory = source / "aot" / part
            weights = b"\x00\x3c"
            graph = pb.Graph(schema_version=GRAPH_SCHEMA, checkpoint_sha256=checkpoint,
                arena_alignment=256, arena_bytes=256, weights_bytes=2,
                weights_sha256=hashlib.sha256(weights).digest(), inputs=[2], outputs=[3])
            tensor = graph.tensors.add(id=1, name="weight", dtype=pb.FP16, shape=[1])
            tensor.weight.CopyFrom(pb.WeightLocation(offset=0, size_bytes=2))
            graph.tensors.add(id=2, name="x", dtype=pb.FP16, shape=[1])
            tensor = graph.tensors.add(id=3, name="feature", dtype=pb.FP16, shape=[1])
            tensor.arena.CopyFrom(pb.ArenaAllocation(offset=0, size_bytes=256, birth=0, death=1))
            graph.nodes.add(op=pb.ADD, inputs=[1, 2], outputs=[3])
            write_message(directory / "graph.pb", graph)
            (directory / "weights.bin").write_bytes(weights)
        write_message(source / "aot/manifest.pb", pb.AotManifest(schema_version=1,
            checkpoint_sha256=checkpoint))
        destination = root / "canonical" / profile / "64x64"
        with contextlib.redirect_stdout(io.StringIO()):
            package.assemble(argparse.Namespace(profile=profile, source_dir=source,
                output_dir=destination, overwrite=False))
        return destination

    def test_assemble_tamper_and_obsolete_schema(self):
        with tempfile.TemporaryDirectory() as name:
            directory = self.make_bundle(Path(name), "mlvc-s-psnr-v1")
            manifest = package.verify_package(directory)
            self.assertEqual(len(manifest["artifacts"]), 8)
            weights = directory / "aot/MLVCEncoder/weights.bin"
            weights.write_bytes(b"\x01\x3c")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                package.verify_package(directory)
            weights.write_bytes(b"\x00\x3c")
            graph_path = directory / "aot/MLVCEncoder/graph.pb"
            graph = read_message(graph_path, pb.Graph)
            graph.schema_version = 2
            write_message(graph_path, graph)
            bundle = read_message(directory / "model_bundle.pb", pb.ModelBundle)
            for artifact in bundle.artifacts:
                if artifact.path == "aot/MLVCEncoder/graph.pb":
                    artifact.size_bytes = graph_path.stat().st_size
                    artifact.sha256 = bytes.fromhex(package.sha256(graph_path))
            write_message(directory / "model_bundle.pb", bundle)
            with self.assertRaisesRegex(ValueError, "schema"):
                package.verify_package(directory)

    def test_embedding_is_reproducible_and_tracks_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for profile in package.load_json(package.PROFILE_PATH)["profiles"]:
                self.make_bundle(root, profile)
            args = argparse.Namespace(model_root=root / "canonical", output_data=root / "models.bin",
                output_index=root / "index.inc", depfile=root / "models.d")
            with contextlib.redirect_stdout(io.StringIO()):
                package.embed(args)
            first = args.output_data.read_bytes()
            index = args.output_index.read_text()
            with contextlib.redirect_stdout(io.StringIO()):
                package.embed(args)
            self.assertEqual(first, args.output_data.read_bytes())
            self.assertEqual(index, args.output_index.read_text())
            self.assertEqual(len(index.splitlines()), 18)
            self.assertIn("graph.pb", args.depfile.read_text())


if __name__ == "__main__":
    unittest.main()
