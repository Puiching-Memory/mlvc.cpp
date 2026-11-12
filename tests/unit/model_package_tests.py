"""Exercise AOT-only bundle assembly, integrity checks, and deterministic embedding."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import model_package as package


class PackageTests(unittest.TestCase):
    def make_bundle(self, root, profile):
        registration = package.load_json(package.PROFILE_PATH)["profiles"][profile]
        source = root / "source" / profile
        source.mkdir(parents=True)
        metadata = dict(name=profile, params=dict(
            full_model_params=dict(model_version=registration["model_version"]),
            split_model_params=dict(model_width=64, model_height=64)))
        (source / "metadata.json").write_text(json.dumps(metadata))
        for name in ("gaussian_pmf.json", "bit_estimator_pmf.json"):
            (source / name).write_text("{}")
        for part in ("MLVCEncoder", "MLVCDecoder"):
            directory = source / "aot" / part
            directory.mkdir(parents=True)
            (directory / "graph.json").write_text(json.dumps(dict(
                schema_version=2, source="checkpoint", source_sha256=registration["weights_sha256"])))
            (directory / "weights.bin").write_bytes(b"weights")
        (source / "aot/manifest.json").write_text(json.dumps(dict(source_sha256=registration["weights_sha256"])))
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
            weights.write_bytes(b"tamper!")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                package.verify_package(directory)
            weights.write_bytes(b"weights")
            graph_path = directory / "aot/MLVCEncoder/graph.json"
            graph = package.load_json(graph_path)
            graph["schema_version"] = 1
            graph_path.write_text(json.dumps(graph))
            record = manifest["artifacts"]["aot/MLVCEncoder/graph.json"]
            record.update(bytes=graph_path.stat().st_size, sha256=package.sha256(graph_path))
            (directory / "model_bundle.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema 2"):
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
            self.assertIn("weights.bin", args.depfile.read_text())


if __name__ == "__main__":
    unittest.main()
