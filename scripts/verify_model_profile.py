#!/usr/bin/env python3
"""Strictly validate a checkpoint against a registered deployment profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "third_party" / "mlvc" / "video"
sys.path.insert(0, str(VIDEO_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from conversion._full_model import _model_factory  # noqa: E402
from model_profiles import configure_model_factory, verify_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    verify_checkpoint(args.model_version, checkpoint)
    configure_model_factory(_model_factory)
    model = _model_factory.full_model_factory(
        args.model_version,
        weights_path=checkpoint,
    )
    params = model.model_params
    print(
        f"verified {params.model_version}: feature={params.feature_channels}, "
        f"y={params.latent_channels}, z={params.hyperprior_channels}, "
        f"hyper-downsample={params.downsample_hyperprior}"
    )


if __name__ == "__main__":
    main()
