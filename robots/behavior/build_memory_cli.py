# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rebuild the BEHAVIOR DINO cache from the official MemoryManager corpus."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from robots.behavior.dino_v2.encoder import (
    EXPECTED_SOURCE_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    Dinov2DeploymentPaths,
    Dinov2Engine,
    Dinov2RevisionIdentity,
    _sha256_file,
)
from robots.behavior.dino_v2.index import rebuild_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robots.behavior.build_memory_cli",
        description=(
            "Rebuild the derived BEHAVIOR DINO cache from the official "
            "MemoryManager corpus under --memory-dir."
        ),
    )
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--cuda-device",
        required=True,
        help="Single CUDA device id made visible to the DINO compiler.",
    )
    return parser


def _single_cuda_device(value: str) -> str:
    device = str(value).strip()
    if not device or "," in device or not device.isdigit():
        raise ValueError("--cuda-device must be one numeric device id")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    encoder = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = _single_cuda_device(args.cuda_device)
        source_archive = args.source_archive.expanduser().resolve()
        weights = args.weights.expanduser().resolve()
        # Heavy imports stay in the CLI path; the DINO index module itself remains
        # a lightweight derived-cache reader/writer.
        import torch
        import torchvision

        identity = Dinov2RevisionIdentity(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            source_commit=EXPECTED_SOURCE_COMMIT,
            source_archive_sha256=_sha256_file(source_archive, label="source_archive"),
            weights_sha256=_sha256_file(weights, label="weights"),
            torch_version=str(torch.__version__),
            torchvision_version=str(torchvision.__version__),
            device="cuda",
        )
        encoder = Dinov2Engine(
            identity,
            Dinov2DeploymentPaths(
                source_archive_path=source_archive,
                weights_path=weights,
                cache_dir=None
                if args.cache_dir is None
                else args.cache_dir.expanduser().resolve(),
            ),
        )
        result = rebuild_index(args.memory_dir.expanduser().resolve(), encoder)
    except ValueError as error:
        parser.error(str(error))
    finally:
        if encoder is not None:
            encoder.close()
    print(json.dumps(dict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
