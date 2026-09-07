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

"""Build the BEHAVIOR DINO episode catalog from official demonstration data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from robots.behavior.sft_offline_converter import compile_runtime_catalog


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behavior-build-memory",
        description=(
            "Build the derived BEHAVIOR DINO episode-memory catalog from "
            "official demonstration media and reviewed episode rollups."
        ),
    )
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path, action="append")
    parser.add_argument("--rollups-dir", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--cuda-device",
        required=True,
        help="Single CUDA device id made visible to the DINO compiler.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _single_cuda_device(value: str) -> str:
    device = str(value).strip()
    if not device or "," in device or not device.isdigit():
        raise ValueError("--cuda-device must be one numeric device id")
    return device


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = _single_cuda_device(args.cuda_device)
        result = compile_runtime_catalog(
            selection_manifest=args.selection_manifest.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            video_roots=tuple(path.expanduser().resolve() for path in args.video_root),
            rollups_dir=args.rollups_dir.expanduser().resolve(),
            source_archive=args.source_archive.expanduser().resolve(),
            weights=args.weights.expanduser().resolve(),
            cache_dir=None
            if args.cache_dir is None
            else args.cache_dir.expanduser().resolve(),
            batch_size=int(args.batch_size),
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
