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

"""Download and verify the external assets required by BEHAVIOR."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from robots.behavior.dino_v2.encoder import (
    EXPECTED_SOURCE_ARCHIVE_SHA256,
    EXPECTED_WEIGHTS_SHA256,
)
from robots.behavior.policy_checkpoint import validate_policy_checkpoint

_DOWNLOAD_SNIPPET = """
import sys

from omnigibson.utils.asset_utils import (
    download_2025_challenge_task_instances,
    download_behavior_1k_assets,
    download_omnigibson_robot_assets,
)

actions = set(filter(None, sys.argv[1].split(",")))
accept_license = sys.argv[2] == "1"
if "robot" in actions:
    download_omnigibson_robot_assets()
if "behavior" in actions:
    download_behavior_1k_assets(accept_license=accept_license)
if "challenge" in actions:
    download_2025_challenge_task_instances()
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behavior-download-assets",
        description=(
            "Download BEHAVIOR assets through the official OmniGibson API, "
            "or verify an existing installation."
        ),
    )
    parser.add_argument(
        "--behavior-python",
        type=Path,
        default=None,
        help=(
            "Python from the BEHAVIOR venv. Defaults to BEHAVIOR_PYTHON, "
            "RPENT_BEHAVIOR_PYTHON, or $RPENT_REPRO_ROOT/venvs/behavior/bin/python."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="BEHAVIOR data root (defaults to OMNIGIBSON_DATA_PATH).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Pi0.5 checkpoint directory (defaults to PI05_CHECKPOINT_PATH).",
    )
    parser.add_argument(
        "--dino-source-archive",
        type=Path,
        default=None,
        help="Pinned DINOv2 source archive (defaults to DINOV2_SOURCE_ARCHIVE).",
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        default=None,
        help="DINOv2 ViT-S/14 weights (defaults to DINOV2_WEIGHTS).",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=(
            "Confirm acceptance of the BEHAVIOR data licence non-interactively. "
            "Without this flag, the official downloader prompts when needed."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not call a downloader for an asset directory already present.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing simulator, checkpoint, and DINO assets without downloading.",
    )
    return parser


def _env_path(value: Path | None, *names: str) -> Path | None:
    if value is not None:
        return value.expanduser().resolve()
    for name in names:
        configured = os.environ.get(name)
        if configured:
            return Path(configured).expanduser().resolve()
    return None


def _behavior_python(value: Path | None) -> Path:
    resolved = _env_path(value, "BEHAVIOR_PYTHON", "RPENT_BEHAVIOR_PYTHON")
    if resolved is None:
        repro_root = os.environ.get("RPENT_REPRO_ROOT")
        if repro_root:
            resolved = (
                Path(repro_root).expanduser().resolve()
                / "venvs"
                / "behavior"
                / "bin"
                / "python"
            )
    if resolved is None or not resolved.is_file():
        raise ValueError(
            "BEHAVIOR Python is unavailable; pass --behavior-python or set "
            "BEHAVIOR_PYTHON"
        )
    return resolved


def _require_data_path(value: Path | None) -> Path:
    resolved = _env_path(value, "OMNIGIBSON_DATA_PATH")
    if resolved is None:
        raise ValueError(
            "BEHAVIOR data root is unavailable; pass --data-path or set "
            "OMNIGIBSON_DATA_PATH"
        )
    return resolved


def _require_file(path: Path | None, *, label: str) -> Path:
    if path is None or not path.is_file():
        raise ValueError(f"missing required {label}: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(
    *,
    data_path: Path,
    checkpoint: Path | None,
    dino_source_archive: Path | None,
    dino_weights: Path | None,
) -> None:
    required_directories = (
        data_path / "behavior-1k-assets" / "scenes",
        data_path / "omnigibson-robot-assets",
        data_path / "2025-challenge-task-instances",
    )
    for path in required_directories:
        if not path.is_dir():
            raise ValueError(f"missing required directory: {path}")
    _require_file(data_path / "omnigibson.key", label="OmniGibson licence key")

    if checkpoint is None:
        raise ValueError(
            "Pi0.5 checkpoint is unavailable; pass --checkpoint or set "
            "PI05_CHECKPOINT_PATH"
        )
    validate_policy_checkpoint(checkpoint)

    source = _require_file(dino_source_archive, label="DINOv2 source archive")
    weights = _require_file(dino_weights, label="DINOv2 weights")
    source_sha256 = _sha256_file(source)
    if source_sha256 != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "DINOv2 source archive SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_ARCHIVE_SHA256}, got {source_sha256}"
        )
    weights_sha256 = _sha256_file(weights)
    if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            "DINOv2 weights SHA-256 mismatch: "
            f"expected {EXPECTED_WEIGHTS_SHA256}, got {weights_sha256}"
        )
    print("BEHAVIOR assets: OK")


def download_assets(
    *,
    behavior_python: Path,
    data_path: Path,
    accept_license: bool,
    skip_existing: bool,
) -> None:
    actions = ["robot", "behavior", "challenge"]
    if skip_existing:
        present = {
            "robot": (data_path / "omnigibson-robot-assets").is_dir(),
            "behavior": (
                (data_path / "behavior-1k-assets").is_dir()
                and (data_path / "omnigibson.key").is_file()
            ),
            "challenge": (data_path / "2025-challenge-task-instances").is_dir(),
        }
        actions = [name for name in actions if not present[name]]
    if not actions:
        print("BEHAVIOR assets already exist; nothing to download.")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMNIGIBSON_DATA_PATH"] = str(data_path)
    subprocess.run(
        [
            str(behavior_python),
            "-c",
            _DOWNLOAD_SNIPPET,
            ",".join(actions),
            "1" if accept_license else "0",
        ],
        check=True,
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        data_path = _require_data_path(args.data_path)
        if args.verify:
            verify_assets(
                data_path=data_path,
                checkpoint=_env_path(args.checkpoint, "PI05_CHECKPOINT_PATH"),
                dino_source_archive=_env_path(
                    args.dino_source_archive, "DINOV2_SOURCE_ARCHIVE"
                ),
                dino_weights=_env_path(args.dino_weights, "DINOV2_WEIGHTS"),
            )
        else:
            download_assets(
                behavior_python=_behavior_python(args.behavior_python),
                data_path=data_path,
                accept_license=bool(args.accept_license),
                skip_existing=bool(args.skip_existing),
            )
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
