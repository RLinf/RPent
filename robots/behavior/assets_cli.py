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

"""Download and verify BEHAVIOR assets.

The downloader delegates simulator data installation to OmniGibson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
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

accept_license = sys.argv[1] == "1"
download_omnigibson_robot_assets()
download_behavior_1k_assets(accept_license=accept_license)
download_2025_challenge_task_instances()
"""

_DATASETS = (
    "omnigibson-robot-assets",
    "behavior-1k-assets",
    "2025-challenge-task-instances",
)
_DOWNLOAD_RECORD = ".rpent-behavior-download.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robots.behavior.assets_cli",
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
        help="Skip assets only when a matching completion record exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check the simulator download record, checkpoint, and DINO files.",
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
    configured = value
    if configured is None:
        for name in ("BEHAVIOR_PYTHON", "RPENT_BEHAVIOR_PYTHON"):
            if os.environ.get(name):
                configured = Path(os.environ[name])
                break
    if configured is None and os.environ.get("RPENT_REPRO_ROOT"):
        configured = Path(os.environ["RPENT_REPRO_ROOT"]) / "venvs/behavior/bin/python"
    # A venv's bin/python may be a symlink to the system interpreter.
    executable = configured.expanduser().absolute() if configured is not None else None
    if executable is None or not executable.is_file():
        raise ValueError(
            "BEHAVIOR Python is unavailable; pass --behavior-python or set "
            "BEHAVIOR_PYTHON"
        )
    return executable


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


def _dataset_inventory(path: Path) -> dict[str, int | str]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"missing or unsupported asset directory: {path}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted(path.rglob("*")):
        if ".git" in item.relative_to(path).parts:
            continue
        if item.is_symlink():
            raise ValueError(f"unsupported symlink in asset directory: {item}")
        if not item.is_file():
            continue
        size = item.stat().st_size
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    if not file_count or not total_bytes:
        raise ValueError(f"asset directory has no downloaded files: {path}")
    return {
        "files": file_count,
        "bytes": total_bytes,
        "path_size_sha256": digest.hexdigest(),
    }


def _current_download_record(data_path: Path) -> dict[str, object]:
    key = _require_file(data_path / "omnigibson.key", label="OmniGibson licence key")
    if key.stat().st_size == 0:
        raise ValueError(f"empty OmniGibson licence key: {key}")
    return {
        "format": 1,
        "datasets": {
            dataset: _dataset_inventory(data_path / dataset) for dataset in _DATASETS
        },
    }


def _check_download_record(data_path: Path, *, required: bool) -> bool:
    record_path = data_path / _DOWNLOAD_RECORD
    if record_path.is_symlink() or (record_path.exists() and not record_path.is_file()):
        raise ValueError(f"invalid BEHAVIOR asset completion record: {record_path}")
    if not record_path.is_file():
        if required or any(
            (data_path / dataset).exists() or (data_path / dataset).is_symlink()
            for dataset in _DATASETS
        ):
            raise ValueError(
                f"unverified BEHAVIOR assets at {data_path}: no completion record. "
                "Use a new data root for a fresh download; do not remove or "
                "overwrite existing assets without checking their ownership."
            )
        return False
    try:
        recorded = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid BEHAVIOR asset completion record: {record_path}"
        ) from error
    current = _current_download_record(data_path)
    if recorded != current:
        raise ValueError(
            f"BEHAVIOR asset inventory changed since download: {data_path}. "
            "Use a new data root or inspect the existing assets manually."
        )
    return True


def _write_download_record(data_path: Path) -> None:
    record_path = data_path / _DOWNLOAD_RECORD
    payload = json.dumps(_current_download_record(data_path), indent=2, sort_keys=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=data_path,
        prefix=f"{_DOWNLOAD_RECORD}.tmp-",
        delete=False,
    )
    temporary_path = Path(stream.name)
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, record_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_assets(
    *,
    data_path: Path,
    checkpoint: Path | None,
    dino_source_archive: Path | None,
    dino_weights: Path | None,
) -> None:
    _check_download_record(data_path, required=True)

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
    completed = _check_download_record(data_path, required=False)
    if completed and skip_existing:
        print("BEHAVIOR assets match their completion record; nothing to download.")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMNIGIBSON_DATA_PATH"] = str(data_path)
    subprocess.run(
        [
            str(behavior_python),
            "-c",
            _DOWNLOAD_SNIPPET,
            "1" if accept_license else "0",
        ],
        check=True,
        env=env,
    )
    _write_download_record(data_path)


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
