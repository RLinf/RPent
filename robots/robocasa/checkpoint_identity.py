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

"""Content identity and reusable attestation for the frozen RLDX checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

AUTHORITY_PATH = Path(__file__).with_name("checkpoint_manifest.json")
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authority() -> tuple[dict[str, Any], str]:
    raw = AUTHORITY_PATH.read_bytes()
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("checkpoint_id") != "RLDX-1-FT-RC365"
        or value.get("algorithm") != "sha256"
        or not isinstance(value.get("files"), dict)
    ):
        raise ValueError("unsupported checkpoint authority manifest")
    for name, record in value["files"].items():
        path = Path(name)
        if (
            path.as_posix() != name
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in {"model", "vlm"}
            or not isinstance(record, dict)
            or type(record.get("size")) is not int
            or record["size"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"invalid checkpoint authority entry: {name!r}")
    return value, _sha256_bytes(raw)


def expected_fingerprint() -> str:
    manifest, _ = _authority()
    payload = {
        "checkpoint_id": manifest["checkpoint_id"],
        "files": manifest["files"],
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return _sha256_bytes(canonical)


def authority_manifest() -> dict[str, Any]:
    """Return the validated immutable authority manifest."""
    manifest, _ = _authority()
    return manifest


def _root_problem(path: Path, label: str) -> str | None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        return f"cannot inspect {label}: {exc}"
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return f"{label} must be a real directory"
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        return f"{label} must be owned by the current user and not group/other writable"
    return None


def _file_metadata(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"checkpoint input must be a regular non-symlink file: {path}")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise ValueError(
            f"checkpoint input must be owned by the current user and not writable "
            f"by group/other: {path}"
        )
    return metadata


def _stat_record(metadata: os.stat_result) -> dict[str, int]:
    return {name: int(getattr(metadata, name)) for name in STAT_FIELDS}


def _path_for(name: str, model_root: Path, vlm_root: Path) -> Path:
    root = model_root if name.startswith("model/") else vlm_root
    return root / name.split("/", 1)[1]


def _validate_weight_index(model_root: Path, manifest: dict[str, Any]) -> None:
    declared = {
        Path(name).name
        for name in manifest["files"]
        if name.startswith("model/") and name.endswith(".safetensors")
    }
    actual = {path.name for path in model_root.glob("*.safetensors")}
    if actual != declared:
        raise ValueError(
            f"checkpoint shard set mismatch: expected={sorted(declared)}, "
            f"actual={sorted(actual)}"
        )
    index_path = model_root / "model.safetensors.index.json"
    _file_metadata(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8", errors="strict"))
    indexed = set(index.get("weight_map", {}).values())
    if indexed != declared:
        raise ValueError("checkpoint index shard set differs from authority")
    total = sum(manifest["files"][f"model/{name}"]["size"] for name in declared)
    if (
        index.get("metadata", {}).get("total_size") != total
        or manifest.get("weight_total_size") != total
    ):
        raise ValueError("checkpoint index total_size differs from authority")


def verify_checkpoint(
    model_root: Path,
    vlm_root: Path,
    *,
    hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Hash every authority input once and return a reusable attestation."""
    model_input = Path(model_root)
    vlm_input = Path(vlm_root)
    for root, label in ((model_input, "model root"), (vlm_input, "VLM root")):
        if problem := _root_problem(root, label):
            raise ValueError(problem)
    model_root = model_input.resolve()
    vlm_root = vlm_input.resolve()
    manifest, authority_hash = _authority()
    _validate_weight_index(model_root, manifest)
    files: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(manifest["files"].items()):
        path = _path_for(name, model_root, vlm_root)
        before = _file_metadata(path)
        if before.st_size != expected["size"]:
            raise ValueError(f"checkpoint size mismatch: {name}")
        digest = hasher(path)
        after = _file_metadata(path)
        if _stat_record(before) != _stat_record(after):
            raise ValueError(f"checkpoint input changed while hashing: {name}")
        if digest != expected["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {name}")
        files[name] = {
            "size": expected["size"],
            "sha256": digest,
            "stat": _stat_record(after),
        }
    return {
        "schema_version": 1,
        "checkpoint_id": manifest["checkpoint_id"],
        "authority_manifest_sha256": authority_hash,
        "fingerprint": expected_fingerprint(),
        "model_root": str(model_root),
        "vlm_root": str(vlm_root),
        "files": files,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def validate_attestation(
    value: Any, model_root: Path, vlm_root: Path
) -> dict[str, Any]:
    """Validate an attestation and cheaply prove its inputs have not changed."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint attestation")
    manifest, authority_hash = _authority()
    model_input = Path(model_root)
    vlm_input = Path(vlm_root)
    for root, label in ((model_input, "model root"), (vlm_input, "VLM root")):
        if problem := _root_problem(root, label):
            raise ValueError(problem)
    model_root = model_input.resolve()
    vlm_root = vlm_input.resolve()
    if value.get("checkpoint_id") != manifest["checkpoint_id"]:
        raise ValueError("checkpoint attestation id mismatch")
    if value.get("authority_manifest_sha256") != authority_hash:
        raise ValueError("checkpoint authority manifest mismatch")
    if value.get("fingerprint") != expected_fingerprint():
        raise ValueError("checkpoint fingerprint mismatch")
    if value.get("model_root") != str(model_root) or value.get("vlm_root") != str(
        vlm_root
    ):
        raise ValueError("checkpoint attestation root mismatch")
    records = value.get("files")
    if not isinstance(records, dict) or set(records) != set(manifest["files"]):
        raise ValueError("checkpoint attestation file set mismatch")
    _validate_weight_index(model_root, manifest)
    for name, expected in manifest["files"].items():
        record = records.get(name)
        if (
            not isinstance(record, dict)
            or record.get("size") != expected["size"]
            or record.get("sha256") != expected["sha256"]
        ):
            raise ValueError(f"checkpoint attestation content mismatch: {name}")
        current = _file_metadata(_path_for(name, model_root, vlm_root))
        if record.get("stat") != _stat_record(current):
            raise ValueError(f"checkpoint input changed since verification: {name}")
        if not name.endswith(".safetensors") and sha256_file(
            _path_for(name, model_root, vlm_root)
        ) != record.get("sha256"):
            raise ValueError(f"checkpoint input changed since verification: {name}")
    return value


def load_attestation(path: Path, model_root: Path, vlm_root: Path) -> dict[str, Any]:
    """Securely read and validate a private checkpoint attestation file."""
    path = Path(path)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("checkpoint attestation must be owned, regular, and mode 0600")
    if metadata.st_size <= 0 or metadata.st_size > 16 * 1024 * 1024:
        raise ValueError("checkpoint attestation has an invalid size")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_record(opened) != _stat_record(metadata):
            raise ValueError("checkpoint attestation changed while opening")
        raw = os.read(descriptor, metadata.st_size + 1)
        after = os.fstat(descriptor)
        if _stat_record(opened) != _stat_record(after) or len(raw) != metadata.st_size:
            raise ValueError("checkpoint attestation changed while reading")
    finally:
        os.close(descriptor)
    value = json.loads(raw.decode("utf-8", errors="strict"))
    return validate_attestation(value, model_root, vlm_root)
