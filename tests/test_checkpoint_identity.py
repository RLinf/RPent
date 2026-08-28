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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robots.robocasa import checkpoint_identity


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model"
    vlm = tmp_path / "vlm"
    model.mkdir()
    vlm.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"frozen-weights")
    index = model / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {"layer": shard.name},
            }
        ),
        encoding="utf-8",
    )
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (vlm / "config.json").write_text("{}\n", encoding="utf-8")
    files = {}
    for prefix, root in (("model", model), ("vlm", vlm)):
        for path in sorted(root.iterdir()):
            files[f"{prefix}/{path.name}"] = {
                "size": path.stat().st_size,
                "sha256": _digest(path),
            }
    authority = tmp_path / "checkpoint_manifest.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_id": "RLDX-1-FT-RC365",
                "algorithm": "sha256",
                "weight_total_size": shard.stat().st_size,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(checkpoint_identity, "AUTHORITY_PATH", authority)
    return model, vlm


def test_checkpoint_attestation_binds_content_and_file_identity(tmp_path, monkeypatch):
    model, vlm = _fixture(tmp_path, monkeypatch)
    attestation = checkpoint_identity.verify_checkpoint(model, vlm)
    assert len(attestation["fingerprint"]) == 64
    assert checkpoint_identity.validate_attestation(attestation, model, vlm)

    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(attestation), encoding="utf-8")
    path.chmod(0o600)
    assert checkpoint_identity.load_attestation(path, model, vlm) == attestation

    (model / "config.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since verification"):
        checkpoint_identity.load_attestation(path, model, vlm)


def test_checkpoint_verification_rejects_hash_and_shard_set_changes(
    tmp_path, monkeypatch
):
    model, vlm = _fixture(tmp_path, monkeypatch)
    (model / "config.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint_identity.verify_checkpoint(model, vlm)

    model, vlm = _fixture(tmp_path / "other", monkeypatch)
    (model / "extra.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError, match="shard set mismatch"):
        checkpoint_identity.verify_checkpoint(model, vlm)


def test_checkpoint_verification_rejects_symlinked_root(tmp_path, monkeypatch):
    model, vlm = _fixture(tmp_path, monkeypatch)
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        checkpoint_identity.verify_checkpoint(alias, vlm)
