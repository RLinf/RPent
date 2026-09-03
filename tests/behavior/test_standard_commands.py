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
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior import assets_cli, install_runtime, policy_checkpoint, runtime
from rpent.robots.components.pi05_vla_server import _validate_checkpoint_manifest


def test_behavior_console_scripts_are_registered() -> None:
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert 'behavior-download-assets = "robots.behavior.assets_cli:main"' in pyproject
    assert (
        'behavior-install-runtime = "robots.behavior.install_runtime:main"' in pyproject
    )
    assert 'include = ["rpent*", "robots", "robots.behavior*"]' in pyproject
    assert "robots.libero*" not in pyproject
    assert "robots.robocasa*" not in pyproject
    assert "robots.robotwin*" not in pyproject
    assert "rpent.cli." + "behavior" not in pyproject


def test_behavior_standard_command_modules_document_source_checkout_contract() -> None:
    assert "requires an RPent source checkout" in (assets_cli.__doc__ or "")
    assert "requires an RPent source checkout" in (install_runtime.__doc__ or "")


@pytest.mark.parametrize("entry", [assets_cli.main, install_runtime.main])
def test_behavior_console_scripts_have_help(entry) -> None:
    with pytest.raises(SystemExit) as raised:
        entry(["--help"])
    assert raised.value.code == 0


def test_install_runtime_delegates_to_packaged_shell(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert install_runtime.main([]) == 7
    assert calls == [
        [
            "bash",
            str(
                Path(install_runtime.__file__).with_name("install_behavior_runtime.sh")
            ),
        ]
    ]


def _make_simulator_layout(root: Path) -> None:
    (root / "behavior-1k-assets" / "scenes").mkdir(parents=True)
    (root / "omnigibson-robot-assets").mkdir()
    (root / "2025-challenge-task-instances").mkdir()
    (root / "omnigibson.key").write_bytes(b"key")


def test_assets_verify_checks_layout_checkpoint_and_dino(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "data"
    _make_simulator_layout(data_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = tmp_path / "dinov2-source.tar.gz"
    weights = tmp_path / "dinov2-vits14.pth"
    source.write_bytes(b"source")
    weights.write_bytes(b"weights")

    validated: list[Path] = []
    monkeypatch.setattr(
        assets_cli,
        "validate_policy_checkpoint",
        lambda path: validated.append(path),
    )
    hashes = {
        source: assets_cli.EXPECTED_SOURCE_ARCHIVE_SHA256,
        weights: assets_cli.EXPECTED_WEIGHTS_SHA256,
    }
    monkeypatch.setattr(assets_cli, "_sha256_file", hashes.__getitem__)

    assets_cli.verify_assets(
        data_path=data_path,
        checkpoint=checkpoint,
        dino_source_archive=source,
        dino_weights=weights,
    )
    assert validated == [checkpoint]


def test_assets_download_skip_existing_avoids_behavior_subprocess(
    tmp_path, monkeypatch
) -> None:
    data_path = tmp_path / "data"
    _make_simulator_layout(data_path)
    behavior_python = tmp_path / "python"
    behavior_python.write_text("")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("all existing assets must skip the subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    assets_cli.download_assets(
        behavior_python=behavior_python,
        data_path=data_path,
        accept_license=False,
        skip_existing=True,
    )


def test_assets_download_uses_behavior_python_and_official_api(
    tmp_path, monkeypatch
) -> None:
    data_path = tmp_path / "data"
    behavior_python = tmp_path / "behavior-python"
    behavior_python.write_text("")
    captured: dict[str, object] = {}

    def fake_run(command, *, check, env):
        captured.update(command=command, check=check, env=env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assets_cli.download_assets(
        behavior_python=behavior_python,
        data_path=data_path,
        accept_license=True,
        skip_existing=True,
    )

    command = captured["command"]
    assert command[0] == str(behavior_python)
    assert command[1] == "-c"
    assert command[3:] == ["robot,behavior,challenge", "1"]
    assert captured["check"] is True
    assert captured["env"]["OMNIGIBSON_DATA_PATH"] == str(data_path)


def test_pi05_server_uses_generic_checkpoint_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.bin"
    model.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    manifest = {
        "schema_version": 1,
        "profile_id": "test-profile",
        "resolved_path": str(checkpoint.resolve()),
        "files": {
            "model.bin": {
                "size_bytes": len(b"fixture"),
                "sha256": digest,
            }
        },
        "binding_sha256": "test-binding",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved, binding = _validate_checkpoint_manifest(checkpoint, manifest_path)

    assert resolved == str(checkpoint.resolve())
    assert binding == manifest


def test_policy_checkpoint_writes_validated_binding_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    payload = {
        "schema_version": 1,
        "profile_id": "fixture",
        "resolved_path": str(tmp_path / "checkpoint"),
        "files": {},
        "binding_sha256": "fixture-binding",
    }
    binding = SimpleNamespace(as_dict=lambda: payload)
    validated: list[Path] = []

    def fake_validate(path: Path):
        validated.append(path)
        return binding

    monkeypatch.setattr(policy_checkpoint, "validate_policy_checkpoint", fake_validate)
    destination = tmp_path / "runtime" / "checkpoint-manifest.json"

    result = policy_checkpoint.write_policy_checkpoint_manifest(
        destination,
        tmp_path / "checkpoint",
    )

    assert result is binding
    assert validated == [tmp_path / "checkpoint"]
    assert json.loads(destination.read_text(encoding="utf-8")) == payload


def test_behavior_vla_spawn_passes_generated_checkpoint_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeDaemon:
        def __init__(self, name, cmd, *, env_overrides, log_path):
            captured.update(
                name=name,
                cmd=cmd,
                env_overrides=env_overrides,
                log_path=log_path,
            )
            self.started = False

        def start(self) -> None:
            self.started = True

    def fake_write_manifest(destination: Path, checkpoint: Path):
        captured.update(manifest=destination, checkpoint=checkpoint)

    rpc = object()
    monkeypatch.setattr(runtime, "ProcessDaemon", FakeDaemon)
    monkeypatch.setattr(runtime, "pick_free_port", lambda: 45678)
    monkeypatch.setattr(runtime, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr(
        runtime,
        "write_policy_checkpoint_manifest",
        fake_write_manifest,
    )
    checkpoint = tmp_path / "checkpoint"
    args = SimpleNamespace(
        vla_endpoint=None,
        behavior_model_cuda_device=None,
        cuda_device=None,
        behavior_python=Path(__file__),
        policy_checkpoint=checkpoint,
    )

    daemon, returned_rpc = runtime._spawn_vla_server(args, tmp_path / "output")

    manifest = tmp_path / "output" / "policy_checkpoint_manifest.json"
    assert daemon.started is True
    assert returned_rpc is rpc
    assert captured["manifest"] == manifest
    assert captured["checkpoint"] == checkpoint
    command = captured["cmd"]
    manifest_index = command.index("--checkpoint-manifest")
    assert command[manifest_index + 1] == str(manifest)
