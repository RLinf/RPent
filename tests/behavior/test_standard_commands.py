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

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.cli import behavior as behavior_cli
from robots.behavior import assets_cli, install_runtime


def test_behavior_console_scripts_are_registered() -> None:
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    assert 'behavior-download-assets = "rpent.cli.behavior:download_assets"' in pyproject
    assert 'behavior-install-runtime = "rpent.cli.behavior:install_runtime"' in pyproject


@pytest.mark.parametrize(
    ("entry", "module"),
    [
        (behavior_cli.download_assets, "robots.behavior.assets_cli"),
        (behavior_cli.install_runtime, "robots.behavior.install_runtime"),
    ],
)
def test_packaged_console_dispatches_to_source_plugin(monkeypatch, entry, module) -> None:
    calls: list[str] = []
    monkeypatch.setattr(behavior_cli, "_run_source_module", calls.append)

    assert entry() is None
    assert calls == [module]


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
                Path(install_runtime.__file__).with_name(
                    "install_behavior_runtime.sh"
                )
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


def test_assets_download_uses_behavior_python_and_official_api(tmp_path, monkeypatch) -> None:
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
