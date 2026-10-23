# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""BEHAVIOR asset downloads do not mistake a directory for a completed bundle."""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

from robots.behavior import assets_cli


def test_behavior_finds_the_official_rlinf_env_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robots.behavior import rlinf_env

    env_file = tmp_path / "rlinf/envs/sim/behavior/behavior_env.py"
    env_file.parent.mkdir(parents=True)
    env_file.touch()
    monkeypatch.setattr(rlinf_env, "get_rlinf_repo_path", lambda: tmp_path)

    assert rlinf_env.discover_rlinf_root() == tmp_path


def test_behavior_resolves_the_official_challenge_instance_layout(
    tmp_path: Path,
) -> None:
    from robots.behavior.rlinf_env import (
        _bootstrap_template_path,
        _resolve_activity_instance_dir,
    )

    dataset_root = tmp_path / "2025-challenge-task-instances"
    json_dir = dataset_root / "scenes" / "house" / "json"
    instance_dir = json_dir / "house_task_demo_instances"
    instance_dir.mkdir(parents=True)
    bootstrap = json_dir / "house_task_demo_0_0_template.json"
    bootstrap.write_text("{}")
    selected = instance_dir / "house_task_demo_0_242_template-tro_state.json"
    selected.write_text("{}")

    resolved = _resolve_activity_instance_dir(
        dataset_root,
        scene_model="house",
        task_name="demo",
        activity_definition_id=0,
        activity_instance_id=242,
    )

    assert resolved == instance_dir
    assert (
        _bootstrap_template_path(
            resolved,
            scene_model="house",
            task_name="demo",
            activity_definition_id=0,
        )
        == bootstrap
    )


def test_behavior_python_keeps_the_venv_interpreter_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_path = tmp_path / "behavior"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_path)
    executable = venv_path / "bin/python"
    assert executable.is_symlink()
    monkeypatch.setenv("BEHAVIOR_PYTHON", str(executable))

    selected = assets_cli._behavior_python(None)

    assert selected == executable.absolute()
    assert (
        Path(
            subprocess.check_output(
                [str(selected), "-c", "import sys; print(sys.prefix)"], text=True
            ).strip()
        )
        == venv_path
    )
    assert selected != Path(sys.executable).resolve()


@pytest.mark.parametrize("has_partial_file", [False, True])
@pytest.mark.parametrize("skip_existing", [False, True])
def test_existing_unverified_assets_are_not_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_partial_file: bool,
    skip_existing: bool,
) -> None:
    data_path = tmp_path / "data"
    partial = data_path / "behavior-1k-assets"
    partial.mkdir(parents=True)
    if has_partial_file:
        (partial / "VERSION").write_text("3.7.2rc1", encoding="utf-8")
    monkeypatch.setattr(
        assets_cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("existing assets must not be overwritten"),
    )

    with pytest.raises(ValueError, match="unverified BEHAVIOR assets"):
        assets_cli.download_assets(
            behavior_python=Path(sys.executable),
            data_path=data_path,
            accept_license=True,
            skip_existing=skip_existing,
        )

    assert partial.is_dir()


def test_download_record_detects_missing_or_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data"
    downloaded_files = [
        data_path / dataset / "payload" for dataset in assets_cli._DATASETS
    ]
    calls = []

    def official_download(command, *, check, env):
        calls.append(command)
        assert check is True
        assert env["OMNIGIBSON_DATA_PATH"] == str(data_path)
        for path in downloaded_files:
            path.parent.mkdir(parents=True)
            path.write_bytes(b"official data")
        (data_path / "omnigibson.key").write_bytes(b"licence key")

    monkeypatch.setattr(assets_cli.subprocess, "run", official_download)
    assets_cli.download_assets(
        behavior_python=Path(sys.executable),
        data_path=data_path,
        accept_license=True,
        skip_existing=True,
    )
    assert len(calls) == 1

    assets_cli.download_assets(
        behavior_python=Path(sys.executable),
        data_path=data_path,
        accept_license=True,
        skip_existing=True,
    )
    assert len(calls) == 1

    downloaded_files[0].write_bytes(b"truncated")
    with pytest.raises(ValueError, match="inventory changed"):
        assets_cli.verify_assets(
            data_path=data_path,
            checkpoint=None,
            dino_source_archive=None,
            dino_weights=None,
        )
    with pytest.raises(ValueError, match="inventory changed"):
        assets_cli.download_assets(
            behavior_python=Path(sys.executable),
            data_path=data_path,
            accept_license=True,
            skip_existing=True,
        )
    assert len(calls) == 1


def test_failed_download_does_not_create_completion_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data"

    def interrupted_download(command, *, check, env):
        (data_path / "behavior-1k-assets").mkdir()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(assets_cli.subprocess, "run", interrupted_download)
    with pytest.raises(subprocess.CalledProcessError):
        assets_cli.download_assets(
            behavior_python=Path(sys.executable),
            data_path=data_path,
            accept_license=True,
            skip_existing=True,
        )
    assert not (data_path / assets_cli._DOWNLOAD_RECORD).exists()
    with pytest.raises(ValueError, match="unverified BEHAVIOR assets"):
        assets_cli.verify_assets(
            data_path=data_path,
            checkpoint=None,
            dino_source_archive=None,
            dino_weights=None,
        )
