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

"""Focused tests for the minimal RoboCasa integration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import RoboCasaEnvFacade
from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.robot_spec import _parse_config, get_robot_spec
from rpent.memory import MemoryManager
from rpent.prompt.utils import format_prompt
from rpent.utils.resources import ensure_resources


class _DirectRpc:
    def __init__(self, facade: RoboCasaEnvFacade) -> None:
        self.facade = facade

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        timeout_s: float | None = None,
    ):
        del timeout_s
        handler = getattr(self.facade, method.removeprefix("env."))
        return handler(*args, **(kwargs or {}))


def _args(
    tmp_path: Path,
    *,
    memory_dir: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        task_name="OpenDrawer",
        split="target",
        seed=1,
        output_dir=tmp_path / "run",
        memory_dir=memory_dir,
    )


def test_parse_config_uses_default_results_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))

    config = _parse_config(
        _args(tmp_path, memory_dir=None),
    )

    assert config.prompt_vars["memory_dir"] == str(
        tmp_path / "resources" / "robocasa" / "results"
    )


def test_default_resources_sync_uses_robocasa_subtree(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("RPENT_RESOURCES_HF_REPO", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resources_dir = ensure_resources(get_robot_spec())

    assert resources_dir == tmp_path / "resources" / "robocasa"
    assert calls == {
        "repo_id": "RLinf/RPent-memory",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "resources"),
        "allow_patterns": ["robocasa/**"],
    }


def test_results_corpus_is_readable_through_memory_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    robot_resources = tmp_path / "resources" / "robocasa"
    results_dir = robot_resources / "results"
    results_dir.mkdir(parents=True)
    audit = results_dir / "OpenDrawer_s0.json"
    audit.write_text('{"success": true}\n')

    manager = MemoryManager(root=robot_resources / "memory")
    read_text_file = manager.get_common_tool_bindings()["read_text_file"][1]

    assert read_text_file(path=str(audit))["content"] == '{"success": true}\n'


def test_parse_config_resolves_local_results_dir(tmp_path):
    memory_dir = tmp_path / "local-memory"

    config = _parse_config(
        _args(tmp_path, memory_dir=memory_dir),
    )

    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())


def test_prompt_names_only_current_task_memory(tmp_path):
    memory_dir = tmp_path / "results"
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "memory_dir": str(memory_dir),
        },
    )

    assert str(memory_dir / "OpenDrawer_s0.json") in rendered
    assert str(memory_dir / "recipe_OpenDrawer_s0.jsonl") in rendered
    assert str(memory_dir / "OpenDrawer.md") in rendered
    assert "read every existing file" in rendered
    assert "ArrangeTea_s0" not in rendered
    assert "GLOBAL_MEMORY" not in rendered
    assert "{{" not in rendered


def test_real_navview_follows_mobile_base(monkeypatch):
    if os.environ.get("RPENT_RUN_ROBOCASA_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_ROBOCASA_INTEGRATION=1 for the GPU integration test")

    pytest.importorskip("robocasa")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("ROBOT_PLATFORM", "ROBOCASA")
    monkeypatch.delenv("RLDX_RESET_SEED", raising=False)

    facade = RoboCasaEnvFacade(
        task_name="OpenDrawer",
        split="target",
        seed=1,
        camera_h=256,
        camera_w=256,
    )
    try:
        assert facade.env.sim is None
        client = RoboCasaEnvClient(
            _DirectRpc(facade),
            expected_meta=facade.get_env_meta(),
        )

        rgb_before, depth = client.render_camera("navview", depth=True)
        world = client.world_map("navview")
        meta_before = client.get_camera_meta("navview")

        assert rgb_before.shape == (256, 256, 3)
        assert depth.shape == (256, 256)
        assert world.shape == (256, 256, 3)
        assert np.isfinite(rgb_before).all()
        assert np.isfinite(depth).all()
        assert np.isfinite(world).all()

        action = np.zeros(int(facade.env.action_dim), dtype=np.float64)
        assert action.shape == (12,)
        action[6] = -1.0
        action[7] = 1.0
        action[11] = 1.0
        for _ in range(8):
            client.step(action)

        rgb_after = client.render_camera("navview")
        meta_after = client.get_camera_meta("navview")
        pose_before = np.asarray(meta_before["extrinsic_cam2world"])
        pose_after = np.asarray(meta_after["extrinsic_cam2world"])

        assert np.linalg.norm(pose_after[:3, 3] - pose_before[:3, 3]) > 1e-4
        assert not np.array_equal(rgb_before, rgb_after)
    finally:
        close = getattr(facade.env, "close", None)
        if close is not None:
            close()
