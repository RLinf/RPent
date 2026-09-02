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

"""Opt-in real-environment smoke coverage for every Target50 task."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "robots" / "robocasa" / "eval" / "target50.json"


def _target50_tasks() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [task for split in manifest["splits"].values() for task in split["tasks"]]


class _DirectRpc:
    def __init__(self, facade) -> None:
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


@pytest.mark.parametrize("task_name", _target50_tasks())
def test_target50_environment_contract(task_name, monkeypatch):
    if os.environ.get("RPENT_RUN_ROBOCASA_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_ROBOCASA_INTEGRATION=1 to run RoboCasa smoke tests")

    pytest.importorskip("robocasa")
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.env_server import DEFAULT_CAMS, RoboCasaEnvFacade

    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("ROBOT_PLATFORM", "ROBOCASA")
    monkeypatch.delenv("RLDX_RESET_SEED", raising=False)

    facade = RoboCasaEnvFacade(
        task_name=task_name,
        split="target",
        seed=1,
        camera_h=64,
        camera_w=64,
    )
    try:
        assert facade.env.sim is None
        client = RoboCasaEnvClient(
            _DirectRpc(facade),
            expected_meta=facade.get_env_meta(),
        )

        assert facade.env.action_dim == 12
        task_language = client.get_task_language()
        assert isinstance(task_language, str) and task_language.strip()

        action = np.zeros(12, dtype=np.float64)
        client.step(action)
        assert isinstance(client.check_success(), bool)

        for camera_name in DEFAULT_CAMS:
            rgb = client.render_camera(camera_name)
            assert rgb.shape == (64, 64, 3)
            assert np.isfinite(rgb).all()

        nav_rgb, nav_depth = client.render_camera("navview", depth=True)
        nav_world = client.world_map("navview")
        assert nav_rgb.shape == (64, 64, 3)
        assert nav_depth.shape == (64, 64)
        assert nav_world.shape == (64, 64, 3)
        assert np.isfinite(nav_rgb).all()
        assert np.isfinite(nav_depth).all()
        assert np.isfinite(nav_world).all()
    finally:
        facade.close()
