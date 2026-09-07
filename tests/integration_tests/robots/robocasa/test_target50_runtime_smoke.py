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


def _target50_cells() -> list[tuple[str, int]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        (task, seed)
        for split in manifest["splits"].values()
        for task in split["tasks"]
        for seed in split["seeds"]
    ]


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


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("task_name", "seed"),
    _target50_cells(),
    ids=lambda value: str(value),
)
def test_target50_environment_contract(task_name, seed, monkeypatch):
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
        seed=seed,
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


@pytest.mark.timeout(180)
def test_nav_camera_follows_mobile_base(monkeypatch):
    if os.environ.get("RPENT_RUN_ROBOCASA_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_ROBOCASA_INTEGRATION=1 to run RoboCasa smoke tests")

    pytest.importorskip("robocasa")
    from robots.robocasa.env_client import RoboCasaEnvClient
    from robots.robocasa.env_server import RoboCasaEnvFacade

    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("ROBOT_PLATFORM", "ROBOCASA")
    monkeypatch.delenv("RLDX_RESET_SEED", raising=False)

    facade = RoboCasaEnvFacade(
        task_name="OpenDrawer",
        split="target",
        seed=1,
        camera_h=64,
        camera_w=64,
    )
    try:
        client = RoboCasaEnvClient(
            _DirectRpc(facade),
            expected_meta=facade.get_env_meta(),
        )
        base_before = np.asarray(
            client.current_raw_obs["robot0_base_pos"], dtype=np.float64
        )
        camera_before = np.asarray(
            client.get_camera_meta("navview")["extrinsic_cam2world"],
            dtype=np.float64,
        )
        image_before = client.render_camera("navview").astype(np.float64)

        action = np.zeros(12, dtype=np.float64)
        action[6] = 1.0
        action[7] = 0.2
        action[11] = 1.0
        for _ in range(8):
            client.step(action)

        base_after = np.asarray(
            client.current_raw_obs["robot0_base_pos"], dtype=np.float64
        )
        camera_after = np.asarray(
            client.get_camera_meta("navview")["extrinsic_cam2world"],
            dtype=np.float64,
        )
        image_after = client.render_camera("navview").astype(np.float64)

        assert np.linalg.norm(base_after - base_before) > 1e-4
        assert np.linalg.norm(camera_after - camera_before) > 1e-4
        assert np.mean(np.abs(image_after - image_before)) > 0.1
    finally:
        facade.close()
