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

"""RoboCasa localization capture RPC, client aliases and step artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa import tools
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import RoboCasaEnvFacade
from rpent.session import EnvState
from rpent.tools.localization.localization import LocalizationFrame


def _sim():
    names = {"robot0_agentview_left": 0, "mobilebase0_navview": 1}
    xmat = np.eye(3).reshape(1, 9).repeat(2, axis=0)
    data = SimpleNamespace(cam_xpos=np.zeros((2, 3)), cam_xmat=xmat)

    def render(*, width, height, camera_name, depth):
        assert depth and camera_name in names
        return np.zeros((height, width, 3), np.uint8), np.full(
            (height, width), 0.5, np.float32
        )

    model = SimpleNamespace(
        camera_name2id=names.__getitem__,
        cam_fovy=np.array([60.0, 75.0]),
        stat=SimpleNamespace(extent=1.0),
        tex_type=np.array([2]),
        vis=SimpleNamespace(
            map=SimpleNamespace(znear=0.01, zfar=50.0),
            quality=SimpleNamespace(numslices=28),
        ),
    )
    return SimpleNamespace(model=model, data=data, render=render)


def test_facade_captures_current_pose_of_moving_cameras():
    sim = _sim()
    sim.data.cam_xpos[1] = [2.0, 0, 1]
    facade = RoboCasaEnvFacade.__new__(RoboCasaEnvFacade)
    facade.env = SimpleNamespace(sim=sim, timestep=12)
    facade.camera_h, facade.camera_w = 4, 6
    records = facade.capture_localization(
        ["robot0_agentview_left", ("mobilebase0_navview", 8, 8)]
    )
    first = [LocalizationFrame.from_canonical(record) for record in records]
    assert first[0].shape == (4, 6) and first[1].shape == (8, 8)
    assert first[0].metadata["env_step"] == 12
    sim.data.cam_xpos[1] = [2.5, 0.3, 1]  # the base moved the navigation camera
    moved = LocalizationFrame.from_canonical(
        facade.capture_localization([("mobilebase0_navview", 8, 8)])[0]
    )
    np.testing.assert_allclose(moved.T_world_from_cv[:3, 3], [2.5, 0.3, 1])


class _Rpc:
    def __init__(self):
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs))
        if method == "env.get_env_meta":
            return {"camera_h": 4, "camera_w": 6}
        if method == "env.reset":
            return {}
        assert method == "env.capture_localization"
        records = []
        for request in kwargs["cameras"]:
            shape = (request["height"] or 4, request["width"] or 6)
            records.append(
                {
                    "camera": request["camera"],
                    "rgb": np.zeros((*shape, 3), np.uint8),
                    "depth_m": np.ones(shape),
                    "intrinsic_K": np.eye(3),
                    "T_world_from_cv": np.eye(4),
                }
            )
        return records


def test_client_resolves_aliases_and_world_map_uses_capture():
    rpc = _Rpc()
    client = RoboCasaEnvClient(rpc, expected_meta={"camera_h": 4, "camera_w": 6})
    frames = client.capture_frames(["wrist", ("navview", 8, 8)])
    assert [frame.camera for frame in frames] == [
        "robot0_eye_in_hand",
        "mobilebase0_navview",
    ]
    world = client.world_map("agentview")
    assert world.shape == (4, 6, 3) and world.dtype == np.float32
    assert rpc.calls[-1][1]["cameras"] == [
        {"camera": "robot0_agentview_left", "height": 4, "width": 6}
    ]
    assert [call[0] for call in rpc.calls].count("env.reset") == 1


def _frame(height, width, z):
    depth = np.full((height, width), 1.0)
    depth[0, 0] = np.nan
    pose = np.eye(4)
    pose[2, 3] = z
    return LocalizationFrame.from_canonical(
        {
            "camera": "cam",
            "rgb": np.full((height, width, 3), 100, np.uint8),
            "depth_m": depth,
            "intrinsic_K": [
                [5.0, 0, width / 2 - 0.5],
                [0, 5.0, height / 2 - 0.5],
                [0, 0, 1],
            ],
            "T_world_from_cv": pose,
        }
    )


def test_step_artifacts_come_from_one_capture(tmp_path):
    requests = []
    frames = [_frame(4, 6, 2.0), _frame(4, 6, 2.0), _frame(4, 6, -1.0), _frame(8, 8, 2)]

    def capture_frames(cameras):
        requests.append(cameras)
        return frames

    primitives = SimpleNamespace(
        env=SimpleNamespace(capture_frames=capture_frames), hi_res=8
    )
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        tools._save_observation_artifacts(primitives, state, step)
    assert requests == [["agentview", "wrist", "navview", ("agentview", 8, 8)]]
    artifacts = state.get(0).artifacts
    for name in tools._HEAVY_ARTIFACTS:
        assert name in artifacts, name
    assert state.load("agentview_world_high.npz", step=0).dtype == np.float32
    assert state.load("navview_metadata.json", step=0)["valid_pixels"] == 4 * 6 - 1
    floor = state.load("navview_floor.png", step=0)
    # Camera z = -1 and depth 1 put valid navview points at z = 0 (floor);
    # the invalid pixel keeps its RGB.
    assert tuple(floor[0, 0]) == (100, 100, 100)
    assert tuple(floor[1, 1]) == (0, 255, 0)


@pytest.mark.parametrize("translation", [[0.5, -1.5, 0.0], [0.0, 0.0, -1.0]])
def test_back_project_batch_keeps_zero_sum_points_and_rejects_invalid_depth(
    tmp_path, translation
):
    pose = np.eye(4)
    pose[:3, 3] = translation
    frame = LocalizationFrame.from_canonical(
        {
            "camera": "agentview",
            "rgb": np.zeros((1, 4, 3), np.uint8),
            "depth_m": np.array([[1.0, 1.0, np.nan, np.inf]]),
            "intrinsic_K": np.eye(3),
            "T_world_from_cv": pose,
        }
    )
    np.testing.assert_array_equal(frame.valid_mask, [[True, True, False, False]])
    state = EnvState(tmp_path)
    with state.record_step(state={}) as step:
        state.save("agentview_world.npz", frame.world_xyz, step=step)
    data = tools.back_project_batch([[0, col] for col in range(4)], state=state).data
    assert [sample["valid"] for sample in data["results"]] == [True, True, False, False]
    for col in range(2):
        np.testing.assert_allclose(
            data["results"][col]["world_xyz"], frame.world_xyz[0, col]
        )
    assert all(sample["world_xyz"] is None for sample in data["results"][2:])
    assert data["summary"] == {
        "valid_count": 2,
        "total_count": 4,
        "median_xyz": np.median(frame.world_xyz[0, :2], axis=0).tolist(),
    }
