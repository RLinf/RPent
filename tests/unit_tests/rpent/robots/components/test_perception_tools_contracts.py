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

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero import tools as libero
from robots.robodojo import tools as robodojo
from rpent.robots.components.perception_tools import (
    back_project_depth,
    view_recorded_state,
)


@pytest.mark.parametrize("backend", [libero, robodojo])
def test_recorded_view_keeps_backend_images_and_log(backend):
    images = (
        ["agentview_policy.png", "agentview_high.png", "wrist_high.png"]
        if backend is libero
        else [name for _, name in robodojo.CAMERA_ARTIFACTS]
    )
    record = SimpleNamespace(
        step_idx=3,
        terminated=True,
        truncated=False,
        state={"joints": [1]},
        extras={"task_language": "pick"},
        command={"action": "move_to"},
        result={"reached": True},
        elapsed_s=0.1,
        artifacts=set(images),
    )
    state = SimpleNamespace(
        get=lambda step: record, load_bytes=lambda name, **kw: name.encode()
    )
    result = backend.view_env_state(state=state)
    assert result["state"] == {"joints": [1]}
    assert result["terminated"] is True
    assert result["task_language"] == "pick"
    assert result["log"] == {
        "command": record.command,
        "result": record.result,
        "elapsed_s": 0.1,
    }
    for slot, name in zip(
        ("_image_bytes", "_image_cam_bytes", "_image_wrist_bytes"), images
    ):
        assert result[slot] == name.encode()


def test_missing_record_and_image_keep_error_and_no_fallback():
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    assert view_recorded_state(
        -1, state=SimpleNamespace(get=missing), image_artifacts={}
    ) == {"error": "state step not available: missing"}
    record = SimpleNamespace(
        step_idx=0,
        terminated=False,
        truncated=False,
        state={},
        extras={},
        command=None,
        result=None,
        elapsed_s=None,
        artifacts={"high.png", "low.png"},
    )
    attempted = []

    def load(name, **kwargs):
        attempted.append(name)
        raise FileNotFoundError(name)

    result = view_recorded_state(
        -1,
        state=SimpleNamespace(get=lambda step: record, load_bytes=load),
        image_artifacts={"_image_bytes": ("high.png", "low.png")},
    )
    assert "_image_bytes" not in result
    assert attempted == ["high.png"]


@pytest.mark.parametrize("sign", [-1, 1])
def test_calibrated_depth_preserves_camera_convention(sign):
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    camera = {
        "depth": np.full((3, 3), 2.0),
        "intrinsic_matrix": np.diag([2, 4, 1]),
        "extrinsic_matrix": pose,
    }
    result = back_project_depth(camera, 2, 1, camera="test", camera_z_sign=sign)
    assert result == {
        "camera": "test",
        "pixel": [2, 1],
        "depth_m": 2.0,
        "world_xyz": [2.0, 3.0, 3.0 + sign * 2],
    }


@pytest.mark.parametrize(
    "row,col,depth,error",
    [
        (2, 0, 1, "pixel out of range"),
        (0, 0, 0, "invalid depth"),
        (0, 0, np.nan, "invalid depth"),
    ],
)
def test_depth_errors_remain_structured(row, col, depth, error):
    camera = {
        "depth": np.full((2, 2), depth),
        "intrinsic_matrix": np.eye(3),
        "extrinsic_matrix": np.eye(4),
    }
    result = back_project_depth(camera, row, col, camera="test", camera_z_sign=-1)
    assert error in result["error"]
