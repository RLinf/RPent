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

"""Offline contracts for the VLABench toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.vlabench.agent_toolkit import VLABenchToolkit
from robots.vlabench.motion import VLABenchMotion
from robots.vlabench.observations import VLABenchGeometryObservations
from robots.vlabench.spec import get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.utils import templates


class _FakeEnv:
    """Env client stand-in that records the motions it is asked to run."""

    def __init__(self) -> None:
        self.success = False
        self.pose_calls: list[dict[str, Any]] = []
        self.gripper_calls: list[dict[str, Any]] = []

    def robot_model_state(self) -> dict[str, Any]:
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        return {"world_from_eef": pose, "arm_qpos": np.zeros(7)}

    def sensor_snapshot(
        self, *, cameras: list[str], height: int = 512, width: int = 512
    ) -> dict[str, Any]:
        calibration = {
            "intrinsic_K": np.eye(3),
            "extrinsic_cam2world": np.eye(4),
            "height": 4,
            "width": 4,
        }
        frames = [
            {
                "camera": camera,
                "rgb_top_down": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth_meters": np.ones((4, 4), dtype=np.float32),
                "calibration": calibration,
            }
            for camera in cameras
        ]
        return {
            "observation_id": "obs_1",
            "cameras": frames,
            "robot": {"model": self.robot_model_state()},
        }

    def task_status(self) -> dict[str, Any]:
        return {
            "task_success": self.success,
            "terminated": self.success,
            "truncated": False,
        }

    def get_task_language(self) -> str:
        return "pick the painting"

    def execute_pose(self, **kwargs: Any) -> dict[str, Any]:
        self.pose_calls.append(kwargs)
        return {"ok": True, "steps_used": 3}

    def execute_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(kwargs)
        return {"ok": True, "steps_used": kwargs["steps"]}


@pytest.fixture
def rig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[VLABenchToolkit, _FakeEnv]:
    # Tool descriptions resolve {{output_dir}}, which needs a configured run.
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path / "out")}
    )
    env = _FakeEnv()
    toolkit = VLABenchToolkit(
        runtime_kwargs={"env": env},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "out",
    )
    return toolkit, env


def test_registers_the_planner_tools(rig) -> None:
    toolkit, _ = rig
    names = {tool.name for tool in toolkit.list_tools()}
    assert {
        "finish",
        "move_pose",
        "move_delta",
        "set_gripper",
        "back_project",
        "render",
        "view_env_state",
    } <= names


def test_initial_observation_records_rgb_depth_and_calibration_per_camera(rig) -> None:
    toolkit, _ = rig
    record = toolkit.state.get(0)
    assert len(record.artifacts) == 12
    assert "wrist_camera_meta.json" in record.artifacts
    assert record.extras["task_language"] == "pick the painting"


def test_move_delta_adds_the_offset_to_the_measured_pose(rig) -> None:
    toolkit, env = rig
    result = toolkit.execute_tool("move_delta", {"dxyz": [0.0, 0.0, 0.1]})
    assert "error" not in result.data
    call = env.pose_calls[-1]
    np.testing.assert_allclose(call["xyz"], [0.1, 0.2, 0.4])
    np.testing.assert_allclose(call["quaternion_xyzw"], [0, 0, 0, 1], atol=1e-9)
    assert toolkit.state.latest_step == 1


def test_set_gripper_forwards_the_command_and_step_count(rig) -> None:
    toolkit, env = rig
    toolkit.execute_tool("set_gripper", {"gripper": 1.0, "steps": 3})
    assert env.gripper_calls[-1]["gripper"] == 1.0
    assert env.gripper_calls[-1]["steps"] == 3


def test_finish_ends_the_run_but_does_not_trust_a_success_claim(rig) -> None:
    toolkit, env = rig
    claimed = toolkit.execute_tool("finish", {"status": "success", "summary": "x"})
    assert claimed.data["_finish"] is True
    assert claimed.data["status"] == "failure"
    assert claimed.data["success"] is False
    env.success = True
    verified = toolkit.execute_tool("finish", {"status": "success", "summary": "x"})
    assert verified.data["status"] == "success"
    assert verified.data["success"] is True


def test_solved_follows_the_native_flag(rig) -> None:
    toolkit, env = rig
    assert toolkit.solved() is False
    env.success = True
    assert toolkit.solved() is True


def test_back_project_reads_the_recorded_depth(rig) -> None:
    toolkit, _ = rig
    result = toolkit.execute_tool(
        "back_project",
        {
            "camera": "agentview",
            "selection": {"image_shape": [4, 4], "points_rc": [[2, 3]]},
        },
    )
    assert "error" not in result.data
    assert result.data["camera"] == "agentview"
    assert result.data["step"] == 0


def test_move_pose_validates_unit_quaternion_and_supports_partial_targets(rig) -> None:
    toolkit, env = rig
    quat = [0.0, 0.0, 0.70710678, 0.70710678]
    full = toolkit.execute_tool(
        "move_pose", {"xyz": [0.2, 0.3, 0.4], "quaternion_xyzw": quat}
    )
    assert "error" not in full.data
    assert env.pose_calls[-1]["xyz"] == [0.2, 0.3, 0.4]
    np.testing.assert_allclose(env.pose_calls[-1]["quaternion_xyzw"], quat, atol=1e-6)

    pos_only = toolkit.execute_tool("move_pose", {"xyz": [0.3, 0.2, 0.1]})
    assert "error" not in pos_only.data
    assert env.pose_calls[-1]["xyz"] == [0.3, 0.2, 0.1]
    np.testing.assert_allclose(
        env.pose_calls[-1]["quaternion_xyzw"], [0, 0, 0, 1], atol=1e-9
    )

    ori_only = toolkit.execute_tool("move_pose", {"quaternion_xyzw": [0, 0, 0, 1]})
    assert "error" not in ori_only.data
    np.testing.assert_allclose(env.pose_calls[-1]["xyz"], [0.1, 0.2, 0.3])

    bad = toolkit.execute_tool(
        "move_pose", {"xyz": [0.1, 0.2, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, 2.0]}
    )
    assert "finite unit 4-vector" in bad.data["error"]
    motion = VLABenchMotion(env=env, check_cancelled=lambda: None)
    with pytest.raises(ValueError, match="finite unit 4-vector"):
        motion.move_pose(xyz=[0.1, 0.2, 0.3], quaternion_xyzw=[0.0, 0.0, 0.0, 2.0])


def test_render_and_view_env_state_and_observations(rig, tmp_path: Path) -> None:
    toolkit, _ = rig
    rendered = toolkit.execute_tool("render", {})
    assert "error" not in rendered.data
    assert toolkit.state.latest_step == 1

    viewed = toolkit.execute_tool("view_env_state", {"step": 0})
    assert "error" not in viewed.data
    assert viewed.data["step"] == 0
    assert len(viewed.images) == 4

    missing = toolkit.execute_tool("view_env_state", {"step": 99})
    assert "state step not available" in missing.data["error"]

    primary = toolkit._observations.get_observation(camera="agentview", step=0)
    others = [
        obs.camera for obs in toolkit._observations.additional_observations(primary)
    ]
    assert others == ["left", "right", "wrist"]
    with pytest.raises(ValueError, match="VLABench cameras are"):
        toolkit._observations.get_observation(camera="overhead", step=0)

    partial_state = EnvState(tmp_path / "partial")
    with partial_state.record_step(state={"step_idx": 0}) as step0:
        partial_state.save(
            "wrist_rgb.png", np.zeros((4, 4, 3), dtype=np.uint8), step=step0
        )
    partial_obs = VLABenchGeometryObservations(partial_state)
    with pytest.raises(ValueError, match="no calibrated RGB-D pair"):
        partial_obs.get_observation(camera="wrist", step=0)


def test_evaluation_metrics_handles_normal_and_error_status(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    toolkit, env = rig
    toolkit.execute_tool("set_gripper", {"gripper": -1.0, "steps": 4})
    metrics = toolkit.evaluation_metrics(success=False)
    assert metrics["native_actions"] == 4
    assert metrics["episode_status"]["task_success"] is False

    monkeypatch.setattr(
        env, "task_status", lambda: (_ for _ in ()).throw(RuntimeError("rpc down"))
    )
    degraded = toolkit.evaluation_metrics(success=None)
    assert degraded["native_actions"] == 4
    assert degraded["status_error"] == "RuntimeError: rpc down"


def test_prompts_align_with_registered_tools_and_guide() -> None:
    spec = get_robot_spec()
    variables = {
        "task_name": "select_painting",
        "track": "track_1_in_distribution",
        "episode_index": 0,
        "seed": 42,
        "mode": "eval",
        "memory_dir": "/memory",
        "recipe_tag": "vlabench_select_painting_e0_s42",
        "output_dir": Path("/output"),
    }
    system = spec.prompts.render("system", variables=variables)
    user = spec.prompts.render("user", variables=variables)
    assert "offset_xyz" not in system
    assert "view_env_state(step=0)" in user
    for tool_name in ("back_project", "move_pose", "move_delta", "set_gripper"):
        assert tool_name in system
