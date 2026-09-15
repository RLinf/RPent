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

import sys
import types
from dataclasses import make_dataclass
from typing import Any

import pytest

from rpent.tools.toolkit import readonly


class FakeSingleArmPrimitives:
    """No-runtime primitive surface shared by LIBERO and RoboCasa tests."""

    instances: list[FakeSingleArmPrimitives] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = 0
        self.recording_started = False
        type(self).instances.append(self)

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return {"success": True}

    def reset_episode(self, reason: str) -> dict[str, Any]:
        return {"success": True, "reason": reason}

    def start_recording(self) -> None:
        self.recording_started = True

    def recorded_frame_count(self) -> int:
        return 0

    def frame_slice(self, start: int) -> list[Any]:
        del start
        return []

    def stop_recording(self) -> list[Any]:
        return []

    def dump_success_criteria(self) -> str:
        return "offline success criteria"

    @readonly
    def segment(self, **kwargs: Any) -> dict[str, Any]:
        return {"segment": kwargs}

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"operation": name, "arguments": kwargs}

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_to", **kwargs)

    def pi0_pick(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_pick", **kwargs)

    def pi0_doubled(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_doubled", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("release", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("set_gripper", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_wrist", **kwargs)

    def rotate_pitch(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_pitch", **kwargs)

    def move_pose(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_pose", **kwargs)

    def move_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_delta", **kwargs)

    def scripted_grasp(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("scripted_grasp", **kwargs)

    def rldx_skill(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_skill", **kwargs)

    def rldx_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_arm", **kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("navigate_to", **kwargs)

    def move_base(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_base", **kwargs)


@pytest.fixture
def fake_single_arm_primitives() -> type[FakeSingleArmPrimitives]:
    FakeSingleArmPrimitives.instances.clear()
    return FakeSingleArmPrimitives


def _fake_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_rlinf_realworld_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install minimal RLinf realworld modules for config-contract tests.

    Importing RLinf's realworld package can start/clean ROS/Ray processes as an
    import side effect on lab machines.  These tests only need RLinf dataclass
    field names, so they use inert stand-ins instead of importing hardware code.
    """

    franka_robot_fields = [
        "enable_camera_player",
        "enable_camera_depth",
        "camera_resize",
        "max_num_steps",
        "reward_threshold",
        "action_scale",
        "enable_gripper_penalty",
        "compliance_param",
        "precision_param",
        "task_description",
        "camera_names",
        "target_ee_pose",
        "reset_ee_pose",
        "ee_pose_limit_min",
        "ee_pose_limit_max",
    ]
    franka_hardware_fields = [
        "robot_ip",
        "camera_serials",
        "camera_type",
        "gripper_type",
        "gripper_connection",
        "node_rank",
    ]
    dual_robot_fields = [
        "max_num_steps",
        "task_description",
        "joint_reset_qpos",
        "target_ee_pose",
        "ee_pose_limit_min",
        "ee_pose_limit_max",
    ]
    dual_hardware_fields = [
        "left_robot_ip",
        "right_robot_ip",
        "base_camera_serials",
        "base_camera_type",
        "left_camera_serials",
        "left_camera_type",
        "right_camera_serials",
        "right_camera_type",
        "left_gripper_type",
        "right_gripper_type",
        "left_gripper_connection",
        "right_gripper_connection",
        "left_controller_node_rank",
        "right_controller_node_rank",
        "node_rank",
    ]

    def dataclass_for(name: str, fields: list[str]) -> type:
        return make_dataclass(name, [(field, object, None) for field in fields])

    class FakeFrankaEnv:
        pass

    modules = {
        "rlinf": _fake_module("rlinf", __path__=[]),
        "rlinf.envs": _fake_module("rlinf.envs", __path__=[]),
        "rlinf.envs.realworld": _fake_module("rlinf.envs.realworld", __path__=[]),
        "rlinf.envs.realworld.common": _fake_module(
            "rlinf.envs.realworld.common", __path__=[]
        ),
        "rlinf.envs.realworld.common.wrappers": _fake_module(
            "rlinf.envs.realworld.common.wrappers",
            apply_single_arm_wrappers=lambda env, _env_cfg: env,
        ),
        "rlinf.envs.realworld.franka": _fake_module(
            "rlinf.envs.realworld.franka", __path__=[]
        ),
        "rlinf.envs.realworld.franka.franka_env": _fake_module(
            "rlinf.envs.realworld.franka.franka_env",
            FrankaEnv=FakeFrankaEnv,
            FrankaRobotConfig=dataclass_for("FrankaRobotConfig", franka_robot_fields),
        ),
        "rlinf.envs.realworld.franka.tasks": _fake_module(
            "rlinf.envs.realworld.franka.tasks", __path__=[]
        ),
        "rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env": _fake_module(
            "rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env",
            DualFrankaTCPRobotConfig=dataclass_for(
                "DualFrankaTCPRobotConfig", dual_robot_fields
            ),
        ),
        "rlinf.scheduler": _fake_module("rlinf.scheduler", __path__=[]),
        "rlinf.scheduler.hardware": _fake_module(
            "rlinf.scheduler.hardware", __path__=[]
        ),
        "rlinf.scheduler.hardware.robots": _fake_module(
            "rlinf.scheduler.hardware.robots", __path__=[]
        ),
        "rlinf.scheduler.hardware.robots.franka": _fake_module(
            "rlinf.scheduler.hardware.robots.franka",
            FrankaConfig=dataclass_for("FrankaConfig", franka_hardware_fields),
        ),
        "rlinf.scheduler.hardware.robots.dual_franka": _fake_module(
            "rlinf.scheduler.hardware.robots.dual_franka",
            DualFrankaConfig=dataclass_for("DualFrankaConfig", dual_hardware_fields),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
