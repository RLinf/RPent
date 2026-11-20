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

"""Real requests to existing BEHAVIOR Env/VLA services; no process ownership."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robots.behavior.primitives import BehaviorPrimitives
from robots.behavior.tools import decode_robot_state, extract_public_proprio
from rpent.evaluation import write_json_atomic
from tests.e2e_tests.common import publish_check

if TYPE_CHECKING:
    from tests.e2e_tests.behavior.scenario import BehaviorScenario


def test_environment_component(behavior_scenario: BehaviorScenario) -> None:
    _, result = behavior_scenario.environment
    publish_check("behavior_environment_component", result)


def test_vla_component(behavior_scenario: BehaviorScenario) -> None:
    publish_check("behavior_vla_component", behavior_scenario.vla)


def test_native_motion_component(behavior_scenario: BehaviorScenario) -> None:
    """Execute a bounded native arm hold, then check read-only IK for both arms."""
    if os.environ.get("RPENT_RUN_BEHAVIOR_MOTION_INTEGRATION") != "1":
        pytest.skip("set RPENT_RUN_BEHAVIOR_MOTION_INTEGRATION=1 with CuRobo Env")
    env = behavior_scenario.env
    snapshot = env.get_motion_snapshot()
    state = decode_robot_state(extract_public_proprio(snapshot["observation"]))
    primitive = BehaviorPrimitives(
        env=env,
        model=behavior_scenario.model,
        task_id=behavior_scenario.args.task_id,
        task_language=behavior_scenario.args.task_language,
        check_cancelled=lambda: None,
        execution_horizon=32,
    )
    hold = primitive.move_to(
        arm="left",
        xyz=state["left"]["eef_position"],
        quat=state["left"]["eef_quaternion_xyzw"],
        settle_steps=8,
    )
    assert hold.data.get("completed") is True, hold.data
    assert hold.data["confirmed_actions"] > 0
    assert hold.data["unconfirmed_actions"] == 0
    # Reaching the pose tolerance may return before joint velocity settles.
    # Supply counted hold frames before asking for a read-only static plan.
    settled = primitive.rotate_wrist_joint(
        arm="right", joint="roll", delta_deg=0.0, steps=1, settle_steps=12
    )
    assert settled.data["confirmed_actions"] == 13, settled.data
    assert settled.data["unconfirmed_actions"] == 0
    snapshot = env.get_motion_snapshot()
    token = snapshot["observation_token"]
    state = decode_robot_state(extract_public_proprio(snapshot["observation"]))
    limits = env.arm_joint_limits
    plans = {}
    for arm in ("left", "right"):
        bounds = np.asarray(limits[arm])
        assert bounds.shape == (7, 2) and np.all(bounds[:, 0] < bounds[:, 1])
        target = np.r_[state[arm]["eef_position"], state[arm]["eef_quaternion_xyzw"]]
        plan = env.plan_arm_path(
            arm=arm, target_pose=target, expected_observation_token=token
        )
        assert plan["status"] == "Success", plan
        assert plan["observation_token"] == token
        assert plan["joint_names"] == [f"{arm}_arm_joint{i}" for i in range(1, 8)]
        np.testing.assert_allclose(plan["joint_position_limits_rad"], bounds)
        assert env.get_motion_snapshot()["observation_token"] == token
        plans[arm] = plan
    result = {
        "status": "passed",
        "definition": "bounded arm hold followed by read-only native USD IK; no reset",
        "hold_execution": hold.data,
        "settle_execution": settled.data,
        "plans": plans,
        "observation_token": token,
        "official_task_success": None,
    }
    write_json_atomic(behavior_scenario.output_dir / "motion-component.json", result)
    publish_check("behavior_native_motion_component", result)
