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


"""Opt-in UniWAM checks using the production runtime and scripted planner."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import get_robot_spec
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_UNIWAM_INTEGRATION") != "1",
    reason="requires UniWAM model environment, weights, simulator assets and a free GPU",
)


def _argv():
    return [
        "--vla-backend",
        "uniwam",
        "--task-name",
        "beat_block_hammer",
        "--seed",
        "100000",
        "--task-config",
        "demo_randomized",
        "--max-episode-steps",
        "48",
        "--cuda-device",
        os.environ["CUDA_VISIBLE_DEVICES"],
        "--uniwam-python",
        os.environ["UNIWAM_PYTHON"],
    ]


def test_uniwam_mixed_actions():
    spec = get_robot_spec()
    args = parse_runtime_args(spec, _argv())
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "uniwam-mixed"
    with runtime_phase(spec, args, output, {"env", "vla"}) as runtime:
        env = runtime["env"]
        driver = RoboTwinPrimitives(**runtime, check_cancelled=lambda: None)
        result = driver.uniwam_act(chunks=1)
        assert result.data["executed_steps"] > 0
        if env.terminated or env.truncated:
            env.reset()
        value = float(env.last_info["robot_state"]["qpos_target14"][6])
        driver.apply_qpos_updates([{"arm": "left", "gripper": value}])
        np.testing.assert_array_equal(
            env.qpos_history[-1], env.last_info["robot_state"]["qpos_target14"]
        )
        assert driver.uniwam_act(chunks=1).data["executed_steps"] > 0
        env.reset()
        assert env.qpos_history.shape == (1, 14)


def test_uniwam_policy_chain():
    run_scripted_policy_chain(
        robot="robotwin",
        robot_argv=_argv(),
        output_dir=Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "uniwam-chain",
        action=ScriptedToolCall("uniwam_act", {"chunks": 1}),
        action_count_field="executed_steps",
    )
