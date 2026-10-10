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

import time
from contextlib import closing

import numpy as np
import pytest

from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES, get_robot_spec
from tests.e2e_tests.common import (
    parse_runtime_args,
    publish_check,
    require_array,
    runtime_phase,
)
from tests.e2e_tests.robotwin.scenario import RoboTwinScenario


def test_environment_component(robotwin_scenario: RoboTwinScenario) -> None:
    _, _, result = robotwin_scenario.environment
    publish_check("robotwin_environment_component", result)


def test_lingbot_component(robotwin_scenario: RoboTwinScenario) -> None:
    _, result = robotwin_scenario.lingbot
    publish_check("lingbot_component", result)


@pytest.mark.timeout(1200)
def test_wam_component(wam_argv, tmp_path, record_property) -> None:
    spec = get_robot_spec()
    args = parse_runtime_args(spec, wam_argv)
    with runtime_phase(spec, args, tmp_path, {"env", "wam"}) as runtime:
        env = runtime["env"]
        with closing(runtime["model"]) as model:
            caps = model.wam.get_capabilities()
            record_property("checkpoint", caps["checkpoint"])
            env.reset()
            model.reset()
            observation = {
                "views": {
                    name: {"rgb": env.render_camera(name)}
                    for name in ROBOTWIN_CAMERA_NAMES
                },
                "robot_state": env.last_info["robot_state"],
                "task_language": env.get_task_language(),
            }
            started = time.perf_counter()
            actions = require_array(
                model.predict(observation), "WAM actions", ndim=2, last_dim=14
            )
            record_property("inference_seconds", time.perf_counter() - started)
            assert actions.shape == (caps["chunk_size"], 14)
            assert model.action_type == "qpos"
            np.save(tmp_path / "actions.npy", actions)
            *_, info = env.chunk_step(actions, action_type=model.action_type)
            assert info["executed_actions"] > 0
            require_array(
                info["robot_state"]["qpos_target14"], "next joints", ndim=1, last_dim=14
            )
            record_property("executed_actions", info["executed_actions"])
            record_property("native_success", info["episode_status"]["eval_success"])
            model.reset()
