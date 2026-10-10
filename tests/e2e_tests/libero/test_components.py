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

from robots.libero.robot_spec import get_robot_spec
from tests.e2e_tests.common import (
    parse_runtime_args,
    publish_check,
    require_array,
    runtime_phase,
)
from tests.e2e_tests.libero.scenario import LiberoScenario


def test_environment_component(libero_scenario: LiberoScenario) -> None:
    _, result = libero_scenario.environment
    publish_check("libero_environment_component", result)


def test_environment_action(libero_scenario: LiberoScenario) -> None:
    publish_check("libero_environment_action", libero_scenario.environment_action)


def test_pi05_component(libero_scenario: LiberoScenario) -> None:
    if libero_scenario.variant != "pro":
        pytest.skip("shared Pi0.5 component is covered by LIBERO-PRO")
    _, result = libero_scenario.pi05
    publish_check("pi05_component", result)


def test_sam3_component(libero_scenario: LiberoScenario) -> None:
    if libero_scenario.variant != "pro":
        pytest.skip("shared SAM3 component is covered by LIBERO-PRO")
    publish_check("sam3_component", libero_scenario.sam3)


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
            raw = {**env.raw_obs(), "task_descriptions": env.get_task_language()}
            started = time.perf_counter()
            actions = require_array(
                model.predict(raw), "WAM actions", ndim=2, last_dim=7
            )
            record_property("inference_seconds", time.perf_counter() - started)
            assert actions.shape == (caps["chunk_size"], 7)
            np.save(tmp_path / "actions.npy", actions)
            obs, *_ = env.chunk_step(actions)
            require_array(obs["main_images"], "next image", ndim=3, last_dim=3)
            record_property("native_success", env.terminated)
            record_property("truncated", env.truncated)
            model.reset()


@pytest.mark.timeout(1200)
def test_environment_horizon_excludes_settling(wam_argv, tmp_path) -> None:
    spec = get_robot_spec()
    args = parse_runtime_args(spec, wam_argv)
    with runtime_phase(spec, args, tmp_path, {"env"}) as runtime:
        env = runtime["env"]
        env.reset()
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1
        for step in range(args.max_episode_steps):
            env.step(action)
            assert not env.terminated
            assert env.truncated == (step == args.max_episode_steps - 1)
