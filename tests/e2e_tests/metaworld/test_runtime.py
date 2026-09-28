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

"""Real EGL/RPC/tool execution without a model checkpoint or external provider."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.metaworld.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.rpc import RpcError
from tests.e2e_tests.common import (
    parse_runtime_args,
    prepare_suite,
    publish_check,
    require_depth,
    require_rgb,
    runtime_phase,
)


@pytest.fixture(scope="module")
def output():
    return prepare_suite(
        output_dir=Path(os.environ["RPENT_E2E_OUTPUT_DIR"]),
        stack="metaworld",
        extra="metaworld",
        details={"control": "Cartesian, no VLA"},
    )


@pytest.mark.parametrize("task", ["reach-v3", "push-v3", "pick-place-v3"])
def test_real_rgbd_and_episode_boundary(output, task):
    spec = get_robot_spec()
    args = parse_runtime_args(spec, ["--task", task, "--max-episode-steps", "2"])
    with runtime_phase(spec, args, output / task, {"env"}) as runtime:
        client = runtime["env_client"]
        obs = client.get_obs()
        rgb = require_rgb(obs["rgb"], "camera")
        require_depth(obs["depth"], "depth", rgb)
        assert rgb.std() > 1
        assert set(obs) == {
            "eef_position",
            "gripper_opening",
            "rgb",
            "depth",
            "camera",
            "instruction",
            "steps",
            "terminated",
            "truncated",
        }
        assert np.asarray(obs["eef_position"]).shape == (3,)
        with pytest.raises(RpcError, match="four finite values"):
            client.step([0, 0, 0, 2])
        assert client.get_obs()["steps"] == 0
        client.step([0, 0, 0, -1])
        obs, _, _, truncated, _ = client.step([0, 0, 0, -1])
        assert truncated and obs["steps"] == 2
        with pytest.raises(RpcError, match="episode ended"):
            client.step([0, 0, 0, -1])
        assert client.get_obs()["steps"] == 2
        publish_check(
            "metaworld_rgbd",
            {"task": task, "shape": rgb.shape, "budget_enforced": True},
        )


def test_tool_dispatch_records_motion_and_finish(output):
    spec = get_robot_spec()
    args = parse_runtime_args(spec, ["--task", "reach-v3", "--max-episode-steps", "8"])
    args.output_dir = output / "tool-chain"
    args.memory_dir = output / "memory"
    args.memory_profile = "local"
    config = spec.parse_config(args)
    spec.prepare_memory(args, config)
    with runtime_phase(spec, args, config.output_dir, {"env"}) as runtime:
        toolkit = get_toolkit(
            runtime_kwargs=runtime,
            dashboard_events=NullDashboardEventSink(),
            config=config,
        )
        try:
            read = toolkit.execute_tool("view_env_state", {})
            assert read.result["_image_bytes"]
            move = toolkit.execute_tool("set_gripper", {"gripper": -1, "steps": 2})
            assert "error" not in move.result
            assert move.result["motion_result"]["steps"] == 2
            assert move.result["steps"] == 2
            finish = toolkit.execute_tool(
                "finish",
                {"status": "stuck", "summary": "bounded real simulator tool check"},
            )
            assert finish.is_finish
            assert (config.output_dir / "states.json").exists()
        finally:
            toolkit.close()
    assert (config.output_dir / "tool_snapshots.mp4").is_file()
