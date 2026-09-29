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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.metaworld.env_server import MetaWorldEnvFacade
from robots.metaworld.robot_spec import get_robot_spec
from robots.metaworld.toolkit import MetaWorldToolkit, back_project_pixel
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager


def test_task_prompt_contains_the_actual_run_configuration():
    prompt = get_robot_spec().prompts.render(
        "user",
        variables={
            "instruction": "Open the test drawer",
            "seed": 17,
            "max_episode_steps": 123,
            "memory_dir": "/tmp/test-memory",
        },
    )
    for expected in ("Open the test drawer", "17", "123", "/tmp/test-memory"):
        assert expected in prompt
    assert "{" not in prompt


def test_reading_proprioception_does_not_build_privileged_stacked_observations():
    native = SimpleNamespace(
        get_endeff_pos=Mock(return_value=np.array([1.0, 2.0, 3.0])),
        data=SimpleNamespace(
            body=lambda name: SimpleNamespace(
                xpos=np.array([0.0, 0.0, 0.05 if name == "leftclaw" else 0.0])
            )
        ),
        _get_obs=Mock(side_effect=AssertionError("changes the native frame history")),
    )
    env = MetaWorldEnvFacade.__new__(MetaWorldEnvFacade)
    env._env = SimpleNamespace(unwrapped=native)
    env._meta = {"task": "reach-v3", "camera": "corner2"}
    env._steps = 0
    env._terminated = env._truncated = False
    obs = env.get_obs(include_images=False)
    np.testing.assert_array_equal(obs["eef_position"], [1, 2, 3])
    assert obs["gripper_opening"] == 0.5
    native._get_obs.assert_not_called()


def test_out_of_range_action_is_rejected_before_float32_rounding():
    env = MetaWorldEnvFacade.__new__(MetaWorldEnvFacade)
    env._env = Mock()
    env._terminated = env._truncated = env._success = False
    env._steps = 0
    env._meta = {"max_episode_steps": 10}
    env._env.step.return_value = ({}, 0.0, False, False, {})
    env.get_obs = lambda **kwargs: {}
    with pytest.raises(ValueError):
        env.step([np.nextafter(1.0, 2.0), 0.0, 0.0, 0.0])
    env._env.step.assert_not_called()


def test_projection_stays_bound_to_visible_snapshot_after_capture_failure(tmp_path):
    obs = {
        "eef_position": np.zeros(3),
        "gripper_opening": 1.0,
        "steps": 0,
        "terminated": False,
        "truncated": False,
        "instruction": "test",
        "rgb": np.zeros((3, 3, 3), dtype=np.uint8),
        "depth": np.full((3, 3), 2.0),
        "camera": {
            "intrinsics": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
            "rotation_world_from_camera": np.eye(3),
            "position_world": [1, 2, 3],
        },
    }
    moved = {
        **{k: v for k, v in obs.items() if k not in ("rgb", "depth", "camera")},
        "eef_position": np.array([0.01, 0, 0]),
        "steps": 1,
    }
    client = SimpleNamespace(
        last_obs=obs,
        get_obs=Mock(side_effect=[obs, OSError("capture failed")]),
        step=Mock(return_value=(moved, 0.0, False, False, {})),
    )
    toolkit = MetaWorldToolkit(
        runtime_kwargs={"env_client": client},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(root=tmp_path / "memory"),
        state_output_dir=tmp_path / "output",
    )
    motion = toolkit.execute_tool(
        "move_to", {"target_xyz": [0.01, 0, 0], "gripper": -1, "max_steps": 1}
    )
    assert "state_capture_error" in motion.result
    projection = toolkit.execute_tool("back_project", {"row": 1, "col": 2})
    assert "error" not in projection.result
    np.testing.assert_allclose(projection.result["world_xyz"], [2, 2, 1])


def test_far_plane_is_not_reported_as_a_surface():
    with pytest.raises(ValueError, match="far clipping plane"):
        back_project_pixel(
            {"depth": np.full((2, 2), 100.0), "camera": {"clip_range": [0.01, 100.0]}},
            0,
            0,
        )


def test_recording_failure_does_not_skip_renderer_or_environment_cleanup():
    events = []

    def failed_video_close():
        events.append("video")
        raise OSError("encoder failed")

    env = MetaWorldEnvFacade.__new__(MetaWorldEnvFacade)
    env._closed = False
    env._video_writer = Mock(close=Mock(side_effect=failed_video_close))
    env._renderer = Mock(close=Mock(side_effect=lambda: events.append("renderer")))
    env._env = Mock(close=Mock(side_effect=lambda: events.append("env")))
    with pytest.raises(OSError, match="encoder failed"):
        env.close()
    env.close()
    assert events == ["video", "renderer", "env"]
    assert env._video_writer is None


@pytest.mark.parametrize("missing_camera", [False, True])
def test_constructor_cleans_up_without_hiding_startup_errors(
    monkeypatch, missing_camera
):
    native = Mock()
    native.close.side_effect = OSError("cleanup also failed")
    renderer = Mock(side_effect=ValueError("renderer creation failed"))
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        SimpleNamespace(
            registry={"Meta-World/MT1": object()}, make=Mock(return_value=native)
        ),
    )
    monkeypatch.setitem(sys.modules, "metaworld", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            Renderer=renderer,
            mj_name2id=Mock(return_value=-1 if missing_camera else 0),
            mjtObj=SimpleNamespace(mjOBJ_CAMERA=0),
        ),
    )
    message = (
        "absent from the native model" if missing_camera else "renderer creation failed"
    )
    with pytest.raises(ValueError, match=message):
        MetaWorldEnvFacade("reach-v3", 0, 10, "corner2")
    native.close.assert_called_once()
    if missing_camera:
        renderer.assert_not_called()
