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

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.metaworld.config import CAMERAS, TASKS
from robots.metaworld.env_server import MetaWorldEnvFacade
from robots.metaworld.robot_spec import get_robot_spec, get_toolkit
from robots.metaworld.toolkit import back_project_pixel
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


def count_video_frames(path):
    import imageio_ffmpeg

    frames = imageio_ffmpeg.read_frames(str(path))
    metadata = next(frames)
    assert metadata["size"] == (480, 480)
    return sum(1 for _ in frames)


def test_task_catalog_matches_installed_mt10():
    from metaworld.env_dict import MT10_V3

    assert set(TASKS) == set(MT10_V3)


@pytest.mark.parametrize("task", TASKS)
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
    recording = json.loads((output / task / "environment.json").read_text())
    assert recording["benchmark"] == "MT1" and recording["num_tasks"] == 1
    assert recording["versions"]["metaworld"] == "3.0.0"
    assert count_video_frames(Path(recording["video_path"])) == 3


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
    assert count_video_frames(config.output_dir / "videos/episode_0000.mp4") == 3


@pytest.mark.parametrize("camera", CAMERAS)
def test_camera_observation_does_not_mutate_native_history(camera):
    env = MetaWorldEnvFacade("reach-v3", 0, 5, camera)
    try:
        native = env._env.unwrapped
        previous = np.full_like(native._prev_obs, 0.123)
        native._prev_obs = previous.copy()
        before_qpos = native.data.qpos.copy()
        before_time = native.data.time
        first = env.get_obs()
        second = env.get_obs()
        assert first["rgb"].std() > 1
        np.testing.assert_array_equal(native._prev_obs, previous)
        np.testing.assert_array_equal(native.data.qpos, before_qpos)
        assert native.data.time == before_time
        np.testing.assert_array_equal(first["eef_position"], second["eef_position"])
    finally:
        env.close()


def test_pixel_centers_match_a_rendered_plane():
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <visual><global offwidth="480" offheight="480"/><quality offsamples="0"/></visual>
      <worldbody>
        <camera name="plane" pos="0 -1 1" xyaxes="1 0 0 0 1 1" fovy="60"/>
        <geom type="plane" size="10 10 0.1" euler="10 15 0"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    facade = MetaWorldEnvFacade.__new__(MetaWorldEnvFacade)
    facade._env = SimpleNamespace(unwrapped=SimpleNamespace(model=model, data=data))
    facade._meta = {"camera": "plane"}
    facade._camera_id = 0
    with mujoco.Renderer(model, height=480, width=480) as renderer:
        renderer.update_scene(data, camera="plane")
        renderer.enable_depth_rendering()
        obs = {"depth": renderer.render(), "camera": facade.get_camera_meta("plane")}
        for row, col in [(80, 80), (160, 300), (240, 240), (400, 360)]:
            # The fixture's tilted plane independently constrains both image
            # axes, pixel centers and camera translation.
            point = np.asarray(back_project_pixel(obs, row, col))
            normal = data.geom_xmat[0].reshape(3, 3)[:, 2]
            assert abs(np.dot(point - data.geom_xpos[0], normal)) < 2e-5


def test_reset_preserves_prior_episode_video(output):
    spec = get_robot_spec()
    args = parse_runtime_args(spec, ["--task", "push-v3", "--max-episode-steps", "4"])
    video_dir = output / "reset-videos/videos"
    video_dir.mkdir(parents=True)
    previous = video_dir / "episode_0000.mp4"
    previous.write_bytes(b"prior recording must not be overwritten")
    with runtime_phase(spec, args, output / "reset-videos", {"env"}) as runtime:
        client = runtime["env_client"]
        initial = client.get_obs()["eef_position"]
        client.step([0, 0, 0, -1])
        reset = client.reset()
        np.testing.assert_allclose(reset["eef_position"], initial)
        assert reset["steps"] == 0
        client.step([0, 0, 0, -1])
        client.step([0, 0, 0, -1])
    assert previous.read_bytes() == b"prior recording must not be overwritten"
    assert count_video_frames(video_dir / "episode_0001.mp4") == 2
    assert count_video_frames(video_dir / "episode_0002.mp4") == 3
