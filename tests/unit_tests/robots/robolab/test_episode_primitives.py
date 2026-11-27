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

"""Measured RoboLab action and observation integration below the toolkit."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from rpent.session import EnvState
from rpent.utils.transforms import rotation_from_wxyz


class NativeScene:
    """Instantaneous flange tracking with a TCP offset that changes on closure."""

    def __init__(self):
        self.step = 0
        self.xyz = np.array([0.4, 0.0, 0.2])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.opening = 1.0
        self.end_step = None
        self.success_step = None
        self.image_reads = 0

    def coordinate_calibration(self):
        return {
            "coordinate_calibration": {
                "base_xyz": [0.0, 0.0, 0.0],
                "base_rotation": np.eye(3),
                "table_top_xyz": [0.0, 0.0, 0.0],
            },
            "left_to_right_in_flange": [0.0, 0.08, 0.0],
            "flange_to_tcp_in_flange": [0.13, 0.0, 0.0],
        }

    def get_env_meta(self):
        return {"task": "BananaInBowlTask", "native_max_episode_steps": 40}

    def get_task_language(self):
        return "Put a banana in the bowl."

    def get_status(self):
        success = self.step == self.success_step
        return {
            "step": self.step,
            "success": success,
            "terminated": success,
            "truncated": self.step == self.end_step,
        }

    def get_motion_state(self):
        axes = rotation_from_wxyz(self.quat).as_matrix()
        return {
            "status": self.get_status(),
            "flange_world_xyz": self.xyz,
            "flange_quaternion_wxyz": self.quat,
            "tcp_world_xyz": self.xyz + axes @ [0.13 + 0.02 * (1 - self.opening), 0, 0],
            "pad_centres_world_m": np.array([[0, 0.04, 0], [0, -0.04, 0]]),
            "opening": self.opening,
        }

    def step_to(self, xyz, quaternion_wxyz, opening, *, expected_step):
        assert expected_step == self.step
        self.xyz = xyz
        self.quat = quaternion_wxyz
        self.opening = float(opening >= 0.5)
        self.step += 1
        return self.get_motion_state()

    def get_obs(self):
        from robots.robolab.access import CAMERAS

        self.image_reads += 1
        frame = {
            "color": np.zeros((2, 2, 3), dtype=np.uint8),
            "distance_to_image_plane": np.array([[1.0, 0.0], [2.0, np.inf]]),
            "intrinsic_matrix": np.eye(3),
            "camera_to_world": np.eye(4),
        }
        return {"vision": dict.fromkeys(CAMERAS, frame), "status": self.get_status()}


def make_primitives(env, tmp_path):
    from robots.rl.envs.robolab import RoboLabRLPrimitives

    return RoboLabRLPrimitives(
        env=env,
        sam3_client=None,
        state=EnvState(tmp_path),
        record_artifacts=True,
        check_cancelled=lambda: None,
    )


def test_orientation_only_move_keeps_measured_contact_midpoint(tmp_path):
    env = NativeScene()
    primitive = make_primitives(env, tmp_path)
    start = primitive.robot_state()
    target = [np.pi, 0.0, 0.0]
    result = primitive.move_to(orientation_rpy=target)
    assert result.data["status"] == "reached"
    np.testing.assert_allclose(
        primitive.measure().xyz, start["pose"]["xyz"], atol=0.004
    )
    np.testing.assert_allclose(
        primitive.measure().rotation[:, 2], [0, 0, -1], atol=0.02
    )


def test_binary_gripper_closure_keeps_contact_midpoint(tmp_path):
    env = NativeScene()
    primitive = make_primitives(env, tmp_path)
    start = primitive.measure().xyz
    result = primitive.set_gripper(opening=0.0)
    assert result.data["status"] == "reached"
    assert primitive.robot_state()["opening"] == 0
    np.testing.assert_allclose(primitive.measure().xyz, start, atol=0.004)


def test_recorded_observation_preserves_native_timeout_and_depth(tmp_path):
    env = NativeScene()
    env.end_step = 1
    primitive = make_primitives(env, tmp_path)
    result = primitive.move_to(xyz=[0.60, 0.03, 0.20])
    assert result.data["reason"] == "truncated"
    assert primitive.ended
    observation = primitive.record_observation()
    assert observation.truncated and not observation.terminated
    world = primitive._state.load("wrist_cam_world.npz", step=observation.step_idx)
    np.testing.assert_allclose(world[0, 0], [0, 0, -1])
    assert np.isnan(world[0, 1]).all()


def test_lost_receipt_latches_execution_and_preserves_last_confirmed_state(tmp_path):
    class LostReceipt(NativeScene):
        def step_to(self, *args, **kwargs):
            super().step_to(*args, **kwargs)
            raise TimeoutError("lost action receipt")

    env = LostReceipt()
    primitive = make_primitives(env, tmp_path)
    with pytest.raises(TimeoutError, match="lost action receipt"):
        primitive.move_to(xyz=[0.6, 0.0, 0.2])
    assert primitive.infrastructure_error["execution_uncertain"]
    assert primitive.robot_state()["state_stale"]
    assert primitive.move_to(xyz=[0.7, 0.0, 0.2]).data == {"error": "episode_ended"}
    assert env.step == 1
    with pytest.raises(RuntimeError, match="RoboLab"):
        primitive.record_observation()
    assert primitive.infrastructure_error["execution_uncertain"]
    assert primitive.env_steps == 0


def test_settled_close_with_residual_opening_stops_batch():
    from robots.rl.toolkit import RLToolkit

    class ContactScene(NativeScene):
        def step_to(self, xyz, quaternion_wxyz, opening, *, expected_step):
            super().step_to(xyz, quaternion_wxyz, opening, expected_step=expected_step)
            if opening == 0:
                self.opening = 0.4
            return self.get_motion_state()

    env = ContactScene()
    toolkit = RLToolkit(environment="robolab", runtime_kwargs={"env": env})
    result = toolkit.execute_tool(
        "execute_actions",
        {
            "actions": [
                {"action": "set_gripper", "opening": 0},
                {"action": "move_to", "xyz": [60, 0, 25]},
            ]
        },
    ).data

    assert result["executed_count"] == 1
    assert result["stop_reason"] == "blocked"
    closed = result["results"][0]
    assert closed["status"] == "partial"
    assert closed["opening"] == 0.4
    assert "grasp_status" not in closed
    assert env.step > 0


def test_closure_that_remains_fully_open_is_partial(tmp_path):
    class JammedScene(NativeScene):
        def step_to(self, xyz, quaternion_wxyz, opening, *, expected_step):
            return super().step_to(
                xyz, quaternion_wxyz, 1.0, expected_step=expected_step
            )

    primitive = make_primitives(JammedScene(), tmp_path)

    result = primitive.set_gripper(opening=0.0)

    assert result.data["status"] == "partial" and result.data["opening"] == 1.0


def test_command_at_the_current_gripper_target_uses_zero_native_steps(tmp_path):
    env = NativeScene()
    primitive = make_primitives(env, tmp_path)
    assert primitive.robot_state()["opening"] == 1

    result = primitive.set_gripper(opening=1.0)

    assert result.data["status"] == "reached"
    assert result.data["env_steps"] == 0
    assert env.step == 0


@pytest.mark.parametrize(
    "native_timeout, native_success, configured_limit",
    [(1, None, 100), (None, None, 1), (1, 1, 1)],
)
def test_pool_exposes_two_views_and_consistent_terminal_flags(
    monkeypatch, tmp_path, native_timeout, native_success, configured_limit
):
    from robots.rl.server import (
        _WorkerSessionHost,
        pack_execute_response,
        unpack_execute_response,
    )

    class Scene(NativeScene):
        def close(self):
            self.closed = True

    env = Scene()
    env.end_step = native_timeout
    env.success_step = native_success
    host = _WorkerSessionHost("robolab", None, 10.0, None, None, 2)
    monkeypatch.setattr(host, "_build_env", lambda *args, **kwargs: env)
    opened = host.open(
        {
            "task": {"task_name": "BananaInBowlTask", "seed": 3},
            "max_episode_steps": configured_limit,
            "output_dir": str(tmp_path),
        }
    )
    assert opened["environment"] == "robolab"
    assert "wrist_cam" in opened["system_prompt"]
    assert "table_height_above_base_cm" in opened["system_prompt"]
    initial = json.loads(opened["initial_observation"])["state"]
    assert initial["max_episode_steps"] == min(configured_limit, 40)
    assert initial["native_max_episode_steps"] == 40
    schemas = {tool["function"]["name"] for tool in opened["tools"]}
    assert schemas == {"view_env", "segment", "back_project", "execute_actions"}

    rendered = host.execute({"name": "view_env", "arguments": {"camera": "all"}})
    images = rendered.pop("images")
    meta, roundtrip = unpack_execute_response(pack_execute_response(rendered, images))
    assert meta["done"] is False and len(roundtrip) == 2
    assert all(image.startswith(b"\x89PNG") for image in roundtrip)

    result = host.execute(
        {
            "name": "execute_actions",
            "arguments": {
                "actions": [
                    {"action": "move_to", "xyz": [60, 3, 20]},
                    {"action": "set_gripper", "opening": 0},
                ]
            },
        }
    )
    success = native_success is not None
    assert result["done"] and result["truncated"] is (not success)
    assert result["success"] is success
    assert env.step == 1 and env.opening == 1
    reply = json.loads(result["text"])
    assert reply["executed_count"] == 1
    assert reply["stop_reason"] == ("terminated" if success else "truncated")
    for state in (reply["results"][0], reply["final_state"]):
        assert state["terminated"] is success
        assert state["truncated"] is (not success)
    host.close()
    assert env.closed


def test_robolab_captures_a_named_camera(tmp_path):
    class StereoScene(NativeScene):
        def get_obs(self):
            observation = super().get_obs()
            vision = observation["vision"]
            overview = vision["over_shoulder_left_camera"]
            vision["wrist_cam"] = {
                **overview,
                "color": np.full_like(overview["color"], 125),
            }
            return observation

    primitive = make_primitives(StereoScene(), tmp_path)
    rgb, world, calibration = primitive.capture("wrist_cam")

    assert rgb.shape == (2, 2, 3)
    assert np.all(rgb == 125)
    np.testing.assert_allclose(world[0, 0], [0, 0, -1])
    assert calibration["env_steps"] == 0


def test_display_size_preserves_aspect_and_pixel_projection():
    from robots.rl.toolkit import RLToolkit

    class WideScene(NativeScene):
        def get_obs(self):
            observation = super().get_obs()
            for frame in observation["vision"].values():
                frame["color"] = np.zeros((4, 8, 3), dtype=np.uint8)
                frame["distance_to_image_plane"] = np.ones((4, 8))
            return observation

    toolkit = RLToolkit(
        environment="robolab", image_size=4, runtime_kwargs={"env": WideScene()}
    )
    view = toolkit.execute_tool("view_env", {"camera": "wrist_cam"})

    with Image.open(io.BytesIO(view.images[0])) as image:
        assert image.size == (4, 2)
    assert view.data["images"] == [{"camera": "wrist_cam", "size": [2, 4]}]
    point = toolkit.execute_tool(
        "back_project",
        {"pixel": [1, 2], "camera": "wrist_cam", "step": view.data["step"]},
    )
    assert point.data["world_xyz"] == [500.0, -300.0, -100.0]
    box = toolkit.execute_tool(
        "back_project",
        {"box": [0, 1, 2, 3], "camera": "wrist_cam", "step": view.data["step"]},
    )
    assert box.data["world_xyz"] == [350.0, -150.0, -100.0]


def test_unrecorded_steps_capture_lazily_and_preserve_history():
    from robots.rl.toolkit import RLToolkit

    env = NativeScene()
    toolkit = RLToolkit(environment="robolab", runtime_kwargs={"env": env})
    assert env.image_reads == 0

    view = toolkit.execute_tool("view_env", {"camera": "all"})
    assert view.data["step"] == 0 and len(view.images) == 2
    assert env.image_reads == 1

    for x in (60, 65):
        toolkit.execute_tool(
            "execute_actions", {"actions": [{"action": "move_to", "xyz": [x, 0, 20]}]}
        )
    assert env.image_reads == 1

    historical = toolkit.execute_tool(
        "back_project", {"pixel": [0, 0], "camera": "wrist_cam", "step": 0}
    )
    assert historical.data["world_xyz"] == [0.0, 0.0, -100.0]
    assert historical.data["historical_geometry"]
    missing = toolkit.execute_tool(
        "back_project", {"pixel": [0, 0], "camera": "wrist_cam", "step": 1}
    )
    assert "step 1" in missing.data["error"]
    assert env.image_reads == 1

    latest = toolkit.execute_tool("view_env", {"camera": "wrist_cam"})
    assert latest.data["step"] == 2 and len(latest.images) == 1
    assert env.image_reads == 2
