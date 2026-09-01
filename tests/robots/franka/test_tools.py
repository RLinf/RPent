"""Offline Franka primitive and state-capture tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robots.franka.perception import back_project
from robots.franka.runtime_config import set_calibration_path
from robots.franka.tools import (
    FrankaPrimitives,
    coerce_vec3,
    dump_state,
    view_camera_meta,
    view_env_state,
)
from rpent.tools.state import EnvState


class FakeEnv:
    def __init__(self) -> None:
        self.moves: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        self.gripper_open = True
        self.chunks: list[np.ndarray] = []

    def reset(self):
        return {"ok": True}

    def move_delta(self, value):
        self.moves.append(np.asarray(value))
        return {"ok": True}

    def rotate_delta(self, value):
        self.rotations.append(np.asarray(value))
        return {"ok": True}

    def set_gripper(self, *, open: bool):
        self.gripper_open = open
        return {"ok": True, "open": open}

    def get_observation(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.ones((1, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((1, 8, 8), dtype=np.float32) * 2,
            "states": np.zeros(8, dtype=np.float32),
        }

    def get_robot_state(self):
        return {"tcp_pose": [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}

    def get_camera_metadata(self):
        return {"depth_unit": "m", "cameras": {"wrist_1": {"fx": 100.0}}}

    def step_chunk(self, actions):
        self.chunks.append(np.asarray(actions))
        return {"terminated": False, "truncated": False}


class FakeModel:
    def predict_action_batch(self, observation, mode="eval"):
        assert observation["task_descriptions"] == "pick up the cube"
        assert mode == "eval"
        return np.zeros((2, 7), dtype=np.float32), {}


def _primitives(env: FakeEnv, *, model=None, check_cancelled=lambda: None):
    return FrankaPrimitives(
        env=env,
        model=model,
        task_description="default task",
        check_cancelled=check_cancelled,
    )


def test_vec3_validation_and_motion_forwarding():
    env = FakeEnv()
    primitives = _primitives(env)

    primitives.move_delta([0.01, 0.0, -0.02])
    primitives.rotate_delta([0.0, 0.0, 0.1])

    np.testing.assert_allclose(env.moves[0], [0.01, 0.0, -0.02])
    np.testing.assert_allclose(env.rotations[0], [0.0, 0.0, 0.1])
    with pytest.raises(ValueError, match="exactly 3"):
        coerce_vec3([1.0, 2.0], name="delta")


def test_dump_state_saves_canonical_rgbd_artifacts(tmp_path: Path):
    env = FakeEnv()
    primitives = _primitives(env)
    state = EnvState(tmp_path)

    record = dump_state(
        primitives,
        state,
        command={"action": "move_delta"},
        result={"ok": True},
        elapsed_s=0.2,
    )

    assert record.artifacts == {
        "camera.png",
        "camera_depth.npy",
        "camera_meta.json",
        "wrist.png",
        "wrist_depth.npy",
    }
    output = view_env_state(state=state)
    assert output["_image_wrist_bytes"]
    assert output["_image_cam_bytes"]
    assert view_camera_meta(state=state)["camera_meta"]["depth_unit"] == "m"


def test_vla_grasp_runs_bounded_chunks():
    env = FakeEnv()
    primitives = _primitives(env, model=FakeModel())

    result = primitives.vla_grasp("pick up the cube", max_chunks=3)

    assert result["chunks_executed"] == 3
    assert len(env.chunks) == 3


def test_back_project_reads_rpent_state_artifacts(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={"raw_base_state": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]}}
    ) as step:
        state.save("wrist_depth.npy", np.full((4, 4), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "observation_camera_map": {"main": "wrist_cam"},
                "cameras": {
                    "wrist_cam": {
                        "intrinsic_K": [[100, 0, 2], [0, 100, 2], [0, 0, 1]]
                    }
                },
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = back_project(row=2, col=2, camera="wrist", state=state)

    assert result["coordinate_frame"] == "franka_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_base"]) == 3
