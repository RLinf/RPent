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

from types import SimpleNamespace

import numpy as np

from robots.libero import robot_spec
from robots.libero.tools import LiberoPrimitives
from rpent.flywheel.episode import validate_episode


def _obs(value: int) -> dict:
    return {
        "main_images": np.full((256, 256, 3), value, np.uint8),
        "wrist_images": np.full((256, 256, 3), value + 1, np.uint8),
        "states": np.full(8, value, np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


class _Env:
    def __init__(self):
        self.terminated = False
        self.truncated = False
        self.return_all_frames = False
        self.chunk_return_all_frames = None

    def reset(self):
        return _obs(0), {}

    def step(self, action):
        del action
        return _obs(1), 0, False, False, {}

    def chunk_step(self, actions, *, return_all_frames=None):
        self.chunk_return_all_frames = return_all_frames
        observations = [_obs(index + 2) for index in range(len(actions))]
        result = observations if return_all_frames else observations[-1]
        terminated = np.zeros(len(actions), np.bool_)
        terminated[-1] = True
        self.terminated = True
        return result, np.zeros(len(actions)), terminated, np.zeros(len(actions)), {}


class _Model:
    def predict_action_batch(self, observation, *, mode):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert mode == "eval"
        return np.ones((2, 7), np.float32), {}


def _primitives(env, config=None):
    return LiberoPrimitives(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        flywheel_config=config,
    )


def test_collection_records_scripted_and_vla_actions(tmp_path):
    env = _Env()
    primitives = _primitives(
        env,
        {
            "root": tmp_path,
            "suite": "libero_object",
            "task_id": 2,
            "seed": 3,
        },
    )
    primitives.reset()
    primitives.begin_primitive("move_to")
    primitives._step_env(np.zeros(7))
    primitives.end_primitive()
    primitives.begin_primitive("pi0_pick")
    primitives._vlm_chunk("pick up the bowl")
    primitives.end_primitive()

    path = primitives.finalize_flywheel()
    metadata = validate_episode(path)
    assert metadata["step_count"] == 3
    assert metadata["training_step_count"] == 3
    assert env.chunk_return_all_frames is True
    with np.load(path / "transitions.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action_source"], [0, 1, 1])
        np.testing.assert_array_equal(data["primitive_id"], [0, 1, 1])


def test_collection_disabled_keeps_fast_chunk_path():
    env = _Env()
    primitives = _primitives(env)
    primitives.reset()
    primitives._vlm_chunk("pick up the bowl")
    assert env.chunk_return_all_frames is None
    assert primitives.finalize_flywheel() is None


def test_dashboard_flywheel_config_belongs_to_unique_env(tmp_path, monkeypatch):
    args = SimpleNamespace(
        collect_flywheel_data=True,
        flywheel_root=str(tmp_path),
        suite="libero_object",
        task=2,
        seed=3,
    )
    monkeypatch.setattr(
        robot_spec,
        "try_spawn_server",
        lambda owned, events, component, starter: (None, component),
    )
    monkeypatch.setattr(
        robot_spec,
        "try_wait_server",
        lambda *args, **kwargs: {},
    )

    _, shared = robot_spec._init_runtime(args, tmp_path, None, {"vla", "sam3"})
    _, unique = robot_spec._init_runtime(args, tmp_path, None, {"env"})

    assert "flywheel_config" not in shared
    assert unique["flywheel_config"] == {
        "root": str(tmp_path),
        "suite": "libero_object",
        "task_id": 2,
        "seed": 3,
    }
