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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import robot_spec
from robots.libero import toolkit as libero_toolkit
from robots.libero.flywheel import LIBERO_SPEC
from robots.libero.toolkit import LiberoRuntime
from robots.libero.tools import _step_env, _vlm_chunk
from rpent.flywheel.episode import validate_episode
from rpent.tools import ToolContext


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
    def predict(self, observation, *, options):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert options["mode"] == "eval"
        return np.ones((2, 7), np.float32)


def _runtime(env, config=None, molmo_client=None):
    return LiberoRuntime(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        flywheel_config=config,
        molmo_client=molmo_client,
    )


def _context(runtime):
    return ToolContext(
        robot=runtime,
        state=None,
        memory=None,
        output_dir=None,
        record_frame=lambda frame: None,
        _cancel_event=threading.Event(),
    )


@pytest.mark.parametrize("with_molmo", [False, True])
def test_collection_records_scripted_and_vla_actions(tmp_path, with_molmo):
    env = _Env()
    molmo = SimpleNamespace() if with_molmo else None
    runtime = _runtime(
        env,
        {
            "root": tmp_path,
            "suite": "libero_object",
            "task_id": 2,
            "seed": 3,
        },
        molmo_client=molmo,
    )
    assert runtime.molmo_client is molmo
    runtime.reset()
    ctx = _context(runtime)
    runtime.execute_primitive("move_to", _step_env, ctx=ctx, action=np.zeros(7))
    runtime.execute_primitive(
        "pi0_pick", _vlm_chunk, ctx=ctx, instruction="pick up the bowl"
    )

    path = runtime.finalize_flywheel()
    metadata = validate_episode(path, spec=LIBERO_SPEC)
    assert metadata["step_count"] == 3
    assert metadata["training_step_count"] == 3
    assert env.chunk_return_all_frames is True
    with np.load(path / "transitions.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action_source"], [0, 1, 1])
        np.testing.assert_array_equal(data["primitive_id"], [0, 1, 1])


def test_optional_molmo_keeps_collection_disabled():
    molmo = SimpleNamespace()
    runtime = LiberoRuntime(_Env(), _Model(), SimpleNamespace(), molmo)
    runtime.reset()
    assert runtime.molmo_client is molmo
    assert runtime.finalize_flywheel() is None


def test_collection_disabled_keeps_native_frame_recording():
    env = _Env()
    runtime = _runtime(env)
    runtime.reset()
    _vlm_chunk(_context(runtime), "pick up the bowl")
    assert env.chunk_return_all_frames is True
    assert runtime.finalize_flywheel() is None


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


@pytest.mark.parametrize("failure", [None, "finalize", "save"])
def test_close_handles_collection_and_video_independently(
    tmp_path, monkeypatch, failure
):
    toolkit = libero_toolkit.LiberoToolkit.__new__(libero_toolkit.LiberoToolkit)
    frames = [_obs(0)["main_images"]]
    finalize = Mock(return_value=tmp_path / "episode")
    save = Mock()
    if failure is not None:
        {"finalize": finalize, "save": save}[failure].side_effect = RuntimeError(
            failure
        )
    toolkit._robot = SimpleNamespace(finalize_flywheel=finalize)
    toolkit._frames = frames
    toolkit._state = SimpleNamespace(save=save)
    logger = Mock()
    monkeypatch.setattr(libero_toolkit, "logger", logger)

    toolkit.close()

    finalize.assert_called_once_with()
    save.assert_called_once_with("episode.mp4", frames, step=None, fps=20)
    if failure == "finalize":
        logger.info.assert_not_called()
    else:
        logger.info.assert_called_once_with(
            "flywheel episode finalized: %s", tmp_path / "episode"
        )
    assert logger.warning.call_count == (failure is not None)


@pytest.mark.parametrize("fails", [False, True])
def test_native_action_brackets_flywheel_recording(make_toolkit, monkeypatch, fails):
    from robots.libero import flywheel

    writer = Mock()
    monkeypatch.setattr(flywheel, "create_episode_writer", lambda config, obs: writer)
    toolkit, env, _ = make_toolkit(flywheel_config={"enabled": True})
    if fails:
        monkeypatch.setattr(env, "step", Mock(side_effect=RuntimeError("step failed")))

    result = toolkit.execute_tool("release", {})

    assert result.is_error is fails
    writer.begin_primitive.assert_called_once_with("release")
    writer.end_primitive.assert_called_once_with()
    if fails:
        writer.add_transition.assert_not_called()
    else:
        assert writer.add_transition.call_count > 0
