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
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import robot_spec
from robots.libero import toolkit as libero_toolkit
from robots.libero.flywheel import LIBERO_SPEC
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
    def predict(self, observation, *, options):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert options["mode"] == "eval"
        return np.ones((2, 7), np.float32)


def _primitives(env, config=None, molmo_client=None):
    return LiberoPrimitives(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        flywheel_config=config,
        molmo_client=molmo_client,
    )


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_wam_refreshes_observations_and_stops_on_native_end(terminated, truncated):
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.side_effect = [{"position": 0}, {"position": 1}]
    env.reset.return_value = (_obs(0), {})
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    model.wam.get_capabilities.return_value = {"backend": "test"}
    primitives = LiberoPrimitives(env, model, Mock(), lambda: None, policy_kind="wam")
    primitives.reset()
    model.reset.assert_called_once()
    env.chunk_step.return_value = (_obs(1), 0, False, False, {})
    assert primitives.wam_act("subtask").data["chunks"] == 1

    def end(actions):
        env.terminated, env.truncated = terminated, truncated
        return _obs(2), 0, terminated, truncated, {}

    env.chunk_step.side_effect = end
    result = primitives.wam_act(max_chunks=4).data
    assert result["chunks"] == 1
    assert result["success"] == terminated and result["truncated"] == truncated
    assert primitives.wam_act().data["chunks"] == 0
    calls = model.predict.call_args_list
    assert [call.args[0]["position"] for call in calls] == [0, 1]
    assert [call.args[0]["task_descriptions"] for call in calls] == [
        "subtask",
        "put the bowl on the plate",
    ]


@pytest.mark.parametrize("with_molmo", [False, True])
def test_collection_records_scripted_and_vla_actions(tmp_path, with_molmo):
    env = _Env()
    molmo = SimpleNamespace() if with_molmo else None
    primitives = _primitives(
        env,
        {
            "root": tmp_path,
            "suite": "libero_object",
            "task_id": 2,
            "seed": 3,
        },
        molmo_client=molmo,
    )
    assert primitives.molmo_client is molmo
    primitives.reset()
    primitives.begin_primitive("move_to")
    primitives._step_env(np.zeros(7))
    primitives.end_primitive()
    primitives.begin_primitive("pi0_pick")
    primitives._vlm_chunk("pick up the bowl")
    primitives.end_primitive()

    path = primitives.finalize_flywheel()
    metadata = validate_episode(path, spec=LIBERO_SPEC)
    assert metadata["step_count"] == 3
    assert metadata["training_step_count"] == 3
    assert env.chunk_return_all_frames is True
    with np.load(path / "transitions.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action_source"], [0, 1, 1])
        np.testing.assert_array_equal(data["primitive_id"], [0, 1, 1])


def test_molmo_positional_argument_keeps_collection_disabled():
    molmo = SimpleNamespace()
    primitives = LiberoPrimitives(
        _Env(), _Model(), SimpleNamespace(), lambda: None, molmo
    )
    primitives.reset()
    assert primitives.molmo_client is molmo
    assert primitives.finalize_flywheel() is None


def test_collection_disabled_keeps_fast_chunk_path():
    env = _Env()
    primitives = _primitives(env)
    primitives.reset()
    primitives._vlm_chunk("pick up the bowl")
    assert env.chunk_return_all_frames is None
    assert primitives.finalize_flywheel() is None


@pytest.mark.parametrize("failure", [None, "predict", "step"])
def test_policy_subinstruction_does_not_mutate_environment_observations(failure):
    env = _Env()
    primitives = _primitives(env)
    initial, _ = primitives.reset()
    initial_keys = set(initial)

    def predict(observation, *, options):
        assert initial["task_descriptions"] == "put the bowl on the plate"
        assert observation["task_descriptions"] == "pick up the bowl"
        if failure == "predict":
            raise RuntimeError("prediction failed")
        return np.zeros((2, 7))

    primitives.model.predict = predict
    if failure == "step":
        env.chunk_step = Mock(side_effect=RuntimeError("step failed"))
    if failure is None:
        primitives.pi0_doubled("pick up the bowl", max_chunks=1)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            primitives.pi0_doubled("pick up the bowl", max_chunks=1)

    assert set(initial) == initial_keys
    assert initial["task_descriptions"] == "put the bowl on the plate"
    assert primitives._last_obs["task_descriptions"] == "put the bowl on the plate"


def test_dashboard_flywheel_config_belongs_to_unique_env(tmp_path, monkeypatch):
    args = SimpleNamespace(
        collect_flywheel_data=True,
        vla_backend=None,
        wam_backend=None,
        vla_endpoint=None,
        wam_endpoint=None,
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


@pytest.mark.parametrize("failure", [None, "finalize", "stop", "save"])
def test_close_handles_collection_and_video_independently(
    tmp_path, monkeypatch, failure
):
    toolkit = libero_toolkit.LiberoToolkit.__new__(libero_toolkit.LiberoToolkit)
    frames = [_obs(0)["main_images"]]
    finalize = Mock(return_value=tmp_path / "episode")
    stop = Mock(return_value=frames)
    save = Mock()
    if failure is not None:
        {"finalize": finalize, "stop": stop, "save": save}[
            failure
        ].side_effect = RuntimeError(failure)
    toolkit._primitives = SimpleNamespace(
        finalize_flywheel=finalize, stop_recording=stop
    )
    toolkit._state = SimpleNamespace(save=save)
    logger = Mock()
    monkeypatch.setattr(libero_toolkit, "logger", logger)

    toolkit.close()

    finalize.assert_called_once_with()
    stop.assert_called_once_with()
    if failure == "stop":
        save.assert_not_called()
    else:
        save.assert_called_once_with("episode.mp4", frames, step=None, fps=20)
    if failure == "finalize":
        logger.info.assert_not_called()
    else:
        logger.info.assert_called_once_with(
            "flywheel episode finalized: %s", tmp_path / "episode"
        )
    assert logger.warning.call_count == (failure is not None)
