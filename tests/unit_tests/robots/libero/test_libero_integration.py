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

import json
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


def test_seeded_vla_chunks_repeat_after_reset():
    requests = []

    class Model(_Model):
        def predict(self, observation, *, options):
            requests.append(options.copy())
            return super().predict(observation, options=options)

    primitives = LiberoPrimitives(_Env(), Model(), None, lambda: None, vla_seed=42)
    for _ in range(2):
        primitives.reset()
        primitives._vlm_chunk("pick up the bowl")
        primitives._vlm_chunk("pick up the bowl")
    assert requests[:2] == requests[2:]
    assert requests[0]["seed"] != requests[1]["seed"]


def _rl_episode_args(tmp_path, monkeypatch, init_runtime, toolkit):
    from robots.libero import rl_episode

    spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(prompt_vars={}),
        init_runtime=init_runtime,
        prompts=SimpleNamespace(render=lambda *args, **kwargs: "Move the robot."),
    )
    monkeypatch.setattr(rl_episode, "init_output_dir", lambda path: None)
    monkeypatch.setattr(rl_episode, "get_robot_spec", lambda: spec)
    monkeypatch.setattr(rl_episode, "get_toolkit", lambda **kwargs: toolkit())
    request = tmp_path / "request.json"
    planner = {
        "base_url": "http://trainer/v1",
        "model_name": "policy",
        "max_tokens": 64,
        "max_turns": 2,
        "request_timeout_s": 5,
        "episode_timeout_s": 5,
        "no_images": True,
        "history_compaction": None,
    }
    task = {"suite": "libero_goal", "task": 7, "seed": 3, "vla_seed": 1}
    request.write_text(json.dumps({"task": task, "planner": planner}))
    return SimpleNamespace(
        output_dir=tmp_path,
        request=request,
        frozen_memory_dir=None,
        tokenizer_path=None,
    )


def test_rl_episode_setup_failure_releases_started_services(tmp_path, monkeypatch):
    from robots.libero import rl_episode

    daemon = SimpleNamespace(stop=Mock())

    def toolkit():
        raise RuntimeError("env did not answer")

    args = _rl_episode_args(
        tmp_path, monkeypatch, lambda *args: ([daemon], {}), toolkit
    )
    assert rl_episode.run(args) == rl_episode.EXIT_SETUP_FAILED
    daemon.stop.assert_called_once()
    assert not (tmp_path / "outcome.json").exists()


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


@pytest.mark.parametrize("failure", [None, "finalize", "stop", "save"])
def test_close_handles_collection_and_video_independently(
    tmp_path, monkeypatch, failure
):
    toolkit = libero_toolkit.LiberoToolkit.__new__(libero_toolkit.LiberoToolkit)
    from rpent.memory import MemoryManager
    from rpent.tools import Toolkit

    Toolkit.__init__(
        toolkit, dashboard_events=None, memory=MemoryManager(tmp_path / "memory")
    )
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


@pytest.mark.parametrize("after_reset", [False, True])
def test_recipe_exports_each_segment_once_and_keeps_repeated_calls(
    tmp_path, after_reset
):
    import json

    from robots.libero.tools import write_recipe_from_states
    from rpent.session import EnvState

    state = EnvState(tmp_path)
    command = {"action": "segment", "prompt": "cup", "camera": "agentview"}
    artifact = {
        "mode": "text",
        "prompt": "cup",
        "camera": "agentview",
        "segment_index": 0,
    }
    if after_reset:
        with state.record_step(state={}, command={"action": "move_to"}, result={}):
            state.save("segment_00.json", artifact)
    with state.record_step(
        state={}, command={"action": "reset"} if after_reset else None, result={}
    ):
        pass
    for index in range(2):
        state.save("segment_00.json", artifact)
        with state.record_step(
            state={}, command=command, result={}, terminated=index == 1
        ):
            pass
    name = write_recipe_from_states(state, "test", output_dir=tmp_path)
    commands = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert commands == [command, command]


def test_rl_episode_reports_outcome_after_releasing_the_scene(tmp_path, monkeypatch):
    from robots.libero import rl_episode

    released, seen = [], {}
    daemon = SimpleNamespace(stop=lambda: released.append("env"))

    def init_runtime(args, output, events, components):
        seen.update(output=output, components=components, task=args.task)
        return [daemon], {}

    def toolkit():
        return SimpleNamespace(
            primitives=SimpleNamespace(reset_info={"init_state_id": 3}),
            solved=lambda: True,
            state=SimpleNamespace(get=lambda: SimpleNamespace(truncated=False)),
            close=lambda: released.append("toolkit"),
        )

    async def solve(*args, **kwargs):
        assert not released and not (tmp_path / "outcome.json").exists()
        return SimpleNamespace(
            messages=[],
            stats={"turns_used": 1},
            finish_result=None,
            error=None,
            error_kind=None,
        )

    args = _rl_episode_args(tmp_path, monkeypatch, init_runtime, toolkit)
    monkeypatch.setattr(rl_episode, "solve", solve)
    assert rl_episode.run(args) == 0
    assert seen == {"output": tmp_path, "components": {"env", "vla", "sam3"}, "task": 7}
    assert released == ["toolkit", "env"]
    # The trainer reads exactly these fields.
    assert json.loads((tmp_path / "outcome.json").read_text()) == {
        "success": True,
        "truncated": False,
        "terminal": True,
        "termination_reason": "success",
        "error": None,
        "error_kind": None,
        "init_state_id": 3,
        "memory_entries": [],
    }
    task = json.loads(args.request.read_text())
    task["task"]["memory_entries"] = [{"id": "task-family/task.md", "sha256": "0"}]
    args.request.write_text(json.dumps(task))
    with pytest.raises(ValueError, match="given together"):
        rl_episode.run(args)
