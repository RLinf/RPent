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
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.libero import toolkit as libero_toolkit
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.flywheel.flywheel import select_vla_segments
from rpent.flywheel.reader import iter_trajectories
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.robots.components.action_spec import validate_action
from rpent.session import EnvState, Payload
from rpent.session.env_capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.rpc import RpcError


def _obs(value: int) -> dict:
    return {
        "main_images": np.full((256, 256, 3), value, np.uint8),
        "wrist_images": np.full((256, 256, 3), value + 1, np.uint8),
        "states": np.full(8, value, np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


class _Env:
    action_specs = {"default": {"low": [-1.0] * 7, "high": [1.0] * 7}}

    def __init__(self):
        self.terminated = False
        self.truncated = False
        self.return_all_frames = False
        self.chunk_return_all_frames = None
        self.last_obs = _obs(0)

    def reset(self):
        self.last_obs = _obs(0)
        return self.last_obs, {}

    def _step(self, action):
        del action
        return _obs(1), 0, False, False, {}

    def validate_action(self, values):
        return validate_action(values, self.action_specs["default"])

    def _chunk_step(self, actions, *, return_all_frames=None):
        observations = [_obs(index + 2) for index in range(len(actions))]
        result = observations if return_all_frames else observations[-1]
        terminated = np.zeros(len(actions), np.bool_)
        terminated[-1] = True
        self.terminated = True
        return (
            result,
            np.zeros(len(actions)),
            terminated,
            np.zeros(len(actions)),
            [{} for _ in actions],
        )

    def step(self, action):
        result = self._step(action)
        self.last_obs = result[0]
        return result

    def chunk_step(self, actions, *, return_all_frames=None):
        self.chunk_return_all_frames = return_all_frames
        all_frames = (
            self.return_all_frames if return_all_frames is None else return_all_frames
        )
        result = self._chunk_step(actions, return_all_frames=all_frames)
        self.last_obs = result[0][-1] if all_frames else result[0]
        return result


class _Model:
    def predict(self, observation, *, options):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert options["mode"] == "eval"
        return np.ones((2, 7), np.float32)


def _primitives(env, molmo_client=None):
    return LiberoPrimitives(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        molmo_client=molmo_client,
    )


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_wam_refreshes_observations_and_stops_on_native_end(terminated, truncated):
    env = Mock(terminated=False, truncated=False, return_all_frames=False)
    env.raw_obs.side_effect = [{"position": 0}, {"position": 1}]
    env.reset.return_value = (_obs(0), {})
    model = Mock(predict=Mock(return_value=np.zeros((16, 7))))
    model.wam.model_meta = {"backend": "test"}
    primitives = LiberoPrimitives(env, model, Mock(), lambda: None, policy_kind="wam")
    primitives.reset()
    model.reset.assert_called_once()
    env.chunk_step.return_value = (_obs(1), 0, False, False, {})
    assert primitives.wam_act("subtask").data["chunks"] == 1

    def end(actions, *, return_all_frames=None):
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


@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_wam_toolkit_rejects_motion_after_episode_end(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "dump_state",
        lambda *args, **kwargs: SimpleNamespace(step_idx=0, terminated=False),
    )
    env = _Env()
    toolkit = libero_toolkit.LiberoToolkit(
        runtime_kwargs={"env": env, "model": Mock(), "sam3_client": Mock()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
        policy_kind="wam",
        enable_direct_action=True,
    )
    toolkit.get_env_state = lambda **kwargs: ToolResult(data={})
    setattr(env, terminal, True)
    env.step = Mock()
    env.chunk_step = Mock()
    for name, arguments in (
        ("release", {"max_steps": 1}),
        ("execute_action", {"values": [0.0] * 7}),
        ("wam_act", {}),
    ):
        result = toolkit.execute_tool(name, arguments)
        assert result.is_error and "Episode already ended" in result.data["error"]
    env.step.assert_not_called()
    env.chunk_step.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "record_trajectory"),
    [("evaluation", True), ("exploration", True), ("evaluation", False)],
)
def test_actions_follow_api_tool_calls(tmp_path, monkeypatch, mode, record_trajectory):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr("rpent.planner.api_loop.substitute", lambda value: value)
    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "dump_state",
        lambda *args, **kwargs: SimpleNamespace(step_idx=0, terminated=False),
    )
    monkeypatch.setattr(
        libero_toolkit.LiberoToolkit,
        "get_env_state",
        lambda self, *, command, result, elapsed_s: ToolResult(data=result),
    )
    env = _Env()
    events = NullDashboardEventSink()
    toolkit = libero_toolkit.LiberoToolkit(
        runtime_kwargs={
            "env": env,
            "model": _Model(),
            "sam3_client": SimpleNamespace(),
        },
        dashboard_events=events,
        memory=MemoryManager(tmp_path / "memory"),
        mode=mode,
        enable_direct_action=True,
        state_output_dir=tmp_path / "state",
    )
    if record_trajectory:
        toolkit.state.start_task(
            instruction="release, use VLA, then move",
            config=Payload(format="test", data={}),
        )
    executed = []

    def step(action):
        executed.append(action)
        result = (
            _obs(len(executed)),
            np.float32(0.5),
            np.bool_(False),
            np.bool_(False),
            {
                "step": np.int64(len(executed)),
                "episode": {"episode_len": np.array([len(executed)], np.int32)},
            },
        )
        return result

    monkeypatch.setattr(env, "_step", step)
    chunk_calls = []

    def chunk_step(actions, *, return_all_frames=None):
        chunk_calls.append(return_all_frames)
        returned = [step(action) for action in actions]
        observations, rewards, terminated, truncated, infos = zip(*returned)
        return (
            list(observations) if return_all_frames else observations[-1],
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            list(infos),
        )

    monkeypatch.setattr(env, "_chunk_step", chunk_step)
    direct = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0]
    calls = iter(
        [
            ("release", {"max_steps": 2}),
            ("pi0_doubled", {"prompt": "pick up the bowl", "max_chunks": 2}),
            ("execute_action", {"values": direct}),
            ("finish", {"status": "success", "summary": "done"}),
        ]
    )

    async def stream(messages, info):
        name, arguments = next(calls)
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

    # Manual execution must not borrow an API call's identity.
    toolkit.execute_tool("release", {"max_steps": 1})
    result = ApiAgentLoop(
        model=FunctionModel(stream_function=stream), dashboard_events=events
    ).solve(
        system_prompt="Use the robot toolkit.",
        user_message="release, use VLA, then move",
        toolkit=toolkit,
        max_turns=4,
    )
    toolkit.resume_calls()
    toolkit.execute_tool("release", {"max_steps": 1})

    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert len(executed) == 9
    assert chunk_calls == [True, True]
    actions = toolkit.state.action_records()
    if not record_trajectory:
        assert toolkit.state.get_task() is None
        assert toolkit.state.planner_records() == []
        assert actions == []
        assert toolkit.state.prediction_records() == []
        assert not (tmp_path / "state" / "trajectories").exists()
        return
    if mode == "exploration":
        assert actions == [] and toolkit.state.prediction_records() == []
        return

    assert [action.source for action in actions] == ["tool_computed"] * 2 + [
        "vla"
    ] * 4 + ["direct"]
    np.testing.assert_array_equal(
        [action.action.data for action in actions], executed[1:-1]
    )
    for name, group in (
        ("release", actions[:2]),
        ("pi0_doubled", actions[2:6]),
        ("execute_action", actions[6:]),
    ):
        call = toolkit.state.get_tool_call(group[0].tool_call_id)
        assert call.name == name and all(
            a.tool_call_id == call.tool_call_id for a in group
        )
        assert call.tool_call_id in [
            t.tool_call_id
            for t in toolkit.state.get_planner(call.planner_id).tool_calls
        ]
    assert [
        prediction.action_ids for prediction in toolkit.state.prediction_records()
    ] == [
        [action.action_id for action in actions[2:4]],
        [action.action_id for action in actions[4:6]],
    ]


@pytest.fixture
def captured_primitives(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="release", config=Payload(format="test", data={}))
    planner_id = state.register_planner(backend="api")
    tool_call_id = state.register_tool_call(
        planner_id=planner_id,
        name="release",
        arguments=Payload(format="json", data={"max_steps": 3}),
    )
    capture = ToolCallCapture(state, tool_call_id)
    capture.start()
    env = _Env()
    primitives = _primitives(
        CapturingEnvClient(
            env, state, capture_call=partial(libero_toolkit._capture_libero_call, env)
        )
    )
    primitives.reset()
    return primitives, capture


@pytest.mark.parametrize("failure", ["step", "cancel", "frame"])
def test_action_capture_keeps_confirmed_prefix(
    captured_primitives, monkeypatch, failure
):
    primitives, capture = captured_primitives
    if failure == "step":
        monkeypatch.setattr(
            primitives.env.wrapped,
            "_step",
            Mock(side_effect=[(_obs(1), 0, False, False, {}), RuntimeError("step")]),
        )
    elif failure == "cancel":
        monkeypatch.setattr(
            primitives,
            "_check_cancelled",
            Mock(side_effect=[None, ToolCancelled("cancel")]),
        )
    else:
        primitives.start_recording()
        monkeypatch.setattr(
            primitives, "record_frame", Mock(side_effect=RuntimeError("frame"))
        )

    with capture.bind(), pytest.raises((RuntimeError, ToolCancelled), match=failure):
        primitives.release(max_steps=3)

    actions = capture.state.action_records()
    assert len(actions) == 1
    call = capture.state.get_tool_call(capture.tool_call_id)
    assert call.before_observation_id == actions[0].before_observation_id
    if failure == "step":
        assert call.after_observation_id is None
        assert call.missing["after_observation_id"].reason == "capture_failed"
    else:
        assert call.after_observation_id == actions[0].after_observation_id
        assert "after_observation_id" not in call.missing
    assert capture.state.get_observation(actions[0].after_observation_id)


@pytest.mark.parametrize("policy_kind", ["vla", "wam"])
def test_policy_actions_survive_video_failure(
    captured_primitives, monkeypatch, policy_kind
):
    primitives, capture = captured_primitives
    primitives.policy_kind = policy_kind
    if policy_kind == "wam":
        primitives.env.wrapped.raw_obs = lambda: primitives._last_obs
        primitives.model.wam = SimpleNamespace(model_meta={"backend": "test"})
        act = primitives.wam_act
    else:
        act = primitives._vlm_chunk
    primitives.start_recording()
    monkeypatch.setattr(
        primitives, "record_frame", Mock(side_effect=RuntimeError("video"))
    )
    with capture.bind(), pytest.raises(RuntimeError, match="video"):
        act("pick up the bowl")
    actions = capture.state.action_records()
    assert len(actions) == 2
    assert [a.source for a in actions] == ["vla", "vla"]
    assert [a.result.data["terminated"] for a in actions] == [False, True]
    (prediction,) = capture.state.prediction_records()
    assert prediction.action_ids == [a.action_id for a in actions]


def test_cancelling_a_prediction_before_execution_records_no_actions(
    captured_primitives, monkeypatch
):
    primitives, capture = captured_primitives
    monkeypatch.setattr(
        primitives,
        "_check_cancelled",
        Mock(side_effect=[None, ToolCancelled("cancel")]),
    )
    with capture.bind(), pytest.raises(ToolCancelled):
        primitives._vlm_chunk("pick up the bowl")
    (prediction,) = capture.state.prediction_records()
    assert prediction.action_ids == []
    assert capture.state.action_records() == []
    assert primitives.env.chunk_return_all_frames is None


@pytest.mark.parametrize("capture_actions", [False, True])
def test_no_capture_context_keeps_fast_chunk_path(capture_actions, tmp_path):
    env = _Env()
    state = EnvState(tmp_path)
    primitives = _primitives(
        CapturingEnvClient(
            env, state, capture_call=partial(libero_toolkit._capture_libero_call, env)
        )
        if capture_actions
        else env
    )
    primitives.reset()
    primitives._vlm_chunk("pick up the bowl")
    assert env.chunk_return_all_frames is None
    assert state.action_records() == state.prediction_records() == []
    assert not (tmp_path / "trajectories").exists()


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


@pytest.mark.parametrize("failure", [None, "stop", "save"])
def test_close_saves_video_and_handles_video_errors(tmp_path, monkeypatch, failure):
    toolkit = libero_toolkit.LiberoToolkit.__new__(libero_toolkit.LiberoToolkit)
    from rpent.memory import MemoryManager
    from rpent.tools import Toolkit

    Toolkit.__init__(
        toolkit, dashboard_events=None, memory=MemoryManager(tmp_path / "memory")
    )
    frames = [_obs(0)["main_images"]]
    stop = Mock(return_value=frames)
    save = Mock()
    if failure is not None:
        {"stop": stop, "save": save}[failure].side_effect = RuntimeError(failure)
    toolkit._primitives = SimpleNamespace(stop_recording=stop)
    toolkit._state = SimpleNamespace(save=save)
    logger = Mock()
    monkeypatch.setattr(libero_toolkit, "logger", logger)

    toolkit.close()

    stop.assert_called_once_with()
    if failure == "stop":
        save.assert_not_called()
    else:
        save.assert_called_once_with("episode.mp4", frames, step=None, fps=20)
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


class _LiberoBackend:
    def __init__(self):
        self.executed = []
        self.wrap_count = 0
        self.reset_count = 0
        self.fail_at = None
        self.success_at = 3

    def _observation(self):
        return _obs(len(self.executed))

    def _wrap_obs(self, observations):
        self.wrap_count += 1
        return {
            key: np.stack([obs[key] for obs in observations])
            if isinstance(value, np.ndarray)
            else [obs[key] for obs in observations]
            for key, value in observations[0].items()
        }

    def reset(self):
        self.reset_count += 1
        self.executed.clear()
        self.current_raw_obs = [self._observation()]
        return self._wrap_obs(self.current_raw_obs), {}

    def step(self, actions):
        assert actions.shape == (1, 7)
        self.executed.append(actions[0].copy())
        self.current_raw_obs = [self._observation()]
        if len(self.executed) == self.fail_at:
            raise RuntimeError("native step failed")
        return (
            self._wrap_obs(self.current_raw_obs),
            np.array([0.5], dtype=np.float32),
            np.array([len(self.executed) == self.success_at]),
            np.array([False]),
            {"episode": {"episode_len": np.array([len(self.executed)])}},
        )

    def chunk_step(self, actions):
        steps = [self.step(actions[:, index]) for index in range(actions.shape[1])]
        return (
            [step[0] for step in steps],
            *(np.stack([step[index] for step in steps], axis=1) for index in (1, 2, 3)),
            [step[4] for step in steps],
        )


@pytest.mark.parametrize(
    ("record", "all_frames", "default_all_frames"),
    [(True, False, True), (True, None, True), (False, None, False)],
)
def test_libero_client_facade_capture_and_raw_trajectory_roundtrip(
    make_server_and_client, tmp_path, all_frames, default_all_frames, record
):
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade
    from rpent.flywheel.reader import read_array

    backend = _LiberoBackend()
    state = EnvState(tmp_path)
    capture = None
    if record:
        state.start_task(
            instruction="put the bowl on the plate",
            config=Payload(
                "test.config", {"arguments": {"robot_name": "libero", "seed": 0}}
            ),
        )
        tool_id = state.register_tool_call(
            planner_id=state.register_planner(backend="api"),
            name="pi0_pick",
            arguments=Payload("json", {}),
        )
        capture = ToolCallCapture(state, tool_id)
    commands = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)
    with make_server_and_client(LiberoEnvFacade(backend, meta={}), "socket") as rpc:
        client = LiberoEnvClient(
            rpc, expected_meta={}, return_all_frames=default_all_frames
        )
        env = CapturingEnvClient(
            client,
            state,
            capture_call=partial(libero_toolkit._capture_libero_call, client),
        )
        with capture.bind() if capture else nullcontext():
            env.step(np.zeros(7, dtype=np.float32))
            with env.prediction("pick up the bowl"):
                result = env.chunk_step(commands, return_all_frames=all_frames)
        assert client.terminated and not client.truncated
        np.testing.assert_array_equal(result[2], [False, True, False, False, False])
        np.testing.assert_array_equal(client.last_obs["states"], np.full(8, 6))
        expected_all_frames = default_all_frames if all_frames is None else all_frames
        assert isinstance(result[0], list) is expected_all_frames
        assert (result[0][-1] if expected_all_frames else result[0]) is client.last_obs
    assert backend.reset_count == 1
    assert backend.wrap_count == 7
    np.testing.assert_array_equal(backend.executed[1:], commands)
    if not record:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    trajectory = next(iter_trajectories(root))
    actions = trajectory["actions"]
    (prediction,) = trajectory["predictions"]
    assert prediction["action_ids"] == [action["action_id"] for action in actions[1:]]
    assert prediction["instruction"] == "pick up the bowl"
    assert [action["result"]["data"]["terminated"] for action in actions] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    for index, (action, command) in enumerate(
        zip(actions, backend.executed, strict=True)
    ):
        saved_action = read_array(root, action["action"]["data"])
        np.testing.assert_array_equal(saved_action, command)
        assert saved_action.dtype == command.dtype
        for side, value in (("before", index), ("after", index + 1)):
            data = action[f"{side}_observation"]["payload"]["data"]
            np.testing.assert_array_equal(
                read_array(root, data["states"]), np.full(8, value)
            )
            np.testing.assert_array_equal(
                read_array(root, data["main_images"]), np.full((256, 256, 3), value)
            )


def test_libero_failed_chunk_discards_unconfirmed_actions(
    make_server_and_client, tmp_path
):
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade

    state = EnvState(tmp_path)
    state.start_task(
        instruction="move",
        config=Payload("test.config", {"arguments": {"robot_name": "libero"}}),
    )
    tool_id = state.register_tool_call(
        planner_id=state.register_planner(backend="api"),
        name="move",
        arguments=Payload("json", {}),
    )
    backend = _LiberoBackend()
    with make_server_and_client(LiberoEnvFacade(backend, meta={}), "http") as rpc:
        client = LiberoEnvClient(rpc, expected_meta={})
        env = CapturingEnvClient(
            client,
            state,
            capture_call=partial(libero_toolkit._capture_libero_call, client),
        )
        with ToolCallCapture(state, tool_id).bind():
            env.step(np.zeros(7))
            backend.fail_at = 3
            with pytest.raises(RpcError, match="native step failed"):
                with env.prediction("move"):
                    env.chunk_step(np.ones((4, 7)))
            assert len(state.action_records()) == 1
            assert state.prediction_records()[0].action_ids == []
            call = state.get_tool_call(tool_id)
            assert call.missing["after_observation_id"].reason == "capture_failed"
            np.testing.assert_array_equal(client.last_obs["states"], np.full(8, 1))
            np.testing.assert_array_equal(
                backend.current_raw_obs[0]["states"], np.full(8, 3)
            )
            backend.success_at = 5
            with env.prediction("move"):
                env.chunk_step(np.ones((2, 7)))
    assert len(backend.executed) == 5
    first, recovered, following = state.action_records()
    assert recovered.before_observation_id != first.after_observation_id
    before = state.get_observation(recovered.before_observation_id).payload
    assert before.data is None
    assert before.completeness == "partial"
    assert before.missing[""].reason == "capture_failed"
    assert following.before_observation_id == recovered.after_observation_id
    np.testing.assert_array_equal(
        state.get_observation(following.before_observation_id).payload.data["states"],
        np.full(8, 4),
    )
    state.end_task(
        status="completed",
        stop_reason="finish",
        result=Payload("test", {"environment_success": True}),
    )
    trajectory = next(iter_trajectories(state.save_planner_trajectory()))
    selected = select_vla_segments(trajectory)
    assert selected["segments"] == [
        {"start_index": 2, "stop_index": 3, "instruction": "move"}
    ]
    assert [item["action_id"] for item in selected["rejected"]] == [recovered.action_id]
