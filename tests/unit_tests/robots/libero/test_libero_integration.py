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
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.libero import robot_spec
from robots.libero import toolkit as libero_toolkit
from robots.libero.flywheel import LIBERO_SPEC
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.flywheel.episode import validate_episode
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.robots.components.action_spec import validate_action
from rpent.session import ArtifactRef, EnvState, Payload
from rpent.session.capture import ToolCallCapture
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled


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

    def reset(self):
        return _obs(0), {}

    def step(self, action):
        del action
        return _obs(1), 0, False, False, {}

    def validate_action(self, values):
        return validate_action(values, self.action_specs["default"])

    def chunk_step(self, actions, *, return_all_frames=None):
        self.chunk_return_all_frames = return_all_frames
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


class _Model:
    def predict(self, observation, *, options):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert options["mode"] == "eval"
        return np.ones((2, 7), np.float32)


def _primitives(env, config=None, molmo_client=None, *, capture_actions=False):
    return LiberoPrimitives(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        flywheel_config=config,
        molmo_client=molmo_client,
        capture_actions=capture_actions,
    )


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
    results = []

    def step(action):
        executed.append(action)
        result = (
            _obs(len(executed)),
            np.float32(0.5),
            np.bool_(False),
            np.bool_(False),
            {
                "step": np.int64(len(executed)),
                "custom_stat": np.array([42], np.int32),
                "episode": {
                    "success_once": np.array([False]),
                    "return": np.array([0.5 * len(executed)], np.float32),
                    "episode_len": np.array([len(executed)], np.int32),
                    "reward": np.array([0.5], np.float32),
                },
            },
        )
        results.append(result)
        return result

    monkeypatch.setattr(env, "step", step)
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

    monkeypatch.setattr(env, "chunk_step", chunk_step)
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
    toolkit.state.end_task(status="completed", stop_reason="finish")
    root = toolkit.state.save_planner_trajectory()
    saved = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    saved_predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
    ]
    if mode == "exploration":
        assert actions == []
        assert saved == []
        assert toolkit.state.prediction_records() == []
        assert saved_predictions == []
        return

    assert [row["action_id"] for row in saved] == [
        action.action_id for action in actions
    ]
    assert [row["source"] for row in saved] == [action.source for action in actions]
    image = saved[-1]["after_observation"]["payload"]["data"]["main_images"]
    np.testing.assert_array_equal(
        np.load(root / image["path"], allow_pickle=False), _obs(8)["main_images"]
    )
    assert [action.source for action in actions] == [
        "tool_computed",
        "tool_computed",
        "vla",
        "vla",
        "vla",
        "vla",
        "direct",
    ]
    assert actions[0].tool_call_id == actions[1].tool_call_id
    assert actions[1].tool_call_id != actions[2].tool_call_id
    for previous, current in zip(actions, actions[1:]):
        assert previous.after_observation_id == current.before_observation_id
    for index, action in enumerate(actions, start=1):
        row = saved[index - 1]
        inline = [(row["action"]["data"], action.action.data)]
        np.testing.assert_array_equal(action.action.data, executed[index])
        assert action.action.data.dtype == executed[index].dtype
        assert action.result.data == dict(
            zip(("reward", "terminated", "truncated", "info"), results[index][1:])
        )
        assert isinstance(action.result.data["reward"], np.float32)
        assert action.executed_at is None
        assert action.missing["executed_at"].reason == "not_exposed"
        for side, observation_id, value in (
            ("before", action.before_observation_id, index),
            ("after", action.after_observation_id, index + 1),
        ):
            observation = toolkit.state.get_observation(observation_id)
            inline.append(
                (
                    row[f"{side}_observation"]["payload"]["data"]["states"],
                    observation.payload.data["states"],
                )
            )
            for key, expected in _obs(value).items():
                stored = observation.payload.data[key]
                if isinstance(stored, ArtifactRef):
                    stored = np.load(root / stored.path, allow_pickle=False)
                np.testing.assert_array_equal(stored, expected)
        info = row["result"]["data"]["info"]
        inline.extend(
            (info["episode"][key], value)
            for key, value in action.result.data["info"]["episode"].items()
        )
        for encoded, native in inline:
            assert encoded == {
                "$type": "ndarray",
                "dtype": str(native.dtype),
                "shape": list(native.shape),
                "value": native.tolist(),
            }
        np.testing.assert_array_equal(
            np.load(root / info["custom_stat"]["path"], allow_pickle=False), [42]
        )

    tool_calls = [
        toolkit.state.get_tool_call(group[0].tool_call_id)
        for group in (actions[:2], actions[2:6], actions[6:])
    ]
    assert [call.name for call in tool_calls] == [
        "release",
        "pi0_doubled",
        "execute_action",
    ]
    assert len({call.planner_id for call in tool_calls}) == 3
    for call, group in zip(tool_calls, (actions[:2], actions[2:6], actions[6:])):
        assert {action.tool_call_id for action in group} == {call.tool_call_id}
        assert call.before_observation_id == group[0].before_observation_id
        assert call.after_observation_id == group[-1].after_observation_id
        assert call.status == "completed"
        assert "before_observation_id" not in call.missing
        assert "after_observation_id" not in call.missing
        planner = toolkit.state.get_planner(call.planner_id)
        assert call.tool_call_id in {item.tool_call_id for item in planner.tool_calls}

    vla_call = tool_calls[1]
    assert vla_call.effective_instruction == "pick up the bowl"
    assert vla_call.model_config.data == {"predict_options": {"mode": "eval"}}
    assert vla_call.model_config.completeness == "partial"
    assert vla_call.model_config.missing["/server_config"].reason == "not_exposed"
    assert "effective_instruction" not in vla_call.missing
    assert "model_config" not in vla_call.missing

    assert len(saved_predictions) == 2
    for row, group in zip(saved_predictions, (actions[2:4], actions[4:6])):
        assert row["planner_id"] == vla_call.planner_id
        assert row["tool_call_id"] == vla_call.tool_call_id
        assert row["instruction"] == "pick up the bowl"
        assert row["action_ids"] == [action.action_id for action in group]
        assert (
            row["input_observation"]["observation_id"]
            == row["input_observation_id"]
            == group[0].before_observation_id
        )
        assert row["actions"]["format"] == "libero.action_chunk"
        assert row["actions"]["completeness"] == "complete"
        assert row["actions"]["data"] == {
            "$type": "ndarray",
            "dtype": "float32",
            "shape": [2, 7],
            "value": [action.action.data.tolist() for action in group],
        }

    executed[1][:] = 99
    results[1][0]["states"][:] = 99
    stored = toolkit.state.get_action(actions[0].action_id)
    np.testing.assert_array_equal(stored.action.data, [0, 0, 0, 0, 0, 0, -1])
    observation = toolkit.state.get_observation(stored.after_observation_id)
    np.testing.assert_array_equal(observation.payload.data["states"], _obs(2)["states"])


@pytest.fixture
def captured_primitives(tmp_path):
    primitives = _primitives(_Env(), capture_actions=True)
    primitives.reset()
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
    return primitives, capture


@pytest.mark.parametrize("failure", ["step", "cancel", "frame"])
def test_action_capture_keeps_confirmed_prefix(
    captured_primitives, monkeypatch, failure
):
    primitives, capture = captured_primitives
    if failure == "step":
        monkeypatch.setattr(
            primitives.env,
            "step",
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


@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_action_capture_preserves_terminal_flags(
    captured_primitives, monkeypatch, terminal
):
    primitives, capture = captured_primitives

    def step(action):
        setattr(primitives.env, terminal, True)
        return (
            _obs(1),
            np.float32(1),
            np.bool_(primitives.env.terminated),
            np.bool_(primitives.env.truncated),
            {},
        )

    monkeypatch.setattr(primitives.env, "step", step)
    with capture.bind():
        primitives.release(max_steps=3)

    actions = capture.state.action_records()
    assert len(actions) == 1
    assert isinstance(actions[0].result.data[terminal], np.bool_)
    assert actions[0].result.data[terminal]


@pytest.mark.parametrize("failure", [False, True])
def test_unrecorded_action_breaks_observation_reuse(
    captured_primitives, monkeypatch, failure
):
    primitives, capture = captured_primitives
    monkeypatch.setattr(
        primitives.env,
        "step",
        Mock(
            side_effect=[
                (_obs(1), 0, False, False, {}),
                RuntimeError("manual step")
                if failure
                else (_obs(2), 0, False, False, {}),
                (_obs(3), 0, False, False, {}),
                (_obs(4), 0, False, False, {}),
            ]
        ),
    )
    with capture.bind():
        primitives.release(max_steps=1)
    if failure:
        with pytest.raises(RuntimeError, match="manual step"):
            primitives.release(max_steps=1)
    else:
        primitives.release(max_steps=1)
    call = capture.state.get_tool_call(capture.tool_call_id)
    next_id = capture.state.register_tool_call(
        planner_id=call.planner_id,
        name="release",
        arguments=Payload(format="json", data={"max_steps": 2}),
    )
    with ToolCallCapture(capture.state, next_id).bind():
        primitives.release(max_steps=2)

    first, second, third = capture.state.action_records()
    assert first.after_observation_id != second.before_observation_id
    assert second.after_observation_id == third.before_observation_id
    next_call = capture.state.get_tool_call(next_id)
    assert next_call.before_observation_id == second.before_observation_id
    assert next_call.after_observation_id == third.after_observation_id
    if failure:
        assert second.before_observation_id is None
        assert second.missing["before_observation_id"].reason == "capture_failed"
        assert next_call.missing["before_observation_id"].reason == "capture_failed"
    else:
        observation = capture.state.get_observation(second.before_observation_id)
        np.testing.assert_array_equal(
            observation.payload.data["states"], _obs(2)["states"]
        )


@pytest.mark.parametrize("vla", [False, True])
@pytest.mark.parametrize("failed_write", [1, 3])
def test_attachment_failure_preserves_confirmed_actions(
    captured_primitives, monkeypatch, vla, failed_write
):
    primitives, capture = captured_primitives
    save = capture.state.save_trajectory_artifact
    writes = 0

    def save_with_failure(value, **kwargs):
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("image write failed")
        return save(value, **kwargs)

    monkeypatch.setattr(capture.state, "save_trajectory_artifact", save_with_failure)
    with capture.bind():
        if vla:
            primitives._vlm_chunk("pick up the bowl")
        else:
            primitives.release(max_steps=2)

    actions = capture.state.action_records()
    assert len(actions) == 2
    assert actions[0].after_observation_id == actions[1].before_observation_id
    assert writes == 6
    assert isinstance(primitives._last_obs["main_images"], np.ndarray)
    capture.state.end_task(status="completed", stop_reason="finish")
    root = capture.state.save_planner_trajectory()
    saved = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    assert len(saved) == 2
    side = "before" if failed_write == 1 else "after"
    observation = saved[0][f"{side}_observation"]["payload"]
    assert observation["completeness"] == "partial"
    assert "main_images" not in observation["data"]
    assert observation["missing"]["/main_images"] == {
        "reason": "capture_failed",
        "detail": "OSError: image write failed",
    }
    assert (root / observation["data"]["wrist_images"]["path"]).is_file()
    assert not list((root / "artifacts").glob(".*.tmp.*"))
    if failed_write == 3:
        assert saved[0]["after_observation"] == saved[1]["before_observation"]
    if vla:
        prediction = json.loads((root / "predictions.jsonl").read_text())
        assert prediction["input_observation"] == saved[0]["before_observation"]
        assert prediction["action_ids"] == [row["action_id"] for row in saved]


@pytest.mark.parametrize("frame_failure", [False, True])
def test_vla_capture_keeps_all_confirmed_actions(
    captured_primitives, monkeypatch, frame_failure
):
    primitives, capture = captured_primitives
    initial = primitives._last_obs
    predict = primitives.model.predict

    def predict_separately(observation, *, options):
        assert primitives._observation_id is not None
        captured = capture.state.get_observation(primitives._observation_id)
        np.testing.assert_array_equal(
            captured.payload.data["states"], initial["states"]
        )
        assert observation is not initial
        assert initial["task_descriptions"] == "put the bowl on the plate"
        assert "extra_view_images" not in initial
        return predict(observation, options=options)

    monkeypatch.setattr(primitives.model, "predict", predict_separately)
    if frame_failure:
        primitives.start_recording()
        monkeypatch.setattr(
            primitives, "record_frame", Mock(side_effect=RuntimeError("frame"))
        )
    with capture.bind():
        if frame_failure:
            with pytest.raises(RuntimeError, match="frame"):
                primitives._vlm_chunk("pick up the bowl")
        else:
            primitives._vlm_chunk("pick up the bowl")

    assert primitives.env.chunk_return_all_frames is True
    actions = capture.state.action_records()
    assert [action.source for action in actions] == ["vla", "vla"]
    assert actions[0].after_observation_id == actions[1].before_observation_id
    assert not actions[0].result.data["terminated"]
    assert actions[1].result.data["terminated"]
    assert primitives._last_obs["task_descriptions"] == initial["task_descriptions"]
    call = capture.state.get_tool_call(capture.tool_call_id)
    assert call.before_observation_id == actions[0].before_observation_id
    assert call.after_observation_id == actions[-1].after_observation_id
    observation = capture.state.get_observation(call.after_observation_id)
    np.testing.assert_array_equal(observation.payload.data["states"], _obs(3)["states"])
    (prediction,) = capture.state.prediction_records()
    assert prediction.input_observation_id == actions[0].before_observation_id
    assert prediction.action_ids == [action.action_id for action in actions]
    np.testing.assert_array_equal(prediction.actions.data, np.ones((2, 7), np.float32))


@pytest.mark.parametrize(
    "terminal,step_count", [("terminated", 5), ("truncated", 5), ("terminated", 3)]
)
def test_predictions_keep_each_calls_context_and_confirmed_actions(
    captured_primitives, monkeypatch, terminal, step_count
):
    primitives, capture = captured_primitives
    predicted = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)
    chunks = (predicted, -predicted)
    instructions = ("move toward the bowl", "lift the bowl")
    flags = {name: np.zeros(step_count, bool) for name in ("terminated", "truncated")}
    flags[terminal][2] = True
    predict = Mock(side_effect=chunks)
    monkeypatch.setattr(primitives.model, "predict", predict)
    monkeypatch.setattr(
        primitives.env,
        "chunk_step",
        Mock(
            side_effect=[
                (
                    [
                        _obs(call * step_count + index + 1)
                        for index in range(step_count)
                    ],
                    np.zeros(step_count),
                    flags["terminated"] if call else np.zeros(step_count, bool),
                    flags["truncated"] if call else np.zeros(step_count, bool),
                    [{} for _ in range(step_count)],
                )
                for call in range(2)
            ]
        ),
    )
    with capture.bind():
        for instruction in instructions:
            primitives._vlm_chunk(instruction)

    predictions = capture.state.prediction_records()
    actions = capture.state.action_records()
    assert len(predictions) == 2 and len(actions) == 2 * step_count
    for index, prediction in enumerate(predictions):
        group = actions[index * step_count : (index + 1) * step_count]
        request = predict.call_args_list[index].args[0]
        assert (
            prediction.instruction
            == request["task_descriptions"]
            == instructions[index]
        )
        assert prediction.input_observation_id == group[0].before_observation_id
        observation = capture.state.get_observation(prediction.input_observation_id)
        np.testing.assert_array_equal(
            observation.payload.data["states"], request["states"]
        )
        assert prediction.action_ids == [action.action_id for action in group]
        np.testing.assert_array_equal(
            prediction.actions.data, chunks[index][:step_count]
        )


@pytest.mark.parametrize("failure", ["predict", "cancel", "chunk"])
@pytest.mark.parametrize("confirmed_prefix", [False, True])
def test_unexecuted_vla_actions_are_not_recorded(
    captured_primitives, monkeypatch, failure, confirmed_prefix
):
    primitives, capture = captured_primitives
    if confirmed_prefix:
        with capture.bind():
            primitives._vlm_chunk("pick up the bowl")
    confirmed = capture.state.action_records()
    predictions = capture.state.prediction_records()
    initial = primitives._last_obs
    execute_chunk = primitives.env.chunk_step
    chunk_step = Mock(side_effect=RuntimeError("chunk"))
    monkeypatch.setattr(primitives.env, "chunk_step", chunk_step)
    if failure == "predict":
        monkeypatch.setattr(
            primitives.model, "predict", Mock(side_effect=RuntimeError("predict"))
        )
    elif failure == "cancel":
        monkeypatch.setattr(
            primitives,
            "_check_cancelled",
            Mock(side_effect=[None, ToolCancelled("cancel")]),
        )
    with capture.bind(), pytest.raises((RuntimeError, ToolCancelled), match=failure):
        primitives._vlm_chunk("pick up the bowl")

    assert [a.action_id for a in capture.state.action_records()] == [
        a.action_id for a in confirmed
    ]
    assert [p.prediction_id for p in capture.state.prediction_records()] == [
        p.prediction_id for p in predictions
    ]
    assert primitives._last_obs is initial
    assert initial["task_descriptions"] == "put the bowl on the plate"
    assert "extra_view_images" not in initial
    assert chunk_step.call_count == (failure == "chunk")
    call = capture.state.get_tool_call(capture.tool_call_id)
    if failure == "chunk":
        assert call.after_observation_id is None
        assert call.missing["after_observation_id"].reason == "capture_failed"
        assert "unconfirmed" in call.missing["after_observation_id"].detail
        next_id = capture.state.register_tool_call(
            planner_id=call.planner_id,
            name="pi0_doubled",
            arguments=Payload("json", {"prompt": "pick up the bowl"}),
        )
        monkeypatch.setattr(primitives.env, "chunk_step", execute_chunk)
        with ToolCallCapture(capture.state, next_id).bind():
            primitives._vlm_chunk("pick up the bowl")
        first, second = capture.state.action_records()[-2:]
        assert first.before_observation_id is None
        assert first.missing["before_observation_id"].reason == "capture_failed"
        assert second.before_observation_id == first.after_observation_id
        assert capture.state.get_tool_call(next_id).before_observation_id is None
        prediction = capture.state.prediction_records()[-1]
        assert prediction.input_observation_id is None
        assert prediction.missing["input_observation_id"].reason == "capture_failed"
    elif confirmed_prefix:
        assert call.after_observation_id == confirmed[-1].after_observation_id
        assert "after_observation_id" not in call.missing


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


@pytest.mark.parametrize("capture_actions", [False, True])
def test_no_capture_context_keeps_fast_chunk_path(capture_actions):
    env = _Env()
    primitives = _primitives(env, capture_actions=capture_actions)
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
