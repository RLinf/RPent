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

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import ArtifactRef, EnvState, MissingInfo, Payload
from rpent.session.env_capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.direct_action import direct_action_tool


class _Client:
    def __init__(self):
        self.position = 0
        self.reset_count = 0
        self.calls = []
        self.chunk_limit = None
        self.fail = False
        self.last_obs = self._observation()

    def _observation(self):
        return {
            "states": np.array([self.position], dtype=np.float32),
            "joint/history~": np.array([[self.position]], dtype=np.float32),
            "cameras": {
                "front": [np.full((2, 3, 3), self.position, dtype=np.uint8)],
                "wrist": np.full((2, 3, 3), self.position + 1, dtype=np.uint8),
            },
        }

    def reset(self):
        self.reset_count += 1
        self.position = 0
        self.last_obs = self._observation()
        return self.last_obs

    def get_task_language(self):
        return "Move the arm."

    def step(self, action, **kwargs):
        return self._run("step", [action], **kwargs)

    def chunk_step(self, actions, **kwargs):
        return self._run("chunk_step", actions[: self.chunk_limit], **kwargs)

    def _run(self, method, actions, **kwargs):
        self.calls.append((method, kwargs))
        before = self.last_obs
        steps = []
        observations = []
        for action in actions:
            native = np.asarray(action, dtype=np.float32)
            self.position += float(native[0])
            self.last_obs = self._observation()
            if self.fail:
                raise RuntimeError("step failed after moving")
            observations.append(self.last_obs)
            action_type = kwargs.get("action_type", "joint")
            steps.append(
                {
                    "action": {
                        "format": f"test.action.{action_type}",
                        "data": {"target": native} if action_type == "ee" else native,
                        "completeness": "complete",
                        "inline_array_paths": ["/target"]
                        if action_type == "ee"
                        else [""],
                    },
                    "after_observation": self._observation_payload(self.last_obs),
                    "result": {
                        "format": "test.result",
                        "data": {"reward": self.position, "info": {"success": False}},
                        "completeness": "complete",
                    },
                }
            )
        self.last_result = (
            observations if kwargs.get("return_all_frames") else self.last_obs,
            self.position,
            False,
            False,
            {"executed_actions": len(steps)},
        )
        self.last_response = {
            "result": self.last_result,
            "before_observation": self._observation_payload(before),
            "steps": steps,
        }
        return self.last_response if kwargs.get("capture") else self.last_result

    @staticmethod
    def _observation_payload(obs):
        return {
            "format": "test.observation",
            "format_version": "1",
            "data": obs,
            "completeness": "partial",
            "missing": {
                "/depth": {"reason": "not_exposed", "detail": "No depth sensor."}
            },
            "inline_array_paths": ["/states", "/joint~1history~0"],
        }


@pytest.fixture
def capture(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="Move the arm.", config=Payload("test.config", {}))
    planner = state.register_planner(backend="api")
    tool = state.register_tool_call(
        planner_id=planner,
        name="move",
        arguments=Payload("json", {}),
        missing={
            name: MissingInfo("not_recorded", "No observation yet.")
            for name in ("before_observation_id", "after_observation_id")
        },
    )
    return ToolCallCapture(state, tool)


@pytest.fixture
def env(capture):
    return CapturingEnvClient(_Client(), capture.state)


@pytest.fixture
def toolkit(capture, tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=capture.state,
        memory=MemoryManager(tmp_path / "memory"),
    )
    monkeypatch.setattr(
        toolkit, "get_env_state", lambda **kwargs: ToolResult(data={"ok": True})
    )
    return toolkit


def test_records_executed_steps_and_reuses_observations(env, capture):
    assert env.wrapped.reset_count == 0
    assert env.calls == []
    assert env.get_task_language() == "Move the arm."
    env.wrapped.chunk_limit = 2
    with capture.bind():
        capture.start()
        result = env.chunk_step([[1], [2], [9]], return_all_frames=True)
        assert result is env.wrapped.last_result
        assert env.last_obs is result[0][-1]
        assert env.step([4], action_type="ee") is env.wrapped.last_result

    assert env.calls == [
        ("chunk_step", {"capture": True, "return_all_frames": True}),
        ("step", {"capture": True, "action_type": "ee"}),
    ]
    actions = capture.state.action_records()
    assert len(actions) == 3
    assert [item.action.format for item in actions] == [
        "test.action.joint",
        "test.action.joint",
        "test.action.ee",
    ]
    for index, (action, value) in enumerate(zip(actions, [1, 2, 4], strict=True)):
        data = action.action.data["target"] if index == 2 else action.action.data
        np.testing.assert_array_equal(data, [value])
        assert action.source == "tool_computed"
        assert action.tool_call_id == capture.tool_call_id
        before = capture.state.get_observation(action.before_observation_id)
        after = capture.state.get_observation(action.after_observation_id)
        np.testing.assert_array_equal(
            before.payload.data["states"], [[0], [1], [3]][index]
        )
        np.testing.assert_array_equal(
            after.payload.data["states"], [[1], [3], [7]][index]
        )
        if index:
            assert (
                action.before_observation_id == actions[index - 1].after_observation_id
            )
    tool = capture.state.get_tool_call(capture.tool_call_id)
    assert tool.before_observation_id == actions[0].before_observation_id
    assert tool.after_observation_id == actions[-1].after_observation_id
    assert tool.missing == {}


@pytest.mark.parametrize(
    ("method", "values", "kwargs", "expected_actions"),
    [
        ("step", [2], {"action_type": "ee"}, [[2]]),
        ("chunk_step", [[1], [2]], {"return_all_frames": True}, [[1], [2]]),
    ],
)
def test_custom_capture_records_actions_and_predictions(
    capture, method, values, kwargs, expected_actions
):
    client = _Client()
    methods = []

    def capture_call(method, *args, **kwargs):
        methods.append(method)
        getattr(client, method)(*args, **kwargs)
        return client.last_response

    env = CapturingEnvClient(client, capture.state, capture_call=capture_call)
    with capture.bind(), env.prediction("Move."):
        result = getattr(env, method)(values, **kwargs)

    assert methods == [method]
    assert client.calls == [(method, kwargs)]
    assert result is client.last_result
    actions = capture.state.action_records()
    np.testing.assert_array_equal(
        [
            action.action.data["target"] if method == "step" else action.action.data
            for action in actions
        ],
        expected_actions,
    )
    assert all(action.source == "vla" for action in actions)
    (prediction,) = capture.state.prediction_records()
    assert prediction.action_ids == [action.action_id for action in actions]


def test_custom_capture_failure_is_reported_without_retry(capture):
    client = _Client()
    capture_call = Mock(side_effect=RuntimeError("capture failed"))
    env = CapturingEnvClient(client, capture.state, capture_call=capture_call)
    with capture.bind(), pytest.raises(RuntimeError, match="capture failed"):
        env.step([1])

    capture_call.assert_called_once_with("step", [1])
    assert client.calls == []
    assert capture.state.action_records() == []
    call = capture.state.get_tool_call(capture.tool_call_id)
    assert call.missing["after_observation_id"].reason == "capture_failed"


def test_observation_reuse_across_tool_calls(env, capture):
    with capture.bind():
        env.step([1])
    other_tool = capture.state.register_tool_call(
        planner_id=capture.state.get_tool_call(capture.tool_call_id).planner_id,
        name="move_again",
        arguments=Payload("json", {}),
    )
    with ToolCallCapture(capture.state, other_tool).bind():
        env.step([2])
    first, second = capture.state.action_records()
    assert first.after_observation_id == second.before_observation_id
    assert second.tool_call_id == other_tool
    assert (
        capture.state.get_tool_call(other_tool).before_observation_id
        == first.after_observation_id
    )


def test_payload_snapshots_and_nested_attachments(env, capture, tmp_path):
    with capture.bind():
        result = env.step([1])
    raw = env.wrapped.last_response
    assert isinstance(result[0]["cameras"]["front"][0], np.ndarray)
    result[0]["states"][:] = 99
    result[0]["joint/history~"][:] = 99
    result[0]["cameras"]["front"][0][:] = 99
    raw["steps"][0]["action"]["data"][:] = 99
    raw["steps"][0]["result"]["data"]["info"]["success"] = True

    action = capture.state.action_records()[0]
    np.testing.assert_array_equal(action.action.data, [1])
    assert action.result.data["info"]["success"] is False
    payload = capture.state.get_observation(action.after_observation_id).payload
    assert payload.format == "test.observation" and payload.format_version == "1"
    assert payload.completeness == "partial"
    assert payload.missing["/depth"] == MissingInfo("not_exposed", "No depth sensor.")
    np.testing.assert_array_equal(payload.data["states"], [1])
    np.testing.assert_array_equal(payload.data["joint/history~"], [[1]])
    image = payload.data["cameras"]["front"][0]
    assert isinstance(image, ArtifactRef)
    root = tmp_path / "trajectories" / capture.state.get_task().task_id
    np.testing.assert_array_equal(
        np.load(root / image.path), np.ones((2, 3, 3), dtype=np.uint8)
    )
    assert len(list((root / "artifacts").iterdir())) == 4
    capture.state.end_task(status="completed", stop_reason="finish")
    capture.state.save_planner_trajectory()
    assert len(list((root / "artifacts").iterdir())) == 4


@pytest.mark.parametrize(
    "interruption", ["reset", "failure", "uncaptured_step", "uncaptured_failure"]
)
def test_interruptions_break_observation_reuse(env, capture, interruption):
    with capture.bind():
        env.step([1])
    if interruption == "reset":
        env.reset()
        expected_before = 0
    elif interruption == "failure":
        env.wrapped.fail = True
        with capture.bind(), pytest.raises(RuntimeError, match="step failed"):
            env.chunk_step([[2], [3]])
        assert len(capture.state.action_records()) == 1
        tool = capture.state.get_tool_call(capture.tool_call_id)
        assert tool.after_observation_id is None
        assert tool.missing["after_observation_id"].reason == "capture_failed"
        env.wrapped.fail = False
        expected_before = 3
    else:
        if interruption == "uncaptured_failure":
            env.wrapped.fail = True
            with pytest.raises(RuntimeError, match="step failed"):
                env.step([2])
            env.wrapped.fail = False
        else:
            env.step([2])
        assert env.calls[-1] == ("step", {})
        expected_before = 3
    with capture.bind():
        env.step([4])
    first, second = capture.state.action_records()
    assert first.after_observation_id != second.before_observation_id
    before = capture.state.get_observation(second.before_observation_id)
    np.testing.assert_array_equal(before.payload.data["states"], [expected_before])
    tool = capture.state.get_tool_call(capture.tool_call_id)
    assert tool.before_observation_id == first.before_observation_id
    assert tool.after_observation_id == second.after_observation_id
    assert tool.missing == {}


@pytest.mark.parametrize("custom_capture", [False, True])
def test_no_matching_capture_passes_through_without_a_task(
    capture, tmp_path, custom_capture
):
    state = EnvState(tmp_path / "other")
    capture_call = Mock()
    env = CapturingEnvClient(
        _Client(), state, capture_call=capture_call if custom_capture else None
    )
    with env.prediction("Move."):
        assert env.step([1]) is env.wrapped.last_result
    with capture.bind():
        with env.prediction("Move again."):
            assert (
                env.chunk_step([[2]], return_all_frames=True) is env.wrapped.last_result
            )
    assert env.calls == [("step", {}), ("chunk_step", {"return_all_frames": True})]
    capture_call.assert_not_called()
    assert state.action_records() == []
    assert capture.state.action_records() == []
    assert state.prediction_records() == capture.state.prediction_records() == []
    assert not (tmp_path / "other" / "trajectories").exists()


def test_prediction_without_actions_creates_no_observations_or_attachments(
    env, capture, tmp_path
):
    env.wrapped.chunk_limit = 0
    with capture.bind(), env.prediction("Move.") as prediction_id:
        assert env.calls == []
        assert not (tmp_path / "trajectories").exists()
        assert env.chunk_step([[1]]) is env.wrapped.last_result
    (prediction,) = capture.state.prediction_records()
    assert prediction.prediction_id == prediction_id
    assert prediction.instruction == "Move."
    assert prediction.action_ids == []
    assert capture.prediction_id is None
    assert capture.state.action_records() == []
    assert (
        capture.state.get_tool_call(capture.tool_call_id).before_observation_id is None
    )
    assert not (tmp_path / "trajectories").exists()


@pytest.mark.parametrize("failed_observation", [0, 1], ids=["before", "after"])
def test_attachment_failure_preserves_executed_actions(
    env, capture, monkeypatch, failed_observation, tmp_path
):
    save = capture.state.save_trajectory_artifact
    image_count = 0

    def save_image(value):
        nonlocal image_count
        if value.ndim == 3:
            image_count += 1
            if image_count == 2 * failed_observation + 1:
                raise OSError("disk full")
        return save(value)

    monkeypatch.setattr(capture.state, "save_trajectory_artifact", save_image)
    with capture.bind(), env.prediction("Move."):
        env.step([1])
        env.step([2])
    first, second = capture.state.action_records()
    np.testing.assert_array_equal([first.action.data, second.action.data], [[1], [2]])
    assert first.after_observation_id == second.before_observation_id
    prediction = capture.state.prediction_records()[0]
    assert prediction.action_ids == [first.action_id, second.action_id]
    observations = [
        capture.state.get_observation(observation_id).payload
        for observation_id in (
            first.before_observation_id,
            first.after_observation_id,
            second.after_observation_id,
        )
    ]
    payload = observations[failed_observation]
    assert payload.data["cameras"]["front"] == [None]
    assert payload.completeness == "partial"
    assert payload.missing["/cameras/front/0"] == MissingInfo(
        "capture_failed", "OSError: disk full"
    )
    root = tmp_path / "trajectories" / capture.state.get_task().task_id
    for index, observation in enumerate(observations):
        assert (root / observation.data["cameras"]["wrist"].path).is_file()
        if index != failed_observation:
            image = observation.data["cameras"]["front"][0]
            assert (root / image.path).is_file()
    assert image_count == 6


@pytest.mark.parametrize("source", ["tool_computed", "direct"])
def test_tool_dispatch_sets_source_and_restores_it_after_vla(
    env, capture, toolkit, source
):
    def move(values: list[float]) -> ToolResult:
        """Move the arm."""
        env.step(values)
        with env.prediction("Adjust the arm."):
            env.step([2])
        env.step(values)
        return ToolResult(data={"ok": True})

    if source == "direct":
        declaration = direct_action_tool(
            {"default": {"low": [-10.0], "high": [10.0]}}, move
        )
        declaration = replace(declaration.with_handler(move), name="move")
    else:
        declaration = tool(move)
    toolkit.add_tool(declaration)
    published = next(item for item in toolkit.list_tools() if item.name == "move")
    assert set(published.input_schema["properties"]) == {"values"}

    with capture.bind():
        result = toolkit.execute_tool("move", {"values": [1.0]})
    assert not result.is_error
    actions = capture.state.action_records()
    assert [action.source for action in actions] == [source, "vla", source]
    (prediction,) = capture.state.prediction_records()
    assert prediction.action_ids == [actions[1].action_id]
    assert capture.state.get_tool_call(capture.tool_call_id).status == "completed"


def test_predictions_group_executed_actions_without_extra_observations(
    env, capture, tmp_path
):
    with capture.bind():
        with env.prediction("Pick up the cup."):
            env.step([1])
            env.step([2])
        env.step([4])
        with env.prediction("Place it down."):
            env.wrapped.chunk_limit = 1
            env.chunk_step([[5], [6]])
        env.step([7])

    actions = capture.state.action_records()
    first, second = capture.state.prediction_records()
    assert [action.source for action in actions] == [
        "vla",
        "vla",
        "tool_computed",
        "vla",
        "tool_computed",
    ]
    assert first.action_ids == [actions[0].action_id, actions[1].action_id]
    assert second.action_ids == [actions[3].action_id]
    assert first.instruction == "Pick up the cup."
    assert second.instruction == "Place it down."
    for prediction in (first, second):
        assert prediction.tool_call_id == capture.tool_call_id
        assert (
            prediction.planner_id
            == capture.state.get_tool_call(capture.tool_call_id).planner_id
        )
    np.testing.assert_array_equal(actions[3].action.data, [5])
    assert [
        (action.before_observation_id, action.after_observation_id)
        for action in actions
    ] == [(f"O{index + 1}", f"O{index + 2}") for index in range(len(actions))]
    before = capture.state.get_observation(actions[0].before_observation_id).payload
    np.testing.assert_array_equal(before.data["states"], [0])
    root = tmp_path / "trajectories" / capture.state.get_task().task_id
    assert len(list((root / "artifacts").iterdir())) == 2 * (len(actions) + 1)


@pytest.mark.parametrize("failure", ["predict", "execute"])
def test_prediction_failure_keeps_completed_actions_and_restores_source(
    env, capture, failure
):
    with capture.bind():
        with pytest.raises(RuntimeError):
            with env.prediction("Move."):
                if failure == "predict":
                    raise RuntimeError("model prediction failed")
                env.step([1])
                env.wrapped.fail = True
                env.chunk_step([[2], [3]])
        env.wrapped.fail = False
        env.step([4])

    actions = capture.state.action_records()
    (prediction,) = capture.state.prediction_records()
    assert actions[-1].source == "tool_computed"
    assert prediction.action_ids == [action.action_id for action in actions[:-1]]
    assert len(prediction.action_ids) == (1 if failure == "execute" else 0)
    if failure == "execute":
        assert actions[0].source == "vla"
        assert actions[1].before_observation_id != actions[0].after_observation_id
