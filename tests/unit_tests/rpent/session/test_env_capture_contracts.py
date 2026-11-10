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

from unittest.mock import Mock

import pytest

from rpent.session import EnvState, Payload
from rpent.session.env_capture import CapturingEnvClient, ToolCallCapture


@pytest.fixture
def capture(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="Move.", config=Payload("json", {}))
    tool = state.register_tool_call(
        planner_id=None, name="move", arguments=Payload("json", {})
    )
    return ToolCallCapture(state, tool)


def captured_step(before, after, *, cached=False):
    return {
        "result": after,
        "before_observation": {"format": "test.position", "data": before},
        "before_observation_cached": cached,
        "steps": [
            {
                "action": {"format": "test.action", "data": [1]},
                "after_observation": {"format": "test.position", "data": after},
                "result": {"format": "test.result", "data": {"done": False}},
            }
        ],
    }


@pytest.mark.parametrize("cached", [False, True])
def test_failed_step_breaks_continuity_until_a_new_observation(capture, cached):
    client = Mock()
    client.step.side_effect = [
        captured_step(0, 1),
        RuntimeError("response lost"),
        captured_step(1 if cached else 2, 3, cached=cached),
        captured_step(3, 4),
    ]
    env = CapturingEnvClient(client, capture.state)
    with capture.bind():
        env.step([1])
        with (
            env.prediction("Move."),
            pytest.raises(RuntimeError, match="response lost"),
        ):
            env.step([1])
        failed = capture.state.get_tool_call(capture.tool_call_id)
        assert failed.after_observation_id is None
        assert failed.missing["after_observation_id"].reason == "capture_failed"
        env.step([1])
        env.step([1])

    first, recovered, last = capture.state.action_records()
    assert capture.state.prediction_records()[0].action_ids == []
    assert recovered.source == "tool_computed"
    assert recovered.before_observation_id != first.after_observation_id
    before = capture.state.get_observation(recovered.before_observation_id).payload
    if cached:
        assert before.data is None and before.missing[""].reason == "capture_failed"
    else:
        assert before.data == 2
    assert last.before_observation_id == recovered.after_observation_id


@pytest.mark.parametrize("interruption", ["reset", "uncaptured_step"])
def test_reset_or_unrecorded_motion_starts_a_new_observation_chain(
    capture, interruption
):
    client = Mock()
    client.step.side_effect = [captured_step(0, 1), captured_step(5, 6)]
    env = CapturingEnvClient(client, capture.state)
    with capture.bind():
        env.step([1])
    if interruption == "reset":
        env.reset()
    else:
        client.chunk_step.return_value = 5
        env.chunk_step([[4]])
    with capture.bind():
        env.step([1])
    first, second = capture.state.action_records()
    assert first.after_observation_id != second.before_observation_id
    assert capture.state.get_observation(second.before_observation_id).payload.data == 5


def test_a_failed_camera_write_keeps_the_executed_action(capture, monkeypatch):
    client = Mock()
    client.step.return_value = captured_step(0, {"position": 1, "camera": b"image"})
    env = CapturingEnvClient(client, capture.state)
    monkeypatch.setattr(
        capture.state,
        "save_trajectory_artifact",
        Mock(side_effect=OSError("disk full")),
    )
    with capture.bind(), env.prediction("Move."):
        env.step([1])
    client.step.assert_called_once_with([1], capture=True)
    (action,) = capture.state.action_records()
    assert capture.state.prediction_records()[0].action_ids == [action.action_id]
    obs = capture.state.get_observation(action.after_observation_id).payload
    assert obs.data == {"position": 1, "camera": None}
    assert (
        obs.completeness == "partial"
        and obs.missing["/camera"].reason == "capture_failed"
    )
