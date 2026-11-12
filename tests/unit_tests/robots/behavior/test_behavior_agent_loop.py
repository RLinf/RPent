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

"""Offline Planner loop: initial views, one VLA chunk, fresh views and feedback."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic_ai import BinaryContent, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.toolkit import BehaviorToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.tools.toolkit import ToolCancelled
from tests.unit_tests.robots.behavior.conftest import (
    _execution_feedback,
    _template_toolkit,
)

_CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


_TASK = "Put the popcorn bag in the microwave and cook the popcorn."


class _Environment:
    arm_joint_limits = {arm: np.tile([-3.0, 3.0], (7, 1)) for arm in ("left", "right")}

    def __init__(self) -> None:
        self.action_chunks: list[np.ndarray] = []
        self.actions_executed = 0

    def get_observation(self) -> dict[str, Any]:
        observation = {
            key: np.full(
                (1, 6, 8, 3),
                10 * (index + 1) + self.actions_executed,
                dtype=np.uint8,
            )
            for index, key in enumerate(_CAMERAS)
        }
        proprio = np.zeros(61, dtype=np.float32)
        proprio[[23, 48]] = 1
        proprio[17:20] = [0.4 + self.actions_executed / 1000, 0.2, 0.8]
        proprio[42:45] = [0.4, -0.2, 0.8]
        proprio[24:26] = 0.05
        proprio[49:51] = 0.03
        observation["robot_r1::proprio"] = proprio[None]
        observation["task_id"] = np.array([40])
        return observation

    def get_session_status(self) -> dict[str, Any]:
        return {"ended": False, "reason": None, "outcome": "unknown", "generation": 0}

    def chunk_step(self, actions: np.ndarray) -> tuple:
        self.action_chunks.append(actions.copy())
        count = len(actions)
        self.actions_executed += count
        return (
            self.get_observation(),
            None,
            None,
            None,
            _execution_feedback(count),
        )


class _VLA:
    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []
        self.prediction = np.arange(64 * 23, dtype=np.float32).reshape(64, 23)

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        self.inputs.append(observation)
        return self.prediction.copy()


def _model_observation(part: ToolReturnPart) -> tuple[dict[str, Any], list[int]]:
    """Read the JSON and three PNG images actually delivered through the SDK."""
    assert isinstance(part.content, list)
    text = [item for item in part.content if isinstance(item, str)]
    images = [item for item in part.content if isinstance(item, BinaryContent)]
    assert len(text) == 1 and len(images) == 3
    pixels = []
    for image in images:
        assert image.media_type == "image/png"
        with Image.open(BytesIO(image.data)) as decoded:
            pixels.append(decoded.getpixel((0, 0))[0])
    return json.loads(text[0]), pixels


def test_api_planner_observes_then_executes_one_chunk_and_sees_new_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    env, vla = _Environment(), _VLA()
    events = NullDashboardEventSink()
    toolkit = BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": vla,
            "task_id": 40,
            "task_language": _TASK,
            "execution_horizon": 32,
            "max_chunks_per_call": 1,
            "instruction_mode": "main",
            "tool_allowlist": ["view_env_state", "vla_act", "finish"],
        },
        dashboard_events=events,
        memory=MemoryManager(tmp_path / "memory", memory_access="read_only"),
        state_output_dir=tmp_path / "state",
    )
    received_views: list[list[int]] = []
    received_feedback: list[dict[str, Any]] = []

    async def respond(messages, info):
        returned = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returned:
            assert not env.action_chunks and not vla.inputs
            yield {0: DeltaToolCall(name="view_env_state", json_args='{"step": 0}')}
            return

        latest = returned[-1]
        observation, pixels = _model_observation(latest)
        received_views.append(pixels)
        assert observation["state"]["task_language"] == _TASK
        robot_state = observation["state"]["robot_state"]
        assert robot_state["available"] is True
        assert robot_state["source_step"] == observation["step"]
        assert robot_state["frame"] == "robot_base"
        assert robot_state["units"]["finger_qpos"] == "metre per finger"
        assert robot_state["left"]["eef_quaternion_xyzw"] == [0, 0, 0, 1]
        if latest.tool_name == "view_env_state":
            assert pixels == [10, 20, 30]
            assert not env.action_chunks and not vla.inputs
            assert observation["step"] == 0
            assert robot_state["left"]["eef_position"] == [0.4, 0.2, 0.8]
            yield {0: DeltaToolCall(name="vla_act", json_args='{"chunks": 1}')}
            return

        assert latest.tool_name == "vla_act"
        assert pixels == [42, 52, 62]
        assert observation["step"] == 1
        assert robot_state["left"]["eef_position"] == [0.432, 0.2, 0.8]
        feedback = observation["log"]["result"]
        received_feedback.append(feedback)
        assert feedback["requested_actions"] == feedback["confirmed_actions"] == 32
        assert feedback["observation_fresh"] is True
        yield {
            0: DeltaToolCall(
                name="finish",
                json_args=json.dumps(
                    {
                        "status": "success",
                        "summary": "Observed the next three views.",
                    }
                ),
            )
        }

    prompts = get_robot_spec().prompts
    variables = {
        "task_id": 40,
        "task_name": "make_microwave_popcorn",
        "task_language": _TASK,
        "execution_horizon": 32,
        "max_chunks_per_call": 1,
        "instruction_mode": "main",
        "tool_allowlist": ["view_env_state", "vla_act", "finish"],
        "memory_dir": str(tmp_path / "memory"),
    }
    planner = ApiAgentLoop(
        model=FunctionModel(stream_function=respond),
        dashboard_events=events,
        timeout_s=30,
    )
    try:
        result = planner.solve(
            system_prompt=prompts.render("system", variables=variables),
            user_message=prompts.render("user", variables=variables),
            toolkit=toolkit,
            max_turns=3,
        )
        assert result.error is None
        assert result.stats["turns_used"] == 3
        assert result.stats["tool_calls"] == 3
        assert received_views == [[10, 20, 30], [42, 52, 62]]
        assert len(received_feedback) == len(vla.inputs) == len(env.action_chunks) == 1
        assert env.actions_executed == 32
        np.testing.assert_array_equal(env.action_chunks[0], vla.prediction[:32])
        assert vla.inputs[0]["task_descriptions"] == _TASK
        assert vla.inputs[0][_CAMERAS[0]].shape == (6, 8, 3)
        assert toolkit.state.latest_step == 1
        assert result.finish_result["status"] == "unverified"
        assert [
            entry["name"] for entry in result.messages if entry["role"] == "tool"
        ] == [
            "view_env_state",
            "vla_act",
            "finish",
        ]
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("set_gripper", {"val": 1}),
        ("release", {}),
        ("rotate_wrist_joint", {"joint": "yaw", "delta_deg": 1}),
    ],
)
def test_invalid_joint_feedback_is_saved_as_unknown_with_previous_observation(
    tmp_path, monkeypatch, tool, arguments
):
    env, model = _Environment(), _VLA()
    execute = env.chunk_step

    def invalid_feedback(actions):
        returned = execute(actions)
        returned[4].update(confirmed_actions=2, executed_actions=2)
        return returned

    monkeypatch.setattr(env, "chunk_step", invalid_feedback)
    with _template_toolkit(
        tmp_path,
        ["view_env_state", tool, "finish"],
        env=env,
        model=model,
        task_language=_TASK,
    ) as toolkit:
        result = toolkit.execute_tool(
            tool, {"arm": "left", "steps": 1, "settle_steps": 0, **arguments}
        )
        assert result.is_error
        feedback = result.data["log"]["result"]
        assert feedback["requested_actions"] == 1
        assert feedback["confirmed_actions"] is None
        assert feedback["code"] == "invalid_execution_feedback"
        assert feedback["observation_fresh"] is False
        stored = toolkit.execute_tool("view_env_state", {"step": 1})
        assert stored.data["log"]["result"] == feedback
        assert stored.data["state"]["robot_state"]["left"]["eef_position"][
            0
        ] == pytest.approx(0.4)
        assert len(env.action_chunks) == 1 and not model.inputs


@pytest.mark.parametrize(
    "failure", ["observation", "prediction", "cancel", "execution", "feedback"]
)
def test_later_vla_failure_preserves_confirmed_prefix_in_saved_result(
    tmp_path, monkeypatch, failure
):
    env, model = _Environment(), _VLA()
    with _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "finish"],
        env=env,
        model=model,
        task_language=_TASK,
        execution_horizon=2,
        max_chunks_per_call=3,
    ) as toolkit:
        if failure == "observation":
            observe = env.get_observation
            reads = 0

            def fail_observation():
                nonlocal reads
                reads += 1
                if reads == 3:  # First prediction, its feedback, then the next input.
                    raise ConnectionError("next observation unavailable")
                return observe()

            monkeypatch.setattr(env, "get_observation", fail_observation)
        elif failure == "prediction":
            predict = model.predict

            def fail_prediction(observation):
                if model.inputs:
                    raise RuntimeError("second prediction failed")
                return predict(observation)

            monkeypatch.setattr(model, "predict", fail_prediction)
        elif failure == "cancel":
            checks = 0

            def cancel_after_first_chunk():
                nonlocal checks
                checks += 1
                if checks == 3:
                    raise ToolCancelled("cancelled before the second chunk")

            monkeypatch.setattr(
                toolkit._primitives, "_check_cancelled", cancel_after_first_chunk
            )
        else:
            execute = env.chunk_step

            def fail_execution(actions):
                if env.action_chunks and failure == "execution":
                    env.action_chunks.append(actions.copy())
                    env.actions_executed += (
                        1  # Lost reply after a possible side effect.
                    )
                    raise TimeoutError("second execution outcome unknown")
                returned = execute(actions)
                if len(env.action_chunks) == 2:
                    returned[4]["issued_actions"] = len(actions) + 1
                return returned

            monkeypatch.setattr(env, "chunk_step", fail_execution)
        result = toolkit.execute_tool("vla_act", {"chunks": 3})
        feedback = result.data["log"]["result"]
        assert result.is_error
        assert feedback["requested_chunks"] == 3
        assert feedback["observation_fresh"] is False
        assert len(env.action_chunks) <= 2  # No replay or third chunk after failure.
        if failure in {"execution", "feedback"}:
            assert feedback["requested_actions"] == 4
            assert feedback["known_execution_before_failure"] == {
                "requested_actions": 2,
                "issued_actions": 2,
                "confirmed_actions": 2,
                "unconfirmed_actions": 0,
                "executed_actions": 2,
            }
            for name in (
                "issued_actions",
                "confirmed_actions",
                "unconfirmed_actions",
                "executed_actions",
            ):
                assert feedback[name] is None
            assert toolkit._primitives._command_history_unknown is True
        else:
            assert len(env.action_chunks) == 1
            assert feedback["requested_actions"] == feedback["confirmed_actions"] == 2
            assert feedback["issued_actions"] == feedback["executed_actions"] == 2
            assert feedback["unconfirmed_actions"] == 0
        if failure == "cancel":
            assert feedback["interrupted"] is True
            assert feedback["code"] == "tool_cancelled"
        stored = toolkit.execute_tool("view_env_state", {"step": 1})
        assert stored.data["log"]["result"] == feedback
        assert stored.data["state"]["robot_state"]["left"]["eef_position"][
            0
        ] == pytest.approx(0.402)
