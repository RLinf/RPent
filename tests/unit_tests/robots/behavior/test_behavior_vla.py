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

"""VLA service, policy adaptation and the offline agent loop."""

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

_CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


_TASK = "Put the popcorn bag in the microwave and cook the popcorn."


class _Environment:
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
            {
                "executed_actions": count,
                "issued_actions": count,
                "confirmed_actions": count,
                "unconfirmed_actions": 0,
                "observation_after_last_issued_action": True,
                "stop_reason": "chunk_completed",
            },
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
