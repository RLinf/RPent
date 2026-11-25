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

"""Native turn fixtures and hand-authored expected samples for planner adapters."""

import asyncio
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition

from rpent.flywheel.export import export_vlm_samples
from rpent.flywheel.vlm import VLMMessage, VLMSample
from rpent.flywheel.vlm_api import api_turn_to_sample
from rpent.session import EnvState, MissingInfo, Payload, PlannerRecord, ToolCallRecord
from rpent.session.planner_capture import CapturingModel

TASK = "Move above the basket."
STATE = '{"step":0,"state":{"eef_pos":[0.0,0.0,0.5]}}'
COORDINATES = '{"step":0,"camera":"wrist","world_xyz":[0.1,0.2,0.3]}'
AFTER_RESPONSE = '{"step":1,"state":{"eef_pos":[0.1,0.2,0.4]}}'
REASONING = "Use the recorded basket coordinates."
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
    "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
)
TOOLS = [
    {
        "name": "back_project",
        "description": "Convert image pixels to world coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {"row": {"type": "integer"}, "col": {"type": "integer"}},
            "required": ["row", "col"],
        },
    },
    {
        "name": "move_to",
        "description": "Move the end effector to world coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {"xyz": {"type": "array", "items": {"type": "number"}}},
            "required": ["xyz"],
        },
    },
]


@pytest.fixture(
    params=[
        "api",
        "api_batch",
        "codex_direct",
        "claude_code",
        "missing_image",
        "encrypted_reasoning",
        "redacted_reasoning",
    ]
)
def vlm_case(tmp_path, request):
    """Build a native planner record and its expected VLM sample."""
    case = request.param
    state = EnvState(tmp_path)
    state.start_task(instruction=TASK, config=Payload("test.config", {}))
    artifact = replace(
        state.save_trajectory_artifact(PNG, name="wrist.bin"), media_type="image/png"
    )
    if case == "missing_image":
        (state.trajectory_dir / artifact.path).unlink()

    previous = {
        "id": "old-query",
        "name": "back_project",
        "arguments": {"row": 10, "col": 20},
    }
    calls = [{"id": "call-1", "name": "move_to", "arguments": {"xyz": [0.1, 0.2, 0.4]}}]
    if case == "api_batch":
        calls = [
            {
                "id": f"call-{i}",
                "name": "back_project",
                "arguments": {"row": row, "col": 20},
            }
            for i, row in enumerate((10, 30), start=1)
        ]
    missing = {}

    if case in {"api", "api_batch"}:
        backend = "api"
        native_request = Payload(
            "pydantic_ai.model_request",
            {
                "messages": [
                    {
                        "kind": "request",
                        "parts": [
                            {
                                "part_kind": "user-prompt",
                                "content": [
                                    TASK,
                                    STATE,
                                    {
                                        "kind": "binary",
                                        "data": artifact,
                                        "media_type": "image/png",
                                    },
                                ],
                            }
                        ],
                    },
                    {
                        "kind": "response",
                        "parts": [
                            {
                                "part_kind": "tool-call",
                                "tool_name": previous["name"],
                                "args": previous["arguments"],
                                "tool_call_id": previous["id"],
                            }
                        ],
                    },
                    {
                        "kind": "request",
                        "parts": [
                            {
                                "part_kind": "tool-return",
                                "tool_name": previous["name"],
                                "content": COORDINATES,
                                "tool_call_id": previous["id"],
                            }
                        ],
                    },
                ],
                "model_request_parameters": {
                    "function_tools": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters_json_schema": t["input_schema"],
                        }
                        for t in TOOLS
                    ]
                },
            },
        )
        native_response = Payload(
            "pydantic_ai.model_response",
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": c["name"],
                        "args": c["arguments"],
                        "tool_call_id": c["id"],
                    }
                    for c in calls
                ],
            },
        )
    elif case in {"claude_code", "redacted_reasoning"}:
        backend = "claude_code"
        native_request = Payload(
            "claude.messages_request",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TASK},
                            {"type": "text", "text": STATE},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": artifact,
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": previous["id"],
                                "name": previous["name"],
                                "input": previous["arguments"],
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": previous["id"],
                                "content": COORDINATES,
                            }
                        ],
                    },
                ],
                "tools": TOOLS,
            },
        )
        thinking = REASONING
        if case == "redacted_reasoning":
            thinking = "<REDACTED>"
            missing["/content/0"] = MissingInfo("not_exposed", "Thinking was redacted.")
        native_response = Payload(
            "claude.messages_response",
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking},
                    *[
                        {
                            "type": "tool_use",
                            "id": c["id"],
                            "name": c["name"],
                            "input": c["arguments"],
                        }
                        for c in calls
                    ],
                ],
            },
            completeness="partial" if missing else "complete",
            missing=missing,
        )
    else:
        backend = "codex"
        native_request = Payload(
            "codex.expanded_request",
            {
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "tools": [
                            {
                                "type": "function",
                                "name": t["name"],
                                "description": t["description"],
                                "parameters": t["input_schema"],
                            }
                            for t in TOOLS
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": TASK},
                            {"type": "input_text", "text": STATE},
                            {"type": "input_image", "image_url": artifact},
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": previous["id"],
                        "name": previous["name"],
                        "arguments": json.dumps(previous["arguments"]),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": previous["id"],
                        "output": COORDINATES,
                    },
                ]
            },
        )
        output = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": REASONING}],
            }
        ]
        if case == "encrypted_reasoning":
            output = [
                {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-reasoning",
                }
            ]
            missing["reasoning"] = MissingInfo(
                "not_exposed", "Only encrypted reasoning is available."
            )
        output.extend(
            {
                "type": "function_call",
                "call_id": c["id"],
                "name": c["name"],
                "arguments": json.dumps(c["arguments"]),
            }
            for c in calls
        )
        native_response = Payload("codex.inference_response", {"output_items": output})

    native_request.completeness = "complete"
    if native_response.completeness != "partial":
        native_response.completeness = "complete"
    source = PlannerRecord(
        planner_id="P2",
        backend=backend,
        status="completed",
        request=native_request,
        response=native_response,
        tool_calls=[
            ToolCallRecord(
                tool_call_id=c["id"],
                planner_id="P2",
                origin="model",
                name=c["name"],
                arguments=Payload("json", c["arguments"]),
                status="completed",
                feedback=Payload(
                    "json", AFTER_RESPONSE if c["name"] == "move_to" else COORDINATES
                ),
                before_observation_id="O0",
                after_observation_id="O1" if c["name"] == "move_to" else "O0",
            )
            for c in calls
        ],
    )
    expected = VLMSample(
        episode_id=state.get_task().task_id,
        turn_id=source.planner_id,
        messages=[
            VLMMessage(
                "user",
                content=[
                    {"type": "text", "text": TASK},
                    {"type": "text", "text": STATE},
                    {"type": "image", "artifact": artifact, "observation_id": "O0"},
                ],
            ),
            VLMMessage("assistant", tool_calls=[previous]),
            VLMMessage(
                "tool",
                content=[{"type": "text", "text": COORDINATES}],
                tool_call_id=previous["id"],
            ),
        ],
        tools=TOOLS,
        target=VLMMessage("assistant", tool_calls=calls),
        metadata={"backend": backend, "missing": missing},
    )
    return {
        "name": case,
        "root": state.trajectory_dir,
        "source": source,
        "expected": expected,
    }


def test_sample_keeps_one_response_and_excludes_its_tool_results(vlm_case):
    """Verify that feedback from target calls stays out of the sample input."""
    source, sample = vlm_case["source"], vlm_case["expected"]
    assert sample.turn_id == source.planner_id
    assert sample.target.role == "assistant"
    assert [(c["name"], c["arguments"]) for c in sample.target.tool_calls] == [
        (c.name, c.arguments.data) for c in source.tool_calls
    ]
    current_ids = {c["id"] for c in sample.target.tool_calls}
    assert current_ids.isdisjoint(m.tool_call_id for m in sample.messages)
    assert AFTER_RESPONSE not in [
        p["text"] for m in sample.messages for p in m.content if p["type"] == "text"
    ]
    if vlm_case["name"] == "api_batch":
        assert [c["id"] for c in sample.target.tool_calls] == ["call-1", "call-2"]


def test_sample_serializes_without_loading_or_embedding_images(vlm_case):
    """Verify that serialized samples reference images by path."""
    sample = vlm_case["expected"]
    row = json.loads(json.dumps(asdict(sample), allow_nan=False))
    assert set(row) == {
        "episode_id",
        "turn_id",
        "messages",
        "tools",
        "target",
        "metadata",
    }
    image = row["messages"][0]["content"][2]
    assert image["artifact"]["path"] == "artifacts/wrist.bin"
    assert image["artifact"]["kind"] == "bytes"
    assert image["observation_id"] == "O0"
    assert (vlm_case["root"] / image["artifact"]["path"]).exists() == (
        vlm_case["name"] != "missing_image"
    )
    result = row["messages"][-1]
    assert result["tool_call_id"] == row["messages"][-2]["tool_calls"][0]["id"]
    assert json.loads(result["content"][0]["text"])["world_xyz"] == [0.1, 0.2, 0.3]


def test_sample_omits_reasoning_from_messages(vlm_case):
    """Check that messages omit reasoning and metadata retains missing-data details."""
    row = asdict(vlm_case["expected"])
    assert all("reasoning" not in m for m in [*row["messages"], row["target"]])
    if vlm_case["name"] in {"encrypted_reasoning", "redacted_reasoning"}:
        assert {v["reason"] for v in row["metadata"]["missing"].values()} == {
            "not_exposed"
        }
        assert "opaque-reasoning" not in json.dumps(row)
        assert "<REDACTED>" not in json.dumps(row)
    elif vlm_case["name"] in {"api", "api_batch"}:
        assert row["metadata"]["missing"] == {}


def test_plain_text_target_needs_no_image_or_tool_call():
    """Check that a plain text answer can serve as the entire target."""
    sample = VLMSample(
        episode_id="task-1",
        turn_id="P1",
        messages=[
            VLMMessage("user", content=[{"type": "text", "text": "Report completion."}])
        ],
        tools=[],
        target=VLMMessage(
            "assistant", content=[{"type": "text", "text": "Task completed."}]
        ),
    )
    row = json.loads(json.dumps(asdict(sample)))
    assert row["target"]["content"] == [{"type": "text", "text": "Task completed."}]
    assert row["target"]["tool_calls"] == []


@pytest.fixture
def api_opening(tmp_path):
    """Record an opening API turn with two calls to the same tool."""
    state = EnvState(tmp_path)
    state.start_task(instruction=TASK, config=Payload("test.config", {}))
    request = ModelRequest(
        parts=[
            SystemPromptPart("Robot instructions."),
            UserPromptPart(STATE),
            UserPromptPart(TASK),
        ]
    )
    response = ModelResponse(
        parts=[
            ThinkingPart(REASONING, signature="opaque"),
            TextPart("Check both points."),
            ToolCallPart("back_project", '{"row":10,"col":20}', "call-1"),
            ToolCallPart("back_project", {"row": 30, "col": 20}, "call-2"),
        ]
    )
    query = TOOLS[0]
    parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name=query["name"],
                parameters_json_schema=query["input_schema"],
                description=query["description"],
            )
        ],
        output_tools=[
            ToolDefinition(
                name="finish",
                parameters_json_schema={"type": "object"},
                description="End the task.",
            )
        ],
    )
    model = CapturingModel(
        FunctionModel(lambda messages, info: response), state, "conversation"
    )
    asyncio.run(model.request([request], None, parameters))
    call = state.tool_call_records()[0]
    state.update_tool_call(
        call.tool_call_id, status="completed", feedback=Payload("json", AFTER_RESPONSE)
    )
    state.end_task(status="completed", stop_reason="request_limit")
    root = state.save_planner_trajectory()
    return json.loads((root / "planners.jsonl").read_text())


def test_convert_api_opening_keeps_one_target_for_both_calls(api_opening):
    """Verify that both calls retain their IDs and order in one target."""
    sample = api_turn_to_sample({"task_id": "episode"}, api_opening)
    assert (sample.episode_id, sample.turn_id) == ("episode", "P1")
    assert sample.messages == [
        VLMMessage(role, content=[{"type": "text", "text": text}])
        for role, text in [
            ("system", "Robot instructions."),
            ("user", STATE),
            ("user", TASK),
        ]
    ]
    assert sample.target == VLMMessage(
        "assistant",
        content=[{"type": "text", "text": "Check both points."}],
        tool_calls=[
            {
                "id": "call-1",
                "name": "back_project",
                "arguments": {"row": 10, "col": 20},
            },
            {
                "id": "call-2",
                "name": "back_project",
                "arguments": {"row": 30, "col": 20},
            },
        ],
    )
    assert sample.tools == [
        TOOLS[0],
        {
            "name": "finish",
            "description": "End the task.",
            "input_schema": {"type": "object"},
        },
    ]


def test_convert_api_text_answer_after_thinking_only_response(api_opening):
    """Skip a thinking-only response as a target and in a later turn's history."""
    reply = api_opening["response"]["data"]
    reply["parts"] = [reply["parts"][0]]
    with pytest.raises(ValueError, match="no text or tool calls"):
        api_turn_to_sample({"task_id": "episode"}, api_opening)
    api_opening["request"]["data"]["messages"].append(reply)
    api_opening["response"]["data"] = {
        **reply,
        "parts": [{"part_kind": "text", "content": "Done."}],
    }
    sample = api_turn_to_sample({"task_id": "episode"}, api_opening)
    assert [m.role for m in sample.messages] == ["system", "user", "user"]
    assert sample.target == VLMMessage(
        "assistant", content=[{"type": "text", "text": "Done."}]
    )


@pytest.mark.parametrize("args", ["{", "[]"])
def test_convert_api_turn_rejects_invalid_tool_arguments(api_opening, args):
    """Check that invalid tool arguments raise an error identifying the call."""
    api_opening["response"]["data"]["parts"][-1]["args"] = args
    with pytest.raises(ValueError, match="call-2"):
        api_turn_to_sample({"task_id": "episode"}, api_opening)


@pytest.mark.parametrize("data", [{"$type": "artifact_ref"}, "aW1hZ2U="])
def test_convert_api_turn_rejects_unsupported_inputs(api_opening, data):
    """Check that invalid image references and failed turns are rejected."""
    api_opening["request"]["data"]["messages"][0]["parts"][-1]["content"] = [
        TASK,
        {"kind": "binary", "media_type": "image/png", "data": data},
    ]
    with pytest.raises(ValueError, match="image byte artifact reference"):
        api_turn_to_sample({"task_id": "episode"}, api_opening)
    api_opening["status"] = "failed"
    with pytest.raises(ValueError, match="completed API planner turn"):
        api_turn_to_sample({"task_id": "episode"}, api_opening)


def test_convert_two_saved_api_turns_with_observations_and_coordinates(tmp_path):
    """Verify that the second turn receives the first turn's images and coordinates."""
    state = EnvState(tmp_path)
    state.start_task(instruction=TASK, config=Payload("test.config", {}))
    calls = [
        ToolCallPart("view_env_state", {}, "view"),
        ToolCallPart("back_project", {"row": 10, "col": 20}, "point"),
        ToolCallPart("query_world_map", {"view": "wrist", "bbox": [0, 0, 3, 4]}, "map"),
        ToolCallPart("read_image", {"name": "wrist.png", "step": 0}, "read"),
    ]
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name=c.tool_name) for c in calls]
    )
    replies = iter(
        [
            ModelResponse(parts=[ThinkingPart(REASONING), *calls]),
            ModelResponse(parts=[TextPart("Ready.")]),
        ]
    )
    model = CapturingModel(
        FunctionModel(lambda messages, info: next(replies)), state, "conversation"
    )
    history = [ModelRequest(parts=[UserPromptPart(TASK)])]
    response = asyncio.run(model.request(history, None, parameters))
    image = BinaryContent(data=PNG, media_type="image/png")
    world_map = {
        "step": 0,
        "view": "wrist",
        "bbox": [0, 0, 3, 4],
        "points": [[0.1, 0.2, 0.3]],
    }
    image_info = {"artifact": "wrist.png", "step": 0}
    feedback = ModelRequest(
        parts=[
            ToolReturnPart("back_project", [COORDINATES], "point"),
            ToolReturnPart("view_env_state", [STATE, image, "Wrist view."], "view"),
            ToolReturnPart("query_world_map", world_map, "map"),
            ToolReturnPart("read_image", image_info, "read"),
            UserPromptPart([image]),
        ]
    )
    asyncio.run(model.request([*history, response, feedback], None, parameters))
    state.end_task(status="completed", stop_reason="model_output")
    root = state.save_planner_trajectory()
    task = json.loads((root / "task.json").read_text())["task"]
    first, second = [
        api_turn_to_sample(task, json.loads(line))
        for line in (root / "planners.jsonl").read_text().splitlines()
    ]

    assert (first.turn_id, second.turn_id) == ("P1", "P2")
    assert first.messages == [
        VLMMessage("user", content=[{"type": "text", "text": TASK}])
    ]
    assert first.target == VLMMessage(
        "assistant",
        tool_calls=[
            {"id": c.tool_call_id, "name": c.tool_name, "arguments": c.args}
            for c in calls
        ],
    )
    assert second.messages[:2] == [*first.messages, first.target]
    point, view, region, read, attachment = second.messages[2:]
    assert [
        (m.role, m.tool_call_id) for m in (point, view, region, read, attachment)
    ] == [
        ("tool", "point"),
        ("tool", "view"),
        ("tool", "map"),
        ("tool", "read"),
        ("user", None),
    ]
    assert json.loads(point.content[0]["text"]) == [COORDINATES]
    assert json.loads(region.content[0]["text"]) == world_map
    assert json.loads(read.content[0]["text"]) == image_info
    before, picture, after = view.content
    assert [before, after] == [
        {"type": "text", "text": STATE},
        {"type": "text", "text": "Wrist view."},
    ]
    assert attachment.content == [picture]
    assert picture["type"] == "image" and picture["observation_id"] is None
    assert picture["artifact"].media_type == "image/png"
    assert (root / picture["artifact"].path).read_bytes() == PNG
    assert second.target == VLMMessage(
        "assistant", content=[{"type": "text", "text": "Ready."}]
    )


def test_convert_api_retry_preserves_call_id_and_sdk_feedback(api_opening):
    """Check that retry and failure messages retain their call IDs and feedback."""
    history = api_opening["request"]["data"]["messages"]
    history.extend(
        [
            api_opening["response"]["data"],
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "retry-prompt",
                        "tool_name": "back_project",
                        "tool_call_id": "call-2",
                        "content": "Pixel outside the image.",
                    },
                    {
                        "part_kind": "tool-return",
                        "tool_name": "back_project",
                        "tool_call_id": "call-1",
                        "content": "Missing camera.",
                        "outcome": "failed",
                    },
                ],
            },
        ]
    )
    api_opening["planner_id"] = "P2"
    api_opening["response"]["data"] = {
        "parts": [{"part_kind": "text", "content": "Checking the camera."}]
    }
    retry, failure = api_turn_to_sample({"task_id": "episode"}, api_opening).messages[
        -2:
    ]
    assert (retry.role, retry.tool_call_id) == ("tool", "call-2")
    assert (
        retry.content[0]["text"]
        == "Pixel outside the image.\n\nFix the errors and try again."
    )
    assert (failure.role, failure.tool_call_id) == ("tool", "call-1")
    assert json.loads(failure.content[0]["text"]) == {"error": "Missing camera."}


def _write_vlm_episode(root, result, planners):
    """Write an episode with task metadata and ordered native planner records."""
    root.mkdir(parents=True)
    task = {
        "task_id": root.name,
        "status": "completed",
        "stop_reason": "finish",
        "result": result,
    }
    (root / "task.json").write_text(json.dumps({"schema_version": 5, "task": task}))
    (root / "planners.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in planners)
    )


def test_vlm_cli_exports_without_lerobot(api_opening, tmp_path):
    """Exercise CLI startup and sample export with the optional VLA dependency absent."""
    root, output = tmp_path / "source", tmp_path / "dataset"
    _write_vlm_episode(root, {"data": {"environment_success": True}}, [api_opening])
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['lerobot'] = None\n"
            "from rpent.flywheel.cli import main\n"
            "sys.exit(main())",
            "--vlm",
            "--input-root",
            str(root),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    sample = json.loads((output / "samples.jsonl").read_text())
    assert sample == asdict(api_turn_to_sample({"task_id": root.name}, api_opening))


def test_vlm_export_selects_confirmed_episodes_and_counts_rejections(
    api_opening, tmp_path
):
    """Check episode selection and the reason reported for each skipped turn."""
    jobs = tmp_path / "jobs"
    success = {"data": {"environment_success": True}}
    legacy = deepcopy(api_opening)
    del legacy["request"]["data"]["model"]
    invalid_args = deepcopy(api_opening)
    invalid_args["response"]["data"]["parts"][-1]["args"] = "["
    truncated = deepcopy(api_opening)
    truncated["response"]["data"]["finish_reason"] = "length"
    missing_image = deepcopy(api_opening)
    missing_image["request"]["data"]["messages"][-1]["parts"].append(
        {
            "part_kind": "user-prompt",
            "content": [
                {
                    "kind": "binary",
                    "media_type": "image/png",
                    "data": {
                        "$type": "artifact_ref",
                        "kind": "bytes",
                        "path": "artifacts/absent.bin",
                    },
                }
            ],
        }
    )
    records = [
        api_opening,
        dict(api_opening, backend="codex"),
        dict(api_opening, backend="claude_code"),
        dict(api_opening, response=None),
        truncated,
        invalid_args,
        missing_image,
        legacy,
    ]
    _write_vlm_episode(
        jobs / "success",
        success,
        [dict(row, planner_id=f"P{i}") for i, row in enumerate(records, 1)],
    )
    for name, result in [
        (
            "failed",
            {
                "data": {
                    "environment_success": False,
                    "finish_result": {"status": "success"},
                }
            },
        ),
        ("unknown", None),
        (
            "missing",
            {
                **success,
                "missing": {"/environment_success": {"reason": "not_recorded"}},
            },
        ),
        ("nonboolean", {"data": {"environment_success": 1}}),
    ]:
        _write_vlm_episode(jobs / name, result, [api_opening])

    out = tmp_path / "dataset"
    report = export_vlm_samples(jobs, out)
    assert report == json.loads((out / "report.json").read_text())
    assert (
        report["episode_count"],
        report["selected_episode_count"],
        report["exported_episode_count"],
    ) == (5, 1, 1)
    assert (report["turn_count"], report["sample_count"], report["image_count"]) == (
        12,
        2,
        0,
    )
    assert report["excluded_episode_counts"] == {"no_confirmed_environment_success": 4}
    assert report["excluded_turn_counts"] == {
        "no_confirmed_environment_success": 4,
        "unsupported_backend": 2,
        "incomplete_response": 2,
        "invalid_tool_arguments": 1,
        "missing_images": 1,
        "invalid_turn": 0,
    }
    assert report["skipped_backend_counts"] == {"codex": 1, "claude_code": 1}
    assert (
        sum(report["excluded_turn_counts"].values()) + report["sample_count"]
        == report["turn_count"]
    )
    rows = list(map(json.loads, (out / "samples.jsonl").read_text().splitlines()))
    assert [(row["episode_id"], row["turn_id"]) for row in rows] == [
        ("success", "P1"),
        ("success", "P8"),
    ]
    assert rows[1]["messages"] == rows[0]["messages"]
    assert rows[1]["tools"] == rows[0]["tools"]
    assert rows[1]["metadata"]["model"] is None
    assert not (out / "train.jsonl").exists() and not (out / "val.jsonl").exists()
    all_episodes = export_vlm_samples(jobs, tmp_path / "all", include_unsuccessful=True)
    assert (all_episodes["selected_episode_count"], all_episodes["sample_count"]) == (
        5,
        6,
    )


def test_vlm_export_deduplicates_image_bytes_across_episodes(api_opening, tmp_path):
    """Check that relocated samples resolve shared images and distinguish different bytes."""
    jobs = tmp_path / "jobs"
    for name in ("first", "second"):
        record = deepcopy(api_opening)
        parts = record["request"]["data"]["messages"][0]["parts"]
        parts.append(
            {
                "part_kind": "user-prompt",
                "content": [
                    {
                        "kind": "binary",
                        "media_type": "image/png",
                        "data": {
                            "$type": "artifact_ref",
                            "kind": "bytes",
                            "path": f"artifacts/{file}.bin",
                        },
                    }
                    for file in (name, "same-name")
                ],
            }
        )
        root = jobs / name
        _write_vlm_episode(root, {"data": {"environment_success": True}}, [record])
        (root / "artifacts").mkdir()
        (root / f"artifacts/{name}.bin").write_bytes(PNG)
        Image.new("L", (1, 1), 10 if name == "first" else 20).save(
            root / "artifacts/same-name.bin", format="PNG"
        )

    dataset = tmp_path / "dataset"
    report = export_vlm_samples(jobs, dataset)
    assert (
        report["exported_episode_count"],
        report["sample_count"],
        report["image_count"],
    ) == (2, 2, 3)
    moved = dataset.rename(tmp_path / "moved")
    jobs.rename(tmp_path / "original-recordings")
    paths = []
    for row in map(json.loads, (moved / "samples.jsonl").read_text().splitlines()):
        paths.append(
            [
                b["artifact"]["path"]
                for m in row["messages"]
                for b in m["content"]
                if b["type"] == "image"
            ]
        )
    assert paths[0][0] == paths[1][0] and paths[0][1] != paths[1][1]
    assert (moved / paths[0][0]).read_bytes() == PNG
    for i, pair in enumerate(paths, 1):
        assert all(Path(path).parent == Path("images") for path in pair)
        with Image.open(moved / pair[1]) as image:
            assert image.getpixel((0, 0)) == 10 * i
    assert len(list((moved / "images").iterdir())) == 3
