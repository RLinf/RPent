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


"""Offline wire, attachment, compaction and interrupted-stream contracts."""

import asyncio
import base64
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent, ToolReturn
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import SlidingWindowCompaction

from rpent.planner.qwen import QwenChatModel
from rpent.planner.request_trace import RequestTrace, TraceTransport, traced_compaction

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZ8kAAAAASUVORK5CYII="
)


def test_factory_preserves_recorder_when_disabling_strict_tools(tmp_path):
    from rpent.planner.base import build_api_model

    model = build_api_model(
        "openai-chat:qwen-test",
        "http://localhost:39999",
        strict_tool_definitions=False,
        request_trace_dir=tmp_path,
    )
    assert model.request_trace is not None
    assert model.request_trace.directory == tmp_path


def test_post_mapping_multimodal_requests_and_compaction(tmp_path):
    async def run():
        trace = RequestTrace(tmp_path)
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            number = len(calls)
            message = {
                "role": "assistant",
                "content": "done",
                "reasoning_content": "think",
            }
            if number < 7:
                message.update(
                    content="",
                    tool_calls=[
                        {
                            "id": f"call-{number}",
                            "type": "function",
                            "function": {"name": "inspect_image", "arguments": "{}"},
                        }
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "id": f"cmpl-{number}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if number < 7 else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                },
            )

        async with httpx.AsyncClient(
            transport=TraceTransport(trace, httpx.MockTransport(respond))
        ) as http:
            provider = OpenAIProvider(
                openai_client=AsyncOpenAI(api_key="local", http_client=http)
            )
            model = QwenChatModel("qwen-test", provider=provider)
            model.request_trace = trace

            def inspect_image() -> ToolReturn:
                return ToolReturn(
                    return_value=[
                        "observed",
                        BinaryContent(data=PNG, media_type="image/png"),
                    ]
                )

            agent = Agent(
                model,
                tools=[inspect_image],
                capabilities=[
                    traced_compaction(
                        lambda: SlidingWindowCompaction(
                            max_messages=4, keep_messages=2
                        ),
                        trace,
                    )
                ],
            )
            result = await agent.run(
                ["task", BinaryContent(data=PNG, media_type="image/png")]
            )
            assert result.output == "done"
        assert len(calls) == 7
        assert len(list((tmp_path / "assets").glob("*.png"))) == 1
        assert (tmp_path / "compactions.jsonl").exists()
        assert any("<tool_response>" in str(x) for c in calls for x in c["messages"])
        native = list((tmp_path / "native").glob("*.json"))
        assert len(native) == 7
        for path in (tmp_path / "requests").glob("*/metadata.json"):
            meta = json.loads(path.read_text())
            assert meta["complete"] and meta["native_id"]
        first = json.loads((tmp_path / "requests/000001/request.json").read_text())
        ref = first["messages"][0]["content"][1]["image_url"]["url"]
        assert (tmp_path / ref["$asset"]).read_bytes() == PNG
        assert "Authorization" not in str(first)

    asyncio.run(run())


def _episode_fixture(path):
    from rpent.planner.request_trace import write_json

    trace = path / "api_trace"
    directory = trace / "requests/000001"
    write_json(directory / "request.json", {"messages": []})
    write_json(
        directory / "metadata.json",
        {
            "complete": True,
            "status_code": 200,
            "content_type": "application/json",
            "native_id": "000001",
        },
    )
    write_json(
        directory / "response.body",
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    write_json(
        trace / "responses/000001.json",
        [{"parts": [{"part_kind": "text", "content": "done"}]}],
    )
    write_json(
        path / "result.json",
        {"success": False, "agent_error": None, "stats": {"turns_used": 1}},
    )
    write_json(path / "transcript_test.json", {"error": None})
    write_json(path / "states.json", {"steps": [{"terminated": False}]})
    write_json(path / "rollout.json", {})
    return directory


def test_native_task_failure_is_valid_but_success_must_match_environment(tmp_path):
    from rpent.evaluation.trajectory import audit_episode

    _episode_fixture(tmp_path)
    result, turns = audit_episode(tmp_path)
    assert not result["success"] and len(turns) == 1
    path = tmp_path / "result.json"
    result = json.loads(path.read_text())
    result["success"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="native success mismatch"):
        audit_episode(tmp_path)


def test_wire_native_output_mismatch_cannot_enter_sft(tmp_path):
    from rpent.evaluation.trajectory import audit_episode

    _episode_fixture(tmp_path)
    path = tmp_path / "api_trace/responses/000001.json"
    value = json.loads(path.read_text())
    value[0]["parts"][0]["content"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed turn coverage"):
        audit_episode(tmp_path)


def test_changed_asset_bytes_are_rejected(tmp_path):
    from rpent.evaluation.trajectory import restore

    trace = RequestTrace(tmp_path)
    reference = {
        **trace.asset(PNG),
        "$encoding": "data_uri",
        "data_uri_prefix": "data:image/png;base64,",
    }
    (tmp_path / reference["$asset"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore(reference, tmp_path)


def test_interrupted_stream_is_not_complete(tmp_path):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[]}\n\n'
            raise httpx.ReadError("interrupted")

    async def run():
        transport = TraceTransport(
            RequestTrace(tmp_path),
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=Broken(), headers={"content-type": "text/event-stream"}
                )
            ),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            try:
                await client.post(
                    "http://local/v1/chat/completions",
                    json={"messages": [], "stream": True},
                )
            except httpx.ReadError:
                pass
        meta = json.loads((tmp_path / "requests/000001/metadata.json").read_text())
        assert not meta["complete"]
        assert "interrupted" in meta["error"]
        assert (tmp_path / "requests/000001/response.body").read_bytes()

    asyncio.run(run())


def test_openai_stream_cleanup_cannot_truncate_recorded_body(tmp_path):
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    raw = (
        b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"qwen-test",'
        b'"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n'
        b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"qwen-test",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )

    class OneShot(httpx.AsyncByteStream):
        consumed = False

        async def __aiter__(self):
            if not self.consumed:
                self.consumed = True
                yield raw

    async def run():
        trace = RequestTrace(tmp_path)
        async with httpx.AsyncClient(
            transport=TraceTransport(
                trace,
                httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        stream=OneShot(),
                        headers={"content-type": "text/event-stream"},
                    )
                ),
            )
        ) as http:
            model = QwenChatModel(
                "qwen-test",
                provider=OpenAIProvider(
                    openai_client=AsyncOpenAI(api_key="local", http_client=http)
                ),
            )
            model.request_trace = trace
            async with model.request_stream(
                [ModelRequest(parts=[UserPromptPart("hello")])],
                None,
                ModelRequestParameters(),
            ) as stream:
                async for event in stream:
                    pass
            await asyncio.sleep(0)
        assert (tmp_path / "requests/000001/response.body").read_bytes() == raw
        meta = json.loads((tmp_path / "requests/000001/metadata.json").read_text())
        assert meta["complete"] and meta["error"] is None

    asyncio.run(run())


@pytest.mark.parametrize("finish_reason", ["length", None])
def test_output_limit_cannot_enter_sft(tmp_path, finish_reason):
    from rpent.evaluation.trajectory import audit_episode

    directory = _episode_fixture(tmp_path)
    value = json.loads((directory / "response.body").read_text())
    value["choices"][0]["finish_reason"] = finish_reason
    (directory / "response.body").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed turn coverage"):
        audit_episode(tmp_path)


def test_stream_reader_requires_done_and_keeps_tool_fragments(tmp_path):
    from rpent.evaluation.trajectory import read_response
    from rpent.planner.request_trace import write_json

    write_json(
        tmp_path / "metadata.json",
        {
            "complete": True,
            "status_code": 200,
            "content_type": "text/event-stream",
        },
    )
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "think",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "function": {"name": "finish", "arguments": "{"},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "}"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    body = "\n\n".join("data: " + json.dumps(chunk) for chunk in chunks)
    (tmp_path / "response.body").write_text(body)
    with pytest.raises(ValueError, match="SSE lacks DONE"):
        read_response(tmp_path)
    (tmp_path / "response.body").write_text(body + "\n\ndata: [DONE]\n\n")
    message, metadata = read_response(tmp_path)
    assert message["reasoning_content"] == "think"
    assert message["tool_calls"][0]["function"] == {"name": "finish", "arguments": "{}"}
    assert metadata["finish_reason"] == "tool_calls"
