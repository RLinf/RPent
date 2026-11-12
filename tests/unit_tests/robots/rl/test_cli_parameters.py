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


"""Exercise editable launch defaults, CLI-to-wire settings, and resume boundaries offline."""

import asyncio
import json
import signal
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import RunContext, RunUsage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.providers.openai import OpenAIProvider

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import sliding_window_compaction
from rpent.planner.qwen import QwenChatModel, qwen_model_settings
from tools.rl import run_batch as batch

ROOT = Path(__file__).resolve().parents[4]


def batch_args(tmp_path, *options):
    parser, _ = batch.build_parser()
    return parser.parse_args(
        [
            "run",
            str(tmp_path / "batch"),
            "--jobs",
            str(tmp_path / "jobs.json"),
            "--gpus",
            "0",
            "--endpoints",
            "http://localhost:1/v1",
            *options,
        ]
    )


def clean_defaults(monkeypatch):
    for name in (
        "MODEL",
        "REASONING_EFFORT",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "MIN_P",
        "PRESENCE_PENALTY",
        "REPETITION_PENALTY",
        "MAX_TOKENS",
    ):
        monkeypatch.delenv("RPENT_" + name, raising=False)


def test_editable_run_defaults_and_explicit_cli_precedence(tmp_path, monkeypatch):
    clean_defaults(monkeypatch)
    source = (ROOT / "tools/rl/run.sh").read_text()
    assert "--max-tokens" in source
    changed = source.replace("RPENT_MAX_TOKENS:-32768", "RPENT_MAX_TOKENS:-4096")
    changed = changed.replace("  --thinking ", "  --no-thinking ")
    (tmp_path / "run.sh").write_text(changed)
    monkeypatch.setattr(batch, "EVAL_DIR", tmp_path)
    args = batch_args(tmp_path)
    batch.resolve_run_options(args)
    assert args.max_tokens == 4096
    assert args.no_thinking is True
    monkeypatch.setenv("RPENT_TEMPERATURE", "0.8")
    args = batch_args(
        tmp_path, "--max-tokens", "2048", "--temperature", ".65", "--thinking"
    )
    batch.resolve_run_options(args)
    assert args.max_tokens == 2048
    assert args.temperature == 0.65
    assert args.no_thinking is False
    assert "--thinking" in batch.episode_options(args)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--temperature", "nan"),
        ("--top-p", "1.1"),
        ("--min-p", "-.1"),
        ("--top-k", "0"),
        ("--presence-penalty", "3"),
        ("--repetition-penalty", "0"),
        ("--compaction-max-messages", "-1"),
    ],
)
def test_invalid_parameters_rejected_before_batch_launch(tmp_path, flag, value):
    args = batch_args(tmp_path, flag, value)
    with pytest.raises(ValueError):
        batch.resolve_run_options(args)
    assert not (tmp_path / "batch").exists()


def test_official_thinking_defaults_are_explicit(monkeypatch):
    clean_defaults(monkeypatch)
    s = qwen_model_settings(max_tokens=32768, reasoning_effort="xhigh")
    assert s["temperature"] == 1
    assert s["top_p"] == 0.95
    assert s["presence_penalty"] == 0
    assert s["extra_body"]["top_k"] == 20
    assert s["extra_body"]["min_p"] == 0
    assert s["extra_body"]["repetition_penalty"] == 1
    assert s["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_cli_overrides_reach_actual_http_wire_and_window(tmp_path, monkeypatch):
    from rpent.cli import main as cli
    from rpent.planner import base

    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.8-27b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AsyncOpenAI(
                api_key="unused", base_url="http://offline/v1", http_client=http
            )
            model = QwenChatModel(
                "qwen3.8-27b", provider=OpenAIProvider(openai_client=client)
            )
            monkeypatch.setattr(base, "build_api_model", lambda *a, **kw: model)
            args = cli._build_argparser().parse_args(
                [
                    "--robot",
                    "rl",
                    "--max-tokens",
                    "777",
                    "--thinking",
                    "--reasoning-effort",
                    "low",
                    "--context-window",
                    "262144",
                    "--compaction-fraction",
                    ".6",
                    "--compaction-max-messages",
                    "0",
                    "--compaction-keep-messages",
                    "10",
                    "--temperature",
                    ".65",
                    "--top-p",
                    ".87",
                    "--top-k",
                    "13",
                    "--min-p",
                    ".03",
                    "--presence-penalty",
                    ".2",
                    "--repetition-penalty",
                    "1.1",
                ]
            )
            planner = base.build_planner(
                "api",
                output_dir=tmp_path,
                recipe_tag="test",
                robot_name="rl",
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                dashboard_events=NullDashboardEventSink(),
                **cli._api_planner_options(args),
            )
            window = planner.compaction()
            assert window.max_messages is None and window.keep_messages == 10
            assert window.max_fraction == 0.6
            await model.request(
                [ModelRequest(parts=[UserPromptPart("test")])],
                planner.model_settings,
                ModelRequestParameters(),
            )
            await client.close()

    asyncio.run(run())
    b = bodies[0]
    assert b.get("max_completion_tokens", b.get("max_tokens")) == 777
    assert b["temperature"] == 0.65 and b["top_p"] == 0.87
    assert b["top_k"] == 13 and b["min_p"] == 0.03
    assert b["presence_penalty"] == 0.2 and b["repetition_penalty"] == 1.1
    assert b["reasoning_effort"] == "low"


def test_configurable_message_window_keeps_pairs_and_zero_disables_trigger():
    messages = [
        ModelRequest(parts=[SystemPromptPart("system"), UserPromptPart("task")])
    ]
    for i in range(41):
        messages += [
            ModelResponse(parts=[ToolCallPart("view", {}, tool_call_id=str(i))]),
            ModelRequest(
                parts=[ToolReturnPart("view", "measured", tool_call_id=str(i))]
            ),
        ]
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), prompt="task")

    def apply(maximum, keep):
        request = ModelRequestContext(
            model=model,
            messages=list(messages),
            model_settings={},
            model_request_parameters=ModelRequestParameters(),
        )
        window = sliding_window_compaction(
            context_window=262144,
            max_fraction=0.7,
            max_messages=maximum,
            keep_messages=keep,
        )()
        return asyncio.run(window.before_model_request(ctx, request)).messages

    assert apply(0, 10) == messages
    retained = apply(12, 10)
    assert len(retained) == 11 and retained[0] == messages[0]
    calls = {
        p.tool_call_id for m in retained for p in m.parts if isinstance(p, ToolCallPart)
    }
    returns = {
        p.tool_call_id
        for m in retained
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert calls == returns == {"36", "37", "38", "39", "40"}


@pytest.mark.parametrize(
    "flag,change",
    [
        ("--temperature", ".8"),
        ("--max-tokens", "8192"),
        ("--compaction-max-messages", "0"),
        ("--compaction-keep-messages", "20"),
    ],
)
def test_batch_manifest_pins_generation_and_window(tmp_path, monkeypatch, flag, change):
    monkeypatch.setenv("RPENT_MODEL", "openai-chat:qwen3.8-27b")
    (tmp_path / "jobs.json").write_text("[]")
    output = tmp_path / "batch"
    output.mkdir()
    args = batch_args(tmp_path, "--start-interval", "0")
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        batch.collect(args, output)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["run_options"]["--temperature"] == 1
        assert manifest["run_options"]["--compaction-max-messages"] == 80
        altered = batch_args(tmp_path, flag, change, "--start-interval", "0")
        with pytest.raises(ValueError, match="Resume configuration differs"):
            batch.collect(altered, output)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
