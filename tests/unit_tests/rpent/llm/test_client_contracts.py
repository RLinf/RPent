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

"""Offline contracts for provider selection and usage accounting."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from rpent.dashboard.events import NullDashboardEventSink
from rpent.llm import LLMClient, LLMConfig, LLMUsage
from rpent.planner.api_loop import _build_stats
from rpent.planner.base import build_planner


@pytest.mark.parametrize(
    ("config", "model_type"),
    [
        (LLMConfig("openai", "gpt-4o", api_key="test"), "OpenAIResponsesModel"),
        (
            LLMConfig("openai", "gpt-4o", api_key="test", openai_format="chat"),
            "OpenAIChatModel",
        ),
        (LLMConfig("anthropic", "claude-test", api_key="test"), "AnthropicModel"),
    ],
)
def test_provider_configuration_builds_selected_model(
    config: LLMConfig, model_type: str
) -> None:
    model = config.build_model()
    assert type(model).__name__ == model_type
    assert model.model_name == config.model


def test_direct_calls_report_per_call_and_cumulative_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def model(messages: list[Any], info: Any) -> ModelResponse:
        assert info.instructions == "Be concise."
        assert info.model_settings["max_tokens"] == 128
        assert messages
        return ModelResponse(
            parts=[TextPart("done")],
            usage=RequestUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=3,
                cache_write_tokens=2,
                details={"reasoning_tokens": 1},
            ),
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    client = LLMClient(LLMConfig("openai", "offline", api_key="test"))
    sync = client.generate_sync("first", system_prompt="Be concise.", max_tokens=128)
    async_result = asyncio.run(
        client.generate("second", system_prompt="Be concise.", max_tokens=128)
    )

    assert sync.text == async_result.text == "done"
    assert sync.usage.as_dict() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 3,
        "cache_write_tokens": 2,
        "reasoning_output_tokens": 1,
        "requests": 1,
        "cost_usd": None,
        "total_tokens": 15,
    }
    assert client.total_usage.input_tokens == 20
    assert client.total_usage.output_tokens == 10
    assert client.total_usage.cache_read_tokens == 6
    assert client.total_usage.requests == 2


def test_api_planner_uses_explicit_llm_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    monkeypatch.setattr(LLMConfig, "build_model", lambda self: model)
    config = LLMConfig("anthropic", "offline", api_key="test")
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="test",
        llm_config=config,
        dashboard_events=NullDashboardEventSink(),
    )
    assert planner._model is model


def test_planner_usage_normalizes_cache_without_double_counting() -> None:
    api = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "cache_read_tokens": 3,
            "cache_write_tokens": 2,
            "requests": 1,
        }
    )
    claude_sdk = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 5,
            "total_output_tokens": 5,
            "total_cache_read_input_tokens": 3,
            "total_cache_creation_input_tokens": 2,
        }
    )
    codex_sdk = LLMUsage.from_planner_stats(
        {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_cached_input_tokens": 3,
            "total_reasoning_output_tokens": 1,
        }
    )
    assert api.input_tokens == claude_sdk.input_tokens == codex_sdk.input_tokens == 10
    assert api.total_tokens == claude_sdk.total_tokens == codex_sdk.total_tokens == 15
    assert api.cache_read_tokens == claude_sdk.cache_read_tokens == 3
    assert codex_sdk.reasoning_output_tokens == 1


def test_api_planner_preserves_reported_reasoning_and_cost() -> None:
    stats = _build_stats(
        RunUsage(
            input_tokens=10,
            output_tokens=5,
            details={"reasoning_tokens": 2},
            cost=0.01,
        ),
        turns=1,
        n_tool_calls=0,
    )
    usage = LLMUsage.from_planner_stats(stats)
    assert usage.reasoning_output_tokens == 2
    assert usage.cost_usd == 0.01
