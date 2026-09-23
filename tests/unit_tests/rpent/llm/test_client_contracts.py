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
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from rpent.dashboard.events import NullDashboardEventSink
from rpent.llm import LLMClient, LLMConfig, LLMUsage, RetryPolicy
from rpent.llm.retry import RetryLoggingModel
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
    assert isinstance(planner._model, RetryLoggingModel)
    assert planner._model.wrapped is model
    assert planner._model.log_path == tmp_path / "llm_errors.jsonl"


@pytest.mark.parametrize(
    ("status", "expected_attempts"),
    [(400, 1), (401, 1), (429, 3), (503, 3)],
)
def test_http_failures_have_bounded_retries_and_sanitized_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    expected_attempts: int,
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise ModelHTTPError(status, "offline", body={"prompt": "private text"})

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "openai",
            "offline",
            api_key="secret-key",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    with pytest.raises(ModelHTTPError):
        client.generate_sync("private text")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert attempts == expected_attempts
    assert len(records) == expected_attempts
    assert [record["will_retry"] for record in records] == [
        *([True] * (expected_attempts - 1)),
        False,
    ]
    assert all(record["status_code"] == status for record in records)
    assert "private text" not in log_path.read_text()
    assert "secret-key" not in log_path.read_text()


def test_transient_failure_recovers_without_replaying_completed_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelHTTPError(429, "offline")
        return ModelResponse(
            parts=[TextPart("recovered")],
            usage=RequestUsage(input_tokens=4, output_tokens=2),
        )

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "anthropic",
            "offline",
            api_key="test",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    result = asyncio.run(client.generate("task"))
    assert attempts == 2
    assert result.text == "recovered"
    assert result.usage.requests == 1
    assert len(log_path.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("error", "expected_attempts"),
    [(ModelAPIError("offline", "connection failed"), 3), (ValueError("bad data"), 1)],
)
def test_non_http_failures_are_logged_and_only_connection_errors_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    expected_attempts: int,
) -> None:
    attempts = 0

    def model(messages: list[Any], info: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    log_path = tmp_path / "llm_errors.jsonl"
    client = LLMClient(
        LLMConfig(
            "openai",
            "offline",
            api_key="test",
            retry=RetryPolicy(max_retries=2, initial_delay_s=0, max_delay_s=0),
        ),
        log_path=log_path,
    )
    with pytest.raises(type(error)):
        client.generate_sync("task")
    assert attempts == expected_attempts
    assert len(log_path.read_text().splitlines()) == expected_attempts


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_sdk_retries_are_disabled_for_explicit_policy(provider: str) -> None:
    client = LLMClient(LLMConfig(provider, "offline", api_key="test"))
    assert client._model.wrapped.provider.client.max_retries == 0


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
