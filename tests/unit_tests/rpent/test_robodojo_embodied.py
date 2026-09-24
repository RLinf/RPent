# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

"""Exercise the RoboDojo adapter through its real MCP transport."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import BinaryContent

from rpent.benchmarks.robodojo import (
    RoboDojoEmbodiedClient,
    _demo_context,
    _snapshot_messages,
    _tool_system_prompt,
)
from rpent.planner.base import PlannerResult


class _VLMError(RuntimeError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message)
        self.details = kwargs


def _messages() -> list[dict[str, Any]]:
    image = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-data").decode()
    return [
        {"role": "system", "content": "Use RoboDojo commands."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "Example: left move x 5"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "Pick up the cup."},
            ],
        },
    ]


def test_robodojo_client_snapshot_and_submit_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_planner(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["llm_config"].openai_format == "responses"
        assert kwargs["llm_config"].prompt_cache_mode == "explicit"
        assert kwargs["llm_config"].prompt_cache_key == "rpent-robodojo-v1:gpt-test"
        assert kwargs["llm_config"].image_history_groups == 2

        class Planner:
            def solve(self, **parameters: Any) -> PlannerResult:
                toolkit = parameters["toolkit"]
                initial = parameters["user_message"]
                assert isinstance(initial, list)
                assert any(isinstance(part, BinaryContent) for part in initial)
                assert "Example: left move x 5" in str(initial)
                snapshot = toolkit.execute_tool("dojo__snapshot", {})
                images = [
                    block
                    for block in snapshot.content_blocks
                    if block["type"] == "image"
                ]
                assert len(images) == 1
                assert base64.b64decode(images[0]["source"]["data"]) == b"jpeg-data"
                assert "Pick up the cup" in str(snapshot.content_blocks)
                assert "Example: left move x 5" not in str(snapshot.content_blocks)
                submitted = toolkit.execute_tool(
                    "dojo__submit_action",
                    {"commands": ["left move x 5"], "plan": "approach"},
                )
                assert "Action submitted" in str(submitted.result)
                return PlannerResult(
                    stats={
                        "total_input_tokens": 100,
                        "total_output_tokens": 20,
                        "cache_read_tokens": 30,
                        "cache_write_tokens": 5,
                        "requests": 2,
                    }
                )

        return Planner()

    monkeypatch.setattr("rpent.embodied_agent.build_planner", fake_planner)
    client = RoboDojoEmbodiedClient(
        model="gpt-test",
        base_url="https://example.invalid/v1",
        output_dir=tmp_path,
        error_type=_VLMError,
    )
    result = client.complete(_messages())
    assert json.loads(result["text"])["commands"] == ["left move x 5"]
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 30
    assert result["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 5
    assert result["attempts"] == 2
    assert not list(tmp_path.rglob("snapshot.json"))


def test_robodojo_missing_action_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            return PlannerResult(
                error="model ended without tools", stats={"requests": 1}
            )

    monkeypatch.setattr(
        "rpent.embodied_agent.build_planner", lambda *args, **kwargs: Planner()
    )
    client = RoboDojoEmbodiedClient(
        model="gpt-test",
        base_url="https://example.invalid/v1",
        output_dir=tmp_path,
        error_type=_VLMError,
    )
    with pytest.raises(_VLMError, match="produced no RoboDojo action"):
        client.complete(_messages())


def test_robodojo_preserves_permanent_provider_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Planner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            return PlannerResult(
                error="ModelHTTPError: status_code: 403, model_name: test",
                stats={"requests": 1},
            )

    monkeypatch.setattr(
        "rpent.embodied_agent.build_planner", lambda *args, **kwargs: Planner()
    )
    client = RoboDojoEmbodiedClient(
        model="gpt-test",
        base_url="https://example.invalid/v1",
        output_dir=tmp_path,
        error_type=_VLMError,
    )
    with pytest.raises(_VLMError) as caught:
        client.complete(_messages())
    assert caught.value.details["status_code"] == 403


def test_snapshot_keeps_current_images_when_demo_is_large() -> None:
    messages = _messages()[1:]
    messages[0]["content"] = messages[0]["content"][:1] * 8
    output = _snapshot_messages(messages, max_images=3)
    assert len([item for item in output["content"] if item["type"] == "image"]) == 3
    assert output["content"][-1]["text"].endswith("Pick up the cup.")


def test_demo_context_preserves_text_and_bounds_images() -> None:
    parts = _demo_context(_messages()[1:-1], max_images=1)
    assert len([part for part in parts if isinstance(part, BinaryContent)]) == 1
    assert "Example: left move x 5" in str(parts)


def test_robodawn_json_instruction_is_replaced_by_tool_workflow() -> None:
    prompt = "Robot frame rules.\nRESPONSE FORMAT: reply with ONE JSON object"
    adapted = _tool_system_prompt(prompt)
    assert "Robot frame rules" in adapted
    assert "dojo__snapshot" in adapted
    assert "dojo__submit_action" in adapted
    assert "ONE JSON object" not in adapted
