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

"""Image delivery policy contracts for Toolkit (issue #305)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Annotated

import pytest
from pydantic import Field

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.toolkit import ImageDeliveryPolicy

IMAGES_PER_ACTION = 3  # LIBERO-style robots embed 3 PNGs per action result


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _ImageToolkit(Toolkit):
    """Mimic a robot toolkit that embeds images after every stateful tool."""

    def __init__(
        self,
        output_dir: Path,
        *,
        image_policy: ImageDeliveryPolicy | None = None,
    ) -> None:
        self.events = _RecordingEventSink()
        super().__init__(
            dashboard_events=self.events,
            state=EnvState(output_dir),
            memory=MemoryManager(output_dir / "memory"),
            image_policy=image_policy,
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        with self.state.record_step(
            state={"step": 0},
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return ToolResult(
            data={"observation": "ok"},
            images=[b"main", b"agentview", b"wrist"],
        )

    def solved(self) -> bool:
        return False


def _register_action_toolkit(toolkit: _ImageToolkit) -> None:
    @tool
    def move(
        distance: Annotated[float, Field(description="meters to move")],
    ) -> ToolResult:
        """Move the robot."""
        return ToolResult(data={"moved": distance})


    toolkit.add_tool(move)


def test_unbounded_policy_replays_every_image_linearly(tmp_path: Path) -> None:
    """Bug repro: history image count grows linearly with the number of actions."""
    toolkit = _ImageToolkit(tmp_path / "state")
    _register_action_toolkit(toolkit)
    policy = ImageDeliveryPolicy(max_embedded_images=None)
    toolkit._image_policy = policy

    total_images = 0
    for step in range(5):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        total_images += len(result.images)

    assert total_images == 5 * IMAGES_PER_ACTION


def test_default_policy_caps_total_images_in_history(tmp_path: Path) -> None:
    """Fix check: total delivered images stay within the conversation budget."""
    toolkit = _ImageToolkit(tmp_path / "state")
    _register_action_toolkit(toolkit)

    payload_images: list[bytes] = []
    for step in range(10):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        payload_images.extend(result.images)

    budget = ImageDeliveryPolicy().max_embedded_images
    assert budget is not None
    assert len(payload_images) == budget
    assert len(payload_images) < 10 * IMAGES_PER_ACTION


def test_exhausted_budget_reports_omitted_images(tmp_path: Path) -> None:
    toolkit = _ImageToolkit(tmp_path / "state")
    _register_action_toolkit(toolkit)

    result = toolkit.execute_tool("move", {"distance": 0.0})
    assert len(result.images) == IMAGES_PER_ACTION
    assert "images_omitted" not in result.data

    for step in range(1, 5):
        result = toolkit.execute_tool("move", {"distance": float(step)})

    assert result.images == []
    assert result.data["images_omitted"] == IMAGES_PER_ACTION
    assert result.data["observation"] == "ok"


def test_per_result_cap_trims_single_result(tmp_path: Path) -> None:
    toolkit = _ImageToolkit(tmp_path / "state")
    toolkit._image_policy = ImageDeliveryPolicy(
        max_embedded_images=None, max_per_result=1
    )
    _register_action_toolkit(toolkit)

    for step in range(3):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        assert len(result.images) == 1
        assert result.data["images_omitted"] == IMAGES_PER_ACTION - 1
