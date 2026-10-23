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

from pathlib import Path
from typing import Annotated, Any

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


def _register_view_toolkit(toolkit: _ImageToolkit) -> None:
    @tool(readonly=True)
    def view_env_state() -> ToolResult:
        """Explicit observation request; mirrors robots' view_env_state."""
        return ToolResult(
            data={"view": "ok"},
            images=[b"agentview", b"wrist"],
        )

    toolkit.add_tool(view_env_state)


def test_default_policy_delivers_every_image(tmp_path: Path) -> None:
    """Default stays unlimited: no behavior change for any existing user."""
    assert ImageDeliveryPolicy().max_embedded_images is None

    toolkit = _ImageToolkit(tmp_path / "state")
    _register_action_toolkit(toolkit)

    total_images = 0
    for step in range(5):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        total_images += len(result.images)

    assert total_images == 5 * IMAGES_PER_ACTION


def test_opt_in_budget_caps_auto_attached_images(tmp_path: Path) -> None:
    """With an explicit budget, auto-attached images stay within the quota."""
    toolkit = _ImageToolkit(
        tmp_path / "state", image_policy=ImageDeliveryPolicy(max_embedded_images=12)
    )
    _register_action_toolkit(toolkit)

    payload_images: list[bytes] = []
    for step in range(10):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        payload_images.extend(result.images)

    assert len(payload_images) == 12
    assert len(payload_images) < 10 * IMAGES_PER_ACTION


def test_exhausted_budget_reports_omitted_images(tmp_path: Path) -> None:
    toolkit = _ImageToolkit(
        tmp_path / "state", image_policy=ImageDeliveryPolicy(max_embedded_images=12)
    )
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
    toolkit = _ImageToolkit(
        tmp_path / "state",
        image_policy=ImageDeliveryPolicy(max_embedded_images=None, max_per_result=1),
    )
    _register_action_toolkit(toolkit)

    for step in range(3):
        result = toolkit.execute_tool("move", {"distance": float(step)})
        assert len(result.images) == 1
        assert result.data["images_omitted"] == IMAGES_PER_ACTION - 1


def test_readonly_results_bypass_quota(tmp_path: Path) -> None:
    """Explicit observation requests always deliver images, even with no budget left.

    The planner has no other way to re-fetch scene images, so read-only results
    (e.g. view_env_state) must never be throttled by the auto-attach budget.
    """
    toolkit = _ImageToolkit(
        tmp_path / "state", image_policy=ImageDeliveryPolicy(max_embedded_images=3)
    )
    _register_action_toolkit(toolkit)
    _register_view_toolkit(toolkit)

    toolkit.execute_tool("move", {"distance": 0.0})  # budget fully spent
    result = toolkit.execute_tool("move", {"distance": 1.0})
    assert result.images == []
    assert result.data["images_omitted"] == IMAGES_PER_ACTION

    view = toolkit.execute_tool("view_env_state", {})
    assert view.images == [b"agentview", b"wrist"]
    assert "images_omitted" not in view.data
