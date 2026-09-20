# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single read-only tool; no file, image, finish, or motion capabilities."""

from __future__ import annotations

from typing import Any

from robots.lynsense.ros2_adapter import Ros2StateAdapter
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import Toolkit, readonly


class LynsenseToolkit(Toolkit):
    include_image_reader = False

    def __init__(
        self,
        *,
        adapter: Ros2StateAdapter,
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
    ) -> None:
        super().__init__(dashboard_events=dashboard_events, memory=memory)
        self._adapter = adapter
        self.add_tool(
            "read_robot_state",
            {
                "name": "read_robot_state",
                "description": (
                    "Read the latest right-arm state snapshot. Status ok means "
                    "readable data, not robot health or permission to move."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            self.read_robot_state,
        )

    def _register_common_tools(self) -> None:
        """This backend deliberately exposes no common file tools."""

    @readonly
    def read_robot_state(self) -> dict[str, Any]:
        return self._adapter.get_snapshot()

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        return self._adapter.get_snapshot()

    def solved(self) -> bool:
        return False

    def close(self) -> None:
        self._adapter.close()
