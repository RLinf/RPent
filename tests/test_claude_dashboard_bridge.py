from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any

from rpent.planner.claude_code import _ClaudeDashboardBridge


class _Interaction:
    def __init__(self, calls: list[Any]) -> None:
        self.task_replacement_requested = True
        self._calls = calls

    def complete_task_replacement(self, error: str | None = None) -> None:
        self._calls.append(("replacement_complete", error))


class _BlockingToolkit:
    def __init__(self, calls: list[Any]) -> None:
        self._calls = calls
        self.cancel_started = threading.Event()
        self.allow_cancel_to_finish = threading.Event()

    def cancel_active_and_wait(self) -> None:
        self._calls.append("cancel_started")
        self.cancel_started.set()
        if not self.allow_cancel_to_finish.wait(timeout=1.0):
            raise TimeoutError("test did not release active tool")
        self._calls.append("cancel_finished")


class _Driver:
    def __init__(self, calls: list[Any]) -> None:
        self._calls = calls

    async def interrupt(self) -> None:
        self._calls.append("planner_interrupted")


class ClaudeDashboardBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_replacement_waits_for_active_tool_before_interrupt(
        self,
    ) -> None:
        calls: list[Any] = []
        interaction = _Interaction(calls)
        toolkit = _BlockingToolkit(calls)
        driver = _Driver(calls)
        bridge = _ClaudeDashboardBridge(
            interaction=interaction,
            toolkit=toolkit,
            emit_user=lambda _text: None,
        )

        replacement = asyncio.create_task(bridge._handle_task_replacement(driver))
        cancel_started = await asyncio.to_thread(toolkit.cancel_started.wait, 1.0)

        self.assertTrue(cancel_started)
        self.assertFalse(replacement.done())
        self.assertEqual(calls, ["cancel_started"])

        toolkit.allow_cancel_to_finish.set()

        self.assertTrue(await asyncio.wait_for(replacement, timeout=1.0))
        self.assertEqual(
            calls,
            [
                "cancel_started",
                "cancel_finished",
                "planner_interrupted",
                ("replacement_complete", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
