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

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import execute_tool
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool


@pytest.mark.parametrize("cancel_queued", [False, True])
def test_cancellation_drains_capture_and_executor_queue_with_full_pool(
    tmp_path, cancel_queued
):
    class TestToolkit(Toolkit):
        def _register_common_tools(self):
            pass

        def get_env_state(self, **kwargs):
            loop.call_soon_threadsafe(capturing.set)
            assert release.wait(5)
            return ToolResult(data=kwargs["result"])

    toolkit = TestToolkit(
        dashboard_events=NullDashboardEventSink(),
        state=EnvState(tmp_path),
        memory=SimpleNamespace(),
    )
    release = threading.Event()
    executed = []

    @tool
    def move(label: str) -> ToolResult:
        executed.append(label)
        if label == "active":
            loop.call_soon_threadsafe(started.set)
            while not release.wait(0.005):
                toolkit.raise_if_cancelled()
        return ToolResult(data={"label": label})

    toolkit.add_tool(move)

    async def scenario():
        nonlocal loop, started, capturing
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        started, capturing = asyncio.Event(), asyncio.Event()
        active = asyncio.create_task(execute_tool(toolkit, "move", {"label": "active"}))
        tasks = [active]
        try:
            await asyncio.wait_for(started.wait(), 3)
            # One worker runs the action, the other waits in Toolkit, and the
            # remaining invocations are still queued in the default executor.
            tasks.extend(
                asyncio.create_task(execute_tool(toolkit, "move", {"label": str(i)}))
                for i in range(4)
            )
            await asyncio.sleep(0.05)
            cancelled = tasks[-1] if cancel_queued else active
            cancelled.cancel()
            await asyncio.wait_for(capturing.wait(), 3)
            assert not cancelled.done()
            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 3
            )
            assert isinstance(results[tasks.index(cancelled)], asyncio.CancelledError)
            assert all(
                isinstance(result, asyncio.CancelledError) or result.is_error
                for result in results
            )
            assert executed == ["active"]
            toolkit.resume_calls()
            result = await execute_tool(toolkit, "move", {"label": "next"})
            assert not result.is_error
            assert executed == ["active", "next"]
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            toolkit.close()

    loop = started = capturing = None
    asyncio.run(scenario())
