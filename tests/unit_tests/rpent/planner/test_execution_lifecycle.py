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

from rpent.memory import MemoryManager
from rpent.planner.base import execute_tool
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, readonly, tool


@tool
def finish(status: str, summary: str, *, ctx: ToolContext) -> ToolResult:
    """Accept the requested outcome for this test toolkit."""
    return ToolResult(data={"_finish": True, "status": status, "summary": summary})


@tool
@readonly
def moving(*, ctx: ToolContext) -> ToolResult:
    """Wait at a cooperative cancellation boundary."""
    ctx.robot.entered()
    assert ctx._cancel_event.wait(3)
    ctx.check_cancelled()
    raise AssertionError("cancelled handler continued")


def test_cancel_drains_jobs_even_when_default_tool_pool_is_full(tmp_path):
    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        queued = asyncio.Event()

        class ObservedExecutor(ThreadPoolExecutor):
            submissions = 0

            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                self.submissions += 1
                if self.submissions == 2:
                    queued.set()
                return future

        loop.set_default_executor(ObservedExecutor(max_workers=1))
        toolkit = Toolkit(
            state=EnvState(tmp_path),
            memory=MemoryManager(tmp_path / "memory"),
            robot=SimpleNamespace(
                entered=lambda: loop.call_soon_threadsafe(entered.set)
            ),
            output_dir=tmp_path,
            tools=(finish, moving),
        )
        try:
            active = asyncio.create_task(execute_tool(toolkit, "moving", {}))
            await asyncio.wait_for(entered.wait(), timeout=3)
            pending = asyncio.create_task(execute_tool(toolkit, "moving", {}))
            await asyncio.wait_for(queued.wait(), timeout=3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=3)
            assert (
                await asyncio.wait_for(active, timeout=3)
            ).error == "Tool call cancelled."
            assert (
                toolkit.execute_tool("list_dir", {}).error == "Tool calls are paused."
            )
            toolkit.resume_calls()
            assert not toolkit.execute_tool("list_dir", {}).is_error
        finally:
            toolkit.close()

    asyncio.run(scenario())


def test_request_cancellation_waits_for_capture_before_returning(tmp_path):
    async def scenario():
        loop = asyncio.get_running_loop()
        capturing = asyncio.Event()
        cancelled = asyncio.Event()
        release = threading.Event()

        @tool
        def action(*, ctx: ToolContext) -> ToolResult:
            """Perform an action before its final observation."""
            return ToolResult()

        class CapturingToolkit(Toolkit):
            def _capture_observation(self, **kwargs):
                loop.call_soon_threadsafe(capturing.set)
                assert release.wait(4)
                return {"step": 0}, []

            def cancel_active_and_wait(self):
                loop.call_soon_threadsafe(cancelled.set)
                super().cancel_active_and_wait()

        toolkit = CapturingToolkit(
            state=EnvState(tmp_path),
            memory=MemoryManager(tmp_path / "memory"),
            robot=None,
            output_dir=tmp_path,
            tools=(finish, action),
        )
        request = asyncio.create_task(execute_tool(toolkit, "action", {}))
        try:
            await asyncio.wait_for(capturing.wait(), timeout=3)
            request.cancel()
            await asyncio.wait_for(cancelled.wait(), timeout=3)
            assert not request.done()
            with pytest.raises(RuntimeError, match="cleanup"):
                toolkit.resume_calls()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(request, timeout=3)
            toolkit.resume_calls()
            assert not toolkit.execute_tool("list_dir", {}).is_error
        finally:
            release.set()
            toolkit.close()

    asyncio.run(scenario())
