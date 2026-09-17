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

import threading
from concurrent.futures import ThreadPoolExecutor

from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolContext, Toolkit, ToolResult, readonly, tool


def test_overlapping_calls_are_rejected_and_cancel_waits_for_active_call(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    contexts = []

    @tool
    @readonly
    def read(*, ctx: ToolContext) -> ToolResult:
        """Read until the current operation reaches its cancellation boundary."""
        contexts.append(ctx)
        entered.set()
        assert release.wait(5)
        ctx.check_cancelled()
        return ToolResult(data={"ok": True})

    toolkit = Toolkit(
        state=EnvState(tmp_path),
        memory=MemoryManager(tmp_path / "memory"),
        robot=None,
        output_dir=tmp_path,
        tools=(read,),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(toolkit.execute_tool, "read", {})
        try:
            assert entered.wait(5)
            rejected = toolkit.execute_tool("read", {})
            assert rejected.error == "another tool operation is still active"
            assert len(contexts) == 1
            cancelled = pool.submit(toolkit.cancel_active_and_wait)
            assert contexts[0]._cancel_event.wait(5)
            assert not cancelled.done()
        finally:
            release.set()
        assert active.result(5).error == "Tool call cancelled."
        cancelled.result(5)
    assert not toolkit.execute_tool("read", {}).is_error
    assert len(contexts) == 2
    assert not contexts[1]._cancel_event.is_set()
