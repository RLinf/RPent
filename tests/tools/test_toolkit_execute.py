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

"""Tests for :meth:`Toolkit.execute_tool` concurrency and state capture.

Read-only tools share the RWLock's read lock (concurrent); stateful tools
take the exclusive write lock (mutually exclusive, cancellable).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from rpent.tools.tool_spec import ToolSpec, readonly
from rpent.tools.toolkit import Toolkit


class _FakeState:
    def __init__(self, record=None):
        self.record = record

    def latest_record(self):
        return self.record


class _FakeMemory:
    def get_common_tool_bindings(self):
        return {}


class _FakeSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class _StubToolkit(Toolkit):
    """Minimal ``Toolkit`` subclass registering synthetic tools."""

    _FRAME_ARTIFACTS = {}

    def __init__(self, *, record=None):
        super().__init__(
            dashboard_events=_FakeSink(),
            state=_FakeState(record=record),
            memory=_FakeMemory(),
        )
        self._tools = {}  # isolate from the common file tools
        self.captured_commands = []

    def get_env_state(self, *, command, result, elapsed_s):
        self.captured_commands.append(command)
        return {"state": "captured"}


@pytest.fixture
def toolkit() -> _StubToolkit:
    return _StubToolkit()


def _add(toolkit, name, handler) -> None:
    toolkit.add_tool(
        name, {"name": name, "description": "d", "input_schema": {}}, handler
    )


# --------------------------------------------------------------------------
# Concurrency semantics
# --------------------------------------------------------------------------


def test_readonly_tools_run_concurrently(toolkit):
    gate = threading.Barrier(2)

    @readonly
    def handler(x, gate=None):
        if gate is not None:
            gate.wait(timeout=5)
        time.sleep(0.2)
        return {"value": x}

    _add(toolkit, "r1", handler)
    _add(toolkit, "r2", handler)

    results = {}

    def run(name):
        results[name] = toolkit.execute_tool(name, {"x": name, "gate": gate})

    threads = [threading.Thread(target=run, args=(n,)) for n in ("r1", "r2")]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    # Two 0.2s sleeps that overlap finish in well under 0.4s.
    assert elapsed < 0.4, f"read-only tools should overlap, took {elapsed:.3f}s"
    assert {results["r1"].result["value"], results["r2"].result["value"]} == {
        "r1",
        "r2",
    }


def test_stateful_tools_are_serialized(toolkit):
    def handler(x):
        time.sleep(0.2)
        return {"value": x}

    _add(toolkit, "s1", handler)
    _add(toolkit, "s2", handler)

    threads = [
        threading.Thread(target=lambda n=n: toolkit.execute_tool(n, {"x": n}))
        for n in ("s1", "s2")
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.4, f"stateful tools must not overlap, took {elapsed:.3f}s"
    assert {c["action"] for c in toolkit.captured_commands} == {"s1", "s2"}


def test_readonly_waits_for_in_flight_stateful_tool(toolkit):
    entered = threading.Event()
    release = threading.Event()

    def stateful():
        entered.set()
        release.wait(timeout=5)
        return {"ok": True}

    _add(toolkit, "write_hold", stateful)

    t1 = threading.Thread(target=lambda: toolkit.execute_tool("write_hold", {}))
    t1.start()
    assert entered.wait(timeout=2)

    done = threading.Event()

    @readonly
    def rd(x):
        done.set()
        return {"value": x}

    _add(toolkit, "rd", rd)
    t2 = threading.Thread(target=lambda: toolkit.execute_tool("rd", {"x": 1}))
    t2.start()

    # While the stateful tool holds the write lock, a read-only call is blocked.
    assert not done.wait(timeout=0.2), "read-only tool must wait for the write lock"

    release.set()
    t1.join(timeout=2)
    assert done.wait(timeout=2), (
        "read-only tool should proceed after the write lock drops"
    )
    t2.join(timeout=2)


# --------------------------------------------------------------------------
# Read-only vs stateful bookkeeping
# --------------------------------------------------------------------------


def test_readonly_has_no_active_operation(toolkit):
    seen = {}

    @readonly
    def rd():
        seen["active"] = toolkit._active_operation
        toolkit.raise_if_cancelled()  # must be a no-op for read-only tools
        return {"ok": True}

    _add(toolkit, "rd", rd)

    result = toolkit.execute_tool("rd", {})
    assert seen["active"] is None
    assert result.result["ok"] is True


def test_readonly_does_not_capture_state_and_stateful_does(toolkit):
    @readonly
    def rd(x):
        return {"value": x}

    def st(x):
        return {"value": x}

    _add(toolkit, "rd", rd)
    _add(toolkit, "st", st)

    toolkit.execute_tool("rd", {"x": 1})
    assert toolkit.captured_commands == []

    toolkit.execute_tool("st", {"x": 2})
    assert [c["action"] for c in toolkit.captured_commands] == ["st"]


def test_stateful_publishes_step_record(toolkit):
    toolkit = _StubToolkit(record=object())  # latest_record() returns a record
    _add(toolkit, "st", lambda: {"ok": True})

    toolkit.execute_tool("st", {})
    assert len(toolkit._dashboard_events.events) == 1


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_stateful_tool_can_be_cancelled(toolkit):
    started = threading.Event()

    def cancellable():
        started.set()
        while True:
            toolkit.raise_if_cancelled()
            time.sleep(0.005)

    _add(toolkit, "cancellable", cancellable)

    holder = {}
    thread = threading.Thread(
        target=lambda: holder.update(r=toolkit.execute_tool("cancellable", {}))
    )
    thread.start()
    assert started.wait(timeout=2)

    toolkit.cancel_active_and_wait()
    thread.join(timeout=2)

    result = holder["r"].result
    assert result.get("code") == "tool_cancelled"
    assert result.get("interrupted") is True


def test_cancel_with_no_active_operation_is_noop(toolkit):
    toolkit.cancel_active_and_wait()  # must return immediately, not hang


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_unknown_tool_returns_error(toolkit):
    result = toolkit.execute_tool("missing", {})
    assert result.result["error"] == "unknown tool: missing"


def test_add_tool_spec_registers_and_executes(toolkit):
    spec = ToolSpec(
        name="spec_tool",
        description="registered via add_tool_spec",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: {"ok": True},
        readonly=True,
    )
    toolkit.add_tool_spec(spec)

    names = [t.name for t in toolkit.get_tools_spec()]
    assert "spec_tool" in names

    result = toolkit.execute_tool("spec_tool", {})
    assert result.result == {"ok": True}


def test_add_tool_spec_rejects_non_tool_spec(toolkit):
    with pytest.raises(TypeError):
        toolkit.add_tool_spec({"name": "x", "description": "d", "input_schema": {}})


# --------------------------------------------------------------------------
# Async dispatch (as in ``_build_rpent_server``)
# --------------------------------------------------------------------------
#
# ``rpent/planner/claude_code.py:_build_rpent_server`` wraps each tool in an
# ``async def run_tool`` that calls ``asyncio.to_thread(toolkit.execute_tool,
# ...)`` and fans calls out via the SDK. These tests reproduce that dispatch
# to prove the parallelism comes from ``execute_tool``'s RWLock, not from any
# lock in the async layer.


def test_async_to_thread_readonly_tools_run_in_parallel(toolkit):
    """Two read-only calls dispatched via ``asyncio.to_thread`` overlap."""
    gate = threading.Barrier(2)

    @readonly
    def rd(x):
        gate.wait(timeout=5)
        time.sleep(0.2)
        return {"value": x}

    _add(toolkit, "rd", rd)

    async def dispatch_and_measure() -> float:
        async def dispatch(name, args):
            return await asyncio.to_thread(toolkit.execute_tool, name, args)

        start = time.perf_counter()
        r1, r2 = await asyncio.wait_for(
            asyncio.gather(
                dispatch("rd", {"x": "a"}),
                dispatch("rd", {"x": "b"}),
            ),
            timeout=5,
        )
        elapsed = time.perf_counter() - start
        assert {r1.result["value"], r2.result["value"]} == {"a", "b"}
        return elapsed

    elapsed = asyncio.run(dispatch_and_measure())
    # Two 0.2s sleeps that overlap finish in well under 0.4s; the Barrier(2)
    # doubles as a hard proof that both threads entered the handler together.
    assert elapsed < 0.4, f"async read-only tools should overlap, took {elapsed:.3f}s"


def test_async_to_thread_stateful_tools_serialize(toolkit):
    """Two stateful calls dispatched via ``asyncio.to_thread`` do not overlap."""

    def st(x):
        time.sleep(0.2)
        return {"value": x}

    _add(toolkit, "s1", st)
    _add(toolkit, "s2", st)

    async def dispatch_and_measure() -> float:
        async def dispatch(name):
            await asyncio.to_thread(toolkit.execute_tool, name, {"x": name})

        start = time.perf_counter()
        await asyncio.wait_for(
            asyncio.gather(dispatch("s1"), dispatch("s2")),
            timeout=5,
        )
        return time.perf_counter() - start

    elapsed = asyncio.run(dispatch_and_measure())
    assert elapsed >= 0.4, f"async stateful tools must not overlap, took {elapsed:.3f}s"
    assert {c["action"] for c in toolkit.captured_commands} == {"s1", "s2"}
