from __future__ import annotations

import asyncio
from typing import Any

from rpent.planner.api_loop import _ApiDashboardSession


class _Control:
    def __init__(self) -> None:
        self.completions = 0

    async def complete(self, driver: Any) -> None:
        self.completions += 1


def _session(control: _Control) -> _ApiDashboardSession:
    return _ApiDashboardSession(
        agent=object(),  # type: ignore[arg-type]
        control=control,  # type: ignore[arg-type]
        observer=object(),  # type: ignore[arg-type]
        max_turns=10,
        no_images=False,
    )


def test_busy_dashboard_messages_start_independent_api_runs() -> None:
    async def scenario() -> None:
        control = _Control()
        session = _session(control)
        first_running = asyncio.Event()
        tool_boundary = asyncio.Event()
        prompts: list[str] = []

        async def run_agent(seed: str) -> bool:
            prompts.append(seed)
            if seed == "initial":
                first_running.set()
                await tool_boundary.wait()
            return True

        session._run_agent = run_agent  # type: ignore[method-assign]

        assert await session.submit("initial") == 1
        await first_running.wait()
        assert await session.submit("queued") == 1
        tool_boundary.set()

        task = session._run_task
        assert task is not None
        await task

        assert prompts == ["initial", "queued"]
        assert control.completions == 2

    asyncio.run(scenario())


def test_dashboard_message_after_interrupt_starts_independent_api_run() -> None:
    async def scenario() -> None:
        control = _Control()
        session = _session(control)
        first_running = asyncio.Event()
        prompts: list[str] = []

        async def run_agent(seed: str) -> bool:
            prompts.append(seed)
            if seed == "interrupted":
                first_running.set()
                await asyncio.Event().wait()
            return True

        session._run_agent = run_agent  # type: ignore[method-assign]

        assert await session.submit("interrupted") == 1
        await first_running.wait()
        assert await session.interrupt() == 1
        assert await session.submit("after interrupt") == 1

        task = session._run_task
        assert task is not None
        await task

        assert prompts == ["interrupted", "after interrupt"]
        assert control.completions == 1

    asyncio.run(scenario())
