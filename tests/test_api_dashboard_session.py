from __future__ import annotations

import asyncio

from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.dashboard.state import DashboardState
from rpent.planner.api_loop import _ApiDashboardSession

_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task",
        "fields": (),
        "display": "task",
        "output_slug": "task",
    },
    "runtime_components": (),
    "frame_channels": (),
}


def _setup(tmp_path):
    state = DashboardState(
        run_id="test-session",
        output_dir=tmp_path,
        dashboard_spec=_DASHBOARD_SPEC,
    )
    state.set_planner_activity("idle", accepting_input=True)
    state.submit_input("first")
    state.submit_input("queued")
    emitted: list[str] = []
    control = DashboardPlannerControl(
        interaction=state,
        cancel_active_and_wait=lambda: None,
        emit_user=emitted.append,
        emit_initial_user=lambda: None,
        defer_message_ack=True,
    )
    session = _ApiDashboardSession(
        agent=object(),  # type: ignore[arg-type]
        control=control,
        observer=object(),  # type: ignore[arg-type]
        max_turns=10,
        no_images=False,
    )
    return state, control, session, emitted


def _statuses(state: DashboardState) -> list[str]:
    messages = state.snapshot()["interaction"]["messages"]
    return [message["status"] for message in messages]


def test_interrupt_restores_queued_api_messages_as_unsent(tmp_path) -> None:
    async def scenario() -> None:
        state, control, session, emitted = _setup(tmp_path)
        first_started = asyncio.Event()

        async def run_agent(seed: str) -> bool:
            assert seed == "first"
            first_started.set()
            await asyncio.Event().wait()
            return True

        session._run_agent = run_agent  # type: ignore[method-assign]

        await control._flush(session)
        assert _statuses(state) == ["sending", "sending"]

        await first_started.wait()
        assert _statuses(state) == ["sent", "sending"]
        assert state.request_interrupt() == "accepted"

        await control._process(session)

        assert _statuses(state) == ["sent", "unsent"]
        assert emitted == ["first"]
        assert state.planner_activity == "idle"

    asyncio.run(scenario())


def test_end_preserves_api_messages_that_never_started_as_unsent(tmp_path) -> None:
    async def scenario() -> None:
        state, control, session, emitted = _setup(tmp_path)

        await control._flush(session)
        control.end()
        await session.close()

        assert _statuses(state) == ["unsent", "unsent"]
        assert emitted == []

    asyncio.run(scenario())
