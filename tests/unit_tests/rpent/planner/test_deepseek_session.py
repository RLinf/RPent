# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Host control identity, acknowledgement, and irreversible stop contracts."""

from unittest.mock import AsyncMock, Mock

import pytest

from rpent.dashboard.interaction import DashboardMessage
from rpent.planner.deepseek_session import (
    DeepSeekSessionDriver,
    HostCommandRejected,
    HostControl,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_control_serializes_identity_and_sequence_without_retry(
    tmp_path, monkeypatch
):
    host = HostControl(tmp_path / "host.sock", "run-identity")
    requests = []

    def exchange(request):
        requests.append(request)
        return {**request, "ok": True}

    monkeypatch.setattr(host, "_exchange", exchange)
    await host.request("status")
    await host.request("pause")
    assert [(r["runId"], r["seq"], r["op"]) for r in requests] == [
        ("run-identity", 1, "status"),
        ("run-identity", 2, "pause"),
    ]

    def lost(request):
        requests.append(request)
        raise TimeoutError("ACK lost")

    monkeypatch.setattr(host, "_exchange", lost)
    with pytest.raises(TimeoutError):
        await host.request("submit", text="one action")
    assert len(requests) == 3


@pytest.mark.anyio
async def test_foreign_ack_cannot_complete_control(tmp_path, monkeypatch):
    host = HostControl(tmp_path / "host.sock", "new-task")
    monkeypatch.setattr(
        host,
        "_exchange",
        lambda request: {
            "runId": "old-task",
            "seq": request["seq"],
            "ok": True,
        },
    )
    with pytest.raises(RuntimeError, match="foreign"):
        await host.request("submit", text="move")


def driver_fixture():
    boundary = Mock(execution_uncertain=False, finish_result=None)
    host = Mock(request=AsyncMock())
    driver = DeepSeekSessionDriver(host, boundary)
    driver.control = Mock()
    return driver, host, boundary


@pytest.mark.anyio
async def test_consumption_can_arrive_before_submit_ack_and_is_deduplicated():
    driver, host, boundary = driver_fixture()

    async def submit(*args, **kwargs):
        driver.message_consumed("receipt")
        return {"receipt": "receipt", "newTurn": False}

    host.request.side_effect = submit
    assert await driver.submit(DashboardMessage("message", "move once", "sending")) == 0
    driver.message_consumed("receipt")
    driver.control.message_started.assert_called_once_with("message", "move once")
    boundary.resume_calls.assert_not_called()


@pytest.mark.anyio
async def test_missing_submit_ack_permanently_seals_admission():
    driver, host, boundary = driver_fixture()
    host.request.side_effect = TimeoutError("submission may already be accepted")
    message = DashboardMessage("message", "move once", "sending")
    with pytest.raises(TimeoutError):
        await driver.submit(message)
    boundary.stopped.set.assert_called_once()
    with pytest.raises(RuntimeError):
        await driver.submit(message)
    assert host.request.await_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["budget", "finish"])
async def test_verified_terminal_rejection_restores_unconsumed_input(reason):
    driver, host, boundary = driver_fixture()
    if reason == "finish":
        boundary.finish_result = {"_finish": True, "status": "success"}
    host.request.side_effect = HostCommandRejected(
        {"stopReason": reason, "error": "run has ended"}
    )
    assert await driver.submit(DashboardMessage("m", "later", "sending")) == 0
    assert driver.error is None
    assert driver.stop_reason == reason
    driver.control.message_discarded.assert_called_once_with("m")
    boundary.stopped.set.assert_called_once()
    host.request.assert_awaited_once()


@pytest.mark.anyio
async def test_remote_finish_claim_without_native_finish_cannot_hide_rejection():
    driver, host, boundary = driver_fixture()
    host.request.side_effect = HostCommandRejected(
        {"stopReason": "finish", "error": "unverified finish"}
    )
    with pytest.raises(HostCommandRejected):
        await driver.submit(DashboardMessage("m", "later", "sending"))
    assert driver.error
    assert driver.stop_reason is None
    driver.control.message_discarded.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "uncertain,state,quiescent",
    [
        (True, "paused", True),
        (False, "terminal", True),
        (False, "paused", False),
    ],
)
async def test_unsafe_interrupt_never_reopens(uncertain, state, quiescent):
    driver, host, boundary = driver_fixture()
    boundary.execution_uncertain = uncertain
    host.request.return_value = {"state": state, "quiescent": quiescent}
    with pytest.raises(RuntimeError, match="resumable"):
        await driver.interrupt()
    assert driver.error
    assert not driver.paused
    boundary.resume_calls.assert_not_called()


@pytest.mark.anyio
async def test_pause_restores_only_unconsumed_messages():
    driver, host, boundary = driver_fixture()
    driver.receipts = {
        "pending": DashboardMessage("p", "later", "sending"),
        "sent": DashboardMessage("s", "already delivered", "sending"),
    }
    driver.message_consumed("sent")
    host.request.return_value = {
        "state": "paused",
        "quiescent": True,
        "discarded": ["pending", "sent"],
    }
    assert await driver.interrupt() == 0
    driver.control.message_discarded.assert_called_once_with("p")
    assert driver.paused
    boundary.resume_calls.assert_not_called()


@pytest.mark.anyio
async def test_withdraw_does_not_resubmit_or_interrupt_other_input():
    driver, host, boundary = driver_fixture()
    driver.receipts["r"] = DashboardMessage("m", "withdraw this", "sending")
    host.request.return_value = {"withdrawn": True}
    assert await driver.withdraw("m")
    host.request.assert_awaited_once_with("withdraw", receipt="r")
    boundary.stopped.set.assert_not_called()
