# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Host-only control of one pinned dsh runtime; never exposed as a model tool."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.dashboard.interaction import DashboardMessage

if TYPE_CHECKING:
    from rpent.dashboard.planner_control import DashboardPlannerControl
    from rpent.planner.deepseek_harness import _ToolBoundary


class HostCommandRejected(RuntimeError):
    """A verified rejection, distinct from loss of a command acknowledgement."""

    def __init__(self, reply: dict):
        super().__init__(reply.get("error", "dsh host command rejected"))
        self.reply = reply


class HostControl:
    """Serialize acknowledged commands without retrying an ambiguous submission."""

    def __init__(self, path: Path, run_id: str, timeout_s: float = 30):
        self.path = path
        self.run_id = run_id
        self.timeout_s = timeout_s
        self.seq = 0
        self._lock = asyncio.Lock()

    async def request(self, op: str, **values) -> dict:
        """Send exactly one command and validate its identity and acknowledgement."""
        async with self._lock:
            self.seq += 1
            request = {"runId": self.run_id, "seq": self.seq, "op": op, **values}
            result = await asyncio.to_thread(self._exchange, request)
            if result.get("runId") != self.run_id or result.get("seq") != self.seq:
                raise RuntimeError(
                    "dsh host control returned a foreign acknowledgement"
                )
            if result.get("ok") is not True:
                raise HostCommandRejected(result)
            return result

    def _exchange(self, request: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout_s)
            channel.connect(str(self.path))
            channel.sendall(json.dumps(request).encode() + b"\n")
            data = bytearray()
            while b"\n" not in data:
                chunk = channel.recv(65536)
                if not chunk:
                    raise RuntimeError(
                        "dsh host control closed without acknowledgement"
                    )
                data.extend(chunk)
                if len(data) > 1024 * 1024:
                    raise RuntimeError("dsh host control acknowledgement is too large")
            return json.loads(data.partition(b"\n")[0])


class DeepSeekSessionDriver:
    """Queue messages, acknowledge consumption, and establish safe pauses."""

    def __init__(
        self,
        host: HostControl,
        boundary: _ToolBoundary,
        *,
        operator_inputs: dict[str, dict] | None = None,
    ):
        self.host = host
        self.boundary = boundary
        self.control: DashboardPlannerControl | None = None
        self.receipts: dict[str, DashboardMessage] = {}
        self.acknowledged: set[str] = set()
        self.consumed: set[str] = set()
        self.paused = False
        self.error: str | None = None
        self.stop_reason: str | None = None
        self.operator_inputs = operator_inputs if operator_inputs is not None else {}

    async def submit(self, message: DashboardMessage) -> int:
        """Queue ordinary input at the next safe model decision boundary."""
        try:
            if self.error:
                raise RuntimeError(self.error)
            if self.paused:
                self.boundary.resume_calls()
                await self.host.request("resume")
                self.paused = False
            values = {"text": message.text, "messageId": message.message_id}
            operator_text = message.operator_text
            if operator_text is None and message.source == "user":
                operator_text = message.text
            if (
                message.source in {"user", "dashboard_continuation"}
                and isinstance(operator_text, str)
                and operator_text.strip()
            ):
                # Retain the submitted candidate even if its ACK is lost. A final
                # consumed receipt can prove delivery; submission alone cannot.
                self.operator_inputs.setdefault(
                    message.message_id, {"text": operator_text}
                )
            if getattr(message, "observation", None) is not None:
                from rpent.planner.utils.http_mcp_server import mcp_result

                values["observation"] = mcp_result(
                    message.observation, include_structured=True
                )
            reply = await self.host.request("submit", **values)
            self.receipts[reply["receipt"]] = message
            if message.message_id in self.operator_inputs:
                self.operator_inputs[message.message_id]["receipt"] = reply["receipt"]
            if reply["receipt"] in self.consumed:
                self.message_consumed(reply["receipt"])
            return int(reply["newTurn"])
        except Exception as exc:
            if isinstance(exc, HostCommandRejected) and (
                exc.reply.get("stopReason") == "budget"
                or (
                    exc.reply.get("stopReason") == "finish"
                    and self.boundary.finish_result is not None
                )
            ):
                # A verified rejection after native finish is a terminal race,
                # not an ambiguous submission or an uncertain robot action.
                self.stop_reason = exc.reply["stopReason"]
                self.boundary.stopped.set()
                if self.control is not None:
                    self.control.message_discarded(message.message_id)
                return 0
            # A missing ACK cannot establish whether input was accepted. Seal
            # admission immediately; a retry could submit the same action twice.
            self.boundary.stopped.set()
            self.error = str(exc)
            raise

    def message_consumed(self, receipt: str) -> None:
        """Commit each delivered UI message once, using the runtime receipt."""
        self.consumed.add(receipt)
        message = self.receipts.get(receipt)
        if message is not None and receipt not in self.acknowledged:
            self.acknowledged.add(receipt)
            if self.control is not None:
                self.control.message_started(message.message_id, message.text)

    async def withdraw(self, message_id: str) -> bool:
        """Withdraw only a message that the runtime has not consumed."""
        receipt = next(
            (
                key
                for key, message in self.receipts.items()
                if message.message_id == message_id
            ),
            None,
        )
        if receipt is None or receipt in self.consumed:
            return False
        try:
            reply = await self.host.request("withdraw", receipt=receipt)
        except Exception as exc:
            self.boundary.stopped.set()
            self.error = str(exc)
            raise
        if reply.get("withdrawn"):
            self.acknowledged.add(receipt)
            return True
        return False

    async def interrupt(self) -> int:
        """Pause only after Toolkit drain and runtime quiescence are confirmed."""
        try:
            reply = await self.host.request("pause")
            for receipt in reply.get("discarded", []):
                message = self.receipts.get(receipt)
                if message is not None and receipt not in self.acknowledged:
                    if self.control is not None:
                        self.control.message_discarded(message.message_id)
                    self.acknowledged.add(receipt)
            if (
                not reply.get("quiescent")
                or reply.get("state") != "paused"
                or self.boundary.execution_uncertain
                or self.boundary.finish_result is not None
            ):
                raise RuntimeError("dsh stopped without a resumable, certain state")
            self.paused = True
            return 0  # turn/end remains responsible for its completion count.
        except Exception as exc:
            self.boundary.stopped.set()
            self.error = str(exc)
            raise
