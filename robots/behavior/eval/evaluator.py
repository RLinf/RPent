# Copyright 2026 The RPent Authors.
# Copyright 2026 The RLinf Authors.
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

"""Join official BEHAVIOR WebSocket observations to RPent action chunks.

``EvaluatorStepBridge`` owns observations, pending actions and execution counts.
``serve_websocket`` exposes that bridge to one official evaluator connection.
The Env service exposes the same bridge through RPC. Simulation and scoring
remain in the separately launched official evaluator.
"""

from __future__ import annotations

import asyncio
import http
import math
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from robots.behavior.env_client import validate_native_actions
from robots.behavior.tools import extract_public_proprio
from rpent.utils.logging import get_logger

logger = get_logger("behavior_policy_transport")


class BridgeCancelled(RuntimeError):
    """A reset, timeout or service shutdown cancelled an operation."""


class BridgeSessionEnded(BridgeCancelled):
    """The evaluator disconnected without reporting a task outcome."""


@dataclass
class _Generation:
    number: int
    issued: int = 0
    confirmed: int = 0
    cancelled: str | None = None
    session_end_reason: str | None = None


@dataclass
class _Chunk:
    actions: np.ndarray
    return_all_frames: bool
    issued: int = 0
    confirmed: int = 0
    final_observation: dict[str, Any] | None = None
    final_motion_snapshot: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    last_progress_at: float = field(default_factory=time.monotonic)


class EvaluatorStepBridge:
    """Hand observations and actions between one evaluator and RPent tools.

    Args:
        simulation_timeout_s: Maximum interval without a new observation.
        action_timeout_s: Maximum time an observation waits for an RPent action.
        chunk_timeout_s: Absolute deadline for a whole action chunk.
    """

    def __init__(
        self,
        *,
        simulation_timeout_s: float,
        action_timeout_s: float,
        chunk_timeout_s: float,
    ) -> None:
        for name, value in (
            ("simulation_timeout_s", simulation_timeout_s),
            ("action_timeout_s", action_timeout_s),
            ("chunk_timeout_s", chunk_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        self._simulation_timeout_s = float(simulation_timeout_s)
        self._action_timeout_s = float(action_timeout_s)
        self._chunk_timeout_s = float(chunk_timeout_s)
        self._condition = threading.Condition()
        self._generations = [_Generation(0)]
        self._closed = False
        self.last_obs: dict[str, Any] | None = None
        self._observation_generation: int | None = None
        self._active_chunk: _Chunk | None = None
        self._inflight: _Chunk | None = None
        self._receiver: object | None = None
        self._observation_index = -1
        self._snapshot: dict[str, Any] | None = None

    def _wait(
        self,
        record: _Generation,
        ready: Callable[[], bool],
        deadline: float,
        operation: str,
        timeout_s: float,
        *,
        allow_ended: bool = False,
    ) -> None:
        """Check session state and wait for a condition under the shared lock."""
        while True:
            if record.cancelled is not None or self._closed:
                raise BridgeCancelled(
                    f"BEHAVIOR bridge generation {record.number}: {record.cancelled or 'closed'}"
                )
            complete = ready()
            if record.session_end_reason is not None and not (allow_ended and complete):
                raise BridgeSessionEnded(
                    f"BEHAVIOR bridge generation {record.number}: session ended ({record.session_end_reason}); outcome unknown"
                )
            if complete:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop(
                    f"{operation} timed out after {timeout_s:g}s", state="cancelled"
                )
                continue
            self._condition.wait(remaining)

    def reset(self) -> None:
        """Cancel the old generation and wait for a new generation's first frame."""
        with self._condition:
            if self._closed:
                raise BridgeCancelled("BEHAVIOR bridge is closed")
            self.stop("reset", state="cancelled")
            self._generations.append(_Generation(len(self._generations)))
            self._observation_index = -1
            self._snapshot = None

    def stop(
        self,
        reason: str = "closed",
        *,
        state: Literal["closed", "cancelled", "ended"] = "closed",
    ) -> None:
        """Wake waiters after shutdown, cancellation or evaluator disconnection.

        Disconnection returns partial execution feedback. Cancellation raises
        instead; a later disconnection never hides an existing cancellation.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("session end reason must be a non-empty string")
        with self._condition:
            record = self._generations[-1]
            if state == "ended":
                if record.cancelled is not None or self._closed:
                    return
                if record.session_end_reason is None:
                    record.session_end_reason = reason
            else:
                if record.cancelled is None:
                    record.cancelled = reason
                self._closed = self._closed or state == "closed"
                self._receiver = None
            self._active_chunk = None
            self._inflight = None
            self._condition.notify_all()

    def session_status(self) -> dict[str, Any]:
        """Report connection state without claiming task success."""
        with self._condition:
            record = self._generations[-1]
            return {
                "ended": record.session_end_reason is not None,
                "reason": record.session_end_reason,
                "outcome": "unknown",
                "generation": record.number,
            }

    def get_observation(self) -> dict[str, Any]:
        """Read the current generation's latest frame, waiting for its first."""
        with self._condition:
            record = self._generations[-1]
            self._wait(
                record,
                lambda: self._observation_generation == record.number,
                time.monotonic() + self._simulation_timeout_s,
                "get_observation",
                self._simulation_timeout_s,
                allow_ended=True,
            )
            return self.last_obs

    def get_motion_snapshot(self) -> dict[str, Any]:
        """Copy one observation with its matching frame ID and control history."""
        with self._condition:
            record = self._generations[-1]
            self._wait(
                record,
                lambda: self._observation_generation == record.number,
                time.monotonic() + self._simulation_timeout_s,
                "get_motion_snapshot",
                self._simulation_timeout_s,
            )
            snapshot = deepcopy(self._snapshot)
            snapshot["motion_context"].update(
                issued_actions=record.issued,
                confirmed_actions=record.confirmed,
                unconfirmed_actions=record.issued - record.confirmed,
                actions_pending=self._active_chunk is not None,
            )
            return snapshot

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
        expected_observation_token: dict[str, int] | None = None,
    ) -> tuple:
        """Submit native actions and return only confirmed execution feedback."""
        array = validate_native_actions(actions, chunk=True).copy()
        deadline = time.monotonic() + self._chunk_timeout_s
        with self._condition:
            record = self._generations[-1]
            self._wait(
                record,
                lambda: True,
                deadline,
                "chunk_step",
                self._chunk_timeout_s,
                allow_ended=True,
            )
            if self._active_chunk is not None:
                raise RuntimeError("BEHAVIOR bridge already has an outstanding chunk")
            if expected_observation_token is not None:
                token = {
                    "generation": record.number,
                    "observation_index": self._observation_index,
                }
                if (
                    not isinstance(expected_observation_token, dict)
                    or set(expected_observation_token) != set(token)
                    or any(
                        type(v) is not int or v < 0
                        for v in expected_observation_token.values()
                    )
                ):
                    raise ValueError(
                        "expected_observation_token must identify one public frame"
                    )
                if (
                    expected_observation_token != token
                    or self._observation_generation != record.number
                ):
                    raise RuntimeError(
                        "stale_observation: public planning frame changed"
                    )
                if self._inflight is not None:
                    raise RuntimeError(
                        "unconfirmed_action: planning frame predates an action"
                    )
            chunk = _Chunk(array, return_all_frames)
            if record.session_end_reason is None:
                self._active_chunk = chunk
            self._condition.notify_all()
            while chunk.confirmed < len(array) and record.session_end_reason is None:
                progress = chunk.last_progress_at
                progress_deadline = progress + self._simulation_timeout_s
                inactivity_first = progress_deadline < deadline
                self._wait(
                    record,
                    lambda: (
                        chunk.last_progress_at != progress
                        or record.session_end_reason is not None
                    ),
                    min(deadline, progress_deadline),
                    "chunk_step without observation"
                    if inactivity_first
                    else "chunk_step",
                    self._simulation_timeout_s
                    if inactivity_first
                    else self._chunk_timeout_s,
                    allow_ended=True,
                )
            self._wait(
                record,
                lambda: self._observation_generation == record.number,
                deadline,
                "chunk_step",
                self._chunk_timeout_s,
                allow_ended=True,
            )
            observation = (
                chunk.frames
                if return_all_frames
                else (
                    chunk.final_observation
                    if chunk.final_observation is not None
                    else self.last_obs
                )
            )
            snapshot = chunk.final_motion_snapshot or self._snapshot
            return (
                observation,
                None,
                None,
                None,
                {
                    "bridge_generation": record.number,
                    "requested_actions": len(array),
                    "executed_actions": chunk.confirmed,
                    "issued_actions": chunk.issued,
                    "confirmed_actions": chunk.confirmed,
                    "unconfirmed_actions": chunk.issued - chunk.confirmed,
                    "stop_reason": "session_ended"
                    if record.session_end_reason is not None
                    else "chunk_completed",
                    "session_outcome": "unknown",
                    "last_known_observation_available": True,
                    "observation_after_last_issued_action": chunk.issued > 0
                    and chunk.confirmed == chunk.issued,
                    "observation_token": snapshot["observation_token"],
                    "motion_context": {
                        **deepcopy(snapshot["motion_context"]),
                        "issued_actions": record.issued,
                        "confirmed_actions": record.confirmed,
                        "unconfirmed_actions": record.issued - record.confirmed,
                        "actions_pending": self._active_chunk is not None,
                    },
                },
            )

    def observe_and_get_action(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Save an official frame, confirm the previous action and issue the next."""
        if not isinstance(raw_obs, dict):
            raise TypeError("BEHAVIOR evaluator observation must be a dictionary")
        deadline = time.monotonic() + self._action_timeout_s
        with self._condition:
            record = self._generations[-1]
            self._wait(
                record,
                lambda: True,
                deadline,
                "observe_and_get_action",
                self._action_timeout_s,
            )
            if self._receiver is not None:
                raise RuntimeError(
                    "BEHAVIOR bridge requires serialized evaluator calls"
                )
            receiver = object()
            self._receiver = receiver
            try:
                # Save the frame and confirm the action sent on the previous call.
                self.last_obs = raw_obs
                self._observation_generation = record.number
                if self._active_chunk is not None:
                    self._active_chunk.last_progress_at = time.monotonic()
                confirmed_chunk = self._inflight
                if confirmed_chunk is not None:
                    self._inflight = None
                    confirmed_chunk.confirmed += 1
                    record.confirmed += 1
                    confirmed_chunk.final_observation = raw_obs
                    if confirmed_chunk.return_all_frames:
                        confirmed_chunk.frames.append(raw_obs)
                    if confirmed_chunk.confirmed == len(confirmed_chunk.actions):
                        self._active_chunk = None

                # Keep the two adjacent public control frames needed by motion tools.
                previous = self._snapshot["motion_context"] if self._snapshot else None
                self._observation_index += 1
                token = {
                    "generation": record.number,
                    "observation_index": self._observation_index,
                }
                try:
                    proprio = extract_public_proprio(raw_obs).copy()
                except (ValueError, TypeError, UnicodeError):
                    proprio = None
                context = {
                    "source": "confirmed_public_control_observations",
                    "control_dt_s": 1 / 30,
                    "current_token": token,
                    "previous_token": deepcopy(previous["current_token"])
                    if previous
                    else None,
                    "current_proprio": proprio,
                    "previous_proprio": previous["current_proprio"].copy()
                    if previous is not None and previous["current_proprio"] is not None
                    else None,
                    "pair_confirmed": confirmed_chunk is not None
                    and previous is not None
                    and previous["current_token"]["generation"] == record.number
                    and previous["current_token"]["observation_index"]
                    == self._observation_index - 1
                    and record.issued == record.confirmed,
                    "initial_bootstrap": self._observation_index == 0
                    and record.issued == record.confirmed == 0
                    and confirmed_chunk is None,
                    "issued_actions": record.issued,
                    "confirmed_actions": record.confirmed,
                    "unconfirmed_actions": record.issued - record.confirmed,
                    "actions_pending": self._active_chunk is not None,
                }
                self._snapshot = {
                    "observation": raw_obs,
                    "observation_token": token,
                    "motion_context": context,
                }
                if confirmed_chunk is not None:
                    confirmed_chunk.final_motion_snapshot = self._snapshot
                self._condition.notify_all()

                # Wait for a tool's action batch, then return exactly one action.
                self._wait(
                    record,
                    lambda: self._active_chunk is not None,
                    deadline,
                    "observe_and_get_action",
                    self._action_timeout_s,
                )
                chunk = self._active_chunk
                action = chunk.actions[chunk.issued].copy()
                chunk.issued += 1
                record.issued += 1
                self._inflight = chunk
                return action
            finally:
                if self._receiver is receiver:
                    self._receiver = None

    def stats(self) -> dict[str, Any]:
        """Expose execution counters for diagnostics without advancing simulation."""
        with self._condition:
            record = self._generations[-1]
            issued = sum(item.issued for item in self._generations)
            confirmed = sum(item.confirmed for item in self._generations)
            return {
                "generation": record.number,
                "issued": issued,
                "confirmed": confirmed,
                "unconfirmed": issued - confirmed,
                "closed": self._closed,
                "session_ended": record.session_end_reason is not None,
                "session_end_reason": record.session_end_reason,
                "last_observation_generation": self._observation_generation,
                "history": [
                    {
                        "generation": item.number,
                        "issued": item.issued,
                        "confirmed": item.confirmed,
                        "unconfirmed": item.issued - item.confirmed,
                        "cancel_reason": item.cancelled,
                        "session_end_reason": item.session_end_reason,
                    }
                    for item in self._generations
                ],
            }


def pack_data(value: Any) -> Any:
    """Encode NumPy values using the official evaluator's binary field names."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype for msgpack: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def unpack_data(value: dict) -> Any:
    """Restore official NumPy fields from a decoded MessagePack dictionary."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


@asynccontextmanager
async def serve_websocket(
    bridge: EvaluatorStepBridge,
    *,
    port: int,
    task_id: int,
    host: str = "127.0.0.1",
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """Listen for one official evaluator; finish worker calls before releasing it."""
    import msgpack
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed

    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise ValueError("task_id must be a non-negative integer")
    active_session = None
    handshake = {
        **(metadata or {}),
        "task_id": task_id,
        "single_session": True,
        "requires_reset": True,
    }
    handshake.pop("chunk_protocol", None)

    async def handle_connection(websocket):
        nonlocal active_session
        if active_session is not None:
            await websocket.close(code=1013, reason="Policy session is busy")
            return
        active_session = websocket
        disconnected = asyncio.create_task(websocket.wait_closed())
        work = None
        packer = msgpack.Packer(default=pack_data)
        reset_received = False
        try:
            await websocket.send(packer.pack(handshake))
            while True:
                request = msgpack.unpackb(
                    await websocket.recv(),
                    object_hook=unpack_data,
                    strict_map_key=False,
                )
                if not isinstance(request, dict):
                    raise ValueError("Policy request must be a mapping")
                started = time.monotonic()
                resetting = "reset" in request
                if resetting:
                    if request["reset"] is not True:
                        raise ValueError("Planner reset must be true")
                    if bridge.last_obs is not None:
                        bridge.stop()
                        raise RuntimeError(
                            "BEHAVIOR official policy supports one episode per process"
                        )
                    if bridge.session_status()["ended"]:
                        raise RuntimeError("BEHAVIOR policy session is closed")
                    work = asyncio.create_task(asyncio.to_thread(bridge.reset))
                else:
                    if not reset_received:
                        raise ValueError("Reset is required before requesting actions")
                    if request.get("type") == "infer_chunk":
                        raise ValueError(
                            "Planner accepts single-action observations only"
                        )
                    work = asyncio.create_task(
                        asyncio.to_thread(
                            bridge.observe_and_get_action, deepcopy(request)
                        )
                    )
                done, _ = await asyncio.wait(
                    (work, disconnected), return_when=asyncio.FIRST_COMPLETED
                )
                if disconnected in done:
                    break
                try:
                    result = await work
                except Exception:
                    if resetting:
                        bridge.stop()
                    raise
                if resetting:
                    reset_received = True
                    continue
                action = np.asarray(result, dtype=np.float32)
                if action.shape != (23,) or not np.isfinite(action).all():
                    raise ValueError(
                        f"Policy must return one finite [23] action, got {action.shape}"
                    )
                await websocket.send(
                    packer.pack(
                        {
                            "action": action,
                            "server_timing": {
                                "total_ms": (time.monotonic() - started) * 1000.0
                            },
                        }
                    )
                )
        except ConnectionClosed:
            logger.info("BEHAVIOR evaluator connection closed")
        except Exception:
            logger.exception("BEHAVIOR evaluator connection failed")
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            # Stopping the bridge releases a thread blocked in action/observation wait.
            # Keep the connection occupied until that thread has actually returned.
            if work is not None and resetting:
                await asyncio.gather(work, return_exceptions=True)
            bridge.stop("connection_closed", state="ended")
            if work is not None:
                await asyncio.gather(work, return_exceptions=True)
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)
            active_session = None

    async with serve(
        handle_connection,
        host,
        port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        yield server


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
