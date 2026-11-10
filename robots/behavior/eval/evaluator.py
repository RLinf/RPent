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

"""Adapt the official BEHAVIOR evaluator to RPent's environment interface.

The official evaluator owns simulation, reset and scoring. This module joins
its single-action WebSocket protocol to RPent tools that submit action chunks:

* ``WebsocketPolicyServer`` handles connections and NumPy message encoding.
* ``OfficialEvaluatorPolicy`` implements the evaluator's reset/act lifecycle.
* ``EvaluatorStepBridge`` hands observations/actions between policy and tools,
  confirming each action only when its next official observation arrives.

``create_official_policy_server`` binds the policy and connection lifecycle.
The Env service owns these objects; model inference remains in the VLA service.
MessagePack and WebSocket dependencies are imported only when serving messages.
"""

from __future__ import annotations

import asyncio
import copy
import http
import math
import random
import sys
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robots.behavior.env_client import validate_native_actions
from robots.behavior.tools import extract_public_proprio
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

logger = get_logger("behavior_policy_transport")


class BridgeCancelled(RuntimeError):
    """A bridge operation stopped before completion due to reset/close/timeout."""


class BridgeSessionEnded(BridgeCancelled):
    """The evaluator connection ended without reporting a task outcome."""


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
    """Coordinate one evaluator stream and one outstanding action chunk.

    Args:
        timeout_s: Maximum wait for each public blocking operation. A timeout
            cancels the current generation; the evaluator must reset before
            further work. This timeout must exceed expected model latency.
        simulation_timeout_s: Optional maximum interval without a new evaluator
            observation while executing a chunk. Each received observation
            refreshes this interval. Also bounds waiting for the first frame.
        action_timeout_s: Maximum time an evaluator observation waits for an
            RPent action. Defaults to ``timeout_s``.
        chunk_timeout_s: Absolute chunk deadline, including all progress. When
            omitted, legacy calls use ``timeout_s``; with simulation_timeout_s
            configured, it is ``number of actions * simulation_timeout_s``.
    """

    def __init__(
        self,
        timeout_s: float = 120.0,
        *,
        simulation_timeout_s: float | None = None,
        action_timeout_s: float | None = None,
        chunk_timeout_s: float | None = None,
    ) -> None:
        for name, value in (
            ("timeout_s", timeout_s),
            ("simulation_timeout_s", simulation_timeout_s),
            ("action_timeout_s", action_timeout_s),
            ("chunk_timeout_s", chunk_timeout_s),
        ):
            if value is None and name != "timeout_s":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        self._timeout_s = float(timeout_s)
        self._simulation_timeout_s = simulation_timeout_s or self._timeout_s
        self._action_timeout_s = action_timeout_s or self._timeout_s
        self._chunk_timeout_s = chunk_timeout_s
        self._progress_timeout_enabled = simulation_timeout_s is not None
        self._condition = threading.Condition()
        self._generations = [_Generation(0)]
        self._closed = False
        self._last_obs: dict[str, Any] | None = None
        self._observation_generation: int | None = None
        self._active_chunk: _Chunk | None = None
        self._inflight: _Chunk | None = None
        self._receiver: object | None = None
        self._observation_index = -1
        self._motion_context: dict[str, Any] | None = None

    @property
    def last_obs(self) -> dict[str, Any] | None:
        """Retain the latest received observation even after cancellation."""
        with self._condition:
            return self._last_obs

    def _check_generation(self, generation: int, *, allow_ended: bool = False) -> None:
        record = self._generations[generation]
        if record.cancelled is not None or self._closed:
            reason = record.cancelled or "closed"
            raise BridgeCancelled(f"BEHAVIOR bridge generation {generation}: {reason}")
        if record.session_end_reason is not None and not allow_ended:
            raise BridgeSessionEnded(
                f"BEHAVIOR bridge generation {generation}: "
                f"session ended ({record.session_end_reason}); outcome unknown"
            )

    def _cancel(self, reason: str) -> None:
        record = self._generations[-1]
        if record.cancelled is None:
            record.cancelled = reason
        self._active_chunk = None
        self._inflight = None
        self._receiver = None
        self._motion_context = None
        self._condition.notify_all()

    def _wait(
        self,
        deadline: float,
        generation: int,
        operation: str,
        timeout_s: float,
        *,
        allow_ended: bool = False,
    ) -> None:
        self._check_generation(generation, allow_ended=allow_ended)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._cancel(f"{operation} timed out after {timeout_s:g}s")
            self._check_generation(generation)
        self._condition.wait(remaining)

    def reset(self) -> None:
        """Cancel the previous episode; only the official owner calls this."""
        with self._condition:
            if self._closed:
                raise BridgeCancelled("BEHAVIOR bridge is closed")
            self._cancel("reset")
            self._generations.append(_Generation(len(self._generations)))
            self._observation_index = -1

    def close(self) -> None:
        """Cancel all waiters without claiming outstanding actions completed."""
        with self._condition:
            if not self._closed:
                self._cancel("closed")
                self._closed = True

    def end_session(self, reason: str = "connection_closed") -> None:
        """Release pending chunks when the evaluator connection actually ends.

        Connection closure reports neither task success nor action completion.
        Issued actions without a following observation remain unconfirmed. Use
        ``close`` instead for owner cancellation or faults; an existing fault
        is never converted into a normal session end by this method.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("session end reason must be a non-empty string")
        with self._condition:
            record = self._generations[-1]
            if record.cancelled is not None or self._closed:
                return
            if record.session_end_reason is None:
                record.session_end_reason = reason
                self._active_chunk = None
                self._inflight = None
                self._condition.notify_all()

    def session_status(self) -> dict[str, Any]:
        """Return transport lifecycle only; official task outcome is unknown."""
        with self._condition:
            record = self._generations[-1]
            return {
                "ended": record.session_end_reason is not None,
                "reason": record.session_end_reason,
                "outcome": "unknown",
                "generation": record.number,
            }

    def get_observation(self) -> dict[str, Any]:
        """Read this generation's latest frame, including after session end.

        The retained frame is not evidence that the last issued action executed.
        ``session_status`` and chunk feedback describe that distinction. Frames
        from a prior reset generation are never returned.
        """
        deadline = time.monotonic() + self._simulation_timeout_s
        with self._condition:
            generation = self._generations[-1].number
            self._check_generation(generation, allow_ended=True)
            while self._observation_generation != generation:
                self._wait(
                    deadline, generation, "get_observation", self._simulation_timeout_s
                )
            self._check_generation(generation, allow_ended=True)
            return self._last_obs

    def _motion_snapshot(self, *, copy_observation: bool = False) -> dict[str, Any]:
        """Build a snapshot while holding the observation/action condition."""
        record = self._generations[-1]
        context = deepcopy(self._motion_context or {})
        context.update(
            issued_actions=record.issued,
            confirmed_actions=record.confirmed,
            unconfirmed_actions=record.issued - record.confirmed,
            actions_pending=self._active_chunk is not None,
        )
        return {
            "observation": deepcopy(self._last_obs)
            if copy_observation
            else self._last_obs,
            "observation_token": {
                "generation": record.number,
                "observation_index": self._observation_index,
            },
            "motion_context": context,
        }

    def get_motion_snapshot(self) -> dict[str, Any]:
        """Atomically bind public observation, control history and generation.

        Reading never creates another control frame. Only evaluator callbacks
        may extend the finite-difference history; no simulator state is read.
        """
        deadline = time.monotonic() + self._simulation_timeout_s
        with self._condition:
            generation = self._generations[-1].number
            self._check_generation(generation)
            while self._observation_generation != generation:
                self._wait(
                    deadline,
                    generation,
                    "get_motion_snapshot",
                    self._simulation_timeout_s,
                )
            self._check_generation(generation)
            return self._motion_snapshot(copy_observation=True)

    def _check_observation_token(self, expected: Any) -> None:
        if expected is None:
            return
        keys = {"generation", "observation_index"}
        if (
            not isinstance(expected, dict)
            or set(expected) != keys
            or any(type(expected[key]) is not int for key in keys)
            or any(expected[key] < 0 for key in keys)
        ):
            raise ValueError(
                "expected_observation_token must identify one public frame"
            )
        current = {
            "generation": self._generations[-1].number,
            "observation_index": self._observation_index,
        }
        if expected != current or self._observation_generation != current["generation"]:
            raise RuntimeError("stale_observation: public planning frame changed")
        if self._inflight is not None:
            raise RuntimeError("unconfirmed_action: planning frame predates an action")

    def _record_motion_observation(
        self, raw_obs: dict[str, Any], *, confirmed_action: bool
    ) -> None:
        """Capture a public61 pair before another action may be issued."""
        previous = self._motion_context
        self._observation_index += 1
        record = self._generations[-1]
        token = {
            "generation": record.number,
            "observation_index": self._observation_index,
        }
        try:
            state = extract_public_proprio(raw_obs)
        except (ValueError, TypeError, UnicodeError):
            state = None
        pair_confirmed = (
            confirmed_action
            and previous is not None
            and previous["current_token"]["generation"] == record.number
            and previous["current_token"]["observation_index"]
            == self._observation_index - 1
            and record.issued == record.confirmed
        )
        self._motion_context = {
            "source": "confirmed_public_control_observations",
            "control_dt_s": 1 / 30,
            "current_token": token,
            "previous_token": deepcopy(previous["current_token"]) if previous else None,
            "current_proprio": state,
            "previous_proprio": (
                previous["current_proprio"].copy()
                if previous is not None and previous["current_proprio"] is not None
                else None
            ),
            "pair_confirmed": pair_confirmed,
            "initial_bootstrap": (
                self._observation_index == 0
                and record.issued == record.confirmed == 0
                and not confirmed_action
            ),
        }

    def step(self, action: Any) -> tuple:
        """Issue one native action and wait for a subsequent observation."""
        array = validate_native_actions(action, chunk=False)
        return self.chunk_step(array[None, :])

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool = False,
        expected_observation_token: dict[str, int] | None = None,
    ) -> tuple:
        """Wait until every action is confirmed by the next evaluator observation.

        Returns:
            A Gym 5-tuple. Reward and terminal flags remain None because the
            policy observation protocol does not supply them. ``executed_actions``
            counts only actions acknowledged by a subsequent observation.
            Session end returns partial counts and the last known observation,
            explicitly marked when it predates the last issued action. With
            ``return_all_frames``, only confirmed frames are returned, so that
            list can be empty on session end.
        """
        array = validate_native_actions(actions, chunk=True).copy()
        timeout_s = self._chunk_timeout_s
        if timeout_s is None:
            timeout_s = (
                len(array) * self._simulation_timeout_s
                if self._progress_timeout_enabled
                else self._timeout_s
            )
        deadline = time.monotonic() + timeout_s
        with self._condition:
            generation = self._generations[-1].number
            self._check_generation(generation, allow_ended=True)
            record = self._generations[generation]
            if self._active_chunk is not None:
                raise RuntimeError("BEHAVIOR bridge already has an outstanding chunk")
            self._check_observation_token(expected_observation_token)
            chunk = _Chunk(array, return_all_frames)
            if record.session_end_reason is None:
                self._active_chunk = chunk
            self._condition.notify_all()
            while chunk.confirmed < len(array):
                self._check_generation(generation, allow_ended=True)
                if record.session_end_reason is not None:
                    break
                progress_deadline = chunk.last_progress_at + self._simulation_timeout_s
                wait_for_progress = progress_deadline < deadline
                self._wait(
                    min(deadline, progress_deadline),
                    generation,
                    "chunk_step without observation"
                    if wait_for_progress
                    else "chunk_step",
                    self._simulation_timeout_s if wait_for_progress else timeout_s,
                    allow_ended=True,
                )
            self._check_generation(generation, allow_ended=True)
            has_observation = self._observation_generation == generation
            if not has_observation:
                self._check_generation(generation)
            observation = (
                chunk.frames
                if return_all_frames
                else (
                    chunk.final_observation
                    if chunk.final_observation is not None
                    else self._last_obs
                )
            )
            motion_snapshot = chunk.final_motion_snapshot or self._motion_snapshot()
            return (
                observation,
                None,
                None,
                None,
                {
                    "bridge_generation": generation,
                    "requested_actions": len(array),
                    "executed_actions": chunk.confirmed,
                    "issued_actions": chunk.issued,
                    "confirmed_actions": chunk.confirmed,
                    "unconfirmed_actions": chunk.issued - chunk.confirmed,
                    "stop_reason": (
                        "session_ended"
                        if record.session_end_reason is not None
                        else "chunk_completed"
                    ),
                    "session_outcome": "unknown",
                    "last_known_observation_available": has_observation,
                    "observation_after_last_issued_action": (
                        chunk.issued > 0 and chunk.confirmed == chunk.issued
                    ),
                    "observation_token": motion_snapshot["observation_token"],
                    "motion_context": {
                        **motion_snapshot["motion_context"],
                        "unconfirmed_actions": record.issued - record.confirmed,
                        "issued_actions": record.issued,
                        "confirmed_actions": record.confirmed,
                        "actions_pending": self._active_chunk is not None,
                    },
                },
            )

    def observe_and_get_action(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Accept one evaluator observation and wait for the next native action.

        The evaluator must serialize calls and invoke ``reset`` before the
        first observation of a new episode. Call this directly on the bridge,
        outside the Env facade lock held by a waiting ``chunk_step``.
        """
        if not isinstance(raw_obs, dict):
            raise TypeError("BEHAVIOR evaluator observation must be a dictionary")
        deadline = time.monotonic() + self._action_timeout_s
        with self._condition:
            record = self._generations[-1]
            generation = record.number
            self._check_generation(generation)
            if self._receiver is not None:
                raise RuntimeError(
                    "BEHAVIOR bridge requires serialized evaluator calls"
                )
            receiver = object()
            self._receiver = receiver
            try:
                self._last_obs = raw_obs
                self._observation_generation = generation
                if self._active_chunk is not None:
                    self._active_chunk.last_progress_at = time.monotonic()
                confirmed_chunk = self._inflight
                if confirmed_chunk is not None:
                    chunk = self._inflight
                    self._inflight = None
                    chunk.confirmed += 1
                    record.confirmed += 1
                    chunk.final_observation = raw_obs
                    if chunk.return_all_frames:
                        chunk.frames.append(raw_obs)
                    if chunk.confirmed == len(chunk.actions):
                        self._active_chunk = None
                self._record_motion_observation(
                    raw_obs, confirmed_action=confirmed_chunk is not None
                )
                if confirmed_chunk is not None:
                    confirmed_chunk.final_motion_snapshot = self._motion_snapshot()
                self._condition.notify_all()
                while True:
                    self._check_generation(generation)
                    chunk = self._active_chunk
                    if chunk is not None and chunk.issued < len(chunk.actions):
                        action = chunk.actions[chunk.issued].copy()
                        chunk.issued += 1
                        record.issued += 1
                        self._inflight = chunk
                        return action
                    self._wait(
                        deadline,
                        generation,
                        "observe_and_get_action",
                        self._action_timeout_s,
                    )
            finally:
                if self._receiver is receiver:
                    self._receiver = None

    def stats(self) -> dict[str, Any]:
        """Return cumulative and per-generation counts, retaining cancellations."""
        with self._condition:
            history = [
                {
                    "generation": record.number,
                    "issued": record.issued,
                    "confirmed": record.confirmed,
                    "unconfirmed": record.issued - record.confirmed,
                    "cancel_reason": record.cancelled,
                    "session_end_reason": record.session_end_reason,
                }
                for record in self._generations
            ]
            issued = sum(record.issued for record in self._generations)
            confirmed = sum(record.confirmed for record in self._generations)
            return {
                "generation": self._generations[-1].number,
                "issued": issued,
                "confirmed": confirmed,
                "unconfirmed": issued - confirmed,
                "closed": self._closed,
                "session_ended": self._generations[-1].session_end_reason is not None,
                "session_end_reason": self._generations[-1].session_end_reason,
                "last_observation_generation": self._observation_generation,
                "history": history,
            }


class OfficialEvaluatorPolicy:
    """Implement the official policy protocol for one episode per process.

    Args:
        bridge: Borrowed bridge shared with the internal RPent Env facade.
        task_id: Configured official task identity.
        reset_policy: Optional callback to reset the borrowed VLA policy before
            the first observation. It must serialize against model inference.
    """

    single_session = True
    requires_reset = True
    supports_chunk = False

    def __init__(
        self,
        bridge: EvaluatorStepBridge,
        *,
        task_id: int,
        reset_policy: Callable[[], None] | None = None,
    ) -> None:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        self.bridge = bridge
        self.task_id = task_id
        self._reset_policy = reset_policy
        self._state_lock = threading.Lock()
        self._act_lock = threading.Lock()
        self._observations = 0
        self._closed = False
        self._resetting = False

    def reset(self) -> None:
        """Prepare the first episode; a later reset cancels and rejects reuse."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("BEHAVIOR policy session is closed")
            if self._resetting:
                raise RuntimeError("BEHAVIOR policy reset is already in progress")
            if self._observations:
                self.bridge.close()
                raise RuntimeError(
                    "BEHAVIOR official policy supports one episode per process"
                )
            self._resetting = True
        try:
            if self._reset_policy is not None:
                self._reset_policy()
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("BEHAVIOR policy session closed during reset")
                self.bridge.reset()
        except Exception:
            with self._state_lock:
                if not self._closed:
                    self.bridge.close()
            raise
        finally:
            with self._state_lock:
                self._resetting = False

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Accept one raw observation and issue a single native 23D action."""
        if not isinstance(observation, dict):
            raise TypeError("BEHAVIOR evaluator observation must be a dictionary")
        if not self._act_lock.acquire(blocking=False):
            raise RuntimeError("BEHAVIOR policy requires serialized evaluator calls")
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("BEHAVIOR policy session is closed")
                if self._resetting:
                    raise RuntimeError("BEHAVIOR policy reset is in progress")
                self._observations += 1
            return self.bridge.observe_and_get_action(observation)
        finally:
            self._act_lock.release()

    def close(self, reason: str = "connection_closed") -> None:
        """Mark real connection closure; leave task outcome and execution unknown."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self.bridge.end_session(reason)


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
    """Restore evaluator arrays without importing Torch or the simulator."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def Packer(*args: Any, **kwargs: Any) -> Any:
    """Create the official NumPy MessagePack encoder when transport is used."""
    import msgpack

    return msgpack.Packer(*args, **{"default": pack_data, **kwargs})


def unpackb(*args: Any, **kwargs: Any) -> Any:
    """Decode official messages without importing transport during discovery."""
    import msgpack

    return msgpack.unpackb(*args, **{"object_hook": unpack_data, **kwargs})


class WebsocketPolicyServer:
    """Serve one reset-required evaluator session using native 23D actions.

    Args:
        policy: Object exposing ``reset()`` and ``act(observation)``. The caller
            owns policy shutdown and wakes blocked policy calls on disconnect.
        host: Interface on which to listen.
        port: WebSocket/health-check port.
        metadata: Additional fields sent once immediately after connection.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "127.0.0.1",
        port: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = {
            **(metadata or {}),
            "single_session": True,
            "requires_reset": True,
        }
        self.metadata.pop("chunk_protocol", None)
        self._active_session: ServerConnection | None = None
        self._policy_busy = threading.Event()

    async def run(self) -> None:
        """Serve official policy requests and the evaluator's readiness check."""
        from websockets.asyncio.server import serve

        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _call_policy(self, function: Callable[..., Any], *args: Any) -> Any:
        self._policy_busy.set()

        def run() -> Any:
            try:
                return function(*args)
            finally:
                self._policy_busy.clear()

        job = asyncio.create_task(asyncio.to_thread(run))

        def consume_exception(task: asyncio.Task) -> None:
            if not task.cancelled():
                task.exception()

        job.add_done_callback(consume_exception)
        # Cancellation cannot stop a worker thread; do not admit a new session
        # until the old policy call has actually returned.
        return await asyncio.shield(job)

    async def _handler(self, websocket: ServerConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        if self._active_session is not None or self._policy_busy.is_set():
            await websocket.close(code=1013, reason="Policy session is busy")
            return
        self._active_session = websocket
        try:
            await self._serve_connection(websocket)
        except ConnectionClosed:
            logger.info("BEHAVIOR evaluator connection closed")
        except Exception:
            logger.exception("BEHAVIOR evaluator connection failed")
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            if self._active_session is websocket:
                self._active_session = None

    async def _serve_connection(self, websocket: ServerConnection) -> None:
        packer = Packer()
        await websocket.send(packer.pack(self.metadata))
        reset_received = False
        while True:
            request = unpackb(await websocket.recv(), strict_map_key=False)
            if not isinstance(request, dict):
                raise ValueError("Policy request must be a mapping")
            started = time.monotonic()
            if "reset" in request:
                if request["reset"] is not True:
                    raise ValueError("Planner reset must be true")
                seed = request.get("seed")
                if "seed" in request and (
                    type(seed) is not int or not 0 <= seed < 2**32
                ):
                    raise ValueError("reset seed must be an integer in [0, 2**32)")
                await self._call_policy(self.policy.reset)
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    # Preserve a loaded caller's RNG without making Torch an
                    # Env-service dependency. VLA inference has its own seed.
                    torch = sys.modules.get("torch")
                    if torch is not None:
                        torch.manual_seed(seed)
                reset_received = True
                continue  # The official reset request has no response frame.
            if not reset_received:
                raise ValueError("Reset is required before requesting actions")
            if request.get("type") == "infer_chunk":
                raise ValueError("Planner accepts single-action observations only")
            action = await self._call_policy(self.policy.act, copy.deepcopy(request))
            action = np.asarray(action, dtype=np.float32)
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


def _health_check(connection: ServerConnection, request: Any) -> Any:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def create_official_policy_server(
    policy: OfficialEvaluatorPolicy,
    *,
    host: str = "127.0.0.1",
    port: int,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Build the optional official WebSocket server; run with ``await server.run()``.

    Install the lightweight ``behavior`` extra for WebSocket dependencies.
    Ordinary imports do not require the simulator or model runtime.
    Every connection exit releases tool waiters with an unknown task outcome,
    including a disconnect caused by transport or evaluator failure.
    """

    class SessionServer(WebsocketPolicyServer):
        async def _serve_connection(self, websocket: Any) -> None:
            connection = asyncio.create_task(super()._serve_connection(websocket))
            closed = asyncio.create_task(websocket.wait_closed())
            try:
                completed, _ = await asyncio.wait(
                    (connection, closed), return_when=asyncio.FIRST_COMPLETED
                )
                if closed in completed:
                    await closed
                    policy.close()
                    if not connection.done():
                        connection.cancel()
                        return
                await connection
            finally:
                policy.close()
                # The official server shields its policy worker thread. Ending
                # the bridge first wakes that worker even if the connection
                # coroutine is cancelled while awaiting its action.
                for task in (connection, closed):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(connection, closed, return_exceptions=True)

    return SessionServer(
        policy,
        host=host,
        port=port,
        metadata={
            **(metadata or {}),
            "task_id": policy.task_id,
            "single_session": True,
        },
    )
