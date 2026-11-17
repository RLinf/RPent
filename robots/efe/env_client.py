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

"""EFE standard-mode robot client.

The task container ships ``rcb_client.py`` as its only route to the
simulator: observations, action chunks, status, and kinematics. This module
wraps that client with the checks RPent relies on before a command can move
the robot. The service clips out-of-range targets instead of rejecting them;
this client rejects them first so a tool call never reports a command the
robot did not receive.

The adapter keeps the protocol's failure rules. It never retries a call that
may already have executed, because a lost response does not prove the service
skipped the action. After a connection error the caller must call
``wait_ready`` on the wrapped client and read ``status`` before acting again.

The wrapped client publishes ``spec``, ``get_obs``, ``submit``, ``status``,
``wait_ready``, and ``ik``. It publishes no reset or finish method, so the
adapter relies on two further attributes: it calls ``/reset`` through the RCB1
helper ``_post(path, meta, parts, timeout)``, the same path ``submit`` takes,
and builds ``/finish`` from the client's base URL (``base``), falling back to
the ``base_url`` it was constructed with. Both live behind one adapter method
each, so a change in either contract fails in one place.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np

# The protocol fixes the longest interpolation step per row; the service spec
# carries no field for it.
_MAX_DT_S = 0.5
_DONE_ACTION = "done"
# Versions this adapter was written and run against. ``2.1`` and ``2.3`` publish
# ``control_dt`` instead of ``control_hz`` and ship a different ``rcb_client``
# (``get_obs(include_depth=...)``, ``Result.depth``, no ``wait_ready``), so they
# need their own adapter rather than a silent fallback here.
_SUPPORTED_PROTOCOLS = ("eai-standard/2.2", "eai-standard/2.4")
# Reset and finish restart or end a whole episode; they get the protocol's
# longer operation timeout rather than the request timeout.
_SERVICE_POST_TIMEOUT_S = 600


class EfeEnvClient:
    """Validated access to one EFE episode.

    Args:
        base_url: Service URL. The service client probes ``sim:8000`` and then
            ``localhost:8000`` when this is unset. The resolved endpoint is
            kept for ``/finish``.
        connect_timeout_s: How long to wait for a simulator that is still
            starting. This wait counts against the episode's wall clock.
        robot: An existing ``rcb_client.Robot`` for tests. Do not combine it
            with ``base_url``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        connect_timeout_s: float = 3600.0,
        deadline: float | None = None,
        robot: Any | None = None,
    ) -> None:
        if robot is not None:
            if base_url is not None:
                raise ValueError("Pass either base_url or robot, not both.")
        started_at = time.monotonic()
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite.")
        if deadline is not None and deadline <= started_at:
            raise TimeoutError("The EFE trial deadline expired before connection.")
        if deadline is not None:
            connect_timeout_s = min(connect_timeout_s, deadline - started_at)
        if connect_timeout_s <= 0 or not math.isfinite(connect_timeout_s):
            raise ValueError("connect_timeout_s must be positive and finite.")
        if robot is None:
            from rcb_client import Robot

            robot = Robot(base_url, timeout=connect_timeout_s)
        self._robot = robot
        self._deadline = deadline
        # rcb_client exposes no public reset/finish endpoint or request-timeout
        # hook. Keep its private RCB1 post method as the narrow adapter point.
        self._robot_post = robot._post
        # Its action methods use fixed 600–1800 s HTTP read timeouts. Clamp
        # those requests too, so a nearly expired trial cannot keep executing
        # a long chunk after RPent has handed off.
        robot._post = self._post_with_deadline
        client_base = robot.base
        resolved_base = client_base or base_url
        self._base_url = resolved_base.rstrip("/") if resolved_base else None
        self._spec = robot.spec()
        self._validate_spec(self._spec)
        self._done = False
        self.done_reason: str | None = None
        self.last_meta: dict[str, Any] = {}
        self.submit_calls = 0
        self.observe_calls = 0
        self.control_steps = 0
        self.reset_calls = 0
        self.finish_calls = 0
        self._resets_left: int | None = None

    # -- read-only views -------------------------------------------------

    @property
    def spec(self) -> dict[str, Any]:
        """Return the service spec for the running task."""
        return self._spec

    @property
    def done(self) -> bool:
        """Whether a response reported the end of the episode."""
        return self._done

    @property
    def reset_budget(self) -> int:
        """Return how many resets the task grants; zero when it grants none."""
        return int((self._spec.get("budget") or {}).get("reset") or 0)

    @property
    def resets_left(self) -> int:
        """Return the service-reported reset allowance when available."""
        if self._resets_left is not None:
            return self._resets_left
        return max(self.reset_budget - self.reset_calls, 0)

    def set_deadline(self, deadline: float | None) -> None:
        """Set the monotonic deadline for service calls in this trial."""
        self._deadline = deadline

    @property
    def remaining_s(self) -> float | None:
        """Seconds remaining in the trial, or ``None`` when unbounded."""
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    @property
    def success_hidden(self) -> bool:
        """Whether the service hides success and leaves the end to the agent.

        The service spells the flag out: ``/spec`` returns
        ``"hidden: the service never tells you whether you have succeeded..."``,
        while only the grader's private ``final.json`` holds the bare
        ``"hidden"``. Match the prefix, not the exact string.
        """
        signal = self._spec.get("success_signal")
        return isinstance(signal, str) and (
            signal == "hidden" or signal.startswith("hidden:")
        )

    @property
    def control_hz(self) -> float:
        """Return the control rate the service executes chunks at."""
        return float(self._spec["control_hz"])

    @property
    def control_period_s(self) -> float:
        """Return the duration of one control step."""
        return 1.0 / self.control_hz

    # -- calls -----------------------------------------------------------

    def observe(
        self,
        cameras: list[str] | None = None,
        *,
        depth: bool = False,
        video: str = "frames",
    ) -> Any:
        """Return the current frames and proprioception without advancing time.

        The protocol's ``seconds`` argument always stays 0 here: a nonzero value
        holds the last command and moves the robot, which a read must not do.

        Args:
            cameras: Camera names from ``spec["cameras"]``. Default: the
                task's default cameras.
            depth: Request the last frame's depth map for each camera.
            video: ``"frames"`` (default), ``"mp4"``, ``"both"``, or ``"none"``.
                The service always captures the last frame of each camera;
                ``"none"`` omits the per-frame arrays, so a read that needs
                only proprioception returns a smaller payload.
        """
        self._raise_if_done("observe")
        cameras = self._validated_cameras(cameras)
        result = self._bounded_call(
            self._robot.get_obs,
            cameras=cameras,
            seconds=0,
            depth=True if depth else None,
            video=video,
        )
        self.observe_calls += 1
        self._record_state(result.meta)
        return result

    def submit(
        self,
        action_type: str,
        actions: Any,
        *,
        dt: float | None = None,
        cameras: list[str] | None = None,
        video: str = "none",
    ) -> Any:
        """Execute one validated action chunk.

        Args:
            action_type: A key of ``spec["action_types"]``.
            actions: A ``(T, dim)`` array in ``layout`` order. A 1-D array is
                one row. ``done`` takes an empty array.
            dt: Time between rows. Default: one control period. Must be a
                whole number of control periods, at most 0.5 s, and the chunk
                must not exceed ``spec["max_chunk_s"]``.
            cameras: Cameras to render for this chunk.
            video: ``"none"``, ``"mp4"``, or ``"frames"``. Rendering costs CPU,
                so the default returns no images.
        """
        self._raise_if_done("submit")
        cameras = self._validated_cameras(cameras)
        action_spec = self._action_spec(action_type)
        rows = self._validate_actions(action_type, action_spec, actions)
        resolved_dt = (
            None if action_type == _DONE_ACTION else self._resolve_dt(dt, len(rows))
        )
        result = self._bounded_call(
            self._robot.submit,
            action_type,
            rows,
            dt=resolved_dt,
            cameras=cameras,
            video=video,
        )
        self.submit_calls += 1
        self.control_steps += int(result.meta.get("control_steps_chunk") or 0)
        self._record_state(result.meta)
        return result

    def status(self) -> dict[str, Any]:
        """Return the service's call, step, and episode counters."""
        status = self._bounded_call(self._robot.status)
        # Shell code can call rcb_client directly in Codex's full-access mode.
        # The service snapshot must replace our local state, including a prior
        # done=True after an external reset.
        self._record_state(status, authoritative=True)
        return status

    def wait_ready(self, timeout_s: float = 600.0) -> dict[str, Any]:
        """Wait for a crashed service to resume the same episode, then report status.

        The service resumes in the state after the last call it completed. That
        call may have executed even if its response was lost, so the caller must
        read the status before deciding what to send next.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        remaining = self.remaining_s
        if remaining is not None:
            if remaining <= 0:
                raise TimeoutError(
                    "The trial budget is exhausted; no further service call can run."
                )
            timeout_s = min(timeout_s, remaining)
        status = self._bounded_call(self._robot.wait_ready, timeout=timeout_s)
        self._record_state(status)
        return status

    def reset(self, *, cameras: list[str] | None = None) -> dict[str, Any]:
        """Abandon the running episode and start a new one from the frozen state.

        Only a task that declares a reset budget exposes ``/reset``, and the
        service refuses it after success, agent finish, or when the budget is
        spent. Only the last episode is graded.
        """
        if self.reset_budget <= 0:
            raise ValueError("This task declares no reset budget")
        if self._done and not self._reset_allowed_after_end():
            raise RuntimeError(
                f"Cannot reset: the episode ended "
                f"({self.done_reason or 'no reason reported'})"
            )
        cameras = self._validated_cameras(cameras)
        meta: dict[str, Any] = {"video": "none"}
        if cameras is not None:
            meta["cameras"] = list(cameras)
        result = self._bounded_call(
            self._service_post, "/reset", meta, [], _SERVICE_POST_TIMEOUT_S
        )
        self.reset_calls += 1
        self._done = False
        self.done_reason = None
        self.last_meta = {}
        self._record_state(result.meta)
        return dict(result.meta)

    def finish(self) -> dict[str, Any]:
        """End the episode now (``POST /finish``) and return the service status.

        Only a task with agent-ended episodes accepts it: the service hides
        success, so the agent must say when its episode is over. The verifier
        grades the state the episode ended in; ending before the goal is a
        failure.
        """
        if not self.success_hidden:
            raise ValueError("This task does not use agent-ended episodes")
        if self._done:
            raise RuntimeError(
                f"Cannot finish: the episode ended "
                f"({self.done_reason or 'no reason reported'})"
            )
        if self._base_url is None:
            raise RuntimeError("The service client has no base URL")
        remaining = self.remaining_s
        timeout_s = 60.0 if remaining is None else min(60.0, remaining)
        status = self._bounded_call(
            self._post_json, f"{self._base_url}/finish", {}, timeout_s
        )
        self.finish_calls += 1
        self._done = True
        self.done_reason = status.get("reason") or "finished"
        self._record_state(status)
        return status

    def ik(
        self,
        targets: dict[str, list[float]],
        *,
        seed: list[float] | None = None,
    ) -> dict[str, Any]:
        """Return arm joint positions that place tool frames at ``targets``.

        IK is stateless: it never moves the robot and never advances time, so a
        target can be checked before any chunk is submitted.

        Args:
            targets: Tool frame name to ``[x, y, z, qw, qx, qy, qz]`` in the
                world frame.
            seed: Joint positions in ``spec["kinematics"]["joint_names"]``
                order. Default: the service's current position.
        """
        kinematics = self._spec.get("kinematics")
        if not kinematics:
            raise ValueError("The service exposes no kinematics")
        known = set(kinematics["ik_targets"])
        unknown = sorted(set(targets) - known)
        if unknown:
            raise ValueError(f"Unknown IK targets: {unknown}. Known: {sorted(known)}")
        for frame, pose in targets.items():
            array = np.asarray(pose, dtype=np.float64)
            if array.shape != (7,):
                raise ValueError(f"IK target {frame} needs 7 values, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"IK target {frame} has non-finite values")
        joint_names = kinematics.get("joint_names") or []
        if seed is not None:
            seed_array = np.asarray(seed, dtype=np.float64)
            if seed_array.ndim != 1 or seed_array.shape[0] != len(joint_names):
                raise ValueError(
                    f"IK seed needs {len(joint_names)} joint positions, "
                    f"got {seed_array.shape}"
                )
            if not np.isfinite(seed_array).all():
                raise ValueError("IK seed has non-finite values")
        return self._bounded_call(
            self._robot.ik, targets, seed=None if seed is None else list(seed)
        )

    def _bounded_call(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Stop waiting for a service call when the trial deadline expires.

        Requests that time out may still have reached the service. A timed-out
        write is therefore uncertain and must not be retried automatically.
        """
        remaining = self.remaining_s
        if remaining is None:
            return function(*args, **kwargs)
        if remaining <= 0:
            raise TimeoutError(
                "The trial budget is exhausted; no further service call can run."
            )

        outcome: dict[str, Any] = {}

        def invoke() -> None:
            try:
                outcome["value"] = function(*args, **kwargs)
            except BaseException as exc:  # re-raised in the calling thread
                outcome["error"] = exc

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            raise TimeoutError(
                "The service call exceeded the remaining trial budget "
                f"({remaining:.1f} s)."
            )
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    # -- validation ------------------------------------------------------

    def _service_post(
        self,
        path: str,
        meta: dict[str, Any],
        parts: list[Any],
        timeout_s: float | None = None,
    ) -> Any:
        """POST through the client's RCB1 helper and return its result.

        ``rcb_client.Robot`` has no public reset or finish method. Reusing its
        ``_post`` keeps one dependency instead of two: the same helper that
        carries ``submit`` also carries these endpoint calls.
        """
        timeout = _SERVICE_POST_TIMEOUT_S if timeout_s is None else timeout_s
        remaining = self.remaining_s
        if remaining is not None:
            if remaining <= 0:
                raise TimeoutError(
                    "The trial budget is exhausted; no further service call can run."
                )
            timeout = min(timeout, remaining)
        return self._robot_post(path, meta, parts, timeout)

    def _post_with_deadline(
        self, path: str, meta: dict[str, Any], parts: list[Any], timeout: float
    ) -> Any:
        """Limit the wrapped RCB1 request to the remaining trial time."""
        remaining = self.remaining_s
        if remaining is not None:
            if remaining <= 0:
                raise TimeoutError(
                    "The trial budget is exhausted; no further service call can run."
                )
            timeout = min(timeout, remaining)
        return self._robot_post(path, meta, parts, timeout)

    @staticmethod
    def _post_json(
        url: str, payload: dict[str, Any], timeout_s: float = 60.0
    ) -> dict[str, Any]:
        """POST a JSON body and return the decoded response.

        ``/reset`` uses the client's RCB1 helper, while ``/finish`` takes a
        JSON body. Keep the two requests distinct until the service contract
        is verified; switching this endpoint to RCB1 would be a guess.
        """
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"POST {url} failed with {exc.code}: {detail}") from exc

    def _validated_cameras(self, cameras: list[str] | None) -> list[str] | None:
        """Return the camera list after rejecting names the spec does not list."""
        if cameras is None:
            return None
        unknown = [name for name in cameras if name not in self._spec["cameras"]]
        if unknown:
            raise ValueError(
                f"Unknown cameras: {unknown}. Known: {sorted(self._spec['cameras'])}"
            )
        return list(cameras)

    @staticmethod
    def _validate_spec(spec: dict[str, Any]) -> None:
        protocol = spec.get("protocol")
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported EFE service protocol {protocol!r}; this adapter "
                f"handles {list(_SUPPORTED_PROTOCOLS)}. ``2.1`` and ``2.3`` "
                "publish ``control_dt`` and a different ``rcb_client``."
            )
        for key in ("action_types", "cameras", "control_hz", "max_chunk_s"):
            if key not in spec:
                raise ValueError(f"The service spec has no {key!r}: {sorted(spec)}")
        if float(spec["control_hz"]) <= 0:
            raise ValueError("control_hz must be positive")
        if float(spec["max_chunk_s"]) <= 0:
            raise ValueError("max_chunk_s must be positive")

    def _action_spec(self, action_type: str) -> dict[str, Any]:
        # The spec also carries ``not_yet``: a list of unimplemented names.
        action_types = {
            name: spec
            for name, spec in self._spec["action_types"].items()
            if isinstance(spec, dict)
        }
        if action_type not in action_types:
            raise ValueError(
                f"Unknown action type {action_type!r}. "
                f"Available: {sorted(action_types)}"
            )
        spec = action_types[action_type]
        if int(spec["dim"]) != len(spec["layout"]):
            raise ValueError(
                f"Action type {action_type!r} declares dim {spec['dim']} "
                f"with {len(spec['layout'])} layout entries"
            )
        return spec

    @staticmethod
    def _validate_actions(
        action_type: str,
        spec: dict[str, Any],
        actions: Any,
    ) -> np.ndarray:
        dim = int(spec["dim"])
        array = np.asarray(actions, dtype=np.float64)
        if dim == 0:
            if array.size:
                raise ValueError(
                    f"Action type {action_type!r} takes no actions, got shape {array.shape}"
                )
            return array
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != dim:
            raise ValueError(
                f"Action type {action_type!r} needs shape (T, {dim}), got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"Action type {action_type!r} needs at least one row")
        layout = list(spec["layout"])
        finite = np.isfinite(array).all(axis=0)
        if not finite.all():
            bad = [name for name, ok in zip(layout, finite, strict=True) if not ok]
            raise ValueError(f"Non-finite action values for {bad}")
        limits = spec.get("limits") or {}
        for column, name in enumerate(layout):
            bounds = limits.get(name)
            if bounds is None:
                continue
            # A continuous or unbounded joint declares ``[null, null]`` (or one
            # ``null`` bound) rather than omitting the key, so a bound may be
            # absent. Only check the bounds the task actually declares.
            low = None if bounds[0] is None else float(bounds[0])
            high = None if bounds[1] is None else float(bounds[1])
            values = array[:, column]
            if (low is not None and (values < low).any()) or (
                high is not None and (values > high).any()
            ):
                raise ValueError(
                    f"Action {name} outside [{low}, {high}]: "
                    f"[{values.min()}, {values.max()}]"
                )
        return array

    def _resolve_dt(self, dt: float | None, rows: int) -> float | None:
        """Return the validated row spacing, or ``None`` for one control period."""
        period = self.control_period_s
        if dt is None:
            duration = rows * period
            if duration > float(self._spec["max_chunk_s"]):
                raise ValueError(
                    f"A {rows}-row chunk lasts {duration:.3f} s, over "
                    f"max_chunk_s={self._spec['max_chunk_s']}"
                )
            return None
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a positive finite number, got {dt}")
        if dt > _MAX_DT_S:
            raise ValueError(f"dt must be at most {_MAX_DT_S} s, got {dt}")
        periods = dt / period
        if abs(periods - round(periods)) > 1e-6:
            raise ValueError(
                f"dt must be a whole multiple of the {period:.4f} s control period"
            )
        if rows * dt > float(self._spec["max_chunk_s"]):
            raise ValueError(
                f"A {rows}-row chunk at dt={dt} lasts {rows * dt:.3f} s, over "
                f"max_chunk_s={self._spec['max_chunk_s']}"
            )
        return dt

    # -- episode state ---------------------------------------------------

    def _record_state(
        self, payload: dict[str, Any], *, authoritative: bool = False
    ) -> None:
        """Keep service state and reconcile counters when a response provides them."""
        previous_reset_calls = self.reset_calls
        episode = payload.get("episode")
        if isinstance(episode, int) and not isinstance(episode, bool) and episode >= 1:
            # The standard-mode protocol numbers its initial episode as 1.
            self.reset_calls = episode - 1
        resets_left = payload.get("resets_left")
        if (
            isinstance(resets_left, int)
            and not isinstance(resets_left, bool)
            and resets_left >= 0
        ):
            self._resets_left = resets_left
            if "episode" not in payload and not payload.get("done"):
                self.reset_calls = max(self.reset_budget - resets_left, 0)
        total_steps = payload.get("control_steps_total")
        if isinstance(total_steps, int) and not isinstance(total_steps, bool):
            self.control_steps = total_steps

        if authoritative and self.reset_calls != previous_reset_calls:
            # A direct /reset starts a new episode. Do not attach the prior
            # episode's observation or camera frames to the new one.
            self.last_meta = {}
        self.last_meta = {**self.last_meta, **payload}

        if "done" in payload and (authoritative or payload["done"]):
            self._done = bool(payload["done"])
            self.done_reason = payload.get("reason") if self._done else None

    def _reset_allowed_after_end(self) -> bool:
        """Whether the service still accepts ``/reset`` after this ending.

        The service refuses a reset after success, agent finish, or the time
        limit. An attempt that ended early on a task failure or on
        ``max_control_steps`` can still be abandoned for a fresh attempt, which
        is what the reset budget is for. A reason this adapter does not
        recognise is left to the service, which stays authoritative.
        """
        reason = (self.done_reason or "").strip().lower()
        if not reason:
            return True
        return not any(
            settled in reason
            for settled in (
                "success",
                "goal",
                "timeout",
                "timed out",
                "finished",
                "agent_finished",
            )
        )

    def _raise_if_done(self, call: str) -> None:
        if self._done:
            raise RuntimeError(
                f"Cannot {call}: the episode ended "
                f"({self.done_reason or 'no reason reported'})"
            )
