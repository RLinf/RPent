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
adapter calls ``/reset`` through its RCB1 helper
``_post(path, meta, parts, timeout)``, the same path ``submit`` takes, and
calls ``/finish`` as a JSON request. Both live behind one adapter method each,
so a change in either contract fails in one place.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
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
            ``localhost:8000`` when this is unset.
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
        robot: Any | None = None,
    ) -> None:
        if robot is not None:
            if base_url is not None:
                raise ValueError("Pass either base_url or robot, not both.")
        else:
            from rcb_client import Robot

            robot = Robot(base_url, timeout=connect_timeout_s)
        self._robot = robot
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
        result = self._robot.get_obs(
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
        result = self._robot.submit(
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
        status = self._robot.status()
        self._record_state(status)
        return status

    def wait_ready(self, timeout_s: float = 600.0) -> dict[str, Any]:
        """Wait for a crashed service to resume the same episode, then report status.

        The service resumes in the state after the last call it completed. That
        call may have executed even if its response was lost, so the caller must
        read the status before deciding what to send next.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        status = self._robot.wait_ready(timeout=timeout_s)
        self._record_state(status)
        return status

    def reset(self, *, cameras: list[str] | None = None) -> dict[str, Any]:
        """Abandon the running episode and start a new one from the frozen state.

        Only a task that declares a reset budget exposes ``/reset``, and the
        service refuses it after success or once the budget is spent. Only the
        last episode is graded.
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
        result = self._service_post("/reset", meta, [])
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
        base = getattr(self._robot, "base", None)
        if base is None:
            raise RuntimeError("The service client has no base URL")
        status = self._post_json(f"{base}/finish", {})
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
        return self._robot.ik(targets, seed=None if seed is None else list(seed))

    # -- validation ------------------------------------------------------

    def _service_post(
        self,
        path: str,
        meta: dict[str, Any],
        parts: list[Any],
    ) -> Any:
        """POST through the client's RCB1 helper and return its result.

        ``rcb_client.Robot`` has no public reset or finish method. Reusing its
        ``_post`` keeps one dependency instead of two: the same helper that
        carries ``submit`` also carries these endpoint calls.
        """
        post = getattr(self._robot, "_post", None)
        if post is None:
            raise RuntimeError(
                "The service client exposes no _post path; reset needs "
                "rcb_client.Robot._post(path, meta, parts, timeout)"
            )
        return post(path, meta, parts, _SERVICE_POST_TIMEOUT_S)

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            with urllib.request.urlopen(request, timeout=60) as response:
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

    def _record_state(self, payload: dict[str, Any]) -> None:
        """Keep the latest known values; ``/status`` reports only counters."""
        self.last_meta = {**self.last_meta, **payload}
        if payload.get("done"):
            self._done = True
            self.done_reason = payload.get("reason")

    def _reset_allowed_after_end(self) -> bool:
        """Whether the service still accepts ``/reset`` after this ending.

        The service refuses a reset only once the trial is settled: success or
        the time limit. An attempt that ended early on a task failure or on
        ``max_control_steps`` can still be abandoned for a fresh attempt, which
        is what the reset budget is for. A reason this adapter does not
        recognise is left to the service, which stays authoritative.
        """
        reason = (self.done_reason or "").strip().lower()
        if not reason:
            return True
        return not any(
            settled in reason for settled in ("success", "goal", "timeout", "timed out")
        )

    def _raise_if_done(self, call: str) -> None:
        if self._done:
            raise RuntimeError(
                f"Cannot {call}: the episode ended "
                f"({self.done_reason or 'no reason reported'})"
            )
