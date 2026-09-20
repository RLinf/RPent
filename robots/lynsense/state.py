# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pure state validation; the adapter owns synchronization and clocks."""

from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from typing import Any


def finite_number(value: Any) -> float:
    """Reject non-numeric, boolean and non-finite message/config values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _array(value: Any) -> list:
    if isinstance(value, (str, bytes)):
        raise ValueError("expected an array, not text")
    return list(value)


class StateCache:
    """Cache independent stream receipts, never a motion-readiness verdict."""

    def __init__(self, stale_after_s: float):
        self.stale_after_s = finite_number(stale_after_s)
        if self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self._received: dict[str, float | None] = {"joints": None, "driver": None}
        self._fields: dict[str, Any] = {
            "observed_at": None,
            "joint_names": None,
            "positions": None,
        }
        self._invalid: str | None = None
        self._clock_failure: str | None = None
        self._warnings: list[str] = []

    def _receive(self, stream: str, received_mono: float) -> None:
        received = finite_number(received_mono)
        previous = self._received[stream]
        if previous is not None and received < previous:
            self._clock_failure = f"monotonic clock regressed for {stream}"
        self._received[stream] = received

    def update_driver(self, received_mono: float) -> None:
        """Record presence only; RobotMsg diagnostics remain uninterpreted."""
        self._receive("driver", received_mono)

    def update_joints(
        self, message: Any, received_mono: float, observed_at: str
    ) -> None:
        """Validate the newest sample, invalidating any previously usable one."""
        self._receive("joints", received_mono)
        try:
            names = _array(message.name)
            positions = [finite_number(v) for v in _array(message.position)]
            if not names or len(names) != len(positions):
                raise ValueError("joint names and positions must be nonempty and aligned")
            if not all(isinstance(n, str) and n.strip() for n in names):
                raise ValueError("joint names must be nonempty strings")
            if len(set(names)) != len(names):
                raise ValueError("joint names must be unique")
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self._invalid = f"invalid JointState: {exc}"
            return

        fields: dict[str, Any] = {
            "observed_at": observed_at,
            "joint_names": names,
            "positions": positions,
        }
        warnings = []
        for source, target in (("velocity", "velocities"), ("effort", "efforts")):
            try:
                values = _array(getattr(message, source, []))
                if not values:
                    continue
                if len(values) != len(names):
                    raise ValueError("length does not match joint names")
                fields[target] = [finite_number(v) for v in values]
            except (TypeError, ValueError, OverflowError) as exc:
                warnings.append(f"omitted {target}: {exc}")
        self._fields = fields
        self._warnings = warnings
        self._invalid = None

    def snapshot(self, now_mono: float) -> dict[str, Any]:
        """Return copied data with validity and separate receipt ages."""
        now = finite_number(now_mono)
        ages = {
            stream: None if received is None else now - received
            for stream, received in self._received.items()
        }
        if any(age is not None and (age < 0 or not math.isfinite(age)) for age in ages.values()):
            self._clock_failure = "invalid monotonic receipt age"
        missing = [stream for stream, age in ages.items() if age is None]
        stale = [
            stream for stream, age in ages.items()
            if age is not None and age > self.stale_after_s
        ]
        if self._clock_failure:
            status, reason = "failed", self._clock_failure
        elif self._invalid:
            status, reason = "invalid", self._invalid
        elif missing:
            status, reason = "unavailable", f"waiting for {', '.join(missing)}"
        elif stale:
            status, reason = "stale", f"no fresh receipt from {', '.join(stale)}"
        else:
            status, reason = "ok", "both state streams received recently; not a safety verdict"
        # Invalid clock ages are diagnostic errors, not JSON NaN/Infinity payloads.
        ages = {key: age if age is None or math.isfinite(age) else None for key, age in ages.items()}
        return deepcopy({
            **self._fields,
            "status": status,
            "reason": reason,
            "robot_state_received": self._received["driver"] is not None,
            "age_s": ages,
            "warnings": self._warnings,
        })
