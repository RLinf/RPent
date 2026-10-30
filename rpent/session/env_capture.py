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

"""Capture tool execution and environment actions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from rpent.session import EnvState, MissingInfo, Payload


class ToolCallCapture:
    """Bind execution updates to one registered tool call."""

    def __init__(self, state: EnvState, tool_call_id: str) -> None:
        self.state = state
        self.tool_call_id = tool_call_id
        self.started = False
        self.finished = False
        self._result_captured = False
        self.action_source: Literal["tool_computed", "direct"] = "tool_computed"
        self.prediction_id: str | None = None

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Propagate this call through async tasks and to_thread workers."""
        token = current_tool_call.set(self)
        try:
            yield
        finally:
            current_tool_call.reset(token)

    def start(
        self, *, action_source: Literal["tool_computed", "direct"] = "tool_computed"
    ) -> None:
        """Record actual admission to execution."""
        self.state.update_tool_call(
            self.tool_call_id, status="running", started_at=datetime.now(timezone.utc)
        )
        self.action_source = action_source
        self.started = True

    def result(self, payload: Payload) -> None:
        """Save the original result before any adapter transformation."""
        self.state.update_tool_call(
            self.tool_call_id, raw_result=payload, missing={"raw_result": None}
        )
        self._result_captured = True

    def finish(
        self, status: Literal["completed", "failed", "cancelled"], detail: str = ""
    ) -> None:
        """Finish execution without inventing timestamps for unstarted calls."""
        missing = {}
        if not self._result_captured:
            missing["raw_result"] = MissingInfo(
                "not_recorded" if self.started else "not_applicable", detail
            )
        self.state.update_tool_call(
            self.tool_call_id,
            status=status if self.started else "not_executed",
            ended_at=datetime.now(timezone.utc) if self.started else None,
            missing=missing,
        )
        self.finished = True


current_tool_call: ContextVar[ToolCallCapture | None] = ContextVar(
    "rpent_tool_call", default=None
)


class CapturingEnvClient:
    """Record executed environment actions for the current API tool call."""

    def __init__(
        self,
        wrapped: Any,
        state: EnvState,
        *,
        capture_call: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.state = state
        self._capture_call = capture_call
        self._observation_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the client and discard the previous observation association."""
        self._observation_id = None
        return self.wrapped.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Execute and record one action, returning the original client result."""
        return self._execute("step", *args, **kwargs)

    def chunk_step(self, *args: Any, **kwargs: Any) -> Any:
        """Execute and record an action batch, returning the original client result."""
        return self._execute("chunk_step", *args, **kwargs)

    @contextmanager
    def prediction(self, instruction: str) -> Iterator[str | None]:
        """Group executed actions under one VLA prediction."""
        capture = current_tool_call.get()
        if capture is None or capture.state is not self.state:
            yield None
            return
        prediction_id = self.state.register_prediction(
            tool_call_id=capture.tool_call_id,
            instruction=instruction,
        )
        previous = capture.prediction_id
        capture.prediction_id = prediction_id
        try:
            yield prediction_id
        finally:
            capture.prediction_id = previous

    def _execute(
        self, method: Literal["step", "chunk_step"], *args: Any, **kwargs: Any
    ) -> Any:
        operation = getattr(self.wrapped, method)
        capture = current_tool_call.get()
        if capture is None or capture.state is not self.state:
            self._observation_id = None
            return operation(*args, **kwargs)
        try:
            response = (
                self._capture_call(method, *args, **kwargs)
                if self._capture_call is not None
                else operation(*args, capture=True, **kwargs)
            )
        except BaseException as exc:
            self._observation_id = None
            self.state.update_tool_call(
                capture.tool_call_id,
                missing={
                    "after_observation_id": MissingInfo(
                        "capture_failed",
                        f"Environment execution is unconfirmed: {type(exc).__name__}: {exc}",
                    )
                },
            )
            raise
        if not response["steps"]:
            return response["result"]

        before_id = self._observation_id or self.state.register_observation(
            payload=self._payload(response["before_observation"])
        )
        first_before_id = before_id
        for step in response["steps"]:
            after_id = self.state.register_observation(
                payload=self._payload(step["after_observation"])
            )
            self.state.register_action(
                tool_call_id=capture.tool_call_id,
                source="vla"
                if capture.prediction_id is not None
                else capture.action_source,
                prediction_id=capture.prediction_id,
                action=self._payload(step["action"]),
                before_observation_id=before_id,
                after_observation_id=after_id,
                result=self._payload(step["result"]),
            )
            before_id = after_id
        self._observation_id = before_id
        tool_call = self.state.get_tool_call(capture.tool_call_id)
        self.state.update_tool_call(
            capture.tool_call_id,
            before_observation_id=tool_call.before_observation_id or first_before_id,
            after_observation_id=before_id,
            missing={"before_observation_id": None, "after_observation_id": None},
        )
        return response["result"]

    def _payload(self, data: dict[str, Any]) -> Payload:
        """Decode a payload and save its non-inline arrays and bytes as attachments."""
        payload = Payload(**data)
        payload.missing = {
            path: MissingInfo(**reason) for path, reason in payload.missing.items()
        }

        def store(value: Any, pointer: str = "") -> Any:
            if isinstance(value, np.ndarray) and pointer in payload.inline_array_paths:
                return value
            if isinstance(value, (np.ndarray, bytes)):
                try:
                    return self.state.save_trajectory_artifact(value)
                except OSError as exc:
                    payload.completeness = "partial"
                    payload.missing[pointer] = MissingInfo(
                        "capture_failed", f"{type(exc).__name__}: {exc}"
                    )
                    return None
            if isinstance(value, dict):
                return {
                    key: store(
                        item, f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
                    )
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [
                    store(item, f"{pointer}/{index}")
                    for index, item in enumerate(value)
                ]
            return value

        payload.data = store(payload.data)
        return payload
