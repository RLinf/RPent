# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Server-side validation and session lifecycle for WAM workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rpent.robots.components.wam_contracts import (
    WAMCapabilities,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.rpc_facade import DEFAULT_SESSION_TIMEOUT_S

WAMAdapter = tuple[
    Callable[[dict[str, Any]], dict[str, Any]],
    Callable[[Any], WAMPrediction],
]


@dataclass(frozen=True)
class WAMAdapterSpec:
    """Register server-side transforms and their checkpoint capabilities together."""

    encode: Callable[..., dict[str, Any]]
    decode: Callable[..., WAMPrediction]
    capabilities: Callable[..., WAMCapabilities]


class BaseWAMFacade(RpcFacade):
    """Own the common ``wam.*`` routes, independent of model and simulator."""

    PREDICT_METHOD = "wam.predict"

    def __init__(
        self,
        capabilities: WAMCapabilities,
        *,
        adapter: WAMAdapter | None = None,
        session_timeout_s: float = DEFAULT_SESSION_TIMEOUT_S,
    ) -> None:
        self._capabilities = capabilities
        self._adapter = adapter
        super().__init__(
            enable_sessions=capabilities["uses_sessions"],
            session_timeout_s=session_timeout_s,
        )
        self._rpc.update(
            {
                self.PREDICT_METHOD: self.predict,
                "wam.capabilities": self.get_capabilities,
                "wam.reset": self.reset,
            }
        )
        self._readonly_methods.add("wam.capabilities")

    def get_capabilities(self, *, session_id: str | None = None) -> WAMCapabilities:
        """Return this checkpoint's input and execution contract."""
        return self._capabilities

    def predict(
        self, request: WAMRequest, *, session_id: str | None = None
    ) -> WAMPrediction:
        """Check incoming observations, then encode, infer and decode directly."""
        control = self._capabilities["control"]
        if any(request[key] != control[key] for key in ("embodiment", "action_space")):
            raise ValueError("WAM request does not match the worker's control contract")
        if (
            not isinstance(request["instruction"], str)
            or not request["instruction"].strip()
        ):
            raise ValueError("instruction must be a non-empty string")
        for role in self._capabilities["camera_roles"]:
            image = np.asarray(request["images"].get(role))
            if (
                image.ndim != 3
                or image.shape[-1] != 3
                or 0 in image.shape
                or image.dtype != np.uint8
            ):
                raise ValueError(
                    f"images.{role} must have non-empty [H, W, 3] uint8 RGB shape"
                )
        for name, size in self._capabilities["state_schema"].items():
            state = np.asarray(request["state"].get(name))
            if (
                state.shape != (size,)
                or state.dtype.kind not in "iuf"
                or not np.isfinite(state).all()
            ):
                raise ValueError(
                    f"state.{name} must have shape ({size},) without NaN or Inf"
                )
        model_input = self._adapter[0](request) if self._adapter else request
        result = self.predict_native(model_input, session_id=session_id)
        return self._adapter[1](result) if self._adapter else result

    def predict_native(
        self, request: dict[str, Any], *, session_id: str | None = None
    ) -> Any:
        """Infer from encoded input; a configured adapter decodes the native result."""
        raise NotImplementedError

    def reset(self, *, session_id: str | None = None) -> dict[str, bool]:
        """Release episode history; stateless workers need no reset hook."""
        self.reset_native(session_id=session_id)
        return {"ok": True}

    def reset_native(self, *, session_id: str | None = None) -> None:
        """Stateful workers must discard this session's cached model state."""
        if self._capabilities["uses_sessions"]:
            raise NotImplementedError(
                "session-based WAM workers must implement reset_native"
            )

    def _on_session_drop(self, session_id: str) -> None:
        self.reset_native(session_id=session_id)

    def close(self) -> None:
        """Release remaining model histories during server shutdown."""
        for session_id in list(self._sessions):
            self.drop_session(session_id)
        super().close()
