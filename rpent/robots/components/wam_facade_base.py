# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Server-side validation and session lifecycle for WAM workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rpent.robots.components.wam_rpc_protocol import (
    WAMCapabilities,
    WAMPrediction,
    normalize_wam_request,
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
        self._capabilities = WAMCapabilities.from_wire(capabilities.to_wire())
        self._adapter = adapter
        super().__init__(
            enable_sessions=self._capabilities.uses_sessions,
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

    def get_capabilities(self, *, session_id: str | None = None) -> dict[str, Any]:
        """Return this checkpoint's input and execution contract."""
        return self._capabilities.to_wire()

    def predict(
        self, request: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """Validate input and output and attach the negotiated contract identity."""
        normalized = normalize_wam_request(request)
        self._capabilities.validate_request(normalized)
        model_input = self._adapter[0](normalized) if self._adapter else normalized
        result = self.predict_native(model_input, session_id=session_id)
        if self._adapter:
            result = self._adapter[1](result)
        prediction = WAMPrediction.from_wire(
            result.to_wire() if isinstance(result, WAMPrediction) else result
        )
        self._capabilities.validate_prediction(prediction)
        prediction.metadata.update(
            self._capabilities.prediction_metadata(normalized["embodiment"])
        )
        return prediction.to_wire()

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
        if self._capabilities.uses_sessions:
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
