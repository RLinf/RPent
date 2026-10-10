# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Transport-independent client for RPent WAM workers."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.wam_rpc_protocol import (
    WAMCapabilities,
    WAMPrediction,
    normalize_wam_request,
)
from rpent.utils.rpc import RpcClient


class BaseWAMClient:
    """Client for the versioned ``wam.*`` RPC contract."""

    PREDICT_METHOD = "wam.predict"

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_backend: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self._client = client
        self._expected_backend = expected_backend
        self._timeout_s = float(timeout_s)
        self._capabilities: WAMCapabilities | None = None

    def get_capabilities(self, *, refresh: bool = False) -> WAMCapabilities:
        """Fetch, validate, and cache checkpoint capabilities."""
        if refresh or self._capabilities is None:
            payload = self._client.call("wam.capabilities", timeout_s=30.0)
            capabilities = WAMCapabilities.from_wire(payload)
            if (
                self._expected_backend is not None
                and capabilities.backend != self._expected_backend
            ):
                raise ValueError(
                    f"expected backend {self._expected_backend!r}, got {capabilities.backend!r}"
                )
            self._capabilities = capabilities
        return self._capabilities

    def predict(self, request: dict[str, Any]) -> WAMPrediction:
        """Validate an observation and the returned executable action chunk."""
        normalized = normalize_wam_request(request)
        capabilities = self.get_capabilities()
        capabilities.validate_request(normalized)
        payload = self._client.call(
            self.PREDICT_METHOD, args=(normalized,), timeout_s=self._timeout_s
        )
        prediction = WAMPrediction.from_wire(payload)
        capabilities.validate_prediction(prediction)
        for name, expected in capabilities.prediction_metadata(
            normalized["embodiment"]
        ).items():
            actual = prediction.metadata.get(name)
            if name == "action_schema" and isinstance(actual, tuple):
                actual = list(actual)
            if actual != expected:
                raise ValueError(
                    f"prediction.metadata.{name} must be {expected!r}, got {actual!r}"
                )
        return prediction

    def reset(self) -> None:
        """Clear this session's model history before a new episode."""
        self._client.call("wam.reset", timeout_s=self._timeout_s)

    def close(self) -> None:
        """Release the transport session and its model state."""
        self._client.close()
