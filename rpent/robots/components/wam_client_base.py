# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Transport-independent client for RPent WAM workers."""

from __future__ import annotations

import numpy as np

from rpent.robots.components.wam_contracts import (
    WAMCapabilities,
    WAMControlSpec,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcClient


class BaseWAMClient:
    """Client for the ``wam.*`` RPC contract."""

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

    def get_capabilities(self) -> WAMCapabilities:
        """Fetch, validate, and cache checkpoint capabilities."""
        if self._capabilities is None:
            capabilities = self._client.call("wam.capabilities", timeout_s=30.0)
            if (
                self._expected_backend is not None
                and capabilities["backend"] != self._expected_backend
            ):
                raise ValueError(
                    f"expected backend {self._expected_backend!r}, got {capabilities['backend']!r}"
                )
            self._capabilities = capabilities
        return self._capabilities

    def validate_contract(self, expected: WAMControlSpec) -> None:
        """Connect and require the platform's native control contract."""
        actual = self.get_capabilities()["control"]
        if actual != expected:
            raise ValueError(
                f"WAM control mismatch: expected={expected!r}, actual={actual!r}"
            )

    def predict(self, request: WAMRequest) -> WAMPrediction:
        """Request actions and check their shape and values before execution."""
        capabilities = self.get_capabilities()
        prediction = self._client.call(
            self.PREDICT_METHOD, args=(request,), timeout_s=self._timeout_s
        )
        actions = np.asarray(prediction["actions"])
        action_dim = len(capabilities["control"]["action_schema"])
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != action_dim:
            raise ValueError(
                f"WAM actions must have non-empty [chunk, {action_dim}] shape"
            )
        chunk_size = capabilities["chunk_size"]
        if chunk_size is not None and len(actions) != chunk_size:
            raise ValueError(f"WAM prediction must contain {chunk_size} actions")
        if actions.dtype.kind not in "iuf" or not np.isfinite(actions).all():
            raise ValueError("WAM actions must be numeric without NaN or Inf")
        prediction["actions"] = actions
        return prediction

    def reset(self) -> None:
        """Clear this session's model history before a new episode."""
        self._client.call("wam.reset", timeout_s=self._timeout_s)

    def close(self) -> None:
        """Release the transport session and its model state."""
        self._client.close()
