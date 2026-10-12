# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared action-model transport and VLA/WAM client contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.am_contracts import (
    AMControlSpec,
    WAMCapabilities,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcClient


class BaseAMClient:
    """Hold an RPC-compatible transport; subclasses own prediction contracts."""

    PREDICT_METHOD: str

    def __init__(self, client: Any, *, timeout_s: float) -> None:
        self._client = client
        self._timeout_s = float(timeout_s)

    def _predict(self, *args: Any) -> Any:
        """Forward the subclass's prediction arguments without conversion."""
        return self._client.call(
            self.PREDICT_METHOD, args=args, timeout_s=self._timeout_s
        )


class BaseVLAClient(BaseAMClient):
    """Preserve the VLA RPC contract for existing model clients."""

    PREDICT_METHOD = "vla.predict"
    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client: Any) -> None:
        super().__init__(client, timeout_s=self._TIMEOUT_S["predict"])

    def predict(self, obs: Any, options: dict | None = None) -> Any:
        """Request a single VLA action chunk from observations and options."""
        return self._predict(obs, options)


class BaseWAMClient(BaseAMClient):
    """Client for the ``wam.*`` RPC contract."""

    PREDICT_METHOD = "wam.predict"

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_backend: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(client, timeout_s=timeout_s)
        self._expected_backend = expected_backend
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

    def validate_contract(self, expected: AMControlSpec) -> None:
        """Connect and require the platform's native control contract."""
        actual = self.get_capabilities()["control"]
        if actual != expected:
            raise ValueError(
                f"WAM control mismatch: expected={expected!r}, actual={actual!r}"
            )

    def predict(self, request: WAMRequest) -> WAMPrediction:
        """Request actions and check their shape and values before execution."""
        capabilities = self.get_capabilities()
        prediction = self._predict(request)
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
