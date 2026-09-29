# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Transport-independent client for RPent action-model bridges."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
    normalize_action_model_request,
)
from rpent.robots.components.policy_client_base import BasePolicyClient
from rpent.utils.rpc import RpcClient


class BaseActionModelClient(BasePolicyClient):
    """Client for the unified ``action_model.*`` RPC contract."""

    PREDICT_METHOD = "action_model.predict"

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_backend: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(client)
        self._expected_backend = expected_backend
        self._TIMEOUT_S = {**self._TIMEOUT_S, "predict": float(timeout_s)}
        self._capabilities: ActionModelCapabilities | None = None

    def get_capabilities(self, *, refresh: bool = False) -> ActionModelCapabilities:
        """Fetch, validate, and cache bridge capabilities."""
        if refresh or self._capabilities is None:
            payload = self._client.call("action_model.capabilities", timeout_s=30.0)
            capabilities = ActionModelCapabilities.from_wire(payload)
            if (
                self._expected_backend is not None
                and capabilities.backend != self._expected_backend
            ):
                raise ActionModelProtocolError(
                    f"expected backend {self._expected_backend!r}, got "
                    f"{capabilities.backend!r}"
                )
            self._capabilities = capabilities
        return self._capabilities

    def predict(self, request: dict[str, Any]) -> ActionModelPrediction:
        """Validate a request, call the bridge, and validate its prediction."""
        normalized = normalize_action_model_request(request)
        capabilities = self.get_capabilities()
        capabilities.require_embodiment(normalized["embodiment"])
        payload = self._request_prediction(normalized)
        prediction = ActionModelPrediction.from_wire(payload)
        if prediction.actions.shape[1] != capabilities.action_dim:
            raise ActionModelProtocolError(
                f"prediction action dimension {prediction.actions.shape[1]} does not "
                f"match advertised {capabilities.action_dim}"
            )
        metadata = prediction.metadata
        for name, expected in (
            ("backend", capabilities.backend),
            ("checkpoint", capabilities.checkpoint),
            ("embodiment", normalized["embodiment"]),
        ):
            actual = metadata.get(name)
            if actual != expected:
                raise ActionModelProtocolError(
                    f"prediction.metadata.{name} must be {expected!r}, got {actual!r}"
                )
        action_schema = metadata.get("action_schema")
        if not isinstance(action_schema, (list, tuple)) or tuple(
            action_schema
        ) != tuple(capabilities.action_schema):
            raise ActionModelProtocolError(
                "prediction.metadata.action_schema must match advertised "
                f"{list(capabilities.action_schema)!r}, got {action_schema!r}"
            )
        return prediction

    def close(self) -> None:
        """Close the underlying transport session when one is enabled."""
        self._client.close()
