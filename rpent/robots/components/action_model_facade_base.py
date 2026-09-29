# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Server-side base for unified action-model bridges."""

from __future__ import annotations

from typing import Any

from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
    normalize_action_model_request,
)
from rpent.utils.rpc import RpcFacade


class BaseActionModelFacade(RpcFacade):
    """Register and validate the common ``action_model.*`` RPC methods."""

    def __init__(self, capabilities: ActionModelCapabilities) -> None:
        super().__init__()
        self._capabilities = capabilities
        self._rpc["action_model.capabilities"] = self.get_capabilities
        self._rpc["action_model.predict"] = self.predict
        self._readonly_methods.add("action_model.capabilities")

    def get_capabilities(self) -> dict[str, Any]:
        """Return the bridge's stable execution contract."""
        return self._capabilities.to_wire()

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate input and normalize the backend prediction."""
        normalized = normalize_action_model_request(request)
        self._capabilities.require_embodiment(normalized["embodiment"])
        prediction = self.predict_native(normalized)
        if not isinstance(prediction, ActionModelPrediction):
            prediction = ActionModelPrediction.from_wire(prediction)
        if prediction.actions.shape[1] != self._capabilities.action_dim:
            raise ActionModelProtocolError(
                f"native prediction action dimension {prediction.actions.shape[1]} "
                f"does not match advertised action_dim "
                f"{self._capabilities.action_dim}"
            )
        return prediction.to_wire()

    def predict_native(
        self, request: dict[str, Any]
    ) -> ActionModelPrediction | dict[str, Any]:
        """Run backend inference. Subclasses must implement this method."""
        raise NotImplementedError
