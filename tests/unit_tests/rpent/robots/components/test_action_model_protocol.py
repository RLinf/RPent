# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import numpy as np
import pytest

from rpent.robots.components.action_model_client_base import BaseActionModelClient
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelCompatibilityError,
    ActionModelPrediction,
    ActionModelProtocolError,
    normalize_action_model_request,
)

CAPABILITIES = {
    "backend": "cosmos_policy",
    "checkpoint": "predict2-2b-libero",
    "supported_embodiments": ["libero_7d"],
    "action_dim": 7,
    "camera_roles": ["primary", "wrist"],
    "action_schema": ["a0", "a1", "a2", "a3", "a4", "a5", "a6"],
}


def _request() -> dict:
    return {
        "images": {
            "primary": np.zeros((4, 5, 3), np.uint8),
            "wrist": np.ones((4, 5, 3), np.uint8),
            "extra": [],
        },
        "proprio": np.arange(8, dtype=np.float32),
        "instruction": " pick the bowl ",
        "embodiment": "libero_7d",
        "metadata": {"step": 2},
    }


def test_protocol_round_trip_preserves_arrays_and_optional_outputs() -> None:
    request = normalize_action_model_request(_request())
    assert request["instruction"] == "pick the bowl"
    assert request["images"]["primary"].dtype == np.uint8

    prediction = ActionModelPrediction.from_wire(
        {
            "actions": np.ones((3, 7), np.float64),
            "future_observation": {"image": np.zeros((2, 2, 3), np.uint8)},
            "value": np.asarray(0.75, np.float32),
            "metadata": {"backend": "cosmos_policy"},
        }
    )
    assert prediction.actions.shape == (3, 7)
    assert prediction.actions.dtype == np.float32
    assert prediction.value == pytest.approx(0.75)
    assert ActionModelPrediction.from_wire(prediction.to_wire()).actions.shape == (
        3,
        7,
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda request: request.update(instruction=" "), "instruction"),
        (lambda request: request.update(proprio=[0.0, np.nan]), "NaN"),
        (
            lambda request: request["images"].update(primary=np.zeros((4, 5))),
            "shape",
        ),
    ],
)
def test_request_rejects_malformed_inputs(mutate, message: str) -> None:
    request = _request()
    mutate(request)
    with pytest.raises(ActionModelProtocolError, match=message):
        normalize_action_model_request(request)


@pytest.mark.parametrize(
    "actions, message",
    [
        ([], "non-empty"),
        (np.zeros(7), "shape"),
        (np.full((1, 7), np.inf), "NaN or Inf"),
    ],
)
def test_prediction_rejects_unsafe_actions(actions, message: str) -> None:
    with pytest.raises(ActionModelProtocolError, match=message):
        ActionModelPrediction.from_wire({"actions": actions})


def test_capabilities_require_explicit_embodiment_and_schema() -> None:
    capabilities = ActionModelCapabilities.from_wire(CAPABILITIES)
    capabilities.require_embodiment("libero_7d", action_dim=7)
    with pytest.raises(ActionModelCompatibilityError, match="droid"):
        capabilities.require_embodiment("droid")
    with pytest.raises(ActionModelCompatibilityError, match="action_schema"):
        capabilities.require_embodiment("libero_7d", action_schema=("different",) * 7)


class _Rpc:
    def __init__(self) -> None:
        self.calls = []
        self.prediction_action_schema = list(CAPABILITIES["action_schema"])

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs, timeout_s))
        if method == "action_model.capabilities":
            return CAPABILITIES
        request = args[0]
        return {
            "actions": np.ones((2, 7), np.float32),
            "metadata": {
                "backend": "cosmos_policy",
                "checkpoint": "predict2-2b-libero",
                "embodiment": request["embodiment"],
                "action_schema": self.prediction_action_schema,
            },
        }

    def close(self) -> None:
        pass


def test_client_caches_capabilities_and_validates_prediction_metadata() -> None:
    rpc = _Rpc()
    client = BaseActionModelClient(rpc, expected_backend="cosmos_policy")
    assert client.predict(_request()).actions.shape == (2, 7)
    assert client.predict(_request()).actions.shape == (2, 7)
    assert [call[0] for call in rpc.calls].count("action_model.capabilities") == 1


def test_client_rejects_prediction_with_mismatched_action_schema() -> None:
    rpc = _Rpc()
    rpc.prediction_action_schema = ["wrong"] * 7
    client = BaseActionModelClient(rpc, expected_backend="cosmos_policy")

    with pytest.raises(ActionModelProtocolError, match="action_schema"):
        client.predict(_request())
