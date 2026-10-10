# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import numpy as np
import pytest

from rpent.robots.components.wam_client_base import BaseWAMClient
from rpent.robots.components.wam_rpc_protocol import (
    WAMCapabilities,
    WAMPrediction,
    normalize_wam_request,
)

CAPABILITIES = {
    "protocol_version": 1,
    "backend": "cosmos_policy",
    "checkpoint": "predict2-2b-libero",
    "supported_embodiments": ["libero_7d"],
    "action_space": "test.delta.v1",
    "state_schema": {"joint_position": 8},
    "camera_roles": ["primary", "wrist"],
    "action_schema": ["a0", "a1", "a2", "a3", "a4", "a5", "a6"],
}


def _request() -> dict:
    return {
        "images": {
            "primary": np.zeros((4, 5, 3), np.uint8),
            "wrist": np.ones((4, 5, 3), np.uint8),
        },
        "state": {"joint_position": np.arange(8, dtype=np.float32)},
        "protocol_version": 1,
        "action_space": "test.delta.v1",
        "instruction": " pick the bowl ",
        "embodiment": "libero_7d",
        "metadata": {"step": 2},
    }


def test_protocol_round_trip_preserves_arrays_and_optional_outputs() -> None:
    request = normalize_wam_request(_request())
    assert request["instruction"] == "pick the bowl"
    assert request["images"]["primary"].dtype == np.uint8

    prediction = WAMPrediction.from_wire(
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
    assert WAMPrediction.from_wire(prediction.to_wire()).actions.shape == (
        3,
        7,
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda request: request.update(instruction=" "), "instruction"),
        (lambda request: request["state"].update(joint_position=[0.0, np.nan]), "NaN"),
        (
            lambda request: request["images"].update(primary=np.zeros((4, 5))),
            "shape",
        ),
    ],
)
def test_request_rejects_malformed_inputs(mutate, message: str) -> None:
    request = _request()
    mutate(request)
    with pytest.raises(ValueError, match=message):
        normalize_wam_request(request)


@pytest.mark.parametrize(
    "actions, message",
    [
        ([], "non-empty"),
        (np.zeros(7), "shape"),
        (np.full((1, 7), np.inf), "NaN or Inf"),
    ],
)
def test_prediction_rejects_unsafe_actions(actions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WAMPrediction.from_wire({"actions": actions})


def test_capabilities_require_explicit_embodiment_and_schema() -> None:
    capabilities = WAMCapabilities.from_wire(CAPABILITIES)
    capabilities.require_execution("libero_7d", "test.delta.v1", action_dim=7)
    with pytest.raises(ValueError, match="unsupported_arm"):
        capabilities.require_execution("unsupported_arm", "test.delta.v1")
    with pytest.raises(ValueError, match="action_space"):
        capabilities.require_execution("libero_7d", "test.absolute.v1")
    with pytest.raises(ValueError, match="action_schema"):
        capabilities.require_execution(
            "libero_7d",
            "test.delta.v1",
            action_schema=tuple(reversed(capabilities.action_schema)),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"protocol_version": None}, "protocol_version"),
        ({"protocol_version": True}, "protocol_version"),
        ({"state_schema": {"joint_position": 8.5}}, "positive integer"),
        ({"chunk_size": True}, "positive integer"),
        ({"uses_sessions": "false"}, "boolean"),
        ({"action_schema": ["x", "x"]}, "duplicates"),
    ],
)
def test_capabilities_reject_ambiguous_contracts(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        WAMCapabilities.from_wire({**CAPABILITIES, **changes})


class _Rpc:
    def __init__(self) -> None:
        self.calls = []
        self.prediction_action_schema = list(CAPABILITIES["action_schema"])

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs, timeout_s))
        if method == "wam.capabilities":
            return CAPABILITIES
        request = args[0]
        return {
            "actions": np.ones((2, 7), np.float32),
            "metadata": {
                "backend": "cosmos_policy",
                "checkpoint": "predict2-2b-libero",
                "embodiment": request["embodiment"],
                "action_schema": self.prediction_action_schema,
                "action_space": "test.delta.v1",
            },
        }

    def close(self) -> None:
        pass


def test_client_caches_capabilities_and_validates_prediction_metadata() -> None:
    rpc = _Rpc()
    client = BaseWAMClient(rpc, expected_backend="cosmos_policy")
    assert client.predict(_request()).actions.shape == (2, 7)
    assert client.predict(_request()).actions.shape == (2, 7)
    assert [call[0] for call in rpc.calls].count("wam.capabilities") == 1


def test_client_rejects_prediction_with_mismatched_action_schema() -> None:
    rpc = _Rpc()
    rpc.prediction_action_schema = ["wrong"] * 7
    client = BaseWAMClient(rpc, expected_backend="cosmos_policy")

    with pytest.raises(ValueError, match="action_schema"):
        client.predict(_request())


@pytest.mark.parametrize("field", ["state", "images", "action_space"])
def test_incompatible_request_fails_before_inference(field) -> None:
    rpc = _Rpc()
    request = _request()
    request[field] = "other.controller.v1" if field == "action_space" else {}
    with pytest.raises(ValueError):
        BaseWAMClient(rpc).predict(request)
    assert [call[0] for call in rpc.calls] == ["wam.capabilities"]
