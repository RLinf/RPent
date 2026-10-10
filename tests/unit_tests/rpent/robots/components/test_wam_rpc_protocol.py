# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Check the client boundary without object serialization or metadata echoes."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components.wam_client_base import BaseWAMClient

CONTROL = {
    "embodiment": "test_arm",
    "action_space": "test.delta.v1",
    "action_schema": ["x", "y"],
    "action_type": None,
}
CAPABILITIES = {
    "backend": "test",
    "control": CONTROL,
    "chunk_size": 3,
}


def _client(prediction, capabilities=None):
    rpc = Mock()
    rpc.call.side_effect = lambda method, **kw: (
        (capabilities if capabilities is not None else CAPABILITIES)
        if method == "wam.capabilities"
        else prediction
    )
    return BaseWAMClient(rpc, expected_backend="test"), rpc


def test_client_reuses_capabilities_and_passes_optional_outputs_unchanged():
    prediction = {
        "actions": np.ones((3, 2), np.float32),
        "future_observation": {"image": np.zeros((2, 2, 3), np.uint8)},
        "value": np.asarray(0.75, np.float32),
    }
    client, rpc = _client(prediction)
    client.validate_contract(CONTROL)
    request = {"instruction": "move"}
    for _ in range(2):
        result = client.predict(request)
        assert result is prediction
        assert result["actions"] is prediction["actions"]
    assert [call.args[0] for call in rpc.call.call_args_list].count(
        "wam.capabilities"
    ) == 1
    assert rpc.call.call_args.kwargs["args"][0] is request


@pytest.mark.parametrize(
    "field", ["embodiment", "action_space", "action_schema", "action_type"]
)
def test_client_rejects_incompatible_control_before_prediction(field):
    capabilities = deepcopy(CAPABILITIES)
    capabilities["control"][field] = ["y", "x"] if field == "action_schema" else "other"
    client, rpc = _client({}, capabilities)
    with pytest.raises(ValueError, match="control mismatch"):
        client.validate_contract(CONTROL)
    assert [call.args[0] for call in rpc.call.call_args_list] == ["wam.capabilities"]


def test_client_rejects_wrong_backend():
    client, _ = _client({}, {**CAPABILITIES, "backend": "other"})
    with pytest.raises(ValueError, match="expected backend"):
        client.get_capabilities()


@pytest.mark.parametrize(
    "actions",
    [
        [],
        np.zeros(2),
        np.zeros((3, 3)),
        np.zeros((2, 2)),
        np.full((3, 2), np.nan),
        np.full((3, 2), np.inf),
        np.full((3, 2), "invalid"),
    ],
)
def test_client_rejects_invalid_actions_before_execution(actions):
    client, _ = _client({"actions": actions})
    with pytest.raises(ValueError):
        client.predict({})
