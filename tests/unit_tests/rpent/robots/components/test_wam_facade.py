# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Check input validation at the service boundary, before model inference."""

from unittest.mock import Mock

import numpy as np
import pytest

from rpent.robots.components.wam_client_base import BaseWAMClient
from rpent.robots.components.wam_facade_base import BaseWAMFacade


def _facade():
    facade = BaseWAMFacade(
        {
            "backend": "test",
            "control": {
                "embodiment": "test_arm",
                "action_space": "test.delta.v1",
                "action_schema": ["x", "y"],
                "action_type": None,
            },
            "camera_roles": ("primary",),
            "state_schema": {"joint_position": 2},
            "chunk_size": None,
            "uses_sessions": False,
        }
    )
    facade.predict_native = Mock(return_value={"actions": np.zeros((3, 2), np.float32)})
    return facade


def _request():
    return {
        "images": {"primary": np.zeros((2, 2, 3), np.uint8)},
        "state": {"joint_position": np.zeros(2, np.float32)},
        "action_space": "test.delta.v1",
        "instruction": "move",
        "embodiment": "test_arm",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"instruction": " "},
        {"embodiment": "other"},
        {"action_space": "other"},
        {"images": {}},
        {"images": {"primary": np.zeros((2, 2))}},
        {"images": {"primary": np.zeros((2, 2, 3), np.float32)}},
        {"state": {}},
        {"state": {"joint_position": np.zeros(3)}},
        {"state": {"joint_position": np.array([0.0, np.nan])}},
    ],
)
def test_invalid_request_fails_before_inference(changes):
    facade = _facade()
    with pytest.raises(ValueError):
        facade.predict({**_request(), **changes})
    facade.predict_native.assert_not_called()


def test_invalid_model_output_is_rejected_at_client_boundary():
    facade = _facade()
    facade.predict_native.return_value = {"actions": np.zeros((3, 3), np.float32)}
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    with pytest.raises(ValueError, match="shape"):
        BaseWAMClient(rpc).predict(_request())
    facade.predict_native.assert_called_once()
