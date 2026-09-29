# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from rpent.robots.components.action_model_protocol import ActionModelProtocolError
from scripts.wam.dreamzero_rpc_bridge import (
    DROID_PROPRIO_SCHEMA,
    DreamZeroBridge,
)


def _install_dreamzero_client(monkeypatch, client_type: type) -> None:
    package = types.ModuleType("eval_utils")
    module = types.ModuleType("eval_utils.policy_client")
    module.WebsocketClientPolicy = client_type
    package.policy_client = module
    monkeypatch.setitem(sys.modules, "eval_utils", package)
    monkeypatch.setitem(sys.modules, "eval_utils.policy_client", module)


def test_dreamzero_bridge_requires_session_aware_native_server(monkeypatch) -> None:
    class Client:
        def __init__(self, **kwargs):
            del kwargs

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": False,
                "action_space": "joint_position",
            }

    _install_dreamzero_client(monkeypatch, Client)

    with pytest.raises(RuntimeError, match="session IDs"):
        DreamZeroBridge("127.0.0.1", 8000, "DreamZero-DROID")


def test_dreamzero_bridge_maps_explicit_droid_camera_and_proprio_schema(
    monkeypatch,
) -> None:
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.observations = []
            created.append(self)

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": True,
                "action_space": "joint_position",
            }

        def infer(self, observation):
            self.observations.append(observation)
            return np.ones((2, 8), np.float32)

        def reset(self, options):
            del options

    _install_dreamzero_client(monkeypatch, Client)
    bridge = DreamZeroBridge("dreamzero", 8000, "DreamZero-DROID")
    result = bridge.predict(
        {
            "images": {
                "primary": np.zeros((4, 4, 3), np.uint8),
                "wrist": np.ones((4, 4, 3), np.uint8),
                "extra": [np.full((4, 4, 3), 2, np.uint8)],
            },
            "proprio": np.arange(14, dtype=np.float32),
            "instruction": "pick the cup",
            "embodiment": "droid_joint_position_8d",
            "metadata": {
                "proprio_schema": list(DROID_PROPRIO_SCHEMA),
                "episode_id": "episode-7",
            },
        }
    )

    assert result["actions"].shape == (2, 8)
    native = created[0].observations[0]
    np.testing.assert_array_equal(native["observation/joint_position"], np.arange(7))
    np.testing.assert_array_equal(
        native["observation/cartesian_position"], np.arange(7, 13)
    )
    np.testing.assert_array_equal(native["observation/gripper_position"], [13])
    assert native["session_id"] == "episode-7"
    assert native["prompt"] == "pick the cup"


def test_dreamzero_bridge_requires_episode_id_and_switches_native_session(
    monkeypatch,
) -> None:
    created = []

    class Client:
        def __init__(self, **kwargs):
            del kwargs
            self.observations = []
            created.append(self)

        def get_server_metadata(self):
            return {
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_session_id": True,
                "action_space": "joint_position",
            }

        def infer(self, observation):
            self.observations.append(observation)
            return np.ones((1, 8), np.float32)

        def reset(self, options):
            del options

    _install_dreamzero_client(monkeypatch, Client)
    bridge = DreamZeroBridge("dreamzero", 8000, "DreamZero-DROID")
    request = {
        "images": {
            "primary": np.zeros((4, 4, 3), np.uint8),
            "wrist": np.ones((4, 4, 3), np.uint8),
            "extra": [np.full((4, 4, 3), 2, np.uint8)],
        },
        "proprio": np.arange(14, dtype=np.float32),
        "instruction": "pick the cup",
        "embodiment": "droid_joint_position_8d",
        "metadata": {"proprio_schema": list(DROID_PROPRIO_SCHEMA)},
    }

    with pytest.raises(ActionModelProtocolError, match="episode_id"):
        bridge.predict(request)

    for episode_id in ("episode-1", "episode-2"):
        request["metadata"]["episode_id"] = episode_id
        bridge.predict(request)
    with pytest.raises(ActionModelProtocolError, match="14 proprioception fields"):
        bridge.predict({**request, "proprio": np.zeros(7, np.float32)})
    assert [obs["session_id"] for obs in created[0].observations] == [
        "episode-1",
        "episode-2",
    ]
