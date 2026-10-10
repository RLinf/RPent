# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import numpy as np
import pytest

from rpent.robots.components.wam_facade_base import BaseWAMFacade
from rpent.robots.components.wam_rpc_protocol import (
    WAMCapabilities,
    WAMPrediction,
)


class _Facade(BaseWAMFacade):
    def __init__(self, actions: np.ndarray) -> None:
        self._actions = actions
        super().__init__(
            WAMCapabilities(
                backend="test",
                checkpoint="test-checkpoint",
                supported_embodiments=("test_arm",),
                action_space="test.delta.v1",
                state_schema={"joint_position": 2},
                camera_roles=("primary",),
                action_schema=("x", "y"),
            )
        )

    def predict_native(self, request, *, session_id=None):
        return WAMPrediction(
            actions=self._actions,
            metadata={
                "backend": "test",
                "checkpoint": "test-checkpoint",
                "embodiment": request["embodiment"],
            },
        )


def _request() -> dict:
    return {
        "images": {"primary": np.zeros((2, 2, 3), np.uint8)},
        "protocol_version": 1,
        "state": {"joint_position": np.zeros(2, np.float32)},
        "action_space": "test.delta.v1",
        "instruction": "move",
        "embodiment": "test_arm",
    }


def test_facade_rejects_native_action_dimension_that_breaks_capabilities() -> None:
    facade = _Facade(np.zeros((3, 3), np.float32))

    with pytest.raises(ValueError, match="advertised action_dim"):
        facade.predict(_request())
