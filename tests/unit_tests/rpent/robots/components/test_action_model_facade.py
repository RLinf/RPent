# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import numpy as np
import pytest

from rpent.robots.components.action_model_facade_base import BaseActionModelFacade
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
)


class _Facade(BaseActionModelFacade):
    def __init__(self, actions: np.ndarray) -> None:
        self._actions = actions
        super().__init__(
            ActionModelCapabilities(
                backend="test",
                checkpoint="test-checkpoint",
                supported_embodiments=("test_arm",),
                action_dim=2,
                camera_roles=("primary",),
                action_schema=("x", "y"),
            )
        )

    def predict_native(self, request):
        return ActionModelPrediction(
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
        "proprio": np.zeros(2, np.float32),
        "instruction": "move",
        "embodiment": "test_arm",
    }


def test_facade_rejects_native_action_dimension_that_breaks_capabilities() -> None:
    facade = _Facade(np.zeros((3, 3), np.float32))

    with pytest.raises(ActionModelProtocolError, match="advertised action_dim"):
        facade.predict(_request())
