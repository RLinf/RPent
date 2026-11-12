# Copyright 2026 The RPent Authors.
"""Model selection for the opt-in RoboDojo component and CLI checks."""

import os

import pytest


@pytest.fixture
def vla_backend():
    backend = os.environ.get("RPENT_ROBODOJO_VLA_BACKEND", "vpp2")
    if backend not in {"vpp2", "awomo05"}:
        pytest.fail(f"Unknown RoboDojo VLA backend: {backend}")
    model_path = "VPP2_MODEL_PATH" if backend == "vpp2" else "AWOMO05_MODEL_PATH"
    if not (
        os.environ.get(model_path) or os.environ.get("RPENT_ROBODOJO_VLA_ENDPOINT")
    ):
        pytest.skip(f"requires {model_path} or RPENT_ROBODOJO_VLA_ENDPOINT")
    return backend
