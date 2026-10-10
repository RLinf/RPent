# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""WAM control and RPC field declarations, using plain dictionaries and arrays."""

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from typing_extensions import NotRequired

# --- Platform control: native action semantics ---


class WAMControlSpec(TypedDict):
    """Describe field order and native execution mode, without implementing control.

    The versioned action_space identifies units, frame, scaling, gripper
    convention and control rate. action_type selects a native environment mode.
    """

    embodiment: str
    action_space: str
    action_schema: list[str]
    action_type: str | None


# --- Connection: worker capabilities and input requirements ---


class WAMCapabilities(TypedDict):
    """One worker's control contract and checkpoint input requirements."""

    backend: str
    checkpoint: str
    control: WAMControlSpec
    camera_roles: Sequence[str]
    state_schema: dict[str, int]
    chunk_size: int | None
    uses_sessions: bool
    returns_future_observation: NotRequired[bool]
    returns_value: NotRequired[bool]
    metadata: NotRequired[dict[str, Any]]


# --- Prediction request: physical observations and execution identity ---


class WAMRequest(TypedDict):
    """Physical RGB and named state, with the requested execution identity."""

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str
    embodiment: str
    action_space: str


# --- Prediction result: executable actions and optional model outputs ---


class WAMPrediction(TypedDict):
    """Actions are required; optional predictions are separate from simulator state."""

    actions: np.ndarray
    future_observation: NotRequired[Any]
    value: NotRequired[float | np.ndarray | None]
    metadata: NotRequired[dict[str, Any]]
