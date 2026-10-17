# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared physical observations/control and WAM RPC field declarations.

VLA model payloads retain their native formats; physical observations can be
consumed by either VLA clients or WAM adapters before model-specific encoding.
"""

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from typing_extensions import NotRequired

# --- Shared platform observations and native control ---


class AMObservation(TypedDict):
    """Physical RGB views, named robot state and task instruction."""

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str


class AMControlSpec(TypedDict):
    """Describe field order and native execution mode, without implementing control.

    The versioned action_space identifies units, frame, scaling, gripper
    convention and control rate. action_type selects a native environment mode.
    """

    embodiment: str
    action_space: str
    action_schema: list[str]
    action_type: str | None


# --- WAM connection: worker capabilities and input requirements ---


class WAMCapabilities(TypedDict):
    """One worker's control contract and checkpoint input requirements."""

    backend: str
    checkpoint: str
    control: AMControlSpec
    camera_roles: Sequence[str]
    state_schema: dict[str, int]
    chunk_size: int | None
    uses_sessions: bool
    returns_future_observation: NotRequired[bool]
    returns_value: NotRequired[bool]
    metadata: NotRequired[dict[str, Any]]


# --- WAM request: physical observations and execution identity ---


class WAMRequest(AMObservation):
    """Physical RGB and named state, with the requested execution identity."""

    embodiment: str
    action_space: str


# --- WAM result: executable actions and optional model outputs ---


class WAMPrediction(TypedDict):
    """Actions are required; optional predictions are separate from simulator state."""

    actions: np.ndarray
    future_observation: NotRequired[Any]
    value: NotRequired[float | np.ndarray | None]
    metadata: NotRequired[dict[str, Any]]
