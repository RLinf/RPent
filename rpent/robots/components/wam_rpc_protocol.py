# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""WAM message fields; the shared RPC transport serializes dictionaries and arrays."""

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from typing_extensions import NotRequired

from rpent.robots.components.wam_control_spec import WAMControlSpec


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


class WAMRequest(TypedDict):
    """Physical RGB and named state, with the requested execution identity."""

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str
    embodiment: str
    action_space: str


class WAMPrediction(TypedDict):
    """Actions are required; optional predictions are separate from simulator state."""

    actions: np.ndarray
    future_observation: NotRequired[Any]
    value: NotRequired[float | np.ndarray | None]
    metadata: NotRequired[dict[str, Any]]
