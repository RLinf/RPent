# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sensor names and policy inputs shared by the BEHAVIOR Env and model client."""

from __future__ import annotations

import numpy as np

CONTROL_DT_S = 1.0 / 30.0
CAMERA_SUFFIXES = {
    "head": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}


def valid_observation_token(token: object) -> bool:
    """Recognize an episode generation and its nonnegative frame index."""
    return (
        isinstance(token, dict)
        and token.keys() == {"generation", "observation_index"}
        and all(type(value) is int and value >= 0 for value in token.values())
    )


def confirmation_matches(info: dict, expected: dict, confirmed: int) -> bool:
    """Bind a step reply to the requested episode and its confirmed frame count."""
    token = info.get("observation_token")
    context = info.get("motion_context")
    return (
        valid_observation_token(token)
        and type(confirmed) is int
        and confirmed >= 0
        and token["generation"] == expected["generation"]
        and token["observation_index"] == expected["observation_index"] + confirmed
        and isinstance(context, dict)
        and context.get("current_token") == token
    )


def public_proprio(value, *, batched: bool = False) -> np.ndarray:
    """Validate the measured 61D R1Pro state without copying it."""
    state = np.asarray(value)
    expected = (1, 61) if batched else (61,)
    if (
        state.shape != expected
        or state.dtype.kind not in "fiu"
        or not np.isfinite(state).all()
    ):
        raise ValueError(
            f"expected finite real 61D R1Pro proprio with shape {expected}"
        )
    return state


def sensor_key(raw: dict, suffix: str) -> str | bytes:
    """Find one native or flattened sensor by its SDK suffix."""
    matches = [
        key
        for key in raw
        if (key.decode() if isinstance(key, bytes) else str(key)).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one sensor ending in {suffix!r}: {matches}")
    return matches[0]


def native_policy_observation(raw: dict, instruction: str | list[str] | None) -> dict:
    """Select the batched RLinf inputs for one R1Pro without repacking the cameras or joints."""
    if isinstance(instruction, list) and len(instruction) == 1:
        instruction = instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            "BEHAVIOR task_descriptions must contain one nonempty instruction"
        )
    result = {
        "states": public_proprio(raw["states"], batched=True),
        "task_descriptions": [instruction],
    }
    for key, batch in (("main_images", (1,)), ("wrist_images", (1, 2))):
        rgb = np.asarray(raw[key])
        if (
            rgb.shape[:-3] != batch
            or rgb.shape[-1] != 3
            or min(rgb.shape[-3:-1]) < 1
            or rgb.dtype != np.uint8
        ):
            raise ValueError(f"Expected uint8 RGB images for {key} with batch {batch}.")
        result[key] = rgb
    return result
