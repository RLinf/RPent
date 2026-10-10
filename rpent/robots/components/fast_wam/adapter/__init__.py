# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select matched Fast-WAM platform encoders, decoders and controller contracts."""

from functools import partial
from typing import Any

from robots.libero.control import LIBERO_OSC
from robots.robotwin.control import ROBOTWIN_QPOS
from robots.robotwin.observation import ROBOTWIN_CAMERAS
from rpent.robots.components.fast_wam import USES_SESSIONS
from rpent.robots.components.fast_wam.adapter.decode import (
    decode_libero,
    decode_robotwin,
)
from rpent.robots.components.fast_wam.adapter.encode import (
    encode_libero,
    encode_robotwin,
)
from rpent.robots.components.wam_facade_base import (
    WAMAdapter,
    WAMAdapterSpec,
)
from rpent.robots.components.wam_rpc_protocol import WAMCapabilities


def _libero_capabilities(
    *, processor: Any, binarize_gripper: bool, **common: Any
) -> WAMCapabilities:
    if processor.num_output_cameras not in (1, 2):
        raise ValueError("Fast-WAM LIBERO supports one or two cameras")
    return WAMCapabilities(
        control=LIBERO_OSC,
        camera_roles=("primary", "wrist")[: processor.num_output_cameras],
        state_schema={"eef_position": 3, "eef_quaternion_xyzw": 4, "gripper_qpos": 2},
        **common,
    )


def _robotwin_capabilities(
    *, processor: Any, binarize_gripper: bool, **common: Any
) -> WAMCapabilities:
    if processor.num_output_cameras != 3:
        raise ValueError("Fast-WAM RoboTwin requires three cameras")
    if binarize_gripper:
        raise ValueError("RoboTwin qpos does not use LIBERO gripper binarization")
    return WAMCapabilities(
        control=ROBOTWIN_QPOS,
        camera_roles=ROBOTWIN_CAMERAS,
        state_schema={"joint_targets": 14},
        **common,
    )


ADAPTERS = {
    "libero": WAMAdapterSpec(encode_libero, decode_libero, _libero_capabilities),
    "robotwin": WAMAdapterSpec(
        encode_robotwin, decode_robotwin, _robotwin_capabilities
    ),
}


def make_adapter(
    platform: str,
    checkpoint: str,
    *,
    processor: Any,
    device: str,
    dtype: Any,
    prompt_template: str,
    video_size: tuple[int, int],
    action_horizon: int,
    execute_steps: int,
    concat: str,
    binarize_gripper: bool = False,
) -> tuple[WAMCapabilities, WAMAdapter]:
    """Bind checkpoint preprocessing and a supported execution prefix."""
    if platform not in ADAPTERS:
        raise ValueError(f"unsupported Fast-WAM platform: {platform!r}")
    if (
        type(action_horizon) is not int
        or type(execute_steps) is not int
        or not 0 < execute_steps <= action_horizon
    ):
        raise ValueError(
            "Fast-WAM requires 0 < execute_steps <= action_horizon (integers)"
        )
    spec = ADAPTERS[platform]
    capabilities = spec.capabilities(
        processor=processor,
        binarize_gripper=binarize_gripper,
        backend="fast_wam",
        checkpoint=checkpoint,
        chunk_size=execute_steps,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
        metadata={"prediction_horizon": action_horizon},
    )
    return capabilities, (
        partial(
            spec.encode,
            processor=processor,
            device=device,
            dtype=dtype,
            prompt_template=prompt_template,
            video_size=video_size,
            concat=concat,
        ),
        partial(
            spec.decode,
            processor=processor,
            horizon=action_horizon,
            execute_steps=execute_steps,
            binarize_gripper=binarize_gripper,
        ),
    )
