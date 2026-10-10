# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fast-WAM encoders matching the official LIBERO and RoboTwin evaluation paths."""

from typing import Any

import numpy as np
from PIL import Image


def _crop_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source = Image.fromarray(image)
    scale = max(width / source.width, height / source.height)
    source = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.BILINEAR,
    )
    left, top = (source.width - width) // 2, (source.height - height) // 2
    return np.asarray(source.crop((left, top, left + width, top + height)))


def _axis_angle(quaternion: np.ndarray) -> np.ndarray:
    # Match Fast-WAM/robosuite's quaternion branch, including negative w.
    # scipy's shortest-rotation convention differs for that branch.
    w = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    if denominator == 0:
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * np.arccos(w) / denominator)


def _model_input(
    request: dict,
    image: np.ndarray,
    state: np.ndarray,
    *,
    processor: Any,
    device: str,
    dtype: Any,
    prompt_template: str,
) -> dict[str, Any]:
    import torch

    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Fast-WAM requires one merged state key")
    key = state_meta[0]["key"]
    batch = {"state": {key: torch.tensor(state, dtype=torch.float32)[None]}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    pixels = torch.tensor(image).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
    return {
        "prompt": prompt_template.format(task=request["instruction"]),
        "input_image": pixels * (2.0 / 255.0) - 1.0,
        "proprio": batch["state"][key],
    }


def encode_libero(
    request: dict,
    *,
    processor: Any,
    device: str,
    dtype: Any,
    prompt_template: str,
    video_size: tuple[int, int],
    concat: str = "horizontal",
) -> dict[str, Any]:
    """Encode 1/2 cameras and position/axis-angle/gripper state for LIBERO."""
    count = processor.num_output_cameras
    if count not in (1, 2) or concat not in ("horizontal", "vertical"):
        raise ValueError(
            "LIBERO requires 1/2 cameras and horizontal/vertical concatenation"
        )
    metas = processor.shape_meta["images"]
    if len(metas) < count:
        raise ValueError("Fast-WAM image metadata omits required cameras")
    images = []
    for role, meta in zip(("primary", "wrist")[:count], metas):
        channels, height, width = meta["shape"]
        if channels != 3 or min(height, width) <= 0:
            raise ValueError("Fast-WAM camera shape must be positive [3,H,W]")
        # Physical images already have the raw vertical flip. Upstream rotates
        # raw LIBERO images 180 degrees, so only a horizontal flip remains.
        images.append(_crop_resize(request["images"][role][:, ::-1], height, width))
    image = np.concatenate(images, axis=1 if concat == "horizontal" else 0)
    if image.shape[:2] != tuple(video_size):
        raise ValueError(
            f"Fast-WAM camera layout does not match video_size={video_size}"
        )
    state = request["state"]
    proprio = np.concatenate(
        (
            state["eef_position"],
            _axis_angle(state["eef_quaternion_xyzw"]),
            state["gripper_qpos"],
        )
    ).astype(np.float32)
    return _model_input(
        request,
        image,
        proprio,
        processor=processor,
        device=device,
        dtype=dtype,
        prompt_template=prompt_template,
    )


def encode_robotwin(
    request: dict,
    *,
    processor: Any,
    device: str,
    dtype: Any,
    prompt_template: str,
    video_size: tuple[int, int],
    concat: str = "robotwin",
) -> dict[str, Any]:
    """Match upstream's head-over-two-wrists layout and joint_action.vector."""
    if (
        tuple(video_size) != (384, 320)
        or concat != "robotwin"
        or processor.num_output_cameras != 3
    ):
        raise ValueError(
            "RoboTwin requires three cameras in the [384,320] robotwin layout"
        )
    images = [
        np.asarray(
            Image.fromarray(request["images"][role]).resize(
                size, Image.Resampling.BILINEAR
            )
        )
        for role, size in (
            ("head", (320, 256)),
            ("left_wrist", (160, 128)),
            ("right_wrist", (160, 128)),
        )
    ]
    image = np.concatenate((images[0], np.concatenate(images[1:], axis=1)), axis=0)
    return _model_input(
        request,
        image,
        request["state"]["joint_targets"],
        processor=processor,
        device=device,
        dtype=dtype,
        prompt_template=prompt_template,
    )
