#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Proxy the official DreamZero WebSocket service through RPent RPC.

DreamZero-DROID advertises its native 8-D joint-position action contract. It is
intentionally not marked compatible with LIBERO's 7-D OSC action space.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np

from rpent.robots.components.action_model_facade_base import BaseActionModelFacade
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
)

DROID_PROPRIO_SCHEMA = tuple(
    [f"joint_position_{index}" for index in range(7)]
    + [f"cartesian_position_{index}" for index in range(6)]
    + ["gripper_position"]
)
DROID_ACTION_SCHEMA = tuple(
    [f"joint_position_{index}" for index in range(7)] + ["gripper_position"]
)


class DreamZeroBridge(BaseActionModelFacade):
    """Adapt DreamZero's official ``WebsocketClientPolicy`` interface."""

    def __init__(self, host: str, port: int, checkpoint: str) -> None:
        try:
            from eval_utils.policy_client import WebsocketClientPolicy
        except ImportError as exc:
            raise RuntimeError(
                "DreamZero is not installed; run this bridge inside the official "
                "DreamZero environment"
            ) from exc
        self._policy = WebsocketClientPolicy(host=host, port=port)
        server_metadata = self._policy.get_server_metadata()
        if not isinstance(server_metadata, dict):
            raise RuntimeError("DreamZero server returned invalid metadata")
        if server_metadata.get("n_external_cameras") != 2:
            raise RuntimeError("DreamZero-DROID must advertise two external cameras")
        if not server_metadata.get("needs_wrist_camera"):
            raise RuntimeError("DreamZero-DROID must advertise a wrist camera")
        if not server_metadata.get("needs_session_id"):
            raise RuntimeError("DreamZero-DROID must advertise session IDs")
        if server_metadata.get("action_space") != "joint_position":
            raise RuntimeError("DreamZero-DROID must advertise joint_position actions")
        super().__init__(
            ActionModelCapabilities(
                backend="dreamzero",
                checkpoint=checkpoint,
                supported_embodiments=("droid_joint_position_8d",),
                action_dim=8,
                camera_roles=("primary", "extra_0", "wrist"),
                action_schema=DROID_ACTION_SCHEMA,
                proprio_schema=DROID_PROPRIO_SCHEMA,
                metadata={"native_server": server_metadata},
            )
        )

    def predict_native(self, request: dict[str, Any]) -> ActionModelPrediction:
        metadata = request["metadata"]
        schema = tuple(metadata.get("proprio_schema", ()))
        if schema != DROID_PROPRIO_SCHEMA:
            raise ActionModelProtocolError(
                "DreamZero-DROID requires its explicit 14-field proprio_schema"
            )
        extra = request["images"]["extra"]
        wrist = request["images"]["wrist"]
        if len(extra) != 1 or wrist is None:
            raise ActionModelProtocolError(
                "DreamZero-DROID requires two external views and one wrist view"
            )
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        episode_id = metadata.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ActionModelProtocolError(
                "DreamZero-DROID requires a non-empty metadata.episode_id"
            )
        session_id = episode_id.strip()
        native_observation = {
            "observation/exterior_image_0_left": request["images"]["primary"],
            "observation/exterior_image_1_left": extra[0],
            "observation/wrist_image_left": wrist,
            "observation/joint_position": proprio[:7],
            "observation/cartesian_position": proprio[7:13],
            "observation/gripper_position": proprio[13:14],
            "prompt": request["instruction"],
            "session_id": session_id,
        }
        started = time.monotonic()
        actions = self._policy.infer(native_observation)
        return ActionModelPrediction(
            actions=actions,
            metadata={
                "backend": "dreamzero",
                "checkpoint": self._capabilities.checkpoint,
                "embodiment": request["embodiment"],
                "action_schema": list(DROID_ACTION_SCHEMA),
                "inference_seconds": time.monotonic() - started,
            },
        )

    def close(self) -> None:
        """Reset the native session so DreamZero can release cached rollout state."""
        try:
            self._policy.reset({})
        except Exception:
            logging.getLogger(__name__).exception("DreamZero reset failed during close")


def main() -> None:
    parser = argparse.ArgumentParser(description="RPent DreamZero RPC bridge")
    parser.add_argument("--dreamzero-host", default="127.0.0.1")
    parser.add_argument("--dreamzero-port", type=int, default=8000)
    parser.add_argument("--checkpoint", default="GEAR-Dreams/DreamZero-DROID")
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8121)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    DreamZeroBridge(args.dreamzero_host, args.dreamzero_port, args.checkpoint).serve(
        transport=args.transport, host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
