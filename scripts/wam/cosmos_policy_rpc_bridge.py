#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Expose the official Cosmos Policy LIBERO checkpoint through RPent RPC.

Run this file inside the official ``cosmos-policy`` environment with the RPent
checkout on ``PYTHONPATH``. Model preprocessing and normalization remain owned
by the upstream implementation.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import numpy as np

from rpent.robots.components.action_model_facade_base import BaseActionModelFacade
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
)

ACTION_SCHEMA = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)
PROPRIO_SCHEMA = (
    "gripper_qpos_0",
    "gripper_qpos_1",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "eef_quat_w",
)


def resolve_cosmos_checkpoint(checkpoint: str) -> str:
    """Resolve a local Cosmos policy directory to its model file.

    The official LIBERO download stores the policy weights beside the dataset
    statistics and T5 embeddings.  The upstream model loader, however,
    requires the concrete ``.pt`` file path.
    """
    if not os.path.isdir(checkpoint):
        return checkpoint
    preferred = os.path.join(checkpoint, "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    if os.path.isfile(preferred):
        return preferred
    candidates = sorted(
        os.path.join(checkpoint, name)
        for name in os.listdir(checkpoint)
        if name.endswith(".pt") and os.path.isfile(os.path.join(checkpoint, name))
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cosmos checkpoint directory {checkpoint!r} must contain "
        "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    )


def normalize_cosmos_actions(actions: Any) -> np.ndarray:
    """Normalize the upstream action list/array to a finite float32 matrix."""
    normalized = np.asarray(actions, dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != len(ACTION_SCHEMA):
        raise ActionModelProtocolError(
            f"Cosmos Policy actions must have [T, 7] shape, got {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ActionModelProtocolError("Cosmos Policy actions contain NaN or Inf")
    return normalized


class CosmosPolicyBridge(BaseActionModelFacade):
    """Load and serve NVIDIA's Predict2 2B LIBERO policy."""

    def __init__(self, checkpoint: str) -> None:
        try:
            from cosmos_policy.experiments.robot.cosmos_utils import (
                get_action,
                get_model,
                init_t5_text_embeddings_cache,
                load_dataset_stats,
                t5_text_embeddings_cache,
            )
            from cosmos_policy.experiments.robot.libero.run_libero_eval import (
                PolicyEvalConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Cosmos Policy is not installed; run this bridge inside the "
                "official cosmos-policy LIBERO environment"
            ) from exc

        checkpoint_root = (
            os.path.dirname(checkpoint) if os.path.isfile(checkpoint) else checkpoint
        )
        cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=resolve_cosmos_checkpoint(checkpoint),
            config_file="cosmos_policy/config/config.py",
            dataset_stats_path=f"{checkpoint_root}/libero_dataset_statistics.json",
            t5_text_embeddings_path=f"{checkpoint_root}/libero_t5_embeddings.pkl",
            use_wrist_image=True,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            chunk_size=16,
            num_open_loop_steps=16,
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            flip_images=True,
            num_denoising_steps_action=5,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
        )
        dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        model, _cosmos_config = get_model(cfg)
        self._cfg = cfg
        self._model = model
        self._dataset_stats = dataset_stats
        self._get_action = get_action
        self._t5_text_embeddings_cache = t5_text_embeddings_cache
        super().__init__(
            ActionModelCapabilities(
                backend="cosmos_policy",
                checkpoint=checkpoint,
                supported_embodiments=("libero_7d",),
                action_dim=7,
                camera_roles=("primary", "wrist"),
                action_schema=ACTION_SCHEMA,
                proprio_schema=PROPRIO_SCHEMA,
                returns_future_observation=True,
                returns_value=True,
                metadata={"model_family": "Cosmos Policy Predict2 2B"},
            )
        )

    def predict_native(self, request: dict[str, Any]) -> ActionModelPrediction:
        instruction = request["instruction"]
        if instruction not in self._t5_text_embeddings_cache:
            raise ActionModelProtocolError(
                f"instruction {instruction!r} is not present in the precomputed "
                "T5 embedding cache; this bridge does not load T5-11B online"
            )
        wrist = request["images"]["wrist"]
        if wrist is None:
            raise ValueError("Cosmos Policy LIBERO requires a wrist image")
        schema = tuple(request["metadata"].get("proprio_schema", ()))
        if schema != PROPRIO_SCHEMA:
            raise ActionModelProtocolError(
                "Cosmos Policy LIBERO requires its explicit 9-field proprio_schema"
            )
        proprio = np.asarray(request["proprio"], dtype=np.float32)
        if proprio.shape != (len(PROPRIO_SCHEMA),):
            raise ActionModelProtocolError(
                f"Cosmos Policy LIBERO proprio must have shape "
                f"({len(PROPRIO_SCHEMA)},), got {proprio.shape}"
            )
        primary = request["images"]["primary"]
        if self._cfg.flip_images:
            primary = np.flipud(primary)
            wrist = np.flipud(wrist)
        observation = {
            "primary_image": primary,
            "wrist_image": wrist,
            "proprio": proprio,
        }
        started = time.monotonic()
        result = self._get_action(
            self._cfg,
            self._model,
            self._dataset_stats,
            observation,
            instruction,
            num_denoising_steps_action=self._cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
        )
        return ActionModelPrediction(
            actions=normalize_cosmos_actions(result["actions"]),
            future_observation=result.get("future_image_predictions"),
            value=result.get("value_prediction"),
            metadata={
                "backend": "cosmos_policy",
                "checkpoint": self._capabilities.checkpoint,
                "embodiment": request["embodiment"],
                "action_schema": list(ACTION_SCHEMA),
                "inference_seconds": time.monotonic() - started,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RPent Cosmos Policy RPC bridge")
    parser.add_argument(
        "--checkpoint", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    )
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    CosmosPolicyBridge(args.checkpoint).serve(
        transport=args.transport, host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
