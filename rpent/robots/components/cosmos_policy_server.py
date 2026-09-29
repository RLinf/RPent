# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve NVIDIA's LIBERO Cosmos Policy in its own environment over RPent RPC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from rpent.robots.components.action_model_facade_base import BaseActionModelFacade
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
)
from rpent.robots.components.cosmos_policy_protocol import ACTION_SCHEMA, PROPRIO_SCHEMA
from rpent.utils.logging import get_logger

logger = get_logger("cosmos_policy_server")
DEFAULT_CHECKPOINT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"


def prepare_observation(request: dict[str, Any]) -> dict[str, np.ndarray]:
    """Match NVIDIA's LIBERO image orientation and 9D proprioception layout."""
    observation = {}
    for source, target in (
        ("primary", "primary_image"),
        ("wrist", "wrist_image"),
    ):
        image = np.asarray(request["images"][source])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"{source} must be a uint8 RGB image shaped [256, 256, 3]")
        observation[target] = np.ascontiguousarray(np.flipud(image))
    if tuple(request["metadata"].get("proprio_schema", ())) != PROPRIO_SCHEMA:
        raise ValueError("Cosmos Policy requires its explicit 9-field proprio_schema")
    proprio = np.asarray(request["proprio"], dtype=np.float32)
    if proprio.shape != (9,) or not np.isfinite(proprio).all():
        raise ValueError("Cosmos Policy proprio must be a finite vector of length 9")
    observation["proprio"] = proprio
    return observation


def resolve_cosmos_checkpoint(checkpoint: str) -> str:
    """Resolve the official local download directory to its policy weights."""
    root = Path(checkpoint).expanduser()
    if not root.is_dir():
        return str(root)
    weights = root / "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing Cosmos Policy checkpoint: {weights}")
    return str(weights)


class CosmosPolicyFacade(BaseActionModelFacade):
    """Own the official LIBERO policy; serialize inference through the RPC lock."""

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        dataset_stats: str | None = None,
        text_embeddings: str | None = None,
        denoising_steps: int = 5,
        seed: int = 1,
        predict_future: bool = False,
        cached_instructions_only: bool = False,
    ) -> None:
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        # Cosmos pins its own Torch/Transformers stack. Keep imports in its worker.
        from cosmos_policy.experiments.robot import cosmos_utils
        from cosmos_policy.experiments.robot.libero.run_libero_eval import (
            PolicyEvalConfig,
        )
        from cosmos_policy.utils.utils import set_seed_everywhere

        checkpoint = resolve_cosmos_checkpoint(checkpoint)
        root = (
            str(Path(checkpoint).parent) if Path(checkpoint).is_file() else checkpoint
        )
        dataset_stats = dataset_stats or f"{root}/libero_dataset_statistics.json"
        text_embeddings = text_embeddings or f"{root}/libero_t5_embeddings.pkl"
        self._predict_future = predict_future
        self._cached_instructions_only = cached_instructions_only
        self._cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=checkpoint,
            dataset_stats_path=dataset_stats,
            t5_text_embeddings_path=text_embeddings,
            num_denoising_steps_action=denoising_steps,
            seed=seed,
        )
        set_seed_everywhere(seed)
        self._dataset_stats = cosmos_utils.load_dataset_stats(dataset_stats)
        cosmos_utils.init_t5_text_embeddings_cache(text_embeddings)
        self._text_embeddings = cosmos_utils.t5_text_embeddings_cache
        self._model, config = cosmos_utils.get_model(self._cfg)
        if config.dataloader_train.dataset.chunk_size != self._cfg.chunk_size:
            raise ValueError("Cosmos Policy checkpoint must use 16-action chunks")
        self._get_action = cosmos_utils.get_action
        super().__init__(
            ActionModelCapabilities(
                backend="cosmos_policy",
                checkpoint=checkpoint,
                supported_embodiments=("libero_7d",),
                action_dim=7,
                camera_roles=("primary", "wrist"),
                action_schema=ACTION_SCHEMA,
                proprio_schema=PROPRIO_SCHEMA,
                returns_future_observation=predict_future,
                returns_value=predict_future,
                metadata={
                    "chunk_size": 16,
                    "cached_instructions_only": cached_instructions_only,
                },
            )
        )

    def predict_native(self, request: dict) -> ActionModelPrediction:
        """Generate one native 16x7 action chunk using the official policy API."""
        instruction = request["instruction"]
        if self._cached_instructions_only and instruction not in self._text_embeddings:
            raise ValueError(
                "instruction is not present in the precomputed T5 embedding cache"
            )
        observation = prepare_observation(request)
        result = self._get_action(
            self._cfg,
            self._model,
            self._dataset_stats,
            observation,
            instruction,
            seed=self._cfg.seed,
            num_denoising_steps_action=self._cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=self._predict_future,
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (16, 7):
            raise ValueError("Cosmos Policy must return actions shaped [16, 7]")
        return ActionModelPrediction(
            actions=actions,
            future_observation=result.get("future_image_predictions")
            if self._predict_future
            else None,
            value=result.get("value_prediction") if self._predict_future else None,
            metadata={
                "backend": "cosmos_policy",
                "checkpoint": self._capabilities.checkpoint,
                "embodiment": "libero_7d",
                "action_schema": list(ACTION_SCHEMA),
            },
        )


def main() -> None:
    """Start the model service inside the official Cosmos Policy environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-stats",
    )
    parser.add_argument("--text-embeddings")
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--predict-future", action="store_true")
    parser.add_argument("--cached-instructions-only", action="store_true")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    args = parser.parse_args()
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["DETERMINISTIC"] = "True"
    facade = CosmosPolicyFacade(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embeddings=args.text_embeddings,
        denoising_steps=args.denoising_steps,
        seed=args.seed,
        predict_future=args.predict_future,
        cached_instructions_only=args.cached_instructions_only,
    )
    logger.info(
        "Cosmos Policy ready on %s://%s:%s", args.transport, args.host, args.port
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
