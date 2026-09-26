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

"""Serve NVIDIA Cosmos Policy from its own Python environment over RPent RPC."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger

logger = get_logger("cosmos_policy_server")
DEFAULT_CHECKPOINT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"


def prepare_observation(raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Match NVIDIA's LIBERO image orientation and 9D proprioception layout."""
    observation = {}
    for source, target in (
        ("agentview_image", "primary_image"),
        ("robot0_eye_in_hand_image", "wrist_image"),
    ):
        image = np.asarray(raw_obs[source])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"{source} must be a uint8 RGB image shaped [256, 256, 3]")
        observation[target] = np.ascontiguousarray(np.flipud(image))
    parts = []
    for key, size in (
        ("robot0_gripper_qpos", 2),
        ("robot0_eef_pos", 3),
        ("robot0_eef_quat", 4),
    ):
        part = np.asarray(raw_obs[key], dtype=np.float32)
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"{key} must be a finite vector of length {size}")
        parts.append(part)
    observation["proprio"] = np.concatenate(parts)
    return observation


class CosmosPolicyFacade(BaseVLAFacade):
    """Own the official LIBERO policy; serialize inference through the RPC lock."""

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        dataset_stats: str = DEFAULT_CHECKPOINT + "/libero_dataset_statistics.json",
        text_embeddings: str = DEFAULT_CHECKPOINT + "/libero_t5_embeddings.pkl",
        denoising_steps: int = 5,
        seed: int = 1,
    ) -> None:
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        super().__init__()
        # Cosmos pins its own Torch/Transformers stack. Keep imports in its worker.
        from cosmos_policy.experiments.robot import cosmos_utils
        from cosmos_policy.experiments.robot.libero.run_libero_eval import (
            PolicyEvalConfig,
        )
        from cosmos_policy.utils.utils import set_seed_everywhere

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
        self._model, config = cosmos_utils.get_model(self._cfg)
        if config.dataloader_train.dataset.chunk_size != self._cfg.chunk_size:
            raise ValueError("Cosmos Policy checkpoint must use 16-action chunks")
        self._get_action = cosmos_utils.get_action

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        """Generate one native 16x7 action chunk using the official policy API."""
        if options and options != {"mode": "eval"}:
            raise ValueError("Cosmos Policy supports only evaluation options")
        instruction = obs.get("task_descriptions")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("task_descriptions must be a non-empty task instruction")
        observation = prepare_observation(obs)
        result = self._get_action(
            self._cfg,
            self._model,
            self._dataset_stats,
            observation,
            instruction,
            seed=self._cfg.seed,
            num_denoising_steps_action=self._cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        return np.asarray(result["actions"], dtype=np.float32)


def main() -> None:
    """Start the model service inside the official Cosmos Policy environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-stats",
        default=DEFAULT_CHECKPOINT + "/libero_dataset_statistics.json",
    )
    parser.add_argument(
        "--text-embeddings", default=DEFAULT_CHECKPOINT + "/libero_t5_embeddings.pkl"
    )
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
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
    )
    logger.info(
        "Cosmos Policy ready on %s://%s:%s", args.transport, args.host, args.port
    )
    facade.serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
