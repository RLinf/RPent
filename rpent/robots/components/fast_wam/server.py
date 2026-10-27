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

"""Serve RLinf's LIBERO policy or the official Fast-WAM RoboTwin model."""

from __future__ import annotations

import argparse
from typing import Any

from rpent.robots.components.am_facade_base import BaseWAMFacade
from rpent.robots.components.fast_wam.adapter import ADAPTERS, make_adapter


class FastWAMFacade(BaseWAMFacade):
    """Bridge platform observations to model inference without a simulator."""

    def __init__(
        self,
        model: Any,
        *,
        checkpoint: str,
        platform: str,
        processor: Any = None,
        config: Any = None,
        prompt_template: str = "{task}",
    ) -> None:
        options = (
            dict(config.get("inference_options", {})) if config is not None else {}
        )
        if set(options) & {"prompt", "input_image", "proprio", "action_horizon"}:
            raise ValueError(
                "inference_options must not override encoded observations or action_horizon"
            )
        self._model = model
        self._platform = platform
        self._inference_options = (
            {**options, "action_horizon": config.action_horizon}
            if platform == "robotwin"
            else {}
        )
        model_meta, adapter = make_adapter(
            platform,
            checkpoint,
            model=model,
            processor=processor,
            config=config,
            prompt_template=prompt_template,
        )
        super().__init__(model_meta, adapter=adapter)

    def predict_native(self, request: dict, *, session_id: str | None = None) -> Any:
        """Use RLinf's LIBERO policy pipeline or RoboTwin's native inference."""
        if self._platform == "libero":
            return self._model.predict_action_batch(request, mode="eval")
        import torch

        with torch.no_grad():
            return self._model.infer_action(**request, **self._inference_options)


def load_libero_policy(cfg: Any, checkpoint: str, stats: str, dtype: Any) -> Any:
    """Pass native RLinf model configuration through with deployment paths."""
    from rlinf.models.embodiment.fastwam import get_model

    if "execute_steps" in cfg or "inference_options" in cfg:
        raise ValueError(
            "Fast-WAM LIBERO requires RLinf model YAML: replace execute_steps with "
            "num_action_chunks and move inference_options to top-level model fields"
        )
    cfg.model_path = checkpoint
    cfg.dataset_stats_path = stats
    return get_model(cfg, torch_dtype=dtype).eval()


def main() -> None:
    """Load the platform's Fast-WAM configuration in its provisioned env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=tuple(ADAPTERS), required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="LIBERO: RLinf model YAML; RoboTwin: resolved model/processor YAML",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    if args.platform == "libero":
        model = load_libero_policy(
            cfg,
            args.checkpoint,
            args.dataset_stats,
            getattr(torch, args.dtype),
        ).to(args.device)
        facade = FastWAMFacade(
            model, checkpoint=args.checkpoint, platform=args.platform
        )
    else:
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )
        from hydra.utils import instantiate

        model = instantiate(
            cfg.model, model_dtype=getattr(torch, args.dtype), device=args.device
        )
        model.load_checkpoint(args.checkpoint)
        model = model.to(args.device).eval()
        processor = instantiate(cfg.processor).eval()
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(args.dataset_stats)
        )
        facade = FastWAMFacade(
            model,
            checkpoint=args.checkpoint,
            platform=args.platform,
            processor=processor,
            config=cfg,
            prompt_template=DEFAULT_PROMPT,
        )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
