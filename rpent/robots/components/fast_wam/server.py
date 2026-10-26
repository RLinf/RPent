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
from pathlib import Path
from typing import Any

from rpent.robots.components.am_facade_base import BaseWAMFacade
from rpent.robots.components.fast_wam.adapter import ADAPTERS, make_adapter


class FastWAMFacade(BaseWAMFacade):
    """Bridge platform observations to model inference without a simulator."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        checkpoint: str,
        platform: str,
        action_horizon: int,
        execute_steps: int,
        video_size: tuple[int, int],
        concat: str,
        prompt_template: str,
        inference_options: dict[str, Any] | None = None,
        binarize_gripper: bool = False,
    ) -> None:
        options = dict(inference_options or {})
        if set(options) & {"prompt", "input_image", "proprio", "action_horizon"}:
            raise ValueError(
                "inference_options must not override encoded observations or action_horizon"
            )
        self._model = model
        self._platform = platform
        self._inference_options = {**options, "action_horizon": action_horizon}
        model_meta, adapter = make_adapter(
            platform,
            checkpoint,
            processor=processor,
            device=model.device,
            dtype=model.torch_dtype,
            prompt_template=prompt_template,
            video_size=video_size,
            concat=concat,
            action_horizon=action_horizon,
            execute_steps=execute_steps,
            binarize_gripper=binarize_gripper,
        )
        super().__init__(model_meta, adapter=adapter)

    def predict_native(self, request: dict, *, session_id: str | None = None) -> Any:
        """Use RLinf's LIBERO policy pipeline or RoboTwin's native inference."""
        if self._platform == "libero":
            return self._model.predict_action_batch(request, mode="eval")
        import torch

        with torch.no_grad():
            return self._model.infer_action(**request, **self._inference_options)


def load_libero_policy(
    config_path: str, cfg: Any, checkpoint: str, stats: str, dtype: Any
) -> Any:
    """Load RLinf's policy while preserving the resolved RPent worker YAML."""
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.fastwam import get_model

    options = dict(cfg.get("inference_options", {}))
    # RLinf's rollout API does not expose upstream's optional compiled path.
    if options.pop("compile_action_infer", False):
        raise ValueError("RLinf Fast-WAM does not support compile_action_infer")
    supported = {
        "num_inference_steps",
        "text_cfg_scale",
        "negative_prompt",
        "sigma_shift",
        "seed",
        "rand_device",
        "tiled",
    }
    if unknown := options.keys() - supported:
        raise ValueError(
            f"Unsupported RLinf Fast-WAM inference options: {sorted(unknown)}"
        )
    path = Path(config_path).resolve()
    model_cfg = OmegaConf.create(
        {
            "model_path": checkpoint,
            "dataset_stats_path": stats,
            "action_horizon": cfg.action_horizon,
            "num_action_chunks": cfg.execute_steps,
            "binarize_gripper": cfg.get("binarize_gripper", False),
            "num_inference_steps": 20,
            **options,
            "fastwam": {
                "config_dir": str(path.parent),
                "config_name": path.name,
                "overrides": [
                    r"data.train.processor=\${processor}",
                    f"data.train.num_frames={cfg.action_horizon + 1}",
                    f"data.train.concat_multi_camera={cfg.concat}",
                ],
            },
        }
    )
    return get_model(model_cfg, torch_dtype=dtype).eval()


def main() -> None:
    """Load a resolved Fast-WAM model/processor config in its provisioned env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=tuple(ADAPTERS), required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved YAML containing model, processor and inference settings",
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
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    if args.platform == "libero":
        model = load_libero_policy(
            args.config,
            cfg,
            args.checkpoint,
            args.dataset_stats,
            getattr(torch, args.dtype),
        ).to(args.device)
        processor = model.processor
    else:
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
        processor,
        checkpoint=args.checkpoint,
        platform=args.platform,
        action_horizon=cfg.action_horizon,
        execute_steps=cfg.execute_steps,
        video_size=tuple(cfg.video_size),
        concat=cfg.concat,
        prompt_template=DEFAULT_PROMPT,
        inference_options=OmegaConf.to_container(
            cfg.get("inference_options", OmegaConf.create({})), resolve=True
        ),
        binarize_gripper=cfg.get("binarize_gripper", False),
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
