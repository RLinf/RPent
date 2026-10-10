# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Serve an official Fast-WAM model with paired platform adapters."""

from __future__ import annotations

import argparse
from typing import Any

from rpent.robots.components.fast_wam.adapter import ADAPTERS, make_adapter
from rpent.robots.components.wam_facade_base import BaseWAMFacade


class FastWAMFacade(BaseWAMFacade):
    """Use the official model and processor without importing a simulator."""

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
        self._inference_options = {**options, "action_horizon": action_horizon}
        capabilities, adapter = make_adapter(
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
        super().__init__(capabilities, adapter=adapter)

    def predict_native(self, request: dict, *, session_id: str | None = None) -> dict:
        """Invoke the official action-only inference API on encoded inputs."""
        import torch

        with torch.no_grad():
            return self._model.infer_action(**request, **self._inference_options)


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
