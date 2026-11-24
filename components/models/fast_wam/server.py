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

import numpy as np

from components.models.fast_wam.adapter import ADAPTERS
from rpent.components.models.contracts import WAMPrediction
from rpent.components.models.facade import BaseWAMFacade


def _axis_angle(quaternion: np.ndarray) -> np.ndarray:
    # Match Fast-WAM/robosuite's quaternion branch, including negative w.
    # scipy's shortest-rotation convention differs for that branch.
    w = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    if denominator == 0:
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * np.arccos(w) / denominator)


class FastWAMLiberoFacade(BaseWAMFacade):
    """Bridge LIBERO observations to RLinf's policy, which owns the rest."""

    def __init__(self, model: Any, *, checkpoint: str) -> None:
        self._model = model
        self._processor = model.processor
        super().__init__(ADAPTERS["libero"](checkpoint=checkpoint, model=model))

    def encode(self, request: dict) -> dict[str, Any]:
        """Map physical observations to RLinf's batched LIBERO policy input."""
        state = request["state"]
        proprio = np.concatenate(
            (
                state["eef_position"],
                _axis_angle(state["eef_quaternion_xyzw"]),
                state["gripper_qpos"],
            )
        ).astype(np.float32)
        # Physical views have the vertical flip; RLinf's policy expects the
        # official 180-degree orientation. RLinf owns resizing and normalization.
        images = request["images"]
        return {
            "main_images": np.ascontiguousarray(images["primary"][:, ::-1])[None],
            "wrist_images": (
                np.ascontiguousarray(images["wrist"][:, ::-1])[None]
                if self._processor.num_output_cameras == 2
                else None
            ),
            "states": proprio[None],
            "task_descriptions": [request["instruction"]],
        }

    def decode(self, result: tuple[np.ndarray, dict]) -> WAMPrediction:
        """Unbatch executable actions already decoded and truncated by RLinf."""
        actions, metadata = result
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError("RLinf Fast-WAM must return one batched action chunk")
        return WAMPrediction(actions=actions[0], metadata=metadata)

    def predict_native(self, request: dict, *, session_id: str | None = None) -> Any:
        """Run RLinf's LIBERO policy pipeline on the encoded observations."""
        return self._model.predict_action_batch(request, mode="eval")


class FastWAMRoboTwinFacade(BaseWAMFacade):
    """Drive the official model with its three-camera, qpos14 input layout."""

    def __init__(
        self,
        model: Any,
        *,
        checkpoint: str,
        processor: Any,
        config: Any,
        prompt_template: str = "{task}",
    ) -> None:
        options = dict(config.get("inference_options", {}))
        if set(options) & {"prompt", "input_image", "proprio", "action_horizon"}:
            raise ValueError(
                "inference_options must not override encoded observations or action_horizon"
            )
        self._model = model
        self._processor = processor
        self._prompt_template = prompt_template
        self._horizon = config.action_horizon
        self._execute_steps = config.execute_steps
        self._inference_options = {**options, "action_horizon": self._horizon}
        super().__init__(
            ADAPTERS["robotwin"](
                checkpoint=checkpoint, processor=processor, config=config
            )
        )
        # The contract above already required one merged state and action key.
        self._state_key = processor.shape_meta["state"][0]["key"]
        action_key = processor.shape_meta["action"][0]["key"]
        self._normalizer = processor.normalizer.normalizers["action"][action_key]

    def encode(self, request: dict) -> dict[str, Any]:
        """Match upstream's head-over-two-wrists layout and joint_action.vector."""
        import torch
        from PIL import Image

        images = request["images"]
        resized = [
            np.asarray(
                Image.fromarray(images[role]).resize(size, Image.Resampling.BILINEAR)
            )
            for role, size in (
                ("head", (320, 256)),
                ("left_wrist", (160, 128)),
                ("right_wrist", (160, 128)),
            )
        ]
        image = np.concatenate(
            (resized[0], np.concatenate(resized[1:], axis=1)), axis=0
        )
        state = torch.tensor(request["state"]["joint_targets"], dtype=torch.float32)
        batch = self._processor.action_state_transform(
            {"state": {self._state_key: state[None]}}
        )
        batch = self._processor.normalizer.forward(batch)
        pixels = (
            torch.tensor(image)
            .permute(2, 0, 1)[None]
            .to(device=self._model.device, dtype=self._model.torch_dtype)
        )
        return {
            "prompt": self._prompt_template.format(task=request["instruction"]),
            "input_image": pixels * (2.0 / 255.0) - 1.0,
            "proprio": batch["state"][self._state_key],
        }

    def decode(self, result: dict) -> WAMPrediction:
        """Return the official 14D qpos vector unchanged after denormalization."""
        import torch

        action = result["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if tuple(action.shape) != (1, self._horizon, 14):
            raise ValueError(f"Fast-WAM actions must have [1,{self._horizon},14] shape")
        decoded = self._normalizer.backward(
            action.detach().to(dtype=torch.float32, device="cpu")
        )
        actions = decoded[0].numpy().astype(np.float32)
        if actions.shape != (self._horizon, 14):
            raise ValueError("Fast-WAM action normalizer changed the execution shape")
        # Check before prefix selection can hide invalid values.
        if not np.isfinite(actions).all():
            raise ValueError("Fast-WAM actions contain NaN or Inf")
        return WAMPrediction(actions=actions[: self._execute_steps])

    def predict_native(self, request: dict, *, session_id: str | None = None) -> Any:
        """Call the official inference entry point on the encoded batch."""
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
        facade = FastWAMLiberoFacade(model, checkpoint=args.checkpoint)
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
        facade = FastWAMRoboTwinFacade(
            model,
            checkpoint=args.checkpoint,
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
