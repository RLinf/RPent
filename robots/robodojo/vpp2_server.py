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

"""RPent RPC adapter for the official RoboDojo VPP2 history8 policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpent.robots.components.vla_facade_base import BaseVLAFacade


def model_paths(model_path: str | Path | None) -> dict[str, Path]:
    """Resolve one resource root, retaining the official weight directory layout."""
    if not model_path:
        raise ValueError("Set --vla-model-path / VPP2_MODEL_PATH or --vla-endpoint")
    root = Path(model_path).expanduser().resolve()
    bundle = root / "checkpoints" / "joint2b_s100000"
    paths = {
        "policy_dir": root / "policy" / "src",
        "checkpoint_path": bundle / "action.pt",
        "video_checkpoint_path": bundle / "video.pt",
        "dataset_stats_path": bundle / "dataset_stats.json",
        "wan_model_dir": root / "checkpoints" / "Wan2.1-I2V-14B-480P",
    }
    for path in (*paths.values(), paths["policy_dir"] / "vpp2" / "policy.py"):
        if not path.exists():
            raise FileNotFoundError(f"VPP2 resource missing: {path}")
    return paths


class VPP2Facade(BaseVLAFacade):
    """Share weights, not episode caches; each request owns its history and seed."""

    def __init__(self, policy):
        self._policy = policy
        super().__init__()

    def predict(self, obs: dict, options=None) -> np.ndarray:
        """Use official preprocessing and inference under the RPC write lock."""
        import torch
        from vpp2.data import DEFAULT_PROMPT
        from vpp2.policy import _normalized_to_uint8, _uint8_to_normalized

        if options is not None:
            raise ValueError("VPP2 predict does not accept options")
        instruction = obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty subtask string")
        index = obs["inference_index"]
        if type(index) is not int or index < 0:
            raise ValueError("inference_index must be a nonnegative integer")
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("state must be finite EE16 proprioception")
        history = obs["history"]
        if len(history) != 9:
            raise ValueError("history must contain the anchor and eight sampled frames")
        policy = self._policy
        frames = []
        for views in history:
            if len(views) != 3:
                raise ValueError("each history frame requires head, left and right RGB")
            images = {}
            for name, rgb in zip(("head_camera", "left_camera", "right_camera"), views):
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                    raise ValueError("history images must be uint8 HWC RGB")
                images[name] = {"rgb": rgb}
            image = policy._compose_observation_image({"observation": images})
            frames.append(_normalized_to_uint8(image))
        # Official history passes through CPU uint8, but the latest image does not.
        condition = _uint8_to_normalized(frames).unsqueeze(0).to(image)
        seed = policy.base_seed + index
        with torch.no_grad():
            prediction = policy.model.infer_action(
                prompt=DEFAULT_PROMPT.format(task=instruction),
                input_image=image,
                action_horizon=policy.action_horizon,
                proprio=policy._normalize_state(state),
                condition_video=condition.contiguous(),
                negative_prompt=policy.negative_prompt,
                text_cfg_scale=policy.text_cfg_scale,
                num_inference_steps=policy.num_inference_steps,
                sigma_shift=policy.sigma_shift,
                seed=seed,
                video_seed=seed + policy.video_seed_offset,
                rand_device=policy.rand_device,
                tiled=policy.tiled,
                num_video_frames=policy.num_video_frames,
                return_one_step_video=False,
                return_full_video=False,
                full_video_num_inference_steps=None,
            )
        return policy._denormalize_action(prediction["action"])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    paths = model_paths(args.model_path)
    sys.path.insert(0, str(paths.pop("policy_dir")))
    from vpp2.policy import get_model

    policy = get_model({key: str(value) for key, value in paths.items()})
    VPP2Facade(policy).serve(
        transport="socket",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
