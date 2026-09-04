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

"""RPC server wrapping the dual-Franka TCP-rot6d Pi0.5 policy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger

logger = get_logger("dual_franka_vla_server")

# Resolve the RLinf checkout before the deferred ``import rlinf`` executes.
RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))


def build_model_cfg(model_path: str, repo_id: str) -> Any:
    """Build the RLinf OpenPI inference config for dual Franka."""
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 20,
            "action_dim": 20,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_dualfranka_tcp_rot6d",
                "num_images_in_input": 3,
                "noise_level": 0.5,
                "action_chunk": 20,
                "num_steps": 5,
                "train_expert_only": False,
                "action_env_dim": 20,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "use_dsrl": False,
            },
            "openpi_data": {"repo_id": repo_id},
        }
    )


class DualFrankaVLAFacade(BaseVLAFacade):
    """Serve one loaded dual-Franka Pi0.5 model over RPent RPC."""

    def __init__(self, model_path: str, repo_id: str):
        super().__init__()
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(model_path, repo_id)
        started_at = time.time()
        logger.info(
            "loading dual-Franka Pi0.5 (model_path=%s, repo_id=%s) ...",
            model_path,
            repo_id,
        )
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        logger.info("model ready in %.1fs", time.time() - started_at)

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        """Run one inference on the openpi wire obs produced by ``Pi05VLAClient``."""
        mode = (options or {}).get("mode", "eval")
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(obs, mode=mode)
        return (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PI05_CHECKPOINT_PATH"),
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("DUAL_FRANKA_REPO_ID"),
        help="SFT dataset repo ID used to locate norm_stats.json in the checkpoint",
    )
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    args = parser.parse_args()

    if not args.model_path:
        raise RuntimeError("provide --model-path or set PI05_CHECKPOINT_PATH")
    if not args.repo_id:
        raise RuntimeError("provide --repo-id or set DUAL_FRANKA_REPO_ID")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    facade = DualFrankaVLAFacade(args.model_path, args.repo_id)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
