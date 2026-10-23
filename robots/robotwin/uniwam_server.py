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


"""RPent RPC adapter for the official UniWAM RoboTwin deployment."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.robotwin.uniwam_client import ACTION_CHUNK_SIZE
from rpent.robots.components.vla_facade_base import BaseVLAFacade


class UniWAMFacade(BaseVLAFacade):
    """Use request-owned history; never carry episode state between requests."""

    def __init__(self, policy: Any, reset_policy: Callable[[Any], None]):
        self._policy = policy
        self._reset_policy = reset_policy
        super().__init__()

    def predict(self, obs: dict[str, Any], options: Any = None) -> np.ndarray:
        """Rebuild official policy inputs under the inherited RPC write lock."""
        if options is not None:
            raise ValueError("UniWAM predict does not accept options")
        history = np.asarray(obs["history"], dtype=np.float32)
        if (
            history.ndim != 2
            or history.shape[1] != 14
            or not 1 <= len(history) <= ACTION_CHUNK_SIZE
        ):
            raise ValueError("history must have shape [1..16, 14]")
        if not np.isfinite(history).all():
            raise ValueError("history must be finite")
        instruction = obs["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        self._reset_policy(self._policy)
        for state in history:
            self._policy.record_executed_qpos({"joint_action": {"vector": state}})
        self._policy.set_instruction(instruction)
        self._policy.update_obs(obs)
        return self._policy.get_action()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    policy_dir = args.policy_dir.expanduser().resolve()
    if not (policy_dir / "deploy_policy.py").is_file():
        raise FileNotFoundError(f"UniWAM deploy_policy.py not found in {policy_dir}")
    sys.path.insert(0, str(policy_dir))
    from deploy_policy import get_model, reset_model

    policy = get_model(
        {
            "ckpt_setting": args.model_path,
            "wan_path": args.wan_path,
            "vlm_path": args.vlm_path,
            "inference_mode": "history_flow",
            "num_inference_timesteps": 4,
        }
    )
    if policy.model.config.action_chunk_size != ACTION_CHUNK_SIZE:
        raise ValueError("This adapter requires the UniWAM RoboTwin 16-step checkpoint")
    UniWAMFacade(policy, reset_model).serve(
        transport="socket",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
