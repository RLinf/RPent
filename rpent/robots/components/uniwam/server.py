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

"""Serve the official UniWAM history-flow policy over the shared WAM RPC."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rpent.robots.components.am_facade_base import BaseWAMFacade
from rpent.robots.components.uniwam import ACTION_CHUNK_SIZE
from rpent.robots.components.uniwam.adapter import make_adapter
from rpent.robots.components.uniwam.runtime_config import validate_options


class UniWAMFacade(BaseWAMFacade):
    """Rebuild policy history per request under the inherited RPC write lock."""

    def __init__(
        self,
        policy: Any,
        reset_policy: Callable[[Any], None],
        *,
        checkpoint: str = "",
        platform: str = "robotwin",
    ):
        self._policy = policy
        self._reset_policy = reset_policy
        metadata, adapter = make_adapter(platform, checkpoint)
        super().__init__(metadata, adapter=adapter)

    def predict_native(
        self, obs: dict[str, Any], *, session_id: str | None = None
    ) -> np.ndarray:
        """Use executed history, never actions predicted by a previous request."""
        self._reset_policy(self._policy)
        for state in obs["history"]:
            self._policy.record_executed_qpos({"joint_action": {"vector": state}})
        self._policy.set_instruction(obs["instruction"])
        self._policy.update_obs(obs)
        return self._policy.get_action()

    def reset_native(self, *, session_id: str | None = None) -> None:
        """Discard the official policy cache at episode reset."""
        self._reset_policy(self._policy)


def main() -> None:
    """Load the official deployment in the selected model environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--platform", choices=("robotwin",), default="robotwin")
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8118)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    validate_options(argparse.Namespace(am_checkpoint=args.checkpoint))
    root = Path(args.checkpoint).expanduser().resolve()
    sys.path.insert(0, str((root / "policy").resolve()))
    from deploy_policy import get_model, reset_model

    policy = get_model(
        {
            "ckpt_setting": str((root / "checkpoint/pytorch_model").resolve()),
            "wan_path": str((root / "wan").resolve()),
            "vlm_path": str((root / "vlm").resolve()),
            "inference_mode": "history_flow",
            "num_inference_timesteps": 4,
        }
    )
    if policy.model.config.action_chunk_size != ACTION_CHUNK_SIZE:
        raise ValueError("This adapter requires the UniWAM RoboTwin 16-step checkpoint")
    UniWAMFacade(
        policy, reset_model, checkpoint=str(root), platform=args.platform
    ).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
