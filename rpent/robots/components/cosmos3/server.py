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

"""Expose an existing Cosmos3 SGLang service through RPent's WAM routes."""

from __future__ import annotations

import argparse
from typing import Any

from rpent.robots.components.am_facade_base import BaseWAMFacade
from rpent.robots.components.cosmos3.adapter import ADAPTERS, make_adapter


class Cosmos3Facade(BaseWAMFacade):
    """Reuse RLinf's Cosmos3 policy transforms and synchronous HTTP transport."""

    def __init__(
        self, policy: Any, http_client: Any, *, platform: str, checkpoint: str
    ) -> None:
        model = policy.model_cfg
        if (model.raw_action_dim, model.action_dim, model.action_normalization) != (
            10,
            7,
            "quantile_rot",
        ):
            raise ValueError(
                "Cosmos3 LIBERO requires rot6d10 to axis-angle7 with quantile_rot statistics"
            )
        if not model.get("action_stats_path") or int(model.num_action_chunks) <= 0:
            raise ValueError(
                "Cosmos3 requires action statistics and a positive action chunk size"
            )
        self._policy = policy
        self._http = http_client
        self._timeout = float(
            policy.cfg_rollout.get("sglang", {}).get("http_timeout_s", 600)
        )
        meta, adapter = make_adapter(platform, checkpoint, policy=policy)
        # Cover the native request budget plus encoding and RPC overhead.
        meta["prediction_timeout_s"] = self._timeout + 30.0
        super().__init__(meta, adapter=adapter)

    def predict_native(
        self, request: dict, *, session_id: str | None = None
    ) -> tuple[dict, dict]:
        """Send one environment request using RLinf's existing msgpack client."""
        response = self._http.post(
            self._policy.action_path,
            request["payload"],
            msgpack=True,
            timeout_s=self._timeout,
        )
        return response, request["state"]


def main() -> None:
    """Start an RPC worker; the SGLang process remains externally owned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=tuple(ADAPTERS), required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Identity of the model served by SGLang"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--sglang-endpoint", required=True)
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8118)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter
    from rlinf.utils.http_client import InferenceHTTPClient

    cfg = OmegaConf.load(args.config)
    if cfg.env.eval.env_type != args.platform:
        raise ValueError("Cosmos3 configuration must select the requested environment")
    client = InferenceHTTPClient(args.sglang_endpoint)
    if not client.health():
        raise RuntimeError(
            f"Cosmos3 SGLang service is not ready: {args.sglang_endpoint}"
        )
    facade = Cosmos3Facade(
        Cosmos3SGLangAdapter(cfg, rank=0),
        client,
        platform=args.platform,
        checkpoint=args.checkpoint,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
