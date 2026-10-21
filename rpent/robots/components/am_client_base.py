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

"""Shared action-model transport and VLA/WAM client contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.am_contracts import (
    AMControlSpec,
    WAMModelMeta,
    WAMPrediction,
    WAMRequest,
)
from rpent.utils.rpc import RpcClient


class BaseAMClient:
    """Hold an RPC-compatible transport; subclasses own prediction contracts."""

    PREDICT_METHOD: str

    def __init__(self, client: Any, *, timeout_s: float) -> None:
        self._client = client
        self._timeout_s = float(timeout_s)

    def _predict(self, *args: Any) -> Any:
        """Forward the subclass's prediction arguments without conversion."""
        return self._client.call(
            self.PREDICT_METHOD, args=args, timeout_s=self._timeout_s
        )


class BaseVLAClient(BaseAMClient):
    """Preserve the VLA RPC contract for existing model clients."""

    PREDICT_METHOD = "vla.predict"
    _TIMEOUT_S: dict[str, float] = {"default": 30.0, "predict": 120.0}

    def __init__(self, client: Any) -> None:
        super().__init__(client, timeout_s=self._TIMEOUT_S["predict"])

    def predict(self, obs: Any, options: dict | None = None) -> Any:
        """Request a single VLA action chunk from observations and options."""
        return self._predict(obs, options)


class BaseWAMClient(BaseAMClient):
    """Client for ``wam.*`` RPCs, with metadata fetched once on connection."""

    PREDICT_METHOD = "wam.predict"

    def __init__(
        self,
        client: RpcClient,
        *,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(client, timeout_s=timeout_s)
        self.model_meta: WAMModelMeta = self._client.call(
            "wam.get_model_meta", timeout_s=30.0
        )

    def validate_contract(self, expected: AMControlSpec) -> None:
        """Require the platform's native control contract in cached metadata."""
        actual = self.model_meta["control"]
        if actual != expected:
            raise ValueError(
                f"WAM control mismatch: expected={expected!r}, actual={actual!r}"
            )

    def predict(self, request: WAMRequest) -> WAMPrediction:
        """Request actions and check their shape and values before execution."""
        model_meta = self.model_meta
        prediction = self._predict(request)
        actions = np.asarray(prediction["actions"])
        action_dim = len(model_meta["control"]["action_schema"])
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != action_dim:
            raise ValueError(
                f"WAM actions must have non-empty [chunk, {action_dim}] shape"
            )
        chunk_size = model_meta["chunk_size"]
        if chunk_size is not None and len(actions) != chunk_size:
            raise ValueError(f"WAM prediction must contain {chunk_size} actions")
        if actions.dtype.kind not in "iuf" or not np.isfinite(actions).all():
            raise ValueError("WAM actions must be numeric without NaN or Inf")
        prediction["actions"] = actions
        return prediction

    def reset(self) -> None:
        """Clear this session's model history before a new episode."""
        self._client.call("wam.reset", timeout_s=self._timeout_s)

    def close(self) -> None:
        """Release the transport session and its model state."""
        self._client.close()
