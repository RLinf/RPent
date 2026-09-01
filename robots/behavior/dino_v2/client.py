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

"""RPC client for the optional BEHAVIOR DINOv2 component."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION, l2_normalize_row
from rpent.utils.rpc import RpcClient


class BehaviorDinoClient:
    """Small checked RPC wrapper around a DINOv2 encoder service."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._close_lock = threading.Lock()
        self._transport_closed = False
        meta = self.healthz()
        if expected_meta:
            mismatches = {
                key: {"expected": expected, "actual": meta.get(key)}
                for key, expected in expected_meta.items()
                if meta.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"dino_meta mismatch: {mismatches!r}")
        self.server_meta = dict(meta)

    def _call(self, method: str, **kwargs: Any) -> Any:
        return self._client.call(method, kwargs=kwargs, timeout_s=120.0)

    def healthz(self) -> dict[str, Any]:
        payload = self._client.call("healthz", timeout_s=5.0)
        if not isinstance(payload, dict):
            raise TypeError("dino healthz must return a mapping")
        if payload.get("dimension") != DINOV2_DIMENSION:
            raise RuntimeError("DINO service dimension does not match CLS384")
        return payload

    def encode_batch(
        self, images: list[np.ndarray | None]
    ) -> tuple[np.ndarray | None, ...]:
        payload = self._call("dino.encode_batch", images=images)
        if not isinstance(payload, list):
            raise TypeError("dino.encode_batch must return a list")
        result: list[np.ndarray | None] = []
        for index, item in enumerate(payload):
            if item is None:
                result.append(None)
                continue
            result.append(l2_normalize_row(item, path=f"dino.output[{index}]"))
        return tuple(result)

    def close_transport(self) -> None:
        with self._close_lock:
            if self._transport_closed:
                return
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._transport_closed = True


__all__ = ["BehaviorDinoClient"]
