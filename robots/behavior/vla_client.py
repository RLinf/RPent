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

"""HTTP client for the BEHAVIOR Pi0.5 VLA sidecar."""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Mapping

import httpx
import numpy as np

from robots.behavior.policy_checkpoint import (
    PolicyCheckpointBinding,
    assert_matching_policy_checkpoint_binding,
)
from robots.behavior.schemas import extract_policy_state, validate_action_chunk


def _png_b64(img: np.ndarray) -> str:
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"image must be [H,W,3 or 4], got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, np.ascontiguousarray(arr), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class BehaviorVLAClient:
    """Client for a BEHAVIOR-compatible /predict endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 600.0,
        binding_id: str | None = None,
    ) -> None:
        self._base_url = str(base_url).rstrip("/")
        self._binding_id = str(binding_id) if binding_id is not None else None
        self._client = httpx.Client(
            timeout=timeout_s,
            trust_env=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=0),
        )

    @property
    def endpoint(self) -> str:
        return self._base_url

    def healthz(
        self,
        *,
        timeout_ms: int | None = None,
        expected_checkpoint_binding: (
            PolicyCheckpointBinding | Mapping[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if timeout_ms is not None:
            kwargs["timeout"] = timeout_ms / 1000.0
        response = self._client.get(f"{self._base_url}/healthz", **kwargs)
        response.raise_for_status()
        payload = response.json()
        if expected_checkpoint_binding is not None:
            assert_matching_policy_checkpoint_binding(
                payload.get("checkpoint_binding"),
                expected_checkpoint_binding,
            )
        return payload

    def wait_for_healthz(
        self,
        *,
        timeout_s: float = 600.0,
        poll_timeout_ms: int = 1000,
        expected_checkpoint_binding: (
            PolicyCheckpointBinding | Mapping[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        deadline = time.time() + float(timeout_s)
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                return self.healthz(
                    timeout_ms=poll_timeout_ms,
                    expected_checkpoint_binding=expected_checkpoint_binding,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(
            f"BEHAVIOR vla server not healthy after {timeout_s:.0f}s "
            f"(last error: {last_error})"
        )

    def disable_actions(self, *, timeout_ms: int = 5000) -> dict[str, Any]:
        body = (
            {"binding_id": self._binding_id} if self._binding_id is not None else None
        )
        response = self._client.post(
            f"{self._base_url}/control/disable-actions",
            json=body,
            timeout=max(float(timeout_ms) / 1000.0, 0.001),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("actions_enabled") is not False:
            raise RuntimeError(f"VLA server did not disable actions: {payload!r}")
        return payload

    def bind_actions(
        self, binding_id: str, *, timeout_ms: int = 5000
    ) -> dict[str, Any]:
        if not isinstance(binding_id, str) or not binding_id.strip():
            raise ValueError("binding_id must be a non-empty string")
        normalized = binding_id.strip()
        response = self._client.post(
            f"{self._base_url}/control/bind-actions",
            json={"binding_id": normalized},
            timeout=max(float(timeout_ms) / 1000.0, 0.001),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("actions_enabled") is not False:
            raise RuntimeError("VLA binding did not preserve disabled actions")
        self._binding_id = normalized
        return payload

    def enable_actions(self, *, timeout_ms: int = 5000) -> dict[str, Any]:
        body = (
            {"binding_id": self._binding_id} if self._binding_id is not None else None
        )
        response = self._client.post(
            f"{self._base_url}/control/enable-actions",
            json=body,
            timeout=max(float(timeout_ms) / 1000.0, 0.001),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("actions_enabled") is not True:
            raise RuntimeError(f"VLA server did not enable actions: {payload!r}")
        return payload

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        main = np.asarray(env_obs["main_images"])
        wrists = np.asarray(env_obs["wrist_images"])
        if main.ndim != 3:
            raise ValueError(f"main_images must be [H,W,3], got {main.shape}")
        if wrists.ndim != 4 or wrists.shape[0] != 2:
            raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")
        states = np.asarray(env_obs["states"], dtype=np.float32)
        if states.ndim != 1:
            raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
        extract_policy_state(states)
        body = {
            "instruction": str(env_obs.get("task_descriptions") or ""),
            "images": {
                "main": {"format": "png", "data": _png_b64(main)},
                "left_wrist": {"format": "png", "data": _png_b64(wrists[0])},
                "right_wrist": {"format": "png", "data": _png_b64(wrists[1])},
            },
            "state": [states.tolist()],
            "mode": mode,
            "binding_id": self._binding_id,
        }
        response = self._client.post(f"{self._base_url}/predict", json=body)
        if response.status_code != 200:
            try:
                payload = response.json()
                detail = payload.get("detail") or payload.get("error") or payload
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"BEHAVIOR VLA /predict failed (HTTP {response.status_code}): {detail}"
            )
        payload = response.json()
        action_batch = np.asarray(payload["actions"], dtype=np.float32)
        if action_batch.ndim != 3 or action_batch.shape[0] != 1:
            raise ValueError(
                "BEHAVIOR VLA response actions must be [1,T,23], "
                f"got {action_batch.shape}"
            )
        return validate_action_chunk(action_batch[0]), {
            "shape": payload.get("shape"),
            "dtype": payload.get("dtype"),
        }

    def close(self) -> None:
        self._client.close()


__all__ = ["BehaviorVLAClient"]
