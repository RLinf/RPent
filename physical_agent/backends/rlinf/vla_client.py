"""HTTP client for the Pi0.5 VLA `/predict` server.

Used by ``LiberoPrimitiveDriver`` in place of an in-process openpi model.
The server lifecycle is the caller's responsibility: bring up
``physical_agent.backends.rlinf.vla_server`` (or any compatible ``/predict``
implementation) before constructing this client.

Wire format (see also ``vla_server.py``):

    POST {base_url}/predict
    {
      "instruction": "<task_descriptions[0]>",
      "images": {
        "main":  {"format": "png", "data": "<base64>"},
        "extra": {"format": "png", "data": "<base64>"}    # optional
      },
      "state": [[s0..sN]],          # shape [B, state_dim]
      "mode":  "eval"
    }

    -> 200 OK
    {
      "actions": [[[a0..a6], ...]],  # shape [B, chunk, action_dim]
      "shape":   [B, chunk, action_dim],
      "dtype":   "float32"
    }
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import httpx
import imageio.v2 as imageio
import numpy as np
import torch


def _png_b64(img: np.ndarray) -> str:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class VLAClient:
    """HTTP client wrapping a remote Pi0.5 VLA `/predict` server.

    Only call site is ``LiberoPrimitiveDriver``, which uses one method:
    ``predict_action_batch(env_obs, mode="eval")``.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 60.0):
        # Strip trailing slashes so users can pass either ``http://h:p`` or
        # ``http://h:p/``; we always append ``/predict`` etc.
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)

    def healthz(self, *, timeout_ms: int | None = None) -> dict[str, Any]:
        """GET /healthz — used by callers polling for server readiness.

        ``timeout_ms`` overrides the per-call HTTP timeout; pass a short value
        when polling so each attempt fails fast.
        """
        kwargs: dict[str, Any] = {}
        if timeout_ms is not None:
            kwargs["timeout"] = timeout_ms / 1000.0
        resp = self._client.get(f"{self._base_url}/healthz", **kwargs)
        resp.raise_for_status()
        return resp.json()

    def wait_for_healthz(
        self, *, timeout_s: float = 300.0, poll_timeout_ms: int = 1000
    ) -> None:
        """Block until /healthz returns 200, or *timeout_s* elapses.

        Each probe uses ``healthz(timeout_ms=poll_timeout_ms)`` as the loop
        cadence — no extra sleep is needed since a connection refused / read
        timeout returns within ``poll_timeout_ms``.
        """
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                self.healthz(timeout_ms=poll_timeout_ms)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise TimeoutError(
            f"vla_server failed to become healthy within {timeout_s:.0f}s "
            f"(last error: {last_err})"
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        main_images = _to_numpy(env_obs["main_images"])
        if main_images.ndim != 4:
            raise ValueError(
                f"main_images expected shape [B,H,W,3]; got {main_images.shape}"
            )
        # LiberoPrimitiveDriver always runs with B=1 (num_envs=1). Encode the
        # first batch element. If batch>1 ever surfaces, callers should fan
        # out one request per item.
        images: dict[str, Any] = {
            "main": {"format": "png", "data": _png_b64(main_images[0])},
        }
        extra_view = env_obs.get("extra_view_images")
        if extra_view is not None:
            extra_arr = _to_numpy(extra_view)
            if extra_arr.size > 0 and extra_arr.ndim == 4:
                images["extra"] = {
                    "format": "png",
                    "data": _png_b64(extra_arr[0]),
                }

        task_descriptions = env_obs.get("task_descriptions") or [""]
        states = _to_numpy(env_obs["states"]).astype(np.float32)
        if states.ndim != 2:
            raise ValueError(
                f"states must be [B, state_dim]; got {states.shape}"
            )

        body = {
            "instruction": str(task_descriptions[0]),
            "images": images,
            "state": states.tolist(),
            "mode": mode,
        }

        resp = self._client.post(f"{self._base_url}/predict", json=body)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error")
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"VLA /predict failed (HTTP {resp.status_code}): {detail}"
            )
        payload = resp.json()
        actions = np.asarray(payload["actions"], dtype=np.float32)
        return actions, {}
