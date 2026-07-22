"""HTTP RPC client for the Pi0.5 VLA `/predict` server.

The server lifecycle is the caller's responsibility: bring up
``robots/libero/vla_server.py`` (or any compatible ``/predict``
implementation) before constructing this client.

Wire format (see also ``vla_server.py``):

    POST {base_url}/predict
    {
      "instruction": "<task_descriptions>",
      "images": {
        "main":  {"format": "png", "data": "<base64>"},
        "wrist": {"format": "png", "data": "<base64>"},   # optional
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
import numpy as np


def _png_b64(img: np.ndarray) -> str:
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLAClient:
    """HTTP client wrapping a remote Pi0.5 VLA `/predict` server.

    Only call site is ``LiberoPrimitives``, which uses one method:
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
        main_images = np.asarray(env_obs["main_images"])
        if main_images.ndim != 3:
            raise ValueError(
                f"main_images expected shape [H,W,3]; got {main_images.shape}"
            )
        images: dict[str, Any] = {
            "main": {"format": "png", "data": _png_b64(main_images)},
        }
        for src_key, wire_key in (("wrist_images", "wrist"), ("extra_view_images", "extra")):
            view = env_obs.get(src_key)
            if view is None:
                continue
            arr = np.asarray(view)
            if arr.size > 0 and arr.ndim == 3:
                images[wire_key] = {
                    "format": "png",
                    "data": _png_b64(arr),
                }

        instruction = env_obs.get("task_descriptions") or ""

        states = np.asarray(env_obs["states"]).astype(np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        body = {
            "instruction": instruction,
            "images": images,
            # vla_server's wire still expects [B, state_dim]
            "state": [states.tolist()],
            "mode": mode,
        }

        resp = self._client.post(f"{self._base_url}/predict", json=body)
        if resp.status_code != 200:
            try:
                payload = resp.json()
                detail = (
                    payload.get("detail")
                    or payload.get("error")
                    or payload
                )
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"VLA /predict failed (HTTP {resp.status_code}): {detail}"
            )
        payload = resp.json()
        # Wire returns [B=1, chunk, action_dim]; strip B so callers see
        # [chunk, action_dim] without thinking in num_envs.
        actions = np.asarray(payload["actions"], dtype=np.float32)[0]
        return actions, {}
