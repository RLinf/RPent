"""Thin client wrapping the Pi0.5 VLA RPC server.

The server lifecycle is the caller's responsibility: bring up
``robots/libero/vla_server.py`` (or any compatible ``predict`` /
``healthz`` implementation) before constructing this client.

Wire schema (see also ``vla_server.py``):

    call("predict", kwargs={
        "instruction": "<task_descriptions>",
        "images": {
            "main":  {"format": "png", "data": "<base64>"},
            "wrist": {"format": "png", "data": "<base64>"},  # optional
            "extra": {"format": "png", "data": "<base64>"},  # optional
        },
        "state": [[s0..sN]],           # shape [B, state_dim]
        "mode":  "eval",
    })
    -> {"actions": [[[a0..a6], ...]], "shape": [B, chunk, action_dim], "dtype": "float32"}
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient


def _png_b64(img: np.ndarray) -> str:
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLAClient:
    """Client wrapping a remote Pi0.5 VLA over any :class:`RpcClient` transport.

    Only call site is ``LiberoPrimitives``, which uses one method:
    ``predict_action_batch(env_obs, mode="eval")``.
    """

    def __init__(self, client: RpcClient):
        self._client = client

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self._client.call("healthz", timeout_s=timeout_s)

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
                images[wire_key] = {"format": "png", "data": _png_b64(arr)}

        states = np.asarray(env_obs["states"]).astype(np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        payload = self._client.call(
            "predict",
            kwargs={
                "instruction": env_obs.get("task_descriptions") or "",
                "images": images,
                # vla_server's wire still expects [B, state_dim]
                "state": [states.tolist()],
                "mode": mode,
            },
        )
        # Wire returns [B=1, chunk, action_dim]; strip B so callers see
        # [chunk, action_dim] without thinking in num_envs.
        actions = np.asarray(payload["actions"], dtype=np.float32)[0]
        return actions, {}
