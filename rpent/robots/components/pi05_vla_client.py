"""Thin client wrapping the Pi0.5 VLA RPC server, used by LIBERO.

The server lifecycle is the caller's responsibility: bring up
``rpent.robots.components.pi05_vla_server`` (or any compatible ``predict`` /
``healthz`` implementation) before constructing this client.

Wire schema (see also ``pi05_vla_server``):

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

from rpent.robots.components.vla_client_base import BaseVLAClient


def _png_b64(img: np.ndarray) -> str:
    """Encode an RGB uint8 image to a base64-encoded PNG string."""
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_libero_obs(obs: dict[str, Any]) -> tuple[str, dict[str, Any], list]:
    """Extract Pi0.5 wire-protocol parameters from a LIBERO observation dict.

    Returns ``(instruction, images, state)`` suitable for passing to
    ``Pi05VLAClient._predict_protocol`` or directly to the Pi0.5 VLA server's
    RPC ``predict`` method.

    Other environments implement their own ``_encode_*`` function and reuse
    ``_predict_protocol``.
    """
    main_images = np.asarray(obs["main_images"])
    if main_images.ndim != 3:
        raise ValueError(
            f"main_images expected shape [H,W,3]; got {main_images.shape}"
        )
    images: dict[str, Any] = {
        "main": {"format": "png", "data": _png_b64(main_images)},
    }
    for src_key, wire_key in (("wrist_images", "wrist"), ("extra_view_images", "extra")):
        view = obs.get(src_key)
        if view is None:
            continue
        arr = np.asarray(view)
        if arr.size > 0 and arr.ndim == 3:
            images[wire_key] = {"format": "png", "data": _png_b64(arr)}

    states = np.asarray(obs["states"]).astype(np.float32)
    if states.ndim != 1:
        raise ValueError(
            f"states must be single-env shape [state_dim]; got {states.shape}"
        )
    return obs.get("task_descriptions") or "", images, [states.tolist()]


class Pi05VLAClient(BaseVLAClient):
    """Pi0.5 VLA client. ``predict(obs, mode)`` is the LIBERO entry point.

    ``_predict_protocol`` is the pure Pi0.5 wire-protocol layer — other
    environments can reuse it after encoding their own obs.
    """

    def predict(
        self,
        obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        instruction, images, state = _encode_libero_obs(obs)
        return self._predict_protocol(instruction, images, state, mode)

    def _predict_protocol(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Pure Pi0.5 wire-protocol: RPC call + response unpacking.

        Args:
            instruction: task description string.
            images: dict like ``{"main": {"format": "png", "data": "..."}}``.
            state: list of shape ``[[B, state_dim]]``.
            mode: ``"eval"`` or ``"train"``.

        Returns:
            ``(actions, info)`` where actions is ``[chunk, action_dim]``.
        """
        payload = self._client.call(
            "predict",
            kwargs={
                "instruction": instruction,
                "images": images,
                "state": state,
                "mode": mode,
            },
        )
        # Wire returns [B=1, chunk, action_dim]; strip B so callers see
        # [chunk, action_dim] without thinking in num_envs.
        actions = np.asarray(payload["actions"], dtype=np.float32)[0]
        return actions, {}