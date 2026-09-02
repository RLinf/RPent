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

"""Thin client wrapping the Pi0.5 VLA RPC server.

The server lifecycle is the caller's responsibility: bring up
``rpent.robots.components.pi05_vla_server`` (or any compatible ``vla.predict`` /
``healthz`` implementation) before constructing this client.

Embodiment-specific obs encoding is dispatched by the ``_ENCODE_OBS`` registry.
Add a new encoder function and register it there per embodiment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient

# ---------------------------------------------------------------------------
# Obs encoder registry
# ---------------------------------------------------------------------------


def _encode_obs_libero(env_obs: dict) -> dict:
    """LIBERO single-env obs → openpi batched wire obs.

    openpi expects ``main_images [B,H,W,3]``, ``wrist_images [B,H,W,3]``
    or ``None``, ``extra_view_images [B,H,W,3]`` or ``None``,
    ``states [B,state_dim] float32``, ``task_descriptions [str]``.
    Images are cast to ``uint8`` (openpi ``Normalize`` expects ``[0,255]``
    uint8 input) and validated to be single ``[H,W,3]`` views.
    """

    def _batch_view(v):
        if v is None:
            return None
        arr = np.asarray(v)
        if arr.ndim != 3:
            raise ValueError(f"expected [H,W,3] image, got shape {arr.shape}")
        return arr.astype(np.uint8)[None]

    main = np.asarray(env_obs["main_images"])
    if main.ndim != 3:
        raise ValueError(f"expected [H,W,3] image, got shape {main.shape}")
    return {
        "main_images": main.astype(np.uint8)[None],
        "wrist_images": _batch_view(env_obs.get("wrist_images")),
        "extra_view_images": _batch_view(env_obs.get("extra_view_images")),
        "states": np.asarray(env_obs["states"], dtype=np.float32)[None],
        "task_descriptions": [str(env_obs.get("task_descriptions") or "")],
    }


# NOTE: an embodiment registered here must also exist in the server's
# ``PI05_EMBODIMENTS`` (and ``PI05_ROBOT_PLATFORMS`` if it sets ROBOT_PLATFORM);
# the two registries are kept in sync manually.
_ENCODE_OBS: dict[str, Any] = {
    "libero": _encode_obs_libero,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Pi05VLAClient(BaseVLAClient):
    """Client wrapping a remote Pi0.5 VLA over any :class:`RpcClient` transport.

    Construction requires an ``embodiment`` name (e.g. ``"libero"``) that
    selects the obs encoder from ``_ENCODE_OBS``.
    """

    def __init__(self, client, *, embodiment: str):
        super().__init__(client)
        if embodiment not in _ENCODE_OBS:
            raise ValueError(
                f"unknown pi05 client embodiment: {embodiment!r}; "
                f"registered={list(_ENCODE_OBS)}"
            )
        self._embodiment = embodiment

    # ---- obs encode (symmetric with server decode_obs_<name>) ----

    def encode_obs(self, env_obs: dict) -> dict:
        """Dispatch to the embodiment's encoder from ``_ENCODE_OBS``."""
        return _ENCODE_OBS[self._embodiment](env_obs)

    # ---- inference ----

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Encode obs, request ``vla.predict``, strip batch dim, return ``[chunk, action_dim]``."""
        openpi_obs = self.encode_obs(env_obs)
        actions = super().predict(openpi_obs, options)
        return np.asarray(actions)[0]
