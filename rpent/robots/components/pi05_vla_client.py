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

_BEHAVIOR_ACTION_DIM = 23
_BEHAVIOR_RAW_PROPRIO_SEGMENTS: dict[str, slice] = {
    "left_arm": slice(158, 165),
    "left_gripper": slice(193, 195),
    "right_arm": slice(197, 204),
    "right_gripper": slice(232, 234),
    "trunk": slice(236, 240),
    "base": slice(253, 256),
}

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


def _encode_obs_behavior(env_obs: dict) -> dict:
    """BEHAVIOR/R1Pro single-env obs -> openpi batched wire obs."""

    if not isinstance(env_obs, dict):
        raise TypeError("BEHAVIOR observation must be a mapping")

    main = np.asarray(env_obs.get("main_images"))
    if main.ndim != 3 or main.shape[-1] != 3:
        raise ValueError(f"main_images must be [H,W,3], got {main.shape}")
    if main.dtype != np.uint8:
        raise TypeError(f"main_images must have dtype uint8, got {main.dtype}")

    wrists = np.asarray(env_obs.get("wrist_images"))
    if wrists.ndim != 4 or wrists.shape[0] != 2 or wrists.shape[-1] != 3:
        raise ValueError(f"wrist_images must be [2,H,W,3], got {wrists.shape}")
    if wrists.dtype != np.uint8:
        raise TypeError(f"wrist_images must have dtype uint8, got {wrists.dtype}")

    states = np.asarray(env_obs.get("states"), dtype=np.float32)
    if states.ndim != 1:
        raise ValueError(f"states must be [raw_proprio_dim], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("states contains NaN or infinity")
    _extract_behavior_policy_state(states)

    task_description = env_obs.get("task_descriptions")
    if isinstance(task_description, (list, tuple)):
        instruction = next(
            (
                item.strip()
                for item in task_description
                if isinstance(item, str) and item.strip()
            ),
            "",
        )
    else:
        instruction = str(task_description or "")

    return {
        "main_images": np.ascontiguousarray(main)[None],
        "wrist_images": np.ascontiguousarray(wrists)[None],
        "extra_view_images": None,
        "states": np.ascontiguousarray(states)[None],
        "task_descriptions": [instruction],
    }


def _validate_behavior_action_chunk(
    actions: Any, *, max_horizon: int | None = None
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != _BEHAVIOR_ACTION_DIM or array.shape[0] < 1:
        raise ValueError(
            f"BEHAVIOR actions must be [T,{_BEHAVIOR_ACTION_DIM}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("BEHAVIOR actions contain NaN or infinity")
    if max_horizon is not None and array.shape[0] > int(max_horizon):
        raise ValueError(
            f"BEHAVIOR action horizon {array.shape[0]} exceeds {int(max_horizon)}"
        )
    return array


def _extract_behavior_policy_state(raw_proprio: Any) -> np.ndarray:
    raw = np.asarray(raw_proprio, dtype=np.float32)
    if raw.ndim != 1 or raw.shape[0] < _BEHAVIOR_RAW_PROPRIO_SEGMENTS["base"].stop:
        raise ValueError(
            "raw R1Pro proprio must be a vector with at least "
            f"{_BEHAVIOR_RAW_PROPRIO_SEGMENTS['base'].stop} values, got {raw.shape}"
        )
    compact = np.concatenate(
        [
            raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["base"]],
            raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["trunk"]],
            raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["left_arm"]],
            raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["right_arm"]],
            np.asarray([raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["left_gripper"]].sum()]),
            np.asarray([raw[_BEHAVIOR_RAW_PROPRIO_SEGMENTS["right_gripper"]].sum()]),
        ]
    )
    if compact.shape != (_BEHAVIOR_ACTION_DIM,):
        raise ValueError(
            f"compact policy state must be [{_BEHAVIOR_ACTION_DIM}], got {compact.shape}"
        )
    if not np.isfinite(compact).all():
        raise ValueError("compact policy state contains NaN or infinity")
    return compact


# NOTE: an embodiment registered here must also exist in the server's
# ``PI05_EMBODIMENTS`` (and ``PI05_ROBOT_PLATFORMS`` if it sets ROBOT_PLATFORM);
# the two registries are kept in sync manually.
_ENCODE_OBS: dict[str, Any] = {
    "behavior": _encode_obs_behavior,
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
        if embodiment == "behavior":
            self._TIMEOUT_S = {**self._TIMEOUT_S, "predict": 600.0}

    # ---- obs encode (symmetric with server decode_obs_<name>) ----

    def encode_obs(self, env_obs: dict) -> dict:
        """Dispatch to the embodiment's encoder from ``_ENCODE_OBS``."""
        return _ENCODE_OBS[self._embodiment](env_obs)

    # ---- inference ----

    def predict(self, env_obs: dict, options: dict | None = None) -> np.ndarray:
        """Encode obs, request ``vla.predict``, strip batch dim, return ``[chunk, action_dim]``."""
        openpi_obs = self.encode_obs(env_obs)
        actions = np.asarray(super().predict(openpi_obs, options))
        if self._embodiment == "behavior":
            if actions.ndim != 3 or actions.shape[0] != 1:
                raise ValueError(
                    f"BEHAVIOR Pi0.5 actions must be [1,T,23], got {actions.shape}"
                )
            return _validate_behavior_action_chunk(actions[0])
        return actions[0]
