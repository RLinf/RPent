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

"""APXInf VLA RPC server.

Wraps an APXInf :class:`~apxinf.policies.base.Policy` behind RPent's
``vla.predict`` RPC contract. The policy is loaded once at startup via
``apxinf_robo.presets.build_robot_policy``; each ``predict`` call
translates the RPent wire observation into the APXInf observation dict,
runs inference, and returns the deployable action array.

APXInf-robo, the APXInf engine, and ``apxinf_py`` must be importable from the
Python environment that runs this server. See the APXInf documentation for
build instructions.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from rpent.robots.components.am_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin

logger = get_logger("apxinf_vla_server")

# ---------------------------------------------------------------------------
# Embodiment registry: obs decode + policy construction parameters
# ---------------------------------------------------------------------------

# Each embodiment declares how to convert RPent wire obs (as produced by
# ``Pi05VLAClient``) into the APXInf observation dict, and
# which ``from_pretrained`` kwargs to use.
#
# RPent wire format (openpi convention, what the client sends):
#   {
#     "main_images":      [B, H, W, 3] uint8,
#     "wrist_images":     [B, H, W, 3] uint8 | None,
#     "extra_view_images":[B, N, H, W, 3] uint8 | None,
#     "states":           [B, state_dim] float32,
#     "task_descriptions":[str],
#   }
#
# APXInf observation format (what policy.infer expects):
#   {
#     <image_keys[0]>: [H, W, 3] uint8,
#     <image_keys[1]>: [H, W, 3] uint8,   (if num_views >= 2)
#     ...
#     "prompt": str,
#     <state_key>: [state_dim] float32,   (when the policy declares a state key)
#   }

APXINF_EMBODIMENTS: dict[str, dict[str, Any]] = {
    "libero": {
        "policy": "franka_libero",
        "action_chunk": 5,
        "action_dim": 7,
        # LIBERO pi05 checkpoint: 2 views (base + wrist), action_dim=7,
        # no state injection (openpi pi05_libero convention).
        "policy_kwargs": {
            "action_dim": 7,
            "discrete_state": False,
            "num_flow_steps": 5,
            "action_horizon": 10,
            "num_views": 2,
            "norm_dtype": "float64",
        },
        # Maps RPent wire keys to APXInf VIEW_SLOTS order.
        # index 0 -> base_0_rgb, index 1 -> left_wrist_0_rgb
        "view_sources": [("main_images", None), ("wrist_images", None)],
    },
    "dual_franka": {
        "policy": "dual_franka_tcp_rot6d",
        "action_chunk": 20,
        "action_dim": 20,
        "policy_kwargs": {"num_flow_steps": 5, "num_views": 3, "action_horizon": 20},
        "view_sources": [
            ("extra_view_images", 0),
            ("main_images", None),
            ("extra_view_images", 1),
        ],
    },
    "yam": {
        "policy": "yam_joint",
        "action_chunk": 30,
        "action_dim": 14,
        "policy_kwargs": {"num_flow_steps": 5, "num_views": 3, "action_horizon": 30},
        "view_sources": [
            ("main_images", None),
            ("extra_view_images", 0),
            ("extra_view_images", 1),
        ],
    },
}


def apxinf_format_decode(
    wire_obs: dict[str, Any], embodiment: str, policy
) -> dict[str, Any]:
    """Convert one batched RPent wire observation to an APXInf observation dict.

    Strips the leading batch dimension (B=1) from all arrays and renames
    image keys to the policy's configured ``image_keys``.
    """
    emb_cfg = APXINF_EMBODIMENTS[embodiment]
    view_sources = emb_cfg["view_sources"]
    image_keys = policy.image_keys

    apx_obs: dict[str, Any] = {}

    # Map each view slot to its source wire key.
    for image_key, (wire_key, view_index) in zip(image_keys, view_sources):
        value = wire_obs.get(wire_key)
        if value is None:
            continue
        arr = np.asarray(value)
        if view_index is not None:
            arr = arr[0, view_index]
        elif arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        apx_obs[image_key] = arr.astype(np.uint8)

    # Prompt
    descriptions = wire_obs.get("task_descriptions", [""])
    apx_obs[policy.prompt_key] = str(descriptions[0]) if descriptions else ""

    # State is also needed for action recovery when state tokens are disabled.
    if policy.state_key is not None:
        states = np.asarray(wire_obs.get("states", []), dtype=np.float32)
        if states.ndim == 2:
            states = states[0]
        apx_obs[policy.state_key] = states

    return apx_obs


# ---------------------------------------------------------------------------
# VLA facade
# ---------------------------------------------------------------------------


class ApxInfVLAFacade(MainThreadServeMixin, BaseVLAFacade):
    """VLA facade backed by an APXInf policy.

    Loads the policy via ``apxinf_robo.presets.build_robot_policy`` at startup and
    serves ``vla.predict`` by delegating to ``policy.infer``.

    Uses :class:`MainThreadServeMixin` so all RPC dispatch runs on the
    main thread, which is required because
    ``apxinf_py::PySentencePieceTokenizer`` is unsendable across threads.
    """

    def __init__(
        self,
        *,
        model_path: str,
        embodiment: str,
        precision: str = "bf16",
        device: str = "cuda:0",
        tokenizer_path: str | None = None,
        norm_stats_path: str | None = None,
        extra_policy_kwargs: dict[str, Any] | None = None,
    ):
        if embodiment not in APXINF_EMBODIMENTS:
            raise ValueError(
                f"unknown apxinf server embodiment: {embodiment!r}; "
                f"registered={list(APXINF_EMBODIMENTS)}"
            )
        self._embodiment = embodiment
        self._action_chunk = APXINF_EMBODIMENTS[embodiment]["action_chunk"]
        stats_path = (
            Path(norm_stats_path)
            if norm_stats_path
            else Path(model_path) / "norm_stats.json"
        )
        if not stats_path.is_file():
            raise FileNotFoundError(f"normalization statistics not found: {stats_path}")
        super().__init__()
        try:
            from apxinf_robo.presets import build_robot_policy
        except ImportError as exc:
            raise ImportError(
                "apxinf-robo is not importable. Ensure APXInf-robo, APXInf "
                "and the apxinf_py native binding are installed. "
                "See the APXInf documentation for build instructions."
            ) from exc

        emb_cfg = APXINF_EMBODIMENTS[embodiment]
        policy_kwargs = dict(emb_cfg["policy_kwargs"])
        if extra_policy_kwargs:
            policy_kwargs.update(extra_policy_kwargs)

        t0 = time.time()
        logger.info(
            "loading APXInf policy (embodiment=%s, model_path=%s, precision=%s) ...",
            embodiment,
            model_path,
            precision,
        )
        self._policy = build_robot_policy(
            emb_cfg["policy"],
            model_path,
            device=device,
            precision=precision,
            tokenizer_path=tokenizer_path,
            norm_stats=str(stats_path),
            **policy_kwargs,
        )
        if (
            self._policy.action_dim != emb_cfg["action_dim"]
            or self._policy.action_horizon != emb_cfg["policy_kwargs"]["action_horizon"]
            or len(self._policy.image_keys) != len(emb_cfg["view_sources"])
        ):
            self.close()
            raise ValueError(
                f"loaded policy does not match {embodiment} runtime contract"
            )
        logger.info("APXInf policy ready in %.1fs", time.time() - t0)
        logger.info(
            "policy metadata: image_keys=%s prompt_key=%s state_key=%s "
            "action_dim=%d action_horizon=%d",
            list(self._policy.image_keys),
            self._policy.prompt_key,
            self._policy.state_key,
            self._policy.action_dim,
            self._policy.action_horizon,
        )

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["vla.status"] = self.get_status
        self._readonly_methods.add("vla.status")
        if self._embodiment == "yam":
            self._rpc["vla.get_model_meta"] = self.get_model_meta
            self._readonly_methods.add("vla.get_model_meta")

    def get_model_meta(self) -> dict:
        """Return the YAM contract required by its existing runtime."""
        from robots.yam.contracts import vla_runtime_contract

        return vla_runtime_contract()

    def get_status(self) -> dict:
        """Return the executable chunk length, as the Pi0.5 server does."""
        return {"action_chunk": self._action_chunk}

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        """Return one batched action chunk in the selected robot's environment units."""
        if (options or {}).get("mode", "eval") != "eval":
            raise ValueError("APXInf VLA supports eval mode only")
        apx_obs = apxinf_format_decode(obs, self._embodiment, self._policy)
        result = self._policy.infer(apx_obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected = (self._policy.action_horizon, self._policy.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(
                f"invalid APXInf actions: expected finite {expected}, got {actions.shape}"
            )
        return actions[None, : self._action_chunk]

    def close(self) -> None:
        if hasattr(self, "_policy") and self._policy is not None:
            self._policy.close()
            self._policy = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="APXInf VLA RPC server for RPent",
    )
    p.add_argument(
        "--embodiment",
        required=True,
        help="Embodiment preset name (e.g. 'libero'); see APXINF_EMBODIMENTS",
    )
    p.add_argument(
        "--model-path",
        required=True,
        help="Path to the checkpoint directory (openpi or APXInf format)",
    )
    p.add_argument(
        "--precision",
        default="bf16",
        choices=["auto", "fp8", "bf16", "int8"],
        help="Inference precision (default: bf16)",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device string (default: cuda:0)",
    )
    p.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to the SentencePiece tokenizer (if not in checkpoint dir)",
    )
    p.add_argument(
        "--norm-stats-path",
        default=None,
        help="Path to norm_stats.json (if not in checkpoint dir)",
    )
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    facade = ApxInfVLAFacade(
        model_path=args.model_path,
        embodiment=args.embodiment,
        precision=args.precision,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        norm_stats_path=args.norm_stats_path,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
