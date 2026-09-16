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

"""RPC server wrapping the Pi0.5 VLA.

Embodiment-specific settings (openpi config name, action dim, …) are
selected by the ``--embodiment`` CLI flag and looked up in
``PI05_EMBODIMENTS``.

The ``--model-backend`` flag picks the RLinf loader an embodiment uses
(``openpi_pytorch``, the default, or ``openpi_rlinf``). Presets whose loader
resolves normalisation statistics from a directory take ``--norm-stats-path``
(or the ``PI05_NORM_STATS_PATH`` environment variable). Use ``--repo-id``
to select dataset normalization statistics within a checkpoint.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.config import get_pi05_checkpoint_path
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")

# ---------------------------------------------------------------------------
# Embodiment registry
# ---------------------------------------------------------------------------

# NOTE: an embodiment added here must also be registered in the client's
# ``_ENCODE_OBS`` (obs encoding); the two registries are kept in sync manually.
PI05_EMBODIMENTS: dict[str, dict] = {
    "dual_franka": {
        "num_action_chunks": 20,
        "action_dim": 20,
        "use_proprio": True,
        "num_steps": 5,
        "add_value_head": False,
        "openpi": {
            "config_name": "pi05_dualfranka_tcp_rot6d",
            "num_images_in_input": 3,
            "action_chunk": 20,
            "num_steps": 5,
            "train_expert_only": False,
            "action_env_dim": 20,
            "add_value_head": False,
            "detach_critic_input": True,
        },
    },
    "libero": {
        "num_action_chunks": 5,
        "action_dim": 7,
        "use_proprio": True,
        "num_steps": 5,
        "add_value_head": False,
        "openpi": {
            "config_name": "pi05_libero",
            "num_images_in_input": 2,
            "action_chunk": 5,
            "num_steps": 5,
            "action_env_dim": 7,
            "add_value_head": False,
        },
    },
}

PI05_ROBOT_PLATFORMS: dict[str, str] = {
    "libero": "LIBERO",
}

# RLinf model loaders an embodiment preset may select via ``model_backend``.
PI05_MODEL_BACKENDS: tuple[str, ...] = ("openpi_rlinf", "openpi_pytorch")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str, emb_cfg: dict) -> Any:
    """OmegaConf for the RLinf ``openpi`` / ``openpi_rlinf`` ``get_model``.

    Two-level merge ``emb_cfg`` into a default config template.  ``emb_cfg``
    mirrors the OmegaConf structure (top-level keys + ``openpi`` sub-dict),
    so adding a new key to an embodiment preset automatically flows into
    the model config.  ``model_path`` is set at runtime, not from the
    embodiment preset.
    """
    cfg = {
        "model_type": "openpi",
        "model_path": model_path,
        "precision": None,
        "is_lora": False,
        "lora_rank": 32,
        "openpi": {
            "noise_level": 0.5,
            "train_expert_only": True,
            "noise_method": "flow_sde",
            "value_after_vlm": False,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": None,
            "use_dsrl": False,
        },
    }
    # Deep merge: top-level keys override, openpi sub-dict merges into cfg.openpi
    for k, v in emb_cfg.items():
        if k == "openpi":
            cfg["openpi"].update(v)
        else:
            cfg[k] = v

    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# VLA facade
# ---------------------------------------------------------------------------


class Pi05VLAFacade(BaseVLAFacade):
    """Pi0.5 VLA inference backed by an openpi model.

    Wires ``vla.predict`` to :meth:`predict` (registered by the base class).
    Embodiment-specific behavior (model config, loader, obs decode) is driven
    by the ``embodiment`` name passed at construction. ``model_backend``
    selects the RLinf loader — ``openpi_pytorch`` (default) or ``openpi_rlinf``
    for the real-robot YAM joint policy.

    Session-isolation is not supported (``reset_session`` is not registered).
    """

    def __init__(
        self,
        *,
        model_path: str,
        embodiment: str,
        model_backend: str = "openpi_pytorch",
        norm_stats_path: str | None = None,
        repo_id: str | None = None,
    ):
        if embodiment not in PI05_EMBODIMENTS:
            raise ValueError(
                f"unknown pi05 server embodiment: {embodiment!r}; "
                f"registered={list(PI05_EMBODIMENTS)}"
            )
        emb_cfg = PI05_EMBODIMENTS[embodiment]
        if model_backend not in PI05_MODEL_BACKENDS:
            raise ValueError(
                f"unsupported pi05 model backend: {model_backend!r}; "
                f"supported={list(PI05_MODEL_BACKENDS)}"
            )
        if embodiment == "dual_franka" and not (repo_id or norm_stats_path):
            raise ValueError("dual_franka requires repo_id or norm_stats_path")
        self._embodiment = embodiment
        super().__init__()

        if model_backend == "openpi_rlinf":
            from rlinf.models.embodiment.openpi_rlinf import get_model
        else:
            from rlinf.models.embodiment.openpi import get_model

        platform = PI05_ROBOT_PLATFORMS.get(embodiment)
        if platform is not None:
            os.environ.setdefault("ROBOT_PLATFORM", platform)

        cfg = build_model_cfg(model_path=model_path, emb_cfg=emb_cfg)
        if repo_id is not None or norm_stats_path is not None:
            cfg.openpi_data = {}
            if repo_id is not None:
                cfg.openpi_data.repo_id = repo_id
            if norm_stats_path is not None:
                cfg.openpi_data.norm_stats_path = norm_stats_path
        t0 = time.time()
        logger.info(
            "loading Pi0.5 (embodiment=%s, model_backend=%s, model_path=%s) ...",
            embodiment,
            model_backend,
            cfg["model_path"],
        )
        self._model = get_model(cfg, torch_dtype=None).cuda().eval()
        logger.info("model ready in %.1fs", time.time() - t0)

    # ---- inference ----

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        """Run one inference and return the action ndarray.

        The caller (client) is responsible for encoding env-native obs into
        the openpi wire format (see ``Pi05VLAClient.encode_obs``).
        """
        mode = (options or {}).get("mode", "eval")
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(obs, mode=mode)
        return (
            actions.detach().cpu().numpy()
            if (
                hasattr(actions, "detach")
                and hasattr(actions, "cpu")
                and hasattr(actions, "numpy")
            )
            else np.asarray(actions)
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embodiment",
        required=True,
        help="Embodiment preset name (e.g. 'libero'); see PI05_EMBODIMENTS",
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
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--model-backend",
        choices=list(PI05_MODEL_BACKENDS),
        default="openpi_pytorch",
        help="RLinf model loader (default: openpi_pytorch)",
    )
    p.add_argument(
        "--norm-stats-path",
        default=os.environ.get("PI05_NORM_STATS_PATH"),
        help="norm_stats directory, for presets whose loader needs it "
        "(defaults to PI05_NORM_STATS_PATH env)",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="SFT dataset repo ID used to locate checkpoint normalization statistics",
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

    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    facade = Pi05VLAFacade(
        model_path=model_path,
        embodiment=args.embodiment,
        model_backend=args.model_backend,
        norm_stats_path=args.norm_stats_path,
        repo_id=args.repo_id,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
