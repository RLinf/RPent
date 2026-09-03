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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))

# ---------------------------------------------------------------------------
# Embodiment registry
# ---------------------------------------------------------------------------

_BEHAVIOR_ACTION_DIM = 23


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_manifest(
    path: str | Path,
    manifest_path: str | Path,
) -> tuple[str, dict[str, Any]]:
    requested = Path(path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"your Pi0.5 model checkpoint is unavailable: {error}"
        ) from error
    if not resolved.is_dir():
        raise ValueError(f"your Pi0.5 model checkpoint is not a directory: {resolved}")

    manifest_file = Path(manifest_path).expanduser()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"checkpoint manifest is unavailable: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"checkpoint manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")

    manifest_resolved_path = manifest.get("resolved_path")
    if manifest_resolved_path is not None:
        manifest_resolved = Path(str(manifest_resolved_path)).expanduser()
        try:
            manifest_resolved = manifest_resolved.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"checkpoint manifest resolved_path is unavailable: {error}"
            ) from error
        if manifest_resolved != resolved:
            raise ValueError(
                "checkpoint manifest resolved_path does not match --model-path: "
                f"{manifest_resolved} != {resolved}"
            )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint manifest files must be a non-empty object")
    for relative_path, requirement in files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("checkpoint manifest file paths must be non-empty strings")
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(
                f"checkpoint manifest file path is not a safe relative path: {relative_path}"
            )
        if not isinstance(requirement, Mapping):
            raise ValueError(
                f"checkpoint manifest entry must be an object: {relative_path}"
            )
        try:
            expected_size = int(requirement["size_bytes"])
            expected_sha256 = str(requirement["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint manifest entries require size_bytes and sha256: "
                f"{relative_path}"
            ) from error
        candidate = resolved / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"checkpoint file is missing or unsafe: {candidate}")
        size = candidate.stat().st_size
        if size != expected_size:
            raise ValueError(
                "checkpoint file size mismatch for "
                f"{relative_path}: expected {expected_size}, got {size}"
            )
        actual_sha256 = _checkpoint_sha256(candidate)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "checkpoint file SHA256 mismatch for "
                f"{relative_path}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    return str(resolved), manifest


# NOTE: an embodiment added here must also be registered in the client's
# ``_ENCODE_OBS`` (obs encoding); the two registries are kept in sync manually.
PI05_EMBODIMENTS: dict[str, dict] = {
    "behavior": {
        "seed": 0,
        "num_action_chunks": 32,
        "action_dim": 32,
        "use_proprio": True,
        "num_steps": 4,
        "add_value_head": False,
        "openpi_data": {
            "assets": {
                "assets_dir": None,
                "asset_id": "assets/behavior-1k/2025-challenge-demos",
            },
            "extra_delta_transform": False,
            "extract_state_from_proprio": True,
            "use_all_wrist_images": True,
            "use_quantile_norm": True,
        },
        "openpi": {
            "config_name": "pi05_behavior",
            "num_images_in_input": 3,
            "action_dim": 32,
            "action_horizon": 32,
            "action_chunk": 32,
            "action_env_dim": 23,
            "num_steps": 4,
            "add_value_head": False,
            "noise_level": 0.0,
            "noise_method": "flow_sde",
            "joint_logprob": False,
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
    "behavior": "BEHAVIOR",
    "libero": "LIBERO",
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str, emb_cfg: dict) -> Any:
    """OmegaConf for ``rlinf.models.embodiment.openpi.get_model``.

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
        elif k == "openpi_data":
            cfg["openpi_data"] = dict(v)
        else:
            cfg[k] = v
    openpi_data = cfg.get("openpi_data")
    if isinstance(openpi_data, dict) and isinstance(openpi_data.get("assets"), dict):
        if not openpi_data["assets"].get("assets_dir"):
            openpi_data["assets"]["assets_dir"] = os.fspath(Path(model_path))
    from omegaconf import OmegaConf

    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# VLA facade
# ---------------------------------------------------------------------------


class Pi05VLAFacade(BaseVLAFacade):
    """Pi0.5 VLA inference backed by an openpi model.

    Wires ``vla.predict`` to :meth:`predict` (registered by the base class).
    Embodiment-specific behavior (model config, obs decode) is driven by
    the ``embodiment`` name passed at construction.

    Session-isolation is not supported (``reset_session`` is not registered).
    """

    def __init__(
        self,
        *,
        model_path: str,
        embodiment: str,
        checkpoint_manifest: str | None = None,
    ):
        if embodiment not in PI05_EMBODIMENTS:
            raise ValueError(
                f"unknown pi05 server embodiment: {embodiment!r}; "
                f"registered={list(PI05_EMBODIMENTS)}"
            )
        emb_cfg = PI05_EMBODIMENTS[embodiment]
        self._embodiment = embodiment
        self._model_path = os.fspath(Path(model_path).expanduser())
        self._checkpoint_binding: dict[str, Any] | None = None
        self._predict_lock = threading.Lock()
        super().__init__()

        import torch
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        platform = PI05_ROBOT_PLATFORMS.get(embodiment)
        if platform is not None:
            os.environ.setdefault("ROBOT_PLATFORM", platform)

        if checkpoint_manifest is not None:
            self._model_path, self._checkpoint_binding = _validate_checkpoint_manifest(
                model_path,
                checkpoint_manifest,
            )
        seed = emb_cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        cfg = build_model_cfg(model_path=self._model_path, emb_cfg=emb_cfg)
        t0 = time.time()
        logger.info(
            "loading Pi0.5 (embodiment=%s, model_path=%s) ...",
            embodiment,
            cfg["model_path"],
        )
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        self._model_meta = {
            "status": "ok",
            "runtime": "pi05_vla",
            "embodiment": embodiment,
            "config_name": emb_cfg.get("openpi", {}).get("config_name"),
            "action_horizon": emb_cfg.get("openpi", {}).get("action_chunk")
            or emb_cfg.get("num_action_chunks"),
            "action_dim": emb_cfg.get("openpi", {}).get("action_env_dim")
            or emb_cfg.get("action_dim"),
            "model_path": self._model_path,
            "device": "cuda",
            "load_elapsed_s": round(time.time() - t0, 2),
        }
        if self._checkpoint_binding is not None:
            self._model_meta["checkpoint_binding"] = self._checkpoint_binding
        logger.info("model ready in %.1fs", self._model_meta["load_elapsed_s"])

    def _builtin_dispatch(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if method == "healthz":
            return {**self._model_meta, "pid": os.getpid()}
        return super()._builtin_dispatch(method, args, kwargs)

    # ---- inference ----

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        """Run one inference and return the action ndarray.

        The caller (client) is responsible for encoding env-native obs into
        the openpi wire format (see ``Pi05VLAClient.encode_obs``).
        """
        if self._embodiment == "behavior":
            if options is None:
                options = {}
            if not isinstance(options, dict):
                raise TypeError("VLA options must be a mapping")
            unexpected = set(options) - {"mode"}
            if unexpected:
                raise ValueError(f"unsupported VLA options: {sorted(unexpected)!r}")
        mode = (options or {}).get("mode", "eval")
        if self._embodiment == "behavior":
            if mode != "eval":
                raise ValueError("BEHAVIOR VLA inference mode must be 'eval'")
        predict_kwargs = {"mode": mode}
        if self._embodiment == "behavior":
            predict_kwargs["compute_values"] = False
        lock = self._predict_lock if self._embodiment == "behavior" else nullcontext()
        import torch

        with lock, torch.no_grad():
            actions, _ = self._model.predict_action_batch(obs, **predict_kwargs)
        result = (
            actions.detach().cpu().numpy()
            if (
                hasattr(actions, "detach")
                and hasattr(actions, "cpu")
                and hasattr(actions, "numpy")
            )
            else np.asarray(actions)
        ).astype(np.float32)
        if self._embodiment == "behavior":
            if (
                result.ndim != 3
                or result.shape[0] != 1
                or result.shape[1] < 1
                or result.shape[2] != _BEHAVIOR_ACTION_DIM
                or not np.isfinite(result).all()
            ):
                raise ValueError(
                    "Pi0.5 returned invalid "
                    f"[1,T,{_BEHAVIOR_ACTION_DIM}] shape {result.shape}"
                )
        return result


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
        "--checkpoint-manifest",
        default=None,
        help=(
            "Optional generic JSON manifest describing required checkpoint files "
            "and SHA-256 values."
        ),
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
        checkpoint_manifest=args.checkpoint_manifest,
    )
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
