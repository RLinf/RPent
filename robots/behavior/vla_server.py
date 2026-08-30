"""Pi0.5 HTTP sidecar for BEHAVIOR; this process never imports OmniGibson."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))

from robots.behavior.policy_checkpoint import (  # noqa: E402
    SHARED_POLICY_CHECKPOINT_PATH,
    validate_policy_checkpoint,
)
from robots.behavior.schemas import ACTION_DIM, DEFAULT_ACTION_CHUNK  # noqa: E402

NORM_STATS_REL = Path("assets/behavior-1k/2025-challenge-demos/norm_stats.json")
NORM_STATS_ASSET_ID = NORM_STATS_REL.parent.as_posix()


class ImageBlock(BaseModel):
    format: str = "png"
    data: str


class PredictRequest(BaseModel):
    instruction: str
    images: dict[str, ImageBlock]
    state: list[list[float]]
    mode: str = "eval"
    binding_id: str | None = None


class BindingRequest(BaseModel):
    binding_id: str


_MODEL: Any = None
_MODEL_META: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()
_ACTIONS_ENABLED = True
_ACTIONS_LOCK = threading.Lock()
_ACTION_BINDING_ID: str | None = None


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("--cuda-device must be one physical GPU ordinal")
    return device


def _binding_digest(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_matching_binding(value: str | None) -> None:
    if _ACTION_BINDING_ID is None:
        if value is not None:
            raise ValueError("VLA server is not bound to this attempt")
        return
    if value != _ACTION_BINDING_ID:
        raise ValueError("VLA attempt binding mismatch")


def validate_checkpoint(path: str | Path) -> Path:
    """Return the verified shared checkpoint root."""

    return Path(validate_policy_checkpoint(path).resolved_path)


def build_model_config(checkpoint: str | Path) -> Any:
    from omegaconf import OmegaConf

    checkpoint = Path(checkpoint).absolute()
    return OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "precision": None,
            "openpi_data": {
                # RLinf forwards this object into OpenPI's DataConfigFactory.  Its
                # checkpoint loader resolves norm stats as
                # ``checkpoint / asset_id / norm_stats.json``; the validated
                # BEHAVIOR checkpoint keeps them under the pinned assets tree.
                "assets": {
                    "assets_dir": str(checkpoint),
                    "asset_id": NORM_STATS_ASSET_ID,
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
                "action_horizon": DEFAULT_ACTION_CHUNK,
                "action_chunk": DEFAULT_ACTION_CHUNK,
                "action_env_dim": ACTION_DIM,
                "num_steps": 4,
                "add_value_head": False,
                "noise_level": 0.0,
                "noise_method": "flow_sde",
                "joint_logprob": False,
            },
        }
    )


def load_model(checkpoint: str | Path, *, seed: int) -> None:
    """Load Pi0.5 after caller has already applied CUDA_VISIBLE_DEVICES."""

    global _ACTION_BINDING_ID, _ACTIONS_ENABLED, _MODEL, _MODEL_META
    import torch

    try:
        from rlinf.models.embodiment.openpi import get_model
    except Exception as exc:
        raise RuntimeError(
            "RLinf OpenPI model dependency is unavailable for BEHAVIOR VLA"
        ) from exc

    checkpoint_binding = validate_policy_checkpoint(checkpoint)
    resolved = Path(checkpoint_binding.resolved_path)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.time()
    model = get_model(build_model_config(resolved))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = model.to(device).eval()
    with _ACTIONS_LOCK:
        _ACTIONS_ENABLED = True
        _ACTION_BINDING_ID = None
    _MODEL_META = {
        "status": "ok",
        "runtime": "behavior_vla",
        "config_name": "pi05_behavior",
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "action_dim": ACTION_DIM,
        "device": str(device),
        "checkpoint": str(resolved),
        "checkpoint_binding": checkpoint_binding.as_dict(),
        "seed": int(seed),
        "load_elapsed_s": round(time.time() - started, 2),
    }


def _decode_image(block: dict[str, Any]) -> np.ndarray:
    import imageio.v2 as imageio

    if str(block.get("format", "png")).lower() != "png":
        raise ValueError("only PNG image blocks are supported")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block is missing base64 data")
    image = np.asarray(imageio.imread(io.BytesIO(base64.b64decode(data))))
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError(f"image must be [H,W,3 or 4], got {image.shape}")
    return image[..., :3].astype(np.uint8, copy=False)


def build_env_observation(request: dict[str, Any]) -> dict[str, Any]:
    import torch

    images = request.get("images") or {}
    required = ("main", "left_wrist", "right_wrist")
    missing = [name for name in required if name not in images]
    if missing:
        raise ValueError(f"missing image(s): {missing}")
    state = np.asarray(request.get("state"), dtype=np.float32)
    if state.ndim != 2 or state.shape[0] != 1 or state.shape[1] < 256:
        raise ValueError(
            "state must contain one raw R1Pro proprio vector [1,N>=256], "
            f"got {state.shape}"
        )
    main = _decode_image(images["main"])
    left = _decode_image(images["left_wrist"])
    right = _decode_image(images["right_wrist"])
    return {
        "main_images": torch.from_numpy(main[None]),
        "wrist_images": torch.from_numpy(np.stack([left, right], axis=0)[None]),
        "states": torch.from_numpy(state),
        "task_descriptions": [str(request.get("instruction") or "")],
        "extra_view_images": None,
    }


def build_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="RPent BEHAVIOR Pi0.5")

    @app.get("/healthz")
    def healthz():
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        with _ACTIONS_LOCK:
            actions_enabled = bool(_ACTIONS_ENABLED)
            binding_digest = _binding_digest(_ACTION_BINDING_ID)
        return {
            **_MODEL_META,
            "pid": os.getpid(),
            "actions_enabled": actions_enabled,
            "binding_digest": binding_digest,
        }

    @app.post("/control/disable-actions")
    def disable_actions(request: BindingRequest | None = None):
        global _ACTIONS_ENABLED
        with _MODEL_LOCK, _ACTIONS_LOCK:
            if request is not None:
                try:
                    _require_matching_binding(request.binding_id)
                except ValueError as error:
                    raise HTTPException(status_code=409, detail=str(error)) from error
            _ACTIONS_ENABLED = False
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": False,
            "binding_digest": _binding_digest(_ACTION_BINDING_ID),
        }

    @app.post("/control/bind-actions")
    def bind_actions(request: BindingRequest):
        global _ACTION_BINDING_ID
        binding_id = request.binding_id.strip()
        if not binding_id or len(binding_id) > 256:
            raise HTTPException(status_code=400, detail="invalid binding_id")
        with _MODEL_LOCK, _ACTIONS_LOCK:
            if _ACTIONS_ENABLED:
                raise HTTPException(
                    status_code=409,
                    detail="disable VLA actions before binding a fresh attempt",
                )
            _ACTION_BINDING_ID = binding_id
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": False,
            "binding_digest": _binding_digest(binding_id),
        }

    @app.post("/control/enable-actions")
    def enable_actions(request: BindingRequest | None = None):
        global _ACTIONS_ENABLED
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        with _MODEL_LOCK, _ACTIONS_LOCK:
            try:
                _require_matching_binding(
                    request.binding_id if request is not None else None
                )
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            _ACTIONS_ENABLED = True
        return {
            "status": "ok",
            "pid": os.getpid(),
            "actions_enabled": True,
            "binding_digest": _binding_digest(_ACTION_BINDING_ID),
        }

    @app.post("/predict")
    def predict(request: PredictRequest):
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        with _ACTIONS_LOCK:
            try:
                _require_matching_binding(request.binding_id)
            except ValueError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if not _ACTIONS_ENABLED:
                raise HTTPException(
                    status_code=409, detail="VLA action inference is disabled"
                )
        try:
            import torch

            env_obs = build_env_observation(request.model_dump())
            with _MODEL_LOCK:
                with _ACTIONS_LOCK:
                    try:
                        _require_matching_binding(request.binding_id)
                    except ValueError as error:
                        raise HTTPException(
                            status_code=409, detail=str(error)
                        ) from error
                    if not _ACTIONS_ENABLED:
                        raise HTTPException(
                            status_code=409, detail="VLA action inference is disabled"
                        )
                with torch.no_grad():
                    actions, _ = _MODEL.predict_action_batch(
                        env_obs,
                        mode="eval",
                        compute_values=False,
                    )
            if torch.is_tensor(actions):
                actions = actions.detach().float().cpu().numpy()
            actions = np.asarray(actions, dtype=np.float32)
            if (
                actions.ndim != 3
                or actions.shape[0] != 1
                or actions.shape[2] != ACTION_DIM
                or actions.shape[1] < 1
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    f"Pi0.5 returned invalid [1,T,{ACTION_DIM}] shape {actions.shape}"
                )
            return {
                "actions": actions.tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            }
        except HTTPException:
            raise
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"}, status_code=500
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", default=str(SHARED_POLICY_CHECKPOINT_PATH))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    cuda_device = _single_cuda_device(args.cuda_device)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    load_model(args.checkpoint, seed=args.seed)

    if args.parent_watch:
        from rpent.utils.daemon import watch_parent_death

        watch_parent_death(lambda: os._exit(0))

    import uvicorn

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()


__all__ = [
    "NORM_STATS_REL",
    "build_app",
    "build_env_observation",
    "build_model_config",
    "load_model",
    "main",
    "validate_checkpoint",
]
