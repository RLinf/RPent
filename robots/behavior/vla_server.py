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

"""Pi0.5 HTTP sidecar for BEHAVIOR; this process never imports OmniGibson."""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

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
from rpent.robots.components.vla_facade_base import BaseVLAFacade  # noqa: E402
from rpent.utils.rpc.http_rpc import _from_json, _NumpyEncoder  # noqa: E402
from rpent.utils.rpc.rpc_facade import make_error_response  # noqa: E402

NORM_STATS_REL = Path("assets/behavior-1k/2025-challenge-demos/norm_stats.json")
NORM_STATS_ASSET_ID = NORM_STATS_REL.parent.as_posix()


class ImageBlock(BaseModel):
    format: str = "png"
    data: str


class PredictRequest(BaseModel):
    instruction: str
    images: dict[str, ImageBlock]
    state: list[list[float]]
    mode: Literal["eval"] = "eval"


_MODEL: Any = None
_MODEL_META: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("--cuda-device must be one physical GPU ordinal")
    return device


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
                "norm_stats_path": str(checkpoint / NORM_STATS_REL),
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

    global _MODEL, _MODEL_META
    import torch

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
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
    finally:
        if gc_was_enabled:
            gc.enable()


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


def _run_model(env_obs: dict[str, Any], *, mode: str) -> np.ndarray:
    if _MODEL is None:
        raise RuntimeError("model not loaded")
    import torch

    with _MODEL_LOCK, torch.no_grad():
        actions, _ = _MODEL.predict_action_batch(
            env_obs,
            mode=mode,
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
    return actions


def _rpc_env_observation(observation: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not isinstance(observation, dict):
        raise TypeError("BEHAVIOR VLA observation must be a mapping")
    main = np.asarray(observation.get("main_images"))
    wrists = np.asarray(observation.get("wrist_images"))
    states = np.asarray(observation.get("states"), dtype=np.float32)
    descriptions = observation.get("task_descriptions")
    if main.ndim != 4 or main.shape[0] != 1 or main.shape[-1] != 3:
        raise ValueError(f"main_images must be [1,H,W,3], got {main.shape}")
    if wrists.ndim != 5 or wrists.shape[:2] != (1, 2) or wrists.shape[-1] != 3:
        raise ValueError(f"wrist_images must be [1,2,H,W,3], got {wrists.shape}")
    if states.ndim != 2 or states.shape[0] != 1 or states.shape[1] < 256:
        raise ValueError(f"states must be [1,N>=256], got {states.shape}")
    if not isinstance(descriptions, list) or len(descriptions) != 1:
        raise ValueError("task_descriptions must contain one string")
    return {
        "main_images": torch.from_numpy(
            np.ascontiguousarray(main.astype(np.uint8, copy=False))
        ),
        "wrist_images": torch.from_numpy(
            np.ascontiguousarray(wrists.astype(np.uint8, copy=False))
        ),
        "states": torch.from_numpy(np.ascontiguousarray(states)),
        "task_descriptions": [str(descriptions[0])],
        "extra_view_images": None,
    }


class BehaviorVLAFacade(BaseVLAFacade):
    """Expose the legacy BEHAVIOR Pi0.5 model through common VLA RPC."""

    def _builtin_dispatch(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if method == "healthz":
            if _MODEL is None:
                raise RuntimeError("model not loaded")
            return {**_MODEL_META, "pid": os.getpid()}
        return super()._builtin_dispatch(method, args, kwargs)

    def predict(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        options = {} if options is None else options
        if not isinstance(options, dict):
            raise TypeError("VLA options must be a mapping")
        unexpected = set(options) - {"mode"}
        if unexpected:
            raise ValueError(f"unsupported VLA options: {sorted(unexpected)!r}")
        mode = str(options.get("mode", "eval"))
        if mode != "eval":
            raise ValueError("BEHAVIOR VLA inference mode must be 'eval'")
        return _run_model(_rpc_env_observation(observation), mode=mode)


def build_app(facade: BehaviorVLAFacade | None = None) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response

    facade = facade or BehaviorVLAFacade()
    app = FastAPI(title="RPent BEHAVIOR Pi0.5")

    @app.get("/healthz")
    def healthz():
        try:
            return facade._dispatch("healthz", (), {})
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            actions = _run_model(
                build_env_observation(request.model_dump()),
                mode=request.mode,
            )
            return {
                "actions": actions.tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            }
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"}, status_code=500
            )

    @app.post("/call")
    def rpc_call(request: dict[str, Any]):
        try:
            method = request["method"]
            args = tuple(_from_json(value) for value in request.get("args", []))
            kwargs = {
                key: _from_json(value)
                for key, value in request.get("kwargs", {}).items()
            }
            response = {"ok": True, "result": facade._dispatch(method, args, kwargs)}
        except Exception as exc:
            response = make_error_response(exc)
        return Response(
            content=json.dumps(response, cls=_NumpyEncoder),
            media_type="application/json",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help="Path to your Pi05-Behavior model checkpoint.",
    )
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

    uvicorn.run(
        build_app(BehaviorVLAFacade()),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BehaviorVLAFacade",
    "NORM_STATS_REL",
    "build_app",
    "build_env_observation",
    "build_model_config",
    "load_model",
    "main",
    "validate_checkpoint",
]
