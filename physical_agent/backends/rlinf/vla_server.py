"""HTTP server wrapping the Pi0.5 VLA behind a single ``/predict`` endpoint.

Run::

    python -m physical_agent.backends.rlinf.vla_server \
        --host 0.0.0.0 --port 8000 [--model-path /path/to/pi05_libero]

Endpoints:

- ``POST /predict``  — single inference call. See ``vla_client.py`` for the
  exact request / response schema.
- ``GET  /healthz``  — readiness probe (``{"status":"ok"}`` once the model is
  loaded; HTTP 503 otherwise). Useful for clients that start in parallel
  with the server.

The server loads Pi0.5 once at startup (same path as ``repl_driver.main``
today) and serves until killed.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from physical_agent.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
)
from physical_agent.backends import add_external_rlinf_to_path

PHYSICALAGENT_ROOT = get_repo_root()
RLINF_REPO_PATH = add_external_rlinf_to_path(PHYSICALAGENT_ROOT)
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from physical_agent.backends.rlinf.primitives import build_model_cfg  # noqa: E402
from physical_agent.utils.logging import get_logger  # noqa: E402

logger = get_logger("vla_server")


# ----- model singleton ----------------------------------------------------


_MODEL = None


def load_model(model_path: str | None) -> None:
    """Build the openpi model and stash it on the module."""
    global _MODEL
    from rlinf.models.embodiment.openpi import get_model as get_openpi_model

    cfg = build_model_cfg(model_path=model_path or get_pi05_checkpoint_path())
    t0 = time.time()
    logger.info("loading Pi0.5 (model_path=%s) ...", cfg["model_path"])
    model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
    _MODEL = model
    logger.info("model ready in %.1fs", time.time() - t0)


# ----- request decoding ---------------------------------------------------


def _decode_image_block(block: dict[str, Any]) -> np.ndarray:
    fmt = (block.get("format") or "png").lower()
    if fmt != "png":
        raise ValueError(f"unsupported image format: {fmt!r} (only 'png')")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block missing base64 'data'")
    raw = base64.b64decode(data)
    img = np.asarray(imageio.imread(io.BytesIO(raw)))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _build_env_obs(req: dict[str, Any]) -> dict[str, Any]:
    images = req.get("images") or {}
    if not isinstance(images, dict) or "main" not in images:
        raise ValueError("'images.main' is required")
    main = _decode_image_block(images["main"])
    main_batch = main[None]
    obs: dict[str, Any] = {
        "main_images": main_batch,
        "task_descriptions": [str(req.get("instruction", ""))]
        * main_batch.shape[0],
        "extra_view_images": None,
    }
    extra = images.get("extra")
    if isinstance(extra, dict):
        obs["extra_view_images"] = _decode_image_block(extra)[None]

    state = req.get("state")
    if state is None:
        raise ValueError("'state' is required")
    states = np.asarray(state, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(
            f"state must be [B, state_dim]; got shape {states.shape}"
        )
    obs["states"] = states
    return obs


# ----- HTTP app -----------------------------------------------------------


def build_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="Pi0.5 VLA")

    class ImageBlock(BaseModel):
        format: str = "png"
        data: str

    class PredictRequest(BaseModel):
        instruction: str = ""
        images: dict[str, ImageBlock]
        state: list[list[float]]
        mode: str = "eval"

    @app.get("/healthz")
    def healthz():
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if _MODEL is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        try:
            env_obs = _build_env_obs(req.model_dump())
            with torch.no_grad():
                actions, _ = _MODEL.predict_action_batch(env_obs, mode=req.mode)
            actions_np = (
                actions.detach().cpu().numpy()
                if isinstance(actions, torch.Tensor)
                else np.asarray(actions)
            ).astype(np.float32)
            return JSONResponse(
                {
                    "actions": actions_np.tolist(),
                    "shape": list(actions_np.shape),
                    "dtype": "float32",
                }
            )
        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning("bad /predict request: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("/predict failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


# ----- subprocess helper --------------------------------------------------


def _find_free_port() -> int:
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def spawn_local_server(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    model_path: str | None = None,
):
    """Spawn a `vla_server` child process bound to ``host:port``.

    Returns ``(base_url, proc)``. ``port`` defaults to a free local port.
    The caller is responsible for stopping ``proc`` (e.g. ``proc.terminate()``
    in an ``atexit`` hook). No readiness probe — callers that need to wait
    for ``/healthz`` should use ``VLAClient.healthz()`` directly.
    """
    import subprocess
    import sys

    bind_port = port if port is not None else _find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "physical_agent.backends.rlinf.vla_server",
        "--host", host,
        "--port", str(bind_port),
    ]
    if model_path:
        cmd += ["--model-path", model_path]
    proc = subprocess.Popen(cmd)
    return f"http://{host}:{bind_port}", proc


# ----- CLI ----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    args = p.parse_args()

    load_model(args.model_path)
    app = build_app()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
