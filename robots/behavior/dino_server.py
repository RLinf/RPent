"""DINOv2 encoder RPC server for BEHAVIOR memory retrieval."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("--cuda-device must be one physical GPU ordinal")
    return device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_required_path(value: str | None, *, env_name: str, label: str) -> Path:
    raw = value or os.environ.get(env_name)
    if not raw:
        raise RuntimeError(
            f"DINO {label} path is required; set --{label.replace('_', '-')} "
            f"or {env_name}"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"DINO {label} path is missing: {path}")
    return path


def _backend_loader_from_env() -> Any:
    spec = os.environ.get("RPENT_BEHAVIOR_DINOV2_BACKEND_FACTORY")
    if not spec:
        return None
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise RuntimeError(
            "RPENT_BEHAVIOR_DINOV2_BACKEND_FACTORY must be 'module:callable'"
        )
    module = importlib.import_module(module_name)
    loader = getattr(module, attr)
    if not callable(loader):
        raise RuntimeError("configured DINO backend factory is not callable")
    return loader


class DinoRpc:
    def __init__(self, encoder: Any, meta: dict[str, Any]) -> None:
        self._encoder = encoder
        self._meta = dict(meta)

    def healthz(self) -> dict[str, Any]:
        return {**self._meta, "pid": os.getpid()}

    def encode_batch(self, *, images: list[Any]) -> list[Any]:
        result = self._encoder.encode_batch(
            [None if image is None else np.asarray(image, dtype=np.uint8) for image in images]
        )
        return [None if item is None else np.asarray(item, dtype=np.float32) for item in result]

    def close(self) -> dict[str, Any]:
        self._encoder.close()
        return {"status": "closed", "pid": os.getpid()}

    def dispatch(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if method == "healthz":
            return self.healthz()
        if method == "dino.encode_batch":
            return self.encode_batch(*args, **kwargs)
        if method == "dino.close":
            return self.close()
        raise AttributeError(f"unknown DINO RPC method: {method}")


def _materialize_encoder(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    # Heavy imports begin only after main() has applied CUDA_VISIBLE_DEVICES.
    import torch
    import torchvision

    from robots.behavior.memory_embeddings_dinov2 import (
        DINOV2_DIMENSION,
        MODEL_ID,
        MODEL_REVISION,
        Dinov2DeploymentPaths,
        Dinov2Encoder,
        Dinov2RevisionIdentity,
    )

    source_archive = _resolve_required_path(
        args.source_archive,
        env_name="RPENT_BEHAVIOR_DINOV2_SOURCE_ARCHIVE",
        label="source_archive",
    )
    weights = _resolve_required_path(
        args.weights,
        env_name="RPENT_BEHAVIOR_DINOV2_WEIGHTS",
        label="weights",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("DINO service requires CUDA; CPU fallback is not a BEHAVIOR runtime component")
    identity = Dinov2RevisionIdentity(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        source_commit=MODEL_REVISION.rsplit("@", 1)[-1],
        source_archive_sha256=_sha256_file(source_archive),
        weights_sha256=_sha256_file(weights),
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
        device=device,
    )
    deployment = Dinov2DeploymentPaths(
        source_archive_path=source_archive,
        weights_path=weights,
        cache_dir=Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None,
    )
    encoder = Dinov2Encoder(
        identity,
        deployment,
        backend_loader=_backend_loader_from_env(),
    )
    # Force backend construction now so healthz never advertises a placeholder.
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    encoder.encode_batch([blank])
    return encoder, {
        "status": "ok",
        "runtime": "behavior_dino",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": DINOV2_DIMENSION,
        "device": device,
        "checkpoint_binding": identity.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument("--source-archive", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    cuda_device = _single_cuda_device(args.cuda_device)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    from rpent.utils.daemon import watch_parent_death
    from rpent.utils.rpc.http_rpc import HttpRpcServer

    encoder, meta = _materialize_encoder(args)
    rpc = DinoRpc(encoder, meta)
    server = HttpRpcServer((args.host, args.port), rpc.dispatch)
    if args.parent_watch:
        watch_parent_death(server.shutdown)
    try:
        server.serve_forever()
    finally:
        try:
            encoder.close()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()


__all__ = ["DinoRpc", "main"]
