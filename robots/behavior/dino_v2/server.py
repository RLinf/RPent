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

"""DINOv2 encoder RPC server for BEHAVIOR memory retrieval."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))

from rpent.utils.rpc import RpcFacade  # noqa: E402


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


class BehaviorDinoFacade(RpcFacade):
    """Expose the BEHAVIOR DINOv2 engine through the shared RPC facade."""

    def __init__(self, encoder: Any, meta: dict[str, Any]) -> None:
        super().__init__()
        self._encoder = encoder
        self._meta = dict(meta)
        self._close_lock = threading.Lock()
        self._closed = False
        self._register_rpc()

    def _register_rpc(self) -> None:
        self._rpc["dino.encode_batch"] = self.encode_batch
        self._rpc["dino.get_meta"] = self.get_meta
        self._readonly_methods.add("dino.get_meta")

    def get_meta(self) -> dict[str, Any]:
        return {**self._meta, "pid": os.getpid()}

    def encode_batch(self, *, images: list[Any]) -> list[Any]:
        result = self._encoder.encode_batch(
            [
                None if image is None else np.asarray(image, dtype=np.uint8)
                for image in images
            ]
        )
        return [
            None if item is None else np.asarray(item, dtype=np.float32)
            for item in result
        ]

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._encoder.close()
            self._closed = True


def _materialize_encoder(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    # Heavy imports begin only after main() has applied CUDA_VISIBLE_DEVICES.
    import torch
    import torchvision

    from robots.behavior.dino_v2.encoder import (
        DINOV2_DIMENSION,
        MODEL_ID,
        MODEL_REVISION,
        Dinov2DeploymentPaths,
        Dinov2Engine,
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
        raise RuntimeError(
            "DINO service requires CUDA; CPU fallback is not a BEHAVIOR runtime component"
        )
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
        cache_dir=Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else None,
    )
    encoder = Dinov2Engine(identity, deployment)
    # Force backend construction before get_meta advertises the deployment.
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

    encoder, meta = _materialize_encoder(args)
    facade = BehaviorDinoFacade(encoder, meta)
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()


__all__ = ["BehaviorDinoFacade", "main"]
