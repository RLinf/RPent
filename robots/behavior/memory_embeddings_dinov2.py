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

"""Pinned DINOv2 ViT-S/14 RGB224 CLS384 embedding contract.

The encoder identity is portable and path-free.  Deployment paths are checked
only when an actual backend is materialized.  Tests and offline builders may
inject a backend; the default backend imports torch lazily after verifying both
frozen assets, so importing this module itself remains lightweight.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from robots.behavior.memory_schema import MemoryValidationError, fail, require_sha256

MODEL_ID = "facebookresearch/dinov2_vits14"
MODEL_REVISION = "facebookresearch/dinov2@7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EXPECTED_SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
)
EXPECTED_WEIGHTS_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
PREPROCESS_ID = "rpent_dinov2_vits14_rgb224_bicubic_antialias_v1"
EXTRACTOR_ID = "dinov2_vits14_cls_token_v1"
DINOV2_DIMENSION = 384
MAX_BATCH_SIZE = 32
DISTANCE_METRIC = "one_minus_cosine_on_l2_cls384"


class Dinov2Backend(Protocol):
    torch_version: str
    torchvision_version: str
    device: str
    eval_mode: bool
    parameters_frozen: bool
    inference_only: bool

    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray: ...
    def close(self) -> None: ...


BackendLoader = Callable[
    ["Dinov2RevisionIdentity", "Dinov2DeploymentPaths"], Dinov2Backend
]


@dataclass(frozen=True, slots=True)
class Dinov2RevisionIdentity:
    model_id: str
    model_revision: str
    source_commit: str
    source_archive_sha256: str
    weights_sha256: str
    torch_version: str
    torchvision_version: str
    device: str
    preprocess_id: str = PREPROCESS_ID
    extractor_id: str = EXTRACTOR_ID
    dimension: int = DINOV2_DIMENSION

    def __post_init__(self) -> None:
        expected = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "device": "cuda",
            "preprocess_id": PREPROCESS_ID,
            "extractor_id": EXTRACTOR_ID,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                fail(
                    "MEMORY_DINOV2_IDENTITY_MISMATCH",
                    f"embedding.{field}",
                    f"expected {value!r}",
                )
        require_sha256(
            self.source_archive_sha256, path="embedding.source_archive_sha256"
        )
        require_sha256(self.weights_sha256, path="embedding.weights_sha256")
        if self.dimension != DINOV2_DIMENSION:
            fail(
                "MEMORY_DINOV2_DIMENSION_INVALID", "embedding.dimension", "expected 384"
            )
        for field in ("torch_version", "torchvision_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value.strip() != value:
                fail(
                    "MEMORY_DINOV2_IDENTITY_INVALID",
                    f"embedding.{field}",
                    "must be exact non-empty version",
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Dinov2RevisionIdentity":
        return cls(**dict(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "source_commit": self.source_commit,
            "source_archive_sha256": self.source_archive_sha256,
            "weights_sha256": self.weights_sha256,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "device": self.device,
            "preprocess_id": self.preprocess_id,
            "extractor_id": self.extractor_id,
            "dimension": self.dimension,
        }


@dataclass(frozen=True, slots=True)
class Dinov2DeploymentPaths:
    source_archive_path: Path
    weights_path: Path
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        for field in ("source_archive_path", "weights_path"):
            value = getattr(self, field)
            if not isinstance(value, Path) or not value.is_absolute():
                fail("MEMORY_DINOV2_DEPLOYMENT_INVALID", field, "must be absolute Path")
        if self.cache_dir is not None and (
            not isinstance(self.cache_dir, Path) or not self.cache_dir.is_absolute()
        ):
            fail(
                "MEMORY_DINOV2_DEPLOYMENT_INVALID",
                "cache_dir",
                "must be absolute Path or None",
            )


def l2_normalize_row(value: Any, *, path: str) -> np.ndarray:
    row = np.asarray(value, dtype=np.float64)
    if row.shape != (DINOV2_DIMENSION,) or not np.isfinite(row).all():
        fail("MEMORY_DINOV2_VECTOR_INVALID", path, "expected finite vector[384]")
    norm = float(np.linalg.norm(row))
    if norm <= 0.0:
        fail("MEMORY_DINOV2_VECTOR_INVALID", path, "cannot normalize zero vector")
    result = np.asarray(row / norm, dtype=np.float32)
    second = float(np.linalg.norm(result.astype(np.float64)))
    if second <= 0.0:
        fail("MEMORY_DINOV2_VECTOR_INVALID", path, "float32 normalization collapsed")
    result = result / np.float32(second)
    return np.ascontiguousarray(result, dtype=np.float32)


def l2_matrix(values: Any, *, path: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != DINOV2_DIMENSION
        or not np.isfinite(matrix).all()
    ):
        fail("MEMORY_DINOV2_MATRIX_INVALID", path, "expected finite matrix[N,384]")
    return (
        np.stack(
            [
                l2_normalize_row(row, path=f"{path}[{index}]")
                for index, row in enumerate(matrix)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if matrix.shape[0]
        else np.zeros((0, DINOV2_DIMENSION), dtype=np.float32)
    )


def one_minus_cosine(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = l2_matrix(query, path="query")
    c = l2_matrix(candidates, path="candidates")
    return np.asarray(1.0 - np.clip(q @ c.T, -1.0, 1.0), dtype=np.float32)


def _sha256_file(path: Path, *, label: str) -> str:
    if not path.is_file():
        fail("MEMORY_DINOV2_ASSET_MISSING", label, f"missing file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail("MEMORY_DINOV2_ASSET_UNREADABLE", label, f"{type(exc).__name__}: {exc}")
    return digest.hexdigest()


def _safe_extract_source(source_archive: Path, destination: Path) -> Path:
    try:
        with tarfile.open(source_archive, mode="r:*") as archive:
            members = archive.getmembers()
            if not members:
                fail(
                    "MEMORY_DINOV2_SOURCE_ARCHIVE_INVALID",
                    "source_archive",
                    "archive is empty",
                )
            for member in members:
                portable = PurePosixPath(member.name)
                if (
                    portable.is_absolute()
                    or ".." in portable.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    fail(
                        "MEMORY_DINOV2_SOURCE_ARCHIVE_INVALID",
                        "source_archive",
                        f"unsafe archive member {member.name!r}",
                    )
            archive.extractall(destination)
    except MemoryValidationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        fail(
            "MEMORY_DINOV2_SOURCE_ARCHIVE_INVALID",
            "source_archive",
            f"{type(exc).__name__}: {exc}",
        )
    hubconf_paths = tuple(destination.rglob("hubconf.py"))
    if len(hubconf_paths) != 1:
        fail(
            "MEMORY_DINOV2_SOURCE_ARCHIVE_INVALID",
            "source_archive",
            f"expected exactly one hubconf.py, found {len(hubconf_paths)}",
        )
    return hubconf_paths[0].parent


class _TorchDinov2Backend:
    def __init__(
        self,
        identity: Dinov2RevisionIdentity,
        deployment: Dinov2DeploymentPaths,
    ) -> None:
        source_sha = _sha256_file(
            deployment.source_archive_path, label="source_archive"
        )
        weights_sha = _sha256_file(deployment.weights_path, label="weights")
        if source_sha != identity.source_archive_sha256:
            fail(
                "MEMORY_DINOV2_ASSET_SHA256_MISMATCH",
                "source_archive",
                f"expected {identity.source_archive_sha256}, actual {source_sha}",
            )
        if weights_sha != identity.weights_sha256:
            fail(
                "MEMORY_DINOV2_ASSET_SHA256_MISMATCH",
                "weights",
                f"expected {identity.weights_sha256}, actual {weights_sha}",
            )

        # Heavy imports remain after complete asset validation and after the
        # service entry point has set CUDA_VISIBLE_DEVICES.
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        if str(torch.__version__) != identity.torch_version:
            fail(
                "MEMORY_DINOV2_BACKEND_IDENTITY_MISMATCH",
                "torch_version",
                f"expected {identity.torch_version!r}, actual {torch.__version__!r}",
            )
        if str(torchvision.__version__) != identity.torchvision_version:
            fail(
                "MEMORY_DINOV2_BACKEND_IDENTITY_MISMATCH",
                "torchvision_version",
                f"expected {identity.torchvision_version!r}, actual {torchvision.__version__!r}",
            )
        if identity.device != "cuda" or not torch.cuda.is_available():
            fail(
                "MEMORY_DINOV2_CUDA_UNAVAILABLE",
                "device",
                "the frozen encoder requires a visible CUDA device",
            )

        temporary_parent = deployment.cache_dir
        if temporary_parent is None and Path("/dev/shm").is_dir():
            temporary_parent = Path("/dev/shm")
        if temporary_parent is not None:
            try:
                temporary_parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                fail(
                    "MEMORY_DINOV2_CACHE_INVALID",
                    "cache_dir",
                    f"{type(exc).__name__}: {exc}",
                )
        self._temporary = tempfile.TemporaryDirectory(
            prefix="rpent-dinov2-source-",
            dir=os.fspath(temporary_parent) if temporary_parent is not None else None,
        )
        source_root = _safe_extract_source(
            deployment.source_archive_path,
            Path(self._temporary.name),
        )
        try:
            model = torch.hub.load(
                os.fspath(source_root),
                "dinov2_vits14",
                source="local",
                pretrained=False,
            )
            state = torch.load(
                deployment.weights_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            model.requires_grad_(False)
            model.eval()
            model.to(device="cuda")
        except Exception as exc:
            self._temporary.cleanup()
            fail(
                "MEMORY_DINOV2_MODEL_LOAD_FAILED",
                "encoder.backend",
                f"{type(exc).__name__}: {exc}",
            )
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            self._temporary.cleanup()
            fail(
                "MEMORY_DINOV2_MODEL_NOT_FROZEN",
                "encoder.backend",
                "model must be eval-only and frozen",
            )
        self._torch = torch
        self._functional = importlib.import_module("torchvision.transforms.functional")
        transforms = importlib.import_module("torchvision.transforms")
        self._bicubic = transforms.InterpolationMode.BICUBIC
        self._model = model
        self.torch_version = str(torch.__version__)
        self.torchvision_version = str(torchvision.__version__)
        self.device = "cuda"
        self.eval_mode = True
        self.parameters_frozen = True
        self.inference_only = True

    def _preprocess(self, image: np.ndarray) -> Any:
        torch = self._torch
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        height, width = image.shape[:2]
        if height <= width:
            resized_height = 256
            resized_width = int(round(width * 256.0 / height))
        else:
            resized_width = 256
            resized_height = int(round(height * 256.0 / width))
        tensor = self._functional.resize(
            tensor,
            [resized_height, resized_width],
            interpolation=self._bicubic,
            antialias=True,
        )
        tensor = self._functional.center_crop(tensor, [224, 224])
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        return self._functional.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def encode_batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if self._model.training or any(
            parameter.requires_grad for parameter in self._model.parameters()
        ):
            fail(
                "MEMORY_DINOV2_MODEL_NOT_FROZEN",
                "encoder.backend",
                "model state changed after admission",
            )
        batch = self._torch.stack([self._preprocess(image) for image in images])
        batch = batch.to(device="cuda", non_blocking=False)
        with self._torch.inference_mode():
            output = self._model(batch)
        if not isinstance(output, self._torch.Tensor):
            fail(
                "MEMORY_DINOV2_OUTPUT_INVALID",
                "encoder.output",
                f"expected Tensor, got {type(output).__name__}",
            )
        return output.detach().to(device="cpu", dtype=self._torch.float32).numpy()

    def close(self) -> None:
        self._model = None
        self._temporary.cleanup()


def _default_backend_loader(
    identity: Dinov2RevisionIdentity,
    deployment: Dinov2DeploymentPaths,
) -> Dinov2Backend:
    return _TorchDinov2Backend(identity, deployment)


class Dinov2Encoder:
    def __init__(
        self,
        identity: Dinov2RevisionIdentity,
        deployment: Dinov2DeploymentPaths,
        *,
        backend_loader: BackendLoader | None = None,
    ) -> None:
        self._identity = identity
        self._deployment = deployment
        self._loader = backend_loader or _default_backend_loader
        self._backend: Dinov2Backend | None = None
        self._closed = False

    def revision_metadata(self) -> dict[str, Any]:
        return self._identity.as_dict()

    def _backend_instance(self) -> Dinov2Backend:
        if self._closed:
            fail("MEMORY_DINOV2_ENCODER_CLOSED", "encoder", "encoder is closed")
        if self._backend is None:
            backend = self._loader(self._identity, self._deployment)
            expected = {
                "torch_version": self._identity.torch_version,
                "torchvision_version": self._identity.torchvision_version,
                "device": self._identity.device,
                "eval_mode": True,
                "parameters_frozen": True,
                "inference_only": True,
            }
            for field, wanted in expected.items():
                actual = getattr(backend, field, None)
                if actual != wanted:
                    fail(
                        "MEMORY_DINOV2_BACKEND_IDENTITY_MISMATCH",
                        field,
                        f"expected {wanted!r}, actual {actual!r}",
                    )
            self._backend = backend
        return self._backend

    def encode_batch(
        self, values: Sequence[np.ndarray | None]
    ) -> tuple[np.ndarray | None, ...]:
        if len(values) > MAX_BATCH_SIZE:
            fail(
                "MEMORY_DINOV2_BATCH_TOO_LARGE",
                "embedding_input",
                "max batch size is 32",
            )
        result: list[np.ndarray | None] = [None] * len(values)
        positions: list[int] = []
        images: list[np.ndarray] = []
        for index, value in enumerate(values):
            if value is None:
                continue
            image = np.asarray(value)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                fail(
                    "MEMORY_DINOV2_INPUT_INVALID",
                    f"embedding_input[{index}]",
                    "expected RGB8 [H,W,3]",
                )
            positions.append(index)
            images.append(np.ascontiguousarray(image))
        if not images:
            if self._closed:
                fail("MEMORY_DINOV2_ENCODER_CLOSED", "encoder", "encoder is closed")
            return tuple(result)
        raw = np.asarray(self._backend_instance().encode_batch(tuple(images)))
        if raw.shape != (len(images), DINOV2_DIMENSION):
            fail("MEMORY_DINOV2_OUTPUT_INVALID", "encoder.output", "expected [N,384]")
        for row, position in enumerate(positions):
            result[position] = l2_normalize_row(raw[row], path=f"encoder.output[{row}]")
        return tuple(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()


__all__ = [
    "DINOV2_DIMENSION",
    "DISTANCE_METRIC",
    "EXPECTED_SOURCE_ARCHIVE_SHA256",
    "Dinov2DeploymentPaths",
    "Dinov2Encoder",
    "Dinov2RevisionIdentity",
    "MemoryValidationError",
    "one_minus_cosine",
    "l2_matrix",
    "l2_normalize_row",
]
