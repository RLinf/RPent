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

"""Per-session robot state and artifact storage."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger("env_state")

_MANIFEST_NAME = "states.json"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_TEXT_SUFFIXES = {".txt", ".md"}
_SUPPORTED_SUFFIXES = (
    _IMAGE_SUFFIXES
    | {
        ".npy",
        ".npz",
        ".json",
        ".jsonl",
        ".mp4",
        ".bin",
    }
    | _TEXT_SUFFIXES
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass
class StepRecord:
    """One motion step and the artifact base names captured for it."""

    step_idx: int
    state: dict[str, Any]
    terminated: bool = False
    truncated: bool = False
    artifacts: set[str] = field(default_factory=set)
    command: dict | None = None
    result: dict | None = None
    elapsed_s: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_blob(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "step_idx": self.step_idx,
            "state": self.state,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "artifacts": sorted(self.artifacts),
        }
        if self.command is not None:
            blob["command"] = self.command
        if self.result is not None:
            blob["result"] = self.result
        if self.elapsed_s is not None:
            blob["elapsed_s"] = self.elapsed_s
        if self.extras:
            blob["extras"] = self.extras
        return blob


class EnvState:
    """Own a session's step trace and all state-related files in its output root."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self.reset()

    # -- private file resolution -----------------------------------------

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("artifact name must be a non-empty string")
        path = Path(name)
        if path.name != name or path.is_absolute() or name in {".", ".."}:
            raise ValueError(f"artifact name must be a base filename: {name!r}")
        if name == _MANIFEST_NAME:
            raise ValueError(f"{_MANIFEST_NAME!r} is reserved")
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported artifact suffix: {path.suffix or '<none>'}")
        return name

    def _artifact_file(self, name: str, step: int | None) -> Path:
        name = self._validate_name(name)
        if step is None:
            return self._output_dir / name
        if step < 0:
            raise ValueError("artifact writes require a nonnegative step")
        artifact = Path(name)
        return self._output_dir / name / f"{step:02d}{artifact.suffix}"

    def _manifest_file(self) -> Path:
        return self._output_dir / _MANIFEST_NAME

    def _temporary_file(self, destination: Path) -> Path:
        return destination.with_name(f".{destination.stem}.tmp{destination.suffix}")

    def _record_for(self, step: int) -> StepRecord:
        """Return the live step record at ``step`` for in-place updates."""
        if 0 <= step < len(self._steps):
            return self._steps[step]
        raise KeyError(f"step {step} not present in state trace")

    # -- manifest --------------------------------------------------------

    def _write_manifest(self) -> None:
        destination = self._manifest_file()
        temporary = self._temporary_file(destination)
        manifest = {
            "run_artifacts": sorted(self._run_artifacts),
            "steps": [record.to_blob() for record in self._steps],
        }
        try:
            with temporary.open("w") as file:
                json.dump(manifest, file, indent=2, default=_json_default)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    # -- lifecycle and counters -----------------------------------------

    def reset(self) -> None:
        """Reset the in-memory trace without removing on-disk artifacts."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list[StepRecord] = []
        self._run_artifacts: set[str] = set()
        self._step_open = False

    @property
    def latest_step(self) -> int | None:
        if not self._steps:
            return None
        return self._steps[-1].step_idx

    def latest_record(self) -> StepRecord | None:
        """Return the most recently recorded step (live reference, no copy)."""
        return self._steps[-1] if self._steps else None

    def _resolve_read_step(self, step: int | None) -> int | None:
        if step is None:
            return None
        if step == -1:
            latest = self.latest_step
            if latest is None:
                raise LookupError("no steps available")
            return latest
        if step < 0:
            raise ValueError("step must be -1, None, or nonnegative")
        return step

    # -- generic artifact operations ------------------------------------

    def save(
        self,
        name: str,
        value: Any,
        *,
        step: int | None = -1,
        **options: Any,
    ) -> str | None:
        """Serialize ``value`` according to ``name`` and return its base name.

        ``step`` defaults to ``-1`` (the most recently recorded step), so calls
        made inside (or right after) a :meth:`record_step` block attach to that
        step without an explicit index. Pass an ``int`` to target a specific
        step, or ``None`` for a session-level artifact such as an episode video.
        """
        step = self._resolve_read_step(step)
        destination = self._artifact_file(name, step)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_file(destination)
        suffix = destination.suffix.lower()
        record: StepRecord | None = self._record_for(step) if step is not None else None
        try:
            if suffix in _IMAGE_SUFFIXES:
                array = np.asarray(value)
                if array.dtype != np.uint8:
                    array = array.astype(np.uint8)
                imageio.imwrite(temporary, array)
            elif suffix == ".npy":
                np.save(temporary, np.asarray(value))
            elif suffix == ".npz":
                np.savez_compressed(temporary, array=np.asarray(value))
            elif suffix == ".json":
                with temporary.open("w") as file:
                    json.dump(value, file, indent=2, default=_json_default)
            elif suffix == ".jsonl":
                with temporary.open("w") as file:
                    if isinstance(value, str):
                        file.write(value)
                    else:
                        for item in value:
                            file.write(json.dumps(item, default=_json_default) + "\n")
            elif suffix == ".mp4":
                if isinstance(value, (bytes, bytearray, memoryview)):
                    temporary.write_bytes(bytes(value))
                else:
                    imageio.mimwrite(
                        temporary,
                        list(value),
                        fps=int(options.get("fps", 20)),
                    )
            elif suffix in _TEXT_SUFFIXES:
                temporary.write_text(str(value))
            elif suffix == ".bin":
                temporary.write_bytes(bytes(value))
            else:
                raise ValueError(f"unsupported artifact suffix: {suffix}")
            os.replace(temporary, destination)
            if record is not None:
                record.artifacts.add(name)
            else:
                self._run_artifacts.add(name)
            if not self._step_open:
                self._write_manifest()
            return name
        except Exception as exc:
            logger.warning("failed to save artifact %s: %s", name, exc)
            return None
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, name: str, *, step: int | None = -1) -> Any:
        """Load an artifact; ``step=-1`` selects the latest recorded step."""
        resolved_step = self._resolve_read_step(step)
        source = self._artifact_file(name, resolved_step)
        suffix = source.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            return imageio.imread(source)
        if suffix == ".npy":
            return np.load(source)
        if suffix == ".npz":
            with np.load(source) as archive:
                return archive["array"]
        if suffix == ".json":
            with source.open() as file:
                return json.load(file)
        if suffix == ".jsonl":
            with source.open() as file:
                return [json.loads(line) for line in file if line.strip()]
        if suffix in _TEXT_SUFFIXES:
            return source.read_text()
        return source.read_bytes()

    def load_bytes(self, name: str, *, step: int | None = -1) -> bytes:
        resolved_step = self._resolve_read_step(step)
        return self._artifact_file(name, resolved_step).read_bytes()

    def artifact_path(self, name: str, *, step: int | None = -1) -> Path:
        """Return the canonical filesystem path for an artifact."""
        resolved_step = self._resolve_read_step(step)
        return self._artifact_file(name, resolved_step)

    def exists(self, name: str, *, step: int | None = -1) -> bool:
        try:
            resolved_step = self._resolve_read_step(step)
        except LookupError:
            return False
        return self._artifact_file(name, resolved_step).exists()

    # -- step records ----------------------------------------------------

    @contextmanager
    def record_step(
        self,
        *,
        state: dict[str, Any],
        terminated: bool = False,
        truncated: bool = False,
        command: dict | None = None,
        result: dict | None = None,
        elapsed_s: float | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Generator[int, None, None]:
        """Append a new step record and yield its index.

        The step is committed to the trace immediately, so any subsequent
        :meth:`save` without an explicit ``step`` attaches to it. If the block
        raises, the step and any artifacts written to it are rolled back.
        """
        if self._step_open:
            raise RuntimeError("a step record is already open")
        record = StepRecord(
            step_idx=len(self._steps),
            state=copy.deepcopy(state),
            terminated=terminated,
            truncated=truncated,
            command=copy.deepcopy(command),
            result=copy.deepcopy(result),
            elapsed_s=elapsed_s,
            extras=copy.deepcopy(extras or {}),
        )
        self._steps.append(record)
        self._step_open = True
        try:
            yield record.step_idx
        except BaseException as e:
            if self._steps and self._steps[-1] is record:
                self._steps.pop()
            logger.warning(
                "step %d is discarded due to exception during record: %s",
                record.step_idx,
                e,
            )
        finally:
            self._step_open = False
            self._write_manifest()

    def get(self, step: int = -1) -> StepRecord:
        resolved_step = self._resolve_read_step(step)
        if resolved_step is None:
            raise ValueError(f"step {step} must be -1 or nonnegative")
        return copy.deepcopy(self._record_for(resolved_step))

    def records(self) -> list[StepRecord]:
        return copy.deepcopy(self._steps)
