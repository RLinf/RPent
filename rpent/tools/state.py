"""Per-run environment state and artifact storage."""
from __future__ import annotations

import copy
import fnmatch
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger("env_state")

_MANIFEST_NAME = "states.json"
_MANIFEST_VERSION = 2
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_TEXT_SUFFIXES = {".txt", ".md"}
_SUPPORTED_SUFFIXES = _IMAGE_SUFFIXES | {
    ".npy",
    ".json",
    ".jsonl",
    ".mp4",
    ".bin",
} | _TEXT_SUFFIXES


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
    artifacts: set[str] = field(default_factory=set)
    command: dict | None = None
    result: dict | None = None
    elapsed_s: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_blob(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "step_idx": self.step_idx,
            "state": self.state,
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

    @classmethod
    def from_blob(cls, blob: dict[str, Any]) -> "StepRecord":
        return cls(
            step_idx=int(blob["step_idx"]),
            state=dict(blob.get("state") or {}),
            artifacts={str(name) for name in blob.get("artifacts") or []},
            command=blob.get("command"),
            result=blob.get("result"),
            elapsed_s=blob.get("elapsed_s"),
            extras=dict(blob.get("extras") or {}),
        )


class EnvState:
    """Own a run's step trace and all state-related files in its output root."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list[StepRecord] = []
        self._pending_step: StepRecord | None = None
        self._run_artifacts: set[str] = set()
        self._next_step = 0
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
        return self._output_dir / f"{step:02d}_{name}"

    def _manifest_file(self) -> Path:
        return self._output_dir / _MANIFEST_NAME

    def _temporary_file(self, destination: Path) -> Path:
        return destination.with_name(
            f".{destination.stem}.tmp{destination.suffix}"
        )

    def _record_for_write(self, step: int) -> tuple[StepRecord, bool]:
        if self._pending_step is not None and self._pending_step.step_idx == step:
            return self._pending_step, False
        for record in self._steps:
            if record.step_idx == step:
                return record, True
        raise KeyError(f"step {step} not present in state trace")

    # -- manifest --------------------------------------------------------

    def _write_manifest(self) -> None:
        destination = self._manifest_file()
        temporary = self._temporary_file(destination)
        manifest = {
            "version": _MANIFEST_VERSION,
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
        """Remove state-owned artifacts and start a fresh trace."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        records = list(self._steps)
        if self._pending_step is not None:
            records.append(self._pending_step)
        for record in records:
            for name in record.artifacts:
                self._artifact_file(name, record.step_idx).unlink(missing_ok=True)
        for name in self._run_artifacts:
            self._artifact_file(name, None).unlink(missing_ok=True)
        self._manifest_file().unlink(missing_ok=True)
        self._steps = []
        self._pending_step = None
        self._run_artifacts = set()
        self._next_step = 0

    @property
    def next_step_idx(self) -> int:
        return self._next_step

    @property
    def latest_step(self) -> int | None:
        if not self._steps:
            return None
        return self._steps[-1].step_idx

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
        step: int | None,
        **options: Any,
    ) -> str | None:
        """Serialize ``value`` according to ``name`` and return its base name."""
        destination = self._artifact_file(name, step)
        temporary = self._temporary_file(destination)
        suffix = destination.suffix.lower()
        record: StepRecord | None = None
        committed = False
        if step is not None:
            record, committed = self._record_for_write(step)
        try:
            if suffix in _IMAGE_SUFFIXES:
                array = np.asarray(value)
                if array.dtype != np.uint8:
                    array = array.astype(np.uint8)
                imageio.imwrite(temporary, array)
            elif suffix == ".npy":
                np.save(temporary, np.asarray(value))
            elif suffix == ".json":
                with temporary.open("w") as file:
                    json.dump(value, file, indent=2, default=_json_default)
            elif suffix == ".jsonl":
                with temporary.open("w") as file:
                    if isinstance(value, str):
                        file.write(value)
                    else:
                        for item in value:
                            file.write(
                                json.dumps(item, default=_json_default) + "\n"
                            )
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
                if committed:
                    self._write_manifest()
            else:
                self._run_artifacts.add(name)
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

    def exists(self, name: str, *, step: int | None = -1) -> bool:
        try:
            resolved_step = self._resolve_read_step(step)
        except LookupError:
            return False
        return self._artifact_file(name, resolved_step).exists()

    def remove(self, name: str, *, step: int | None) -> bool:
        destination = self._artifact_file(name, step)
        if not destination.exists():
            return False
        record: StepRecord | None = None
        committed = False
        if step is not None:
            record, committed = self._record_for_write(step)
        destination.unlink()
        if step is None and name in self._run_artifacts:
            self._run_artifacts.remove(name)
            self._write_manifest()
        elif record is not None:
            was_recorded = name in record.artifacts
            record.artifacts.discard(name)
            if was_recorded and committed:
                self._write_manifest()
        return True

    def list(
        self,
        pattern: str = "*",
        *,
        step: int | None = -1,
    ) -> list[str]:
        resolved_step = self._resolve_read_step(step)
        if resolved_step is None:
            names = [
                name
                for name in self._run_artifacts
                if self._artifact_file(name, None).exists()
            ]
        else:
            record = self.get(resolved_step)
            names = [
                name
                for name in record.artifacts
                if self._artifact_file(name, resolved_step).exists()
            ]
        return sorted(name for name in names if fnmatch.fnmatch(name, pattern))

    def list_all(self, pattern: str = "*") -> list[tuple[int | None, str]]:
        artifacts: list[tuple[int | None, str]] = [
            (None, name) for name in self.list(pattern, step=None)
        ]
        for record in self._steps:
            artifacts.extend(
                (record.step_idx, name)
                for name in record.artifacts
                if fnmatch.fnmatch(name, pattern)
                and self._artifact_file(name, record.step_idx).exists()
            )
        return sorted(
            artifacts,
            key=lambda item: (-1 if item[0] is None else item[0], item[1]),
        )

    # -- step records ----------------------------------------------------

    @contextmanager
    def record_step(
        self,
        *,
        state: dict[str, Any],
        command: dict | None = None,
        result: dict | None = None,
        elapsed_s: float | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Iterator[int]:
        if self._pending_step is not None:
            raise RuntimeError("a step record is already open")
        record = StepRecord(
            step_idx=self._next_step,
            state=copy.deepcopy(state),
            command=copy.deepcopy(command),
            result=copy.deepcopy(result),
            elapsed_s=elapsed_s,
            extras=copy.deepcopy(extras or {}),
        )
        self._pending_step = record
        try:
            yield record.step_idx
        except BaseException:
            raise
        else:
            self._steps.append(record)
            try:
                self._write_manifest()
            except Exception:
                self._steps.pop()
                raise
            self._next_step = record.step_idx + 1
        finally:
            self._pending_step = None

    def get(self, step: int = -1) -> StepRecord:
        resolved_step = self._resolve_read_step(step)
        if resolved_step is None:
            raise ValueError("step records are not run-level artifacts")
        for record in self._steps:
            if record.step_idx == resolved_step:
                return copy.deepcopy(record)
        raise KeyError(f"step {resolved_step} not present in state trace")

    def records(self) -> list[StepRecord]:
        return copy.deepcopy(self._steps)

    # -- LLM-facing view -------------------------------------------------

    def view(
        self,
        step: int = -1,
        *,
        image_slots: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            record = self.get(step)
        except Exception as exc:
            return {"error": f"state step not available: {exc}"}

        metadata: dict[str, Any] = {}
        metadata_suffix = "_metadata.json"
        for name in sorted(record.artifacts):
            if not name.endswith(metadata_suffix):
                continue
            key = name.removesuffix(metadata_suffix)
            try:
                loaded = self.load(name, step=record.step_idx)
                if isinstance(loaded, dict):
                    loaded = {
                        field: value
                        for field, value in loaded.items()
                        if field not in {"K", "T_base_cam"}
                    }
                metadata[key] = loaded
            except Exception as exc:
                metadata[key] = {"error": str(exc)}

        out: dict[str, Any] = {
            "step": record.step_idx,
            "state": record.state,
            "artifacts": sorted(record.artifacts),
            "camera_meta": metadata,
            "log": {
                "command": record.command,
                "result": record.result,
                "elapsed_s": record.elapsed_s,
            },
        }
        if record.extras:
            out["extras"] = record.extras
        if image_slots:
            for slot, name in image_slots.items():
                if name not in record.artifacts:
                    continue
                try:
                    out[slot] = self.load_bytes(name, step=record.step_idx)
                except FileNotFoundError:
                    continue
        return out
