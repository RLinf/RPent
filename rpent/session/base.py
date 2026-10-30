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
import math
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import IO, Any, Literal
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np

from rpent.session.records import (
    ActionRecord,
    ArtifactRef,
    DataRef,
    MissingInfo,
    ObservationRecord,
    Payload,
    PlannerRecord,
    PredictionRecord,
    TaskRecord,
    ToolCallRecord,
)
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
    """Own task records, legacy tool steps, and stored artifacts."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self._artifact_lock = threading.Lock()
        self._records_lock = threading.Lock()
        self._task: TaskRecord | None = None
        self._planners: dict[str, PlannerRecord] = {}
        self._tool_calls: dict[str, ToolCallRecord] = {}
        self._observations: dict[str, ObservationRecord] = {}
        self._actions: dict[str, ActionRecord] = {}
        self._predictions: dict[str, PredictionRecord] = {}
        self._id_counters = dict.fromkeys(("P", "T", "O", "A", "V"), 0)
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

    # -- task lifecycle -------------------------------------------------

    def start_task(self, *, instruction: str, config: Payload) -> str:
        """Start this store's single task and return its ID.

        A new task needs a new EnvState; reset() does not clear task records.
        """
        with self._records_lock:
            if self._task is not None:
                raise RuntimeError("a task has already been started")
            self._task = TaskRecord(
                task_id=uuid4().hex,
                instruction=instruction,
                config=copy.deepcopy(config),
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            return self._task.task_id

    def end_task(
        self,
        *,
        status: Literal["completed", "failed", "cancelled"],
        stop_reason: str,
        result: Payload | None = None,
    ) -> None:
        """Record task completion after all execution and capture have settled."""
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid task end status: {status!r}")
        with self._records_lock:
            task = self._running_task()
            result = copy.deepcopy(result)
            task.status = status
            task.stop_reason = stop_reason
            task.result = result
            task.ended_at = datetime.now(timezone.utc)

    def get_task(self) -> TaskRecord | None:
        """Return a task snapshot, or None before start_task()."""
        with self._records_lock:
            return copy.deepcopy(self._task)

    def _running_task(self) -> TaskRecord:
        if self._task is None or self._task.status != "running":
            raise RuntimeError("task is not running")
        return self._task

    def _new_record_id(self, kind: Literal["P", "T", "O", "A", "V"]) -> str:
        """Allocate an ID while holding the records lock."""
        self._id_counters[kind] += 1
        return f"{kind}{self._id_counters[kind]}"

    # -- task record registration ---------------------------------------

    def register_planner(
        self,
        *,
        backend: str,
        status: Literal["open", "completed", "failed", "cancelled"] = "open",
        request: Payload | None = None,
        response: Payload | None = None,
        source_ids: dict[str, str] | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        missing: dict[str, MissingInfo] | None = None,
    ) -> str:
        """Register one model call; tool calls are registered separately."""
        with self._records_lock:
            self._running_task()
            record = copy.deepcopy(
                PlannerRecord(
                    planner_id=self._new_record_id("P"),
                    backend=backend,
                    status=status,
                    request=request,
                    response=response,
                    source_ids=source_ids or {},
                    started_at=started_at,
                    ended_at=ended_at,
                    missing=missing or {},
                )
            )
            self._planners[record.planner_id] = record
            return record.planner_id

    def register_tool_call(
        self,
        *,
        planner_id: str,
        name: str,
        arguments: Payload,
        status: Literal[
            "requested",
            "running",
            "completed",
            "failed",
            "cancelled",
            "not_executed",
            "unknown",
        ] = "requested",
        source_ids: dict[str, str] | None = None,
        effective_instruction: str | None = None,
        model_config: Payload | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        raw_result: Payload | None = None,
        feedback: Payload | None = None,
        before_observation_id: str | None = None,
        after_observation_id: str | None = None,
        missing: dict[str, MissingInfo] | None = None,
    ) -> str:
        """Register a model-issued tool call in response order and index it."""
        with self._records_lock:
            self._running_task()
            planner = self._planners[planner_id]
            record = copy.deepcopy(
                ToolCallRecord(
                    tool_call_id=self._new_record_id("T"),
                    planner_id=planner_id,
                    origin="model",
                    name=name,
                    arguments=arguments,
                    status=status,
                    source_ids=source_ids or {},
                    effective_instruction=effective_instruction,
                    model_config=model_config,
                    started_at=started_at,
                    ended_at=ended_at,
                    raw_result=raw_result,
                    feedback=feedback,
                    before_observation_id=before_observation_id,
                    after_observation_id=after_observation_id,
                    missing=missing or {},
                )
            )
            planner.tool_calls.append(record)
            self._tool_calls[record.tool_call_id] = record
            return record.tool_call_id

    def register_observation(
        self,
        *,
        payload: Payload,
        captured_at: datetime | None = None,
        missing: dict[str, MissingInfo] | None = None,
    ) -> str:
        """Register a new capture in acquisition order; reuse its ID to share it."""
        with self._records_lock:
            self._running_task()
            record = copy.deepcopy(
                ObservationRecord(
                    observation_id=self._new_record_id("O"),
                    payload=payload,
                    captured_at=captured_at,
                    missing=missing or {},
                )
            )
            self._observations[record.observation_id] = record
            return record.observation_id

    def register_action(
        self,
        *,
        tool_call_id: str,
        source: Literal["vla", "tool_computed", "direct"],
        action: Payload,
        before_observation_id: str | None,
        after_observation_id: str | None,
        executed_at: datetime | None = None,
        result: Payload | None = None,
        prediction_id: str | None = None,
        missing: dict[str, MissingInfo] | None = None,
    ) -> str:
        """Register a confirmed action and link it to its VLA prediction, if given."""
        with self._records_lock:
            self._running_task()
            if tool_call_id not in self._tool_calls:
                raise KeyError(tool_call_id)
            prediction = (
                self._predictions[prediction_id] if prediction_id is not None else None
            )
            if prediction is not None and (
                source != "vla" or prediction.tool_call_id != tool_call_id
            ):
                raise ValueError(
                    "prediction requires a VLA action from the same tool call"
                )
            record = copy.deepcopy(
                ActionRecord(
                    action_id=self._new_record_id("A"),
                    tool_call_id=tool_call_id,
                    source=source,
                    action=action,
                    before_observation_id=before_observation_id,
                    after_observation_id=after_observation_id,
                    executed_at=executed_at,
                    result=result,
                    missing=missing or {},
                )
            )
            self._actions[record.action_id] = record
            if prediction is not None:
                prediction.action_ids.append(record.action_id)
            return record.action_id

    def register_prediction(
        self,
        *,
        tool_call_id: str,
        instruction: str,
        action_ids: list[str] | None = None,
        missing: dict[str, MissingInfo] | None = None,
    ) -> str:
        """Register a VLA prediction, optionally linking already executed actions."""
        with self._records_lock:
            self._running_task()
            tool_call = self._tool_calls[tool_call_id]
            action_ids = action_ids or []
            for action_id in action_ids:
                if self._actions[action_id].tool_call_id != tool_call_id:
                    raise ValueError(
                        f"action {action_id!r} does not belong to tool call {tool_call_id!r}"
                    )
            record = copy.deepcopy(
                PredictionRecord(
                    prediction_id=self._new_record_id("V"),
                    planner_id=tool_call.planner_id,
                    tool_call_id=tool_call_id,
                    instruction=instruction,
                    action_ids=action_ids,
                    missing=missing or {},
                )
            )
            self._predictions[record.prediction_id] = record
            return record.prediction_id

    # -- task record updates --------------------------------------------

    def update_planner(
        self,
        planner_id: str,
        *,
        status: Literal["open", "completed", "failed", "cancelled"] | None = None,
        response: Payload | None = None,
        source_ids: dict[str, str] | None = None,
        ended_at: datetime | None = None,
        missing: dict[str, MissingInfo | None] | None = None,
    ) -> None:
        """Update a model call while its task is running.

        None leaves fields unchanged; response can only be filled once.
        Source IDs merge by key; None values in ``missing`` remove reasons.
        """
        with self._records_lock:
            self._running_task()
            record = self._planners[planner_id]
            if status is not None and status not in {
                "open",
                "completed",
                "failed",
                "cancelled",
            }:
                raise ValueError(f"invalid planner status: {status!r}")
            self._update_call_record(
                record,
                values={"status": status, "response": response, "ended_at": ended_at},
                source_ids=source_ids,
                missing=missing,
            )

    def update_tool_call(
        self,
        tool_call_id: str,
        *,
        status: Literal[
            "requested",
            "running",
            "completed",
            "failed",
            "cancelled",
            "not_executed",
            "unknown",
        ]
        | None = None,
        effective_instruction: str | None = None,
        model_config: Payload | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        raw_result: Payload | None = None,
        feedback: Payload | None = None,
        before_observation_id: str | None = None,
        after_observation_id: str | None = None,
        source_ids: dict[str, str] | None = None,
        missing: dict[str, MissingInfo | None] | None = None,
    ) -> None:
        """Update a tool call while its task is running.

        None leaves fields unchanged; model config, raw result, and feedback
        can each only be filled once. Source IDs merge by key; None values in
        ``missing`` remove reasons. Observation IDs marked missing are cleared.
        """
        with self._records_lock:
            self._running_task()
            record = self._tool_calls[tool_call_id]
            if status is not None and status not in {
                "requested",
                "running",
                "completed",
                "failed",
                "cancelled",
                "not_executed",
                "unknown",
            }:
                raise ValueError(f"invalid tool call status: {status!r}")
            self._update_call_record(
                record,
                values={
                    "status": status,
                    "effective_instruction": effective_instruction,
                    "model_config": model_config,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "raw_result": raw_result,
                    "feedback": feedback,
                    "before_observation_id": before_observation_id,
                    "after_observation_id": after_observation_id,
                },
                source_ids=source_ids,
                missing=missing,
            )
            for name in ("before_observation_id", "after_observation_id"):
                if missing and missing.get(name) is not None:
                    setattr(record, name, None)

    @staticmethod
    def _update_call_record(
        record: PlannerRecord | ToolCallRecord,
        *,
        values: dict[str, Any],
        source_ids: dict[str, str] | None,
        missing: dict[str, MissingInfo | None] | None,
    ) -> None:
        """Apply a call update in place while holding the records lock."""
        values = {name: value for name, value in values.items() if value is not None}
        for name in ("response", "model_config", "raw_result", "feedback"):
            if name in values and getattr(record, name) is not None:
                raise ValueError(f"{name} has already been captured")
        values, source_ids, missing = copy.deepcopy(
            (values, source_ids or {}, missing or {})
        )
        for name, value in values.items():
            setattr(record, name, value)
        record.source_ids.update(source_ids)
        for name, reason in missing.items():
            if reason is None:
                record.missing.pop(name, None)
            else:
                record.missing[name] = reason

    # -- task record snapshots ------------------------------------------

    def get_planner(self, planner_id: str) -> PlannerRecord:
        """Return a model call and its tool calls as a snapshot."""
        with self._records_lock:
            return copy.deepcopy(self._planners[planner_id])

    def get_tool_call(self, tool_call_id: str) -> ToolCallRecord:
        """Return a tool call snapshot by ID."""
        with self._records_lock:
            return copy.deepcopy(self._tool_calls[tool_call_id])

    def get_observation(self, observation_id: str) -> ObservationRecord:
        """Return an observation snapshot by ID."""
        with self._records_lock:
            return copy.deepcopy(self._observations[observation_id])

    def get_action(self, action_id: str) -> ActionRecord:
        """Return an executed action snapshot by ID."""
        with self._records_lock:
            return copy.deepcopy(self._actions[action_id])

    def planner_records(self) -> list[PlannerRecord]:
        """Return model call snapshots in registration order."""
        with self._records_lock:
            return copy.deepcopy(list(self._planners.values()))

    def action_records(self) -> list[ActionRecord]:
        """Return executed action snapshots in registration order."""
        with self._records_lock:
            return copy.deepcopy(list(self._actions.values()))

    def prediction_records(self) -> list[PredictionRecord]:
        """Return VLA prediction snapshots in registration order."""
        with self._records_lock:
            return copy.deepcopy(list(self._predictions.values()))

    # -- trajectory storage ---------------------------------------------

    def save_trajectory_artifact(
        self, value: np.ndarray | bytes, *, name: str | None = None
    ) -> ArtifactRef:
        """Store an array or bytes in the task's trajectory and return its reference."""
        if self._task is None:
            raise RuntimeError(
                "start_task() must run before saving trajectory artifacts"
            )
        if not isinstance(value, (np.ndarray, bytes)):
            raise TypeError("trajectory artifacts must be numpy arrays or bytes")
        is_bytes = isinstance(value, bytes)
        suffix = ".bin" if is_bytes else ".npy"
        name = self._validate_name(
            name if name is not None else f"capture-{uuid4().hex}{suffix}"
        )
        if Path(name).suffix != suffix:
            raise ValueError(f"trajectory artifact name must end with {suffix}")
        path = f"artifacts/{name}"
        destination = self._output_dir / "trajectories" / self._task.task_id / path
        with self._artifact_lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self._write_trajectory_file(destination, binary=True) as file:
                if is_bytes:
                    file.write(value)
                else:
                    np.save(file, value, allow_pickle=False)
        return ArtifactRef(
            path=path,
            kind="bytes" if is_bytes else "ndarray",
            dtype=None if is_bytes else str(value.dtype),
            shape=None if is_bytes else value.shape,
        )

    def save_planner_trajectory(self) -> Path:
        """Save task, planner, action, and prediction records after end_task().

        Attachment I/O failures are marked as missing in the saved payload.
        Record-file write failures propagate to the caller. Each file is
        replaced after a successful write; the directory is not atomic.

        Returns:
            The output directory's trajectories/<task_id> subdirectory.
        """
        if self._task is None or self._task.status == "running":
            raise RuntimeError("end_task() must finish before saving the trajectory")
        root = self._output_dir / "trajectories" / self._task.task_id
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        reserved_paths: set[Path] = set()
        byte_artifacts: dict[str, ArtifactRef] = {}
        attachment_count = 0

        def reserve_artifacts(value: Any) -> None:
            if isinstance(value, ArtifactRef):
                reserved_paths.add((root / value.path).resolve())
            elif is_dataclass(value):
                for item in fields(value):
                    reserve_artifacts(getattr(value, item.name))
            elif isinstance(value, dict):
                for item in value.values():
                    reserve_artifacts(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    reserve_artifacts(item)

        reserve_artifacts(self._task)
        for planner in self._planners.values():
            reserve_artifacts(planner)
        for observation in self._observations.values():
            reserve_artifacts(observation)
        for action in self._actions.values():
            reserve_artifacts(action)
        for prediction in self._predictions.values():
            reserve_artifacts(prediction)

        def encode(
            value: Any,
            *,
            missing: dict[str, MissingInfo] | None = None,
            pointer: str = "",
            inline_array_paths: tuple[str, ...] = (),
        ) -> Any:
            nonlocal attachment_count
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, np.ndarray) and pointer in inline_array_paths:
                return {
                    "$type": "ndarray",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "value": encode(value.tolist(), missing=missing, pointer=pointer),
                }
            if isinstance(value, (np.ndarray, bytes)):
                is_bytes = isinstance(value, bytes)
                digest = sha256(value).hexdigest() if is_bytes else None
                if digest is not None and digest in byte_artifacts:
                    return encode(byte_artifacts[digest])
                suffix = ".bin" if is_bytes else ".npy"
                while True:
                    attachment_count += 1
                    path = f"artifacts/payload-{attachment_count:06d}{suffix}"
                    destination = root / path
                    if not any(
                        candidate.resolve() in reserved_paths
                        for candidate in (
                            destination,
                            self._temporary_file(destination),
                        )
                    ):
                        break
                try:
                    value = self.save_trajectory_artifact(value, name=destination.name)
                except OSError as exc:
                    if missing is None:
                        raise
                    missing[pointer] = MissingInfo(
                        "capture_failed", f"{type(exc).__name__}: {exc}"
                    )
                    return None
                if digest is not None:
                    byte_artifacts[digest] = value
            if is_dataclass(value):
                is_payload = isinstance(value, Payload)
                failures = {} if is_payload else None
                data = {
                    item.name: encode(
                        getattr(value, item.name),
                        missing=failures if item.name == "data" else None,
                        inline_array_paths=(
                            value.inline_array_paths
                            if is_payload and item.name == "data"
                            else ()
                        ),
                    )
                    for item in fields(value)
                    if not (is_payload and item.name == "inline_array_paths")
                }
                if failures:
                    data["missing"].update(encode(failures))
                    data["completeness"] = "partial"
                if isinstance(value, (ArtifactRef, DataRef)):
                    data["$type"] = (
                        "artifact_ref" if isinstance(value, ArtifactRef) else "data_ref"
                    )
                return data
            if isinstance(value, dict):
                data = {
                    key: encode(
                        item,
                        missing=missing,
                        pointer=f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}",
                        inline_array_paths=inline_array_paths,
                    )
                    for key, item in value.items()
                }
                return {"$type": "dict", "value": data} if "$type" in value else data
            if isinstance(value, (list, tuple)):
                return [
                    encode(
                        item,
                        missing=missing,
                        pointer=f"{pointer}/{index}",
                        inline_array_paths=inline_array_paths,
                    )
                    for index, item in enumerate(value)
                ]
            if isinstance(value, float) and not math.isfinite(value):
                return {"$type": "float", "value": str(value)}
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Path):
                return str(value)
            return value

        with self._write_trajectory_file(root / "planners.jsonl") as file:
            for planner in self._planners.values():
                json.dump(encode(planner), file, ensure_ascii=False, allow_nan=False)
                file.write("\n")
        observations: dict[str, Any] = {}
        for filename, records, sides in (
            ("actions.jsonl", self._actions.values(), ("before", "after")),
            ("predictions.jsonl", self._predictions.values(), ()),
        ):
            with self._write_trajectory_file(root / filename) as file:
                for record in records:
                    data = encode(record)
                    for side in sides:
                        observation_id = getattr(record, f"{side}_observation_id")
                        if (
                            observation_id is not None
                            and observation_id not in observations
                        ):
                            observations[observation_id] = encode(
                                self._observations[observation_id]
                            )
                        data[f"{side}_observation"] = (
                            observations[observation_id]
                            if observation_id is not None
                            else None
                        )
                    json.dump(data, file, ensure_ascii=False, allow_nan=False)
                    file.write("\n")
        with self._write_trajectory_file(root / "task.json") as file:
            json.dump(
                {"schema_version": 5, "task": encode(self._task)},
                file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        return root

    @contextmanager
    def _write_trajectory_file(
        self, destination: Path, *, binary: bool = False
    ) -> Generator[IO[Any], None, None]:
        """Replace a trajectory file only after its write succeeds."""
        temporary = self._temporary_file(destination)
        try:
            with temporary.open(
                "wb" if binary else "w", encoding=None if binary else "utf-8"
            ) as file:
                yield file
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    # -- legacy lifecycle and counters ----------------------------------

    def reset(self) -> None:
        """Clear legacy tool state, preserving task records and disk artifacts."""
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
        with self._artifact_lock:
            step = self._resolve_read_step(step)
            destination = self._artifact_file(name, step)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._temporary_file(destination)
            suffix = destination.suffix.lower()
            record: StepRecord | None = (
                self._record_for(step) if step is not None else None
            )
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
                            codec="libx264",
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

    def prune_artifacts(
        self, names: tuple[str, ...], *, step: int, keep_last: int
    ) -> None:
        """Remove named files for the step leaving the retention window.

        Call for each new step. Step records retain the names of the artifacts
        originally captured, including files that have since been pruned.
        """
        old = step - keep_last
        if old < 0:
            return
        for name in names:
            path = self._artifact_file(name, old)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "failed to prune artifact %s at step %d: %s", name, old, exc
                )

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
            for name in record.artifacts:
                try:
                    self._artifact_file(name, record.step_idx).unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning(
                        "failed to remove artifact %s from discarded step %d: %s",
                        name,
                        record.step_idx,
                        cleanup_error,
                    )
            if self._steps and self._steps[-1] is record:
                self._steps.pop()
            logger.warning(
                "step %d is discarded due to exception during record: %s",
                record.step_idx,
                e,
            )
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(f"failed to record step {record.step_idx}: {e}") from e
        finally:
            self._step_open = False
            self._write_manifest()

    def get(self, step: int = -1) -> StepRecord:
        resolved_step = self._resolve_read_step(step)
        if resolved_step is None:
            raise ValueError(f"step {step} must be -1 or nonnegative")
        with self._artifact_lock:
            return copy.deepcopy(self._record_for(resolved_step))

    def records(self) -> list[StepRecord]:
        return copy.deepcopy(self._steps)
