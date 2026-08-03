"""EnvState: the per-run state-trace owner.

A **step** is one motion-primitive tool call that produced a dumped state
snapshot, indexed from ``0`` where ``0`` is the post-``reset()`` baseline
(no command). Non-motion tool calls (``get_ee_pose``, ``back_project``,
``view_driver_state``, file/memory tools) are NOT steps and leave the trace
untouched.

``EnvState`` replaces the per-env ``_append_state`` / ``_load_states`` /
``_latest_step`` / ``_load_step`` / ``_load_image`` / ``_load_depth`` free
functions (and the toolkit's ``_next_step`` counter) with one explicit owner
constructed with an ``output_dir`` (no process-global). The toolkit and the
reader tools (``view_driver_state``, ``back_project``) hold a non-owning
reference to one ``EnvState`` per run; the composition root owns its lifecycle.

Artefact naming follows the LIBERO layout. Each saved artefact is a named
*stream* turned into a path by a fixed rule::

    stream "image"        -> images/image_NN.png
    stream "image_wrist"  -> images_wrist/image_wrist_NN.png
    stream "depth"         -> depths/depth_NN.npy
    stream "depth_wrist"  -> depths_wrist/depth_wrist_NN.npy
    stream "world"         -> world/world_NN.npy        (libero)
    stream "wrist_meta"    -> wrist_meta/wrist_meta_NN.json

i.e. the stream name is the file prefix; the directory pluralises the
artefact type (``image``->``images``, ``depth``->``depths``; ``world`` and
``wrist_meta`` stay singular, matching libero) and appends the camera suffix.
Franka maps scene->``image``/``depth`` and wrist->``image_wrist``/
``depth_wrist``; lerobot maps scene->``image``/``depth`` and
arm->``image_arm``. No per-env layout class is needed: the env's thin
``dump_state`` wrapper just picks the stream names.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np

from rpent.utils.logging import get_logger

logger = get_logger("env_state")

# Artefact type (the leading segment of a stream name) -> file extension.
_ARTIFACT_EXT: dict[str, str] = {
    "image": ".png",
    "depth": ".npy",
    "world": ".npy",
    "wrist_meta": ".json",
}

# Artefact type -> directory name (plural where libero pluralises).
_PLURAL: dict[str, str] = {
    "image": "images",
    "depth": "depths",
}

def _split_stream(stream: str) -> tuple[str, str]:
    for artifact_type in sorted(_ARTIFACT_EXT, key=len, reverse=True):
        if stream == artifact_type:
            return artifact_type, ""
        prefix = artifact_type + "_"
        if stream.startswith(prefix):
            return artifact_type, stream[len(prefix):]
    art, sep, suffix = stream.partition("_")
    return art, (suffix if sep else "")


def stream_dir(output_dir: Path, stream: str) -> Path:
    """Directory for a stream's artefacts (e.g. ``images_wrist``)."""
    art, suffix = _split_stream(stream)
    base = _PLURAL.get(art, art)
    return output_dir / (base if not suffix else f"{base}_{suffix}")


def stream_path(output_dir: Path, stream: str, step_idx: int, ext: str | None = None) -> Path:
    """Full path for one stream artefact at ``step_idx``."""
    art, _ = _split_stream(stream)
    if ext is None:
        ext = _ARTIFACT_EXT.get(art, ".bin")
    return stream_dir(output_dir, stream) / f"{stream}_{step_idx:02d}{ext}"


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One dumped step: the unit the LLM reads back via ``view_driver_state``.

    ``step_idx`` is the motion-primitive index (0 = post-reset baseline, which
    has ``command``/``result``/``elapsed_s`` = None). ``frames``/``depth`` are
    the stream names saved for this step (e.g. ``["image", "image_wrist"]``).
    ``camera_meta`` is the per-camera metadata dict at capture time (intrinsics
    + extrinsics + calibration status). ``extras`` absorbs env-specific fields
    (libero's world maps, ``libero_terminated``, task_language, ...).
    """

    step_idx: int
    state: dict
    command: dict | None = None
    result: dict | None = None
    elapsed_s: float | None = None
    frames: list[str] = field(default_factory=list)
    depth: list[str] = field(default_factory=list)
    camera_meta: dict | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_blob(self) -> dict[str, Any]:
        blob: dict[str, Any] = {
            "step_idx": self.step_idx,
            "state": self.state,
            "frames": self.frames,
            "depth": self.depth,
        }
        if self.camera_meta is not None:
            blob["camera_meta"] = self.camera_meta
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
    def from_blob(cls, blob: dict) -> "StepRecord":
        return cls(
            step_idx=int(blob.get("step_idx", -1)),
            state=blob.get("state", {}),
            command=blob.get("command"),
            result=blob.get("result"),
            elapsed_s=blob.get("elapsed_s"),
            frames=list(blob.get("frames", [])),
            depth=list(blob.get("depth", [])),
            camera_meta=blob.get("camera_meta"),
            extras=dict(blob.get("extras") or {}),
        )


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------


class EnvState:
    """Owns the on-disk state trace for one run.

    Constructed with an explicit ``output_dir`` (the composition root passes
    it; ``EnvState`` never reaches into a process-global). Owns the step
    counter, the ``states.json`` trace (atomic append), and the image/depth
    artefact layout. The toolkit and the reader tools (``view_driver_state``,
    ``back_project``) hold a non-owning reference to one ``EnvState`` per run.
    """

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list[dict] = self._read_states()
        self._next_step = self._derive_next_step()

    # -- internal: states.json I/O ----------------------------------------

    def _states_path(self) -> Path:
        return self.output_dir / "states.json"

    def _read_states(self) -> list[dict]:
        path = self._states_path()
        if not path.exists():
            return []
        try:
            with open(path) as f:
                arr = json.load(f)
            return [s for s in arr if isinstance(s, dict)] if isinstance(arr, list) else []
        except Exception as e:
            logger.warning("could not parse %s: %s; starting fresh", path, e)
            return []

    def _write_states_atomically(self, steps: list[dict]) -> None:
        path = self._states_path()
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(steps, f, indent=2, default=str)
        os.replace(tmp, path)

    def _derive_next_step(self) -> int:
        if not self._steps:
            return 0
        return max(int(s["step_idx"]) for s in self._steps if "step_idx" in s) + 1

    # -- lifecycle --------------------------------------------------------

    def reset(self, *, wipe_streams: Iterable[str] = ()) -> None:
        """Wipe stale artefact dirs + ``states.json`` for a fresh run.

        Resets the in-memory trace and counter. Step 0 (baseline) is dumped by
        the caller right after via :meth:`append`.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stream in wipe_streams:
            d = stream_dir(self.output_dir, stream)
            if d.exists():
                shutil.rmtree(d)
        states = self._states_path()
        if states.exists():
            states.unlink()
        self._steps = []
        self._next_step = 0

    # -- counter ----------------------------------------------------------

    @property
    def next_step_idx(self) -> int:
        """The step_idx the next dumped step will get (does NOT advance)."""
        return self._next_step

    @property
    def latest_step(self) -> int | None:
        """The highest step_idx successfully written (None if trace is empty)."""
        if not self._steps:
            return None
        return int(self._steps[-1]["step_idx"])

    # -- writing ----------------------------------------------------------

    def save_image(self, step_idx: int, stream: str, frame) -> str | None:
        """Save one RGB frame under ``<stream>_<NN>.png``; return ``stream`` on success."""
        path = stream_path(self.output_dir, stream, step_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        try:
            imageio.imwrite(path, arr)
            return stream
        except Exception as e:
            logger.warning("frame dump failed for stream %s: %s", stream, e)
            return None

    def save_depth(self, step_idx: int, stream: str, depth) -> str | None:
        """Save one depth map under ``<stream>_<NN>.npy``; return ``stream`` on success."""
        path = stream_path(self.output_dir, stream, step_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(path, np.asarray(depth, dtype=np.float32))
            return stream
        except Exception as e:
            logger.warning("depth dump failed for stream %s: %s", stream, e)
            return None

    def artifact_path(
        self,
        step_idx: int,
        stream: str,
        *,
        ext: str | None = None,
    ) -> Path:
        """Return the canonical path for one step artifact."""
        return stream_path(self.output_dir, stream, step_idx, ext=ext)

    def append(self, record: StepRecord) -> StepRecord:
        """Atomically append ``record`` to ``states.json`` and advance the counter.

        The counter and ``states.json`` are kept in sync: this write is the
        single source of truth for "latest step". If the write raises, the
        counter is NOT advanced (so the next caller reuses the same
        ``step_idx``) -- no desync between the in-memory counter and disk.
        """
        blob = record.to_blob()
        self._write_states_atomically(self._steps + [blob])
        self._steps.append(blob)
        self._next_step = max(self._next_step, record.step_idx + 1)
        return record

    # -- reading ----------------------------------------------------------

    def get(self, step_idx: int) -> StepRecord:
        """Look up the step record for ``step_idx``."""
        for blob in self._steps:
            if int(blob.get("step_idx", -1)) == step_idx:
                return StepRecord.from_blob(blob)
        raise KeyError(f"step {step_idx} not present in states.json")

    def records(self) -> list[StepRecord]:
        """Return the persisted step records in trace order."""
        return [StepRecord.from_blob(blob) for blob in self._steps]

    def load_image_bytes(self, step_idx: int, stream: str) -> bytes | None:
        path = stream_path(self.output_dir, stream, step_idx)
        if not path.exists():
            return None
        return path.read_bytes()

    def load_depth(self, step_idx: int, stream: str) -> np.ndarray:
        path = stream_path(self.output_dir, stream, step_idx)
        return np.load(path)

    # -- LLM-facing view (ex-view_driver_state) ---------------------------

    def view(self, step: int | None = None, *, image_slots: dict[str, str] | None = None) -> dict:
        """Build the ``view_driver_state`` dict for one step.

        ``image_slots`` maps a ToolResult image slot (``_image_bytes``,
        ``_image_cam_bytes``, ``_image_wrist_bytes``) to a stream name whose
        saved PNG should be embedded as bytes. The env decides which cameras
        map to which slots.
        """
        latest = self.latest_step
        if latest is None:
            return {"error": "no driver state entries; driver not ready"}
        nn = latest if step is None else int(step)
        try:
            rec = self.get(nn)
        except Exception as e:
            return {"error": f"step {nn} not present in driver state trace: {e}"}
        out: dict[str, Any] = {
            "step": nn,
            "state": rec.state,
            "frames": rec.frames,
            "depth": rec.depth,
            "camera_meta": {
                name: {k: v for k, v in meta.items() if k not in {"K", "T_base_cam"}}
                for name, meta in (rec.camera_meta or {}).items()
            },
            "log": {
                "command": rec.command,
                "result": rec.result,
                "elapsed_s": rec.elapsed_s,
            },
        }
        if rec.extras:
            out["extras"] = rec.extras
        if image_slots:
            for slot, stream in image_slots.items():
                b = self.load_image_bytes(nn, stream)
                if b:
                    out[slot] = b
        return out
