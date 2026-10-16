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

"""Bounded episode video recording for BEHAVIOR."""

from __future__ import annotations

import os
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.session import EnvState


class VideoArtifactWriter:
    """Streaming MP4 writer that registers the artifact only on successful close."""

    def __init__(
        self,
        env_state: EnvState,
        *,
        name: str,
        step: int | None,
        fps: int,
        max_frames: int | None,
    ) -> None:
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if max_frames is not None and int(max_frames) <= 0:
            raise ValueError("max_frames must be positive when provided")
        self._env_state = env_state
        self._name = name
        self._step = step
        self._fps = fps
        self._max_frames = None if max_frames is None else int(max_frames)
        self._destination = env_state.artifact_path(self._name, step=step)
        self._temporary = self._destination.with_name(
            f".{self._destination.stem}.{os.getpid()}.{id(self)}.tmp"
            f"{self._destination.suffix}"
        )
        self._writer: Any | None = None
        self._closed = False
        self._aborted = False
        self.frames_written = 0
        self.frames_dropped = 0

    def append(self, frame: Any) -> bool:
        """Append one RGB frame; return False when the configured cap is reached."""

        if self._closed:
            raise RuntimeError("video writer is closed")
        if self._aborted:
            return False
        if self._max_frames is not None and self.frames_written >= self._max_frames:
            self.frames_dropped += 1
            return False
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("video frame must have shape [H, W, C>=3]")
        array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if self._writer is None:
            self._destination.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(self._temporary, fps=self._fps)
        self._writer.append_data(np.ascontiguousarray(array))
        self.frames_written += 1
        return True

    def close(self) -> str | None:
        """Finalize the video and register it in the EnvState manifest."""

        if self._closed:
            return self._name if self._destination.exists() else None
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._aborted or self.frames_written <= 0:
                self._temporary.unlink(missing_ok=True)
                return None
            saved = self._env_state.save(
                self._name, self._temporary.read_bytes(), step=self._step
            )
            if saved is None:
                raise RuntimeError(f"failed to save {self._name}")
            return saved
        except Exception:
            self.abort()
            raise
        finally:
            self._temporary.unlink(missing_ok=True)

    def abort(self) -> None:
        """Close and remove the temporary file without publishing the artifact."""

        if self._aborted:
            return
        self._aborted = True
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        finally:
            self._closed = True
            self._temporary.unlink(missing_ok=True)
