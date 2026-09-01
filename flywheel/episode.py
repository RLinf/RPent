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

"""Small in-memory writer for one LIBERO episode."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
_SUITE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _observation(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy the policy inputs before LIBERO can reuse their buffers."""
    main = np.array(obs["main_images"], dtype=np.uint8, copy=True, order="C")
    wrist = np.array(obs["wrist_images"], dtype=np.uint8, copy=True, order="C")
    state = np.array(obs["states"], dtype=np.float32, copy=True, order="C")
    if main.shape != (256, 256, 3) or wrist.shape != main.shape:
        raise ValueError("LIBERO policy images must have shape (256, 256, 3)")
    if state.shape != (8,):
        raise ValueError("LIBERO policy state must have shape (8,)")
    return main, wrist, state


def _action(value: Any) -> np.ndarray:
    action = np.array(value, dtype=np.float32, copy=True, order="C")
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("LIBERO action must be finite with shape (7,)")
    return action


class EpisodeWriter:
    """Collect one normal evaluation episode and publish it on close."""

    def __init__(
        self,
        root: Path | str,
        *,
        suite: str,
        task_id: int,
        seed: int,
        initial_observation: dict[str, Any],
    ) -> None:
        if not _SUITE.fullmatch(suite):
            raise ValueError(f"invalid LIBERO suite: {suite!r}")
        if any(type(value) is not int or value < 0 for value in (task_id, seed)):
            raise ValueError("task_id and seed must be non-negative integers")
        language = initial_observation.get("task_descriptions")
        if not isinstance(language, str) or not language:
            raise ValueError("LIBERO observation has no task description")
        first_observation = _observation(initial_observation)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.episode_id = f"episode_{stamp}_{uuid.uuid4().hex[:8]}"
        parent = (
            Path(root).expanduser().resolve()
            / "raw"
            / "libero"
            / suite
            / f"task_{task_id:02d}"
            / f"seed_{seed:03d}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / self.episode_id
        self._partial = self.path.with_name(f"{self.episode_id}.partial")
        self._partial.mkdir()

        self._metadata = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "suite": suite,
            "task_id": task_id,
            "seed": seed,
            "task_language": language,
        }
        self._observations = [first_observation]
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._primitive_ids: list[int] = []
        self._vla_ids: list[int] = []
        self._proposal_indices: list[int] = []
        self._primitive_names: list[str] = []
        self._active_primitive = -1
        self._proposals: list[dict[str, Any]] = []
        self._closed = False

    @property
    def step_count(self) -> int:
        return len(self._actions)

    def begin_primitive(self, name: str) -> None:
        self._active_primitive = len(self._primitive_names)
        self._primitive_names.append(name)

    def end_primitive(self) -> None:
        self._active_primitive = -1

    def add_proposal(self, instruction: str, actions: Any) -> int:
        proposal = np.array(actions, dtype=np.float32, copy=True, order="C")
        if (
            proposal.ndim != 2
            or proposal.shape[1] != 7
            or not np.isfinite(proposal).all()
        ):
            raise ValueError("VLA proposal must be finite with shape (horizon, 7)")
        vla_id = len(self._proposals)
        self._proposals.append(
            {
                "created_step": self.step_count,
                "primitive_id": self._active_primitive,
                "instruction": instruction,
                "actions": proposal,
            }
        )
        return vla_id

    def add_transition(
        self,
        action: Any,
        next_observation: dict[str, Any],
        reward: Any,
        terminated: Any,
        truncated: Any,
        *,
        vla_id: int = -1,
        proposal_index: int = -1,
    ) -> None:
        self._actions.append(_action(action))
        self._observations.append(_observation(next_observation))
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._primitive_ids.append(self._active_primitive)
        self._vla_ids.append(vla_id)
        self._proposal_indices.append(proposal_index)

    def finalize(self) -> Path:
        if self._closed:
            return self.path
        main, wrist, state = map(np.stack, zip(*self._observations, strict=True))
        actions = (
            np.stack(self._actions) if self._actions else np.empty((0, 7), np.float32)
        )
        with (self._partial / "transitions.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                main_images=main,
                wrist_images=wrist,
                states=state,
                actions=actions,
                rewards=np.asarray(self._rewards, np.float32),
                terminated=np.asarray(self._terminated, np.bool_),
                truncated=np.asarray(self._truncated, np.bool_),
                action_source=(
                    np.asarray(self._vla_ids, np.int32) >= 0
                ).astype(np.uint8),
                primitive_id=np.asarray(self._primitive_ids, np.int32),
                vla_chunk_id=np.asarray(self._vla_ids, np.int32),
                proposal_index=np.asarray(self._proposal_indices, np.int16),
            )

        proposal_actions = (
            np.stack([item["actions"] for item in self._proposals])
            if self._proposals
            else np.empty((0, 0, 7), np.float32)
        )
        with (self._partial / "proposals.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                actions=proposal_actions,
                created_step=np.asarray(
                    [item["created_step"] for item in self._proposals], np.int32
                ),
                primitive_id=np.asarray(
                    [item["primitive_id"] for item in self._proposals], np.int32
                ),
                instruction=np.asarray(
                    [item["instruction"] for item in self._proposals], dtype=np.str_
                ),
            )

        success_steps = np.flatnonzero(np.asarray(self._terminated, np.bool_))
        success = bool(success_steps.size)
        metadata = {
            **self._metadata,
            "is_success": success,
            "stop_reason": (
                "env_terminated"
                if success
                else "env_truncated"
                if any(self._truncated)
                else "agent_stopped"
            ),
            "step_count": self.step_count,
            "training_step_count": int(success_steps[0] + 1) if success else 0,
            "primitive_names": self._primitive_names,
            "proposal_count": len(self._proposals),
        }
        (self._partial / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_episode(self._partial)
        os.replace(self._partial, self.path)
        self._closed = True
        return self.path


def validate_episode(path: Path | str) -> dict[str, Any]:
    """Validate the alignment needed by the exporter and training loader."""
    root = Path(path)
    metadata = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema in {root}")
    with np.load(root / "transitions.npz", allow_pickle=False) as data:
        count = int(metadata["step_count"])
        if data["actions"].shape != (count, 7):
            raise ValueError(f"invalid action shape in {root}")
        for key in ("main_images", "wrist_images"):
            if data[key].shape != (count + 1, 256, 256, 3):
                raise ValueError(f"invalid {key} shape in {root}")
        if data["states"].shape != (count + 1, 8):
            raise ValueError(f"invalid state shape in {root}")
        for key in (
            "rewards",
            "terminated",
            "truncated",
            "action_source",
            "primitive_id",
            "vla_chunk_id",
            "proposal_index",
        ):
            if data[key].shape != (count,):
                raise ValueError(f"invalid {key} shape in {root}")
        terminated = data["terminated"]
        expected = int(np.flatnonzero(terminated)[0] + 1) if terminated.any() else 0
        if (
            metadata["is_success"] != bool(terminated.any())
            or metadata["training_step_count"] != expected
        ):
            raise ValueError(f"invalid training boundary in {root}")
        if (
            not np.isfinite(data["actions"]).all()
            or not np.isfinite(data["states"]).all()
        ):
            raise ValueError(f"non-finite training data in {root}")
    proposal_count = metadata["proposal_count"]
    with np.load(root / "proposals.npz", allow_pickle=False) as proposals:
        actions = proposals["actions"]
        if (
            actions.ndim != 3
            or actions.shape[0] != proposal_count
            or actions.shape[2] != 7
        ):
            raise ValueError(f"invalid proposal action shape in {root}")
        for key in ("created_step", "primitive_id", "instruction"):
            if proposals[key].shape != (proposal_count,):
                raise ValueError(f"invalid proposal {key} shape in {root}")
    return metadata
