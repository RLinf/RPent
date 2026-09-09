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

"""Small in-memory episode writer using caller-supplied data rules."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _array(value: Any, field: dict[str, Any]) -> np.ndarray:
    """Copy one policy input before the caller can reuse its buffer."""
    array = np.array(value, dtype=field["dtype"], copy=True, order="C")
    if array.shape != field["shape"]:
        raise ValueError(f"expected array shape {field['shape']}; got {array.shape}")
    return array


def _observation(obs: dict[str, Any], spec: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: _array(obs[key], field)
        for key, field in spec["arrays"].items()
        if key != "actions"
    }


def _training_step_count(transitions: Any, spec: dict[str, Any]) -> int:
    mask = np.asarray(spec["success_mask"](transitions))
    if mask.dtype != np.bool_ or mask.shape != (len(transitions["actions"]),):
        raise ValueError("success_mask must return one boolean per action")
    steps = np.flatnonzero(mask)
    return int(steps[0] + 1) if steps.size else 0


class EpisodeWriter:
    """Collect one normal evaluation episode and publish it on close."""

    def __init__(
        self,
        parent: Path | str,
        *,
        metadata: dict[str, Any],
        spec: dict[str, Any],
        initial_observation: dict[str, Any],
    ) -> None:
        self._spec = spec
        first_observation = _observation(initial_observation, spec)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.episode_id = f"episode_{stamp}_{uuid.uuid4().hex[:8]}"
        parent = Path(parent).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / self.episode_id
        self._partial = self.path.with_name(f"{self.episode_id}.partial")
        self._partial.mkdir()

        self._metadata = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
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
        field = self._spec["arrays"]["actions"]
        proposal = np.array(actions, dtype=field["dtype"], copy=True, order="C")
        if proposal.shape[1:] != field["shape"] or not np.isfinite(proposal).all():
            raise ValueError(
                "VLA proposal must be finite with shape (horizon, *action_shape)"
            )
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
        action = _array(action, self._spec["arrays"]["actions"])
        if not np.isfinite(action).all():
            raise ValueError("action must be finite")
        self._actions.append(action)
        self._observations.append(_observation(next_observation, self._spec))
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._primitive_ids.append(self._active_primitive)
        self._vla_ids.append(vla_id)
        self._proposal_indices.append(proposal_index)

    def finalize(self) -> Path:
        if self._closed:
            return self.path
        field = self._spec["arrays"]["actions"]
        actions = (
            np.stack(self._actions)
            if self._actions
            else np.empty((0, *field["shape"]), field["dtype"])
        )
        transitions = {
            key: np.stack([obs[key] for obs in self._observations])
            for key in self._observations[0]
        }
        transitions.update(
            actions=actions,
            rewards=np.asarray(self._rewards, np.float32),
            terminated=np.asarray(self._terminated, np.bool_),
            truncated=np.asarray(self._truncated, np.bool_),
            action_source=(np.asarray(self._vla_ids, np.int32) >= 0).astype(np.uint8),
            primitive_id=np.asarray(self._primitive_ids, np.int32),
            vla_chunk_id=np.asarray(self._vla_ids, np.int32),
            proposal_index=np.asarray(self._proposal_indices, np.int16),
        )
        with (self._partial / "transitions.npz").open("wb") as stream:
            np.savez_compressed(stream, **transitions)

        proposal_actions = (
            np.stack([item["actions"] for item in self._proposals])
            if self._proposals
            else np.empty((0, 0, *field["shape"]), field["dtype"])
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

        training_steps = _training_step_count(transitions, self._spec)
        metadata = {
            **self._metadata,
            "is_success": bool(training_steps),
            "stop_reason": (
                "env_terminated"
                if any(self._terminated)
                else "env_truncated"
                if any(self._truncated)
                else "agent_stopped"
            ),
            "step_count": self.step_count,
            "training_step_count": training_steps,
            "primitive_names": self._primitive_names,
            "proposal_count": len(self._proposals),
        }
        (self._partial / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_episode(self._partial, spec=self._spec)
        os.replace(self._partial, self.path)
        self._closed = True
        return self.path


def validate_episode(path: Path | str, *, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the alignment needed by the exporter and training loader."""
    root = Path(path)
    metadata = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema in {root}")
    with np.load(root / "transitions.npz", allow_pickle=False) as data:
        count = int(metadata["step_count"])
        for key, field in spec["arrays"].items():
            array = data[key]
            length = count if key == "actions" else count + 1
            if array.shape != (length, *field["shape"]):
                raise ValueError(f"invalid {key} shape in {root}")
            if array.dtype != np.dtype(field["dtype"]):
                raise ValueError(f"invalid {key} dtype in {root}")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"non-finite {key} in {root}")
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
        expected = _training_step_count(data, spec)
        if (
            metadata["is_success"] != bool(expected)
            or metadata["training_step_count"] != expected
        ):
            raise ValueError(f"invalid training boundary in {root}")
    proposal_count = metadata["proposal_count"]
    with np.load(root / "proposals.npz", allow_pickle=False) as proposals:
        actions = proposals["actions"]
        field = spec["arrays"]["actions"]
        if (
            actions.ndim != len(field["shape"]) + 2
            or actions.shape[0] != proposal_count
            or actions.shape[2:] != field["shape"]
        ):
            raise ValueError(f"invalid proposal action shape in {root}")
        for key in ("created_step", "primitive_id", "instruction"):
            if proposals[key].shape != (proposal_count,):
                raise ValueError(f"invalid proposal {key} shape in {root}")
    return metadata
