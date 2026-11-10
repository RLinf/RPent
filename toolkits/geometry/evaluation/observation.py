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

"""Read and write sensor-only initial frames, including the original captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import numpy as np

from rpent.evaluation.result import write_json_atomic
from rpent.session import EnvState
from toolkits.geometry.evaluation.checklist import frame_id, manifest_digest
from toolkits.geometry.observation import GeometryObservation

SENSOR_FILES = (
    "agentview_high.png",
    "agentview_world_high.npz",
    "agentview_depth_high.npz",
    "agentview_valid_high.npz",
    "agentview_metadata_high.json",
)


def file_digest(path: Path) -> str:
    """Hash an input without loading a large point map into a second byte buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sensor_path(root: Path, name: str) -> Path:
    """Resolve only the fixed sensor whitelist in an EnvState artifact layout."""
    if name not in SENSOR_FILES:
        raise ValueError(f"not a sensor artifact: {name}")
    return root / "episode" / name / ("00" + Path(name).suffix)


def save_observation(
    root: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    observation: GeometryObservation,
) -> None:
    """Save one step-zero RGB/metric snapshot, never simulator object state."""
    if (
        observation.step != 0
        or observation.camera != "agentview"
        or observation.world_frame_id != frame_id(case["case_id"])
    ):
        raise ValueError("capture does not match the fixed initial-frame protocol")
    state = EnvState(root / "episode")
    arrays = {
        "agentview_high.png": observation.rgb,
        "agentview_world_high.npz": observation.world_xyz.astype(np.float32),
        "agentview_valid_high.npz": observation.valid_mask,
        "agentview_metadata_high.json": {
            "intrinsic_K": None
            if observation.intrinsics is None
            else observation.intrinsics.tolist(),
            "extrinsic_cam2world": None
            if observation.camera_to_world is None
            else observation.camera_to_world.tolist(),
        },
    }
    if observation.depth_m is not None:
        arrays["agentview_depth_high.npz"] = observation.depth_m.astype(np.float32)
    with state.record_step(state={}):
        for name, value in arrays.items():
            if state.save(name, value) is None:
                raise RuntimeError(f"could not persist {name}")
    write_json_atomic(
        root / "observation.json",
        {
            "schema_version": 1,
            "manifest_sha256": manifest_digest(manifest),
            "case_id": case["case_id"],
            "frame_id": observation.world_frame_id,
            "step": 0,
            "camera": "agentview",
            "units": "m",
            "artifact_sha256": {
                name: file_digest(sensor_path(root, name)) for name in arrays
            },
        },
    )


def load_observation(
    root: Path, manifest: dict[str, Any], case: dict[str, Any]
) -> GeometryObservation:
    """Verify sensor identity and hashes before exposing a snapshot to the tool."""
    identity = json.loads((root / "observation.json").read_text())
    for key, expected in {
        "schema_version": 1,
        "manifest_sha256": manifest_digest(manifest),
        "case_id": case["case_id"],
        "frame_id": frame_id(case["case_id"]),
        "step": 0,
        "camera": "agentview",
        "units": "m",
    }.items():
        if identity.get(key) != expected:
            raise ValueError(f"observation {key} mismatch: {case['case_id']}")
    steps = json.loads((root / "episode/states.json").read_text())["steps"]
    if len(steps) != 1 or steps[0]["step_idx"] != 0:
        raise ValueError("expected exactly one recorded initial frame")
    hashes = identity["artifact_sha256"]
    required = {"agentview_high.png", "agentview_world_high.npz"}
    if not required <= hashes.keys():
        raise ValueError("capture requires native RGB and aligned metric coordinates")
    values = {}
    for name in SENSOR_FILES:
        if name not in hashes:
            continue
        path = sensor_path(root, name)
        if file_digest(path) != hashes[name]:
            raise ValueError(f"sensor hash mismatch: {name}")
        if name.endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                values[name] = archive["array"]
        elif name.endswith(".json"):
            values[name] = json.loads(path.read_text())
        else:
            values[name] = imageio.imread(path)
    meta = values.get("agentview_metadata_high.json", {})
    return GeometryObservation(
        camera="agentview",
        step=0,
        world_frame_id=identity["frame_id"],
        rgb=values["agentview_high.png"],
        world_xyz=values["agentview_world_high.npz"],
        depth_m=values.get("agentview_depth_high.npz"),
        valid_mask=values.get("agentview_valid_high.npz"),
        intrinsics=meta.get("intrinsic_K"),
        camera_to_world=meta.get("extrinsic_cam2world"),
        up_direction=np.array([0.0, 0.0, 1.0]),
        source_refs={name: hashes[name] for name in SENSOR_FILES if name in hashes},
    )


class FrozenObservationProvider:
    """Serve only the verified frame pinned to an individual checklist case."""

    def __init__(self, observation: GeometryObservation) -> None:
        self.observation = observation

    def get_observation(self, *, camera: str, step: int) -> GeometryObservation:
        """Resolve latest to step zero without capturing another view."""
        if camera != self.observation.camera or step not in (-1, 0):
            raise ValueError("the evaluation only contains agentview step zero")
        return self.observation
