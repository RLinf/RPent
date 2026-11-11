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

"""Resolve independently fixed model points at each captured simulator pose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.geometry.evaluation.checklist import frame_id, manifest_digest
from toolkits.geometry.evaluation.evaluation_scene import REFERENCE_OBJECTS
from toolkits.geometry.evaluation.metrics import truth_template, validate_truth
from toolkits.geometry.evaluation.observation import file_digest
from toolkits.geometry.evaluation.reference_profiles import REFERENCE_PROFILES


def mesh_fingerprint(geom: dict[str, Any]) -> str:
    """Match the frozen catalog's body-local mesh fingerprint at nanometre rounding."""
    vertices = np.asarray(geom["vertices_body"], dtype=float)
    faces = np.asarray(geom["faces"], dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(vertices)
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("invalid compiled reference mesh")
    return hashlib.sha256(np.round(vertices, 9).tobytes() + faces.tobytes()).hexdigest()


def build_truth(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    inputs: Path,
    *,
    profiles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check model identity, then transform fixed local points using same-frame FK.

    A model point remains evaluable when occluded. This function never reads
    masks, selected pixels or predictions and never substitutes a body origin.
    """
    profiles = REFERENCE_PROFILES if profiles is None else profiles
    if (
        profiles.get("schema_version") != 1
        or profiles.get("units") != "m"
        or not profiles.get("frozen_at")
    ):
        raise ValueError(
            "reference profiles must be frozen metre-based model annotations"
        )
    truth = truth_template(manifest, cases)
    annotations = {row["request_id"]: row for row in truth["annotations"]}
    profile_hash = manifest_digest(profiles)
    truth["profile_sha256"] = profile_hash
    for case in cases:
        path = inputs / case["case_id"] / "evaluator_snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot_hash = file_digest(path)
        for key, expected in {
            "schema_version": 1,
            "case_id": case["case_id"],
            "frame_id": frame_id(case["case_id"]),
            "units": "m",
            "step": 0,
            "camera": "agentview",
        }.items():
            if snapshot.get(key) != expected:
                raise ValueError(
                    f"reference snapshot {key} mismatch: {case['case_id']}"
                )
        for request in case["requests"]:
            name = request["reference"]
            profile = profiles["references"][name]
            obj = snapshot["objects"][REFERENCE_OBJECTS[name]]
            matches = [
                geom
                for geom in obj["visual_geoms"]
                if geom["mesh_name"] == profile["compiled_mesh_name"]
                and mesh_fingerprint(geom) == profile["compiled_mesh_sha256"]
            ]
            if not matches:
                raise ValueError(
                    f"compiled model differs from frozen reference: {name}"
                )
            pose = np.eye(4)
            pose[:3, :3] = obj["body_rotation_world"]
            pose[:3, 3] = obj["body_position_world"]
            annotations[request["request_id"]].update(
                evaluable=True,
                model_local_xyz=profile["body_local_xyz"],
                model_to_world=pose.tolist(),
                annotation_source=(
                    f"Frozen complete-model profile {profile_hash}; {profile['method']}; "
                    f"same-frame MuJoCo FK; snapshot {snapshot_hash}"
                ),
                profile_sha256=profile_hash,
                snapshot_sha256=snapshot_hash,
                body_name=obj["body_name"],
            )
    validate_truth(manifest, truth, cases)
    return truth
