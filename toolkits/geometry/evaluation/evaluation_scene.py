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

"""Opt-in evaluator snapshots of compiled LIBERO geometry and current body poses.

This module is used only by reference collection. Snapshots are never observation
artifacts and are not passed to GeometryAgent or the visual grounding model.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

REFERENCE_OBJECTS = {
    "plate": "plate_1",
    "ramekin": "glazed_rim_porcelain_ramekin_1",
    "cheese": "cream_cheese_1",
    "soup": "alphabet_soup_1",
    "black_bowl": "akita_black_bowl_1",
    "wine_bottle": "wine_bottle_1",
    "gray_white_mug": "porcelain_mug_1",
    "yellow_white_mug": "white_yellow_mug_1",
    "microwave": "microwave_1",
}


def snapshot_simulator(rob: Any) -> dict[str, Any]:
    """Read mesh vertices and FK poses directly from the same settled simulator."""
    sim = rob.sim
    model, data = sim.model, sim.data
    objects = {}
    for name in REFERENCE_OBJECTS.values():
        if name not in rob.obj_body_id:
            continue
        body_id = int(rob.obj_body_id[name])
        position = np.asarray(data.body_xpos[body_id], dtype=float)
        rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
        geoms = []
        for gid in np.flatnonzero(np.asarray(model.geom_bodyid) == body_id):
            if int(model.geom_group[gid]) != 1:
                continue
            mesh_id = int(model.geom_dataid[gid])
            # MuJoCo mjGEOM_MESH=7. Primitive-only visual geoms are recorded
            # by name/type but cannot masquerade as a labelled mesh reference.
            if int(model.geom_type[gid]) != 7 or mesh_id < 0:
                continue
            start, count = (
                int(model.mesh_vertadr[mesh_id]),
                int(model.mesh_vertnum[mesh_id]),
            )
            vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
            face_start, face_count = (
                int(model.mesh_faceadr[mesh_id]),
                int(model.mesh_facenum[mesh_id]),
            )
            faces = np.asarray(
                model.mesh_face[face_start : face_start + face_count], dtype=int
            )
            geom_rotation = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            geom_position = np.asarray(data.geom_xpos[gid], dtype=float)
            local = (vertices @ geom_rotation.T + geom_position - position) @ rotation
            geoms.append(
                {
                    "geom_name": model.geom_id2name(int(gid)),
                    "mesh_name": model.mesh_id2name(mesh_id),
                    "vertices_body": local.tolist(),
                    "faces": faces.tolist(),
                    "mesh_sha256": hashlib.sha256(
                        np.round(local, 9).tobytes() + faces.tobytes()
                    ).hexdigest(),
                }
            )
        objects[name] = {
            "body_name": model.body_id2name(body_id),
            "body_id": body_id,
            "body_position_world": position.tolist(),
            "body_rotation_world": rotation.tolist(),
            "visual_geoms": geoms,
        }
    return {
        "schema_version": 1,
        "units": "m",
        "source": "MuJoCo compiled visual meshes and sim.data body/geom FK at the captured initial state",
        "objects": objects,
    }


class ReferenceEnvAdapter:
    """Add one evaluator attribute without changing reset, rendering or physics."""

    def __init__(self, env: Any) -> None:
        self._wrapped = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def rpent_reference_snapshot(self) -> dict[str, Any]:
        rob = self._wrapped
        while hasattr(rob, "env"):
            rob = rob.env
        return snapshot_simulator(rob)


def evaluation_env_class(base: type) -> type:
    """Wrap worker factories using RLinf's existing get-env-attribute transport."""

    class EvaluationEnv(base):
        _rpent_evaluation_enabled = True

        def get_env_fns(self):
            factories = super().get_env_fns()
            return [
                lambda factory=factory: ReferenceEnvAdapter(factory())
                for factory in factories
            ]

    return EvaluationEnv
