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

"""Evaluator snapshots preserve complete model geometry in its body frame."""

from types import SimpleNamespace

import numpy as np

from robots.libero.env_server import LiberoEnvFacade
from toolkits.geometry.evaluation.evaluation_scene import (
    ReferenceEnvAdapter,
    snapshot_simulator,
)


def test_compiled_mesh_transform_and_child_geometry_exclusion():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    geom_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    position = np.array([0.4, -0.2, 0.8])
    local_offset = np.array([0.01, 0.02, 0.03])
    vertices = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.04, 0.0]])
    model = SimpleNamespace(
        geom_bodyid=np.array([1, 2]),
        geom_group=np.array([1, 1]),
        geom_dataid=np.array([0, 0]),
        geom_type=np.array([7, 7]),
        mesh_vertadr=np.array([0]),
        mesh_vertnum=np.array([3]),
        mesh_vert=vertices,
        mesh_faceadr=np.array([0]),
        mesh_facenum=np.array([1]),
        mesh_face=np.array([[0, 1, 2]]),
        geom_id2name=lambda _: "plate_visual",
        mesh_id2name=lambda _: "plate_mesh",
        body_id2name=lambda _: "plate_body",
    )
    data = SimpleNamespace(
        body_xpos=np.stack((np.zeros(3), position, position)),
        body_xmat=np.stack((np.eye(3), rotation, rotation)),
        geom_xpos=np.stack((position + rotation @ local_offset, position)),
        geom_xmat=np.stack((rotation @ geom_rotation, rotation)),
    )
    rob = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data), obj_body_id={"plate_1": 1}
    )
    snapshot = snapshot_simulator(rob)
    obj = snapshot["objects"]["plate_1"]
    assert len(obj["visual_geoms"]) == 1
    np.testing.assert_allclose(
        obj["visual_geoms"][0]["vertices_body"],
        vertices @ geom_rotation.T + local_offset,
        atol=1e-12,
    )
    np.testing.assert_allclose(obj["body_position_world"], position)
    assert snapshot["units"] == "m"
    adapter = ReferenceEnvAdapter(SimpleNamespace(env=rob, value=7))
    assert adapter.value == 7 and adapter.rpent_reference_snapshot == snapshot


def test_reference_rpc_is_absent_from_normal_env_servers():
    normal = LiberoEnvFacade(SimpleNamespace(), meta={})
    assert "evaluation.reference_snapshot" not in normal._rpc
    worker = SimpleNamespace(get_env_attr=lambda name: {"name": name})
    evaluation = LiberoEnvFacade(
        SimpleNamespace(
            _rpent_evaluation_enabled=True, env=SimpleNamespace(workers=[worker])
        ),
        meta={},
    )
    assert evaluation.reference_snapshot() == {"name": "rpent_reference_snapshot"}
    assert "evaluation.reference_snapshot" in evaluation._readonly_methods
