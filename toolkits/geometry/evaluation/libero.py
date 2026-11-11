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

"""Capture and load fixed LIBERO frames with separate evaluator-only snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import numpy as np

from rpent.evaluation.result import write_json_atomic
from rpent.session import EnvState
from rpent.utils.serialization import to_numpy_tree
from toolkits.geometry.evaluation.checklist import frame_id, manifest_digest
from toolkits.geometry.observation import GeometryObservation

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


def make_environment(case: dict[str, Any]) -> Any:
    """Build one opt-in reference-reading environment with the existing config."""
    os.environ.setdefault("LIBERO_TYPE", "pro")
    if os.environ["LIBERO_TYPE"] != "pro":
        raise ValueError("the fixed checklist requires LIBERO_TYPE=pro")
    # Import the production configuration before RLinf imports MuJoCo. This
    # retains its GL initialization order without changing or patching its code.
    configuration = import_module("robots.libero.env_server")
    libero = import_module("rlinf.envs.sim.libero.libero_env")
    benchmark = import_module("rlinf.envs.sim.libero.utils").benchmark

    suite = benchmark.get_benchmark(case["suite"])()
    task = case["task"]
    if not 0 <= task < suite.n_tasks:
        raise ValueError("checklist task is absent from the installed LIBERO suite")
    trials = len(suite.get_task_init_states(task))
    if not trials:
        raise ValueError("the selected task has no initial states")
    reset_id = sum(len(suite.get_task_init_states(i)) for i in range(task))
    reset_id += case["seed"] % trials
    cfg = configuration.build_env_cfg(
        task_suite_name=case["suite"],
        specific_reset_id=reset_id,
        seed=case["seed"],
        max_episode_steps=1,
    )
    return evaluation_env_class(libero.LiberoEnv)(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def capture_case(
    case: dict[str, Any],
    *,
    height: int = 1024,
    width: int = 1024,
    environment_factory: Callable[[dict[str, Any]], Any] = make_environment,
) -> tuple[GeometryObservation, dict[str, Any]]:
    """Capture sensors and independent model geometry at the same settled reset.

    No action or planner is constructed. Only the environment created by this
    call is closed, including when rendering or reference collection fails.
    """
    from robots.libero.tools import _metric_depth, _world_from_depth

    env = environment_factory(case)
    try:
        env.reset()
        rgb, depth = to_numpy_tree(
            env.render_camera("agentview", height=height, width=width, depth=True)
        )
        meta = to_numpy_tree(
            env.get_camera_meta("agentview", height=height, width=width)
        )
        if not isinstance(meta, dict):
            raise ValueError("LIBERO did not return camera calibration")
        # Match the existing LIBERO dump convention: flip RGB and depth
        # together, linearize depth, then project with that camera's metadata.
        rgb = np.ascontiguousarray(np.asarray(rgb)[::-1])
        depth = np.ascontiguousarray(_metric_depth(depth, meta)[::-1])
        observation = GeometryObservation(
            camera="agentview",
            step=0,
            world_frame_id=frame_id(case["case_id"]),
            rgb=rgb,
            depth_m=depth,
            world_xyz=_world_from_depth(depth, meta),
            intrinsics=meta["intrinsic_K"],
            camera_to_world=meta["extrinsic_cam2world"],
            up_direction=np.array([0.0, 0.0, 1.0]),
        )
        snapshot = to_numpy_tree(
            env.env.workers[0].get_env_attr("rpent_reference_snapshot")
        )
        snapshot.update(
            case_id=case["case_id"],
            frame_id=observation.world_frame_id,
            step=0,
            camera="agentview",
        )
        return observation, snapshot
    finally:
        env.env.close()


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
