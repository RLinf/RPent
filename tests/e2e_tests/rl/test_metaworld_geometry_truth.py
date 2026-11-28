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

"""Compare actual planner pixel queries with independent native visual-surface rays."""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robots.rl.envs.metaworld import LocalMetaWorldEnv
from robots.rl.toolkit import RLToolkit

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("metaworld")


def sample_rays(scene, camera, size):
    # Select table/object interiors using a separate native segmentation render.
    # All rays below use native camera convention, independent of RPent calibration.
    m, d = scene.model, scene.data
    renderer = scene.renderer
    renderer.update_scene(d, camera=camera)
    renderer.enable_segmentation_rendering()
    try:
        labels = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    native_size = labels.shape[0]
    scale = native_size // size
    iy = np.arange(size) * scale + scale // 2
    if camera == "corner2":
        iy = native_size - 1 - iy
    ids = labels[iy[:, None], iy[None, :], 0]
    types = labels[iy[:, None], iy[None, :], 1]
    angle = np.deg2rad(m.cam(camera).fovy.item())
    focal = native_size / (2 * np.tan(angle / 2))
    mid = (native_size - 1) / 2
    for body in ("tablelink", "obj"):
        mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & (
            m.geom_bodyid[ids.clip(min=0)] == m.body(body).id
        )
        # Keep a one-pixel margin against antialiasing at silhouette boundaries.
        from scipy.ndimage import binary_erosion

        candidates = np.argwhere(binary_erosion(mask))
        assert len(candidates) >= 3, (camera, size, body)
        for i in np.linspace(0, len(candidates) - 1, 5, dtype=int):
            row, col = candidates[i]
            ray = d.cam(camera).xmat.reshape(3, 3) @ [
                (iy[col] - mid) / focal,
                (mid - iy[row]) / focal,
                -1.0,
            ]
            ray /= np.linalg.norm(ray)
            hit = np.array([-1], np.int32)
            origin = d.cam(camera).xpos
            dist = mujoco.mj_ray(
                m, d, origin, ray, mujoco.MjvOption().geomgroup, 1, -1, hit
            )
            assert dist > 0 and m.geom_bodyid[hit.item()] == m.body(body).id, (
                body,
                row,
                col,
                hit,
            )
            xyz = origin + dist * ray
            if body == "tablelink":
                assert abs(xyz[2]) < 0.005
            yield body, [int(row), int(col)], xyz


def test_metaworld_geometry_truth(tmp_path):
    env = LocalMetaWorldEnv("pick-place-v3", 49)
    tk = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": None},
        environment="metaworld",
        record_artifacts=False,
    )
    rows = []
    try:
        for pose in ("initial", "moved"):
            if pose == "moved":
                target = np.asarray(tk._robot_state()["pose"]["xyz"]) + [0.016, 0, 0.02]
                result = tk.execute_tool(
                    "execute_actions",
                    {
                        "actions": [
                            {"action": "move_to", "xyz": (100 * target).tolist()}
                        ]
                    },
                )
                assert result.data["results"][0]["status"] == "reached", result.data
                assert tk._primitives.latest_step == 1
            for size in (256, 512, 1024):
                tk._primitives.display_size = size
                for cam in tk._primitives.cameras:
                    view = tk.execute_tool("view_env", {"camera": cam})
                    assert Image.open(io.BytesIO(view.images[0])).size == (size, size)
                    errors = {"obj": [], "tablelink": []}
                    for body, pixel, native in sample_rays(env.scene, cam, size):
                        expected = np.array([native[1], -native[0], native[2]])
                        res = tk.execute_tool(
                            "back_project",
                            {
                                "camera": cam,
                                "step": tk._primitives.latest_step,
                                "pixel": pixel,
                            },
                        )
                        assert not res.is_error, res.data
                        error = float(
                            np.linalg.norm(
                                np.asarray(res.data["world_xyz"]) / 100 - expected
                            )
                        )
                        errors[body].append(1000 * error)
                    for body, values in errors.items():
                        rows.append(
                            {
                                "camera": cam,
                                "pose": pose,
                                "display_size": size,
                                "surface": body,
                                "count": len(values),
                                "max_mm": max(values),
                                "median_mm": float(np.median(values)),
                            }
                        )
        report = Path(
            os.getenv(
                "RPENT_METAWORLD_GEOMETRY_REPORT",
                tmp_path / "geometry-truth-report.json",
            )
        )
        report.write_text(json.dumps(rows, indent=2))
        assert len(rows) == 24 and max(r["max_mm"] for r in rows) < 5, rows
    finally:
        env.close()
