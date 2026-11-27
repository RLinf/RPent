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

"""Opt-in geometry and scene-isolation checks against the native Isaac runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest

from robots.rl.envs.robolab import RoboLabRLPrimitives
from robots.robolab.access import CAMERAS
from robots.robolab.episode_server import RoboLabEpisodeRuntime, _prim

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOLAB_INTEGRATION") != "1",
    reason="requires the RoboLab Isaac Sim environment and an available RTX device",
)


def _scene_signature(episode):
    states = {
        "joints": episode.robot.data.joint_pos.detach().cpu().numpy().copy(),
    }
    for name, asset in episode.env.scene.rigid_objects.items():
        states[name] = asset.data.root_state_w.detach().cpu().numpy().copy()
    return states


def _usd_camera(episode, name):
    from pxr import Gf, Usd, UsdGeom

    prim = _prim(episode.env, getattr(episode.env.cfg.scene, name))
    camera = UsdGeom.Camera(prim).GetCamera(Usd.TimeCode.Default())
    if name == "wrist_cam":
        # PhysX provides link transforms in xyzw. Compose the authored camera
        # offset using USD's row-vector API, independently of RPent's arrays.
        link = episode.robot.body_names.index(prim.GetParent().GetName())
        pose = episode.robot.root_physx_view.get_link_transforms()[0, link].tolist()
        link_to_world = Gf.Matrix4d(1.0)
        link_to_world.SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
        link_to_world.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
        camera.transform = (
            UsdGeom.Xformable(prim).GetLocalTransformation() * link_to_world
        )
    return camera


def _table_ray_errors(episode, primitive, camera, table_path):
    import omni.physx
    from pxr import Gf

    _, points, _ = primitive.capture(camera)
    sensor = episode.env.scene[camera].data
    kind = "instance_id_segmentation_fast"
    info = sensor.info if isinstance(sensor.info, dict) else sensor.info[0]
    labels = info[kind]["idToLabels"]
    table_ids = [
        int(label)
        for label, prim in labels.items()
        if str(prim) == table_path or str(prim).startswith(table_path + "/")
    ]
    assert table_ids, (
        f"No table renderer instance for {camera} at {table_path}: {labels}"
    )
    table_pixels = np.isin(
        sensor.output[kind][0, ..., 0].detach().cpu().numpy(), table_ids
    )
    view = _usd_camera(episode, camera)
    origin = view.transform.ExtractTranslation()
    queries = omni.physx.get_physx_scene_query_interface()
    height, width = points.shape[:2]

    errors, tops = [], []
    for row in np.linspace(0.2 * height, 0.85 * height, 9, dtype=int):
        for col in np.linspace(0.15 * width, 0.85 * width, 11, dtype=int):
            # Accept pixels classified as table by both RTX and PhysX.
            # This excludes visible meshes with different collision coverage.
            region = np.s_[row - 1 : row + 2, col - 1 : col + 2]
            if not table_pixels[region].all():
                continue
            nearby = points[region]
            if (
                not np.isfinite(nearby).all()
                or np.max(np.ptp(nearby, axis=(0, 1))) > 0.02
            ):
                continue
            # Generate rays using the authored USD optics and PhysX pose.
            local = Gf.Vec3d(
                ((float(col) + 0.5) / width - 0.5) * view.horizontalAperture
                + view.horizontalApertureOffset,
                (0.5 - (float(row) + 0.5) / height) * view.verticalAperture
                + view.verticalApertureOffset,
                -view.focalLength,
            )
            direction = view.transform.TransformDir(local).GetNormalized()
            hit = queries.raycast_closest(tuple(origin), tuple(direction), 4.0)
            if not hit["hit"] or not str(hit["collision"]).startswith(table_path):
                continue
            if hit["normal"][2] < 0.98:
                continue
            point = np.asarray(hit["position"])
            errors.append(float(np.linalg.norm(points[row, col] - point)))
            tops.append(float(point[2]))
    assert len(errors) >= 3, f"{camera} returned only {len(errors)} visible table hits"
    return errors, tops


def _run_native_probe():
    """Check real RGB-D against PhysX and revisit a seed after a different task."""
    gpu = int(os.environ.get("RPENT_ROBOLAB_GPU", "0"))
    started = time.perf_counter()
    runtime = RoboLabEpisodeRuntime(cuda_device=gpu)
    print(f"RoboLab cold app startup {time.perf_counter() - started:.2f}s", flush=True)
    from robolab.core.environments import runtime as native_runtime

    create_env = native_runtime.create_env

    def with_scene_queries(cfg, **kwargs):
        cfg.sim.enable_scene_query_support = True
        for name in CAMERAS:
            camera = getattr(cfg.scene, name)
            camera.data_types.append("instance_id_segmentation_fast")
            camera.colorize_instance_id_segmentation = False
        return create_env(cfg, **kwargs)

    original_scene = {
        "task": "BananaInBowlTask",
        "seed": 0,
        "max_episode_steps": 10000,
        "save_dir": None,
    }
    try:
        with patch.object(native_runtime, "create_env", with_scene_queries):
            episode = runtime.open_episode(**original_scene)
        assert episode.get_status()["step"] == 0
        original = _scene_signature(episode)
        primitive = RoboLabRLPrimitives(env=episode, check_cancelled=lambda: None)
        coordinates = primitive.coordinates
        table_asset = (
            getattr(episode.env.cfg.scene, "table", None)
            or episode.env.cfg.scene.table_fixture
        )
        table_path = str(_prim(episode.env, table_asset).GetPath())
        table = np.asarray(
            episode.coordinate_calibration()["coordinate_calibration"]["table_top_xyz"]
        )
        to_table = table - coordinates.origin
        to_table[2] = 0.0
        forward = float(
            coordinates.rotation[:, 0] @ (to_table / np.linalg.norm(to_table))
        )
        assert forward > 0.9

        for camera in CAMERAS:
            errors, tops = _table_ray_errors(episode, primitive, camera, table_path)
            max_error = max(errors)
            height_error = abs(
                np.median(tops) - coordinates.origin[2] - coordinates.table_height_m
            )
            print(
                f"{camera}: {len(errors)} rays, max {1000 * max_error:.2f} mm, "
                f"table height error {1000 * height_error:.2f} mm",
                flush=True,
            )
            assert max_error < 0.005
            assert height_error < 0.005

        moved_rpy = primitive.robot_state()["pose"]["orientation_rpy"]
        moved_rpy[2] += 0.15
        move = primitive.move_to(orientation_rpy=moved_rpy)
        assert move.data["status"] == "reached"
        assert episode.get_status()["step"] > 0
        errors, _ = _table_ray_errors(episode, primitive, "wrist_cam", table_path)
        print(
            f"wrist_cam after motion: {len(errors)} rays, max {1000 * max(errors):.2f} mm",
            flush=True,
        )
        assert max(errors) < 0.005

        runtime.close_episode()
        start = time.perf_counter()
        other = runtime.open_episode(
            **{**original_scene, "task": "FruitsOrangesOnPlateTask", "seed": 1}
        )
        assert other.get_status()["step"] == 0
        assert other.get_task_language() != episode.get_task_language()
        assert other.get_env_meta()["seed"] == 1
        print(
            f"RoboLab warm cross-task scene open {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        runtime.close_episode()

        start = time.perf_counter()
        with patch.object(native_runtime, "create_env", with_scene_queries):
            revisited = runtime.open_episode(**original_scene)
        assert revisited.get_status()["step"] == 0
        repeated = _scene_signature(revisited)
        assert repeated.keys() == original.keys()
        for name, value in original.items():
            np.testing.assert_allclose(
                repeated[name], value, rtol=0, atol=1e-5, err_msg=name
            )
        print(
            f"RoboLab warm repeat seed open {time.perf_counter() - start:.2f}s",
            flush=True,
        )
        print("ROBO_LAB_NATIVE_PROBE_PASSED", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        runtime.close()


def test_native_geometry_and_scene_reuse():
    """Run Isaac in its own process because Kit owns its shutdown."""
    script = f"import runpy; runpy.run_path({__file__!r}, run_name='__main__')"
    run = subprocess.run(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    print(run.stdout)
    assert run.returncode == 0 and "ROBO_LAB_NATIVE_PROBE_PASSED" in run.stdout


if __name__ == "__main__":
    _run_native_probe()
